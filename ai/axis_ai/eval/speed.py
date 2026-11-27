"""추론 속도: 입력 크기별 모델 처리 시간, ONNX/TFLite 변환과 PyTorch 대비 속도·오차.

python -m axis_ai.eval.speed --model models/<v> [--sizes 256,320,480,640] [--image <jpg>] [--out ...]
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import emit


def time_fn(fn, n: int = 20, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95))}


class _HeadModule(torch.nn.Module):
    """정규화까지 포함한 Guide Head(ONNX 내보내기용)."""

    def __init__(self, head):
        super().__init__()
        self.net = head.net
        for k in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(k, getattr(head, k).clone().float())

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


def export_head_onnx(head, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = _HeadModule(head).eval()
    torch.onnx.export(m, torch.zeros(1, m.x_mean.shape[0]), str(path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
    return path


def compare_onnx(head, path: Path, X: np.ndarray) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x1 = X[:1].astype(np.float32)
    want = head.predict(X.astype(np.float32))
    got = sess.run(None, {"x": X.astype(np.float32)})[0]
    return {"torch_ms": time_fn(lambda: head.predict(x1))["mean_ms"],
            "onnx_ms": time_fn(lambda: sess.run(None, {"x": x1}))["mean_ms"],
            "max_abs_err": float(np.abs(want - got).max())}


def _yolo_rows(weights: Path, img: np.ndarray, sizes: list[int]) -> list[tuple]:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    return [(weights.stem, s, time_fn(lambda s=s: m(img, imgsz=s, verbose=False), n=15)) for s in sizes]


def _export_yolo(weights: Path, out_dir: Path, img: np.ndarray, size: int) -> list[str]:
    from ultralytics import YOLO

    lines = []
    pt = YOLO(str(weights))
    torch_t = time_fn(lambda: pt(img, imgsz=size, verbose=False), n=15)
    for fmt in ("onnx", "tflite"):
        try:
            exported = Path(YOLO(str(weights)).export(format=fmt, imgsz=size, verbose=False))
            dest = out_dir / exported.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(exported), dest)
            files = [dest] if dest.is_file() else [p for p in dest.rglob("*") if p.is_file()]
            mb = sum(p.stat().st_size for p in files) / 1e6
            if fmt == "onnx":
                om = YOLO(str(dest), task=pt.task)
                t = time_fn(lambda: om(img, imgsz=size, verbose=False), n=15)
                lines.append(f"| {weights.stem} | {fmt} | 성공 | {mb:.1f} | {torch_t['mean_ms']:.1f} | "
                             f"{t['mean_ms']:.1f} |")
            else:
                lines.append(f"| {weights.stem} | {fmt} | 성공 | {mb:.1f} | - | - |")
        except Exception as e:  # noqa: BLE001 — 변환 실패도 결과로 기록한다
            lines.append(f"| {weights.stem} | {fmt} | 실패: {str(e).splitlines()[0][:80]} | - | - | - |")
    return lines


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="입력 크기별 추론 속도와 모델 변환 비교")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--sizes", default="256,320,480,640")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    from ..bundle import ModelBundle
    from ..depth import DepthEstimator
    from ..detect import DEFAULT_FOOD_WEIGHTS
    from ..embed import Embedder
    from ..head.model import INPUT_DIM
    from ..pose import DEFAULT_WEIGHTS

    sizes = [int(s) for s in args.sizes.split(",")]
    if args.image:
        pil = Image.open(args.image).convert("RGB")
    else:
        pil = Image.new("RGB", (480, 640), (120, 120, 120))
    img = np.asarray(pil)
    result: dict = {"sizes": {}}
    lines = ["# 추론 속도와 모델 변환", "", f"입력 사진 {pil.size[0]}x{pil.size[1]}, 같은 사진을 반복 처리한 평균(ms).", "",
             "## 입력 크기별 처리 시간", "", "| 모델 | 입력 크기 | 평균 | p95 |", "|---|---|---|---|"]
    rows = _yolo_rows(DEFAULT_WEIGHTS, img, sizes) + _yolo_rows(DEFAULT_FOOD_WEIGHTS, img, sizes)
    depth_models = {s: DepthEstimator(size=s) for s in sizes}
    rows += [("depth-anything-v2-small", s, time_fn(lambda s=s: depth_models[s].estimate(pil), n=10)) for s in sizes]
    emb = Embedder()
    rows.append(("mobilenet-v3-small", 224, time_fn(lambda: emb.embed(pil))))
    for name, s, t in rows:
        result["sizes"][f"{name}@{s}"] = t
        lines.append(f"| {name} | {s} | {t['mean_ms']:.1f} | {t['p95_ms']:.1f} |")

    out_dir = args.model / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines += ["", "## 모델 변환", "", "| 모델 | 형식 | 결과 | 크기(MB) | PyTorch(ms) | 변환본(ms) |",
              "|---|---|---|---|---|---|"]
    lines += _export_yolo(DEFAULT_WEIGHTS, out_dir, img, 320)
    bundle = ModelBundle.load(args.model)
    result["head"] = {}
    X = np.random.default_rng(0).normal(size=(64, INPUT_DIM)).astype(np.float32)
    for sid, head in sorted(bundle.heads.items()):
        p = export_head_onnx(head, out_dir / f"head_{sid}.onnx")
        r = compare_onnx(head, p, X)
        result["head"][sid] = r
        lines.append(f"| Guide Head {sid} | onnx | 성공(최대 오차 {r['max_abs_err']:.1e}) | "
                     f"{p.stat().st_size / 1e6:.2f} | {r['torch_ms']:.3f} | {r['onnx_ms']:.3f} |")
    emit("\n".join(lines), args.out)
    return result


if __name__ == "__main__":
    main()
