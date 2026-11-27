"""이미지 기반 합성 평가: test 사진을 잘라내기·padding으로 어긋나게 만들고, stat·rank·head가 같은 이미지를 보고
안내를 따라가 OK에 닿는지 잰다. head는 보정값의 MAE·MSE와 작가 간 교차 비교도 잰다.

python -m axis_ai.eval.image_guide --processed data/processed --raw data/raw --model models/<v>
       [--scorers stat,rank,head] [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..bundle import ModelBundle
from ..data.build import load_processed
from ..data.depth_maps import load_depth_maps
from ..data.manifest import open_image
from ..depth import depth_features
from ..guide import MOVES, decide
from ..head.augment import correction, warp
from ..head.model import correction_to_direction, head_input
from . import emit, pct
from .synthetic_guide import DIRECTIONS, MAGNITUDES, MAX_STEPS

ROTATIONS = (-10.0, 10.0)
VALUE_NAMES = ("dx", "dy", "log배율", "각도")


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def variant_params(direction: str, m: float) -> tuple[float, float, float, float]:
    """정답이 direction이 되도록 어긋나게 하는 (dx, dy, scale, angle). synthetic_guide.perturb와 같은 뜻."""
    if direction == "CLOSER":
        return 0.0, 0.0, 1 / (1 + m), 0.0
    if direction == "FARTHER":
        return 0.0, 0.0, 1 + m, 0.0
    dx, dy, _ = MOVES[direction]
    return -m * _sign(dx), -m * _sign(dy), 1.0, 0.0


class _Frame:
    """평가 중 한 장면(이미지, 인물, 깊이 지도)."""

    def __init__(self, img, person, dmap):
        self.img, self.person, self.dmap = img, person, dmap

    def moved(self, dx, dy, scale, angle=0.0):
        img, p, d = warp(self.img, self.person, self.dmap, dx, dy, scale, angle)
        return None if p is None else _Frame(img, p, d)


def _head_correction(frame: _Frame, head, embedder) -> np.ndarray:
    emb = embedder.embed(Image.fromarray(frame.img))
    dfeat = depth_features(frame.dmap, frame.person.box) if frame.dmap is not None else None
    return head.predict(head_input(emb, frame.person, dfeat))


def _direction(kind: str, frame: _Frame, style: str, bundle: ModelBundle, embedder) -> str:
    if kind == "head":
        return correction_to_direction(_head_correction(frame, bundle.heads[style], embedder))[0]
    return decide(frame.person, bundle.scorers[kind][style]).direction


def _follow(kind, frame, style, bundle, embedder) -> int | None:
    for step in range(MAX_STEPS + 1):
        d = _direction(kind, frame, style, bundle, embedder)
        if d == "OK":
            return step
        if d not in MOVES or step == MAX_STEPS:
            return None
        frame = frame.moved(*MOVES[d])
        if frame is None:
            return None
    return None


def _available(kind: str, style: str, bundle: ModelBundle) -> bool:
    return style in bundle.heads if kind == "head" else style in bundle.scorers.get(kind, {})


def evaluate(rows: list[dict], raw_dir: Path, depth_maps: dict, bundle: ModelBundle, embedder,
             kinds: list[str]) -> dict:
    out: dict = {}
    frames = []  # (style, 원본 Frame)
    for r in rows:
        dmap = depth_maps.get(r["image_id"])
        img = np.asarray(open_image(Path(raw_dir) / r["file"]).convert("RGB"))
        frames.append((r["style_id"], _Frame(img, r["person"], None if dmap is None else dmap.astype(np.float32))))

    for kind in kinds:
        per_dir = {d: [0, 0] for d in DIRECTIONS}
        rec_dir = {d: 0 for d in DIRECTIONS}
        steps, ok_count, total = [], 0, 0
        errs = []
        for style, base in frames:
            if not _available(kind, style, bundle):
                continue
            for d in DIRECTIONS:
                for m in MAGNITUDES:
                    params = variant_params(d, m)
                    v = base.moved(*params)
                    if v is None:
                        continue
                    if kind == "head":
                        c = _head_correction(v, bundle.heads[style], embedder)
                        errs.append(c - correction(*params))
                        got = correction_to_direction(c)[0]
                    else:
                        got = _direction(kind, v, style, bundle, embedder)
                    total += 1
                    ok_count += got == "OK"
                    per_dir[d][1] += 1
                    per_dir[d][0] += got == d
                    k = _follow(kind, v, style, bundle, embedder)
                    if k is not None:
                        rec_dir[d] += 1
                        steps.append(k)
            if kind == "head":
                for a in ROTATIONS:
                    v = base.moved(0.0, 0.0, 1.0, a)
                    if v is not None:
                        errs.append(_head_correction(v, bundle.heads[style], embedder) - correction(0, 0, 1, a))
        if total == 0:
            continue
        res = {
            "accuracy": sum(c for c, _ in per_dir.values()) / total,
            "n": total,
            "ok_share": ok_count / total,
            "per_direction": {d: (c / n if n else None) for d, (c, n) in per_dir.items()},
            "recovery": len(steps) / total,
            "mean_steps": (sum(steps) / len(steps)) if steps else None,
        }
        if kind == "head":
            e = np.stack(errs)
            res["mae"] = [float(v) for v in np.abs(e).mean(axis=0)]
            res["mse"] = [float(v) for v in (e ** 2).mean(axis=0)]
        out[kind] = res

    if "head" in kinds and bundle.heads:
        cross: dict[str, float] = {}
        for style in sorted(bundle.heads):
            diff = n = 0
            for s, base in frames:
                if s != style:
                    continue
                for d in DIRECTIONS:
                    v = base.moved(*variant_params(d, MAGNITUDES[-1]))
                    if v is None:
                        continue
                    own = correction_to_direction(_head_correction(v, bundle.heads[style], embedder))[0]
                    for other in bundle.heads:
                        if other == style:
                            continue
                        n += 1
                        diff += own != correction_to_direction(
                            _head_correction(v, bundle.heads[other], embedder))[0]
            if n:
                cross[style] = diff / n
        out["cross"] = cross
    return out


def _report(result: dict, n_rows: int, history: dict) -> str:
    kinds = [k for k in result if k != "cross"]
    lines = ["# 안내 평가 (이미지 기반 합성)", "",
             f"test 사진 {n_rows}장을 잘라내기·반사 padding으로 어긋나게 만들었다(방향 6개 x 크기 {MAGNITUDES}). "
             f"모든 방식이 같은 이미지를 보고, 안내대로 이미지를 다시 옮기며 최대 {MAX_STEPS}단계 따라간다.", "",
             "## 방식 비교", "",
             "| 방식 | 첫 안내 정확도 | OK 도달 | 평균 단계 | OK 응답 비율 | 건수 |", "|---|---|---|---|---|---|"]
    for k in kinds:
        v = result[k]
        ms = "-" if v["mean_steps"] is None else f"{v['mean_steps']:.1f}"
        lines.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['recovery'])} | {ms} | {pct(v['ok_share'])} | {v['n']} |")
    lines += ["", "| 방식 | " + " | ".join(DIRECTIONS) + " |", "|---|" + "---|" * len(DIRECTIONS)]
    for k in kinds:
        pdir = result[k]["per_direction"]
        lines.append(f"| {k} | " + " | ".join("-" if pdir[d] is None else pct(pdir[d]) for d in DIRECTIONS) + " |")
    if "head" in result:
        h = result["head"]
        lines += ["", "## Guide Head 보정값 오차 (MAE / MSE)", "",
                  "각도는 30°로 나눈 값이다. 회전 ±10° 변형을 함께 넣었다.", "",
                  "| 값 | " + " | ".join(VALUE_NAMES) + " |", "|---|" + "---|" * len(VALUE_NAMES),
                  "| MAE | " + " | ".join(f"{v:.4f}" for v in h["mae"]) + " |",
                  "| MSE | " + " | ".join(f"{v:.4f}" for v in h["mse"]) + " |"]
    if result.get("cross"):
        lines += ["", "## 작가 간 교차 비교", "",
                  "그 작가 사진의 어긋난 변형을 다른 작가 Head에 넣었을 때, 자기 Head와 다른 방향이 나온 비율.", "",
                  "| 작가 | 다른 방향 비율 |", "|---|---|"]
        lines += [f"| {s} | {pct(v)} |" for s, v in result["cross"].items()]
    if history:
        lines += ["", "## 학습 곡선 (MSE)", "", "| 작가 | epoch | train | val |", "|---|---|---|---|"]
        for s, hist in history.items():
            pick = [h for h in hist if h["epoch"] % 50 == 0 or h is hist[0] or h is hist[-1]]
            lines += [f"| {s} | {h['epoch']} | {h['train_mse']:.4f} | {h['val_mse']:.4f} |" for h in pick]
    return "\n".join(lines)


def main(argv: list[str] | None = None, embedder=None) -> dict:
    ap = argparse.ArgumentParser(description="이미지 기반 안내 평가")
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--scorers", default="stat,rank,head")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    bundle = ModelBundle.load(args.model)
    kinds = [k.strip() for k in args.scorers.split(",") if k.strip()]
    if "head" in kinds and embedder is None:
        from ..embed import Embedder

        embedder = Embedder()
    rows = [r for r in load_processed(args.processed) if r["split"] == "test" and r["person"] is not None]
    result = evaluate(rows, args.raw, load_depth_maps(args.processed), bundle, embedder, kinds)
    hist_path = args.model / "head_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.is_file() else {}
    emit(_report(result, len(rows), history), args.out)
    return result


if __name__ == "__main__":
    main()
