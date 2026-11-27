"""작가별 Guide Head 학습: train 사진을 이미지 단위로 어긋나게 만들고 되돌리는 보정값을 MSE로 학습한다.

python -m axis_ai.head.train --processed data/processed --raw data/raw --model models/<v> [--aug 20] [--epochs 300]
(먼저 python -m axis_ai.data.depth_maps 로 depth.npz를 만든다. 없으면 깊이 특징은 0)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.build import load_processed
from ..data.depth_maps import load_depth_maps
from ..data.manifest import open_image
from ..depth import depth_features
from .augment import correction, sample_misalignment, warp
from .model import GuideHead, head_input


def make_samples(rows: list[dict], raw_dir: Path, depth_maps: dict, embedder, aug: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """사진마다 원본 1개(보정 0)와 변형 aug개. (X, Y)."""
    X, Y = [], []
    for r in rows:
        img = np.asarray(open_image(Path(raw_dir) / r["file"]).convert("RGB"))
        dmap = depth_maps.get(r["image_id"])
        dmap = None if dmap is None else dmap.astype(np.float32)
        params = [(0.0, 0.0, 1.0, 0.0)] + [sample_misalignment(rng) for _ in range(aug)]
        for dx, dy, s, a in params:
            out, person, dout = warp(img, r["person"], dmap, dx, dy, s, a)
            if person is None:
                continue
            emb = embedder.embed(Image.fromarray(out))
            dfeat = depth_features(dout, person.box) if dout is not None else None
            X.append(head_input(emb, person, dfeat))
            Y.append(correction(dx, dy, s, a))
    return np.stack(X), np.stack(Y)


def train_heads(processed_dir: Path, raw_dir: Path, model_dir: Path, embedder, aug: int = 20,
                epochs: int = 300, seed: int = 0) -> dict[str, list[dict]]:
    model_dir = Path(model_dir)
    version = (model_dir / "version.txt").read_text(encoding="utf-8").strip()
    styles = [s["id"] for s in json.loads((model_dir / "styles.json").read_text(encoding="utf-8"))]
    rows = [r for r in load_processed(processed_dir) if r["person"] is not None and r["split"] == "train"]
    depth_maps = load_depth_maps(processed_dir)
    rng = np.random.default_rng(seed)
    histories = {}
    for sid in styles:
        X, Y = make_samples([r for r in rows if r["style_id"] == sid], raw_dir, depth_maps, embedder, aug, rng)
        head, hist = GuideHead.fit(sid, X, Y, version, epochs=epochs, seed=seed)
        head.save(model_dir / "head" / f"{sid}.pt")
        histories[sid] = hist
        print(f"{sid}: 샘플 {len(X)}개, val_mse {hist[0]['val_mse']:.4f} -> {hist[-1]['val_mse']:.4f}")
    (model_dir / "head_history.json").write_text(json.dumps(histories, indent=1), encoding="utf-8")
    return histories


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="작가별 Guide Head 학습")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--aug", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args(argv)
    from ..embed import Embedder

    train_heads(args.processed, args.raw, args.model, Embedder(), aug=args.aug, epochs=args.epochs)


if __name__ == "__main__":
    main()
