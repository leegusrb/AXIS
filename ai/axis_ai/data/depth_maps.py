"""학습 사진마다 깊이 지도를 만들어 processed/depth.npz에 둔다(Guide Head 학습용).

python -m axis_ai.data.depth_maps --processed data/processed --raw data/raw
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .manifest import open_image


def load_depth_maps(processed_dir: Path) -> dict[str, np.ndarray]:
    p = Path(processed_dir) / "depth.npz"
    if not p.is_file():
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def make_depth_maps(processed_dir: Path, raw_dir: Path, estimator) -> int:
    """아직 없는 사진만 추정한다. 새로 만든 개수를 돌려준다."""
    processed_dir = Path(processed_dir)
    maps = load_depth_maps(processed_dir)
    new = 0
    with open(processed_dir / "features.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        if r["image_id"] in maps:
            continue
        maps[r["image_id"]] = estimator.estimate(open_image(Path(raw_dir) / r["file"])).astype(np.float16)
        new += 1
    if new:
        np.savez(processed_dir / "depth.npz", **maps)
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="학습 사진 깊이 지도 만들기")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args(argv)
    from ..depth import DepthEstimator

    print(f"새 깊이 지도 {make_depth_maps(args.processed, args.raw, DepthEstimator())}장 -> {args.processed / 'depth.npz'}")


if __name__ == "__main__":
    main()
