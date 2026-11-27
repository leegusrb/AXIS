"""처리된 데이터로 스타일별 scorer와 임베딩 중심을 학습해 모델 번들을 만든다.

python -m axis_ai.train --processed data/processed --styles data/styles.json --out models/<version> [--rank]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .data.build import load_embeddings, load_processed
from .scorers import StatScorer
from .scorers.rank import RankScorer, make_pairs
from .scorers.stat import MIN_SAMPLES


def _hash_key(image_id: str) -> str:
    return hashlib.sha1(image_id.encode("utf-8")).hexdigest()


def train(processed_dir: Path, styles_path: Path, out_dir: Path, rank: bool = False) -> None:
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    version = out_dir.name
    styles = json.loads(Path(styles_path).read_text(encoding="utf-8"))
    rows = [r for r in load_processed(processed_dir) if r["features"] is not None]
    embeddings = load_embeddings(processed_dir)

    trained, scorers, centroids = [], {}, {}
    for style in styles:
        sid = style["id"]
        train_rows = [r for r in rows if r["style_id"] == sid and r["split"] == "train"]
        if len(train_rows) < MIN_SAMPLES:
            print(f"경고: 스타일 {sid}는 train 사진이 {len(train_rows)}장이라 건너뛰어요 (최소 {MIN_SAMPLES}장)")
            continue
        scorers[sid] = StatScorer.fit(sid, [r["features"] for r in train_rows], version)
        mean = np.mean([embeddings[r["image_id"]] for r in train_rows], axis=0)
        centroids[sid] = (mean / max(np.linalg.norm(mean), 1e-12)).astype(np.float32)
        trained.append(style)
    if not trained:
        raise SystemExit(f"학습할 수 있는 스타일이 없어요: 스타일마다 train 사진(인물 검출)이 {MIN_SAMPLES}장 이상 필요해요")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("stat", "rank"):  # 다시 학습할 때 이전 결과가 섞이지 않게 지운다
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    (out_dir / "stat").mkdir()
    for sid, s in scorers.items():
        s.save(out_dir / "stat" / f"{sid}.json")
    np.savez(out_dir / "centroids.npz", **centroids)
    (out_dir / "styles.json").write_text(json.dumps(trained, ensure_ascii=False, indent=2), encoding="utf-8")
    taste = sorted(
        (r for r in rows if r["split"] == "taste" and r["style_id"] in scorers),
        key=lambda r: _hash_key(r["image_id"]),
    )
    (out_dir / "taste.json").write_text(
        json.dumps([{"image_id": r["image_id"], "file": r["file"]} for r in taste], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(processed_dir / "features.jsonl", out_dir / "photos.jsonl")
    shutil.copyfile(processed_dir / "embeddings.npz", out_dir / "embeddings.npz")
    if rank:
        (out_dir / "rank").mkdir()
        for sid in scorers:
            own = [r["person"] for r in rows if r["style_id"] == sid and r["split"] == "train"]
            others = [r["person"] for r in rows
                      if r["style_id"] != sid and r["style_id"] in scorers and r["split"] == "train"]
            RankScorer.fit(sid, make_pairs(own, others), version).save(out_dir / "rank" / f"{sid}.pt")
    (out_dir / "version.txt").write_text(version, encoding="utf-8")
    print(f"스타일 {len(trained)}개 학습 -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="스타일별 scorer를 학습해 모델 번들을 만든다")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--styles", type=Path, default=Path("data/styles.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", action="store_true")
    args = ap.parse_args(argv)
    train(args.processed, args.styles, args.out, rank=args.rank)


if __name__ == "__main__":
    main()
