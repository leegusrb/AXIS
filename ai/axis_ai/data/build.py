"""manifest 검증 → 분할 → 포즈·임베딩 추출 → data/processed 저장.

python -m axis_ai.data.build --manifest data/manifest.csv --raw data/raw --out data/processed
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..embed import Embedder
from ..features import compute_features, select_subject
from ..pose import PoseDetector
from ..types import FEATURE_NAMES, CompositionFeatures, Person
from .manifest import PhotoRecord, assign_splits, load_manifest, open_image, split_warnings, validate


def _person_to_json(p: Person) -> dict:
    return {
        "box": [float(v) for v in p.box],
        "keypoints": [[float(v) for v in row] for row in p.keypoints],
        "conf": float(p.conf),
    }


def build(
    records: list[PhotoRecord],
    raw_dir: Path,
    out_dir: Path,
    detector,
    embedder,
    problems: list[str] = (),
) -> None:
    """features.jsonl, embeddings.npz, report.txt를 out_dir에 쓴다. 인물이 없어도 행과 임베딩은 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    no_person: list[str] = []
    with open(out_dir / "features.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            img = open_image(Path(raw_dir) / r.file)
            subject = select_subject(detector.detect(img))
            embeddings[r.image_id] = embedder.embed(img).astype(np.float32)
            if subject is None:
                no_person.append(r.image_id)
            feats = compute_features(subject) if subject else None
            row = {
                "image_id": r.image_id,
                "style_id": r.style_id,
                "split": r.split,
                "file": r.file,
                "author": r.author,
                "license": r.license,
                "source_url": r.source_url,
                "person": _person_to_json(subject) if subject else None,
                "features": {n: float(getattr(feats, n)) for n in FEATURE_NAMES} if feats else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez(out_dir / "embeddings.npz", **embeddings)
    lines = ["# 검증 문제", *problems, "", "# 인물 미검출", *no_person]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_processed(out_dir: Path) -> list[dict]:
    """features.jsonl을 읽어 person은 Person, features는 CompositionFeatures로 복원한다(없으면 None)."""
    rows = []
    with open(Path(out_dir) / "features.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["person"] is not None:
                p = row["person"]
                row["person"] = Person(
                    box=tuple(p["box"]), keypoints=np.array(p["keypoints"], dtype=float), conf=p["conf"]
                )
            if row["features"] is not None:
                row["features"] = CompositionFeatures(**row["features"])
            rows.append(row)
    return rows


def load_embeddings(out_dir: Path) -> dict[str, np.ndarray]:
    with np.load(Path(out_dir) / "embeddings.npz") as z:
        return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="manifest를 검증하고 특징을 만든다")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args(argv)

    ok, problems = validate(load_manifest(args.manifest), args.raw)
    records = assign_splits(ok)
    problems += split_warnings(records)
    build(records, args.raw, args.out, PoseDetector(), Embedder(), problems)
    for p in problems:
        print(p)
    print(f"{len(records)}장 처리, 문제/경고 {len(problems)}건 -> {args.out}")


if __name__ == "__main__":
    main()
