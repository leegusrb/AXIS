"""학습 결과 폴더(모델 번들)를 읽는다. 서버가 쓴다."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .data.build import load_embeddings
from .scorers import load_scorers
from .types import CompositionFeatures

_PHOTO_KEYS = ("image_id", "file", "author", "license", "source_url")


def _hash_key(image_id: str) -> str:
    return hashlib.sha1(image_id.encode("utf-8")).hexdigest()


class ModelBundle:
    def __init__(self, version, styles, scorers, taste, centroids, rows, embeddings, heads=None):
        self.version: str = version
        self.styles: list[dict] = styles
        self.scorers: dict[str, dict] = scorers
        self.taste: list[dict] = taste
        self.centroids: dict[str, np.ndarray] = centroids
        self._rows = {r["image_id"]: r for r in rows}
        self._embeddings = embeddings
        self.heads: dict = heads or {}  # style_id -> GuideHead (head/ 폴더가 없으면 비어 있음)

    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        d = Path(model_dir)
        if not (d / "version.txt").is_file():
            raise FileNotFoundError(f"모델 번들이 아니에요(version.txt 없음): {d}")

        def read(name: str):
            return json.loads((d / name).read_text(encoding="utf-8"))

        with np.load(d / "centroids.npz") as z:
            centroids = {k: z[k] for k in z.files}
        rows = _load_photos(d)
        styles = read("styles.json")
        ids = {s["id"] for s in styles}  # 이전 학습이 남긴 다른 스타일의 scorer 파일은 무시한다
        return cls(
            version=(d / "version.txt").read_text(encoding="utf-8").strip(),
            styles=styles,
            scorers={k: {sid: sc for sid, sc in load_scorers(d, k).items() if sid in ids}
                     for k in ("stat", "rank")},
            taste=read("taste.json"),
            centroids=centroids,
            rows=rows,
            embeddings=load_embeddings(d),
            heads=_load_heads(d, ids),
        )

    @property
    def rows(self) -> list[dict]:
        return list(self._rows.values())

    def photo_features(self, image_id: str) -> tuple[CompositionFeatures, np.ndarray]:
        row = self._rows.get(image_id)
        if row is None or row["features"] is None or image_id not in self._embeddings:
            raise KeyError(image_id)
        return row["features"], self._embeddings[image_id]

    def sample_photos(self, style_id: str, n: int = 3) -> list[dict]:
        rows = [r for r in self._rows.values()
                if r["style_id"] == style_id and r["split"] == "train" and r["features"] is not None]
        rows.sort(key=lambda r: _hash_key(r["image_id"]))
        return [{k: r[k] for k in _PHOTO_KEYS} for r in rows[:n]]


def _load_photos(d: Path) -> list[dict]:
    rows = []
    with open(d / "photos.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["features"] is not None:
                row["features"] = CompositionFeatures(**row["features"])
            rows.append(row)
    return rows


def _load_heads(d: Path, ids: set[str]) -> dict:
    folder = d / "head"
    if not folder.is_dir():
        return {}
    from .head.model import GuideHead

    heads = (GuideHead.load(p) for p in sorted(folder.glob("*.pt")))
    return {h.style_id: h for h in heads if h.style_id in ids}
