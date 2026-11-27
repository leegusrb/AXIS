import json

import numpy as np
import pytest
from axis_ai.bundle import ModelBundle
from axis_ai.features import compute_features
from axis_ai.train import train
from axis_ai.types import Person

STYLES = [{"id": s, "name": f"이름 {s}", "description": f"설명 {s}"} for s in ("s_left", "s_mid", "s_right")]
CX = {"s_left": 0.25, "s_mid": 0.5, "s_right": 0.75}
NAMES = ("cx", "eye_y", "area", "headroom", "face_x", "shot")

# (박스 높이 기준 y 비율, 박스 너비 기준 x 오프셋 비율), COCO 17점
_LAYOUT = [
    (0.10, 0.0), (0.09, -0.08), (0.09, 0.08), (0.10, -0.16), (0.10, 0.16),
    (0.25, -0.30), (0.25, 0.30), (0.40, -0.40), (0.40, 0.40), (0.52, -0.42), (0.52, 0.42),
    (0.55, -0.15), (0.55, 0.15), (0.75, -0.15), (0.75, 0.15), (0.95, -0.15), (0.95, 0.15),
]


def make_person(cx=0.5, top=0.2, w=0.3, h=0.6, conf=0.9) -> Person:
    kp = np.zeros((17, 3), dtype=float)
    for i, (fy, fx) in enumerate(_LAYOUT):
        kp[i] = (cx + fx * w, top + fy * h, 0.9)
    return Person(box=(cx - w / 2, top, w, h), keypoints=kp, conf=conf)


def write_processed(d, n_per_style=20):
    """스타일마다 앞 4장 taste, 다음 4장 test, 나머지 train."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    emb = {}
    with open(d / "features.jsonl", "w", encoding="utf-8") as f:
        for s in CX:
            for i in range(n_per_style):
                split = "taste" if i < 4 else "test" if i < 8 else "train"
                iid = f"{s}_{i}"
                p = make_person(cx=CX[s] + rng.normal(0, 0.02), top=0.2 + rng.normal(0, 0.02),
                                w=0.3 + rng.normal(0, 0.01), h=0.6 + rng.normal(0, 0.02))
                ft = compute_features(p)
                row = {"image_id": iid, "style_id": s, "split": split, "file": f"{s}/{i}.jpg",
                       "author": "A", "license": "CC0", "source_url": "http://x/" + iid,
                       "person": {"box": list(p.box), "keypoints": p.keypoints.tolist(), "conf": p.conf},
                       "features": {k: float(getattr(ft, k)) for k in NAMES}}
                f.write(json.dumps(row) + "\n")
                emb[iid] = rng.normal(size=576).astype(np.float32)
    np.savez(d / "embeddings.npz", **emb)


@pytest.fixture
def bundle(tmp_path) -> ModelBundle:
    write_processed(tmp_path / "proc")
    (tmp_path / "styles.json").write_text(json.dumps(STYLES), encoding="utf-8")
    out = tmp_path / "models" / "v1"
    train(tmp_path / "proc", tmp_path / "styles.json", out)
    return ModelBundle.load(out)
