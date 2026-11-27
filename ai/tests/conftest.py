from axis_ai.types import CompositionFeatures, Person
import numpy as np

# COCO 순서: 0 코, 1-2 눈, 3-4 귀, 5-6 어깨, 7-8 팔꿈치, 9-10 손목, 11-12 엉덩이, 13-14 무릎, 15-16 발목
_VISIBLE = {
    "all": set(range(17)),
    "upper": set(range(11)),
    "face": set(range(5)),
    "no_eyes": set(range(17)) - {1, 2},
    "no_face": set(range(17)) - {0, 1, 2},
}

# (박스 높이 기준 y 비율, 박스 너비 기준 x 오프셋 비율)
_LAYOUT = [
    (0.10, 0.0), (0.09, -0.08), (0.09, 0.08), (0.10, -0.16), (0.10, 0.16),
    (0.25, -0.30), (0.25, 0.30), (0.40, -0.40), (0.40, 0.40), (0.52, -0.42), (0.52, 0.42),
    (0.55, -0.15), (0.55, 0.15), (0.75, -0.15), (0.75, 0.15), (0.95, -0.15), (0.95, 0.15),
]


def make_person(cx=0.5, top=0.2, w=0.3, h=0.6, conf=0.9, visible="all") -> Person:
    vis = _VISIBLE[visible]
    kp = np.zeros((17, 3), dtype=float)
    for i, (fy, fx) in enumerate(_LAYOUT):
        kp[i, 0] = cx + fx * w
        kp[i, 1] = top + fy * h
        kp[i, 2] = 0.9 if i in vis else 0.0
    return Person(box=(cx - w / 2, top, w, h), keypoints=kp, conf=conf)


def feat(**overrides) -> CompositionFeatures:
    v = dict(cx=0.5, eye_y=0.3, area=0.2, headroom=0.15, shot=0.0)
    v.update(overrides)
    v.setdefault("face_x", v["cx"])
    return CompositionFeatures(**v)


class FakeEmbedder:
    """이미지 평균색으로 만든 576차원 벡터(빠른 테스트용)."""

    def embed(self, img):
        m = np.asarray(img.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255
        return np.tile(m, 192).astype(np.float32)


def make_head_data(root, styles=("s_left", "s_mid", "s_right"), n=12):
    """스타일마다 앞 2장 taste, 다음 3장 test, 나머지 train. 이미지·깊이 지도까지 만든 처리 폴더와 학습된 번들."""
    import json

    from PIL import Image

    from axis_ai.features import compute_features
    from axis_ai.train import train

    cx = {"s_left": 0.3, "s_mid": 0.5, "s_right": 0.7}
    proc, raw = root / "proc", root / "raw"
    proc.mkdir(parents=True), raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    emb, depth = {}, {}
    with open(proc / "features.jsonl", "w", encoding="utf-8") as f:
        for s in styles:
            for i in range(n):
                split = "taste" if i < 2 else "test" if i < 5 else "train"
                iid = f"{s}_{i}"
                p = make_person(cx=cx[s] + rng.normal(0, 0.02), top=0.2, w=0.25, h=0.6)
                img = np.full((80, 60, 3), 30, np.uint8)
                x, y, w, h = p.box
                img[int(y * 80):int((y + h) * 80), int(x * 60):int((x + w) * 60)] = 220
                Image.fromarray(img).save(raw / f"{iid}.jpg")
                ft = compute_features(p)
                row = {"image_id": iid, "style_id": s, "split": split, "file": f"{iid}.jpg",
                       "author": "A", "license": "CC0", "source_url": "http://x/" + iid,
                       "person": {"box": list(p.box), "keypoints": p.keypoints.tolist(), "conf": p.conf},
                       "features": {k: float(getattr(ft, k)) for k in
                                    ("cx", "eye_y", "area", "headroom", "face_x", "shot")}}
                f.write(json.dumps(row) + "\n")
                emb[iid] = FakeEmbedder().embed(Image.fromarray(img))
                depth[iid] = (img[:, :, 0] / 255).astype(np.float16)
    np.savez(proc / "embeddings.npz", **emb)
    np.savez(proc / "depth.npz", **depth)
    (root / "styles.json").write_text(json.dumps([{"id": s, "name": s, "description": ""} for s in styles]))
    model = root / "models" / "v1"
    train(proc, root / "styles.json", model)
    return proc, raw, model
