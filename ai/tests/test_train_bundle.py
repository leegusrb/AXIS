import json

import numpy as np
import pytest

from axis_ai.bundle import ModelBundle
from axis_ai.features import compute_features
from axis_ai.train import main, train
from conftest import make_person

STYLES = [{"id": s, "name": s, "description": ""} for s in ("s_left", "s_mid", "s_right")]
CX = {"s_left": 0.25, "s_mid": 0.5, "s_right": 0.75}
NAMES = ("cx", "eye_y", "area", "headroom", "face_x", "shot")


def _write_processed(d, n_per_style=20, styles=("s_left", "s_mid", "s_right")):
    """스타일마다 앞 4장 taste(첫 장은 인물 없음), 다음 4장 test, 나머지 train."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    emb = {}
    n_taste_person = 0
    with open(d / "features.jsonl", "w", encoding="utf-8") as f:
        for s in styles:
            for i in range(n_per_style):
                split = "taste" if i < 4 else "test" if i < 8 else "train"
                iid = f"{s}_{i}"
                p = make_person(cx=CX[s] + rng.normal(0, 0.02), top=0.2 + rng.normal(0, 0.02),
                                w=0.3 + rng.normal(0, 0.01), h=0.6 + rng.normal(0, 0.02))
                row = {"image_id": iid, "style_id": s, "split": split, "file": f"{s}/{i}.jpg",
                       "author": "A", "license": "CC0", "source_url": "http://x/" + iid,
                       "person": None, "features": None}
                if not (split == "taste" and i == 0):
                    ft = compute_features(p)
                    row["features"] = {k: float(getattr(ft, k)) for k in NAMES}
                    row["person"] = {"box": list(p.box), "keypoints": p.keypoints.tolist(), "conf": p.conf}
                    n_taste_person += split == "taste"
                f.write(json.dumps(row) + "\n")
                emb[iid] = rng.normal(size=576).astype(np.float32)
    np.savez(d / "embeddings.npz", **emb)
    return n_taste_person


def test_main_and_bundle(tmp_path):
    n_taste = _write_processed(tmp_path / "proc")
    (tmp_path / "styles.json").write_text(json.dumps(STYLES))
    out = tmp_path / "models" / "v1"
    main(["--processed", str(tmp_path / "proc"), "--styles", str(tmp_path / "styles.json"), "--out", str(out)])
    b = ModelBundle.load(out)
    assert b.version == "v1" == (out / "version.txt").read_text()
    assert [s["id"] for s in b.styles] == ["s_left", "s_mid", "s_right"]
    assert set(b.scorers["stat"]) == set(CX) and b.scorers["rank"] == {}
    for s, cx in CX.items():
        mine = compute_features(make_person(cx=cx))
        for o, ocx in CX.items():
            if o != s:
                other = compute_features(make_person(cx=ocx))
                assert b.scorers["stat"][s].score(mine) > b.scorers["stat"][s].score(other)
    assert len(b.taste) == n_taste == 9
    assert all(set(t) == {"image_id", "file"} for t in b.taste)
    assert all(abs(np.linalg.norm(c) - 1) < 1e-5 for c in b.centroids.values())
    f, e = b.photo_features("s_mid_10")
    assert e.shape == (576,) and 0.4 < f.cx < 0.6
    with pytest.raises(KeyError):
        b.photo_features("s_left_0")  # 인물 없음
    with pytest.raises(KeyError):
        b.photo_features("nope")
    ph = b.sample_photos("s_mid")
    assert len(ph) == 3 and ph == b.sample_photos("s_mid")
    assert set(ph[0]) == {"image_id", "file", "author", "license", "source_url"}
    assert all(p["image_id"].startswith("s_mid_") and int(p["image_id"].split("_")[-1]) >= 8 for p in ph)


def test_skips_thin_style_and_fails_when_none(tmp_path, capsys):
    _write_processed(tmp_path / "proc", styles=("s_left", "s_mid"))
    (tmp_path / "styles.json").write_text(json.dumps(STYLES))
    train(tmp_path / "proc", tmp_path / "styles.json", tmp_path / "m" / "v2")
    assert "s_right" in capsys.readouterr().out
    b = ModelBundle.load(tmp_path / "m" / "v2")
    assert [s["id"] for s in b.styles] == ["s_left", "s_mid"]
    _write_processed(tmp_path / "thin", n_per_style=9)  # train 1장
    with pytest.raises(SystemExit):
        train(tmp_path / "thin", tmp_path / "styles.json", tmp_path / "m" / "v3")


def test_load_rejects_non_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match="번들"):
        ModelBundle.load(tmp_path)


def test_train_with_rank(tmp_path):
    from axis_ai.guide import decide

    _write_processed(tmp_path / "proc")
    (tmp_path / "styles.json").write_text(json.dumps(STYLES))
    out = tmp_path / "models" / "v2"
    main(["--processed", str(tmp_path / "proc"), "--styles", str(tmp_path / "styles.json"),
          "--out", str(out), "--rank"])
    b = ModelBundle.load(out)
    assert set(b.scorers["rank"]) == set(CX)
    assert all(s.version == "v2" for s in b.scorers["rank"].values())
    assert decide(make_person(cx=0.6), b.scorers["rank"]["s_mid"]).direction


def test_retrain_removes_stale_scorers(tmp_path):
    (tmp_path / "styles.json").write_text(json.dumps(STYLES))
    out = tmp_path / "m" / "v1"
    _write_processed(tmp_path / "proc")
    train(tmp_path / "proc", tmp_path / "styles.json", out, rank=True)
    assert (out / "stat" / "s_right.json").exists() and (out / "rank" / "s_right.pt").exists()
    _write_processed(tmp_path / "proc2", styles=("s_left", "s_mid"))
    train(tmp_path / "proc2", tmp_path / "styles.json", out)  # rank 없이, 스타일 2개로 재학습
    assert sorted(p.name for p in (out / "stat").iterdir()) == ["s_left.json", "s_mid.json"]
    assert not (out / "rank").exists()


def test_load_ignores_scorers_of_unlisted_styles(tmp_path):
    (tmp_path / "styles.json").write_text(json.dumps(STYLES))
    out = tmp_path / "m" / "v1"
    _write_processed(tmp_path / "proc")
    train(tmp_path / "proc", tmp_path / "styles.json", out)
    (out / "styles.json").write_text(json.dumps(STYLES[:2]))  # 번들 스타일 목록에서 s_right 제외
    b = ModelBundle.load(out)
    assert set(b.scorers["stat"]) == {"s_left", "s_mid"}
