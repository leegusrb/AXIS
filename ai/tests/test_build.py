import json

import numpy as np
from PIL import Image

from axis_ai.data.build import build, load_embeddings, load_processed, main
from axis_ai.data.manifest import PhotoRecord
from axis_ai.features import compute_features
from conftest import make_person


class FakeDetector:
    def __init__(self, persons):
        self.persons = persons

    def detect(self, img):
        return self.persons


class FakeEmbedder:
    def embed(self, img):
        v = np.zeros(576, dtype=np.float32)
        v[0] = 1.0
        return v


def _setup(tmp_path, ids=("a", "b")):
    raw = tmp_path / "raw"
    (raw / "s1").mkdir(parents=True)
    recs = []
    for i, n in enumerate(ids):
        Image.new("RGB", (500, 500), (i, i, i)).save(raw / "s1" / f"{n}.jpg")
        recs.append(PhotoRecord(n, "s1", f"s1/{n}.jpg", "http://u/" + n, "Jane", "CC0", "test"))
    return raw, recs


def test_build_writes_null_features_when_no_person(tmp_path):
    raw, recs = _setup(tmp_path)
    out = tmp_path / "out"
    build(recs, raw, out, FakeDetector([]), FakeEmbedder())
    rows = [json.loads(line) for line in (out / "features.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["person"] is None and r["features"] is None for r in rows)
    assert set(np.load(out / "embeddings.npz").files) == {"a", "b"}
    assert "a" in (out / "report.txt").read_text()


def test_build_roundtrip(tmp_path):
    raw, recs = _setup(tmp_path, ids=("a",))
    out = tmp_path / "out"
    p = make_person(cx=0.4, top=0.1, w=0.3, h=0.7, conf=0.8)
    small = make_person(conf=0.9, w=0.05, h=0.1)
    build(recs, raw, out, FakeDetector([small, p]), FakeEmbedder())
    row = json.loads((out / "features.jsonl").read_text())
    assert row["file"] == "s1/a.jpg" and row["author"] == "Jane"
    assert row["license"] == "CC0" and row["source_url"] == "http://u/a" and row["split"] == "test"
    assert len(row["person"]["box"]) == 4 and len(row["person"]["keypoints"]) == 17
    assert set(row["features"]) == {"cx", "eye_y", "area", "headroom", "face_x", "shot"}
    (loaded,) = load_processed(out)
    got = loaded["person"]
    assert np.allclose(got.box, p.box) and np.allclose(got.keypoints, p.keypoints)
    assert got.conf == p.conf
    assert loaded["features"] == compute_features(p)
    assert loaded["image_id"] == "a" and loaded["style_id"] == "s1" and loaded["split"] == "test"
    emb = load_embeddings(out)
    assert emb["a"].shape == (576,) and emb["a"].dtype == np.float32


def test_load_processed_null_person(tmp_path):
    raw, recs = _setup(tmp_path, ids=("a",))
    build(recs, raw, tmp_path / "out", FakeDetector([]), FakeEmbedder())
    (row,) = load_processed(tmp_path / "out")
    assert row["person"] is None and row["features"] is None


def test_main_validates_and_splits(tmp_path, monkeypatch):
    raw, _ = _setup(tmp_path)
    m = tmp_path / "m.csv"
    m.write_text("image_id,style_id,file,source_url,author,license\n"
                 "a,s1,s1/a.jpg,u,Jane,CC0\nb,s1,s1/b.jpg,u,Jane,All rights reserved\n")
    monkeypatch.setattr("axis_ai.data.build.PoseDetector", lambda: FakeDetector([make_person()]))
    monkeypatch.setattr("axis_ai.data.build.Embedder", lambda: FakeEmbedder())
    out = tmp_path / "out"
    main(["--manifest", str(m), "--raw", str(raw), "--out", str(out)])
    rows = load_processed(out)
    assert [r["image_id"] for r in rows] == ["a"]
    report = (out / "report.txt").read_text()
    assert "b" in report
    assert "train 사진이 0장" in report
