import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from axis_ai.bundle import ModelBundle
from axis_ai.eval import OPPOSITE, latency, real_moves, recommend_consistency, separability, synthetic_guide, user_test
from axis_ai.features import compute_features
from axis_ai.train import train
from conftest import make_person

STYLES = [{"id": s, "name": s, "description": ""} for s in ("s_left", "s_mid", "s_right")]
CX = {"s_left": 0.25, "s_mid": 0.5, "s_right": 0.75}
NAMES = ("cx", "eye_y", "area", "headroom", "face_x", "shot")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """스타일마다 앞 4장 taste, 다음 4장 test, 나머지 train(총 24장). 모든 사진에 인물이 있다."""
    root = tmp_path_factory.mktemp("eval")
    proc = root / "proc"
    proc.mkdir()
    rng = np.random.default_rng(0)
    emb = {}
    with open(proc / "features.jsonl", "w", encoding="utf-8") as f:
        for s in CX:
            for i in range(24):
                split = "taste" if i < 4 else "test" if i < 8 else "train"
                iid = f"{s}_{i}"
                p = make_person(cx=CX[s] + rng.normal(0, 0.02), top=0.2 + rng.normal(0, 0.02),
                                w=0.3 + rng.normal(0, 0.01), h=0.6 + rng.normal(0, 0.02))
                ft = compute_features(p)
                row = {"image_id": iid, "style_id": s, "split": split, "file": f"{s}/{i}.jpg",
                       "author": "A", "license": "CC0", "source_url": "http://x/" + iid,
                       "features": {k: float(getattr(ft, k)) for k in NAMES},
                       "person": {"box": list(p.box), "keypoints": p.keypoints.tolist(), "conf": p.conf}}
                f.write(json.dumps(row) + "\n")
                emb[iid] = rng.normal(size=576).astype(np.float32)
    np.savez(proc / "embeddings.npz", **emb)
    (root / "styles.json").write_text(json.dumps(STYLES))
    train(proc, root / "styles.json", root / "models" / "v1")
    return proc, root / "models" / "v1"


def test_separability(data, tmp_path):
    proc, _ = data
    out = tmp_path / "sub" / "sep.md"
    r = separability.main(["--processed", str(proc), "--out", str(out)])
    assert r["composition"] > 0.9
    assert r["chance"] == pytest.approx(1 / 3)
    assert 0.0 <= r["mobilenet"] <= 1.0 and 0.0 <= r["both"] <= 1.0
    assert "구도" in out.read_text(encoding="utf-8")


def test_synthetic_guide_stat(data, tmp_path):
    proc, model = data
    r = synthetic_guide.main(["--processed", str(proc), "--model", str(model), "--scorers", "stat,rank",
                              "--out", str(tmp_path / "g.md")])
    assert set(r) == {"stat"}  # 번들에 rank가 없다
    assert r["stat"]["accuracy"] > 0.7
    assert set(r["stat"]["per_direction"]) == set(OPPOSITE)
    assert 0.0 <= r["stat"]["ok_share"] <= 1.0


def test_synthetic_guide_reports_recovery(data, tmp_path):
    proc, model = data
    r = synthetic_guide.main(["--processed", str(proc), "--model", str(model), "--scorers", "stat",
                              "--out", str(tmp_path / "g.md")])
    s = r["stat"]
    assert 0.0 <= s["recovery"] <= 1.0
    assert s["recovery"] > 0.8  # 안내를 따라가면 대부분 OK에 닿는다
    assert 0.0 < s["mean_steps"] <= synthetic_guide.MAX_STEPS
    assert set(s["recovery_per_direction"]) == set(OPPOSITE)
    assert "OK 도달" in (tmp_path / "g.md").read_text(encoding="utf-8")


def test_follow_guidance_counts_steps_and_gives_up():
    from axis_ai.guide import MOVES
    from axis_ai.scorers.rule import RuleScorer

    sc = RuleScorer("t", target={"cx": 0.5}, spread={"cx": 0.1})
    assert synthetic_guide.follow_guidance(make_person(cx=0.5), sc) == 0
    far = make_person(cx=0.5 - 2 * abs(MOVES["LEFT"][0]))
    assert synthetic_guide.follow_guidance(far, sc) == 2
    never = RuleScorer("t", target={"cx": 0.5}, spread={"cx": 1e-6})  # 사실상 OK에 닿을 수 없음
    assert synthetic_guide.follow_guidance(make_person(cx=0.43), never, max_steps=3) is None


def test_perturb_direction_semantics():
    p = make_person(cx=0.5)
    cx = lambda q: q.box[0] + q.box[2] / 2  # noqa: E731
    assert cx(synthetic_guide.perturb(p, "RIGHT", 0.1)) == pytest.approx(0.6)  # 오른쪽으로 옮긴 변형 -> 정답 RIGHT
    assert cx(synthetic_guide.perturb(p, "LEFT", 0.1)) == pytest.approx(0.4)
    assert synthetic_guide.perturb(p, "UP", 0.1).box[1] < p.box[1]  # 인물이 위로 치우침 -> 카메라를 UP으로 올려야 함
    assert synthetic_guide.perturb(p, "CLOSER", 0.2).box[2] < p.box[2]
    assert synthetic_guide.perturb(p, "FARTHER", 0.2).box[2] > p.box[2]


def test_recommend_consistency(data, tmp_path):
    _, model = data
    r = recommend_consistency.main(["--model", str(model), "--out", str(tmp_path / "r.md")])
    assert r["consistency"] > 0.8
    assert r["n"] == 3 * 15
    assert all(v["n"] == 15 for v in r["per_style"].values())


def _db(path: Path, ddl: str, rows, sql: str):
    con = sqlite3.connect(path)
    con.execute(ddl)
    con.executemany(sql, rows)
    con.commit()
    con.close()


def test_latency(tmp_path):
    db = tmp_path / "a.db"
    _db(db, "CREATE TABLE inference_logs (id INTEGER, decode_ms INT, pose_ms INT, guide_ms INT, total_ms INT)",
        [(1, 1, 10, 2, 20), (2, 2, 20, 4, 40), (3, 3, 30, 6, 60)],
        "INSERT INTO inference_logs VALUES (?,?,?,?,?)")
    r = latency.main(["--db", str(db), "--out", str(tmp_path / "l.md")])
    assert r["count"] == 3
    assert r["pose_ms"]["p50"] == 20 and r["total_ms"]["p50"] == 40
    assert r["total_ms"]["p95"] == pytest.approx(58)


def test_latency_empty(tmp_path):
    db = tmp_path / "e.db"
    _db(db, "CREATE TABLE inference_logs (id INTEGER, decode_ms INT, pose_ms INT, guide_ms INT, total_ms INT)",
        [], "INSERT INTO inference_logs VALUES (?,?,?,?,?)")
    assert latency.main(["--db", str(db)]) == {"count": 0}


def test_user_test(tmp_path):
    db = tmp_path / "s.db"
    rows = [("1", "d1", 0.4, 0, "2026-11-20 10:00:00"), ("2", "d1", 0.6, 1, "2026-11-20 10:05:00"),
            ("3", "d1", 0.5, 0, "2026-11-20 10:10:00"), ("4", "d1", 0.4, 1, "2026-11-20 10:15:00"),
            ("5", "d1", 0.3, 1, "2026-11-20 10:16:00"),  # 앞 켬 사진은 이미 쓰였으니 짝 없음
            ("6", "d2", 0.9, 1, "2026-11-21 10:00:00"),  # 앞에 끔이 없는 켬 사진은 제외
            ("7", "d2", 0.2, 0, "2026-11-21 10:05:00"), ("8", "d2", 0.5, 0, "2026-11-21 10:06:00"),
            ("9", "d2", 0.8, 1, "2026-11-21 10:07:00")]  # 끔이 연달아면 가장 나중 것과 짝
    _db(db, "CREATE TABLE shots (id TEXT, device_id TEXT, score REAL, guide_used BOOLEAN, created_at TEXT)",
        rows, "INSERT INTO shots VALUES (?,?,?,?,?)")
    r = user_test.main(["--db", str(db)])
    assert r["n_pairs"] == 3
    assert r["mean_delta"] == pytest.approx((0.2 - 0.1 + 0.3) / 3)
    assert r["improved_share"] == pytest.approx(2 / 3)
    r = user_test.main(["--db", str(db), "--since", "2026-11-21"])
    assert r["n_pairs"] == 1 and r["mean_delta"] == pytest.approx(0.3)
    assert user_test.main(["--db", str(db), "--since", "2027-01-01"])["n_pairs"] == 0


class _FakeDetector:
    """이미지 왼쪽 위 픽셀의 빨강값으로 인물 중심 x를 정한다(0이면 인물 없음)."""

    def detect(self, img):
        r = img.getpixel((0, 0))[0]
        return [] if r == 0 else [make_person(cx=r / 100)]


def test_parse_name():
    assert real_moves.parse_name(Path("s_mid_p1_LEFT.jpg")) == ("s_mid", "p1", "LEFT")
    assert real_moves.parse_name(Path("wide_full_p2_base.jpg")) == ("wide_full", "p2", "base")
    assert real_moves.parse_name(Path("bad_name.jpg")) is None
    assert real_moves.parse_name(Path("s_p1_SIDEWAYS.jpg")) is None


def test_real_moves(data, tmp_path):
    _, model = data
    d = tmp_path / "moves"
    d.mkdir()
    for name, red in [("s_mid_p1_base", 50), ("s_mid_p1_LEFT", 70), ("s_mid_p1_RIGHT", 30),
                      ("s_mid_p1_UP", 0), ("other_p1_LEFT", 50)]:
        Image.new("RGB", (8, 8), (red, 0, 0)).save(d / f"{name}.jpg")
    Image.new("RGB", (8, 8), (70, 0, 0)).save(d / "s_mid_p2_LEFT.JPEG")  # 대문자 확장자도 읽는다
    (d / "s_mid_p3_LEFT.HEIC").write_bytes(b"x")
    bundle = ModelBundle.load(model)
    r = real_moves.run(d, bundle, _FakeDetector(), ["stat"])
    assert r["skipped_non_jpeg"] == ["s_mid_p3_LEFT.HEIC"]
    # LEFT 이동 -> 인물이 오른쪽(0.7) -> 정답 RIGHT, RIGHT 이동 -> 인물이 왼쪽(0.3) -> 정답 LEFT
    assert r["scorers"]["stat"]["n"] == 3 and r["scorers"]["stat"]["accuracy"] == 1.0
    assert r["no_person"] == ["s_mid_p1_UP.jpg"]
    assert r["no_scorer"] == ["other_p1_LEFT.jpg"]
    r = real_moves.run(d, bundle, _FakeDetector(), ["stat"], style_filter="s_left")
    assert r["scorers"]["stat"]["n"] == 0 and r["scorers"]["stat"]["accuracy"] is None


def test_user_test_groups_by_user_when_present(tmp_path):
    db = tmp_path / "u.db"
    rows = [("1", None, 1, 0.4, 0, "2026-11-28 10:00:00"), ("2", None, 2, 0.9, 1, "2026-11-28 10:01:00"),
            ("3", None, 1, 0.7, 1, "2026-11-28 10:02:00")]  # 사용자 2는 끔 사진이 없어 짝 없음
    _db(db, "CREATE TABLE shots (id TEXT, device_id TEXT, user_id INTEGER, score REAL, guide_used BOOLEAN,"
            " created_at TEXT)", rows, "INSERT INTO shots VALUES (?,?,?,?,?,?)")
    r = user_test.main(["--db", str(db)])
    assert r["n_pairs"] == 1 and r["mean_delta"] == pytest.approx(0.3)


def test_latency_includes_depth_and_embed_when_present(tmp_path):
    db = tmp_path / "d.db"
    _db(db, "CREATE TABLE inference_logs (id INTEGER, decode_ms INT, pose_ms INT, guide_ms INT,"
            " depth_ms INT, embed_ms INT, total_ms INT, scorer TEXT)",
        [(1, 1, 10, 2, 50, 0, 70, "stat"), (2, 1, 10, 2, 0, 20, 40, "head")],
        "INSERT INTO inference_logs VALUES (?,?,?,?,?,?,?,?)")
    r = latency.main(["--db", str(db), "--out", str(tmp_path / "l.md")])
    assert r["depth_ms"]["p50"] == 25 and r["embed_ms"]["p50"] == 10
    assert r["by_scorer"]["head"]["total_ms"]["p50"] == 40


@pytest.fixture(scope="module")
def head_data(tmp_path_factory):
    from axis_ai.head.train import train_heads
    from conftest import FakeEmbedder, make_head_data
    proc, raw, model = make_head_data(tmp_path_factory.mktemp("imgguide"))
    train_heads(proc, raw, model, FakeEmbedder(), aug=4, epochs=20)
    return proc, raw, model


def test_image_guide_reports_all(head_data, tmp_path):
    from axis_ai.eval import image_guide
    from conftest import FakeEmbedder
    proc, raw, model = head_data
    r = image_guide.main(["--processed", str(proc), "--raw", str(raw), "--model", str(model),
                          "--scorers", "stat,head", "--out", str(tmp_path / "g.md")], embedder=FakeEmbedder())
    assert set(r["stat"]) >= {"accuracy", "ok_share", "recovery", "mean_steps", "per_direction", "n"}
    assert r["stat"]["n"] > 0 and r["stat"]["recovery"] > 0.5
    assert len(r["head"]["mae"]) == 4 and len(r["head"]["mse"]) == 4
    assert all(v >= 0 for v in r["head"]["mae"])
    assert set(r["cross"]) == {"s_left", "s_mid", "s_right"}
    assert all(0.0 <= v <= 1.0 for v in r["cross"].values())
    text = (tmp_path / "g.md").read_text(encoding="utf-8")
    assert "MAE" in text and "교차" in text and "학습 곡선" in text


def test_image_perturb_matches_synthetic_semantics():
    from axis_ai.eval.image_guide import variant_params
    assert variant_params("RIGHT", 0.1) == (0.1, 0.0, 1.0, 0.0)  # 인물이 오른쪽으로 밀림 -> 정답 RIGHT
    assert variant_params("UP", 0.1) == (0.0, -0.1, 1.0, 0.0)
    assert variant_params("CLOSER", 0.25) == (0.0, 0.0, pytest.approx(0.8), 0.0)
    assert variant_params("FARTHER", 0.2) == (0.0, 0.0, pytest.approx(1.2), 0.0)


def test_time_fn_shape():
    from axis_ai.eval import speed
    r = speed.time_fn(lambda: sum(range(1000)), n=5, warmup=1)
    assert r["mean_ms"] >= 0 and r["p95_ms"] >= 0 and set(r) == {"mean_ms", "p95_ms"}


def test_head_onnx_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    from axis_ai.eval import speed
    from axis_ai.head.model import GuideHead
    X = np.random.default_rng(0).normal(size=(50, 588)).astype(np.float32)
    head, _ = GuideHead.fit("s", X, X[:, :4], "v", epochs=5)
    r = speed.compare_onnx(head, speed.export_head_onnx(head, tmp_path / "h.onnx"), X)
    assert r["max_abs_err"] < 1e-4 and r["torch_ms"] >= 0 and r["onnx_ms"] >= 0
