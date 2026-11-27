import numpy as np
import pytest
from axis_ai.head.augment import correction, sample_misalignment, warp
from axis_ai.head.model import GuideHead, correction_to_direction, head_decide, head_input
from axis_ai.params import MOVE_STEP
from axis_ai.scorers.rule import RuleScorer
from conftest import make_person


def test_warp_moves_person_and_keypoints_consistently():
    img = np.zeros((200, 150, 3), np.uint8)
    p = make_person(cx=0.5)
    _, q, _ = warp(img, p, None, 0.1, 0.0, 1.0, 0.0)
    assert q.box[0] == pytest.approx(p.box[0] + 0.1, abs=1e-3)
    assert q.keypoints[0, 0] == pytest.approx(p.keypoints[0, 0] + 0.1, abs=1e-3)
    assert q.keypoints[0, 1] == pytest.approx(p.keypoints[0, 1], abs=1e-3)


def test_warp_moves_pixels_with_person():
    img = np.zeros((100, 100, 3), np.uint8)
    img[40:60, 40:60] = 255
    out, _, _ = warp(img, make_person(), None, 0.2, 0.0, 1.0, 0.0)
    assert out[50, 70].max() == 255 and out[50, 45].max() == 0


def test_warp_shrink_uses_padding_and_keeps_size():
    img = np.random.default_rng(0).integers(0, 255, (200, 150, 3), dtype=np.uint8)
    out, q, d = warp(img, make_person(), np.ones((50, 40), np.float32), 0, 0, 0.7, 0)
    assert out.shape == img.shape and d.shape == (50, 40) and q.box[2] < make_person().box[2]
    assert out[0, 0].sum() > 0  # 반사 padding이라 검은 테두리가 아니다


def test_warp_rotation_tilts_shoulders():
    from axis_ai.rotation import estimate_tilt
    _, q, _ = warp(np.zeros((100, 100, 3), np.uint8), make_person(), None, 0, 0, 1.0, 10.0)
    assert estimate_tilt(q) == pytest.approx(10.0, abs=0.5)  # 시계 방향으로 돌면 오른쪽이 내려간다


def test_warp_out_of_frame_is_none():
    _, q, _ = warp(np.zeros((100, 100, 3), np.uint8), make_person(cx=0.5, w=0.2), None, 0.9, 0, 1.0, 0)
    assert q is None


def test_correction_and_direction_semantics():
    # 피사체가 오른쪽으로 0.1 밀린 변형 -> 보정은 왼쪽(-0.1) -> 카메라를 RIGHT로
    assert correction_to_direction(correction(0.1, 0, 1.0, 0))[0] == "RIGHT"
    assert correction_to_direction(correction(-0.1, 0, 1.0, 0))[0] == "LEFT"
    assert correction_to_direction(correction(0, 0.1, 1.0, 0))[0] == "DOWN"
    assert correction_to_direction(correction(0, 0, 0.75, 0))[0] == "CLOSER"  # 너무 작음
    assert correction_to_direction(correction(0, 0, 1.3, 0))[0] == "FARTHER"
    assert correction_to_direction(np.array([0.2 * MOVE_STEP, 0, 0, 0])) == ("OK", "SMALL")
    assert correction_to_direction(np.array([-3 * MOVE_STEP, 0, 0, 0])) == ("RIGHT", "LARGE")


def test_head_input_dim():
    x = head_input(np.zeros(576), make_person(), None)
    assert x.shape == (588,) and x.dtype == np.float32


def test_fit_reduces_mse_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 588)).astype(np.float32)
    W = rng.normal(size=(588, 4)).astype(np.float32) * 0.05
    Y = X @ W
    head, hist = GuideHead.fit("s", X, Y, "v", epochs=200)
    assert hist[-1]["train_mse"] < hist[0]["train_mse"] * 0.5 and "val_mse" in hist[-1]
    assert [h["epoch"] for h in hist][:2] == [10, 20]
    head.save(tmp_path / "s.pt")
    loaded = GuideHead.load(tmp_path / "s.pt")
    assert loaded.style_id == "s" and loaded.version == "v"
    assert np.allclose(loaded.predict(X[0]), head.predict(X[0]), atol=1e-5)


def test_head_decide_uses_stat_score_and_rotation():
    sc = RuleScorer("t", {"cx": 0.5}, {"cx": 0.1})
    p = make_person(cx=0.6)
    d, rot = head_decide(np.array([-0.1, 0, 0, -10 / 30]), p, sc)
    assert d.direction == "RIGHT" and rot == pytest.approx(10.0)
    assert d.score == pytest.approx(sc.score(__import__("axis_ai.features").features.compute_features(p)))
    assert d.target_box[0] == pytest.approx(p.box[0] - 0.1)
    none, rot = head_decide(np.zeros(4), None, sc)
    assert none.direction == "NO_SUBJECT" and rot is None


def test_sample_misalignment_ranges():
    rng = np.random.default_rng(1)
    zero_angle = 0
    for _ in range(300):
        dx, dy, s, a = sample_misalignment(rng)
        assert abs(dx) <= 0.2 and abs(dy) <= 0.2 and 0.7 <= s <= 1.4 and abs(a) <= 15
        zero_angle += a == 0
    assert 100 < zero_angle < 200
