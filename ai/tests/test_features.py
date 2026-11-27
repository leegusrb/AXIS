import numpy as np
import pytest

from axis_ai.features import compute_features, select_subject
from axis_ai.geometry import transform_person
from axis_ai.types import FEATURE_NAMES
from conftest import make_person


def test_select_largest_times_conf():
    big = make_person(w=0.4, h=0.7, conf=0.6)
    small = make_person(w=0.1, h=0.2, conf=0.99)
    assert select_subject([small, big]) is big


def test_select_ignores_low_conf_and_empty():
    assert select_subject([make_person(conf=0.3)]) is None
    assert select_subject([]) is None


def test_features_basic():
    f = compute_features(make_person(cx=0.3, top=0.2, w=0.2, h=0.5, visible="all"))
    assert f.cx == pytest.approx(0.3)
    assert f.headroom == pytest.approx(0.2)
    assert f.area == pytest.approx(0.1)
    assert f.shot == 1.0
    assert f.eye_y == pytest.approx(0.2 + 0.09 * 0.5)
    assert f.face_x == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize("visible,shot", [("all", 1.0), ("upper", 0.0), ("face", 0.0)])
def test_shot_type(visible, shot):
    assert compute_features(make_person(visible=visible)).shot == shot


def test_shot_knees_only():
    p = make_person()
    p.keypoints[15:17, 2] = 0.0
    assert compute_features(p).shot == 0.5


def test_eye_fallback_to_nose_then_box():
    f = compute_features(make_person(top=0.2, h=0.5, visible="no_eyes"))
    assert f.eye_y == pytest.approx(0.2 + 0.10 * 0.5)  # 코 y
    f = compute_features(make_person(top=0.2, h=0.5, visible="no_face"))
    assert f.eye_y == pytest.approx(0.25)
    assert f.face_x == pytest.approx(0.5)  # 얼굴 키포인트 없음 -> cx


def test_features_finite_for_clipped_person():
    q = transform_person(make_person(cx=0.85, w=0.3), dx=0.1)
    assert np.isfinite(compute_features(q).vector()).all()


def test_vector_order():
    f = compute_features(make_person())
    assert len(FEATURE_NAMES) == 6
    assert f.vector().tolist() == [getattr(f, n) for n in FEATURE_NAMES]
