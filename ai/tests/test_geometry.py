import pytest

from axis_ai.geometry import transform_person
from conftest import make_person


def test_shift_moves_box_and_keypoints():
    p = make_person(cx=0.5)
    q = transform_person(p, dx=0.1)
    assert q.box[0] == pytest.approx(p.box[0] + 0.1)
    assert q.keypoints[0, 0] == pytest.approx(p.keypoints[0, 0] + 0.1)


def test_scale_about_box_center():
    p = make_person(cx=0.5, top=0.2, w=0.2, h=0.4)
    q = transform_person(p, scale=1.25)
    assert q.box[2] == pytest.approx(0.25) and q.box[3] == pytest.approx(0.5)
    assert q.box[0] + q.box[2] / 2 == pytest.approx(0.5)


def test_partially_outside_is_clipped_and_keypoints_hidden():
    q = transform_person(make_person(cx=0.85, w=0.3), dx=0.1)
    assert q.box[0] + q.box[2] <= 1.0
    assert (q.keypoints[q.keypoints[:, 0] > 1.0, 2] == 0).all()


def test_mostly_outside_returns_none():
    assert transform_person(make_person(cx=0.9, w=0.3), dx=0.3) is None
