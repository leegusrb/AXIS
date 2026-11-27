from dataclasses import replace

import numpy as np
from axis_ai.rotation import ROTATE_MESSAGES, estimate_tilt, rotate_hint
from conftest import make_person


def _tilt(p, dy):
    kp = p.keypoints.copy()
    kp[6, 1] += dy  # 오른쪽 어깨(화면 오른쪽, x 큰 쪽)를 아래로
    return replace(p, keypoints=kp)


def test_level_shoulders_zero():
    assert abs(estimate_tilt(make_person())) < 1e-6


def test_right_lower_is_positive_and_cw():
    d = estimate_tilt(_tilt(make_person(w=0.3), 0.05))
    assert d > 8 and rotate_hint(d) == "CW"
    assert rotate_hint(-d) == "CCW" and rotate_hint(3.0) is None and rotate_hint(None) is None
    assert ROTATE_MESSAGES["CW"] == "휴대폰을 시계 방향으로 살짝 돌리세요"


def test_falls_back_to_eyes_without_shoulders():
    p = make_person()
    kp = p.keypoints.copy()
    kp[[5, 6], 2] = 0.0
    kp[2, 1] += 0.02
    assert estimate_tilt(replace(p, keypoints=kp)) > 8  # 눈 간격 0.048 ≥ 박스 너비 0.3의 10%
    kp[2, 0] = kp[1, 0] + 0.01  # 눈이 겹침(옆얼굴)
    assert estimate_tilt(replace(p, keypoints=kp)) is None


def test_tilt_none_for_side_view():
    p = make_person()
    kp = p.keypoints.copy()
    kp[6, 0] = kp[5, 0] + 0.01  # 어깨가 거의 겹침(옆모습)
    assert estimate_tilt(replace(p, keypoints=kp)) is None
    kp2 = p.keypoints.copy()
    kp2[[1, 2, 5, 6], 2] = 0.0
    assert estimate_tilt(replace(p, keypoints=kp2)) is None


def test_food_has_no_tilt():
    p = replace(make_person(), label="cake", keypoints=np.zeros((17, 3)))
    assert estimate_tilt(p) is None
