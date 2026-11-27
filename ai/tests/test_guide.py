import math

import pytest

from axis_ai.features import compute_features
from axis_ai.geometry import transform_person
from axis_ai.guide import MOVES, decide
from axis_ai.scorers import RULE_PRESETS
from conftest import make_person


class CenterScorer:
    style_id = "center"
    version = "test"

    def __init__(self, target: dict[str, float]):
        self.target = target

    def score(self, f) -> float:
        return math.exp(-sum(((getattr(f, k) - t) / 0.1) ** 2 for k, t in self.target.items()))


def test_moves_signs():
    assert MOVES["LEFT"] == (0.08, 0, 1.0) and MOVES["RIGHT"] == (-0.08, 0, 1.0)
    assert MOVES["UP"] == (0, 0.08, 1.0) and MOVES["DOWN"] == (0, -0.08, 1.0)
    assert MOVES["CLOSER"][2] == 1.25 and MOVES["FARTHER"][2] == 0.8


@pytest.mark.parametrize("person_kwargs,target,expected", [
    (dict(cx=0.2), dict(cx=0.5), "LEFT"),
    (dict(cx=0.8), dict(cx=0.5), "RIGHT"),
    (dict(top=0.05), dict(eye_y=0.35), "UP"),
    (dict(top=0.45, h=0.5), dict(eye_y=0.25), "DOWN"),
    (dict(w=0.1, h=0.25), dict(area=0.2), "CLOSER"),
    (dict(w=0.5, h=0.9), dict(area=0.15), "FARTHER"),
])
def test_direction_sign_table(person_kwargs, target, expected):
    d = decide(make_person(**person_kwargs), CenterScorer(target))
    assert d.direction == expected


def test_ok_when_at_target():
    p = make_person()
    f = compute_features(p)
    d = decide(p, CenterScorer({"cx": f.cx, "eye_y": f.eye_y, "area": f.area}))
    assert d.direction == "OK"
    assert d.message == "좋아요! 지금 찍으세요"
    assert d.magnitude == "SMALL"
    assert d.confidence == p.conf
    assert d.target_box == p.box


def test_no_subject():
    d = decide(None, CenterScorer({"cx": 0.5}))
    assert d.direction == "NO_SUBJECT"
    assert d.subject_box is None and d.target_box is None
    assert d.score == 0.0 and d.score_delta == 0.0 and d.confidence == 0.0
    assert d.magnitude == "SMALL"
    assert d.message == "인물을 화면에 담아주세요"


def test_small_then_large_magnitude():
    d = decide(make_person(cx=0.5), CenterScorer({"cx": 0.55}))
    assert d.direction == "LEFT" and d.magnitude == "SMALL"
    assert d.message == "카메라를 왼쪽으로 조금 옮기세요"
    d = decide(make_person(cx=0.4), CenterScorer({"cx": 0.5}))
    assert d.direction == "LEFT" and d.magnitude == "LARGE"
    assert d.message == "카메라를 왼쪽으로 옮기세요"


@pytest.mark.parametrize("person_kwargs,scorer,expected", [
    (dict(), RULE_PRESETS["tight_close"], "CLOSER"),
    (dict(cx=0.85), RULE_PRESETS["thirds_upper"], "RIGHT"),
    (dict(cx=0.2), CenterScorer({"cx": 0.5}), "LEFT"),
])
def test_far_subject_still_guided_with_full_confidence(person_kwargs, scorer, expected):
    p = make_person(**person_kwargs)
    d = decide(p, scorer)
    assert d.direction == expected
    assert d.confidence == p.conf >= 0.5


def test_no_improvement_halves_confidence():
    p = make_person(cx=0.5)
    s = CenterScorer({"cx": 0.5})
    d = decide(p, s)  # 목표 위, 단 s0 >= OK_SCORE 이므로 OK
    assert d.direction == "OK"
    class Flat:
        style_id = "flat"
        version = "t"
        def score(self, f):
            return 0.3
    d = decide(p, Flat())  # 어느 후보도 올리지 못함, s0 < OK_SCORE
    assert d.direction != "OK"
    assert d.confidence == pytest.approx(p.conf * 0.5)


def test_all_candidates_none_is_guarded(monkeypatch):
    import axis_ai.guide as g
    monkeypatch.setattr(g, "transform_person", lambda *a, **k: None)
    p = make_person()
    d = decide(p, CenterScorer({"cx": 0.9}))
    assert d.direction == "FARTHER" and d.confidence == pytest.approx(p.conf * 0.5)
    f = compute_features(p)
    d = decide(p, CenterScorer({"cx": f.cx}))
    assert d.direction == "OK"


def test_never_pushes_subject_off_screen():
    p = make_person(cx=0.98, w=0.2)
    assert transform_person(p, *MOVES["LEFT"]) is None  # 인물이 오른쪽으로 사라지는 후보
    d = decide(p, CenterScorer({"cx": 1.2}))
    assert d.direction != "LEFT"
    assert d.direction in MOVES
    assert transform_person(p, *MOVES[d.direction]) is not None


def test_target_box_moves_toward_target():
    p = make_person(cx=0.4)
    d = decide(p, CenterScorer({"cx": 0.5}))
    assert d.direction == "LEFT"
    x0 = p.box[0] + p.box[2] / 2
    x1 = d.target_box[0] + d.target_box[2] / 2
    assert x1 > x0
    assert x1 - x0 <= 5 * 0.08 + 1e-9
    assert abs(x1 - 0.5) < abs(x0 - 0.5)
    assert d.subject_box == p.box


def test_confidence_scaled_by_person_conf():
    s = CenterScorer({"cx": 0.5})
    hi = decide(make_person(cx=0.4, conf=0.9), s)
    lo = decide(make_person(cx=0.4, conf=0.45), s)
    assert hi.direction == lo.direction
    assert lo.confidence == pytest.approx(hi.confidence * 0.5)
    assert hi.confidence <= 0.9


def test_rule_preset_drives_left():
    p = make_person(cx=0.15, w=0.3, h=0.83, top=0.2)
    d = decide(p, RULE_PRESETS["thirds_upper"])
    assert d.direction == "LEFT"
