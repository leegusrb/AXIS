import pytest

from axis_ai.scorers.rule import RULE_PRESETS, RuleScorer
from conftest import feat


def test_rule_scorer_peaks_at_target():
    s = RULE_PRESETS["thirds_upper"]
    assert s.score(feat(cx=0.33, eye_y=0.33, area=0.25)) == pytest.approx(1.0)
    assert s.score(feat(cx=0.6, eye_y=0.33, area=0.25)) < 0.5


def test_presets_cover_fake_style_ids():
    assert set(RULE_PRESETS) == {"wide_full", "tight_close", "thirds_upper"}
    for sid, s in RULE_PRESETS.items():
        assert s.style_id == sid and s.version == "rule-v0"


def test_only_target_keys_are_used():
    s = RuleScorer("x", {"cx": 0.5}, {"cx": 0.1})
    assert s.score(feat(cx=0.5, area=0.99, eye_y=0.9)) == pytest.approx(1.0)


def test_one_sigma_off_gives_expected_value():
    s = RuleScorer("x", {"cx": 0.5}, {"cx": 0.1})
    assert s.score(feat(cx=0.6)) == pytest.approx(0.6065306597)
