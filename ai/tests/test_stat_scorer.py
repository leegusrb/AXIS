import json

import numpy as np
import pytest

from axis_ai.scorers import load_scorers
from axis_ai.scorers.stat import StatScorer
from axis_ai.types import FEATURE_NAMES
from conftest import feat


def _cluster(cx, n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.normal(cx, 0.03)
        out.append(feat(
            cx=c,
            eye_y=rng.normal(0.3, 0.02),
            area=rng.normal(0.2, 0.02),
            headroom=rng.normal(0.15, 0.02),
            face_x=rng.normal(c, 0.02),
        ))
    return out


def test_score_higher_near_style_center():
    s = StatScorer.fit("a", _cluster(0.33), "t")
    assert s.score(feat(cx=0.33)) > s.score(feat(cx=0.7))


def test_score_in_unit_range():
    s = StatScorer.fit("a", _cluster(0.33), "t")
    assert 0.0 <= s.score(feat(cx=0.99)) <= s.score(feat(cx=0.33)) <= 1.0


def test_score_finite_for_extreme_input():
    s = StatScorer.fit("a", _cluster(0.33), "t")
    v = s.score(feat(cx=50.0, shot=1.0))
    assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_two_modes_both_score_high():
    s = StatScorer.fit("a", _cluster(0.25) + _cluster(0.75, seed=1), "t")
    assert len(s.means) >= 2
    assert min(s.score(feat(cx=0.25)), s.score(feat(cx=0.75))) > s.score(feat(cx=0.5))


def test_component_count_capped_by_sample_count():
    s = StatScorer.fit("a", _cluster(0.25, n=15) + _cluster(0.75, n=15, seed=1), "t")
    assert len(s.means) <= 3


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        StatScorer.fit("a", _cluster(0.3, n=4), "t")


def test_five_samples_ok_single_component():
    s = StatScorer.fit("a", _cluster(0.3, n=5), "t")
    assert len(s.means) == 1


def test_save_load_roundtrip(tmp_path):
    s = StatScorer.fit("a", _cluster(0.25) + _cluster(0.75, seed=1), "v1")
    p = tmp_path / "a.json"
    s.save(p)
    data = json.loads(p.read_text())
    assert data["kind"] == "stat" and data["style_id"] == "a" and data["version"] == "v1"
    assert np.array(data["means"]).shape[1] == len(FEATURE_NAMES)
    assert np.array(data["inv_covs"]).shape[1:] == (len(FEATURE_NAMES),) * 2
    r = StatScorer.load(p)
    assert (r.style_id, r.version) == ("a", "v1")
    for cx in (0.1, 0.25, 0.5, 0.75):
        assert r.score(feat(cx=cx)) == pytest.approx(s.score(feat(cx=cx)))


def test_load_scorers_reads_stat_folder(tmp_path):
    (tmp_path / "stat").mkdir()
    StatScorer.fit("a", _cluster(0.3), "t").save(tmp_path / "stat" / "a.json")
    StatScorer.fit("b", _cluster(0.6), "t").save(tmp_path / "stat" / "b.json")
    got = load_scorers(tmp_path, "stat")
    assert set(got) == {"a", "b"}
    assert got["a"].score(feat(cx=0.3)) > got["b"].score(feat(cx=0.3))


def test_load_scorers_missing_folder_is_empty(tmp_path):
    assert load_scorers(tmp_path, "stat") == {}
    assert load_scorers(tmp_path, "rank") == {}


def test_load_scorers_unknown_kind_raises(tmp_path):
    with pytest.raises(ValueError):
        load_scorers(tmp_path, "bogus")


def test_single_framing_style_tolerates_half_shot_step():
    s = StatScorer.fit("a", [f.__class__(**{**f.__dict__, "shot": 1.0}) for f in _cluster(0.33)], "t")
    same = feat(cx=0.33, eye_y=0.3, area=0.2, headroom=0.15, face_x=0.33, shot=1.0)
    near = feat(cx=0.33, eye_y=0.3, area=0.2, headroom=0.15, face_x=0.33, shot=0.5)
    assert s.score(near) > 0.3
    assert s.score(near) < s.score(same)
