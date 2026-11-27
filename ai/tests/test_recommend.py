import numpy as np
import pytest

from axis_ai.recommend import recommend
from axis_ai.scorers import RuleScorer
from conftest import feat

SPREAD = {"cx": 0.1, "eye_y": 0.08, "area": 0.08}


def _scorers():
    return {
        "a": RuleScorer("a", {"cx": 0.5, "eye_y": 0.3, "area": 0.12}, SPREAD),
        "b": RuleScorer("b", {"cx": 0.5, "eye_y": 0.33, "area": 0.45}, SPREAD),
        "c": RuleScorer("c", {"cx": 0.33, "eye_y": 0.33, "area": 0.25}, SPREAD),
    }


def _centroids():
    return {k: np.eye(4)[i] for i, k in enumerate("abc")}


def test_prefers_style_whose_composition_matches():
    liked = [(feat(cx=0.5, eye_y=0.33, area=0.45), np.ones(4))]
    result = recommend(liked, _scorers(), _centroids())
    assert result[0][0] == "b"


def test_embedding_breaks_tie_when_composition_equal():
    scorers = {
        "a": RuleScorer("a", {"cx": 0.5}, SPREAD),
        "b": RuleScorer("b", {"cx": 0.5}, SPREAD),
    }
    centroids = {"a": np.array([1.0, 0, 0, 0]), "b": np.array([0, 1.0, 0, 0])}
    liked = [(feat(cx=0.5), np.array([0.1, 0.9, 0, 0]))]
    result = recommend(liked, scorers, centroids)
    assert [s for s, _ in result] == ["b", "a"]
    assert result[0][1] > result[1][1]


def test_percentages_sum_to_100_and_sorted():
    liked = [(feat(), np.array([1.0, 0.2, 0, 0])),
             (feat(cx=0.4, area=0.3), np.array([0, 1.0, 0, 0]))]
    result = recommend(liked, _scorers(), _centroids())
    pcts = [p for _, p in result]
    assert abs(sum(pcts) - 100) < 1e-6
    assert pcts == sorted(pcts, reverse=True)
    assert len(result) == 3


def test_ties_broken_by_style_id():
    scorers = {k: RuleScorer(k, {"cx": 0.5}, SPREAD) for k in ("z", "m", "a")}
    centroids = {k: np.ones(4) for k in scorers}
    result = recommend([(feat(), np.ones(4))], scorers, centroids)
    assert [s for s, _ in result] == ["a", "m", "z"]


def test_zero_norm_embedding_counts_as_zero_similarity():
    result = recommend([(feat(), np.zeros(4))], _scorers(), _centroids())
    assert abs(sum(p for _, p in result) - 100) < 1e-6


def test_empty_raises():
    with pytest.raises(ValueError):
        recommend([], _scorers(), _centroids())


def test_missing_centroid_raises_key_error_with_message():
    centroids = _centroids()
    del centroids["c"]
    with pytest.raises(KeyError, match="c"):
        recommend([(feat(), np.ones(4))], _scorers(), centroids)
