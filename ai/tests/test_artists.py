from axis_ai.artists import ARTIST_INFO, CATEGORIES, RULE_ARTISTS, RULE_DIAGRAM, learned_diagram
from axis_ai.scorers.rule import AnyOfScorer, RuleScorer
from conftest import feat


def test_artist_names_and_categories():
    assert [c["id"] for c in CATEGORIES] == ["person", "food"]
    assert ARTIST_INFO["wide_full"]["name"] == "작가 A — 여백과 인물 중심"
    assert ARTIST_INFO["sym_back"]["name"] == "작가 B — 대칭과 배경 강조"
    assert ARTIST_INFO["tight_close"]["name"].startswith("작가 C")
    assert ARTIST_INFO["thirds_upper"]["name"].startswith("작가 D")
    assert {ARTIST_INFO[s]["category"] for s in ("food_center", "food_thirds")} == {"food"}
    assert set(RULE_ARTISTS) == set(RULE_DIAGRAM) == {"sym_back", "food_center", "food_thirds"}


def test_any_of_takes_best_part():
    left = RuleScorer("t", {"cx": 0.33}, {"cx": 0.05})
    right = RuleScorer("t", {"cx": 0.67}, {"cx": 0.05})
    s = AnyOfScorer("t", [left, right])
    assert s.score(feat(cx=0.67)) == right.score(feat(cx=0.67)) > 0.99
    assert s.score(feat(cx=0.5)) < 0.01


def test_sym_back_differs_from_wide_full_framing():
    b = RULE_ARTISTS["sym_back"]
    # 작가 A처럼 인물이 적당히 큰 가운데 전신(area 0.12)은 B 기준으로 낮아야 한다
    assert b.score(feat(cx=0.5, face_x=0.5, eye_y=0.4, area=0.12)) < 0.3
    assert b.score(feat(cx=0.5, face_x=0.5, eye_y=0.4, area=0.05)) > 0.8
    assert b.score(feat(cx=0.42, face_x=0.42, eye_y=0.4, area=0.05)) < 0.3  # 엄격한 가운데


def test_learned_diagram_is_mean_train_box():
    rows = [{"style_id": "s", "split": "train", "person": {"box": [0.1, 0.2, 0.3, 0.4]}},
            {"style_id": "s", "split": "train", "person": {"box": [0.3, 0.2, 0.3, 0.6]}},
            {"style_id": "s", "split": "test", "person": {"box": [0.9, 0.9, 0.1, 0.1]}},
            {"style_id": "s", "split": "train", "person": None}]
    assert learned_diagram(rows, "s") == (0.2, 0.2, 0.3, 0.5)
    assert learned_diagram(rows, "other") is None
