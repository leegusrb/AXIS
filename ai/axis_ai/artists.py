"""화면·발표에서 쓰는 작가(스타일) 이름과 카테고리, 사진 없이 규칙으로 평가하는 작가."""
import numpy as np

from .scorers.rule import AnyOfScorer, RuleScorer

CATEGORIES: list[dict] = [{"id": "person", "name": "인물"}, {"id": "food", "name": "음식"}]

ARTIST_INFO: dict[str, dict] = {
    "wide_full": {"name": "작가 A — 여백과 인물 중심", "category": "person",
                  "description": "머리부터 발끝까지 담고, 인물 주변에 여백을 넉넉히 두는 구도"},
    "sym_back": {"name": "작가 B — 대칭과 배경 강조", "category": "person",
                 "description": "인물을 정확히 가운데에 작게 두고, 배경을 넓게 보여 주는 대칭 구도"},
    "tight_close": {"name": "작가 C — 꽉 찬 클로즈업", "category": "person",
                    "description": "얼굴과 어깨가 화면 가운데를 크게 채우는 구도"},
    "thirds_upper": {"name": "작가 D — 삼분할 상반신", "category": "person",
                     "description": "상반신을 화면 왼쪽이나 오른쪽 1/3 선 근처에 두는 구도"},
    "food_center": {"name": "작가 E — 가운데 크게", "category": "food",
                    "description": "음식을 화면 가운데에 크게 담는 구도"},
    "food_thirds": {"name": "작가 F — 삼분할과 여백", "category": "food",
                    "description": "음식을 1/3 선 근처에 두고 나머지에 여백을 남기는 구도"},
}

_FOOD_THIRDS_SPREAD = {"cx": 0.06, "area": 0.06, "headroom": 0.1}

RULE_ARTISTS: dict = {
    "sym_back": RuleScorer(
        "sym_back", {"cx": 0.5, "face_x": 0.5, "area": 0.05, "eye_y": 0.4},
        {"cx": 0.03, "face_x": 0.03, "area": 0.02, "eye_y": 0.1}),
    "food_center": RuleScorer(
        "food_center", {"cx": 0.5, "area": 0.35, "headroom": 0.2},
        {"cx": 0.08, "area": 0.1, "headroom": 0.08}),
    "food_thirds": AnyOfScorer("food_thirds", [
        RuleScorer("food_thirds", {"cx": c, "area": 0.15, "headroom": 0.3}, _FOOD_THIRDS_SPREAD)
        for c in (0.33, 0.67)]),
}

# 작가 카드의 구도 그림 박스 (x, y, w, h). 규칙 목표값에 맞춘다.
RULE_DIAGRAM: dict[str, tuple[float, float, float, float]] = {
    "sym_back": (0.45, 0.33, 0.1, 0.5),
    "food_center": (0.2, 0.2, 0.6, 0.6),
    "food_thirds": (0.14, 0.3, 0.38, 0.38),
}


def learned_diagram(rows: list[dict], style_id: str) -> tuple[float, float, float, float] | None:
    """학습 작가의 구도 그림: train 사진 인물 박스의 평균."""
    boxes = [r["person"]["box"] for r in rows
             if r["style_id"] == style_id and r["split"] == "train" and r.get("person")]
    if not boxes:
        return None
    return tuple(round(float(v), 4) for v in np.mean(boxes, axis=0))
