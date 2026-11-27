import math

from axis_ai.types import CompositionFeatures


class RuleScorer:
    """사진 없이 쓰는 임시 Scorer. target에 있는 특징만 계산한다."""

    def __init__(self, style_id: str, target: dict[str, float],
                 spread: dict[str, float], version: str = "rule-v0"):
        self.style_id = style_id
        self.target = dict(target)
        self.spread = dict(spread)
        self.version = version

    def score(self, f: CompositionFeatures) -> float:
        z2 = sum(((getattr(f, k) - t) / self.spread[k]) ** 2 for k, t in self.target.items())
        return float(math.exp(-0.5 * z2))


_SPREAD = {"cx": 0.1, "eye_y": 0.08, "area": 0.08}

RULE_PRESETS: dict[str, RuleScorer] = {
    "wide_full": RuleScorer("wide_full", {"cx": 0.5, "eye_y": 0.3, "area": 0.12}, _SPREAD),
    "tight_close": RuleScorer("tight_close", {"cx": 0.5, "eye_y": 0.33, "area": 0.45}, _SPREAD),
    "thirds_upper": RuleScorer("thirds_upper", {"cx": 0.33, "eye_y": 0.33, "area": 0.25}, _SPREAD),
}


class AnyOfScorer:
    """여러 규칙 중 가장 잘 맞는 것의 점수(예: 왼쪽·오른쪽 삼분할 어느 쪽이든)."""

    def __init__(self, style_id: str, parts: list[RuleScorer], version: str = "rule-v0"):
        self.style_id = style_id
        self.parts = list(parts)
        self.version = version

    def score(self, f: CompositionFeatures) -> float:
        return max(p.score(f) for p in self.parts)
