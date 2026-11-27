from pathlib import Path
from typing import Literal

from axis_ai.scorers.base import Scorer
from axis_ai.scorers.rank import RankScorer
from axis_ai.scorers.rule import RULE_PRESETS, AnyOfScorer, RuleScorer
from axis_ai.scorers.stat import StatScorer

__all__ = ["Scorer", "StatScorer", "RankScorer", "RuleScorer", "AnyOfScorer", "RULE_PRESETS", "load_scorers"]


def load_scorers(model_dir: Path, kind: Literal["stat", "rank"]) -> dict[str, Scorer]:
    if kind not in ("stat", "rank"):
        raise ValueError(f"알 수 없는 scorer 종류: {kind}")
    if kind == "rank":
        rank_dir = Path(model_dir) / "rank"
        if not rank_dir.is_dir():
            return {}
        ranked = (RankScorer.load(p) for p in sorted(rank_dir.glob("*.pt")))
        return {s.style_id: s for s in ranked}
    folder = Path(model_dir) / "stat"
    if not folder.is_dir():
        return {}
    scorers = (StatScorer.load(p) for p in sorted(folder.glob("*.json")))
    return {s.style_id: s for s in scorers}
