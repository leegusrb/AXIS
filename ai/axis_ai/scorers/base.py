from typing import Protocol

from axis_ai.types import CompositionFeatures


class Scorer(Protocol):
    style_id: str
    version: str

    def score(self, f: CompositionFeatures) -> float: ...
