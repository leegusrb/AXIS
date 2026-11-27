import json
import math
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from axis_ai.params import SHOT_MIN_STD
from axis_ai.types import FEATURE_NAMES, CompositionFeatures

MIN_SAMPLES = 5
MAX_COMPONENTS = 3
REG_COVAR = 1e-3
_DIM = len(FEATURE_NAMES)
_SHOT = FEATURE_NAMES.index("shot")


class StatScorer:
    """구도 특징 분포(가우시안 혼합)와의 마할라노비스 거리로 점수를 낸다."""

    def __init__(self, style_id: str, version: str, means: np.ndarray, inv_covs: np.ndarray):
        self.style_id = style_id
        self.version = version
        self.means = np.asarray(means, dtype=float)
        self.inv_covs = np.asarray(inv_covs, dtype=float)

    @classmethod
    def fit(cls, style_id: str, feats: list[CompositionFeatures], version: str) -> "StatScorer":
        n = len(feats)
        if n < MIN_SAMPLES:
            raise ValueError(f"표본이 {MIN_SAMPLES}개 이상 필요합니다: {n}")
        x = np.stack([f.vector() for f in feats])
        best, best_bic = None, math.inf
        for k in range(1, max(1, min(MAX_COMPONENTS, n // 10)) + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 reg_covar=REG_COVAR, random_state=0).fit(x)
            bic = gm.bic(x)
            if bic < best_bic:
                best, best_bic = gm, bic
        covs = best.covariances_.copy()
        for c in covs:  # shot은 0/0.5/1 값이라 한 가지 샷만 있으면 분산이 0에 가깝다
            c[_SHOT, _SHOT] += max(0.0, SHOT_MIN_STD ** 2 - c[_SHOT, _SHOT])
        return cls(style_id, version, best.means_, np.linalg.inv(covs))

    def score(self, f: CompositionFeatures) -> float:
        d = f.vector() - self.means  # (K, 6)
        d2 = np.einsum("ki,kij,kj->k", d, self.inv_covs, d)
        s = math.exp(-float(d2.min()) / (2 * _DIM))
        return min(1.0, max(0.0, s))

    def save(self, path: Path) -> None:
        data = {
            "kind": "stat",
            "style_id": self.style_id,
            "version": self.version,
            "means": self.means.tolist(),
            "inv_covs": self.inv_covs.tolist(),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StatScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["style_id"], data["version"], np.array(data["means"]),
                   np.array(data["inv_covs"]))
