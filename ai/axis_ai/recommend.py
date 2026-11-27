import numpy as np

from axis_ai.params import REC_TEMPERATURE, REC_W_COMPOSITION, REC_W_EMBED
from axis_ai.scorers.base import Scorer
from axis_ai.types import CompositionFeatures


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recommend(liked: list[tuple[CompositionFeatures, np.ndarray]],
              scorers: dict[str, Scorer],
              centroids: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """좋아요한 사진으로 스타일 순위를 낸다. (style_id, 백분율) 내림차순, 합 100."""
    if not liked:
        raise ValueError("좋아요한 사진이 없습니다")
    missing = sorted(set(scorers) - set(centroids))
    if missing:
        raise KeyError(f"centroid가 없는 스타일: {', '.join(missing)}")

    mean_emb = np.mean([np.asarray(e, dtype=float) for _, e in liked], axis=0)
    styles = sorted(scorers)
    values = []
    for s in styles:
        comp = float(np.mean([scorers[s].score(f) for f, _ in liked]))
        emb = _cosine(mean_emb, np.asarray(centroids[s], dtype=float))
        values.append(REC_W_COMPOSITION * comp + REC_W_EMBED * emb)

    logits = np.array(values) / REC_TEMPERATURE
    exp = np.exp(logits - logits.max())
    pct = exp / exp.sum() * 100
    return sorted(zip(styles, (float(p) for p in pct)), key=lambda t: (-t[1], t[0]))
