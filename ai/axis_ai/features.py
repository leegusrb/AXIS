import numpy as np

from .params import KEYPOINT_VISIBLE, MIN_PERSON_CONF
from .types import CompositionFeatures, Person

_NOSE, _EYES, _KNEES, _ANKLES = 0, (1, 2), (13, 14), (15, 16)


def select_subject(persons: list[Person]) -> Person | None:
    """신뢰도 기준 이상인 인물 중 박스 면적 × 신뢰도가 가장 큰 사람."""
    cands = [p for p in persons if p.conf >= MIN_PERSON_CONF]
    if not cands:
        return None
    return max(cands, key=lambda p: p.box[2] * p.box[3] * p.conf)


def _visible(p: Person, idx) -> list[int]:
    return [i for i in idx if p.keypoints[i, 2] >= KEYPOINT_VISIBLE]


def compute_features(p: Person) -> CompositionFeatures:
    x, y, w, h = p.box
    cx = x + w / 2
    kp = p.keypoints

    eyes = _visible(p, _EYES)
    nose = _visible(p, (_NOSE,))
    if eyes:
        eye_y = float(np.mean(kp[eyes, 1]))
    elif nose:
        eye_y = float(kp[_NOSE, 1])
    else:
        eye_y = y + 0.1 * h

    face = _visible(p, (_NOSE, *_EYES))
    face_x = float(np.mean(kp[face, 0])) if face else cx

    if _visible(p, _ANKLES):
        shot = 1.0
    elif _visible(p, _KNEES):
        shot = 0.5
    else:
        shot = 0.0

    return CompositionFeatures(cx=cx, eye_y=eye_y, area=w * h, headroom=y, face_x=face_x, shot=shot)
