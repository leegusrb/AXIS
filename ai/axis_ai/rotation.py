"""화면 기울기 추정: 어깨(없으면 눈) 선의 기울기. 양수면 화면 오른쪽이 아래 → 휴대폰을 시계 방향으로."""
import math
from typing import Literal

from .params import KEYPOINT_VISIBLE, ROTATE_DEG
from .types import Person

ROTATE_MESSAGES = {"CW": "휴대폰을 시계 방향으로 살짝 돌리세요",
                   "CCW": "휴대폰을 반시계 방향으로 살짝 돌리세요"}
OK_TILTED_MESSAGE = "좋아요! 수평만 맞추면 완벽해요"

# (키포인트 쌍, 박스 너비 대비 최소 x 간격). 간격이 좁으면 옆모습이라 기울기를 믿을 수 없다.
_PAIRS = (((5, 6), 0.25), ((1, 2), 0.10))


def estimate_tilt(p: Person) -> float | None:
    if p.label != "person":
        return None
    kp = p.keypoints
    for (a, b), min_gap in _PAIRS:
        if kp[a, 2] < KEYPOINT_VISIBLE or kp[b, 2] < KEYPOINT_VISIBLE:
            continue
        (x0, y0), (x1, y1) = sorted([tuple(kp[a, :2]), tuple(kp[b, :2])])
        if x1 - x0 < min_gap * p.box[2]:
            return None
        return math.degrees(math.atan2(y1 - y0, x1 - x0))
    return None


def rotate_hint(deg: float | None, threshold: float = ROTATE_DEG) -> Literal["CW", "CCW"] | None:
    if deg is None or abs(deg) <= threshold:
        return None
    return "CW" if deg > 0 else "CCW"
