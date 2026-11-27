"""안내 결정: `direction`은 항상 사용자가 카메라를 움직일 방향이며, MOVES는 그 이동으로 화면 속 인물이 움직이는 양이다."""
import math
from dataclasses import dataclass
from typing import Literal

from .features import compute_features
from .geometry import transform_person
from .params import LARGE_LOG_GAIN, MOVE_STEP, OK_DELTA, OK_SCORE, ZOOM_IN, ZOOM_OUT
from .scorers.base import Scorer
from .types import Box, Person

Direction = Literal["LEFT", "RIGHT", "UP", "DOWN", "CLOSER", "FARTHER", "OK", "NO_SUBJECT"]

# 카메라 이동 방향 -> 화면 속 인물의 (dx, dy, scale). 카메라를 왼쪽으로 돌리면 인물은 오른쪽으로 간다.
MOVES: dict[str, tuple[float, float, float]] = {
    "LEFT": (+MOVE_STEP, 0.0, 1.0),
    "RIGHT": (-MOVE_STEP, 0.0, 1.0),
    "UP": (0.0, +MOVE_STEP, 1.0),
    "DOWN": (0.0, -MOVE_STEP, 1.0),
    "CLOSER": (0.0, 0.0, ZOOM_IN),
    "FARTHER": (0.0, 0.0, ZOOM_OUT),
}

_MESSAGES = {
    "LEFT": ("카메라를 왼쪽으로 조금 옮기세요", "카메라를 왼쪽으로 옮기세요"),
    "RIGHT": ("카메라를 오른쪽으로 조금 옮기세요", "카메라를 오른쪽으로 옮기세요"),
    "UP": ("카메라를 위로 조금 옮기세요", "카메라를 위로 옮기세요"),
    "DOWN": ("카메라를 아래로 조금 옮기세요", "카메라를 아래로 옮기세요"),
    "CLOSER": ("인물에게 조금 다가가세요", "인물에게 다가가세요"),
    "FARTHER": ("조금 뒤로 물러나세요", "뒤로 물러나세요"),
}
_MSG_OK = "좋아요! 지금 찍으세요"
_MAX_TARGET_STEPS = 5


def message_for(direction: str, magnitude: str, noun: str = "인물") -> str:
    if direction == "OK":
        return _MSG_OK
    if direction == "NO_SUBJECT":
        return f"{noun}을 화면에 담아주세요"  # 인물·음식 모두 받침이 있어 "을"
    msg = _MESSAGES[direction][0 if magnitude == "SMALL" else 1]
    return msg if noun == "인물" else msg.replace("인물에게", f"{noun}에")


@dataclass
class GuideDecision:
    direction: Direction
    magnitude: Literal["SMALL", "LARGE"]
    message: str
    score: float
    score_delta: float
    confidence: float
    subject_box: Box | None
    target_box: Box | None


def _score(p: Person, scorer: Scorer) -> float:
    return float(scorer.score(compute_features(p)))


def _target_box(person: Person, move: tuple[float, float, float], scorer: Scorer) -> Box:
    current, best = person, _score(person, scorer)
    for _ in range(_MAX_TARGET_STEPS):
        nxt = transform_person(current, *move)
        if nxt is None:
            break
        s = _score(nxt, scorer)
        if s <= best:
            break
        current, best = nxt, s
    return current.box


def decide(person: Person | None, scorer: Scorer, noun: str = "인물") -> GuideDecision:
    if person is None:
        return GuideDecision("NO_SUBJECT", "SMALL", message_for("NO_SUBJECT", "SMALL", noun),
                             0.0, 0.0, 0.0, None, None)

    s0 = _score(person, scorer)
    scores: dict[str, float] = {}
    for name, move in MOVES.items():
        cand = transform_person(person, *move)
        if cand is not None:
            scores[name] = _score(cand, scorer)
    if not scores:  # 모든 후보가 화면 밖(실제로는 FARTHER가 있어 드묾)
        if s0 >= OK_SCORE:
            return GuideDecision("OK", "SMALL", _MSG_OK, s0, 0.0, person.conf, person.box, person.box)
        return GuideDecision("FARTHER", "SMALL", message_for("FARTHER", "SMALL", noun), s0, 0.0,
                             person.conf * 0.5, person.box, person.box)

    # 멀리 떨어져 점수가 거의 0이어도 방향이 보이도록 로그 점수 상승폭으로 방향을 고른다.
    log0 = math.log(max(s0, 1e-12))
    gains = {n: math.log(max(sc, 1e-12)) - log0 for n, sc in scores.items()}
    best = max(gains, key=gains.get)
    delta = scores[best] - s0

    if delta < OK_DELTA and s0 >= OK_SCORE:
        return GuideDecision("OK", "SMALL", _MSG_OK, s0, delta, person.conf, person.box, person.box)

    g = gains[best]
    magnitude = "LARGE" if g >= LARGE_LOG_GAIN else "SMALL"
    conf = person.conf * 0.5 if g <= 0 else person.conf
    msg = message_for(best, magnitude, noun)
    return GuideDecision(best, magnitude, msg, s0, delta, conf, person.box,
                         _target_box(person, MOVES[best], scorer))
