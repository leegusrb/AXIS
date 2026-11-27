"""이미지 단위 어긋남 변형: 이미지·키포인트·박스·깊이 지도에 같은 아핀 변환을 적용한다."""
import math
from dataclasses import replace

import cv2
import numpy as np

from ..params import ANGLE_NORM, MIN_VISIBLE_AREA
from ..types import Person


def _matrix(w: int, h: int, dx: float, dy: float, scale: float, angle_deg: float) -> np.ndarray:
    """픽셀 좌표 2x3 정방향 행렬: 화면 중심 기준 회전(양수=시계 방향)·배율 뒤 (dx, dy) 이동."""
    a = math.radians(angle_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    cx, cy = w / 2, h / 2
    return np.array([[c, -s, cx - c * cx + s * cy + dx * w],
                     [s, c, cy - s * cx - c * cy + dy * h]], dtype=np.float64)


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def warp(img: np.ndarray, person: Person, depth: np.ndarray | None, dx: float, dy: float,
         scale: float, angle_deg: float) -> tuple[np.ndarray, Person | None, np.ndarray | None]:
    h, w = img.shape[:2]
    m = _matrix(w, h, dx, dy, scale, angle_deg)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    dout = None
    if depth is not None:
        dh, dw = depth.shape
        dout = cv2.warpAffine(depth.astype(np.float32), _matrix(dw, dh, dx, dy, scale, angle_deg), (dw, dh),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    size = np.array([w, h], dtype=np.float64)
    x, y, bw, bh = person.box
    corners = np.array([[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]]) * size
    c = _apply(m, corners) / size
    x0, y0 = c.min(axis=0)
    x1, y1 = c.max(axis=0)
    full = (x1 - x0) * (y1 - y0)
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)
    vis = max(cx1 - cx0, 0.0) * max(cy1 - cy0, 0.0)
    if full <= 0 or vis / full < MIN_VISIBLE_AREA:
        return out, None, dout
    kp = person.keypoints.copy()
    kp[:, :2] = _apply(m, kp[:, :2] * size) / size
    outside = (kp[:, 0] < 0) | (kp[:, 0] > 1) | (kp[:, 1] < 0) | (kp[:, 1] > 1)
    kp[outside, 2] = 0.0
    box = (float(cx0), float(cy0), float(cx1 - cx0), float(cy1 - cy0))
    return out, replace(person, box=box, keypoints=kp), dout


def sample_misalignment(rng: np.random.Generator) -> tuple[float, float, float, float]:
    d = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.2)) for _ in range(2)]
    if rng.random() < 1 / 3:
        d[int(rng.integers(2))] = 0.0
    scale = float(math.exp(rng.uniform(math.log(0.7), math.log(1.4))))
    angle = 0.0 if rng.random() < 0.5 else float(rng.choice([-1.0, 1.0]) * rng.uniform(3, 15))
    return d[0], d[1], scale, angle


def correction(dx: float, dy: float, scale: float, angle_deg: float) -> np.ndarray:
    """변형을 되돌리는 보정값(dx, dy, log 배율, 각도/ANGLE_NORM). 화면 속 피사체가 움직여야 할 양."""
    return np.array([-dx, -dy, -math.log(scale), -angle_deg / ANGLE_NORM], dtype=np.float32)
