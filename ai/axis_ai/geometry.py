from dataclasses import replace

from .params import MIN_VISIBLE_AREA
from .types import Person


def transform_person(p: Person, dx: float = 0, dy: float = 0, scale: float = 1.0) -> Person | None:
    """박스 중심 기준으로 scale배 한 뒤 (dx, dy)만큼 옮긴다. 인물이 화면 밖으로 나가면 None."""
    x, y, w, h = p.box
    cx, cy = x + w / 2, y + h / 2
    ncx, ncy = cx + dx, cy + dy
    nw, nh = w * scale, h * scale
    x0, y0, x1, y1 = ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2

    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)
    cw, ch = max(cx1 - cx0, 0.0), max(cy1 - cy0, 0.0)
    full = nw * nh
    if full <= 0 or (cw * ch) / full < MIN_VISIBLE_AREA:
        return None

    kp = p.keypoints.copy()
    kp[:, 0] = (kp[:, 0] - cx) * scale + ncx
    kp[:, 1] = (kp[:, 1] - cy) * scale + ncy
    outside = (kp[:, 0] < 0) | (kp[:, 0] > 1) | (kp[:, 1] < 0) | (kp[:, 1] > 1)
    kp[outside, 2] = 0.0
    return replace(p, box=(cx0, cy0, cw, ch), keypoints=kp)
