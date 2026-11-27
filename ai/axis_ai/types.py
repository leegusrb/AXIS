from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]  # x, y, w, h (좌상단 기준, 0-1 정규화)


@dataclass(frozen=True)
class Person:
    box: Box
    keypoints: np.ndarray  # (17, 3) x, y, conf — COCO 순서
    conf: float
    label: str = "person"  # 검출 클래스 이름(음식은 키포인트가 모두 0)


FEATURE_NAMES: tuple[str, ...] = ("cx", "eye_y", "area", "headroom", "face_x", "shot")


@dataclass(frozen=True)
class CompositionFeatures:
    cx: float
    eye_y: float
    area: float
    headroom: float
    face_x: float
    shot: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=float)
