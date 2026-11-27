"""음식 검출: YOLO 검출 모델(COCO)에서 음식 관련 클래스만 남겨 키포인트 없는 Person으로 돌려준다."""
from pathlib import Path

import numpy as np
from PIL import Image

from .pose import _ensure_weights
from .types import Person

DEFAULT_FOOD_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "weights" / "yolo11n.pt"

FOOD_CLASSES = frozenset({"bowl", "cup", "wine glass", "banana", "apple", "sandwich", "orange",
                          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"})


def boxes_to_food(xyxyn: np.ndarray, confs: np.ndarray, names: list[str]) -> list[Person]:
    out = []
    for b, c, n in zip(xyxyn, confs, names):
        if n not in FOOD_CLASSES:
            continue
        x0, y0, x1, y1 = (float(v) for v in np.clip(b, 0, 1))
        out.append(Person(box=(x0, y0, x1 - x0, y1 - y0), keypoints=np.zeros((17, 3)),
                          conf=float(c), label=n))
    return out


class FoodDetector:
    def __init__(self, weights: str | Path = DEFAULT_FOOD_WEIGHTS):
        from ultralytics import YOLO

        self._model = YOLO(_ensure_weights(Path(weights)))

    def detect(self, img: Image.Image) -> list[Person]:
        res = self._model(np.asarray(img.convert("RGB")), verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        names = [res.names[int(k)] for k in res.boxes.cls.cpu().numpy()]
        return boxes_to_food(res.boxes.xyxyn.cpu().numpy(), res.boxes.conf.cpu().numpy(), names)
