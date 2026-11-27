"""YOLO 포즈 검출 래퍼. 출력은 0-1 정규화 좌표의 Person 목록."""
from pathlib import Path

import numpy as np
from PIL import Image

from .types import Person

# 저장소 루트의 models/weights (gitignore 대상). 파일이 없으면 처음 한 번 내려받는다.
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "weights" / "yolo11n-pose.pt"


def _ensure_weights(path: Path) -> str:
    if path.is_file():
        return str(path)
    from ultralytics.utils.downloads import attempt_download_asset

    path.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(path))  # path.name을 GitHub 릴리스에서 path 위치로 내려받는다
    return str(path)


class PoseDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS):
        from ultralytics import YOLO

        self._model = YOLO(_ensure_weights(Path(weights)))

    def detect(self, img: Image.Image) -> list[Person]:
        res = self._model(np.asarray(img.convert("RGB")), verbose=False)[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []
        xyxyn = res.boxes.xyxyn.cpu().numpy().clip(0, 1)
        confs = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xyn.cpu().numpy().clip(0, 1)
        kconf = res.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(kxy.shape[:2])
        persons = []
        for b, c, xy, kc in zip(xyxyn, confs, kxy, kconf):
            x0, y0, x1, y1 = (float(v) for v in b)
            kp = np.concatenate([xy, kc[:, None]], axis=1).astype(float)
            persons.append(Person(box=(x0, y0, x1 - x0, y1 - y0), keypoints=kp, conf=float(c)))
        return persons
