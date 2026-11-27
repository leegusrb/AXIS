"""단안 깊이 추정(Depth Anything V2 Small, Apache-2.0)과 피사체·배경 깊이 특징."""
import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .params import DEPTH_PREVIEW_W, DEPTH_SIZE
from .types import Box

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass(frozen=True)
class DepthFeatures:
    subject: float  # 피사체 박스 안 상대 깊이 중앙값(1이 가까움)
    background: float  # 박스 밖 중앙값
    diff: float  # subject - background

    def vector(self) -> np.ndarray:
        return np.array([self.subject, self.background, self.diff], dtype=np.float32)


def depth_features(depth: np.ndarray, box: Box) -> DepthFeatures:
    h, w = depth.shape
    x, y, bw, bh = box
    x0, y0 = int(np.clip(round(x * w), 0, w - 1)), int(np.clip(round(y * h), 0, h - 1))
    x1, y1 = int(np.clip(round((x + bw) * w), x0 + 1, w)), int(np.clip(round((y + bh) * h), y0 + 1, h))
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    subject = float(np.median(depth[mask]))
    background = float(np.median(depth[~mask])) if (~mask).any() else subject
    return DepthFeatures(subject, background, subject - background)


def preview_png(depth: np.ndarray, width: int = DEPTH_PREVIEW_W) -> str:
    """흑백 깊이 지도(가까울수록 밝게) base64 PNG."""
    img = Image.fromarray((np.clip(depth, 0, 1) * 255).astype(np.uint8), "L")
    img = img.resize((width, max(1, round(img.height * width / img.width))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DepthEstimator:
    def __init__(self, model_id: str = DEFAULT_DEPTH_MODEL, size: int = DEPTH_SIZE):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.size = size
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._proc = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self._device).eval()

    def estimate(self, img: Image.Image) -> np.ndarray:
        """짧은 변이 size인 0–1 상대 깊이(1이 가까움) float32 배열."""
        img = img.convert("RGB")
        s = self.size / min(img.size)
        out_w, out_h = max(1, round(img.width * s)), max(1, round(img.height * s))
        side = (self.size // 14) * 14  # 모델 패치 크기(14)의 배수
        inputs = self._proc(images=img, return_tensors="pt", size={"height": side, "width": side},
                            keep_aspect_ratio=True)
        with self._torch.no_grad():
            pred = self._model(pixel_values=inputs["pixel_values"].to(self._device)).predicted_depth
            pred = self._torch.nn.functional.interpolate(pred.unsqueeze(1), size=(out_h, out_w),
                                                         mode="bilinear", align_corners=False)[0, 0]
        d = pred.float().cpu().numpy()
        lo, hi = float(d.min()), float(d.max())
        return ((d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)).astype(np.float32)
