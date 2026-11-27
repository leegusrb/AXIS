"""MobileNetV3-small 이미지 임베딩 (576차원, L2 정규화). 추천과 B 실험에만 쓴다."""
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class Embedder:
    def __init__(self):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self._model = mobilenet_v3_small(weights=weights).eval()
        self._transform = weights.transforms()

    def embed(self, img: Image.Image) -> np.ndarray:
        x = self._transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f = self._model.avgpool(self._model.features(x)).flatten(1)[0]
        v = f.numpy().astype(np.float32)
        return v / np.linalg.norm(v)
