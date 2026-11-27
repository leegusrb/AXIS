from pathlib import Path

import numpy as np
import torch
from torch import nn

from axis_ai.features import compute_features
from axis_ai.geometry import transform_person
from axis_ai.types import CompositionFeatures, Person

_DIM = 6
_STD_FLOOR = 1e-6
# 점수 보정: 원본 원점수 중앙값 -> 0.9, 변형 원점수 중앙값 -> 0.5 (StatScorer와 같은 쓰임새의 0–1 점수)
_CALIB_GOOD = 0.9


def _rand_shift(rng: np.random.Generator) -> tuple[float, float]:
    d = [float(rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.25)) for _ in range(2)]
    if rng.random() < 1 / 3:  # 한 축만 움직이는 변형도 만든다
        d[int(rng.integers(2))] = 0.0
    return d[0], d[1]


def make_pairs(own: list[Person], others: list[Person], n_aug: int = 20,
               seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(더 나은 쪽, 못한 쪽) 특징 벡터 쌍. 원본 > 가상 변형, 자기 원본 > 다른 스타일 원본."""
    rng = np.random.default_rng(seed)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for p in own:
        base = compute_features(p).vector()
        for _ in range(n_aug):
            dx, dy = _rand_shift(rng)
            v = transform_person(p, dx, dy, float(rng.uniform(0.7, 1.4)))
            if v is not None:
                pairs.append((base, compute_features(v).vector()))
        if others:
            o = others[int(rng.integers(len(others)))]
            pairs.append((base, compute_features(o).vector()))
    return pairs


class RankScorer:
    """순위 학습으로 만든 작은 MLP. 원본이 변형·다른 스타일보다 높은 점수를 받도록 학습한다."""

    kind = "rank"

    def __init__(self, style_id: str, version: str, mean: torch.Tensor, std: torch.Tensor, net: nn.Module,
                 calib: torch.Tensor | None = None):
        self.style_id = style_id
        self.version = version
        self.mean = mean
        self.std = std
        self.net = net.eval()
        # score = sigmoid(a * (raw - b)). 보정 정보가 없는 예전 파일은 a=1, b=0(보정 없음)
        self.calib = calib if calib is not None else torch.tensor([1.0, 0.0])

    @staticmethod
    def _net() -> nn.Module:
        return nn.Sequential(nn.Linear(_DIM, 32), nn.ReLU(), nn.Linear(32, 1))

    @classmethod
    def fit(cls, style_id: str, pairs: list[tuple[np.ndarray, np.ndarray]], version: str,
            epochs: int = 200, seed: int = 0) -> "RankScorer":
        if not pairs:
            raise ValueError("학습 쌍이 없어요")
        torch.manual_seed(seed)
        good = torch.tensor(np.stack([a for a, _ in pairs]), dtype=torch.float32)
        bad = torch.tensor(np.stack([b for _, b in pairs]), dtype=torch.float32)
        allv = torch.cat([good, bad])
        mean, std = allv.mean(0), allv.std(0).clamp_min(_STD_FLOOR)
        good, bad = (good - mean) / std, (bad - mean) / std
        net = cls._net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.MarginRankingLoss(margin=0.1)
        target = torch.ones(len(pairs))
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(good).squeeze(1), net(bad).squeeze(1), target)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            g_med = float(net(good).squeeze(1).median())
            b_med = float(net(bad).squeeze(1).median())
        if g_med > b_med:
            a = float(np.log(_CALIB_GOOD / (1 - _CALIB_GOOD))) / (g_med - b_med)
            calib = torch.tensor([a, b_med])
        else:  # 학습이 원본과 변형을 가르지 못한 경우: 보정하지 않는다
            calib = torch.tensor([1.0, 0.0])
        return cls(style_id, version, mean, std, net, calib)

    def score(self, f: CompositionFeatures) -> float:
        x = (torch.tensor(f.vector(), dtype=torch.float32) - self.mean) / self.std
        with torch.no_grad():
            raw = self.net(x.unsqueeze(0))[0, 0]
            return float(torch.sigmoid(self.calib[0] * (raw - self.calib[1])))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "rank", "style_id": self.style_id, "version": self.version,
                    "mean": self.mean, "std": self.std, "state_dict": self.net.state_dict(),
                    "calib": self.calib}, path)

    @classmethod
    def load(cls, path: Path) -> "RankScorer":
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        net = cls._net()
        net.load_state_dict(d["state_dict"])
        return cls(d["style_id"], d["version"], d["mean"], d["std"], net, d.get("calib"))
