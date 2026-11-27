"""통합 가이드 모델: MobileNetV3 특징 + YOLO 구도 특징 + 깊이 특징 -> 작가별 Guide Head(MLP) -> 보정값."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..depth import DepthFeatures
from ..features import compute_features
from ..geometry import transform_person
from ..guide import MOVES, GuideDecision, message_for
from ..params import ANGLE_NORM, HEAD_OK, MOVE_STEP, ZOOM_IN
from ..scorers.base import Scorer
from ..types import Person

EMB_DIM = 576
INPUT_DIM = EMB_DIM + 9 + 3
OUTPUT_DIM = 4  # dx, dy, log 배율, 각도/ANGLE_NORM
_STD_FLOOR = 1e-6


def head_input(emb: np.ndarray, p: Person, depth: DepthFeatures | None) -> np.ndarray:
    comp = compute_features(p).vector()
    box = np.array([p.box[2], p.box[3], p.conf])
    d = depth.vector() if depth is not None else np.zeros(3)
    return np.concatenate([np.asarray(emb, dtype=np.float64), comp, box, d]).astype(np.float32)


def correction_to_direction(c: np.ndarray) -> tuple[str, str]:
    """c는 화면 속 피사체가 움직여야 할 양이고, MOVES는 카메라 방향마다 피사체가 움직이는 양이다(LEFT=+dx)."""
    u = np.array([c[0] / MOVE_STEP, c[1] / MOVE_STEP, c[2] / math.log(ZOOM_IN)])
    i = int(np.argmax(np.abs(u)))
    if abs(u[i]) < HEAD_OK:
        return "OK", "SMALL"
    names = (("LEFT", "RIGHT"), ("UP", "DOWN"), ("CLOSER", "FARTHER"))[i]
    return names[0] if u[i] > 0 else names[1], "LARGE" if abs(u[i]) >= 2 else "SMALL"


def head_decide(c: np.ndarray, person: Person | None, stat_scorer: Scorer,
                noun: str = "인물") -> tuple[GuideDecision, float | None]:
    """보정값으로 안내를 정한다. 점수는 같은 작가의 stat 점수로 채운다. (결정, 회전 각도)."""
    if person is None:
        return GuideDecision("NO_SUBJECT", "SMALL", message_for("NO_SUBJECT", "SMALL", noun),
                             0.0, 0.0, 0.0, None, None), None
    direction, magnitude = correction_to_direction(c)
    s0 = float(stat_scorer.score(compute_features(person)))
    delta = 0.0
    if direction in MOVES:
        moved = transform_person(person, *MOVES[direction])
        if moved is not None:
            delta = float(stat_scorer.score(compute_features(moved))) - s0
    target = transform_person(person, float(c[0]), float(c[1]), float(math.exp(c[2])))
    d = GuideDecision(direction, magnitude, message_for(direction, magnitude, noun), s0, delta,
                      person.conf, person.box, target.box if target is not None else person.box)
    return d, float(-c[3] * ANGLE_NORM)


def _net() -> nn.Module:
    return nn.Sequential(nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, OUTPUT_DIM))


class GuideHead:
    def __init__(self, style_id: str, version: str, x_mean, x_std, y_mean, y_std, net: nn.Module):
        self.style_id, self.version = style_id, version
        self.x_mean, self.x_std, self.y_mean, self.y_std = x_mean, x_std, y_mean, y_std
        self.net = net.eval()

    @classmethod
    def fit(cls, style_id: str, X: np.ndarray, Y: np.ndarray, version: str, epochs: int = 300,
            seed: int = 0, val_frac: float = 0.2) -> tuple["GuideHead", list[dict]]:
        if len(X) < 5:
            raise ValueError("학습 샘플이 너무 적어요")
        torch.manual_seed(seed)
        idx = np.random.default_rng(seed).permutation(len(X))
        n_val = max(1, int(len(X) * val_frac))
        val, tr = idx[:n_val], idx[n_val:]
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        x_mean, x_std = x[tr].mean(0), x[tr].std(0).clamp_min(_STD_FLOOR)
        y_mean, y_std = y[tr].mean(0), y[tr].std(0).clamp_min(_STD_FLOOR)
        xn, yn = (x - x_mean) / x_std, (y - y_mean) / y_std
        net = _net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        history = []
        for ep in range(1, epochs + 1):
            net.train()
            opt.zero_grad()
            loss = loss_fn(net(xn[tr]), yn[tr])
            loss.backward()
            opt.step()
            if ep % 10 == 0:
                net.eval()
                with torch.no_grad():
                    pred = net(xn) * y_std + y_mean  # 원래 단위로 MSE를 기록한다
                history.append({"epoch": ep,
                                "train_mse": float(((pred[tr] - y[tr]) ** 2).mean()),
                                "val_mse": float(((pred[val] - y[val]) ** 2).mean())})
        return cls(style_id, version, x_mean, x_std, y_mean, y_std, net), history

    def predict(self, x: np.ndarray) -> np.ndarray:
        t = (torch.tensor(np.asarray(x), dtype=torch.float32) - self.x_mean) / self.x_std
        with torch.no_grad():
            out = self.net(t.reshape(-1, INPUT_DIM)) * self.y_std + self.y_mean
        out = out.numpy()
        return out[0] if np.asarray(x).ndim == 1 else out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"style_id": self.style_id, "version": self.version, "x_mean": self.x_mean,
                    "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std,
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "GuideHead":
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        net = _net()
        net.load_state_dict(d["state_dict"])
        return cls(d["style_id"], d["version"], d["x_mean"], d["x_std"], d["y_mean"], d["y_std"], net)
