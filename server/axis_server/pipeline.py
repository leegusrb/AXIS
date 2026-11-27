import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axis_ai.artists import RULE_ARTISTS
from axis_ai.depth import DepthFeatures, depth_features, preview_png
from axis_ai.features import select_subject
from dataclasses import replace

from axis_ai.guide import GuideDecision, decide, message_for
from axis_ai.head.model import head_decide, head_input
from axis_ai.rotation import OK_TILTED_MESSAGE, estimate_tilt, rotate_hint
from axis_ai.types import Person
from axis_ai.scorers import Scorer
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError


class InvalidImageError(ValueError):
    """analyze()가 디코드할 수 없는 입력을 받았을 때 던진다(클라이언트 오류)."""


@dataclass
class AnalysisResult:
    decision: GuideDecision
    timing_ms: dict[str, int]  # decode, pose, guide
    scorer_used: str = ""  # 실제로 쓴 평가 방식(규칙 작가는 요청과 상관없이 "rule")
    subject: Person | None = None
    rotation_deg: float | None = None
    rotate: str | None = None  # 휴대폰을 돌릴 방향 CW/CCW
    category: str = "person"
    depth: DepthFeatures | None = None
    depth_preview: str | None = None  # base64 PNG (깊이 보기를 켰을 때만)


class AnalysisPipeline(Protocol):
    model_version: str
    available_scorers: set[str]

    def analyze(self, jpeg: bytes, style_id: str, scorer: str, *, category: str = "person",
                depth_preview: bool = False) -> AnalysisResult:
        """디코드할 수 없는 입력이면 InvalidImageError를 던진다. 그 밖의 예외는 서버 오류(500)다."""
        ...


class Catalog(Protocol):
    raw_dir: Path

    def styles(self) -> list[dict]: ...  # id, name, description, samples[{file, author, license}]
    def taste_photos(self) -> list[dict]: ...  # {image_id, file}
    def recommend(self, image_ids: list[str]) -> list[tuple[str, float]]: ...  # KeyError: 알 수 없는 id


_DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "CLOSER", "FARTHER", "OK", "NO_SUBJECT"]


class FakePipeline:
    """밝기 평균으로 방향을 결정적으로 고르는 가짜 파이프라인(테스트·앱 개발용)."""

    model_version = "fake-0"
    available_scorers = {"stat"}

    def analyze(self, jpeg: bytes, style_id: str, scorer: str, *, category: str = "person",
                depth_preview: bool = False) -> AnalysisResult:
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("L")
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError(str(e)) from e
        brightness = ImageStat.Stat(img).mean[0]
        direction = _DIRECTIONS[min(int(brightness / 256 * len(_DIRECTIONS)), len(_DIRECTIONS) - 1)]
        magnitude = "LARGE" if int(brightness) % 2 else "SMALL"
        noun = "음식" if category == "food" else "인물"
        if direction == "NO_SUBJECT":
            decision = GuideDecision(direction, "SMALL", message_for(direction, "SMALL", noun),
                                     0.0, 0.0, 0.0, None, None)
            return AnalysisResult(decision, {"decode": 1, "pose": 0, "guide": 0}, scorer, category=category)
        subject = (0.31, 0.12, 0.28, 0.71)
        target = subject if direction == "OK" else (0.40, 0.10, 0.28, 0.71)
        decision = GuideDecision(direction, magnitude, message_for(direction, magnitude, noun),
                                 0.54, 0.0 if direction == "OK" else 0.07, 0.82, subject, target)
        return AnalysisResult(decision, {"decode": 1, "pose": 1, "guide": 1},
                              "rule" if style_id in RULE_ARTISTS else scorer, category=category)


class RealPipeline:
    """YOLO 검출 + scorer(또는 Guide Head)로 안내를 계산한다. scorers[종류][style_id]."""

    def __init__(self, scorers: dict[str, dict[str, Scorer]], model_version: str, detector,
                 food_detector=None, depth=None, heads: dict | None = None, embedder=None):
        self._scorers = scorers
        self.model_version = model_version
        self._heads = heads or {}
        self._embedder = embedder
        self.available_scorers = {k for k, v in scorers.items() if v}
        if self._heads and embedder is not None:
            self.available_scorers.add("head")
        self._detector = detector
        self._food_detector = food_detector
        self._depth = depth
        self._depth_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._detect_lock = threading.Lock()  # 스레드풀 핸들러가 YOLO 예측기 하나를 공유한다

    def warmup(self) -> None:
        with self._detect_lock:
            for d in (self._detector, self._food_detector):
                if d is not None:
                    d.detect(Image.new("RGB", (320, 320), (0, 0, 0)))

    def _resolve(self, style_id: str, scorer: str) -> str:
        """실제로 쓸 방식. 규칙 작가는 rule, head가 없는 작가는 stat으로 대신한다."""
        if style_id in RULE_ARTISTS:
            return "rule"
        if scorer == "head" and (style_id not in self._heads or self._embedder is None):
            return "stat"
        return scorer

    def analyze(self, jpeg: bytes, style_id: str, scorer: str, *, category: str = "person",
                depth_preview: bool = False) -> AnalysisResult:
        ms = lambda a, b: int((b - a) * 1000)  # noqa: E731
        t0 = time.perf_counter()
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise InvalidImageError(str(e)) from e
        t1 = time.perf_counter()
        food = category == "food"
        noun = "음식" if food else "인물"
        detector = self._food_detector if food else self._detector
        with self._detect_lock:
            persons = detector.detect(img) if detector is not None else []
        t2 = time.perf_counter()
        subject = select_subject(persons)
        used = self._resolve(style_id, scorer)

        dmap = None
        if self._depth is not None and (depth_preview or (used == "head" and subject is not None)):
            with self._depth_lock:
                dmap = self._depth.estimate(img)
        t3 = time.perf_counter()
        depth = depth_features(dmap, subject.box) if dmap is not None and subject is not None else None

        embed_ms = 0
        if used == "head" and subject is not None:
            te = time.perf_counter()
            with self._embed_lock:
                emb = self._embedder.embed(img)
            embed_ms = ms(te, time.perf_counter())
            c = self._heads[style_id].predict(head_input(emb, subject, depth))
            decision, tilt = head_decide(c, subject, self._scorers["stat"][style_id], noun)
        else:
            kind = "stat" if used == "head" else used  # head인데 인물이 없을 때
            sc = RULE_ARTISTS[style_id] if style_id in RULE_ARTISTS else self._scorers[kind][style_id]
            decision = decide(subject, sc, noun)
            tilt = estimate_tilt(subject) if subject is not None else None
        rotate = rotate_hint(tilt)
        if rotate and decision.direction == "OK":  # OK 판정은 그대로, 문구만 수평을 권한다
            decision = replace(decision, message=OK_TILTED_MESSAGE)
        t4 = time.perf_counter()
        return AnalysisResult(decision, {"decode": ms(t0, t1), "pose": ms(t1, t2),
                                         "guide": max(ms(t3, t4) - embed_ms, 0), "depth": ms(t2, t3),
                                         "embed": embed_ms}, used,
                              subject=subject, rotation_deg=tilt, rotate=rotate, category=category,
                              depth=depth, depth_preview=preview_png(dmap) if depth_preview and dmap is not None
                              else None)
