"""서버 실행: AXIS_MODE=fake|rule|model python -m axis_server --port 8000"""
import argparse
import sys

import uvicorn
from axis_ai.bundle import ModelBundle
from axis_ai.detect import FoodDetector
from axis_ai.pose import PoseDetector
from axis_ai.scorers import RULE_PRESETS

from .config import Settings
from .fake_seed import make_fake_catalog
from .main import BundleCatalog, create_app
from .pipeline import FakePipeline, RealPipeline


def _depth_factory():
    from axis_ai.depth import DepthEstimator

    return DepthEstimator()


def build_components(settings: Settings, detector_factory=PoseDetector, food_detector_factory=FoodDetector,
                     depth_factory=_depth_factory):
    """모드에 맞는 (pipeline, catalog, settings)를 만든다."""
    if settings.mode == "fake":
        return FakePipeline(), make_fake_catalog(settings.var_dir / "fake_media"), settings
    if settings.mode == "rule":
        catalog = make_fake_catalog(settings.var_dir / "fake_media")
        pipeline = RealPipeline({"rule": RULE_PRESETS}, "rule-v0", detector_factory(),
                                food_detector_factory(), depth_factory() if settings.depth else None)
        return pipeline, catalog, settings.model_copy(update={"default_scorer": "rule"})
    if settings.models_dir is None:
        sys.exit("AXIS_MODELS_DIR를 지정하세요")
    bundle = ModelBundle.load(settings.models_dir)
    embedder = None
    if bundle.heads:
        from axis_ai.embed import Embedder

        embedder = Embedder()
    pipeline = RealPipeline(bundle.scorers, bundle.version, detector_factory(), food_detector_factory(),
                            depth_factory() if settings.depth else None, heads=bundle.heads,
                            embedder=embedder)
    return pipeline, BundleCatalog(bundle, settings.raw_dir), settings


def check_default_scorer(pipeline, settings: Settings) -> None:
    """기본 scorer를 쓸 수 없으면 시작할 때 바로 알린다(요청마다 400이 나는 것을 막는다)."""
    if settings.default_scorer not in pipeline.available_scorers:
        sys.exit(f"기본 scorer '{settings.default_scorer}'를 쓸 수 없어요. "
                 f"사용 가능: {', '.join(sorted(pipeline.available_scorers)) or '없음'}. "
                 "AXIS_DEFAULT_SCORER를 바꾸거나 모델을 다시 학습하세요(rank는 학습 때 --rank 필요)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AXIS 서버")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    pipeline, catalog, settings = build_components(Settings())
    check_default_scorer(pipeline, settings)
    if isinstance(pipeline, RealPipeline):
        pipeline.warmup()
    uvicorn.run(create_app(settings, pipeline, catalog), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
