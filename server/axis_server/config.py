from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AXIS_")

    models_dir: Path | None = None
    raw_dir: Path = Path("data/raw")
    var_dir: Path = Path("server/var")
    default_scorer: str = "stat"
    mode: Literal["fake", "rule", "model"] = "model"
    max_upload_bytes: int = 5_000_000
    max_shot_bytes: int = 15_000_000
    depth: bool = True  # AXIS_DEPTH=0이면 깊이 모델을 불러오지 않는다
