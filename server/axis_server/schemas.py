from typing import Literal

from pydantic import BaseModel, Field

USERNAME_PATTERN = r"^[A-Za-z0-9_가-힣]{3,20}$"

Direction = Literal["LEFT", "RIGHT", "UP", "DOWN", "CLOSER", "FARTHER", "OK", "NO_SUBJECT"]
BoxOut = tuple[float, float, float, float]


class SampleOut(BaseModel):
    url: str
    author: str
    license: str


class StyleOut(BaseModel):
    id: str
    name: str
    description: str
    samples: list[SampleOut]
    category: str = "person"
    kind: Literal["learned", "rule"] = "learned"
    model_version: str = ""
    diagram_box: BoxOut | None = None


class CategoryOut(BaseModel):
    id: str
    name: str


class TastePhotoOut(BaseModel):
    id: str
    url: str


class StyleScoreOut(BaseModel):
    style_id: str
    score: float


class RecommendIn(BaseModel):
    liked_photo_ids: list[str] = Field(min_length=1)


class TimingOut(BaseModel):
    decode: int
    pose: int
    guide: int
    total: int
    depth: int = 0
    embed: int = 0


class DepthOut(BaseModel):
    subject: float
    background: float
    diff: float


class SubjectOut(BaseModel):
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    area: float
    conf: float


class GuideResult(BaseModel):
    direction: Direction
    magnitude: Literal["SMALL", "LARGE"]
    message: str
    score: float
    score_delta: float
    confidence: float
    subject_box: BoxOut | None
    target_box: BoxOut | None
    scorer: str
    model_version: str
    timing_ms: TimingOut
    category: str = "person"
    subject: SubjectOut | None = None
    rotation_deg: float | None = None
    rotate: Literal["CW", "CCW"] | None = None
    depth: DepthOut | None = None
    depth_preview: str | None = None


class ShotOut(BaseModel):
    id: str
    style_id: str
    score: float
    guide_used: bool
    url: str
    created_at: str


class HealthOut(BaseModel):
    status: str
    model_version: str
    scorers: list[str]


class AuthIn(BaseModel):
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=6, max_length=64)


class LoginIn(BaseModel):
    username: str
    password: str


class AuthOut(BaseModel):
    token: str
    username: str


class MeOut(BaseModel):
    id: int
    username: str
