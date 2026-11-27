import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from axis_ai.artists import ARTIST_INFO, CATEGORIES, RULE_ARTISTS, RULE_DIAGRAM, learned_diagram
from axis_ai.bundle import ModelBundle
from axis_ai.recommend import recommend as recommend_styles
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from . import auth

from .config import Settings
from .db import InferenceLog, Shot, User, make_session_factory, sync_styles
from .pipeline import AnalysisPipeline, Catalog, InvalidImageError
from .schemas import (AuthIn, AuthOut, CategoryOut, DepthOut, GuideResult, SubjectOut, HealthOut, LoginIn, MeOut,
                      RecommendIn, ShotOut, StyleOut,
                      StyleScoreOut, TastePhotoOut, TimingOut)

logger = logging.getLogger(__name__)


def _is_jpeg(b: bytes) -> bool:
    return b[:3] == b"\xff\xd8\xff"


def _safe_path(base: Path, rel: str) -> Path:
    base = base.resolve()
    p = (base / rel).resolve()
    if not p.is_relative_to(base) or not p.is_file():
        raise HTTPException(404, "not found")
    return p


THUMB_MIN_W, THUMB_MAX_W = 64, 1200
KNOWN_SCORERS = {"stat", "rank", "head", "rule"}


def _thumbnail(src: Path, cache_root: Path, rel: str, w: int) -> Path:
    """긴 변이 아니라 너비가 w인 JPEG 썸네일을 만들어 캐시한다(비율 유지, EXIF 방향 적용)."""
    w = min(max(w, THUMB_MIN_W), THUMB_MAX_W)
    dest = (cache_root / str(w) / rel).resolve()
    if not dest.is_relative_to((cache_root / str(w)).resolve()):
        raise HTTPException(404, "not found")
    if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    try:
        with Image.open(src) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(415, "cannot decode image") from e
    if img.width > w:
        img = img.resize((w, max(1, round(img.height * w / img.width))), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
    img.save(tmp, "JPEG", quality=85)
    tmp.replace(dest)
    return dest


def _read_limited(f: UploadFile, limit: int) -> bytes:
    data = f.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "file too large")
    return data


def _subject_out(p) -> SubjectOut | None:
    if p is None:
        return None
    x, y, w, h = p.box
    return SubjectOut(class_name=p.label, cx=x + w / 2, cy=y + h / 2, w=w, h=h, area=w * h, conf=p.conf)


def _shot_out(r: Shot) -> ShotOut:
    ts = r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=timezone.utc)
    return ShotOut(id=r.id, style_id=r.style_id, score=r.score, guide_used=r.guide_used,
                   url=f"/media/shots/{r.file}", created_at=ts.isoformat())


class BundleCatalog:
    """학습된 모델 번들로 만든 카탈로그."""

    def __init__(self, bundle: ModelBundle, raw_dir: Path):
        self.bundle = bundle
        self.raw_dir = Path(raw_dir)

    def styles(self) -> list[dict]:
        return [{"id": s["id"], "name": s["name"], "description": s["description"],
                 "samples": [{"file": p["file"], "author": p["author"], "license": p["license"]}
                             for p in self.bundle.sample_photos(s["id"], 3)],
                 "diagram_box": learned_diagram(self.bundle.rows, s["id"])}
                for s in self.bundle.styles]

    def taste_photos(self) -> list[dict]:
        return list(self.bundle.taste)

    def recommend(self, image_ids: list[str]) -> list[tuple[str, float]]:
        liked = [self.bundle.photo_features(i) for i in image_ids]  # 모르는 id만 KeyError(404)
        try:
            return recommend_styles(liked, self.bundle.scorers["stat"], self.bundle.centroids)
        except KeyError as e:  # 번들 내부 불일치(centroid 없음 등)는 서버 오류(500)
            raise RuntimeError(f"모델 번들이 일관되지 않아요: {e}") from e


def all_styles(catalog: Catalog, model_version: str) -> list[dict]:
    """카탈로그(학습 작가)에 화면용 이름·카테고리를 입히고 규칙 작가를 붙인다."""
    out = []
    for s in catalog.styles():
        info = ARTIST_INFO.get(s["id"], {})
        out.append({**s, "name": info.get("name", s["name"]),
                    "description": info.get("description", s["description"]),
                    "category": info.get("category", "person"), "kind": "learned",
                    "model_version": model_version, "diagram_box": s.get("diagram_box")})
    for sid in RULE_ARTISTS:
        info = ARTIST_INFO[sid]
        out.append({"id": sid, "name": info["name"], "description": info["description"],
                    "category": info["category"], "kind": "rule", "model_version": "rule-v0",
                    "diagram_box": RULE_DIAGRAM[sid], "samples": []})
    return out


def create_app(settings: Settings, pipeline: AnalysisPipeline, catalog: Catalog) -> FastAPI:
    var_dir = Path(settings.var_dir)
    shots_dir = var_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    Session = make_session_factory(var_dir / "axis.db")
    secret = auth.load_secret(var_dir)
    app = FastAPI(title="AXIS")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def current_user(authorization: str | None) -> User:
        uid = auth.read_token(auth.bearer(authorization), secret)
        with Session() as s:
            user = s.get(User, uid)
        if user is None:
            raise HTTPException(401, "로그인이 필요해요")
        return user

    @app.post("/auth/signup", response_model=AuthOut)
    def signup(body: AuthIn):
        user = User(username=body.username, password_hash=auth.hash_password(body.password),
                    created_at=datetime.now(timezone.utc))
        with Session() as s:
            s.add(user)
            try:
                s.commit()
            except IntegrityError:
                raise HTTPException(409, "이미 있는 아이디예요")
        return AuthOut(token=auth.make_token(user.id, secret), username=user.username)

    @app.post("/auth/login", response_model=AuthOut)
    def login(body: LoginIn):
        with Session() as s:
            user = s.scalars(select(User).where(User.username == body.username)).first()
        if user is None or not auth.verify_password(body.password, user.password_hash):
            raise HTTPException(401, "아이디나 비밀번호가 틀려요")
        return AuthOut(token=auth.make_token(user.id, secret), username=user.username)

    @app.get("/me", response_model=MeOut)
    def me(authorization: str | None = Header(None)):
        u = current_user(authorization)
        return MeOut(id=u.id, username=u.username)

    @app.get("/health", response_model=HealthOut)
    def health():
        return HealthOut(status="ok", model_version=pipeline.model_version,
                         scorers=sorted(pipeline.available_scorers))

    styles_now = all_styles(catalog, pipeline.model_version)
    sync_styles(Session, CATEGORIES, styles_now)
    style_category = {s["id"]: s["category"] for s in styles_now}
    style_ids = set(style_category)

    @app.get("/categories", response_model=list[CategoryOut])
    def categories():
        return CATEGORIES

    @app.get("/styles", response_model=list[StyleOut])
    def styles(category: str | None = None):
        return [StyleOut(**{**s, "samples": [dict(url=f"/media/raw/{x['file']}", author=x["author"],
                                                  license=x["license"]) for x in s["samples"]]})
                for s in all_styles(catalog, pipeline.model_version)
                if category is None or s["category"] == category]

    @app.get("/taste/photos", response_model=list[TastePhotoOut])
    def taste_photos():
        return [TastePhotoOut(id=p["image_id"], url=f"/media/raw/{p['file']}")
                for p in catalog.taste_photos()]

    @app.post("/recommend", response_model=list[StyleScoreOut])
    def recommend(body: RecommendIn):
        try:
            ranked = catalog.recommend(body.liked_photo_ids)
        except KeyError as e:
            raise HTTPException(404, f"unknown photo id: {e.args[0]}")
        return [StyleScoreOut(style_id=s, score=sc) for s, sc in ranked]

    @app.post("/analyze", response_model=GuideResult)
    def analyze(frame: UploadFile = File(...), style_id: str = Form(...),
                      scorer: str | None = Form(None), depth_preview: bool = Form(False)):
        data = _read_limited(frame, settings.max_upload_bytes)
        if not _is_jpeg(data):
            raise HTTPException(400, "frame must be JPEG")
        if style_id not in style_ids:
            raise HTTPException(404, "unknown style_id")
        scorer = scorer or settings.default_scorer
        if scorer in KNOWN_SCORERS and scorer not in pipeline.available_scorers:
            # 앱에 저장된 방식이 이 서버에 없으면(예: rule 모드, head 없는 번들) 기본 방식으로 대신한다
            scorer = settings.default_scorer
        if scorer not in pipeline.available_scorers and style_id not in RULE_ARTISTS:
            raise HTTPException(400, f"scorer not available: {scorer}")
        t0 = time.perf_counter()
        try:
            res = pipeline.analyze(data, style_id, scorer, category=style_category[style_id],
                                   depth_preview=depth_preview)
        except InvalidImageError as e:
            raise HTTPException(400, "cannot decode frame") from e
        except Exception:
            logger.exception("analyze failed")
            raise
        total = int((time.perf_counter() - t0) * 1000)
        timing = {"depth": 0, "embed": 0, **res.timing_ms, "total": total}
        d = res.decision
        scorer = res.scorer_used or scorer
        with Session() as s:
            s.add(InferenceLog(created_at=datetime.now(timezone.utc), style_id=style_id,
                               scorer=scorer, direction=d.direction, score=d.score,
                               confidence=d.confidence, decode_ms=timing["decode"],
                               pose_ms=timing["pose"], guide_ms=timing["guide"],
                               depth_ms=timing["depth"], embed_ms=timing["embed"], total_ms=total))
            s.commit()
        return GuideResult(direction=d.direction, magnitude=d.magnitude, message=d.message,
                           score=d.score, score_delta=d.score_delta, confidence=d.confidence,
                           subject_box=d.subject_box, target_box=d.target_box, scorer=scorer,
                           model_version=pipeline.model_version, timing_ms=TimingOut(**timing),
                           category=res.category, subject=_subject_out(res.subject),
                           rotation_deg=res.rotation_deg, rotate=res.rotate,
                           depth=DepthOut(**vars(res.depth)) if res.depth else None,
                           depth_preview=res.depth_preview)

    @app.post("/shots", response_model=ShotOut)
    def post_shot(photo: UploadFile = File(...), style_id: str = Form(...), score: float = Form(...),
                  guide_used: bool = Form(...), authorization: str | None = Header(None)):
        user = current_user(authorization)
        data = _read_limited(photo, settings.max_shot_bytes)
        if not _is_jpeg(data):
            raise HTTPException(400, "photo must be JPEG")
        if style_id not in style_ids:
            raise HTTPException(404, "unknown style_id")
        name = f"{uuid.uuid4()}.jpg"
        (shots_dir / name).write_bytes(data)
        shot = Shot(id=name[:-4], user_id=user.id, style_id=style_id, score=score,
                    guide_used=guide_used, file=name, created_at=datetime.now(timezone.utc))
        with Session() as s:
            s.add(shot)
            s.commit()
        return _shot_out(shot)

    @app.get("/shots", response_model=list[ShotOut])
    def list_shots(style_id: str | None = None, authorization: str | None = Header(None)):
        user = current_user(authorization)
        q = select(Shot).where(Shot.user_id == user.id)
        if style_id is not None:
            q = q.where(Shot.style_id == style_id)
        with Session() as s:
            rows = s.scalars(q
                             .order_by(Shot.created_at.desc())).all()
        return [_shot_out(r) for r in rows]

    @app.get("/media/raw/{path:path}")
    def media_raw(path: str, w: int | None = None):
        src = _safe_path(Path(catalog.raw_dir), path)
        if w is None:
            return FileResponse(src)
        return FileResponse(_thumbnail(src, var_dir / "thumbs", path, w), media_type="image/jpeg")

    @app.get("/media/shots/{path:path}")
    def media_shots(path: str):
        return FileResponse(_safe_path(shots_dir, path))

    return app
