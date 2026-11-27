from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class Style(Base):
    __tablename__ = "styles"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    category_id: Mapped[str] = mapped_column(ForeignKey("categories.id"))
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)  # learned | rule
    model_version: Mapped[str] = mapped_column(String)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Shot(Base):
    __tablename__ = "shots"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    device_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)  # 로그인 전 기록
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    style_id: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    guide_used: Mapped[bool] = mapped_column(Boolean)
    file: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    style_id: Mapped[str] = mapped_column(String)
    scorer: Mapped[str] = mapped_column(String)
    direction: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    decode_ms: Mapped[int] = mapped_column(Integer)
    pose_ms: Mapped[int] = mapped_column(Integer)
    guide_ms: Mapped[int] = mapped_column(Integer)
    depth_ms: Mapped[int] = mapped_column(Integer, default=0)
    embed_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[int] = mapped_column(Integer)


_MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def migrate(db_path) -> None:
    """Alembic으로 최신 스키마까지 올린다. 11/16 스키마로 만든 기존 DB는 0001로 표시한 뒤 올린다."""
    from alembic import command
    from alembic.config import Config

    url = f"sqlite:///{db_path}"
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    if "shots" in tables and "alembic_version" not in tables:
        command.stamp(cfg, "0001")
    command.upgrade(cfg, "head")


def make_session_factory(db_path) -> sessionmaker:
    migrate(db_path)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    return sessionmaker(engine, expire_on_commit=False)


def sync_styles(Session: sessionmaker, categories: list[dict], styles: list[dict]) -> None:
    """서버 시작 시 카테고리·작가 목록과 모델 버전을 DB에 맞춘다(있으면 갱신)."""
    with Session() as s:
        for c in categories:
            s.merge(Category(id=c["id"], name=c["name"]))
        s.flush()
        for st in styles:
            s.merge(Style(id=st["id"], category_id=st["category"], name=st["name"],
                          description=st["description"], kind=st["kind"], model_version=st["model_version"]))
        s.commit()
