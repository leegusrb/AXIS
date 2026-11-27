"""11/16 MVP 스키마: 촬영 기록(기기 단위)과 추론 로그

Revision ID: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade():
    op.create_table(
        "shots",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("device_id", sa.String, nullable=False, index=True),
        sa.Column("style_id", sa.String, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("guide_used", sa.Boolean, nullable=False),
        sa.Column("file", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "inference_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("style_id", sa.String, nullable=False),
        sa.Column("scorer", sa.String, nullable=False),
        sa.Column("direction", sa.String, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("decode_ms", sa.Integer, nullable=False),
        sa.Column("pose_ms", sa.Integer, nullable=False),
        sa.Column("guide_ms", sa.Integer, nullable=False),
        sa.Column("total_ms", sa.Integer, nullable=False),
    )


def downgrade():
    op.drop_table("inference_logs")
    op.drop_table("shots")
