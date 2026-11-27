"""추론 로그에 깊이·임베딩 단계 시간

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade():
    with op.batch_alter_table("inference_logs") as b:
        b.add_column(sa.Column("depth_ms", sa.Integer, nullable=False, server_default="0"))
        b.add_column(sa.Column("embed_ms", sa.Integer, nullable=False, server_default="0"))


def downgrade():
    with op.batch_alter_table("inference_logs") as b:
        b.drop_column("embed_ms")
        b.drop_column("depth_ms")
