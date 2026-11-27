"""촬영 카테고리와 작가(스타일) 테이블, 작가별 모델 버전

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade():
    op.create_table(
        "categories",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
    )
    op.create_table(
        "styles",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("category_id", sa.String, sa.ForeignKey("categories.id"), nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("description", sa.String, nullable=False),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("model_version", sa.String, nullable=False),
    )


def downgrade():
    op.drop_table("styles")
    op.drop_table("categories")
