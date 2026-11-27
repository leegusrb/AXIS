"""회원(users)과 사용자별 촬영 기록

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("username", sa.String, nullable=False, unique=True, index=True),
        sa.Column("password_hash", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    with op.batch_alter_table("shots") as b:
        b.alter_column("device_id", existing_type=sa.String, nullable=True)
        b.add_column(sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", name="fk_shots_user"),
                               nullable=True))
        b.create_index("ix_shots_user_id", ["user_id"])


def downgrade():
    with op.batch_alter_table("shots") as b:
        b.drop_index("ix_shots_user_id")
        b.drop_column("user_id")
        b.alter_column("device_id", existing_type=sa.String, nullable=False)
    op.drop_table("users")
