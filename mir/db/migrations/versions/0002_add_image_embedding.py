"""Add image_embedding and nsfw_score to posts

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 384-d mean-pooled + projected CLIP embedding stored per post
    op.add_column(
        "posts",
        sa.Column("image_embedding", Vector(384), nullable=True),
    )
    # Max NSFW score across all images in the post; stored for SFW posts too
    # so the threshold can be tuned retrospectively without re-running CLIP.
    op.add_column(
        "posts",
        sa.Column("nsfw_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("posts", "nsfw_score")
    op.drop_column("posts", "image_embedding")
