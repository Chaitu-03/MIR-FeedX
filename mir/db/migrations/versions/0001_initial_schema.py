"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-22

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Extensions — idempotent, safe if already created by init.sql
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # accounts
    op.create_table(
        "accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("blog_name", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("avatar_url", sa.Text()),
        sa.Column("total_posts", sa.Integer()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_accounts_blog_name_trgm",
        "accounts",
        ["blog_name"],
        postgresql_using="gin",
        postgresql_ops={"blog_name": "gin_trgm_ops"},
    )

    # posts
    op.create_table(
        "posts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tumblr_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("post_type", sa.Text()),
        sa.Column("body_raw", sa.Text()),
        sa.Column("body_clean", sa.Text()),
        sa.Column("image_urls", ARRAY(sa.Text())),
        sa.Column("note_count", sa.Integer(), server_default="0"),
        sa.Column("reblog_key", sa.Text()),
        sa.Column(
            "reblogged_from",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "reblogged_root",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("nsfw", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("lang", sa.Text()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        # ts_body is added below as a GENERATED column via raw SQL
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # SQLAlchemy ORM can't declare GENERATED ALWAYS AS — add via raw DDL.
    op.execute(
        """
        ALTER TABLE posts
        ADD COLUMN ts_body tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(body_clean, ''))) STORED
        """
    )

    op.create_index("ix_posts_ts_body", "posts", ["ts_body"], postgresql_using="gin")
    op.create_index("ix_posts_nsfw", "posts", ["nsfw"])

    # tags
    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("usage_count", sa.Integer(), server_default="0"),
        sa.Column("embedding", Vector(384), nullable=True),
    )

    # post_tags
    op.create_table(
        "post_tags",
        sa.Column(
            "post_id",
            sa.Integer(),
            sa.ForeignKey("posts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id",
            sa.Integer(),
            sa.ForeignKey("tags.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )

    # communities
    op.create_table(
        "communities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text()),
        sa.Column("centroid", Vector(384)),
        sa.Column("type", sa.Text()),
        sa.Column("member_ids", ARRAY(sa.Integer())),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )

    # crawl_state
    op.create_table(
        "crawl_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("blog_name", sa.Text(), nullable=False, unique=True),
        sa.Column("last_timestamp", sa.BigInteger()),
        sa.Column("last_crawled_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("fail_count", sa.Integer(), server_default="0"),
    )

    # api_keys
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("label", sa.Text()),
        sa.Column("is_admin", sa.Boolean(), server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("crawl_state")
    op.drop_table("communities")
    op.drop_table("post_tags")
    op.drop_table("tags")
    op.drop_index("ix_posts_nsfw", table_name="posts")
    op.drop_index("ix_posts_ts_body", table_name="posts")
    op.drop_table("posts")
    op.drop_index("ix_accounts_blog_name_trgm", table_name="accounts")
    op.drop_table("accounts")
    # Leave extensions installed — other databases/schemas may depend on them.
