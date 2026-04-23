from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    blog_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    avatar_url: Mapped[str | None] = mapped_column(Text)
    total_posts: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    posts: Mapped[list[Post]] = relationship(
        back_populates="account", foreign_keys="Post.account_id"
    )

    __table_args__ = (
        Index(
            "ix_accounts_blog_name_trgm",
            "blog_name",
            postgresql_using="gin",
            postgresql_ops={"blog_name": "gin_trgm_ops"},
        ),
    )


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tumblr_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False
    )
    post_type: Mapped[str | None] = mapped_column(Text)
    body_raw: Mapped[str | None] = mapped_column(Text)
    body_clean: Mapped[str | None] = mapped_column(Text)
    image_urls: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    note_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    reblog_key: Mapped[str | None] = mapped_column(Text)
    reblogged_from: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    reblogged_root: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    nsfw: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    # Max NSFW score across all images; stored even for SFW posts so the
    # threshold can be tuned without re-running CLIP.
    nsfw_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    lang: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Mean-pooled + projected CLIP embedding (384-d) stored after image processing.
    image_embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    # ts_body is a generated column populated by Postgres via the Alembic migration:
    #   GENERATED ALWAYS AS (to_tsvector('english', coalesce(body_clean, ''))) STORED
    # Declared here as a plain column so SQLAlchemy can read it; Postgres owns writes.
    ts_body: Mapped[str | None] = mapped_column(TSVECTOR)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    account: Mapped[Account] = relationship(
        back_populates="posts", foreign_keys=[account_id]
    )

    __table_args__ = (
        Index("ix_posts_ts_body", "ts_body", postgresql_using="gin"),
        Index("ix_posts_nsfw", "nsfw"),
    )


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)


class PostTag(Base):
    __tablename__ = "post_tags"

    post_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )


class Community(Base):
    __tablename__ = "communities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    centroid: Mapped[list[float] | None] = mapped_column(Vector(384))
    type: Mapped[str | None] = mapped_column(Text)  # 'tag_cluster' | 'account_cluster'
    member_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CrawlState(Base):
    __tablename__ = "crawl_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    blog_name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    last_timestamp: Mapped[int | None] = mapped_column(BigInteger)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 'pending' | 'active' | 'done' | 'dead' | 'paused'
    status: Mapped[str] = mapped_column(Text, default="pending", server_default="pending")
    fail_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    label: Mapped[str | None] = mapped_column(Text)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),)


__all__ = [
    "Base",
    "Account",
    "Post",
    "Tag",
    "PostTag",
    "Community",
    "CrawlState",
    "APIKey",
]
