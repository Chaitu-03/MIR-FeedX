from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://mir:mir@localhost:5432/mir"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # Tumblr API keys — comma-separated in env, parsed into a list
    tumblr_api_keys: list[str] = Field(default_factory=list)

    @field_validator("tumblr_api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v: object) -> list[str]:
        # pydantic-settings v2 JSON-decodes list fields from .env before this runs,
        # so v may already be a list. Fall back to comma-split for plain strings.
        if isinstance(v, list):
            return [str(k).strip() for k in v if str(k).strip()]
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return []

    # NSFW filtering
    nsfw_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    nsfw_blocklist_path: Path = Path("data/nsfw_blocklist.txt")

    # Security
    api_keys_enabled: bool = True

    # Embedding / ML
    account_embedding_window: Annotated[int, Field(gt=0)] = 100

    # Post embedding fusion weights (text, image, tags).
    # Must sum to 1.0; validated and tuned by the eval harness (Prompt 16b).
    post_weight_text: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    post_weight_image: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    post_weight_tag: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15

    # Crawl stop condition
    # Crawling stops completely once this many SFW, non-video posts are indexed.
    # Videos and NSFW posts filtered during ingestion do NOT count toward this total.
    # The live count is checked via: SELECT COUNT(*) FROM posts WHERE nsfw = false.
    target_post_count: Annotated[int, Field(gt=0)] = 50_000

    # Model artefacts
    models_dir: Path = Path("models")
    clip_miniLM_projection_path: Path = Path("models/clip_to_miniLM_projection.pt")

    # fasttext language-identification model (LID-176)
    # Download: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fasttext_model_path: Path = Path("data/lid.176.bin")

    # Search tuning — validated by eval harness (Prompt 16b)
    rrf_k: int = 60                         # RRF constant
    note_count_boost: float = 0.1           # engagement boost coefficient
    recency_half_life_days: float = 90.0    # recency decay half-life in days

    # Logging
    log_level: str = "INFO"
    log_json: bool = False  # JSON renderer in prod, console for dev

    # Cache (Redis)
    cache_ttl_seconds: int = 300
    cache_key_prefix: str = "mir:cache:"


settings = Settings()
