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

    # Tumblr API credentials — list of {"consumer_key": ..., "secret": ...} pairs.
    # JSON-encoded in .env. Keys rotated round-robin; on rate-limit / auth failure
    # the offending key is put on cooldown and the next is tried.
    tumblr_api_credentials: list[dict[str, str]] = Field(default_factory=list)

    # Legacy: bare list of consumer keys. Used as fallback if credentials empty.
    tumblr_api_keys: list[str] = Field(default_factory=list)

    @field_validator("tumblr_api_credentials", mode="before")
    @classmethod
    def _parse_credentials(cls, v: object) -> list[dict[str, str]]:
        if isinstance(v, list):
            out = []
            for entry in v:
                if isinstance(entry, dict) and entry.get("consumer_key"):
                    out.append({
                        "consumer_key": str(entry["consumer_key"]).strip(),
                        "secret": str(entry.get("secret", "")).strip(),
                    })
            return out
        return []

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

    @property
    def all_consumer_keys(self) -> list[str]:
        """Unified accessor: prefer credentials, fall back to legacy api_keys."""
        if self.tumblr_api_credentials:
            return [c["consumer_key"] for c in self.tumblr_api_credentials]
        return list(self.tumblr_api_keys)

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

    # Beat: how often to enqueue crawl_active_blogs (env: CRAWL_INTERVAL_MINUTES).
    # Default 15 (verification). Staging 30–60, production 360.
    crawl_interval_minutes: Annotated[int, Field(gt=0)] = 15

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
