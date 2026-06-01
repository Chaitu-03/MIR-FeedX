"""
tests/conftest.py
Session-wide test configuration.

Must be loaded before any mir.* imports so that:
  1. API_KEYS_ENABLED=true forces auth on — test suite validates 401/403 paths.
  2. Session-scoped asyncio event loop (set via pyproject.toml
     asyncio_default_fixture_loop_scope=session) prevents asyncpg
     "Future attached to different loop" errors.
"""
import os

# Force auth ON for the test session.
# Overrides the .env value (os.environ > .env in pydantic-settings v2 priority).
# conftest.py is loaded before test modules import mir.*, so settings singleton
# picks this up on first instantiation.
os.environ["API_KEYS_ENABLED"] = "true"
