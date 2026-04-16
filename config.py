"""
Centralised configuration – every setting is loaded from the environment
(or a .env file) with sensible defaults.

Usage
-----
    from config import settings
    print(settings.LLM_MODEL)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file).
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _env(key: str, default: str | None = None) -> str:
    """Return an env-var value or *default*; raise if neither exists."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example → .env and fill in the blanks."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable application-wide settings."""

    # ── LLM Provider ────────────────────────────────────────────────
    LLM_PROVIDER: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "groq"),
    )

    # ── API keys ────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
    )
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
    )
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", ""),
    )

    # ── Models ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    LLM_MODEL: str = field(
        default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile"),
    )

    # ── Chunking ────────────────────────────────────────────────────
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(_env("CHUNK_SIZE", "512")),
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(_env("CHUNK_OVERLAP", "64")),
    )

    # ── Vector DB ───────────────────────────────────────────────────
    VECTOR_DB_PATH: str = field(
        default_factory=lambda: _env("VECTOR_DB_PATH", "./vector_store"),
    )

    # ── Logging ─────────────────────────────────────────────────────
    LOG_LEVEL: str = field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO"),
    )


# Singleton used throughout the code-base.
settings = Settings()
