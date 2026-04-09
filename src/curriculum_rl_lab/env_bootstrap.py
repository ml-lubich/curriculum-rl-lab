"""Load `.env` and map MiniMax settings for OpenAI-compatible clients."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_minimax_env() -> None:
    """Load environment variables from the repo `.env` (and cwd) if present."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError("python-dotenv is required to load .env files") from exc

    load_dotenv()
    load_dotenv(_repo_root() / ".env")

    minimax_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if minimax_key:
        os.environ.setdefault("OPENAI_API_KEY", minimax_key)
    base = os.environ.get("MINIMAX_BASE_URL", "").strip()
    if base:
        os.environ.setdefault("OPENAI_BASE_URL", base)
        os.environ.setdefault("OPENAI_API_BASE", base)
