from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.5")
    planner_api_key: str | None = os.getenv("PLANNER_API_KEY") or openrouter_api_key
    planner_base_url: str = os.getenv("PLANNER_BASE_URL", openrouter_base_url)
    planner_model: str = os.getenv("PLANNER_MODEL", "anthropic/claude-3.5-sonnet")
    reflector_api_key: str | None = os.getenv("REFLECTOR_API_KEY") or openrouter_api_key
    reflector_base_url: str = os.getenv("REFLECTOR_BASE_URL", openrouter_base_url)
    reflector_model: str = os.getenv("REFLECTOR_MODEL", "openai/gpt-5.1")
    enable_reflection: bool = _get_bool("ENABLE_REFLECTION", True)
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    encode_format: str = os.getenv("ENCODE_FORMAT", "JPEG")
    verify_delay_ms: int = int(os.getenv("VERIFY_DELAY_MS", "200"))
    max_steps: int = int(os.getenv("MAX_STEPS", "50"))
    max_failures: int = int(os.getenv("MAX_FAILURES", "5"))
    max_wall_clock_seconds: int | None = (
        int(os.getenv("MAX_WALL_CLOCK_SECONDS", "0")) or None
    )

    enable_hid: bool = _get_bool("ENABLE_HID", False)
    enable_semantic: bool = _get_bool("ENABLE_SEMANTIC", False)
    use_openrouter: bool = _get_bool("USE_OPENROUTER", True)
    planner_use_openrouter: bool = use_openrouter
    enable_embeddings: bool = _get_bool("ENABLE_EMBEDDINGS", False)
    memory_root: str | None = os.getenv("MEMORY_ROOT")
