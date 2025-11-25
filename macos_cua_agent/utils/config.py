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
