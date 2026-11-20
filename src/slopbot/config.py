"""Runtime settings, read from environment variables (and `.env` via the CLI)."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    fallbacks: bool = True
    braintrust_project: str = "Slopbot"
    braintrust_enabled: bool = False
    slack_bot_token: str | None = None
    slack_app_token: str | None = None


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def get_settings() -> Settings:
    effort = os.getenv("SLOPBOT_EFFORT", "medium").strip().lower()
    if effort not in VALID_EFFORTS:
        raise ValueError(f"SLOPBOT_EFFORT must be one of {VALID_EFFORTS}, got {effort!r}")
    return Settings(
        model=os.getenv("SLOPBOT_MODEL", "claude-opus-5-5"),
        effort=effort,
        max_tokens=int(os.getenv("SLOPBOT_MAX_TOKENS", "16000")),
        fallbacks=_bool(os.getenv("SLOPBOT_FALLBACKS"), True),
        braintrust_project=os.getenv("BRAINTRUST_PROJECT", "Slopbot"),
        braintrust_enabled=bool(os.getenv("BRAINTRUST_API_KEY"))
        and _bool(os.getenv("SLOPBOT_BRAINTRUST_LOGGING"), True),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
        slack_app_token=os.getenv("SLACK_APP_TOKEN"),
    )
