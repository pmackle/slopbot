"""Runtime settings, read from environment variables (and `.env` via the CLI)."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL = "claude-opus-5-5"


class ConfigError(ValueError):
    """The environment is misconfigured in a way the user needs to fix."""


@dataclass(frozen=True)
class Settings:
    # Models (names as the API or gateway knows them)
    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    gen_model: str = DEFAULT_MODEL
    effort: str = "medium"
    max_tokens: int = 16000
    fallbacks: bool = True

    # Where requests go. With base_url set, api_key is a LiteLLM virtual key.
    base_url: str | None = None
    api_key: str | None = None

    # Cloud IAP in front of the gateway (LITELLM_IAP=1)
    iap: bool = False
    iap_service_account: str | None = None
    iap_audience: str | None = None

    braintrust_project: str = "Slopbot"
    braintrust_project_id: str | None = None
    braintrust_enabled: bool = False
    slack_bot_token: str | None = None
    slack_app_token: str | None = None

    @property
    def uses_gateway(self) -> bool:
        return bool(self.base_url)

    @property
    def braintrust_target(self) -> dict[str, str]:
        """Which Braintrust project to use: the ID when set (it wins over the name)."""
        if self.braintrust_project_id:
            return {"project_id": self.braintrust_project_id}
        return {"project": self.braintrust_project}


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _str(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def get_settings() -> Settings:
    effort = os.getenv("SLOPBOT_EFFORT", "medium").strip().lower()
    if effort not in VALID_EFFORTS:
        raise ConfigError(f"SLOPBOT_EFFORT must be one of {VALID_EFFORTS}, got {effort!r}")

    base_url = _str("ANTHROPIC_BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
    model = _str("SLOPBOT_MODEL") or DEFAULT_MODEL

    iap = _bool(os.getenv("LITELLM_IAP"), False)
    iap_service_account = _str("LITELLM_IAP_SERVICE_ACCOUNT")
    if iap and not base_url:
        raise ConfigError("LITELLM_IAP=1 needs ANTHROPIC_BASE_URL set to the gateway URL.")
    if iap and not iap_service_account:
        raise ConfigError(
            "LITELLM_IAP=1 needs LITELLM_IAP_SERVICE_ACCOUNT (the service account IAM signs as)."
        )

    return Settings(
        model=model,
        judge_model=_str("SLOPBOT_JUDGE_MODEL") or model,
        gen_model=_str("SLOPBOT_GEN_MODEL") or model,
        effort=effort,
        max_tokens=int(os.getenv("SLOPBOT_MAX_TOKENS", "16000")),
        # Anthropic's server-side refusal fallback uses a `fallbacks` request field, which LiteLLM
        # also reserves for its own router. Default it off behind a gateway; opt in if yours
        # passes it through.
        fallbacks=_bool(os.getenv("SLOPBOT_FALLBACKS"), not base_url),
        base_url=base_url,
        api_key=_str("ANTHROPIC_API_KEY"),
        iap=iap,
        iap_service_account=iap_service_account,
        # IAP matches a self-signed JWT's `aud` against the requested URL *including its path*,
        # so the bare gateway URL never matches; `<gateway>/*` covers every endpoint.
        iap_audience=_str("LITELLM_IAP_AUDIENCE") or (f"{base_url}/*" if base_url else None),
        braintrust_project=os.getenv("BRAINTRUST_PROJECT", "Slopbot"),
        braintrust_project_id=_str("BRAINTRUST_PROJECT_ID"),
        braintrust_enabled=bool(os.getenv("BRAINTRUST_API_KEY"))
        and _bool(os.getenv("SLOPBOT_BRAINTRUST_LOGGING"), True),
        slack_bot_token=_str("SLACK_BOT_TOKEN"),
        slack_app_token=_str("SLACK_APP_TOKEN"),
    )
