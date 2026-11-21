"""The one Claude call that decides whether something is slop."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from braintrust import current_span, traced, wrap_anthropic
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .gateway import SPEND, make_anthropic_client
from .prompts import SYSTEM_PROMPT, wrap_content

FALLBACK_BETA = "server-side-fallback-2026-07-01"
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


class SlopVerdict(BaseModel):
    """Structured output Claude must return."""

    verdict: Literal["ai", "human", "mixed", "unsure"] = Field(
        description="ai, human, mixed (part AI, part human), or unsure (not enough signal)."
    )
    ai_probability: int = Field(
        description="Honest probability from 0 to 100 that the content is AI-generated."
    )
    reasons: list[str] = Field(
        description="1-3 short bullets, each citing specific evidence present in the content."
    )
    quip: str = Field(description="One deadpan sentence in Slopbot's voice, at most ~20 words.")


@dataclass(frozen=True)
class ImageInput:
    media_type: str
    data: bytes
    name: str = "image"


@dataclass
class DetectionResult:
    verdict: SlopVerdict
    served_model: str
    fell_back: bool
    span_id: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None  # as charged by the gateway; None when talking to Anthropic


class SlopbotError(Exception):
    """Detection failed in a way we can explain to the user."""


class DetectionRefused(SlopbotError):
    def __init__(self, category: str | None):
        super().__init__(f"Claude declined to judge this content (category: {category}).")
        self.category = category


_client: Any = None


def get_client(settings: Settings | None = None) -> Any:
    """Shared Anthropic client, wrapped so every call is traced in Braintrust (no-op when off).

    Points at the LiteLLM gateway when ANTHROPIC_BASE_URL is set (see gateway.py); otherwise
    Anthropic directly, with credentials from ANTHROPIC_API_KEY or an `ant auth login` profile.
    """
    global _client
    if _client is None:
        _client = wrap_anthropic(make_anthropic_client(settings or get_settings()))
    return _client


def fallback_kwargs(settings: Settings) -> dict[str, Any]:
    """Server-side refusal fallback: if the model declines, the API re-runs the request on a
    fallback model inside the same call. Off by default behind a gateway (see config.py)."""
    if not settings.fallbacks:
        return {}
    return {"betas": [FALLBACK_BETA], "fallbacks": "default"}


def build_messages(text: str | None, images: Sequence[ImageInput]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        if image.media_type not in SUPPORTED_IMAGE_TYPES:
            raise SlopbotError(f"Unsupported image type: {image.media_type}")
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64.standard_b64encode(image.data).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": wrap_content(text, len(images))})
    return [{"role": "user", "content": content}]


def build_request(
    text: str | None, images: Sequence[ImageInput], settings: Settings
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": build_messages(text, images),
        "output_format": SlopVerdict,
        "output_config": {"effort": settings.effort},
        **fallback_kwargs(settings),
    }
    return request


def _usage_dict(usage: Any) -> dict[str, Any]:
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    return {k: getattr(usage, k, None) for k in keys if getattr(usage, k, None) is not None}


@traced(name="slopbot.detect", notrace_io=True)
def detect(
    text: str | None,
    images: Sequence[ImageInput] = (),
    *,
    surface: str = "cli",
    settings: Settings | None = None,
    client: Any = None,
) -> DetectionResult:
    """Judge `text` and/or `images`. Raises SlopbotError on refusals or unusable output."""
    settings = settings or get_settings()
    client = client or get_client(settings)
    span = current_span()
    span.log(
        input={"text": text, "images": [i.name for i in images]},
        metadata={
            "surface": surface,
            "model": settings.model,
            "effort": settings.effort,
            "has_images": bool(images),
        },
    )

    SPEND.take_last()  # clear any stale value on this thread
    response = client.beta.messages.parse(**build_request(text, images, settings))
    cost_usd = SPEND.take_last()

    fell_back = any(
        getattr(entry, "type", None) == "fallback_message"
        for entry in (getattr(response.usage, "iterations", None) or [])
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise DetectionRefused(getattr(details, "category", None))
    if response.stop_reason == "max_tokens":
        raise SlopbotError("Ran out of tokens before reaching a verdict.")

    verdict = response.parsed_output
    if verdict is None:
        raise SlopbotError("Claude returned no parseable verdict.")

    result = DetectionResult(
        verdict=verdict,
        served_model=response.model,
        fell_back=fell_back,
        span_id=getattr(span, "id", None) or None,
        usage=_usage_dict(response.usage),
        cost_usd=cost_usd,
    )
    span.log(
        output=verdict.model_dump(),
        metadata={"served_model": response.model, "fell_back": fell_back, "cost_usd": cost_usd},
    )
    return result
