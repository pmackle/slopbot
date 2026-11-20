from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from slopbot.detector import SlopVerdict


def png_bytes(size: tuple[int, int] = (64, 48), mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color=(200, 100, 50) if mode == "RGB" else (200, 100, 50, 128)).save(
        out, format="PNG"
    )
    return out.getvalue()


def make_verdict(**overrides: Any) -> SlopVerdict:
    data = {
        "verdict": "ai",
        "ai_probability": 92,
        "reasons": ['Opens with "delve into the rich tapestry"', "Three tidy lists of three"],
        "quip": "More em dashes than a Victorian novel.",
    }
    data.update(overrides)
    return SlopVerdict(**data)


def make_response(
    verdict: SlopVerdict | None = None,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    iterations: list[Any] | None = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=verdict,
        model=model,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300, iterations=iterations),
    )


class FakeMessages:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: Any):
        self.messages = FakeMessages(response)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def verdict() -> SlopVerdict:
    return make_verdict()
