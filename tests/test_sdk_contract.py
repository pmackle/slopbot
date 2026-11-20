"""Run detect() through the real Anthropic SDK against a mock HTTP transport.

Catches SDK-level problems (bad parameter names, schema generation, response parsing) that the
fake-client tests can't, without spending tokens.
"""

import json

import anthropic
import httpx2
import pytest
from braintrust import wrap_anthropic

from slopbot.config import Settings
from slopbot.detector import FALLBACK_BETA, detect

VERDICT = {
    "verdict": "ai",
    "ai_probability": 88,
    "reasons": ['"delve into the rich tapestry"'],
    "quip": "Synergy detected.",
}


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "braintrust-wrapped"])
def test_detect_through_real_sdk(wrapped):
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": json.dumps(VERDICT)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 900, "output_tokens": 120},
            },
        )

    client = anthropic.Anthropic(
        api_key="sk-ant-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    if wrapped:
        client = wrap_anthropic(client)
    result = detect(
        "We must delve into the rich tapestry of synergy.",
        settings=Settings(effort="medium"),
        client=client,
    )

    body = captured["body"]
    assert captured["url"].split("?")[0].endswith("/v1/messages")
    assert FALLBACK_BETA in captured["headers"]["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "medium"
    schema = body["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"verdict", "ai_probability", "reasons", "quip"}
    assert schema["properties"]["verdict"]["enum"] == ["ai", "human", "mixed", "unsure"]

    assert result.verdict.model_dump() == VERDICT
    assert result.usage == {"input_tokens": 900, "output_tokens": 120}
