import json as jsonlib
from types import SimpleNamespace

import httpx
import httpx2
import pytest

from slopbot.config import ConfigError, Settings, get_settings
from slopbot.detector import FALLBACK_BETA, detect
from slopbot.gateway import (
    JWT_LIFETIME_S,
    REFRESH_MARGIN_S,
    GatewayError,
    IapJwtSource,
    fetch_gateway_models,
    make_anthropic_client,
    verify_models,
)

SA = "bot@example-project.iam.gserviceaccount.com"
GATEWAY = "https://gateway.example.com"
ENV_VARS = [
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "LITELLM_IAP",
    "LITELLM_IAP_SERVICE_ACCOUNT",
    "LITELLM_IAP_AUDIENCE",
    "SLOPBOT_MODEL",
    "SLOPBOT_JUDGE_MODEL",
    "SLOPBOT_GEN_MODEL",
    "SLOPBOT_FALLBACKS",
]


class FakeSession:
    """Stands in for google.auth's AuthorizedSession calling IAM signJwt."""

    def __init__(self, status: int = 200):
        self.status = status
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, jsonlib.loads(json["payload"])))
        n = len(self.calls)
        return SimpleNamespace(
            status_code=self.status,
            text="Permission 'iam.serviceAccounts.signJwt' denied",
            json=lambda: {"keyId": "k1", "signedJwt": f"jwt-{n}"},
        )


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --- settings ---------------------------------------------------------------------------------


def test_direct_anthropic_defaults(clean_env):
    s = get_settings()
    assert not s.uses_gateway and not s.iap
    assert s.fallbacks is True
    assert s.model == s.judge_model == s.gen_model == "claude-opus-5-5"


def test_gateway_settings(clean_env):
    clean_env.setenv("ANTHROPIC_BASE_URL", GATEWAY + "/")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-litellm")
    clean_env.setenv("LITELLM_IAP", "1")
    clean_env.setenv("LITELLM_IAP_SERVICE_ACCOUNT", SA)
    clean_env.setenv("SLOPBOT_MODEL", "claude-sonnet-5-5")
    clean_env.setenv("SLOPBOT_JUDGE_MODEL", "claude-opus-5-5")
    s = get_settings()
    assert s.base_url == GATEWAY  # trailing slash dropped
    assert s.api_key == "sk-litellm"
    assert s.iap and s.iap_service_account == SA
    assert s.iap_audience == GATEWAY + "/*"  # IAP matches aud against the request path
    assert s.fallbacks is False  # LiteLLM reserves `fallbacks`; off unless opted in
    assert (s.model, s.judge_model, s.gen_model) == (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    )
    clean_env.setenv("SLOPBOT_FALLBACKS", "on")
    clean_env.setenv("LITELLM_IAP_AUDIENCE", GATEWAY + "/v1/*")
    s = get_settings()
    assert s.fallbacks is True and s.iap_audience == GATEWAY + "/v1/*"


def test_braintrust_project_id_wins_over_name(clean_env):
    clean_env.delenv("BRAINTRUST_PROJECT_ID", raising=False)
    assert get_settings().braintrust_target == {"project": "Slopbot"}
    clean_env.setenv("BRAINTRUST_PROJECT_ID", "proj-123")
    assert get_settings().braintrust_target == {"project_id": "proj-123"}


def test_iap_needs_gateway_and_service_account(clean_env):
    clean_env.setenv("LITELLM_IAP", "1")
    with pytest.raises(ConfigError, match="ANTHROPIC_BASE_URL"):
        get_settings()
    clean_env.setenv("ANTHROPIC_BASE_URL", GATEWAY)
    with pytest.raises(ConfigError, match="LITELLM_IAP_SERVICE_ACCOUNT"):
        get_settings()


# --- IAP JWTs ---------------------------------------------------------------------------------


def test_iap_jwt_claims_and_endpoint():
    session = FakeSession()
    source = IapJwtSource(SA, GATEWAY, session=session, clock=Clock(1000))
    assert source.header_value() == "Bearer jwt-1"
    ((url, claims),) = session.calls
    assert url == (
        f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{SA}:signJwt"
    )
    assert claims == {
        "iss": SA,
        "sub": SA,
        "aud": GATEWAY,
        "iat": 1000,
        "exp": 1000 + JWT_LIFETIME_S,
    }


def test_iap_jwt_is_cached_then_refreshed_before_expiry():
    session, clock = FakeSession(), Clock(1000)
    source = IapJwtSource(SA, GATEWAY, session=session, clock=clock)
    assert source.token() == "jwt-1"
    clock.now = 1000 + JWT_LIFETIME_S - REFRESH_MARGIN_S - 1
    assert source.token() == "jwt-1"
    assert len(session.calls) == 1
    clock.now = 1000 + JWT_LIFETIME_S - REFRESH_MARGIN_S
    assert source.token() == "jwt-2"
    assert session.calls[1][1]["exp"] == clock.now + JWT_LIFETIME_S


def test_iap_permission_error_is_explained():
    source = IapJwtSource(SA, GATEWAY, session=FakeSession(status=403), clock=Clock(1000))
    with pytest.raises(GatewayError, match="serviceAccountTokenCreator"):
        source.token()


# --- Claude client through the gateway --------------------------------------------------------

VERDICT = {"verdict": "human", "ai_probability": 10, "reasons": ["typo"], "quip": "Human."}


def gateway_settings(**overrides) -> Settings:
    base = dict(
        base_url=GATEWAY,
        api_key="sk-litellm",
        iap=True,
        iap_service_account=SA,
        iap_audience=GATEWAY,
        fallbacks=False,
        model="claude-opus-5-5",
    )
    base.update(overrides)
    return Settings(**base)


def test_detect_through_gateway_sends_key_and_iap_jwt():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(
            url=str(request.url),
            headers=dict(request.headers),
            body=jsonlib.loads(request.content),
        )
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": jsonlib.dumps(VERDICT)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    settings = gateway_settings()
    iap = IapJwtSource(SA, GATEWAY, session=FakeSession(), clock=Clock(1000))
    client = make_anthropic_client(settings, iap=iap, transport=httpx2.MockTransport(handler))
    result = detect("helo wrld", settings=settings, client=client)

    assert result.verdict.verdict == "human"
    assert seen["url"].split("?")[0] == f"{GATEWAY}/v1/messages"
    assert seen["headers"]["authorization"] == "Bearer sk-litellm"
    assert seen["headers"]["proxy-authorization"] == "Bearer jwt-1"
    assert "x-api-key" not in seen["headers"]
    assert "fallbacks" not in seen["body"]
    assert FALLBACK_BETA not in seen["headers"].get("anthropic-beta", "")


# --- /model/info ------------------------------------------------------------------------------


def model_info_transport(status=200, payload=None, headers=None, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.update(url=str(request.url), headers=dict(request.headers))
        if status != 200:
            return httpx.Response(status, text="Invalid IAP credentials", headers=headers or {})
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_model_info_lists_models_with_both_auth_headers():
    seen = {}
    payload = {
        "data": [
            {"model_name": "claude-opus-5-5", "litellm_params": {"model": "anthropic/x"}},
            {"model_name": "claude-haiku-5-5"},
            {"model_name": "claude-opus-5-5"},
        ]
    }
    iap = IapJwtSource(SA, GATEWAY, session=FakeSession(), clock=Clock(1000))
    names = fetch_gateway_models(
        gateway_settings(), iap=iap, transport=model_info_transport(payload=payload, seen=seen)
    )
    assert names == ["claude-opus-5-5", "claude-haiku-5-5"]
    assert seen["url"] == f"{GATEWAY}/model/info"
    assert seen["headers"]["authorization"] == "Bearer sk-litellm"
    assert seen["headers"]["proxy-authorization"] == "Bearer jwt-1"


def test_verify_models_reports_missing_names():
    payload = {"data": [{"model_name": "claude-opus-5-5"}]}
    settings = gateway_settings(iap=False)
    transport = model_info_transport(payload=payload)
    assert verify_models(settings, {"SLOPBOT_MODEL": "claude-opus-5-5"}, transport=transport)
    with pytest.raises(GatewayError) as e:
        verify_models(settings, {"SLOPBOT_JUDGE_MODEL": "claude-nope"}, transport=transport)
    assert "SLOPBOT_JUDGE_MODEL=claude-nope" in str(e.value)
    assert "claude-opus-5-5" in str(e.value)


def test_verify_models_is_a_noop_without_gateway():
    assert verify_models(Settings(), {"SLOPBOT_MODEL": "anything"}) == []


@pytest.mark.parametrize(
    ("iap_on", "expected"),
    [(False, "Set LITELLM_IAP=1"), (True, "Cloud IAP rejected the signed JWT")],
)
def test_iap_rejection_is_explained(iap_on, expected):
    transport = model_info_transport(status=401, headers={"x-goog-iap-generated-response": "true"})
    iap = IapJwtSource(SA, GATEWAY, session=FakeSession(), clock=Clock(1000)) if iap_on else None
    with pytest.raises(GatewayError, match=expected):
        fetch_gateway_models(gateway_settings(iap=iap_on), iap=iap, transport=transport)


def test_litellm_key_rejection_is_explained():
    transport = model_info_transport(status=401)
    with pytest.raises(GatewayError, match="ANTHROPIC_API_KEY"):
        fetch_gateway_models(gateway_settings(iap=False), transport=transport)


def test_gateway_cost_header_is_recorded_per_call():
    from slopbot.gateway import SPEND

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"x-litellm-response-cost": "0.0123", "x-litellm-model-group": "test-model"},
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": jsonlib.dumps(VERDICT)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    settings = gateway_settings(iap=False, model="test-model")
    before = SPEND.snapshot().get("test-model", 0.0)
    client = make_anthropic_client(settings, transport=httpx2.MockTransport(handler))
    result = detect("text", settings=settings, client=client)
    assert result.cost_usd == pytest.approx(0.0123)
    assert SPEND.snapshot()["test-model"] == pytest.approx(before + 0.0123)
