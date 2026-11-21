"""Talking to Claude through a LiteLLM gateway, optionally behind Google Cloud IAP.

- The LiteLLM virtual key (ANTHROPIC_API_KEY) goes in `Authorization: Bearer ...`.
- With LITELLM_IAP=1, every request also carries `Proxy-Authorization: Bearer <jwt>`. The JWT
  is signed by IAM Credentials `signJwt` as LITELLM_IAP_SERVICE_ACCOUNT, using your local gcloud
  Application Default Credentials (`gcloud auth application-default login`). Its audience
  defaults to `<gateway>/*` because IAP matches `aud` against the request path. It lives in
  memory only and is re-signed a few minutes before it expires. No Google credentials are stored.
- Model names are checked against the gateway's `GET /model/info`.
- Every response's `x-litellm-response-cost` header is recorded in SPEND, so evals report
  what they actually cost on the key's budget.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import anthropic
import httpx

from .config import Settings

SIGN_JWT_URL = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:signJwt"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
JWT_LIFETIME_S = 3600
REFRESH_MARGIN_S = 300
ADC_HINT = "Run `gcloud auth application-default login` and try again."


class GatewayError(RuntimeError):
    """Something about the gateway setup needs fixing; the message says what."""


class IapJwtSource:
    """Mints and caches IAP JWTs signed by IAM as a service account."""

    def __init__(
        self,
        service_account: str,
        audience: str,
        *,
        session: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        self.service_account = service_account
        self.audience = audience
        self._session = session
        self._clock = clock
        self._lock = threading.Lock()
        self._jwt: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        with self._lock:
            if self._jwt is None or self._clock() >= self._expires_at - REFRESH_MARGIN_S:
                self._jwt, self._expires_at = self._mint()
            return self._jwt

    def header_value(self) -> str:
        return f"Bearer {self.token()}"

    def _authorized_session(self) -> Any:
        if self._session is None:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
            from google.auth.transport.requests import AuthorizedSession

            try:
                credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            except DefaultCredentialsError as e:
                raise GatewayError(
                    f"No Google Application Default Credentials found. {ADC_HINT}"
                ) from e
            self._session = AuthorizedSession(credentials)
        return self._session

    def _mint(self) -> tuple[str, float]:
        from google.auth.exceptions import RefreshError, TransportError

        now = int(self._clock())
        claims = {
            "iss": self.service_account,
            "sub": self.service_account,
            "aud": self.audience,
            "iat": now,
            "exp": now + JWT_LIFETIME_S,
        }
        url = SIGN_JWT_URL.format(sa=quote(self.service_account, safe="@"))
        try:
            response = self._authorized_session().post(
                url, json={"payload": json.dumps(claims)}, timeout=30
            )
        except RefreshError as e:
            raise GatewayError(f"Your Google ADC login has expired. {ADC_HINT}") from e
        except TransportError as e:
            raise GatewayError(f"Couldn't reach Google to sign the IAP token: {e}") from e

        if response.status_code == 403:
            raise GatewayError(
                f"IAM refused to sign as {self.service_account}. Your Google account needs "
                "roles/iam.serviceAccountTokenCreator on that service account."
            )
        if response.status_code >= 400:
            raise GatewayError(
                f"signJwt failed (HTTP {response.status_code}): {response.text[:300]}"
            )
        return response.json()["signedJwt"], float(now + JWT_LIFETIME_S)


@lru_cache(maxsize=4)
def _shared_iap_source(service_account: str, audience: str) -> IapJwtSource:
    return IapJwtSource(service_account, audience)


def iap_source(settings: Settings) -> IapJwtSource | None:
    """One JWT source per process, shared by the Claude client and the /model/info check."""
    if not settings.iap:
        return None
    assert settings.iap_service_account and settings.iap_audience  # validated in get_settings
    return _shared_iap_source(settings.iap_service_account, settings.iap_audience)


class SpendMeter:
    """Adds up what the gateway says each call cost (`x-litellm-response-cost`), per model.

    `take_last()` returns the cost of the most recent call made on the current thread, which is
    how detect() attaches a cost to its own call even when evals run calls in parallel threads.
    """

    COST_HEADER = "x-litellm-response-cost"
    MODEL_HEADER = "x-litellm-model-group"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self.cost_by_model: defaultdict[str, float] = defaultdict(float)
        self.calls_by_model: defaultdict[str, int] = defaultdict(int)

    def record(self, response: Any) -> None:
        raw = response.headers.get(self.COST_HEADER)
        if raw is None:
            return
        try:
            cost = float(raw)
        except ValueError:
            return
        model = response.headers.get(self.MODEL_HEADER) or "unknown"
        with self._lock:
            self.cost_by_model[model] += cost
            self.calls_by_model[model] += 1
        self._local.last = cost

    def take_last(self) -> float | None:
        cost = getattr(self._local, "last", None)
        self._local.last = None
        return cost

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.cost_by_model)

    @property
    def total(self) -> float:
        with self._lock:
            return sum(self.cost_by_model.values())


SPEND = SpendMeter()


def make_anthropic_client(
    settings: Settings,
    *,
    iap: IapJwtSource | None = None,
    transport: Any = None,
) -> anthropic.Anthropic:
    """Anthropic client pointed at the gateway (or Anthropic directly when no base URL is set).

    The SDK sends `api_key` as `Authorization: Bearer ...`, which is where LiteLLM expects its
    virtual key. `transport` is for tests.
    """
    kwargs: dict[str, Any] = {}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key

    iap = iap or iap_source(settings)
    if settings.uses_gateway or transport is not None:
        http_kwargs: dict[str, Any] = {"event_hooks": {"request": [], "response": []}}
        if transport is not None:
            http_kwargs["transport"] = transport
        if settings.uses_gateway:
            http_kwargs["event_hooks"]["response"].append(SPEND.record)
        if iap is not None:

            def add_iap_header(request: Any) -> None:
                request.headers["Proxy-Authorization"] = iap.header_value()

            http_kwargs["event_hooks"]["request"].append(add_iap_header)
            # IAP answers unauthenticated requests with a redirect to a Google login page.
            # Surface that as an error instead of following it into HTML.
            http_kwargs["follow_redirects"] = False
        kwargs["http_client"] = anthropic.DefaultHttpxClient(**http_kwargs)
    return anthropic.Anthropic(**kwargs)


def gateway_headers(settings: Settings, iap: IapJwtSource | None = None) -> dict[str, str]:
    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    iap = iap or iap_source(settings)
    if iap is not None:
        headers["Proxy-Authorization"] = iap.header_value()
    return headers


def _explain_failure(response: httpx.Response, settings: Settings) -> str:
    iap_generated = response.headers.get("x-goog-iap-generated-response", "").lower() == "true"
    to_google_login = "accounts.google.com" in response.headers.get("location", "")
    if iap_generated or to_google_login:
        if not settings.iap:
            return (
                "The gateway is behind Google Cloud IAP. Set LITELLM_IAP=1 and "
                "LITELLM_IAP_SERVICE_ACCOUNT."
            )
        return (
            f"Cloud IAP rejected the signed JWT ({response.text.strip()[:200]}). Check that "
            "LITELLM_IAP_SERVICE_ACCOUNT is allowed through IAP and that the JWT audience "
            f"({settings.iap_audience}) matches the request URL (IAP compares the path too; "
            "`<gateway>/*` is the default). Note: when the Proxy-Authorization JWT fails, IAP "
            "falls back to Authorization, so its error can mention the wrong header."
        )
    if response.status_code in (401, 403):
        return "LiteLLM rejected the key. Check ANTHROPIC_API_KEY (your LiteLLM virtual key)."
    return f"HTTP {response.status_code}: {response.text.strip()[:300]}"


def fetch_gateway_models(
    settings: Settings, *, iap: IapJwtSource | None = None, transport: Any = None
) -> list[str]:
    """Model names this key can use, from LiteLLM's `GET /model/info`."""
    if not settings.base_url:
        raise GatewayError("ANTHROPIC_BASE_URL isn't set, so there's no gateway to ask.")
    with httpx.Client(transport=transport, timeout=30, follow_redirects=False) as client:
        try:
            response = client.get(
                f"{settings.base_url}/model/info", headers=gateway_headers(settings, iap)
            )
        except httpx.HTTPError as e:
            raise GatewayError(f"Couldn't reach {settings.base_url}: {e}") from e
    if response.status_code != 200:
        raise GatewayError(f"/model/info failed. {_explain_failure(response, settings)}")
    try:
        entries = response.json().get("data") or []
    except ValueError as e:
        raise GatewayError("/model/info didn't return JSON. Is ANTHROPIC_BASE_URL right?") from e
    names = [e.get("model_name") for e in entries if isinstance(e, dict)]
    return list(dict.fromkeys(n for n in names if n))


def verify_models(
    settings: Settings,
    models: Mapping[str, str],
    *,
    iap: IapJwtSource | None = None,
    transport: Any = None,
) -> list[str]:
    """Raise GatewayError if any configured model isn't on the gateway. No-op without a gateway.

    `models` maps a setting name (e.g. "SLOPBOT_MODEL") to the model it's configured with.
    Returns the gateway's model list.
    """
    if not settings.uses_gateway:
        return []
    available = fetch_gateway_models(settings, iap=iap, transport=transport)
    missing = {env: name for env, name in models.items() if name not in available}
    if missing:
        wrong = ", ".join(f"{env}={name}" for env, name in missing.items())
        listed = ", ".join(available) or "(none)"
        raise GatewayError(f"Not available on the gateway: {wrong}. This key can use: {listed}")
    return available
