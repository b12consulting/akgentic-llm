"""Wire-level guards on the retry layers under ``create_model`` (Epic 32).

Every request is counted at an ``httpx2.MockTransport`` placed *under* the
``AsyncHTTPX2TenacityTransport`` built by ``create_http_client()``, so a count
covers both our transport and anything above it — the provider SDK included.
"""

from collections.abc import Callable

import httpx2
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport

from akgentic.llm.config import ModelConfig
from akgentic.llm.providers import create_http_client, create_model

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_Handler = Callable[[httpx2.Request], httpx2.Response]

_MAX_ATTEMPTS = 3

_ERROR_BODY = {"error": {"message": "request timed out upstream", "type": "timeout"}}

# Every environment variable the providers below read; each case clears them all
# and sets only its own, so neither a local ``.env`` nor the CI environment leaks in.
_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_AD_TOKEN",
    "OPENAI_API_VERSION",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "NVIDIA_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_APP_URL",
    "OPENROUTER_APP_TITLE",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
)

_OPENAI_ENV = {"OPENAI_API_KEY": "test-key"}
_AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://acme.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "test-key",
    "OPENAI_API_VERSION": "2024-12-01-preview",
}

# provider -> (model, env)
_STAINLESS_CASES: dict[str, tuple[str, dict[str, str]]] = {
    "openai": ("gpt-4o", _OPENAI_ENV),
    "openai-chat": ("gpt-4o", _OPENAI_ENV),
    "azure": ("gpt-4o", _AZURE_ENV),
    "azure-chat": ("gpt-4o", _AZURE_ENV),
    "anthropic": ("claude-opus-5-5", {"ANTHROPIC_API_KEY": "test-key"}),
    "nvidia": ("meta/llama-3.1-8b-instruct", _OPENAI_ENV),
    "openrouter": ("openai/gpt-4o", {"OPENROUTER_API_KEY": "test-key"}),
}

_OTHER_CASES: dict[str, tuple[str, dict[str, str]]] = {
    "google-gla": ("gemini-2.0-flash", {"GOOGLE_API_KEY": "test-key"}),
    "mistral": ("mistral-small-latest", {"MISTRAL_API_KEY": "test-key"}),
}


class _Recorder:
    """A mock-transport handler that records every request it is sent."""

    def __init__(self, respond: _Handler) -> None:
        self.requests: list[httpx2.Request] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._respond(request)


def _counting_client(handler: _Handler) -> httpx2.AsyncClient:
    """Build ``create_http_client()`` with ``handler`` as the transport it wraps.

    The single place the counting point is located: the mock replaces only the
    *wrapped* transport, so the retry transport above it stays in the path.
    """
    client = create_http_client(max_attempts=_MAX_ATTEMPTS, exp_multiplier=0.01, exp_max_s=0.05)
    assert isinstance(client._transport, AsyncHTTPX2TenacityTransport)
    client._transport.wrapped = httpx2.MockTransport(handler)
    return client


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _status_408(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(408, json=_ERROR_BODY, request=request)


def _read_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


async def _run(provider: str, model: str, recorder: _Recorder) -> None:
    config = ModelConfig(provider=provider, model=model)
    await Agent(create_model(config, http_client=_counting_client(recorder))).run("hi")


# ---------------------------------------------------------------------------
# The provider SDKs do not retry (Story 32.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", list(_STAINLESS_CASES))
async def test_stainless_sdk_sends_one_request_on_408(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 408 returned by our transport is not retried by the SDK: one request."""
    model, env = _STAINLESS_CASES[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_status_408)

    with pytest.raises(ModelAPIError):
        await _run(provider, model, recorder)

    assert len(recorder.requests) == 1


@pytest.mark.parametrize("provider", list(_STAINLESS_CASES))
async def test_stainless_sdk_sends_one_request_on_timeout(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout our transport does not retry is not retried by the SDK: one request."""
    model, env = _STAINLESS_CASES[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_read_timeout)

    with pytest.raises(ModelAPIError):
        await _run(provider, model, recorder)

    assert len(recorder.requests) == 1


@pytest.mark.parametrize("provider", list(_STAINLESS_CASES))
async def test_stainless_sdk_does_not_multiply_transport_retries(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A network error our transport retries costs ``max_attempts`` requests, not 3x that."""
    model, env = _STAINLESS_CASES[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_connect_error)

    with pytest.raises(ModelAPIError):
        await _run(provider, model, recorder)

    assert len(recorder.requests) == _MAX_ATTEMPTS


@pytest.mark.parametrize("provider", list(_OTHER_CASES))
async def test_non_stainless_sdk_sends_one_request_on_408(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Google and Mistral are built without an SDK retry policy: one request on a 408."""
    model, env = _OTHER_CASES[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_status_408)

    with pytest.raises(ModelAPIError):
        await _run(provider, model, recorder)

    assert len(recorder.requests) == 1
