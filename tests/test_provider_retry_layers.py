"""Wire-level guards on the retry layers under ``create_model`` (Epic 32).

Every request is counted at an ``httpx2.MockTransport`` placed *under* the
transport chain built by ``create_http_client()``, so a count covers every
layer above it — the provider SDK included.

A retryable status is answered with an *unread* body (``_UnreadBody``). A
``json=`` or ``content=`` response reads itself in its constructor, which would
make the body buffering under test a no-op that no assertion can see.
"""

import json
import time
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Self

import httpx2
import openai
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport

from akgentic.llm.config import ModelConfig
from akgentic.llm.providers import (
    _BufferErrorBody,
    _ReturnFinalStatus,
    create_http_client,
    create_model,
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_Handler = Callable[[httpx2.Request], httpx2.Response]

_MAX_ATTEMPTS = 3

_ERROR_BODY = {"error": {"message": "request timed out upstream", "type": "timeout"}}

_RATE_LIMIT_MESSAGE = "Rate limit reached for gpt-4o in organization org-acme on tokens per min"
_RATE_LIMIT_BODY = {
    "error": {
        "message": _RATE_LIMIT_MESSAGE,
        "type": "rate_limit_exceeded",
        "param": None,
        "code": "rate_limit_exceeded",
    }
}

_QUOTA_MESSAGE = "You have no credits remaining"
_QUOTA_BODY = {
    "error": {
        "message": _QUOTA_MESSAGE,
        "type": "insufficient_quota",
        "param": None,
        "code": "credit_balance_exhausted",
    }
}

_OVERLOADED_MESSAGE = "The server is overloaded, please retry later"
_OVERLOADED_BODY = {
    "error": {"message": _OVERLOADED_MESSAGE, "type": "server_error", "param": None, "code": None}
}

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
    """A mock-transport handler that records every request it is sent, and when."""

    def __init__(self, respond: _Handler) -> None:
        self.requests: list[httpx2.Request] = []
        self.stamps: list[float] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.stamps.append(time.monotonic())
        return self._respond(request)


class _UnreadBody(httpx2.AsyncByteStream):
    """A response body that stays on the wire until someone iterates it."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._payload


class _LifecycleSpy(httpx2.AsyncBaseTransport):
    """A base transport that records the lifecycle calls that reach it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __aenter__(self) -> Self:
        self.calls.append("enter")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self.calls.append("exit")

    async def aclose(self) -> None:
        self.calls.append("aclose")


def _client_over(base: httpx2.AsyncBaseTransport) -> httpx2.AsyncClient:
    """Build ``create_http_client()`` with ``base`` as the transport at the bottom of it.

    The single place the counting point is located. The chain is
    ``_ReturnFinalStatus`` → ``AsyncHTTPX2TenacityTransport`` → ``_BufferErrorBody``
    → base, and ``base`` replaces only the last link: assigning it to the retry
    transport's ``wrapped`` would swap the body buffer out of the path and leave
    every body assertion below silently testing nothing.
    """
    client = create_http_client(max_attempts=_MAX_ATTEMPTS, exp_multiplier=0.01, exp_max_s=0.05)
    outer = client._transport
    assert isinstance(outer, _ReturnFinalStatus)
    retrying = outer.wrapped
    assert isinstance(retrying, AsyncHTTPX2TenacityTransport)
    buffer = retrying.wrapped
    assert isinstance(buffer, _BufferErrorBody)
    buffer.wrapped = base
    return client


def _counting_client(handler: _Handler) -> httpx2.AsyncClient:
    """Build ``create_http_client()`` over a mock transport answering with ``handler``."""
    return _client_over(httpx2.MockTransport(handler))


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _unread(
    status: int,
    payload: bytes,
    request: httpx2.Request,
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    """A response whose body has not been read yet, as one off the wire."""
    return httpx2.Response(status, headers=headers, stream=_UnreadBody(payload), request=request)


def _always(status: int, payload: bytes, headers: dict[str, str] | None = None) -> _Handler:
    """A handler answering every request with ``status`` and an unread ``payload``."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        return _unread(status, payload, request, headers)

    return respond


def _always_json(status: int, body: object) -> _Handler:
    """A handler answering every request with ``status`` and an unread JSON ``body``."""
    return _always(status, json.dumps(body).encode(), {"content-type": "application/json"})


def _status_408(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(408, json=_ERROR_BODY, request=request)


def _read_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


def _responses_body() -> dict[str, object]:
    """A minimal valid OpenAI Responses API response."""
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }


def _chat_completion_body() -> dict[str, object]:
    """A minimal valid OpenAI chat-completion response."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "fallback ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


async def _run_config(config: ModelConfig, recorder: _Recorder, *, streamed: bool = False) -> str:
    agent = Agent(create_model(config, http_client=_counting_client(recorder)))
    if streamed:
        async with agent.run_stream("hi") as result:
            return await result.get_output()
    return (await agent.run("hi")).output


async def _run(provider: str, model: str, recorder: _Recorder, *, streamed: bool = False) -> str:
    config = ModelConfig(provider=provider, model=model)
    return await _run_config(config, recorder, streamed=streamed)


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


# ---------------------------------------------------------------------------
# The final retryable status reaches the caller with its body (Story 32.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_stream"])
@pytest.mark.parametrize("provider", ["openai", "openai-chat"])
async def test_an_exhausted_rate_limit_reaches_the_caller_with_its_body(
    provider: str, streamed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 on every attempt is retried, then surfaces as ``ModelHTTPError`` with its message."""
    model, env = _STAINLESS_CASES[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_always_json(429, _RATE_LIMIT_BODY))

    with pytest.raises(ModelHTTPError) as caught:
        await _run(provider, model, recorder, streamed=streamed)

    assert len(recorder.requests) == _MAX_ATTEMPTS
    assert caught.value.status_code == 429
    assert isinstance(caught.value.body, dict)
    assert caught.value.body["message"] == _RATE_LIMIT_MESSAGE


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "run_stream"])
async def test_an_insufficient_quota_429_fails_fast_with_its_message(
    streamed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A billing failure is not retried: one request, and the user reads the provider's words."""
    model, env = _STAINLESS_CASES["openai"]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_always_json(429, _QUOTA_BODY))

    with pytest.raises(ModelHTTPError) as caught:
        await _run("openai", model, recorder, streamed=streamed)

    assert len(recorder.requests) == 1
    assert caught.value.status_code == 429
    assert isinstance(caught.value.body, dict)
    assert _QUOTA_MESSAGE in caught.value.body["message"]
    assert _QUOTA_MESSAGE in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [b"", b"<html>Too Many Requests</html>", b"[]", b'{"error": "a string"}'],
    ids=["empty", "html", "json-array", "error-string"],
)
async def test_an_unparseable_429_body_keeps_the_retry(
    payload: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 whose body cannot say ``insufficient_quota`` is retried as a rate limit."""
    model, env = _STAINLESS_CASES["openai"]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_always(429, payload))

    with pytest.raises(ModelHTTPError) as caught:
        await _run("openai", model, recorder)

    assert len(recorder.requests) == _MAX_ATTEMPTS
    assert caught.value.status_code == 429


async def test_an_exhausted_5xx_reaches_the_caller_with_its_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 on every attempt is retried, then surfaces as ``ModelHTTPError`` with its message."""
    model, env = _STAINLESS_CASES["openai"]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_always_json(503, _OVERLOADED_BODY))

    with pytest.raises(ModelHTTPError) as caught:
        await _run("openai", model, recorder)

    assert len(recorder.requests) == _MAX_ATTEMPTS
    assert caught.value.status_code == 503
    assert isinstance(caught.value.body, dict)
    assert caught.value.body["message"] == _OVERLOADED_MESSAGE


async def test_retry_after_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 carrying ``Retry-After: 1`` waits about a second before the retry that succeeds.

    The test client's backoff is capped at 0.05 s, so only the header can produce the gap.
    """
    model, env = _STAINLESS_CASES["openai"]
    _set_env(monkeypatch, env)
    rate_limited = _always(
        429,
        json.dumps(_RATE_LIMIT_BODY).encode(),
        {"content-type": "application/json", "retry-after": "1"},
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        if len(recorder.requests) == 1:
            return rate_limited(request)
        return httpx2.Response(200, json=_responses_body(), request=request)

    recorder = _Recorder(respond)

    output = await _run("openai", model, recorder)

    assert output == "ok"
    assert len(recorder.requests) == 2
    assert recorder.stamps[1] - recorder.stamps[0] >= 0.9


async def test_the_fallback_chain_advances_on_an_exhausted_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited primary hands the run to its fallback, which answers."""
    _set_env(monkeypatch, _OPENAI_ENV)
    rate_limited = _always_json(429, _RATE_LIMIT_BODY)

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/responses"):
            return rate_limited(request)
        assert request.url.path.endswith("/chat/completions")
        return httpx2.Response(200, json=_chat_completion_body(), request=request)

    recorder = _Recorder(respond)
    config = ModelConfig(
        provider="openai",
        model="gpt-4o",
        fallback_models=[ModelConfig(provider="openai-chat", model="gpt-4o")],
    )

    output = await _run_config(config, recorder)

    paths = [request.url.path for request in recorder.requests]
    assert output == "fallback ok"
    assert sum(path.endswith("/responses") for path in paths) == _MAX_ATTEMPTS
    assert sum(path.endswith("/chat/completions") for path in paths) == 1


async def test_a_2xx_is_never_buffered() -> None:
    """A streamed 2xx comes back unread: buffering it would defeat every streamed completion."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        return _unread(200, b"data: chunk\n\n", request)

    client = _counting_client(_Recorder(respond))
    request = client.build_request("POST", "https://api.example.com/v1/responses")

    response = await client.send(request, stream=True)
    try:
        assert response.status_code == 200
        assert response.is_stream_consumed is False
    finally:
        await response.aclose()
        await client.aclose()


async def test_a_timeout_is_not_converted_into_a_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a retryable status is returned: a timeout still reaches the SDK as a timeout."""
    model, env = _STAINLESS_CASES["openai"]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_read_timeout)

    with pytest.raises(ModelAPIError) as caught:
        await _run("openai", model, recorder)

    assert len(recorder.requests) == 1
    assert isinstance(caught.value.__cause__, openai.APITimeoutError)


@pytest.mark.parametrize("provider", [*_STAINLESS_CASES, *_OTHER_CASES])
async def test_every_provider_surfaces_an_exhausted_429_as_a_typed_error(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No provider lets a raw ``HTTPStatusError`` escape on an exhausted rate limit.

    The message is asserted too: Anthropic's SDK raises a typed 429 even when the
    body is lost, so type and status alone would not notice the buffer missing.
    """
    model, env = {**_STAINLESS_CASES, **_OTHER_CASES}[provider]
    _set_env(monkeypatch, env)
    recorder = _Recorder(_always_json(429, _RATE_LIMIT_BODY))

    with pytest.raises(ModelHTTPError) as caught:
        await _run(provider, model, recorder)

    assert len(recorder.requests) == _MAX_ATTEMPTS
    assert caught.value.status_code == 429
    assert _RATE_LIMIT_MESSAGE in str(caught.value)


async def test_aclose_reaches_the_base_transport() -> None:
    """Closing the client releases the connection pool at the bottom of the chain."""
    spy = _LifecycleSpy()
    client = _client_over(spy)

    await client.aclose()

    assert spy.calls == ["aclose"]


async def test_async_with_reaches_the_base_transport() -> None:
    """``async with create_http_client()`` opens and closes the base transport."""
    spy = _LifecycleSpy()

    async with _client_over(spy) as client:
        assert isinstance(client, httpx2.AsyncClient)

    assert spy.calls == ["enter", "exit"]
