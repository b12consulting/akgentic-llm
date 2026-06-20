"""Behavior tests for ``ReactAgent.aclose()`` and persistent-loop ``run_sync``.

Covers Story 10-1 (Epic 10, ADR-008): the instance-held ``httpx.AsyncClient``
handle, ``aclose()`` releasing that pool, and ``run_sync`` reusing the actor's
persistent event loop (falling back to ``asyncio.run()`` when none is supplied).

All tests are zero-egress: ``aclose()`` is exercised on the real client the
agent builds (no request is ever sent), and the ``run_sync`` loop-selection
tests patch ``ReactAgent.run`` with a stub coroutine so no model is contacted.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from akgentic.llm import ModelConfig, ReactAgent, ReactAgentConfig


@pytest.fixture
def minimal_config() -> ReactAgentConfig:
    """Minimal config; the OpenAI provider builds with no key (no egress here)."""
    return ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))


# ---------------------------------------------------------------------------
# AC #4 — instance handle wires the pool
# ---------------------------------------------------------------------------


def test_http_client_held_on_instance(minimal_config: ReactAgentConfig) -> None:
    """The agent exposes the ``httpx.AsyncClient`` it built as ``_http_client``."""
    agent = ReactAgent(config=minimal_config)
    assert isinstance(agent._http_client, httpx.AsyncClient)
    assert not agent._http_client.is_closed


def test_model_uses_the_held_client(minimal_config: ReactAgentConfig) -> None:
    """The client passed to ``create_model`` is the same one ``aclose()`` closes."""
    import akgentic.llm.agent as agent_mod

    captured: dict[str, Any] = {}
    real_create_model = agent_mod.create_model

    def spy_create_model(model_cfg: Any, http_client: httpx.AsyncClient) -> Any:
        captured["http_client"] = http_client
        return real_create_model(model_cfg, http_client)

    with patch.object(agent_mod, "create_model", side_effect=spy_create_model):
        agent = ReactAgent(config=minimal_config)

    # The client handed to create_model() is the very instance reachable for aclose().
    assert captured["http_client"] is agent._http_client


# ---------------------------------------------------------------------------
# AC #1 — aclose() releases the client and is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_client(minimal_config: ReactAgentConfig) -> None:
    """After ``aclose()`` the underlying client is closed."""
    agent = ReactAgent(config=minimal_config)
    assert not agent._http_client.is_closed

    await agent.aclose()

    assert agent._http_client.is_closed


@pytest.mark.asyncio
async def test_aclose_is_idempotent(minimal_config: ReactAgentConfig) -> None:
    """A second ``aclose()`` is harmless (does not raise)."""
    agent = ReactAgent(config=minimal_config)

    await agent.aclose()
    # Second close must not raise — teardown may run more than once.
    await agent.aclose()

    assert agent._http_client.is_closed


@pytest.mark.asyncio
async def test_aclose_targets_only_owned_client(minimal_config: ReactAgentConfig) -> None:
    """``aclose()`` closes only the client the agent created (no internals reach)."""
    agent = ReactAgent(config=minimal_config)
    owned = agent._http_client

    await agent.aclose()

    # The closed client is exactly the agent's own handle.
    assert agent._http_client is owned
    assert owned.is_closed


# ---------------------------------------------------------------------------
# AC #2 — run_sync reuses the actor's persistent loop
# ---------------------------------------------------------------------------


def test_run_sync_reuses_persistent_loop(minimal_config: ReactAgentConfig) -> None:
    """``run_sync`` runs on the supplied idle loop and leaves it usable."""
    loop = asyncio.new_event_loop()
    used_loop: list[asyncio.AbstractEventLoop] = []

    async def stub_run(*_: Any, **__: Any) -> str:
        used_loop.append(asyncio.get_running_loop())
        return "ran-on-persistent-loop"

    try:
        agent = ReactAgent(config=minimal_config, event_loop=loop)
        with patch.object(ReactAgent, "run", new=stub_run):
            result = agent.run_sync("q")

        assert result == "ran-on-persistent-loop"
        # The coroutine executed on the very loop we supplied...
        assert used_loop == [loop]
        # ...and that loop is still usable afterwards (not closed by run_sync).
        assert not loop.is_closed()
        assert loop.run_until_complete(asyncio.sleep(0)) is None
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# AC #3 — run_sync falls back to asyncio.run() without a persistent loop
# ---------------------------------------------------------------------------


def test_run_sync_falls_back_without_loop(minimal_config: ReactAgentConfig) -> None:
    """With no ``event_loop``, ``run_sync`` returns the result via ``asyncio.run()``."""
    used_loop: list[asyncio.AbstractEventLoop] = []

    async def stub_run(*_: Any, **__: Any) -> str:
        used_loop.append(asyncio.get_running_loop())
        return "ran-via-asyncio-run"

    agent = ReactAgent(config=minimal_config, event_loop=None)
    with patch.object(ReactAgent, "run", new=stub_run):
        result = agent.run_sync("q")

    assert result == "ran-via-asyncio-run"
    # A loop was created for the call; it has been torn down by asyncio.run().
    assert len(used_loop) == 1
    assert used_loop[0].is_closed()


def test_run_sync_falls_back_with_closed_loop(minimal_config: ReactAgentConfig) -> None:
    """A closed ``event_loop`` is skipped; the call still completes via fallback."""
    closed = asyncio.new_event_loop()
    closed.close()
    used_loop: list[asyncio.AbstractEventLoop] = []

    async def stub_run(*_: Any, **__: Any) -> str:
        used_loop.append(asyncio.get_running_loop())
        return "ran-via-fallback"

    agent = ReactAgent(config=minimal_config, event_loop=closed)
    with patch.object(ReactAgent, "run", new=stub_run):
        result = agent.run_sync("q")

    assert result == "ran-via-fallback"
    # The fallback loop is a fresh one, never the closed loop we supplied.
    assert used_loop and used_loop[0] is not closed
