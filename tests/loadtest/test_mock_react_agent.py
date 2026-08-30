"""Tests for MockReactAgent + the YAML scenario state machine (Epic 9 / ADR-007).

These tests stub ``deps`` (an object exposing ``.config.name``) and a
``StructuredOutput``-shaped output model — no akgentic-agent import — and assert
per-agent returned messages, ordered observer events, the zero-token / provider
guard, latency, scenario caching, and determinism.
"""

import asyncio
import inspect
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from akgentic.llm import ModelConfig, ModelSwitchError, ReactAgent, ReactAgentConfig
from akgentic.llm.event import (
    LlmUsageEvent,
    ToolCallEvent,
    ToolReturnEvent,
)
from akgentic.llm.loadtest import (
    MockProviderReachedError,
    MockReactAgent,
    ScenarioConfig,
    load_scenario,
)
from akgentic.llm.loadtest.scenario import SCENARIO_ENV_VAR, _resolve_scenario_ref

SANDPILE = str(Path(__file__).parent / "scenarios" / "sandpile-research.yaml")


# ---------------------------------------------------------------------------
# Stubs (duck-typed — no akgentic-agent import)
# ---------------------------------------------------------------------------


class _ModelCfg(BaseModel):
    model: str


class _Config(BaseModel):
    name: str
    model_cfg: _ModelCfg


class _Deps:
    """Minimal stand-in for a BaseAgent: only ``.config.name`` is read."""

    def __init__(self, name: str) -> None:
        self.config = _Config(name=name, model_cfg=_ModelCfg(model=SANDPILE))


class _Request(BaseModel):
    message_type: str
    recipient: str
    message: str


class _StructuredOutput(BaseModel):
    """Mirrors the dynamic StructuredOutput subclass act() passes as output_type."""

    messages: list[_Request]


class _Recorder:
    """Captures domain events in emission order."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


def _make_config(name: str, model: str = SANDPILE) -> _Config:
    return _Config(name=name, model_cfg=_ModelCfg(model=model))


def _make_agent(name: str, observer: Any | None = None) -> MockReactAgent:
    return MockReactAgent(config=_make_config(name), observer=observer)


# ---------------------------------------------------------------------------
# FR1 — drop-in surface
# ---------------------------------------------------------------------------


def test_drop_in_surface() -> None:
    """Every ReactAgent member used by BaseAgent/state-restore is present."""
    agent = _make_agent("@Expert")
    assert hasattr(agent, "context")
    for member in (
        "run",
        "run_sync",
        "subscribe_context",
        "restore_context",
        "system_prompt",
        "tool",
    ):
        assert callable(getattr(agent, member))

    def fn() -> None:  # decorator passthrough
        return None

    assert agent.system_prompt(fn) is fn
    assert agent.tool(fn) is fn
    # checkpoint/rewind removed in Epic 12 pre-cleanup — parity holds without them.
    assert not hasattr(agent, "checkpoint")
    assert not hasattr(agent, "rewind")


# ---------------------------------------------------------------------------
# Epic 13 — `capabilities` constructor-argument parity (accept-and-ignore)
# ---------------------------------------------------------------------------


def test_capabilities_accepted_and_ignored() -> None:
    """`capabilities=[...]` constructs without error and does not change run() output."""
    from pydantic_ai.capabilities import Capability

    cap = Capability(id="custom-cap")

    without_caps = MockReactAgent(config=_make_config("@Manager"))
    with_caps = MockReactAgent(config=_make_config("@Manager"), capabilities=[cap])

    out_without = without_caps.run_sync(
        "sandpile please", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )
    out_with = with_caps.run_sync(
        "sandpile please", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )

    assert out_with.messages == out_without.messages


def test_limit_recovery_accepted_and_ignored() -> None:
    """`limit_recovery=` constructs without error and does not change run() output.

    The same accept-and-ignore mirror `capabilities` gets, and for a stronger
    reason: the mock enforces no run-tier budget and raises no usage-limit error on any path,
    so there is no breach for a recovery policy to decide about. Only the keyword is mirrored;
    a behavioural mirror would have nothing to mirror.
    """
    from akgentic.llm import LimitRecoveryCapability

    without_policy = MockReactAgent(config=_make_config("@Manager"))
    with_policy = MockReactAgent(
        config=_make_config("@Manager"), limit_recovery=LimitRecoveryCapability()
    )

    out_without = without_policy.run_sync(
        "sandpile please", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )
    out_with = with_policy.run_sync(
        "sandpile please", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )

    assert out_with.messages == out_without.messages


# ---------------------------------------------------------------------------
# FR2 — zero tokens, no egress
# ---------------------------------------------------------------------------


def test_no_model_built() -> None:
    """The constructor never builds a model or HTTP client."""
    agent = _make_agent("@Manager")
    assert agent._model is None
    assert agent._http_client is None


def test_provider_guard_raises() -> None:
    """Reaching the provider factory during a mock run raises."""
    agent = _make_agent("@Manager")
    with pytest.raises(MockProviderReachedError):
        agent._build_model()


@pytest.mark.asyncio
async def test_aclose_is_harmless_noop() -> None:
    """``aclose()`` returns None, builds no model/provider, and never raises."""
    agent = _make_agent("@Manager")
    with patch.object(agent, "_build_model", side_effect=AssertionError("must not build")):
        result = await agent.aclose()
        # A second call must also be harmless (drop-in parity with ReactAgent).
        assert await agent.aclose() is None
    assert result is None
    # The zero-token guarantee holds: no model/provider was ever constructed.
    assert agent._model is None
    assert agent._http_client is None


def test_no_token_usage_emitted() -> None:
    """Every emitted usage event reports zero input/output tokens."""
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)
    agent.run_sync("research the sandpile model", deps=_Deps("@Manager"))
    usage = [e for e in rec.events if isinstance(e, LlmUsageEvent)]
    assert usage
    assert all(e.input_tokens == 0 and e.output_tokens == 0 for e in usage)


# ---------------------------------------------------------------------------
# FR3 — identity from deps, structured return
# ---------------------------------------------------------------------------


def test_manager_returns_structured_output() -> None:
    """@Manager returns a StructuredOutput with the two recorded routed messages."""
    agent = _make_agent("@Manager")
    out = agent.run_sync(
        "Please research the sandpile model",
        deps=_Deps("@Manager"),
        output_type=_StructuredOutput,
    )
    assert isinstance(out, _StructuredOutput)
    assert [m.recipient for m in out.messages] == ["@Assistant", "@Expert"]
    assert [m.message_type for m in out.messages] == ["instruction", "notification"]


def test_str_output_path() -> None:
    """With a str/None output_type the agent returns respond.text or ''."""
    agent = _make_agent("@Expert")
    out = agent.run_sync("anything", deps=_Deps("@Expert"))
    assert out == ""


def test_identity_resolved_from_deps() -> None:
    """The agent name is duck-typed from deps.config.name, not the config."""
    # Construct as @Manager but run as @Expert via deps — deps wins.
    agent = _make_agent("@Manager")
    out = agent.run_sync("hello", deps=_Deps("@Expert"), output_type=_StructuredOutput)
    assert out.messages == []


# ---------------------------------------------------------------------------
# FR4 — input-driven state selection
# ---------------------------------------------------------------------------


def test_state_matching() -> None:
    """A matching `when` selects the state; a non-match falls to default."""
    agent = _make_agent("@Manager")
    hit = agent.run_sync("about sandpile", deps=_Deps("@Manager"), output_type=_StructuredOutput)
    assert len(hit.messages) == 2

    miss = agent.run_sync("unrelated topic", deps=_Deps("@Manager"), output_type=_StructuredOutput)
    assert miss.messages == []


def test_expert_default_empty_twice() -> None:
    """@Expert is served by `default` (empty) for both inbound messages."""
    agent = _make_agent("@Expert")
    deps = _Deps("@Expert")
    first = agent.run_sync("notification one", deps=deps, output_type=_StructuredOutput)
    second = agent.run_sync("notification two", deps=deps, output_type=_StructuredOutput)
    assert first.messages == []
    assert second.messages == []


def test_matching_state_consumed_once() -> None:
    """A matched state is consumed; a second matching prompt falls to default."""
    agent = _make_agent("@Manager")
    deps = _Deps("@Manager")
    first = agent.run_sync("sandpile please", deps=deps, output_type=_StructuredOutput)
    second = agent.run_sync("sandpile again", deps=deps, output_type=_StructuredOutput)
    assert len(first.messages) == 2
    assert second.messages == []


# ---------------------------------------------------------------------------
# FR5 / FR6 — faithful event stream incl. simulated tools
# ---------------------------------------------------------------------------


def test_event_stream_order() -> None:
    """Observer sees inbound request, tool call/return, then final response."""
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput)

    kinds = [type(e).__name__ for e in rec.events]
    # Key subsequence in order: inbound msg, tool-call msg+event, tool-return
    # msg+event, final msg + usage.
    assert kinds.index("ToolCallEvent") < kinds.index("ToolReturnEvent")
    assert kinds.index("ToolReturnEvent") < _last_index(kinds, "LlmUsageEvent")

    tool_call = next(e for e in rec.events if isinstance(e, ToolCallEvent))
    tool_ret = next(e for e in rec.events if isinstance(e, ToolReturnEvent))
    assert tool_call.tool_name == "update_planning"
    assert tool_ret.tool_name == "update_planning"
    assert tool_call.tool_call_id == tool_ret.tool_call_id
    assert tool_ret.success is True


def test_tool_calls_simulated_not_executed() -> None:
    """The synthetic call/return parts exist; no real tool is referenced."""
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput)
    messages = agent.context.messages
    call_parts = [
        p
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    return_parts = [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert call_parts and return_parts
    args = call_parts[0].args
    assert isinstance(args, str) and args.startswith('{"create_tasks"')
    assert return_parts[0].content == "Done"


# ---------------------------------------------------------------------------
# FR7 — YAML config, fully Pydantic
# ---------------------------------------------------------------------------


def test_scenario_load() -> None:
    """The scenario parses into typed Pydantic models with a raw-JSON tool arg."""
    scenario = load_scenario(SANDPILE)
    assert isinstance(scenario, ScenarioConfig)
    assert set(scenario.agents) == {"@Manager", "@Assistant", "@Expert"}
    manager_state = scenario.agents["@Manager"].states[0]
    assert isinstance(manager_state.tools[0].args, str)
    assert manager_state.tools[0].name == "update_planning"


def test_scenario_cached() -> None:
    """load_scenario returns the same object for the same resolved path."""
    assert load_scenario(SANDPILE) is load_scenario(SANDPILE)


def test_resolve_ref_from_config_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution prefers config.model_cfg.model, then the env var."""
    assert _resolve_scenario_ref(_make_config("@Manager", model="path.yaml")) == "path.yaml"

    class _NoModel:
        model_cfg = None

    monkeypatch.setenv(SCENARIO_ENV_VAR, "env-scenario.yaml")
    assert _resolve_scenario_ref(_NoModel()) == "env-scenario.yaml"


def test_resolve_ref_unresolved_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing config field and no env var raises ValueError."""
    monkeypatch.delenv(SCENARIO_ENV_VAR, raising=False)

    class _NoModel:
        model_cfg = None

    with pytest.raises(ValueError):
        _resolve_scenario_ref(_NoModel())


# ---------------------------------------------------------------------------
# FR8 — sandpile scenario reproduced
# ---------------------------------------------------------------------------


def test_sandpile_manager() -> None:
    """@Manager: simulated update_planning + instruction/notification routing."""
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)
    out = agent.run_sync(
        "the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )
    assert [m.recipient for m in out.messages] == ["@Assistant", "@Expert"]
    assert "web search" in out.messages[0].message.lower()
    call = next(e for e in rec.events if isinstance(e, ToolCallEvent))
    assert call.tool_name == "update_planning"


def test_sandpile_assistant() -> None:
    """@Assistant: simulated web_search + findings routed to @Expert."""
    rec = _Recorder()
    agent = _make_agent("@Assistant", observer=rec)
    out = agent.run_sync(
        "research the sandpile model", deps=_Deps("@Assistant"), output_type=_StructuredOutput
    )
    assert [m.recipient for m in out.messages] == ["@Expert"]
    assert out.messages[0].message_type == "response"
    call = next(e for e in rec.events if isinstance(e, ToolCallEvent))
    assert call.tool_name == "web_search"


def test_sandpile_expert() -> None:
    """@Expert: empty StructuredOutput for both inbound messages."""
    agent = _make_agent("@Expert")
    deps = _Deps("@Expert")
    a = agent.run_sync("notification from @Manager", deps=deps, output_type=_StructuredOutput)
    b = agent.run_sync("notification from @Assistant", deps=deps, output_type=_StructuredOutput)
    assert a.messages == [] and b.messages == []


# ---------------------------------------------------------------------------
# FR9 — optional latency
# ---------------------------------------------------------------------------


def test_latency_applied() -> None:
    """A non-zero per-state latency injects measurable think-time."""

    async def _run() -> None:
        agent = _make_agent("@Manager")
        agent._scenario.agents["@Manager"].states[0].latency_ms = 50
        start = time.perf_counter()
        await agent.run("the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput)
        assert time.perf_counter() - start >= 0.045

    asyncio.run(_run())


def test_zero_latency_no_delay() -> None:
    """The default zero latency adds no measurable delay."""

    async def _run() -> None:
        agent = _make_agent("@Expert")
        start = time.perf_counter()
        await agent.run("anything", deps=_Deps("@Expert"))
        assert time.perf_counter() - start < 0.5

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Determinism / boundary
# ---------------------------------------------------------------------------


def test_determinism_across_instances() -> None:
    """Two fresh agents produce identical routed output for the same input."""
    a = _make_agent("@Manager").run_sync(
        "the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )
    b = _make_agent("@Manager").run_sync(
        "the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )
    assert a.model_dump() == b.model_dump()


def test_no_sibling_import() -> None:
    """The loadtest source must not import sibling akgentic packages."""
    src = Path(__file__).parents[2] / "src" / "akgentic" / "llm" / "loadtest"
    for module in ("mock_agent.py", "scenario.py", "__init__.py"):
        text = (src / module).read_text(encoding="utf-8")
        for sibling in (
            "akgentic.core",
            "akgentic.tool",
            "akgentic.agent",
            "akgentic.team",
            "akgentic.catalog",
            "akgentic.infra",
        ):
            assert sibling not in text


def _last_index(items: list[str], value: str) -> int:
    """Return the index of the last occurrence of value."""
    return len(items) - 1 - items[::-1].index(value)


# ---------------------------------------------------------------------------
# Additional surface / branch coverage
# ---------------------------------------------------------------------------


def test_restore_context_filters_llm_messages() -> None:
    """restore_context loads only LlmMessageEvent payloads into the context."""
    agent = _make_agent("@Manager")
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput)
    snapshot_messages = agent.context.messages

    class _Carrier:
        def __init__(self, event: Any) -> None:
            self.event = event

    from akgentic.llm.event import LlmMessageEvent

    fresh = _make_agent("@Manager")
    events: list[Any] = [_Carrier(LlmMessageEvent(message=m)) for m in snapshot_messages]
    events.append(_Carrier(object()))  # ignored — not an LlmMessageEvent
    fresh.restore_context(events)
    assert len(fresh.context.messages) == len(snapshot_messages)


def test_multimodal_list_prompt_matches() -> None:
    """A multimodal list prompt is flattened to text for state matching."""
    agent = _make_agent("@Manager")
    out = agent.run_sync(
        ["please discuss the", "sandpile model"],
        deps=_Deps("@Manager"),
        output_type=_StructuredOutput,
    )
    assert len(out.messages) == 2


def test_regex_and_from_sender_matchers() -> None:
    """The regex and from_sender matchers gate state selection."""
    scenario_dict = {
        "name": "matchers",
        "agents": {
            "@A": {
                "states": [
                    {"id": "rx", "when": {"regex": r"\bfoo\d+"}, "respond": {"text": "rx-hit"}},
                    {"id": "snd", "when": {"from_sender": "@Boss"}, "respond": {"text": "snd-hit"}},
                ],
                "default": {"respond": {"text": "default"}},
            }
        },
    }
    scenario = ScenarioConfig.model_validate(scenario_dict)
    agent = _make_agent("@A")
    agent._scenario = scenario

    assert agent.run_sync("foo42 here", deps=_Deps("@A")) == "rx-hit"
    assert agent.run_sync("from @Boss now", deps=_Deps("@A")) == "snd-hit"
    assert agent.run_sync("nothing matches", deps=_Deps("@A")) == "default"


# ---------------------------------------------------------------------------
# Story 11-3 (FR7) — synchronous close() parity with ReactAgent
# ---------------------------------------------------------------------------


def test_init_owns_open_nonrunning_loop() -> None:
    """``__init__`` owns its own loop, open and not running, current on the thread (AC #4)."""
    agent = _make_agent("@Manager")
    try:
        assert isinstance(agent._loop, asyncio.AbstractEventLoop)
        assert not agent._loop.is_closed()
        assert not agent._loop.is_running()
        assert agent._closed is False
        # set_event_loop made the agent's loop current on the constructing thread.
        assert asyncio.get_event_loop() is agent._loop
    finally:
        agent.close()


def test_close_is_callable_and_returns_none() -> None:
    """``close()`` is synchronous, returns None, and closes the owned loop (AC #1)."""
    agent = _make_agent("@Manager")
    assert agent.close() is None
    assert agent._loop.is_closed()
    assert agent._closed is True


def test_close_is_idempotent() -> None:
    """A second and third ``close()`` is a harmless no-op (AC #2)."""
    agent = _make_agent("@Manager")
    agent.close()
    # Repeated close() must not raise and must not error on the closed loop.
    agent.close()
    agent.close()
    assert agent._loop.is_closed()


def test_close_builds_no_model_or_client() -> None:
    """``close()`` never builds a model/client; the provider guard is never tripped (AC #3)."""
    agent = _make_agent("@Manager")
    assert agent._model is None and agent._http_client is None
    with patch.object(agent, "_build_model", side_effect=AssertionError("must not build")):
        agent.close()
    # Zero-token guarantee preserved across teardown.
    assert agent._model is None and agent._http_client is None


def test_close_cancels_pending_tasks() -> None:
    """``close()`` cancels stragglers left on the loop before closing it (AC #5)."""
    agent = _make_agent("@Manager")

    async def forever() -> None:
        await asyncio.sleep(3600)

    # Schedule a long-lived task on the agent's loop without running it, so it is
    # still pending when close() runs the cancel step.
    pending = agent._loop.create_task(forever())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent.close()

    assert pending.cancelled()
    assert agent._loop.is_closed()
    messages = [str(w.message) for w in caught]
    assert not any("pending" in m.lower() for m in messages), messages


def test_close_logs_teardown_failure_and_still_closes_loop() -> None:
    """A failing ``aclose()`` is logged (not raised) and the loop still closes (AC #5)."""
    import akgentic.llm.loadtest.mock_agent as mock_mod

    async def boom(_self: MockReactAgent) -> None:
        raise RuntimeError("teardown boom")

    agent = _make_agent("@Manager")
    with (
        patch.object(MockReactAgent, "aclose", new=boom),
        patch.object(mock_mod, "logger") as log,
    ):
        # close() must NOT propagate the teardown failure.
        agent.close()

    log.warning.assert_called_once()
    assert log.warning.call_args.kwargs.get("exc_info") is True
    assert agent._loop.is_closed()


def test_run_sync_runs_on_owned_loop() -> None:
    """``run_sync`` runs the coroutine on ``self._loop`` (AC #7)."""
    used_loop: list[asyncio.AbstractEventLoop] = []

    async def stub_run(*_: Any, **__: Any) -> str:
        used_loop.append(asyncio.get_running_loop())
        return "ran-on-owned-loop"

    agent = _make_agent("@Expert")
    try:
        with patch.object(MockReactAgent, "run", new=stub_run):
            first = agent.run_sync("q", deps=_Deps("@Expert"))
            second = agent.run_sync("q", deps=_Deps("@Expert"))
        assert first == "ran-on-owned-loop"
        assert second == "ran-on-owned-loop"
        assert used_loop == [agent._loop, agent._loop]
        assert not agent._loop.is_closed()
    finally:
        agent.close()


def test_run_sync_after_close_raises() -> None:
    """Calling ``run_sync`` on a closed mock raises (AC #7)."""
    agent = _make_agent("@Expert")
    agent.close()
    with pytest.raises(RuntimeError):
        agent.run_sync("q", deps=_Deps("@Expert"))


# ---------------------------------------------------------------------------
# Epic 21 / FR5 — conclude_without_tools parity with ReactAgent
# ---------------------------------------------------------------------------


def test_conclude_without_tools_emits_no_tool_events() -> None:
    """The mock's conclusion replays a state with NO tool events (AC #17).

    ``@Manager``'s matching state carries a tool stub, so a plain ``run()`` on the
    same prompt emits ``ToolCallEvent`` + ``ToolReturnEvent`` — the control that
    gives this assertion teeth. "No tools" for the mock means that loop is skipped.
    """
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)

    out = agent.conclude_without_tools_sync(
        "the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )

    assert [m.recipient for m in out.messages] == ["@Assistant", "@Expert"]
    assert not [e for e in rec.events if isinstance(e, (ToolCallEvent, ToolReturnEvent))]


def test_run_on_the_same_state_does_emit_tool_events() -> None:
    """Control for the test above: the state really does carry a tool stub."""
    rec = _Recorder()
    agent = _make_agent("@Manager", observer=rec)

    agent.run_sync("the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput)

    assert [e for e in rec.events if isinstance(e, ToolCallEvent)]
    assert [e for e in rec.events if isinstance(e, ToolReturnEvent)]


def test_conclude_without_tools_writes_no_tool_parts_to_context() -> None:
    """No synthetic tool call/return parts reach the context either (AC #17)."""
    agent = _make_agent("@Manager")
    agent.conclude_without_tools_sync(
        "the sandpile model", deps=_Deps("@Manager"), output_type=_StructuredOutput
    )

    messages = agent.context.messages
    assert not [
        p
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    assert not [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]


@pytest.mark.asyncio
async def test_conclude_without_tools_consumes_the_state() -> None:
    """A conclusion is a real turn: it consumes the state it matched (AC #17).

    The honest mirror of the real class, where a conclusion is an actual run. A
    bypass that preserved the state for a later ``run()`` would make the mock
    diverge from what it stands in for.
    """
    agent = _make_agent("@Manager")
    deps = _Deps("@Manager")

    first = await agent.conclude_without_tools(
        "sandpile please", deps=deps, output_type=_StructuredOutput
    )
    second = await agent.run("sandpile again", deps=deps, output_type=_StructuredOutput)

    assert len(first.messages) == 2
    assert second.messages == []


def test_conclude_without_tools_sync_after_close_raises() -> None:
    """The sync bridge carries the mock's own closed-agent guard (AC #18)."""
    agent = _make_agent("@Expert")
    agent.close()
    with pytest.raises(RuntimeError, match="MockReactAgent is closed"):
        agent.conclude_without_tools_sync("wrap it up", deps=_Deps("@Expert"))


def test_conclude_without_tools_sync_runs_on_the_owned_loop() -> None:
    """The bridge uses ``run_until_complete`` on the mock's own loop (AC #18)."""
    used_loops: list[asyncio.AbstractEventLoop] = []

    async def stub_conclude(*_: Any, **__: Any) -> str:
        used_loops.append(asyncio.get_running_loop())
        return "ran-on-owned-loop"

    agent = _make_agent("@Expert")
    try:
        with patch.object(MockReactAgent, "conclude_without_tools", new=stub_conclude):
            first = agent.conclude_without_tools_sync("once", deps=_Deps("@Expert"))
            second = agent.conclude_without_tools_sync("twice", deps=_Deps("@Expert"))
        assert first == second == "ran-on-owned-loop"
        assert used_loops == [agent._loop, agent._loop]
        assert not agent._loop.is_closed()
    finally:
        agent.close()


def test_conclusion_signatures_match_the_real_agent() -> None:
    """Same signatures on both classes, so a substitution cannot pass falsely (AC #16).

    A mock whose conclusion took different parameters would let an
    ``akgentic-agent`` test go green on a call the real ``ReactAgent`` rejects —
    which is exactly what drop-in parity exists to prevent.

    The model-roster trio is here for the same reason and for a sharper one: the mock is
    a **standalone class**, not a ``ReactAgent`` subclass, so it inherits nothing. A
    ``BaseAgent`` that delegates ``switch_model`` would ``AttributeError`` against a mock
    under the load-test flag unless these three are written out here too.
    """
    for name in (
        "conclude_without_tools",
        "conclude_without_tools_sync",
        "switch_model",
        "active_model",
        "model_roster",
    ):
        assert inspect.signature(getattr(MockReactAgent, name)) == inspect.signature(
            getattr(ReactAgent, name)
        )


def test_the_mock_refuses_to_switch_models() -> None:
    """A mock cannot switch: the scenario is bound to ``model_cfg.model`` at construction.

    Refusing is the honest answer — accepting would leave the config naming one scenario
    while the state machine replays another, silently. And ``_model``/``_http_client`` are
    ``None`` by the zero-token guarantee, so there is nothing to build.
    """
    agent = _make_agent("@Expert")
    try:
        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model("openai:gpt-4o")
        assert "openai:gpt-4o" in str(exc.value)
        assert agent._model is None
        assert agent._http_client is None
    finally:
        agent.close()


def test_the_mock_raises_the_real_agents_switch_error_class() -> None:
    """One class object, imported not redefined — cf. the usage-limit tier classes.

    A second definition here would be a different class, so an ``except ModelSwitchError``
    written against the real agent would silently not catch the mock's.
    """
    from akgentic.llm import agent as real_agent_module
    from akgentic.llm.loadtest import mock_agent as mock_module

    assert mock_module.ModelSwitchError is real_agent_module.ModelSwitchError
    assert issubclass(ModelSwitchError, ValueError)


def test_the_mock_readers_answer_truthfully() -> None:
    """``active_model`` reports what ``config`` carries; a config with no roster reads empty.

    ``model_roster`` uses ``getattr``: ``config`` is duck-typed here, and a load-test
    caller's config need not carry a roster field at all.
    """
    agent = _make_agent("@Expert")
    try:
        assert agent.active_model() is agent._config.model_cfg
        assert agent.model_roster() == []
    finally:
        agent.close()


def test_mock_defines_no_usage_limit_exception_classes() -> None:
    """The tier classes have exactly one definition, in ``akgentic.llm.agent`` (AC #19).

    A second definition in the mock module would be a different class object, so an
    ``except RunUsageLimitError`` written against the real agent would silently not
    catch the mock's — divergence that no behavioural test would surface.
    """
    from akgentic.llm import agent as real_agent_module
    from akgentic.llm.loadtest import mock_agent as mock_module

    for name in ("UsageLimitError", "RunUsageLimitError", "AgentUsageLimitError"):
        defined_here = getattr(mock_module, name, None)
        assert defined_here is None or defined_here is getattr(real_agent_module, name)


def test_real_and_mock_close_without_branching() -> None:
    """A caller can ``close()`` real and mock with no isinstance/type branch (AC #9)."""
    real = ReactAgent(
        config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    )
    mock = _make_agent("@Manager")

    # No isinstance/type/hasattr branch at the call site — only close() is called
    # on each agent. The real agent never runs (zero token egress).
    for agent in (real, mock):
        agent.close()
        agent.close()  # idempotent second call
        assert agent._loop.is_closed()
