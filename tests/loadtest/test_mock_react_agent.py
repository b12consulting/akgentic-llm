"""Tests for MockReactAgent + the YAML scenario state machine (Epic 9 / ADR-007).

These tests stub ``deps`` (an object exposing ``.config.name``) and a
``StructuredOutput``-shaped output model — no akgentic-agent import — and assert
per-agent returned messages, ordered observer events, the zero-token / provider
guard, latency, scenario caching, and determinism.
"""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

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

SANDPILE = str(
    Path(__file__).parent / "scenarios" / "sandpile-research.yaml"
)


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
    for member in ("run", "run_sync", "subscribe_context", "checkpoint", "rewind",
                   "restore_context", "system_prompt", "tool"):
        assert callable(getattr(agent, member))

    def fn() -> None:  # decorator passthrough
        return None

    assert agent.system_prompt(fn) is fn
    assert agent.tool(fn) is fn
    snap = agent.checkpoint("cp1")
    assert snap.checkpoint_id == "cp1"
    agent.rewind("cp1")


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
    hit = agent.run_sync("about sandpile", deps=_Deps("@Manager"),
                         output_type=_StructuredOutput)
    assert len(hit.messages) == 2

    miss = agent.run_sync("unrelated topic", deps=_Deps("@Manager"),
                          output_type=_StructuredOutput)
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
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"),
                   output_type=_StructuredOutput)

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
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"),
                   output_type=_StructuredOutput)
    messages = agent.context.messages
    call_parts = [
        p for m in messages if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ToolCallPart)
    ]
    return_parts = [
        p for m in messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, ToolReturnPart)
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
    out = agent.run_sync("the sandpile model", deps=_Deps("@Manager"),
                         output_type=_StructuredOutput)
    assert [m.recipient for m in out.messages] == ["@Assistant", "@Expert"]
    assert "web search" in out.messages[0].message.lower()
    call = next(e for e in rec.events if isinstance(e, ToolCallEvent))
    assert call.tool_name == "update_planning"


def test_sandpile_assistant() -> None:
    """@Assistant: simulated web_search + findings routed to @Expert."""
    rec = _Recorder()
    agent = _make_agent("@Assistant", observer=rec)
    out = agent.run_sync("research the sandpile model", deps=_Deps("@Assistant"),
                         output_type=_StructuredOutput)
    assert [m.recipient for m in out.messages] == ["@Expert"]
    assert out.messages[0].message_type == "response"
    call = next(e for e in rec.events if isinstance(e, ToolCallEvent))
    assert call.tool_name == "web_search"


def test_sandpile_expert() -> None:
    """@Expert: empty StructuredOutput for both inbound messages."""
    agent = _make_agent("@Expert")
    deps = _Deps("@Expert")
    a = agent.run_sync("notification from @Manager", deps=deps,
                       output_type=_StructuredOutput)
    b = agent.run_sync("notification from @Assistant", deps=deps,
                       output_type=_StructuredOutput)
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
        await agent.run("the sandpile model", deps=_Deps("@Manager"),
                        output_type=_StructuredOutput)
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
        for sibling in ("akgentic.core", "akgentic.tool", "akgentic.agent",
                        "akgentic.team", "akgentic.catalog", "akgentic.infra"):
            assert sibling not in text


def _last_index(items: list[str], value: str) -> int:
    """Return the index of the last occurrence of value."""
    return len(items) - 1 - items[::-1].index(value)


# ---------------------------------------------------------------------------
# Additional surface / branch coverage
# ---------------------------------------------------------------------------


def test_run_sync_with_non_running_loop_falls_back() -> None:
    """A provided-but-idle event loop still routes through asyncio.run."""
    loop = asyncio.new_event_loop()
    try:
        agent = MockReactAgent(config=_make_config("@Expert"), event_loop=loop)
        assert agent.run_sync("hi", deps=_Deps("@Expert")) == ""
    finally:
        loop.close()


def test_restore_context_filters_llm_messages() -> None:
    """restore_context loads only LlmMessageEvent payloads into the context."""
    agent = _make_agent("@Manager")
    agent.run_sync("the sandpile model", deps=_Deps("@Manager"),
                   output_type=_StructuredOutput)
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
    out = agent.run_sync(["please discuss the", "sandpile model"],
                         deps=_Deps("@Manager"), output_type=_StructuredOutput)
    assert len(out.messages) == 2


def test_regex_and_from_sender_matchers() -> None:
    """The regex and from_sender matchers gate state selection."""
    scenario_dict = {
        "name": "matchers",
        "agents": {
            "@A": {
                "states": [
                    {"id": "rx", "when": {"regex": r"\bfoo\d+"},
                     "respond": {"text": "rx-hit"}},
                    {"id": "snd", "when": {"from_sender": "@Boss"},
                     "respond": {"text": "snd-hit"}},
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
