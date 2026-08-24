"""Tests for dangling-tool-call healing, now ``HealingCapability``'s.

Covers Story 5.1 / ADR-003: When the REACT loop fails mid-execution, any
``ModelResponse`` whose ``ToolCallPart`` entries never received results is
healed by appending a ``ModelRequest`` with matching ``ToolReturnPart``
entries. This prevents the 'unprocessed tool calls' error on the next
``run()``.

The unit tests invoke the healer directly, against a manually populated context, so
the healing logic can be exercised without a real LLM round-trip. Its subject moved
in story 23-2 — ``ReactAgent._heal_unprocessed_tool_calls`` is gone and the healer is
``HealingCapability._heal`` — but the context, the assertions and the event checks are
unchanged; ``_heal_through_the_capability`` is the one-line adapter.

The run-level tests underneath are driven by a **real** model: healing is
``HealingCapability.on_run_error`` now, and no capability hook fires when ``iter()``
is stubbed out.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

from akgentic.llm import (
    ConclusionDecision,
    HealingCapability,
    LimitRecoveryCapability,
    LlmMessageEvent,
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
    RunUsageLimits,
    ToolReturnEvent,
    UsageLimitError,
)
from akgentic.llm.agent import RUN_LIMIT_HEALING_MESSAGE

RUN_ID = uuid.UUID("cf92c35f-4ee9-4cff-8361-b8ce3827e021")


class _NeverConcludes(LimitRecoveryCapability):
    """The opt-out seam: a run-tier breach raises instead of concluding.

    This file's subject is what the BREACHING run leaves in the context. The default
    recovery policy answers a breach with a second, tool-free run whose own messages then
    sit on top of the healing request, so the trailing message is no longer the one under
    test. Declining keeps the run under test the only one there is.
    """

    async def handle_limit_exceeded(
        self, ctx: RunContext[Any], *, error: UsageLimitExceeded
    ) -> ConclusionDecision | None:
        """Decline to conclude."""
        return None


class _EventCapture:
    """Capture all domain events emitted on a ContextManager."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


def _make_agent() -> ReactAgent:
    """Construct a ReactAgent with a minimal offline config.

    The underlying pydantic-ai Agent is never invoked by the unit tests — they run the
    healer directly on a manually populated context.
    """
    config = ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    return ReactAgent(config=config)


def _heal_through_the_capability(agent: ReactAgent, model_message: str) -> None:
    """Heal the agent's context — the one line that changed when the subject moved.

    ``HealingCapability`` needs nothing from the agent but its ``ContextManager``, which
    is exactly why healing could move out of ``ReactAgent`` at all.
    """
    HealingCapability(context=agent._context)._heal(model_message)


def _tool_calling_model(tool_name: str) -> FunctionModel:
    """A real model that answers every request with the same single tool call."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={})])

    return FunctionModel(model_fn)


def _response_with_tool_calls(*call_specs: tuple[str, str]) -> ModelResponse:
    """Build a ModelResponse with the given (tool_name, tool_call_id) tuples."""
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name=name, tool_call_id=call_id, args="{}")
            for name, call_id in call_specs
        ],
        timestamp=datetime.now(),
        run_id=RUN_ID,
    )


# ---------------------------------------------------------------------------
# AC-1 / AC-2: healing appends ModelRequest with ToolReturnParts
# ---------------------------------------------------------------------------


class TestHealingAppendsToolReturns:
    """AC-1 / AC-2: Last message is ModelResponse with tool calls → heal."""

    def test_appends_model_request_with_one_tool_return_per_call(self) -> None:
        """Single tool call is healed with a single ToolReturnPart."""
        agent = _make_agent()
        agent._context.add_message(_response_with_tool_calls(("my_tool", "call_xyz")))

        before = len(agent._context.messages)
        _heal_through_the_capability(agent, "test error")

        messages = agent._context.messages
        assert len(messages) == before + 1
        healed = messages[-1]
        assert isinstance(healed, ModelRequest)
        tool_returns = [p for p in healed.parts if isinstance(p, ToolReturnPart)]
        assert len(tool_returns) == 1

    def test_tool_return_matches_tool_call_fields(self) -> None:
        """AC-1/AC-2: Each ToolReturnPart mirrors the ToolCallPart's name and id."""
        agent = _make_agent()
        agent._context.add_message(_response_with_tool_calls(("search", "call_001")))

        _heal_through_the_capability(agent, "boom")

        healed = agent._context.messages[-1]
        assert isinstance(healed, ModelRequest)
        part = healed.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == "search"
        assert part.tool_call_id == "call_001"

    def test_tool_return_content_is_exactly_the_message_passed_in(self) -> None:
        """The content is the caller's string verbatim — no wrapper added (AC #2).

        ``in`` would still pass with the old
        ``"Error: tool call aborted due to failure: …"`` prefix back in place; the
        equality is what pins its removal, and what makes each call site responsible
        for supplying a complete sentence.
        """
        agent = _make_agent()
        agent._context.add_message(_response_with_tool_calls(("search", "call_001")))

        _heal_through_the_capability(agent, "RuntimeError: connection reset")

        healed = agent._context.messages[-1]
        assert isinstance(healed, ModelRequest)
        part = healed.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.content == "RuntimeError: connection reset"


# ---------------------------------------------------------------------------
# AC-3: no-op conditions
# ---------------------------------------------------------------------------


class TestNoOpConditions:
    """AC-3: Healing is a no-op when there is nothing to heal."""

    def test_noop_when_context_empty(self) -> None:
        """Empty context → no message appended."""
        agent = _make_agent()
        assert agent._context.messages == []

        _heal_through_the_capability(agent, "error")

        assert agent._context.messages == []

    def test_noop_when_last_message_is_model_request(self) -> None:
        """Last message is ModelRequest → no heal."""
        agent = _make_agent()
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="hi")]))

        before = len(agent._context.messages)
        _heal_through_the_capability(agent, "error")

        assert len(agent._context.messages) == before

    def test_noop_when_last_response_has_no_tool_calls(self) -> None:
        """Last ModelResponse is text-only → no heal."""
        agent = _make_agent()
        agent._context.add_message(
            ModelResponse(
                parts=[TextPart(content="just text")],
                timestamp=datetime.now(),
                run_id=RUN_ID,
            )
        )

        before = len(agent._context.messages)
        _heal_through_the_capability(agent, "error")

        assert len(agent._context.messages) == before


# ---------------------------------------------------------------------------
# AC-4: warning log
# ---------------------------------------------------------------------------


class TestWarningLog:
    """AC-4: A WARNING is logged with the count of healed tool calls."""

    def test_warning_logged_with_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """Emits WARNING-level log 'Healing %d unprocessed tool call(s) after error'."""
        agent = _make_agent()
        agent._context.add_message(
            _response_with_tool_calls(("t1", "c1"), ("t2", "c2"), ("t3", "c3"))
        )

        with caplog.at_level(logging.WARNING, logger="akgentic.llm.capabilities"):
            _heal_through_the_capability(agent, "error")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "Healing 3 unprocessed tool call(s) after error" in r.getMessage()
            for r in warning_records
        )

    def test_no_warning_when_noop(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning emitted when there is nothing to heal."""
        agent = _make_agent()

        with caplog.at_level(logging.WARNING, logger="akgentic.llm.capabilities"):
            _heal_through_the_capability(agent, "error")

        assert not any(
            "Healing" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )


# ---------------------------------------------------------------------------
# AC-1: parallel tool calls — all healed
# ---------------------------------------------------------------------------


class TestParallelToolCalls:
    """AC-1: Multiple pending tool calls are all healed in one ModelRequest."""

    def test_multiple_calls_all_healed(self) -> None:
        """Parallel tool-use: one ToolReturnPart per pending ToolCallPart."""
        agent = _make_agent()
        agent._context.add_message(
            _response_with_tool_calls(
                ("alpha", "call_a"),
                ("beta", "call_b"),
                ("gamma", "call_c"),
            )
        )

        _heal_through_the_capability(agent, "error")

        healed = agent._context.messages[-1]
        assert isinstance(healed, ModelRequest)
        tool_returns = [p for p in healed.parts if isinstance(p, ToolReturnPart)]
        assert len(tool_returns) == 3
        healed_ids = {p.tool_call_id for p in tool_returns}
        assert healed_ids == {"call_a", "call_b", "call_c"}
        healed_names = {p.tool_name for p in tool_returns}
        assert healed_names == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# AC-5: observer events emitted for healing message
# ---------------------------------------------------------------------------


class TestObserverEventsOnHeal:
    """AC-5: Healing goes through add_message() so observer events fire."""

    def test_llm_message_event_and_tool_return_events_emitted(self) -> None:
        """LlmMessageEvent emitted once, one ToolReturnEvent per healed part.

        ContextManager maps every ``ToolReturnPart`` to
        ``ToolReturnEvent(success=True)`` — the flag reflects part kind, not
        tool outcome — so the test asserts on presence, counts, and identity
        fields only, not on ``success``.
        """
        agent = _make_agent()
        capture = _EventCapture()
        agent.subscribe_context(capture)

        agent._context.add_message(_response_with_tool_calls(("t1", "c1"), ("t2", "c2")))
        # Drain events from adding the ModelResponse so we only inspect heal-time events.
        capture.events.clear()

        _heal_through_the_capability(agent, "error")

        llm_events = [e for e in capture.events if isinstance(e, LlmMessageEvent)]
        assert len(llm_events) == 1
        healed_msg = llm_events[0].message
        assert isinstance(healed_msg, ModelRequest)

        tool_return_events = [e for e in capture.events if isinstance(e, ToolReturnEvent)]
        assert len(tool_return_events) == 2
        assert {e.tool_call_id for e in tool_return_events} == {"c1", "c2"}
        assert {e.tool_name for e in tool_return_events} == {"t1", "t2"}


# ---------------------------------------------------------------------------
# AC-1 / AC-2: a failing run heals the context on its way out
# ---------------------------------------------------------------------------


class _RaisingRun:
    """Async context manager whose __aenter__ raises the configured error."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def __aenter__(self) -> "_RaisingRun":
        raise self._error

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return False


class TestRunInvokesHealing:
    """Run-level integration: a failing run heals the context through the capability.

    Real runs, not doubles: ``HealingCapability.on_run_error`` is a capability hook, and
    a stubbed ``iter()`` fires no hook at all. Each test asserts on the two trailing
    messages — the dangling ``ModelResponse`` the failure left, and the single healing
    ``ModelRequest`` that closes it out — which is the structural claim the old
    ``before + 1`` length check made against a hand-seeded context.
    """

    @pytest.mark.asyncio
    async def test_usage_limit_exceeded_heals_pending_tool_calls(self) -> None:
        """AC-1: a run-tier breach heals the context and still raises.

        Also AC #3: what lands in the model's context is the instruction sentence,
        not a stack. The two no-traceback assertions are the regression guard —
        restoring ``traceback.format_exc()`` as the healing wording turns them red
        while the structural assertions above stay green.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        )
        agent = ReactAgent(config=config, limit_recovery=_NeverConcludes())

        @agent.pydantic_agent.tool_plain
        def tool_a() -> str:
            return "ok"

        with agent.pydantic_agent.override(model=_tool_calling_model("tool_a")):
            with pytest.raises(UsageLimitError):
                await agent.run("test")

        messages = agent._context.messages
        healed = messages[-1]
        assert isinstance(healed, ModelRequest)
        tool_returns = [p for p in healed.parts if isinstance(p, ToolReturnPart)]
        assert len(tool_returns) == 1
        content = str(tool_returns[0].content)
        assert content == RUN_LIMIT_HEALING_MESSAGE
        assert "Traceback (most recent call last)" not in content
        assert "pydantic_ai" not in content

        dangling = messages[-2]
        assert isinstance(dangling, ModelResponse)
        assert tool_returns[0].tool_call_id == dangling.tool_calls[0].tool_call_id

    @pytest.mark.asyncio
    async def test_generic_exception_heals_and_reraises(self) -> None:
        """AC-2: a non-usage failure heals context and re-raises unchanged.

        AC #4: the healed content names the exception's type and message — what a
        model can act on — and carries no stack.
        """
        agent = _make_agent()

        class MyBoomError(Exception):
            pass

        @agent.pydantic_agent.tool_plain
        def tool_b() -> str:
            raise MyBoomError("kaboom")

        with agent.pydantic_agent.override(model=_tool_calling_model("tool_b")):
            with pytest.raises(MyBoomError, match="kaboom"):
                await agent.run("test")

        messages = agent._context.messages
        healed = messages[-1]
        assert isinstance(healed, ModelRequest)
        tool_returns = [p for p in healed.parts if isinstance(p, ToolReturnPart)]
        assert len(tool_returns) == 1
        assert tool_returns[0].tool_name == "tool_b"
        content = str(tool_returns[0].content)
        assert "MyBoomError: kaboom" in content
        assert "Traceback (most recent call last)" not in content

        dangling = messages[-2]
        assert isinstance(dangling, ModelResponse)
        assert tool_returns[0].tool_call_id == dangling.tool_calls[0].tool_call_id

    @pytest.mark.asyncio
    async def test_generic_branch_preserves_the_exception_object_and_traceback(self) -> None:
        """The traceback is removed from the LLM context ONLY — not from the caller.

        The whole of FR3 rests on this: operators still receive the stack, because
        ``Akgent._handle_failure`` formats it off the exception that leaves
        ``run()``. What this pins is that a non-usage failure is neither wrapped,
        replaced nor swallowed on its way out — anything that raised a new error, or
        returned instead of propagating, would break debugging to fix a prompt.
        Asserted by identity, which no message-level check can substitute for.

        Story 23-2 made that guarantee structural rather than behavioural: the
        ``except Exception`` clause that used to re-raise is deleted, not emptied, so
        there is no handler on this path to get wrong. The assertion is kept anyway,
        because it now also covers ``HealingCapability.on_run_error``, which sits on
        the same path and *could* suppress the error by returning a result — it
        deliberately re-raises the original object instead.
        """
        agent = _make_agent()
        agent._context.add_message(_response_with_tool_calls(("tool_c", "call_c")))
        sentinel = RuntimeError("sentinel failure")

        raising = _RaisingRun(sentinel)
        with patch.object(agent._pydantic_agent, "iter", return_value=raising):
            with pytest.raises(RuntimeError) as exc_info:
                await agent.run("test")

        assert exc_info.value is sentinel
        assert exc_info.value.__traceback__ is not None
