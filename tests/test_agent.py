"""Unit tests for ReactAgent implementation."""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from akgentic.llm import (
    AgentUsageLimitError,
    AgentUsageLimits,
    CompactionCapability,
    CompactionConfig,
    CompactionResult,
    ConclusionDecision,
    ContextManager,
    EventSourcingCapability,
    HealingCapability,
    LifetimeBudgetCapability,
    LimitRecoveryCapability,
    ModelConfig,
    ModelSwitchError,
    ReactAgent,
    ReactAgentConfig,
    RunUsageLimitError,
    RunUsageLimits,
    UsageLimitError,
    UserPrompt,
)
from akgentic.llm.agent import RUN_LIMIT_HEALING_MESSAGE
from akgentic.llm.capabilities import DEFAULT_CONCLUSION_REASON
from akgentic.llm.compaction import SummarizingCompaction
from akgentic.llm.config import model_roster_key
from akgentic.llm.event import (
    LlmContextClearedEvent,
    LlmContextCompactedEvent,
    LlmMessageEvent,
    LlmSystemPromptEvent,
    LlmUsageEvent,
    SystemPromptPartSnapshot,
    ToolCallEvent,
)


class _RecordingCompaction:
    """Fake async CompactionStrategy that records calls and returns a fixed result."""

    def __init__(self, result: CompactionResult) -> None:
        self._result = result
        self.calls = 0

    async def compact(self, messages: list) -> CompactionResult:
        self.calls += 1
        return self._result


def _over_budget_config() -> ReactAgentConfig:
    """Config whose auto-trigger threshold is 850 (context_length 1000 * 0.85)."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
        compaction_cfg=CompactionConfig(auto_trigger=True, trigger_ratio=0.85),
    )


class MockObserver:
    """Mock observer for context notifications."""

    def __init__(self):
        self.events = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


def _text_model(text: str = "test result") -> FunctionModel:
    """A model answering every request with one TextPart, for driving a REAL run.

    Persistence, system-prompt recording, healing and the agent-lifetime budget are
    capability hooks now, and a stubbed ``iter()`` fires none of them — so every test whose
    subject is one of those four has to reach the model rather than a double.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(stub)


def _spending_model(input_tokens: int, output_tokens: int = 0, text: str = "ok") -> FunctionModel:
    """A ``_text_model`` that also reports a caller-chosen token spend, for a REAL run.

    The replacement for the stubbed-``iter()`` spending double this file used to carry: the
    lifetime token accumulator is folded in ``LifetimeBudgetCapability.wrap_run``, which a
    stubbed ``iter()`` never reaches.

    The spend is exact, not estimated: ``FunctionModel`` only guesses usage when the
    response it is handed carries none (``models/function.py`` — ``if not
    response.usage.has_values()``), so setting it here pins the run's cost to the number
    the test asked for. One request per run, so a run costs exactly ``input + output``.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=text)],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    return FunctionModel(stub)


def _calling_model(
    tool_name: str, args: dict[str, Any] | None = None, **usage: int
) -> FunctionModel:
    """A model that answers every request with the same single tool call.

    Two REAL failure shapes are built on it: a tool that raises (the run dies partway) and
    a run-tier ``RunUsageLimits(run_request_limit=1)`` (the second request is refused). Both
    used to be injected by making a stubbed ``iter()`` raise, which no longer reaches the
    capability under test.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool_name, args=args or {})],
            usage=RequestUsage(**usage),
        )

    return FunctionModel(stub)


def _counting_model(requests: list[str]) -> FunctionModel:
    """A ``_text_model`` that records one entry per model request it is handed.

    The replacement for ``iter_call.assert_not_called()`` wherever the claim was "nothing
    was paid for": the refusal lives in ``LifetimeBudgetCapability.wrap_run``, which fires
    INSIDE ``iter()``, so the run must start for the refusal to happen at all. A request
    count of zero is the claim that survives, and it is the stronger of the two.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append("request")
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(stub)


def _history_recording_model(seen: list[list[ModelMessage]]) -> FunctionModel:
    """A ``_text_model`` that records the history list it is handed on each request.

    The only way to assert what auto-compaction actually delivered: the strategy's call
    count says a fold was computed, and the observer says an event was emitted, but neither
    says the RUN read the folded history. This does.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(stub)


def _tool_then_text_model(tool_name: str) -> FunctionModel:
    """Calls ``tool_name`` on the first request of a run, then answers on the second.

    Two model requests in one turn, which is what makes "at most once per turn" a real
    claim: auto-compaction fires from ``wrap_run``, once per RUN. Moved onto
    ``before_model_request`` — the other hook a fold could plausibly live on — it would
    fire twice here.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(m, ModelResponse) and m.tool_calls for m in messages):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={})])

    return FunctionModel(stub)


def _bare_run_context() -> RunContext[None]:
    """A synthetic RunContext for driving a capability hook outside a real run."""
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


@dataclass
class _NeverConcludes(LimitRecoveryCapability):
    """The documented opt-out seam: a run-tier breach raises, exactly as it used to.

    Mounted by every test whose subject is what a breach does to something *else* — the
    lifetime run counter, the healed context, the conclusion's own inputs. Those claims are
    about one run, and the default policy adds a second one on top of it; declining keeps
    each test's subject the single run it was written about. The recovery path itself has
    its own tests, which use the default.
    """

    async def handle_limit_exceeded(
        self, ctx: RunContext[Any], *, error: UsageLimitExceeded
    ) -> ConclusionDecision | None:
        """Decline to conclude."""
        return None


@pytest.fixture
def minimal_config():
    """Minimal ReactAgentConfig for testing."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
    )


@pytest.fixture
def config_with_limits():
    """ReactAgentConfig with usage limits."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        run_usage_limits=RunUsageLimits(run_request_limit=5, total_tokens_limit=1000),
    )


class TestUsageLimitError:
    """Test UsageLimitError exception."""

    def test_exception_creation(self):
        """Test UsageLimitError can be raised."""
        with pytest.raises(UsageLimitError):
            raise UsageLimitError("Limit exceeded")

    def test_exception_message(self):
        """Test UsageLimitError preserves message."""
        try:
            raise UsageLimitError("Request limit reached")
        except UsageLimitError as e:
            assert str(e) == "Request limit reached"


class TestReactAgentInit:
    """Test ReactAgent initialization."""

    def test_init_minimal_config(self, minimal_config):
        """Test agent initializes with minimal config."""
        agent = ReactAgent(config=minimal_config)
        assert agent is not None
        assert agent.context is not None
        assert agent.pydantic_agent is not None

    def test_init_with_deps_type(self, minimal_config):
        """Test agent initializes with deps_type."""

        class MyDeps:
            value: str = "test"

        agent = ReactAgent(config=minimal_config, deps_type=MyDeps)
        assert agent is not None

    def test_init_with_tools(self, minimal_config):
        """Test agent initializes with tools."""

        def my_tool(query: str) -> str:
            return f"Result: {query}"

        agent = ReactAgent(config=minimal_config, tools=[my_tool])
        assert agent is not None

    def test_init_with_result_type(self, minimal_config):
        """Test agent initializes with custom result_type (Pydantic model)."""
        from pydantic import BaseModel

        class CustomResult(BaseModel):
            value: str

        agent = ReactAgent(config=minimal_config, result_type=CustomResult)
        assert agent is not None

    def test_init_with_capabilities_present_in_root_capability(self, minimal_config):
        """A supplied capability is present in the constructed Agent's public capability surface."""
        from pydantic_ai.capabilities import Capability

        cap = Capability(id="custom-cap")
        agent = ReactAgent(config=minimal_config, capabilities=[cap])
        assert cap in agent.pydantic_agent.root_capability.capabilities

    def test_capabilities_omitted_equals_explicit_empty_list(self, minimal_config):
        """Omitting `capabilities` is behaviourally identical to passing `capabilities=[]`."""
        agent_omitted = ReactAgent(config=minimal_config)
        agent_explicit_empty = ReactAgent(config=minimal_config, capabilities=[])
        # Compared by type rather than by value: both stacks now carry the four internal
        # capabilities, each bound to ITS OWN agent's ContextManager, and dataclass
        # equality makes two capabilities on different contexts unequal. The claim being
        # made is about the shape of the stack, which the sequence of types expresses.
        assert [type(c) for c in agent_omitted.pydantic_agent.root_capability.capabilities] == [
            type(c) for c in agent_explicit_empty.pydantic_agent.root_capability.capabilities
        ]

    def test_internal_capabilities_precede_the_callers(self, minimal_config):
        """[LifetimeBudget, Compaction, EventSourcing, LimitRecovery, Healing, *caller].

        The one guard on an order that nothing else pins. Two behavioural couplings depend
        on it and have their own tests. The budget refuses a spent agent **before** compaction
        pays for a summarizer (``test_rejection_happens_before_compaction`` and its token
        twin, both verified red when Compaction is moved ahead of LifetimeBudget).
        LimitRecovery sits immediately BEFORE Healing so that Healing fires FIRST: the
        ``on_run_error`` walk is over ``reversed(self.capabilities)``, so the later entry
        runs first, and the recovery seam must see the healed context
        (``test_healing_has_already_run_when_the_seam_is_consulted``, verified red with the
        two swapped back).
        Compaction ahead of persistence is belt-and-braces rather than load-bearing —
        swapping those two leaves the outcome unchanged, because ``_anchor`` re-opens the
        cursor at the first node hook. First-in-the-list is outermost: pydantic-ai builds
        each ``wrap_run`` chain over ``reversed(self.capabilities)``.

        The five internal classes are asserted POSITIONALLY, as a contiguous block in that
        exact sequence, with the caller's after them. That is stronger than pairwise
        ``<`` comparisons: a swap of any adjacent pair fails it outright. Where the block
        *starts* is deliberately not asserted — pydantic-ai composes a base capability of
        its own into that surface and where that one sits is not this package's contract.
        Matched by type rather than by instance ref for a second reason: ``for_run`` hands
        some runs a fresh copy, so an instance ref would not match one either.

        **Standing caveat, deliberately not asserted.** pydantic-ai's ``CombinedCapability``
        topologically re-sorts the whole chain as soon as ANY capability declares
        ``get_ordering()``, so this is the shipped default for callers that declare nothing,
        not an invariant. Do not make an internal capability declare an ordering to pin it —
        that is a behavioural change owed its own decision.
        """
        from pydantic_ai.capabilities import Capability

        caller_cap = Capability(id="custom-cap")
        agent = ReactAgent(config=minimal_config, capabilities=[caller_cap])

        mounted = list(agent.pydantic_agent.root_capability.capabilities)
        types = [type(c) for c in mounted]
        expected = [
            LifetimeBudgetCapability,
            CompactionCapability,
            EventSourcingCapability,
            LimitRecoveryCapability,
            HealingCapability,
        ]
        start = types.index(LifetimeBudgetCapability)
        assert types[start : start + len(expected)] == expected
        assert start + len(expected) <= mounted.index(caller_cap)


class TestReactAgentCapabilityHook:
    """AC8: a supplied capability's before_model_request hook fires during a real run()."""

    @pytest.mark.asyncio
    async def test_before_model_request_hook_fires_during_run(self, minimal_config):
        """The capability chain runs for real — not just accepted into the signature."""
        from pydantic_ai.capabilities import AbstractCapability
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        class _RecordingCapability(AbstractCapability):
            """Minimal capability recording whether before_model_request fired."""

            def __init__(self) -> None:
                self.invoked = False

            async def before_model_request(self, ctx, request_context):
                self.invoked = True
                return request_context

        cap = _RecordingCapability()
        agent = ReactAgent(config=minimal_config, capabilities=[cap])

        def stub_model(messages: list, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="ok")])

        with agent.pydantic_agent.override(model=FunctionModel(stub_model)):
            result = await agent.run("hello")

        assert cap.invoked is True
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_capability_orphaning_a_tool_call_pair_is_repaired_by_the_framework(
        self, minimal_config
    ):
        """AC3 (story 17-4): the pre-v2 "no re-fold" claim does not hold under v2.

        `agent.py`'s ``capabilities`` docstring used to say the framework "does not
        re-run its orphan role=tool fold after capabilities run." Direct source
        verification against pydantic-ai 2.21.0 (``_agent_graph.py``'s
        ``_prepare_request``) found this false: on every model request,
        ``_clean_message_history(..., repair_last_response=True)`` runs AFTER
        ``before_model_request`` and silently synthesizes a matching
        ``ToolReturnPart`` for any dangling ``ToolCallPart`` — including one a
        capability itself just created. This test pins that corrected behavior
        against a capability that does exactly what the old docstring warned about:
        splits a tool call/return pair by deleting the return.
        """
        from dataclasses import replace as dc_replace

        from pydantic_ai.capabilities import AbstractCapability
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        tool_call_id = "call-1"

        class _OrphaningCapability(AbstractCapability):
            """Strips the ToolReturnPart matching tool_call_id from every message."""

            async def before_model_request(self, ctx, request_context):
                request_context.messages = [
                    dc_replace(
                        m,
                        parts=[
                            p
                            for p in m.parts
                            if not (
                                isinstance(p, ToolReturnPart) and p.tool_call_id == tool_call_id
                            )
                        ],
                    )
                    if isinstance(m, ModelRequest)
                    else m
                    for m in request_context.messages
                ]
                return request_context

        agent = ReactAgent(config=minimal_config, capabilities=[_OrphaningCapability()])
        # Seed a completed tool round-trip directly; before_model_request strips its
        # ToolReturnPart on the next run, orphaning the preceding ToolCallPart.
        agent.context.restore(
            [
                ModelRequest(parts=[UserPromptPart(content="first")]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name="foo", args={}, tool_call_id=tool_call_id)]
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name="foo", content="ok", tool_call_id=tool_call_id)]
                ),
            ]
        )

        received_messages: list = []

        def stub_model(messages: list, info: AgentInfo) -> ModelResponse:
            received_messages.extend(messages)
            return ModelResponse(parts=[TextPart(content="done")])

        with agent.pydantic_agent.override(model=FunctionModel(stub_model)):
            result = await agent.run("continue")

        assert result == "done"
        # The capability deleted the only ToolReturnPart for tool_call_id — if the
        # model still received one, the framework synthesized it after the
        # capability ran, contradicting the old "no re-fold" docstring claim.
        returned_ids = {
            p.tool_call_id
            for m in received_messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart)
        }
        assert tool_call_id in returned_ids

    @pytest.mark.asyncio
    async def test_run_usage_fold_emits_no_usage_deprecation_warning(self, minimal_config):
        """`_fold_run_usage`'s `run.usage` read (property, no parens) stays warning-free.

        Regression test for ADR-014 Phase 0 / FR1: pins the currently-correct
        accessor form against a real pydantic-ai ``AgentRun``. The deprecated form
        is calling ``run.usage()`` like the old method — nothing in this codebase
        does that, and this test fails immediately if it ever starts to.
        """
        import warnings

        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        agent = ReactAgent(config=minimal_config)

        def stub_model(messages: list, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="ok")])

        with agent.pydantic_agent.override(model=FunctionModel(stub_model)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await agent.run("hello")

        usage_warnings = [w for w in caught if "usage" in str(w.message).lower()]
        assert usage_warnings == []
        assert result == "ok"


class TestReactAgentEndStrategyRetryWins:
    """Story 17-5: pins v2's 'exhaustive'-strategy retry-wins invariant.

    `RuntimeConfig.end_strategy` defaults to ``"exhaustive"`` and is always passed
    explicitly to ``Agent(...)`` (`agent.py`), so pydantic-ai's own constructor
    default (which flipped `'early'`->`'graceful'` in v2) never applies here -- but
    v2's *implementation* of `'exhaustive'` itself changed underneath that default.
    v2 moved tool-call processing into `pydantic_ai/_tool_execution.py`
    (`_ExhaustiveProcessor`, `_apply_retry_wins`, `_is_retry_wins_trigger`): when a
    function tool call in the same round as an already-successful output tool call
    produces a `RetryPromptPart` (from `ModelRetry` or arg-validation failure), the
    output is suppressed and the run stays open for a further model turn. The pinned
    v1.107.0 baseline (`pydantic_ai/_agent_graph.py::process_tool_calls`) has no code
    path that reads a function tool's `RetryPromptPart` to affect an already-set
    `final_result` -- the output would win immediately, ending the run on that turn.
    """

    @pytest.mark.asyncio
    async def test_exhaustive_strategy_keeps_run_open_when_a_concurrent_function_tool_retries(
        self, monkeypatch
    ):
        from pydantic import BaseModel
        from pydantic_ai import ModelRetry
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        # google-gla is a non-native provider: get_output_type() returns the raw
        # BaseModel unwrapped, so pydantic-ai uses tool-based (ToolOutput) output --
        # the model must emit a discrete output ToolCallPart, matching the scenario.
        # The API key is never dereferenced: FunctionModel replaces the real model
        # before any run happens.
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        class RouteDecision(BaseModel):
            target: str

        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
        )
        assert config.runtime_cfg.end_strategy == "exhaustive"
        agent = ReactAgent(config=config, result_type=RouteDecision)

        @agent.pydantic_agent.tool_plain
        def flaky_tool(value: str) -> str:
            raise ModelRetry("needs correction")

        call_count = 0

        def stub_model(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            output_tool_name = info.output_tools[0].name
            if call_count == 1:
                # One round, two tool calls: an already-valid output AND a function
                # tool call that will retry.
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=output_tool_name,
                            args={"target": "billing"},
                            tool_call_id="out-1",
                        ),
                        ToolCallPart(
                            tool_name="flaky_tool",
                            args={"value": "x"},
                            tool_call_id="fn-1",
                        ),
                    ]
                )
            # Second turn: no more function tool calls, so nothing can retry-win --
            # this output finalizes the run.
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool_name,
                        args={"target": "billing"},
                        tool_call_id="out-2",
                    )
                ]
            )

        with agent.pydantic_agent.override(model=FunctionModel(stub_model)):
            result = await agent.run("route this")

        # Under v1.107.0, the first turn's output would have won immediately
        # (call_count == 1). Under the real v2 install, `flaky_tool`'s ModelRetry
        # sets `retry_wins_triggered`, `_apply_retry_wins` nulls the already-set
        # `final_result`, and the graph loops for a second model turn.
        assert call_count == 2
        assert result == RouteDecision(target="billing")


class TestReactAgentRun:
    """Test ReactAgent.run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_result(self, minimal_config):
        """Test run() returns result from pydantic-ai agent."""
        agent = ReactAgent(config=minimal_config)

        run = _StubRun(
            output="test result",
            yields=True,
            new_messages=[ModelRequest(parts=[UserPromptPart(content="test")])],
        )
        # Patch iter to return context manager directly
        with patch.object(agent._pydantic_agent, "iter", return_value=run):
            result = await agent.run("test query")
            assert result == "test result"

    @pytest.mark.asyncio
    async def test_run_updates_context(self, minimal_config):
        """Test context messages updated after run().

        Driven by a real model: the messages reach the context through
        ``EventSourcingCapability``, and no capability hook fires under a stubbed
        ``iter()``.
        """
        agent = ReactAgent(config=minimal_config)
        assert len(agent.context.messages) == 0

        with agent.pydantic_agent.override(model=_text_model()):
            result = await agent.run("test query")

        # Context should have messages after run: this run's request and its response.
        assert result == "test result"
        assert len(agent.context.messages) == 2

    @pytest.mark.asyncio
    async def test_usage_limit_error_raised(self, minimal_config):
        """Test UsageLimitError raised when pydantic-ai raises UsageLimitExceeded."""
        agent = ReactAgent(config=minimal_config)

        run = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))
        with patch.object(agent._pydantic_agent, "iter", return_value=run):
            with pytest.raises(UsageLimitError) as exc_info:
                await agent.run("test query")
            assert "Request limit exceeded" in str(exc_info.value)


class TestReactAgentProperties:
    """Test ReactAgent properties and methods."""

    def test_context_property(self, minimal_config):
        """Test context property returns ContextManager."""
        agent = ReactAgent(config=minimal_config)
        context = agent.context
        assert context is not None
        assert hasattr(context, "messages")

    def test_pydantic_agent_property(self, minimal_config):
        """Test pydantic_agent property returns pydantic-ai Agent."""
        agent = ReactAgent(config=minimal_config)
        pydantic_agent = agent.pydantic_agent
        assert pydantic_agent is not None
        # Check it has pydantic-ai Agent methods
        assert hasattr(pydantic_agent, "tool")
        assert hasattr(pydantic_agent, "system_prompt")


class TestReactAgentContextMethods:
    """Test ReactAgent context management methods."""

    @pytest.mark.asyncio
    async def test_subscribe_context(self, minimal_config):
        """Test subscribe_context() observer notified on message add.

        A real model, because the messages reach the context through
        ``EventSourcingCapability`` and no hook fires under a stubbed ``iter()``.
        """
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config)
        agent.subscribe_context(observer)

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("test query")

        # Observer should have been notified, for every message the run persisted
        notified = [e.message for e in observer.events if isinstance(e, LlmMessageEvent)]
        assert notified == agent.context.messages
        assert len(notified) == 2

    def test_checkpoint_and_rewind_wrappers_removed(self, minimal_config):
        """ReactAgent no longer exposes checkpoint/rewind wrappers (AC 4)."""
        agent = ReactAgent(config=minimal_config)
        assert not hasattr(agent, "checkpoint")
        assert not hasattr(agent, "rewind")


class TestReactAgentSyncMethod:
    """Test ReactAgent.run_sync() method."""

    def test_run_sync_works_synchronously(self, minimal_config):
        """Test run_sync() executes synchronously."""
        agent = ReactAgent(config=minimal_config)

        run = _StubRun(
            output="test result",
            yields=True,
            new_messages=[ModelRequest(parts=[UserPromptPart(content="test")])],
        )
        with patch.object(agent._pydantic_agent, "iter", return_value=run):
            result = agent.run_sync("test query")
            assert result == "test result"


class TestReactAgentSystemPrompts:
    """Test system prompt registration."""

    def test_current_datetime_prompt_registered(self, minimal_config):
        """Test current_datetime_prompt always registered."""
        agent = ReactAgent(config=minimal_config)
        # Datetime prompt is always registered
        # We verify by checking agent initialized without error
        assert agent is not None

    def test_system_prompt_decorator_wrapper(self, minimal_config):
        """Test system_prompt() decorator wrapper registers custom prompt."""
        agent = ReactAgent(config=minimal_config)

        # Register custom prompt
        @agent.system_prompt
        def custom_prompt(ctx):
            return "Custom system prompt"

        # Verify decorator returned a callable
        assert callable(custom_prompt)


class TestReactAgentToolDecorator:
    """Test tool decorator wrapper."""

    def test_tool_decorator_wrapper(self, minimal_config):
        """Test tool() decorator wrapper registers tool function."""

        # Create agent with deps_type to satisfy pydantic-ai requirement
        class MyDeps:
            pass

        agent = ReactAgent(config=minimal_config, deps_type=MyDeps)

        # Register tool with proper RunContext annotation
        from pydantic_ai import RunContext

        @agent.tool
        def search_tool(ctx: RunContext[MyDeps], query: str) -> list[str]:
            """Search for items matching query.

            Args:
                query: The search query string

            Returns:
                List of matching items
            """
            return [f"Result: {query}"]

        # Verify decorator returned a callable
        assert callable(search_tool)

    def test_decorators_preserve_the_decorated_function_type(self, minimal_config):
        """`@agent.tool` / `@agent.system_prompt` must not erase the function's type.

        A wrapper annotated ``(func: Any) -> Any`` silently retypes every
        registered function to ``Any``, so mypy stops checking all of its
        callers — a regression no behavioural test can observe. It has to be
        checked at runtime rather than with ``assert_type``, because CI
        type-checks ``src/`` only; an ``assert_type`` under ``tests/`` is a
        no-op nothing would ever read.

        Two assertions, together equivalent to the static claim:

        1. The same type variable goes in and comes out, so a caller keeps the
           signature it registered. Reverting either wrapper to ``-> Any``
           fails here.
        2. The premise that annotation rests on still holds — pydantic-ai hands
           back the original function object rather than a wrapper. If a future
           version starts wrapping, ``-> F`` becomes a lie and this goes red.
           Asserted for both wrappers: they reach pydantic-ai through different
           code paths (``tool()`` versus the ``system_prompt(dynamic=True)``
           closure), so one holding says nothing about the other.
        """
        for method in (ReactAgent.tool, ReactAgent.system_prompt):
            hints = get_type_hints(method)
            assert isinstance(hints["func"], TypeVar)
            assert hints["return"] is hints["func"]

        class MyDeps:
            pass

        agent = ReactAgent(config=minimal_config, deps_type=MyDeps)

        from pydantic_ai import RunContext

        def search_tool(ctx: RunContext[MyDeps], query: str) -> list[str]:
            """Search for items matching query.

            Args:
                query: The search query string

            Returns:
                List of matching items
            """
            return [f"Result: {query}"]

        def dynamic_prompt(ctx: RunContext[MyDeps]) -> str:
            return "Dynamic system prompt"

        assert agent.tool(search_tool) is search_tool
        assert agent.system_prompt(dynamic_prompt) is dynamic_prompt


class TestReactAgentUsageLimits:
    """Test usage limits conversion."""

    def test_usage_limits_converted(self, config_with_limits):
        """Test usage limits from config converted to pydantic-ai format."""
        agent = ReactAgent(config=config_with_limits)
        # Test that conversion happens without error
        pydantic_limits = agent._to_pydantic_limits(config_with_limits.run_usage_limits)
        assert pydantic_limits is not None
        # run_request_limit maps onto pydantic-ai's request_limit
        assert pydantic_limits.request_limit == 5
        assert pydantic_limits.total_tokens_limit == 1000

    def test_none_usage_limits(self, minimal_config):
        """Test None usage limits returns None."""
        agent = ReactAgent(config=minimal_config)
        pydantic_limits = agent._to_pydantic_limits(None)
        assert pydantic_limits is None

    def test_converter_reads_the_limits_object_it_is_handed(self):
        """Test the converter reads its argument's request limit, not the agent tier.

        Narrow by construction: _to_pydantic_limits takes the tier as a parameter, so
        this pins the conversion only. Which tier run() hands it is the part that can
        actually regress — see test_run_hands_the_run_tier_to_pydantic_ai.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(run_request_limit=5),
            agent_usage_limits=AgentUsageLimits(agent_request_limit=1),
        )
        agent = ReactAgent(config=config)
        pydantic_limits = agent._to_pydantic_limits(config.run_usage_limits)
        assert pydantic_limits.request_limit == 5


class _StubRun:
    """Full ``AgentRun``-protocol test double: the single shared component every
    test in this file constructs (directly or via a small factory) instead of
    redefining the protocol locally.

    ``usage`` is a property, not a method, matching pydantic-ai's real
    ``AgentRun.usage`` (calling it as a method emits the deprecation warning
    story 16-1 guards against). The constructor accepts and discards ``iter()``'s
    real ``*args``/``**kwargs`` so it survives a future ``iter()`` signature
    change untouched; pass ``captured=`` (or use ``_capturing_stub_run`` below)
    to record those kwargs instead of discarding them.

    ``ReactAgent`` drives ``Agent.run()``, which is implemented on top of
    ``iter()`` — so patching ``iter`` still intercepts it — but then drives what
    ``iter()`` returned through a wider protocol: it reads ``next_node`` before
    its loop and ends on ``assert agent_run.result is not None``. ``next_node``
    is therefore any non-``End`` sentinel; because ``result`` is always a
    non-``None`` ``MagicMock``, ``run()`` breaks out on the first iteration and
    never calls ``next()``.

    A stubbed ``iter()`` means **no capability hook fires**, so a run driven by
    this double persists nothing and records no system prompt — both now belong
    to ``EventSourcingCapability``. ``yields`` and ``new_messages`` drive the
    ``async for`` protocol only; a test asserting on persistence, on
    system-prompt events or on healing must use a real model instead.
    """

    def __init__(
        self,
        *args,
        output="ok",
        new_messages=None,
        yields=False,
        enter_raises=None,
        captured=None,
        **kwargs,
    ):
        if captured is not None:
            captured.update(kwargs)
        self.result = MagicMock(output=output)
        # Non-End sentinel: run() reads this before its loop, then breaks on `result`.
        self.next_node = MagicMock()
        self._usage = RunUsage()
        self._new_messages = new_messages if new_messages is not None else []
        self._yields = yields
        self._enter_raises = enter_raises

    @property
    def usage(self):
        return self._usage

    async def __aenter__(self):
        if self._enter_raises is not None:
            raise self._enter_raises
        return self

    async def __aexit__(self, *_):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._yields and not hasattr(self, "_iterated"):
            self._iterated = True
            return None
        raise StopAsyncIteration

    def new_messages(self):
        return self._new_messages


def _capturing_stub_run(captured: dict, **stub_kwargs):
    """iter() ``side_effect`` that records the call's real kwargs into `captured`.

    Complements passing ``captured=`` directly into ``_StubRun(...)`` (which only
    works with ``return_value=``, a single fixed instance): a ``side_effect=``
    factory must be invoked fresh, with ``iter()``'s actual args/kwargs, on every
    call — this returns such a factory.

    It reports no token spend, and cannot: the lifetime fold's anchor is
    ``wrap_run``'s ``ctx.usage``, which a stubbed ``iter()`` never reaches, and
    ``ReactAgent`` no longer passes a ``run(usage=...)`` for a double to write
    through. A test whose subject is the spend drives ``_spending_model`` instead.
    """

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return _StubRun(**stub_kwargs)

    return factory


def _agent_limit_config(limit):
    """Config carrying only an agent-tier run budget."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        agent_usage_limits=AgentUsageLimits(agent_request_limit=limit),
    )


def _agent_token_config(**token_limits):
    """Config carrying only agent-tier TOKEN limits (no run budget)."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        agent_usage_limits=AgentUsageLimits(**token_limits),
    )


def _armed_compaction_config(**agent_limits):
    """Config with an ARMED auto-trigger (threshold 850) plus agent-tier limits.

    Both halves are load-bearing wherever this is used. "The summarizer never ran" is only
    a claim about the refusal if compaction would otherwise have fired: with the gate
    disarmed the assertion holds for the wrong reason and stays green under the mutation
    that deletes the budget checks.
    """
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
        compaction_cfg=CompactionConfig(auto_trigger=True, trigger_ratio=0.85),
        agent_usage_limits=AgentUsageLimits(**agent_limits),
    )


def _both_tier_config(*, run_request_limit, agent_request_limit):
    """Config carrying a run-tier request budget AND an agent-tier run budget.

    The two are deliberately given different values wherever it is used, so a test
    distinguishes "reads the run tier" from "reads whichever tier is set".
    """
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        run_usage_limits=RunUsageLimits(run_request_limit=run_request_limit),
        agent_usage_limits=AgentUsageLimits(agent_request_limit=agent_request_limit),
    )


class TestReactAgentRunCountEnforcement:
    """Test the agent-lifetime run budget: pre-flight, check-then-consume."""

    def test_fresh_agent_starts_at_zero(self, minimal_config):
        """Test a newly constructed agent has consumed no runs."""
        agent = ReactAgent(config=minimal_config)
        assert agent._agent_run_count == 0

    def test_runs_up_to_the_limit_succeed(self):
        """Test calls 1..N execute and consume exactly N."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            agent.run_sync("second")
        assert agent._agent_run_count == 2

    def test_run_past_the_limit_raises_and_does_not_consume(self):
        """Test call N+1 is rejected and leaves the counter pinned at the limit."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("third")
        # The rejected call never executed, so it consumed nothing: runs consumed,
        # never runs attempted.
        assert agent._agent_run_count == 2
        assert str(exc_info.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"

    def test_rejection_does_not_reach_the_tool_call_healing_path(self):
        """Test a rejected run never routes through the healing capability.

        Asserted on the call, not on resulting context: healing is a no-op on an
        empty context, so an emptiness check would pass even from inside the try.
        The patch moved with its subject — healing is ``HealingCapability._heal``
        now — and the claim is unchanged: a pre-flight rejection never reaches it.
        """
        agent = ReactAgent(config=_agent_limit_config(1))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            with patch.object(HealingCapability, "_heal") as heal:
                with pytest.raises(UsageLimitError):
                    agent.run_sync("second")
        heal.assert_not_called()

    def test_run_hands_the_run_tier_to_pydantic_ai(self):
        """Test run() converts the run tier for pydantic-ai, never the agent tier."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(run_request_limit=5),
            agent_usage_limits=AgentUsageLimits(agent_request_limit=1),
        )
        agent = ReactAgent(config=config)
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return _StubRun()

        with patch.object(agent._pydantic_agent, "iter", side_effect=capture):
            agent.run_sync("only run")
        # 5 (run tier), never 1 (agent tier): differently valued so this distinguishes
        # "reads the run tier" from "reads whichever tier is set".
        assert captured["usage_limits"].request_limit == 5

    def test_rejection_happens_before_compaction(self):
        """Test the budget check precedes compaction: a rejected run pays no summarizer.

        Re-expressed against the STRATEGY, because ``_maybe_compact`` — the old patch
        target — no longer exists: auto-compaction is ``CompactionCapability.wrap_run``,
        mounted second, nested inside ``LifetimeBudgetCapability.wrap_run``. The claim is
        strictly stronger than the one it replaces: no summarizer **LLM call** at all,
        rather than no method call. Nothing in the code says so — only the assembly order
        does, so moving Compaction ahead of LifetimeBudget turns this red.

        The gate is deliberately ARMED (900 > 850): with compaction off the assertion
        would hold for the wrong reason.
        """
        agent = ReactAgent(config=_armed_compaction_config(agent_request_limit=1))
        strategy = _RecordingCompaction(CompactionResult("S", 1))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            agent._compactor.strategy = strategy
            agent._context._last_input_tokens = 900
            with pytest.raises(UsageLimitError):
                agent.run_sync("second")
        assert strategy.calls == 0

    def test_counter_advances_when_the_wrapped_call_raises(self):
        """Test a run that fails partway has already been counted.

        The failure is real: the model calls a tool that raises, so the run dies after
        the counter advanced. Injecting it by making ``iter()`` raise no longer works —
        the counter lives in a ``wrap_run`` hook that a stubbed ``iter()`` never reaches.
        """
        agent = ReactAgent(config=_agent_limit_config(3))

        @agent.pydantic_agent.tool_plain
        def boom_tool() -> str:
            raise RuntimeError("boom")

        with agent.pydantic_agent.override(model=_calling_model("boom_tool")):
            with pytest.raises(RuntimeError):
                agent.run_sync("first")
        assert agent._agent_run_count == 1

    def test_counter_advances_when_the_run_tier_limit_fires(self):
        """Test a run-tier breach still consumes the agent-tier budget.

        The breach is real: one request is all the run tier allows, and the model asks
        for a tool, so the follow-up request pydantic-ai needs is the one refused.

        Recovery is declined so the claim stays about ONE run: the default policy answers a
        breach with a tool-free conclusion, which is a second run through the same stack and
        consumes a second unit of the very budget this test counts.
        """
        agent = ReactAgent(
            config=_both_tier_config(run_request_limit=1, agent_request_limit=3),
            limit_recovery=_NeverConcludes(),
        )

        @agent.pydantic_agent.tool_plain
        def noop() -> str:
            return "ok"

        with agent.pydantic_agent.override(model=_calling_model("noop")):
            with pytest.raises(RunUsageLimitError) as exc_info:
                agent.run_sync("first")
        assert agent._agent_run_count == 1
        assert "request_limit of 1" in str(exc_info.value)

    def test_repeated_run_tier_failures_exhaust_the_agent_tier(self):
        """Test the two tiers interact: a run-level loop cannot spin forever.

        Which tier fired is asserted by CLASS; the message assertions that follow
        pin the wording, they do not identify the tier.

        Recovery is declined for the same reason as the twin above: each loop iteration must
        consume exactly one unit of the agent tier for the arithmetic to mean anything.
        """
        agent = ReactAgent(
            config=_both_tier_config(run_request_limit=1, agent_request_limit=2),
            limit_recovery=_NeverConcludes(),
        )

        @agent.pydantic_agent.tool_plain
        def noop() -> str:
            return "ok"

        with agent.pydantic_agent.override(model=_calling_model("noop")):
            for _ in range(2):
                with pytest.raises(RunUsageLimitError) as run_tier:
                    agent.run_sync("burn a turn")
                assert "The next request would exceed" in str(run_tier.value)
            with pytest.raises(AgentUsageLimitError) as agent_tier:
                agent.run_sync("one turn too many")
        assert str(agent_tier.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"

    def test_unset_limit_never_blocks(self, minimal_config):
        """Test agent_request_limit=None (the default) blocks nothing but still counts."""
        assert minimal_config.agent_usage_limits.agent_request_limit is None
        agent = ReactAgent(config=minimal_config)
        with agent.pydantic_agent.override(model=_text_model()):
            for _ in range(5):
                agent.run_sync("unbounded")
        assert agent._agent_run_count == 5

    async def test_async_run_enforces_the_same_budget(self):
        """Test the async entry point holds the budget (run_sync only delegates to it)."""
        agent = ReactAgent(config=_agent_limit_config(1))
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("first")
            with pytest.raises(UsageLimitError):
                await agent.run("second")
        assert agent._agent_run_count == 1

    def test_run_count_is_not_persisted_config_state(self):
        """Test the counter is runtime-only: no config field, nothing serialized."""
        assert "_run_count" not in ReactAgentConfig.model_fields
        assert "run_count" not in ReactAgentConfig.model_fields
        dumped = _agent_limit_config(2).model_dump()
        assert "_run_count" not in dumped
        assert "run_count" not in dumped


def _usage_event(run_id: str) -> LlmUsageEvent:
    """One model round-trip's usage record, tagged with the run it belongs to."""
    return LlmUsageEvent(
        run_id=run_id,
        model_name="gpt-4o",
        provider_name="openai",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        requests=1,
    )


class TestReactAgentRunCountRestore:
    """restore_context() recomputes the agent-lifetime run budget from replayed events."""

    def test_three_distinct_runs_seed_three(self):
        """Test one event per run over three runs seeds a count of three."""
        agent = ReactAgent(config=_agent_limit_config(10))
        events = [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2", "r3")]
        agent.restore_context(events)
        assert agent._agent_run_count == 3

    def test_one_run_emitting_three_events_seeds_one(self):
        """Test a run with three tool-call round-trips counts once, not three times.

        The discriminating test. One ``run()`` emits one ``LlmUsageEvent`` per
        ``ModelResponse``, all sharing one ``run_id``. Two plausible-but-wrong
        implementations fail here and only here: counting events seeds 3, and
        aggregating without ``by_run=True`` leaves ``runs`` empty and seeds 0.
        A fixture with one event per run passes under all three.
        """
        agent = ReactAgent(config=_agent_limit_config(10))
        events = [FakeEventMessage(event=_usage_event("same-run")) for _ in range(3)]
        agent.restore_context(events)
        assert agent._agent_run_count == 1

    def test_empty_event_list_seeds_zero(self):
        """Test restoring nothing seeds zero — on a fresh agent and on a spent one.

        The second half is what makes this bite. A fresh agent reads ``0``
        whether or not seeding ran at all, so asserting only that proves
        nothing; driving the counter above zero first turns the assertion into
        a real one. It also pins **assignment** over a high-water mark (AC #6):
        ``max(self._run_count, ...)`` satisfies every other test in this class.
        """
        agent = ReactAgent(config=_agent_limit_config(10))
        agent.restore_context([])
        assert agent._agent_run_count == 0

        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        assert agent._agent_run_count == 2
        agent.restore_context([])
        assert agent._agent_run_count == 0

    def test_events_without_usage_seed_zero(self):
        """Test envelopes carrying non-usage payloads are ignored.

        Seeded above zero first, for the same reason: this distinguishes "the
        ignore path assigns zero" from "the counter was simply never touched".
        """
        agent = ReactAgent(config=_agent_limit_config(10))
        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        events = [
            FakeEventMessage(event=LlmMessageEvent(message=msg)),
            FakeEventMessage(
                event=ToolCallEvent(
                    run_id="r1", tool_name="lookup", tool_call_id="c1", arguments="{}"
                )
            ),
        ]
        agent.restore_context(events)
        assert agent._agent_run_count == 0

    def test_restored_agent_at_its_limit_raises_on_the_next_run(self):
        """Test the seeded value is enforced, not merely stored.

        Reading ``_run_count`` back only proves it was written. This drives the
        seeded value through ``_check_and_consume_agent_budget`` and asserts the
        message shape 15-2 pinned.

        The driver is a real model rather than a stubbed ``iter()``: the only refusal
        lives in ``LifetimeBudgetCapability.wrap_run``, which fires inside ``iter()``, so
        a stub reaches nothing. Assertions unchanged.
        """
        agent = ReactAgent(config=_agent_limit_config(2))
        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        with agent.pydantic_agent.override(model=_text_model()):
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("one turn too many")
        assert str(exc_info.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"

    def test_restored_agent_below_its_limit_spends_only_the_remainder(self):
        """Test the seeded count is the budget's starting point, not a blanket block.

        Complements the at-limit test: without this, an implementation that seeded
        the limit itself (or any value >= it) would look correct.
        """
        agent = ReactAgent(config=_agent_limit_config(2))
        agent.restore_context([FakeEventMessage(event=_usage_event("r1"))])
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("the one run left")
            with pytest.raises(UsageLimitError):
                agent.run_sync("one turn too many")
        assert agent._agent_run_count == 2

    def test_restore_is_idempotent(self):
        """Test seeding assigns rather than accumulates: restoring twice is stable."""
        agent = ReactAgent(config=_agent_limit_config(10))
        events = [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")]
        agent.restore_context(events)
        agent.restore_context(events)
        assert agent._agent_run_count == 2

    def test_never_restored_agent_starts_at_zero(self, minimal_config):
        """Test seeding runs on restore only — construction still yields zero."""
        assert ReactAgent(config=minimal_config)._agent_run_count == 0

    def test_seeding_leaves_the_message_fold_intact(self):
        """Test a mixed stream still restores exactly its LlmMessageEvent messages."""
        agent = ReactAgent(config=_agent_limit_config(10))
        msg1 = ModelRequest(parts=[UserPromptPart(content="first")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="second")])
        agent.restore_context(
            [
                FakeEventMessage(event=LlmMessageEvent(message=msg1)),
                FakeEventMessage(event=_usage_event("r1")),
                FakeEventMessage(event=LlmMessageEvent(message=msg2)),
                FakeEventMessage(event=_usage_event("r2")),
            ]
        )
        assert agent.context.messages == [msg1, msg2]
        assert agent._agent_run_count == 2


@dataclass
class _RunUsageProbe(AbstractCapability[Any]):
    """Records the run's own usage accumulator, as handed to a co-mounted capability.

    Mounted as a caller capability it nests **inside** ``LifetimeBudgetCapability``, so its
    ``wrap_run`` runs after the budget's pre-flight and before the run spends anything —
    the moment at which "this run starts at zero" is a claim worth making.
    """

    seen: list[tuple[RunUsage, int]] = field(default_factory=list)

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Snapshot the accumulator's identity and its total at hand-over, then run."""
        # The total is snapshotted here, not read back at the end: the run spends THROUGH
        # this very object, so it is non-zero by the time the assertions run.
        self.seen.append((ctx.usage, ctx.usage.total_tokens))
        return await handler()


class TestReactAgentTokenBudgetEnforcement:
    """Test the agent-lifetime TOKEN budget: accumulate across runs, check pre-flight."""

    def test_fresh_agent_starts_with_nothing_spent(self, minimal_config):
        """Test a newly constructed agent has burned no tokens."""
        agent = ReactAgent(config=minimal_config)
        assert agent._agent_usage.total_tokens == 0

    def test_unset_token_limits_never_block(self, minimal_config):
        """Test the default (all None) blocks nothing, however much is spent."""
        limits = minimal_config.agent_usage_limits
        assert (limits.input_tokens_limit, limits.output_tokens_limit) == (None, None)
        assert limits.total_tokens_limit is None
        agent = ReactAgent(config=minimal_config)
        with agent.pydantic_agent.override(model=_spending_model(1000, 500)):
            for _ in range(5):
                agent.run_sync("unbounded")
        assert agent._agent_usage.total_tokens == 7500

    def test_total_tokens_limit_blocks_once_the_lifetime_budget_is_spent(self):
        """Test the budget spans runs: no single run breaches it, the agent still stops.

        The discriminating test for accumulation. Every run here spends 60 against a
        limit of 100, so an implementation that compared per-run usage — or reset the
        accumulator each run — never raises and passes a single-run test green.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(40, 20)):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("third")
        assert agent._agent_usage.total_tokens == 120
        # pydantic-ai v2 appends a docs-hint suffix to UsageLimitExceeded; ADR-013 only
        # requires the prefix to keep matching.
        assert str(exc_info.value).startswith(
            "Exceeded the total_tokens_limit of 100 (total_tokens=120)"
        )

    def test_input_tokens_limit_blocks_independently(self):
        """Test input_tokens_limit is live on its own, not only via the total."""
        agent = ReactAgent(config=_agent_token_config(input_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(60)):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("third")
        # pydantic-ai v2 appends a docs-hint suffix to UsageLimitExceeded; ADR-013 only
        # requires the prefix to keep matching.
        assert str(exc_info.value).startswith(
            "Exceeded the input_tokens_limit of 100 (input_tokens=120)"
        )

    def test_output_tokens_limit_blocks_independently(self):
        """Test output_tokens_limit is live on its own — output tokens only here."""
        agent = ReactAgent(config=_agent_token_config(output_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(0, 60)):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("third")
        # pydantic-ai v2 appends a docs-hint suffix to UsageLimitExceeded; ADR-013 only
        # requires the prefix to keep matching.
        assert str(exc_info.value).startswith(
            "Exceeded the output_tokens_limit of 100 (output_tokens=120)"
        )

    def test_a_run_may_overshoot_the_budget(self):
        """Test the contract is "do not START once spent", not "never exceed".

        A run's token cost is unknown until it completes, so the run that crosses the
        line finishes normally and returns its output; only the next one is refused.
        Pinned as behaviour so a reader meeting the overshoot does not file it as a bug.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(900, 100)):
            assert agent.run_sync("one very expensive run") == "ok"
            assert agent._agent_usage.total_tokens == 1000
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused")

    def test_usage_folds_even_when_the_run_fails_partway(self):
        """Test tokens a failed run burned are still counted — the provider billed them.

        The model spends an exact 40/20 asking for a tool, and the tool raises: the run
        dies **after** the request that cost the tokens, which is the shape the fold in
        ``wrap_run``'s ``finally`` exists for. See
        ``test_a_real_failing_run_still_folds_its_usage`` for the same claim asserted
        loosely against pydantic-ai's own estimate.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))

        @agent.pydantic_agent.tool_plain
        def boom_tool() -> str:
            raise RuntimeError("boom")

        model = _calling_model("boom_tool", input_tokens=40, output_tokens=20)
        with agent.pydantic_agent.override(model=model):
            with pytest.raises(RuntimeError):
                agent.run_sync("fails after spending")
        assert agent._agent_usage.total_tokens == 60

    async def test_a_real_failing_run_still_folds_its_usage(self):
        """A REAL run that raises after spending still contributes what it burned.

        The design rests on ``usage=`` being the same object pydantic-ai's graph mutates
        in place for the whole run; a stubbed run proves only that the stub cooperates.
        Here the model is real, the token count is pydantic-ai's own, and the run dies
        inside a tool — after the model request that spent them. Move the fold out of
        ``_run_with_limits``' ``finally`` onto the success path and this goes red.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1_000_000))

        def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name="boom_tool", args={})])

        @agent.pydantic_agent.tool_plain
        def boom_tool() -> str:
            raise RuntimeError("boom")

        assert agent._agent_usage.total_tokens == 0
        with agent.pydantic_agent.override(model=FunctionModel(tool_calling_model)):
            with pytest.raises(RuntimeError, match="boom"):
                await agent.run("spend, then fail")

        assert agent._agent_usage.total_tokens > 0

    def test_token_rejection_consumes_no_run_budget(self):
        """Test the two agent-tier gates are independent: a token refusal costs no run.

        The token check runs first precisely so a refused call does not also burn a
        unit of agent_request_limit, which would make repeated refusals shrink an
        unrelated budget.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            agent_usage_limits=AgentUsageLimits(agent_request_limit=5, total_tokens_limit=100),
        )
        agent = ReactAgent(config=config)
        with agent.pydantic_agent.override(model=_spending_model(150)):
            agent.run_sync("first")
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused on tokens")
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused again")
        assert agent._agent_run_count == 1

    def test_token_rejection_happens_before_compaction(self):
        """Test the token check precedes compaction: a refused run pays no summarizer.

        The token half of the same regression guard, re-expressed the same way and for the
        same reason (see ``test_rejection_happens_before_compaction``). The gate is armed,
        so the strategy would have been called had the run been admitted.
        """
        agent = ReactAgent(config=_armed_compaction_config(total_tokens_limit=100))
        strategy = _RecordingCompaction(CompactionResult("S", 1))
        with agent.pydantic_agent.override(model=_spending_model(150)):
            agent.run_sync("first")
            agent._compactor.strategy = strategy
            agent._context._last_input_tokens = 900
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused")
        assert strategy.calls == 0

    def test_run_tier_never_receives_the_lifetime_accumulator(self):
        """Test every run starts at zero usage — the run tier never sees the accumulator.

        The same claim as before the budget moved, re-expressed against the new shape:
        the object the run spends through is now the one pydantic-ai's graph creates and
        hands to every hook as ``ctx.usage``, and ``LifetimeBudgetCapability`` folds it in
        rather than being it. The mutation this exists to kill — making the capability's
        accumulator the run's own budget object (``usage=self._budget.usage`` on the
        ``run()`` call) — raises nothing and logs nothing: it checks the RUN tier's limits
        against lifetime totals, silently turning a per-run cap into a lifetime one.

        Under the pre-capability shape no other test in this file went red for it. It now
        also inflates every lifetime token total, so several token tests fall over too —
        but on arithmetic, not on the claim. This is the only test that names it: it asserts
        the accumulator's IDENTITY, and that each run is handed a total of zero.
        """
        probe = _RunUsageProbe()
        # Mounted as a CALLER capability, so it sits inside the budget's wrap_run and
        # sees exactly the object the run itself will spend through.
        agent = ReactAgent(config=_agent_token_config(), capabilities=[probe])

        with agent.pydantic_agent.override(model=_spending_model(100, 50)):
            agent.run_sync("first")
            agent.run_sync("second")

        # Non-zero by the second run, so "fresh" is a real claim there, not a tautology.
        assert agent._agent_usage.total_tokens == 300
        assert len(probe.seen) == 2
        for usage, total_at_handover in probe.seen:
            assert usage is not agent._agent_usage
            assert total_at_handover == 0

    async def test_async_run_enforces_the_same_token_budget(self):
        """Test the async entry point holds the budget (run_sync only delegates to it)."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(150)):
            await agent.run("first")
            with pytest.raises(UsageLimitError):
                await agent.run("second")

    def test_accumulator_is_not_persisted_config_state(self):
        """Test the accumulator is runtime-only: no config field, nothing serialized."""
        assert "_agent_usage" not in ReactAgentConfig.model_fields
        assert "agent_usage" not in ReactAgentConfig.model_fields
        assert "_agent_usage" not in _agent_token_config(total_tokens_limit=100).model_dump()


class TestReactAgentTokenBudgetRestore:
    """restore_context() reseeds the token accumulator from the same replayed events."""

    def test_restore_seeds_tokens_from_replayed_events(self):
        """Test three replayed runs seed the summed token totals, not zero."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        agent.restore_context(
            [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2", "r3")]
        )
        # _usage_event spends 10 input / 5 output per event.
        assert agent._agent_usage.input_tokens == 30
        assert agent._agent_usage.output_tokens == 15

    def test_every_event_of_a_run_contributes_its_tokens(self):
        """Test tokens sum over EVENTS within a run, unlike the run counter.

        The counterpart to the run counter's "three events, one run". Both seeds come
        from the same aggregation and they are deliberately different reductions of it:
        a run with three round-trips counts once but spent three times.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        agent.restore_context([FakeEventMessage(event=_usage_event("same-run")) for _ in range(3)])
        assert agent._agent_run_count == 1
        assert agent._agent_usage.total_tokens == 45

    def test_restored_agent_over_budget_raises_on_the_next_run(self):
        """Test the seeded tokens are enforced, not merely stored.

        Also the test that catches a dropped ``by_run=True``: that mutation seeds an
        empty summary, so the agent looks fresh and this run succeeds.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=40))
        agent.restore_context(
            [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2", "r3")]
        )
        with agent.pydantic_agent.override(model=_spending_model(1)):
            with pytest.raises(UsageLimitError) as exc_info:
                agent.run_sync("one run too many")
        # pydantic-ai v2 appends a docs-hint suffix to UsageLimitExceeded; ADR-013 only
        # requires the prefix to keep matching.
        assert str(exc_info.value).startswith(
            "Exceeded the total_tokens_limit of 40 (total_tokens=45)"
        )

    def test_restored_agent_below_its_limit_spends_only_the_remainder(self):
        """Test the seeded total is the budget's starting point, not a blanket block."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        agent.restore_context(
            [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2", "r3")]
        )
        with agent.pydantic_agent.override(model=_spending_model(60)):
            agent.run_sync("the one run left")  # 45 + 60 = 105, over only afterwards
            with pytest.raises(UsageLimitError):
                agent.run_sync("one run too many")

    def test_restore_is_idempotent_for_tokens(self):
        """Test seeding assigns rather than accumulates: restoring twice is stable."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        events = [FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")]
        agent.restore_context(events)
        agent.restore_context(events)
        assert agent._agent_usage.total_tokens == 30

    def test_empty_event_list_seeds_zero_tokens(self):
        """Test restoring nothing zeroes the accumulator — on a spent agent, not a fresh one.

        Seeded above zero first for the reason 15-3's review established: asserting
        ``0`` on something already ``0`` passes under an implementation that never
        touches the accumulator, and under a high-water-mark one.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        assert agent._agent_usage.total_tokens == 30
        agent.restore_context([])
        assert agent._agent_usage.total_tokens == 0

    def test_events_without_usage_seed_zero_tokens(self):
        """Test envelopes carrying non-usage payloads are ignored."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        agent.restore_context(
            [
                FakeEventMessage(event=LlmMessageEvent(message=msg)),
                FakeEventMessage(
                    event=ToolCallEvent(
                        run_id="r1", tool_name="lookup", tool_call_id="c1", arguments="{}"
                    )
                ),
            ]
        )
        assert agent._agent_usage.total_tokens == 0

    def test_live_spend_continues_from_the_restored_total(self):
        """Test a restored agent folds new runs on top of the seed, not over it."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))
        agent.restore_context([FakeEventMessage(event=_usage_event("r1"))])
        with agent.pydantic_agent.override(model=_spending_model(100, 50)):
            agent.run_sync("one more")
        assert agent._agent_usage.input_tokens == 110
        assert agent._agent_usage.output_tokens == 55


class TestUsageLimitErrorTierSplit:
    """Test the two tiers raise distinct classes, told apart by isinstance only.

    Every assertion here is on the exception's CLASS. The tier is never derived
    from the message text — asserting the text of a message is a separate concern
    and lives in the message-identity test below.
    """

    def test_run_tier_breach_raises_the_run_subclass(self, minimal_config):
        """Test pydantic-ai's mid-run breach surfaces as RunUsageLimitError."""
        agent = ReactAgent(config=minimal_config)
        run = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))
        with patch.object(agent._pydantic_agent, "iter", return_value=run):
            with pytest.raises(RunUsageLimitError):
                agent.run_sync("test query")

    def test_agent_tier_token_breach_raises_the_agent_subclass(self):
        """Test a pre-flight token breach surfaces as AgentUsageLimitError."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(80, 40)):
            agent.run_sync("first")
            with pytest.raises(AgentUsageLimitError):
                agent.run_sync("second")

    def test_agent_tier_run_breach_raises_the_agent_subclass(self):
        """Test the N+1 run surfaces as AgentUsageLimitError."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(AgentUsageLimitError):
                agent.run_sync("third")

    def test_base_class_still_catches_both_tiers(self, minimal_config):
        """Test the additive claim: one `except UsageLimitError` catches both tiers.

        This is the guard behind "no deprecation shim is required". If either
        subclass ever stops descending from the base, every existing handler in
        akgentic-agent and downstream breaks silently — this test goes red first.
        """
        assert issubclass(RunUsageLimitError, UsageLimitError)
        assert issubclass(AgentUsageLimitError, UsageLimitError)

        run_tier_agent = ReactAgent(config=minimal_config)
        breach = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))
        with patch.object(run_tier_agent._pydantic_agent, "iter", return_value=breach):
            try:
                run_tier_agent.run_sync("burn the turn")
            except UsageLimitError as err:
                assert isinstance(err, RunUsageLimitError)
            else:
                pytest.fail("run tier did not raise")

        agent_tier_agent = ReactAgent(config=_agent_limit_config(1))
        with agent_tier_agent.pydantic_agent.override(model=_text_model()):
            agent_tier_agent.run_sync("first")
            try:
                agent_tier_agent.run_sync("second")
            except UsageLimitError as err:
                assert isinstance(err, AgentUsageLimitError)
            else:
                pytest.fail("agent tier did not raise")

    def test_the_two_tiers_are_not_each_other(self):
        """Test the split is a real discrimination, not two aliases of one class."""
        assert not issubclass(RunUsageLimitError, AgentUsageLimitError)
        assert not issubclass(AgentUsageLimitError, RunUsageLimitError)

        run_tier = RunUsageLimitError("turn exhausted")
        agent_tier = AgentUsageLimitError("lifetime exhausted")
        assert not isinstance(run_tier, AgentUsageLimitError)
        assert not isinstance(agent_tier, RunUsageLimitError)

    def test_the_base_class_is_unchanged(self):
        """Test UsageLimitError stays a plain Exception subclass, not abstract."""
        assert issubclass(UsageLimitError, Exception)
        assert UsageLimitError.__bases__ == (Exception,)
        # Still directly instantiable: nothing downstream that constructs the base
        # (tests, fakes, re-raises) is broken by the split.
        assert str(UsageLimitError("still constructible")) == "still constructible"

    def test_message_text_is_unchanged_by_the_split(self, minimal_config):
        """Test the split moved the class, never the wording, at both tiers."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("first")
            agent.run_sync("second")
            with pytest.raises(AgentUsageLimitError) as agent_tier:
                agent.run_sync("third")
        assert str(agent_tier.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"

        run_agent = ReactAgent(config=minimal_config)
        breach = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))
        with patch.object(run_agent._pydantic_agent, "iter", return_value=breach):
            with pytest.raises(RunUsageLimitError) as run_tier:
                run_agent.run_sync("burn the turn")
        # pydantic-ai's own wording, translated verbatim by str(e).
        assert "Request limit exceeded" in str(run_tier.value)


def weather_lookup(city: str) -> str:
    """Look up the weather for a city.

    Args:
        city: The city to look up.

    Returns:
        A canned forecast string.
    """
    return f"sunny in {city}"


def _tool_capturing_model(seen: dict[str, list[str]]) -> FunctionModel:
    """A model stub recording the tool names the agent OFFERED it on each request.

    ``AgentInfo.function_tools`` is what the model may call, so it is the outcome
    the no-tools claim is about — as opposed to whether ``override`` was called,
    which is the implementation detail the epic explicitly refuses to assert on.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(stub)


class TestReactAgentConcludeWithoutTools:
    """conclude_without_tools(): one follow-up run, no tools, its own budget."""

    async def test_conclusion_offers_the_model_no_tools(self, minimal_config):
        """The conclusion run reaches the model with an empty tool list (AC #9).

        Asserted on what the model was **offered**, not on the ``override`` call.
        A per-run ``toolsets=[]`` — the plausible wrong implementation, since
        ``iter()`` accepts one — leaves every registered tool in place and fails
        here; only ``override(tools=[], toolsets=[])`` replaces them.
        """
        agent = ReactAgent(config=minimal_config, tools=[weather_lookup])
        seen: dict[str, list[str]] = {}

        with agent.pydantic_agent.override(model=_tool_capturing_model(seen)):
            await agent.conclude_without_tools("budget spent, answer now")

        assert seen["tools"] == []

    async def test_an_ordinary_run_still_offers_the_tool(self, minimal_config):
        """Control case: the same agent, the same stub, ``run()`` (AC #9).

        Without this the assertion above passes even if ``weather_lookup`` was
        never registered — i.e. even if the conclusion removed nothing at all.
        """
        agent = ReactAgent(config=minimal_config, tools=[weather_lookup])
        seen: dict[str, list[str]] = {}

        with agent.pydantic_agent.override(model=_tool_capturing_model(seen)):
            await agent.run("ordinary turn")

        assert seen["tools"] == ["weather_lookup"]

    async def test_conclusion_carries_its_own_single_request_limit(self):
        """The conclusion is bounded by request_limit=1, never the config tier (AC #10).

        The configured run tier is 7, so this distinguishes "uses its own budget"
        from "uses whichever budget is set" — including the budget that was just
        exhausted, which is the one it must not inherit.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(run_request_limit=7),
        )
        agent = ReactAgent(config=config)
        captured: dict = {}

        with patch.object(agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured)):
            await agent.conclude_without_tools("wrap it up")

        assert captured["usage_limits"].request_limit == 1

    async def test_reason_reaches_the_model_as_the_user_prompt(self, minimal_config):
        """``reason`` is the run's prompt, not a log line (AC #11)."""
        agent = ReactAgent(config=minimal_config)
        captured: dict = {}

        with patch.object(agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured)):
            await agent.conclude_without_tools("your tool budget is spent; answer now")

        assert captured["user_prompt"] == "your tool budget is spent; answer now"

    async def test_conclusion_returns_the_runs_output(self, minimal_config):
        """The conclusion returns the run output the way ``run()`` does (AC #8)."""
        agent = ReactAgent(config=minimal_config)

        with patch.object(agent._pydantic_agent, "iter", return_value=_StubRun(output="concluded")):
            result = await agent.conclude_without_tools("wrap it up")

        assert result == "concluded"

    def test_spent_run_budget_refuses_the_conclusion_before_any_model_call(self):
        """An agent at its lifetime run limit cannot conclude (AC #12).

        Terminal by design: the budget that would pay for the conclusion is exactly
        the one that is spent.

        "Before any model call" is asserted as a request count of zero rather than as
        ``iter`` never being called. The old assertion is **false by construction** now:
        the only refusal lives in ``LifetimeBudgetCapability.wrap_run``, which fires
        *inside* ``iter()``, so the run must start for the refusal to happen at all. What
        that assertion existed to pin — the conclusion costs nothing — is exactly the
        request count, and a stubbed ``iter()`` could never have observed it anyway.
        """
        agent = ReactAgent(config=_agent_limit_config(1))
        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("the only run this agent gets")

        requests: list[str] = []
        with agent.pydantic_agent.override(model=_counting_model(requests)):
            with pytest.raises(AgentUsageLimitError):
                agent.conclude_without_tools_sync("wrap it up")

        assert requests == []

    def test_spent_token_budget_refuses_the_conclusion(self):
        """The token half of the agent tier refuses the conclusion too (AC #12).

        Same re-expression, same reason (see the run-budget twin above): the exception
        class, the terminality claim and the "before any spend" claim all survive; only
        the way "nothing was paid for" is observed has moved from the stub to the model.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with agent.pydantic_agent.override(model=_spending_model(80, 40)):
            agent.run_sync("burn the lifetime token budget")

        requests: list[str] = []
        with agent.pydantic_agent.override(model=_counting_model(requests)):
            with pytest.raises(AgentUsageLimitError):
                agent.conclude_without_tools_sync("wrap it up")

        assert requests == []

    async def test_conclusion_runs_on_top_of_the_healed_context(self):
        """The healing ToolReturnPart is in the history the conclusion is given (AC #13).

        This is the whole point of healing before concluding: the tool result the
        model reads as the reason it must answer now is already in the context the
        follow-up run is handed.

        The breaching run must be real — healing is ``HealingCapability.on_run_error``
        and no hook fires under a stubbed ``iter()``. Only the conclusion stays stubbed,
        to capture the history it was handed. The run-tier breach also puts the trailing
        dangling ``ModelResponse`` there for real, which the second half asserts.

        Recovery is declined so the breaching run leaves the healed context as its LAST
        word: the default policy would drive its own conclusion here, whose messages would
        then sit between the healing request and the one this test captures.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        )
        agent = ReactAgent(
            config=config, tools=[weather_lookup], limit_recovery=_NeverConcludes()
        )

        def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="weather_lookup", args={"city": "Paris"})]
            )

        with agent.pydantic_agent.override(model=FunctionModel(tool_calling_model)):
            with pytest.raises(RunUsageLimitError):
                await agent.run("do the thing")

        captured: dict = {}
        with patch.object(agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured)):
            await agent.conclude_without_tools("wrap it up")

        history = captured["message_history"]
        healing = history[-1]
        assert isinstance(healing, ModelRequest)
        assert [str(p.content) for p in healing.parts if isinstance(p, ToolReturnPart)] == [
            RUN_LIMIT_HEALING_MESSAGE
        ]
        # The healing request closes out the response the breach left dangling.
        dangling = history[-2]
        assert isinstance(dangling, ModelResponse)
        assert dangling.tool_calls

    def test_sync_bridge_returns_the_same_output(self, minimal_config):
        """conclude_without_tools_sync() returns what the async form returns (AC #14)."""
        agent = ReactAgent(config=minimal_config)

        with patch.object(agent._pydantic_agent, "iter", return_value=_StubRun(output="concluded")):
            assert agent.conclude_without_tools_sync("wrap it up") == "concluded"

    def test_sync_bridge_raises_after_close(self, minimal_config):
        """The closed-agent guard is the same one run_sync/compact carry (AC #14)."""
        agent = ReactAgent(config=minimal_config)
        agent.close()
        with pytest.raises(RuntimeError, match="ReactAgent is closed"):
            agent.conclude_without_tools_sync("wrap it up")

    def test_sync_bridge_runs_on_the_agents_own_loop(self, minimal_config):
        """One loop strategy, not two (AC #14).

        ``asyncio.run()`` or a fresh loop per call would detach the pooled httpx
        connections from the loop that owns them, so ``aclose()`` raises on stop
        and the pool leaks. Two calls, so a per-call loop shows up as two distinct
        loops rather than one.
        """
        used_loops: list[asyncio.AbstractEventLoop] = []

        async def stub_conclude(*_: Any, **__: Any) -> str:
            used_loops.append(asyncio.get_running_loop())
            return "ran-on-owned-loop"

        agent = ReactAgent(config=minimal_config)
        try:
            with patch.object(ReactAgent, "conclude_without_tools", new=stub_conclude):
                first = agent.conclude_without_tools_sync("once")
                second = agent.conclude_without_tools_sync("twice")
            assert first == second == "ran-on-owned-loop"
            assert used_loops == [agent._loop, agent._loop]
            assert not agent._loop.is_closed()
        finally:
            agent.close()

    # ``test_run_still_raises_and_never_concludes_on_its_own`` stood here until story 26-2.
    # It asserted the contract this story reverses, and would have STAYED GREEN against the
    # new behaviour: a stubbed ``iter()`` fires no capability hook, so no decision is ever
    # recorded and the breach raises on the today's-behaviour branch whatever the policy is.
    # Its two claims live on in ``TestReactAgentLimitRecovery`` — the default now concludes
    # (asserted against a REAL breach), a seam returning ``None`` still raises — and its
    # ``iter_call.call_count == 1`` intent survives as "exactly one conclusion".

    def test_run_signature_gained_no_limits_parameter(self):
        """The limits override is internal; run()'s public signature is untouched (AC #15).

        The extraction that makes the conclusion possible must not leak a knob onto
        ``run()`` — a caller passing its own run-tier budget per call is a different
        feature, and not this one.
        """
        assert list(inspect.signature(ReactAgent.run).parameters) == [
            "self",
            "user_prompt",
            "deps",
            "output_type",
        ]


def _recovery_config(**limits: int) -> ReactAgentConfig:
    """Config whose run tier is tight enough for a REAL breach to be cheap to provoke."""
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        run_usage_limits=RunUsageLimits(**limits),
    )


def _breaching_then_concluding_model(
    offered: list[list[str]], text: str = "concluded"
) -> FunctionModel:
    """Asks for two tools while any are offered; answers plainly once they are gone.

    The test model HAS to behave differently once the tools are gone, or the recovery tests
    cannot pass at all: a model that always answers with a ``ToolCallPart`` breaches the run
    (good) and then breaches the *conclusion* too, because ``override(tools=[], toolsets=[])``
    leaves it calling a tool that no longer exists. ``AgentInfo.function_tools`` is empty
    exactly when the toolset was overridden away, which is the condition to branch on —
    ``_tool_then_text_model`` answers on the second *request* instead, which is not the same.

    Two calls in one response, against ``tool_calls_limit=1``: pydantic-ai checks the
    projected count for the whole response before running any of them, so the breach lands on
    the FIRST request and the run costs exactly one model call. That is what makes "two
    ``LlmUsageEvent``s for a rescued turn" a claim about the shape rather than about how
    chatty the stub is.

    ``offered`` records the tool names each request was given, so the caller can assert both
    what the conclusion was offered (nothing) and how many conclusions happened (one).
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = [t.name for t in info.function_tools]
        offered.append(names)
        if names:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="weather_lookup", args={"city": "Paris"}),
                    ToolCallPart(tool_name="weather_lookup", args={"city": "Lyon"}),
                ]
            )
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(stub)


def _always_calling_model() -> FunctionModel:
    """Asks for the same tool on every request, tools or no tools.

    The conclusion therefore breaches too: with the toolset overridden away the call names a
    tool that does not exist, pydantic-ai answers with a retry prompt, and the follow-up
    request exceeds the conclusion's own ``run_request_limit=1``. That is the recursion
    guard's scenario.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="weather_lookup", args={"city": "Paris"})]
        )

    return FunctionModel(stub)


def _is_healing_message(message: ModelMessage) -> bool:
    """Whether ``message`` is the ``ModelRequest`` ``HealingCapability`` writes on a breach."""
    return isinstance(message, ModelRequest) and any(
        isinstance(p, ToolReturnPart) and str(p.content) == RUN_LIMIT_HEALING_MESSAGE
        for p in message.parts
    )


@dataclass
class _RecordingSeam(LimitRecoveryCapability):
    """Records each consultation and the context as it stood at that moment."""

    consulted: list[UsageLimitExceeded] = field(default_factory=list)
    context_seen: list[list[ModelMessage]] = field(default_factory=list)
    context: ContextManager | None = None

    async def handle_limit_exceeded(
        self, ctx: RunContext[Any], *, error: UsageLimitExceeded
    ) -> ConclusionDecision | None:
        """Snapshot the durable context, then decide exactly as the base class would."""
        self.consulted.append(error)
        if self.context is not None:
            self.context_seen.append(list(self.context.messages))
        return await super().handle_limit_exceeded(ctx, error=error)


class TestReactAgentLimitRecovery:
    """run() degrades a run-tier breach into a tool-free conclusion, behind the seam."""

    async def test_a_run_tier_breach_returns_the_concluded_answer(self):
        """Default policy: the turn answers instead of raising (AC #7).

        Driven by a REAL breach, never a stubbed ``iter()``: the decision is written by
        ``LimitRecoveryCapability.on_run_error``, which fires *inside* ``iter()``, so a stub
        reaches no hook and would prove nothing here — it is exactly how the test this
        replaces stayed green while asserting the opposite behaviour.

        Three claims, one setup: the conclusion's output is what ``run()`` returns, the
        conclusion was offered no tools, and it happened exactly once.
        """
        offered: list[list[str]] = []
        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            result = await agent.run("do the thing")

        assert result == "concluded"
        assert offered == [["weather_lookup"], []]

    async def test_a_seam_returning_none_reproduces_todays_behaviour(self):
        """The opt-out is byte-for-byte the pre-recovery contract (AC #8).

        The compatibility guarantee for every deployment that has not opted in, and for
        ``akgentic-agent`` during the window before its own recovery epic lands: the breach
        surfaces as ``RunUsageLimitError`` carrying pydantic-ai's own wording, the original
        exception is its ``__cause__``, no conclusion is attempted, and healing still ran.
        """
        offered: list[list[str]] = []
        agent = ReactAgent(
            config=_recovery_config(tool_calls_limit=1),
            tools=[weather_lookup],
            limit_recovery=_NeverConcludes(),
        )

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            with patch.object(agent, "conclude_without_tools") as conclude:
                with pytest.raises(RunUsageLimitError) as exc_info:
                    await agent.run("do the thing")

        conclude.assert_not_called()
        assert offered == [["weather_lookup"]]
        cause = exc_info.value.__cause__
        assert isinstance(cause, UsageLimitExceeded)
        assert str(exc_info.value) == str(cause)
        assert "tool_calls_limit" in str(exc_info.value)
        assert _is_healing_message(agent.context.messages[-1])

    async def test_a_stale_decision_never_drives_a_later_turn(self, minimal_config):
        """A decision that outlived its own turn is discarded at the head of the next (AC #5).

        The ``except`` clause normally consumes the decision, but it only fires for a
        ``UsageLimitExceeded``: a co-mounted capability can transform the breach into another
        class on its way out, and the decision would then survive into a later run and
        conclude a turn nobody decided about.

        The stub is the point here rather than a shortcut — the claim is about a breach that
        reaches NO hook, so nothing overwrites what the previous turn left behind. The
        decision is planted through the hook itself, not by writing the field.
        """
        agent = ReactAgent(config=minimal_config)
        with pytest.raises(UsageLimitExceeded):
            await agent._limit_recovery.on_run_error(
                _bare_run_context(), error=UsageLimitExceeded("an earlier turn")
            )

        breach = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))
        with patch.object(agent._pydantic_agent, "iter", return_value=breach):
            with patch.object(agent, "conclude_without_tools") as conclude:
                with pytest.raises(RunUsageLimitError):
                    await agent.run("burn the turn")

        conclude.assert_not_called()
        assert agent._limit_recovery.consume_decision() is None

    async def test_healing_has_already_run_when_the_seam_is_consulted(self):
        """Healing fires FIRST, and neither hook is skipped (AC #6b).

        The whole reason ``LimitRecoveryCapability`` is mounted *before*
        ``HealingCapability``: pydantic-ai walks ``on_run_error`` over
        ``reversed(self.capabilities)``, so the later entry runs first. Swap the two back and
        the seam sees a context whose last message is still the dangling ``ModelResponse``,
        so a policy that reads the context to decide decides on the wrong one.

        What the order does NOT protect is the conclusion's own starting context. The walk
        runs every hook and re-raises only afterwards, so healing has always written its
        ``ToolReturnPart`` by the time ``_run_with_limits`` drives the conclusion — swapping
        the two leaves ``test_a_run_tier_breach_returns_the_concluded_answer`` and
        ``test_a_rescued_turn_emits_two_runs_worth_of_events`` green, and this test is what
        sees it. Keeping a dangling tool call out of the conclusion is the job of using
        ``on_run_error`` instead of ``wrap_run``, not of this ordering.
        """
        offered: list[list[str]] = []
        seam = _RecordingSeam()
        agent = ReactAgent(
            config=_recovery_config(tool_calls_limit=1),
            tools=[weather_lookup],
            limit_recovery=seam,
        )
        seam.context = agent.context

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            result = await agent.run("do the thing")

        assert result == "concluded"
        assert len(seam.consulted) == 1, "the recovery hook ran exactly once"
        assert _is_healing_message(seam.context_seen[0][-1]), "healing had not run yet"

    async def test_the_conclusion_keeps_the_runs_output_type_and_deps(self):
        """Both are threaded verbatim from the breached call (AC #9).

        Pinned with a recorder rather than read out of a real second run: the real run is
        what the recovery test above already proves, and a recorder is the only way to see
        exactly what ``conclude_without_tools`` was handed.
        """

        class Verdict(BaseModel):
            answer: str

        recorded: dict[str, Any] = {}

        async def recorder(reason: str, *, deps: Any = None, output_type: Any = None) -> Verdict:
            recorded.update(reason=reason, deps=deps, output_type=output_type)
            return Verdict(answer="concluded")

        offered: list[list[str]] = []
        agent = ReactAgent(
            config=_recovery_config(tool_calls_limit=1),
            tools=[weather_lookup],
            deps_type=str,
        )

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            with patch.object(agent, "conclude_without_tools", new=recorder):
                result = await agent.run("do the thing", "tenant-7", Verdict)

        assert result == Verdict(answer="concluded")
        assert recorded["deps"] == "tenant-7"
        assert recorded["output_type"] is Verdict
        assert recorded["reason"] == DEFAULT_CONCLUSION_REASON

    async def test_a_structured_output_type_survives_a_real_recovery(self, monkeypatch):
        """The recovered value is an instance of the caller's own type (AC #9).

        The conclusion is NOT forced to plain text: its structured output is what routes
        through the caller's normal path downstream. google-gla is a non-native provider, so
        ``get_output_type`` leaves the type raw and pydantic-ai uses an output tool — which
        the conclusion still has, ``override(tools=[], toolsets=[])`` replacing only the
        function tools.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        class Verdict(BaseModel):
            answer: str

        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
            run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        )
        agent = ReactAgent(config=config, tools=[weather_lookup])

        def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name="weather_lookup", args={"city": "Paris"}),
                        ToolCallPart(tool_name="weather_lookup", args={"city": "Lyon"}),
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name, args={"answer": "concluded"}
                    )
                ]
            )

        with agent.pydantic_agent.override(model=FunctionModel(stub)):
            result = await agent.run("do the thing", output_type=Verdict)

        assert result == Verdict(answer="concluded")

    async def test_a_breach_inside_the_conclusion_does_not_conclude_again(self):
        """The recursion guard, asserted rather than commented (AC #10).

        The conclusion runs through the same capability stack, so the recovery capability is
        mounted on it too and records a decision for its own breach. What stops a second
        conclusion is the explicit recovery parameter ``conclude_without_tools`` leaves off —
        drop it and this run concludes until the interpreter runs out of stack.
        """
        reasons: list[str] = []
        unguarded = ReactAgent.conclude_without_tools

        async def counting(
            self: ReactAgent, reason: str, *, deps: Any = None, output_type: Any = None
        ) -> Any:
            reasons.append(reason)
            return await unguarded(self, reason, deps=deps, output_type=output_type)

        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])

        with agent.pydantic_agent.override(model=_always_calling_model()):
            with patch.object(ReactAgent, "conclude_without_tools", new=counting):
                with pytest.raises(RunUsageLimitError) as exc_info:
                    await agent.run("do the thing")

        assert reasons == [DEFAULT_CONCLUSION_REASON]
        assert "tool_calls_limit" in str(exc_info.value), "the ORIGINAL breach surfaced"

    async def test_a_direct_conclusion_never_recovers_itself(self):
        """Called directly, a conclusion that breaches raises — it does not conclude (AC #10)."""
        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])
        reasons: list[str] = []
        unguarded = ReactAgent.conclude_without_tools

        async def counting(
            self: ReactAgent, reason: str, *, deps: Any = None, output_type: Any = None
        ) -> Any:
            reasons.append(reason)
            return await unguarded(self, reason, deps=deps, output_type=output_type)

        with agent.pydantic_agent.override(model=_always_calling_model()):
            with patch.object(ReactAgent, "conclude_without_tools", new=counting):
                with pytest.raises(RunUsageLimitError):
                    await agent.conclude_without_tools("wrap it up")

        assert reasons == ["wrap it up"]

    @pytest.mark.parametrize(
        "secondary",
        [RuntimeError("the conclusion blew up"), AgentUsageLimitError("lifetime budget spent")],
        ids=["generic", "agent-tier"],
    )
    async def test_a_raising_conclusion_surfaces_the_original_breach(self, secondary, caplog):
        """Whatever the conclusion raises, the caller sees the breach (AC #11).

        Escalation parity with what ``akgentic-agent`` does today. The agent-tier case is the
        one worth stating: an ``AgentUsageLimitError`` from the conclusion's own pre-flight is
        correct, not a bug — but it is not what this caller asked about, and the terminal
        signal is not lost either way, since the NEXT ``run()`` is refused pre-flight with it
        and a direct ``conclude_without_tools()`` still raises it unchanged.
        """
        offered: list[list[str]] = []
        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])

        async def failing(*_: Any, **__: Any) -> Any:
            raise secondary

        with caplog.at_level(logging.ERROR, logger="akgentic.llm.agent"):
            with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
                with patch.object(agent, "conclude_without_tools", new=failing):
                    with pytest.raises(RunUsageLimitError) as exc_info:
                        await agent.run("do the thing")

        assert "tool_calls_limit" in str(exc_info.value)
        assert str(secondary) not in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, UsageLimitExceeded)
        assert any(record.exc_info for record in caplog.records), "the secondary was logged"

    @pytest.mark.parametrize("output", [None, "   "], ids=["none", "whitespace"])
    async def test_an_unusable_conclusion_output_surfaces_the_original_breach(
        self, output, caplog
    ):
        """Nothing usable is the same as nothing at all (AC #11).

        Narrow by design: ``None``, or a ``str`` that is empty or whitespace-only. Richer
        emptiness — a structured output carrying no requests — is the caller's judgement and
        stays out of this package.

        The log line is asserted, not incidental: nothing raised on this path, so the warning
        is the ONLY trace that a turn was rescued and the rescue came back empty. Its sibling
        above has an exception to carry that signal; this one has nothing else.
        """
        offered: list[list[str]] = []
        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])

        async def useless(*_: Any, **__: Any) -> Any:
            return output

        with caplog.at_level(logging.WARNING, logger="akgentic.llm.agent"):
            with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
                with patch.object(agent, "conclude_without_tools", new=useless):
                    with pytest.raises(RunUsageLimitError) as exc_info:
                        await agent.run("do the thing")

        assert "tool_calls_limit" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, UsageLimitExceeded)
        assert any("no usable output" in r.message for r in caplog.records), (
            "an empty rescue must leave a trace; nothing raised to carry one"
        )

    async def test_an_agent_tier_breach_never_consults_the_seam(self):
        """The terminal tier stays terminal: no seam, no conclusion (AC #12).

        ``AgentUsageLimitError`` is this package's own class, not a ``UsageLimitExceeded``,
        so the hook's ``isinstance`` check excludes it by construction rather than by a
        special case.
        """
        seam = _RecordingSeam()
        agent = ReactAgent(config=_agent_limit_config(1), limit_recovery=seam)

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("the only run this agent gets")

            with patch.object(agent, "conclude_without_tools") as conclude:
                with pytest.raises(AgentUsageLimitError):
                    await agent.run("one run too many")

        assert seam.consulted == []
        assert seam.consume_decision() is None
        conclude.assert_not_called()

    async def test_a_rescued_turn_consumes_two_units_of_the_lifetime_budget(self):
        """A rescued turn costs TWO runs of the agent-lifetime budget, not one.

        Inherent to the sibling-run design rather than a regression — the conclusion is a
        second run through the same stack and pays the same agent-tier pre-flight. But it is
        exactly what ``test_counter_advances_when_the_run_tier_limit_fires`` and its loop
        twin stopped covering when they took the opt-out seam to keep their own subject a
        single run, so without this the arithmetic a caller sizing ``agent_request_limit``
        depends on would be stated only in prose.
        """
        offered: list[list[str]] = []
        agent = ReactAgent(config=_recovery_config(tool_calls_limit=1), tools=[weather_lookup])

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            assert await agent.run("do the thing") == "concluded"

        assert agent._agent_run_count == 2

    async def test_a_rescued_turn_emits_two_runs_worth_of_events(self):
        """The event stream is unchanged in shape by recovery (AC #15).

        A rescued turn is the outer run's events (``run_id`` A) → the healing
        ``ToolReturnPart`` → the conclusion's events (``run_id`` B), which is byte-for-byte
        what a rescued turn already emitted when the conclusion was driven from another
        package. No event dataclass and no ``EventSourcingCapability`` change was needed, and
        none should be made: the conclusion is simply a second run.
        """
        offered: list[list[str]] = []
        observer = MockObserver()
        agent = ReactAgent(
            config=_recovery_config(tool_calls_limit=1),
            tools=[weather_lookup],
            observer=observer,
        )

        with agent.pydantic_agent.override(model=_breaching_then_concluding_model(offered)):
            assert await agent.run("do the thing") == "concluded"

        usage = [(i, e) for i, e in enumerate(observer.events) if isinstance(e, LlmUsageEvent)]
        assert len(usage) == 2
        assert usage[0][1].run_id != usage[1][1].run_id
        healing_at = next(
            i
            for i, e in enumerate(observer.events)
            if isinstance(e, LlmMessageEvent) and _is_healing_message(e.message)
        )
        assert usage[0][0] < healing_at < usage[1][0]


class TestReactAgentMultimodalPrompt:
    """Test ReactAgent multimodal UserPrompt support."""

    def test_str_prompt_passes_through(self, minimal_config):
        """Test str user_prompt passes through to pydantic-ai unchanged."""
        agent = ReactAgent(config=minimal_config)
        captured_kwargs: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured_kwargs)
        ):
            agent.run_sync("plain text")

        assert captured_kwargs["user_prompt"] == "plain text"

    def test_list_prompt_passes_through_unchanged(self, minimal_config):
        """Test list[str | BinaryContent] passes to pydantic-ai unchanged."""
        agent = ReactAgent(config=minimal_config)
        captured_kwargs: dict = {}
        multimodal = ["describe: ", BinaryContent(data=b"imgbytes", media_type="image/png")]

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured_kwargs)
        ):
            agent.run_sync(multimodal)

        assert captured_kwargs["user_prompt"] is multimodal  # exact same object, no copy

    def test_user_prompt_importable(self):
        """Test UserPrompt type alias importable from akgentic.llm."""
        from akgentic.llm import UserPrompt as ImportedUserPrompt

        assert ImportedUserPrompt is not None

    def test_user_prompt_alias_in_module_scope(self):
        """Test UserPrompt imported at top of test file is not None."""
        assert UserPrompt is not None

    def test_user_prompt_is_union_type(self):
        """Test UserPrompt type alias resolves to a union containing str and list."""
        import types

        # UserPrompt = str | list[str | BinaryContent] is a UnionType in Python 3.10+
        assert isinstance(UserPrompt, types.UnionType)
        # Both str and list must be args of the union
        union_args = UserPrompt.__args__
        assert str in union_args
        # list type should be present (as a generic alias)
        list_args = [a for a in union_args if hasattr(a, "__origin__") and a.__origin__ is list]
        assert len(list_args) == 1

    def test_no_conversion_in_run(self, minimal_config):
        """Test no BinaryContent construction or list conversion inside run()."""
        agent = ReactAgent(config=minimal_config)
        captured_kwargs: dict = {}
        # Use a list prompt to verify it passes through as-is (same identity)
        bc = BinaryContent(data=b"x", media_type="image/png")
        multimodal: list[str | BinaryContent] = ["text", bc]

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured_kwargs)
        ):
            agent.run_sync(multimodal)

        # The exact same list object must be passed — no copy, no wrapping
        assert captured_kwargs["user_prompt"] is multimodal


# --- Helper event wrappers for restore_context tests ---


@dataclass
class FakeEventMessage:
    """Mimics EventMessage from akgentic-core with an .event payload."""

    event: object


class TestReactAgentFoldsPendingOperatorActions:
    """ReactAgent.run folds buffered pre-first-run operator actions (FR4, FR5)."""

    @staticmethod
    def _capturing_run_factory(captured: dict):
        """Return an iter() side_effect that records the kwargs passed to iter()."""
        return _capturing_stub_run(captured)

    @pytest.mark.asyncio
    async def test_str_prompt_gets_preamble_prepended(self, minimal_config):
        """FR4: a buffered entry is prepended to a str prompt with a blank-line join."""
        agent = ReactAgent(config=minimal_config)
        agent.context.append_user_prompt("[Operator action] ran /reset")
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("hello")

        assert captured["user_prompt"] == "[Operator action] ran /reset\n\nhello"

    @pytest.mark.asyncio
    async def test_multiple_entries_joined_in_order(self, minimal_config):
        """FR4: multiple buffered entries join with blank lines, then the prompt."""
        agent = ReactAgent(config=minimal_config)
        agent.context.append_user_prompt("first")
        agent.context.append_user_prompt("second")
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("body")

        assert captured["user_prompt"] == "first\n\nsecond\n\nbody"

    @pytest.mark.asyncio
    async def test_list_prompt_gets_preamble_as_leading_element(self, minimal_config):
        """FR4: a multimodal list prompt gets the preamble inserted as its first element."""
        agent = ReactAgent(config=minimal_config)
        agent.context.append_user_prompt("op-A")
        agent.context.append_user_prompt("op-B")
        bc = BinaryContent(data=b"img", media_type="image/png")
        multimodal: list = ["describe", bc]
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run(multimodal)

        assert captured["user_prompt"] == ["op-A\n\nop-B", "describe", bc]

    @pytest.mark.asyncio
    async def test_run_clears_buffer(self, minimal_config):
        """FR4: after a run the operator-action buffer is empty."""
        agent = ReactAgent(config=minimal_config)
        agent.context.append_user_prompt("once")
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("q")

        assert agent.context.drain_pending_user_prompts() == []

    @pytest.mark.asyncio
    async def test_empty_buffer_leaves_str_prompt_unchanged(self, minimal_config):
        """FR4: with nothing buffered, a str prompt passes through unchanged."""
        agent = ReactAgent(config=minimal_config)
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("plain")

        assert captured["user_prompt"] == "plain"

    @pytest.mark.asyncio
    async def test_empty_buffer_leaves_list_prompt_identical(self, minimal_config):
        """FR4: with nothing buffered, a list prompt passes through as the same object."""
        agent = ReactAgent(config=minimal_config)
        multimodal: list = ["text", BinaryContent(data=b"x", media_type="image/png")]
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run(multimodal)

        assert captured["user_prompt"] is multimodal

    @pytest.mark.asyncio
    async def test_message_history_stays_empty_so_system_prompt_injects(self, minimal_config):
        """FR5: folding leaves message_history empty so pydantic-ai injects the system prompt."""
        agent = ReactAgent(config=minimal_config)
        agent.context.append_user_prompt("[Operator action] ran /help")
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("question")

        # message_history is the drain-independent run buffer — empty before the
        # first run, so pydantic-ai's `if not messages` injection path runs.
        assert captured["message_history"] == []


class TestReactAgentRestoreContext:
    """Test ReactAgent.restore_context() method."""

    def test_filters_llm_message_events(self, minimal_config):
        """Test restore_context filters LlmMessageEvent from mixed event list."""
        agent = ReactAgent(config=minimal_config)

        msg1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="World")])

        events = [
            FakeEventMessage(event=LlmMessageEvent(message=msg1)),
            FakeEventMessage(
                event=ToolCallEvent(run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}")
            ),
            FakeEventMessage(event=LlmMessageEvent(message=msg2)),
        ]

        agent.restore_context(events)

        assert len(agent.context.messages) == 2
        assert agent.context.messages[0] is msg1
        assert agent.context.messages[1] is msg2

    def test_ignores_non_llm_events(self, minimal_config):
        """Test restore_context ignores non-LlmMessageEvent events."""
        agent = ReactAgent(config=minimal_config)

        events = [
            FakeEventMessage(
                event=ToolCallEvent(run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}")
            ),
            FakeEventMessage(event="arbitrary string"),
        ]

        agent.restore_context(events)

        assert len(agent.context.messages) == 0

    def test_handles_empty_event_list(self, minimal_config):
        """Test restore_context handles empty event list gracefully."""
        agent = ReactAgent(config=minimal_config)

        # Pre-populate context to verify it gets cleared/replaced
        agent.context.add_message(ModelRequest(parts=[UserPromptPart(content="pre")]))
        assert len(agent.context.messages) == 1

        agent.restore_context([])

        assert len(agent.context.messages) == 0

    def test_handles_zero_llm_events(self, minimal_config):
        """Test restore_context handles list with zero LlmMessageEvent events."""
        agent = ReactAgent(config=minimal_config)

        agent.context.add_message(ModelRequest(parts=[UserPromptPart(content="pre")]))

        events = [
            FakeEventMessage(
                event=ToolCallEvent(run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}")
            ),
        ]

        agent.restore_context(events)

        # Should restore empty list (no LlmMessageEvents found)
        assert len(agent.context.messages) == 0

    def test_preserves_message_order(self, minimal_config):
        """Test restore_context preserves original order of LlmMessageEvent messages."""
        agent = ReactAgent(config=minimal_config)

        msgs = [ModelRequest(parts=[UserPromptPart(content=f"msg-{i}")]) for i in range(5)]
        events = [FakeEventMessage(event=LlmMessageEvent(message=m)) for m in msgs]

        agent.restore_context(events)

        assert len(agent.context.messages) == 5
        for i, m in enumerate(agent.context.messages):
            assert m.parts[0].content == f"msg-{i}"  # type: ignore[attr-defined]


# --- System prompt rendering events: run-lifecycle wiring (Story 6-2) ---


def _system_request_with_run_id(
    *system_parts: tuple[str | None, str],
    run_id: str = "run-1",
) -> ModelRequest:
    """Build a first ModelRequest with system parts + a user part and a run_id.

    Mirrors the shape pydantic-ai stamps on a run's first ModelRequest: one
    SystemPromptPart per (dynamic_ref, content) pair, a trailing UserPromptPart,
    and the run's run_id set on the message.
    """
    parts: list[object] = [
        SystemPromptPart(content=content, dynamic_ref=dynamic_ref)
        for dynamic_ref, content in system_parts
    ]
    parts.append(UserPromptPart(content="Hello"))
    return ModelRequest(parts=parts, run_id=run_id)  # type: ignore[arg-type]


def _make_mock_run(new_messages: list[ModelRequest]):
    """Return a run double (via return_value=) whose new_messages() yields `new_messages`."""
    return _StubRun(new_messages=new_messages, yields=True)


def _system_events(observer: MockObserver) -> list[LlmSystemPromptEvent]:
    """Filter an observer's captured events to LlmSystemPromptEvent instances."""
    return [e for e in observer.events if isinstance(e, LlmSystemPromptEvent)]


def _register_backstory(agent: ReactAgent, rendering: str) -> None:
    """Register one dynamic system prompt, so two agents render an identical block.

    The function's name is what pydantic-ai stamps as ``dynamic_ref``, and the ref is
    part of what the rendering hash is taken over — hence one shared registration
    helper rather than a locally-named closure per test.
    """

    @agent.system_prompt
    def backstory() -> str:
        return rendering


class TestReactAgentRunRecordsSystemPrompt:
    """Per-run system prompt recording, now EventSourcingCapability's (AC 1, 2, 3).

    Driven by real models throughout: the recording rides the capability's closing
    sweep, and a stubbed ``iter()`` fires no hook at all.
    """

    @pytest.mark.asyncio
    async def test_run_records_one_event_with_run_id(self, minimal_config):
        """AC 1/2: one LlmSystemPromptEvent emitted, run_id matches the run's messages."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        _register_backstory(agent, "B.")

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("query")

        events = _system_events(observer)
        assert len(events) == 1
        assert events[0].run_id == str(agent.context.messages[-1].run_id)

    @pytest.mark.asyncio
    async def test_dedup_across_two_unchanged_runs(self, minimal_config):
        """AC 3: two runs with identical rendering emit exactly one event total."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        _register_backstory(agent, "B.")

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("query 1")
            await agent.run("query 2")

        assert len(_system_events(observer)) == 1

    @pytest.mark.asyncio
    async def test_changed_rendering_emits_second_event(self, minimal_config):
        """AC 2: a changed dynamic block emits a second, distinct event."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        renderings = iter(["Day 1.", "Day 2."])

        @agent.system_prompt
        def current_date() -> str:
            return next(renderings)

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("query 1")
            await agent.run("query 2")

        events = _system_events(observer)
        assert len(events) == 2
        assert events[0].content_hash != events[1].content_hash

    @pytest.mark.asyncio
    async def test_no_system_prompt_records_nothing(self, minimal_config):
        """AC 1 edge: a run with nothing to record records no event.

        Retargeted from ``test_no_new_messages_records_nothing``: ``run()`` always
        produces messages, so "no new messages" is not reachable through ReactAgent —
        the reachable form of "nothing to record" is a run with no system prompt at
        all. The companion below drives the run_id half of the guard directly.
        """
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("query")

        assert _system_events(observer) == []

    @pytest.mark.asyncio
    async def test_messages_without_run_id_record_nothing(self, minimal_config):
        """AC 1 edge: a run whose last message lacks a run_id skips the recording.

        Retargeted at the subject, which moved into ``EventSourcingCapability``: a real
        run always stamps a run_id, so the guard is only reachable by driving the
        capability's own ``wrap_run`` hook. The assertion is the original one.
        """
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        capability = EventSourcingCapability(context=agent.context)
        ctx = _bare_run_context()

        async def handler():
            ctx.messages.append(_system_request_with_run_id(("backstory", "B."), run_id=None))  # type: ignore[arg-type]
            return MagicMock()

        await capability.wrap_run(ctx, handler=handler)

        assert _system_events(observer) == []

    @pytest.mark.asyncio
    async def test_a_run_that_added_nothing_records_nothing(self, minimal_config):
        """AC 1 edge: a run that appended no message of its own records no event.

        The other half of the guard the companion above drives, and reachable the same
        way: ``run()`` always produces messages, so "this run added nothing" exists only
        at the capability's own ``wrap_run`` hook. Everything the recording needs is
        deliberately in place — the history ends in a system request carrying a run_id,
        and the ``ContextManager`` holds that same request to hash — so the one thing
        suppressing the event is the run having added nothing. That makes it a real
        guard: drop the length check and this goes red rather than staying vacuously
        green.
        """
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        seeded = _system_request_with_run_id(("backstory", "B."), run_id="r1")
        agent.context.add_message(seeded)
        capability = EventSourcingCapability(context=agent.context)
        ctx = _bare_run_context()
        ctx.messages.append(seeded)

        async def handler():
            return MagicMock()

        await capability.wrap_run(ctx, handler=handler)

        assert _system_events(observer) == []


class TestReactAgentRestoreSeedsSystemPromptHash:
    """restore_context() seeds the dedup hash from persisted events (AC 4, 5, 6, 7)."""

    def test_seed_from_persisted_event(self, minimal_config):
        """AC 4: the seeded hash equals the persisted event's content_hash."""
        agent = ReactAgent(config=minimal_config)

        event = LlmSystemPromptEvent(
            run_id="r1",
            parts=(SystemPromptPartSnapshot(dynamic_ref="b", content="B."),),
            content_hash="abc123",
        )
        agent.restore_context([FakeEventMessage(event=event)])

        assert agent.context._last_system_prompt_hash == "abc123"

    def test_seed_suppresses_unchanged_reemission(self, minimal_config):
        """AC 4: a run matching the seeded rendering emits nothing.

        The seed hash is learned from a probe agent's own real run of the same
        rendering, rather than computed off a hand-built request: the run is real now,
        so the rendering pydantic-ai stamps is the one that must match the seed.
        """
        probe_observer = MockObserver()
        probe = ReactAgent(config=minimal_config, observer=probe_observer)
        _register_backstory(probe, "B.")
        with probe.pydantic_agent.override(model=_text_model()):
            probe.run_sync("probe")
        known_hash = _system_events(probe_observer)[0].content_hash

        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        _register_backstory(agent, "B.")

        event = LlmSystemPromptEvent(
            run_id="r1",
            parts=(SystemPromptPartSnapshot(dynamic_ref="backstory", content="B."),),
            content_hash=known_hash,
        )
        agent.restore_context([FakeEventMessage(event=event)])

        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("query")

        assert _system_events(observer) == []

    def test_post_restore_change_emits(self, minimal_config):
        """AC 5: a run whose rendering differs from the seed emits one event."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        _register_backstory(agent, "New.")

        event = LlmSystemPromptEvent(
            run_id="r1",
            parts=(SystemPromptPartSnapshot(dynamic_ref="backstory", content="Old."),),
            content_hash="seeded-hash-that-differs",
        )
        agent.restore_context([FakeEventMessage(event=event)])

        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("query")

        assert len(_system_events(observer)) == 1

    def test_latest_event_wins_on_restore(self, minimal_config):
        """AC 4: with two persisted events, the later one's hash is seeded."""
        agent = ReactAgent(config=minimal_config)

        first = LlmSystemPromptEvent(run_id="r1", parts=(), content_hash="first-hash")
        second = LlmSystemPromptEvent(run_id="r2", parts=(), content_hash="second-hash")
        agent.restore_context([FakeEventMessage(event=first), FakeEventMessage(event=second)])

        assert agent.context._last_system_prompt_hash == "second-hash"

    def test_no_persisted_event_leaves_hash_none(self, minimal_config):
        """AC 6: only LlmMessageEvents present ⇒ dedup hash stays None."""
        agent = ReactAgent(config=minimal_config)

        msg = ModelRequest(parts=[UserPromptPart(content="Hi")])
        agent.restore_context([FakeEventMessage(event=LlmMessageEvent(message=msg))])

        assert agent.context._last_system_prompt_hash is None

    def test_pre_event_history_first_run_emits(self, minimal_config):
        """AC 6: after a seed-less restore, the first run emits the None → hash event.

        An older team persisted its run-1 messages (whose first ModelRequest
        carries the system parts) but never persisted an LlmSystemPromptEvent, so
        restore leaves the dedup hash at None and the next record emits.
        """
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)

        # Pre-event history: a persisted first ModelRequest with system parts,
        # but no LlmSystemPromptEvent to seed from.
        msg = _system_request_with_run_id(("backstory", "B."), run_id="r1")
        agent.restore_context([FakeEventMessage(event=LlmMessageEvent(message=msg))])
        assert agent.context._last_system_prompt_hash is None

        with agent.pydantic_agent.override(model=_text_model()):
            agent.run_sync("query")

        assert len(_system_events(observer)) == 1

    def test_messages_still_restored_with_seed(self, minimal_config):
        """AC 7: message restore is unchanged when a seed event is also present."""
        agent = ReactAgent(config=minimal_config)

        msg1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="World")])
        seed_event = LlmSystemPromptEvent(run_id="r1", parts=(), content_hash="abc")

        agent.restore_context(
            [
                FakeEventMessage(event=LlmMessageEvent(message=msg1)),
                FakeEventMessage(event=seed_event),
                FakeEventMessage(event=LlmMessageEvent(message=msg2)),
            ]
        )

        assert len(agent.context.messages) == 2
        assert agent.context.messages[0] is msg1
        assert agent.context.messages[1] is msg2
        assert agent.context._last_system_prompt_hash == "abc"


# --- Epic 12 / Story 12-3: compaction wiring ---


class TestReactAgentResolvesCompaction:
    """__init__ resolves the strategy as runtime state on the shared client (AC 4, 10)."""

    def test_init_sets_compaction_attr(self, minimal_config):
        """The agent holds a resolved strategy as a plain attribute, not a config field."""
        agent = ReactAgent(config=minimal_config)
        assert agent._compaction is not None
        assert "_compaction" not in type(agent._config).model_fields

    def test_summarizer_reuses_shared_http_client(self, minimal_config):
        """The default 'summarize' strategy reuses the agent's shared httpx client (no 2nd pool)."""
        agent = ReactAgent(config=minimal_config)
        assert isinstance(agent._compaction, SummarizingCompaction)
        assert agent._compaction._http_client is agent._http_client

    def test_summary_model_cfg_overrides_primary(self):
        """When summary_model_cfg is set, the summarizer is built from it, not model_cfg."""
        summary_cfg = ModelConfig(provider="openai", model="gpt-4o-mini")
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            compaction_cfg=CompactionConfig(summary_model_cfg=summary_cfg),
        )
        agent = ReactAgent(config=config)
        assert isinstance(agent._compaction, SummarizingCompaction)
        assert agent._compaction._model_cfg is summary_cfg

    @pytest.mark.asyncio
    async def test_aclose_closes_the_single_shared_client_once(self, minimal_config):
        """aclose() releases the one client the summarizer shares; a 2nd call is a no-op."""
        agent = ReactAgent(config=minimal_config)
        assert agent._compaction._http_client is agent._http_client  # type: ignore[attr-defined]
        assert not agent._http_client.is_closed

        await agent.aclose()
        assert agent._http_client.is_closed
        await agent.aclose()  # idempotent
        assert agent._http_client.is_closed


class TestReactAgentAutoCompaction:
    """Auto-trigger arithmetic and once-per-turn firing (AC 5).

    The gate moved into ``CompactionCapability.wrap_run``, so every driver here is a REAL
    run: ``await agent._maybe_compact()`` no longer exists as a target, and a stubbed
    ``iter()`` reaches no capability hook. The claims — ``fake.calls == 1`` / ``== 0`` and
    the threshold arithmetic — are unchanged, and each still runs against the same
    ``ReactAgentConfig`` wiring it always did.

    The strategy is swapped on the mounted capability, which is the object the gate reads.
    ``ReactAgent._compaction`` is a read-only read-through property for exactly this
    reason: an assignment to it raises instead of silently testing nothing.
    """

    def test_threshold_arithmetic(self):
        """_compaction_threshold == int(context_length * trigger_ratio)."""
        agent = ReactAgent(config=_over_budget_config())
        assert agent._compaction_threshold() == 850

    def test_threshold_none_when_context_length_unset(self, minimal_config):
        """No context_length ⇒ threshold None (compaction disabled)."""
        agent = ReactAgent(config=minimal_config)
        assert agent._compaction_threshold() is None

    def test_threshold_none_when_auto_trigger_disabled(self):
        """auto_trigger=False ⇒ threshold None: one concept, read once (AC #14).

        The clause the docstring always claimed ("or None when compaction is off") and
        the gate now relies on, rather than re-deriving ``auto_trigger`` itself.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
            compaction_cfg=CompactionConfig(auto_trigger=False, trigger_ratio=0.85),
        )
        assert ReactAgent(config=config)._compaction_threshold() is None

    @pytest.mark.asyncio
    async def test_compacts_when_usage_over_threshold(self):
        """Usage above the threshold compacts via the strategy."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 900  # > 850
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_noop_when_usage_at_or_below_threshold(self):
        """Usage at/below the threshold no-ops."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 850  # == threshold, not strictly above
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_noop_when_no_usage_reported(self):
        """last_input_tokens is None (no-usage provider) ⇒ never mis-fires."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        assert agent._context.last_input_tokens is None
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_noop_when_context_length_none(self, minimal_config):
        """context_length None (threshold None) ⇒ no-op even with huge usage."""
        agent = ReactAgent(config=minimal_config)
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 10_000_000
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_noop_when_auto_trigger_disabled(self):
        """auto_trigger=False ⇒ no-op regardless of usage."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
            compaction_cfg=CompactionConfig(auto_trigger=False, trigger_ratio=0.85),
        )
        agent = ReactAgent(config=config)
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 999
        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_run_auto_compacts_at_most_once_per_turn(self):
        """run() invokes the auto-trigger exactly once per turn, not per model request.

        The turn is deliberately multi-step — the model calls a tool, then answers — so
        two model requests are issued under one run. ``wrap_run`` fires once per RUN, so
        the fold happens once; a fold moved onto ``before_model_request`` would fire twice
        here and this is what would catch it.
        """
        agent = ReactAgent(config=_over_budget_config())

        @agent.pydantic_agent.tool_plain
        def noop() -> str:
            return "ok"

        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 900  # over threshold

        with agent.pydantic_agent.override(model=_tool_then_text_model("noop")):
            await agent.run("q")

        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_conclude_without_tools_keeps_auto_compaction_parity(self):
        """A conclusion arms the same gate a run does (AC #12).

        It always did, via ``_maybe_compact`` in the shared ``_run_with_limits``; it still
        does, via the same capability stack on the same ``run()`` call. Asserted rather
        than assumed, because the mechanism changed underneath it.
        """
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context._last_input_tokens = 900  # over threshold

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.conclude_without_tools("wrap it up")

        assert fake.calls == 1

    def test_swapping_the_strategy_on_the_agent_is_refused(self):
        """``agent._compaction = fake`` raises rather than testing nothing (AC #11).

        Eleven call sites used to do exactly that, against a plain attribute the capability
        would no longer consult. A read-only property makes that mistake loud instead of
        leaving a green test wired to nothing.
        """
        agent = ReactAgent(config=_over_budget_config())
        with pytest.raises(AttributeError):
            agent._compaction = _RecordingCompaction(CompactionResult("S", 1))

    def test_the_agent_reads_the_strategy_the_capability_holds(self):
        """``_compaction`` reports the mounted capability's strategy, not a copy (AC #11)."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        assert agent._compaction is fake

    @pytest.mark.asyncio
    async def test_the_fold_reaches_the_model_on_the_path_react_agent_actually_drives(self):
        """An auto-compacted ``ReactAgent`` run hands the MODEL the folded history.

        The capability-level proof
        (``test_capabilities.test_the_model_receives_exactly_the_post_fold_durable_history``)
        mounts ``CompactionCapability`` on a **bare** ``Agent``. That leaves the shipped
        path — this class's own mount plus ``_run_with_limits``' ``message_history=
        self._context.messages`` hand-off — pinned nowhere: disabling the live write in
        ``compact_now`` turns two ``test_capabilities.py`` tests red and **zero** tests in
        this file. Auto-compaction would be dead code end to end and every test here would
        still be green, because every other one of them asserts on the strategy's call count
        or on the observer's events, never on what the model was handed.

        Both halves are asserted: the summary arrived, and the message it replaced did not.
        """
        seen: list[list[ModelMessage]] = []
        agent = ReactAgent(config=_over_budget_config())
        agent._compactor.strategy = _RecordingCompaction(CompactionResult("S", 1))
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="earlier")]))
        agent._context._last_input_tokens = 900  # over the 850 threshold

        with agent.pydantic_agent.override(model=_history_recording_model(seen)):
            await agent.run("q")

        assert seen, "the model was never called"
        prompts = [
            p.content
            for m in seen[0]
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        assert "[Conversation summary] S" in prompts, "the fold never reached the run"
        assert "earlier" not in prompts, "the folded-away message came back"

    @pytest.mark.asyncio
    async def test_the_persistence_cursor_opens_against_the_post_fold_history(self):
        """The synthetic summary is never persisted as a message event (AC #6).

        The replay rule this pins is real and load-bearing: the
        ``LlmContextCompactedEvent`` already carries the summary, so persisting the
        synthetic ``"[Conversation summary] …"`` request as an ``LlmMessageEvent`` too
        would double-apply it on restore. ``EventSourcingCapability.wrap_run`` nests
        INSIDE ``CompactionCapability``'s and opens its cursor on the post-fold list,
        which is what puts the summary behind the cursor.

        **The stack order is NOT what delivers this, and that was verified rather than
        assumed.** Swapping Compaction and EventSourcing in ``__init__``'s assembly list
        leaves this test green: ``EventSourcingCapability._anchor`` re-opens the cursor
        against the normalised list the first time a node hook hands it over (story 23-4),
        which absorbs any history rewrite performed before that point — including a fold
        applied by a capability mounted further in. Probed directly with a 10-message
        history folded to one: under the swapped order the summary is still not persisted
        and the run's own messages still are. The order guard is
        ``test_internal_capabilities_precede_the_callers``; this is a behavioural guard on
        the outcome, and it is the outcome that matters.

        The run's own messages are all still persisted, which is the other half — a fix
        that stopped persisting the summary by persisting nothing would pass the first
        assertion alone.
        """
        observer = MockObserver()
        agent = ReactAgent(config=_over_budget_config(), observer=observer)
        agent._compactor.strategy = _RecordingCompaction(CompactionResult("S", 1))
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="earlier")]))
        agent._context._last_input_tokens = 900  # over the 850 threshold

        with agent.pydantic_agent.override(model=_text_model()):
            await agent.run("q")

        compacted = [e for e in observer.events if isinstance(e, LlmContextCompactedEvent)]
        assert len(compacted) == 1
        persisted = [e.message for e in observer.events if isinstance(e, LlmMessageEvent)]
        summaries = [
            p.content
            for m in persisted
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and str(p.content).startswith("[Conversation summary]")
        ]
        assert summaries == [], "the synthetic summary was persisted as a message event"
        # The other half, and it has to be COUNTED rather than merely located: a persistence
        # sweep that emitted nothing at all would satisfy the assertion above, and one that
        # emitted only the tail would satisfy any single-message membership check. Scoped to
        # the events after the fold, since the setup's own add_message emits one too.
        fold_at = observer.events.index(compacted[0])
        after_fold = [
            e.message for e in observer.events[fold_at:] if isinstance(e, LlmMessageEvent)
        ]
        assert [type(m) for m in after_fold] == [ModelRequest, ModelResponse]
        prompts = [
            p.content
            for m in after_fold
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        assert prompts == ["q"], "the run's own prompt was not the one persisted"
        assert after_fold[-1] is agent.context.messages[-1]


class TestReactAgentManualCompact:
    """Manual compact() forces, bypassing the budget gate (AC 6)."""

    def test_compact_forces_even_with_compaction_disabled_budget(self, minimal_config):
        """compact() folds even when context_length is None (auto path would no-op)."""
        agent = ReactAgent(config=minimal_config)  # auto_trigger True, context_length None
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compactor.strategy = fake
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))

        status = agent.compact()

        assert fake.calls == 1
        assert "Compacted" in status
        assert agent.context.messages[0].parts[0].content == "[Conversation summary] S"

    def test_compact_raises_after_close(self, minimal_config):
        """compact() raises RuntimeError once the agent is closed."""
        agent = ReactAgent(config=minimal_config)
        agent.close()
        with pytest.raises(RuntimeError, match="ReactAgent is closed"):
            agent.compact()

    @pytest.mark.asyncio
    async def test_compact_now_zero_replacement_emits_no_event(self, minimal_config):
        """A zero-replacement result no-ops: no event, no synthetic message, history untouched."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        agent._compactor.strategy = _RecordingCompaction(CompactionResult("", 0))
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))

        status = await agent._compact_now()

        assert status == "Nothing to compact."
        assert [e for e in observer.events if isinstance(e, LlmContextCompactedEvent)] == []
        assert len(agent.context.messages) == 1

    @pytest.mark.asyncio
    async def test_emitted_event_carries_strategy_tokens_after(self, minimal_config):
        """Story 12-4: the strategy's tokens_after is forwarded onto the emitted event."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        agent._compactor.strategy = _RecordingCompaction(CompactionResult("S", 1, tokens_after=123))
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))

        await agent._compact_now()

        events = [e for e in observer.events if isinstance(e, LlmContextCompactedEvent)]
        assert len(events) == 1
        assert events[0].tokens_after == 123


class TestReactAgentClearContextWrapper:
    """Sync clear_context() wrapper (AC 7)."""

    def test_returns_status_string_and_wipes_history(self, minimal_config):
        """clear_context() returns a status string and empties the history."""
        agent = ReactAgent(config=minimal_config)
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u2")]))

        status = agent.clear_context()

        assert status == "Cleared 2 message(s); system prompt regenerates on the next run."
        assert agent.context.messages == []

    def test_emits_cleared_event(self, minimal_config):
        """clear_context() emits an LlmContextClearedEvent through the context layer."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config, observer=observer)
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))

        agent.clear_context()

        cleared = [e for e in observer.events if isinstance(e, LlmContextClearedEvent)]
        assert len(cleared) == 1
        assert cleared[0].cleared_message_count == 1

    def test_no_loop_interaction(self, minimal_config):
        """clear_context() never touches the event loop (no run_until_complete)."""
        agent = ReactAgent(config=minimal_config)
        agent._context.add_message(ModelRequest(parts=[UserPromptPart(content="u1")]))

        with patch.object(agent._loop, "run_until_complete") as ruc:
            agent.clear_context()

        ruc.assert_not_called()


class TestReactAgentConfigValidatorsAtConstruction:
    """The two ReactAgentConfig validators fire at construction (AC 11)."""

    def test_auto_trigger_with_max_messages_raises(self):
        """auto_trigger=True + max_messages is rejected (window-exclusivity)."""
        with pytest.raises(ValueError, match="max_messages"):
            ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
                compaction_cfg=CompactionConfig(auto_trigger=True),
                max_messages=10,
            )

    def test_threshold_at_or_above_usage_limit_raises(self):
        """An effective threshold >= a set token limit is rejected (dead-trigger guard)."""
        with pytest.raises(ValueError, match="strictly below"):
            ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
                compaction_cfg=CompactionConfig(auto_trigger=True, trigger_ratio=0.85),
                run_usage_limits=RunUsageLimits(input_tokens_limit=800),  # 850 >= 800
            )

    def test_valid_config_builds_agent(self):
        """A valid auto-compaction config builds an agent without error."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o", context_length=10_000),
            compaction_cfg=CompactionConfig(auto_trigger=True, trigger_ratio=0.85),
            run_usage_limits=RunUsageLimits(input_tokens_limit=50_000),
        )
        agent = ReactAgent(config=config)
        assert agent is not None


# ---------------------------------------------------------------------------
# Story 22-2 — switch_model(), the per-run model, and the three couplings
# ---------------------------------------------------------------------------

OPENAI_KEY = "openai:gpt-4o"
GOOGLE_KEY = "google-gla:gemini-2.0-flash"
ANTHROPIC_KEY = "anthropic:claude-sonnet-4-5"


class _SwitchAnswer(BaseModel):
    """A structured output type, so the effective output type is observable at all."""

    answer: str


def _echo_key_stub(key: str):
    """A model function that answers with the roster key it was built for."""

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=key)])

    return stub


class _ModelFactory:
    """Stand-in for ``create_model``: records every call, returns a per-key FunctionModel.

    Monkeypatched over ``akgentic.llm.agent.create_model``, so BOTH construction and
    ``switch_model`` build through it and every model object is identifiable.

    Deliberately NOT ``pydantic_agent.override(model=...)``: pydantic-ai resolves the
    override context-var AHEAD of the per-run ``model=`` argument (``_pick_raw_model``,
    ``pydantic_ai/agent/__init__.py``), so a switch test written inside an ``override``
    block is green whether or not ``switch_model`` does anything at all. That is also why
    the ~800 existing tests, which all use ``override``, are untouched by this story.
    """

    def __init__(self, stub_factory=_echo_key_stub, fail_on: tuple[str, ...] = ()) -> None:
        self.configs: list[ModelConfig] = []
        self.clients: list[Any] = []
        self.built: list[FunctionModel] = []
        self.by_key: dict[str, FunctionModel] = {}
        self._stub_factory = stub_factory
        self._fail_on = set(fail_on)

    def __call__(self, config: ModelConfig, http_client: Any = None) -> FunctionModel:
        self.configs.append(config)
        self.clients.append(http_client)
        key = model_roster_key(config)
        if key in self._fail_on:
            raise ValueError(f"{key} needs an endpoint this environment does not set")
        model = FunctionModel(self._stub_factory(key))
        self.built.append(model)
        self.by_key[key] = model
        return model


@pytest.fixture
def model_factory(monkeypatch):
    """Install ``_ModelFactory`` over ``akgentic.llm.agent.create_model``."""
    factory = _ModelFactory()
    monkeypatch.setattr("akgentic.llm.agent.create_model", factory)
    return factory


def _two_model_config(**kwargs: Any) -> ReactAgentConfig:
    """An openai-active agent with an anthropic entry to switch to."""
    return ReactAgentConfig(
        model_cfg=[
            ModelConfig(provider="openai", model="gpt-4o"),
            ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ],
        **kwargs,
    )


class TestReactAgentModelRosterReaders:
    """``active_model()`` and ``model_roster()`` read the live config (AC #1)."""

    def test_active_model_reads_the_config_at_the_moment_of_the_call(self, model_factory):
        """Not a value cached at construction: a switch moves what the reader returns."""
        agent = ReactAgent(config=_two_model_config())

        assert agent.active_model() is agent._config.model_roster[0]
        entry = agent.switch_model(ANTHROPIC_KEY)
        assert agent.active_model() is entry
        assert agent.active_model() is agent._config.model_roster[1]

    def test_model_roster_returns_a_fresh_list(self, model_factory):
        """The copy is the contract: a tool that sorts the result must not edit the agent.

        The entries themselves are shared on purpose — nothing mutates a ModelConfig in
        place, and ``switch_model`` installs one of them by identity.
        """
        agent = ReactAgent(config=_two_model_config())

        roster = agent.model_roster()
        roster.clear()
        roster.append(ModelConfig(provider="mistral", model="mistral-large-latest"))

        assert len(agent._config.model_roster) == 2
        assert [model_roster_key(e) for e in agent.model_roster()] == [OPENAI_KEY, ANTHROPIC_KEY]
        assert agent.model_roster()[0] is agent._config.model_roster[0]

    def test_both_readers_answer_on_a_single_model_agent(self, model_factory):
        """No roster is not an error for a reader — only for a switch (AC #3)."""
        agent = ReactAgent(
            config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
        )

        assert agent.model_roster() == []
        assert agent.active_model() is agent._config.model_cfg


class TestReactAgentSwitchModelRefusals:
    """Every refusal is one class, and none of them writes anything (AC #2, #3, #4)."""

    def test_model_switch_error_is_a_value_error(self):
        """``except ValueError`` written before this story still catches a refusal.

        The subclassing is the whole reason ``validate_compaction_bounds`` can go on
        raising a plain ``ValueError`` for Pydantic while ``switch_model`` raises this.
        """
        assert issubclass(ModelSwitchError, ValueError)

    def _assert_untouched(self, agent, before) -> None:
        """The three-way identity check: a refusal changed no object (AC #4)."""
        config, model, strategy = before
        assert agent._config is config
        assert agent._model is model
        assert agent._compactor.strategy is strategy

    @staticmethod
    def _snapshot(agent):
        return (agent._config, agent._model, agent._compactor.strategy)

    def test_an_unknown_key_names_every_available_key(self, model_factory):
        """The message is the only diagnosis a tool-driven caller gets, so it lists them."""
        agent = ReactAgent(config=_two_model_config())
        before = self._snapshot(agent)

        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model("openai:gpt-5")

        message = str(exc.value)
        assert "openai:gpt-5" in message
        assert OPENAI_KEY in message
        assert ANTHROPIC_KEY in message
        self._assert_untouched(agent, before)

    def test_an_agent_with_no_roster_gets_its_own_message(self, model_factory):
        """Distinct from the unknown-key message — never 'available keys: ' with none."""
        agent = ReactAgent(
            config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
        )
        before = self._snapshot(agent)

        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model(OPENAI_KEY)

        message = str(exc.value)
        assert "no model roster" in message
        assert "available keys" not in message
        self._assert_untouched(agent, before)

    def test_an_unbuildable_entry_is_wrapped_with_its_cause(self, monkeypatch):
        """ADR-018 Trap 2: a roster entry can fail at SWITCH time, not at construction.

        The provider's own wording is the only diagnosis available, so it is preserved in
        the message and the original exception is kept as ``__cause__``.
        """
        factory = _ModelFactory(fail_on=(ANTHROPIC_KEY,))
        monkeypatch.setattr("akgentic.llm.agent.create_model", factory)
        agent = ReactAgent(config=_two_model_config())
        before = self._snapshot(agent)

        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model(ANTHROPIC_KEY)

        assert "needs an endpoint this environment does not set" in str(exc.value)
        assert isinstance(exc.value.__cause__, ValueError)
        assert not isinstance(exc.value.__cause__, ModelSwitchError)
        self._assert_untouched(agent, before)

    def test_a_provider_exception_is_wrapped_too(self, monkeypatch):
        """AC #2 has no exception clause — a provider's own class is translated as well.

        ``create_model`` reaches third-party provider constructors, and their failures are
        NOT a ``ValueError`` hierarchy: pydantic-ai raises ``UserError``, a
        ``RuntimeError``, for a missing ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``. That
        is the commonest way an entry fails at SWITCH time rather than at construction —
        an agent built on the one provider whose key is set, switching to one whose is
        not — and the two ``ValueError`` cases akgentic pre-checks itself (azure's
        endpoint, google's key) hide it.

        It matters across the package boundary: ``akgentic-agent`` catches this one class
        and must not catch ``Exception``, so an untranslated provider error reaches the
        tool layer as a crash instead of a message the model can correct itself from.
        """
        built = _ModelFactory()

        def failing(config: ModelConfig, http_client: Any = None) -> FunctionModel:
            if model_roster_key(config) == ANTHROPIC_KEY:
                raise UserError("Set the `ANTHROPIC_API_KEY` environment variable")
            return built(config, http_client)

        monkeypatch.setattr("akgentic.llm.agent.create_model", failing)
        agent = ReactAgent(config=_two_model_config())
        before = self._snapshot(agent)

        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model(ANTHROPIC_KEY)

        assert "ANTHROPIC_API_KEY" in str(exc.value)
        assert isinstance(exc.value.__cause__, UserError)
        assert not isinstance(exc.value.__cause__, ValueError)
        self._assert_untouched(agent, before)

    def test_a_switch_that_would_strand_auto_compaction_is_refused(self, model_factory):
        """FR5 at switch time: the candidate entry's threshold is checked before commit.

        The 100k-token entry pushes the trigger to exactly 85_000, which is the run tier's
        cap — the configuration the constructor would have refused outright. Sitting **on**
        the boundary rather than far past it is deliberate: it is what makes this test
        sensitive to the ``>=``/``>`` mutation, and therefore what proves it shares one
        implementation with the construction-time rejection rather than a second copy.
        """
        agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=[
                    ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
                    ModelConfig(
                        provider="anthropic", model="claude-sonnet-4-5", context_length=100_000
                    ),
                ],
                compaction_cfg=CompactionConfig(auto_trigger=True, trigger_ratio=0.85),
                run_usage_limits=RunUsageLimits(input_tokens_limit=85_000),
            )
        )
        before = self._snapshot(agent)

        with pytest.raises(ModelSwitchError) as exc:
            agent.switch_model(ANTHROPIC_KEY)

        message = str(exc.value)
        assert "85000" in message
        assert "input_tokens_limit" in message
        self._assert_untouched(agent, before)


class _ReactAgentConfigWithExtraField(ReactAgentConfig):
    """A config carrying a field ``switch_model``'s write path has never heard of."""

    extra_field: str = "sentinel"


class TestReactAgentSwitchModelCommit:
    """F1 and F2 — what the commit installs, and what it does not (AC #5, #6)."""

    def test_the_commit_keeps_a_field_the_write_path_never_heard_of(self, model_factory):
        """Golden Rule 12's guard, in the only formulation that works.

        The obvious guard — populate every field, switch, compare whole models — is
        insufficient, and provably so: a hand-enumerated ``ReactAgentConfig(...)`` naming
        EVERY field that exists today passes it green. A whole-model comparison can only
        compare fields that exist *now*; a field added later sits at its default on both
        sides, so dropping it is invisible.

        A field the rebuild has never heard of is what makes the defect visible. An
        enumerated reconstruction returns a plain ``ReactAgentConfig`` and fails the
        isinstance outright; ``model_copy(update=...)`` returns the subclass with the
        field intact.
        """
        agent = ReactAgent(
            config=_ReactAgentConfigWithExtraField(
                model_cfg=[
                    ModelConfig(provider="openai", model="gpt-4o"),
                    ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
                ]
            )
        )

        agent.switch_model(ANTHROPIC_KEY)

        assert isinstance(agent._config, _ReactAgentConfigWithExtraField)
        assert agent._config.extra_field == "sentinel"

    def test_the_switch_installs_the_roster_element_itself_from_instances(self, model_factory):
        """F1: ``model_cfg is model_roster[i]`` — the entry, never a copy of it."""
        agent = ReactAgent(config=_two_model_config())

        entry = agent.switch_model(ANTHROPIC_KEY)

        assert entry is agent._config.model_roster[1]
        assert agent._config.model_cfg is agent._config.model_roster[1]

    def test_the_switch_installs_the_roster_element_itself_from_dicts(self, model_factory):
        """F1's other input path: dict entries through ``model_validate``.

        The path where the divergence was filed — Pydantic validates the ``model_cfg``
        dict and the roster dicts into *different* objects, so before any switch
        ``model_cfg is not model_roster[0]``. After a switch the two paths are identical,
        which is what closes F1.
        """
        config = ReactAgentConfig.model_validate(
            {
                "model_cfg": [
                    {"provider": "openai", "model": "gpt-4o"},
                    {"provider": "anthropic", "model": "claude-sonnet-4-5"},
                ]
            }
        )
        assert config.model_cfg is not config.model_roster[0]
        agent = ReactAgent(config=config)

        entry = agent.switch_model(ANTHROPIC_KEY)

        assert entry is agent._config.model_roster[1]
        assert agent._config.model_cfg is agent._config.model_roster[1]

    def test_the_roster_entrys_non_key_fields_replace_the_active_ones(self, model_factory):
        """F2: ``provider:model`` is the identity; the roster entry is the definition.

        A hand-set ``model_cfg`` differing from its roster entry on ``temperature`` loses
        that difference on the first switch back to its key. Merging would mean
        enumerating which fields to keep — the Golden Rule 12 defect — and refusing would
        reject a config that legitimately validated.
        """
        active = ModelConfig(provider="openai", model="gpt-4o", temperature=0.9)
        rostered = ModelConfig(provider="openai", model="gpt-4o", temperature=0.1)
        agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=active,
                model_roster=[
                    rostered,
                    ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
                ],
            )
        )
        assert agent.active_model().temperature == 0.9

        entry = agent.switch_model(OPENAI_KEY)

        assert entry is rostered
        assert agent.active_model() is rostered
        assert agent.active_model().temperature == 0.1

    def test_switching_to_the_already_active_key_is_not_short_circuited(self, model_factory):
        """The F2 rule has no exception, so the no-op case runs resolve/validate/commit.

        A short-circuit would make "the roster entry wins" true except when the key
        happens to already be active — one rule with a hidden branch.
        """
        agent = ReactAgent(config=_two_model_config())
        built_before = len(model_factory.built)
        first_model = agent._model

        entry = agent.switch_model(OPENAI_KEY)

        assert entry is agent._config.model_roster[0]
        assert len(model_factory.built) == built_before + 1
        assert agent._model is not first_model
        assert agent._model is model_factory.built[-1]


class TestReactAgentSwitchModelCouplings:
    """FR4 / NFR1 / NFR2 — a switch replaces the model and nothing else (AC #7, #9, #10)."""

    async def test_the_run_is_served_the_post_switch_model(self, model_factory):
        """The per-run ``model=`` argument, asserted through what the model answers."""
        agent = ReactAgent(config=_two_model_config())

        assert await agent.run("first") == OPENAI_KEY
        agent.switch_model(ANTHROPIC_KEY)
        assert await agent.run("second") == ANTHROPIC_KEY

    async def test_the_pydantic_agent_is_never_rebuilt(self, model_factory):
        """AC #7: the ``Agent`` object survives N switches; only ``_model`` moves."""
        agent = ReactAgent(config=_two_model_config())
        pydantic_agent = agent._pydantic_agent

        for key in (ANTHROPIC_KEY, OPENAI_KEY, ANTHROPIC_KEY, OPENAI_KEY):
            agent.switch_model(key)
            assert agent._pydantic_agent is pydantic_agent

        assert await agent.run("still one agent") == OPENAI_KEY

    def test_every_switch_reuses_the_one_connection_pool(self, model_factory):
        """NFR1: one httpx client for the agent's life, handed to every model built."""
        agent = ReactAgent(config=_two_model_config())
        client = agent._http_client

        for key in (ANTHROPIC_KEY, OPENAI_KEY, ANTHROPIC_KEY):
            agent.switch_model(key)

        assert agent._http_client is client
        assert model_factory.clients == [client] * len(model_factory.clients)
        assert len(model_factory.clients) == 4  # construction + three switches

        agent.close()
        assert client.is_closed

    async def test_a_switch_between_two_runs_loses_nothing(self, model_factory):
        """NFR2: history, run counter and lifetime usage all survive; run 2 sees run 1."""
        seen: list[list[ModelMessage]] = []

        def recording_stub(key: str):
            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                seen.append(list(messages))
                return ModelResponse(
                    parts=[TextPart(content=key)],
                    usage=RequestUsage(input_tokens=7, output_tokens=3),
                )

            return stub

        model_factory._stub_factory = recording_stub
        agent = ReactAgent(config=_two_model_config())
        await agent.run("first")

        messages = list(agent.context.messages)
        run_count = agent._agent_run_count
        usage = (agent._agent_usage.input_tokens, agent._agent_usage.output_tokens)

        agent.switch_model(ANTHROPIC_KEY)

        assert agent.context.messages == messages
        assert agent._agent_run_count == run_count
        assert (agent._agent_usage.input_tokens, agent._agent_usage.output_tokens) == usage

        assert await agent.run("second") == ANTHROPIC_KEY
        assert seen[-1][: len(messages)] == messages
        assert agent._agent_run_count == run_count + 1

    async def test_tools_and_system_prompts_survive_a_switch(self, model_factory):
        """They survive because the ``Agent`` does — nothing re-registers them (AC #7)."""
        offered: list[list[str]] = []
        prompts: list[list[str]] = []

        def inspecting_stub(key: str):
            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                offered.append([t.name for t in info.function_tools])
                prompts.append(
                    [
                        p.content
                        for m in messages
                        if isinstance(m, ModelRequest)
                        for p in m.parts
                        if isinstance(p, SystemPromptPart)
                    ]
                )
                return ModelResponse(parts=[TextPart(content=key)])

            return stub

        model_factory._stub_factory = inspecting_stub
        agent = ReactAgent(config=_two_model_config(), tools=[weather_lookup])

        @agent.system_prompt
        def _backstory(ctx: RunContext[None]) -> str:
            return "you are a switchable agent"

        await agent.run("first")
        agent.switch_model(ANTHROPIC_KEY)
        await agent.run("second")

        assert offered == [["weather_lookup"], ["weather_lookup"]]
        assert prompts[-1] == ["you are a switchable agent"]


def _output_mode_stub(key: str):
    """Answers ``_SwitchAnswer`` however the run's effective output mode asks for it.

    ``info.model_request_parameters.output_mode`` is the direct observable of the type
    pydantic-ai was handed: ``'native'`` for the constructor's ``NativeOutput`` wrapper,
    ``'tool'`` for a raw Pydantic model on a provider without native structured output.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        params = info.model_request_parameters
        if params.output_tools:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=params.output_tools[0].name, args={"answer": key})]
            )
        return ModelResponse(parts=[TextPart(content=f'{{"answer": "{key}"}}')])

    return stub


class TestReactAgentOutputTypeFollowsTheActiveModel:
    """FR6 — the effective output type is resolved per run from the live config (AC #14, #15)."""

    async def test_a_switch_to_a_non_native_provider_unwraps_the_output_type(self, monkeypatch):
        """The latent bug, forbidden form A: a type bound at construction (AC #15).

        Constructed on ``openai``, so the ``Agent``'s constructor ``output_type`` is
        ``NativeOutput(_SwitchAnswer)``. After switching to ``google-gla`` — which has no
        native structured output — a ``run(output_type=None)`` must use the UNWRAPPED
        type. Before this story ``_run_with_limits`` passed ``output_type=None`` on this
        path, pydantic-ai fell back to the constructor's wrapper, and the run went out in
        ``'native'`` mode against a provider that does not support it.
        """
        modes: list[str] = []

        def mode_recording_stub(key: str):
            inner = _output_mode_stub(key)

            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                modes.append(info.model_request_parameters.output_mode)
                return inner(messages, info)

            return stub

        monkeypatch.setattr(
            "akgentic.llm.agent.create_model", _ModelFactory(stub_factory=mode_recording_stub)
        )
        agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=[
                    ModelConfig(provider="openai", model="gpt-4o"),
                    ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
                ]
            ),
            result_type=_SwitchAnswer,
        )

        first = await agent.run("before")
        assert modes == ["native"]
        assert first.answer == OPENAI_KEY

        agent.switch_model(GOOGLE_KEY)
        second = await agent.run("after")

        assert modes == ["native", "tool"]
        assert second.answer == GOOGLE_KEY

    async def test_the_constructor_wrapper_is_never_again_the_effective_type(self, monkeypatch):
        """AC #14: even with no switch, the type is resolved per run from the live config.

        Same wrapper, same mode — the point is that it now comes from
        ``get_output_type(self._config.model_cfg, ...)`` at the call rather than from the
        ``Agent``'s constructor argument, which stays an unused default.
        """
        monkeypatch.setattr(
            "akgentic.llm.agent.create_model", _ModelFactory(stub_factory=_output_mode_stub)
        )
        agent = ReactAgent(config=_two_model_config(), result_type=_SwitchAnswer)

        result = await agent.run("no switch here")

        assert result.answer == OPENAI_KEY


class TestReactAgentMidRunSwitchBoundary:
    """AC #16 — forbidden form B: a value re-read per call but hoisted before the call."""

    @staticmethod
    def _flipping_agent(monkeypatch, target: str, stub_factory, result_type=str, **config_kwargs):
        """An agent whose ``flip_model`` tool switches to ``target`` mid-run."""
        factory = _ModelFactory(stub_factory=stub_factory)
        monkeypatch.setattr("akgentic.llm.agent.create_model", factory)
        agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=[
                    ModelConfig(provider="openai", model="gpt-4o", context_length=1000),
                    ModelConfig(
                        provider="google-gla", model="gemini-2.0-flash", context_length=8000
                    ),
                ],
                **config_kwargs,
            ),
            result_type=result_type,
        )

        @agent.tool
        def flip_model(ctx: RunContext[None]) -> str:
            """Switch the agent's model from inside the run."""
            return model_roster_key(agent.switch_model(target))

        return agent, factory

    async def test_a_mid_run_switch_does_not_change_the_run_in_flight(self, monkeypatch):
        """pydantic-ai binds the model once per ``run()``; the switch lands on the next one.

        The auto-compaction gate is the documented exception: ``_compaction_threshold``
        reads ``context_length`` live, so it moves for the remainder of the run even
        though the model does not. That is invariant 3 working as designed, not a defect
        to cache away.
        """
        answered: list[str] = []

        def tool_then_key_stub(key: str):
            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                already_flipped = any(
                    isinstance(m, ModelRequest)
                    and any(isinstance(p, ToolReturnPart) for p in m.parts)
                    for m in messages
                )
                if not already_flipped:
                    return ModelResponse(parts=[ToolCallPart(tool_name="flip_model", args={})])
                answered.append(key)
                return ModelResponse(parts=[TextPart(content=key)])

            return stub

        agent, _ = self._flipping_agent(monkeypatch, GOOGLE_KEY, tool_then_key_stub)
        assert agent._compaction_threshold() == 850

        first = await agent.run("flip mid-run")

        assert first == OPENAI_KEY
        assert answered == [OPENAI_KEY]
        assert agent.active_model() is agent._config.model_roster[1]
        assert agent._compaction_threshold() == 6800

        assert await agent.run("and now") == GOOGLE_KEY

    async def test_a_conclusion_after_a_breach_is_served_the_post_switch_model(
        self, monkeypatch
    ):
        """A conclusion IS a next run, so it must not be served a hoisted model or type.

        The tool switches to ``google-gla`` on the first request; the second request
        breaches ``tool_calls_limit`` and the recovery seam concludes. The conclusion runs
        with the tools overridden away, so the stub answers — and both what it answers
        with and the mode it was asked for come from the POST-switch config.

        A ``run()`` that hoisted the model and the output type into locals and threaded
        them into both ``_run_with_limits`` and the conclusion would serve the conclusion
        the pre-switch pair, and this test is what catches it.
        """
        seen: list[tuple[str, str]] = []

        def breaching_stub(key: str):
            answer = _output_mode_stub(key)

            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                seen.append((key, info.model_request_parameters.output_mode))
                if info.function_tools:
                    return ModelResponse(parts=[ToolCallPart(tool_name="flip_model", args={})])
                return answer(messages, info)

            return stub

        agent, _ = self._flipping_agent(
            monkeypatch,
            GOOGLE_KEY,
            breaching_stub,
            result_type=_SwitchAnswer,
            run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        )

        result = await agent.run("flip then breach")

        assert isinstance(result, _SwitchAnswer)
        assert result.answer == GOOGLE_KEY
        assert seen[-1] == (GOOGLE_KEY, "tool")
        assert seen[0][0] == OPENAI_KEY


class TestReactAgentSwitchModelSummarizer:
    """FR7 — the summarizer follows the active model, unless it was chosen (AC #17)."""

    def test_the_summarizer_is_rebuilt_on_the_new_entry(self, model_factory):
        """``summary_model_cfg is None`` means "follow the active model", so it follows.

        A new strategy object, built on the switched-to entry and on the agent's own
        client — never a second connection pool.
        """
        agent = ReactAgent(config=_two_model_config())
        before = agent._compactor.strategy

        entry = agent.switch_model(ANTHROPIC_KEY)

        strategy = agent._compactor.strategy
        assert strategy is not before
        assert isinstance(strategy, SummarizingCompaction)
        assert strategy._model_cfg is entry
        assert strategy._http_client is agent._http_client

    def test_a_chosen_summarizer_is_left_alone(self, model_factory):
        """An escalation must not drag a cheap dedicated summarizer up with it."""
        agent = ReactAgent(
            config=_two_model_config(
                compaction_cfg=CompactionConfig(
                    summary_model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini")
                )
            )
        )
        before = agent._compactor.strategy

        agent.switch_model(ANTHROPIC_KEY)

        assert agent._compactor.strategy is before
        assert before._model_cfg.model == "gpt-4o-mini"

    def test_the_compaction_property_stays_read_only(self, model_factory):
        """FR7 assigns ``_compactor.strategy``; ``_compaction`` is deliberately a reader."""
        agent = ReactAgent(config=_two_model_config())
        agent.switch_model(ANTHROPIC_KEY)

        assert agent._compaction is agent._compactor.strategy
        with pytest.raises(AttributeError):
            agent._compaction = _RecordingCompaction(
                CompactionResult(summary="s", replaced_message_count=1, tokens_after=None)
            )


class TestReactAgentSwitchDoesNotSanitizeHistory:
    """Trap 1 — a switch hands the next provider the previous one's messages (AC #18)."""

    async def test_the_accumulated_history_crosses_a_provider_switch_unchanged(
        self, model_factory
    ):
        """What this proves, and what it does not.

        It proves *akgentic* performs no sanitization: the same message objects, in the
        same order, in the same number, reach the run after the switch — nothing is
        rewritten, dropped or re-tagged by ``switch_model``.

        It does NOT prove that a real provider accepts another provider's parts. Whether
        Google tolerates an OpenAI tool-call part is the provider's business, and it stays
        best-effort; story 22-3 documents that. A test cannot settle it against
        ``FunctionModel``.
        """
        seen: list[list[ModelMessage]] = []

        def recording_stub(key: str):
            def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                seen.append(list(messages))
                return ModelResponse(parts=[TextPart(content=key)])

            return stub

        model_factory._stub_factory = recording_stub
        agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=[
                    ModelConfig(provider="openai", model="gpt-4o"),
                    ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
                ]
            )
        )

        await agent.run("on openai")
        carried = list(agent.context.messages)

        agent.switch_model(GOOGLE_KEY)
        await agent.run("on google")

        handed = seen[-1][: len(carried)]
        assert len(handed) == len(carried)
        assert all(a is b for a, b in zip(handed, carried, strict=True))
