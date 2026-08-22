"""Unit tests for ReactAgent implementation."""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, TypeVar, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import UsageLimitExceeded
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
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from akgentic.llm import (
    AgentUsageLimitError,
    AgentUsageLimits,
    CompactionConfig,
    CompactionResult,
    EventSourcingCapability,
    HealingCapability,
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
    RunUsageLimitError,
    RunUsageLimits,
    UsageLimitError,
    UserPrompt,
)
from akgentic.llm.agent import RUN_LIMIT_HEALING_MESSAGE
from akgentic.llm.compaction import SummarizingCompaction
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

    Persistence, system-prompt recording and healing are capability hooks now, and a
    stubbed ``iter()`` fires none of them — so every test whose subject is one of those
    three has to reach the model rather than a double.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(stub)


def _bare_run_context() -> RunContext[None]:
    """A synthetic RunContext for driving a capability hook outside a real run."""
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


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
        # Compared by type rather than by value: both stacks now carry the two internal
        # capabilities, each bound to ITS OWN agent's ContextManager, and dataclass
        # equality makes two capabilities on different contexts unequal. The claim being
        # made is about the shape of the stack, which the sequence of types expresses.
        assert [type(c) for c in agent_omitted.pydantic_agent.root_capability.capabilities] == [
            type(c) for c in agent_explicit_empty.pydantic_agent.root_capability.capabilities
        ]

    def test_internal_capabilities_precede_the_callers(self, minimal_config):
        """The stack is [EventSourcing, Healing, *caller] — the internal two first.

        Asserted as RELATIVE order by index, never as absolute positions:
        pydantic-ai composes a base capability of its own into that surface and
        where that one sits is not this package's contract. Matched by type rather
        than by instance ref for a second reason — ``for_run`` hands every run a
        fresh copy, so an instance ref would not match one either.
        """
        from pydantic_ai.capabilities import Capability

        caller_cap = Capability(id="custom-cap")
        agent = ReactAgent(config=minimal_config, capabilities=[caller_cap])

        mounted = list(agent.pydantic_agent.root_capability.capabilities)
        types = [type(c) for c in mounted]
        assert types.index(EventSourcingCapability) < types.index(HealingCapability)
        assert types.index(HealingCapability) < mounted.index(caller_cap)


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
        spent=None,
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
        self._usage = spent if spent is not None else RunUsage()
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


def _spend_through_the_run_anchor(kwargs: dict, spend: RunUsage) -> None:
    """Report a run's cost the way pydantic-ai does: in place, on the handed-in RunUsage.

    ``ReactAgent`` folds the ``RunUsage`` it passes as ``run(usage=...)``, because that
    object is the only anchor that survives a run which raised. A double that only set
    ``_StubRun.usage`` would therefore report a cost of zero.
    """
    handed_in = kwargs.get("usage")
    if handed_in is not None:
        handed_in.incr(spend)


def _capturing_stub_run(captured: dict, **stub_kwargs):
    """iter() ``side_effect`` that records the call's real kwargs into `captured`.

    Complements passing ``captured=`` directly into ``_StubRun(...)`` (which only
    works with ``return_value=``, a single fixed instance): a ``side_effect=``
    factory must be invoked fresh, with ``iter()``'s actual args/kwargs, on every
    call — this returns such a factory. A ``spent=`` is reported through the
    handed-in accumulator as well, since that is what the fold reads.
    """

    def factory(*args, **kwargs):
        captured.update(kwargs)
        spent = stub_kwargs.get("spent")
        if spent is not None:
            _spend_through_the_run_anchor(kwargs, spent)
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


def _stub_run_spending(input_tokens, output_tokens=0):
    """iter() side_effect returning a run that reports a fixed token spend."""
    spend = RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def factory(*args, **kwargs):
        _spend_through_the_run_anchor(kwargs, spend)
        return _StubRun(spent=spend)

    return factory


class TestReactAgentRunCountEnforcement:
    """Test the agent-lifetime run budget: pre-flight, check-then-consume."""

    def test_fresh_agent_starts_at_zero(self, minimal_config):
        """Test a newly constructed agent has consumed no runs."""
        agent = ReactAgent(config=minimal_config)
        assert agent._agent_run_count == 0

    def test_runs_up_to_the_limit_succeed(self):
        """Test calls 1..N execute and consume exactly N."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
            agent.run_sync("first")
            agent.run_sync("second")
        assert agent._agent_run_count == 2

    def test_run_past_the_limit_raises_and_does_not_consume(self):
        """Test call N+1 is rejected and leaves the counter pinned at the limit."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        """Test the budget check precedes compaction: a rejected run costs nothing."""
        agent = ReactAgent(config=_agent_limit_config(1))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
            agent.run_sync("first")
        with patch.object(agent, "_maybe_compact", new=AsyncMock()) as compact:
            with pytest.raises(UsageLimitError):
                agent.run_sync("second")
        compact.assert_not_called()

    def test_counter_advances_when_the_wrapped_call_raises(self):
        """Test a run that fails partway has already been counted."""
        agent = ReactAgent(config=_agent_limit_config(3))
        with patch.object(agent._pydantic_agent, "iter", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                agent.run_sync("first")
        assert agent._agent_run_count == 1

    def test_counter_advances_when_the_run_tier_limit_fires(self):
        """Test a run-tier breach still consumes the agent-tier budget."""
        agent = ReactAgent(config=_agent_limit_config(3))
        breach = UsageLimitExceeded("The next request would exceed the request_limit of 1")
        with patch.object(agent._pydantic_agent, "iter", side_effect=breach):
            with pytest.raises(RunUsageLimitError) as exc_info:
                agent.run_sync("first")
        assert agent._agent_run_count == 1
        assert "request_limit of 1" in str(exc_info.value)

    def test_repeated_run_tier_failures_exhaust_the_agent_tier(self):
        """Test the two tiers interact: a run-level loop cannot spin forever.

        Which tier fired is asserted by CLASS; the message assertions that follow
        pin the wording, they do not identify the tier.
        """
        agent = ReactAgent(config=_agent_limit_config(2))
        breach = UsageLimitExceeded("The next request would exceed the request_limit of 1")
        with patch.object(agent._pydantic_agent, "iter", side_effect=breach):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
            for _ in range(5):
                agent.run_sync("unbounded")
        assert agent._agent_run_count == 5

    async def test_async_run_enforces_the_same_budget(self):
        """Test the async entry point holds the budget (run_sync only delegates to it)."""
        agent = ReactAgent(config=_agent_limit_config(1))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        """
        agent = ReactAgent(config=_agent_limit_config(2))
        agent.restore_context([FakeEventMessage(event=_usage_event(rid)) for rid in ("r1", "r2")])
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(1000, 500)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(40, 20)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(60)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(0, 60)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(900, 100)):
            assert agent.run_sync("one very expensive run") == "ok"
            assert agent._agent_usage.total_tokens == 1000
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused")

    def test_usage_folds_even_when_the_run_fails_partway(self):
        """Test tokens a failed run burned are still counted — the provider billed them.

        The failure is injected from the stub factory (spend, then raise) rather than
        by patching a private method that no longer exists. See
        ``test_a_real_failing_run_still_folds_its_usage`` for the same claim proved
        against a real model, where the anchor is pydantic-ai's own accumulator.
        """
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=1000))

        def spend_then_fail(*args, **kwargs):
            _spend_through_the_run_anchor(kwargs, RunUsage(input_tokens=40, output_tokens=20))
            raise RuntimeError("boom")

        with patch.object(agent._pydantic_agent, "iter", side_effect=spend_then_fail):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(150)):
            agent.run_sync("first")
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused on tokens")
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused again")
        assert agent._agent_run_count == 1

    def test_token_rejection_happens_before_compaction(self):
        """Test the token check precedes compaction: a refused run pays no summarizer."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(150)):
            agent.run_sync("first")
        with patch.object(agent, "_maybe_compact", new=AsyncMock()) as compact:
            with pytest.raises(UsageLimitError):
                agent.run_sync("refused")
        compact.assert_not_called()

    def test_run_tier_never_receives_the_lifetime_accumulator(self):
        """Test iter() starts every run at zero usage — it never sees the accumulator.

        The mutation this exists to kill (``usage=self._agent_usage`` on the ``iter()``
        call) raises nothing and logs nothing: it checks the RUN tier's limits against
        lifetime totals, silently turning a per-run cap into a lifetime one. No other
        test in this file goes red for it.
        """
        agent = ReactAgent(config=_agent_token_config())
        captured = []

        def capture(*args, **kwargs):
            usage = kwargs.get("usage")
            # The total is snapshotted at hand-over, not read back at the end: the run
            # spends THROUGH this very object (that is the fold's anchor), so the object
            # is non-zero by the time the assertions below run.
            captured.append((usage, None if usage is None else usage.total_tokens))
            _spend_through_the_run_anchor(kwargs, RunUsage(input_tokens=100, output_tokens=50))
            return _StubRun()

        with patch.object(agent._pydantic_agent, "iter", side_effect=capture):
            agent.run_sync("first")
            agent.run_sync("second")

        # Non-zero by the second call, so "fresh" is a real claim there, not a tautology.
        assert agent._agent_usage.total_tokens == 300
        assert len(captured) == 2
        for usage, total_at_handover in captured:
            assert usage is None or (usage is not agent._agent_usage and total_at_handover == 0)

    async def test_async_run_enforces_the_same_token_budget(self):
        """Test the async entry point holds the budget (run_sync only delegates to it)."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(150)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(1)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(60)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(100, 50)):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(80, 40)):
            agent.run_sync("first")
            with pytest.raises(AgentUsageLimitError):
                agent.run_sync("second")

    def test_agent_tier_run_breach_raises_the_agent_subclass(self):
        """Test the N+1 run surfaces as AgentUsageLimitError."""
        agent = ReactAgent(config=_agent_limit_config(2))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        with patch.object(agent_tier_agent._pydantic_agent, "iter", side_effect=_StubRun):
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
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
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

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured)
        ):
            await agent.conclude_without_tools("wrap it up")

        assert captured["usage_limits"].request_limit == 1

    async def test_reason_reaches_the_model_as_the_user_prompt(self, minimal_config):
        """``reason`` is the run's prompt, not a log line (AC #11)."""
        agent = ReactAgent(config=minimal_config)
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=_capturing_stub_run(captured)
        ):
            await agent.conclude_without_tools("your tool budget is spent; answer now")

        assert captured["user_prompt"] == "your tool budget is spent; answer now"

    async def test_conclusion_returns_the_runs_output(self, minimal_config):
        """The conclusion returns the run output the way ``run()`` does (AC #8)."""
        agent = ReactAgent(config=minimal_config)

        with patch.object(
            agent._pydantic_agent, "iter", return_value=_StubRun(output="concluded")
        ):
            result = await agent.conclude_without_tools("wrap it up")

        assert result == "concluded"

    def test_spent_run_budget_refuses_the_conclusion_before_any_model_call(self):
        """An agent at its lifetime run limit cannot conclude (AC #12).

        Terminal by design: the budget that would pay for the conclusion is exactly
        the one that is spent. ``iter`` is asserted un-called — the refusal must be
        pre-flight, not a model round-trip that then fails.
        """
        agent = ReactAgent(config=_agent_limit_config(1))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun):
            agent.run_sync("the only run this agent gets")

        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun) as iter_call:
            with pytest.raises(AgentUsageLimitError):
                agent.conclude_without_tools_sync("wrap it up")

        iter_call.assert_not_called()

    def test_spent_token_budget_refuses_the_conclusion(self):
        """The token half of the agent tier refuses the conclusion too (AC #12)."""
        agent = ReactAgent(config=_agent_token_config(total_tokens_limit=100))
        with patch.object(agent._pydantic_agent, "iter", side_effect=_stub_run_spending(80, 40)):
            agent.run_sync("burn the lifetime token budget")

        with patch.object(agent._pydantic_agent, "iter", side_effect=_StubRun) as iter_call:
            with pytest.raises(AgentUsageLimitError):
                agent.conclude_without_tools_sync("wrap it up")

        iter_call.assert_not_called()

    async def test_conclusion_runs_on_top_of_the_healed_context(self):
        """The healing ToolReturnPart is in the history the conclusion is given (AC #13).

        This is the whole point of healing before concluding: the tool result the
        model reads as the reason it must answer now is already in the context the
        follow-up run is handed.

        The breaching run must be real — healing is ``HealingCapability.on_run_error``
        and no hook fires under a stubbed ``iter()``. Only the conclusion stays stubbed,
        to capture the history it was handed. The run-tier breach also puts the trailing
        dangling ``ModelResponse`` there for real, which the second half asserts.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        )
        agent = ReactAgent(config=config, tools=[weather_lookup])

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

        with patch.object(
            agent._pydantic_agent, "iter", return_value=_StubRun(output="concluded")
        ):
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

    def test_run_still_raises_and_never_concludes_on_its_own(self, minimal_config):
        """run()'s contract is unchanged (AC #15).

        A run-tier breach propagates out of ``run()`` exactly as before, and the
        conclusion is not attempted from inside it: exactly one ``iter()`` call, and
        ``conclude_without_tools`` never invoked. Recovering inside ``run()`` would
        make the run tier a brake nobody can observe.
        """
        agent = ReactAgent(config=minimal_config)
        breach = _StubRun(enter_raises=UsageLimitExceeded("Request limit exceeded"))

        with patch.object(agent._pydantic_agent, "iter", return_value=breach) as iter_call:
            with patch.object(agent, "conclude_without_tools") as conclude:
                with pytest.raises(RunUsageLimitError):
                    agent.run_sync("burn the turn")

        assert iter_call.call_count == 1
        conclude.assert_not_called()

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
        agent.context.record_operator_action("[Operator action] ran /reset")
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
        agent.context.record_operator_action("first")
        agent.context.record_operator_action("second")
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
        agent.context.record_operator_action("op-A")
        agent.context.record_operator_action("op-B")
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
        agent.context.record_operator_action("once")
        captured: dict = {}

        with patch.object(
            agent._pydantic_agent, "iter", side_effect=self._capturing_run_factory(captured)
        ):
            await agent.run("q")

        assert agent.context.drain_pending_operator_actions() == []

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
        agent.context.record_operator_action("[Operator action] ran /help")
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
                event=ToolCallEvent(
                    run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}"
                )
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
                event=ToolCallEvent(
                    run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}"
                )
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
                event=ToolCallEvent(
                    run_id="r1", tool_name="t", tool_call_id="c1", arguments="{}"
                )
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
        agent.restore_context(
            [FakeEventMessage(event=first), FakeEventMessage(event=second)]
        )

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


class TestReactAgentMaybeCompact:
    """Auto-trigger arithmetic and once-per-turn firing (AC 5)."""

    def test_threshold_arithmetic(self):
        """_compaction_threshold == int(context_length * trigger_ratio)."""
        agent = ReactAgent(config=_over_budget_config())
        assert agent._compaction_threshold() == 850

    def test_threshold_none_when_context_length_unset(self, minimal_config):
        """No context_length ⇒ threshold None (compaction disabled)."""
        agent = ReactAgent(config=minimal_config)
        assert agent._compaction_threshold() is None

    @pytest.mark.asyncio
    async def test_compacts_when_usage_over_threshold(self):
        """Usage above the threshold compacts via the strategy."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
        agent._context._last_input_tokens = 900  # > 850
        await agent._maybe_compact()
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_noop_when_usage_at_or_below_threshold(self):
        """Usage at/below the threshold no-ops."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
        agent._context._last_input_tokens = 850  # == threshold, not strictly above
        await agent._maybe_compact()
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_noop_when_no_usage_reported(self):
        """last_input_tokens is None (no-usage provider) ⇒ never mis-fires."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
        assert agent._context.last_input_tokens is None
        await agent._maybe_compact()
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_noop_when_context_length_none(self, minimal_config):
        """context_length None (threshold None) ⇒ no-op even with huge usage."""
        agent = ReactAgent(config=minimal_config)
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
        agent._context._last_input_tokens = 10_000_000
        await agent._maybe_compact()
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
        agent._compaction = fake
        agent._context._last_input_tokens = 999
        await agent._maybe_compact()
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_run_auto_compacts_at_most_once_per_turn(self):
        """run() invokes the auto-trigger exactly once before iter()."""
        agent = ReactAgent(config=_over_budget_config())
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
        agent._context._last_input_tokens = 900  # over threshold

        with patch.object(agent._pydantic_agent, "iter", return_value=_make_mock_run([])):
            await agent.run("q")

        assert fake.calls == 1


class TestReactAgentManualCompact:
    """Manual compact() forces, bypassing the budget gate (AC 6)."""

    def test_compact_forces_even_with_compaction_disabled_budget(self, minimal_config):
        """compact() folds even when context_length is None (auto path would no-op)."""
        agent = ReactAgent(config=minimal_config)  # auto_trigger True, context_length None
        fake = _RecordingCompaction(CompactionResult("S", 1))
        agent._compaction = fake
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
        agent._compaction = _RecordingCompaction(CompactionResult("", 0))
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
        agent._compaction = _RecordingCompaction(CompactionResult("S", 1, tokens_after=123))
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
