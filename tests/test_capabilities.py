"""Tests for LifetimeBudget, EventSourcing and Healing capabilities, mounted standalone.

Every run here is driven by a **bare pydantic-ai ``Agent``** — never a ``ReactAgent``. That is
the point of the decomposition: each capability must be mountable and provable on its own, on
any agent, before ``ReactAgent`` cuts over to ``run()``.

Recording-observer pattern and event-ordering assertion style follow ``test_tool_events.py``
and ``test_system_prompt_event.py``. ``asyncio_mode = "auto"`` — plain ``async def`` tests, no
``@pytest.mark.asyncio``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent, RunCancelled, UsageLimitExceeded
from pydantic_ai.capabilities import AbstractCapability, AgentNode, ProcessHistory, WrapRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from akgentic.llm import (
    AgentUsageLimitError,
    AgentUsageLimits,
    ContextManager,
    EventSourcingCapability,
    HealingCapability,
    LifetimeBudgetCapability,
)
from akgentic.llm.capabilities import RUN_LIMIT_HEALING_MESSAGE
from akgentic.llm.event import (
    LlmMessageEvent,
    LlmSystemPromptEvent,
    LlmUsageEvent,
    ToolCallEvent,
    ToolReturnEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EventCapture:
    """Captures all domain events in emission order."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_event(self, event: object) -> None:
        """Append each received event for ordered assertion."""
        self.events.append(event)


def _manager_with_capture() -> tuple[ContextManager, EventCapture]:
    """Return a ContextManager wired to an EventCapture observer."""
    manager = ContextManager()
    capture = EventCapture()
    manager.subscribe(capture)
    return manager, capture


def _message_events(capture: EventCapture) -> list[LlmMessageEvent]:
    """Every LlmMessageEvent seen, in emission order."""
    return [e for e in capture.events if isinstance(e, LlmMessageEvent)]


def _persisted(capture: EventCapture) -> list[ModelMessage]:
    """The messages the observer saw persisted, in emission order."""
    return [e.message for e in _message_events(capture)]


def _healing_contents(capture: EventCapture) -> list[str]:
    """The content of every ToolReturnPart persisted through a healing ModelRequest."""
    contents: list[str] = []
    for message in _persisted(capture):
        if isinstance(message, ModelRequest):
            contents.extend(str(p.content) for p in message.parts if isinstance(p, ToolReturnPart))
    return contents


def _tool_calling_model() -> FunctionModel:
    """A model that answers every request with the same single tool call."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name="noop", args={})])

    return FunctionModel(model_fn)


def _agent_with_tool(capabilities: list[Any]) -> Agent[None, str]:
    """A bare Agent whose model always calls a trivial tool."""
    agent: Agent[None, str] = Agent(model=_tool_calling_model(), capabilities=capabilities)

    @agent.tool_plain
    def noop() -> str:
        return "ok"

    return agent


def _bare_run_context() -> RunContext[None]:
    """A synthetic RunContext — the healing hooks never read it."""
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


@dataclass
class _CancelAfterResponse(AbstractCapability[Any]):
    """Cancel the run once the model response at ``from_step`` is in durable history.

    This is the empirically verified way (pydantic-ai 2.27.1) to leave a **blind tail**: the
    response is already in the durable history, and cancellation skips ``after_node_run`` for
    the node that put it there, so only ``wrap_run``'s closing sweep can still persist it.

    ``from_step=1`` leaves the run's *first* ``ModelRequest`` unpersisted until that sweep,
    which is what makes the ordering inside the ``finally`` observable.
    """

    from_step: int = 2

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if ctx.run_step >= self.from_step:
            ctx.cancel()
        return response


# ---------------------------------------------------------------------------
# AC #2 — steady-state persistence and the event train
# ---------------------------------------------------------------------------


async def test_completed_run_persists_every_message_once_in_run_order() -> None:
    """A completed run persists what it produced, in run order, exactly once (AC #2)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context)]
    )

    result = await agent.run("hello")

    produced = list(result.all_messages())
    assert _persisted(capture) == produced
    assert context.messages == produced
    assert len(produced) == len({id(m) for m in produced})


async def test_event_train_order_is_message_then_tool_then_usage() -> None:
    """Per message: LlmMessageEvent → tool events → LlmUsageEvent (AC #2)."""
    context, capture = _manager_with_capture()
    agent = _agent_with_tool([EventSourcingCapability(context)])

    with pytest.raises(UsageLimitExceeded):
        await agent.run("hello", usage_limits=UsageLimits(tool_calls_limit=1))

    kinds = [type(e).__name__ for e in capture.events]
    # First message is the user ModelRequest: no tool parts, no usage.
    assert kinds[0] == LlmMessageEvent.__name__
    # Second is the ModelResponse carrying the tool call, followed by its tool event and
    # then its usage event — the train, in that order.
    assert kinds[1:4] == [
        LlmMessageEvent.__name__,
        ToolCallEvent.__name__,
        LlmUsageEvent.__name__,
    ]
    # The tool return rides its own message, ahead of the next response's train.
    assert ToolReturnEvent.__name__ in kinds


# ---------------------------------------------------------------------------
# AC #3 — incoming history is never re-persisted
# ---------------------------------------------------------------------------


async def test_incoming_history_is_not_re_persisted() -> None:
    """Only the messages this run added are persisted (AC #3)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context)]
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="earlier question")]),
        ModelResponse(parts=[TextPart(content="earlier answer")]),
    ]

    result = await agent.run("hello", message_history=history)

    assert _persisted(capture) == list(result.all_messages())[len(history) :]
    for message in history:
        assert message not in _persisted(capture)


async def test_a_history_pydantic_ai_normalises_does_not_shift_the_cursor() -> None:
    """The cursor must index the list the sweep reads, not the one it was measured on (AC #2).

    ``UserPromptNode`` rebinds the run's history to a *normalised copy* of what it was handed:
    consecutive ``ModelRequest``s are merged into one, orphaned tool results dropped. Two
    back-to-back ``record_operator_action`` calls — the shape ``ReactAgent`` drives whenever
    the mailbox delivers twice between turns — merge, so the copy is one message shorter than
    the snapshot ``wrap_run`` measured its cursor against. A cursor carried across that rebind
    sits one past where the run's own messages begin, and the user's prompt is silently never
    persisted.
    """
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context)]
    )

    await agent.run("one")
    context.record_operator_action("[operator] first note")
    context.record_operator_action("[operator] second note")
    before = len(_persisted(capture))

    second = await agent.run("two", message_history=context.messages)

    added = _persisted(capture)[before:]
    assert added == list(second.new_messages()), "the run's own messages must all be persisted"
    assert any(isinstance(m, ModelRequest) for m in added), "the user's prompt was dropped"


# ---------------------------------------------------------------------------
# AC #4 — the cursor is per-run (Trap 3)
# ---------------------------------------------------------------------------


async def test_two_sequential_runs_on_one_instance_persist_each_message_once() -> None:
    """One capability instance, two runs, every message persisted exactly once (AC #4)."""
    context, capture = _manager_with_capture()
    capability = EventSourcingCapability(context)
    agent: Agent[None, str] = Agent(model=TestModel(), capabilities=[capability])

    first = await agent.run("one")
    first_messages = list(first.all_messages())
    second = await agent.run("two", message_history=context.messages)
    second_new = list(second.all_messages())[len(first_messages) :]

    persisted = _persisted(capture)
    assert persisted == first_messages + second_new
    assert len(persisted) == len({id(m) for m in persisted})
    assert context.messages == persisted


async def test_a_message_recorded_between_runs_is_not_re_persisted() -> None:
    """A cursor carried between runs would re-persist what happened between them (AC #4).

    ``record_operator_action`` appends to the context between turns — the production shape
    ``ReactAgent`` drives. The next run's cursor must open at the history it is actually
    handed, not where the previous run stopped, or every message added in between is
    persisted a second time.
    """
    context, capture = _manager_with_capture()
    capability = EventSourcingCapability(context)
    agent: Agent[None, str] = Agent(model=TestModel(), capabilities=[capability])

    await agent.run("one")
    context.record_operator_action("[operator] noted between turns")
    handed_over = context.messages
    second = await agent.run("two", message_history=handed_over)

    persisted = _persisted(capture)
    assert len(persisted) == len({id(m) for m in persisted}), "a message was persisted twice"
    assert persisted == list(second.all_messages())
    assert context.messages == persisted


async def test_a_second_run_on_a_fresh_history_is_not_skipped() -> None:
    """A cursor carried between runs would skip the second run entirely (AC #4).

    The sequential-runs test above hands the first run's history back, which happens to make a
    stale cursor land on the right index. This one clears the conversation between runs — the
    ``/clear`` shape — so the second run starts from an empty history. A cursor that survived
    the first run would sit past the end of the new history and silently persist nothing.
    """
    context, capture = _manager_with_capture()
    capability = EventSourcingCapability(context)
    agent: Agent[None, str] = Agent(model=TestModel(), capabilities=[capability])

    await agent.run("one")
    context.clear_context()
    before_second = len(_persisted(capture))
    second = await agent.run("two")

    second_messages = list(second.all_messages())
    assert _persisted(capture)[before_second:] == second_messages
    assert context.messages == second_messages


# ---------------------------------------------------------------------------
# AC #5 — the closing sweep catches a cancelled run (Trap 1)
# ---------------------------------------------------------------------------


async def test_cancelled_run_still_persists_its_tail() -> None:
    """A run cancelled mid-node persists the tail present when it unwinds (AC #5).

    ``after_node_run`` is skipped for a cancelled node, so the trailing messages can only
    reach the observer through ``wrap_run``'s ``finally``.
    """
    context, capture = _manager_with_capture()
    agent = _agent_with_tool([EventSourcingCapability(context), _CancelAfterResponse()])

    with pytest.raises(RunCancelled):
        await agent.run("hello")

    persisted = _persisted(capture)
    # The tail is the second turn's request/response pair, neither of which any
    # `after_node_run` ever saw.
    assert len(persisted) == 4
    tail = persisted[-1]
    assert isinstance(tail, ModelResponse)
    assert tail.tool_calls
    assert context.messages == persisted


# ---------------------------------------------------------------------------
# AC #6 — system-prompt recording moves with the sweep
# ---------------------------------------------------------------------------


async def test_system_prompt_recorded_once_with_the_run_id_of_the_last_message() -> None:
    """One rendering, one event, correlated to the run that produced it (AC #6)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(),
        system_prompt="You are a fixture.",
        capabilities=[EventSourcingCapability(context)],
    )

    await agent.run("hello")

    events = [e for e in capture.events if isinstance(e, LlmSystemPromptEvent)]
    assert len(events) == 1
    assert events[0].run_id == str(context.messages[-1].run_id)
    # The recording rides the closing sweep, so it lands after every message event.
    assert capture.events.index(events[0]) > capture.events.index(_message_events(capture)[-1])


async def test_unchanged_rendering_across_two_runs_emits_one_event() -> None:
    """Hash dedup is unchanged: a static system prompt emits once, not twice (AC #6)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(),
        system_prompt="You are a fixture.",
        capabilities=[EventSourcingCapability(context)],
    )

    await agent.run("one")
    await agent.run("two", message_history=context.messages)

    assert len([e for e in capture.events if isinstance(e, LlmSystemPromptEvent)]) == 1


async def test_changed_rendering_emits_a_second_event() -> None:
    """A dynamic prompt that renders differently emits a second event (AC #6)."""
    context, capture = _manager_with_capture()
    renderings = iter(["first rendering", "second rendering"])
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context)]
    )

    @agent.system_prompt(dynamic=True)
    def _rendering() -> str:
        return next(renderings)

    await agent.run("one")
    await agent.run("two", message_history=context.messages)

    events = [e for e in capture.events if isinstance(e, LlmSystemPromptEvent)]
    assert len(events) == 2
    assert events[0].content_hash != events[1].content_hash


async def test_the_sweep_runs_before_the_system_prompt_recording() -> None:
    """The closing sweep must put the first ModelRequest in place first (AC #6, Trap 2).

    ``record_system_prompt`` scans the first ``ModelRequest`` **in the ContextManager**. Here
    the run is cancelled before any node boundary that would have persisted it, so the
    ``finally``'s sweep is the only thing that puts one there. Record before sweep and the
    scan finds an empty context, so the rendering is silently never recorded.
    """
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(),
        system_prompt="You are a fixture.",
        capabilities=[EventSourcingCapability(context), _CancelAfterResponse(from_step=1)],
    )

    with pytest.raises(RunCancelled):
        await agent.run("hello")

    assert _persisted(capture), "the sweep must still persist the cancelled run's messages"
    assert len([e for e in capture.events if isinstance(e, LlmSystemPromptEvent)]) == 1


async def test_no_system_parts_records_nothing() -> None:
    """A run with no system prompt records nothing (AC #6)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context)]
    )

    await agent.run("hello")

    assert [e for e in capture.events if isinstance(e, LlmSystemPromptEvent)] == []


async def test_hooks_outside_a_wrapped_run_are_a_no_op() -> None:
    """A sweep on an instance whose ``wrap_run`` never ran persists nothing (AC #1)."""
    context, capture = _manager_with_capture()
    capability = EventSourcingCapability(context)
    ctx = _bare_run_context()
    ctx.messages.append(ModelRequest(parts=[UserPromptPart(content="stray")]))

    sentinel = object()
    result = await capability.after_node_run(ctx, node=None, result=sentinel)  # type: ignore[arg-type]

    assert result is sentinel
    assert capture.events == []
    assert context.messages == []


async def test_for_run_returns_a_fresh_instance_on_the_same_context() -> None:
    """Per-run state isolation is the documented ``for_run`` mechanism (AC #1, #4)."""
    context, _ = _manager_with_capture()
    capability = EventSourcingCapability(context)

    bound = await capability.for_run(_bare_run_context())

    assert bound is not capability
    assert isinstance(bound, EventSourcingCapability)
    assert bound.context is context


# ---------------------------------------------------------------------------
# AC #7 / #8 — healing re-raises, with the single-sourced wording
# ---------------------------------------------------------------------------


async def test_healing_completes_dangling_calls_and_re_raises_the_same_object() -> None:
    """Every dangling call is closed out, then the original error re-raised (AC #7)."""
    context, capture = _manager_with_capture()
    context.add_message(ModelRequest(parts=[UserPromptPart(content="q")]))
    context.add_message(
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="alpha", args={}, tool_call_id="c1"),
                ToolCallPart(tool_name="beta", args={}, tool_call_id="c2"),
            ]
        )
    )
    error = RuntimeError("kaboom")

    with pytest.raises(RuntimeError) as exc_info:
        await HealingCapability(context).on_run_error(_bare_run_context(), error=error)

    assert exc_info.value is error
    healed = context.messages[-1]
    assert isinstance(healed, ModelRequest)
    returns = [p for p in healed.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_name, p.tool_call_id) for p in returns] == [("alpha", "c1"), ("beta", "c2")]
    assert _healing_contents(capture) == [
        "Tool call aborted: RuntimeError: kaboom",
        "Tool call aborted: RuntimeError: kaboom",
    ]


async def test_run_tier_breach_heals_with_the_shared_constant() -> None:
    """A ``UsageLimitExceeded`` heals with ``RUN_LIMIT_HEALING_MESSAGE`` (AC #8)."""
    context, capture = _manager_with_capture()
    context.add_message(
        ModelResponse(parts=[ToolCallPart(tool_name="alpha", args={}, tool_call_id="c1")])
    )
    error = UsageLimitExceeded("budget spent")

    with pytest.raises(UsageLimitExceeded) as exc_info:
        await HealingCapability(context).on_run_error(_bare_run_context(), error=error)

    assert exc_info.value is error
    assert _healing_contents(capture) == [RUN_LIMIT_HEALING_MESSAGE]


def test_the_healing_constant_has_one_definition_in_the_package() -> None:
    """``agent.py`` imports the constant rather than forking a second wording (AC #8)."""
    from akgentic.llm import agent as agent_module
    from akgentic.llm import capabilities as capabilities_module

    assert agent_module.RUN_LIMIT_HEALING_MESSAGE is capabilities_module.RUN_LIMIT_HEALING_MESSAGE


async def test_healing_no_ops_on_an_empty_context_and_still_re_raises() -> None:
    """Nothing to heal, error still propagates (AC #7)."""
    context, capture = _manager_with_capture()
    error = RuntimeError("kaboom")

    with pytest.raises(RuntimeError) as exc_info:
        await HealingCapability(context).on_run_error(_bare_run_context(), error=error)

    assert exc_info.value is error
    assert capture.events == []


async def test_healing_no_ops_when_the_trailing_response_has_no_tool_calls() -> None:
    """A trailing text-only response is not dangling (AC #7)."""
    context, capture = _manager_with_capture()
    context.add_message(ModelResponse(parts=[TextPart(content="done")]))
    before = len(capture.events)
    error = RuntimeError("kaboom")

    with pytest.raises(RuntimeError) as exc_info:
        await HealingCapability(context).on_run_error(_bare_run_context(), error=error)

    assert exc_info.value is error
    assert len(capture.events) == before
    assert _healing_contents(capture) == []


async def test_healing_no_ops_when_the_trailing_message_is_a_request() -> None:
    """A trailing ModelRequest means the previous turn already closed out (AC #7)."""
    context, capture = _manager_with_capture()
    context.add_message(
        ModelResponse(parts=[ToolCallPart(tool_name="alpha", args={}, tool_call_id="c1")])
    )
    context.add_message(
        ModelRequest(parts=[ToolReturnPart(tool_name="alpha", content="ok", tool_call_id="c1")])
    )
    error = RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        await HealingCapability(context).on_run_error(_bare_run_context(), error=error)

    assert _healing_contents(capture) == ["ok"]


# ---------------------------------------------------------------------------
# AC #9 — the composed pair, in production order (Trap 2)
# ---------------------------------------------------------------------------


async def test_composed_pair_persists_the_dangling_response_before_the_healing_request() -> None:
    """Sweep before heal, and the original breach reaches the caller (AC #9)."""
    context, capture = _manager_with_capture()
    agent = _agent_with_tool([EventSourcingCapability(context), HealingCapability(context)])

    with pytest.raises(UsageLimitExceeded) as exc_info:
        await agent.run("hello", usage_limits=UsageLimits(tool_calls_limit=1))

    persisted = _persisted(capture)
    healing = persisted[-1]
    dangling = persisted[-2]
    assert isinstance(dangling, ModelResponse)
    assert dangling.tool_calls, "the run must end with an unanswered tool call"
    assert isinstance(healing, ModelRequest)
    assert [str(p.content) for p in healing.parts if isinstance(p, ToolReturnPart)] == [
        RUN_LIMIT_HEALING_MESSAGE
    ]
    # Ordering: the two assertions above read the last two persisted messages by position,
    # so the dangling response reaching the observer *before* the healing request is what
    # puts each of them where its isinstance check found it.
    assert "tool_calls_limit" in str(exc_info.value)


async def test_composed_pair_propagates_a_generic_exception_with_its_wording() -> None:
    """Generic failures heal with type-and-message and propagate unchanged (AC #8, #9)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=_tool_calling_model(),
        capabilities=[EventSourcingCapability(context), HealingCapability(context)],
    )

    @agent.tool_plain
    def noop() -> str:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await agent.run("hello")

    assert _healing_contents(capture) == ["Tool call aborted: RuntimeError: kaboom"]
    persisted = _persisted(capture)
    assert isinstance(persisted[-2], ModelResponse)
    assert persisted[-2].tool_calls


async def test_composed_pair_leaves_a_successful_run_unhealed() -> None:
    """No error, no healing — the pair is inert on the happy path (AC #9)."""
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(),
        capabilities=[EventSourcingCapability(context), HealingCapability(context)],
    )

    result: AgentRunResult[str] = await agent.run("hello")

    assert _persisted(capture) == list(result.all_messages())
    assert _healing_contents(capture) == []


# ---------------------------------------------------------------------------
# Story 23.4 — the sweep survives a co-mounted capability that changes the
# length of durable history
# ---------------------------------------------------------------------------

REFERENCE_BLOCK = "[source references]"
"""The marker a deployment's injected source-reference block carries."""


def _is_reference_block(message: ModelMessage) -> bool:
    """Whether ``message`` is the reference block standing on its own."""
    return (
        isinstance(message, ModelRequest)
        and len(message.parts) == 1
        and isinstance(part := message.parts[0], UserPromptPart)
        and part.content == REFERENCE_BLOCK
    )


def _prepend_reference_block(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Prepend a source-reference block — the seam ``README.md`` documents.

    Idempotent, because a ``ProcessHistory`` processor runs on **every** model request rather
    than once per run: an unconditional prepend would stack one block per step. Prepending is
    also the only safe direction — pydantic-ai rejects a processed list that does not end with
    a ``ModelRequest``.

    Across runs it does prepend again: the block persisted by the previous run is merged into
    the following user request when pydantic-ai normalises the incoming history, so it is no
    longer a message of its own for this check to find. That is exactly the shift the sweep
    has to survive.
    """
    if messages and _is_reference_block(messages[0]):
        return messages
    return [ModelRequest(parts=[UserPromptPart(content=REFERENCE_BLOCK)]), *messages]


def _drop_oldest_message(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop the oldest message — the removal mirror of ``_prepend_reference_block``.

    Bounded by pydantic-ai's own validation of a processed list: never empty, and the trailing
    ``ModelRequest`` is the one message it can never take. One message per model request, so a
    single-step run drops exactly one.
    """
    return messages[1:] if len(messages) > 1 else messages


def _usage_event_count(capture: EventCapture) -> int:
    """How many ``LlmUsageEvent``s the observer saw."""
    return len([e for e in capture.events if isinstance(e, LlmUsageEvent)])


async def _two_runs(
    extra: list[Any],
) -> tuple[ContextManager, EventCapture, AgentRunResult[str], AgentRunResult[str]]:
    """Drive two consecutive runs on one ``ContextManager``, ``extra`` co-mounted.

    The second run is handed the context's own messages, so a message the first run recorded
    is in front of the second run's cursor — which is what makes a duplicate observable.
    """
    context, capture = _manager_with_capture()
    agent: Agent[None, str] = Agent(
        model=TestModel(), capabilities=[EventSourcingCapability(context), *extra]
    )

    first = await agent.run("one")
    second = await agent.run("two", message_history=context.messages)
    return context, capture, first, second


async def test_a_capability_that_prepends_does_not_re_persist_the_previous_run() -> None:
    """A co-mounted prepend must not make the sweep record a message twice (AC #1, #2, #4).

    pydantic-ai writes a ``before_model_request`` chain's processed list back into durable
    history **in place** — the same list object, new contents. A capability that prepends a
    message therefore shifts every index the cursor was measured against, and a sweep that
    trusts the cursor alone re-records the message the shift moved under it: a phantom
    ``LlmMessageEvent``, a duplicated turn re-sent to the model on the next run, and its
    tokens counted twice when the context is restored.
    """
    context, capture, first, second = await _two_runs(
        [ProcessHistory(processor=_prepend_reference_block)]
    )

    persisted = _persisted(capture)
    assert len(persisted) == len({id(m) for m in persisted}), "a message was persisted twice"
    for response in [m for m in first.all_messages() if isinstance(m, ModelResponse)]:
        assert sum(1 for m in persisted if m is response) == 1, (
            "the first run's response reappeared in the second run's persisted delta"
        )
    # Nothing skipped either, in either run: a fix that stops duplicating by dropping fails
    # here. Both runs, because the co-mount shifts the first run's messages too.
    persisted_ids = {id(m) for m in persisted}
    assert {id(m) for m in first.new_messages()} <= persisted_ids
    assert {id(m) for m in second.new_messages()} <= persisted_ids
    assert context.messages == persisted


async def test_a_capability_that_prepends_does_not_change_the_usage_event_count() -> None:
    """The co-mount must not add a usage event (AC #4).

    ``restore_context`` seeds the lifetime budget by summing ``LlmUsageEvent``s, so a
    re-persisted ``ModelResponse`` double-counts its tokens. Stated as a differential against
    the same two runs with nothing co-mounted, so it does not depend on how many usage events
    a ``TestModel`` run happens to produce.
    """
    _, plain, _, _ = await _two_runs([])
    _, prepending, _, _ = await _two_runs([ProcessHistory(processor=_prepend_reference_block)])

    assert _usage_event_count(prepending) == _usage_event_count(plain)


async def test_a_capability_that_removes_does_not_skip_the_run_s_own_messages() -> None:
    """The removal mirror: a shift the other way must not drop a message (AC #2, #3).

    Position and identity have mirror blind spots — position breaks on insertion and removal,
    identity on a rebuild. A removal ahead of the cursor leaves the cursor sitting past where
    the run's own messages begin, and everything behind it is silently never persisted.
    """
    context, capture, _, second = await _two_runs([ProcessHistory(processor=_drop_oldest_message)])

    persisted = _persisted(capture)
    assert {id(m) for m in second.new_messages()} <= {id(m) for m in persisted}
    assert len(persisted) == len({id(m) for m in persisted}), "a message was persisted twice"
    assert context.messages == persisted


async def test_two_plain_runs_persist_every_message_each_run_produced() -> None:
    """The same no-skipping contract with nothing co-mounted (AC #2).

    The baseline the two tests above are differentials against: whatever the sweep's bound is
    derived from, a run's own messages all reach the observer on the plain path too.
    """
    _, capture, first, second = await _two_runs([])

    persisted_ids = {id(m) for m in _persisted(capture)}
    assert {id(m) for m in first.new_messages()} <= persisted_ids
    assert {id(m) for m in second.new_messages()} <= persisted_ids


# ---------------------------------------------------------------------------
# Story 24.1 — LifetimeBudgetCapability, mounted on a bare Agent
# ---------------------------------------------------------------------------


def _budget(**limits: int | None) -> LifetimeBudgetCapability:
    """A budget capability carrying only the agent-tier limits named."""
    return LifetimeBudgetCapability(limits=AgentUsageLimits(**limits))


def _answering_model(input_tokens: int = 0, output_tokens: int = 0) -> FunctionModel:
    """A model answering with one ``TextPart`` and an exact, caller-chosen token spend.

    ``FunctionModel`` estimates usage only when the response carries none, so setting it
    here pins what the run costs — the fold's assertions are on exact totals.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content="ok")],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    return FunctionModel(model_fn)


def _spending_tool_caller(input_tokens: int, output_tokens: int) -> FunctionModel:
    """A model that spends an exact amount and then asks for the ``boom`` tool."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="boom", args={})],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    return FunctionModel(model_fn)


def _agent_whose_tool_raises(model: FunctionModel, capabilities: list[Any]) -> Agent[None, str]:
    """A bare Agent whose only tool raises, so a run dies after the model was paid for."""
    agent: Agent[None, str] = Agent(model=model, capabilities=capabilities)

    @agent.tool_plain
    def boom() -> str:
        raise RuntimeError("boom")

    return agent


@dataclass
class _HookProbe(AbstractCapability[Any]):
    """Records every hook pydantic-ai fires on it, in order."""

    calls: list[str] = field(default_factory=list)

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Record that the run reached this capability, then run it."""
        self.calls.append("wrap_run")
        return await handler()

    async def before_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any]
    ) -> AgentNode[Any]:
        """Record a node boundary and pass the node through untouched."""
        self.calls.append("before_node_run")
        return node

    async def on_run_error(
        self, ctx: RunContext[Any], *, error: BaseException
    ) -> AgentRunResult[Any]:
        """Record the error hook, then re-raise unchanged."""
        self.calls.append("on_run_error")
        raise error


@dataclass
class _UsageProbe(AbstractCapability[Any]):
    """Captures the run's own usage accumulator, as the run itself will spend through it."""

    seen: list[tuple[RunUsage, int]] = field(default_factory=list)

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Record the accumulator the graph made for this run, and its total right now.

        The total is snapshotted here rather than read back later: the run spends THROUGH
        this very object — that is what makes it the fold's anchor — so it is non-zero by
        the time the assertions run.
        """
        self.seen.append((ctx.usage, ctx.usage.total_tokens))
        return await handler()


async def test_it_mounts_on_a_bare_agent_and_counts_its_runs() -> None:
    """It needs nothing from ReactAgent: one instance, one bare Agent, two runs (AC #1, #5).

    Lifetime state must survive run boundaries, which is why the class does NOT override
    ``for_run``: pydantic-ai's default hands back ``self``, so both runs land on the same
    counters. Reintroduce a ``for_run`` returning a copy and both totals fall back to one
    run's worth.
    """
    budget = _budget(agent_request_limit=5)
    agent: Agent[None, str] = Agent(model=_answering_model(10, 5), capabilities=[budget])

    await agent.run("first")
    await agent.run("second")

    assert budget.run_count == 2
    assert budget.usage.total_tokens == 30


async def test_a_spent_run_budget_refuses_the_run() -> None:
    """The N+1 run is refused with the documented message (AC #2b)."""
    budget = _budget(agent_request_limit=2)
    agent: Agent[None, str] = Agent(model=_answering_model(), capabilities=[budget])

    await agent.run("first")
    await agent.run("second")
    with pytest.raises(AgentUsageLimitError) as exc_info:
        await agent.run("third")

    assert str(exc_info.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"


async def test_the_rejection_itself_consumes_nothing() -> None:
    """The counter reports runs CONSUMED, never runs attempted (AC #3).

    Two refusals in a row, and the message is byte-identical both times: an increment on
    the rejection path would make the second read ``run_count=2`` and the counter would
    drift every time a caller retried.
    """
    budget = _budget(agent_request_limit=1)
    agent: Agent[None, str] = Agent(model=_answering_model(), capabilities=[budget])

    await agent.run("the only run")
    messages = []
    for _ in range(2):
        with pytest.raises(AgentUsageLimitError) as exc_info:
            await agent.run("refused")
        messages.append(str(exc_info.value))

    assert budget.run_count == 1
    assert messages == ["Exceeded the agent_request_limit of 1 (run_count=1)"] * 2


async def test_a_spent_token_budget_refuses_the_run() -> None:
    """The token half refuses too, carrying pydantic-ai's own wording (AC #2a)."""
    budget = _budget(total_tokens_limit=100)
    agent: Agent[None, str] = Agent(model=_answering_model(40, 20), capabilities=[budget])

    await agent.run("first")
    await agent.run("second")
    with pytest.raises(AgentUsageLimitError) as exc_info:
        await agent.run("third")

    assert budget.usage.total_tokens == 120
    assert str(exc_info.value).startswith(
        "Exceeded the total_tokens_limit of 100 (total_tokens=120)"
    )


async def test_a_token_rejection_consumes_no_run_budget() -> None:
    """The two gates are independent, which is why tokens are checked first (AC #2).

    A token refusal that also burned a unit of ``agent_request_limit`` would let repeated
    refusals shrink an unrelated budget.
    """
    budget = _budget(agent_request_limit=5, total_tokens_limit=100)
    agent: Agent[None, str] = Agent(model=_answering_model(150), capabilities=[budget])

    await agent.run("first")
    for _ in range(2):
        with pytest.raises(AgentUsageLimitError):
            await agent.run("refused on tokens")

    assert budget.run_count == 1


async def test_the_counter_advances_before_the_wrapped_call_executes() -> None:
    """A run that fails partway has already been counted (AC #3).

    Check-then-consume: move the increment after ``handler()`` and a run that dies in a
    tool costs the agent nothing, so a failing loop never exhausts its lifetime budget.
    """
    budget = _budget(agent_request_limit=3)
    agent = _agent_whose_tool_raises(_spending_tool_caller(0, 0), [budget])

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("dies in the tool")

    assert budget.run_count == 1


async def test_an_unset_run_limit_never_blocks_but_still_counts() -> None:
    """``agent_request_limit=None`` (the default) blocks nothing and still counts (AC #3)."""
    budget = _budget()
    assert budget.limits.agent_request_limit is None
    agent: Agent[None, str] = Agent(model=_answering_model(), capabilities=[budget])

    for _ in range(4):
        await agent.run("unbounded")

    assert budget.run_count == 4


async def test_a_failed_run_still_folds_what_it_burned() -> None:
    """Tokens a failed run burned are still counted — the provider billed them (AC #4).

    This is also what pins the fold's anchor. ``wrap_run``'s ``ctx.usage`` is
    ``GraphAgentState.usage``, which the graph only ever mutates in place, so it carries
    the run's real cost out through the ``finally`` even though the run has no result to
    read usage off. An anchor that were a snapshot instead would fold zero here, silently.
    """
    budget = _budget(total_tokens_limit=1000)
    agent = _agent_whose_tool_raises(_spending_tool_caller(40, 20), [budget])

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("spend, then fail")

    assert budget.usage.total_tokens == 60


async def test_the_accumulator_is_never_the_runs_own_budget_object() -> None:
    """The lifetime total is folded into, never handed to, the run (AC #4).

    Making the accumulator the object the run spends through raises nothing and logs
    nothing: it would check the RUN tier's limits against lifetime totals, turning every
    per-run cap into a lifetime one. Asserted on identity, and on the run starting at zero
    when the lifetime total is already non-zero.
    """
    budget = _budget()
    probe = _UsageProbe()
    agent: Agent[None, str] = Agent(model=_answering_model(100, 50), capabilities=[budget, probe])

    await agent.run("first")
    await agent.run("second")

    assert budget.usage.total_tokens == 300
    # Non-zero lifetime total by the second run, so "the run starts at zero" is a real
    # claim there rather than a tautology.
    assert [total_at_handover for _, total_at_handover in probe.seen] == [0, 0]
    assert all(usage is not budget.usage for usage, _ in probe.seen)


async def test_a_refused_run_reaches_no_downstream_capability() -> None:
    """A spent agent is refused before anything downstream runs (AC #9).

    The property the position buys, and the reason the budget is mounted outermost: every
    inner capability — including, once compaction joins the stack, the one that pays for a
    summarizer LLM call — is downstream of this refusal.

    It is a property of the ORDER, not of the class. Nothing pins the budget outermost:
    pydantic-ai re-sorts the whole chain topologically as soon as any capability declares
    ``get_ordering()``, so a caller declaring ``position='outermost'`` legitimately lands
    ahead of it. Swap the two list entries below and this test goes red.
    """
    budget = _budget(agent_request_limit=1)
    probe = _HookProbe()
    agent: Agent[None, str] = Agent(model=_answering_model(), capabilities=[budget, probe])

    await agent.run("the only run")
    assert probe.calls, "the probe saw nothing even on the admitted run"
    probe.calls.clear()

    with pytest.raises(AgentUsageLimitError):
        await agent.run("refused")

    assert probe.calls == []


def test_seeding_assigns_rather_than_accumulates() -> None:
    """Restore seeding is assignment: idempotent, and a shorter stream lowers it (AC #6).

    Driven above zero first, so an ``incr`` and a ``max(...)`` high-water mark both go red
    rather than passing on an untouched fresh instance.
    """
    budget = _budget(agent_request_limit=10)
    budget.seed(run_count=3, usage=RunUsage(input_tokens=30, output_tokens=15))
    assert (budget.run_count, budget.usage.total_tokens) == (3, 45)

    budget.seed(run_count=3, usage=RunUsage(input_tokens=30, output_tokens=15))
    assert (budget.run_count, budget.usage.total_tokens) == (3, 45), "seeding accumulated"

    budget.seed(run_count=1, usage=RunUsage(input_tokens=10, output_tokens=5))
    assert (budget.run_count, budget.usage.total_tokens) == (1, 15), (
        "a shorter stream did not lower it"
    )


async def test_a_seeded_budget_is_what_the_next_run_is_enforced_against() -> None:
    """Seeded values are enforced, not merely stored (AC #6, #8)."""
    budget = _budget(agent_request_limit=2)
    agent: Agent[None, str] = Agent(model=_answering_model(), capabilities=[budget])
    budget.seed(run_count=2, usage=RunUsage())

    with pytest.raises(AgentUsageLimitError) as exc_info:
        await agent.run("one run too many")

    assert str(exc_info.value) == "Exceeded the agent_request_limit of 2 (run_count=2)"


def test_it_does_not_override_for_run() -> None:
    """The default ``for_run`` (return ``self``) is what keeps the counters alive (AC #5).

    ``EventSourcingCapability`` overrides it because its cursor is per-run; this state is
    per-AGENT. A per-run copy would reset both counters every run and the limit would never
    fire — with no error and no log line. Asserted structurally as well as behaviourally
    because the behavioural failure is quiet.
    """
    assert LifetimeBudgetCapability.for_run is AbstractCapability.for_run
