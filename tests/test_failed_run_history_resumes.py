"""A run that fails mid-tool leaves a history the next run accepts.

pydantic-ai (2.38+) closes out the dangling tool call itself: the graph appends an
interrupted-request marker when a tool-call turn fails, and the next run synthesizes one
``ToolReturnPart`` per unanswered call when it builds its first request. This package used to
close the call out itself and tested that; nothing here does, and every claim is about what
pydantic-ai hands the model on the run after a failure, exercised through ``ReactAgent``.

Every run is real: the failure has to travel the actual graph for the marker to be appended,
and only a model that records the requests it receives can say what the next run was given.
Assertions are on ``tool_call_id`` equality, ``isinstance(part, ToolReturnPart)`` and
``part.outcome == "interrupted"`` — never on the synthesized content string, the marker's
shape, or message counts, which are pydantic-ai internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
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
    LimitRecoveryCapability,
    LlmMessageEvent,
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
    RunUsageLimitError,
    RunUsageLimits,
)


class _NeverConcludes(LimitRecoveryCapability):
    """The opt-out seam: a run-tier breach raises instead of concluding.

    Declining keeps the breaching run the only run, so the run under test is the one that
    resumes on top of it rather than a conclusion the default policy drove in between.
    """

    async def handle_limit_exceeded(
        self, ctx: RunContext[Any], *, error: UsageLimitExceeded
    ) -> ConclusionDecision | None:
        """Decline to conclude."""
        return None


class _EventCapture:
    """Collect every domain event emitted on a ContextManager."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


@dataclass
class _EventEnvelope:
    """Mimics ``EventMessage`` from akgentic-core with an ``.event`` payload."""

    event: object


RESUME_PROMPT = "second"
"""The prompt of the run that resumes on top of the failed one."""


def _recording_model(received: list[list[ModelMessage]], tool_name: str) -> FunctionModel:
    """A model that records every request it receives.

    Until the resume prompt appears in the history, every request is answered with a call
    to ``tool_name``, so the failing run always has a tool-call turn to die in — whether the
    tool raises on the first call or the second call breaches a one-call limit. Once the
    resume prompt is there, the model answers with text so the resumed run completes.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(list(messages))
        if _prompt_present(messages, RESUME_PROMPT):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={})])

    return FunctionModel(model_fn)


def _closes_out(received: list[ModelMessage], dangling_id: str) -> bool:
    """Whether a request after the response carrying ``dangling_id`` closes that call out.

    True when a ``ModelRequest`` positioned after the ``ModelResponse`` whose tool call has
    ``dangling_id`` carries a ``ToolReturnPart`` with that id and ``outcome == "interrupted"``.
    """
    dangling_at: int | None = None
    for i, message in enumerate(received):
        if isinstance(message, ModelResponse) and any(
            c.tool_call_id == dangling_id for c in message.tool_calls
        ):
            dangling_at = i
            break
    if dangling_at is None:
        return False
    return any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(p, ToolReturnPart)
            and p.tool_call_id == dangling_id
            and p.outcome == "interrupted"
            for p in message.parts
        )
        for message in received[dangling_at + 1 :]
    )


def _dangling_call_id(agent: ReactAgent) -> str:
    """The id of the tool call the failed run left unanswered in the agent's context."""
    responses = [m for m in agent.context.messages if isinstance(m, ModelResponse)]
    assert responses, "the failed run persisted no ModelResponse"
    calls = responses[-1].tool_calls
    assert len(calls) == 1
    return calls[0].tool_call_id


def _prompt_present(received: list[ModelMessage], content: str) -> bool:
    """Whether some ``ModelRequest`` in ``received`` carries a ``UserPromptPart`` ``content``."""
    return any(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content == content for p in m.parts)
        for m in received
    )


def _agent(
    run_usage_limits: RunUsageLimits | None = None,
    limit_recovery: LimitRecoveryCapability | None = None,
    observer: _EventCapture | None = None,
) -> ReactAgent:
    """A ReactAgent on an offline config."""
    model_cfg = ModelConfig(provider="openai", model="gpt-4o")
    if run_usage_limits is None:
        config = ReactAgentConfig(model_cfg=model_cfg)
    else:
        config = ReactAgentConfig(model_cfg=model_cfg, run_usage_limits=run_usage_limits)
    return ReactAgent(config=config, limit_recovery=limit_recovery, observer=observer)


async def test_a_crashed_tool_call_is_closed_out_on_the_next_run() -> None:
    """A tool that raises leaves a history the next run's model receives repaired (AC 2)."""
    received: list[list[ModelMessage]] = []
    agent = _agent()

    @agent.pydantic_agent.tool_plain
    def kaboom() -> str:
        raise RuntimeError("kaboom")

    with agent.pydantic_agent.override(model=_recording_model(received, "kaboom")):
        with pytest.raises(RuntimeError, match="kaboom"):
            await agent.run("first")
        dangling_id = _dangling_call_id(agent)

        assert await agent.run("second") == "done"

    last_received = received[-1]
    assert _closes_out(last_received, dangling_id)
    assert _prompt_present(last_received, "second")


async def test_a_generic_failure_reaches_the_caller_as_the_same_object() -> None:
    """A non-usage failure leaves ``run()`` as the same exception object, traceback intact.

    The operator's stack is formatted off the object that leaves ``run()``, so a capability
    on the error path that wrapped, replaced or suppressed it would break debugging without
    any message-level check noticing. Asserted by identity, which ``match=`` cannot substitute
    for: a re-raised copy matches the same text.
    """
    received: list[list[ModelMessage]] = []
    agent = _agent()
    sentinel = RuntimeError("sentinel failure")

    @agent.pydantic_agent.tool_plain
    def kaboom() -> str:
        raise sentinel

    with agent.pydantic_agent.override(model=_recording_model(received, "kaboom")):
        with pytest.raises(RuntimeError) as exc_info:
            await agent.run("first")

    assert exc_info.value is sentinel
    assert exc_info.value.__traceback__ is not None


async def test_a_run_tier_breach_is_closed_out_on_the_next_run() -> None:
    """A tool-call-limit breach leaves a history the next run's model receives repaired (AC 3)."""
    received: list[list[ModelMessage]] = []
    agent = _agent(
        run_usage_limits=RunUsageLimits(tool_calls_limit=1),
        limit_recovery=_NeverConcludes(),
    )

    @agent.pydantic_agent.tool_plain
    def noop() -> str:
        return "ok"

    with agent.pydantic_agent.override(model=_recording_model(received, "noop")):
        with pytest.raises(RunUsageLimitError):
            await agent.run("first")
        dangling_id = _dangling_call_id(agent)

        assert await agent.run("second") == "done"

    assert _closes_out(received[-1], dangling_id)


async def test_the_failure_survives_event_sourcing_and_restore() -> None:
    """The events a crashed run emitted restore into a history a fresh agent resumes (AC 5).

    The worker-restart path: whatever pydantic-ai appended to close out the failed turn is
    persisted by ``EventSourcingCapability``, serialized to the wire and back, folded by
    ``restore_context`` on a fresh agent, and repaired by pydantic-ai on that agent's first run.

    The repair keys on the marker request's ``state == "interrupted"``; a serializer that
    dropped that field would turn every resumed agent into a hard error on its first prompt,
    so the messages go through pydantic-ai's own JSON round-trip before being restored and the
    marker's ``state`` is asserted on the restored side.
    """
    received: list[list[ModelMessage]] = []
    capture = _EventCapture()
    crashing = _agent(observer=capture)

    @crashing.pydantic_agent.tool_plain
    def kaboom() -> str:
        raise RuntimeError("kaboom")

    with crashing.pydantic_agent.override(model=_recording_model(received, "kaboom")):
        with pytest.raises(RuntimeError, match="kaboom"):
            await crashing.run("first")
    dangling_id = _dangling_call_id(crashing)

    persisted = [e.message for e in capture.events if isinstance(e, LlmMessageEvent)]
    wire = ModelMessagesTypeAdapter.dump_json(persisted)
    restored = ModelMessagesTypeAdapter.validate_json(wire)
    marker = restored[-1]
    assert isinstance(marker, ModelRequest) and marker.state == "interrupted"

    envelopes = [_EventEnvelope(event=LlmMessageEvent(message=m)) for m in restored]
    fresh = _agent()

    @fresh.pydantic_agent.tool_plain
    def kaboom_again() -> str:  # a real restart re-registers the tool
        raise RuntimeError("kaboom")

    fresh.restore_context(envelopes)

    with fresh.pydantic_agent.override(model=_recording_model(received, "kaboom")):
        assert await fresh.run("second") == "done"

    assert _closes_out(received[-1], dangling_id)
