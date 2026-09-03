"""Tests for ``DiscardedOutputCapability`` — story 29-2.

Two harnesses, deliberately.

Most specs drive ``after_model_request`` **directly**, with a hand-built
``ModelRequestContext``: the hook's whole contract is "given this response and these request
parameters, return that response", and calling it directly pins the gates one at a time
without a run's scheduling in the way.

The specs that are *about* the anchor — that the strip lands before history, that mount order
does not matter, that restore rebuilds the stripped history — drive a **bare pydantic-ai
``Agent``** with ``EventSourcingCapability`` co-mounted, since only a real run exercises the
append the anchor is chosen to precede. Never a ``ReactAgent``: the capability must be
provable on any agent.

``asyncio_mode = "auto"`` — plain ``async def`` tests, no ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from akgentic.llm import (
    ContextManager,
    DiscardedOutputCapability,
    EventSourcingCapability,
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
)
from akgentic.llm.capabilities import DEFAULT_STRIP_BUDGET
from akgentic.llm.capabilities.discarded_output import _MAX_SCHEMA_DEPTH, _validates
from akgentic.llm.event import LlmMessageEvent, LlmOutputDiscardedEvent, LlmUsageEvent

# ---------------------------------------------------------------------------
# Fixtures of the domain: an output schema and the text that satisfies it
# ---------------------------------------------------------------------------


class _Request(BaseModel):
    """One routed message, mirroring the shape the recorded run emitted."""

    recipient: str
    message: str


class _Out(BaseModel):
    """The run's structured output type."""

    messages: list[_Request]


# The delegation from the recorded run (process c98a1ce2), reduced to this schema.
VALID_OUTPUT = '{"messages": [{"recipient": "@Assistant", "message": "search the web"}]}'
SECOND_VALID_OUTPUT = '{"messages": [{"recipient": "@Human", "message": "on it"}]}'
PROSE = "I should ask the assistant to look this up before answering."
PARTIAL_JSON = '{"messages":[{"message_ty'
WRONG_SHAPE = '{"unexpected": 1}'


def _output_object() -> OutputObjectDefinition:
    """The run's own output schema, exactly as pydantic generates it."""
    return OutputObjectDefinition(json_schema=_Out.model_json_schema(), name="Out")


def _request_context(
    output_object: OutputObjectDefinition | None = None,
) -> ModelRequestContext:
    """A minimal request context carrying just the output schema the hook reads."""
    return ModelRequestContext(
        model=FunctionModel(lambda messages, info: ModelResponse(parts=[])),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            output_mode="prompted",
            output_object=output_object,
        ),
    )


def _run_context() -> RunContext[Any]:
    """A synthetic run context. The hook reads nothing off it; the signature wants one."""
    return RunContext(
        deps=None,
        model=FunctionModel(lambda messages, info: ModelResponse(parts=[])),
        usage=RunUsage(),
    )


def _response(*parts: ModelResponsePart, run_id: str | None = "run-1") -> ModelResponse:
    return ModelResponse(parts=list(parts), run_id=run_id)


def _tool_call() -> ToolCallPart:
    return ToolCallPart(tool_name="noop", args={})


class _Recorder:
    """Observer that records every emitted domain event in order."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


def _wired() -> tuple[ContextManager, _Recorder, DiscardedOutputCapability]:
    """A context manager, its recorder, and a capability bound to it."""
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    return manager, recorder, DiscardedOutputCapability(context=manager)


def _discards(recorder: _Recorder) -> list[LlmOutputDiscardedEvent]:
    return [e for e in recorder.events if isinstance(e, LlmOutputDiscardedEvent)]


def _texts(response: ModelResponse) -> list[str]:
    return [p.content for p in response.parts if isinstance(p, TextPart)]


# ---------------------------------------------------------------------------
# AC 1 — the anchor
# ---------------------------------------------------------------------------


def test_only_after_model_request_is_implemented() -> None:
    """AC 1: one hook, and no ordering declaration.

    Compared against ``AbstractCapability``'s own functions: an inherited hook is the base
    class's function object, an overridden one is not. A node hook or a ``wrap_run`` here
    would mean the capability edits a response already in history and has to be sequenced
    against the persistence sweep — the design AC 3 exists to keep out.
    """
    cls = DiscardedOutputCapability
    assert cls.after_model_request is not AbstractCapability.after_model_request

    for hook in ("before_node_run", "after_node_run", "wrap_node_run", "wrap_run"):
        assert getattr(cls, hook) is getattr(AbstractCapability, hook), hook

    manager, _, capability = _wired()
    assert capability.get_ordering() is None
    assert "get_ordering" not in vars(cls)
    assert manager.messages == []


# ---------------------------------------------------------------------------
# AC 4 — exhaustive only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["early", "graceful"])
async def test_no_op_under_non_exhaustive_strategies(strategy: str) -> None:
    """AC 4: ``early`` and ``graceful`` return the response unchanged and emit nothing.

    ``early`` must be a no-op because there the text is *not* discarded — stripping would
    destroy a live result. ``graceful`` does take the discard branch, but was never
    measured; extending to it is a decision, not a freebie.
    """
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, end_strategy=strategy)  # type: ignore[arg-type]
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _discards(recorder) == []


async def test_no_op_under_early() -> None:
    """AC 4: named spec for ``early`` — the strategy is read from configuration."""
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, end_strategy="early")
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT]
    assert recorder.events == []


async def test_no_op_under_graceful() -> None:
    """AC 4: named spec for ``graceful`` — same discard branch upstream, still not ours."""
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, end_strategy="graceful")
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT]
    assert recorder.events == []


# ---------------------------------------------------------------------------
# AC 5 — strip only what validates against the run's own output schema
# ---------------------------------------------------------------------------


async def test_prose_beside_a_tool_call_survives() -> None:
    """AC 5: plain narration next to a tool call is the model's reasoning — never stripped."""
    manager, recorder, capability = _wired()
    response = _response(TextPart(PROSE), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [PROSE]
    assert _discards(recorder) == []


async def test_partial_json_beside_a_tool_call_survives() -> None:
    """AC 5: truncated JSON does not parse, so it does not validate, so it stays."""
    manager, recorder, capability = _wired()
    response = _response(TextPart(PARTIAL_JSON), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [PARTIAL_JSON]
    assert _discards(recorder) == []


async def test_schema_mismatched_json_survives() -> None:
    """AC 5: valid JSON of the wrong shape is not this run's output — it stays."""
    manager, recorder, capability = _wired()
    response = _response(TextPart(WRONG_SHAPE), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [WRONG_SHAPE]
    assert _discards(recorder) == []


async def test_mixed_parts_strips_only_the_valid_one() -> None:
    """AC 5: one valid output and one prose part ⇒ exactly one stripped, order preserved."""
    manager, recorder, capability = _wired()
    thinking = ThinkingPart(content="weighing options")
    call = _tool_call()
    response = _response(thinking, TextPart(VALID_OUTPUT), TextPart(PROSE), call)

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is not response
    assert list(result.parts) == [thinking, TextPart(PROSE), call]
    assert [e.discarded_content for e in _discards(recorder)] == [(VALID_OUTPUT,)]


async def test_no_output_schema_strips_nothing() -> None:
    """AC 5 / Dev Notes: ``output_object is None`` means nothing can validate.

    ``text`` mode has no schema to be an instance of, and the gate is never relaxed to an
    ``isinstance`` check to compensate.
    """
    manager, recorder, capability = _wired()
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(None), response=response
    )

    assert result is response
    assert _discards(recorder) == []


# ---------------------------------------------------------------------------
# AC 5 — the schema walk itself
#
# This decides what gets deleted from history, so each branch is exercised on its own
# rather than only through the four end-to-end specs above. Every case that reads
# "undecidable" must answer False: an incomplete checker is safe only while it errs
# towards keeping the part.
# ---------------------------------------------------------------------------


def _schema(**keywords: Any) -> OutputObjectDefinition:
    return OutputObjectDefinition(json_schema=dict(keywords))


@pytest.mark.parametrize(
    ("content", "schema", "expected"),
    [
        # -- primitive types, and JSON's bool/int distinction Python does not share --
        ("1", _schema(type="integer"), True),
        ("true", _schema(type="integer"), False),
        ("true", _schema(type="boolean"), True),
        ("1", _schema(type="boolean"), False),
        ("1.5", _schema(type="number"), True),
        ("true", _schema(type="number"), False),
        ('"s"', _schema(type="string"), True),
        ("null", _schema(type="null"), True),
        ("[]", _schema(type="array"), True),
        ("{}", _schema(type="object"), True),
        ("1", _schema(type=["string", "integer"]), True),
        ("1", _schema(type="fantasy"), False),
        # -- required, properties, additionalProperties --
        ('{"a": 1}', _schema(type="object", required=["a", "b"]), False),
        ('{"a": 1}', _schema(type="object", properties={"a": {"type": "integer"}}), True),
        ('{"a": "x"}', _schema(type="object", properties={"a": {"type": "integer"}}), False),
        ('{"z": 1}', _schema(type="object", additionalProperties=False), False),
        ('{"z": 1}', _schema(type="object", additionalProperties={"type": "integer"}), True),
        ('{"z": "x"}', _schema(type="object", additionalProperties={"type": "integer"}), False),
        # -- items --
        ("[1, 2]", _schema(type="array", items={"type": "integer"}), True),
        ('[1, "x"]', _schema(type="array", items={"type": "integer"}), False),
        # -- enum / const --
        ('"a"', _schema(enum=["a", "b"]), True),
        ('"c"', _schema(enum=["a", "b"]), False),
        ('"a"', _schema(enum="not-a-list"), False),
        ("7", _schema(const=7), True),
        ("8", _schema(const=7), False),
        # -- anyOf / oneOf --
        ("1", _schema(anyOf=[{"type": "string"}, {"type": "integer"}]), True),
        ("1.5", _schema(anyOf=[{"type": "string"}, {"type": "integer"}]), False),
        ("1", _schema(anyOf="not-a-list"), False),
        ("1", _schema(oneOf=[{"type": "string"}, {"type": "integer"}]), True),
        ("1", _schema(oneOf=[{"type": "integer"}, {"type": "number"}]), False),
        ("1", _schema(oneOf="not-a-list"), False),
        # -- $ref, resolvable and not --
        (
            '{"a": 1}',
            _schema(**{"$ref": "#/$defs/A", "$defs": {"A": {"type": "object", "required": ["a"]}}}),
            True,
        ),
        ('{"a": 1}', _schema(**{"$ref": "#/$defs/Missing", "$defs": {}}), False),
        ('{"a": 1}', _schema(**{"$ref": "https://example.test/schema"}), False),
        # -- undecidable: a keyword this module does not evaluate --
        ('"abc"', _schema(type="string", pattern="^a"), False),
        ('{"a": 1}', _schema(type="object", allOf=[{"type": "object"}]), False),
        # -- annotations constrain nothing and must not make a schema undecidable --
        ('"s"', _schema(type="string", title="T", description="D", default="x"), True),
        # -- not JSON at all --
        (PROSE, _schema(type="string"), False),
        (PARTIAL_JSON, _schema(type="object"), False),
    ],
)
def test_schema_walk_decides_each_construct(
    content: str, schema: OutputObjectDefinition, expected: bool
) -> None:
    """AC 5: the subset walk answers each construct, and answers False when unsure."""
    assert _validates(content, schema) is expected


def test_a_self_referential_schema_terminates_and_keeps_the_part() -> None:
    """AC 5: a recursive ``$ref`` is bounded, and the bound reads as 'does not validate'.

    Left unbounded this would not terminate; answering True at the bound would strip on a
    schema never actually checked. Both failure modes are worse than a surviving part.
    """
    nested: Any = 1
    for _ in range(_MAX_SCHEMA_DEPTH + 2):
        nested = {"a": nested}
    recursive = OutputObjectDefinition(
        json_schema={
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {"type": "object", "properties": {"a": {"$ref": "#/$defs/Node"}}}},
        }
    )

    import json as _json

    assert _validates(_json.dumps(nested), recursive) is False


@pytest.mark.parametrize(
    "schema",
    [
        _schema(type="object", properties="not-a-mapping"),
        _schema(type="object", properties={"a": "not-a-schema"}),
        _schema(type="object", required="not-a-list", properties={"a": {"type": "integer"}}),
        _schema(type="object", additionalProperties="not-a-schema"),
    ],
    ids=[
        "properties-not-a-mapping",
        "subschema-not-a-mapping",
        "required-not-a-list",
        "additional-properties-not-a-schema",
    ],
)
def test_a_malformed_schema_is_undecidable(schema: OutputObjectDefinition) -> None:
    """AC 5: a schema that is not shaped like one decides nothing, so the part is kept.

    ``required`` here is the case worth naming: ignoring a malformed one would silently
    drop the strongest constraint in the schema and make a mismatched payload look valid.
    """
    assert _validates('{"a": 1}', schema) is False


# ---------------------------------------------------------------------------
# AC 6 — the precondition is text AND tool calls
# ---------------------------------------------------------------------------


async def test_text_without_tool_calls_untouched() -> None:
    """AC 6: a validated output with no tool call is not discarded upstream — leave it.

    That case is the multi-part merge (story 29-3), and this capability must not touch it.
    """
    manager, recorder, capability = _wired()
    response = _response(TextPart(VALID_OUTPUT), TextPart(SECOND_VALID_OUTPUT))

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT, SECOND_VALID_OUTPUT]
    assert _discards(recorder) == []


async def test_tool_calls_without_text_untouched() -> None:
    """AC 6: nothing to strip when the response is tool calls only."""
    manager, recorder, capability = _wired()
    call = _tool_call()
    response = _response(call)

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert list(result.parts) == [call]
    assert _discards(recorder) == []


# ---------------------------------------------------------------------------
# AC 7 — the strip can never empty a response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_parts",
    [
        (),
        (ThinkingPart(content="weighing options"),),
        (TextPart(PROSE),),
    ],
    ids=["bare", "with-thinking", "with-prose"],
)
async def test_response_retains_tool_call_after_strip(
    extra_parts: tuple[ModelResponsePart, ...],
) -> None:
    """AC 7: a tool call is a precondition, so one always survives the strip.

    pydantic-ai's empty-response retry path is therefore unreachable from this capability,
    in every stripped case rather than in the one that happened to be written down.
    """
    _, _, capability = _wired()
    response = _response(TextPart(VALID_OUTPUT), *extra_parts, _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is not response
    assert result.tool_calls
    assert [p for p in result.parts if not isinstance(p, ThinkingPart)]
    assert VALID_OUTPUT not in _texts(result)


# ---------------------------------------------------------------------------
# AC 8 / AC 11 — the per-run budget and its exhaustion event
# ---------------------------------------------------------------------------


async def test_budget_caps_strips_within_a_run() -> None:
    """AC 8: past the budget, responses in the same run come back unchanged."""
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, budget=2)
    ctx, request_context = _run_context(), _request_context(_output_object())

    results = [
        await capability.after_model_request(
            ctx,
            request_context=request_context,
            response=_response(TextPart(VALID_OUTPUT), _tool_call()),
        )
        for _ in range(4)
    ]

    assert [_texts(r) for r in results] == [[], [], [VALID_OUTPUT], [VALID_OUTPUT]]
    assert len([e for e in _discards(recorder) if not e.budget_exhausted]) == 2


async def test_budget_resets_between_runs() -> None:
    """AC 8: the budget is per run, not per agent — a new run id restores it in full."""
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, budget=1)
    ctx, request_context = _run_context(), _request_context(_output_object())

    async def strip(run_id: str) -> ModelResponse:
        return await capability.after_model_request(
            ctx,
            request_context=request_context,
            response=_response(TextPart(VALID_OUTPUT), _tool_call(), run_id=run_id),
        )

    first, spent, second_run = await strip("run-1"), await strip("run-1"), await strip("run-2")

    assert _texts(first) == []
    assert _texts(spent) == [VALID_OUTPUT]
    assert _texts(second_run) == []
    stripped = [e for e in _discards(recorder) if not e.budget_exhausted]
    assert [e.run_id for e in stripped] == ["run-1", "run-2"]


async def test_budget_exhaustion_is_evented() -> None:
    """AC 11: the refusal is recorded, once, on the same event type as a drop.

    The discriminator is ``budget_exhausted``. Content is empty because nothing was
    dropped — the text stayed in the response, and reaches the stream through that
    response's own ``LlmMessageEvent``.
    """
    manager, recorder, _ = _wired()
    capability = DiscardedOutputCapability(context=manager, budget=1)
    ctx, request_context = _run_context(), _request_context(_output_object())

    for _ in range(3):
        await capability.after_model_request(
            ctx,
            request_context=request_context,
            response=_response(TextPart(VALID_OUTPUT), _tool_call()),
        )

    exhausted = [e for e in _discards(recorder) if e.budget_exhausted]
    assert len(exhausted) == 1
    assert exhausted[0].run_id == "run-1"
    assert exhausted[0].discarded_content == ()


def test_default_budget_covers_the_recorded_run() -> None:
    """AC 8: the shipped default leaves headroom over the two strips actually observed."""
    assert DEFAULT_STRIP_BUDGET == 3
    assert DiscardedOutputCapability(context=ContextManager()).budget == DEFAULT_STRIP_BUDGET


# ---------------------------------------------------------------------------
# AC 9 — emission
# ---------------------------------------------------------------------------


async def test_one_event_per_stripped_response() -> None:
    """AC 9: one event per response, not per part, carrying the run id."""
    manager, recorder, capability = _wired()
    response = _response(TextPart(VALID_OUTPUT), TextPart(SECOND_VALID_OUTPUT), _tool_call())

    await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    events = _discards(recorder)
    assert len(events) == 1
    assert events[0].run_id == "run-1"
    assert events[0].budget_exhausted is False


async def test_event_content_is_in_emission_order() -> None:
    """AC 9: the discarded content follows the order the model emitted the parts in.

    A tuple of ``str``, never a tuple of one-character strings: the recorder rejects a bare
    ``str``, and this pins that the capability hands it a list.
    """
    manager, recorder, capability = _wired()
    response = _response(TextPart(SECOND_VALID_OUTPUT), _tool_call(), TextPart(VALID_OUTPUT))

    await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert _discards(recorder)[0].discarded_content == (SECOND_VALID_OUTPUT, VALID_OUTPUT)


# ---------------------------------------------------------------------------
# Live-run harness: the anchor, mount order, restore, and usage
# ---------------------------------------------------------------------------


def _scripted_model(scripted: list[ModelResponse]) -> FunctionModel:
    """A model that replays ``scripted`` in order, then answers with a bare valid output."""
    queue = list(scripted)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if queue:
            return queue.pop(0)
        return ModelResponse(
            parts=[TextPart(SECOND_VALID_OUTPUT)], usage=RequestUsage(input_tokens=5)
        )

    return FunctionModel(model_fn)


def _co_emitting_response() -> ModelResponse:
    """The shape that started this: a validated output beside a tool call, in one response."""
    return ModelResponse(
        parts=[TextPart(VALID_OUTPUT), ToolCallPart(tool_name="noop", args={})],
        usage=RequestUsage(input_tokens=7, output_tokens=41),
    )


def _agent(capabilities: list[Any], scripted: list[ModelResponse]) -> Agent[None, _Out]:
    """A bare ``Agent`` — never a ``ReactAgent`` — with one trivial tool."""
    agent: Agent[None, _Out] = Agent(
        model=_scripted_model(scripted),
        output_type=PromptedOutput(_Out),
        end_strategy="exhaustive",
        capabilities=capabilities,
    )

    @agent.tool_plain
    def noop() -> str:
        return "ok"

    return agent


def _stack(manager: ContextManager, *, strip_first: bool) -> list[Any]:
    """The two capabilities, in the requested mount order."""
    strip = DiscardedOutputCapability(context=manager)
    persist = EventSourcingCapability(context=manager)
    return [strip, persist] if strip_first else [persist, strip]


def _shape(messages: list[ModelMessage]) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
    """Timestamp-independent projection: (message type, ((part type, content), ...))."""
    return [
        (
            type(m).__name__,
            tuple((type(p).__name__, getattr(p, "content", None)) for p in m.parts),
        )
        for m in messages
    ]


def _responses(messages: list[ModelMessage]) -> list[ModelResponse]:
    return [m for m in messages if isinstance(m, ModelResponse)]


async def _run_once(*, strip_first: bool = True) -> tuple[ContextManager, _Recorder]:
    """One full run over a co-emitting response, with both capabilities mounted."""
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    agent = _agent(_stack(manager, strip_first=strip_first), [_co_emitting_response()])
    await agent.run("ask the assistant to search the web")
    return manager, recorder


async def test_stripped_response_is_what_reaches_history() -> None:
    """AC 2: the tool call reaches history; the validated text does not.

    The hook runs before ``ModelRequestNode._append_response``, so persistence never sees
    the raw response at all — there is nothing to un-append.
    """
    manager, _ = await _run_once()

    co_emitted = [r for r in _responses(manager.messages) if r.tool_calls]
    assert len(co_emitted) == 1
    assert co_emitted[0].tool_calls
    assert _texts(co_emitted[0]) == []
    assert VALID_OUTPUT not in [t for r in _responses(manager.messages) for t in _texts(r)]


async def test_stripped_response_is_what_is_evented() -> None:
    """AC 2: ``LlmMessageEvent`` carries the stripped response, never the raw one.

    Annotating instead of stripping would put the stale text back on every restore, with
    every test still green because no test restarts an agent. AC 10 is the other half.
    """
    manager, recorder = await _run_once()

    evented = [
        e.message
        for e in recorder.events
        if isinstance(e, LlmMessageEvent) and isinstance(e.message, ModelResponse)
    ]
    co_emitted = [r for r in evented if r.tool_calls]
    assert len(co_emitted) == 1
    assert _texts(co_emitted[0]) == []
    assert [e.discarded_content for e in _discards(recorder)] == [(VALID_OUTPUT,)]


@pytest.mark.parametrize("strip_first", [True, False], ids=["before", "after"])
async def test_order_independent_against_event_sourcing(strip_first: bool) -> None:
    """AC 3: mounting before or after persistence gives the same history and events.

    This is the whole reason the anchor was chosen. A node-hook implementation would have
    to be sequenced against the persistence sweep, and that sequencing is re-sortable by
    any capability declaring ``get_ordering()``.
    """
    manager, recorder = await _run_once(strip_first=strip_first)
    reference_manager, reference_recorder = await _run_once(strip_first=not strip_first)

    assert _shape(manager.messages) == _shape(reference_manager.messages)
    assert [type(e).__name__ for e in recorder.events] == [
        type(e).__name__ for e in reference_recorder.events
    ]
    assert [e.discarded_content for e in _discards(recorder)] == [
        e.discarded_content for e in _discards(reference_recorder)
    ]


async def test_restore_reproduces_stripped_history() -> None:
    """AC 10: replaying the emitted stream rebuilds the stripped history, not the raw one.

    ``restore_context`` gains no branch for the discard event — it is audit-only — so this
    holds only because ``LlmMessageEvent`` already carries the stripped response.
    """
    manager, recorder = await _run_once()

    agent = ReactAgent(
        config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    )
    agent.restore_context([SimpleNamespace(event=e) for e in recorder.events])

    assert _shape(agent.context.messages) == _shape(manager.messages)
    assert VALID_OUTPUT not in [t for r in _responses(agent.context.messages) for t in _texts(r)]


async def test_usage_event_unchanged_by_strip() -> None:
    """AC 12: ``output_tokens`` still counts the whole generation, discarded text included.

    The recorded message is smaller than the tokens reported for it. That gap is correct —
    the provider billed for the text — and the discard event's content is what closes it.
    """
    stripped_manager, stripped_recorder = await _run_once()

    plain_manager = ContextManager()
    plain_recorder = _Recorder()
    plain_manager.subscribe(plain_recorder)
    plain_agent = _agent(
        [EventSourcingCapability(context=plain_manager)], [_co_emitting_response()]
    )
    await plain_agent.run("ask the assistant to search the web")

    def usages(recorder: _Recorder) -> list[int]:
        return [e.output_tokens for e in recorder.events if isinstance(e, LlmUsageEvent)]

    assert usages(stripped_recorder) == usages(plain_recorder)
    assert 41 in usages(stripped_recorder)
    # ...while the history the tokens were billed for is demonstrably smaller.
    assert _shape(stripped_manager.messages) != _shape(plain_manager.messages)
