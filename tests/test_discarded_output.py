"""Tests for ``DiscardedOutputCapability`` — story 29-2.

Two harnesses, deliberately.

Most specs drive ``after_model_request`` **directly**, with a hand-built
``ModelRequestContext``: the hook's whole contract is "given this response and these request
parameters, return that response", and calling it directly pins the gates one at a time
without a run's scheduling in the way. They drive the instance ``for_run`` hands back, never
the mounted one — that is the object a real run's hooks are called on.

The specs that are *about* the anchor — that the strip lands before history, that mount order
does not matter, that restore rebuilds the stripped history — drive a **bare pydantic-ai
``Agent``** with ``EventSourcingCapability`` co-mounted, since only a real run exercises the
append the anchor is chosen to precede. Never a ``ReactAgent``: the capability must be
provable on any agent, and a bare one declares its output class exactly as ``ReactAgent``
does.

``asyncio_mode = "auto"`` — plain ``async def`` tests, no ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator
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
from akgentic.llm.capabilities.discarded_output import _validates
from akgentic.llm.event import LlmMessageEvent, LlmOutputDiscardedEvent, LlmUsageEvent

# ---------------------------------------------------------------------------
# Fixtures of the domain: an output class and the text that satisfies it
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
FENCED_OUTPUT = f"```json\n{VALID_OUTPUT}\n```"


def _output_object() -> OutputObjectDefinition:
    """The run's own output schema, exactly as pydantic generates it.

    Only its *presence* is read now — it is what says the request declared structured
    output. Conformance is decided against the class, never against this.
    """
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


def _mount(**kwargs: Any) -> tuple[ContextManager, _Recorder, DiscardedOutputCapability]:
    """A context manager, its recorder, and the capability a host would mount."""
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    return manager, recorder, DiscardedOutputCapability(context=manager, **kwargs)


async def _bind(
    mounted: DiscardedOutputCapability, output_type: object = _Out
) -> DiscardedOutputCapability:
    """Declare a run's output class and take the per-run instance, as a real run does."""
    mounted.expect_output_type(output_type)
    bound = await mounted.for_run(_run_context())
    assert isinstance(bound, DiscardedOutputCapability)
    return bound


async def _wired(
    output_type: object = _Out, **kwargs: Any
) -> tuple[ContextManager, _Recorder, DiscardedOutputCapability]:
    """A mounted capability already bound to one run — the common shape of a spec."""
    manager, recorder, mounted = _mount(**kwargs)
    return manager, recorder, await _bind(mounted, output_type)


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
    against the persistence sweep — the design AC 3 exists to keep out. ``for_run`` is
    overridden and is deliberately not in that list: it is not a hook, it is upstream's
    per-run binding step, and it fires before any hook does.
    """
    cls = DiscardedOutputCapability
    assert cls.after_model_request is not AbstractCapability.after_model_request
    assert cls.for_run is not AbstractCapability.for_run

    for hook in ("before_node_run", "after_node_run", "wrap_node_run", "wrap_run"):
        assert getattr(cls, hook) is getattr(AbstractCapability, hook), hook

    manager, _, capability = _mount()
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
    _, recorder, capability = await _wired(end_strategy=strategy)
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _discards(recorder) == []


async def test_no_op_under_early() -> None:
    """AC 4: named spec for ``early`` — the strategy is read from configuration."""
    _, recorder, capability = await _wired(end_strategy="early")
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT]
    assert recorder.events == []


async def test_no_op_under_graceful() -> None:
    """AC 4: named spec for ``graceful`` — same discard branch upstream, still not ours."""
    _, recorder, capability = await _wired(end_strategy="graceful")
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT]
    assert recorder.events == []


# ---------------------------------------------------------------------------
# AC 5 — strip only what validates against the run's own output class
# ---------------------------------------------------------------------------


async def test_prose_beside_a_tool_call_survives() -> None:
    """AC 5: plain narration next to a tool call is the model's reasoning — never stripped."""
    _, recorder, capability = await _wired()
    response = _response(TextPart(PROSE), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [PROSE]
    assert _discards(recorder) == []


async def test_partial_json_beside_a_tool_call_survives() -> None:
    """AC 5: truncated JSON does not parse, so it does not validate, so it stays."""
    _, recorder, capability = await _wired()
    response = _response(TextPart(PARTIAL_JSON), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [PARTIAL_JSON]
    assert _discards(recorder) == []


async def test_schema_mismatched_json_survives() -> None:
    """AC 5: valid JSON of the wrong shape is not this run's output — it stays."""
    _, recorder, capability = await _wired()
    response = _response(TextPart(WRONG_SHAPE), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [WRONG_SHAPE]
    assert _discards(recorder) == []


async def test_mixed_parts_strips_only_the_valid_one() -> None:
    """AC 5: one valid output and one prose part ⇒ exactly one stripped, order preserved."""
    _, recorder, capability = await _wired()
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
    """AC 5 / Dev Notes: ``output_object is None`` means the output is not in a text part.

    ``text`` mode has no structured output at all and ``tool`` mode puts it in a tool call,
    so a text part beside the call is narration whatever it happens to parse as. The gate
    is never relaxed to an ``isinstance`` check to compensate.
    """
    _, recorder, capability = await _wired()
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(None), response=response
    )

    assert result is response
    assert _discards(recorder) == []


# ---------------------------------------------------------------------------
# AC 5 — the validator itself
#
# This decides what gets deleted from history, so each construct is exercised on its own
# rather than only through the four end-to-end specs above. The verdicts are pydantic's,
# which is the point of the swap: the same call pydantic-ai makes, against the same class.
# ---------------------------------------------------------------------------


class _Inner(BaseModel):
    b: int


class _Int(BaseModel):
    a: int


class _Bool(BaseModel):
    a: bool


class _Float(BaseModel):
    a: float


class _Str(BaseModel):
    a: str


class _Null(BaseModel):
    a: None


class _IntList(BaseModel):
    a: list[int]


class _Mapping(BaseModel):
    a: dict[str, int]


class _Either(BaseModel):
    a: str | int


class _Choice(BaseModel):
    a: Literal["a", "b"]


class _Fixed(BaseModel):
    a: Literal[7]


class _Nested(BaseModel):
    a: _Inner


class _Forbidding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int


class _TwoRequired(BaseModel):
    a: int
    b: int


class _Recursive(BaseModel):
    a: _Recursive | None = None


@pytest.mark.parametrize(
    ("content", "output_type", "expected"),
    [
        # -- primitives, and the coercions pydantic performs where JSON Schema would not.
        # The bool/number pairs below are the three verdicts the schema walk got the other
        # way round: it applied JSON Schema's rule, under which `true` is not an `integer`.
        # pydantic's lax mode — the mode pydantic-ai validates the real output in — coerces
        # between them, so these payloads ARE this run's output and the graph does discard
        # them. Answering False here left them in history, which is the defect.
        ('{"a": 1}', _Int, True),
        ('{"a": true}', _Int, True),
        ('{"a": true}', _Bool, True),
        ('{"a": 1}', _Bool, True),
        ('{"a": 1.5}', _Float, True),
        ('{"a": 1}', _Float, True),
        ('{"a": true}', _Float, True),
        ('{"a": "s"}', _Str, True),
        ('{"a": 1}', _Str, False),
        ('{"a": null}', _Null, True),
        ('{"a": []}', _IntList, True),
        ('{"a": {}}', _Mapping, True),
        ('{"a": 1}', _Either, True),
        ('{"a": "5"}', _Int, True),
        # -- required fields, and what an absent or extra key does --
        ('{"a": 1}', _TwoRequired, False),
        ('{"a": 1}', _Int, True),
        ('{"a": "x"}', _Int, False),
        ('{"a": 1, "z": 2}', _Int, True),
        ('{"a": 1, "z": 2}', _Forbidding, False),
        ('{"a": 1}', _Forbidding, True),
        # -- collection members --
        ('{"a": [1, 2]}', _IntList, True),
        ('{"a": [1, "x"]}', _IntList, False),
        ('{"a": {"k": 1}}', _Mapping, True),
        ('{"a": {"k": "x"}}', _Mapping, False),
        # -- closed value sets --
        ('{"a": "a"}', _Choice, True),
        ('{"a": "c"}', _Choice, False),
        ('{"a": 7}', _Fixed, True),
        ('{"a": 8}', _Fixed, False),
        # -- unions --
        ('{"a": 1}', _Either, True),
        ('{"a": 1.5}', _Either, False),
        ('{"a": null}', _Either, False),
        # -- a nested model, which is what a `$ref` in the schema stood for --
        ('{"a": {"b": 1}}', _Nested, True),
        ('{"a": {}}', _Nested, False),
        ('{"a": 1}', _Nested, False),
        # -- not JSON at all, or not this run's output --
        (PROSE, _Out, False),
        (PARTIAL_JSON, _Out, False),
        ("", _Out, False),
        ('{"a": 1}', _Out, False),
        (VALID_OUTPUT, _Out, True),
        (WRONG_SHAPE, _Out, False),
        # pydantic-ai strips markdown fences before validating and this does not, so a
        # fenced payload is KEPT. The divergence is in the direction that costs the fix,
        # never data — and the helper that would close it is private to pydantic-ai.
        (FENCED_OUTPUT, _Out, False),
    ],
)
def test_the_validator_decides_each_construct(
    content: str, output_type: type[BaseModel], expected: bool
) -> None:
    """AC 5: the run's own class answers each construct, through pydantic."""
    assert _validates(content, output_type) is expected


def test_a_numeric_string_validates_because_validation_is_not_strict() -> None:
    """AC 5: ``strict=True`` is deliberately NOT passed, and this is where it would show.

    ``NativeOutput(output_type, strict=True)`` sets a JSON-Schema strictness hint for the
    provider, not pydantic's validation mode: pydantic-ai's own output processor calls
    ``validator.validate_json`` with no strict flag. Validating strictly here would keep a
    part the graph discards anyway, which is the whole defect.
    """
    assert _validates('{"a": "5"}', _Int) is True
    assert _validates('{"a": 5}', _Int) is True


def test_a_self_referential_model_terminates() -> None:
    """AC 5: a recursive output class is decided, not given up on.

    The schema walk this replaced bounded recursion at a fixed depth and called anything
    deeper undecidable, so a legitimately deep output survived a discard the graph
    performs regardless. pydantic recurses over the payload itself and answers.
    """
    deep = "{"
    for _ in range(40):
        deep += '"a": {'
    deep += '"a": null' + "}" * 40 + "}"

    assert _validates(deep, _Recursive) is True
    assert _validates('{"a": {"a": 1}}', _Recursive) is False


class _Exploding(BaseModel):
    """An output class whose validator raises something pydantic does not wrap."""

    a: int

    @field_validator("a")
    @classmethod
    def _boom(cls, value: int) -> int:
        raise RuntimeError("validator blew up")


def test_a_validator_that_raises_keeps_the_part() -> None:
    """AC 5: a caller's output class must not end a run from this hook.

    ``ValueError`` and ``AssertionError`` become ``ValidationError``; anything else
    propagates out of ``model_validate_json``. This hook fires on every model response, so
    that would turn an audit-only capability into a run-ending one. It is logged and read
    as "does not validate" — the direction that keeps the part.
    """
    assert _validates('{"a": 1}', _Exploding) is False


# ---------------------------------------------------------------------------
# AC 5 — the output class comes from the host, once per run
# ---------------------------------------------------------------------------


async def test_no_op_when_the_host_declares_nothing() -> None:
    """AC 5: an unbound mount cannot know what this run produces, so it strips nothing."""
    manager, recorder, mounted = _mount()
    bound = await mounted.for_run(_run_context())
    assert isinstance(bound, DiscardedOutputCapability)
    response = _response(TextPart(VALID_OUTPUT), _tool_call())

    result = await bound.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _discards(recorder) == []
    assert manager.messages == []


@pytest.mark.parametrize(
    "declared", [str, None, int, list[int]], ids=["str", "none", "int", "list"]
)
async def test_no_op_for_an_output_type_that_is_not_a_model(declared: object) -> None:
    """AC 5: only a ``BaseModel`` subclass is decidable here.

    ``str`` and ``None`` are the ordinary cases — no structured output to be an instance
    of. A bare ``int`` or ``list[int]`` is refused for a sharper reason: pydantic-ai wraps
    it in a ``{"response": ...}`` envelope before the model sees it, so validating the
    naked type would answer a different question than the graph does, and in the direction
    that strips.
    """
    _, recorder, capability = await _wired(declared)
    response = _response(TextPart(VALID_OUTPUT), TextPart("5"), _tool_call())

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert capability._output_type is None
    assert _discards(recorder) == []


async def test_a_quoted_string_survives_under_the_default_str_output_type() -> None:
    """AC 5: the single most important spec in this module.

    ``ReactAgent.result_type`` DEFAULTS to ``str``, so ``str`` is the output type most runs
    of this package declare — and *every* JSON string is a valid ``str``, as the first
    assertion below states outright. A gate that accepted ``str`` would therefore not
    weaken this capability, it would invert it into "delete any string co-emitted with a
    tool call", which is a data-deletion regression worse than anything the schema walk it
    replaced could produce.

    Two independent gates stop that, and both are asserted here because a future change
    that relaxes one must not be able to pass by leaning on the other:

    1. ``str`` is not a ``BaseModel`` subclass, so ``expect_output_type`` parks ``None`` and
       the run has no output class at all — the hook returns at its first gate.
    2. A ``str`` output type puts the request in ``text`` mode, where ``output_object`` is
       ``None`` — so even an output class that *had* been accepted would strip nothing.
    """
    narration = '"I will look that up for you."'
    # The premise, stated rather than assumed: this is why gate one has to exist.
    assert TypeAdapter(str).validate_json(narration) == "I will look that up for you."

    _, recorder, capability = await _wired(str)
    response = _response(TextPart(narration), _tool_call())

    # Gate one — no output class was accepted for this run.
    assert capability._output_type is None
    kept = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    # Gate two — a text-mode request carries no output_object, so nothing is discardable.
    _, _, model_typed = await _wired(_Out)
    still_kept = await model_typed.after_model_request(
        _run_context(), request_context=_request_context(None), response=response
    )

    assert _texts(kept) == [narration]
    assert _texts(still_kept) == [narration]
    assert _discards(recorder) == []


async def test_a_str_run_strips_nothing_end_to_end() -> None:
    """AC 5: the same guarantee through a real run, where neither gate is hand-built.

    A ``str``-output agent that co-emits a quoted string with a tool call must reach history
    with the string intact. This is the regression the two gates exist to prevent, proved
    against pydantic-ai's own request construction rather than a fixture — which is what
    makes it evidence that ``text`` mode really does leave ``output_object`` unset.
    """
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    strip = DiscardedOutputCapability(context=manager)
    narration = '"I will look that up for you."'

    agent: Agent[None, str] = Agent(
        model=_scripted_model(
            [
                ModelResponse(
                    parts=[TextPart(narration), ToolCallPart(tool_name="noop", args={})],
                    usage=RequestUsage(input_tokens=7),
                ),
                ModelResponse(parts=[TextPart("done")], usage=RequestUsage(input_tokens=3)),
            ]
        ),
        output_type=str,
        end_strategy="exhaustive",
        capabilities=[strip, EventSourcingCapability(context=manager)],
    )

    @agent.tool_plain
    def noop() -> str:
        return "ok"

    strip.expect_output_type(str)
    await agent.run("look this up")

    assert narration in [t for r in _responses(manager.messages) for t in _texts(r)]
    assert _discards(recorder) == []


async def test_the_declaration_is_consumed_by_for_run() -> None:
    """AC 5: a run that declares nothing never inherits the previous run's class.

    The pending slot lives on the shared mount, so leaving it set would let an undeclared
    run validate against whatever the last one produced — a strip decided against the
    wrong schema, which deletes a part that is not this run's output.
    """
    _, recorder, mounted = _mount()
    first = await _bind(mounted)
    second = await mounted.for_run(_run_context())
    assert isinstance(second, DiscardedOutputCapability)

    stripped = await first.after_model_request(
        _run_context(),
        request_context=_request_context(_output_object()),
        response=_response(TextPart(VALID_OUTPUT), _tool_call()),
    )
    kept = await second.after_model_request(
        _run_context(),
        request_context=_request_context(_output_object()),
        response=_response(TextPart(VALID_OUTPUT), _tool_call(), run_id="run-2"),
    )

    assert _texts(stripped) == []
    assert _texts(kept) == [VALID_OUTPUT]
    assert [e.run_id for e in _discards(recorder)] == ["run-1"]


async def test_react_agent_declares_the_effective_output_class() -> None:
    """AC 5: the wiring hands the capability the class THIS run produces.

    Read at the ``run()`` call, never threaded in from ``run()``'s own frame: a tool may
    call ``switch_model`` mid-run, and the per-call override wins over the constructor's
    ``result_type``. Patching ``run`` leaves ``for_run`` unfired, so the pending slot is
    still readable — which is exactly the value ``for_run`` would have snapshotted.
    """
    agent = ReactAgent(
        config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
        result_type=_Out,
    )
    declared: list[object] = []

    async def _capture(**kwargs: Any) -> SimpleNamespace:
        declared.append(agent._discarded._pending_output_type)
        return SimpleNamespace(output="done")

    with patch.object(agent._pydantic_agent, "run", side_effect=_capture):
        await agent.run("delegate this")
        await agent.run("and this", output_type=_Request)

    assert declared == [_Out, _Request]


# ---------------------------------------------------------------------------
# AC 6 — the precondition is text AND tool calls
# ---------------------------------------------------------------------------


async def test_text_without_tool_calls_untouched() -> None:
    """AC 6: a validated output with no tool call is not discarded upstream — leave it.

    That case is the multi-part merge (story 29-3), and this capability must not touch it.
    """
    _, recorder, capability = await _wired()
    response = _response(TextPart(VALID_OUTPUT), TextPart(SECOND_VALID_OUTPUT))

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(_output_object()), response=response
    )

    assert result is response
    assert _texts(result) == [VALID_OUTPUT, SECOND_VALID_OUTPUT]
    assert _discards(recorder) == []


async def test_tool_calls_without_text_untouched() -> None:
    """AC 6: nothing to strip when the response is tool calls only."""
    _, recorder, capability = await _wired()
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
    _, _, capability = await _wired()
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
    _, recorder, capability = await _wired(budget=2)
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


async def test_budget_is_per_run_because_each_run_gets_its_own_instance() -> None:
    """AC 8: a second run starts with the full budget, and cannot spend the first's.

    ``for_run`` is what makes this true by construction rather than by bookkeeping: the
    counters live on the instance one run holds, so two runs interleaving cannot reset
    each other — the lost bound that a single shared slot allowed.
    """
    _, recorder, mounted = _mount(budget=1)
    first, second = await _bind(mounted), await _bind(mounted)
    request_context = _request_context(_output_object())

    async def strip(capability: DiscardedOutputCapability, run_id: str) -> ModelResponse:
        return await capability.after_model_request(
            _run_context(),
            request_context=request_context,
            response=_response(TextPart(VALID_OUTPUT), _tool_call(), run_id=run_id),
        )

    # Interleaved on purpose: run-2 strips between run-1's two responses.
    one = await strip(first, "run-1")
    two = await strip(second, "run-2")
    one_spent = await strip(first, "run-1")
    two_spent = await strip(second, "run-2")

    assert [_texts(r) for r in (one, two)] == [[], []]
    assert [_texts(r) for r in (one_spent, two_spent)] == [[VALID_OUTPUT], [VALID_OUTPUT]]
    stripped = [e for e in _discards(recorder) if not e.budget_exhausted]
    assert [e.run_id for e in stripped] == ["run-1", "run-2"]


async def test_the_mounted_instance_holds_no_run_state() -> None:
    """AC 8: whatever a run spends, the object the host mounted is untouched.

    The mount is shared across every run of the agent, so a counter left on it is a
    cross-run leak by definition.
    """
    _, _, mounted = _mount(budget=1)
    capability = await _bind(mounted)

    for _ in range(3):
        await capability.after_model_request(
            _run_context(),
            request_context=_request_context(_output_object()),
            response=_response(TextPart(VALID_OUTPUT), _tool_call()),
        )

    assert mounted._strips == 0
    assert mounted._exhausted is False
    assert mounted._output_type is None
    assert mounted._pending_output_type is None


async def test_budget_exhaustion_is_evented() -> None:
    """AC 11: the refusal is recorded, once, on the same event type as a drop.

    The discriminator is ``budget_exhausted``. Content is empty because nothing was
    dropped — the text stayed in the response, and reaches the stream through that
    response's own ``LlmMessageEvent``.
    """
    _, recorder, capability = await _wired(budget=1)
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
    _, recorder, capability = await _wired()
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
    _, recorder, capability = await _wired()
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


def _stack(
    manager: ContextManager, *, strip_first: bool
) -> tuple[list[Any], DiscardedOutputCapability]:
    """The two capabilities, in the requested mount order, plus the strip to declare on."""
    strip = DiscardedOutputCapability(context=manager)
    persist = EventSourcingCapability(context=manager)
    return ([strip, persist] if strip_first else [persist, strip]), strip


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
    capabilities, strip = _stack(manager, strip_first=strip_first)
    agent = _agent(capabilities, [_co_emitting_response()])
    strip.expect_output_type(_Out)
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
