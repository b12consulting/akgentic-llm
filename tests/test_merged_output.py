"""Tests for ``DiscardedOutputCapability``'s merge branch — story 29-3.

The strip (story 29-2) and the merge are the same capability on the same anchor, split on
one predicate — ``response.tool_calls``. The strip's specs live in
``test_discarded_output.py``; this file owns the merge and the two paths' exclusivity.

Two harnesses, as there, and for the same reason. Most specs drive ``after_model_request``
**directly** with a hand-built ``ModelRequestContext``, which pins one gate at a time. The
specs that are *about* the anchor — that the merge lands before history, that mount order
does not matter, that restore rebuilds the merged history, that a type without ``merge``
still takes pydantic-ai's retry path — drive a **bare pydantic-ai ``Agent``**, because only
a real run exercises the concatenation the merge exists to prevent.

**The fixtures are local, and must be.** The reference implementation of the ``merge``
protocol is ``StructuredOutput`` in ``akgentic-agent``: another package, and it has no
``merge`` yet. A spec written against it could not pass, and importing it here would couple
``akgentic-llm`` to a sibling — the coupling AC 7 exists to keep out.

``asyncio_mode = "auto"`` — plain ``async def`` tests, no ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import ast
import logging
import pathlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.messages import (
    BinaryContent,
    FilePart,
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    RetryPromptPart,
    SpeechPart,
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
from akgentic.llm.capabilities import discarded_output
from akgentic.llm.event import LlmMessageEvent, LlmOutputDiscardedEvent

# ---------------------------------------------------------------------------
# Fixtures of the domain: output classes that declare the merge protocol, and ones that
# do not. The protocol, pinned:
#
#     @classmethod
#     def merge(cls, outputs: Sequence[Self]) -> Self: ...
#
# It receives already-validated instances in emission order and returns one instance of
# the same class.
# ---------------------------------------------------------------------------


class _Request(BaseModel):
    """One routed message, mirroring the shape the recorded run emitted."""

    recipient: str
    message: str


class _Plain(BaseModel):
    """An output class that declares no ``merge`` — the production case until the
    ``akgentic-agent`` story lands."""

    messages: list[_Request]


class _Merging(_Plain):
    """The same shape, declaring the protocol: concatenate the lists, in order."""

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        return cls(messages=[message for output in outputs for message in output.messages])


class _Recording(_Plain):
    """Records what ``merge`` was handed, so emission order can be asserted on."""

    handed: ClassVar[list[list[_Request]]] = []

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        cls.handed.append([message for output in outputs for message in output.messages])
        return cls(messages=[message for output in outputs for message in output.messages])


class _Raising(_Plain):
    """A ``merge`` that raises. A hook firing on every response must not end the run."""

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        raise RuntimeError("merge exploded")


class _WrongTyped(_Plain):
    """A ``merge`` returning something that is not an instance of the class."""

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        return "not an instance of this class"  # type: ignore[return-value]


class _Aliased(BaseModel):
    """A class whose field carries an alias, so serialising by field name would produce
    text the class itself refuses to validate."""

    model_config = ConfigDict(populate_by_name=False)

    messages: list[_Request] = Field(alias="msgs")

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        return cls(msgs=[message for output in outputs for message in output.messages])


FIRST = '{"messages": [{"recipient": "@Assistant", "message": "search the web"}]}'
SECOND = '{"messages": [{"recipient": "@Expert", "message": "then review it"}]}'
PROSE = "I should ask the assistant to look this up before answering."
PARTIAL_JSON = '{"messages":[{"message_ty'
WRONG_SHAPE = '{"unexpected": 1}'
ALIASED_FIRST = '{"msgs": [{"recipient": "@Assistant", "message": "search the web"}]}'
ALIASED_SECOND = '{"msgs": [{"recipient": "@Expert", "message": "then review it"}]}'


# ---------------------------------------------------------------------------
# Harness — the hook, driven directly
# ---------------------------------------------------------------------------


def _output_object() -> OutputObjectDefinition:
    """The run's own output schema. Only its *presence* is read by the hook."""
    return OutputObjectDefinition(json_schema=_Plain.model_json_schema(), name="Out")


def _request_context(
    output_object: OutputObjectDefinition | None = None,
) -> ModelRequestContext:
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
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    return manager, recorder, DiscardedOutputCapability(context=manager, **kwargs)


async def _wired(
    output_type: object = _Merging, **kwargs: Any
) -> tuple[ContextManager, _Recorder, DiscardedOutputCapability]:
    """A mounted capability already bound to one run, as a real run's hooks see it."""
    manager, recorder, mounted = _mount(**kwargs)
    mounted.expect_output_type(output_type)
    bound = await mounted.for_run(_run_context())
    assert isinstance(bound, DiscardedOutputCapability)
    return manager, recorder, bound


async def _hook(
    capability: DiscardedOutputCapability,
    response: ModelResponse,
    *,
    output_object: OutputObjectDefinition | None = None,
) -> ModelResponse:
    """Call the anchor the way pydantic-ai does, defaulting to a declared output schema."""
    return await capability.after_model_request(
        _run_context(),
        request_context=_request_context(
            _output_object() if output_object is None else output_object
        ),
        response=response,
    )


def _texts(response: ModelResponse) -> list[str]:
    return [p.content for p in response.parts if isinstance(p, TextPart)]


def _discards(recorder: _Recorder) -> list[LlmOutputDiscardedEvent]:
    return [e for e in recorder.events if isinstance(e, LlmOutputDiscardedEvent)]


def _two_valid_parts() -> ModelResponse:
    """The measured shape: two complete outputs, no tool call, one response."""
    return _response(TextPart(FIRST), TextPart(SECOND))


# ---------------------------------------------------------------------------
# AC 2 / AC 8 — the merge itself
# ---------------------------------------------------------------------------


async def test_two_validating_parts_merge_into_one() -> None:
    """AC 2: two complete outputs become one text part carrying the merged output.

    This is the whole feature: upstream would have concatenated these into ``{...}{...}``
    and spent an output retry on the ``json_invalid`` that follows.
    """
    _, _, capability = await _wired()

    result = await _hook(capability, _two_valid_parts())

    assert len(_texts(result)) == 1
    assert _Merging.model_validate_json(_texts(result)[0]).messages == [
        _Request(recipient="@Assistant", message="search the web"),
        _Request(recipient="@Expert", message="then review it"),
    ]


async def test_merged_part_keeps_the_first_position() -> None:
    """AC 8: the merged part sits where the first text part sat, not appended at the end."""
    thinking = ThinkingPart(content="weighing options")
    _, _, capability = await _wired()
    response = _response(TextPart(FIRST), thinking, TextPart(SECOND))

    result = await _hook(capability, response)

    assert [type(p).__name__ for p in result.parts] == ["TextPart", "ThinkingPart"]
    assert result.parts[1] is thinking


async def test_the_merged_part_carries_the_first_parts_other_fields() -> None:
    """Golden Rule 12: the merged part is *derived* from the first text part, never rebuilt.

    ``_with_merged`` uses ``dataclasses.replace``, so every field it does not name survives —
    ``id`` and ``provider_name`` today, and whatever upstream adds next. Without this spec the
    enumerated form ``TextPart(content=...)`` passes the entire suite green, because every
    other spec here builds its parts from content alone: it is correct on the day it is
    written and silently drops the provider's part identity out of history and out of the
    event stream. Pinning the fields that exist now is what makes that edit impossible, which
    is the protection the rule is after.
    """
    _, _, capability = await _wired()
    first = TextPart(FIRST, id="part-1", provider_name="openai")
    response = _response(first, TextPart(SECOND))

    result = await _hook(capability, response)

    merged = next(p for p in result.parts if isinstance(p, TextPart))
    assert merged.id == "part-1"
    assert merged.provider_name == "openai"
    assert merged.content != FIRST  # genuinely the merged part, not the original carried over
    assert len(_Merging.model_validate_json(merged.content).messages) == 2


async def test_non_text_parts_survive_the_merge() -> None:
    """AC 8: thinking parts and files are carried through untouched, in order."""
    thinking = ThinkingPart(content="weighing options")
    file_part = FilePart(content=BinaryContent(data=b"x", media_type="text/plain"))
    _, _, capability = await _wired()
    response = _response(thinking, TextPart(FIRST), file_part, TextPart(SECOND))

    result = await _hook(capability, response)

    assert result.parts[0] is thinking
    assert result.parts[2] is file_part
    assert [type(p).__name__ for p in result.parts] == ["ThinkingPart", "TextPart", "FilePart"]


async def test_merge_receives_parts_in_emission_order() -> None:
    """AC 8: ``merge`` is handed the validated instances in the order the model emitted them."""
    _Recording.handed.clear()
    _, _, capability = await _wired(_Recording)

    await _hook(capability, _two_valid_parts())

    assert _Recording.handed == [
        [
            _Request(recipient="@Assistant", message="search the web"),
            _Request(recipient="@Expert", message="then review it"),
        ]
    ]


async def test_three_parts_all_reach_merge() -> None:
    """AC 6 / AC 8: the merge is over every text part, not the first two."""
    third = '{"messages": [{"recipient": "@Human", "message": "done"}]}'
    _, _, capability = await _wired()

    result = await _hook(capability, _response(TextPart(FIRST), TextPart(SECOND), TextPart(third)))

    assert len(_Merging.model_validate_json(_texts(result)[0]).messages) == 3


# ---------------------------------------------------------------------------
# AC 6 — merge only when ALL parts validate
#
# A partial merge would silently drop whatever did not parse: data loss dressed as a fix.
# ---------------------------------------------------------------------------


async def test_one_unparseable_part_blocks_the_merge() -> None:
    """AC 6: truncated JSON beside a valid output ⇒ the response comes back untouched."""
    _, recorder, capability = await _wired()
    response = _response(TextPart(FIRST), TextPart(PARTIAL_JSON))

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, PARTIAL_JSON]
    assert recorder.events == []


async def test_one_schema_mismatched_part_blocks_the_merge() -> None:
    """AC 6: valid JSON of the wrong shape blocks it just as a parse failure does."""
    _, recorder, capability = await _wired()
    response = _response(TextPart(FIRST), TextPart(WRONG_SHAPE))

    result = await _hook(capability, response)

    assert result is response
    assert list(result.parts) == list(response.parts)
    assert recorder.events == []


async def test_prose_beside_an_output_blocks_the_merge() -> None:
    """AC 6: narration is not an output — merging around it would delete it."""
    _, _, capability = await _wired()
    response = _response(TextPart(PROSE), TextPart(FIRST))

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [PROSE, FIRST]


# ---------------------------------------------------------------------------
# AC 7 — the gate is "declares merge", and nothing else
# ---------------------------------------------------------------------------


async def test_a_type_without_merge_is_untouched(caplog: pytest.LogCaptureFixture) -> None:
    """AC 7: the branch that runs in production until the sibling story lands.

    **The log assertion is the part that has teeth.** Deleting the declaration gate does
    *not* change the returned response: the missing attribute raises out of ``getattr``
    inside the guard AC 9 put around ``merge``, which swallows it and returns the response
    unchanged — so response-shape assertions alone stay green over a deleted gate, and the
    guard would be decoration. What deleting it *does* change is that every multi-part
    response of every run whose class declares no ``merge`` — today, every one of them —
    logs an ``ERROR`` with a stack trace. That is the observable difference, so that is
    what is asserted.
    """
    _, recorder, capability = await _wired(_Plain)
    response = _two_valid_parts()

    with caplog.at_level(logging.DEBUG, logger=discarded_output.__name__):
        result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, SECOND]
    assert recorder.events == []
    # Not even the debug line: `merge` was never attempted.
    assert [r for r in caplog.records if r.name == discarded_output.__name__] == []


def test_the_module_imports_no_sibling_akgentic_package() -> None:
    """AC 7: the gate is duck-typed, so no sibling package is named anywhere in the module.

    Importing ``StructuredOutput``, or testing for its name, would couple ``akgentic-llm``
    to ``akgentic-agent`` and break the module boundary. Relative imports are allowed up to
    two levels — ``akgentic.llm`` — and no further, because a third would reach the shared
    namespace root and from there a sibling.
    """
    path = pathlib.Path(discarded_output.__file__)
    source = path.read_text()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level <= 2, f"relative import escapes akgentic.llm: level {node.level}"
                continue
            module = node.module or ""
            assert not module.startswith("akgentic.") or module.startswith("akgentic.llm"), module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("akgentic.") or alias.name.startswith(
                    "akgentic.llm"
                ), alias.name

    assert "StructuredOutput" not in source


# ---------------------------------------------------------------------------
# AC 8 — preconditions and shape
# ---------------------------------------------------------------------------


async def test_single_text_part_is_untouched() -> None:
    """AC 8: one part is never concatenated, so re-serialising it would change the record
    for nothing."""
    _, _, capability = await _wired()
    response = _response(TextPart(FIRST))

    result = await _hook(capability, response)

    assert result is response


async def test_no_text_parts_is_untouched() -> None:
    """AC 8: a response with no text at all has nothing to merge."""
    thinking = ThinkingPart(content="weighing options")
    _, _, capability = await _wired()
    response = _response(thinking)

    result = await _hook(capability, response)

    assert result is response


async def test_no_output_schema_is_untouched() -> None:
    """AC 8: ``output_object is None`` means the output does not live in a text part."""
    _, _, capability = await _wired()
    response = _two_valid_parts()

    result = await capability.after_model_request(
        _run_context(), request_context=_request_context(None), response=response
    )

    assert result is response


async def test_undeclared_output_type_is_untouched() -> None:
    """AC 8: a host that declared nothing gets a no-op — the fail-closed direction."""
    manager = ContextManager()
    capability = DiscardedOutputCapability(context=manager)
    bound = await capability.for_run(_run_context())
    assert isinstance(bound, DiscardedOutputCapability)
    response = _two_valid_parts()

    result = await _hook(bound, response)

    assert result is response


async def test_str_result_type_is_untouched() -> None:
    """AC 8: ``ReactAgent.result_type`` defaults to ``str``; a text part is not an instance
    of anything then, and the merge must not fire on ordinary prose responses."""
    _, _, capability = await _wired(str)
    response = _two_valid_parts()

    result = await _hook(capability, response)

    assert result is response


async def test_speech_part_beside_text_blocks_the_merge() -> None:
    """AC 8: ``SpeechPart`` adds its transcript to the very same ``text`` accumulator.

    Merging around it would leave the merged object concatenated beside the transcript —
    still invalid JSON — with the original parts destroyed for nothing.
    """
    _, _, capability = await _wired()
    response = _response(
        TextPart(FIRST), TextPart(SECOND), SpeechPart(speaker="assistant", transcript="hello")
    )

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, SECOND]


async def test_native_tool_call_part_blocks_the_merge() -> None:
    """AC 8: ``NativeToolCallPart`` *resets* the accumulator, so text before it is dropped
    upstream. It is not a ``ToolCallPart``, so ``response.tool_calls`` does not see it —
    this gate is the only thing standing between it and a destructive merge.
    """
    _, _, capability = await _wired()
    response = _response(
        TextPart(FIRST), TextPart(SECOND), NativeToolCallPart(tool_name="web_search", args={})
    )

    assert response.tool_calls == []

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, SECOND]


# ---------------------------------------------------------------------------
# AC 3 — mutually exclusive with the strip, structurally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["early", "graceful", "exhaustive"])
async def test_tool_calls_present_never_merges(strategy: str) -> None:
    """AC 3: a response carrying a tool call takes the strip path, never the merge.

    Asserted under every strategy, which is what makes it a real guard: under ``early`` and
    ``graceful`` the strip is itself a no-op, so a merge leaking onto this response is the
    only thing that could change it — and it would change it by deleting both parts.
    """
    _, _, capability = await _wired(end_strategy=strategy)
    response = _response(TextPart(FIRST), TextPart(SECOND), _tool_call())

    result = await _hook(capability, response)

    if strategy == "exhaustive":
        assert _texts(result) == []  # both validated, so both were stripped
    else:
        assert result is response
        assert _texts(result) == [FIRST, SECOND]


async def test_no_tool_calls_never_strips_and_emits_no_discard_event() -> None:
    """AC 3 / AC 11: nothing on the merge path is recorded as a discard."""
    _, recorder, capability = await _wired()

    result = await _hook(capability, _two_valid_parts())

    assert len(_texts(result)) == 1
    assert _discards(recorder) == []


async def test_merge_emits_no_discard_event() -> None:
    """AC 11: ``LlmOutputDiscardedEvent`` means "this content was thrown away". Nothing is
    thrown away here — the parts' content survives through ``merge`` — so reusing it would
    be a lie, and no new event type is added either."""
    _, recorder, capability = await _wired()

    await _hook(capability, _two_valid_parts())

    assert recorder.events == []


async def test_strip_and_merge_are_exclusive() -> None:
    """AC 3: both shapes through one capability instance; each took exactly one path."""
    _, recorder, capability = await _wired()

    stripped = await _hook(capability, _response(TextPart(FIRST), _tool_call()))
    merged = await _hook(capability, _two_valid_parts())

    assert _texts(stripped) == []  # stripped, not merged
    assert stripped.tool_calls
    assert len(_texts(merged)) == 1  # merged, not stripped
    assert [e.discarded_content for e in _discards(recorder)] == [(FIRST,)]


# ---------------------------------------------------------------------------
# AC 4 — the merge runs under all three end strategies
#
# The upstream ``text += part.content`` accumulation is unconditional; the
# ``end_strategy == 'early'`` branch sits inside ``if tool_calls:``. With no tool calls all
# three strategies concatenate, so gating the merge on the strategy would silently disable
# the fix under two of them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["early", "graceful", "exhaustive"])
async def test_merge_under_every_end_strategy(strategy: str) -> None:
    """AC 4: identical merge under ``early``, ``graceful`` and ``exhaustive``."""
    _, _, capability = await _wired(end_strategy=strategy)

    result = await _hook(capability, _two_valid_parts())

    assert len(_texts(result)) == 1
    assert len(_Merging.model_validate_json(_texts(result)[0]).messages) == 2


# ---------------------------------------------------------------------------
# AC 5 — the merge is not budgeted, and the strip's budget does not reach it
#
# The budget bounds the strip's re-derivation loop: each strip buys an extra model request.
# A merge buys none — it removes a retry, is idempotent per response, and costs nothing.
# ---------------------------------------------------------------------------


async def test_merge_survives_an_exhausted_strip_budget() -> None:
    """AC 5: with the budget fully spent, every subsequent multi-part response still merges."""
    _, _, capability = await _wired(budget=1)

    for _ in range(3):
        await _hook(capability, _response(TextPart(FIRST), _tool_call()))
    assert capability._exhausted

    result = await _hook(capability, _two_valid_parts())

    assert len(_texts(result)) == 1


async def test_merge_does_not_consume_the_strip_budget() -> None:
    """AC 5: merging never increments ``_strips`` nor sets ``_exhausted``."""
    _, _, capability = await _wired(budget=1)

    for _ in range(4):
        await _hook(capability, _two_valid_parts())

    assert capability._strips == 0
    assert capability._exhausted is False


# ---------------------------------------------------------------------------
# AC 9 — a hook that fires on every response never ends the run
# ---------------------------------------------------------------------------


async def test_a_raising_merge_leaves_the_response_unchanged() -> None:
    """AC 9: the exception is logged and swallowed; the run continues on the retry path."""
    _, recorder, capability = await _wired(_Raising)
    response = _two_valid_parts()

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, SECOND]
    assert recorder.events == []


async def test_a_wrong_typed_merge_result_is_refused() -> None:
    """AC 9: a ``merge`` returning something else must not be serialised into history."""
    _, _, capability = await _wired(_WrongTyped)
    response = _two_valid_parts()

    result = await _hook(capability, response)

    assert result is response
    assert _texts(result) == [FIRST, SECOND]


async def test_merged_text_round_trips() -> None:
    """AC 9: the text this capability writes is text the output class accepts.

    Serialising with different settings than the class validates under produces text that
    fails the very validation this capability exists to make succeed. ``_Aliased`` is that
    case made concrete: dumped by field name it reads ``{"messages": ...}``, which the
    class — ``populate_by_name=False`` — refuses.
    """
    _, _, capability = await _wired(_Aliased)
    response = _response(TextPart(ALIASED_FIRST), TextPart(ALIASED_SECOND))

    result = await _hook(capability, response)

    assert len(_texts(result)) == 1
    assert len(_Aliased.model_validate_json(_texts(result)[0]).messages) == 2


async def test_merged_text_round_trips_for_the_plain_shape() -> None:
    """AC 9: the same round trip on the ordinary, alias-free shape."""
    _, _, capability = await _wired()

    result = await _hook(capability, _two_valid_parts())

    assert _Merging.model_validate_json(_texts(result)[0])


# ---------------------------------------------------------------------------
# Live-run harness: the anchor, mount order, restore, and the untouched retry path
# ---------------------------------------------------------------------------


def _scripted_model(scripted: list[ModelResponse]) -> FunctionModel:
    """Replays ``scripted`` in order, then answers with a single valid output."""
    queue = list(scripted)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if queue:
            return queue.pop(0)
        return ModelResponse(parts=[TextPart(FIRST)], usage=RequestUsage(input_tokens=5))

    return FunctionModel(model_fn)


def _multi_part_response() -> ModelResponse:
    """The measured shape: two complete outputs, no tool call, one response."""
    return ModelResponse(
        parts=[TextPart(FIRST), TextPart(SECOND)],
        usage=RequestUsage(input_tokens=7, output_tokens=41),
    )


def _agent(capabilities: list[Any], output_type: type[BaseModel]) -> Agent[None, Any]:
    """A bare ``Agent`` — never a ``ReactAgent``: the capability must be provable on any
    agent, and a bare one declares its output class exactly as ``ReactAgent`` does."""
    return Agent(
        model=_scripted_model([_multi_part_response()]),
        output_type=PromptedOutput(output_type),
        end_strategy="exhaustive",
        capabilities=capabilities,
    )


def _stack(
    manager: ContextManager, *, merge_first: bool
) -> tuple[list[Any], DiscardedOutputCapability]:
    merge = DiscardedOutputCapability(context=manager)
    persist = EventSourcingCapability(context=manager)
    return ([merge, persist] if merge_first else [persist, merge]), merge


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


async def _run_once(
    *, merge_first: bool = True, output_type: type[BaseModel] = _Merging
) -> tuple[ContextManager, _Recorder]:
    manager = ContextManager()
    recorder = _Recorder()
    manager.subscribe(recorder)
    capabilities, merge = _stack(manager, merge_first=merge_first)
    agent = _agent(capabilities, output_type)
    merge.expect_output_type(output_type)
    await agent.run("delegate this")
    return manager, recorder


async def test_merged_response_is_what_reaches_history() -> None:
    """AC 2: history holds exactly one text part, carrying the merged output."""
    manager, _ = await _run_once()

    first = _responses(manager.messages)[0]
    assert len(_texts(first)) == 1
    assert len(_Merging.model_validate_json(_texts(first)[0]).messages) == 2
    assert FIRST not in _texts(first)


async def test_merged_response_is_what_is_evented() -> None:
    """AC 2: ``LlmMessageEvent`` carries the merged response, never the raw one."""
    _, recorder = await _run_once()

    evented = [
        e.message
        for e in recorder.events
        if isinstance(e, LlmMessageEvent) and isinstance(e.message, ModelResponse)
    ]
    assert len(_texts(evented[0])) == 1
    assert len(_Merging.model_validate_json(_texts(evented[0])[0]).messages) == 2


@pytest.mark.parametrize("merge_first", [True, False], ids=["before", "after"])
async def test_merge_is_order_independent_against_event_sourcing(merge_first: bool) -> None:
    """AC 1: mounting before or after persistence gives the same history and events.

    The anchor runs before ``_append_response``, so there is no sweep for the merge to race
    and no order for a ``get_ordering()`` re-sort to break.
    """
    manager, recorder = await _run_once(merge_first=merge_first)
    reference_manager, reference_recorder = await _run_once(merge_first=not merge_first)

    assert _shape(manager.messages) == _shape(reference_manager.messages)
    assert [type(e).__name__ for e in recorder.events] == [
        type(e).__name__ for e in reference_recorder.events
    ]


async def test_restore_reproduces_merged_history() -> None:
    """AC 2: replaying the emitted stream rebuilds the merged history, not the raw one."""
    manager, recorder = await _run_once()

    agent = ReactAgent(
        config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    )
    agent.restore_context([SimpleNamespace(event=e) for e in recorder.events])

    assert _shape(agent.context.messages) == _shape(manager.messages)
    assert FIRST not in [t for r in _responses(agent.context.messages) for t in _texts(r)]


async def test_the_merge_ends_the_run_without_an_output_retry() -> None:
    """AC 2: the point of the feature — one model request, no retry prompt in history."""
    manager, _ = await _run_once()

    assert len(_responses(manager.messages)) == 1
    assert not [p for m in manager.messages for p in m.parts if isinstance(p, RetryPromptPart)]


async def test_a_type_without_merge_still_takes_the_retry_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC 7: the production branch until the sibling story lands, proven equal to today.

    A type without ``merge`` must leave the run byte-identical to one with the capability
    absent altogether: the concatenation still happens, ``json_invalid`` still fires, and
    pydantic-ai still spends an output retry. "No behaviour change" is the claim in this
    story most likely to be untrue, so it is asserted against a control run rather than
    inferred from "no crash".
    """
    with caplog.at_level(logging.DEBUG, logger=discarded_output.__name__):
        manager, recorder = await _run_once(output_type=_Plain)

    control_manager = ContextManager()
    control_recorder = _Recorder()
    control_manager.subscribe(control_recorder)
    control = _agent([EventSourcingCapability(context=control_manager)], _Plain)
    await control.run("delegate this")

    assert _shape(manager.messages) == _shape(control_manager.messages)
    assert [type(e).__name__ for e in recorder.events] == [
        type(e).__name__ for e in control_recorder.events
    ]
    retries = [p for m in manager.messages for p in m.parts if isinstance(p, RetryPromptPart)]
    assert len(retries) == 1
    assert "Invalid JSON" in str(retries[0].content)
    assert [r for r in caplog.records if r.name == discarded_output.__name__] == []
