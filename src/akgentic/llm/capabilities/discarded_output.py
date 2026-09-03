"""``DiscardedOutputCapability`` — strip from a response what pydantic-ai will discard.

See the package docstring for the hook anchors and composition order; this module is the
first entry on the ``after_model_request`` anchor and explains why it uses that one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, TypeGuard, is_typeddict

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_ai import EndStrategy
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from ..context import ContextManager

logger = logging.getLogger(__name__)

DEFAULT_STRIP_BUDGET = 3
"""Strips allowed per run before the capability stands down for the rest of it.

Each strip costs one extra model request: the discarded intent is re-derived on the
next iteration, which is the whole point. Unbounded, a model that co-emits on *every*
iteration is stripped on every iteration and re-derives until pydantic-ai's
``run_request_limit`` — the 1.1M-token single-run shape already on record here.

Three, because the run that motivated this capability co-emitted twice in one turn (at
event idx 19 and idx 41), so two is the observed need and a budget of exactly two would
sit on the boundary with no headroom. Three covers it with one spare and still caps the
extra requests at a small constant, far below any request limit.
"""


def _is_decidable(output_type: object) -> TypeGuard[type[Any]]:
    """Whether validating ``output_type`` here answers the question the graph asks.

    It does for a pydantic model, a dataclass and a ``TypedDict``, and for nothing else.
    That line is not a preference — it is upstream's, drawn in ``_output.py`` where
    ``ObjectOutputProcessor`` builds the validator that decides the real output. A
    *model-like* type generates ``{"type": "object"}`` and is validated as itself; every
    other type is wrapped in a ``{"response": ...}`` envelope and validated as
    ``TypedDict{'response': T | Json[T]}``. So for a bare ``str``, ``int``, ``list[X]`` or
    scalar union, the text the model actually emits is the envelope and the naked type is
    a *different question* — one that answers True on payloads the graph never treated as
    output. ``TypeAdapter(str).validate_json`` accepting every quoted string is the sharp
    end of that, and a wrong True here deletes the model's narration.

    Mirrored rather than imported: ``pydantic_ai._utils.is_model_like`` is private and this
    package's CI floats across pydantic-ai minors. If upstream ever stops enveloping, this
    goes from correct to merely conservative — a no-op, never a bad strip.
    """
    return isinstance(output_type, type) and (
        issubclass(output_type, BaseModel)
        or is_dataclass(output_type)
        or is_typeddict(output_type)
        or getattr(output_type, "__is_model_like__", False)
    )


def _validates(content: str, validator: TypeAdapter[Any]) -> bool:
    """Whether ``content`` is an instance of the run's own output type.

    **The verdict, not an approximation of it.** ``TypeAdapter.validate_json`` is the same
    call pydantic-ai's own output processor makes on the same text against the same type
    (``_output.py`` — ``self.validator.validate_json(...)``, over a ``TypeAdapter`` built
    from the output type), so "this text is the run's structured output" is answered by the
    library that decides it rather than by a re-implementation of JSON Schema. A
    ``TypeAdapter`` and not ``model_validate_json``: the output type is only *usually* a
    ``BaseModel`` — a dataclass and a ``TypedDict`` are equally legal and have no such
    method, and pydantic-ai validates all three the same way.

    **No ``strict=True``.** ``NativeOutput(output_type, strict=True)`` in ``providers.py``
    sets ``OutputObjectDefinition.strict``, which asks the *provider* for schema-constrained
    decoding. It is a different axis from pydantic's ``strict=``, which forbids coercion,
    and it never reaches the validator: what pydantic-ai runs is
    ``validator.validate_json(data, ...)`` with no strict argument — lax. Validating
    strictly here would make this capability stricter than the run it is shadowing and cut
    its hit rate on exactly the coercion-heavy payloads models emit (``"41"`` for an ``int``
    field), leaving them in history. This is the one knob that changes behaviour, so it is
    stated rather than left to be "fixed" later.

    **Known gap: markdown fences.** pydantic-ai runs ``strip_markdown_fences`` over text
    output *before* validating (``_output.py:934``); this does not. A prompted-mode model
    that fences its delegation therefore emits something pydantic-ai accepts and discards,
    and this keeps. "Matches exactly what pydantic-ai accepts" is not literally true for
    that case. Left deliberately: the helper is private, and the gap is under-strip, which
    costs the fix rather than data.

    A type whose validators raise something other than ``ValidationError`` must not end the
    run from a hook that fires on every model response, so anything else is logged and read
    as "does not validate" — the direction that keeps the part.
    """
    try:
        validator.validate_json(content)
    except ValidationError:
        return False
    except Exception:
        logger.exception("Validating a text part raised; leaving the part in place")
        return False
    return True


@dataclass
class DiscardedOutputCapability(AbstractCapability[Any]):
    """Remove the structured output ``CallToolsNode`` is about to discard, before history.

    When a model co-emits a valid structured output *and* a tool call in one response,
    ``end_strategy='exhaustive'`` drops the output, runs the tool and loops — but appends
    the response to ``message_history`` verbatim. The next request therefore shows the
    model its own words ("I've asked @Assistant to research…"), it concludes it already
    delegated, and it returns an empty output that routes nothing. The dropped message
    becomes a permanent one, because a model will not re-derive an intent it believes it
    has already acted on. Stripping the text before the append is what lets it re-derive.

    **The anchor is ``after_model_request``, and that is the whole design.** The hook runs
    between the model response and ``ModelRequestNode._append_response``, so the response
    this returns is the one that reaches history and the one ``EventSourcingCapability``
    later persists. The node-hook alternatives (``before_node_run(CallToolsNode)``,
    ``after_node_run(ModelRequestNode)``) can only mutate a response that is *already* in
    history, which would have to be sequenced against the persistence sweep — an ordering
    that is invisible in a diff and re-sortable by any capability declaring
    ``get_ordering()``. This class declares no ``get_ordering()`` and implements no node
    hook and no ``wrap_run``; mounting it anywhere in the stack gives the same history and
    the same event sequence.

    **It strips only what validates against the run's own output type, and only where the
    request put the output in a text part.** Never on ``isinstance(part, TextPart)``: a
    tool call sitting beside plain narration is ordinary, and stripping on the part type
    alone would silently delete the model's reasoning. Parse failure, type mismatch, and a
    request with no ``output_object`` all mean "keep" — see ``_discardable`` for why that
    last gate is the one that matters most.

    **The host declares the output type, once per run.** ``expect_output_type`` is called
    with the type *this* run will produce, immediately before the run starts; ``for_run``
    builds a validator from it onto the per-run instance and clears the slot. A host that
    declares nothing — a bare ``Agent`` with this capability mounted by hand — gets a
    no-op, which is the same fail-closed direction as an unparseable part. The type is
    never guessed from ``ModelRequestContext``: what the request carries is a JSON Schema,
    and deciding conformance against a schema by hand is exactly what this capability used
    to do and stopped doing.

    **Audit only, never annotation.** The stripped response is what is evented, so
    ``restore_context`` rebuilds the stripped history and the defect does not return on
    the next resume. The discarded text is preserved in ``LlmOutputDiscardedEvent``
    instead, which is what reconciles the recorded content against the
    ``LlmUsageEvent.output_tokens`` the provider billed for the whole generation.
    """

    context: ContextManager
    """The context manager every strip is recorded through, as ``EventSourcingCapability``
    holds one. Emission goes through its public recorders and never through observers of
    this capability's own — one stream, one emission path."""

    end_strategy: EndStrategy = "exhaustive"
    """The run's tool-execution strategy, read from configuration at mount time.

    Never inferred from the response. ``'exhaustive'`` is the only value that discards
    co-emitted text *and* was measured; under ``'early'`` the text is the live result and
    stripping it would destroy it, and ``'graceful'`` takes the same discard branch but
    was never measured, so extending to it is a decision rather than a freebie.
    """

    budget: int = DEFAULT_STRIP_BUDGET
    """Strips allowed per run. See ``DEFAULT_STRIP_BUDGET`` for why three."""

    _pending_output_type: type[Any] | None = field(default=None, init=False)
    """The type the next run will produce, parked here by ``expect_output_type``.

    Lives on the *shared* instance because that is the only object the host holds; it is
    read and cleared by ``for_run`` and never by a hook, so no hook ever consults a slot
    another run can be writing.
    """

    _validator: TypeAdapter[Any] | None = field(default=None, init=False)
    """This run's validator, built by ``for_run`` from the declared type. ``None`` on the
    shared instance and on any run whose host declared nothing decidable — both no-ops.
    Built once per run rather than per response: constructing a ``TypeAdapter`` compiles a
    schema, and this hook fires on every model request."""

    _strips: int = field(default=0, init=False)
    """Strips already made in this run. Per-run because ``for_run`` hands each run its own
    instance; on the shared one it stays at zero forever."""

    _exhausted: bool = field(default=False, init=False)
    """Whether this run's budget has already refused a strip. Once set, the run's
    remaining responses pass through without the budget being tested again."""

    def expect_output_type(self, output_type: object) -> None:
        """Declare the type the next run's structured output will be validated against.

        Called by the host immediately before it starts the run, and at that point
        deliberately: a value resolved a frame earlier goes stale when a tool calls
        ``switch_model`` mid-run, which is the same reason ``ReactAgent`` resolves the
        model and the effective output type at the ``run()`` call itself.

        A type ``_is_decidable`` refuses parks ``None`` and makes the run a no-op — ``str``
        (this package's default output type) and ``None`` being the ordinary cases, since a
        run with no structured output has nothing for a part to be an instance of.

        Args:
            output_type: The run's effective output type, wrapper-free.
        """
        self._pending_output_type = output_type if _is_decidable(output_type) else None

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Hand this run its own instance, carrying its validator and its own budget.

        Upstream's sanctioned per-run isolation, and load-bearing twice over. The budget
        was previously a single slot on the shared instance, so interleaved runs reset
        each other's counters and the bound was lost; the output type on a shared slot
        would be worse — a run could validate against another run's schema and strip a
        part that is not its output at all. Neither is reachable once the state lives on
        the instance only this run holds.

        Clearing the pending slot is what makes an undeclared run a no-op rather than a
        run validated against whatever the previous one declared. Building the validator
        here rather than in the hook keeps schema compilation off the per-response path and
        makes a type pydantic cannot build an adapter for a no-op for the whole run instead
        of a log line per model request.

        Upstream notes that under durable execution per-run state must be derivable from
        ``ctx``. A captured type does not satisfy that — ``RunContext`` carries no output
        schema — and this package does not use durable execution. It is a constraint on
        supporting it later, not a defect today: a durable host would have to declare the
        type through its own serialised state.

        Args:
            ctx: This run's context. Unread — the type comes from the host.

        Returns:
            A copy of this capability bound to this run.
        """
        bound = replace(self)
        bound._validator = self._build_validator(self._pending_output_type)
        self._pending_output_type = None
        return bound

    @staticmethod
    def _build_validator(output_type: type[Any] | None) -> TypeAdapter[Any] | None:
        """A validator for ``output_type``, or None when there is nothing to validate.

        A type pydantic cannot build an adapter for is a no-op rather than a run-ending
        error: this is an audit-only capability and must never be the reason a run fails.
        """
        if output_type is None:
            return None
        try:
            return TypeAdapter(output_type)
        except Exception:
            logger.exception(
                "Could not build a validator for %s; leaving this run's output in place",
                output_type,
            )
            return None

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Return ``response`` with the about-to-be-discarded output removed, or unchanged.

        Unchanged unless every gate passes: the strategy is ``'exhaustive'``, the host
        declared a decidable output type for this run, this run's budget is unspent, the
        response carries at least one ``ToolCallPart``, the request declares an output
        schema, and at least one ``TextPart`` validates against the type.
        """
        if self.end_strategy != "exhaustive" or self._validator is None or self._exhausted:
            return response

        discardable = self._discardable(request_context, response, self._validator)
        if not discardable:
            return response

        if self._strips >= self.budget:
            self._exhausted = True
            logger.warning(
                "Discard budget of %d spent for run %s; leaving output in place",
                self.budget,
                response.run_id,
            )
            self.context.record_discard_budget_exhausted(response.run_id)
            return response

        self._strips += 1
        self.context.record_discarded_output(
            response.run_id, [part.content for part in discardable]
        )
        return self._without(response, discardable)

    @staticmethod
    def _discardable(
        request_context: ModelRequestContext,
        response: ModelResponse,
        validator: TypeAdapter[Any],
    ) -> list[TextPart]:
        """The text parts this response would lose to the discard branch, in emission order.

        Empty unless the response carries a tool call *and* the request declared an output
        schema. Text with no tool call is not discarded by the graph at all — that case is
        the multi-part merge, and this capability must leave it alone.

        **The ``output_object`` gate is not a formality — it is the load-bearing one.** A
        request with no ``output_object`` is in ``text`` mode (no structured output at all)
        or ``tool`` mode (the output arrives in a tool call), so a text part beside the call
        is narration whatever it happens to parse as. Dropping this gate would not weaken
        the capability, it would invert it: ``str`` is this package's default output type,
        and every quoted string validates against it, so the capability would silently
        become "delete any co-emitted string" — a data-deletion regression worse than
        anything the schema walk could do. It is never relaxed to an ``isinstance`` check
        to compensate.
        """
        if not response.tool_calls:
            return []
        if request_context.model_request_parameters.output_object is None:
            return []
        return [
            part
            for part in response.parts
            if isinstance(part, TextPart) and _validates(part.content, validator)
        ]

    @staticmethod
    def _without(response: ModelResponse, dropped: list[TextPart]) -> ModelResponse:
        """A copy of ``response`` without ``dropped``, every other part kept in order.

        A replacement rather than an in-place edit of ``response.parts``: the hook is
        designed to return one, and the caller's object stays intact for anything holding
        it. Parts are matched by identity, so two equal text parts are told apart.
        Thinking parts, files and tool calls all survive — and because a tool call is a
        precondition of getting here, the result always retains one, which is what keeps
        pydantic-ai's empty-response retry path unreachable from here.
        """
        dropped_ids = {id(part) for part in dropped}
        return replace(
            response, parts=[part for part in response.parts if id(part) not in dropped_ids]
        )
