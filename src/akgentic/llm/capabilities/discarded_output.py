"""``DiscardedOutputCapability`` — strip from a response what pydantic-ai will discard.

See the package docstring for the hook anchors and composition order; this module is the
first entry on the ``after_model_request`` anchor and explains why it uses that one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ValidationError
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


def _validates(content: str, output_type: type[BaseModel]) -> bool:
    """Whether ``content`` is an instance of the run's own output class.

    **The verdict, not an approximation of it.** ``model_validate_json`` is the same call
    pydantic-ai's own output processor makes on the same text against the same class
    (``_output.py`` — ``self.validator.validate_json(...)``), so "this text is the run's
    structured output" is answered by the library that decides it rather than by a
    re-implementation of JSON Schema. Prose and truncated JSON fail on the parse pydantic
    performs first, so there is no separate parse gate to keep in step.

    **No ``strict=True``.** ``NativeOutput(output_type, strict=True)`` in ``providers.py``
    sets ``OutputObjectDefinition.strict``, a JSON-Schema strictness hint carried to the
    provider; it is not pydantic's validation mode and never reaches the validator that
    decides the output. Validating strictly here would reject payloads pydantic-ai accepts
    — a numeric string for an ``int`` field being the everyday case — so those parts would
    survive a discard the graph performs anyway, which is the defect this capability
    exists to fix.

    **No markdown-fence stripping.** pydantic-ai runs ``strip_markdown_fences`` over text
    output before validating, so a fenced payload it would discard is kept here instead.
    That helper is private to pydantic-ai, and the divergence falls in the safe direction:
    a kept part costs the fix, never data.

    A model whose validators raise something other than ``ValidationError`` must not end
    the run from a hook that fires on every model response, so anything else is logged and
    read as "does not validate" — the direction that keeps the part.
    """
    try:
        output_type.model_validate_json(content)
    except ValidationError:
        return False
    except Exception:
        logger.exception(
            "Validating a text part against %s raised; leaving the part in place",
            output_type.__name__,
        )
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

    **It strips only what validates against the run's own output class.** Never on
    ``isinstance(part, TextPart)``: a tool call sitting beside plain narration is
    ordinary, and stripping on the type alone would silently delete the model's
    reasoning. Parse failure and schema mismatch both mean "keep".

    **The host declares the output class, once per run.** ``expect_output_type`` is called
    with the class *this* run will produce, immediately before the run starts;
    ``for_run`` snapshots it onto the per-run instance and clears the slot. A host that
    declares nothing — a bare ``Agent`` with this capability mounted by hand — gets a
    no-op, which is the same fail-closed direction as an unparseable part. The class is
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

    _pending_output_type: type[BaseModel] | None = field(default=None, init=False)
    """The class the next run will produce, parked here by ``expect_output_type``.

    Lives on the *shared* instance because that is the only object the host holds; it is
    read and cleared by ``for_run`` and never by a hook, so no hook ever consults a slot
    another run can be writing.
    """

    _output_type: type[BaseModel] | None = field(default=None, init=False)
    """This run's output class, snapshotted by ``for_run``. ``None`` on the shared
    instance and on any run whose host declared nothing — both no-ops."""

    _strips: int = field(default=0, init=False)
    """Strips already made in this run. Per-run because ``for_run`` hands each run its own
    instance; on the shared one it stays at zero forever."""

    _exhausted: bool = field(default=False, init=False)
    """Whether this run's budget has already refused a strip. Once set, the run's
    remaining responses pass through without the budget being tested again."""

    def expect_output_type(self, output_type: object) -> None:
        """Declare the class the next run's structured output will be validated against.

        Called by the host immediately before it starts the run, and at that point
        deliberately: a value resolved a frame earlier goes stale when a tool calls
        ``switch_model`` mid-run, which is the same reason ``ReactAgent`` resolves the
        model and the effective output type at the ``run()`` call itself.

        Anything that is not a ``BaseModel`` subclass parks ``None`` and makes the run a
        no-op. ``str`` and ``None`` are the ordinary cases — a run with no structured
        output has nothing for a part to be an instance of. The others are refused on
        purpose: pydantic-ai wraps a bare output type in a ``{"response": ...}`` envelope
        before the model ever sees it, so validating the naked type here would answer a
        different question than the graph does, and in the direction that strips.

        Args:
            output_type: The run's effective output type, wrapper-free.
        """
        self._pending_output_type = (
            output_type
            if isinstance(output_type, type) and issubclass(output_type, BaseModel)
            else None
        )

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Hand this run its own instance, carrying its output class and its own budget.

        Upstream's sanctioned per-run isolation, and load-bearing twice over. The budget
        was previously a single slot on the shared instance, so interleaved runs reset
        each other's counters and the bound was lost; the output class on a shared slot
        would be worse — a run could validate against another run's schema and strip a
        part that is not its output at all. Neither is reachable once the state lives on
        the instance only this run holds.

        Clearing the pending slot is what makes an undeclared run a no-op rather than a
        run validated against whatever the previous one declared.

        Upstream notes that under durable execution per-run state must be derivable from
        ``ctx``. It is not derivable here — ``RunContext`` carries no output schema — and
        this package does not use durable execution; a durable host would have to declare
        the class through its own serialised state.

        Args:
            ctx: This run's context. Unread — the class comes from the host.

        Returns:
            A copy of this capability bound to this run.
        """
        bound = replace(self)
        bound._output_type = self._pending_output_type
        self._pending_output_type = None
        return bound

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Return ``response`` with the about-to-be-discarded output removed, or unchanged.

        Unchanged unless every gate passes: the strategy is ``'exhaustive'``, the host
        declared an output class for this run, this run's budget is unspent, the response
        carries at least one ``ToolCallPart``, the request declares an output schema, and
        at least one ``TextPart`` validates against the class.
        """
        if self.end_strategy != "exhaustive" or self._output_type is None or self._exhausted:
            return response

        discardable = self._discardable(request_context, response, self._output_type)
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
        output_type: type[BaseModel],
    ) -> list[TextPart]:
        """The text parts this response would lose to the discard branch, in emission order.

        Empty unless the response carries a tool call *and* the request declared an output
        schema. Text with no tool call is not discarded by the graph at all — that case is
        the multi-part merge, and this capability must leave it alone — and a request with
        no ``output_object`` (``text`` mode, or ``tool``-mode output) puts the output
        somewhere other than a text part, so no text part can be one.
        """
        if not response.tool_calls:
            return []
        if request_context.model_request_parameters.output_object is None:
            return []
        return [
            part
            for part in response.parts
            if isinstance(part, TextPart) and _validates(part.content, output_type)
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
