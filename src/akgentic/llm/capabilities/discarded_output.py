"""``DiscardedOutputCapability`` — strip from a response what pydantic-ai will discard.

See the package docstring for the hook anchors and composition order; this module is the
first entry on the ``after_model_request`` anchor and explains why it uses that one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import EndStrategy
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ObjectJsonSchema, RunContext

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

# JSON Schema keywords that constrain nothing — they annotate. Ignored outright.
#
# Membership here is a claim that the keyword cannot change the verdict, so it is not a
# place to park anything merely unfamiliar: an ignored keyword that does affect the
# outcome reads as "valid" and strips a part the schema would have rejected. ``$id`` is
# the near miss and is deliberately **absent** — it re-roots ``$ref`` resolution, which
# this module resolves only against the top-level ``$defs``, so a schema carrying one is
# undecidable rather than ignorable.
_ANNOTATION_KEYWORDS = frozenset(
    {
        "$comment",
        "$defs",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "format",
        "readOnly",
        "title",
        "writeOnly",
    }
)

# The keywords this module actually evaluates. Anything outside these two sets makes the
# schema undecidable here, and an undecidable schema means "does not validate" — which
# keeps the part. See `_validates` for why that direction is the safe one.
_SUPPORTED_KEYWORDS = frozenset(
    {
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "enum",
        "items",
        "oneOf",
        "properties",
        "required",
        "type",
    }
)

_MAX_SCHEMA_DEPTH = 32
"""Recursion bound. A self-referential ``$ref`` would otherwise not terminate; exceeding
the bound is treated as undecidable, so the part survives."""

_REF_PREFIX = "#/$defs/"


def _is_instance_of(value: object, type_name: str) -> bool:
    """Whether ``value`` is an instance of one JSON Schema primitive type name.

    ``bool`` is excluded from the numeric types deliberately: Python makes it a subclass
    of ``int``, JSON Schema does not. An unrecognised name returns False, which makes the
    schema undecidable and keeps the part.
    """
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "null":
        return value is None
    return False


def _matches_type(value: object, declared: object) -> bool:
    """Whether ``value`` satisfies a ``type`` keyword, single-valued or a list of names."""
    names = declared if isinstance(declared, list) else [declared]
    return any(isinstance(name, str) and _is_instance_of(value, name) for name in names)


def _resolve_ref(ref: object, defs: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local ``#/$defs/<name>`` reference; None for anything else.

    Only the local form pydantic emits is understood. A remote or pointer-style ``$ref``
    resolves to None, which the caller reads as undecidable.
    """
    if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
        return None
    target = defs.get(ref[len(_REF_PREFIX) :])
    return target if isinstance(target, dict) else None


def _matches_object(
    value: dict[str, Any], schema: dict[str, Any], defs: dict[str, Any], depth: int
) -> bool:
    """Check ``required``, ``properties`` and ``additionalProperties`` against a mapping.

    A keyword present but malformed is undecidable, never ignored: skipping a ``required``
    that is not a list would drop the strongest constraint in the schema and let a
    mismatched payload read as valid — a strip on doubt, in the one direction that costs
    data. Its *elements* are checked for the same reason and one more: a name that is not
    a string decides nothing, and an unhashable one would raise ``TypeError`` out of a
    hook that runs on every model response.
    """
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        return False
    if any(name not in value for name in required):
        return False

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return False
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        return False

    for key, item in value.items():
        sub_schema = properties.get(key)
        if sub_schema is None:
            if additional is False:
                return False
            if isinstance(additional, dict) and not _matches(item, additional, defs, depth + 1):
                return False
            continue
        if not _matches(item, sub_schema, defs, depth + 1):
            return False
    return True


def _matches_enumeration(value: object, schema: dict[str, Any]) -> bool:
    """Check the two value-listing keywords, ``const`` and ``enum``."""
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema:
        options = schema["enum"]
        return isinstance(options, list) and value in options
    return True


def _matches_composition(
    value: object, schema: dict[str, Any], defs: dict[str, Any], depth: int
) -> bool:
    """Check the two subschema-combining keywords, ``anyOf`` and ``oneOf``."""
    if "anyOf" in schema:
        options = schema["anyOf"]
        if not isinstance(options, list):
            return False
        if not any(_matches(value, option, defs, depth + 1) for option in options):
            return False
    if "oneOf" in schema:
        options = schema["oneOf"]
        if not isinstance(options, list):
            return False
        if sum(_matches(value, option, defs, depth + 1) for option in options) != 1:
            return False
    return True


def _matches(value: object, schema: object, defs: dict[str, Any], depth: int) -> bool:
    """Whether ``value`` conforms to ``schema``, over the subset pydantic emits.

    Every uncertainty resolves to False. A keyword outside the two sets above, a ``$ref``
    that does not resolve, a type name not enumerated, a schema nested past
    ``_MAX_SCHEMA_DEPTH`` — each says "does not validate", so the part is kept. That is
    the direction that cannot destroy the model's reasoning; the opposite direction can.
    """
    if depth > _MAX_SCHEMA_DEPTH or not isinstance(schema, dict):
        return False
    if set(schema) - _ANNOTATION_KEYWORDS - _SUPPORTED_KEYWORDS:
        return False

    if "$ref" in schema:
        # A ``$ref`` alongside another constraining keyword applies both, and this walk
        # follows only the reference. Ignoring the sibling is the permissive direction —
        # a payload the sibling rejects would read as valid and be stripped — so a
        # ``$ref`` that is not alone (annotations aside) is undecidable. Annotations are
        # still allowed beside it: that is the form pydantic actually emits.
        if set(schema) - _ANNOTATION_KEYWORDS - {"$ref"}:
            return False
        target = _resolve_ref(schema["$ref"], defs)
        return target is not None and _matches(value, target, defs, depth + 1)

    if not _matches_enumeration(value, schema):
        return False
    if not _matches_composition(value, schema, defs, depth):
        return False
    if "type" in schema and not _matches_type(value, schema["type"]):
        return False

    if isinstance(value, dict):
        return _matches_object(value, schema, defs, depth)
    if isinstance(value, list) and "items" in schema:
        return all(_matches(item, schema["items"], defs, depth + 1) for item in value)
    return True


def _validates(content: str, output_object: OutputObjectDefinition) -> bool:
    """Whether ``content`` is an instance of the run's own output schema.

    Two gates, both conservative. The text must parse as JSON — prose and truncated JSON
    fail here — and the parsed value must conform to ``output_object.json_schema``.

    **Why a schema walk and not a library.** ``jsonschema`` is not a dependency of this
    package (it reaches this workspace transitively through ``mcp``, which belongs to
    ``akgentic-tool``), and pydantic cannot build a validator from a JSON schema, so the
    only in-package option is to walk the schema. The walk covers the constructs pydantic
    emits for an output object — ``$defs``/``$ref``, ``type``, ``properties``,
    ``required``, ``items``, ``enum``, ``const``, ``anyOf``/``oneOf``,
    ``additionalProperties`` — and calls anything else undecidable rather than guessing.

    That incompleteness is safe in exactly one direction, and it is this one: an
    unrecognised construct makes the capability a no-op for that schema, which costs the
    fix; the opposite failure would delete the model's reasoning from history, which
    costs data. "Never strip on doubt" is what makes an incomplete checker acceptable.
    """
    try:
        value = json.loads(content)
    except ValueError:
        return False
    schema: ObjectJsonSchema = output_object.json_schema
    raw_defs = schema.get("$defs", {})
    defs: dict[str, Any] = raw_defs if isinstance(raw_defs, dict) else {}
    return _matches(value, schema, defs, 0)


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

    **It strips only what validates against the run's own output schema.** Never on
    ``isinstance(part, TextPart)``: a tool call sitting beside plain narration is
    ordinary, and stripping on the type alone would silently delete the model's
    reasoning. Parse failure, schema mismatch, or a schema this module cannot decide all
    mean "keep".

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

    _run_id: str | None = field(default=None, init=False)
    """The run the counters below belong to, taken from the response's own ``run_id`` —
    the same source the recorder is handed. Keying on it rather than on ``for_run`` means
    the budget is per run even for a capability instance mounted across several."""

    _strips: int = field(default=0, init=False)
    """Strips already made in ``_run_id``."""

    _exhausted: bool = field(default=False, init=False)
    """Whether ``_run_id``'s budget has already refused a strip. Once set, the run's
    remaining responses pass through without the budget being tested again."""

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Return ``response`` with the about-to-be-discarded output removed, or unchanged.

        Unchanged unless every gate passes: the strategy is ``'exhaustive'``, this run's
        budget is unspent, the response carries at least one ``ToolCallPart``, the request
        declares an output schema, and at least one ``TextPart`` validates against it.
        """
        if self.end_strategy != "exhaustive":
            return response
        self._track_run(response.run_id)
        if self._exhausted:
            return response

        discardable = self._discardable(request_context, response)
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

    def _track_run(self, run_id: str | None) -> None:
        """Reset the budget when the response belongs to a run this instance has not seen.

        Per run, not per agent: a fresh run starts with the full budget however many runs
        this instance has already driven. Two consecutive runs that both leave ``run_id``
        unset share one budget — a synthetic case, since the agent graph fills the id in
        before this hook is reached.
        """
        if run_id == self._run_id:
            return
        self._run_id = run_id
        self._strips = 0
        self._exhausted = False

    @staticmethod
    def _discardable(
        request_context: ModelRequestContext, response: ModelResponse
    ) -> list[TextPart]:
        """The text parts this response would lose to the discard branch, in emission order.

        Empty unless the response carries a tool call *and* the request declared an output
        schema. Text with no tool call is not discarded by the graph at all — that case is
        the multi-part merge, and this capability must leave it alone — and a request with
        no ``output_object`` (``text`` mode, or ``tool``-mode output) has nothing for a
        part to be an instance of, so nothing can validate.
        """
        if not response.tool_calls:
            return []
        output_object = request_context.model_request_parameters.output_object
        if output_object is None:
            return []
        return [
            part
            for part in response.parts
            if isinstance(part, TextPart) and _validates(part.content, output_object)
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
