"""Run-loop concerns as standalone, mountable pydantic-ai capabilities.

Two hook anchors carry everything here:

- ``after_node_run`` — the steady-state anchor. It fires after every graph node that
  completes, and is where persistence keeps its incremental shape: messages reach
  ``ContextManager.add_message()`` as the run produces them, not in one batch at the end.
- ``wrap_run``'s ``finally`` — the closing anchor. ``after_node_run`` is **not** called for a
  node interrupted by cancellation, and a node can append to history and then die before its
  own boundary, so without a closing sweep the run's tail is never persisted. The sweep in the
  ``finally`` closes that blind tail, and the per-run system-prompt recording rides with it.

**Durable state only.** Persistence reads the run's durable history list — the list
``RunContext.messages`` points at *inside a node hook*. It never reads
``ModelRequestContext.messages`` mid-chain: that request copy legitimately carries other
capabilities' in-flight edits, and pydantic-ai writes the processed list back into durable
history after the ``before_model_request`` chain anyway.

**Composition: the first capability in the list is the outermost.** ``before_*`` hooks fire in
list order, ``after_*`` in reverse, and ``wrap_run``s nest with the first one wrapping all the
rest (pydantic-ai 2.27.1 — ``capabilities/combined.py`` builds each chain over
``reversed(self.capabilities)``). ``ReactAgent`` mounts
``[EventSourcingCapability, HealingCapability, *yours]``. That list order is not final, though:
if **any** capability in the chain declares ``get_ordering()`` (a fixed ``position``, or a
``wraps=`` / ``wrapped_by=`` constraint) pydantic-ai topologically re-sorts the whole chain to
satisfy it, so a caller capability can legitimately end up outside these two. Neither class
here declares one. What a co-mounted capability needs from the order does **not** depend on
where it sits: the closing sweep is in ``wrap_run``'s ``finally``, outside every capability's
node hooks whatever the order, so durable ``after_*`` edits are always the ones persisted.

**The ``wrap_run`` context is a snapshot, not the live list** (pydantic-ai 2.27.1, verified by
running it). ``run_ctx`` is built once, before the graph starts; ``UserPromptNode.run`` then
*rebinds* ``state.message_history`` to a different list object, so ``wrap_run``'s own
``ctx.messages`` stops tracking the run and stays frozen at the incoming history. Node hooks
get a freshly built ``RunContext`` per node and therefore do see the live list, which is why
both node hooks re-anchor the reference the closing sweep later reads — ``before_node_run``
included, so a run that dies inside its very first model request (where ``after_node_run`` has
not once fired against the rebound list) still has its messages swept.

**The rule that follows: open the cursor against the list the sweep will index** — the
normalised list a node hook hands over — never against the incoming history's length. The
normalised copy is routinely *shorter* than what was handed in, so a cursor carried over from
the incoming length sits past where the run's own messages begin and skips everything behind it
in silence. Two back-to-back ``record_operator_action`` calls are enough to trigger it; see
``_anchor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import UsageLimitExceeded
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from .context import ContextManager

logger = logging.getLogger(__name__)

# What the MODEL reads as the tool result of the call the run-tier breach aborted.
# Not a diagnostic: the operator's traceback travels the other channel
# (``ErrorMessage.traceback``, formatted by ``Akgent._handle_failure``). Defined once
# here so the call site and its test never drift into two wordings (ADR-016 §D2).
RUN_LIMIT_HEALING_MESSAGE = (
    "This turn's tool and request budget is exhausted, so this tool call was "
    "aborted and no further tool calls are possible. Answer now using what you "
    "already have, and say plainly what you could not verify."
)


@dataclass
class EventSourcingCapability(AbstractCapability[Any]):
    """Persist every message a run produces through ``ContextManager.add_message()``.

    Mountable on any pydantic-ai ``Agent``; it needs nothing from ``ReactAgent``. Each
    message is handed to ``add_message()`` unchanged and exactly once, so the event train
    (``LlmMessageEvent`` → tool events → ``LlmUsageEvent``) and the sliding window stay
    byte-identical to the hand-rolled drain this replaces.

    Persistence is cursor-based rather than identity-based: the cursor is an index into the
    run's durable history, so nothing depends on ``id()`` of a message object staying stable
    across iterations.

    The cursor is **per-run**. ``for_run`` hands back a fresh instance bound to the same
    ``ContextManager``, so one capability object can drive any number of sequential runs: a
    cursor surviving into the next run would skip that run's opening messages (a cursor too
    high) or re-persist the previous run's (a cursor too low).
    """

    context: ContextManager
    """The context manager every message this run produces is persisted through."""

    _cursor: int | None = field(default=None, init=False)
    """Index into the durable history up to which messages are already persisted.

    ``None`` outside a wrapped run, which makes the sweep a no-op — a hook can fire on a
    capability whose ``wrap_run`` never ran.
    """

    _run_start: int | None = field(default=None, init=False)
    """Durable-history length when this run started; the incoming history is never re-persisted."""

    _history: list[ModelMessage] | None = field(default=None, init=False)
    """The run's live durable history list, as last seen by either node hook.

    Not ``wrap_run``'s ``ctx.messages``: see the module docstring — that one is a snapshot
    taken before ``UserPromptNode`` rebinds the list, and never grows.
    """

    _rebound: bool = field(default=False, init=False)
    """Whether the cursor has been re-opened against the post-rebind history list."""

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Return a fresh instance bound to the same ``ContextManager``.

        The documented mechanism for per-run state isolation. ``replace()`` copies the
        configured fields and leaves every ``init=False`` field at its default, so the new
        instance starts with no cursor and no history reference.
        """
        return replace(self)

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Open the run's cursor, then close its blind tail however the run ends.

        The cursor opens at the incoming history length **before** ``handler()`` runs, which
        is what keeps a run started with a non-empty ``message_history`` from re-persisting
        messages an earlier run already recorded. It is re-opened against the normalised copy
        the graph actually appends to as soon as a node hook hands that list over
        (``_anchor``); this seeding covers the case where no node ever runs.

        The ``finally`` sweeps first and records the system prompt second — in that order,
        because ``record_system_prompt`` scans the first ``ModelRequest`` in the
        ``ContextManager``, and on a first run the sweep is what puts one there.
        """
        self._run_start = self._cursor = len(ctx.messages)
        self._history = ctx.messages
        try:
            return await handler()
        finally:
            self._sweep()
            self._record_system_prompt()

    async def before_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
    ) -> AgentNode[Any]:
        """Re-anchor the live-history reference, and pass the node through untouched.

        Observe-only. It exists because a node can append to durable history and then die
        before its own ``after_node_run`` — a run cancelled during its very first model
        request is the sharpest case, since ``after_node_run`` has then never once fired
        against the rebound list. Anchoring on the way *in* means the closing sweep still has
        the live list to sweep.
        """
        self._anchor(ctx.messages)
        return node

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
        result: NodeResult[Any],
    ) -> NodeResult[Any]:
        """Persist whatever the completed node added, and pass the result through untouched."""
        self._anchor(ctx.messages)
        self._sweep()
        return result

    def _anchor(self, messages: list[ModelMessage]) -> None:
        """Point the sweep at the live durable history, re-opening the cursor on the rebind.

        ``UserPromptNode`` does not hand the run the list it was given: it rebinds
        ``state.message_history`` to a *normalised copy* of it — consecutive ``ModelRequest``s
        merged, orphaned tool results dropped, dangling calls repaired — so the copy's length
        is not the snapshot's. The cursor ``wrap_run`` opened is an index into the snapshot;
        carried onto the copy unchanged it lands at the wrong offset, and a copy that came out
        shorter means the run's own opening messages sit *behind* the cursor and are silently
        never persisted. Two back-to-back ``record_operator_action`` calls are enough to
        trigger it — they merge into one request, and the user's next prompt disappears.

        So the cursor is re-opened against the list the sweep will actually index, the first
        time a node hook hands that list over. That is ``before_node_run`` of the node
        *after* ``UserPromptNode``: the per-node ``RunContext`` is built before its node runs,
        so nothing this run produced is in the list yet. Later re-anchors keep the cursor —
        pydantic-ai rewrites history in place from then on, never by rebinding.
        """
        if messages is self._history:
            return
        self._history = messages
        if not self._rebound:
            self._rebound = True
            self._run_start = self._cursor = len(messages)

    def _sweep(self) -> None:
        """Persist every durable message past the cursor, then advance it.

        The cursor advances **before** the messages are emitted, so an observer that re-enters
        (directly, or by driving another run on this context) cannot see the same message
        twice. A no-op when the cursor is unset — a hook that fired outside a wrapped run.
        """
        if self._cursor is None or self._history is None:
            return
        pending = self._history[self._cursor :]
        self._cursor = len(self._history)
        for message in pending:
            self.context.add_message(message)

    def _record_system_prompt(self) -> None:
        """Record this run's effective system-prompt rendering, once, after the closing sweep.

        The ``run_id`` is read off the last message **this run** added, so the emitted
        ``LlmSystemPromptEvent`` correlates with that run's other events. A run that added no
        message, or whose last message carries no ``run_id``, records nothing — there is
        nothing to correlate. Dedup by rendering hash lives in ``ContextManager`` and is
        untouched: an unchanged rendering emits nothing on the second run.

        Compared against ``_run_start``, not ``_cursor``: the closing sweep has already
        advanced the cursor to the end of the history.
        """
        if self._run_start is None or self._history is None:
            return
        if len(self._history) <= self._run_start:
            return
        run_id = getattr(self._history[-1], "run_id", None)
        if run_id is None:
            return
        self.context.record_system_prompt(str(run_id))


@dataclass
class HealingCapability(AbstractCapability[Any]):
    """Close out dangling tool calls when a run fails, then re-raise the failure unchanged.

    When a run dies mid-turn the trailing message can be a ``ModelResponse`` whose tool calls
    never received results. Left that way, the *next* run is rejected outright for unprocessed
    tool calls. This appends one ``ToolReturnPart`` per outstanding call, as a single
    ``ModelRequest`` through ``ContextManager.add_message()`` — so the healing message is
    persisted and eventized like any other message.

    **Return-to-recover is deliberately not used.** ``on_run_error`` may return an
    ``AgentRunResult`` to suppress the error; this capability always re-raises the original
    object instead. Whether to conclude a broken turn is caller policy, and a capability that
    swallowed a run-tier breach would make that tier unobservable (ADR-016 §D3).

    Mounted alongside ``EventSourcingCapability``, the ordering is structural rather than a
    matter of list position: ``wrap_run``'s ``finally`` runs before ``on_run_error``, so the
    dangling ``ModelResponse`` is already persisted by the time healing looks for it.
    """

    context: ContextManager
    """The context manager whose trailing message is inspected and healed."""

    async def on_run_error(
        self,
        ctx: RunContext[Any],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Heal any dangling tool calls, then re-raise ``error`` — the same object, unchanged.

        The declared return type is the hook's; this implementation never returns. The
        exception and its ``__traceback__`` must reach the caller untouched, because that is
        what ``Akgent._handle_failure`` formats onto ``ErrorMessage.traceback``.
        """
        self._heal(self._healing_message(error))
        raise error

    @staticmethod
    def _healing_message(error: BaseException) -> str:
        """The sentence the **model** reads as the aborted call's tool result.

        A run-tier breach gets the budget wording; anything else gets type and message — what
        a model can act on ("ReadTimeout: pool timeout" → route around it). The stack, which
        it cannot act on, is dropped here and travels the operator channel instead.
        """
        if isinstance(error, UsageLimitExceeded):
            return RUN_LIMIT_HEALING_MESSAGE
        return f"Tool call aborted: {type(error).__name__}: {error}"

    def _heal(self, model_message: str) -> None:
        """Append one ``ToolReturnPart`` per outstanding call on the trailing ``ModelResponse``.

        No-ops on an empty context, or when the trailing message is not a ``ModelResponse``
        with tool calls — there is nothing dangling to close out.

        Args:
            model_message: Used verbatim as each healing part's content, with no wrapper
                added. Each caller supplies a complete, self-contained sentence.
        """
        messages = self.context.messages
        if not messages:
            return

        last = messages[-1]
        if not isinstance(last, ModelResponse) or not last.tool_calls:
            return

        # The union ModelRequest.parts declares, rather than the concrete ToolReturnPart
        # built here, so the list stays open to other part kinds.
        error_parts: list[ModelRequestPart] = [
            ToolReturnPart(
                tool_name=call.tool_name,
                content=model_message,
                tool_call_id=call.tool_call_id,
            )
            for call in last.tool_calls
        ]

        logger.warning("Healing %d unprocessed tool call(s) after error", len(error_parts))
        self.context.add_message(ModelRequest(parts=error_parts))
