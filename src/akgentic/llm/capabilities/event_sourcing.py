"""``EventSourcingCapability`` — persist every message a run produces, incrementally.

See the package docstring for the hook anchors, composition order and cursor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from ..context import ContextManager


@dataclass
class EventSourcingCapability(AbstractCapability[Any]):
    """Persist every message a run produces through ``ContextManager.add_message()``.

    Mountable on any pydantic-ai ``Agent``; it needs nothing from ``ReactAgent``. Each
    message is handed to ``add_message()`` unchanged and exactly once, so the event train
    (``LlmMessageEvent`` → tool events → ``LlmUsageEvent``) and the sliding window stay
    byte-identical to the hand-rolled drain this replaces.

    The sweep is bounded two ways, because position and identity have mirror blind spots:
    position breaks when a co-mounted capability inserts or drops a message ahead of the
    cursor, identity when one rebuilds the history out of equal copies. The last recorded
    message, located by identity, is the primary bound — a shift moves it without changing
    what it is; the positional cursor is the fallback for the rebuild a shift cannot describe.

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

    _recorded_tail: ModelMessage | None = field(default=None, init=False)
    """The last message known to be recorded — this run's, or the incoming history's.

    The sweep's identity anchor. An in-place edit to durable history moves this object's
    position without changing which object it is, so finding it re-derives the bound the
    cursor can no longer be trusted for.
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

        Every per-run field is reset here, so one capability object can drive any number of
        sequential runs even where ``for_run`` was not what produced it.

        The cursor opens at the incoming history length **before** ``handler()`` runs, which
        is what keeps a run started with a non-empty ``message_history`` from re-persisting
        messages an earlier run already recorded. It is re-opened against the normalised copy
        the graph actually appends to as soon as a node hook hands that list over
        (``_anchor``); this seeding covers the case where no node ever runs.

        The ``finally`` sweeps first and records the system prompt second — in that order,
        because ``record_system_prompt`` scans the first ``ModelRequest`` in the
        ``ContextManager``, and on a first run the sweep is what puts one there.
        """
        self._rebound = False
        self._history = ctx.messages
        self._open(ctx.messages)
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
        """Point the sweep at the live durable history, re-opening it on the rebind.

        ``UserPromptNode`` does not hand the run the list it was given: it rebinds
        ``state.message_history`` to a *normalised copy* of it — consecutive ``ModelRequest``s
        merged, orphaned tool results dropped, dangling calls repaired — so the copy's length
        is not the snapshot's. The cursor ``wrap_run`` opened is an index into the snapshot;
        carried onto the copy unchanged it lands at the wrong offset, and a copy that came out
        shorter means the run's own opening messages sit *behind* the cursor and are silently
        never persisted. Two back-to-back ``append_user_prompt`` calls are enough to
        trigger it — they merge into one request, and the user's next prompt disappears.

        So the cursor is re-opened against the list the sweep will actually index, the first
        time a node hook hands that list over. That is ``before_node_run`` of the node
        *after* ``UserPromptNode``: the per-node ``RunContext`` is built before its node runs,
        so nothing this run produced is in the list yet. Later re-anchors keep the bound —
        pydantic-ai rewrites history in place from then on, never by rebinding, and an in-place
        rewrite is what ``_sweep``'s tail anchor is there to absorb.
        """
        if messages is self._history:
            return
        self._history = messages
        if not self._rebound:
            self._rebound = True
            self._open(messages)

    def _open(self, messages: list[ModelMessage]) -> None:
        """Set both of the sweep's bounds against ``messages``: nothing in it is this run's."""
        self._run_start = self._cursor = len(messages)
        self._recorded_tail = messages[-1] if messages else None

    def _sweep(self) -> None:
        """Persist every durable message past the last recorded one, then re-bound.

        The bound advances **before** the messages are emitted, so an observer that re-enters
        (directly, or by driving another run on this context) cannot see the same message
        twice. A no-op when the cursor is unset — a hook that fired outside a wrapped run.
        """
        if self._cursor is None or self._history is None:
            return
        history = self._history
        pending = history[self._bound(history, self._cursor) :]
        self._cursor = len(history)
        if pending:
            self._recorded_tail = pending[-1]
        for message in pending:
            self.context.add_message(message)

    def _bound(self, history: list[ModelMessage], cursor: int) -> int:
        """Index of the first message in ``history`` this run has not recorded yet.

        The last recorded message, located by identity from the end: pydantic-ai writes a
        ``before_model_request`` chain's processed list back into durable history in place, so
        a co-mounted capability that prepends or drops a message moves that message without
        changing which object it is, while ``cursor`` still counts from before the shift.

        ``cursor`` is the fallback for the one edit identity cannot follow — a rebuild that
        replaces messages with equal copies, which leaves their positions intact.

        Two edits defeat both bounds, because each destroys the anchor *and* moves what is
        behind it: rebuilding and shifting in the same pass, and removing the anchor message
        itself. Both fall back to a cursor that no longer describes the list, which duplicates
        or skips exactly as an unanchored cursor did. pydantic-ai's own layered equivalent has
        the same blind spot and reaches for ``run_id`` there; this one does not, and a
        processor that summarises or redacts durable history is the shape that reaches it.
        """
        tail = self._recorded_tail
        if tail is not None:
            for index in range(len(history) - 1, -1, -1):
                if history[index] is tail:
                    return index + 1
        return cursor

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
