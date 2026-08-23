"""Run-loop concerns as standalone, mountable pydantic-ai capabilities.

Three hook anchors carry everything here:

- ``wrap_run``'s **head** — the pre-flight anchor. It runs before the wrapped run does
  anything at all, which is where the agent-lifetime budget refuses a spent agent (the
  outermost ``wrap_run`` is the only place a refusal costs nothing, since every inner
  capability (and every model request) is downstream of it) and where auto-compaction folds
  the history the run is about to read. ``RunContext.messages`` is the graph's own
  ``message_history`` list, which ``UserPromptNode`` *reads* before rebinding ``state`` to a
  normalised copy of it — so a head write reaches the run and a tail write does not.

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
history after the ``before_model_request`` chain anyway. That write-back is **in place** —
same list object, new contents — so it is also the thing the sweep has to survive: a
co-mounted capability that prepends or drops a message shifts every position behind it,
under a cursor measured before the shift. Hence ``_sweep``'s bound is re-derived from the
last recorded message on every pass, with the cursor kept only as the fallback.

**Composition: the first capability in the list is the outermost.** ``before_*`` hooks fire in
list order, ``after_*`` in reverse, and ``wrap_run``s nest with the first one wrapping all the
rest (pydantic-ai 2.27.1 — ``capabilities/combined.py`` builds each chain over
``reversed(self.capabilities)``). ``ReactAgent`` mounts ``[LifetimeBudgetCapability,
CompactionCapability, EventSourcingCapability, HealingCapability, *yours]``. One coupling rides
on exactly that order and nothing else: the budget refuses a spent agent **before** compaction
pays for a summarizer. Compaction sitting ahead of persistence also puts the cursor on the
post-fold history, which is where it belongs — but that one is belt-and-braces, not the
mechanism: ``_anchor`` re-opens the cursor against the normalised list at the first node hook,
which absorbs a fold performed anywhere before it (verified by swapping the two).
That list order is not final, though: if **any** capability in the chain
declares ``get_ordering()`` (a fixed ``position``, or a ``wraps=`` / ``wrapped_by=`` constraint)
pydantic-ai topologically re-sorts the whole chain to satisfy it, so a caller capability can
legitimately end up outside these four. None of the classes here declares one, and the budget
deliberately does not declare one to pin itself outermost — that would be a behavioural change
owed its own decision. What a co-mounted capability needs from the order does **not** depend on
where it sits: the closing sweep is in ``wrap_run``'s ``finally``, outside every capability's
node hooks whatever the order, so durable ``after_*`` edits are always the ones persisted.

**The ``wrap_run`` context stops tracking the run — but it is not a detached copy** (pydantic-ai
2.27.1, verified by running it, both halves). ``run_ctx`` is built once, before the graph starts,
holding ``state.message_history`` *itself*; ``UserPromptNode.run`` then normalises that object
in place and *rebinds* ``state.message_history`` to the result, so from that point on
``wrap_run``'s own ``ctx.messages`` no longer follows the run and stays frozen at the incoming
history.

Read that as a rule about **when**, not about **what**: a write performed before ``handler()``
lands in the object the normalisation reads, so it does reach the run — that is the anchor
``CompactionCapability``'s fold uses, and it is what the two fold-anchor probes in
``tests/test_capabilities.py`` pin. A write performed after ``handler()``, or a *rebind* of the
name at any point, is silently lost. Do not infer from "frozen" that ``ctx.messages`` cannot be
written; infer that it can only be written early, and only in place.

Node hooks get a freshly built ``RunContext`` per node and therefore do see the live list, which
is why both node hooks re-anchor the reference the closing sweep later reads — ``before_node_run``
included, so a run that dies inside its very first model request (where ``after_node_run`` has
not once fired against the rebound list) still has its messages swept.

**The rule that follows: open the cursor against the list the sweep will index** — the
normalised list a node hook hands over — never against the incoming history's length. The
normalised copy is routinely *shorter* than what was handed in, so a cursor carried over from
the incoming length sits past where the run's own messages begin and skips everything behind it
in silence. Two back-to-back ``append_user_prompt`` calls are enough to trigger it; see
``_anchor``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits as PydanticUsageLimits
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
from pydantic_ai.usage import RunUsage

from .compaction import CompactionResult, CompactionStrategy
from .config import AgentUsageLimits
from .context import ContextManager
from .event import LlmContextCompactedEvent

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


class UsageLimitError(Exception):
    """Raised when a usage limit is exceeded during agent execution.

    Base of both tiers — catch this to handle either; catch a subclass to react to
    one. Every breach raises one of the two subclasses below, never this class
    directly, but it stays the documented catch-all: an ``except UsageLimitError``
    written before the tiers were split still catches everything it used to
    (ADR-016 §D1).

    Defined here rather than in ``agent.py`` because ``LifetimeBudgetCapability``
    raises the agent tier and ``agent.py`` imports this module, not the reverse.
    ``akgentic.llm.agent`` re-exports all three names, so every import written
    against the old home keeps working.
    """

    pass


class RunUsageLimitError(UsageLimitError):
    """One run() call exhausted its RunUsageLimits budget.

    Requests, tool calls or tokens spent within the turn — pydantic-ai stopped the
    run mid-graph. The agent may still have lifetime budget, so this is
    **recoverable**: the turn may not call another tool, but the agent can be asked
    to conclude with what it already gathered.
    """

    pass


class AgentUsageLimitError(UsageLimitError):
    """The agent has spent its AgentUsageLimits budget over its whole lifetime.

    Raised pre-flight, by the token check or the run-count check, before the call
    executes. **Terminal** for this agent — no follow-up run can be admitted,
    because the budget that would pay for it is exactly the one that is spent.
    """

    pass


@dataclass
class LifetimeBudgetCapability(AbstractCapability[Any]):
    """Enforce the agent-LIFETIME usage budget: a run count and three token caps.

    Mountable on any pydantic-ai ``Agent``; it needs nothing from ``ReactAgent``. It owns
    both lifetime counters, refuses a spent agent in ``wrap_run`` **before** the wrapped run
    executes, and folds what the run burned back in when it ends.

    **Mount it outermost.** Every inner capability, and every model request, is downstream of
    this ``wrap_run``, so a refusal here is the only one that costs nothing at all. That is a
    property of the position, not of the class: nothing pins it there (see the module
    docstring's note on ``get_ordering()`` re-sorting), so a caller that needs the guarantee
    must not mount work outside it. ``CompactionCapability`` is the concrete case — its
    summarizer issues an LLM call, and it is only mounted *inside* this one that a spent agent
    never pays for it.

    **One enforcement site.** The two refusals live in ``wrap_run``'s head and nowhere else.
    A second, non-consuming pre-flight helper existed while compaction still ran outside the
    run; compaction moved inside, so the reason went with it. Two sites that must agree are
    one site too many.

    **Check-then-consume, tokens first.** The token check runs first so that a token refusal
    consumes no run budget; the run counter then advances *before* the call executes, so a run
    that fails partway — including one that breaches the run tier — has already been counted
    (ADR-013 §D2). The rejection itself never consumes, so the counter reports runs *consumed*,
    never runs *attempted*, and the message stays stable under repeated rejection.

    **A run may overshoot, by construction.** A run's token cost is unknown until it completes,
    so the contract is "do not start a run once the budget is spent", never "never exceed it":
    the last run admitted can carry the total arbitrarily past the limit, and only the next one
    is refused.

    **No ``for_run`` override, deliberately.** ``EventSourcingCapability`` overrides it because
    its cursor is per-run; this state is per-**agent**. pydantic-ai's default hands back
    ``self``, so one instance carries its counters across every run the agent performs. A
    per-run copy would reset them every time and the limit would simply never fire — quietly.
    """

    limits: AgentUsageLimits
    """The agent-tier budget enforced here: a lifetime run count and three token caps."""

    _run_count: int = field(default=0, init=False)
    """Runs consumed over the agent's lifetime.

    In memory only: never a Pydantic field, never persisted. Not lost on resume — ``seed()``
    recomputes it from replayed usage events.
    """

    _usage: RunUsage = field(default_factory=RunUsage, init=False)
    """Lifetime token accumulator, with the same lifecycle as ``_run_count``.

    pydantic-ai's own ``RunUsage``, so folding and comparison are both its code. NEVER handed
    to ``run(usage=…)``: a run takes exactly one usage object, and passing the lifetime total
    would check the *run* tier's limits against it, silently turning a per-run cap into a
    lifetime one.
    """

    @property
    def run_count(self) -> int:
        """Runs consumed over this agent's lifetime."""
        return self._run_count

    @property
    def usage(self) -> RunUsage:
        """The live lifetime token accumulator — the object the checks compare against."""
        return self._usage

    def seed(self, run_count: int, usage: RunUsage) -> None:
        """Set both lifetime counters from a caller's recomputation of them.

        **Assignment, not accumulation.** Replaying the same event stream twice is therefore
        idempotent, and a shorter stream lowers the value. A ``+=``, or a high-water mark,
        would make a restore depend on how many times it ran.

        Args:
            run_count: Runs already consumed, as recomputed by the caller.
            usage: The tokens those runs burned, as recomputed by the caller.
        """
        self._run_count = run_count
        self._usage = usage

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Refuse a spent agent before anything is paid for, then fold what this run burned.

        The two checks run **before** ``handler()``, so a refusal short-circuits the whole
        chain: no inner capability hook fires and no model request is issued.

        The fold is in a ``finally`` because tokens a failed run burned were still burned —
        the provider billed them either way. Its anchor is ``ctx.usage``, which is
        ``GraphAgentState.usage``: the graph only ever mutates it in place
        (``ctx.state.usage.incr(...)``, ``.requests += 1``) and never rebinds it, so it holds
        the run's real cost whether the run returned or raised. It is not this capability's
        own accumulator and must never become it.

        **Do not reuse one ``RunUsage`` across runs as ``Agent.run(usage=…)``.** That object
        becomes ``GraphAgentState.usage`` verbatim (``usage or RunUsage()``), so ``ctx.usage``
        is then a *running total*, not this run's cost, and folding it adds every earlier run
        again. ``ReactAgent`` passes no ``usage=`` at all, which is what makes the fold exact;
        a caller mounting this capability on a bare ``Agent`` must do the same, or pass a
        fresh object per run.

        Raises:
            AgentUsageLimitError: If either lifetime budget is already spent.
        """
        self._check_agent_token_budget()
        self._check_and_consume_agent_budget()
        try:
            return await handler()
        finally:
            self._usage.incr(ctx.usage)

    def _check_agent_token_budget(self) -> None:
        """Refuse to START a run once the agent-lifetime token budget is spent.

        Builds a pydantic-ai ``UsageLimits`` from the three token fields and reuses its
        ``check_tokens()`` against the lifetime accumulator, so an agent-tier breach carries
        pydantic-ai's own message wording. The tier is carried by the **class** —
        ``AgentUsageLimitError`` here, ``RunUsageLimitError`` at the run-tier site — so nothing
        downstream has to parse text to tell the tiers apart. Unset limits (the default, all
        ``None``) make the check a no-op.

        Raises:
            AgentUsageLimitError: If lifetime usage has already exceeded a token limit.
        """
        pydantic_limits = PydanticUsageLimits(
            input_tokens_limit=self.limits.input_tokens_limit,
            output_tokens_limit=self.limits.output_tokens_limit,
            total_tokens_limit=self.limits.total_tokens_limit,
        )
        try:
            pydantic_limits.check_tokens(self._usage)
        except UsageLimitExceeded as e:
            raise AgentUsageLimitError(str(e)) from e

    def _check_agent_run_budget(self) -> None:
        """Refuse to START a run once the lifetime run budget is spent; consume nothing.

        ``agent_request_limit=None`` never blocks.

        Raises:
            AgentUsageLimitError: If the agent has already used its lifetime run budget.
        """
        limit = self.limits.agent_request_limit
        if limit is not None and self._run_count >= limit:
            raise AgentUsageLimitError(
                f"Exceeded the agent_request_limit of {limit} (run_count={self._run_count})"
            )

    def _check_and_consume_agent_budget(self) -> None:
        """Spend one unit of the agent-lifetime run budget, or refuse to run.

        Check-then-consume: the counter advances **before** the call executes, so a run that
        fails partway — including one that raises the run-tier ``RunUsageLimitError`` — has
        already been counted. That ordering is deliberate: an agent whose run-tier limit fires
        repeatedly must also exhaust its agent-tier budget, since both mean "this agent is
        burning too many turns" (ADR-013 §D2). Do not move the increment after the call.

        Raises:
            AgentUsageLimitError: If the agent has already used its lifetime run budget.
        """
        self._check_agent_run_budget()
        self._run_count += 1


@dataclass
class CompactionCapability(AbstractCapability[Any]):
    """Fold the conversation before the run reads it, when the token gate is armed.

    Mountable on any pydantic-ai ``Agent``; it needs nothing from ``ReactAgent``. The gate
    reads provider-reported tokens only — never ``tiktoken``, never an estimate — and the fold
    itself is the strategy's, so this class owns only *when* to fold and *where the result
    goes*.

    **Mount it inside the lifetime budget and outside persistence.** Both neighbours matter
    and neither is pinned by anything but list order:

    - Inside ``LifetimeBudgetCapability``, because the summarizer issues its own LLM call. A
      spent agent must be refused before it pays for one, and the budget's refusal in the
      enclosing ``wrap_run`` head is what delivers that.
    - Outside ``EventSourcingCapability``, so its cursor opens on the POST-fold list and the
      synthetic summary request sits behind it, never re-persisted as an ``LlmMessageEvent``
      (ADR-010 §9's replay rule — the ``LlmContextCompactedEvent`` already carries the
      summary, so persisting both double-applies it on restore). This half is belt-and-braces
      rather than load-bearing: ``EventSourcingCapability._anchor`` re-opens the cursor
      against the normalised list at the first node hook, which absorbs a fold performed
      anywhere ahead of it. Verified by swapping the two — the outcome is unchanged. The
      budget coupling above is the one that genuinely depends on position.

    **No ``for_run`` override, deliberately.** It holds no per-run state — the gate re-reads
    ``context.last_input_tokens`` every time — and pydantic-ai's default hands back ``self``,
    so ``wrap_run`` fires once per run by construction. That once-per-run property is what
    ``before_model_request`` could not give: that hook fires per model *request*, so a fold
    placed there would need an explicit guard to stay once per turn.
    """

    strategy: CompactionStrategy
    """How to summarize and where the fold boundary is; the framework owns the fold itself."""

    context: ContextManager
    """The durable history the fold is applied to, and the observer channel it is emitted on."""

    threshold_fn: Callable[[], int | None]
    """The armed token budget, re-read per run. ``None`` ⇒ auto-compaction is off.

    A callable rather than a value so a config change between runs takes effect, and so
    "compaction is off" stays one concept computed in one place by the owner of the config.
    """

    event_factory: Callable[[CompactionResult], LlmContextCompactedEvent]
    """Builds the append-only event from a strategy result.

    Injected because the event carries configuration this capability deliberately does not
    hold — the strategy id and the summarizer prompt version — which keeps it mountable à la
    carte without a ``ReactAgentConfig``.
    """

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Fold before the run reads its history, then run.

        The fold is performed **before** ``handler()``, which is what makes it take effect:
        ``RunContext.messages`` is ``GraphAgentState.message_history`` itself, the same list
        object the graph holds (pydantic-ai 2.27.1 — ``_agent_graph.build_run_context``), and
        ``UserPromptNode.run`` *reads* that object before rebinding ``state`` to a normalised
        copy of it. A write placed after ``handler()`` would land on a list nothing reads
        again, silently.

        Nothing needs to run on the way out, so there is no ``try``/``finally`` here.
        """
        if self._armed():
            await self.compact_now(ctx.messages)
        return await handler()

    def _armed(self) -> bool:
        """Whether this run's history is over the auto-compaction threshold.

        Three no-ops, in the order they are cheapest to decide: auto-compaction off
        (``threshold_fn()`` is ``None``), no usage reported yet (``last_input_tokens`` is
        ``None`` — never mis-fire on missing data), and usage at or below the threshold
        (strictly above fires).
        """
        threshold = self.threshold_fn()
        used = self.context.last_input_tokens
        return threshold is not None and used is not None and used > threshold

    async def compact_now(self, live_messages: list[ModelMessage] | None = None) -> str:
        """Run the strategy and apply its result to both histories — the only fold site.

        The durable write (``ContextManager.compact``) and the live write
        (``live_messages[:] = …``) are **one logical operation** and are kept in one method
        for that reason: ``Agent.run()`` seeds the run's state from a copy of the history it
        is handed, so mutating the run's list never reaches ``ContextManager`` and folding
        ``ContextManager`` never reaches the run. Split across two methods they drift the
        first time someone adds an early return.

        The live write **mirrors** the durable result rather than folding a second time.
        Applying the fold again to an already-folded list double-folds, and mirroring also
        preserves message *identity*, which ``EventSourcingCapability``'s tail anchor locates
        by.

        Args:
            live_messages: The run's own history list, mutated in place. ``None`` on the
                manual ``/compact`` path, where there is no run in flight — the durable write
                is then the only one, and the next run picks the folded history up from
                ``ContextManager``.

        Returns:
            A human-readable status string.
        """
        result = await self.strategy.compact(self.context.messages)
        if result.replaced_message_count == 0:
            return "Nothing to compact."
        self.context.compact(self.event_factory(result))
        if live_messages is not None:
            live_messages[:] = self.context.messages
        return (
            f"Compacted: replaced {result.replaced_message_count} "
            f"earlier message(s) with a summary."
        )


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
