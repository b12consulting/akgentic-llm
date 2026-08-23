"""REACT-based LLM agent with context management and iteration support."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from pydantic_ai import Agent, AgentCapability, BinaryContent, UsageLimitExceeded
from pydantic_ai import UsageLimits as PydanticUsageLimits
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

# Re-exported, not used here: healing moved into HealingCapability and the usage-limit
# hierarchy moved next to the capability that raises the agent tier, but
# `akgentic.llm.agent.<name>` stays importable for callers written against the old home.
# The capabilities package holds the one definition of each.
from .capabilities import (
    RUN_LIMIT_HEALING_MESSAGE as RUN_LIMIT_HEALING_MESSAGE,
)
from .capabilities import (
    AgentUsageLimitError as AgentUsageLimitError,
)
from .capabilities import (
    CompactionCapability,
    ConclusionDecision,
    EventSourcingCapability,
    HealingCapability,
    LifetimeBudgetCapability,
    LimitRecoveryCapability,
)
from .capabilities import (
    RunUsageLimitError as RunUsageLimitError,
)
from .capabilities import (
    UsageLimitError as UsageLimitError,
)
from .compaction import CompactionResult, CompactionStrategy, create_compaction
from .config import ReactAgentConfig, RunUsageLimits
from .context import ContextManager
from .event import (
    ContextObserver,
    EventMessage,
    LlmContextClearedEvent,
    LlmContextCompactedEvent,
    LlmMessageEvent,
    LlmSystemPromptEvent,
    LlmUsageEvent,
)
from .pricing import aggregate_usage
from .providers import create_http_client, create_model, get_output_type

logger = logging.getLogger(__name__)

UserPrompt = str | list[str | BinaryContent]


def _evict_anyio_run_vars(loop: asyncio.AbstractEventLoop) -> None:
    """Remove anyio's per-loop run-vars entry so the closed loop can be GC'd.

    anyio keeps per-loop state in a module-global ``WeakKeyDictionary``
    (``anyio.lowlevel._run_vars``, keyed by the loop); its value retains the
    finished run ``Task``, which strong-references the loop, so the weak key
    never clears and the loop leaks. Evicting our own loop's entry breaks that
    ``_root_task → loop`` anchor. Best-effort and version-guarded: anyio
    internals are private, and a missing/absent ``_run_vars`` (or no anyio) must
    not break teardown. Local copy of ``akgentic.core.agent._evict_anyio_run_vars``
    (NOT imported — ``akgentic-llm`` must not depend on a sibling package).
    """
    try:
        from anyio.lowlevel import _run_vars  # noqa: PLC0415

        _run_vars.pop(loop, None)
    except Exception:
        pass


class ReactAgent:
    """REACT-based LLM agent.

    Features:
    - REACT pattern support (via pydantic-ai)
    - Dynamic system prompts with registry
    - Context management with observer pattern
    - Iterative execution with context updates
    - Tool integration
    - Usage limit enforcement
    - HTTP retry logic

    Example:
        >>> config = ReactAgentConfig(
        ...     model_cfg=ModelConfig(provider="openai", model="gpt-4o")
        ... )
        >>> # Option 1: Pass observer at initialization
        >>> agent = ReactAgent(
        ...     config=config,
        ...     deps_type=MyDeps,
        ...     tools=[my_tool_func],
        ...     observer=my_observer
        ... )
        >>> # Option 2: Subscribe observer later
        >>> agent = ReactAgent(config=config)
        >>> agent.subscribe_context(my_observer)
        >>> result = await agent.run("User query")
    """

    def __init__(
        self,
        config: ReactAgentConfig,
        deps_type: type[Any] | None = None,
        tools: list[Any] | None = None,
        toolsets: list[Any] | None = None,
        result_type: type[Any] = str,
        observer: ContextObserver | None = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        limit_recovery: LimitRecoveryCapability | None = None,
    ) -> None:
        """Initialize REACT agent.

        Args:
            config: Complete agent configuration
            deps_type: Type for dependency injection (optional)
            tools: List of tool functions (optional)
            toolsets: List of toolsets (optional, e.g., MCP servers)
            result_type: Type for agent result validation (default: str)
            observer: Context observer to register automatically (optional)
            capabilities: Optional sequence of pydantic-ai AgentCapability instances.
                They are NOT forwarded unchanged: four internal capabilities —
                LifetimeBudgetCapability, then CompactionCapability, then
                EventSourcingCapability, then HealingCapability — are mounted ahead of
                them, because those four own the agent-lifetime budget, auto-compaction,
                persistence, system-prompt recording and dangling-tool-call healing for
                every run this agent drives. The budget is outermost so a run it refuses
                reaches none of the others — including the summarizer LLM call.
                That is the MOUNT order, and it is a default rather than a guarantee:
                pydantic-ai's CombinedCapability topologically re-sorts the whole chain
                as soon as ANY capability declares get_ordering(), so a caller declaring
                position='outermost' — or wraps=[...] naming one of the four — lands
                ahead of them. None of the four declares an ordering, so a caller that
                declares nothing does sit inside all four, and the two consequences below
                hold for that caller. A caller that re-sorts itself gets neither.
                First: a capability sees only the POST-compaction history, never what
                compaction folded away — the fold happens in CompactionCapability's
                wrap_run head, which encloses every hook a caller capability has.
                Second: because pydantic-ai unwinds the chain in reverse, a caller
                capability's `after_*` hooks run BEFORE the persistence sweep, so its
                durable edits are the ones persisted. Persistence survives any ordering
                regardless — the closing sweep sits in a `finally` outside every node
                hook — but `on_run_error` precedence does not, and is deliberately left
                uncontracted (see backlog.md).
                Under pydantic-ai 2.x (verified against 2.31.0), a capability that
                orphans a tool call/return pair (e.g. by splitting one while injecting
                content) is NOT left broken: pydantic-ai's own dangling-tool-call
                repair (`_agent_graph._clean_message_history` with
                `repair_last_response=True`) runs on the model request path, AFTER
                the capability chain, and silently synthesizes a matching
                ToolReturnPart before the request reaches the provider. One
                pydantic-ai path skips the repair: resuming a provider-suspended
                response runs the capability chain without it. ReactAgent has no
                deferred-tool or suspend flow, so every request ReactAgent itself
                issues is repaired. This corrects the pre-v2 assumption that no such
                re-fold happened. It is pydantic-ai's own internal pipeline behavior,
                not a documented public guarantee, and could change in a future
                release — a capability should still avoid orphaning tool calls on
                purpose.
            event_loop: Deprecated — accepted and ignored. The agent creates and
                owns its own loop (``self._loop``); the passed loop is neither
                adopted nor used by ``run_sync``. Kept in the signature for one
                release so callers can stop passing it without a flag day.
            limit_recovery: The run-tier recovery policy, as a
                ``LimitRecoveryCapability`` (or a subclass overriding its
                ``handle_limit_exceeded`` seam). Defaults to the base class, whose
                policy is one tool-free conclusion per run-tier breach; a subclass
                returning ``None`` from the seam restores the pre-recovery contract,
                where a breach simply raises. This keyword is the ONLY way to mount
                a subclass: passing one through ``capabilities=`` would mount a
                *second* recovery capability beside the default rather than replacing
                it, and ``_run_with_limits`` reads the decision off the instance held
                here.
        """
        # The agent owns its loop: create it and make it current on the
        # constructing thread BEFORE building the httpx client / model, so the
        # connection pool stays a per-agent resource bound to one stable loop
        # (ADR-008). The deprecated `event_loop=` argument is ignored.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._closed = False

        self._config = config
        self._deps_type = deps_type
        self._result_type = result_type

        # Create context manager (no max_messages by default)
        self._context = ContextManager()

        # Register observer if provided
        if observer:
            self._context.subscribe(observer)

        # Create HTTP client. Held on the instance so its connection pool can be
        # released on stop via aclose(); otherwise the pydantic-ai Model keeps the
        # httpx.AsyncClient (open sockets/TLS buffers) alive past team teardown.
        self._http_client = create_http_client(
            timeout_s=config.runtime_cfg.http_client_config.timeout,
            max_attempts=config.runtime_cfg.http_client_config.max_retries,
            exp_multiplier=config.runtime_cfg.http_client_config.backoff_multiplier,
            exp_max_s=config.runtime_cfg.http_client_config.backoff_max,
        )

        # Create model from config
        self._model = create_model(config.model_cfg, self._http_client)

        # Wrap result_type with provider-aware output strategy for structured output
        wrapped_result_type: Any = get_output_type(config.model_cfg, result_type)

        # The whole agent-lifetime budget — both counters, both pre-flight checks, the
        # usage fold and the restore seeding — lives here. Held on the instance because
        # restore_context() reseeds it and the two read-through properties below report
        # it; it is also the first entry of the capability stack assembled further down.
        self._budget = LifetimeBudgetCapability(limits=config.agent_usage_limits)

        # Resolve the compaction strategy as runtime state (never a Pydantic
        # field). Built AFTER self._http_client so the summarizer reuses the
        # agent's shared httpx client (no second pool); the summarizer model uses
        # summary_model_cfg when set, else the primary model_cfg.
        summary_cfg = config.compaction_cfg.summary_model_cfg or config.model_cfg
        strategy: CompactionStrategy = create_compaction(
            config.compaction_cfg, summary_cfg, self._http_client
        )

        # The whole of auto-compaction — the token gate, the durable fold and the
        # in-place fold of the run's own history — lives here. Held on the instance
        # because the manual /compact bridge delegates to it and the read-through
        # `_compaction` property below reports its strategy; it is also the second
        # entry of the capability stack assembled further down.
        self._compactor = CompactionCapability(
            strategy=strategy,
            context=self._context,
            threshold_fn=self._compaction_threshold,
            event_factory=self._build_compaction_event,
        )

        # The run-tier recovery POLICY — whether a breached turn concludes instead of
        # raising, and with what prompt. Held on the instance for the same reason as the two
        # above: ``_run_with_limits`` reads the decision back off this exact object, which is
        # also why the class must not override ``for_run``. A caller's subclass arrives here
        # and replaces the default; it is never mounted alongside it.
        self._limit_recovery = limit_recovery or LimitRecoveryCapability()

        # The whole capability stack, assembled once, here — the only place its order is
        # decided. The budget is first of all, so a run it refuses reaches none of the
        # others and in particular never pays for compaction's summarizer; nothing but
        # this list holds that. Compaction is second so persistence opens its cursor on
        # the POST-fold history, though that one is belt-and-braces — _anchor re-opens the
        # cursor at the first node hook either way. The internal ones come first so the
        # caller's sit inside them: pydantic-ai unwinds the chain in reverse, so a caller
        # capability's after_* hooks run before the persistence sweep and its durable edits
        # are what gets persisted. Limit recovery sits immediately BEFORE healing, and that
        # position is load-bearing: pydantic-ai walks the on_run_error chain in REVERSE, so
        # the later entry fires first — healing must write its ToolReturnPart before the
        # recovery seam is consulted, or the conclusion would start from a context carrying
        # a dangling tool call.
        capability_stack: list[AgentCapability[Any]] = [
            self._budget,
            self._compactor,
            EventSourcingCapability(context=self._context),
            self._limit_recovery,
            HealingCapability(context=self._context),
            *(capabilities or []),
        ]

        # Create pydantic-ai Agent.
        # pydantic-ai's Agent() overloads declare `deps_type: type[AgentDepsT]
        # = object` (no `None`); ReactAgent forwards its own `deps_type:
        # type[Any] | None`, which the overload stubs reject even though the
        # runtime accepts it.
        self._pydantic_agent = Agent(
            model=self._model,
            tools=tools or [],
            toolsets=toolsets or [],
            retries=config.runtime_cfg.retries,
            deps_type=deps_type,  # type: ignore[arg-type]
            end_strategy=config.runtime_cfg.end_strategy,
            output_type=wrapped_result_type,
            capabilities=capability_stack,
        )

    async def run(
        self, user_prompt: UserPrompt, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Execute agent with REACT pattern.

        One awaited ``pydantic_ai.Agent.run()`` call. Context is still updated
        incrementally as the run produces messages, but that is
        ``EventSourcingCapability``'s ``after_node_run`` sweep, not this method.

        On a run-tier breach the mounted ``LimitRecoveryCapability`` is consulted:
        with the default policy the turn degrades into one tool-free conclusion and
        that conclusion's output is what this returns, instead of raising. A seam
        returning ``None`` restores the raising contract exactly.

        Args:
            user_prompt: User message to process
            deps: Optional dependency object (must match deps_type)
            output_type: Optional per-call output type override. When provided,
                wraps with get_output_type() for provider-aware structured output
                (NativeOutput for OpenAI/Anthropic, raw type for others).
                When None, uses result_type set at construction (default: str).

        Returns:
            Agent result output (type matches output_type if given, else result_type)

        Raises:
            AgentUsageLimitError: If the agent-lifetime budget (tokens or runs)
                rejects this call before it runs. Terminal for this agent.
            RunUsageLimitError: If pydantic-ai breaches a run-tier limit mid-run and
                the recovery seam declined to conclude — or the conclusion it asked
                for failed or produced nothing usable, in which case this carries the
                ORIGINAL breach, never the secondary failure.
            UsageLimitError: Base of both — catch this instead of the subclasses when
                the caller does not care which tier fired.
        """
        return await self._run_with_limits(
            user_prompt,
            deps,
            output_type,
            self._config.run_usage_limits,
            allow_recovery=True,
        )

    async def conclude_without_tools(
        self, reason: str, *, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Turn an interrupted turn into an answer: one follow-up run, no tools.

        **Mechanism only** — nothing here decides *whether* to conclude; that policy
        lives in ``LimitRecoveryCapability``'s seam, and ``run()`` drives this method
        when the seam asks for it. A direct call concludes unconditionally.

        **A conclusion is never itself recovered.** It goes through the same capability
        stack, so the recovery capability is mounted on it too — but this call leaves
        ``_run_with_limits``' recovery off, so a breach *during* a conclusion raises
        rather than starting a second one.

        The tools are removed with ``override(tools=[], toolsets=[])``, the only
        construct that *replaces* what is registered: a per-run ``toolsets=[]`` is
        documented as **additional** toolsets and would leave every tool in place, so
        the conclusion would happily call one. "Zero tool calls" is also not
        expressible as a limit — ``tool_calls_limit`` is ``gt=0``.

        The run carries its own ``RunUsageLimits(run_request_limit=1)`` rather than
        the budget that was just exhausted; with no tools available, one request is
        what the turn needs. It still goes through the normal agent-tier pre-flight,
        so an agent that has *also* spent its lifetime budget raises
        ``AgentUsageLimitError`` from here. That is the caller's signal to stop
        trying, not a defect to swallow — the lifetime counter is what bounds the
        retry loop by construction.

        Args:
            reason: Why the turn must conclude now. Reaches the model as the run's
                user prompt, on top of the healed context — so the
                ``ToolReturnPart`` written by ``HealingCapability`` is already there
                as the tool result the model reasons from.
            deps: Optional dependency object (must match deps_type).
            output_type: Optional per-call output type override (see ``run()``).

        Returns:
            Agent result output, exactly as ``run()`` returns it.

        Raises:
            AgentUsageLimitError: If the agent-lifetime budget rejects this call
                pre-flight. Terminal — do not retry.
            RunUsageLimitError: If even the single-request conclusion breaches a
                run-tier limit.
        """
        with self._pydantic_agent.override(tools=[], toolsets=[]):
            return await self._run_with_limits(
                reason, deps, output_type, RunUsageLimits(run_request_limit=1)
            )

    def conclude_without_tools_sync(
        self, reason: str, *, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Synchronous bridge over :meth:`conclude_without_tools`.

        Mirrors ``run_sync`` and ``compact``: closed-agent guard, then
        ``run_until_complete`` on the agent's own loop. There is deliberately no
        ``asyncio.run()`` fallback and no second loop — a fresh loop per call would
        leave pooled httpx connections attached to already-closed loops, making
        ``aclose()`` raise on stop and leaking the pool.

        Args:
            reason: Why the turn must conclude now (see :meth:`conclude_without_tools`).
            deps: Optional dependency object (must match deps_type).
            output_type: Optional per-call output type override (see ``run()``).

        Returns:
            Agent result output, exactly as :meth:`conclude_without_tools` returns it.

        Raises:
            RuntimeError: If the agent has been closed.
            UsageLimitError: Whatever :meth:`conclude_without_tools` raised.
        """
        if self._closed or self._loop.is_closed():
            raise RuntimeError("ReactAgent is closed")
        return self._loop.run_until_complete(
            self.conclude_without_tools(reason, deps=deps, output_type=output_type)
        )

    async def _run_with_limits(
        self,
        user_prompt: UserPrompt,
        deps: Any,
        output_type: type[Any] | None,
        limits: RunUsageLimits | None,
        *,
        allow_recovery: bool = False,
    ) -> Any:
        """Shared run core: fold the prompt, convert the run tier, one ``run()`` call.

        Everything ``run()`` does, with the run-tier budget as a parameter so the
        tool-free conclusion can substitute its own without ``run()`` growing a
        public knob for it. Reusing this core is what gives the conclusion the same
        pre-flight, the same auto-compaction, the same persistence, the same
        system-prompt recording and the same usage fold — so a conclusion emits an
        ``LlmUsageEvent`` like any other run.

        None of those five is performed here: they belong to the capability stack
        assembled in ``__init__``, and each fires inside the one awaited ``run()``
        below. What is left is the buffered-user-prompt fold, the run-tier limits
        conversion, that call, and the run-tier exception mapping.

        Args:
            user_prompt: The prompt for this turn.
            deps: Optional dependency object.
            output_type: Optional per-call output type override.
            limits: The run-tier budget to bound this turn with, or None.
            allow_recovery: Whether a run-tier breach may degrade into a tool-free
                conclusion when the mounted seam asks for one. ``run()`` passes
                ``True``; ``conclude_without_tools`` leaves it ``False``, which is
                the whole of the recursion guard — a conclusion is driven through
                this same core and must never start another one.

        Returns:
            Agent result output — the run's, or the conclusion's when a breach was
            recovered. ``run()`` either produces a result or raises, so there is no
            "no result" case to return.

        Raises:
            AgentUsageLimitError: Raised pre-flight by either agent-tier check.
            RunUsageLimitError: Raised when pydantic-ai breaches a run-tier limit and
                no conclusion was driven, or the conclusion failed.
            Exception: Anything else the run raised, propagating unchanged — the
                object and its ``__traceback__`` are the caller's, not this
                method's, to alter, because that is what
                ``Akgent._handle_failure`` formats onto ``ErrorMessage.traceback``.
        """
        user_prompt = self._fold_pending_user_prompts(user_prompt)
        pydantic_limits = self._to_pydantic_limits(limits)

        # Discard any decision left over from an earlier turn. The `except` below normally
        # consumes it, but a co-mounted capability may transform the breach into another
        # class on its way out, in which case that clause never fires and the stale decision
        # would drive a conclusion for THIS run's breach instead of the one it was made for.
        self._limit_recovery.consume_decision()

        # No `usage=`: the run's own accumulator is the one pydantic-ai's graph creates,
        # and LifetimeBudgetCapability folds it from `wrap_run`'s ctx. Handing in the
        # lifetime accumulator instead would check the per-run cap against lifetime
        # totals — silently, with no error.
        try:
            result = await self._pydantic_agent.run(
                user_prompt=user_prompt,
                deps=deps,
                usage_limits=pydantic_limits,
                message_history=self._context.messages,
                output_type=get_output_type(self._config.model_cfg, output_type),
            )
            return result.output
        except UsageLimitExceeded as e:
            decision = self._limit_recovery.consume_decision()
            if not allow_recovery or decision is None:
                raise RunUsageLimitError(str(e)) from e
            return await self._conclude_after_breach(
                decision, e, deps=deps, output_type=output_type
            )

    async def _conclude_after_breach(
        self,
        decision: ConclusionDecision,
        breach: UsageLimitExceeded,
        *,
        deps: Any,
        output_type: type[Any] | None,
    ) -> Any:
        """Drive the conclusion the seam asked for, or surface the ORIGINAL breach.

        Escalation parity with what ``akgentic-agent`` does today: **any** failure of the
        attempt — a second run-tier breach, the terminal ``AgentUsageLimitError`` from the
        conclusion's own pre-flight, anything else, or an output nothing can be done with —
        falls through to exactly the behaviour a declined recovery would have produced. The
        caller sees the original breach and never the secondary one, which would otherwise
        replace a "this turn ran out of budget" signal with an unrelated failure.

        "Nothing usable" is deliberately narrow: ``None``, or a ``str`` that is empty or
        whitespace-only. Richer emptiness — a structured output carrying no requests — is
        the caller's judgement and stays out of this package.

        ``deps`` and ``output_type`` are threaded verbatim from the breached call, so the
        conclusion produces the same shape the caller asked for and its structured output
        routes downstream through the caller's normal path.

        Args:
            decision: What the seam decided; its ``reason`` is the conclusion's prompt.
            breach: The original run-tier breach, and the only one the caller may see.
            deps: The breached call's dependency object.
            output_type: The breached call's per-call output type override.

        Returns:
            The conclusion's output.

        Raises:
            RunUsageLimitError: Built from ``breach`` whenever the conclusion did not
                produce a usable answer.
        """
        try:
            output = await self.conclude_without_tools(
                decision.reason, deps=deps, output_type=output_type
            )
        except Exception:
            logger.exception("Tool-free conclusion failed; surfacing the original breach")
            raise RunUsageLimitError(str(breach)) from breach

        if output is None or (isinstance(output, str) and not output.strip()):
            logger.warning("Tool-free conclusion produced no usable output; surfacing the breach")
            raise RunUsageLimitError(str(breach)) from breach
        return output

    @property
    def _agent_run_count(self) -> int:
        """Runs consumed over this agent's lifetime, read through to the capability.

        One source of truth: ``LifetimeBudgetCapability`` owns the counter, enforces it
        and is reseeded by ``restore_context``. This reports it.
        """
        return self._budget.run_count

    @property
    def _agent_usage(self) -> RunUsage:
        """This agent's lifetime token accumulator, read through to the capability."""
        return self._budget.usage

    def _fold_pending_user_prompts(self, user_prompt: UserPrompt) -> UserPrompt:
        """Prepend any user prompts buffered before the first run to the run prompt.

        Drains ``ContextManager._pending_user_prompts`` and folds the entries
        into ``user_prompt`` so they reach the model **as prompt content** rather
        than as a system-less ``ModelRequest`` in ``message_history`` — which
        would make pydantic-ai's history non-empty and suppress system-prompt
        injection on the first run. ``message_history`` is therefore left
        untouched (empty before the first run), preserving injection.

        Folding is per prompt shape:

        - ``str`` prompt → ``f"{preamble}\\n\\n{user_prompt}"``;
        - multimodal ``list`` prompt → the joined ``preamble`` inserted as the
          leading element.

        When nothing is buffered the prompt is returned unchanged.

        Args:
            user_prompt: The caller's prompt for this run (str or multimodal list).

        Returns:
            The prompt with the buffered entries prepended, or the original
            prompt when the buffer is empty.
        """
        pending = self._context.drain_pending_user_prompts()
        if not pending:
            return user_prompt
        preamble = "\n\n".join(pending)
        if isinstance(user_prompt, str):
            return f"{preamble}\n\n{user_prompt}"
        return [preamble, *user_prompt]

    @property
    def _compaction(self) -> CompactionStrategy:
        """The resolved compaction strategy, read through to the mounted capability.

        One source of truth: ``CompactionCapability`` holds the strategy and is what
        actually calls it. Deliberately read-only — a test that swapped a strategy onto
        ``ReactAgent`` instead of onto the capability would be testing nothing, silently,
        so the assignment raises instead. Swap ``agent._compactor.strategy``.
        """
        return self._compactor.strategy

    def _compaction_threshold(self) -> int | None:
        """Token budget that arms the auto-trigger, or None when compaction is off.

        ``int(context_length * trigger_ratio)`` when auto-compaction is on and
        ``model_cfg.context_length`` is set; ``None`` (auto-compaction disabled)
        otherwise. Both ways of being off — the switch and the missing budget — answer
        ``None`` here so "compaction is off" is one concept the capability reads once.
        The arithmetic is unchanged.
        """
        cfg = self._config.compaction_cfg
        context_length = self._config.model_cfg.context_length
        if not cfg.auto_trigger or context_length is None:
            return None
        return int(context_length * cfg.trigger_ratio)

    def _build_compaction_event(self, result: CompactionResult) -> LlmContextCompactedEvent:
        """Build the append-only event from a strategy result (auto + manual share this)."""
        cfg = self._config.compaction_cfg
        return LlmContextCompactedEvent(
            run_id=None,
            strategy_id=cfg.strategy,
            summary=result.summary,
            replaced_message_count=result.replaced_message_count,
            summarizer_prompt_version=cfg.summarizer_prompt_version,
            tokens_before=self._context.last_input_tokens,
            tokens_after=result.tokens_after,
        )

    async def _compact_now(self) -> str:
        """Force a compaction now — the manual half of the one fold site (FR5).

        Delegates to ``CompactionCapability.compact_now`` with no live message list:
        there is no run in flight on this path, so the durable write is the only one and
        the next run picks the folded history up from ``ContextManager``. The auto path
        reaches the same method from the capability's own ``wrap_run``, which is what
        keeps the two from diverging.

        Returns:
            A human-readable status string.
        """
        return await self._compactor.compact_now()

    def compact(self) -> str:
        """Force a compaction now, bypassing the budget gate (manual /compact, FR15).

        Synchronous bridge for the slash-command path: the agent-owned loop is idle
        at dispatch time, so ``run_until_complete`` is legal here (unlike the auto
        path, which awaits inside the running loop). Mirrors ``run_sync``'s
        closed-agent guard.

        Returns:
            A human-readable status string.

        Raises:
            RuntimeError: If the agent has been closed.
        """
        if self._closed or self._loop.is_closed():
            raise RuntimeError("ReactAgent is closed")
        return self._loop.run_until_complete(self._compact_now())

    def clear_context(self) -> str:
        """Wipe the conversation; the system prompt regenerates next run (/clear, FR15).

        Pure synchronous wrapper over ``ContextManager.clear_context`` — no
        summarizer, no ``run_until_complete``, no loop interaction.

        Returns:
            A human-readable status string.
        """
        removed = self._context.clear_context()
        return f"Cleared {removed} message(s); system prompt regenerates on the next run."

    def run_sync(
        self, user_prompt: UserPrompt, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Execute agent synchronously.

        Convenience wrapper around run() for synchronous contexts.

        Args:
            user_prompt: User message to process
            deps: Optional dependency object
            output_type: Optional per-call output type override (see run()).

        Returns:
            Agent result output

        Raises:
            RuntimeError: If the agent has been closed
            UsageLimitError: If usage limits are exceeded. run_sync() surfaces
                whatever run() raised, so either subclass can arrive here:
                RunUsageLimitError for a run-tier breach, AgentUsageLimitError for
                an agent-tier one.
        """
        # Always run on the agent's own loop so the httpx connection pool stays
        # bound to ONE stable loop across calls. There is no asyncio.run()
        # fallback: a fresh loop per call would leave pooled connections attached
        # to already-closed loops, making aclose() raise on stop and leaking the
        # pool (RAM grows per team).
        if self._closed or self._loop.is_closed():
            raise RuntimeError("ReactAgent is closed")
        return self._loop.run_until_complete(self.run(user_prompt, deps, output_type))

    async def aclose(self) -> None:
        """Release async resources (the httpx connection pool); does NOT close the loop.

        Resource-only teardown driven by ``close()`` on a still-open loop. The
        pydantic-ai Model (and its provider) hold this client, so without
        ``aclose()`` its open sockets and connection pool survive team stop and
        accumulate in the worker. Guarded so a second call is harmless.
        """
        if not self._http_client.is_closed:
            await self._http_client.aclose()

    def close(self) -> None:
        """Synchronously tear the agent down; idempotent.

        Drives async resource teardown (``aclose()``) on the still-open loop,
        cancels stragglers, drains async generators and the default executor,
        closes the loop, then evicts anyio's per-loop run-vars anchor — all in
        ``finally``. The five-step order is load-bearing: async exit handlers
        must run before the loop closes, and stragglers / async generators must
        drain before ``loop.close()`` to avoid leaked transports and "Task was
        destroyed but it is pending" warnings; step 5 (the eviction) runs after
        ``loop.close()`` so the closed loop can actually be GC'd (the
        ``_root_task`` ``RunVar`` would otherwise pin it). A second call is a
        harmless no-op; teardown failures are logged, never raised.
        """
        loop = self._loop
        if self._closed or loop.is_closed():
            return
        self._closed = True
        try:
            if not loop.is_running():
                loop.run_until_complete(self.aclose())  # 1. async resource teardown
                self._cancel_pending(loop)  # 2. cancel stragglers
                loop.run_until_complete(loop.shutdown_asyncgens())  # 3. drain async gens
                loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            logger.warning("ReactAgent.close() teardown failed", exc_info=True)
        finally:
            loop.close()  # 4. close the loop
            _evict_anyio_run_vars(loop)  # 5. break anyio's _root_task → loop anchor

    @staticmethod
    def _cancel_pending(loop: asyncio.AbstractEventLoop) -> None:
        """Cancel and await any tasks still pending on ``loop`` before close.

        Mirrors ``Akgent._drain_event_loop``'s straggler step: cancel each
        pending task, then run the loop once to let the cancellations settle.
        """
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if not pending:
            return
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def _to_pydantic_limits(self, limits: RunUsageLimits | None) -> PydanticUsageLimits | None:
        """Convert the config's run tier to pydantic-ai UsageLimits.

        The agent tier is never consulted here: pydantic-ai bounds one run, and the
        agent-lifetime budget is enforced outside the per-run conversion.

        Args:
            limits: Config run-scoped usage limits or None

        Returns:
            Pydantic-ai usage limits or None
        """
        if limits is None:
            return None

        return PydanticUsageLimits(
            request_limit=limits.run_request_limit,
            tool_calls_limit=limits.tool_calls_limit,
            input_tokens_limit=limits.input_tokens_limit,
            output_tokens_limit=limits.output_tokens_limit,
            total_tokens_limit=limits.total_tokens_limit,
        )

    # API wrapper methods

    @property
    def context(self) -> ContextManager:
        """Get context manager for message history access.

        Returns:
            Context manager instance
        """
        return self._context

    def subscribe_context(self, observer: ContextObserver) -> None:
        """Subscribe to context change notifications.

        Args:
            observer: Observer implementing ContextObserver protocol
        """
        self._context.subscribe(observer)

    def restore_context(self, events: Sequence[EventMessage]) -> None:
        """Restore LLM conversation context as an ordered fold over persisted events.

        Folds ``events`` in persisted-sequence order into an accumulator:
        ``LlmMessageEvent`` appends its message; ``LlmContextCompactedEvent``
        applies the **same** mechanical fold as the live ``compact()`` (via the
        shared ``ContextManager.fold_compaction`` helper, **without** notify);
        ``LlmContextClearedEvent`` resets the accumulator to empty (**without**
        notify). Non-matching events (e.g. ``ToolCallEvent``, arbitrary objects)
        are ignored. The final accumulator is bulk-restored once (observers never
        fire during replay), then the dedup hash is seeded from the latest
        ``LlmSystemPromptEvent``. Sharing the fold with the live path keeps live
        and replayed contexts byte-identical, and two sequential compaction events
        compose (the later fold consumes the earlier synthetic summary) with the
        fold applied exactly once per event (FR16).

        The same list also reseeds both agent-lifetime budgets — the run counter
        and the token accumulator — so a resumed agent carries the budget it
        already spent instead of a fresh one (ADR-013 §D3).

        Accepts a ``Sequence`` rather than a ``list`` because ``list`` is
        **invariant**: a caller holding a ``list`` of its own envelope type
        cannot pass it here even when that type satisfies the Protocol.
        ``akgentic-agent`` hit exactly that — it forwards
        ``list[akgentic.core.messages.EventMessage]`` and needed a
        ``type: ignore`` to do it. ``Sequence`` is covariant, so the call
        type-checks. Only iteration and ``reversed()`` are used, both of which
        ``Sequence`` provides.

        Args:
            events: Sequence of event-like objects (typically ``EventMessage``
                instances from ``akgentic-core``). Each is expected to carry a
                ``.event`` payload.
        """
        messages: list[ModelMessage] = []
        for e in events:
            payload = e.event
            if isinstance(payload, LlmMessageEvent):
                messages.append(payload.message)
            elif isinstance(payload, LlmContextCompactedEvent):
                messages = ContextManager.fold_compaction(messages, payload)
            elif isinstance(payload, LlmContextClearedEvent):
                messages = []
        self._context.restore(messages)
        self._seed_system_prompt_from_events(events)
        self._seed_agent_budget_from_events(events)

    def _seed_agent_budget_from_events(self, events: Sequence[EventMessage]) -> None:
        """Recompute both agent-lifetime budgets from replayed usage events, and seed them.

        One ``aggregate_usage`` pass seeds the run counter and the token
        accumulator: the run count is the number of **distinct runs**, never of
        events (one ``run()`` emits one ``LlmUsageEvent`` per ``ModelResponse``, so
        a run with three tool-call round-trips contributes three events under one
        ``run_id``), and the token totals sum those same runs. ``by_run=True`` is
        what populates ``runs`` at all — without it both seeds are zero every time.

        Both values are handed to ``LifetimeBudgetCapability.seed()``, which **assigns**
        rather than accumulates — so replaying the same stream twice is idempotent and a
        shorter stream lowers the value. A ``run()`` that failed before any
        ``ModelResponse`` left no usage event and is invisible here; that is deliberate
        (ADR-013 §Out of scope) — it consumed no model resources.

        Args:
            events: The same event-like list passed to ``restore_context``.
        """
        usage = [e.event for e in events if isinstance(e.event, LlmUsageEvent)]
        summary = aggregate_usage(usage, by_run=True)
        self._budget.seed(
            run_count=len(summary.runs),
            usage=RunUsage(
                input_tokens=sum(r.total_input_tokens for r in summary.runs),
                output_tokens=sum(r.total_output_tokens for r in summary.runs),
            ),
        )

    def _seed_system_prompt_from_events(self, events: Sequence[EventMessage]) -> None:
        """Seed the dedup hash from the latest persisted ``LlmSystemPromptEvent``.

        Scans ``events`` for the **latest** ``LlmSystemPromptEvent`` (the last in
        append/persist order) and seeds ``ContextManager._last_system_prompt_hash``
        from its ``content_hash`` via ``seed_system_prompt_hash`` — without firing
        observers — so a restored agent does not re-emit an unchanged rendering.
        When no such event is present (e.g. an older team), the dedup state is left
        at its current ``None`` so the next run emits the ``None → hash`` transition
        (ADR-004 §3). Additive to the message-restore scan; the same guard style is
        reused.

        Args:
            events: The same event-like list passed to ``restore_context``.
        """
        latest = next(
            (e.event for e in reversed(events) if isinstance(e.event, LlmSystemPromptEvent)),
            None,
        )
        if latest is not None:
            self._context.seed_system_prompt_hash(latest.content_hash)

    def system_prompt[F: Callable[..., Any]](self, func: F) -> F:
        """Register a custom dynamic system prompt.

        Convenience wrapper around pydantic-ai's @agent.system_prompt(dynamic=True).

        Example:
            >>> @agent.system_prompt
            >>> def my_prompt(ctx):
            ...     return f"Context: {ctx.deps.get_context()}"

        Args:
            func: System prompt function

        Returns:
            The same function, with its own type preserved rather than erased.
        """
        # cast, not `return func`: `_pydantic_agent` is Any-typed, so the call's
        # result is Any (warn_return_any). pydantic-ai hands back the original
        # function object, so the cast is a type-level claim only — the runtime
        # value stays whatever pydantic-ai returned.
        return cast(F, self._pydantic_agent.system_prompt(dynamic=True)(func))

    def tool[F: Callable[..., Any]](self, func: F) -> F:
        """Register a tool function.

        Convenience wrapper around pydantic-ai's @agent.tool().

        Generic over the decorated function so its signature survives
        registration: pydantic-ai's own ``tool()`` needs a stack of overloads to
        avoid erasing it, and a wrapper returning ``Any`` would throw that away
        again — leaving every caller of a registered tool unchecked.

        Example:
            >>> @agent.tool
            >>> def search(query: str) -> list[str]:
            ...     return search_database(query)

        Args:
            func: Tool function

        Returns:
            The same function, with its own type preserved rather than erased.
        """
        # cast for the same reason as system_prompt() above.
        return cast(F, self._pydantic_agent.tool(func))

    @property
    def pydantic_agent(self) -> Agent[Any, Any]:
        """Access underlying pydantic-ai Agent for advanced usage.

        Use for features not wrapped by ReactAgent:
        - result_validator()
        - on_error()
        - Direct decorator access

        Returns:
            Pydantic-ai Agent instance
        """
        return self._pydantic_agent
