"""REACT-based LLM agent with context management and iteration support."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from pydantic_ai import Agent, AgentCapability, BinaryContent, UsageLimitExceeded
from pydantic_ai import UsageLimits as PydanticUsageLimits
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

# RUN_LIMIT_HEALING_MESSAGE is re-exported, not used: healing itself moved into
# HealingCapability, but `akgentic.llm.agent.RUN_LIMIT_HEALING_MESSAGE` stays importable
# for callers written against it. capabilities.py holds the one definition.
from .capabilities import (
    RUN_LIMIT_HEALING_MESSAGE as RUN_LIMIT_HEALING_MESSAGE,
)
from .capabilities import EventSourcingCapability, HealingCapability
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


class UsageLimitError(Exception):
    """Raised when a usage limit is exceeded during agent execution.

    Base of both tiers — catch this to handle either; catch a subclass to react to
    one. Every breach raises one of the two subclasses below, never this class
    directly, but it stays the documented catch-all: an ``except UsageLimitError``
    written before the tiers were split still catches everything it used to
    (ADR-016 §D1).
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
                They are NOT forwarded unchanged: two internal capabilities —
                EventSourcingCapability then HealingCapability — are mounted ahead of
                them, because those two own persistence, system-prompt recording and
                dangling-tool-call healing for every run this agent drives.
                Ordering is fixed: a capability's before_model_request hook runs AFTER
                compaction — ContextManager rewrites messages first, the result is
                passed as message_history, and only then does the capability chain
                run. Two consequences, neither guessable from the signature: a
                capability sees only the POST-compaction history, never what
                compaction folded away; and because pydantic-ai unwinds the chain in
                reverse, a caller capability's `after_*` hooks run BEFORE the
                persistence sweep, so its durable edits are the ones persisted.
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

        # Agent-lifetime run counter backing agent_usage_limits.agent_request_limit.
        # In memory only: never a Pydantic field, never persisted. Not lost on resume —
        # restore_context() recomputes it from the replayed LlmUsageEvents.
        self._agent_run_count: int = 0

        # Agent-lifetime token accumulator backing agent_usage_limits' token fields.
        # Same lifecycle as _run_count (in memory, reseeded from the same replayed
        # events). pydantic-ai's own RunUsage, so folding and comparison are both its
        # code. NEVER handed to run(usage=…) — see _check_agent_token_budget.
        self._agent_usage: RunUsage = RunUsage()

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

        # Resolve the compaction strategy as runtime state (never a Pydantic
        # field). Built AFTER self._http_client so the summarizer reuses the
        # agent's shared httpx client (no second pool); the summarizer model uses
        # summary_model_cfg when set, else the primary model_cfg.
        summary_cfg = config.compaction_cfg.summary_model_cfg or config.model_cfg
        self._compaction: CompactionStrategy = create_compaction(
            config.compaction_cfg, summary_cfg, self._http_client
        )

        # Create model from config
        self._model = create_model(config.model_cfg, self._http_client)

        # Wrap result_type with provider-aware output strategy for structured output
        wrapped_result_type: Any = get_output_type(config.model_cfg, result_type)

        # The whole capability stack, assembled once, here. The two internal ones come
        # first so the caller's sit inside them: pydantic-ai unwinds the chain in
        # reverse, so a caller capability's after_* hooks run before the persistence
        # sweep and its durable edits are what gets persisted.
        capability_stack: list[AgentCapability[Any]] = [
            EventSourcingCapability(context=self._context),
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
            RunUsageLimitError: If pydantic-ai breaches a run-tier limit mid-run.
                Recoverable — the agent may still have lifetime budget.
            UsageLimitError: Base of both — catch this instead of the subclasses when
                the caller does not care which tier fired.
        """
        return await self._run_with_limits(
            user_prompt, deps, output_type, self._config.run_usage_limits
        )

    async def conclude_without_tools(
        self, reason: str, *, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Turn an interrupted turn into an answer: one follow-up run, no tools.

        **Mechanism only** — nothing here decides *whether* to conclude; that policy
        lives in the caller (ADR-016 §D3/§D4). ``run()`` never calls this itself and
        keeps raising on every breach.

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
    ) -> Any:
        """Shared run core: pre-flight, compaction, one ``run()`` call.

        Everything ``run()`` does, with the run-tier budget as a parameter so the
        tool-free conclusion can substitute its own without ``run()`` growing a
        public knob for it. Reusing this core is what gives the conclusion the same
        pre-flight, the same persistence, the same system-prompt recording and the
        same usage fold — so a conclusion emits an ``LlmUsageEvent`` like any other
        run.

        Persistence, system-prompt recording and healing are not performed here at
        all: they belong to ``EventSourcingCapability`` and ``HealingCapability``,
        mounted ahead of the caller's capabilities in ``__init__``. What is left is a
        pre-flight and one awaited call.

        Args:
            user_prompt: The prompt for this turn.
            deps: Optional dependency object.
            output_type: Optional per-call output type override.
            limits: The run-tier budget to bound this turn with, or None.

        Returns:
            Agent result output. ``run()`` either produces a result or raises, so
            there is no "no result" case to return.

        Raises:
            AgentUsageLimitError: Raised pre-flight by either agent-tier check.
            RunUsageLimitError: Raised when pydantic-ai breaches a run-tier limit.
            Exception: Anything else the run raised, propagating unchanged — the
                object and its ``__traceback__`` are the caller's, not this
                method's, to alter, because that is what
                ``Akgent._handle_failure`` formats onto ``ErrorMessage.traceback``.
        """
        # Pre-flight: reject before spending anything (a rejected run must not even
        # pay for compaction's summarizer call). Tokens first, so a token rejection
        # consumes no run budget.
        self._check_agent_token_budget()
        self._check_and_consume_agent_budget()
        # Auto-compact (at most once per turn) BEFORE the run reads the history.
        # message_history is read once at entry and the graph only appends, so
        # compacting here takes effect for the whole turn.
        await self._maybe_compact()
        user_prompt = self._fold_pending_operator_actions(user_prompt)
        pydantic_limits = self._to_pydantic_limits(limits)

        # This run's own accumulator, mutated in place by the graph for the whole run.
        # A fresh object, never self._agent_usage: handing pydantic-ai the lifetime
        # accumulator would check the per-run cap against lifetime totals — silently,
        # with no error.
        run_usage = RunUsage()
        try:
            result = await self._pydantic_agent.run(
                user_prompt=user_prompt,
                deps=deps,
                usage_limits=pydantic_limits,
                message_history=self._context.messages,
                output_type=get_output_type(self._config.model_cfg, output_type),
                usage=run_usage,
            )
            return result.output
        except UsageLimitExceeded as e:
            raise RunUsageLimitError(str(e)) from e
        finally:
            # In `finally`: tokens a failed run burned were still burned.
            self._fold_run_usage(run_usage)

    def _check_agent_token_budget(self) -> None:
        """Refuse to START a run once the agent-lifetime token budget is spent.

        Builds a pydantic-ai ``UsageLimits`` from ``agent_usage_limits``' three token
        fields and reuses its ``check_tokens()`` against the lifetime accumulator, so
        an agent-tier breach carries pydantic-ai's own message wording. The tier is
        carried by the **class** — ``AgentUsageLimitError`` here, ``RunUsageLimitError``
        at the run-tier site — so nothing downstream has to parse text to tell the
        tiers apart. Unset limits (the default, all ``None``) make the check a no-op.

        **A run may overshoot the budget, by construction.** A run's token cost is
        unknown until it completes, so this is "do not start a run once the budget is
        spent", never "never exceed it": the last run admitted can carry the total
        arbitrarily past the limit, and only the next one is refused.

        The accumulator is compared here rather than handed to ``run(usage=…)``
        because a run takes exactly one usage object — passing the lifetime total
        would check the *run* tier's limits against it and silently turn a per-run
        cap into a lifetime one (ADR-013 §Out of scope, reopened for the token
        tier).

        Raises:
            AgentUsageLimitError: If lifetime usage has already exceeded a token
                limit. A subclass of UsageLimitError.
        """
        limits = self._config.agent_usage_limits
        pydantic_limits = PydanticUsageLimits(
            input_tokens_limit=limits.input_tokens_limit,
            output_tokens_limit=limits.output_tokens_limit,
            total_tokens_limit=limits.total_tokens_limit,
        )
        try:
            pydantic_limits.check_tokens(self._agent_usage)
        except UsageLimitExceeded as e:
            raise AgentUsageLimitError(str(e)) from e

    def _fold_run_usage(self, run_usage: RunUsage) -> None:
        """Add one run's token usage to the agent-lifetime accumulator.

        The argument is the ``RunUsage`` handed **in** as ``run(usage=…)``, not one
        read back off a result. pydantic-ai defaults that parameter (``usage or
        RunUsage()``), stores that exact object on the graph state and from then on
        only mutates it in place, so it holds the run's real cost whether the run
        returned or raised. That is why it is passed in at all: a run that failed
        partway has no ``AgentRunResult`` to read usage off, so the accumulator the
        graph mutates is the only anchor that survives the exception. Called from a
        ``finally``, so a failed run still contributes what it spent — the provider
        billed it either way.

        Args:
            run_usage: This run's own accumulator, as handed to ``run(usage=…)``.
        """
        self._agent_usage.incr(run_usage)

    def _check_and_consume_agent_budget(self) -> None:
        """Spend one unit of the agent-lifetime run budget, or refuse to run.

        Check-then-consume: the counter advances **before** the call executes, so a
        ``run()`` that fails partway — including one that raises the run-tier
        ``RunUsageLimitError`` — has already been counted. That ordering is deliberate:
        an agent whose run-tier limit fires repeatedly must also exhaust its
        agent-tier budget, since both mean "this agent is burning too many turns"
        (ADR-013 §D2). Do not move the increment after the call.

        The rejection itself does not consume, so the counter reports runs consumed,
        never runs attempted, and the message stays stable under repeated rejection.
        ``agent_request_limit=None`` never blocks.

        Raises:
            AgentUsageLimitError: If the agent has already used its lifetime run
                budget. A subclass of UsageLimitError.
        """
        limit = self._config.agent_usage_limits.agent_request_limit
        if limit is not None and self._agent_run_count >= limit:
            raise AgentUsageLimitError(
                f"Exceeded the agent_request_limit of {limit} (run_count={self._agent_run_count})"
            )
        self._agent_run_count += 1

    def _fold_pending_operator_actions(self, user_prompt: UserPrompt) -> UserPrompt:
        """Prepend any buffered pre-first-run operator actions to the run prompt.

        Drains ``ContextManager._pending_operator_actions`` and folds the entries
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
            The prompt with buffered operator actions prepended, or the original
            prompt when the buffer is empty.
        """
        pending = self._context.drain_pending_operator_actions()
        if not pending:
            return user_prompt
        preamble = "\n\n".join(pending)
        if isinstance(user_prompt, str):
            return f"{preamble}\n\n{user_prompt}"
        return [preamble, *user_prompt]

    def _compaction_threshold(self) -> int | None:
        """Token budget that arms the auto-trigger, or None when compaction is off.

        ``int(context_length * trigger_ratio)`` when ``model_cfg.context_length``
        is set; ``None`` (auto-compaction disabled) otherwise.
        """
        context_length = self._config.model_cfg.context_length
        if context_length is None:
            return None
        return int(context_length * self._config.compaction_cfg.trigger_ratio)

    async def _maybe_compact(self) -> None:
        """Auto-compact once when the last run's input_tokens exceed the threshold.

        Reads the provider-reported ``last_input_tokens`` (no ``tiktoken``). No-ops
        when auto-trigger is off, the budget is unset (``context_length is None``),
        no usage has been reported yet (``last_input_tokens is None`` — never
        mis-fires on missing data), or usage is at/below the threshold. Fires at
        most once per ``run()`` call.
        """
        cfg = self._config.compaction_cfg
        if not cfg.auto_trigger:
            return
        threshold = self._compaction_threshold()
        used = self._context.last_input_tokens
        if threshold is None or used is None or used <= threshold:
            return
        await self._compact_now()

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
        """Run the strategy and fold its result in — the shared async core (R1 + R2).

        Awaits the strategy inside the running loop (ADR-009). A zero-replacement
        result is a clean no-op: no event, no synthetic summary. Both the auto path
        (``_maybe_compact``) and the sync ``compact()`` bridge converge here.

        Returns:
            A human-readable status string.
        """
        result = await self._compaction.compact(self._context.messages)
        if result.replaced_message_count == 0:
            return "Nothing to compact."
        self._context.compact(self._build_compaction_event(result))
        return (
            f"Compacted: replaced {result.replaced_message_count} "
            f"earlier message(s) with a summary."
        )

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
        """Recompute both agent-lifetime budgets from replayed usage events.

        One ``aggregate_usage`` pass seeds the run counter and the token
        accumulator: the run count is the number of **distinct runs**, never of
        events (one ``run()`` emits one ``LlmUsageEvent`` per ``ModelResponse``, so
        a run with three tool-call round-trips contributes three events under one
        ``run_id``), and the token totals sum those same runs. ``by_run=True`` is
        what populates ``runs`` at all — without it both seeds are zero every time.

        Assignment, not accumulation, so replaying the same stream twice is
        idempotent and a shorter stream lowers the value. A ``run()`` that failed
        before any ``ModelResponse`` left no usage event and is invisible here; that
        is deliberate (ADR-013 §Out of scope) — it consumed no model resources.

        Args:
            events: The same event-like list passed to ``restore_context``.
        """
        usage = [e.event for e in events if isinstance(e.event, LlmUsageEvent)]
        summary = aggregate_usage(usage, by_run=True)
        self._agent_run_count = len(summary.runs)
        self._agent_usage = RunUsage(
            input_tokens=sum(r.total_input_tokens for r in summary.runs),
            output_tokens=sum(r.total_output_tokens for r in summary.runs),
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
