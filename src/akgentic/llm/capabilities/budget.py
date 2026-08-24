"""``LifetimeBudgetCapability`` — the agent-lifetime run count and token caps.

See the package docstring for the hook anchors, composition order and cursor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits as PydanticUsageLimits
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ..config import AgentUsageLimits
from .errors import AgentUsageLimitError


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
