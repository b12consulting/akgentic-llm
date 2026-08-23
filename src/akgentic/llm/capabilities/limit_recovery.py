"""``LimitRecoveryCapability`` — decide whether a run-tier breach degrades into an answer.

See the package docstring for the hook anchors, composition order and cursor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

# What the MODEL reads as the prompt of the tool-free conclusion. Adapted from the wording
# ``akgentic-agent`` uses today, with the requester dropped: this package has no notion of a
# requester, and inventing a placeholder would be worse than omitting it (in agent-land an
# unprefixed name is a role to hire). A deployment that wants the requester named returns its
# own ``ConclusionDecision(reason=...)`` from the seam — that is what the seam is for.
DEFAULT_CONCLUSION_REASON = (
    "This turn has run out of its tool-call budget, so you cannot call any further tool "
    "and this is your last chance to answer.\n"
    "Answer now with what you have already gathered. State your conclusion plainly, say "
    "explicitly which parts you could not check or finish, and do not promise follow-up "
    "work — the turn ends with this answer."
)


class ConclusionDecision(BaseModel):
    """The seam's answer to "how should this turn degrade?": conclude, with this prompt.

    Deliberately a model rather than a bare ``str``, so a deployment's policy can grow a
    second dimension (a different budget, a different output shape) without changing the
    seam's signature. Today it carries the prompt and nothing else.
    """

    reason: str = DEFAULT_CONCLUSION_REASON
    """The prompt the tool-free conclusion is started with, on top of the healed context."""


@dataclass
class LimitRecoveryCapability(AbstractCapability[Any]):
    """Decide, on a run-tier breach, whether the turn concludes instead of raising.

    Mountable on any pydantic-ai ``Agent``; it needs nothing from ``ReactAgent``. It
    **decides** and records; it never runs anything. The conclusion itself is a *sibling*
    run driven by whoever mounted this — ``ReactAgent._run_with_limits`` in this package —
    so the recovery never nests a run inside a capability hook.

    **``on_run_error``, never ``wrap_run``.** pydantic-ai gives error hooks their chance only
    once the exception has escaped the entire ``wrap_run`` chain. A capability that caught the
    breach in its own ``wrap_run`` and returned a recovery result would therefore stop
    ``HealingCapability.on_run_error`` from ever running, and the conclusion would start from a
    context still carrying a **dangling tool call** — the exact failure healing exists to
    prevent. The ``wrap_run`` shape is simpler *and broken*, and the suite stays green while it
    is. This class defines no ``wrap_run`` at all.

    **It always re-raises.** ``on_run_error`` may return an ``AgentRunResult`` to suppress the
    error; this one never does. Suppressing it here would make the run tier unobservable and
    would hand the mounter a result it never asked for, when what it needs is the *decision*.

    **Mount it immediately before ``HealingCapability``.** ``CombinedCapability.on_run_error``
    walks ``reversed(self.capabilities)``, so the **last** capability in the mount list fires
    **first**: healing must sit *after* this one to write its ``ToolReturnPart`` before the
    seam is consulted. A capability that raises does not stop the walk — its exception is
    carried into the next capability as ``error`` — so both hooks run either way; only their
    order depends on the list.

    **No ``for_run`` override, deliberately.** pydantic-ai's default hands back ``self``, so
    the object whose hook records the decision **is** the object the mounter holds and reads
    it back off. A ``for_run`` returning a copy — the shape ``EventSourcingCapability`` needs
    for its per-run cursor — would make every decision invisible and recovery would silently
    never happen.
    """

    _decision: ConclusionDecision | None = field(default=None, init=False)
    """The decision this run's breach produced, until the mounter consumes it.

    In memory only, and read-and-clear: see :meth:`consume_decision`.
    """

    async def on_run_error(
        self,
        ctx: RunContext[Any],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Record what the seam decides about a run-tier breach, then re-raise ``error``.

        The declared return type is the hook's; this implementation never returns on any
        path. Anything that is not a ``UsageLimitExceeded`` — including this package's own
        ``AgentUsageLimitError``, which is a different class — passes straight through
        without consulting the seam. The exception and its ``__traceback__`` reach the caller
        untouched, exactly as ``HealingCapability``'s do.
        """
        if not isinstance(error, UsageLimitExceeded):
            raise error
        self._decision = await self.handle_limit_exceeded(ctx, error=error)
        raise error

    async def handle_limit_exceeded(
        self,
        ctx: RunContext[Any],
        *,
        error: UsageLimitExceeded,
    ) -> ConclusionDecision | None:
        """Decide how to degrade this turn. Default: one tool-free conclusion.

        **This is the seam.** Override it in a subclass and mount that subclass through the
        agent's ``limit_recovery=`` keyword to carry a deployment's own policy — a different
        prompt, or no conclusion at all for some breaches.

        Returning ``None`` re-raises unchanged, which reproduces the behaviour this package
        had before recovery existed: the breach surfaces as a run-tier error and nothing else
        happens. That is the opt-out, and it is exact rather than approximate.

        Args:
            ctx: The failed run's context, for a policy that reads ``deps`` or usage.
            error: The run-tier breach pydantic-ai raised.

        Returns:
            The conclusion to drive, or ``None`` to let the breach surface.
        """
        return ConclusionDecision()

    def consume_decision(self) -> ConclusionDecision | None:
        """Return the recorded decision and clear it; a second call returns ``None``.

        Read-and-clear rather than a plain read, because a decision that outlived its own
        turn would drive a conclusion for a *later* run's breach. The mounter also consumes
        (and discards) before each run, which is what covers the case where the breach never
        reaches its ``except`` clause — a co-mounted capability may transform it into another
        class on the way out.
        """
        decision, self._decision = self._decision, None
        return decision
