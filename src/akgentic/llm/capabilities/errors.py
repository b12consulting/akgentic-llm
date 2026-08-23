"""Names shared by more than one capability: the usage-limit hierarchy and the healing message.

Imports nothing, so any sibling may import from it. See the package docstring for how the
capabilities compose.
"""

from __future__ import annotations

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
