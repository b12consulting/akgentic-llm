"""Names shared by more than one capability: the usage-limit hierarchy and the healing message.

Imports nothing, so any sibling may import from it. See the package docstring for how the
capabilities compose.
"""

from __future__ import annotations

# What the MODEL reads as the tool result of the call a run-tier breach aborted.
# Not diagnostics: the operator's traceback travels the other channel
# (``ErrorMessage.traceback``, formatted by ``Akgent._handle_failure``). Defined once
# here so each call site and its test cannot drift into two wordings (ADR-016 §D2).
#
# There is one wording per KIND of limit, because the advice genuinely differs and a single
# sentence has to lie about the others. ``RunUsageLimits`` can breach five ways — requests,
# tool calls, and three token limits — and pydantic-ai raises the same ``UsageLimitExceeded``
# for all of them, so the kind is recovered from usage-vs-limits, never from the message text.

RUN_LIMIT_HEALING_MESSAGE_REQUESTS = (
    "This turn's request budget is exhausted, so this tool call was aborted and this is the "
    "last thing you can do in this turn. Answer now using what you already have, and say "
    "plainly what you could not verify."
)

RUN_LIMIT_HEALING_MESSAGE_TOOL_CALLS = (
    "This turn's tool-call budget is exhausted, so this tool call was aborted and no further "
    "tool calls are possible. You can still reason about what you already have. Answer now, "
    "and say plainly what you could not verify."
)

RUN_LIMIT_HEALING_MESSAGE_TOKENS = (
    "This turn's token budget is exhausted, so this tool call was aborted and no further work "
    "is possible. Answer now and keep it brief, using what you already have, and say plainly "
    "what you could not verify."
)

# The kind-agnostic wording, used when the binding limit cannot be identified. Also the
# historical name: it stays exported so callers written against it keep resolving.
RUN_LIMIT_HEALING_MESSAGE = (
    "This turn's budget is exhausted, so this tool call was aborted and no further tool calls "
    "are possible. Answer now using what you already have, and say plainly what you could not "
    "verify."
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
