"""``HealingCapability`` — close out dangling tool calls when a run fails.

See the package docstring for the hook anchors, composition order and cursor semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import UsageLimitExceeded
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from ..context import ContextManager
from .errors import (
    RUN_LIMIT_HEALING_MESSAGE,
    RUN_LIMIT_HEALING_MESSAGE_REQUESTS,
    RUN_LIMIT_HEALING_MESSAGE_TOKENS,
    RUN_LIMIT_HEALING_MESSAGE_TOOL_CALLS,
)

logger = logging.getLogger(__name__)


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
        self._heal(self._healing_message(error, ctx))
        raise error

    @staticmethod
    def _healing_message(error: BaseException, ctx: RunContext[Any]) -> str:
        """The sentence the **model** reads as the aborted call's tool result.

        A run-tier breach gets the wording for the limit that actually bound (see
        ``_breached_limit_kind``); anything else gets type and message — what a model can act
        on ("ReadTimeout: pool timeout" → route around it). The stack, which it cannot act on,
        is dropped here and travels the operator channel instead.
        """
        if isinstance(error, UsageLimitExceeded):
            return {
                "requests": RUN_LIMIT_HEALING_MESSAGE_REQUESTS,
                "tool_calls": RUN_LIMIT_HEALING_MESSAGE_TOOL_CALLS,
                "tokens": RUN_LIMIT_HEALING_MESSAGE_TOKENS,
            }.get(HealingCapability._breached_limit_kind(ctx), RUN_LIMIT_HEALING_MESSAGE)
        return f"Tool call aborted: {type(error).__name__}: {error}"

    @staticmethod
    def _breached_limit_kind(ctx: RunContext[Any]) -> str:
        """Which KIND of run-tier limit bound: ``requests``, ``tool_calls``, ``tokens``, or ``""``.

        Recovered by comparing the run's usage against the limits it was given — **never by
        reading the exception's message**. pydantic-ai raises one ``UsageLimitExceeded`` for
        all five ``RunUsageLimits`` breaches and exposes no structured discriminator, so its
        wording is the only textual signal and wording is not a contract.

        The comparison order matches pydantic-ai's own (`usage.py`: requests, then tokens, in
        ``check_before_request``; tool calls in ``check_before_tool_call``). Two limits can be
        breached at once, so this reports the first in that order rather than pretending the
        answer is unique. Returns ``""`` when nothing can be identified — no limits attached,
        or a breach whose limit is not one this package sets — and the caller then falls back
        to the kind-agnostic wording rather than guessing.
        """
        limits = getattr(ctx, "usage_limits", None)
        usage = getattr(ctx, "usage", None)
        if limits is None or usage is None:
            return ""

        request_limit = getattr(limits, "request_limit", None)
        if request_limit is not None and usage.requests >= request_limit:
            return "requests"

        tool_calls_limit = getattr(limits, "tool_calls_limit", None)
        if tool_calls_limit is not None and getattr(usage, "tool_calls", 0) > tool_calls_limit:
            return "tool_calls"

        for limit_name, used in (
            ("input_tokens_limit", usage.input_tokens),
            ("output_tokens_limit", usage.output_tokens),
            ("total_tokens_limit", usage.total_tokens),
        ):
            limit = getattr(limits, limit_name, None)
            if limit is not None and used > limit:
                return "tokens"

        return ""

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
