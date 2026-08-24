"""``CompactionCapability`` — fold the conversation before the run reads it.

See the package docstring for the hook anchors, composition order and cursor semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from ..compaction import CompactionResult, CompactionStrategy
from ..context import ContextManager
from ..event import LlmContextCompactedEvent


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
