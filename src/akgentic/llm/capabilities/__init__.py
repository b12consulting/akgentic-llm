"""Run-loop concerns as standalone, mountable pydantic-ai capabilities.

Four hook anchors carry everything here:

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
- ``after_model_request`` — the pre-history anchor, and the only one here that sits *outside*
  the ordering problem the other three live inside. It fires between the model response and
  ``ModelRequestNode._append_response``, so a capability that returns a replacement response
  decides what reaches durable history in the first place — before the sweep, before the
  cursor, before any other capability's node hooks can see it. ``DiscardedOutputCapability``
  is the first capability here to use it, and it uses it **because** of that: stripping the
  output the discard branch is about to throw away has to happen before the append, and doing
  it from a node hook instead would mean editing a response already in history and sequencing
  that edit against persistence. It is deliberately order-independent as a result — it
  declares no ``get_ordering()``, and mounting it before or after ``EventSourcingCapability``
  yields byte-identical history and an identical event sequence.

  The anchor carries **both** of ``CallToolsNode``'s multi-part collapses, split on
  ``response.tool_calls``: with tool calls the co-emitted output is discarded and is stripped
  here; with none the text parts are concatenated into invalid JSON and are merged here
  instead. Same hook, same replacement response, same order-independence — the merge folds
  parts the graph has not yet read, so like the strip it races nothing.

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
CompactionCapability, EventSourcingCapability, LimitRecoveryCapability, HealingCapability,
DiscardedOutputCapability, *yours]``. Two couplings ride on exactly that order and nothing
else: the budget refuses a spent
agent **before** compaction pays for a summarizer, and limit recovery sits immediately *before*
healing so that healing — the later entry, and therefore the first to fire in the **reversed**
``on_run_error`` walk — has written its ``ToolReturnPart`` before the recovery seam is consulted.
Compaction sitting ahead of persistence also puts the cursor on the
post-fold history, which is where it belongs — but that one is belt-and-braces, not the
mechanism: ``_anchor`` re-opens the cursor against the normalised list at the first node hook,
which absorbs a fold performed anywhere before it (verified by swapping the two).
That list order is not final, though: if **any** capability in the chain
declares ``get_ordering()`` (a fixed ``position``, or a ``wraps=`` / ``wrapped_by=`` constraint)
pydantic-ai topologically re-sorts the whole chain to satisfy it, so a caller capability can
legitimately end up outside these five. None of the classes here declares one, and the budget
deliberately does not declare one to pin itself outermost — that would be a behavioural change
owed its own decision. What a co-mounted capability needs from the order does **not** depend on
where it sits: the closing sweep is in ``wrap_run``'s ``finally``, outside every capability's
node hooks whatever the order, so durable ``after_*`` edits are always the ones persisted.
``DiscardedOutputCapability``'s position in that list carries nothing at all — its anchor runs
before the append, so there is no sweep for it to race and no order for a re-sort to break.

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

from .budget import LifetimeBudgetCapability
from .compaction import CompactionCapability
from .discarded_output import DEFAULT_STRIP_BUDGET, DiscardedOutputCapability
from .errors import (
    RUN_LIMIT_HEALING_MESSAGE,
    AgentUsageLimitError,
    RunUsageLimitError,
    UsageLimitError,
)
from .event_sourcing import EventSourcingCapability
from .healing import HealingCapability
from .limit_recovery import (
    DEFAULT_CONCLUSION_REASON,
    ConclusionDecision,
    LimitRecoveryCapability,
)

__all__ = [
    "DEFAULT_CONCLUSION_REASON",
    "DEFAULT_STRIP_BUDGET",
    "RUN_LIMIT_HEALING_MESSAGE",
    "AgentUsageLimitError",
    "CompactionCapability",
    "ConclusionDecision",
    "DiscardedOutputCapability",
    "EventSourcingCapability",
    "HealingCapability",
    "LifetimeBudgetCapability",
    "LimitRecoveryCapability",
    "RunUsageLimitError",
    "UsageLimitError",
]
