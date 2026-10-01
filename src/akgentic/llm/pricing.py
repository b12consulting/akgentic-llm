"""Aggregation models, the public pricer, and aggregate_usage().

``estimate_cost`` prices one response with the ``genai-prices`` library (offline
bundled price snapshot), so provider rates stay current without a hand-maintained
table and cached tokens are priced without double-counting. ``ContextManager``
stamps every ``LlmUsageEvent`` with it at emission.

``aggregate_usage`` folds ``LlmUsageEvent`` lists into hierarchical cost
summaries. Per-model cost is the **sum of the events' stamps**; it is recomputed
from the bucket's aggregate tokens only when that sum is ``0.0`` and the bucket
has tokens — the path a pre-stamp (replayed) bucket takes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypedDict

from genai_prices import Usage, calc_price
from pydantic import BaseModel, Field

from akgentic.llm.event import LlmUsageEvent


class _ModelAccum(TypedDict):
    """Accumulator bucket for per-model token aggregation."""

    provider_name: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    requests: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class ModelUsage:
    """Aggregated token usage and cost for a single model.

    Attributes:
        model_name: Model identifier (e.g. "claude-sonnet-4-20250514").
        provider_name: Provider identifier (e.g. "anthropic").
        input_tokens: Total prompt tokens consumed.
        output_tokens: Total response tokens generated.
        cache_read_tokens: Total tokens read from provider cache.
        cache_write_tokens: Total tokens written to provider cache.
        requests: Total HTTP requests.
        estimated_cost_usd: Sum of the events' ``estimated_cost_usd`` stamps.
            Recomputed from the aggregate tokens via ``estimate_cost`` only when
            that sum is ``0.0`` and the bucket has tokens (pre-stamp events);
            ``0.0`` when the model is not resolvable by genai-prices.
    """

    model_name: str
    provider_name: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    requests: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class RunUsageSummary:
    """Per-run usage summary with per-model breakdown.

    Attributes:
        run_id: Identifier of the agent run.
        models: Per-model usage breakdown for this run.
        total_input_tokens: Sum of input tokens across all models in this run.
        total_output_tokens: Sum of output tokens across all models in this run.
        total_cost_usd: Sum of estimated costs across all models in this run.
    """

    run_id: str
    models: list[ModelUsage]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


class AgentUsageSummary(BaseModel):
    """Hierarchical usage summary for an agent.

    Aggregates LlmUsageEvent data into per-model and optionally per-run
    breakdowns with cost estimates derived via genai-prices.

    Attributes:
        by_model: Mapping of model_name to aggregated ModelUsage.
        runs: Per-run summaries (populated only when by_run=True).
        total_input_tokens: Grand total input tokens.
        total_output_tokens: Grand total output tokens.
        total_cache_read_tokens: Grand total cache read tokens.
        total_cache_write_tokens: Grand total cache write tokens.
        total_requests: Grand total HTTP requests.
        total_cost_usd: Grand total estimated cost in USD.
    """

    by_model: dict[str, ModelUsage] = Field(default_factory=dict)
    runs: list[RunUsageSummary] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_requests: int = 0
    total_cost_usd: float = 0.0


def estimate_cost(
    model_name: str,
    provider_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> float:
    """Estimate the USD cost of one model response via genai-prices (offline snapshot).

    This is the single pricer in the framework: ``ContextManager`` stamps every
    ``LlmUsageEvent`` with it at emission, and ``aggregate_usage`` falls back to
    it for buckets whose events carry no stamp.

    ``input_tokens`` already includes cached tokens; genai-prices splits the
    total internally (cached portion at the cheaper cache-read rate, remainder
    at the input rate), so cache reads are never double-charged.

    Args:
        model_name: Model identifier as reported by the provider; genai-prices'
            ``model_ref``.
        provider_name: Provider identifier (e.g. ``"anthropic"``, ``"openrouter"``).
            An empty string is passed as ``provider_id=None`` so the lookup
            matches by ``model_ref`` across providers instead of scoping to a
            non-existent one.
        input_tokens: Prompt tokens, cached reads included.
        output_tokens: Response tokens.
        cache_read_tokens: Tokens read from provider cache.
        cache_write_tokens: Tokens written to provider cache.

    Returns:
        The estimated cost in USD, or ``0.0`` when genai-prices cannot resolve
        ``model_name`` (``calc_price`` raises ``LookupError``), so an unpriced
        model still aggregates its token counts.
    """
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    try:
        price = calc_price(usage, model_ref=model_name, provider_id=provider_name or None)
    except LookupError:
        return 0.0
    return float(price.total_price)


def _build_model_usage(model_name: str, bucket: _ModelAccum) -> ModelUsage:
    """Build a ModelUsage from an accumulator bucket, deriving its cost.

    The cost is the sum of the bucket's stamps; it is recomputed from the
    aggregate tokens only when that sum is ``0.0`` and the bucket has tokens.
    Unstamped events contribute exactly ``0.0`` and a sum of exact zeros is
    exactly ``0.0``, so the comparison is deliberately exact — no epsilon. A
    bucket with neither stamps nor tokens is ``0.0`` without a price lookup.
    ``estimate_cost`` is called by its module-global name so the fallback can be
    observed by patching ``akgentic.llm.pricing.estimate_cost``.
    """
    stamped = bucket["estimated_cost_usd"]
    tokens = (
        bucket["input_tokens"]
        + bucket["output_tokens"]
        + bucket["cache_read_tokens"]
        + bucket["cache_write_tokens"]
    )
    if stamped == 0.0 and tokens > 0:
        cost = estimate_cost(
            model_name,
            bucket["provider_name"],
            bucket["input_tokens"],
            bucket["output_tokens"],
            bucket["cache_read_tokens"],
            bucket["cache_write_tokens"],
        )
    else:
        cost = stamped
    return ModelUsage(
        model_name=model_name,
        provider_name=bucket["provider_name"],
        input_tokens=bucket["input_tokens"],
        output_tokens=bucket["output_tokens"],
        cache_read_tokens=bucket["cache_read_tokens"],
        cache_write_tokens=bucket["cache_write_tokens"],
        requests=bucket["requests"],
        estimated_cost_usd=cost,
    )


def _new_accum() -> _ModelAccum:
    """Create a zeroed accumulator bucket."""
    return _ModelAccum(
        provider_name="",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        requests=0,
        estimated_cost_usd=0.0,
    )


def _aggregate_events(
    events: list[LlmUsageEvent],
) -> dict[str, _ModelAccum]:
    """Group events by model_name and accumulate token counts and cost stamps."""
    accum: dict[str, _ModelAccum] = defaultdict(_new_accum)
    for ev in events:
        bucket = accum[ev.model_name]
        # First provider wins: the fallback recompute still needs a provider.
        if not bucket["provider_name"]:
            bucket["provider_name"] = ev.provider_name
        bucket["input_tokens"] += ev.input_tokens
        bucket["output_tokens"] += ev.output_tokens
        bucket["cache_read_tokens"] += ev.cache_read_tokens
        bucket["cache_write_tokens"] += ev.cache_write_tokens
        bucket["requests"] += ev.requests
        bucket["estimated_cost_usd"] += ev.estimated_cost_usd
    return accum


def aggregate_usage(
    events: list[LlmUsageEvent],
    *,
    by_run: bool = False,
) -> AgentUsageSummary:
    """Aggregate LlmUsageEvent list into a hierarchical summary.

    Always aggregates totals and by-model breakdown.
    When by_run=True, also provides per-run detail.

    Args:
        events: List of LlmUsageEvent (typically for one agent).
        by_run: Include per-run breakdown (default: False).

    Returns:
        AgentUsageSummary with totals, by-model, and optionally by-run.
    """
    if not events:
        return AgentUsageSummary()

    model_accum = _aggregate_events(events)

    by_model: dict[str, ModelUsage] = {
        model_name: _build_model_usage(model_name, bucket)
        for model_name, bucket in model_accum.items()
    }

    runs: list[RunUsageSummary] = []
    if by_run:
        run_groups: dict[str, list[LlmUsageEvent]] = defaultdict(list)
        for ev in events:
            run_groups[ev.run_id].append(ev)
        for run_id, run_events in run_groups.items():
            run_accum = _aggregate_events(run_events)
            run_models = [
                _build_model_usage(model_name, bucket) for model_name, bucket in run_accum.items()
            ]
            runs.append(
                RunUsageSummary(
                    run_id=run_id,
                    models=run_models,
                    total_input_tokens=sum(m.input_tokens for m in run_models),
                    total_output_tokens=sum(m.output_tokens for m in run_models),
                    total_cost_usd=sum(m.estimated_cost_usd for m in run_models),
                )
            )

    total_cost = sum(m.estimated_cost_usd for m in by_model.values())
    return AgentUsageSummary(
        by_model=by_model,
        runs=runs,
        total_input_tokens=sum(m.input_tokens for m in by_model.values()),
        total_output_tokens=sum(m.output_tokens for m in by_model.values()),
        total_cache_read_tokens=sum(m.cache_read_tokens for m in by_model.values()),
        total_cache_write_tokens=sum(m.cache_write_tokens for m in by_model.values()),
        total_requests=sum(m.requests for m in by_model.values()),
        total_cost_usd=total_cost,
    )
