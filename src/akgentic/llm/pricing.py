"""Pricing table, aggregation models, and aggregate_usage() function.

Provides per-1M-token pricing for supported LLM models and functions to
aggregate ``LlmUsageEvent`` lists into hierarchical cost summaries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, Field

from akgentic.llm.event import LlmUsageEvent

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.0,
        "cache_read": 1.25,
        "cache_write": 2.50,
    },
    "gpt-5.2": {
        "input": 2.0,
        "output": 8.0,
        "cache_read": 0.50,
        "cache_write": 2.0,
    },
}


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
        estimated_cost_usd: Estimated cost in USD based on PRICING table.
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
    breakdowns with cost estimates derived from the PRICING table.

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


def _compute_model_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> float:
    """Compute estimated USD cost for a model using the PRICING table."""
    rates = PRICING.get(model_name, {})
    return (
        input_tokens * rates.get("input", 0.0)
        + output_tokens * rates.get("output", 0.0)
        + cache_read_tokens * rates.get("cache_read", 0.0)
        + cache_write_tokens * rates.get("cache_write", 0.0)
    ) / 1_000_000


def _build_model_usage(
    model_name: str,
    provider_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    requests: int,
) -> ModelUsage:
    """Build a ModelUsage with computed cost."""
    cost = _compute_model_cost(
        model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    )
    return ModelUsage(
        model_name=model_name,
        provider_name=provider_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        requests=requests,
        estimated_cost_usd=cost,
    )


def _aggregate_events(
    events: list[LlmUsageEvent],
) -> dict[str, dict[str, int | str]]:
    """Group events by model_name and accumulate token counts."""
    accum: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {
            "provider_name": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "requests": 0,
        }
    )
    for ev in events:
        bucket = accum[ev.model_name]
        if not bucket["provider_name"]:
            bucket["provider_name"] = ev.provider_name
        bucket["input_tokens"] = int(bucket["input_tokens"]) + ev.input_tokens
        bucket["output_tokens"] = int(bucket["output_tokens"]) + ev.output_tokens
        bucket["cache_read_tokens"] = int(bucket["cache_read_tokens"]) + ev.cache_read_tokens
        bucket["cache_write_tokens"] = int(bucket["cache_write_tokens"]) + ev.cache_write_tokens
        bucket["requests"] = int(bucket["requests"]) + ev.requests
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

    by_model: dict[str, ModelUsage] = {}
    for model_name, bucket in model_accum.items():
        by_model[model_name] = _build_model_usage(
            model_name=model_name,
            provider_name=str(bucket["provider_name"]),
            input_tokens=int(bucket["input_tokens"]),
            output_tokens=int(bucket["output_tokens"]),
            cache_read_tokens=int(bucket["cache_read_tokens"]),
            cache_write_tokens=int(bucket["cache_write_tokens"]),
            requests=int(bucket["requests"]),
        )

    runs: list[RunUsageSummary] = []
    if by_run:
        run_groups: dict[str, list[LlmUsageEvent]] = defaultdict(list)
        for ev in events:
            run_groups[ev.run_id].append(ev)
        for run_id, run_events in run_groups.items():
            run_accum = _aggregate_events(run_events)
            run_models = [
                _build_model_usage(
                    model_name=mn,
                    provider_name=str(b["provider_name"]),
                    input_tokens=int(b["input_tokens"]),
                    output_tokens=int(b["output_tokens"]),
                    cache_read_tokens=int(b["cache_read_tokens"]),
                    cache_write_tokens=int(b["cache_write_tokens"]),
                    requests=int(b["requests"]),
                )
                for mn, b in run_accum.items()
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
