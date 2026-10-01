"""Tests for aggregation models and aggregate_usage() (genai-prices cost path)."""

from __future__ import annotations

import dataclasses

import pytest
from akgentic.core.utils.deserializer import deserialize_object
from genai_prices import Usage, calc_price
from pydantic import BaseModel

import akgentic.llm.pricing
from akgentic.llm.event import LlmUsageEvent
from akgentic.llm.pricing import (
    AgentUsageSummary,
    ModelUsage,
    RunUsageSummary,
    aggregate_usage,
    estimate_cost,
)


def _make_event(
    run_id: str = "run-1",
    model_name: str = "claude-sonnet-4-20250514",
    provider_name: str = "anthropic",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    requests: int = 1,
    estimated_cost_usd: float = 0.0,
) -> LlmUsageEvent:
    # Deliberately unstamped by default: every pre-existing caller exercises the
    # fallback path, so only the explicitly stamped specs below prove the feature.
    return LlmUsageEvent(
        run_id=run_id,
        model_name=model_name,
        provider_name=provider_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        requests=requests,
        estimated_cost_usd=estimated_cost_usd,
    )


def _expected_cost(
    model_name: str,
    provider_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Derive the genai-prices cost the same way the production path does.

    Deterministic against the offline bundled snapshot, so the expected value
    tracks the library rather than a hardcoded flat-formula number.
    """
    price = calc_price(
        Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        model_ref=model_name,
        provider_id=provider_name or None,
    )
    return float(price.total_price)


class TestModelUsage:
    """AC-2: ModelUsage is a frozen dataclass with correct fields."""

    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ModelUsage)
        usage = ModelUsage(
            model_name="m",
            provider_name="p",
            input_tokens=1,
            output_tokens=2,
            cache_read_tokens=3,
            cache_write_tokens=4,
            requests=5,
            estimated_cost_usd=0.1,
        )
        assert dataclasses.fields(usage) is not None
        # Frozen check
        try:
            usage.input_tokens = 999  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except dataclasses.FrozenInstanceError:
            pass

    def test_has_correct_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(ModelUsage)}
        expected = {
            "model_name",
            "provider_name",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "requests",
            "estimated_cost_usd",
        }
        assert field_names == expected


class TestRunUsageSummary:
    """AC-3: RunUsageSummary is a frozen dataclass with correct fields."""

    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(RunUsageSummary)
        summary = RunUsageSummary(
            run_id="r",
            models=[],
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
        )
        try:
            summary.run_id = "x"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except dataclasses.FrozenInstanceError:
            pass

    def test_has_correct_fields(self) -> None:
        field_names = {f.name for f in dataclasses.fields(RunUsageSummary)}
        expected = {
            "run_id",
            "models",
            "total_input_tokens",
            "total_output_tokens",
            "total_cost_usd",
        }
        assert field_names == expected


class TestAgentUsageSummary:
    """AC-4: AgentUsageSummary is a Pydantic BaseModel."""

    def test_is_pydantic_basemodel(self) -> None:
        assert issubclass(AgentUsageSummary, BaseModel)

    def test_defaults(self) -> None:
        s = AgentUsageSummary()
        assert s.by_model == {}
        assert s.runs == []
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0
        assert s.total_cache_read_tokens == 0
        assert s.total_cache_write_tokens == 0
        assert s.total_requests == 0
        assert s.total_cost_usd == 0.0

    def test_serialization_roundtrip(self) -> None:
        events = [
            _make_event(input_tokens=1000, output_tokens=500, cache_read_tokens=100),
        ]
        summary = aggregate_usage(events)
        data = summary.model_dump()
        restored = AgentUsageSummary.model_validate(data)
        assert restored.total_cost_usd == summary.total_cost_usd
        assert restored.total_input_tokens == summary.total_input_tokens
        assert restored.by_model == summary.by_model


class TestAggregateUsageEmpty:
    """AC-6: Empty event list returns zeroed summary."""

    def test_empty_list(self) -> None:
        result = aggregate_usage([])
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.total_cache_read_tokens == 0
        assert result.total_cache_write_tokens == 0
        assert result.total_requests == 0
        assert result.total_cost_usd == 0.0
        assert result.by_model == {}
        assert result.runs == []


class TestAggregateUsageSingleModel:
    """AC-2/#3: Single-model aggregation with genai-prices cost."""

    def test_single_event(self) -> None:
        events = [
            _make_event(
                input_tokens=1_000_000,
                output_tokens=500_000,
            ),
        ]
        result = aggregate_usage(events)
        assert result.total_input_tokens == 1_000_000
        assert result.total_output_tokens == 500_000
        assert result.total_requests == 1
        assert len(result.by_model) == 1
        model = result.by_model["claude-sonnet-4-20250514"]
        assert model.input_tokens == 1_000_000
        assert model.output_tokens == 500_000
        expected_cost = _expected_cost(
            "claude-sonnet-4-20250514",
            "anthropic",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        assert model.estimated_cost_usd == pytest.approx(expected_cost)
        assert model.estimated_cost_usd > 0.0
        assert result.runs == []

    def test_multiple_events_same_model(self) -> None:
        events = [
            _make_event(input_tokens=100, output_tokens=50, requests=1),
            _make_event(input_tokens=200, output_tokens=100, requests=1),
        ]
        result = aggregate_usage(events)
        assert result.total_input_tokens == 300
        assert result.total_output_tokens == 150
        assert result.total_requests == 2
        assert len(result.by_model) == 1


class TestAggregateUsageMultiModel:
    """AC-5: Multi-model aggregation with by_model containing one entry per model."""

    def test_two_models(self) -> None:
        events = [
            _make_event(
                model_name="claude-sonnet-4-20250514",
                provider_name="anthropic",
                input_tokens=1000,
                output_tokens=500,
            ),
            _make_event(
                model_name="gpt-4o",
                provider_name="openai",
                input_tokens=2000,
                output_tokens=1000,
            ),
        ]
        result = aggregate_usage(events)
        assert len(result.by_model) == 2
        assert "claude-sonnet-4-20250514" in result.by_model
        assert "gpt-4o" in result.by_model
        assert result.total_input_tokens == 3000
        assert result.total_output_tokens == 1500

        sonnet = result.by_model["claude-sonnet-4-20250514"]
        assert sonnet.provider_name == "anthropic"
        assert sonnet.input_tokens == 1000

        gpt = result.by_model["gpt-4o"]
        assert gpt.provider_name == "openai"
        assert gpt.input_tokens == 2000


class TestAggregateUsageByRun:
    """by_run=True produces RunUsageSummary per run_id."""

    def test_by_run(self) -> None:
        events = [
            _make_event(run_id="run-1", input_tokens=100, output_tokens=50),
            _make_event(run_id="run-1", input_tokens=200, output_tokens=100),
            _make_event(run_id="run-2", input_tokens=300, output_tokens=150),
        ]
        result = aggregate_usage(events, by_run=True)
        assert len(result.runs) == 2
        run_ids = {r.run_id for r in result.runs}
        assert run_ids == {"run-1", "run-2"}

        run1 = next(r for r in result.runs if r.run_id == "run-1")
        assert run1.total_input_tokens == 300
        assert run1.total_output_tokens == 150
        assert len(run1.models) == 1

        run2 = next(r for r in result.runs if r.run_id == "run-2")
        assert run2.total_input_tokens == 300
        assert run2.total_output_tokens == 150

    def test_by_run_cost_consistency(self) -> None:
        """RunUsageSummary.total_cost_usd must equal sum of its models' costs."""
        events = [
            _make_event(
                run_id="run-1",
                model_name="claude-sonnet-4-20250514",
                input_tokens=1000,
                output_tokens=500,
            ),
            _make_event(
                run_id="run-1",
                model_name="gpt-4o",
                provider_name="openai",
                input_tokens=2000,
                output_tokens=1000,
            ),
        ]
        result = aggregate_usage(events, by_run=True)
        assert len(result.runs) == 1
        run = result.runs[0]
        expected_cost = sum(m.estimated_cost_usd for m in run.models)
        assert run.total_cost_usd == expected_cost

    def test_by_run_false_no_runs(self) -> None:
        events = [_make_event()]
        result = aggregate_usage(events, by_run=False)
        assert result.runs == []

    def test_by_run_multi_model_per_run(self) -> None:
        events = [
            _make_event(run_id="run-1", model_name="claude-sonnet-4-20250514", input_tokens=100),
            _make_event(
                run_id="run-1", model_name="gpt-4o", provider_name="openai", input_tokens=200
            ),
        ]
        result = aggregate_usage(events, by_run=True)
        assert len(result.runs) == 1
        run1 = result.runs[0]
        assert len(run1.models) == 2
        assert run1.total_input_tokens == 300


class TestUnknownModel:
    """AC-4/#5: Unknown model produces estimated_cost_usd == 0.0."""

    def test_unknown_model_cost_zero(self) -> None:
        events = [
            _make_event(
                model_name="unknown-model-xyz",
                input_tokens=5000,
                output_tokens=2000,
            ),
        ]
        result = aggregate_usage(events)
        model = result.by_model["unknown-model-xyz"]
        assert model.estimated_cost_usd == 0.0
        assert model.input_tokens == 5000
        assert model.output_tokens == 2000

    def test_mixed_known_and_unknown(self) -> None:
        events = [
            _make_event(model_name="claude-sonnet-4-20250514", input_tokens=1000),
            _make_event(model_name="unknown-model", input_tokens=500),
        ]
        result = aggregate_usage(events)
        assert result.by_model["unknown-model"].estimated_cost_usd == 0.0
        assert result.by_model["claude-sonnet-4-20250514"].estimated_cost_usd > 0.0
        # AC-5: totals include the unknown model's tokens and its 0.0 cost.
        assert result.total_input_tokens == 1500
        assert result.total_cost_usd == pytest.approx(
            result.by_model["claude-sonnet-4-20250514"].estimated_cost_usd
        )


class TestEstimateCost:
    """NFR5: estimate_cost maps genai-prices LookupError to 0.0 (unknown model)."""

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost(
            model_name="definitely-not-a-real-model-xyz",
            provider_name="anthropic",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost == 0.0

    def test_known_model_returns_positive(self) -> None:
        cost = estimate_cost(
            model_name="claude-sonnet-4-20250514",
            provider_name="anthropic",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost > 0.0

    def test_empty_provider_still_resolves(self) -> None:
        # provider_name="" must be passed as provider_id=None so genai-prices
        # matches by model_ref across providers instead of a non-existent one.
        cost = estimate_cost(
            model_name="claude-sonnet-4-20250514",
            provider_name="",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost > 0.0

    def test_openrouter_route_resolves_under_openrouter_provider(self) -> None:
        cost = estimate_cost(
            model_name="deepseek/deepseek-chat",
            provider_name="openrouter",
            input_tokens=1000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost > 0.0
        assert cost == pytest.approx(
            _expected_cost("deepseek/deepseek-chat", "openrouter", 1000, 1000)
        )

    def test_openrouter_route_without_provider_is_unknown(self) -> None:
        # Slash-prefixed route names only resolve under the openrouter provider id,
        # which is why the provider label stamped on the response matters.
        cost = estimate_cost(
            model_name="deepseek/deepseek-chat",
            provider_name="",
            input_tokens=1000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost == 0.0

    def test_openrouter_unknown_route_returns_zero(self) -> None:
        cost = estimate_cost(
            model_name="acme/not-a-model",
            provider_name="openrouter",
            input_tokens=1000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert cost == 0.0


class TestAggregateUsageOpenRouter:
    """An OpenRouter usage event keeps its provider label and prices through it."""

    def test_openrouter_event_priced_under_openrouter(self) -> None:
        events = [
            _make_event(
                model_name="deepseek/deepseek-chat",
                provider_name="openrouter",
                input_tokens=1000,
                output_tokens=1000,
            )
        ]
        summary = aggregate_usage(events)
        assert len(summary.by_model) == 1
        model = summary.by_model["deepseek/deepseek-chat"]
        assert model.provider_name == "openrouter"
        assert model.estimated_cost_usd == pytest.approx(
            _expected_cost("deepseek/deepseek-chat", "openrouter", 1000, 1000)
        )


class TestCacheTokenPricing:
    """Cache tokens affect cost, priced without double-counting cached reads."""

    def test_cache_tokens_affect_cost(self) -> None:
        # genai-prices treats input_tokens as INCLUDING cached tokens, so a
        # realistic cache-bearing event has input_tokens >= cache_read_tokens.
        events = [
            _make_event(
                model_name="claude-sonnet-4-20250514",
                input_tokens=1_000_000,
                output_tokens=0,
                cache_read_tokens=400_000,
                cache_write_tokens=0,
            ),
        ]
        result = aggregate_usage(events)
        model = result.by_model["claude-sonnet-4-20250514"]
        expected_cost = _expected_cost(
            "claude-sonnet-4-20250514",
            "anthropic",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=400_000,
        )
        assert model.estimated_cost_usd == pytest.approx(expected_cost)
        assert result.total_cache_read_tokens == 400_000

        # Cached reads are cheaper than full-price input: pricing the same 1M
        # input with no cache costs strictly more, confirming cache tokens are
        # priced at the cache rate (no double-count).
        no_cache_cost = _expected_cost(
            "claude-sonnet-4-20250514",
            "anthropic",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=0,
        )
        assert model.estimated_cost_usd < no_cache_cost


class TestTotalRequestsAccumulation:
    """total_requests accumulates correctly across multiple events."""

    def test_total_requests_multi_event(self) -> None:
        events = [
            _make_event(requests=3),
            _make_event(requests=2),
            _make_event(model_name="gpt-4o", provider_name="openai", requests=5),
        ]
        result = aggregate_usage(events)
        assert result.total_requests == 10
        assert result.by_model["claude-sonnet-4-20250514"].requests == 5
        assert result.by_model["gpt-4o"].requests == 5


class TestTotalCostConsistency:
    """AC-7: total_cost_usd equals the sum of the events' stamps (unknown stamps 0.0)."""

    def test_total_equals_sum(self) -> None:
        stamp_sonnet = _expected_cost("claude-sonnet-4-20250514", "anthropic", 1000, 500)
        stamp_gpt = _expected_cost("gpt-4o", "openai", 2000, 1000)
        stamp_unknown = estimate_cost("unknown-model", "anthropic", 500, 250, 0, 0)
        assert stamp_unknown == 0.0
        events = [
            _make_event(
                model_name="claude-sonnet-4-20250514",
                input_tokens=1000,
                output_tokens=500,
                estimated_cost_usd=stamp_sonnet,
            ),
            _make_event(
                model_name="gpt-4o",
                provider_name="openai",
                input_tokens=2000,
                output_tokens=1000,
                estimated_cost_usd=stamp_gpt,
            ),
            _make_event(
                model_name="unknown-model",
                input_tokens=500,
                output_tokens=250,
                estimated_cost_usd=stamp_unknown,
            ),
        ]
        result = aggregate_usage(events)
        assert result.by_model["claude-sonnet-4-20250514"].estimated_cost_usd == pytest.approx(
            stamp_sonnet
        )
        assert result.by_model["gpt-4o"].estimated_cost_usd == pytest.approx(stamp_gpt)
        assert result.by_model["unknown-model"].estimated_cost_usd == 0.0
        assert result.total_cost_usd == pytest.approx(stamp_sonnet + stamp_gpt + stamp_unknown)


class TestStampedAggregation:
    """AC-4: a bucket's cost is the sum of its events' stamps; the pricer is not consulted."""

    def test_synthetic_stamps_are_summed(self) -> None:
        # Inputs, not price expectations: pure arithmetic over whatever is stamped.
        events = [
            _make_event(estimated_cost_usd=0.25),
            _make_event(estimated_cost_usd=0.5),
        ]
        result = aggregate_usage(events)
        bucket = result.by_model["claude-sonnet-4-20250514"]
        assert bucket.estimated_cost_usd == pytest.approx(sum(e.estimated_cost_usd for e in events))
        assert result.total_cost_usd == pytest.approx(bucket.estimated_cost_usd)

    def test_single_provider_realistic_stamps(self) -> None:
        stamp_a = _expected_cost("claude-sonnet-4-20250514", "anthropic", 100, 50)
        stamp_b = _expected_cost("claude-sonnet-4-20250514", "anthropic", 200, 100)
        events = [
            _make_event(input_tokens=100, output_tokens=50, estimated_cost_usd=stamp_a),
            _make_event(input_tokens=200, output_tokens=100, estimated_cost_usd=stamp_b),
        ]
        result = aggregate_usage(events)
        bucket = result.by_model["claude-sonnet-4-20250514"]
        assert bucket.estimated_cost_usd == pytest.approx(stamp_a + stamp_b)
        assert bucket.input_tokens == 300
        assert bucket.requests == 2

    @pytest.mark.parametrize("openrouter_first", [True, False])
    def test_mixed_providers_each_stamp_uses_its_own_provider(self, openrouter_first: bool) -> None:
        # The same model_name under two providers: one resolves, the other does not.
        # A forced recompute would price 2000/2000 tokens at the first-seen provider,
        # which differs from the stamped sum in either order.
        stamp_routed = _expected_cost("deepseek/deepseek-chat", "openrouter", 1000, 1000)
        stamp_bare = estimate_cost("deepseek/deepseek-chat", "", 1000, 1000, 0, 0)
        assert stamp_routed > 0.0
        assert stamp_bare == 0.0
        routed = _make_event(
            model_name="deepseek/deepseek-chat",
            provider_name="openrouter",
            input_tokens=1000,
            output_tokens=1000,
            estimated_cost_usd=stamp_routed,
        )
        bare = _make_event(
            model_name="deepseek/deepseek-chat",
            provider_name="",
            input_tokens=1000,
            output_tokens=1000,
            estimated_cost_usd=stamp_bare,
        )
        events = [routed, bare] if openrouter_first else [bare, routed]
        result = aggregate_usage(events)
        bucket = result.by_model["deepseek/deepseek-chat"]
        assert bucket.estimated_cost_usd == pytest.approx(stamp_routed + stamp_bare)
        assert bucket.input_tokens == 2000

    def test_per_run_buckets_sum_their_own_stamps(self) -> None:
        events = [
            _make_event(run_id="run-1", estimated_cost_usd=0.1),
            _make_event(run_id="run-1", estimated_cost_usd=0.2),
            _make_event(
                run_id="run-1",
                model_name="gpt-4o",
                provider_name="openai",
                estimated_cost_usd=0.7,
            ),
            _make_event(run_id="run-2", estimated_cost_usd=0.4),
        ]
        result = aggregate_usage(events, by_run=True)
        assert len(result.runs) == 2
        for run in result.runs:
            run_events = [e for e in events if e.run_id == run.run_id]
            for model in run.models:
                expected = sum(
                    e.estimated_cost_usd for e in run_events if e.model_name == model.model_name
                )
                assert model.estimated_cost_usd == pytest.approx(expected)
            assert run.total_cost_usd == pytest.approx(
                sum(m.estimated_cost_usd for m in run.models)
            )
        run1 = next(r for r in result.runs if r.run_id == "run-1")
        assert run1.total_cost_usd == pytest.approx(1.0)
        run2 = next(r for r in result.runs if r.run_id == "run-2")
        assert run2.total_cost_usd == pytest.approx(0.4)


class TestFallbackRecompute:
    """AC-5: a zero-stamp bucket with tokens is repriced from its aggregate; no tokens, no call."""

    def test_all_unstamped_bucket_is_repriced_from_aggregate_tokens(self) -> None:
        events = [
            _make_event(input_tokens=100, output_tokens=50),
            _make_event(input_tokens=200, output_tokens=100),
        ]
        result = aggregate_usage(events)
        bucket = result.by_model["claude-sonnet-4-20250514"]
        assert bucket.estimated_cost_usd == pytest.approx(
            _expected_cost("claude-sonnet-4-20250514", "anthropic", 300, 150)
        )
        assert bucket.estimated_cost_usd > 0.0

    def test_zero_token_zero_stamp_bucket_does_not_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def spy(*args: object) -> float:
            calls.append(args)
            return 123.0

        monkeypatch.setattr(akgentic.llm.pricing, "estimate_cost", spy)
        result = aggregate_usage([_make_event(input_tokens=0, output_tokens=0)])
        bucket = result.by_model["claude-sonnet-4-20250514"]
        assert bucket.estimated_cost_usd == 0.0
        assert calls == []
        assert bucket.requests == 1
        assert result.total_requests == 1
        assert result.total_input_tokens == 0

    def test_unknown_model_bucket_falls_back_to_zero(self) -> None:
        result = aggregate_usage(
            [_make_event(model_name="unknown-model-xyz", input_tokens=5000, output_tokens=2000)]
        )
        assert result.by_model["unknown-model-xyz"].estimated_cost_usd == 0.0
        assert result.by_model["unknown-model-xyz"].input_tokens == 5000


_STORED_PRE_STAMP_EVENT = {
    "__model__": "akgentic.llm.event.LlmUsageEvent",
    "run_id": "run-1",
    "model_name": "claude-sonnet-4-20250514",
    "provider_name": "anthropic",
    "input_tokens": 1000,
    "output_tokens": 500,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "requests": 1,
}


class TestReplayGuard:
    """AC-6: a stored event without the stamp key replays at 0.0 and prices via the fallback."""

    def test_stored_dict_without_stamp_key_replays_unstamped(self) -> None:
        restored = deserialize_object(dict(_STORED_PRE_STAMP_EVENT))
        assert isinstance(restored, LlmUsageEvent)
        assert restored.estimated_cost_usd == 0.0
        assert restored.input_tokens == 1000

        result = aggregate_usage([restored])
        assert result.total_input_tokens == 1000
        assert result.by_model["claude-sonnet-4-20250514"].estimated_cost_usd == pytest.approx(
            _expected_cost("claude-sonnet-4-20250514", "anthropic", 1000, 500)
        )

    def test_stored_dict_with_stamp_key_round_trips_the_stamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp = 0.375
        restored = deserialize_object({**_STORED_PRE_STAMP_EVENT, "estimated_cost_usd": stamp})
        assert isinstance(restored, LlmUsageEvent)
        assert restored.estimated_cost_usd == pytest.approx(stamp)

        calls: list[tuple[object, ...]] = []

        def spy(*args: object) -> float:
            calls.append(args)
            return 999.0

        monkeypatch.setattr(akgentic.llm.pricing, "estimate_cost", spy)
        result = aggregate_usage([restored])
        assert result.by_model["claude-sonnet-4-20250514"].estimated_cost_usd == pytest.approx(
            stamp
        )
        assert calls == []


class TestPublicApiExport:
    """AC-2: pricing exports importable from akgentic.llm and present in __all__."""

    def test_all_exports_importable(self) -> None:
        import akgentic.llm

        assert hasattr(akgentic.llm, "AgentUsageSummary")
        assert hasattr(akgentic.llm, "ModelUsage")
        assert hasattr(akgentic.llm, "RunUsageSummary")
        assert hasattr(akgentic.llm, "aggregate_usage")
        assert hasattr(akgentic.llm, "estimate_cost")

    def test_pricing_removed(self) -> None:
        import akgentic.llm

        assert not hasattr(akgentic.llm, "PRICING")
        assert "PRICING" not in akgentic.llm.__all__

    def test_private_pricer_has_no_alias(self) -> None:
        assert not hasattr(akgentic.llm.pricing, "_compute_cost")

    def test_all_in_dunder_all(self) -> None:
        import akgentic.llm

        names = [
            "AgentUsageSummary",
            "ModelUsage",
            "RunUsageSummary",
            "aggregate_usage",
            "estimate_cost",
        ]
        for name in names:
            assert name in akgentic.llm.__all__, f"{name} missing from __all__"
