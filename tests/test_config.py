"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from akgentic.llm import config as config_module
from akgentic.llm.config import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    ReactAgentConfig,
    RuntimeConfig,
    RunUsageLimits,
    TokenUsageLimits,
)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=1000)
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_temperature_validation_too_low(self):
        """Test temperature below 0.0 raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model="gpt-4o", temperature=-0.1)

    def test_temperature_validation_too_high(self):
        """Test temperature above 2.0 raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model="gpt-4o", temperature=2.1)

    def test_temperature_boundary_low(self):
        """Test temperature at lower boundary (0.0) is valid."""
        config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_boundary_high(self):
        """Test temperature at upper boundary (2.0) is valid."""
        config = ModelConfig(provider="openai", model="gpt-4o", temperature=2.0)
        assert config.temperature == 2.0

    def test_optional_fields_none(self):
        """Test optional fields can be None."""
        config = ModelConfig(provider="openai", model="gpt-4o")
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.seed is None
        assert config.reasoning_effort is None

    def test_all_providers(self):
        """Test all supported providers."""
        providers = ["openai", "azure", "nvidia", "google-gla", "mistral", "anthropic"]
        for provider in providers:
            config = ModelConfig(provider=provider, model="test-model")  # type: ignore
            assert config.provider == provider

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(provider="invalid", model="test-model")  # type: ignore

    def test_reasoning_effort_values(self):
        """Test reasoning effort valid values."""
        for effort in ["low", "medium", "high"]:
            config = ModelConfig(provider="openai", model="gpt-4o", reasoning_effort=effort)  # type: ignore
            assert config.reasoning_effort == effort

    def test_serialization(self):
        """Test model serialization."""
        config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, seed=42)
        data = config.model_dump()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.7
        assert data["seed"] == 42

    def test_json_serialization(self):
        """Test JSON serialization."""
        config = ModelConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.7
        )
        json_str = config.model_dump_json()
        assert "anthropic" in json_str
        assert "claude-3-5-sonnet-20241022" in json_str

    def test_context_length_default_none(self):
        """context_length defaults to None (compaction off) — AC 1."""
        config = ModelConfig(provider="openai", model="gpt-4o")
        assert config.context_length is None

    def test_context_length_positive_accepted(self):
        """A positive context_length is accepted — AC 1."""
        config = ModelConfig(provider="openai", model="gpt-4o", context_length=128000)
        assert config.context_length == 128000

    def test_context_length_zero_invalid(self):
        """context_length=0 violates gt=0 — AC 1."""
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model="gpt-4o", context_length=0)

    def test_context_length_negative_invalid(self):
        """A negative context_length violates gt=0 — AC 1."""
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model="gpt-4o", context_length=-1)

    def test_context_length_independent_of_max_tokens(self):
        """context_length and max_tokens are independent — AC 1."""
        config = ModelConfig(
            provider="openai", model="gpt-4o", max_tokens=1000, context_length=200000
        )
        assert config.max_tokens == 1000
        assert config.context_length == 200000

    def test_context_length_round_trip(self):
        """context_length survives model_dump() -> model_validate() — AC 1."""
        config = ModelConfig(provider="openai", model="gpt-4o", context_length=64000)
        restored = ModelConfig.model_validate(config.model_dump())
        assert restored.context_length == 64000
        assert restored == config


class TestModelConfigFallbackModels:
    """Test ModelConfig.fallback_models field and its construction-time guards.

    AC 1, 2: field shape and round-trip. AC 3: nesting guard.
    AC 4, 5, 6: homogeneous native-output-support guard. AC 8: empty-list short-circuit.
    """

    # --- AC 1: the field exists and defaults to empty ---

    def test_fallback_models_defaults_to_empty(self):
        """A bare ModelConfig has an empty fallback chain — AC 1."""
        assert ModelConfig().fallback_models == []

    def test_existing_construction_unchanged(self):
        """An explicit config without fallbacks is unaffected by the new field — AC 1."""
        config = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7)
        assert config.fallback_models == []
        assert config.provider == "openai"
        assert config.temperature == 0.7

    # --- AC 2: a one-deep chain constructs and round-trips ---

    def test_two_entry_chain_round_trips_in_order(self):
        """Chain entries survive dump -> validate in the same order — AC 2."""
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            fallback_models=[
                ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
                ModelConfig(provider="azure", model="gpt-4o-mini"),
            ],
        )
        restored = ModelConfig.model_validate(config.model_dump())
        assert [m.model for m in restored.fallback_models] == [
            "claude-haiku-4-5-20251001",
            "gpt-4o-mini",
        ]
        assert restored == config

    def test_chain_json_round_trips(self):
        """The chain survives a JSON round-trip — no PrivateAttr, no custom encoder — AC 2."""
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            fallback_models=[ModelConfig(provider="azure", model="gpt-4o-mini")],
        )
        assert ModelConfig.model_validate_json(config.model_dump_json()) == config

    def test_fallback_models_is_a_public_pydantic_field(self):
        """fallback_models is a declared field, not a PrivateAttr — AC 2."""
        assert "fallback_models" in ModelConfig.model_fields
        assert "fallback_models" not in ModelConfig.__private_attributes__

    def test_no_arbitrary_types_allowed(self):
        """ModelConfig declares no arbitrary_types_allowed — AC 2."""
        assert not ModelConfig.model_config.get("arbitrary_types_allowed")

    # --- AC 3: nesting is rejected ---

    def test_nested_chain_rejected(self):
        """An entry that declares its own fallback_models is rejected — AC 3."""
        inner = ModelConfig(
            provider="openai",
            model="gpt-4o-mini",
            fallback_models=[ModelConfig(provider="azure", model="gpt-4o")],
        )
        with pytest.raises(ValidationError, match="cannot themselves declare fallback_models"):
            ModelConfig(provider="openai", model="gpt-4o", fallback_models=[inner])

    def test_inner_config_alone_still_constructs(self):
        """Only use *as an entry* fails; the inner config on its own is valid — AC 3."""
        inner = ModelConfig(
            provider="openai",
            model="gpt-4o-mini",
            fallback_models=[ModelConfig(provider="azure", model="gpt-4o")],
        )
        assert len(inner.fallback_models) == 1

    def test_nesting_reported_before_heterogeneity(self):
        """A chain that is both nested and heterogeneous reports the nesting rule — AC 3."""
        inner = ModelConfig(
            provider="mistral",
            model="mistral-large-latest",
            fallback_models=[ModelConfig(provider="google-gla", model="gemini-2.0-flash")],
        )
        with pytest.raises(ValidationError, match="cannot themselves declare fallback_models"):
            ModelConfig(provider="openai", model="gpt-4o", fallback_models=[inner])

    # --- AC 4: heterogeneous chains are rejected, error names the primary's value ---

    def test_supporting_primary_with_non_supporting_fallback_rejected(self):
        """Native-output primary + prompt-based fallback raises, naming True — AC 4."""
        with pytest.raises(ValidationError, match="supports_native_output=True"):
            ModelConfig(
                provider="openai",
                model="gpt-4o",
                fallback_models=[ModelConfig(provider="mistral", model="mistral-large-latest")],
            )

    def test_non_supporting_primary_with_supporting_fallback_rejected(self):
        """Prompt-based primary + native-output fallback raises, naming False — AC 4."""
        with pytest.raises(ValidationError, match="supports_native_output=False"):
            ModelConfig(
                provider="google-gla",
                model="gemini-2.0-flash",
                fallback_models=[ModelConfig(provider="anthropic", model="claude-sonnet-4-5")],
            )

    def test_error_names_the_offending_entry(self):
        """The rejection message identifies the mismatched entry — AC 4."""
        with pytest.raises(ValidationError, match="mistral-large-latest"):
            ModelConfig(
                provider="openai",
                model="gpt-4o",
                fallback_models=[ModelConfig(provider="mistral", model="mistral-large-latest")],
            )

    def test_mismatch_in_second_position_rejected(self):
        """A mismatch anywhere in the chain is caught, not just the first entry — AC 4."""
        with pytest.raises(ValidationError, match="supports_native_output=True"):
            ModelConfig(
                provider="openai",
                model="gpt-4o",
                fallback_models=[
                    ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
                    ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
                ],
            )

    # --- AC 5: homogeneous chains across different providers construct ---

    def test_homogeneous_supporting_chain_across_providers_accepted(self):
        """openai -> anthropic -> azure all support native output — AC 5."""
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            fallback_models=[
                ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
                ModelConfig(provider="azure", model="gpt-4o-mini"),
            ],
        )
        assert [m.provider for m in config.fallback_models] == ["anthropic", "azure"]

    def test_homogeneous_non_supporting_chain_across_providers_accepted(self):
        """google-gla -> mistral: neither supports native output — AC 5."""
        config = ModelConfig(
            provider="google-gla",
            model="gemini-2.0-flash",
            fallback_models=[ModelConfig(provider="mistral", model="mistral-large-latest")],
        )
        assert config.fallback_models[0].provider == "mistral"

    # --- AC 6: same provider, differing support, is rejected ---

    def test_same_provider_differing_support_rejected(self):
        """nvidia openai/* (supports) + nvidia meta/* (does not) raises — AC 6."""
        with pytest.raises(ValidationError, match="supports_native_output=True"):
            ModelConfig(
                provider="nvidia",
                model="openai/gpt-oss-120b",
                fallback_models=[
                    ModelConfig(provider="nvidia", model="meta/llama-3.1-70b-instruct")
                ],
            )

    def test_same_provider_matching_support_accepted(self):
        """Two nvidia openai/* models agree on support and construct — AC 6."""
        config = ModelConfig(
            provider="nvidia",
            model="openai/gpt-oss-120b",
            fallback_models=[ModelConfig(provider="nvidia", model="openai/gpt-oss-20b")],
        )
        assert config.fallback_models[0].model == "openai/gpt-oss-20b"

    # --- AC 8: both validators short-circuit on the empty default ---

    def test_empty_chain_skips_support_probe(self, monkeypatch):
        """With no fallbacks the homogeneity validator never probes support — AC 8."""

        def _boom(config: ModelConfig) -> bool:
            raise AssertionError("support probe must not run for an empty fallback chain")

        monkeypatch.setattr(config_module, "_supports_native_output", _boom)
        assert ModelConfig().fallback_models == []
        assert ModelConfig(provider="openai", model="gpt-4o").fallback_models == []


class TestRunUsageLimits:
    """Test RunUsageLimits model — the per-run tier."""

    def test_all_limits_none(self):
        """Test limits can be None (unlimited)."""
        limits = RunUsageLimits(
            run_request_limit=None,
            tool_calls_limit=None,
            input_tokens_limit=None,
            output_tokens_limit=None,
            total_tokens_limit=None,
        )
        assert limits.run_request_limit is None
        assert limits.tool_calls_limit is None
        assert limits.input_tokens_limit is None
        assert limits.output_tokens_limit is None
        assert limits.total_tokens_limit is None

    def test_specific_limits(self):
        """Test setting specific limits."""
        limits = RunUsageLimits(run_request_limit=10, total_tokens_limit=5000)
        assert limits.run_request_limit == 10
        assert limits.total_tokens_limit == 5000

    def test_all_limits_set(self):
        """Test all limits can be set."""
        limits = RunUsageLimits(
            run_request_limit=100,
            tool_calls_limit=50,
            input_tokens_limit=2000,
            output_tokens_limit=1000,
            total_tokens_limit=3000,
        )
        assert limits.run_request_limit == 100
        assert limits.tool_calls_limit == 50
        assert limits.input_tokens_limit == 2000
        assert limits.output_tokens_limit == 1000
        assert limits.total_tokens_limit == 3000

    def test_default_run_request_limit(self):
        """Test default run_request_limit is 50 — the same brake the pre-split tier had."""
        limits = RunUsageLimits()
        assert limits.run_request_limit == 50

    def test_invalid_negative_limit(self):
        """Test negative limits raise error."""
        with pytest.raises(ValidationError):
            RunUsageLimits(run_request_limit=-1)

    def test_invalid_zero_limit(self):
        """Test zero limits raise error."""
        with pytest.raises(ValidationError):
            RunUsageLimits(total_tokens_limit=0)

    def test_serialization(self):
        """Test model serialization."""
        limits = RunUsageLimits(run_request_limit=10, total_tokens_limit=5000)
        data = limits.model_dump()
        assert data["run_request_limit"] == 10
        assert data["total_tokens_limit"] == 5000

    def test_field_set(self):
        """Exact field set — a field landing on the wrong tier must fail here."""
        assert set(RunUsageLimits.model_fields) == {
            "run_request_limit",
            "tool_calls_limit",
            "input_tokens_limit",
            "output_tokens_limit",
            "total_tokens_limit",
        }

    def test_carries_no_agent_request_limit(self):
        """The run tier does not carry the agent tier's counter."""
        assert "agent_request_limit" not in RunUsageLimits.model_fields


class TestAgentUsageLimits:
    """Test AgentUsageLimits model — the agent-lifetime tier."""

    def test_field_set(self):
        """Exact field set: agent_request_limit plus the three inherited token fields."""
        assert set(AgentUsageLimits.model_fields) == {
            "agent_request_limit",
            "input_tokens_limit",
            "output_tokens_limit",
            "total_tokens_limit",
        }

    def test_carries_no_run_tier_fields(self):
        """tool_calls_limit and run_request_limit belong to the run tier only."""
        assert "tool_calls_limit" not in AgentUsageLimits.model_fields
        assert "run_request_limit" not in AgentUsageLimits.model_fields

    def test_agent_request_limit_defaults_to_none(self):
        """Unlimited by default — no lifetime brake unless one is asked for."""
        assert AgentUsageLimits().agent_request_limit is None

    def test_agent_request_limit_set(self):
        """The lifetime counter accepts a positive value."""
        assert AgentUsageLimits(agent_request_limit=3).agent_request_limit == 3

    def test_invalid_zero_agent_request_limit(self):
        """Zero is rejected (gt=0)."""
        with pytest.raises(ValidationError):
            AgentUsageLimits(agent_request_limit=0)

    def test_token_fields_carry_their_values(self):
        """Token fields round-trip; ReactAgent enforces them pre-flight (test_agent.py)."""
        limits = AgentUsageLimits(input_tokens_limit=100, total_tokens_limit=200)
        assert limits.input_tokens_limit == 100
        assert limits.total_tokens_limit == 200


class TestTokenUsageLimits:
    """Test TokenUsageLimits — the token-only base shared by both tiers."""

    def test_field_set(self):
        """The base carries exactly the three token fields and nothing else."""
        assert set(TokenUsageLimits.model_fields) == {
            "input_tokens_limit",
            "output_tokens_limit",
            "total_tokens_limit",
        }

    def test_request_names_are_not_fields(self):
        """Neither request-count spelling is a field on the base."""
        assert "request_limit" not in TokenUsageLimits.model_fields
        assert "run_request_limit" not in TokenUsageLimits.model_fields
        assert "tool_calls_limit" not in TokenUsageLimits.model_fields

    def test_defaults_are_none(self):
        """All three token limits default to unlimited."""
        limits = TokenUsageLimits()
        assert limits.input_tokens_limit is None
        assert limits.output_tokens_limit is None
        assert limits.total_tokens_limit is None

    def test_both_tiers_are_subclasses(self):
        """Both tiers inherit the token budget."""
        assert issubclass(RunUsageLimits, TokenUsageLimits)
        assert issubclass(AgentUsageLimits, TokenUsageLimits)


class TestAgentRuntimeConfig:
    """Test AgentRuntimeConfig model."""

    def test_defaults(self):
        """Test default values."""
        config = RuntimeConfig()
        assert config.retries == 3
        assert config.end_strategy == "exhaustive"
        assert config.parallel_tool_calls is True
        assert config.http_client_config.timeout == 120.0
        assert config.http_client_config.max_retries == 5
        assert config.http_client_config.backoff_multiplier == 0.5
        assert config.http_client_config.backoff_max == 60.0

    def test_custom_values(self):
        """Test custom configuration."""
        from akgentic.llm.config import HttpClientConfig

        config = RuntimeConfig(
            retries=5,
            end_strategy="early",
            parallel_tool_calls=False,
            http_client_config=HttpClientConfig(timeout=60.0, max_retries=3),
        )
        assert config.retries == 5
        assert config.end_strategy == "early"
        assert config.parallel_tool_calls is False
        assert config.http_client_config.timeout == 60.0
        assert config.http_client_config.max_retries == 3

    def test_invalid_strategy(self):
        """Test invalid end_strategy raises error."""
        with pytest.raises(ValidationError):
            RuntimeConfig(end_strategy="invalid")  # type: ignore

    def test_negative_retries_invalid(self):
        """Test negative retries raise error."""
        with pytest.raises(ValidationError):
            RuntimeConfig(retries=-1)

    def test_zero_retries_valid(self):
        """Test zero retries is valid."""
        config = RuntimeConfig(retries=0)
        assert config.retries == 0

    def test_serialization(self):
        """Test model serialization."""
        from akgentic.llm.config import HttpClientConfig

        config = RuntimeConfig(retries=5, http_client_config=HttpClientConfig(timeout=90.0))
        data = config.model_dump()
        assert data["retries"] == 5
        assert data["http_client_config"]["timeout"] == 90.0


class TestReactAgentConfig:
    """Test ReactAgentConfig model."""

    def test_full_config(self):
        """Test complete agent configuration."""
        from akgentic.llm.config import HttpClientConfig

        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o", temperature=0.7),
            run_usage_limits=RunUsageLimits(run_request_limit=10, total_tokens_limit=5000),
            runtime_cfg=RuntimeConfig(retries=5, http_client_config=HttpClientConfig(timeout=60.0)),
        )
        assert config.model_cfg.provider == "openai"
        assert config.model_cfg.model == "gpt-4o"
        assert config.runtime_cfg.retries == 5
        assert config.runtime_cfg.http_client_config.timeout == 60.0
        assert config.model_cfg.temperature == 0.7
        assert config.run_usage_limits.run_request_limit == 10
        assert config.run_usage_limits.total_tokens_limit == 5000
        assert config.runtime_cfg.retries == 5

    def test_defaults(self):
        """Test default values."""
        config = ReactAgentConfig()
        assert config.model_cfg.provider == "openai"
        assert config.model_cfg.model == "gpt-5.2"
        assert config.run_usage_limits is not None
        assert config.run_usage_limits.run_request_limit == 50
        assert config.agent_usage_limits is not None
        assert config.agent_usage_limits.agent_request_limit is None
        assert config.runtime_cfg.retries == 3
        assert config.runtime_cfg.http_client_config.timeout == 120.0

    def test_usage_limits_is_not_a_declared_field(self):
        """The pre-split field name no longer exists as a model field."""
        assert "usage_limits" not in ReactAgentConfig.model_fields
        assert "run_usage_limits" in ReactAgentConfig.model_fields
        assert "agent_usage_limits" in ReactAgentConfig.model_fields

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        )
        assert config.model_cfg.provider == "anthropic"
        assert config.model_cfg.model == "claude-3-5-sonnet-20241022"
        # Defaults should be set
        assert config.run_usage_limits is not None
        assert config.run_usage_limits.run_request_limit == 50
        assert config.runtime_cfg is not None

    def test_serialization(self):
        """Test model serialization."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            run_usage_limits=RunUsageLimits(run_request_limit=10),
        )
        data = config.model_dump()
        assert data["model_cfg"]["provider"] == "openai"
        assert data["run_usage_limits"]["run_request_limit"] == 10

    def test_json_serialization(self):
        """Test JSON serialization."""
        config = ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
        json_str = config.model_dump_json()
        assert "openai" in json_str
        assert "gpt-4o" in json_str

    def test_nested_validation(self):
        """Test nested validation errors propagate."""
        with pytest.raises(ValidationError):
            ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o", temperature=3.0)
            )


class TestCompactionConfig:
    """Test CompactionConfig model — AC 2, 3."""

    def test_defaults(self):
        """Default field values match the FR2 spec — AC 2."""
        cfg = CompactionConfig()
        assert cfg.strategy == "summarize"
        assert cfg.auto_trigger is True
        assert cfg.trigger_ratio == 0.85
        assert cfg.keep_recent_messages == 4
        assert cfg.summary_target_tokens == 2000
        assert cfg.summarizer_prompt_version == "v1"
        assert cfg.summary_model_cfg is None

    def test_summary_instructions_not_a_config_field(self):
        """The summarizer prompt is NOT stored on the config — only the small version id is,
        keeping the prompt text out of the serialized config / start events. The text is
        resolved via the SUMMARY_INSTRUCTIONS registry, keyed by summarizer_prompt_version."""
        assert "summary_instructions" not in CompactionConfig.model_fields
        assert CompactionConfig().summarizer_prompt_version == "v1"

    def test_strategy_accepts_arbitrary_fqcn(self):
        """strategy is a plain str — an arbitrary dotted FQCN round-trips unchanged — AC 2."""
        fqcn = "mypkg.compaction.HeadlineCompaction"
        cfg = CompactionConfig(strategy=fqcn)
        assert cfg.strategy == fqcn
        assert CompactionConfig.model_validate(cfg.model_dump()).strategy == fqcn

    def test_trigger_ratio_one_accepted(self):
        """trigger_ratio=1.0 is at the le=1.0 boundary — AC 2."""
        assert CompactionConfig(trigger_ratio=1.0).trigger_ratio == 1.0

    def test_trigger_ratio_zero_invalid(self):
        """trigger_ratio=0.0 violates gt=0 — AC 2."""
        with pytest.raises(ValidationError):
            CompactionConfig(trigger_ratio=0.0)

    def test_trigger_ratio_above_one_invalid(self):
        """trigger_ratio>1.0 violates le=1.0 — AC 2."""
        with pytest.raises(ValidationError):
            CompactionConfig(trigger_ratio=1.0001)

    def test_keep_recent_messages_zero_accepted(self):
        """keep_recent_messages=0 is at the ge=0 boundary — AC 2."""
        assert CompactionConfig(keep_recent_messages=0).keep_recent_messages == 0

    def test_keep_recent_messages_negative_invalid(self):
        """keep_recent_messages=-1 violates ge=0 — AC 2."""
        with pytest.raises(ValidationError):
            CompactionConfig(keep_recent_messages=-1)

    def test_summary_target_tokens_zero_invalid(self):
        """summary_target_tokens=0 violates gt=0 — AC 2."""
        with pytest.raises(ValidationError):
            CompactionConfig(summary_target_tokens=0)

    def test_summary_model_cfg_nesting(self):
        """summary_model_cfg accepts a nested ModelConfig — AC 2."""
        nested = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        cfg = CompactionConfig(summary_model_cfg=nested)
        assert cfg.summary_model_cfg == nested

    def test_default_round_trip(self):
        """A default CompactionConfig round-trips equal to itself — AC 2."""
        cfg = CompactionConfig()
        assert CompactionConfig.model_validate(cfg.model_dump()) == cfg

    def test_no_arbitrary_types_allowed(self):
        """CompactionConfig declares no arbitrary_types_allowed — AC 3."""
        assert not CompactionConfig.model_config.get("arbitrary_types_allowed")

    def test_rejects_non_serializable_summary_model_cfg(self):
        """A non-serializable object is rejected by the typed summary_model_cfg field — AC 3."""
        with pytest.raises(ValidationError):
            CompactionConfig(summary_model_cfg=object())  # type: ignore[arg-type]


class TestReactAgentCompactionConfig:
    """Test ReactAgentConfig compaction_cfg / max_messages fields + validators — AC 4, 8, 9."""

    def test_default_exposes_compaction_config(self):
        """A default ReactAgentConfig exposes a default CompactionConfig — AC 4."""
        config = ReactAgentConfig()
        assert isinstance(config.compaction_cfg, CompactionConfig)
        assert config.compaction_cfg.strategy == "summarize"

    def test_max_messages_default_none(self):
        """max_messages defaults to None (unlimited) — AC 4."""
        assert ReactAgentConfig().max_messages is None

    def test_max_messages_zero_accepted(self):
        """max_messages=0 is at the ge=0 boundary (auto_trigger off so window allowed) — AC 4."""
        config = ReactAgentConfig(
            max_messages=0, compaction_cfg=CompactionConfig(auto_trigger=False)
        )
        assert config.max_messages == 0

    def test_max_messages_negative_invalid(self):
        """max_messages=-1 violates ge=0 — AC 4."""
        with pytest.raises(ValidationError):
            ReactAgentConfig(max_messages=-1, compaction_cfg=CompactionConfig(auto_trigger=False))

    def test_compaction_fields_round_trip(self):
        """compaction_cfg and max_messages survive a dump -> validate round-trip — AC 4."""
        config = ReactAgentConfig(
            max_messages=10, compaction_cfg=CompactionConfig(auto_trigger=False, strategy="noop")
        )
        restored = ReactAgentConfig.model_validate(config.model_dump())
        assert restored.max_messages == 10
        assert restored.compaction_cfg.strategy == "noop"
        assert restored == config

    # --- AC 8: window-exclusivity validator ---

    def test_window_with_auto_trigger_rejected(self):
        """auto_trigger=True + a configured window raises — AC 8."""
        with pytest.raises(ValidationError):
            ReactAgentConfig(compaction_cfg=CompactionConfig(auto_trigger=True), max_messages=10)

    def test_window_without_auto_trigger_accepted(self):
        """auto_trigger=False + max_messages=10 constructs — AC 8."""
        config = ReactAgentConfig(
            compaction_cfg=CompactionConfig(auto_trigger=False), max_messages=10
        )
        assert config.max_messages == 10

    def test_auto_trigger_without_window_accepted(self):
        """auto_trigger=True + max_messages=None constructs — AC 8."""
        config = ReactAgentConfig(compaction_cfg=CompactionConfig(auto_trigger=True))
        assert config.max_messages is None

    # --- AC 9: threshold-vs-usage-limit validator ---

    def test_threshold_at_or_above_input_limit_rejected(self):
        """threshold >= input_tokens_limit raises — AC 9."""
        with pytest.raises(ValidationError):
            ReactAgentConfig(
                model_cfg=ModelConfig(context_length=1000),
                run_usage_limits=RunUsageLimits(input_tokens_limit=850),
            )

    def test_threshold_at_or_above_total_limit_rejected(self):
        """threshold >= total_tokens_limit raises — AC 9."""
        with pytest.raises(ValidationError):
            ReactAgentConfig(
                model_cfg=ModelConfig(context_length=1000),
                run_usage_limits=RunUsageLimits(total_tokens_limit=800),
            )

    def test_threshold_strictly_below_both_accepted(self):
        """threshold strictly below both limits constructs — AC 9."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(context_length=1000),
            run_usage_limits=RunUsageLimits(input_tokens_limit=2000, total_tokens_limit=3000),
        )
        assert config.model_cfg.context_length == 1000

    def test_threshold_with_both_limits_none_accepted(self):
        """No token limits set -> threshold check passes — AC 9."""
        config = ReactAgentConfig(model_cfg=ModelConfig(context_length=1000))
        assert config.model_cfg.context_length == 1000

    def test_threshold_skipped_when_context_length_none(self):
        """context_length=None skips the threshold check regardless of limits — AC 9."""
        config = ReactAgentConfig(run_usage_limits=RunUsageLimits(input_tokens_limit=1))
        assert config.run_usage_limits.input_tokens_limit == 1

    def test_threshold_skipped_when_auto_trigger_false(self):
        """auto_trigger=False skips the threshold check regardless of limits — AC 9."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(context_length=1000),
            compaction_cfg=CompactionConfig(auto_trigger=False),
            run_usage_limits=RunUsageLimits(input_tokens_limit=1),
        )
        assert config.compaction_cfg.auto_trigger is False

    def test_agent_tier_token_limits_do_not_arm_the_threshold_validator(self):
        """The validator reads the RUN tier only — an agent-tier limit never trips it.

        Same numbers that raise on run_usage_limits (threshold 850 >= 850) construct
        cleanly here, which pins the validator to one tier rather than to the values.
        """
        config = ReactAgentConfig(
            model_cfg=ModelConfig(context_length=1000),
            agent_usage_limits=AgentUsageLimits(input_tokens_limit=850, total_tokens_limit=800),
        )
        assert config.agent_usage_limits.input_tokens_limit == 850
        assert config.agent_usage_limits.total_tokens_limit == 800


class TestCompactionBudgetIsPrimaryOnly:
    """The compaction threshold reads the PRIMARY model's context_length only.

    A fallback firing mid-run does not change the compaction budget for the rest of
    that run. Regression pin on existing behaviour — story 14-1 changes no production
    code for this. Both fallback entries below share the primary's native-output
    support so these cases exercise the budget rule, not the homogeneity guard.
    """

    def test_primary_threshold_still_rejected_despite_tiny_fallback_window(self):
        """A tiny fallback context_length does not rescue an over-limit primary threshold."""
        with pytest.raises(ValidationError, match="compaction threshold"):
            ReactAgentConfig(
                model_cfg=ModelConfig(
                    provider="openai",
                    model="gpt-4o",
                    context_length=1000,
                    fallback_models=[
                        ModelConfig(
                            provider="anthropic", model="claude-sonnet-4-5", context_length=10
                        )
                    ],
                ),
                run_usage_limits=RunUsageLimits(input_tokens_limit=850),
            )

    def test_primary_threshold_still_accepted_despite_huge_fallback_window(self):
        """A huge fallback context_length does not push a safe primary threshold over."""
        config = ReactAgentConfig(
            model_cfg=ModelConfig(
                provider="openai",
                model="gpt-4o",
                context_length=1000,
                fallback_models=[
                    ModelConfig(
                        provider="anthropic", model="claude-sonnet-4-5", context_length=2_000_000
                    )
                ],
            ),
            run_usage_limits=RunUsageLimits(input_tokens_limit=2000, total_tokens_limit=3000),
        )
        assert config.model_cfg.context_length == 1000
        assert config.model_cfg.fallback_models[0].context_length == 2_000_000
