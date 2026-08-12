"""Configuration models for LLM provider settings.

This module provides Pydantic models for configuring LLM providers,
usage limits, and agent execution settings.

Examples:
    Basic model configuration:

    >>> from akgentic.llm import ModelConfig
    >>> config = ModelConfig(
    ...     provider="openai",
    ...     model="gpt-4o",
    ...     temperature=0.7
    ... )

    Configuration with usage limits:

    >>> from akgentic.llm import ModelConfig, RunUsageLimits, ReactAgentConfig
    >>> config = ReactAgentConfig(
    ...     model=ModelConfig(provider="openai", model="gpt-4o"),
    ...     run_usage_limits=RunUsageLimits(run_request_limit=10, total_tokens_limit=5000)
    ... )
"""

import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Release that deletes the pre-split usage-limits shim. Named in every deprecation
# warning and docstring below so removing it is a scheduled task, not a hunt.
_SHIM_REMOVAL_RELEASE = "akgentic-llm 2.0.0"


class ModelConfig(BaseModel):
    """Configuration for LLM model settings.

    Supports multiple providers with provider-agnostic configuration.
    Provider-specific authentication is handled via environment variables:
    - OpenAI: OPENAI_API_KEY
    - Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    - Anthropic: ANTHROPIC_API_KEY
    - Google: GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS
    - Mistral: MISTRAL_API_KEY
    - NVIDIA: NVIDIA_API_KEY

    Attributes:
        provider: LLM provider name
        model: Model identifier (provider-specific naming)
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = maximum creativity)
        seed: Random seed for reproducible outputs (not supported by all providers)
        max_tokens: Maximum tokens in model response (None = provider default/maximum)
        context_length: Model context window in tokens; the budget that auto-triggers
            compaction. None = compaction off. Distinct from max_tokens (the output cap).
        reasoning_effort: Reasoning effort for o1/o3-style models ('low', 'medium', 'high')
        fallback_models: Models tried in order after this one on API failure. The chain is
            flat (an entry may not declare its own fallbacks) and homogeneous (every entry
            must agree with this config on native structured-output support), both enforced
            at construction. Only this config's context_length governs the compaction budget.

    Example:
        >>> # OpenAI GPT-4o with moderate creativity
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>>
        >>> # Anthropic Claude with deterministic output
        >>> config = ModelConfig(
        ...     provider="anthropic",
        ...     model="claude-3-5-sonnet-20241022",
        ...     temperature=0.0,
        ...     seed=42
        ... )
        >>>
        >>> # OpenAI o1 with high reasoning effort
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="o1",
        ...     reasoning_effort="high"
        ... )
        >>>
        >>> # A fallback chain: gpt-4o first, then Claude, then Azure. All three
        >>> # support native structured output, so the chain is homogeneous.
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     fallback_models=[
        ...         ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ...         ModelConfig(provider="azure", model="gpt-4o-mini"),
        ...     ],
        ... )
    """

    provider: Literal["openai", "azure", "nvidia", "google-gla", "mistral", "anthropic"] = Field(
        default="openai", description="Model provider"
    )

    model: str = Field(
        default="gpt-5.2",
        description="Model identifier (e.g., gpt-5.2, claude-3-5-sonnet-20241022)",
    )

    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )

    seed: int | None = Field(default=None, description="Random seed for reproducible outputs")

    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens in model response"
    )

    context_length: int | None = Field(
        default=None,
        gt=0,
        description="Model context window in tokens; the budget that auto-triggers compaction. None = off.",  # noqa: E501
    )

    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort for o1/o3 models"
    )

    fallback_models: list["ModelConfig"] = Field(
        default_factory=list,
        description=(
            "Models tried in order after this one on API failure (rate limits, 5xx, "
            "auth errors, timeouts). Empty = no fallback (default)."
        ),
    )

    @model_validator(mode="after")
    def _reject_nested_fallback_models(self) -> "ModelConfig":
        """Flat-chain: a fallback entry may not declare fallbacks of its own.

        pydantic-ai's FallbackModel takes (default_model, *fallback_models) and stores a
        flat list, never a tree, so a nested chain would have to be silently flattened at
        build time. Rejecting it here turns that silent flatten into a clear error and
        bounds both the homogeneity check and the primary-governs-config rule to one level.
        """
        if any(entry.fallback_models for entry in self.fallback_models):
            raise ValueError(
                "fallback_models entries cannot themselves declare fallback_models; "
                "the chain is flat — list every model on the primary config"
            )
        return self

    @model_validator(mode="after")
    def _reject_heterogeneous_output_support(self) -> "ModelConfig":
        """Homogeneous-support: every entry must agree with the primary on native output.

        The structured-output wrapper is chosen from the provider alone, once, before any
        request is sent. An entry whose support differs makes that wrapper wrong for
        whichever model actually serves the response — a corruption that only surfaces
        when the primary fails, which is exactly when nobody is watching.
        """
        if not self.fallback_models:
            return self
        primary = _supports_native_output(self)
        mismatched = [
            f"{entry.provider}/{entry.model}"
            for entry in self.fallback_models
            if _supports_native_output(entry) != primary
        ]
        if mismatched:
            raise ValueError(
                f"fallback_models must all match the primary's "
                f"supports_native_output={primary}; mismatched entries: "
                f"{', '.join(mismatched)}"
            )
        return self


def _supports_native_output(config: ModelConfig) -> bool:
    """Check if provider supports native structured output via NativeOutput wrapper.

    Providers with native support (via function calling or tool use APIs):
    - openai: GPT-4o, o1 series, etc.
    - azure: Azure OpenAI Service
    - anthropic: Claude 3.5 Sonnet, etc.
    - nvidia: Only for models with "openai" prefix (e.g., "openai/gpt-oss-120b")

    Providers without native support (use prompt-based extraction):
    - google-gla: Google Gemini models
    - mistral: Mistral AI models
    - nvidia: Non-OpenAI models (e.g., "meta/llama-3.1-70b-instruct")

    Defined here rather than in providers.py so ModelConfig's fallback-chain validator can
    call it: providers.py already imports ModelConfig from this module, so importing the
    predicate back would cycle. providers.py re-imports it — this is its one definition.

    Args:
        config: LLM model configuration.

    Returns:
        True if the provider supports native structured output, False otherwise.

    Example:
        >>> config = ModelConfig(provider="openai", model="gpt-4o")
        >>> _supports_native_output(config)
        True
        >>> config = ModelConfig(provider="google-gla", model="gemini-2.0-flash")
        >>> _supports_native_output(config)
        False
        >>> config = ModelConfig(provider="nvidia", model="openai/gpt-oss-120b")
        >>> _supports_native_output(config)
        True
    """
    if config.provider in ("openai", "azure", "anthropic"):
        return True
    if config.provider == "nvidia":
        return config.model.startswith("openai")
    return False


class CompactionConfig(BaseModel):
    """Configuration for pluggable LLM context compaction.

    Dormant schema until the engine wiring consumes it. ``strategy`` is a plain
    ``str`` (a registry id or a dotted FQCN), never a ``Literal``.

    Attributes:
        strategy: Compaction strategy id or dotted FQCN (e.g. "summarize").
        auto_trigger: Whether usage-based auto-compaction is enabled.
        trigger_ratio: Fraction of context_length that arms the auto-trigger.
        keep_recent_messages: Trailing messages preserved verbatim (counts messages, not pairs).
        summary_target_tokens: Token budget the summarizer aims for.
        summarizer_prompt_version: Version tag selecting the summarizer instructions from
            the ``SUMMARY_INSTRUCTIONS`` registry (compaction.py); also recorded on the
            emitted ``LlmContextCompactedEvent``.
        summary_model_cfg: Optional model for summarization; None reuses the agent's model_cfg.
    """

    strategy: str = Field(default="summarize", description="Strategy id or dotted FQCN")

    auto_trigger: bool = Field(default=True, description="Enable usage-based auto-compaction")

    trigger_ratio: float = Field(
        default=0.85, gt=0, le=1.0, description="Fraction of context_length that arms the trigger"
    )

    keep_recent_messages: int = Field(
        default=4, ge=0, description="Trailing messages preserved verbatim"
    )

    summary_target_tokens: int = Field(
        default=2000, gt=0, description="Token budget the summarizer aims for"
    )

    summarizer_prompt_version: str = Field(
        default="v1", description="Version tag of the summarizer prompt"
    )

    summary_model_cfg: ModelConfig | None = Field(
        default=None, description="Optional summarizer model; None reuses the agent's model_cfg"
    )


class TokenUsageLimits(BaseModel):
    """Token budgets shared by both usage-limit tiers.

    Internal base — callers construct RunUsageLimits or AgentUsageLimits, never this.
    All limits are optional (None = unlimited) and cumulative over their tier's scope.

    Attributes:
        input_tokens_limit: Maximum cumulative input/prompt tokens
        output_tokens_limit: Maximum cumulative output/completion tokens
        total_tokens_limit: Maximum cumulative total tokens (input + output)
    """

    input_tokens_limit: int | None = Field(
        default=None, gt=0, description="Maximum input/prompt tokens"
    )

    output_tokens_limit: int | None = Field(
        default=None, gt=0, description="Maximum output/completion tokens"
    )

    total_tokens_limit: int | None = Field(
        default=None, gt=0, description="Maximum total tokens (input + output)"
    )


class RunUsageLimits(TokenUsageLimits):
    """Per-run usage budget: bounds a single ReactAgent.run() call.

    Enforced by pydantic-ai, which raises UsageLimitExceeded mid-run when a limit is hit;
    ReactAgent surfaces that as UsageLimitError. Token counts reset every run.

    Attributes:
        run_request_limit: Maximum LLM API requests in one run (50 = the default brake)
        tool_calls_limit: Maximum tool invocations in one run

    Example:
        >>> # Basic limits: 10 requests, 5K total tokens
        >>> limits = RunUsageLimits(run_request_limit=10, total_tokens_limit=5000)
        >>>
        >>> # Strict limits for cost control
        >>> limits = RunUsageLimits(
        ...     run_request_limit=50,      # default: 50
        ...     tool_calls_limit=20,       # default: None
        ...     input_tokens_limit=10000,  # default: None
        ...     output_tokens_limit=2000,  # default: None
        ... )
        >>>
        >>> # Unlimited, safety brake included
        >>> limits = RunUsageLimits(run_request_limit=None, total_tokens_limit=None)
    """

    run_request_limit: int | None = Field(
        default=50, gt=0, description="Maximum number of LLM requests per run"
    )

    tool_calls_limit: int | None = Field(
        default=None, gt=0, description="Maximum number of tool calls per run"
    )


class AgentUsageLimits(TokenUsageLimits):
    """Agent-lifetime usage budget: bounds an agent across every run it performs.

    TRAP: nothing here is enforced yet. agent_request_limit is declared for the
    pre-flight run counter that lands later on this epic, and the inherited token
    fields are declared for shape symmetry with the run tier and are never read.

    Attributes:
        agent_request_limit: Maximum ReactAgent.run() calls over the agent's lifetime

    Example:
        >>> limits = AgentUsageLimits(agent_request_limit=100)
    """

    agent_request_limit: int | None = Field(
        default=None, gt=0, description="Maximum number of runs over the agent's lifetime"
    )


def _fold_pre_split_request_limit(data: dict[str, Any], owner: str) -> dict[str, Any]:
    """Fold a pre-split ``request_limit`` key onto ``run_request_limit``.

    Shared by the UsageLimits shim and the ReactAgentConfig keyword shim, which has to
    reach inside a mapping value: a dict routed to ``run_usage_limits`` is validated as
    RunUsageLimits, where ``request_limit`` is an unknown key that Pydantic drops in
    silence — accepted-and-discarded, the one failure mode the shim exists to prevent.

    Raises:
        ValueError: if both spellings are present; which wins would depend on order.
    """
    if "request_limit" not in data:
        return data
    if "run_request_limit" in data:
        raise ValueError(
            f"{owner} received both request_limit (deprecated) and run_request_limit; "
            "which one wins would depend on argument order — pass only run_request_limit"
        )
    mapped = dict(data)
    mapped["run_request_limit"] = mapped.pop("request_limit")
    return mapped


class UsageLimits(RunUsageLimits):
    """DEPRECATED alias of RunUsageLimits, the run tier.

    Accepts the pre-split ``request_limit=`` spelling and maps it onto
    ``run_request_limit``; ``.request_limit`` reads back through to the same field.
    Removed in akgentic-llm 2.0.0 — migrate to ``RunUsageLimits(run_request_limit=...)``.

    ``request_limit`` is a read accessor, never a field: a second storage slot would
    reintroduce the split-brain state the rename exists to remove.
    """

    @model_validator(mode="before")
    @classmethod
    def _map_pre_split_request_limit(cls, data: Any) -> Any:
        """Warn on construction and fold ``request_limit`` into ``run_request_limit``.

        Runs before field validation, so ``request_limit`` never reaches Pydantic as an
        unexpected keyword. Non-dict input (model_validate of an instance) passes through.

        The both-spellings ValueError is raised BEFORE the warning: under
        ``-W error::DeprecationWarning`` a warning emitted first would propagate in place
        of the error, hiding what the caller actually got wrong.
        """
        if not isinstance(data, dict):
            return data
        folded = _fold_pre_split_request_limit(data, "UsageLimits")
        warnings.warn(
            f"UsageLimits is deprecated and will be removed in {_SHIM_REMOVAL_RELEASE}; "
            "use RunUsageLimits(run_request_limit=...) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return folded

    @property
    def request_limit(self) -> int | None:
        """DEPRECATED read accessor for ``run_request_limit``.

        Removed in akgentic-llm 2.0.0. Reflects the underlying field regardless of
        which spelling set it.
        """
        warnings.warn(
            f"UsageLimits.request_limit is deprecated and will be removed in "
            f"{_SHIM_REMOVAL_RELEASE}; read run_request_limit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_request_limit


class HttpClientConfig(BaseModel):
    """HTTP client configuration for LLM API communication.

    Configures timeout and retry behavior with exponential backoff for transient failures:
    - Retries on: HTTP 429 (rate limit), 503 (service unavailable), connection errors
    - Backoff formula: min(max_delay, multiplier * (2 ** attempt))

    Attributes:
        timeout: Maximum time for single LLM request (connection + response)
        max_retries: Maximum retry attempts for transient HTTP failures
        backoff_multiplier: Base delay multiplier for exponential backoff (seconds)
        backoff_max: Cap on backoff delay to prevent excessive waiting

    Example:
        >>> # Default: 120s timeout, 5 retries with exponential backoff
        >>> client = HttpClientConfig()
        >>>
        >>> # Aggressive: fast timeout, fewer retries
        >>> client = HttpClientConfig(
        ...     timeout=30.0,
        ...     max_retries=2,
        ...     backoff_multiplier=0.2
        ... )
        >>>
        >>> # Conservative: long timeout, many retries
        >>> client = HttpClientConfig(
        ...     timeout=300.0,
        ...     max_retries=10,
        ...     backoff_multiplier=1.0,
        ...     backoff_max=120.0
        ... )
    """

    timeout: float = Field(
        default=120.0,
        gt=0,
        description="Maximum duration for single LLM request including connection and response time",  # noqa: E501
    )

    max_retries: int = Field(
        default=5,
        ge=1,
        description="Maximum HTTP retry attempts for transient failures (rate limits, service unavailable)",  # noqa: E501
    )

    backoff_multiplier: float = Field(
        default=0.5,
        gt=0,
        description="Base delay multiplier for exponential backoff: delay = multiplier * (2 ** attempt)",  # noqa: E501
    )

    backoff_max: float = Field(
        default=60.0,
        gt=0,
        description="Maximum backoff delay in seconds to prevent excessive retry waits",
    )


class RuntimeConfig(BaseModel):
    """Runtime configuration for agent behavior.

    Attributes:
        retries: Number of retry attempts for tool call failures and output validation errors
        end_strategy: Tool execution termination strategy
        parallel_tool_calls: Enable concurrent tool execution when model supports it
        http_client_config: HTTP client configuration for API communication
            (timeout and retry settings)

    Tool Execution Strategies:
        - 'early': Stops after first successful result (fast path)
        - 'exhaustive': Executes all tool calls even when result available (complete data gathering)

    Example:
        >>> # Default: resilient with standard HTTP settings
        >>> runtime = RuntimeConfig()
        >>>
        >>> # Aggressive: fast timeout, fewer retries
        >>> runtime = RuntimeConfig(
        ...     http_client_config=HttpClientConfig(
        ...         timeout=30.0, max_retries=2, backoff_multiplier=0.2
        ...     )
        ... )
        >>>
        >>> # Conservative: long timeout, many retries
        >>> runtime = RuntimeConfig(
        ...     http_client_config=HttpClientConfig(
        ...         timeout=300.0, max_retries=10, backoff_multiplier=1.0, backoff_max=120.0
        ...     )
        ... )
    """

    retries: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts for tool call failures and output validation errors",
    )

    end_strategy: Literal["early", "exhaustive"] = Field(
        default="exhaustive",
        description="Tool execution strategy: 'early' stops after first result, 'exhaustive' runs all tools",  # noqa: E501
    )

    parallel_tool_calls: bool = Field(
        default=True,
        description="Enable parallel tool execution when model supports concurrent calls",
    )

    http_client_config: HttpClientConfig = Field(
        default_factory=HttpClientConfig,
        description="HTTP client configuration for API communication",
    )


class ReactAgentConfig(BaseModel):
    """Configuration for REACT (Reasoning + Acting) pattern agent.

    Combines model settings, resource limits, and runtime behavior into a
    unified configuration for ReactAgent execution. This config is passed to
    ReactAgent during initialization and controls all aspects of LLM interaction.

    The REACT pattern alternates between:
    1. Reasoning: LLM generates thoughts and decides on actions
    2. Acting: Execute tools/functions based on LLM decisions
    3. Observing: Feed tool results back to LLM
    4. Repeat until task completion

    Attributes:
        model_cfg: LLM provider and model settings.
        runtime_cfg: Execution behavior and HTTP retry strategy.
        run_usage_limits: Per-run resource limits; the tier pydantic-ai enforces.
        agent_usage_limits: Agent-lifetime resource limits; not enforced yet.
        compaction_cfg: Context-compaction configuration.
        max_messages: Sliding-window size handed to ContextManager; None = unlimited.

    Deprecated:
        ``usage_limits`` survives as a constructor keyword and a read accessor for
        ``run_usage_limits``. Both warn, and both are removed in akgentic-llm 2.0.0.
        Passing ``usage_limits`` and ``run_usage_limits`` together raises ValueError.

    Example:
        >>> # Minimal configuration with defaults
        >>> config = ReactAgentConfig(
        ...     model_cfg=ModelConfig(provider="openai", model="gpt-4o")
        ... )
        >>>
        >>> # Full configuration with limits and custom behavior
        >>> config = ReactAgentConfig(
        ...     model_cfg=ModelConfig(
        ...         provider="anthropic",
        ...         model="claude-3-5-sonnet-20241022",
        ...         temperature=0.7
        ...     ),
        ...     run_usage_limits=RunUsageLimits(
        ...         run_request_limit=10,
        ...         total_tokens_limit=50000
        ...     ),
        ...     runtime_cfg=RuntimeConfig(
        ...         end_strategy="exhaustive",
        ...         http_client=HttpClientConfig(timeout=180.0)
        ...     )
        ... )
    """

    model_cfg: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")

    runtime_cfg: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Runtime behavior configuration"
    )

    run_usage_limits: RunUsageLimits = Field(
        default_factory=RunUsageLimits, description="Per-run usage limits for cost control"
    )

    agent_usage_limits: AgentUsageLimits = Field(
        default_factory=AgentUsageLimits,
        description="Agent-lifetime usage limits (declared, not yet enforced)",
    )

    compaction_cfg: CompactionConfig = Field(
        default_factory=CompactionConfig, description="Context-compaction configuration"
    )

    max_messages: int | None = Field(
        default=None,
        ge=0,
        description="Sliding-window size handed to ContextManager; None = unlimited",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_pre_split_usage_limits(cls, data: Any) -> Any:
        """Warn on the deprecated ``usage_limits=`` keyword and route it to the run tier.

        Runs before field validation, so ``usage_limits`` never reaches Pydantic as an
        unexpected keyword. The value must actually land on ``run_usage_limits``:
        accepting it and discarding it would leave the agent on a budget nobody chose.

        A mapping value gets the same treatment one level down, because it is validated
        as RunUsageLimits — where a pre-split ``request_limit`` key would be dropped in
        silence, which is the same failure with an extra layer of indirection.
        """
        if not isinstance(data, dict) or "usage_limits" not in data:
            return data
        if "run_usage_limits" in data:
            raise ValueError(
                "ReactAgentConfig received both usage_limits (deprecated) and "
                "run_usage_limits; which one wins would depend on argument order — "
                "pass only run_usage_limits"
            )
        mapped = dict(data)
        value = mapped.pop("usage_limits")
        if isinstance(value, dict):
            value = _fold_pre_split_request_limit(value, "ReactAgentConfig(usage_limits=...)")
        warnings.warn(
            f"ReactAgentConfig(usage_limits=...) is deprecated and will be removed in "
            f"{_SHIM_REMOVAL_RELEASE}; use run_usage_limits=... instead",
            DeprecationWarning,
            stacklevel=3,
        )
        mapped["run_usage_limits"] = value
        return mapped

    @property
    def usage_limits(self) -> RunUsageLimits:
        """DEPRECATED read accessor for ``run_usage_limits``.

        Removed in akgentic-llm 2.0.0. Returns the run tier itself, not a copy.
        """
        warnings.warn(
            f"ReactAgentConfig.usage_limits is deprecated and will be removed in "
            f"{_SHIM_REMOVAL_RELEASE}; read run_usage_limits instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_usage_limits

    @model_validator(mode="after")
    def _reject_window_with_auto_compaction(self) -> "ReactAgentConfig":
        """Window-exclusivity: auto-compaction and a sliding window are mutually exclusive.

        The window drops the oldest messages without emitting an event, which would make
        replaced_message_count ambiguous once compaction also rewrites history.
        """
        if self.compaction_cfg.auto_trigger and self.max_messages is not None:
            raise ValueError(
                "max_messages (sliding window) cannot be combined with "
                "compaction_cfg.auto_trigger=True; disable one of them"
            )
        return self

    @model_validator(mode="after")
    def _reject_threshold_above_run_usage_limits(self) -> "ReactAgentConfig":
        """Threshold-vs-usage-limit: keep the auto-trigger reachable before usage limits bite.

        When auto-compaction is live, the effective threshold must sit strictly below every
        set token limit; otherwise pydantic-ai raises UsageLimitExceeded first and the
        auto-trigger is dead code. Reads the RUN tier only — the agent tier's token fields
        are never enforced, so they cannot pre-empt the trigger.
        """
        context_length = self.model_cfg.context_length
        if not (self.compaction_cfg.auto_trigger and context_length is not None):
            return self
        threshold = int(context_length * self.compaction_cfg.trigger_ratio)
        for name in ("input_tokens_limit", "total_tokens_limit"):
            limit = getattr(self.run_usage_limits, name)
            if limit is not None and threshold >= limit:
                raise ValueError(
                    f"compaction threshold {threshold} must be strictly below "
                    f"run_usage_limits.{name} ({limit})"
                )
        return self
