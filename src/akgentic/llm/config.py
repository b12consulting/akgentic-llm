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
    ...     model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
    ...     run_usage_limits=RunUsageLimits(run_request_limit=10, total_tokens_limit=5000)
    ... )
"""

import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Removal schedule for the pre-split usage-limits shim, interpolated into every
# deprecation warning below so the schedule is stated in exactly one place.
#
# There is deliberately no release number here. The shim was announced for removal
# in 2.0.0, but 2.0.0 turned out to be the release that carried the move to
# pydantic-ai v2 -- the major bump was forced by the dependency, not by this
# deprecation, and the two collided on a number. The shim shipped through it. Naming
# any future release here would recreate that same defect at the next forced major,
# so the schedule stays open until someone actually schedules it.
_SHIM_REMOVAL_NOTICE = "no removal release is scheduled"


class ModelConfig(BaseModel):
    """Configuration for LLM model settings.

    Supports multiple providers with provider-agnostic configuration.
    Provider-specific authentication is handled via environment variables:
    - OpenAI: OPENAI_API_KEY
    - Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    - Anthropic: ANTHROPIC_API_KEY
    - Google: GOOGLE_API_KEY or GEMINI_API_KEY (one is required; ADC is not consulted)
    - Mistral: MISTRAL_API_KEY
    - NVIDIA: OPENAI_API_KEY (no api_key is passed to OpenAIProvider, so its own
      OPENAI_API_KEY fallback applies; a missing key surfaces as a 401 at request
      time, not at construction). Endpoint from NVIDIA_BASE_URL, which defaults to
      https://integrate.api.nvidia.com/v1

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
        ...     provider="openai-chat",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>>
        >>> # OpenAI GPT-5.6-luna with responses api
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
        ...     provider="openai-chat",
        ...     model="gpt-4o",
        ...     fallback_models=[
        ...         ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ...         ModelConfig(provider="azure", model="gpt-4o-mini"),
        ...     ],
        ... )
    """

    provider: Literal[
        "openai",
        "openai-chat",
        "azure",
        "azure-chat",
        "nvidia",
        "google-gla",
        "mistral",
        "anthropic",
    ] = Field(default="openai", description="Model provider")

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
    - openai: GPT-4o, o1 series, etc. (Responses API)
    - openai-chat: OpenAI models via the legacy Chat Completions API
    - azure: Azure OpenAI Service (Responses API)
    - azure-chat: Azure OpenAI Service via the legacy Chat Completions API
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
    if config.provider in ("openai", "openai-chat", "azure", "azure-chat", "anthropic"):
        return True
    if config.provider == "nvidia":
        return config.model.startswith("openai")
    return False


def model_roster_key(cfg: ModelConfig) -> str:
    """Return the roster identity of a model config: ``"{provider}:{model}"``.

    The key grammar, defined exactly once. Other packages project a roster onto their
    own row types and must key those rows the same way — a re-spelled grammar is a
    model switch that silently matches nothing. Import this rather than reformatting it.

    Args:
        cfg: The model configuration to key.

    Returns:
        The ``provider:model`` key.

    Example:
        >>> model_roster_key(ModelConfig(provider="azure", model="gpt-4o-mini"))
        'azure:gpt-4o-mini'
    """
    return f"{cfg.provider}:{cfg.model}"


def validate_unique_roster_keys(roster: list[ModelConfig], owner: str) -> None:
    """Reject a roster that names the same ``provider:model`` twice.

    A guard, not a transform: it returns nothing and never rewrites the roster. Two
    entries with one key make a switch request ambiguous, and the ambiguity would only
    surface at switch time, on whichever entry happened to be found first.

    MUST be called from a ``mode="after"`` validator. Before field validation, entries
    may still be raw dicts whose ``provider`` is absent, so ``{"model": "m"}`` and
    ``{"provider": "openai", "model": "m"}`` look distinct although ModelConfig's
    ``"openai"`` default is about to make them identical.

    Args:
        roster: Validated roster entries, in declaration order.
        owner: The model reporting the fault, named in the message.

    Raises:
        ValueError: on the first repeated key.
    """
    seen: set[str] = set()
    for entry in roster:
        key = model_roster_key(entry)
        if key in seen:
            raise ValueError(
                f"{owner} model_roster has a duplicate entry for '{key}'; "
                "every roster entry must name a distinct provider:model"
            )
        seen.add(key)


def normalize_model_roster(data: Any, owner: str) -> Any:
    """Fold a list of model configs into one active model plus a declared roster.

    The single place in the package where a model config may be tested against ``list``.
    A list is an input convenience at the boundary: element 0 becomes ``model_cfg`` and
    the whole list — the active entry included, in declaration order — becomes
    ``model_roster``. Every storage and read path downstream sees one ModelConfig.

    Shared with the packages that build a ReactAgentConfig-shaped model of their own, so
    the roster grammar has one implementation rather than a second copy; mirrors
    ``_fold_pre_split_request_limit``'s owner-carries-the-message convention.

    TRAP: anything other than a list returns ``data`` **unchanged**. It must never fall
    through to an ``else`` that clears ``model_roster``: on the
    ``model_validate(cfg.model_dump())`` round trip, ``model_cfg`` arrives as a single
    dict while ``model_roster`` is already populated, and clearing it there destroys the
    roster on a path no construction test exercises. ``default_factory=list`` supplies
    the empty default when the key is genuinely absent.

    A pure dict transform that copies rather than mutates and removes no key it does not
    own, so it composes with the usage-limits shim in either evaluation order.

    Args:
        data: Raw input handed to a ``mode="before"`` model validator.
        owner: The model reporting the fault, named in the messages.

    Returns:
        The input untouched, or a shallow copy carrying the folded roster.

    Raises:
        ValueError: if the list is empty, or if a list arrives with an explicit
            ``model_roster``.
    """
    if not isinstance(data, dict):
        return data
    if "model_cfg" not in data:
        return data
    value = data["model_cfg"]
    if not isinstance(value, list):
        return data
    if not value:
        raise ValueError(
            f"{owner} received an empty model_cfg list; an agent needs at least one model"
        )
    if "model_roster" in data:
        raise ValueError(
            f"{owner} received both a model_cfg list and an explicit model_roster; "
            "the roster is derived from the list — pass only one"
        )
    mapped = dict(data)
    mapped["model_roster"] = list(value)
    mapped["model_cfg"] = value[0]
    return mapped


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
    ReactAgent surfaces that as RunUsageLimitError. Token counts reset every run.

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

    Both this tier's limits are enforced pre-flight by LifetimeBudgetCapability, which
    owns the counters, accumulates them across every run the agent performs and is
    reseeded from replayed usage events on restore. Breaching either raises
    AgentUsageLimitError, a subclass of UsageLimitError, with pydantic-ai's own message
    text. The run tier raises a different subclass, so the two are told apart by class.

    agent_request_limit is consumed BEFORE the call executes, so a run that fails
    partway still counts.

    TRAP: the inherited token limits bound where a run may START, not where it may
    end — a run's cost is unknown until it completes, so the run that crosses the
    line finishes and only the next one is refused.

    Attributes:
        agent_request_limit: Maximum ReactAgent.run() calls over the agent's lifetime

    Example:
        >>> limits = AgentUsageLimits(agent_request_limit=100, total_tokens_limit=1_000_000)
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
    Still shipped, with no removal release scheduled — migrate to
    ``RunUsageLimits(run_request_limit=...)`` anyway; every use warns.

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
            f"UsageLimits is deprecated ({_SHIM_REMOVAL_NOTICE}); "
            "use RunUsageLimits(run_request_limit=...) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return folded

    @property
    def request_limit(self) -> int | None:
        """DEPRECATED read accessor for ``run_request_limit``.

        Still shipped, with no removal release scheduled. Reflects the underlying
        field regardless of which spelling set it.
        """
        warnings.warn(
            f"UsageLimits.request_limit is deprecated ({_SHIM_REMOVAL_NOTICE}); "
            f"read run_request_limit instead",
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
        parallel_tool_calls: Accepted and validated, but read by nothing in this package.
            ReactAgent reads only retries, end_strategy and http_client_config off
            runtime_cfg, and never derives a parallel_tool_calls model setting from it.
            create_model_settings() is the only function that emits that setting; it takes
            a ModelConfig, never a RuntimeConfig, and has no call site here.
        http_client_config: HTTP client configuration for API communication
            (timeout and retry settings)

    Tool Execution Strategies:
        The three values are exactly pydantic-ai's ``EndStrategy`` vocabulary. akgentic's
        default is 'exhaustive' and is deliberately not pydantic-ai's own default
        ('graceful'): ReactAgent passes end_strategy explicitly on every construction, so
        the upstream default never applies here.

        - 'early': Stops after first successful result (fast path). Output tools run in
          emission order and the run ends at the first one that succeeds; function tools
          are not executed. If every output tool fails, function tools run so the model can
          correct on the next round.
        - 'graceful': Tools run in the order the model emitted them -- function tools that
          precede an output tool complete before it. Output tools run in order and the first
          success wins; subsequent output tools are skipped (their side effects don't run).
          The same "retry-wins" rule described below for 'exhaustive' applies: a function
          tool raising ModelRetry suppresses the output result and surfaces the retry to the
          model instead.
        - 'exhaustive': Executes all tool calls even when result available (complete data
          gathering). Under pydantic-ai v2, this strategy also applies a "retry-wins" rule: if a
          function tool call in the same round as an already-successful output call
          raises ModelRetry (or fails argument validation), the output is suppressed
          and the run continues for another model turn instead of ending immediately.
          v1.107 had no such rule -- an already-successful output always won. The forced extra
          turn counts against run_usage_limits (run_request_limit, total_tokens_limit): a run
          that would have finished successfully on v1 (output wins, retry ignored) can instead
          raise RunUsageLimitError on v2 if that turn pushes it past a run-tier ceiling.

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

    end_strategy: Literal["early", "graceful", "exhaustive"] = Field(
        default="exhaustive",
        description="Tool execution strategy: 'early' stops at the first successful output and skips function tools, 'graceful' runs tools in emission order and lets the first successful output win, 'exhaustive' runs every tool",  # noqa: E501
    )

    parallel_tool_calls: bool = Field(
        default=True,
        description=(
            "Accepted and validated, but read by nothing in this package: ReactAgent "
            "never derives a parallel_tool_calls model setting from it"
        ),
    )

    http_client_config: HttpClientConfig = Field(
        default_factory=HttpClientConfig,
        description="HTTP client configuration for API communication",
    )


def validate_compaction_bounds(
    model_cfg: ModelConfig,
    compaction_cfg: CompactionConfig,
    run_usage_limits: RunUsageLimits,
    owner: str,
) -> None:
    """Keep the auto-compaction trigger reachable before the run tier's token limits bite.

    When auto-compaction is live, the effective threshold must sit strictly below every set
    token limit; otherwise pydantic-ai raises UsageLimitExceeded first and the auto-trigger
    is dead code. Reads the RUN tier only, by choice: the agent tier's token limits are
    enforced too, and one set below the threshold does leave the auto-trigger unreachable —
    the agent refuses the run before compaction can fire. Rejecting that here would change
    which configs are constructible, so it stays a documented consequence rather than a
    validation error.

    The single implementation of the rule, called from two places: ReactAgentConfig's
    after-validator at construction, and ``ReactAgent.switch_model`` against the CANDIDATE
    roster entry before anything is committed. A switch changes ``model_cfg.context_length``,
    which moves the threshold — so a second spelling here would let a switch install a model
    whose configuration the constructor would have refused.

    Args:
        model_cfg: The model whose ``context_length`` sets the threshold.
        compaction_cfg: The compaction settings supplying the trigger and its ratio.
        run_usage_limits: The run tier the threshold must stay strictly below.
        owner: The caller reporting the fault, named in the message.

    Raises:
        ValueError: When the threshold reaches or exceeds a set run-tier token limit.
    """
    context_length = model_cfg.context_length
    if not (compaction_cfg.auto_trigger and context_length is not None):
        return
    threshold = int(context_length * compaction_cfg.trigger_ratio)
    for name in ("input_tokens_limit", "total_tokens_limit"):
        limit = getattr(run_usage_limits, name)
        if limit is not None and threshold >= limit:
            raise ValueError(
                f"{owner} compaction threshold {threshold} must be strictly below "
                f"run_usage_limits.{name} ({limit})"
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
        model_cfg: LLM provider and model settings — always a single ModelConfig once
            stored. At the input boundary it also accepts a **list** of ModelConfig
            (or of dicts, on the catalog path), of which element 0 becomes the active
            model and the whole list becomes ``model_roster``. The list is a
            convenience for declaring a roster, never a stored shape.
        model_roster: The full declared roster, in declaration order, including the
            active entry. Empty means a single-model agent, for which switching is
            unavailable. Entry keys (``provider:model``) must be unique, and the active
            model must be one of them.
        runtime_cfg: Execution behavior and HTTP retry strategy.
        run_usage_limits: Per-run resource limits; the tier pydantic-ai enforces.
        agent_usage_limits: Agent-lifetime resource limits — runs and tokens, both
            enforced pre-flight in ReactAgent.run().
        compaction_cfg: Context-compaction configuration.
        max_messages: Sliding-window size handed to ContextManager; None = unlimited.

    Deprecated:
        ``usage_limits`` survives as a constructor keyword and a read accessor for
        ``run_usage_limits``. Both warn; neither has a scheduled removal release.
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
        ...         http_client_config=HttpClientConfig(timeout=180.0)
        ...     )
        ... )
        >>>
        >>> # A roster: gpt-4o is active, all three are declared and switchable.
        >>> config = ReactAgentConfig(
        ...     model_cfg=[
        ...         ModelConfig(provider="openai", model="gpt-4o"),
        ...         ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ...         ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
        ...     ]
        ... )
        >>> config.model_cfg.model
        'gpt-4o'
        >>> len(config.model_roster)
        3
    """

    model_cfg: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")

    model_roster: list[ModelConfig] = Field(
        default_factory=list,
        description=(
            "Full declared roster in declaration order, including the active entry. "
            "Empty = single-model agent, switching unavailable."
        ),
    )

    runtime_cfg: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Runtime behavior configuration"
    )

    run_usage_limits: RunUsageLimits = Field(
        default_factory=RunUsageLimits, description="Per-run usage limits for cost control"
    )

    agent_usage_limits: AgentUsageLimits = Field(
        default_factory=AgentUsageLimits,
        description="Agent-lifetime usage limits, enforced pre-flight on every run",
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
    def _normalize_model_roster(cls, data: Any) -> Any:
        """Fold a ``model_cfg`` list into the active model plus ``model_roster``.

        The whole body lives in the module-level ``normalize_model_roster`` so other
        packages can call it from their own before-validator instead of re-spelling the
        roster grammar. Deliberately separate from ``_map_pre_split_usage_limits``:
        each before-validator owns one concern and returns its input unchanged when its
        own key is absent, so their evaluation order cannot matter.
        """
        return normalize_model_roster(data, "ReactAgentConfig")

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
            f"ReactAgentConfig(usage_limits=...) is deprecated ({_SHIM_REMOVAL_NOTICE}); "
            f"use run_usage_limits=... instead",
            DeprecationWarning,
            stacklevel=3,
        )
        mapped["run_usage_limits"] = value
        return mapped

    @property
    def usage_limits(self) -> RunUsageLimits:
        """DEPRECATED read accessor for ``run_usage_limits``.

        Still shipped, with no removal release scheduled. Returns the run tier
        itself, not a copy.
        """
        warnings.warn(
            f"ReactAgentConfig.usage_limits is deprecated ({_SHIM_REMOVAL_NOTICE}); "
            f"read run_usage_limits instead",
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

        The whole body lives in the module-level ``validate_compaction_bounds`` because
        ``ReactAgent.switch_model`` must apply the same rule to a candidate roster entry
        before committing it — a switch moves ``context_length``, and therefore the
        threshold. Two spellings would let a switch install what construction refuses.
        """
        validate_compaction_bounds(
            self.model_cfg, self.compaction_cfg, self.run_usage_limits, "ReactAgentConfig"
        )
        return self

    @model_validator(mode="after")
    def _reject_duplicate_roster_keys(self) -> "ReactAgentConfig":
        """Unique-keys: no two roster entries may name the same ``provider:model``.

        Runs after field validation by necessity — before it, an entry's ``provider``
        may still be absent from a raw dict and ModelConfig's ``"openai"`` default has
        not applied, so two spellings of one model look distinct.
        """
        if not self.model_roster:
            return self
        validate_unique_roster_keys(self.model_roster, "ReactAgentConfig")
        return self

    @model_validator(mode="after")
    def _require_active_model_in_roster(self) -> "ReactAgentConfig":
        """Membership: a non-empty roster must contain the active model.

        Normalization satisfies this by construction; it is reachable only when a caller
        hand-sets ``model_roster``. Defined AFTER the uniqueness rule on purpose, so a
        roster that is both duplicated and non-covering reports the duplicate — the
        fault that explains the other.
        """
        if not self.model_roster:
            return self
        active = model_roster_key(self.model_cfg)
        keys = [model_roster_key(entry) for entry in self.model_roster]
        if active not in keys:
            raise ValueError(
                f"model_cfg '{active}' is not in model_roster ({', '.join(keys)}); "
                "the active model must be one of the declared roster entries"
            )
        return self
