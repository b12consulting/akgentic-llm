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

    >>> from akgentic.llm import ModelConfig, UsageLimits, ReactAgentConfig
    >>> config = ReactAgentConfig(
    ...     model=ModelConfig(provider="openai", model="gpt-4o"),
    ...     usage_limits=UsageLimits(request_limit=10, total_tokens_limit=5000)
    ... )
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for LLM model settings.

    Supports multiple providers with provider-agnostic configuration.

    Attributes:
        provider: LLM provider name (openai, azure, anthropic, google-gla, mistral, nvidia)
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
        temperature: Sampling temperature between 0.0-2.0 (default: None = provider default)
        seed: Random seed for reproducibility (default: None = non-deterministic)
        max_tokens: Maximum tokens to generate (default: None = provider default)
        reasoning_effort: Reasoning effort level for supported models (e.g., 'low', 'medium', 'high')

    Example:
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
    """

    provider: Literal["openai", "azure", "nvidia", "google-gla", "mistral", "anthropic"] = Field(
        default="openai", description="Model provider"
    )

    model: str = Field(
        default="gpt-4.1",
        description="Model identifier (e.g., gpt-4.1, claude-3-5-sonnet-20241022)",
    )

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")

    seed: int | None = Field(default=None, description="Random seed for reproducible outputs")

    max_tokens: int | None = Field(default=None, gt=0, description="Maximum tokens in model response")

    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort for o1/o3 models"
    )


class UsageLimits(BaseModel):
    """Usage limits for LLM requests.

    All limits are optional (None = unlimited). Limits are enforced
    during agent execution.

    Attributes:
        request_limit: Maximum number of LLM requests
        tool_calls_limit: Maximum tool calls per request
        input_tokens_limit: Maximum input tokens (cumulative)
        output_tokens_limit: Maximum output tokens (cumulative)
        total_tokens_limit: Maximum total tokens (input + output, cumulative)

    Example:
        >>> limits = UsageLimits(
        ...     request_limit=10,
        ...     total_tokens_limit=5000
        ... )
    """

    request_limit: int | None = Field(default=50, gt=0, description="Maximum number of LLM requests per run")

    tool_calls_limit: int | None = Field(default=None, gt=0, description="Maximum number of tool calls per run")

    input_tokens_limit: int | None = Field(default=None, gt=0, description="Maximum input/prompt tokens")

    output_tokens_limit: int | None = Field(default=None, gt=0, description="Maximum output/completion tokens")

    total_tokens_limit: int | None = Field(default=None, gt=0, description="Maximum total tokens (input + output)")


class AgentRuntimeConfig(BaseModel):
    """Runtime configuration for agent behavior.

    Example:
        >>> runtime = AgentRuntimeConfig(
        ...     retries=3,
        ...     end_strategy="exhaustive",
        ...     timeout_seconds=120.0
        ... )
    """

    retries: int = Field(default=3, ge=0, description="Number of retries for tool calls and validation")

    end_strategy: Literal["early", "exhaustive"] = Field(
        default="exhaustive",
        description="'early' stops after first result, 'exhaustive' runs all tools",
    )

    parallel_tool_calls: bool = Field(default=True, description="Allow parallel tool execution (if model supports)")

    timeout_seconds: float = Field(default=120.0, gt=0, description="Request timeout in seconds")

    max_http_retries: int = Field(default=5, ge=1, description="Maximum HTTP retry attempts for provider API calls")

    retry_backoff_multiplier: float = Field(default=0.5, gt=0, description="Exponential backoff multiplier for retries")

    retry_backoff_max: float = Field(default=60.0, gt=0, description="Maximum backoff time in seconds")


class ReactAgentConfig(BaseModel):
    """Configuration for REACT pattern agent.

    Combines model configuration, usage limits, and execution settings.

    Attributes:
        model: LLM model configuration
        usage_limits: Optional resource limits
        runtime: Runtime behavior configuration
        system_prompts: List of system prompt components

    Example:
        >>> config = ReactAgentConfig(
        ...     model=ModelConfig(provider="openai", model="gpt-4o"),
        ...     usage_limits=UsageLimits(request_limit=10),
        ...     system_prompts=["You are a helpful assistant."]
        ... )
    """

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")

    usage_limits: UsageLimits | None = Field(default=None, description="Usage limits for cost control")

    runtime: AgentRuntimeConfig = Field(
        default_factory=AgentRuntimeConfig, description="Runtime behavior configuration"
    )

    system_prompts: list[str] = Field(default_factory=list, description="System prompt components")
