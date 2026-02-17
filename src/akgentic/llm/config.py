"""Configuration models for LLM agent behavior."""

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model selection and parameters.

    Renamed from LLMConfig for clarity - configures MODEL selection,
    not the entire LLM layer.

    Example:
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="gpt-4.1",
        ...     temperature=0.7,
        ...     seed=42
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
    """Token and request limits for cost control.

    Limits are checked by pydantic-ai during agent runs.
    Set to None to disable specific limits.

    Example:
        >>> limits = UsageLimits(
        ...     request_limit=50,
        ...     total_tokens_limit=10000
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
    """Complete configuration for REACT-based LLM agents.

    Example:
        >>> config = ReactAgentConfig(
        ...     model=ModelConfig(provider="openai", model="gpt-4.1"),
        ...     usage_limits=UsageLimits(request_limit=30),
        ...     runtime=AgentRuntimeConfig(retries=5)
        ... )
    """

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")

    usage_limits: UsageLimits = Field(default_factory=UsageLimits, description="Usage limits for cost control")

    runtime: AgentRuntimeConfig = Field(
        default_factory=AgentRuntimeConfig, description="Runtime behavior configuration"
    )
