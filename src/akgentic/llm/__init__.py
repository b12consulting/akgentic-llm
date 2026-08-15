"""akgentic-llm: LLM integration layer for agent systems.

Clean abstraction for LLM providers with REACT pattern support,
context management, and comprehensive configuration.

Quick Start:
    >>> from akgentic.llm import ModelConfig, ReactAgentConfig, ReactAgent
    >>> config = ReactAgentConfig(
    ...     model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
    ... )
    >>> agent = ReactAgent(config=config)
    >>> result = await agent.run("Hello!")

Key Concepts:
    - REACT pattern: Iterative agent execution with tool calls
    - RunUsageLimits: Per-run token/request budget, enforced by pydantic-ai
    - AgentUsageLimits: Agent-lifetime budget, enforced pre-flight on every run
    - ContextManager: Message history tracking
    - PromptTemplate: Template-based prompts with parameter substitution
"""

from importlib import metadata as _metadata

from .agent import ReactAgent, UsageLimitError, UserPrompt
from .compaction import (
    COMPACTION_STRATEGIES,
    SUMMARY_INSTRUCTIONS,
    CompactionResult,
    CompactionStrategy,
    create_compaction,
)
from .config import (
    AgentUsageLimits,
    CompactionConfig,
    HttpClientConfig,
    ModelConfig,
    ReactAgentConfig,
    RuntimeConfig,
    RunUsageLimits,
    UsageLimits,
)
from .context import ContextManager
from .event import (
    ContextObserver,
    LlmContextClearedEvent,
    LlmContextCompactedEvent,
    LlmMessageEvent,
    LlmSystemPromptEvent,
    LlmUsageEvent,
    SystemPromptPartSnapshot,
    ToolCallEvent,
    ToolReturnEvent,
)
from .pricing import (
    AgentUsageSummary,
    ModelUsage,
    RunUsageSummary,
    aggregate_usage,
)
from .prompts import (
    PromptTemplate,
    current_datetime_prompt,
    json_output_reminder_prompt,
)
from .providers import create_http_client, create_model, create_model_settings, get_output_type

__all__ = [
    # Configuration
    "ModelConfig",
    "RunUsageLimits",
    "AgentUsageLimits",
    "UsageLimits",  # DEPRECATED alias of RunUsageLimits — removal not yet scheduled
    "RuntimeConfig",
    "HttpClientConfig",
    "ReactAgentConfig",
    "CompactionConfig",
    # Agent
    "ReactAgent",
    "UsageLimitError",
    "UserPrompt",
    # Context
    "ContextManager",
    "ContextObserver",
    "LlmMessageEvent",
    "LlmUsageEvent",
    "LlmSystemPromptEvent",
    "SystemPromptPartSnapshot",
    "ToolCallEvent",
    "ToolReturnEvent",
    # Compaction
    "LlmContextCompactedEvent",
    "LlmContextClearedEvent",
    "CompactionStrategy",
    "CompactionResult",
    "COMPACTION_STRATEGIES",
    "SUMMARY_INSTRUCTIONS",
    "create_compaction",
    # Prompts
    "PromptTemplate",
    "current_datetime_prompt",
    "json_output_reminder_prompt",
    # Pricing & Aggregation
    "AgentUsageSummary",
    "ModelUsage",
    "RunUsageSummary",
    "aggregate_usage",
    # Providers
    "create_model",
    "create_http_client",
    "create_model_settings",
    "get_output_type",
]

try:
    __version__ = _metadata.version("akgentic-llm")
except _metadata.PackageNotFoundError:  # pragma: no cover - source tree, never installed
    # Importing from a source tree that was never installed must not fail over a
    # version string. A hardcoded literal here is what drifted from pyproject.toml
    # for eight minor releases; the sentinel is unmistakably "not a real version".
    __version__ = "0.0.0+unknown"
