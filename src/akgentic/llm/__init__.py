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
    - RunUsageLimitError: A run-tier breach — recovered by default rather than
      raised. The mounted LimitRecoveryCapability decides the turn concludes, and
      run() returns that conclusion's answer; this class surfaces only when the
      seam declines (returns None) or the conclusion itself produces nothing usable
    - AgentUsageLimitError: An agent-tier breach — terminal, the agent is finished
    - UsageLimitError: Base of both; catch it to handle either tier, catch a
      subclass to react to one. The tiers are told apart by class, never by
      message text.
    - ContextManager: Message history tracking
    - LifetimeBudgetCapability / CompactionCapability / EventSourcingCapability /
      LimitRecoveryCapability / HealingCapability / DiscardedOutputCapability: the
      run loop's agent-lifetime budget, its auto-compaction, its persistence, its
      run-tier recovery policy, its dangling-tool-call repair and the removal of
      structured output the tool-execution strategy is about to discard, each
      mountable a la carte on any bare pydantic-ai Agent. That is also ReactAgent's
      mount order — though the last one's position in it carries nothing, by design.
    - ConclusionDecision: what LimitRecoveryCapability's handle_limit_exceeded seam
      returns to ask for a tool-free conclusion; None asks for none
    - PromptTemplate: Template-based prompts with parameter substitution
"""

from importlib import metadata as _metadata

from .agent import (
    AgentUsageLimitError,
    ModelSwitchError,
    ReactAgent,
    RunUsageLimitError,
    UsageLimitError,
    UserPrompt,
)
from .capabilities import (
    CompactionCapability,
    ConclusionDecision,
    DiscardedOutputCapability,
    EventSourcingCapability,
    HealingCapability,
    LifetimeBudgetCapability,
    LimitRecoveryCapability,
)
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
    model_roster_key,
    normalize_model_roster,
    validate_unique_roster_keys,
)
from .context import ContextManager
from .event import (
    ContextObserver,
    LlmContextClearedEvent,
    LlmContextCompactedEvent,
    LlmMessageEvent,
    LlmOutputDiscardedEvent,
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
    # The model-roster grammar and its guards — imported by sibling packages that
    # project a roster onto their own row types, so the key is spelled exactly once.
    "model_roster_key",
    "normalize_model_roster",
    "validate_unique_roster_keys",
    # Agent
    "ReactAgent",
    "UsageLimitError",  # base of both tiers — catch it to handle either
    "RunUsageLimitError",
    "AgentUsageLimitError",
    # The one refusal class of ReactAgent.switch_model — a ValueError subclass, so an
    # existing `except ValueError` still catches it.
    "ModelSwitchError",
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
    # Run-loop capabilities
    "LifetimeBudgetCapability",
    "CompactionCapability",
    "EventSourcingCapability",
    "HealingCapability",
    "LimitRecoveryCapability",
    "DiscardedOutputCapability",
    "ConclusionDecision",
    # Compaction
    "LlmContextCompactedEvent",
    "LlmContextClearedEvent",
    "LlmOutputDiscardedEvent",
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
