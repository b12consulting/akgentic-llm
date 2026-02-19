"""akgentic-llm: LLM integration layer for agent systems.

Clean abstraction for LLM providers with REACT pattern support,
context management, and comprehensive configuration.

Quick Start:
    >>> from akgentic.llm import ModelConfig, ReactAgentConfig, ReactAgent
    >>> config = ReactAgentConfig(
    ...     model=ModelConfig(provider="openai", model="gpt-4o"),
    ... )
    >>> agent = ReactAgent(config=config)
    >>> result = await agent.run("Hello!")

Key Concepts:
    - REACT pattern: Iterative agent execution with tool calls
    - UsageLimits: Token/request budgets to control costs
    - ContextManager: Message history with checkpoint/rewind
    - PromptTemplate: Template-based prompts with parameter substitution
"""

from .agent import ReactAgent, UsageLimitError
from .config import HttpClientConfig, ModelConfig, ReactAgentConfig, RuntimeConfig, UsageLimits
from .context import ContextManager, ContextObserver, ContextSnapshot
from .prompts import (
    PromptTemplate,
    current_datetime_prompt,
    json_output_reminder_prompt,
)
from .providers import create_http_client, create_model, create_model_settings, get_output_type

__all__ = [
    # Configuration
    "ModelConfig",
    "UsageLimits",
    "RuntimeConfig",
    "HttpClientConfig",
    "ReactAgentConfig",
    # Agent
    "ReactAgent",
    "UsageLimitError",
    # Context
    "ContextManager",
    "ContextObserver",
    "ContextSnapshot",
    # Prompts
    "PromptTemplate",
    "current_datetime_prompt",
    "json_output_reminder_prompt",
    # Providers
    "create_model",
    "create_http_client",
    "create_model_settings",
    "get_output_type",
]

__version__ = "0.1.0"
