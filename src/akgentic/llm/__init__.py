"""akgentic-llm: LLM integration layer for agent systems.

Clean abstraction for LLM providers with REACT pattern support,
context management, and comprehensive configuration.
"""

from .agent import ReactAgent
from .config import AgentRuntimeConfig, ModelConfig, ReactAgentConfig, UsageLimits
from .context import ContextManager, ContextObserver, ContextSnapshot
from .prompts import (
    PromptProvider,
    SystemPromptRegistry,
    current_datetime_prompt,
    json_output_reminder_prompt,
)
from .providers import create_http_client, create_model

__all__ = [
    # Configuration
    "ModelConfig",
    "UsageLimits",
    "AgentRuntimeConfig",
    "ReactAgentConfig",
    # Agent
    "ReactAgent",
    # Context
    "ContextManager",
    "ContextObserver",
    "ContextSnapshot",
    # Prompts
    "SystemPromptRegistry",
    "PromptProvider",
    "current_datetime_prompt",
    "json_output_reminder_prompt",
    # Providers
    "create_model",
    "create_http_client",
]

__version__ = "0.1.0"
