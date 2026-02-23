from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from akgentic.llm.context import ContextSnapshot


@dataclass(frozen=True)
class LlmMessageEvent:
    """Event emitted when a new model message is added to context."""

    message: ModelMessage


@dataclass(frozen=True)
class LlmCheckpointCreatedEvent:
    """Event emitted when a context checkpoint is created."""

    snapshot: ContextSnapshot


@dataclass(frozen=True)
class LlmCheckpointRestoredEvent:
    """Event emitted when context is restored from a checkpoint."""

    snapshot: ContextSnapshot


@runtime_checkable
class ContextObserver(Protocol):
    """Observer protocol for LLM context changes."""

    def notify_event(self, event: object) -> None:
        """Called when an LLM domain event is emitted.

        Args:
            event: Domain event object
        """
        ...
