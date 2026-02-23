"""Context management with checkpointing and compactification."""

import copy
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from akgentic.llm.event import (
    ContextObserver,
    LlmCheckpointCreatedEvent,
    LlmCheckpointRestoredEvent,
    LlmMessageEvent,
)


def _is_system_message(msg: ModelMessage) -> bool:
    """Check if a message is a system message.

    System messages contain SystemPromptPart and should be preserved
    during sliding window operations.

    Args:
        msg: Message to check

    Returns:
        True if message is a system message
    """
    return isinstance(msg, ModelRequest) and any(
        isinstance(part, SystemPromptPart) for part in msg.parts
    )


class ContextSnapshot(BaseModel):
    """Immutable snapshot of conversation context.

    Used for checkpoint/rewind functionality. Messages are deep-copied
    to ensure immutability.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        timestamp: When the checkpoint was created
        messages: Deep copy of messages at checkpoint
        metadata: Optional custom metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    timestamp: datetime = Field(..., description="When checkpoint was created")
    # FIXME: Using Any instead of list[ModelMessage] due to pydantic-ai 1.60.0 bug
    # pydantic-ai's ModelMessage dataclasses contain forward refs with AliasChoices
    # that cause Pydantic schema generation to fail. Should either:
    # 1. Convert ContextSnapshot to @dataclass(frozen=True) to avoid Pydantic validation
    # 2. Wait for pydantic-ai fix and restore proper type: list[ModelMessage]
    messages: Any = Field(..., description="Deep copy of messages (list[ModelMessage])")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class ContextManager:
    """Manages LLM conversation context with checkpointing.

    Features:
    - Message history tracking
    - Observer pattern for notifications
    - Checkpoint/rewind support
    - Sliding window with system message preservation

    This implementation replicates V1's base_agent.py context management
    with additional checkpoint functionality.

    Observer Behavior:
    - Observers are notified synchronously
    - Exceptions from observers propagate to caller
    - Use try/except in observer methods if exceptions should not interrupt operations

    Example:
        >>> from akgentic.llm import ContextManager
        >>> from pydantic_ai.messages import ModelRequest, UserPromptPart
        >>>
        >>> manager = ContextManager(max_messages=10)
        >>> manager.add_message(ModelRequest(parts=[UserPromptPart(content="Hello")]))
        >>> snapshot = manager.checkpoint("before-llm-call")
        >>> # ... LLM interaction ...
        >>> manager.rewind("before-llm-call")  # Restore if needed
    """

    def __init__(
        self,
        max_messages: int | None = None,
    ) -> None:
        """Initialize context manager.

        Args:
            max_messages: Maximum messages to keep (None = unlimited).
                System messages are always preserved.

        Raises:
            ValueError: If max_messages is negative.
        """
        if max_messages is not None and max_messages < 0:
            raise ValueError(f"max_messages must be non-negative, got {max_messages}")
        self._max_messages = max_messages
        self._messages: list[ModelMessage] = []
        self._checkpoints: dict[str, ContextSnapshot] = {}
        self._checkpoint_order: list[str] = []
        self._observers: list[ContextObserver] = []

    @property
    def messages(self) -> list[ModelMessage]:
        """Get current message history.

        Returns a shallow copy to prevent external mutation.

        Returns:
            Copy of current messages
        """
        return list(self._messages)

    def add_message(self, message: ModelMessage) -> None:
        """Add a message to context.

        Appends message, applies sliding window if configured,
        and notifies observers.

        Args:
            message: Message to add
        """
        self._messages.append(message)
        self._apply_window()

        # Notify observers
        for observer in self._observers:
            observer.notify_event(LlmMessageEvent(message=message))

    def _apply_window(self) -> None:
        """Apply sliding window to messages.

        Keeps most recent max_messages while preserving all system messages.
        System messages don't count toward the limit.
        """
        if self._max_messages is None or len(self._messages) <= self._max_messages:
            return

        # Separate system and non-system messages
        system_msgs = [m for m in self._messages if _is_system_message(m)]
        non_system = [m for m in self._messages if not _is_system_message(m)]

        # Keep as many non-system messages as possible after system messages
        keep_non_system = max(0, self._max_messages - len(system_msgs))
        self._messages = system_msgs + non_system[-keep_non_system:]

    def checkpoint(
        self,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSnapshot:
        """Create a checkpoint of current context.

        Creates a deep copy of messages for immutable snapshot.
        Auto-generates UUID if no id provided.

        Args:
            checkpoint_id: Optional checkpoint identifier (UUID generated if None)
            metadata: Optional metadata to store with checkpoint

        Returns:
            Created snapshot
        """
        if checkpoint_id is None:
            checkpoint_id = str(uuid.uuid4())

        snapshot = ContextSnapshot(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            messages=copy.deepcopy(self._messages),
            metadata=metadata or {},
        )

        self._checkpoints[checkpoint_id] = snapshot
        self._checkpoint_order.append(checkpoint_id)

        # Notify observers
        for observer in self._observers:
            observer.notify_event(LlmCheckpointCreatedEvent(snapshot=snapshot))

        return snapshot

    def rewind(self, checkpoint_id: str) -> None:
        """Restore context to a checkpoint.

        Replaces current messages with copy from checkpoint.
        Snapshot already contains deep copy, so no additional deepcopy needed.

        Args:
            checkpoint_id: Checkpoint to restore

        Raises:
            KeyError: If checkpoint_id not found
        """
        snapshot = self._checkpoints[checkpoint_id]  # Raises KeyError if not found
        self._messages = list(snapshot.messages)

        # Notify observers
        for observer in self._observers:
            observer.notify_event(LlmCheckpointRestoredEvent(snapshot=snapshot))

    def get_checkpoint(self, checkpoint_id: str) -> ContextSnapshot | None:
        """Get a checkpoint by id.

        Args:
            checkpoint_id: Checkpoint to retrieve

        Returns:
            Snapshot if found, None otherwise
        """
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint ids in creation order.

        Returns:
            List of checkpoint ids
        """
        return list(self._checkpoint_order)

    def subscribe(self, observer: ContextObserver) -> None:
        """Subscribe an observer to context events.

        Args:
            observer: Observer to add
        """
        self._observers.append(observer)

    def unsubscribe(self, observer: ContextObserver) -> None:
        """Unsubscribe an observer from context events.

        No-op if observer not present.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def clear(self) -> None:
        """Clear all messages and checkpoints.

        Resets context to empty state.
        """
        self._messages.clear()
        self._checkpoints.clear()
        self._checkpoint_order.clear()
