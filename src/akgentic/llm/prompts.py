"""System prompt registry for dynamic prompt management."""

# TODO: Implementation to be completed by Dev agent
# This is a skeleton file created by Architect

from datetime import UTC, datetime
from typing import Any, Protocol


class PromptProvider(Protocol):
    """Protocol for dynamic system prompt providers.

    Example:
        >>> def my_prompt(ctx):
        ...     return f"User: {ctx.deps.user_id}"
    """

    def __call__(self, ctx: Any) -> str:
        """Generate prompt content from context."""
        ...


class SystemPromptRegistry:
    """Registry for dynamic system prompts with priority ordering.

    Example:
        >>> registry = SystemPromptRegistry()
        >>> registry.register("datetime", current_datetime_prompt, priority=10)

    TODO: Complete implementation following architecture.md specifications
    """

    def __init__(self) -> None:
        """Initialize prompt registry."""
        # TODO: Implement initialization
        raise NotImplementedError("SystemPromptRegistry implementation pending")


# Built-in prompt providers


def current_datetime_prompt(ctx: Any) -> str:
    """Standard datetime system prompt.

    Returns current date and time in user's timezone.
    """
    now = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    return f"The current date and time is {now}."


def json_output_reminder_prompt(ctx: Any) -> str:
    """Reminder for JSON-only structured output.

    Useful for models that sometimes include explanatory text.
    """
    return "\n\nCRITICAL: Return ONLY a valid JSON object. Do not include explanations or markdown code blocks."
