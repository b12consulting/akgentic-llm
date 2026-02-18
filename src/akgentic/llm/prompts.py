"""Prompt templates, rendering, and built-in prompt providers."""

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class PromptProvider(Protocol):
    """Protocol for dynamic system prompt providers.

    Example:
        >>> def my_prompt(ctx):
        ...     return f"User: {ctx.deps.user_id}"
    """

    def __call__(self, ctx: Any) -> str:
        """Generate prompt content from context."""
        ...


class PromptTemplate(BaseModel):
    """Structured prompt with template interpolation.

    Supports {placeholder} syntax for parameter substitution.
    Parameters are resolved at config time, not at runtime.

    Example:
        >>> tpl = PromptTemplate(
        ...     template="You are {role}.\\n{instructions}",
        ...     params={"role": "Architect", "instructions": "Design systems."}
        ... )
        >>> render_prompt(tpl)
        'You are Architect.\\nDesign systems.'
    """

    template: str = Field(..., description="Template string with {placeholder} syntax")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value replacements for template placeholders",
    )


def render_prompt(prompt: str | PromptTemplate) -> str:
    """Render a prompt to its final string form.

    - str: returned as-is (passthrough)
    - PromptTemplate: interpolates params into template

    Args:
        prompt: String prompt or PromptTemplate with params.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If template references a param not in params dict.

    Example:
        >>> render_prompt("Simple string")
        'Simple string'
        >>> tpl = PromptTemplate(
        ...     template="You are {role}.",
        ...     params={"role": "Architect"}
        ... )
        >>> render_prompt(tpl)
        'You are Architect.'
    """
    if isinstance(prompt, str):
        return prompt
    return prompt.template.format(**prompt.params)


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
