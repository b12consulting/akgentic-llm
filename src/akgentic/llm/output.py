"""An output type that merges a split reply and says when a reply could not be read.

A provider may return one reply as several text parts — OpenAI's Responses API emits a
``commentary`` part before the ``final_answer`` — and pydantic-ai joins them, so two valid
objects become ``{...}{...}``, which does not parse. Its stock retry then names only the
parse error, and a model reads that as a format fix rather than as "none of it was used".

- :class:`MergeableOutput` — the base an output model subclasses to opt in. It says how
  several instances fold into one (``merge``), and what the model is told when its reply
  could not be read (``unreadable_output_error``).
- :class:`MergeableOutputCapability` — mounted by ``ReactAgent`` on every run, a no-op for
  any other output type.
"""

import json
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ValidationError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.output import OutputContext


class MergeableOutput(BaseModel):
    """An output model that knows how to merge several of its instances."""

    @classmethod
    @abstractmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        """Fold several outputs, in the order written, into one."""

    @classmethod
    def unreadable_output_error(cls, error: ValidationError) -> str:
        """The retry prompt when the reply could not be read.

        Args:
            error: The validation failure.

        Returns:
            Why the reply was rejected, one line per error, then how to resend.
        """
        lines = ["Your reply could not be read, so none of it was used."]
        lines += error_lines(error)
        lines.append("Resend your whole reply as exactly ONE JSON object.")
        return "\n".join(lines)


@dataclass
class MergeableOutputCapability(AbstractCapability[Any]):
    """Merge a split :class:`MergeableOutput`; word its retry when it cannot be read."""

    # The Any below is pydantic-ai's own hook signature.
    async def before_output_validate(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Replace several JSON objects written back to back with their merge.

        Returns:
            The merged object as JSON; ``output`` unchanged when the output type is not a
            :class:`MergeableOutput`, the output is partial or not text, or it is not two
            or more objects that each validate.
        """
        output_type = _mergeable(output_context)
        if output_type is None or ctx.partial_output or not isinstance(output, str):
            return output
        merged = merge_outputs(output, output_type)
        return output if merged is None else merged

    # Same Any as above: pydantic-ai's hook signature.
    async def on_output_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
        error: ValidationError | ModelRetry,
    ) -> Any:
        """Retry an unreadable :class:`MergeableOutput` with its own wording.

        Raises:
            ModelRetry: ``output_type.unreadable_output_error(error)``, for a
                ``ValidationError`` on a :class:`MergeableOutput`.
            ValidationError | ModelRetry: ``error`` unchanged otherwise.
        """
        output_type = _mergeable(output_context)
        if output_type is None or not isinstance(error, ValidationError):
            raise error
        raise ModelRetry(output_type.unreadable_output_error(error))


def _mergeable(output_context: OutputContext) -> type[MergeableOutput] | None:
    output_type = output_context.output_type
    if isinstance(output_type, type) and issubclass(output_type, MergeableOutput):
        return output_type
    return None


_JSON_WHITESPACE = " \t\n\r"
"""The only whitespace JSON allows between values; bare ``str.strip()`` takes more."""


def merge_outputs(raw: str, output_type: type[MergeableOutput]) -> str | None:
    """One object from several written back to back, through ``output_type.merge``.

    Args:
        raw: The model's text output.
        output_type: The run's output type.

    Returns:
        The merged output as JSON; ``None`` when ``raw`` is a single object, is not
        entirely JSON objects, or holds one that does not validate — validation then
        reports it as usual.
    """
    decoder = json.JSONDecoder()
    outputs: list[MergeableOutput] = []
    index, end = 0, len(raw.rstrip(_JSON_WHITESPACE))
    while index < end:
        # raw_decode rejects leading space
        index = len(raw) - len(raw[index:].lstrip(_JSON_WHITESPACE))
        try:
            value, index = decoder.raw_decode(raw, index)
            outputs.append(output_type.model_validate(value))
        except (ValueError, RecursionError):  # ValidationError is a ValueError
            return None
    if len(outputs) < 2:
        return None
    return output_type.merge(outputs).model_dump_json(by_alias=True)


def error_lines(error: ValidationError) -> list[str]:
    """One ``- loc: msg`` line per validation error, without the rejected input."""
    lines = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        where = ".".join(str(part) for part in detail["loc"])
        lines.append(f"- {where}: {detail['msg']}" if where else f"- {detail['msg']}")
    return lines
