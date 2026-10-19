"""A split ``MergeableOutput`` is merged; an unreadable one is retried with its own wording."""

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.llm import (
    MergeableOutput,
    MergeableOutputCapability,
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
    RuntimeConfig,
)
from akgentic.llm.output import merge_outputs


class _Notes(MergeableOutput):
    notes: list[str]

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        return cls(notes=[note for output in outputs for note in output.notes])


class _Told(_Notes):
    @classmethod
    def unreadable_output_error(cls, error: ValidationError) -> str:
        return "Nothing was kept."


def _parse_error(raw: str = '{"notes": []}Done.') -> ValidationError:
    with pytest.raises(ValidationError) as raised:
        _Notes.model_validate_json(raw)
    return raised.value


def _ctx(*, partial: bool = False) -> Any:
    return SimpleNamespace(partial_output=partial)


def _before(output_type: object, output: object, *, partial: bool = False) -> Any:
    return asyncio.run(
        MergeableOutputCapability().before_output_validate(
            _ctx(partial=partial), output_context=MagicMock(output_type=output_type), output=output
        )
    )


def _on_error(output_type: object, error: ValidationError | ModelRetry) -> None:
    asyncio.run(
        MergeableOutputCapability().on_output_validate_error(
            _ctx(), output_context=MagicMock(output_type=output_type), output="", error=error
        )
    )


class TestTheMerge:
    def test_objects_back_to_back_merge_in_order_through_the_schema(self) -> None:
        merged = merge_outputs('{"notes": ["a"]}\n {"notes": ["b", "c"]}', _Notes)

        assert merged == '{"notes":["a","b","c"]}'

    def test_the_text_is_kept_as_written(self) -> None:
        assert merge_outputs('{"notes": ["I’ve"]}{"notes": []}', _Notes) == '{"notes":["I’ve"]}'

    @pytest.mark.parametrize(
        "raw",
        [
            '{"notes": []}',
            '{"notes": []} ',
            '{"notes": []}Done.',
            '{"notes": []}{"notes": [',
            '{"notes": []}{"other": []}',
            '{"notes": []}[]',
            '{"notes": []} {"notes": []}',
            '{"notes": []}{"notes": [' + "9" * 5000 + "]}",
            '{"notes": []}' + "[" * 100_000,
            "",
        ],
        ids=[
            "one object",
            "one object, trailing space",
            "trailing prose",
            "truncated second object",
            "an object that does not validate",
            "not an object",
            "not JSON whitespace between",
            "an int past the digit limit",
            "nesting past the recursion limit",
            "empty",
        ],
    )
    def test_anything_but_several_valid_objects_is_left_to_validation(self, raw: str) -> None:
        assert merge_outputs(raw, _Notes) is None


class TestTheMergeHook:
    raw = '{"notes": ["a"]}{"notes": ["b"]}'

    def test_a_split_mergeable_output_is_merged(self) -> None:
        assert _before(_Notes, self.raw) == '{"notes":["a","b"]}'

    def test_another_output_type_is_left_alone(self) -> None:
        assert _before(str, self.raw) == self.raw

    def test_partial_output_is_left_alone(self) -> None:
        assert _before(_Notes, self.raw, partial=True) == self.raw

    def test_tool_arguments_are_left_alone(self) -> None:
        args = {"notes": []}

        assert _before(_Notes, args) is args


class TestAnUnreadableOutput:
    def test_the_default_retry_says_none_of_it_was_used_and_names_the_error(self) -> None:
        with pytest.raises(ModelRetry) as raised:
            _on_error(_Notes, _parse_error())

        lines = raised.value.message.splitlines()
        assert lines[0] == "Your reply could not be read, so none of it was used."
        assert lines[1].startswith("- Invalid JSON: trailing characters")
        assert lines[2] == "Resend your whole reply as exactly ONE JSON object."

    def test_a_field_error_is_located_without_the_input(self) -> None:
        text = _Notes.unreadable_output_error(_parse_error('{"notes": "x"}'))

        assert "- notes: Input should be a valid array" in text
        assert '"x"' not in text  # the raw output is already in context

    def test_the_schema_words_its_own_retry(self) -> None:
        with pytest.raises(ModelRetry) as raised:
            _on_error(_Told, _parse_error())

        assert raised.value.message == "Nothing was kept."

    def test_another_output_type_keeps_pydantic_ais_retry(self) -> None:
        error = _parse_error()

        with pytest.raises(ValidationError) as raised:
            _on_error(str, error)

        assert raised.value is error

    def test_a_model_retry_passes_through_unchanged(self) -> None:
        retry = ModelRetry("already rewritten")

        with pytest.raises(ModelRetry) as raised:
            _on_error(_Notes, retry)

        assert raised.value is retry


def _native_run(monkeypatch: pytest.MonkeyPatch, first: list[str]) -> tuple[_Notes, list[str]]:
    """One ``ReactAgent`` run on the native text path: ``first`` as text parts, then a fix."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        runtime_cfg=RuntimeConfig(retries=3),
    )
    react_agent = ReactAgent(config=config)
    retry_prompts: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        retry_prompts.extend(
            part.model_response()
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        texts = first if not retry_prompts else ['{"notes": ["fixed"]}']
        return ModelResponse(parts=[TextPart(content=text) for text in texts])

    try:
        with react_agent.pydantic_agent.override(model=FunctionModel(model)):
            output = react_agent.run_sync("go", output_type=_Notes)
    finally:
        react_agent.close()
    return output, retry_prompts


class TestInARealRun:
    def test_a_reply_split_in_two_text_parts_is_merged_without_a_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output, retry_prompts = _native_run(monkeypatch, ['{"notes": ["a"]}', '{"notes": ["b"]}'])

        assert retry_prompts == []
        assert output == _Notes(notes=["a", "b"])

    def test_an_unmergeable_reply_gets_the_schemas_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output, retry_prompts = _native_run(monkeypatch, ['{"notes": ["a"]}', "Done."])

        assert len(retry_prompts) == 1
        assert "Your reply could not be read, so none of it was used." in retry_prompts[0]
        assert output == _Notes(notes=["fixed"])
