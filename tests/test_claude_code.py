"""Tests for the Claude Code CLI model (claude_code.py).

No test here runs the real CLI. ``fake_cli`` writes a small executable that
records how it was invoked and prints a canned result, so the suite needs
neither the CLI nor a credential and makes no request.
"""

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.tools import ToolDefinition

from akgentic.llm import ModelConfig, create_model, create_model_settings, get_output_type
from akgentic.llm.claude_code import (
    CLAUDE_CODE_CLI_ENV,
    ClaudeCodeModel,
    _render_prompt,
    _resolved_model_name,
    _usage,
)

_FAKE_CLI = """\
#!{python}
import json, os, sys, time
from pathlib import Path

here = Path(__file__).parent
args = sys.argv[1:]
system_prompt = None
if "--system-prompt-file" in args:
    system_prompt = Path(args[args.index("--system-prompt-file") + 1]).read_text()
(here / "invocation.json").write_text(json.dumps({{
    "args": args,
    "stdin": sys.stdin.read(),
    "cwd": os.getcwd(),
    "cwd_files": sorted(os.listdir(os.getcwd())),
    "system_prompt": system_prompt,
}}))
behaviour = json.loads((here / "behaviour.json").read_text())
time.sleep(behaviour.get("sleep", 0))
sys.stdout.write(behaviour.get("stdout", ""))
sys.stderr.write(behaviour.get("stderr", ""))
sys.exit(behaviour.get("exit_code", 0))
"""

_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Current weather in a city.",
    parameters_json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def _result(**overrides: Any) -> dict[str, Any]:
    """A CLI result object, shaped like ``claude --print --output-format json`` prints it."""
    result: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "pong",
        "session_id": "15e36173-70ea-46cc-b871-a9a1104a8f83",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 9,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 1000,
            "output_tokens": 42,
        },
        "modelUsage": {"claude-haiku-4-5-20251001": {"outputTokens": 42}},
    }
    result.update(overrides)
    return result


class FakeCli:
    """Handle on the fake executable: set what it prints, read how it was called."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.path = directory / "claude"
        self.path.write_text(_FAKE_CLI.format(python=sys.executable))
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR)
        self.answers(_result())

    def behaves(self, **behaviour: Any) -> None:
        (self._directory / "behaviour.json").write_text(json.dumps(behaviour))

    def answers(self, result: dict[str, Any]) -> None:
        self.behaves(stdout=json.dumps(result))

    @property
    def invocation(self) -> dict[str, Any]:
        return json.loads((self._directory / "invocation.json").read_text())  # type: ignore[no-any-return]

    def option(self, name: str) -> str:
        args: list[str] = self.invocation["args"]
        return args[args.index(name) + 1]


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    cli = FakeCli(tmp_path)
    monkeypatch.setenv(CLAUDE_CODE_CLI_ENV, str(cli.path))
    return cli


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


async def _request(
    model: ClaudeCodeModel,
    messages: list[Any],
    parameters: ModelRequestParameters | None = None,
    settings: Any = None,
) -> ModelResponse:
    return await model.request(messages, settings, parameters or ModelRequestParameters())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_cli_from_environment(self, fake_cli: FakeCli) -> None:
        model = ClaudeCodeModel("sonnet")
        assert model.model_name == "sonnet"
        assert model.system == "anthropic"
        assert model.base_url is None

    def test_explicit_cli_path_wins(
        self, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CLAUDE_CODE_CLI_ENV, "/nonexistent/claude")
        ClaudeCodeModel("sonnet", cli_path=str(fake_cli.path))

    def test_missing_cli_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CLAUDE_CODE_CLI_ENV, "/nonexistent/claude")
        with pytest.raises(UserError, match="Claude Code CLI not found"):
            ClaudeCodeModel("sonnet")

    @pytest.mark.parametrize("name", ["", "--dangerously-skip-permissions", "-p"])
    def test_model_name_cannot_be_a_flag(self, fake_cli: FakeCli, name: str) -> None:
        with pytest.raises(UserError, match="Invalid Claude Code model name"):
            ClaudeCodeModel(name)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


class TestCommand:
    async def test_runs_as_a_plain_model(self, fake_cli: FakeCli) -> None:
        """No built-in tools, no MCP servers, no user customizations, no session."""
        await _request(ClaudeCodeModel("sonnet"), [_user("hello")])

        args = fake_cli.invocation["args"]
        assert args[:3] == ["--print", "--output-format", "json"]
        assert fake_cli.option("--model") == "sonnet"
        assert fake_cli.option("--tools") == ""
        for flag in (
            "--strict-mcp-config",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
        ):
            assert flag in args
        assert "--json-schema" not in args
        assert "--effort" not in args

    async def test_prompt_travels_on_stdin(self, fake_cli: FakeCli) -> None:
        await _request(ClaudeCodeModel("sonnet"), [_user("/not a command")])
        invocation = fake_cli.invocation
        assert invocation["stdin"] == "/not a command"
        assert "/not a command" not in invocation["args"]

    async def test_runs_in_an_otherwise_empty_directory(self, fake_cli: FakeCli) -> None:
        await _request(ClaudeCodeModel("sonnet"), [_user("hello")])
        invocation = fake_cli.invocation
        assert invocation["cwd_files"] == ["system-prompt.txt"]
        assert not Path(invocation["cwd"]).exists()  # removed once the run is over

    async def test_effort_is_passed_on(self, fake_cli: FakeCli) -> None:
        await _request(ClaudeCodeModel("sonnet", effort="high"), [_user("hello")])
        assert fake_cli.option("--effort") == "high"

    async def test_settings_without_a_flag_are_ignored(self, fake_cli: FakeCli) -> None:
        response = await _request(
            ClaudeCodeModel("sonnet"),
            [_user("hello")],
            settings={"temperature": 0.2, "max_tokens": 100, "seed": 1},
        )
        assert response.parts == [TextPart(content="pong")]
        assert "0.2" not in fake_cli.invocation["args"]


# ---------------------------------------------------------------------------
# System prompt and transcript
# ---------------------------------------------------------------------------


class TestPrompt:
    async def test_default_system_prompt_replaces_the_cli_one(self, fake_cli: FakeCli) -> None:
        await _request(ClaudeCodeModel("sonnet"), [_user("hello")])
        assert fake_cli.invocation["system_prompt"] == "You are a helpful assistant."

    async def test_system_parts_and_instructions_are_joined(self, fake_cli: FakeCli) -> None:
        messages = [
            ModelRequest(
                parts=[SystemPromptPart(content="Be terse."), UserPromptPart(content="hello")],
                instructions="Answer in French.",
            )
        ]
        await _request(ClaudeCodeModel("sonnet"), messages)
        assert fake_cli.invocation["system_prompt"] == "Be terse.\n\nAnswer in French."
        assert fake_cli.invocation["stdin"] == "hello"

    def test_single_user_message_is_sent_as_is(self) -> None:
        assert _render_prompt([_user("hello")]) == "hello"

    def test_empty_history_still_yields_a_prompt(self) -> None:
        assert _render_prompt([]) == "(empty message)"

    def test_history_is_replayed_as_a_transcript(self) -> None:
        messages = [
            _user("Weather in Ghent?"),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather", args={"city": "Ghent"}, tool_call_id="call_1"
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_weather", content="7.5 C, rain", tool_call_id="call_1"
                    )
                ]
            ),
            ModelResponse(parts=[TextPart(content="It rains.")]),
            ModelRequest(
                parts=[RetryPromptPart(content="Answer in one word."), UserPromptPart("And now?")]
            ),
        ]
        prompt = _render_prompt(messages)

        assert prompt.startswith("Conversation so far:")
        assert prompt.endswith("give the next [assistant] turn.")
        assert "[user]\nWeather in Ghent?" in prompt
        assert '"name": "get_weather", "arguments": {"city": "Ghent"}' in prompt
        assert "[result of function call call_1 (get_weather)]\n7.5 C, rain" in prompt
        assert "[assistant]\nIt rains." in prompt
        assert "[correction]\n" in prompt and "Answer in one word." in prompt
        # Order is the order of the conversation.
        assert prompt.index("Weather in Ghent?") < prompt.index("7.5 C, rain")
        assert prompt.index("7.5 C, rain") < prompt.index("It rains.")
        assert prompt.index("It rains.") < prompt.index("And now?")

    def test_retry_of_a_tool_call_names_the_call(self) -> None:
        messages = [
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="city is required", tool_name="get_weather", tool_call_id="call_1"
                    )
                ]
            )
        ]
        assert "[correction of function call call_1 (get_weather)]" in _render_prompt(messages)

    def test_non_text_user_content_is_rejected(self) -> None:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=["look", ImageUrl(url="https://example.com/a.png")])
                ]
            )
        ]
        with pytest.raises(UserError, match="text prompts only"):
            _render_prompt(messages)

    def test_text_items_of_a_user_prompt_are_joined(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content=["one", "two"])])]
        assert _render_prompt(messages) == "one\ntwo"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TestResponse:
    async def test_text_answer(self, fake_cli: FakeCli) -> None:
        response = await _request(ClaudeCodeModel("haiku"), [_user("ping")])

        assert response.parts == [TextPart(content="pong")]
        assert response.finish_reason == "stop"
        assert response.provider_name == "anthropic"
        assert response.provider_response_id == "15e36173-70ea-46cc-b871-a9a1104a8f83"

    async def test_alias_is_resolved_to_the_model_that_answered(self, fake_cli: FakeCli) -> None:
        response = await _request(ClaudeCodeModel("haiku"), [_user("ping")])
        assert response.model_name == "claude-haiku-4-5-20251001"

    async def test_cached_tokens_count_as_input(self, fake_cli: FakeCli) -> None:
        response = await _request(ClaudeCodeModel("haiku"), [_user("ping")])

        assert response.usage.input_tokens == 9 + 100 + 1000
        assert response.usage.cache_write_tokens == 100
        assert response.usage.cache_read_tokens == 1000
        assert response.usage.output_tokens == 42

    def test_usage_is_empty_when_the_cli_reports_none(self) -> None:
        assert _usage({}).input_tokens == 0

    def test_resolved_model_is_the_one_with_most_output(self) -> None:
        result = {
            "modelUsage": {
                "claude-haiku-4-5-20251001": {"outputTokens": 3},
                "claude-sonnet-5-5": {"outputTokens": 40},
            }
        }
        assert _resolved_model_name(result) == "claude-sonnet-5-5"
        assert _resolved_model_name({}) is None

    async def test_result_is_found_among_stray_lines(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(stdout=f"warning: something\n{json.dumps(_result())}\n")
        response = await _request(ClaudeCodeModel("haiku"), [_user("ping")])
        assert response.parts == [TextPart(content="pong")]

    async def test_result_is_found_in_an_event_list(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(stdout=json.dumps([{"type": "system"}, _result()]))
        response = await _request(ClaudeCodeModel("haiku"), [_user("ping")])
        assert response.parts == [TextPart(content="pong")]

    async def test_stream_replays_the_complete_response(self, fake_cli: FakeCli) -> None:
        model = ClaudeCodeModel("haiku")
        async with model.request_stream(
            [_user("ping")], None, ModelRequestParameters()
        ) as stream:
            events = [event async for event in stream]
            response = stream.get()

        assert events
        assert response.parts == [TextPart(content="pong")]


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------


class TestFunctionCalling:
    async def test_functions_are_described_and_a_schema_requested(
        self, fake_cli: FakeCli
    ) -> None:
        fake_cli.answers(_result(structured_output={"content": "hi", "tool_calls": []}))
        await _request(
            ClaudeCodeModel("haiku"),
            [_user("Weather in Ghent?")],
            ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
        )

        system_prompt = fake_cli.invocation["system_prompt"]
        assert "# Function calling" in system_prompt
        assert '"name": "get_weather"' in system_prompt
        assert "Current weather in a city." in system_prompt

        schema = json.loads(fake_cli.option("--json-schema"))
        calls = schema["properties"]["tool_calls"]
        assert calls["items"]["properties"]["name"]["enum"] == ["get_weather"]
        assert "minItems" not in calls
        assert "maxItems" not in calls

    async def test_call_comes_back_as_a_tool_call_part(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(
            _result(
                structured_output={
                    # Text next to a call contradicts it: the call wins.
                    "content": "I cannot reach the weather service.",
                    "tool_calls": [{"name": "get_weather", "arguments": {"city": "Ghent"}}],
                }
            )
        )
        response = await _request(
            ClaudeCodeModel("haiku"),
            [_user("Weather in Ghent?")],
            ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
        )

        assert len(response.parts) == 1
        call = response.parts[0]
        assert isinstance(call, ToolCallPart)
        assert call.tool_name == "get_weather"
        assert call.args == {"city": "Ghent"}
        assert call.tool_call_id.startswith("call_")
        assert response.finish_reason == "tool_call"

    async def test_text_answer_with_tools_available(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(_result(structured_output={"content": "Hello!", "tool_calls": []}))
        response = await _request(
            ClaudeCodeModel("haiku"),
            [_user("hello")],
            ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
        )
        assert response.parts == [TextPart(content="Hello!")]
        assert response.finish_reason == "stop"

    async def test_malformed_calls_are_dropped(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(
            _result(
                structured_output={
                    "content": "",
                    "tool_calls": [
                        "get_weather",
                        {"arguments": {"city": "Ghent"}},
                        {"name": "get_weather", "arguments": "Ghent"},
                    ],
                }
            )
        )
        response = await _request(
            ClaudeCodeModel("haiku"),
            [_user("Weather in Ghent?")],
            ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
        )
        assert [part.args for part in response.parts if isinstance(part, ToolCallPart)] == [{}]

    async def test_call_is_mandatory_when_text_is_not_allowed(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(_result(structured_output={"content": "", "tool_calls": []}))
        await _request(
            ClaudeCodeModel("haiku"),
            [_user("go")],
            ModelRequestParameters(
                output_mode="tool",
                output_tools=[
                    ToolDefinition(name="final_result", parameters_json_schema={"type": "object"})
                ],
                allow_text_output=False,
            ),
        )
        schema = json.loads(fake_cli.option("--json-schema"))
        calls = schema["properties"]["tool_calls"]
        assert calls["minItems"] == 1
        assert calls["items"]["properties"]["name"]["enum"] == ["final_result"]
        assert "You must call at least one function" in fake_cli.invocation["system_prompt"]

    async def test_one_call_at_a_time_without_parallel_tool_calls(
        self, fake_cli: FakeCli
    ) -> None:
        fake_cli.answers(_result(structured_output={"content": "", "tool_calls": []}))
        await _request(
            ClaudeCodeModel("haiku"),
            [_user("go")],
            ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
            settings={"parallel_tool_calls": False},
        )
        schema = json.loads(fake_cli.option("--json-schema"))
        assert schema["properties"]["tool_calls"]["maxItems"] == 1

    async def test_missing_structured_output_is_unexpected_behaviour(
        self, fake_cli: FakeCli
    ) -> None:
        fake_cli.answers(_result(result="I would rather chat."))
        with pytest.raises(UnexpectedModelBehavior, match="no structured output"):
            await _request(
                ClaudeCodeModel("haiku"),
                [_user("go")],
                ModelRequestParameters(function_tools=[_WEATHER_TOOL]),
            )


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class TestFailures:
    async def test_cli_error_is_a_model_api_error(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(
            stdout=json.dumps(_result(is_error=True, result="Not logged in · Please run /login")),
            exit_code=1,
        )
        with pytest.raises(ModelAPIError, match="Not logged in"):
            await _request(ClaudeCodeModel("haiku"), [_user("ping")])

    async def test_api_status_is_a_model_http_error(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(_result(is_error=True, result="Overloaded", api_error_status=529))
        with pytest.raises(ModelHTTPError) as excinfo:
            await _request(ClaudeCodeModel("haiku"), [_user("ping")])
        assert excinfo.value.status_code == 529
        assert excinfo.value.body == "Overloaded"

    async def test_error_without_a_message_names_the_subtype(self, fake_cli: FakeCli) -> None:
        fake_cli.answers(
            _result(is_error=True, result=None, subtype="error_max_structured_output_retries")
        )
        with pytest.raises(ModelAPIError, match="error_max_structured_output_retries"):
            await _request(ClaudeCodeModel("haiku"), [_user("ping")])

    async def test_no_output_reports_stderr(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(stderr="error: unknown option '--safe-mode'", exit_code=1)
        with pytest.raises(ModelAPIError, match="unknown option '--safe-mode'"):
            await _request(ClaudeCodeModel("haiku"), [_user("ping")])

    async def test_unreadable_output(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(stdout="this is not json")
        with pytest.raises(ModelAPIError, match="no readable result"):
            await _request(ClaudeCodeModel("haiku"), [_user("ping")])

    async def test_timeout_kills_the_cli(self, fake_cli: FakeCli) -> None:
        fake_cli.behaves(stdout=json.dumps(_result()), sleep=30)
        with pytest.raises(ModelAPIError, match="did not answer within 0.5s"):
            await _request(ClaudeCodeModel("haiku", timeout_s=0.5), [_user("ping")])

    async def test_cli_that_cannot_start(self, fake_cli: FakeCli) -> None:
        model = ClaudeCodeModel("haiku")
        fake_cli.path.write_text("not an executable format")
        with pytest.raises(ModelAPIError, match="could not be started"):
            await _request(model, [_user("ping")])


# ---------------------------------------------------------------------------
# Through the factory and an agent
# ---------------------------------------------------------------------------


class _Verdict(BaseModel):
    city: str
    advice: str


class TestFactory:
    def test_create_model(self, fake_cli: FakeCli) -> None:
        model = create_model(ModelConfig(provider="claude-code", model="sonnet"))
        assert isinstance(model, ClaudeCodeModel)
        assert model.model_name == "sonnet"

    async def test_reasoning_effort_becomes_effort(self, fake_cli: FakeCli) -> None:
        config = ModelConfig(provider="claude-code", model="sonnet", reasoning_effort="low")
        await _request(create_model(config), [_user("ping")])  # type: ignore[arg-type]
        assert fake_cli.option("--effort") == "low"

    def test_missing_cli_fails_when_the_model_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CLAUDE_CODE_CLI_ENV, "/nonexistent/claude")
        with pytest.raises(UserError, match="Claude Code CLI not found"):
            create_model(ModelConfig(provider="claude-code", model="sonnet"))

    def test_no_native_output(self) -> None:
        config = ModelConfig(provider="claude-code", model="sonnet")
        assert get_output_type(config, _Verdict) is _Verdict
        settings = create_model_settings(config)
        assert settings is not None
        assert settings["parallel_tool_calls"] is False

    def test_can_back_a_prompt_based_primary(self, fake_cli: FakeCli) -> None:
        config = ModelConfig(
            provider="google-gla",
            model="gemini-2.0-flash",
            fallback_models=[ModelConfig(provider="claude-code", model="sonnet")],
        )
        assert config.fallback_models[0].provider == "claude-code"

    def test_cannot_back_a_native_primary(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-5",
                fallback_models=[ModelConfig(provider="claude-code", model="sonnet")],
            )

    def test_in_a_fallback_chain(self, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-a-real-credential")
        config = ModelConfig(
            provider="claude-code",
            model="sonnet",
            fallback_models=[ModelConfig(provider="mistral", model="mistral-large-latest")],
        )
        model = create_model(config)
        assert isinstance(model, FallbackModel)
        assert isinstance(model.models[0], ClaudeCodeModel)


class TestAgentRun:
    async def test_structured_output_through_the_output_tool(self, fake_cli: FakeCli) -> None:
        """A whole agent run: the typed result arrives as a call of the output tool."""
        fake_cli.answers(
            _result(
                structured_output={
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "final_result",
                            "arguments": {"city": "Ghent", "advice": "Take the tram."},
                        }
                    ],
                }
            )
        )
        config = ModelConfig(provider="claude-code", model="haiku")
        agent = Agent(
            create_model(config),
            model_settings=create_model_settings(config),
            output_type=get_output_type(config, _Verdict),
            instructions="You are a weather assistant.",
        )

        result = await agent.run("Should I cycle in Ghent?")

        assert result.output == _Verdict(city="Ghent", advice="Take the tram.")
        assert fake_cli.invocation["system_prompt"].startswith("You are a weather assistant.")
        schema = json.loads(fake_cli.option("--json-schema"))
        assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == [
            "final_result"
        ]
