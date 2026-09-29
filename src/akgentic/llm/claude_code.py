"""Claude Code CLI as a pydantic-ai model.

``ClaudeCodeModel`` answers a model request by running the Claude Code CLI in
print mode (``claude --print --output-format json``) and reading the JSON result
it prints. The CLI brings its own authentication, so the provider works with a
Claude subscription (``claude login`` or ``CLAUDE_CODE_OAUTH_TOKEN`` from
``claude setup-token``) as well as with ``ANTHROPIC_API_KEY``.

The CLI is used as a plain model, not as an agent: its built-in tools, MCP
servers, hooks, skills and ``CLAUDE.md`` discovery are all switched off, and it
runs in an empty temporary directory. Tools stay where they are in every other
provider, in the calling process.

Function calling is emulated, because the CLI takes one prompt and returns one
answer. The function tools and output tools of the request are described in the
system prompt, and the CLI is asked for structured output (``--json-schema``)
that either carries a text answer or names the calls the model wants made.
Those calls come back as ordinary ``ToolCallPart``s, so the agent loop cannot
tell the difference. The conversation so far is replayed as a transcript in the
prompt on every request.

Example:
    >>> from akgentic.llm.claude_code import ClaudeCodeModel
    >>> model = ClaudeCodeModel("sonnet")  # doctest: +SKIP
    >>> from pydantic_ai import Agent
    >>> agent = Agent(model, instructions="You are helpful")  # doctest: +SKIP
    >>> result = await agent.run("Hello!")  # doctest: +SKIP
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    CompletedStreamedResponse,
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger(__name__)

CLAUDE_CODE_CLI_ENV = "CLAUDE_CODE_CLI"
"""Environment variable naming the CLI executable; defaults to ``claude`` on the PATH."""

DEFAULT_TIMEOUT_S = 300.0

type Effort = Literal["low", "medium", "high"]

# Without a system prompt the CLI falls back to its own, which describes a coding
# agent with tools this provider has switched off.
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_FUNCTION_CALLING_PROMPT = """\
# Function calling

You are connected to an application that runs functions on your behalf. You cannot
run them yourself: you name the calls you want, the application runs them and sends
you the results in a later message. These functions are available and working:

<functions>
{functions}
</functions>

Answer through the structured output:
- to call functions, list them in `tool_calls` with `arguments` matching the
  function's `parameters` schema, and leave `content` empty;
- to answer in text, put the answer in `content` and leave `tool_calls` empty.
{rules}
Never invent a function result and never claim a function is unavailable."""

# Settings the CLI has no flag for. They are dropped, not rejected, so a ModelConfig
# written for another provider keeps working when its provider is switched.
_IGNORED_SETTINGS = ("temperature", "max_tokens", "seed", "top_p")


class ClaudeCodeModel(Model[None]):
    """A pydantic-ai model backed by the Claude Code CLI.

    Args:
        model_name: Model alias (``sonnet``, ``opus``, ``haiku``) or full model id,
            passed to the CLI's ``--model``.
        cli_path: CLI executable. Defaults to the ``CLAUDE_CODE_CLI`` environment
            variable, then to ``claude`` on the PATH.
        timeout_s: Seconds a single CLI run may take before it is killed.
        effort: Reasoning effort, passed to the CLI's ``--effort``.
        settings: Model settings used as defaults for this model.
        profile: Model profile override.

    Raises:
        UserError: If the model name could be read as a CLI flag, or the CLI
            executable cannot be found.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cli_path: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        effort: Effort | None = None,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        # The name becomes a command-line argument: it must never be read as a flag.
        if not model_name or model_name.startswith("-"):
            raise UserError(f"Invalid Claude Code model name: {model_name!r}")

        requested = cli_path or os.getenv(CLAUDE_CODE_CLI_ENV) or "claude"
        resolved = shutil.which(requested)
        if resolved is None:
            raise UserError(
                f"Claude Code CLI not found: {requested!r}. Install it "
                f"(https://claude.com/claude-code) or set {CLAUDE_CODE_CLI_ENV} to its path."
            )

        self._model_name = model_name
        self._cli_path = resolved
        self._timeout_s = timeout_s
        self._effort = effort
        super().__init__(settings=settings, profile=profile)

    @property
    def model_name(self) -> str:
        """The model alias or id this model was built with."""
        return self._model_name

    @property
    def system(self) -> str:
        """The provider whose models the CLI serves.

        ``anthropic`` and not ``claude-code``: responses carry the real Anthropic
        model id the CLI resolved the alias to, and genai-prices looks prices up
        under the provider that owns that id.
        """
        return "anthropic"

    @property
    def base_url(self) -> str | None:
        """Always None: the model talks to a local executable, not to a URL."""
        return None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Run the CLI once for this request and turn its result into a response."""
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        _warn_about_ignored_settings(model_settings)

        tools = [
            *model_request_parameters.declared_function_tools,
            *model_request_parameters.output_tools,
        ]
        system_prompt = _system_prompt(
            messages, self._get_instruction_parts(messages, model_request_parameters)
        )
        schema: dict[str, Any] | None = None
        if tools:
            parallel = (model_settings or {}).get("parallel_tool_calls", True)
            section, schema = _function_calling(
                tools,
                allow_text=model_request_parameters.allow_text_output,
                parallel=parallel is not False,
            )
            system_prompt = f"{system_prompt}\n\n{section}"

        result = await self._run_cli(system_prompt, _render_prompt(messages), schema)
        return self._to_response(result, expects_calls=bool(tools))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Answer a streamed request with the complete response, replayed as events.

        The CLI is run to completion first, so nothing arrives incrementally; the
        stream exists so that streaming consumers keep working with this model.
        """
        response = await self.request(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(
            response,
            model_request_parameters=model_request_parameters,
            replay_events=True,
        )

    def _command(self, system_prompt_file: Path, schema: dict[str, Any] | None) -> list[str]:
        command = [
            self._cli_path,
            "--print",
            "--output-format",
            "json",
            "--model",
            self._model_name,
            "--system-prompt-file",
            str(system_prompt_file),
            # A plain model: no built-in tools, no MCP servers, and none of the
            # user's hooks, skills, plugins or CLAUDE.md files.
            "--tools",
            "",
            "--strict-mcp-config",
            "--safe-mode",
            # A prompt that starts with "/" is text, not a command.
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if self._effort is not None:
            command += ["--effort", self._effort]
        if schema is not None:
            command += ["--json-schema", json.dumps(schema)]
        return command

    async def _run_cli(
        self, system_prompt: str, prompt: str, schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Run the CLI in an empty directory and return the result object it printed."""
        with tempfile.TemporaryDirectory(prefix="akgentic-claude-code-") as workdir:
            # A file, not an argument: system prompts outgrow the argument size limit.
            system_prompt_file = Path(workdir) / "system-prompt.txt"
            system_prompt_file.write_text(system_prompt, encoding="utf-8")
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(system_prompt_file, schema),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                )
            except OSError as exc:
                raise ModelAPIError(
                    self._model_name, f"Claude Code CLI could not be started: {exc}"
                ) from exc
            stdout, stderr = await self._communicate(process, prompt)

        if not stdout.strip():
            detail = stderr.decode("utf-8", errors="replace").strip()[:1000]
            raise ModelAPIError(
                self._model_name,
                f"Claude Code CLI exited with code {process.returncode} and no output: {detail}",
            )
        result = _parse_result(self._model_name, stdout)
        if result.get("is_error") or process.returncode != 0:
            self._raise_for_error(result)
        return result

    async def _communicate(
        self, process: asyncio.subprocess.Process, prompt: str
    ) -> tuple[bytes, bytes]:
        try:
            return await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=self._timeout_s
            )
        except TimeoutError as exc:
            await _kill(process)
            raise ModelAPIError(
                self._model_name,
                f"Claude Code CLI did not answer within {self._timeout_s:g}s",
            ) from exc
        except asyncio.CancelledError:
            # The run was cancelled: do not leave the CLI running.
            await _kill(process)
            raise

    def _raise_for_error(self, result: dict[str, Any]) -> None:
        detail = str(result.get("result") or result.get("subtype") or "unknown error")
        status = result.get("api_error_status")
        if isinstance(status, int):
            raise ModelHTTPError(status_code=status, model_name=self._model_name, body=detail)
        raise ModelAPIError(self._model_name, f"Claude Code CLI failed: {detail}")

    def _to_response(self, result: dict[str, Any], *, expects_calls: bool) -> ModelResponse:
        parts: list[ModelResponsePart] = []
        if expects_calls:
            structured = result.get("structured_output")
            if not isinstance(structured, dict):
                raise UnexpectedModelBehavior(
                    "Claude Code CLI returned no structured output",
                    body=str(result.get("result"))[:1000],
                )
            parts = _call_parts(structured)
            if not parts and (content := structured.get("content")):
                parts = [TextPart(content=str(content))]
        elif text := result.get("result"):
            parts = [TextPart(content=str(text))]

        has_calls = any(isinstance(part, ToolCallPart) for part in parts)
        return ModelResponse(
            parts=parts,
            usage=_usage(result),
            model_name=_resolved_model_name(result) or self._model_name,
            provider_name=self.system,
            provider_response_id=result.get("session_id"),
            finish_reason="tool_call" if has_calls else "stop",
        )


async def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


def _warn_about_ignored_settings(model_settings: ModelSettings | None) -> None:
    settings: dict[str, Any] = dict(model_settings or {})
    ignored = [name for name in _IGNORED_SETTINGS if settings.get(name) is not None]
    if ignored:
        logger.debug("Claude Code CLI has no equivalent for %s: ignored", ", ".join(ignored))


def _system_prompt(messages: Sequence[ModelMessage], instruction_parts: Any) -> str:
    """Join the system prompt parts of the history and the instructions of the request."""
    sections = [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    sections += [part.content for part in instruction_parts or []]
    return "\n\n".join(section for section in sections if section) or _DEFAULT_SYSTEM_PROMPT


def _user_text(part: UserPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if not isinstance(item, str):
            raise UserError(
                "The claude-code provider takes text prompts only; "
                f"got {type(item).__name__} in a user prompt"
            )
        texts.append(item)
    return "\n".join(texts)


def _render_request(message: ModelRequest) -> list[str]:
    turns: list[str] = []
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            turns.append(f"[user]\n{_user_text(part)}\n")
        elif isinstance(part, ToolReturnPart):
            turns.append(
                f"[result of function call {part.tool_call_id} ({part.tool_name})]\n"
                f"{part.model_response_str()}\n"
            )
        elif isinstance(part, RetryPromptPart):
            target = f" of function call {part.tool_call_id} ({part.tool_name})"
            header = f"[correction{target if part.tool_name else ''}]"
            turns.append(f"{header}\n{part.model_response()}\n")
    return turns


def _render_response(message: ModelResponse) -> list[str]:
    text = "\n".join(part.content for part in message.parts if isinstance(part, TextPart))
    calls = [
        {"id": part.tool_call_id, "name": part.tool_name, "arguments": part.args_as_dict()}
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    if not text and not calls:
        return []
    lines = ["[assistant]"]
    if text:
        lines.append(text)
    if calls:
        lines.append(f"Function calls requested: {json.dumps(calls)}")
    return ["\n".join(lines) + "\n"]


def _render_prompt(messages: Sequence[ModelMessage]) -> str:
    """Render the conversation as the single prompt the CLI takes."""
    turns: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            turns += _render_request(message)
        else:
            turns += _render_response(message)

    if len(turns) == 1 and turns[0].startswith("[user]\n"):
        return turns[0].removeprefix("[user]\n").rstrip("\n") or "(empty message)"
    if not turns:
        return "(empty message)"
    return "\n".join(
        [
            "Conversation so far:",
            "",
            *turns,
            "Continue the conversation: give the next [assistant] turn.",
        ]
    )


def _function_calling(
    tools: Sequence[ToolDefinition], *, allow_text: bool, parallel: bool
) -> tuple[str, dict[str, Any]]:
    """Build the system prompt section and the output schema that emulate function calling."""
    functions = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters_json_schema,
        }
        for tool in tools
    ]
    calls_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [tool.name for tool in tools]},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
        },
    }
    rules = ""
    if not allow_text:
        calls_schema["minItems"] = 1
        rules += "You must call at least one function; a text answer is not accepted.\n"
    if not parallel:
        calls_schema["maxItems"] = 1
        rules += "Call one function at a time.\n"

    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}, "tool_calls": calls_schema},
        "required": ["content", "tool_calls"],
    }
    section = _FUNCTION_CALLING_PROMPT.format(
        functions=json.dumps(functions, indent=1), rules=rules
    )
    return section, schema


def _call_parts(structured: dict[str, Any]) -> list[ModelResponsePart]:
    parts: list[ModelResponsePart] = []
    for call in structured.get("tool_calls") or []:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        arguments = call.get("arguments")
        parts.append(
            ToolCallPart(
                tool_name=str(call["name"]),
                args=arguments if isinstance(arguments, dict) else {},
                tool_call_id=f"call_{uuid.uuid4().hex[:24]}",
            )
        )
    return parts


def _parse_result(model_name: str, stdout: bytes) -> dict[str, Any]:
    """Read the result object the CLI printed; tolerate stray lines around it."""
    text = stdout.decode("utf-8", errors="replace").strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    # Some CLI versions print the full event list instead of the single result.
    if isinstance(parsed, list):
        parsed = next(
            (e for e in reversed(parsed) if isinstance(e, dict) and e.get("type") == "result"),
            None,
        )
    if not isinstance(parsed, dict):
        raise ModelAPIError(
            model_name, f"Claude Code CLI printed no readable result: {text[:500]!r}"
        )
    return parsed


def _resolved_model_name(result: dict[str, Any]) -> str | None:
    """The model id the CLI resolved the alias to: the one that produced most output."""
    usage = result.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        return None

    def output_tokens(name: str) -> int:
        entry = usage[name]
        return int(entry.get("outputTokens", 0)) if isinstance(entry, dict) else 0

    return str(max(usage, key=output_tokens))


def _usage(result: dict[str, Any]) -> RequestUsage:
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return RequestUsage()
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    return RequestUsage(
        # pydantic-ai counts cached tokens as input tokens; the CLI reports them apart.
        input_tokens=int(usage.get("input_tokens") or 0) + cache_read + cache_write,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=int(usage.get("output_tokens") or 0),
    )
