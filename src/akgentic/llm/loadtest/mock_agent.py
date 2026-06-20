"""Drop-in mock for ``ReactAgent`` driven by a YAML state machine.

Implements ADR-007 §2–§5. ``MockReactAgent`` reproduces a recorded scenario's
routing and LLM event stream at zero token cost: it owns a real
``ContextManager`` and feeds synthetic pydantic-ai messages through it, so a
subscribed observer sees the same event sequence a real run would emit — but no
model or provider is ever built and no real tool is invoked.
"""

import asyncio
import re
import uuid
from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from akgentic.llm.context import ContextManager, ContextSnapshot
from akgentic.llm.event import ContextObserver, LlmMessageEvent
from akgentic.llm.loadtest.scenario import (
    AgentScript,
    ScenarioState,
    ToolStub,
    _resolve_scenario_ref,
    load_scenario,
)


class MockProviderReachedError(RuntimeError):
    """Raised if a provider factory is reached during a mock run (zero-egress guard)."""


class MockReactAgent:
    """Transparent ``ReactAgent`` replacement backed by a scenario state machine."""

    def __init__(
        self,
        config: Any,
        deps_type: type[Any] | None = None,
        tools: list[Any] | None = None,
        toolsets: list[Any] | None = None,
        result_type: type[Any] = str,
        observer: ContextObserver | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Mirror ``ReactAgent.__init__`` without building a model or provider.

        ``_model``/``_http_client`` stay ``None`` (zero-token guarantee). The
        scenario is resolved from ``config`` and loaded eagerly.
        """
        self._config = config
        self._deps_type = deps_type
        self._result_type = result_type
        self._event_loop = event_loop
        self._model: Any = None
        self._http_client: Any = None

        self._context = ContextManager()
        if observer:
            self._context.subscribe(observer)

        self._scenario = load_scenario(_resolve_scenario_ref(config))
        self._consumed: set[int] = set()
        self._run_id: str = str(uuid.uuid4())

    # --- model/provider guard ------------------------------------------------

    def _build_model(self) -> Any:
        """Guard: a mock run must never reach a provider factory."""
        raise MockProviderReachedError(
            "MockReactAgent must never build a model or contact a provider"
        )

    # --- run -----------------------------------------------------------------

    async def run(
        self, user_prompt: Any, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Replay the matching state's event stream and return its output."""
        self._run_id = str(uuid.uuid4())
        name = self._agent_name(deps)
        script = self._scenario.agents[name]
        state = self._select_state(script, user_prompt)
        self._emit_request(user_prompt)
        for stub in state.tools:
            self._emit_tool_call(stub)
            self._emit_tool_return(stub)
        self._emit_final_response(state)
        await self._sleep(self._latency_ms(state))
        return self._build_output(state, output_type)

    def run_sync(
        self, user_prompt: Any, deps: Any = None, output_type: type[Any] | None = None
    ) -> Any:
        """Synchronous wrapper around :meth:`run` (mirrors ``ReactAgent.run_sync``)."""
        if self._event_loop and self._event_loop.is_running():
            return self._event_loop.run_until_complete(
                self.run(user_prompt, deps, output_type)
            )
        return asyncio.run(self.run(user_prompt, deps, output_type))

    async def aclose(self) -> None:
        """No-op teardown (mirrors ``ReactAgent.aclose``; the mock holds no client)."""
        return None

    # --- state selection -----------------------------------------------------

    @staticmethod
    def _agent_name(deps: Any) -> str:
        """Duck-type the agent name from ``deps.config.name`` (no agent import)."""
        return str(getattr(deps.config, "name"))

    def _select_state(self, script: AgentScript, user_prompt: Any) -> ScenarioState:
        """Return the first matching unconsumed state, else the agent's default."""
        prompt = self._prompt_text(user_prompt)
        for state in script.states:
            key = id(state)
            if key in self._consumed:
                continue
            if self._matches(state, prompt):
                self._consumed.add(key)
                return state
        return script.default

    @staticmethod
    def _prompt_text(user_prompt: Any) -> str:
        """Flatten a str or multimodal-list prompt to text for matching."""
        if isinstance(user_prompt, str):
            return user_prompt
        if isinstance(user_prompt, list):
            return " ".join(p for p in user_prompt if isinstance(p, str))
        return str(user_prompt)

    @staticmethod
    def _matches(state: ScenarioState, prompt: str) -> bool:
        """Evaluate a state's optional ``when`` matcher against ``prompt``."""
        spec = state.when
        if spec is None:
            return True
        if spec.contains is not None and spec.contains.lower() not in prompt.lower():
            return False
        if spec.regex is not None and re.search(spec.regex, prompt) is None:
            return False
        if spec.from_sender is not None and spec.from_sender not in prompt:
            return False
        return True

    # --- event emission (synthetic pydantic-ai messages) ---------------------

    def _emit_request(self, user_prompt: Any) -> None:
        """Push the inbound prompt as a ``ModelRequest(UserPromptPart)``."""
        content = self._prompt_text(user_prompt)
        self._context.add_message(
            ModelRequest(parts=[UserPromptPart(content=content)], run_id=self._run_id)
        )

    def _emit_tool_call(self, stub: ToolStub) -> None:
        """Push a ``ModelResponse(ToolCallPart)`` for a simulated tool call."""
        self._context.add_message(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=stub.name,
                        args=stub.args,
                        tool_call_id=self._tool_call_id(stub),
                    )
                ],
                run_id=self._run_id,
            )
        )

    def _emit_tool_return(self, stub: ToolStub) -> None:
        """Push a ``ModelRequest(ToolReturnPart)`` with the canned tool result."""
        self._context.add_message(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=stub.name,
                        content=stub.returns,
                        tool_call_id=self._tool_call_id(stub),
                    )
                ],
                run_id=self._run_id,
            )
        )

    def _emit_final_response(self, state: ScenarioState) -> None:
        """Push the final ``ModelResponse(TextPart=<json>)`` with zero-token usage."""
        self._context.add_message(
            ModelResponse(
                parts=[TextPart(content=self._response_text(state))],
                usage=RequestUsage(),
                model_name="mock",
                provider_name="mock",
                run_id=self._run_id,
            )
        )

    # --- output --------------------------------------------------------------

    def _build_output(self, state: ScenarioState, output_type: type[Any] | None) -> Any:
        """Validate a structured ``output_type`` or return the response text."""
        if output_type is not None and output_type is not str and hasattr(
            output_type, "model_validate"
        ):
            messages = [m.model_dump() for m in state.respond.messages]
            return output_type.model_validate({"messages": messages})
        return state.respond.text or ""

    @staticmethod
    def _response_text(state: ScenarioState) -> str:
        """Build the JSON ``TextPart`` content mirroring a real structured output."""
        if state.respond.text is not None:
            return state.respond.text
        from pydantic import TypeAdapter  # noqa: PLC0415

        adapter: TypeAdapter[Any] = TypeAdapter(list[dict[str, str]])
        messages = [m.model_dump() for m in state.respond.messages]
        return '{"messages": ' + adapter.dump_json(messages).decode("utf-8") + "}"

    # --- latency -------------------------------------------------------------

    def _latency_ms(self, state: ScenarioState) -> int:
        """Per-state latency, falling back to the scenario default."""
        if state.latency_ms is not None:
            return state.latency_ms
        return self._scenario.default_latency_ms

    @staticmethod
    async def _sleep(latency_ms: int) -> None:
        """Inject think-time; a zero/negative value adds no measurable delay."""
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)

    # --- ids -----------------------------------------------------------------

    @staticmethod
    def _tool_call_id(stub: ToolStub) -> str:
        """Deterministic-per-instance tool call id matching call and return."""
        return f"mock-{stub.name}-{id(stub)}"

    # --- drop-in surface delegating to ContextManager ------------------------

    @property
    def context(self) -> ContextManager:
        """Underlying context manager (drop-in parity with ``ReactAgent.context``)."""
        return self._context

    def subscribe_context(self, observer: ContextObserver) -> None:
        """Subscribe an observer to context events."""
        self._context.subscribe(observer)

    def checkpoint(self, checkpoint_id: str | None = None) -> ContextSnapshot:
        """Create a context checkpoint."""
        return self._context.checkpoint(checkpoint_id)

    def rewind(self, checkpoint_id: str) -> None:
        """Restore context to a checkpoint."""
        self._context.rewind(checkpoint_id)

    def restore_context(self, events: list[Any]) -> None:
        """Restore LLM context from persisted events (mirrors ``ReactAgent``)."""
        messages = [
            e.event.message
            for e in events
            if hasattr(e, "event") and isinstance(e.event, LlmMessageEvent)
        ]
        self._context.restore(messages)

    def system_prompt(self, func: Any) -> Any:
        """No-op decorator: returns ``func`` unchanged (no model to register on)."""
        return func

    def tool(self, func: Any) -> Any:
        """No-op decorator: returns ``func`` unchanged (tools are simulated)."""
        return func
