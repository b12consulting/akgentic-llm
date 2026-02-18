"""REACT-based LLM agent with context management and iteration support."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai import UsageLimits as PydanticUsageLimits

from .config import ReactAgentConfig, UsageLimits
from .context import ContextManager, ContextObserver, ContextSnapshot
from .providers import create_http_client, create_model


class UsageLimitError(Exception):
    """Raised when usage limits are exceeded during agent execution."""

    pass


class ReactAgent:
    """REACT-based LLM agent extracted from V1 base_agent.py.

    Features:
    - REACT pattern support (via pydantic-ai)
    - Dynamic system prompts with registry
    - Context management with observer pattern
    - Checkpoint/rewind for error recovery
    - Iterative execution with context updates
    - Tool integration
    - Usage limit enforcement
    - HTTP retry logic

    Example:
        >>> config = ReactAgentConfig(
        ...     model=ModelConfig(provider="openai", model="gpt-4o")
        ... )
        >>> # Option 1: Pass observer at initialization
        >>> agent = ReactAgent(
        ...     config=config,
        ...     deps_type=MyDeps,
        ...     tools=[my_tool_func],
        ...     observer=my_observer
        ... )
        >>> # Option 2: Subscribe observer later
        >>> agent = ReactAgent(config=config)
        >>> agent.subscribe_context(my_observer)
        >>> result = await agent.run("User query")
    """

    def __init__(
        self,
        config: ReactAgentConfig,
        deps_type: type[Any] | None = None,
        tools: list[Any] | None = None,
        toolsets: list[Any] | None = None,
        result_type: type[Any] = str,
        observer: ContextObserver | None = None,
    ) -> None:
        """Initialize REACT agent.

        Args:
            config: Complete agent configuration
            deps_type: Type for dependency injection (optional)
            tools: List of tool functions (optional)
            toolsets: List of toolsets (optional, e.g., MCP servers)
            result_type: Type for agent result validation (default: str)
            observer: Context observer to register automatically (optional)
        """
        self._config = config
        self._deps_type = deps_type
        self._result_type = result_type

        # Create context manager (no max_messages by default)
        self._context = ContextManager()

        # Register observer if provided
        if observer:
            self._context.subscribe(observer)

        # Create HTTP client
        self._http_client = create_http_client(
            timeout_s=config.runtime.timeout_seconds,
            max_attempts=config.runtime.max_http_retries,
            exp_multiplier=config.runtime.retry_backoff_multiplier,
            exp_max_s=config.runtime.retry_backoff_max,
        )

        # Create model from config
        self._model = create_model(config.model, self._http_client)

        # Create pydantic-ai Agent
        # Type ignore for deps_type: pydantic-ai expects type[Any], we allow None
        self._pydantic_agent: Agent[Any, Any] = Agent(
            model=self._model,
            tools=tools or [],
            toolsets=toolsets or [],
            retries=config.runtime.retries,
            deps_type=deps_type,  # type: ignore[arg-type]
            end_strategy=config.runtime.end_strategy,
            history_processors=[],  # Empty for MVP (story 2-1-6b deferred)
        )

        # Register system prompts
        self._register_system_prompts()

    def _register_system_prompts(self) -> None:
        """Register system prompts with the pydantic-ai agent.

        Conditionally registers:
        - config_prompts: From config.system_prompts if provided

        Always registers:
        - current_datetime_prompt: Current date/time in system timezone
        """

        # Register config prompts if provided
        if self._config.system_prompts:

            @self._pydantic_agent.system_prompt(dynamic=True)
            def config_prompts(ctx: Any) -> str:
                return "\n".join(self._config.system_prompts)

        # Always register datetime prompt
        @self._pydantic_agent.system_prompt(dynamic=True)
        def current_datetime_prompt(ctx: Any) -> str:
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
            return f"The current date and time is {now}."

    async def run(self, user_prompt: str, deps: Any = None) -> Any:
        """Execute agent with REACT pattern.

        Runs pydantic-ai agent iteratively, updating context after each
        iteration step.

        Args:
            user_prompt: User message to process
            deps: Optional dependency object (must match deps_type)

        Returns:
            Agent result output (type matches result_type)

        Raises:
            UsageLimitError: If usage limits exceeded
        """
        # Convert usage limits to pydantic-ai format
        pydantic_limits = self._to_pydantic_limits(self._config.usage_limits)

        try:
            # Execute agent with context history
            async with self._pydantic_agent.iter(
                user_prompt=user_prompt,
                deps=deps,
                message_history=self._context.messages,
                usage_limits=pydantic_limits,
            ) as run:
                # Iterate through agent steps (REACT loop)
                async for _ in run:
                    # Add new messages to context
                    # Note: Observers subscribed via subscribe_context() are automatically
                    # notified by ContextManager.add_message()
                    for message in run.new_messages():
                        self._context.add_message(message)

                # Return result if available
                return run.result.output if run.result else None

        except UsageLimitExceeded as e:
            raise UsageLimitError(str(e)) from e

    def run_sync(self, user_prompt: str, deps: Any = None) -> Any:
        """Execute agent synchronously.

        Convenience wrapper around run() for synchronous contexts.

        Args:
            user_prompt: User message to process
            deps: Optional dependency object

        Returns:
            Agent result output

        Raises:
            UsageLimitError: If usage limits exceeded
        """
        return asyncio.run(self.run(user_prompt, deps))

    def _to_pydantic_limits(self, limits: UsageLimits | None) -> PydanticUsageLimits | None:
        """Convert config UsageLimits to pydantic-ai UsageLimits.

        Args:
            limits: Config usage limits or None

        Returns:
            Pydantic-ai usage limits or None
        """
        if limits is None:
            return None

        return PydanticUsageLimits(
            request_limit=limits.request_limit,
            tool_calls_limit=limits.tool_calls_limit,
            input_tokens_limit=limits.input_tokens_limit,
            output_tokens_limit=limits.output_tokens_limit,
            total_tokens_limit=limits.total_tokens_limit,
        )

    # API wrapper methods

    @property
    def context(self) -> ContextManager:
        """Get context manager for message history access.

        Returns:
            Context manager instance
        """
        return self._context

    def subscribe_context(self, observer: ContextObserver) -> None:
        """Subscribe to context change notifications.

        Args:
            observer: Observer implementing ContextObserver protocol
        """
        self._context.subscribe(observer)

    def checkpoint(self, checkpoint_id: str | None = None) -> ContextSnapshot:
        """Create a checkpoint of current context.

        Args:
            checkpoint_id: Optional checkpoint ID (auto-generated if None)

        Returns:
            Created snapshot
        """
        return self._context.checkpoint(checkpoint_id)

    def rewind(self, checkpoint_id: str) -> None:
        """Restore context to a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore

        Raises:
            KeyError: If checkpoint not found
        """
        self._context.rewind(checkpoint_id)

    def system_prompt(self, func: Any) -> Any:
        """Register a custom dynamic system prompt.

        Convenience wrapper around pydantic-ai's @agent.system_prompt(dynamic=True).

        Example:
            >>> @agent.system_prompt
            >>> def my_prompt(ctx):
            ...     return f"Context: {ctx.deps.get_context()}"

        Args:
            func: System prompt function

        Returns:
            Decorated function
        """
        return self._pydantic_agent.system_prompt(dynamic=True)(func)

    def tool(self, func: Any) -> Any:
        """Register a tool function.

        Convenience wrapper around pydantic-ai's @agent.tool().

        Example:
            >>> @agent.tool
            >>> def search(query: str) -> list[str]:
            ...     return search_database(query)

        Args:
            func: Tool function

        Returns:
            Decorated function
        """
        return self._pydantic_agent.tool(func)

    @property
    def pydantic_agent(self) -> Agent[Any, Any]:
        """Access underlying pydantic-ai Agent for advanced usage.

        Use for features not wrapped by ReactAgent:
        - result_validator()
        - on_error()
        - Direct decorator access

        Returns:
            Pydantic-ai Agent instance
        """
        return self._pydantic_agent
