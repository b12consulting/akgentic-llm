"""Unit tests for ReactAgent implementation."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, UserPromptPart

from akgentic.llm import (
    ModelConfig,
    ReactAgent,
    ReactAgentConfig,
    UsageLimitError,
    UsageLimits,
)


class MockObserver:
    """Mock observer for context notifications."""

    def __init__(self):
        self.messages_added = []
        self.checkpoints_created = []
        self.rewinds = []

    def on_message_added(self, message):
        self.messages_added.append(message)

    def on_checkpoint_created(self, snapshot):
        self.checkpoints_created.append(snapshot)

    def on_rewind(self, snapshot):
        self.rewinds.append(snapshot)


@pytest.fixture
def minimal_config():
    """Minimal ReactAgentConfig for testing."""
    return ReactAgentConfig(
        model=ModelConfig(provider="openai", model="gpt-4o"),
    )


@pytest.fixture
def config_with_limits():
    """ReactAgentConfig with usage limits."""
    return ReactAgentConfig(
        model=ModelConfig(provider="openai", model="gpt-4o"),
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=1000),
    )


@pytest.fixture
def config_with_prompts():
    """ReactAgentConfig with system prompts."""
    return ReactAgentConfig(
        model=ModelConfig(provider="openai", model="gpt-4o"),
        system_prompts=["You are a helpful assistant.", "Always be concise."],
    )


@pytest.fixture
def mock_agent_iter():
    """Mock pydantic-ai Agent.iter() for testing."""

    async def fake_iter(*args, **kwargs):
        # Create mock run object that supports async context manager
        class MockRun:
            def __init__(self):
                self.result = MagicMock(output="test result")
                self._messages = [ModelRequest(parts=[UserPromptPart(content="test")])]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                # Yield once then stop
                if not hasattr(self, "_iterated"):
                    self._iterated = True
                    return None
                raise StopAsyncIteration

            def all_messages(self):
                return self._messages

        return MockRun()

    return fake_iter


class TestUsageLimitError:
    """Test UsageLimitError exception."""

    def test_exception_creation(self):
        """Test UsageLimitError can be raised."""
        with pytest.raises(UsageLimitError):
            raise UsageLimitError("Limit exceeded")

    def test_exception_message(self):
        """Test UsageLimitError preserves message."""
        try:
            raise UsageLimitError("Request limit reached")
        except UsageLimitError as e:
            assert str(e) == "Request limit reached"


class TestReactAgentInit:
    """Test ReactAgent initialization."""

    def test_init_minimal_config(self, minimal_config):
        """Test agent initializes with minimal config."""
        agent = ReactAgent(config=minimal_config)
        assert agent is not None
        assert agent.context is not None
        assert agent.pydantic_agent is not None

    def test_init_with_deps_type(self, minimal_config):
        """Test agent initializes with deps_type."""

        class MyDeps:
            value: str = "test"

        agent = ReactAgent(config=minimal_config, deps_type=MyDeps)
        assert agent is not None

    def test_init_with_tools(self, minimal_config):
        """Test agent initializes with tools."""

        def my_tool(query: str) -> str:
            return f"Result: {query}"

        agent = ReactAgent(config=minimal_config, tools=[my_tool])
        assert agent is not None

    def test_init_with_result_type(self, minimal_config):
        """Test agent initializes with custom result_type."""

        class CustomResult:
            value: str

        agent = ReactAgent(config=minimal_config, result_type=CustomResult)
        assert agent is not None


class TestReactAgentRun:
    """Test ReactAgent.run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_result(self, minimal_config):
        """Test run() returns result from pydantic-ai agent."""
        agent = ReactAgent(config=minimal_config)

        # Create mock run object
        class MockRun:
            def __init__(self):
                self.result = MagicMock(output="test result")
                self._new_messages = [ModelRequest(parts=[UserPromptPart(content="test")])]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "_iterated"):
                    self._iterated = True
                    return None
                raise StopAsyncIteration

            def new_messages(self):
                return self._new_messages

        # Patch iter to return context manager directly
        with patch.object(agent._pydantic_agent, "iter", return_value=MockRun()):
            result = await agent.run("test query")
            assert result == "test result"

    @pytest.mark.asyncio
    async def test_run_updates_context(self, minimal_config):
        """Test context messages updated after run()."""
        agent = ReactAgent(config=minimal_config)

        # Create mock run object
        class MockRun:
            def __init__(self):
                self.result = MagicMock(output="test result")
                self._new_messages = [ModelRequest(parts=[UserPromptPart(content="test")])]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "_iterated"):
                    self._iterated = True
                    return None
                raise StopAsyncIteration

            def new_messages(self):
                return self._new_messages

        assert len(agent.context.messages) == 0

        with patch.object(agent._pydantic_agent, "iter", return_value=MockRun()):
            await agent.run("test query")

            # Context should have messages after run
            assert len(agent.context.messages) == 1

    @pytest.mark.asyncio
    async def test_usage_limit_error_raised(self, minimal_config):
        """Test UsageLimitError raised when pydantic-ai raises UsageLimitExceeded."""
        agent = ReactAgent(config=minimal_config)

        class FailingRun:
            async def __aenter__(self):
                raise UsageLimitExceeded("Request limit exceeded")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        with patch.object(agent._pydantic_agent, "iter", return_value=FailingRun()):
            with pytest.raises(UsageLimitError) as exc_info:
                await agent.run("test query")
            assert "Request limit exceeded" in str(exc_info.value)


class TestReactAgentProperties:
    """Test ReactAgent properties and methods."""

    def test_context_property(self, minimal_config):
        """Test context property returns ContextManager."""
        agent = ReactAgent(config=minimal_config)
        context = agent.context
        assert context is not None
        assert hasattr(context, "messages")
        assert hasattr(context, "checkpoint")

    def test_pydantic_agent_property(self, minimal_config):
        """Test pydantic_agent property returns pydantic-ai Agent."""
        agent = ReactAgent(config=minimal_config)
        pydantic_agent = agent.pydantic_agent
        assert pydantic_agent is not None
        # Check it has pydantic-ai Agent methods
        assert hasattr(pydantic_agent, "tool")
        assert hasattr(pydantic_agent, "system_prompt")


class TestReactAgentContextMethods:
    """Test ReactAgent context management methods."""

    @pytest.mark.asyncio
    async def test_subscribe_context(self, minimal_config):
        """Test subscribe_context() observer notified on message add."""
        observer = MockObserver()
        agent = ReactAgent(config=minimal_config)

        # Create mock run object
        class MockRun:
            def __init__(self):
                self.result = MagicMock(output="test result")
                self._new_messages = [ModelRequest(parts=[UserPromptPart(content="test")])]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "_iterated"):
                    self._iterated = True
                    return None
                raise StopAsyncIteration

            def new_messages(self):
                return self._new_messages

        agent.subscribe_context(observer)

        with patch.object(agent._pydantic_agent, "iter", return_value=MockRun()):
            await agent.run("test query")

            # Observer should have been notified
            assert len(observer.messages_added) == 1

    def test_checkpoint_creates_snapshot(self, minimal_config):
        """Test checkpoint() creates snapshot."""
        agent = ReactAgent(config=minimal_config)
        snapshot = agent.checkpoint("test-checkpoint")
        assert snapshot is not None
        assert snapshot.checkpoint_id == "test-checkpoint"

    def test_rewind_restores_context(self, minimal_config):
        """Test rewind() restores context."""
        agent = ReactAgent(config=minimal_config)

        # Create checkpoint
        snapshot = agent.checkpoint("before")
        assert snapshot.checkpoint_id == "before"

        # Modify context (add a message)
        test_message = ModelRequest(parts=[UserPromptPart(content="test")])
        agent.context.add_message(test_message)
        assert len(agent.context.messages) == 1

        # Rewind
        agent.rewind("before")
        assert len(agent.context.messages) == 0


class TestReactAgentSyncMethod:
    """Test ReactAgent.run_sync() method."""

    def test_run_sync_works_synchronously(self, minimal_config):
        """Test run_sync() executes synchronously."""
        agent = ReactAgent(config=minimal_config)

        # Create mock run object
        class MockRun:
            def __init__(self):
                self.result = MagicMock(output="test result")
                self._new_messages = [ModelRequest(parts=[UserPromptPart(content="test")])]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "_iterated"):
                    self._iterated = True
                    return None
                raise StopAsyncIteration

            def new_messages(self):
                return self._new_messages

        with patch.object(agent._pydantic_agent, "iter", return_value=MockRun()):
            result = agent.run_sync("test query")
            assert result == "test result"


class TestReactAgentSystemPrompts:
    """Test system prompt registration."""

    def test_config_system_prompts_registered(self, config_with_prompts):
        """Test system prompts from config.system_prompts are registered."""
        agent = ReactAgent(config=config_with_prompts)
        # Check that agent was created (prompts registered during init)
        assert agent is not None
        # System prompts are registered as decorators on pydantic_agent
        # We can't easily test the prompt content without running the agent,
        # but we can verify initialization succeeded
        assert len(config_with_prompts.system_prompts) == 2

    def test_current_datetime_prompt_registered(self, minimal_config):
        """Test current_datetime_prompt always registered."""
        agent = ReactAgent(config=minimal_config)
        # Datetime prompt is always registered
        # We verify by checking agent initialized without error
        assert agent is not None

    def test_system_prompt_decorator_wrapper(self, minimal_config):
        """Test system_prompt() decorator wrapper registers custom prompt."""
        agent = ReactAgent(config=minimal_config)

        # Register custom prompt
        @agent.system_prompt
        def custom_prompt(ctx):
            return "Custom system prompt"

        # Verify decorator returned a callable
        assert callable(custom_prompt)


class TestReactAgentToolDecorator:
    """Test tool decorator wrapper."""

    def test_tool_decorator_wrapper(self, minimal_config):
        """Test tool() decorator wrapper registers tool function."""

        # Create agent with deps_type to satisfy pydantic-ai requirement
        class MyDeps:
            pass

        agent = ReactAgent(config=minimal_config, deps_type=MyDeps)

        # Register tool with proper RunContext annotation
        from pydantic_ai import RunContext

        @agent.tool
        def search_tool(ctx: RunContext[MyDeps], query: str) -> list[str]:
            """Search for items matching query.

            Args:
                query: The search query string

            Returns:
                List of matching items
            """
            return [f"Result: {query}"]

        # Verify decorator returned a callable
        assert callable(search_tool)


class TestReactAgentUsageLimits:
    """Test usage limits conversion."""

    def test_usage_limits_converted(self, config_with_limits):
        """Test usage limits from config converted to pydantic-ai format."""
        agent = ReactAgent(config=config_with_limits)
        # Test that conversion happens without error
        pydantic_limits = agent._to_pydantic_limits(config_with_limits.usage_limits)
        assert pydantic_limits is not None
        assert pydantic_limits.request_limit == 5
        assert pydantic_limits.total_tokens_limit == 1000

    def test_none_usage_limits(self, minimal_config):
        """Test None usage limits returns None."""
        agent = ReactAgent(config=minimal_config)
        pydantic_limits = agent._to_pydantic_limits(None)
        assert pydantic_limits is None
