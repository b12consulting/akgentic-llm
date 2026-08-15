"""Every Python sample in README.md, transcribed and executed.

The README documented a checkpoint API that does not exist. Its Observer sample
imported ``LlmCheckpointCreatedEvent`` and raised ``ImportError`` as written, and
nothing caught it, because nothing ever ran it. This module is what closes that
gap: each README block is copied in here and executed, so a sample that stops
matching the source turns a test red instead of wasting a reader's afternoon.

**This module never reads README.md.** Asserting on documentation *content* --
grepping the markdown, or checking that a string appears in a docstring -- tests
prose rather than behaviour, and is forbidden (Golden Rule #8). The samples are
protected by *running* them. The cost is that the transcription can drift from
the file; the benefit is that everything transcribed is provably true of the
source, which the markdown alone can never be.

Two rules the samples themselves taught us:

- **Construction alone proves nothing.** Pydantic silently discards unknown
  keyword arguments, so ``ReactAgentConfig(model=...)`` -- where the field is
  ``model_cfg`` -- "succeeds" and yields an agent on a default model. Every
  construction sample below therefore asserts the resulting field *values*.
- **``ReactAgent`` owns an event loop** (``agent.py``: ``__init__`` creates it,
  ``close()`` tears it down). Any sample that builds one closes it, or the suite
  leaks a loop per test.

Zero egress: no sample issues a model request. Provider credentials are faked in
``_fake_provider_env`` purely so the provider factories construct.
"""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import BinaryContent, NativeOutput
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

from akgentic.llm import (
    COMPACTION_STRATEGIES,
    SUMMARY_INSTRUCTIONS,
    AgentUsageLimits,
    CompactionConfig,
    CompactionResult,
    ContextManager,
    HttpClientConfig,
    LlmContextClearedEvent,
    LlmContextCompactedEvent,
    LlmMessageEvent,
    LlmSystemPromptEvent,
    LlmUsageEvent,
    ModelConfig,
    PromptTemplate,
    ReactAgent,
    ReactAgentConfig,
    RuntimeConfig,
    RunUsageLimits,
    SystemPromptPartSnapshot,
    ToolCallEvent,
    ToolReturnEvent,
    UserPrompt,
    aggregate_usage,
    create_compaction,
    current_datetime_prompt,
    get_output_type,
    json_output_reminder_prompt,
)


@pytest.fixture(autouse=True)
def _fake_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every provider factory a key so samples construct without egress.

    The samples name real providers, and ``providers.py`` reads these variables
    when it builds a model. Faking them here rather than relying on the ambient
    environment keeps the suite green on a laptop with no keys and identical to
    CI, which supplies only ``OPENAI_API_KEY``.
    """
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "NVIDIA_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ):
        monkeypatch.setenv(var, "test-key-not-a-real-credential")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")


@pytest.fixture
def agents() -> Any:
    """Build ``ReactAgent``s that are guaranteed to be closed.

    ``ReactAgent.__init__`` creates and owns an asyncio loop; a sample that
    builds an agent and walks away leaks one. Tests append through this factory
    so teardown closes every agent even when the test fails.
    """
    built: list[ReactAgent] = []

    def _make(config: ReactAgentConfig, **kwargs: Any) -> ReactAgent:
        agent = ReactAgent(config=config, **kwargs)
        built.append(agent)
        return agent

    yield _make

    for agent in built:
        agent.close()


def _tool_names(agent: ReactAgent) -> set[str]:
    """Tool names actually registered on the wrapped pydantic-ai agent.

    ``Agent.toolsets`` is public and each function toolset keeps a ``tools``
    mapping. Isolated in one helper so a pydantic-ai reshuffle is a one-line
    repair rather than a hunt; if the attribute ever disappears this returns an
    empty set and the samples go red, which is the honest signal — we would no
    longer be able to prove a documented tool reaches the agent.
    """
    names: set[str] = set()
    for toolset in agent.pydantic_agent.toolsets:
        names.update(getattr(toolset, "tools", {}))
    return names


def _usage_event(run_id: str, model_name: str = "gpt-4o") -> LlmUsageEvent:
    """One usage event, the shape ``ContextManager._emit_usage_event`` produces."""
    return LlmUsageEvent(
        run_id=run_id,
        model_name=model_name,
        provider_name="openai",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        requests=1,
    )


# ---------------------------------------------------------------------------
# Quick Start
# ---------------------------------------------------------------------------


def test_quick_start_sample_builds_the_documented_agent(agents: Any) -> None:
    """§Quick Start — the first block a reader copies.

    ``run_sync`` is not called (it would contact a provider); everything up to
    the request is executed, and the config the agent actually holds is checked
    against the one the sample writes.
    """
    config = ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))

    agent = agents(config)

    # The sample's whole point is that THIS model is the one configured.
    assert agent._config.model_cfg.provider == "openai"
    assert agent._config.model_cfg.model == "gpt-4o"
    assert callable(agent.run_sync)


def test_quick_start_tools_and_output_type_sample(agents: Any) -> None:
    """§Quick Start — with tools and a per-call output type."""

    class Summary(BaseModel):
        title: str
        points: list[str]

    def fetch_data(topic: str) -> str:
        """Retrieve data about a topic."""
        return f"Latest data on {topic}: ..."

    agent = agents(
        ReactAgentConfig(
            model_cfg=ModelConfig(
                provider="anthropic", model="claude-3-5-sonnet-20241022"
            ),
            run_usage_limits=RunUsageLimits(run_request_limit=10, total_tokens_limit=20_000),
        ),
        tools=[fetch_data],
    )

    assert agent._config.model_cfg.provider == "anthropic"
    assert agent._config.run_usage_limits.run_request_limit == 10
    assert agent._config.run_usage_limits.total_tokens_limit == 20_000

    # The sample passes output_type=Summary to run_sync. Issuing the request would
    # contact Anthropic, so what is checked instead is the documented wrapping path
    # (§ReactAgent API: "Both are wrapped with get_output_type()"): a provider with
    # native structured output must yield a NativeOutput wrapper, not the bare type.
    wrapped = get_output_type(agent._config.model_cfg, Summary)
    assert isinstance(wrapped, NativeOutput)

    # The tool the sample passes must reach the wrapped agent, not merely be accepted
    # by the constructor.
    assert "fetch_data" in _tool_names(agent)


# ---------------------------------------------------------------------------
# Configuration — ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_samples() -> None:
    """§ModelConfig — the three documented constructions, values asserted."""
    standard = ModelConfig(provider="openai", model="gpt-4o", temperature=0.7)
    assert standard.temperature == 0.7

    deterministic = ModelConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.0,
        seed=42,
        max_tokens=2000,
    )
    assert (deterministic.temperature, deterministic.seed) == (0.0, 42)
    assert deterministic.max_tokens == 2000

    reasoning = ModelConfig(provider="openai", model="o1", reasoning_effort="high")
    assert reasoning.reasoning_effort == "high"


def test_model_config_table_fields_exist() -> None:
    """§ModelConfig — every row of the field table is a real field with that default.

    The table omitted ``context_length`` and ``fallback_models`` while presenting
    itself as the field list, which is how a reader concludes compaction cannot
    be configured.
    """
    defaults = ModelConfig()
    assert defaults.provider == "openai"
    assert defaults.model == "gpt-5.2"
    assert defaults.temperature is None
    assert defaults.seed is None
    assert defaults.max_tokens is None
    assert defaults.context_length is None
    assert defaults.reasoning_effort is None
    assert defaults.fallback_models == []


# ---------------------------------------------------------------------------
# Configuration — usage limits
# ---------------------------------------------------------------------------


def test_run_usage_limits_samples() -> None:
    """§Usage limits — the run tier's two documented constructions."""
    tight = RunUsageLimits(run_request_limit=10, total_tokens_limit=5_000)
    assert tight.run_request_limit == 10
    assert tight.total_tokens_limit == 5_000

    no_brake = RunUsageLimits(run_request_limit=None)
    assert no_brake.run_request_limit is None


def test_run_usage_limits_table_defaults() -> None:
    """§Usage limits — the run-tier table's defaults, including the 50-request brake."""
    defaults = RunUsageLimits()
    assert defaults.run_request_limit == 50
    assert defaults.tool_calls_limit is None
    assert defaults.input_tokens_limit is None
    assert defaults.output_tokens_limit is None
    assert defaults.total_tokens_limit is None


def test_agent_usage_limits_sample() -> None:
    """§Usage limits — the agent tier's documented construction."""
    limits = AgentUsageLimits(agent_request_limit=100, total_tokens_limit=1_000_000)
    assert limits.agent_request_limit == 100
    assert limits.total_tokens_limit == 1_000_000


def test_agent_usage_limits_table_defaults() -> None:
    """§Usage limits — the agent-tier table's four fields all default to None."""
    defaults = AgentUsageLimits()
    assert defaults.agent_request_limit is None
    assert defaults.input_tokens_limit is None
    assert defaults.output_tokens_limit is None
    assert defaults.total_tokens_limit is None


def test_agent_request_limit_is_enforced_not_merely_declared() -> None:
    """§Usage limits — the agent tier is enforced, which the docs once denied.

    ``__init__.py``'s module docstring called this budget "declared, not yet
    enforced" long after it started refusing runs. A budget of zero runs is
    refused before the model is ever contacted.
    """
    config = ReactAgentConfig(
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        agent_usage_limits=AgentUsageLimits(agent_request_limit=1),
    )
    agent = ReactAgent(config=config)
    try:
        agent._check_and_consume_agent_budget()  # consumes the only unit
        with pytest.raises(Exception, match="agent_request_limit"):
            agent._check_and_consume_agent_budget()
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Configuration — RuntimeConfig and ReactAgentConfig
# ---------------------------------------------------------------------------


def test_runtime_config_table_defaults() -> None:
    """§RuntimeConfig — the documented defaults, including HttpClientConfig's."""
    runtime = RuntimeConfig()
    assert runtime.retries == 3
    assert runtime.end_strategy == "exhaustive"
    assert runtime.parallel_tool_calls is True
    assert isinstance(runtime.http_client_config, HttpClientConfig)

    http = HttpClientConfig()
    assert http.timeout == 120.0
    assert http.max_retries == 5
    assert http.backoff_multiplier == 0.5
    assert http.backoff_max == 60.0


def test_react_agent_config_composition_sample() -> None:
    """§ReactAgentConfig — the full composition block.

    The nested keyword is ``http_client_config``. Written as ``http_client=``
    -- which the docs did -- Pydantic drops it in silence and the sample quietly
    configures nothing, which is exactly what the last assertion here catches.
    """
    config = ReactAgentConfig(
        model_cfg=ModelConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
        ),
        run_usage_limits=RunUsageLimits(
            run_request_limit=10,
            total_tokens_limit=50_000,
        ),
        agent_usage_limits=AgentUsageLimits(
            agent_request_limit=100,
        ),
        runtime_cfg=RuntimeConfig(
            end_strategy="exhaustive",
            http_client_config=HttpClientConfig(timeout=180.0, max_retries=3),
        ),
    )

    assert config.model_cfg.model == "claude-3-5-sonnet-20241022"
    assert config.model_cfg.temperature == 0.7
    assert config.run_usage_limits.run_request_limit == 10
    assert config.agent_usage_limits.agent_request_limit == 100
    assert config.runtime_cfg.end_strategy == "exhaustive"
    assert config.runtime_cfg.http_client_config.timeout == 180.0
    assert config.runtime_cfg.http_client_config.max_retries == 3


def test_the_model_cfg_field_is_named_model_cfg() -> None:
    """The trap behind three false docstrings: ``model=`` is silently discarded.

    Pinned as behaviour rather than as prose. If Pydantic's handling of unknown
    keys ever changes, this test tells us; the docstrings cannot.
    """
    wrong = ReactAgentConfig(model=ModelConfig(provider="anthropic", model="claude-x"))
    assert wrong.model_cfg.provider == "openai"  # the default, NOT what was passed

    right = ReactAgentConfig(model_cfg=ModelConfig(provider="anthropic", model="claude-x"))
    assert right.model_cfg.provider == "anthropic"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_nvidia_provider_samples() -> None:
    """§Providers — the two NVIDIA rows differ on native structured output."""
    from akgentic.llm.config import _supports_native_output

    native = ModelConfig(provider="nvidia", model="openai/gpt-oss-120b")
    assert _supports_native_output(native) is True

    prompt_based = ModelConfig(provider="nvidia", model="meta/llama-3.1-8b-instruct")
    assert _supports_native_output(prompt_based) is False


def test_provider_table_native_output_column() -> None:
    """§Providers — the ✅/❌ column, verified per row against the predicate."""
    from akgentic.llm.config import _supports_native_output

    assert _supports_native_output(ModelConfig(provider="openai", model="gpt-4o"))
    assert _supports_native_output(ModelConfig(provider="azure", model="gpt-4o"))
    assert _supports_native_output(ModelConfig(provider="anthropic", model="claude-x"))
    assert not _supports_native_output(ModelConfig(provider="google-gla", model="gemini-2.0"))
    assert not _supports_native_output(ModelConfig(provider="mistral", model="mistral-large"))


async def test_google_provider_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """§Providers — Google is API-key only; ADC is not consulted.

    The table used to offer ``GOOGLE_APPLICATION_CREDENTIALS``, which reads as
    "an ADC-only deployment works". It does not: the factory raises.
    """
    import httpx

    from akgentic.llm.providers import create_model

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")

    config = ModelConfig(provider="google-gla", model="gemini-2.0-flash")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
            create_model(config, client)


async def test_google_provider_accepts_either_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """§Providers — ``GEMINI_API_KEY`` alone satisfies the requirement."""
    import httpx

    from akgentic.llm.providers import create_model

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    config = ModelConfig(provider="google-gla", model="gemini-2.0-flash")
    async with httpx.AsyncClient() as client:
        assert create_model(config, client) is not None


def test_fallback_chain_sample() -> None:
    """§Fallback chain — the documented chain constructs and stays flat."""
    config = ModelConfig(
        provider="openai",
        model="gpt-5.2",
        fallback_models=[
            ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
            ModelConfig(provider="azure", model="gpt-4o-mini"),
        ],
    )
    assert [m.provider for m in config.fallback_models] == ["anthropic", "azure"]
    assert all(not m.fallback_models for m in config.fallback_models)


def test_fallback_chain_rules_are_enforced_at_construction() -> None:
    """§Fallback chain — flat and homogeneous, both enforced when the config is built."""
    with pytest.raises(ValueError, match="chain is flat"):
        ModelConfig(
            provider="openai",
            model="gpt-4o",
            fallback_models=[
                ModelConfig(
                    provider="anthropic",
                    model="claude-x",
                    fallback_models=[ModelConfig(provider="openai", model="gpt-4o")],
                )
            ],
        )

    with pytest.raises(ValueError, match="supports_native_output"):
        ModelConfig(
            provider="openai",  # native output: True
            model="gpt-4o",
            fallback_models=[ModelConfig(provider="mistral", model="mistral-large")],
        )


# ---------------------------------------------------------------------------
# ReactAgent API — the signature listing, member by member
# ---------------------------------------------------------------------------


def test_react_agent_api_block_members_all_exist() -> None:
    """§ReactAgent API — every member the block lists is real.

    The block is a signature listing, not executable, so it is verified by
    attribute rather than by running it. It previously advertised ``checkpoint``
    and ``rewind``, which never existed on this class.
    """
    for name in (
        "run",
        "run_sync",
        "context",
        "subscribe_context",
        "restore_context",
        "compact",
        "clear_context",
        "system_prompt",
        "tool",
        "aclose",
        "close",
        "pydantic_agent",
    ):
        assert hasattr(ReactAgent, name), f"{name} documented but missing"

    # Documented as gone; must not come back under the old names.
    assert not hasattr(ReactAgent, "checkpoint")
    assert not hasattr(ReactAgent, "rewind")


def test_react_agent_constructor_accepts_every_documented_argument(agents: Any) -> None:
    """§ReactAgent API — the ``__init__`` signature, exercised with all arguments."""

    class _Observer:
        def notify_event(self, event: object) -> None: ...

    def a_tool(x: str) -> str:
        """A tool."""
        return x

    agent = agents(
        ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
        deps_type=None,
        tools=[a_tool],
        toolsets=[],
        result_type=str,
        observer=_Observer(),
        capabilities=[],
        event_loop=None,  # deprecated, accepted and ignored
    )
    assert agent.pydantic_agent is not None


def test_event_loop_argument_is_ignored(agents: Any) -> None:
    """§ReactAgent API — ``event_loop=`` is accepted and ignored, as documented."""
    foreign = asyncio.new_event_loop()
    try:
        agent = agents(
            ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
            event_loop=foreign,
        )
        assert agent._loop is not foreign
    finally:
        foreign.close()


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_process_history_sample(agents: Any) -> None:
    """§Capabilities — the ``ProcessHistory`` example, run as written.

    ``ProcessHistory`` is a pydantic-ai v2 built-in; the import in this sample is
    the one that must not rot when the dependency moves.
    """

    def inject_source_reference(messages):  # noqa: ANN001, ANN202
        """Domain-specific history transformation — not a framework concern."""
        return messages

    agent = agents(
        ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
        capabilities=[ProcessHistory(processor=inject_source_reference)],
    )
    assert agent.pydantic_agent is not None


# ---------------------------------------------------------------------------
# Multimodal prompts
# ---------------------------------------------------------------------------


def test_multimodal_prompt_sample() -> None:
    """§Multimodal Prompts — the documented prompt shape.

    The README opens ``diagram.png``; there is no such file in the repo, so the
    bytes are produced in memory.

    ``isinstance`` on a string literal and on a ``BinaryContent`` the test itself
    just built cannot fail, so it proves nothing. What is checked instead is that
    the keyword names the README uses land in the fields it claims — the same
    silently-discarded-keyword trap that made ``ReactAgentConfig(model=...)``
    configure nothing — and that the pair matches the exported ``UserPrompt``.
    """
    image_bytes = b"\x89PNG\r\n\x1a\n"  # in-memory stand-in for diagram.png

    image = BinaryContent(data=image_bytes, media_type="image/png")
    prompt: UserPrompt = ["Describe what is shown in this architecture diagram.", image]

    assert image.data == image_bytes
    assert image.media_type == "image/png"
    assert image.is_image
    # UserPrompt = str | list[str | BinaryContent]; the list form is the one here.
    assert prompt[1] is image


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def test_context_turns_sample(agents: Any) -> None:
    """§Context Management — history persists across turns.

    The turns themselves would contact a provider, so the messages are appended
    directly; what the sample claims is that ``agent.context.messages`` carries
    them forward, and that is what is checked.
    """
    agent = agents(ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")))

    agent.context.add_message(ModelRequest(parts=[UserPromptPart(content="Start the analysis.")]))
    agent.context.add_message(
        ModelRequest(parts=[UserPromptPart(content="Now summarise your findings.")])
    )

    assert len(agent.context.messages) == 2


def test_context_manager_sample_methods_all_exist_and_run() -> None:
    """§ContextManager — every method the block names, executed.

    The block used to list ``checkpoint``, ``rewind``, ``get_checkpoint`` and
    ``list_checkpoints``. None of them exist; each line here is a real call.
    """
    ctx = ContextManager(max_messages=20)

    msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
    ctx.add_message(msg)
    assert ctx.messages == [msg]
    assert ctx.last_input_tokens is None

    class _Observer:
        def __init__(self) -> None:
            self.events: list[object] = []

        def notify_event(self, event: object) -> None:
            self.events.append(event)

    observer = _Observer()
    ctx.subscribe(observer)
    ctx.unsubscribe(observer)

    # Operator actions: appended directly once history is non-empty, so nothing
    # is left buffered for ReactAgent.run to fold into the prompt.
    ctx.record_operator_action("operator did a thing")
    assert ctx.drain_pending_operator_actions() == []
    assert len(ctx.messages) == 2

    ctx.record_system_prompt("run-1")  # no system parts yet — a documented no-op
    ctx.seed_system_prompt_hash("deadbeef")

    event = LlmContextCompactedEvent(
        run_id=None,
        strategy_id="summarize",
        summary="a summary",
        replaced_message_count=2,
        summarizer_prompt_version="v1",
        tokens_before=None,
        tokens_after=None,
    )
    # The static fold is pure: it returns a new list and leaves the input alone.
    # "summarize" is a full fold — no system parts here, so only the summary survives.
    folded = ContextManager.fold_compaction(ctx.messages, event)
    assert len(folded) == 1
    assert len(ctx.messages) == 2  # input untouched

    ctx.compact(event)  # same fold, applied in place, plus the event
    assert len(ctx.messages) == 1  # the synthetic summary replaced the history

    assert ctx.clear_context() == 1  # returns the number of messages removed
    assert ctx.messages == []

    ctx.restore([msg])
    assert ctx.messages == [msg]
    ctx.clear()
    assert ctx.messages == []


def test_context_manager_has_no_checkpoint_surface() -> None:
    """§ContextManager — the removed API must not return under any name."""
    for name in ("checkpoint", "rewind", "get_checkpoint", "list_checkpoints"):
        assert not hasattr(ContextManager, name), f"{name} is documented nowhere and must not exist"


# ---------------------------------------------------------------------------
# Observer pattern
# ---------------------------------------------------------------------------


def test_observer_sample_imports_and_dispatches() -> None:
    """§Observer Pattern — the canonical case this whole module exists for.

    As written before this story the block imported ``LlmCheckpointCreatedEvent``
    and raised ``ImportError`` on line one. Every branch below is driven with a
    real event so an event that loses a field fails here.
    """
    seen: list[str] = []

    class MyObserver:
        def notify_event(self, event: object) -> None:
            if isinstance(event, ToolCallEvent):
                seen.append(f"Tool called: {event.tool_name} ({event.tool_call_id})")
            elif isinstance(event, ToolReturnEvent):
                status = "success" if event.success else "error"
                seen.append(f"Tool returned: {event.tool_name} ({status})")
            elif isinstance(event, LlmUsageEvent):
                seen.append(
                    f"Usage: {event.model_name} — "
                    f"{event.input_tokens}in/{event.output_tokens}out"
                )
            elif isinstance(event, LlmSystemPromptEvent):
                seen.append(f"System prompt for run {event.run_id} ({event.content_hash[:8]}):")
                for part in event.parts:
                    seen.append(f"  [{part.dynamic_ref or 'static'}] {part.content}")
            elif isinstance(event, LlmMessageEvent):
                seen.append(f"New message: {event.message}")
            elif isinstance(event, LlmContextCompactedEvent):
                seen.append(
                    f"Compacted {event.replaced_message_count} msg(s) "
                    f"via '{event.strategy_id}'"
                )
            elif isinstance(event, LlmContextClearedEvent):
                seen.append(f"Cleared {event.cleared_message_count} msg(s)")

    observer = MyObserver()
    observer.notify_event(
        ToolCallEvent(run_id="r", tool_name="t", tool_call_id="c", arguments="{}")
    )
    observer.notify_event(
        ToolReturnEvent(run_id="r", tool_name="t", tool_call_id="c", success=True)
    )
    observer.notify_event(_usage_event("r"))
    observer.notify_event(
        LlmSystemPromptEvent(
            "r", (SystemPromptPartSnapshot(dynamic_ref=None, content="be helpful"),), "abcdef1234"
        )
    )
    observer.notify_event(
        LlmMessageEvent(message=ModelRequest(parts=[UserPromptPart(content="hi")]))
    )
    observer.notify_event(
        LlmContextCompactedEvent(
            run_id="r",
            strategy_id="summarize",
            summary="s",
            replaced_message_count=3,
            summarizer_prompt_version="v1",
            tokens_before=10,
            tokens_after=5,
        )
    )
    observer.notify_event(LlmContextClearedEvent(run_id="r", cleared_message_count=2))

    assert len(seen) == 8  # 7 events, the system-prompt one contributing two lines


def test_observer_is_accepted_at_construction_and_via_subscribe(agents: Any) -> None:
    """§Observer Pattern — both documented wiring routes actually deliver events.

    Constructing without raising proves nothing here: this test was written that
    way first, and deleting ``self._context.subscribe(observer)`` from
    ``ReactAgent.__init__`` left it green. Each route is therefore driven with a
    real message and the observer must have been notified.
    """

    class MyObserver:
        def __init__(self) -> None:
            self.events: list[object] = []

        def notify_event(self, event: object) -> None:
            self.events.append(event)

    config = ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    message = ModelRequest(parts=[UserPromptPart(content="hi")])

    # Route 1 — observer= at construction.
    at_construction = MyObserver()
    agents(config, observer=at_construction).context.add_message(message)
    assert any(isinstance(e, LlmMessageEvent) for e in at_construction.events)

    # Route 2 — subscribe_context() afterwards.
    via_subscribe = MyObserver()
    later = agents(config)
    later.subscribe_context(via_subscribe)
    later.context.add_message(message)
    assert any(isinstance(e, LlmMessageEvent) for e in via_subscribe.events)


def test_system_prompt_tracer_sample() -> None:
    """§System Prompt Rendering Events — the tracer, driven by a real event."""
    lines: list[str] = []

    class SystemPromptTracer:
        def notify_event(self, event: object) -> None:
            if not isinstance(event, LlmSystemPromptEvent):
                return
            lines.append(f"System prompt @ run {event.run_id} (hash {event.content_hash[:8]})")
            for snapshot in event.parts:
                label = snapshot.dynamic_ref or "static"
                lines.append(f"  [{label}] {snapshot.content}")

    SystemPromptTracer().notify_event(
        LlmSystemPromptEvent(
            "run-7",
            (
                SystemPromptPartSnapshot(dynamic_ref=None, content="static text"),
                SystemPromptPartSnapshot(dynamic_ref="the_date", content="2026-08-15"),
            ),
            "0123456789abcdef",
        )
    )

    assert lines[0] == "System prompt @ run run-7 (hash 01234567)"
    assert lines[1] == "  [static] static text"
    assert lines[2] == "  [the_date] 2026-08-15"


def test_system_prompt_event_is_emitted_and_deduped() -> None:
    """§System Prompt Rendering Events — emitted on change, silent when unchanged."""
    emitted: list[object] = []

    class _Observer:
        def notify_event(self, event: object) -> None:
            if isinstance(event, LlmSystemPromptEvent):
                emitted.append(event)

    ctx = ContextManager()
    ctx.subscribe(_Observer())
    ctx.add_message(ModelRequest(parts=[SystemPromptPart(content="be helpful")]))

    ctx.record_system_prompt("run-1")
    assert len(emitted) == 1  # the None → hash transition

    ctx.record_system_prompt("run-2")
    assert len(emitted) == 1  # unchanged rendering emits nothing


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_compaction_tracer_sample(agents: Any) -> None:
    """§Compaction & Clear Events — the tracer, and both wiring routes."""
    lines: list[str] = []

    class CompactionTracer:
        def notify_event(self, event: object) -> None:
            if isinstance(event, LlmContextCompactedEvent):
                lines.append(
                    f"compacted @ run {event.run_id}: folded "
                    f"{event.replaced_message_count} msg(s) "
                    f"via '{event.strategy_id}' "
                    f"({event.tokens_before} → {event.tokens_after} tok est.)"
                )
                lines.append(f"  summary: {event.summary[:120]}…")
            elif isinstance(event, LlmContextClearedEvent):
                lines.append(
                    f"cleared @ run {event.run_id}: dropped "
                    f"{event.cleared_message_count} msg(s)"
                )

    tracer = CompactionTracer()
    tracer.notify_event(
        LlmContextCompactedEvent(
            run_id="r1",
            strategy_id="summarize",
            summary="the summary",
            replaced_message_count=4,
            summarizer_prompt_version="v1",
            tokens_before=900,
            tokens_after=120,
        )
    )
    tracer.notify_event(LlmContextClearedEvent(run_id="r1", cleared_message_count=6))

    assert "folded 4 msg(s)" in lines[0]
    assert "900 → 120 tok est." in lines[0]
    assert lines[2] == "cleared @ run r1: dropped 6 msg(s)"

    # Both documented wiring routes, each proven by an event actually arriving —
    # a tracer that is merely accepted by the constructor traces nothing.
    config = ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    agent = agents(config, observer=tracer)
    before = len(lines)
    agent.context.clear_context()
    assert len(lines) == before + 1

    subscribed = agents(config)
    subscribed.subscribe_context(tracer)
    subscribed.context.clear_context()
    assert len(lines) == before + 2


def test_compaction_config_sample() -> None:
    """§Compaction Strategies — the documented ``CompactionConfig``, values asserted."""
    cfg = CompactionConfig(
        strategy="summarize",
        auto_trigger=True,
        trigger_ratio=0.85,
        keep_recent_messages=4,
        summary_target_tokens=2000,
        summarizer_prompt_version="v1",
    )
    assert cfg.strategy == "summarize"
    assert cfg.auto_trigger is True
    assert cfg.trigger_ratio == 0.85
    assert cfg.keep_recent_messages == 4
    assert cfg.summary_target_tokens == 2000
    assert cfg.summarizer_prompt_version == "v1"
    assert cfg.summary_model_cfg is None


def test_builtin_strategy_ids_resolve() -> None:
    """§Compaction Strategies — the three built-in ids in the table are registered."""
    for strategy_id in ("summarize", "sliding_window", "none"):
        assert strategy_id in COMPACTION_STRATEGIES


async def test_custom_compaction_strategy_sample() -> None:
    """§Compaction Strategies — the open-extension registry, both routes."""

    class KeepLastOnly:
        async def compact(self, messages):  # noqa: ANN001, ANN202
            return CompactionResult(
                summary="", replaced_message_count=max(0, len(messages) - 1)
            )

    model_cfg = ModelConfig(provider="openai", model="gpt-4o")
    original = dict(COMPACTION_STRATEGIES)
    try:
        # (a) register a factory under a short id. ``strategy`` is a plain str field
        # with no validator, so reading it back proves nothing; what the README
        # promises is that the FRAMEWORK resolves through the registry.
        COMPACTION_STRATEGIES["keep_last"] = lambda cfg, model_cfg, http_client: KeepLastOnly()
        cfg = CompactionConfig(strategy="keep_last")
        assert isinstance(create_compaction(cfg, model_cfg), KeepLastOnly)

        # The registered strategy really is a CompactionStrategy.
        result = await KeepLastOnly().compact([1, 2, 3])
        assert result.replaced_message_count == 2

        # (b) ...or point strategy at a dotted FQCN — no registration needed. That
        # branch is taken only when the id contains a dot: an importable-looking id
        # goes to importlib, while a bare unknown id is rejected outright.
        cfg = CompactionConfig(strategy="my_package.compaction.KeepLastOnly")
        with pytest.raises(ModuleNotFoundError):
            create_compaction(cfg, model_cfg)
        with pytest.raises(ValueError, match="Unknown compaction strategy"):
            create_compaction(CompactionConfig(strategy="not_registered"), model_cfg)
    finally:
        COMPACTION_STRATEGIES.clear()
        COMPACTION_STRATEGIES.update(original)


def test_summary_instructions_override_sample() -> None:
    """§Overriding the Summarizer Prompt — both documented override routes.

    Asserting that the key you just wrote is back in the dict cannot fail. The
    README's actual promise is that the framework reaches the registry *through*
    ``CompactionConfig.summarizer_prompt_version``, so that is what is pinned —
    including the part a reader cannot see: that "v1" is the id a default config
    asks for, which is the only reason overriding "v1" reaches every agent.
    """
    legal = "You are a summarizer for legal documents. Preserve …"
    original = dict(SUMMARY_INSTRUCTIONS)
    try:
        # (a) replace the default in place — every "v1" agent picks it up
        SUMMARY_INSTRUCTIONS["v1"] = legal
        assert CompactionConfig().summarizer_prompt_version == "v1"
        assert SUMMARY_INSTRUCTIONS[CompactionConfig().summarizer_prompt_version] is legal

        # (b) register a named variant and select it per agent
        SUMMARY_INSTRUCTIONS["legal"] = legal
        cfg = CompactionConfig(strategy="summarize", summarizer_prompt_version="legal")
        assert SUMMARY_INSTRUCTIONS[cfg.summarizer_prompt_version] is legal
    finally:
        SUMMARY_INSTRUCTIONS.clear()
        SUMMARY_INSTRUCTIONS.update(original)


# ---------------------------------------------------------------------------
# Cost tracking and aggregation
# ---------------------------------------------------------------------------


def test_aggregate_usage_sample() -> None:
    """§Aggregation — the documented aggregation block, executed end to end.

    Every attribute the sample formats is read here, so a rename on
    ``AgentUsageSummary`` / ``ModelUsage`` / ``RunUsageSummary`` breaks this.
    """
    events: list[LlmUsageEvent] = [
        _usage_event("run-1", "gpt-4o"),
        _usage_event("run-2", "gpt-4o"),
    ]

    summary = aggregate_usage(events)
    # The README prints f"Total cost: ${summary.total_cost_usd:.4f}". Asserting that
    # THAT string starts with "Total cost: $" is a tautology — the needle is the
    # f-string's own literal prefix. The falsifiable claim is that the attribute
    # exists and is a real number; the amount itself is never pinned, because
    # genai-prices ships the price table and it moves (Golden Rule #13).
    assert isinstance(summary.total_cost_usd, float)
    assert summary.total_cost_usd >= 0.0
    assert summary.total_input_tokens == 200
    assert summary.total_output_tokens == 100
    assert set(summary.by_model) == {"gpt-4o"}
    for model_name, usage in summary.by_model.items():
        assert model_name == "gpt-4o"
        assert usage.estimated_cost_usd >= 0.0

    summary = aggregate_usage(events, by_run=True)
    assert [run.run_id for run in summary.runs] == ["run-1", "run-2"]
    for run in summary.runs:
        assert run.total_cost_usd >= 0.0
        assert run.total_input_tokens == 100


def test_aggregate_usage_runs_are_empty_without_by_run() -> None:
    """§Aggregation — ``runs`` is populated only when ``by_run=True``."""
    assert aggregate_usage([_usage_event("run-1")]).runs == []


def test_unpriced_model_aggregates_tokens_at_zero_cost() -> None:
    """§Pricing — an unmatched ``model_ref`` maps to 0.0 with tokens still counted."""
    summary = aggregate_usage([_usage_event("run-1", "not-a-real-model-xyz")])
    assert summary.total_input_tokens == 100
    assert summary.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_prompt_template_sample() -> None:
    """§PromptTemplate — the rendering the README prints, asserted exactly."""
    tpl = PromptTemplate(
        template="You are {role}.\n\nInstructions: {instructions}",
        params={"role": "the Librarian", "instructions": "Extract structured data."},
    )
    assert tpl.render() == "You are the Librarian.\n\nInstructions: Extract structured data."


def test_dynamic_system_prompts_sample(agents: Any) -> None:
    """§Dynamic System Prompts — both built-ins and the decorator form."""

    def get_current_workspace() -> str:
        return "/srv/workspace"

    agent = agents(
        ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o"))
    )

    # Built-in utilities
    agent.system_prompt(current_datetime_prompt)
    agent.system_prompt(json_output_reminder_prompt)

    # Custom prompt
    @agent.system_prompt
    def workspace_context(ctx: Any) -> str:
        return f"Working directory: {get_current_workspace()}"

    assert current_datetime_prompt(None).startswith("The current date and time is ")
    assert "ONLY a valid JSON object" in json_output_reminder_prompt(None)
    assert workspace_context(None) == "Working directory: /srv/workspace"
