# akgentic-llm

[![CI](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/dd80a44fe9e2e27b46f7f3431e19202f/raw/coverage.json)](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml)

LLM integration layer for the [Akgentic](https://github.com/b12consulting/akgentic-quick-start)
multi-agent framework. Wraps pydantic-ai's REACT execution loop with persistent context
management, production HTTP retry logic, and a clean provider abstraction — letting agents
call any LLM without coupling to a specific vendor or framework primitive.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [ModelConfig](#modelconfig)
  - [Usage limits](#usage-limits)
  - [RuntimeConfig](#runtimeconfig)
  - [ReactAgentConfig](#reactagentconfig)
- [Providers](#providers)
- [ReactAgent API](#reactagent-api)
- [Capabilities](#capabilities)
- [Multimodal Prompts](#multimodal-prompts)
- [Context Management](#context-management)
  - [ContextManager](#contextmanager)
  - [Observer Pattern](#observer-pattern)
  - [Tool Event Observability](#tool-event-observability)
  - [System Prompt Rendering Events](#system-prompt-rendering-events)
- [Context Compaction](#context-compaction)
  - [Compaction & Clear Events](#compaction--clear-events)
  - [Compaction Strategies](#compaction-strategies)
  - [Overriding the Summarizer Prompt](#overriding-the-summarizer-prompt)
- [Cost Tracking and Aggregation](#cost-tracking-and-aggregation)
- [Prompts](#prompts)
- [Development](#development)
- [License](#license)

## Overview

`akgentic-llm` is the LLM execution layer between agent logic and LLM providers. It provides:

- **ReactAgent** — a thin wrapper around pydantic-ai's `Agent.iter()` that persists message
  history across calls, deduplicates messages across tool-call iterations, and translates
  pydantic-ai's `UsageLimitExceeded` into a framework-local `UsageLimitError`
- **Provider abstraction** — `create_model()` dispatches to one of six provider factories
  (OpenAI, Azure, Anthropic, Google, Mistral, NVIDIA), wrapping the result in pydantic-ai's
  `FallbackModel` when `ModelConfig.fallback_models` is non-empty; `get_output_type()` wraps
  output types with `NativeOutput` for providers that support structured output, falls back to
  prompt-based extraction for those that don't
- **HTTP retry** — `create_http_client()` configures `AsyncTenacityTransport` with exponential
  backoff, jitter, and `Retry-After` header support; fast-fails on 4xx (except 429)
- **Context management** — `ContextManager` tracks message history across multiple `run()` calls,
  folds it into a summary on compaction or drops it outright on clear, and applies a sliding
  window (system messages always preserved) when a message cap is configured
- **Prompt utilities** — `PromptTemplate` for config-time `{placeholder}` rendering;
  `current_datetime_prompt` and `json_output_reminder_prompt` as ready-made dynamic prompts
- **Multimodal** — `UserPrompt = str | list[str | BinaryContent]`; exported so `akgentic-agent`
  can annotate its own `act()` signature without importing pydantic-ai directly

```
ReactAgent
  │
  ├── run(user_prompt: UserPrompt)           # str | list[str | BinaryContent]
  │     │
  │     ├── pydantic_agent.iter(            # pydantic-ai REACT loop
  │     │       user_prompt,
  │     │       message_history=context.messages,
  │     │       output_type=get_output_type(model_cfg, output_type),
  │     │   )
  │     │     │
  │     │     └── for each step:
  │     │           context.add_message()   # persists + notifies observers
  │     │
  │     └── return run.result.output
  │
  ├── context: ContextManager               # persistent message history
  ├── compact() / clear_context()           # fold history into a summary, or drop it
  └── system_prompt(func)                   # register dynamic system prompt
```

**Runtime dependencies:** `pydantic-ai[mistral]>=2,<3`, `genai-prices>=0.1.0`, `pydantic>=2.0.0`,
`httpx>=0.27.0`, `tenacity>=8.0.0`, `pyyaml>=6.0`. An optional `loadtest` extra pulls in the
token-free mock agent's own `pyyaml` requirement.

**Module boundary:** `akgentic-llm` MUST NOT import from `akgentic-core`, `akgentic-tool`, or
`akgentic-agent`.

## Installation

### Workspace Installation (Recommended)

```bash
git clone git@github.com:b12consulting/akgentic-quick-start.git
cd akgentic-quick-start
git submodule update --init --recursive

uv venv && source .venv/bin/activate
uv sync --all-packages --all-extras
```

### Standalone

```bash
cd packages/akgentic-llm
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

```python
from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig

config = ReactAgentConfig(
    model_cfg=ModelConfig(provider="openai", model="gpt-4o")
)

agent = ReactAgent(config=config)
result = agent.run_sync("Summarise the key priorities for next sprint.")
print(result)
```

With tools and a per-call output type:

```python
from pydantic import BaseModel
from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig, RunUsageLimits

class Summary(BaseModel):
    title: str
    points: list[str]

def fetch_data(topic: str) -> str:
    """Retrieve data about a topic."""
    return f"Latest data on {topic}: ..."

agent = ReactAgent(
    config=ReactAgentConfig(
        model_cfg=ModelConfig(provider="anthropic", model="claude-3-5-sonnet-20241022"),
        run_usage_limits=RunUsageLimits(run_request_limit=10, total_tokens_limit=20_000),
    ),
    tools=[fetch_data],
)

result = agent.run_sync("Summarise AI trends", output_type=Summary)
print(result.title, result.points)
```

## Configuration

### ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `Literal[...]` | `"openai"` | LLM provider |
| `model` | `str` | `"gpt-5.2"` | Model identifier (provider-specific) |
| `temperature` | `float \| None` | `None` | 0.0–2.0; `None` = provider default |
| `seed` | `int \| None` | `None` | Reproducible outputs (not all providers) |
| `max_tokens` | `int \| None` | `None` | Max response tokens; `None` = provider max |
| `context_length` | `int \| None` | `None` | Model context window; the budget that auto-triggers compaction. `None` = compaction off. Distinct from `max_tokens`, which caps output |
| `reasoning_effort` | `Literal["low","medium","high"] \| None` | `None` | For o1/o3-style models only |
| `fallback_models` | `list[ModelConfig]` | `[]` | Models tried in declaration order after this one on API failure — see [Fallback chain](#fallback-chain) |

```python
from akgentic.llm import ModelConfig

# Standard chat model
ModelConfig(provider="openai", model="gpt-4o", temperature=0.7)

# Deterministic with token cap
ModelConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
            temperature=0.0, seed=42, max_tokens=2000)

# Reasoning model
ModelConfig(provider="openai", model="o1", reasoning_effort="high")
```

### Usage limits

Budgets come in two tiers, carried by two separate `ReactAgentConfig` fields, so a limit
meaning "per `run()` call" can never be mistaken for one meaning "over this agent's
lifetime". Both share a token-only base (`TokenUsageLimits`, internal).

#### RunUsageLimits — `ReactAgentConfig.run_usage_limits`

Cumulative across all requests in a **single `run()` call**, and reset on the next one.
Enforced by pydantic-ai; breaching any limit raises `UsageLimitError`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_request_limit` | `int \| None` | `50` | Max LLM API requests per run — acts as a safety brake |
| `tool_calls_limit` | `int \| None` | `None` | Max tool invocations per run |
| `input_tokens_limit` | `int \| None` | `None` | Max cumulative input tokens |
| `output_tokens_limit` | `int \| None` | `None` | Max cumulative output tokens |
| `total_tokens_limit` | `int \| None` | `None` | Max cumulative total tokens |

```python
from akgentic.llm import RunUsageLimits

RunUsageLimits(run_request_limit=10, total_tokens_limit=5_000)  # tight budget
RunUsageLimits(run_request_limit=None)                          # no safety brake
```

> **A tool retry can cost you a request.** Under `end_strategy="exhaustive"` — the default —
> pydantic-ai v2 lets a failing tool call suppress an already-successful output and continue
> the run for another model turn (see [RuntimeConfig](#runtimeconfig)). That forced turn is
> charged to this tier: it consumes one `run_request_limit` unit and its tokens count toward
> `total_tokens_limit`. A run that completed on pydantic-ai v1 can therefore raise
> `UsageLimitError` on v2 without the prompt or the tools having changed.

#### AgentUsageLimits — `ReactAgentConfig.agent_usage_limits`

Spans **every run the agent performs**, not one call.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_request_limit` | `int \| None` | `None` | Max `run()` calls over the agent's lifetime |
| `input_tokens_limit` | `int \| None` | `None` | Max input tokens over the agent's lifetime |
| `output_tokens_limit` | `int \| None` | `None` | Max output tokens over the agent's lifetime |
| `total_tokens_limit` | `int \| None` | `None` | Max total tokens over the agent's lifetime |

Both halves are checked **before** each `run()` executes — tokens first, so a token
refusal costs no run budget — against counters the agent accumulates over its lifetime and
recomputes from persisted usage events on restore.

`agent_request_limit`: once the agent has used its budget, every further call raises
`UsageLimitError` — the same class a run-tier breach raises — with a message of the form
`Exceeded the agent_request_limit of 100 (run_count=100)`.

Four consequences worth knowing before you set it:

- **A run that fails still counts.** The budget is consumed before the call executes, not
  after it returns — including when the call ends in a *run-tier* `UsageLimitError`. An
  agent stuck in a failing loop therefore still runs out of lifetime budget, which is the
  point: both limits mean "this agent is burning too many turns".
- **It counts runs *consumed*, never runs attempted.** A rejected call consumes nothing,
  so repeated rejections leave the count — and the error message — unchanged.
- **The counter is in memory, not persisted** — but resuming does not reset it.
  Nothing is written to a state snapshot; instead `restore_context()` recomputes the count
  from the agent's persisted usage events, grouped by run. Only a genuinely new agent
  starts with a full budget.
- **A run that never reached the model is invisible after a resume.** It emitted no usage
  event, so replay cannot see it. It counted while the agent was live; after a restore the
  count reflects the runs that actually reached the model. Deliberate — a run that produced
  nothing consumed nothing.

The three token limits bound the agent's **lifetime** spend, summed across every run.
Breaching one raises `UsageLimitError` with pydantic-ai's own message text — e.g.
`Exceeded the total_tokens_limit of 1000000 (total_tokens=1000420)` — the same shape a
run-tier breach produces, so nothing downstream has to parse text to tell the tiers apart.

Two consequences here too:

- **A run may overshoot the budget.** A run's token cost is unknown until it finishes, so
  the limit governs where a run may *start*, not where it may end. The run that crosses the
  line completes and returns normally; the next one is refused. Set the limit below the
  spend you actually want to cap if the last run could be expensive.
- **Resuming does not reset it**, for the same reason the run counter survives:
  `restore_context()` sums the agent's persisted usage events. Unlike the counter, tokens
  sum over *events* — a run with three model round-trips counts once but spends three times.

```python
from akgentic.llm import AgentUsageLimits

AgentUsageLimits(agent_request_limit=100, total_tokens_limit=1_000_000)
```

Note that only `run_usage_limits` participates in the compaction-threshold check
(see [Context compaction](#context-compaction)). The check is deliberately not widened to
the agent tier, so an `agent_usage_limits` token limit below the compaction threshold still
constructs — but it does make the auto-trigger unreachable at runtime, because the agent
refuses the run before compaction can fire.

#### Migrating from the pre-split surface

> **Deprecated in 1.7.0. Still shipped — removal is not scheduled for a named release.**
> The pre-split `UsageLimits` class and the `ReactAgentConfig(usage_limits=...)` keyword still
> work and still carry your values through to `run_usage_limits`, but every use emits a
> `DeprecationWarning`. The 2.0.0 major bump was driven by the move to pydantic-ai v2, not by
> this deprecation; the shim shipped through it unchanged.

| Before | After |
|--------|-------|
| `UsageLimits(request_limit=10)` | `RunUsageLimits(run_request_limit=10)` |
| `limits.request_limit` | `limits.run_request_limit` |
| `ReactAgentConfig(usage_limits=...)` | `ReactAgentConfig(run_usage_limits=...)` |
| `config.usage_limits` | `config.run_usage_limits` |

Three things the shim deliberately does **not** do:

- **Passing both names raises `ValueError`.** `ReactAgentConfig(usage_limits=a, run_usage_limits=b)`
  is rejected rather than resolved, because which one won would otherwise depend on the
  order you wrote them in. The same applies to `UsageLimits(request_limit=..., run_request_limit=...)`.
- **Serialization keys are not preserved.** `model_dump()` emits `run_usage_limits` and
  `run_request_limit`. Code that round-trips config through JSON and keys off the old
  names must be updated now; only the constructor keyword and the attribute read are shimmed.
- **Assignment is not shimmed.** `config.usage_limits = ...` raises — the deprecated names
  are read-only views over the real fields. Assign to `run_usage_limits` instead.

### RuntimeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `retries` | `int` | `3` | Retry attempts for tool failures and output validation errors |
| `end_strategy` | `Literal["early","exhaustive"]` | `"exhaustive"` | Tool execution termination |
| `parallel_tool_calls` | `bool` | `True` | Accepted and validated, but read by nothing — see the note below |
| `http_client_config` | `HttpClientConfig` | `HttpClientConfig()` | HTTP timeout and retry tuning |

**End strategies:**
- `"early"` — stops after the first successful result (fast path)
- `"exhaustive"` — runs all tool calls even when a result is available (complete data gathering)

Under pydantic-ai v2, `"exhaustive"` also carries a **retry-wins** rule: when a function tool
called in the same round as an already-successful output call raises `ModelRetry` — or fails
argument validation — the output is **suppressed** and the run continues for another model turn
instead of ending there. pydantic-ai 1.107 had no such rule; an already-successful output always
won. The forced extra turn is charged to `run_usage_limits` (`run_request_limit`,
`total_tokens_limit`), so a run that finished cleanly on v1 can raise `UsageLimitError` on v2 if
that turn pushes it past a run-tier ceiling. See [Usage limits](#usage-limits).

> **Note: `parallel_tool_calls` currently reaches no model.** `ReactAgent.__init__` reads only
> `retries`, `end_strategy` and `http_client_config` off `runtime_cfg`, and never passes a
> `parallel_tool_calls` model setting. The one function that emits that setting,
> `create_model_settings()`, derives it from `ModelConfig` alone — it never sees a
> `RuntimeConfig` — and has no call site in this package: it is an exported helper for callers
> who build their own model, not part of `ReactAgent`'s construction path. Setting this field
> changes nothing about how `ReactAgent` runs.

`HttpClientConfig` fields: `timeout=120.0`, `max_retries=5`, `backoff_multiplier=0.5`,
`backoff_max=60.0` — all configurable.

### ReactAgentConfig

Composes all three layers:

```python
from akgentic.llm import (
    ReactAgentConfig, ModelConfig, RunUsageLimits, AgentUsageLimits,
    RuntimeConfig, HttpClientConfig,
)

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
        agent_request_limit=100,  # max run() calls over this agent's lifetime
    ),
    runtime_cfg=RuntimeConfig(
        end_strategy="exhaustive",
        http_client_config=HttpClientConfig(timeout=180.0, max_retries=3),
    ),
)
```

## Providers

| Provider | `ModelConfig.provider` | Auth env var(s) | Native structured output |
|----------|------------------------|-----------------|--------------------------|
| OpenAI | `"openai"` | `OPENAI_API_KEY` | ✅ |
| Azure OpenAI | `"azure"` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | ✅ |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | ✅ |
| NVIDIA NIM (openai/* models) | `"nvidia"` | `OPENAI_API_KEY` | ✅ |
| NVIDIA NIM (other models) | `"nvidia"` | `OPENAI_API_KEY` | ❌ |
| Google Gemini | `"google-gla"` | `GOOGLE_API_KEY` **or** `GEMINI_API_KEY` (one is mandatory) | ❌ |
| Mistral AI | `"mistral"` | `MISTRAL_API_KEY` | ❌ |

Providers without native structured output use pydantic-ai's prompt-based extraction fallback.

> **NVIDIA reads `OPENAI_API_KEY`, and a missing key fails late.** `_create_nvidia_model` builds
> `OpenAIProvider(base_url=..., http_client=...)` and passes **no** `api_key`, so pydantic-ai's
> `OpenAIProvider` falls back to `OPENAI_API_KEY`. There is no `NVIDIA_API_KEY` lookup anywhere
> in this package. Because a `base_url` **is** supplied, a missing key does not raise at
> construction — pydantic-ai substitutes the placeholder key `'api-key-not-set'` and the failure
> surfaces as a **401 at request time**, not as a configuration error. The endpoint comes from
> `NVIDIA_BASE_URL`, defaulting to `https://integrate.api.nvidia.com/v1`.

> **Google is API-key only.** The provider factory reads `GOOGLE_API_KEY`, falling back to
> `GEMINI_API_KEY`, and raises `ValueError` when neither is set. Application Default
> Credentials are not consulted, so an ADC-only deployment does not work.

```python
# NVIDIA NIM — openai-compatible model (native output)
ModelConfig(provider="nvidia", model="openai/gpt-oss-120b")

# NVIDIA NIM — non-OpenAI model (no native output)
ModelConfig(provider="nvidia", model="meta/llama-3.1-8b-instruct")
```

### Fallback chain

`ModelConfig.fallback_models` lists models tried in declaration order after the primary one on
API failure (rate limits, 5xx, auth errors, timeouts). `create_model()` wraps the chain in
pydantic-ai's `FallbackModel`; an empty list — the default — returns the primary model unwrapped.
Two rules are enforced when the config is constructed: the chain is flat (an entry may not declare
its own `fallback_models`), and every entry must agree with the primary on native structured-output
support, because that wrapper is chosen once from the primary's provider before any request is sent.
`context_length` stays primary-only: a fallback firing mid-run does not change the compaction budget.

Every entry is built eagerly, when the agent is constructed — not lazily, on the first failure. That
is what makes a bad entry fail loudly and early, but it also means each entry's credentials and
environment must be present up front: the example below does not construct without
`AZURE_OPENAI_ENDPOINT`, even while the OpenAI primary is perfectly healthy. All entries share the
one `http_client` passed to `create_model()`.

```python
ModelConfig(
    provider="openai",
    model="gpt-5.2",
    fallback_models=[
        ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ModelConfig(provider="azure", model="gpt-4o-mini"),
    ],
)
```

A chain declared on `ReactAgentConfig.model_cfg` also reaches the compaction summarizer, which
builds its model through the same `create_model()` and falls back to `model_cfg` when
`CompactionConfig.summary_model_cfg` is unset.

## ReactAgent API

```python
class ReactAgent:
    def __init__(
        self,
        config: ReactAgentConfig,
        deps_type: type[Any] | None = None,  # dependency injection type
        tools: list[Any] | None = None,       # tool functions
        toolsets: list[Any] | None = None,    # MCP server toolsets
        result_type: type[Any] = str,         # default output type
        observer: ContextObserver | None = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,  # pydantic-ai AgentCapability sequence
        event_loop: asyncio.AbstractEventLoop | None = None,  # DEPRECATED — accepted and ignored
    ) -> None: ...

    # Execution
    async def run(self, user_prompt: UserPrompt, deps=None, output_type=None) -> Any: ...
    def run_sync(self, user_prompt: UserPrompt, deps=None, output_type=None) -> Any: ...

    # Context
    @property
    def context(self) -> ContextManager: ...
    def subscribe_context(self, observer: ContextObserver) -> None: ...
    def restore_context(self, events: Sequence[EventMessage]) -> None: ...

    # Context compaction (see Context Compaction)
    def compact(self) -> str: ...         # force a fold now, bypassing the budget gate
    def clear_context(self) -> str: ...   # drop history; system prompt regenerates next run

    # Dynamic prompts and tools (decorator API)
    def system_prompt(self, func: F) -> F: ...  # wraps @agent.system_prompt(dynamic=True)
    def tool(self, func: F) -> F: ...            # wraps @agent.tool()

    # Teardown
    async def aclose(self) -> None: ...  # release the httpx pool; leaves the loop open
    def close(self) -> None: ...         # full synchronous teardown; idempotent

    # Advanced
    @property
    def pydantic_agent(self) -> Agent[Any, Any]: ...  # access underlying pydantic-ai Agent
```

`output_type` in `run()` overrides the construction-time `result_type` for that call only.
Both are wrapped with `get_output_type()` to apply the provider-aware `NativeOutput` strategy.

`event_loop=` is **deprecated and ignored**: `ReactAgent.__init__` always creates and owns its
own loop, and `run_sync()` runs on that one. It is kept in the signature for one release so
callers can stop passing it without a flag day.

`ReactAgent.__init__` creates that loop eagerly, so an agent built and discarded without
`close()` leaks it. Call `close()` (or `await aclose()` then `close()`) when you are done.

## Capabilities

`capabilities` is an optional constructor argument on `ReactAgent` (accepted-and-ignored on
`MockReactAgent`) — a sequence of pydantic-ai `AgentCapability` instances, forwarded unchanged
to the wrapped `Agent(...)` as `capabilities or []`. Omitting it is behaviourally identical to
today: `[]` is already `Agent`'s own default.

**Why it exists.** [Context Compaction](#context-compaction) and
[System Prompt Rendering Events](#system-prompt-rendering-events) now cover history
summarization, orphan `role=tool` dropping, and system-prompt dedup — the things consumers
used to reach capabilities for. What's left, and what `akgentic-llm` deliberately does not
own, is **domain-specific history transformation** — e.g. injecting a deployment's
source-reference block (ADR-011 §Division of responsibility). `capabilities` is the supported
seam for that, replacing a workaround that reached three private attributes across two
libraries.

**Example** — `ProcessHistory` is a built-in pydantic-ai capability that wraps a plain
message-transforming function via `before_model_request`, exactly the domain-specific-
transformation use case ADR-011 names:

```python
from pydantic_ai.capabilities import ProcessHistory
from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig

def inject_source_reference(messages):
    """Domain-specific history transformation — not a framework concern."""
    # ... prepend a deployment's source-reference block, etc.
    return messages

agent = ReactAgent(
    config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
    capabilities=[ProcessHistory(processor=inject_source_reference)],
)
```

**Ordering caveats — neither is guessable from the signature:**
- A capability's `before_model_request` hook runs **after** compaction: `ContextManager`
  rewrites messages first, the result is passed as `message_history`, and only then does the
  capability chain run. A capability sees only the **post-compaction** history — it never sees
  what compaction folded away.
- A capability that orphans a tool call/return pair — e.g. by splitting one while injecting
  content — is **not** left broken. pydantic-ai's own dangling-tool-call repair
  (`_agent_graph._clean_message_history` with `repair_last_response=True`) runs on the model
  request path, **after** the capability chain, and synthesizes a matching `ToolReturnPart`
  before the request reaches the provider. One pydantic-ai path skips the repair: resuming a
  provider-suspended response runs the capability chain without it. `ReactAgent` has no
  deferred-tool or suspend flow, so every request `ReactAgent` itself issues is repaired.
  This is pydantic-ai's internal pipeline behaviour, **not a documented public guarantee**, and
  it could change in a future release — a capability should still avoid orphaning tool calls on
  purpose.

## Multimodal Prompts

`UserPrompt = str | list[str | BinaryContent]` is the accepted type for `run()` and
`run_sync()`. Pass a mix of text strings and `BinaryContent` objects:

```python
from pydantic_ai import BinaryContent
from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig

agent = ReactAgent(config=ReactAgentConfig(
    model_cfg=ModelConfig(provider="openai", model="gpt-4o")
))

with open("diagram.png", "rb") as f:
    image_bytes = f.read()

result = agent.run_sync([
    "Describe what is shown in this architecture diagram.",
    BinaryContent(data=image_bytes, media_type="image/png"),
])
```

`UserPrompt` is exported from `akgentic.llm` so consuming layers (`akgentic-agent`) can
annotate their own signatures without importing `pydantic_ai` directly.

> **Note:** Provider support for `BinaryContent` varies — passing an image to a non-vision
> model raises a provider-level error. Multimodal turns are not JSON-serializable and are
> treated as ephemeral (not persisted in history replay).

## Context Management

`ReactAgent` maintains a persistent `ContextManager` across calls. Message history is passed
as `message_history` on every `Agent.iter()` invocation, giving the LLM full conversation
continuity without manual history threading.

```python
agent = ReactAgent(config=config)

# First turn
agent.run_sync("Start the analysis.")

# Second turn — model sees the previous exchange
agent.run_sync("Now summarise your findings.")

# Inspect what the model will see next
print(len(agent.context.messages))
```

### ContextManager

```python
from akgentic.llm import ContextManager

# With optional sliding window (system messages always preserved)
ctx = ContextManager(max_messages=20)

# History
ctx.add_message(msg)          # append + notify observers (message, tool and usage events)
ctx.messages                  # → list[ModelMessage] — a shallow copy, safe to hold
ctx.last_input_tokens         # → int | None — provider-reported size of the last response

# Observers
ctx.subscribe(observer)
ctx.unsubscribe(observer)

# Operator actions recorded outside a run
ctx.record_operator_action("…")     # buffered before the first run, appended after it
ctx.drain_pending_operator_actions()  # → list[str]; ReactAgent.run folds these into the prompt

# System-prompt rendering (see System Prompt Rendering Events)
ctx.record_system_prompt(run_id)
ctx.seed_system_prompt_hash(content_hash)   # restore dedup state without re-emitting

# Compaction and reset
ContextManager.fold_compaction(messages, event)  # static; the shared live/replay fold
ctx.compact(event)            # apply the fold and emit LlmContextCompactedEvent
ctx.clear_context()           # → int removed; emits LlmContextClearedEvent
ctx.restore(messages)         # bulk replace, no observers, no window
ctx.clear()                   # drop every message, silently
```

### Observer Pattern

```python
from akgentic.llm import (
    ContextObserver, LlmMessageEvent, LlmUsageEvent, LlmSystemPromptEvent,
    LlmContextCompactedEvent, LlmContextClearedEvent, ToolCallEvent, ToolReturnEvent,
)

class MyObserver:
    def notify_event(self, event: object) -> None:
        if isinstance(event, ToolCallEvent):
            print(f"Tool called: {event.tool_name} ({event.tool_call_id})")
        elif isinstance(event, ToolReturnEvent):
            status = "success" if event.success else "error"
            print(f"Tool returned: {event.tool_name} ({status})")
        elif isinstance(event, LlmUsageEvent):
            print(f"Usage: {event.model_name} — {event.input_tokens}in/{event.output_tokens}out")
        elif isinstance(event, LlmSystemPromptEvent):
            print(f"System prompt for run {event.run_id} ({event.content_hash[:8]}):")
            for part in event.parts:
                print(f"  [{part.dynamic_ref or 'static'}] {part.content}")
        elif isinstance(event, LlmMessageEvent):
            print(f"New message: {event.message}")
        elif isinstance(event, LlmContextCompactedEvent):
            print(f"Compacted {event.replaced_message_count} msg(s) via '{event.strategy_id}'")
        elif isinstance(event, LlmContextClearedEvent):
            print(f"Cleared {event.cleared_message_count} msg(s)")

agent = ReactAgent(config=config, observer=MyObserver())
# or: agent.subscribe_context(MyObserver())
```

Events: `LlmMessageEvent`, `LlmUsageEvent`, `LlmSystemPromptEvent`,
`LlmContextCompactedEvent`, `LlmContextClearedEvent`, `ToolCallEvent`, `ToolReturnEvent`.
Observers are notified synchronously — exceptions propagate to the caller.

### Tool Event Observability

`ToolCallEvent` and `ToolReturnEvent` are emitted by `ContextManager.add_message()` after
`LlmMessageEvent`, derived from the same message. They provide a clean observability interface
for tool activity without requiring consumers to parse pydantic-ai message internals.

**Part-kind → event mapping:**

| `part_kind` in message | Event emitted | Condition |
|---|---|---|
| `tool-call` | `ToolCallEvent` | One event per part (parallel calls → N events) |
| `tool-return` | `ToolReturnEvent(success=True)` | Always |
| `retry-prompt` | `ToolReturnEvent(success=False)` | Only when tool raised an error |

**Field semantics:**

- `tool_name` — identifies which tool was called; primary routing key in observer handlers
- `tool_call_id` — provider-assigned identifier; correlates a `ToolCallEvent` with its
  corresponding `ToolReturnEvent` within the same message stream
- `arguments` — raw JSON string from the provider. Use `json.loads(event.arguments)` for
  structured access. Stored as `str` to avoid coupling to tool-specific parameter schemas.
- `success` — `True` for clean returns; `False` when the tool raised an error (pydantic-ai
  emits a `retry-prompt` part in that case). The return content is not carried in
  `ToolReturnEvent`; it is already in the accompanying `LlmMessageEvent`.

**Emission ordering:** `LlmMessageEvent` always fires first. Tool events follow immediately.
A consumer receiving `ToolCallEvent` can safely assume the full message is already in context.

### System Prompt Rendering Events

pydantic-ai re-evaluates dynamic system prompts (date, roster, role profiles, mailbox
notices, …) **in place** before each model call, so the rendering actually sent to the
model can change on runs 2+ without any `LlmMessageEvent` being emitted. `LlmSystemPromptEvent`
records that effective rendering so observers (traces, frontends) can show exactly what the
model saw on each run.

**`LlmSystemPromptEvent` payload:**

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | The `ReactAgent` run ID this rendering belongs to — correlates with the run's `LlmMessageEvent` / `ToolCallEvent` / `LlmUsageEvent` |
| `parts` | `tuple[SystemPromptPartSnapshot, ...]` | Full rendering in model order — self-contained, **not** a diff |
| `content_hash` | `str` | sha256 hex over the ordered `(dynamic_ref, content)` pairs; carried in the event so dedup state can be re-seeded on restore without re-hashing |

**`SystemPromptPartSnapshot` fields:**

| Field | Type | Description |
|---|---|---|
| `dynamic_ref` | `str \| None` | Function name for dynamic parts (registered via `@agent.system_prompt(dynamic=True)`); `None` for static parts |
| `content` | `str` | Rendered text actually sent to the model for this part |

**Emission semantics:** emitted by `ContextManager.record_system_prompt(run_id)`, which
`ReactAgent` calls **once per completed run** after pydantic-ai's in-place re-evaluation has
produced the rendering. It scans the first `ModelRequest`'s system parts, hashes the ordered
`(dynamic_ref, content)` pairs, and emits **only when the content hash changed** since the
previous run — run 1 emits via the `None → hash` transition; an unchanged rendering on later
runs emits nothing, so the log does not grow with every run; a context with no system parts
emits nothing. The event store stays **strictly append-only**: emission only appends, and
restoring an agent re-seeds the dedup hash (via `seed_system_prompt_hash`) **without
re-emitting** an unchanged rendering.

**Usage — label each block by its source and render the text the model saw:**

```python
from akgentic.llm import LlmSystemPromptEvent

class SystemPromptTracer:
    def notify_event(self, event: object) -> None:
        if not isinstance(event, LlmSystemPromptEvent):
            return
        print(f"System prompt @ run {event.run_id} (hash {event.content_hash[:8]})")
        for snapshot in event.parts:
            label = snapshot.dynamic_ref or "static"
            print(f"  [{label}] {snapshot.content}")
```

## Context Compaction

Long-running agents accumulate conversation history that eventually approaches the model's
context window. **Compaction** folds the conversation into a summary, preserving only the
system prompt — the default `summarize` strategy replaces the **entire** non-system history
with one summary (no verbatim tail); **clear** drops the history outright so the system
prompt regenerates on the next run. Both are event-sourced: the `ContextManager` emits a
single **primitive** event describing *what changed* (counts + summary text), never the
replaced `ModelMessage` objects — so the log round-trips through the generic serializer and
any subscriber can fold the same change client-side.

Compaction can fire **automatically** (usage-based: when the provider-reported input tokens
cross `trigger_ratio × context_length`, no tokenizer required) or **on demand** via
`ReactAgent.compact()` and `ReactAgent.clear_context()`.

### Compaction & Clear Events

**`LlmContextCompactedEvent`** — emitted when history is folded into a summary:

| Field | Type | Description |
|---|---|---|
| `run_id` | `str \| None` | ReactAgent run the compaction belongs to; `None` if outside a run |
| `strategy_id` | `str` | Resolved strategy id (registry id or FQCN) that produced the summary |
| `summary` | `str` | Summary text that replaced the folded messages |
| `replaced_message_count` | `int` | Non-system messages folded — **observability only**; the `summarize` fold drops *all* non-system content regardless of this count |
| `summarizer_prompt_version` | `str` | Version id selecting the summarizer instructions — see [Overriding the Summarizer Prompt](#overriding-the-summarizer-prompt) |
| `tokens_before` | `int \| None` | Input-token estimate before compaction; `None` if unknown |
| `tokens_after` | `int \| None` | Post-compaction context-size estimate; `None` if the strategy doesn't report one |

**`LlmContextClearedEvent`** — emitted when history is dropped without summarizing:

| Field | Type | Description |
|---|---|---|
| `run_id` | `str \| None` | ReactAgent run the clear belongs to; `None` if outside a run |
| `cleared_message_count` | `int` | Number of messages dropped from context |

Both are append-only: a subscriber reconstructs the resulting context by folding the event
over its own message log. For `summarize` the fold is **full + part-level** — keep only the
system-prompt *parts* (the first request is rebuilt system-parts-only, so a user prompt fused
into it by pydantic-ai is folded away too) and insert the single `summary`; `sliding_window`
keeps the last `keep_recent_messages`; `clear` resets to empty. The fold no longer depends on
`replaced_message_count` (it is observability-only).

**Usage — observe compaction/clear alongside the other LLM events:**

```python
from akgentic.llm import LlmContextCompactedEvent, LlmContextClearedEvent

class CompactionTracer:
    def notify_event(self, event: object) -> None:
        if isinstance(event, LlmContextCompactedEvent):
            print(
                f"compacted @ run {event.run_id}: folded {event.replaced_message_count} msg(s) "
                f"via '{event.strategy_id}' ({event.tokens_before} → {event.tokens_after} tok est.)"
            )
            print(f"  summary: {event.summary[:120]}…")
        elif isinstance(event, LlmContextClearedEvent):
            print(f"cleared @ run {event.run_id}: dropped {event.cleared_message_count} msg(s)")

agent = ReactAgent(config=config, observer=CompactionTracer())
# or: agent.subscribe_context(CompactionTracer())
```

### Compaction Strategies

The strategy is selected by `CompactionConfig.strategy` — a registry id or a dotted FQCN.
Built-ins:

| `strategy` | Behaviour | Calls an LLM? |
|---|---|---|
| `"summarize"` (default) | Replaces the **entire** non-system history with one summary via an awaited LLM call (system prompts kept, part-level — a user prompt fused into the first system request is folded away); **no verbatim tail**. Degrades to a truncation marker if the summarizer errors. Ignores `keep_recent_messages`. | Yes |
| `"sliding_window"` | Deterministic head-drop: keeps the last `keep_recent_messages` verbatim and folds the rest behind a marker, no LLM. | No |
| `"none"` | No-op: never folds a message. | No |

Configure via `CompactionConfig` (nested in `ReactAgentConfig`):

```python
from akgentic.llm import CompactionConfig

cfg = CompactionConfig(
    strategy="summarize",        # or "sliding_window", "none", or "my.module.MyStrategy"
    auto_trigger=True,           # usage-based auto-compaction
    trigger_ratio=0.85,          # fire when input tokens ≥ 0.85 × context_length
    keep_recent_messages=4,      # trailing messages kept verbatim — sliding_window only (summarize ignores it)
    summary_target_tokens=2000,  # token budget the summarizer aims for
    summarizer_prompt_version="v1",
)
```

**Custom strategies (open extension).** A `CompactionStrategy` is any object with
`async def compact(self, messages) -> CompactionResult`. Register a factory in the public,
mutable `COMPACTION_STRATEGIES` registry before building an agent, or reference a class by
its dotted FQCN — the resolver imports it via stdlib `importlib` (akgentic-llm imports no
sibling package):

```python
from akgentic.llm import COMPACTION_STRATEGIES, CompactionConfig, CompactionResult

class KeepLastOnly:
    async def compact(self, messages):
        return CompactionResult(summary="", replaced_message_count=max(0, len(messages) - 1))

# (a) register a factory under a short id...
COMPACTION_STRATEGIES["keep_last"] = lambda cfg, model_cfg, http_client: KeepLastOnly()
cfg = CompactionConfig(strategy="keep_last")

# (b) ...or point strategy at a dotted FQCN — no registration needed
cfg = CompactionConfig(strategy="my_package.compaction.KeepLastOnly")
```

### Overriding the Summarizer Prompt

The `summarize` strategy ships a **domain-agnostic default** system prompt. The prompt text
is **not** stored on `CompactionConfig` — that config is serialized into every agent's start
event, so embedding a multi-line prompt there would duplicate it across the event log.
Instead the config carries only a small `summarizer_prompt_version` id, and the text lives in
the public, mutable `SUMMARY_INSTRUCTIONS` registry keyed by that id (open-extension
precedent: `COMPACTION_STRATEGIES`). The version id is also recorded on each
`LlmContextCompactedEvent` for traceability.

Override programmatically — using the installed package, no source fork — before any agent
is built:

```python
from akgentic.llm import SUMMARY_INSTRUCTIONS, CompactionConfig

# (a) replace the default in place — every "v1" agent picks it up
SUMMARY_INSTRUCTIONS["v1"] = "You are a summarizer for legal documents. Preserve …"

# (b) register a named variant and select it per agent (the id is what lands in the event)
SUMMARY_INSTRUCTIONS["legal"] = "You are a summarizer for legal documents. Preserve …"
cfg = CompactionConfig(strategy="summarize", summarizer_prompt_version="legal")
```

An unknown `summarizer_prompt_version` falls back to the built-in default. For
deployment-driven configuration (env / `.env`), a server's wiring layer can seed
`SUMMARY_INSTRUCTIONS` from its settings at startup — keeping the prompt a process-level
config that never enters the per-agent event stream.

## Cost Tracking and Aggregation

`akgentic-llm` emits an `LlmUsageEvent` for every `ModelResponse` received from a provider.
These events carry per-request token counts and can be aggregated into hierarchical cost
summaries using `aggregate_usage()`.

### Pricing

Model pricing is resolved via the [`genai-prices`](https://github.com/pydantic/genai-prices)
library against its bundled offline snapshot — there is no pricing table maintained in
this package. For each model, `_compute_cost()` builds a `genai_prices.Usage` from the
aggregated token counts and calls `calc_price(usage, model_ref=model_name,
provider_id=provider_name or None)`. An unmatched `model_ref` raises `LookupError`, which
is caught and mapped to `0.0` — unpriced models still have their tokens aggregated.

Because pricing comes from `genai-prices`' bundled snapshot, prices are only as current as
the installed `genai-prices` release (no live/auto-update is wired into this package). The
dependency therefore carries **no upper bound** — capping it would freeze the price table
and make this package report stale costs. Refreshing prices means resolving a newer
`genai-prices`, not editing a pin.

### Aggregation

```python
from akgentic.llm import LlmUsageEvent, aggregate_usage

# Collect events from an observer
events: list[LlmUsageEvent] = my_observer.collected_events

# Aggregate totals and per-model breakdown
summary = aggregate_usage(events)
print(f"Total cost: ${summary.total_cost_usd:.4f}")
print(f"Input tokens: {summary.total_input_tokens}")
for model_name, usage in summary.by_model.items():
    print(f"  {model_name}: ${usage.estimated_cost_usd:.4f}")

# Include per-run breakdown
summary = aggregate_usage(events, by_run=True)
for run in summary.runs:
    print(f"Run {run.run_id}: ${run.total_cost_usd:.4f}")
```

### Data Models

| Model | Description |
|-------|-------------|
| `LlmUsageEvent` | Frozen dataclass emitted per `ModelResponse` — carries `run_id`, `model_name`, `provider_name`, token counts, and `requests` |
| `ModelUsage` | Aggregated tokens and estimated cost for a single model |
| `RunUsageSummary` | Per-run summary with per-model breakdown |
| `AgentUsageSummary` | Top-level summary with `by_model`, optional `runs`, and grand totals |

## Prompts

### PromptTemplate

Config-time `{placeholder}` rendering. Used by `AgentConfig.prompt` in `akgentic-agent`:

```python
from akgentic.llm import PromptTemplate

tpl = PromptTemplate(
    template="You are {role}.\n\nInstructions: {instructions}",
    params={"role": "the Librarian", "instructions": "Extract structured data."},
)
print(tpl.render())
# → "You are the Librarian.\n\nInstructions: Extract structured data."
```

### Dynamic System Prompts

Register callables that are evaluated fresh on every LLM call:

```python
from typing import Any

from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig
from akgentic.llm import current_datetime_prompt, json_output_reminder_prompt

agent = ReactAgent(config=ReactAgentConfig(
    model_cfg=ModelConfig(provider="openai", model="gpt-4o")
))

# Built-in utilities
agent.system_prompt(current_datetime_prompt)       # "The current date and time is …"
agent.system_prompt(json_output_reminder_prompt)   # reminder to output JSON only

# Custom prompt
@agent.system_prompt
def workspace_context(ctx: Any) -> str:
    return f"Working directory: {get_current_workspace()}"
```

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv sync --all-packages --all-extras
```

### Commands

```bash
# Run tests
uv run pytest packages/akgentic-llm/tests/

# Run tests with coverage
uv run pytest packages/akgentic-llm/tests/ --cov=akgentic.llm --cov-fail-under=80

# Lint
uv run ruff check packages/akgentic-llm/src/

# Format
uv run ruff format packages/akgentic-llm/src/

# Type check
uv run mypy packages/akgentic-llm/src/
```

### CI Pipeline

Every pull request runs the full quality gate via GitHub Actions (`.github/workflows/ci.yml`):

CI checks out this repository standalone — not the workspace — so its commands use
repo-relative paths, unlike the workspace-root invocations under [Commands](#commands) above.

| Step | Command | Gate |
|------|---------|------|
| Type check | `uv run mypy src/` (strict, Python 3.12) | Zero errors |
| Lint | `uv run ruff check src/` | Zero errors |
| Tests | `uv run pytest tests/ --cov=akgentic.llm --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80` | All pass, ≥ 80% coverage |

The CI badge at the top of this README reflects the current state of `master`. PRs are
blocked from merging until all steps are green.

### Project Structure

```
src/akgentic/llm/
    __init__.py     # Public API exports
    agent.py        # ReactAgent, UsageLimitError, UserPrompt type alias
    compaction.py   # COMPACTION_STRATEGIES, SUMMARY_INSTRUCTIONS, CompactionStrategy,
                    #   CompactionResult, create_compaction()
    config.py       # ModelConfig, CompactionConfig, TokenUsageLimits, RunUsageLimits,
                    #   AgentUsageLimits, UsageLimits (deprecated), HttpClientConfig,
                    #   RuntimeConfig, ReactAgentConfig, _supports_native_output()
    context.py      # ContextManager
    event.py        # LlmMessageEvent, LlmUsageEvent, LlmSystemPromptEvent,
                    #   SystemPromptPartSnapshot, LlmContextCompactedEvent,
                    #   LlmContextClearedEvent, ToolCallEvent, ToolReturnEvent,
                    #   ContextObserver and EventMessage protocols
    pricing.py      # _compute_cost() (genai-prices), ModelUsage, RunUsageSummary,
                    #   AgentUsageSummary, aggregate_usage()
    prompts.py      # PromptTemplate, current_datetime_prompt, json_output_reminder_prompt
    providers.py    # create_model(), create_http_client(), get_output_type(),
                    #   create_model_settings()
    loadtest/       # Optional `loadtest` extra: token-free mock agent
        __init__.py
        mock_agent.py
        scenario.py
tests/              # Tests organised by module
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-llm/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
