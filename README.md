# akgentic-llm

[![CI](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/dd80a44fe9e2e27b46f7f3431e19202f/raw/coverage.json)](https://github.com/b12consulting/akgentic-llm/actions/workflows/ci.yml)

LLM integration layer for the [Akgentic](https://github.com/b12consulting/akgentic-framework)
multi-agent framework (open-source bundle). Wraps pydantic-ai's REACT execution loop with persistent context
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
  - [Model roster and runtime switching](#model-roster-and-runtime-switching)
    - [Roster vs. fallback chain](#roster-vs-fallback-chain)
    - [Switching at runtime](#switching-at-runtime)
- [Providers](#providers)
- [ReactAgent API](#reactagent-api)
- [Capabilities](#capabilities)
  - [Run-loop capabilities](#run-loop-capabilities)
  - [Hook timeline](#hook-timeline)
  - [Run-tier recovery](#run-tier-recovery)
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

- **ReactAgent** — a thin wrapper around one awaited `pydantic_ai.Agent.run()` call that persists
  message history across calls — through a mounted `EventSourcingCapability`, whose sweep is
  bounded by the last message it recorded, located by **identity**, with a positional cursor as
  the fallback (see [Run-loop capabilities](#run-loop-capabilities)) — and, when a run-tier
  breach is *not* recovered, translates pydantic-ai's `UsageLimitExceeded` into a framework-local
  `RunUsageLimitError` — one of the two tiers under the exported base `UsageLimitError`. By
  default the breach **is** recovered: the turn degrades into one tool-free conclusion and
  `run()` returns its answer (see [Usage limits](#usage-limits) and
  [Run-tier recovery](#run-tier-recovery))
- **Provider abstraction** — `create_model()` dispatches to one of six provider factories
  (OpenAI, Azure, Anthropic, Google, Mistral, NVIDIA), wrapping the result in pydantic-ai's
  `FallbackModel` when `ModelConfig.fallback_models` is non-empty; `get_output_type()` wraps
  output types with `NativeOutput` for providers that support structured output, falls back to
  prompt-based extraction for those that don't
- **HTTP retry** — `create_http_client()` configures `AsyncHTTPX2TenacityTransport` with exponential
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
  │     ├── await pydantic_agent.run(        # pydantic-ai REACT loop, one awaited call
  │     │       user_prompt=…,
  │     │       deps=…,
  │     │       usage_limits=…,              # RUN tier only; no usage= is ever passed
  │     │       message_history=context.messages,
  │     │       model=self._model,           # the ACTIVE model, read here — a switch lands next run
  │     │       output_type=get_output_type( # re-resolved per run from the LIVE model_cfg
  │     │           self._config.model_cfg, output_type or self._result_type),
  │     │   )
  │     │     │
  │     │     ├── LifetimeBudgetCapability  # refuses a spent agent; folds what a run burned
  │     │     ├── CompactionCapability      # folds an over-long history before the run reads it
  │     │     ├── EventSourcingCapability, after each graph node:
  │     │     │     context.add_message()   # persists + notifies observers
  │     │     ├── LimitRecoveryCapability   # on a run-tier breach, decides: conclude, or raise
  │     │     └── HealingCapability         # closes out tool calls a failed run left dangling
  │     │
  │     ├── on a run-tier breach the seam asked to conclude:
  │     │     conclude_without_tools(decision.reason)   # sibling run, no tools
  │     │     return that run's output                  # NOT a RunUsageLimitError
  │     │
  │     └── return result.output
  │
  ├── context: ContextManager               # persistent message history
  ├── compact() / clear_context()           # fold history into a summary, or drop it
  └── system_prompt(func)                   # register dynamic system prompt
```

**Runtime dependencies:** `pydantic-ai-slim[anthropic,google,mistral,openai,retries]>=2.32,<3`,
`genai-prices>=0.1.0`, `pydantic>=2.0.0`, `httpx2>=2.7`, `tenacity>=8.0.0`, `pyyaml>=6.0`. An
optional `loadtest` extra pulls in the token-free mock agent's own `pyyaml` requirement.

**Module boundary:** `akgentic-llm` MUST NOT import from `akgentic-core`, `akgentic-tool`, or
`akgentic-agent`.

## Installation

Published on PyPI. Python 3.12 or newer.

```bash
uv add akgentic-llm
# or
pip install akgentic-llm
```

That is the whole install. `pydantic-ai-slim`, `genai-prices`, `httpx2`, `tenacity`
and `pyyaml` come with it as ordinary dependencies — no workspace checkout, no
submodules.

### Optional Extras

| Extra      | Packages pulled in | Enables                                            |
|------------|--------------------|----------------------------------------------------|
| `loadtest` | `pyyaml`           | `akgentic.llm.loadtest` — token-free scripted mock |

```bash
uv add "akgentic-llm[loadtest]"
```

### As part of the framework bundle

`akgentic-framework` is the meta-distribution that pins every akgentic package
at versions built and tested together. Install `akgentic-llm` through it when
you want the release-wide pin rather than a single package:

```bash
pip install "akgentic-framework[llm]"   # this package alone, release-pinned
pip install "akgentic-framework[all]"   # the whole framework
```

### Working on the package itself

To develop `akgentic-llm` rather than use it, clone the open-source bundle
[akgentic-framework](https://github.com/b12consulting/akgentic-framework), which
carries every package together as submodules:

```bash
git clone git@github.com:b12consulting/akgentic-framework.git
cd akgentic-framework
git submodule update --init
# uncomment the two "SOURCE MODE" blocks in pyproject.toml
uv sync
```

Source mode resolves `akgentic-*` to the local checkouts, editable.

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
Enforced by pydantic-ai; breaching any limit raises `RunUsageLimitError`.

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
> `RunUsageLimitError` on v2 without the prompt or the tools having changed.

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
`AgentUsageLimitError` — the agent tier's own class, distinct from the run tier's — with a
message of the form `Exceeded the agent_request_limit of 100 (run_count=100)`.

Five consequences worth knowing before you set it:

- **A run that fails still counts.** The budget is consumed before the call executes, not
  after it returns — including when the call ends in a run-tier `RunUsageLimitError`. An
  agent stuck in a failing loop therefore still runs out of lifetime budget, which is the
  point: both limits mean "this agent is burning too many turns".
- **A rescued turn costs TWO units, not one.** When a run-tier breach is recovered
  (see [Run-tier recovery](#run-tier-recovery)), the tool-free conclusion is a *sibling run*
  through the same capability stack and pays the same agent-tier pre-flight as any other. So one
  `run()` call that breaches consumes two units of this budget, and a workload that breaches
  often buys half as many turns as one that never does. Size `agent_request_limit` accordingly.
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
Breaching one raises `AgentUsageLimitError` with pydantic-ai's own message text — e.g.
`Exceeded the total_tokens_limit of 1000000 (total_tokens=1000420)`. That wording is the same
shape a run-tier breach produces, deliberately: the **class** is what carries the tier, so
nothing downstream has to parse text to tell the two apart.

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

That check is **one rule with one implementation, applied at two moments**. The rule lives in
`validate_compaction_bounds()` in `config.py`, and it is called both by `ReactAgentConfig`'s
after-validator at construction and by `ReactAgent.switch_model()` against the *candidate* roster
entry, before anything is committed. A switch changes `model_cfg.context_length`, and therefore
moves the threshold — so switching to a longer-context model whose threshold would reach a set
`input_tokens_limit` or `total_tokens_limit` is **refused**, with the produced threshold and the
limit both named in the message (see
[Model roster and runtime switching](#model-roster-and-runtime-switching)). Past that point
pydantic-ai raises before compaction can ever fire and the auto-trigger is dead code with no error
anywhere, which is exactly what the construction-time check exists to prevent — a second spelling
would have let a switch install what the constructor refuses.

`validate_compaction_bounds` is **not** exported from `akgentic.llm`; it is an intra-package
invariant. A consumer that genuinely needs it imports it as
`from akgentic.llm.config import validate_compaction_bounds`.

#### Telling the two tiers apart

A breach raises **one of two classes**, both subclassing `UsageLimitError` — except that a
run-tier breach, by default, **does not raise at all**. Read the whole of this subsection before
writing an `except RunUsageLimitError`.

- **`UsageLimitError`** — the base, and the documented **catch-all**. It stays exported, and no
  enforcement site raises it directly, so an `except UsageLimitError` written before the tiers
  were split still catches everything it used to. The split is **additive**: there is nothing to
  migrate and nothing is deprecated here.
- **`RunUsageLimitError`** — one `run()` call exhausted its `RunUsageLimits` budget.
  **Recovered by default rather than raised**: the mounted `LimitRecoveryCapability` decides the
  turn concludes, `run()` drives one tool-free conclusion and returns *that* answer. This class
  reaches the caller only when the recovery seam **declines**, or when the conclusion it asked
  for failed or produced nothing usable. See [Run-tier recovery](#run-tier-recovery) for the
  seam, and the paragraphs below for what it means for existing code.
- **`AgentUsageLimitError`** — the `AgentUsageLimits` lifetime budget is spent. Raised
  **pre-flight** by either agent-tier check, before the call executes. **Terminal**: no follow-up
  run can be admitted, because the budget that would pay for it is exactly the one that is spent.
  Recovery never applies to it — the seam is consulted for pydantic-ai's `UsageLimitExceeded`
  only, and `AgentUsageLimitError` is a different class that passes straight through.

**Tell them apart with `isinstance`, never by message text.** The token-limit messages come from
pydantic-ai and read alike at both tiers, by design — the class is what carries the tier. Nothing
downstream should branch on an error string.

Both names are exported from `akgentic.llm`, alongside the base:

```python
from akgentic.llm import AgentUsageLimitError, RunUsageLimitError, UsageLimitError

try:
    # A run-tier breach usually does NOT arrive here as an exception: the default policy
    # concludes the turn, and `answer` is that tool-free conclusion's output.
    answer = await agent.run("...")
except RunUsageLimitError:
    ...  # recovery declined, or its conclusion failed — the turn produced nothing usable
except AgentUsageLimitError:
    ...  # this agent is finished; a further run cannot be admitted
```

**`run()` no longer always raises on a run-tier breach — this is the one visible behaviour
change.** Where the call used to end in `RunUsageLimitError`, it now returns the concluded
answer, and an `except RunUsageLimitError:` clause written against the old contract **may stop
firing**. Nothing about the call's *shape* changed: `run()` still returns the output type you
asked for, and the conclusion is produced with the same `deps` and `output_type`, so a structured
result still routes downstream through your normal path.

**The exact opt-out** is a `LimitRecoveryCapability` subclass whose `handle_limit_exceeded`
returns `None`, mounted through the `limit_recovery=` constructor keyword. It reproduces the
previous behaviour exactly — the breach surfaces as `RunUsageLimitError`, chained from
pydantic-ai's own `UsageLimitExceeded`, and nothing else happens:

```python
from akgentic.llm import LimitRecoveryCapability, ReactAgent

class NeverConcludes(LimitRecoveryCapability):
    """Restore the pre-recovery contract: a run-tier breach simply raises."""

    async def handle_limit_exceeded(self, ctx, *, error):
        return None

agent = ReactAgent(config=config, limit_recovery=NeverConcludes())
```

**Escalation parity: you always see the ORIGINAL breach.** If the conclusion itself fails — a
second run-tier breach, a terminal `AgentUsageLimitError` from its own pre-flight, anything at
all — or produces nothing usable (`None`, or a string that is empty or whitespace-only), the
caller gets a `RunUsageLimitError` built from the breach that started it, never from the
secondary failure. A "this turn ran out of budget" signal is never replaced by an unrelated one.

After a run-tier breach the context is left runnable rather than diagnostic: the tool calls the
aborted turn never answered are healed with a short **model-facing instruction** — it tells the
model this turn's budget is spent, that no further tool call is possible, and to answer now with
what it already has — so that sentence, not a traceback, is the tool result a follow-up run
reasons from. That healing is what the recovery seam is consulted *after*: by the time the
conclusion runs, the healed `ToolReturnPart` is already the last thing the model sees, and the
conclusion's own prompt is layered on top of it. When the breach does surface, the operator still
gets the stack: it leaves `run()` as a `RunUsageLimitError` chained from pydantic-ai's own
`UsageLimitExceeded` (`raise ... from e`), and that exception's traceback is what reaches the
event stream.

##### What a consumer has to handle

Degradation is entirely this package's, and it leaves the caller exactly two outcomes — an
answer, or a raise. There is no partial state to interpret and nothing to attempt a second time:

| the turn | `run()` does | you see |
|---|---|---|
| runs normally | returns the output | an ordinary output |
| breaches the **run** tier, seam accepts, conclusion succeeds | returns the conclusion's output | an ordinary output — **indistinguishable** |
| breaches the run tier, seam declines, *or* the conclusion fails or produces nothing usable | re-raises the **ORIGINAL** breach | `RunUsageLimitError` |
| breaches the **agent** tier | raises pre-flight, terminal | `AgentUsageLimitError` |

That row-3 guarantee is what lets a consumer's whole usage-limit policy be a single `except
UsageLimitError` on the base — no tier branch, no retry of its own. `akgentic-agent` is exactly
that: it catches the base, notifies the team's human, and ends the turn.

**"Indistinguishable" is about the returned value, not the whole trace.** A rescue does leave two
marks a consumer can observe if it looks for them: the conclusion's events arrive under a *second*
`run_id`, and the turn costs **two** units of the agent-lifetime run budget. Neither reaches the
return value, which is why no caller has to branch on either — see
[Run-tier recovery](#run-tier-recovery) for both.

**Two limitations follow, and both are open questions on ADR-021 rather than oversights:**

- **The agent tier cannot say goodbye.** `AgentUsageLimitError` is terminal by *construction* —
  the seam is never consulted for it — so an exhausted agent stops mid-conversation with no
  final word. The default is right; that it is not overridable is the gap (§Q1).
- **A conclusion that succeeds emptily tells nobody.** "Nothing usable" is deliberately narrow:
  `None`, or a blank string. A **structured** output carrying nothing — say a list of requests
  that is empty — is an ordinary success, and this package cannot tell otherwise, because it
  receives the output as `Any`. It is returned, your code routes it, nothing goes out, and no
  exception is raised (§Q2).

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
`total_tokens_limit`), so a run that finished cleanly on v1 can raise `RunUsageLimitError` on v2
if that turn pushes it past a run-tier ceiling. See [Usage limits](#usage-limits).

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
    # A list is a roster: element 0 is active, all of them are declared and switchable.
    # Pass a single ModelConfig instead for a one-model agent (model_roster stays []).
    model_cfg=[
        ModelConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
        ),
        ModelConfig(provider="openai", model="gpt-4o-mini"),
    ],
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

### Model roster and runtime switching

An agent may declare a **roster** of models and change which one answers, at runtime, without
being rebuilt. **The pydantic-ai `Agent` is never rebuilt and never mutated by a switch** — the
new model reaches it as a per-run `model=` argument. A switch changes *who answers the next turn*,
not *what the agent remembers*.

`ReactAgentConfig` carries the roster in two fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_cfg` | `ModelConfig` | `ModelConfig()` | The **active** model. Always a single config once stored |
| `model_roster` | `list[ModelConfig]` | `[]` | The full declared roster, in declaration order, active entry included. `[]` = single-model agent, switching unavailable |

`model_cfg` additionally accepts a `list[ModelConfig]` **at the input boundary only**. A
`mode="before"` validator folds that list into element 0 as the active model plus the whole list —
declaration order preserved — as `model_roster`. The union is an input convenience, never a stored
shape: every read path downstream sees one `ModelConfig`, and both fields are real Pydantic fields
that round-trip through `model_dump()` / `model_validate()` unchanged.

```python
from akgentic.llm import ModelConfig, ReactAgentConfig, model_roster_key

config = ReactAgentConfig(
    model_cfg=[
        ModelConfig(provider="openai", model="gpt-4o", context_length=128_000),
        ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
    ],
)

config.model_cfg.model                                    # 'gpt-4o' — element 0 is active
len(config.model_roster)                                  # 3 — the active entry is a member
[model_roster_key(m) for m in config.model_roster]
# ['openai:gpt-4o', 'anthropic:claude-sonnet-4-5', 'google-gla:gemini-2.0-flash']

# A single ModelConfig leaves the roster empty — a one-model agent, which cannot switch.
ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")).model_roster  # []
```

**The key grammar is `f"{provider}:{model}"`**, spelled once, by `model_roster_key()`. That key is
the roster's identity and the argument `switch_model()` takes. Four rules are enforced when the
config is constructed, each with its own message:

- an **empty list** is rejected — an agent needs at least one model;
- passing **both** a `model_cfg` list and an explicit `model_roster` is rejected — the roster is
  derived from the list, so pass only one;
- **duplicate keys** are rejected, naming the repeated key: two entries with one key would make a
  switch request ambiguous, and the ambiguity would only surface at switch time on whichever entry
  happened to be found first;
- the **active model must be a member** of a non-empty roster. Normalization satisfies this by
  construction; it is reachable only when a caller hand-sets `model_roster`.

`model_roster_key()`, `normalize_model_roster()` and `validate_unique_roster_keys()` are all
exported from `akgentic.llm`, deliberately. Sibling packages project a roster onto their own row
types and **import** these rather than re-spelling the grammar — one implementation, not a second
copy. A re-spelled key grammar is a model switch that silently matches nothing.

#### Roster vs. fallback chain

A roster is not a [fallback chain](#fallback-chain), and the two compose: a roster entry keeps its
own `fallback_models`, and switching to it swaps the entry *together with* its chain. Four
differences matter in practice:

| | Roster (`model_roster`) | Fallback chain (`fallback_models`) |
|---|---|---|
| **Trigger** | **Deliberate** — a human or the model asks for a named entry | **Automatic** — fires on API failure with nobody asking |
| **Visibility** | **Observable** — `switch_model()` returns the entry now active | **Invisible** — a chain entry firing is not surfaced |
| **Build time** | **Lazy** — an entry is built only when it is switched to | **Eager** — every entry is built at construction |
| **Homogeneity** | **Heterogeneous allowed** — entries need not agree on native structured-output support | **Forbidden** — every entry must agree with the primary |

The last two are the ones that bite.

**Lazy means a roster entry can fail at switch time.** A chain's eager construction is what makes a
bad entry fail loudly and early (the paragraph under [Fallback chain](#fallback-chain) still
holds). A roster entry is built only on the switching turn, so an entry whose environment is not
satisfied — a missing `ANTHROPIC_API_KEY`, an unset `AZURE_OPENAI_ENDPOINT` — fails *then*, in
front of whoever asked for it, as a `ModelSwitchError`.

**Heterogeneous rosters are safe precisely because the output wrapper is now resolved per run.** A
fallback entry is selected *inside* one request, after the structured-output wrapper for that
request has already been chosen — so a mismatched entry would be served by the wrong wrapper. A
roster entry is selected *before* a request, and the wrapper is re-derived from the newly active
model on the next `run()`. That is why the chain enforces homogeneity and the roster does not.

#### Switching at runtime

`ReactAgent` exposes two readers and one switch:

```python
from akgentic.llm import ModelSwitchError, ReactAgent

agent = ReactAgent(config=config)

agent.active_model()      # the live ModelConfig — after a switch, the roster entry itself
agent.model_roster()      # a fresh list of the declared entries, in declaration order

entry = agent.switch_model("anthropic:claude-sonnet-4-5")
entry.model               # 'claude-sonnet-4-5' — the roster entry now active

try:
    agent.switch_model("openai:o99")
except ModelSwitchError as e:
    str(e)  # "cannot switch to 'openai:o99': available keys: openai:gpt-4o, ..."
```

`model_roster()` returns a **copy** by contract — the list is routinely handed to a tool that
renders or sorts it, and mutating it must not edit the agent's roster. The entries themselves are
shared, and `switch_model()` installs one of them by identity.

**A roster entry is installed wholesale.** `provider:model` is the identity; the entry is the
definition. Its `temperature`, `max_tokens`, `context_length` and `fallback_models` *replace* the
active model's — they are not merged with them. Switching to the already-active key is not
short-circuited, so that rule holds with no exception.

**What a switch preserves.** The pydantic-ai `Agent` is neither rebuilt nor mutated, so tools,
toolsets, registered system prompts, the `ContextManager` message history, the lifetime usage
counters and the one HTTP connection pool all survive it untouched. The summarizer, when it is
rebuilt (below), is rebuilt over that same client — a switch opens no second pool.

**`ModelSwitchError` is the one class every refusal raises.** It is exported from `akgentic.llm`
and subclasses `ValueError`, so an existing `except ValueError` keeps catching it. A consumer needs
exactly **one** `except ModelSwitchError` and must never fall back to `except Exception`. It is
raised in four conditions:

1. **The agent declares no roster** — a distinct message ("this agent declares no model roster, so
   switching is unavailable"), not an empty list of available keys.
2. **The key is unknown** — the message names the requested key *and* every available key. That
   text is what a tool-driven caller gets as its whole diagnosis, and it is the difference between
   a self-correcting failure and a stuck agent.
3. **The entry cannot be built.** Anything `create_model()` raises is translated, with the original
   text preserved and the original kept as `__cause__`. **This covers a provider's own exception
   class**: pydantic-ai raises `UserError` — a `RuntimeError`, *not* a `ValueError` — for a missing
   `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and that is the commonest form of a switch-time build
   failure. Translating it here is what lets the caller keep the single `except`.
4. **The entry would make auto-compaction unreachable** — its compaction threshold would reach or
   pass a set run-tier token limit (see [Usage limits](#usage-limits)).

**A refusal changes nothing.** Every fallible step — resolving the key, building the model,
re-checking the compaction bounds, building the summarizer — runs *before* the first assignment.
There is deliberately no snapshot and no rollback path, because nothing is written until everything
has passed: `_config`, `_model` and the summarizer strategy are left as exactly the objects they
were.

**The summarizer follows the active model, unless it was pinned.** When
`CompactionConfig.summary_model_cfg is None`, the compaction strategy is **rebuilt on every
switch** against the newly active entry, over the same HTTP client. When `summary_model_cfg` is set
explicitly it stays **pinned** across every switch — a dedicated cheap summarizer is the operator's
choice, and an escalation must not drag it along.

**A switch made mid-run takes effect from the next run.** `switch_model()` is reachable from inside
a tool, so this boundary is a real one:

- **the run in flight is unaffected.** pydantic-ai binds the model and the output type once per
  `run()` call, so the turn that made the switch is still answered by the pre-switch model;
- **a conclusion driven after a run-tier breach is a next run** (see
  [Run-tier recovery](#run-tier-recovery)) and *is* served the post-switch model and output type;
- **the one thing that does move mid-run is the auto-compaction gate.** The threshold is computed
  from `model_cfg.context_length` read *live*, so a switch to a larger-context model moves the
  threshold while the run still answers on the pre-switch model. This is deliberate, not an
  oversight: caching the threshold to keep the two in step would break the live-config invariant
  for a consistency nobody asked for, the divergence lasts exactly one run, and it errs toward
  compacting sooner.

> **Trap: a switch does not sanitize the message history.** Provider-specific parts already in the
> history are handed to the next provider exactly as pydantic-ai maps them. Heterogeneous switching
> is therefore **best-effort**. What this package guarantees is narrow and worth stating precisely:
> akgentic performs **no** sanitization — that absence is characterized by test. What a real
> provider does when it is handed another provider's parts is **not** proven by those tests and is
> **not** guaranteed. `/compact` (or `ReactAgent.compact()`) before switching is the mitigation
> available today.

**Under the `loadtest` extra**, `MockReactAgent` answers both readers truthfully — `active_model()`
off its config, `model_roster()` as a fresh list — and **refuses every switch** with the same
`ModelSwitchError` class, imported rather than redefined, so one `except` catches both. It replays
a scenario bound to `model_cfg.model` at construction and builds no model at all.

> **This package calls `switch_model()` nowhere.** `akgentic-llm` provides the mechanism; the
> wiring that makes a switch reachable from a conversation — the tool surface and the agent-side
> configuration — lives in `akgentic-tool` and `akgentic-agent`. Adding a roster changes no agent
> behaviour on its own.

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
builds its model through the same `create_model()`. When `CompactionConfig.summary_model_cfg` is
unset the summarizer **follows the active model**: it is built from `model_cfg` at construction and
rebuilt from the new entry on every `switch_model()`, over the same HTTP client. Setting
`summary_model_cfg` explicitly pins the summarizer, and a switch leaves it alone.

A chain is not a roster, and the two compose — see
[Roster vs. fallback chain](#roster-vs-fallback-chain) for the four differences that matter.

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
        limit_recovery: LimitRecoveryCapability | None = None,       # run-tier recovery policy
    ) -> None: ...

    # Execution
    async def run(self, user_prompt: UserPrompt, deps=None, output_type=None) -> Any: ...
    def run_sync(self, user_prompt: UserPrompt, deps=None, output_type=None) -> Any: ...
    async def conclude_without_tools(self, reason: str, *, deps=None, output_type=None) -> Any: ...
    def conclude_without_tools_sync(self, reason: str, *, deps=None, output_type=None) -> Any: ...

    # Context
    @property
    def context(self) -> ContextManager: ...
    def subscribe_context(self, observer: ContextObserver) -> None: ...
    def restore_context(self, events: Sequence[EventMessage]) -> None: ...

    # Context compaction (see Context Compaction)
    def compact(self) -> str: ...         # force a fold now, bypassing the budget gate
    def clear_context(self) -> str: ...   # drop history; system prompt regenerates next run

    # Model roster (see Model roster and runtime switching)
    def active_model(self) -> ModelConfig: ...        # the live config, read at call time
    def model_roster(self) -> list[ModelConfig]: ...  # a fresh list, declaration order
    def switch_model(self, key: str) -> ModelConfig: ...  # raises ModelSwitchError; next run on

    # Dynamic prompts and tools (decorator API)
    def system_prompt(self, func: F) -> F: ...  # wraps @agent.system_prompt(dynamic=True)
    def tool(self, func: F) -> F: ...            # wraps @agent.tool()

    # Teardown
    async def aclose(self) -> None: ...  # release the httpx2 pool; leaves the loop open
    def close(self) -> None: ...         # full synchronous teardown; idempotent

    # Advanced
    @property
    def pydantic_agent(self) -> Agent[Any, Any]: ...  # access underlying pydantic-ai Agent
```

`output_type` in `run()` overrides the construction-time `result_type` for that call only. **The
effective type is resolved per run, from the model that is live at that moment** —
`get_output_type(self._config.model_cfg, output_type or self._result_type)`, evaluated at the
`pydantic_agent.run()` call itself. The wrapper baked at construction stays on the `Agent` as an
unused default and is **never again the effective type for any run**.

That is not a detail. Before it, an agent built with `result_type=<a Pydantic model>` kept the
wrapper chosen from its *original* provider: switch it across the native-output boundary — an
OpenAI agent moving to Gemini, say — and the run would still be wrapped in `NativeOutput` for a
provider that has none. Re-resolving per run is what makes a
[heterogeneous roster](#roster-vs-fallback-chain) safe.

**`conclude_without_tools()` turns an interrupted turn into an answer — and it is still a
mechanism, not a policy.** The distinction holds; what moved is *where the policy lives*. It is
no longer in `akgentic-agent`: `LimitRecoveryCapability`'s `handle_limit_exceeded` seam, in this
package, is what decides *whether* an interrupted turn concludes, and `run()` drives this method
when the seam asks for it (see [Run-tier recovery](#run-tier-recovery)). This method itself still
decides nothing — a direct call concludes unconditionally, and a conclusion is never itself
recovered: it enters the shared run core with recovery off, so a breach *during* a conclusion
raises rather than starting another one. Note also the `*`: `deps` and `output_type` are
**keyword-only** on both conclusion methods, unlike `run()`'s positional ones.

What the mechanism does, whether a caller invokes it or the seam asks for it:

- **The tools are removed with `override(tools=[], toolsets=[])`** — the only construct that
  *replaces* what is registered. A per-run `toolsets=[]` is documented as **additional** toolsets
  and would leave every tool in place. "Zero tool calls" is not expressible as a limit either:
  `tool_calls_limit` is `gt=0`.
- **The run carries its own `RunUsageLimits(run_request_limit=1)`**, not the budget that was just
  exhausted. With no tools available, one request is what the turn needs.
- **The agent-tier pre-flight still applies.** An agent whose *lifetime* budget is also spent
  raises `AgentUsageLimitError` from the conclusion. That is the caller's signal to stop trying,
  not a defect to swallow — the lifetime counter is what bounds a retry loop by construction.
- **It emits an `LlmUsageEvent`** like any other run: it shares `run()`'s execution core, so the
  usage fold, the system-prompt recording and the persistence sweep are identical.

`reason` reaches the model as the run's user prompt, layered on the healed context — so the
healing instruction described under [Usage limits](#usage-limits) is already there as the tool
result the model reasons from. `conclude_without_tools_sync()` is the synchronous bridge,
mirroring `run_sync()`: closed-agent guard, then the agent's own loop.

`ReactAgent.__init__` creates that loop eagerly, so an agent built and discarded without
`close()` leaks it. Call `close()` (or `await aclose()` then `close()`) when you are done.

## Capabilities

`capabilities` is an optional constructor argument on `ReactAgent` (accepted-and-ignored on
`MockReactAgent`) — a sequence of pydantic-ai `AgentCapability` instances. They are **not**
forwarded unchanged: `ReactAgent` mounts five internal capabilities of its own first and appends
yours after them, so the wrapped `Agent(...)` always receives

```python
[LifetimeBudgetCapability, CompactionCapability, EventSourcingCapability,
 LimitRecoveryCapability, HealingCapability,
 *(capabilities or [])]
```

The stack is never `[]`, even when the argument is omitted — those five are how every run
enforces the agent-lifetime budget, folds an over-long history, persists its messages, decides
whether a run-tier breach degrades into an answer, and closes out its dangling tool calls. See
[Run-loop capabilities](#run-loop-capabilities) and [Run-tier recovery](#run-tier-recovery)
below.

`limit_recovery=` is a separate constructor keyword rather than an entry in this sequence, and
that is deliberate — see [Run-tier recovery](#run-tier-recovery).

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

**Write the processor to be idempotent.** It runs on **every model request**, not once per
run, so an unconditional prepend stacks one block per step within a single run. Injected
content is also persisted, but **asymmetrically**: on the first run there is no recorded tail
yet, so the block falls inside the window the persistence sweep covers and *is* recorded like
any other message; from the second run on it is prepended *ahead* of the recorded tail and is
*not*. Either way, an unguarded prepend would sit on top of the copy already persisted. Guard
the injection on what is already there, as below.

```python
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelRequest, UserPromptPart
from akgentic.llm import ReactAgent, ReactAgentConfig, ModelConfig

SOURCE_REFERENCES = "... a deployment's source-reference block ..."

def _is_source_reference(message):
    return (
        isinstance(message, ModelRequest)
        and len(message.parts) == 1
        and isinstance(message.parts[0], UserPromptPart)
        and message.parts[0].content == SOURCE_REFERENCES
    )

def inject_source_reference(messages):
    """Domain-specific history transformation — not a framework concern."""
    if messages and _is_source_reference(messages[0]):
        return messages
    return [ModelRequest(parts=[UserPromptPart(content=SOURCE_REFERENCES)]), *messages]

agent = ReactAgent(
    config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
    capabilities=[ProcessHistory(processor=inject_source_reference)],
)
```

Prepending is also the only safe direction: pydantic-ai rejects a processed list that is empty
or does not end with a `ModelRequest`.

**Ordering caveats — none is guessable from the signature:**
- A capability's `before_model_request` hook runs **after** compaction. The promise holds, but
  not by the mechanism it once did: the fold is no longer performed by `ReactAgent` ahead of the
  run. It happens in `CompactionCapability.wrap_run`'s **head**, and that `wrap_run` encloses
  every hook a caller capability has. So a capability still sees only the **post-compaction**
  history and never what compaction folded away — because it is mounted *inside* the fold, not
  because the fold ran before the run started.
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
- Your capabilities sit **inside** the five internal ones — unless one of them declares its own
  ordering constraints, see [Run-loop capabilities](#run-loop-capabilities) — so `before_*` hooks
  fire after theirs and `after_*` hooks before theirs. Your durable `after_*` edits are what gets
  persisted: `EventSourcingCapability`'s closing sweep lives in `wrap_run`'s `finally`, outside
  every capability's node hooks.

### Run-loop capabilities

`ReactAgent` does not implement the lifetime budget, compaction, persistence, the run-tier
recovery decision or dangling-tool-call repair in its run method. All five are standalone
pydantic-ai capabilities, exported from `akgentic.llm`, and each is mountable on a bare `Agent`
of your own:

| Class | Owns | Hook anchors |
|---|---|---|
| `LifetimeBudgetCapability(limits=…)` | The agent-**lifetime** budget — one run count and three token caps. Refuses a spent agent before the wrapped run does anything, then folds what the run burned into the lifetime total | `wrap_run`'s head (both refusals — the only enforcement site), `wrap_run`'s `finally` (the usage fold, so tokens a *failed* run burned still count) |
| `CompactionCapability(strategy=…, context=…, threshold_fn=…, event_factory=…)` | Folds the conversation when provider-reported input tokens cross the armed threshold, applying the result to `ContextManager` **and** to the run's own history | `wrap_run`'s head — once per run by construction: it keeps no per-run state and overrides no `for_run` |
| `EventSourcingCapability(context=…)` | Hands every message a run produces to `ContextManager.add_message()`, exactly once, in run order; records the run's system-prompt rendering | `after_node_run` (steady state, keeps emission incremental), `before_node_run` (re-anchors the live history), `wrap_run`'s `finally` (closing sweep + system-prompt recording) |
| `LimitRecoveryCapability()` | The run-tier recovery **policy** — on a `UsageLimitExceeded`, whether the turn degrades into a tool-free conclusion and with what prompt. It only *decides* and records; the conclusion is a sibling run driven by whoever mounted it. Anything that is not a `UsageLimitExceeded` passes straight through without consulting the seam | `on_run_error` — always re-raises, never returns a result to suppress the error |
| `HealingCapability(context=…)` | Appends one `ToolReturnPart` per tool call left dangling by a failed run, so the *next* run is not rejected for unprocessed tool calls | `on_run_error` — it always re-raises the original exception, never returns to recover |

**Ordering.** The stack is
`[LifetimeBudgetCapability, CompactionCapability, EventSourcingCapability,
LimitRecoveryCapability, HealingCapability, *yours]`, and the **first capability is the
outermost**: `before_*` hooks fire in list order,
`after_*` in reverse, and `wrap_run`s nest with the first wrapping all the rest. Each position
earns its place, and they are not equally load-bearing:

- **Budget outermost.** Every inner capability and every model request is downstream of its
  `wrap_run` head, so a refusal there is the only one that costs nothing. Concretely: a
  lifetime-spent agent is refused *before* compaction pays for a summarizer LLM call. This is
  the one coupling that genuinely depends on list position — moving compaction ahead of the
  budget is observable in the test suite.
- **Compaction before event sourcing.** The persistence cursor opens on the post-fold history,
  so the synthetic summary request sits behind it and is never re-persisted as an
  `LlmMessageEvent` (which would double-apply it on replay, since
  `LlmContextCompactedEvent` already carries the summary). Be clear about the strength of this
  one: it is **belt-and-braces, not the mechanism**. `EventSourcingCapability` re-opens its
  cursor against the normalised list at the first node hook, which absorbs a fold performed
  anywhere ahead of it — swapping the two leaves the behaviour unchanged.
- **Internal before caller.** Because the chain unwinds in reverse, a caller capability's
  `after_*` hooks run *before* the persistence sweep, so its durable edits are the ones
  persisted.
- **Limit recovery immediately before healing.** pydantic-ai walks the `on_run_error` chain in
  **reverse**, so the *later* entry fires *first*: healing writes its `ToolReturnPart` before the
  recovery seam is consulted, and a policy that reads the context to decide therefore sees the
  **healed** one. That is the whole of the reason for this position. It is **not** what keeps a
  dangling tool call out of the conclusion — the walk runs every hook and only then re-raises, so
  healing has written its part before any conclusion is driven, whatever the order. What protects
  the conclusion is that recovery uses `on_run_error` rather than `wrap_run`, which is a separate
  statement; see [Run-tier recovery](#run-tier-recovery).
- **Healing last of the internals.** Error hooks fire after `wrap_run` has unwound, so the
  dangling `ModelResponse` is already persisted by the time the healer looks for it. That is
  structural rather than positional.

**The order is a default, not a guarantee.** If **any** capability in the chain declares
`get_ordering()` — a fixed `position`, or a `wraps=` / `wrapped_by=` constraint — pydantic-ai
topologically re-sorts the whole chain to satisfy it, keeping the given order only as a
tiebreaker. None of the five declares one, so the shipped stack is the list above; a caller
capability that declares `position='outermost'` lands ahead of all five, whatever the list says.
What survives that and what does not:

- **Persistence survives any ordering.** The closing sweep is in `wrap_run`'s `finally`, outside
  every capability's node hooks, so durable `after_*` edits are always the ones persisted.
- **`on_run_error` precedence does not**, and is **deliberately uncontracted**. pydantic-ai walks
  that hook from the innermost capability outwards, and this package states no contract about a
  recovering capability pre-empting `HealingCapability`. That is exactly the "limit recovery
  before healing" coupling above: it holds for the shipped list and is **not** guaranteed under a
  re-sort. Pinning the budget outermost by declaring an ordering would be a behavioural change
  owed its own decision, so none of the five does it.

**Compaction writes twice, as one operation.** When the gate arms, `CompactionCapability` applies
the fold to `ContextManager` *and* mirrors the result into the run's own history list, in one
method. Both writes are needed because `Agent.run()` seeds the run's state from a **copy** of the
history it is handed: mutating the run's list never reaches `ContextManager`, and folding
`ContextManager` never reaches the run. The second write **mirrors** the first rather than folding
again — a second fold would double-fold, and mirroring preserves message *identity*, which the
persistence sweep's tail anchor locates by. `ReactAgent.compact()` reaches the same method with no
live list, since there is no run in flight on that path.

**Durable state only.** Persist from `RunContext.messages` **inside a node hook**. Never from
`ModelRequestContext.messages` mid-chain: that request copy legitimately carries other
capabilities' in-flight edits, and pydantic-ai folds the processed list back into durable history
after the `before_model_request` chain anyway. `wrap_run`'s own `ctx.messages` is a snapshot
frozen at the incoming history — `UserPromptNode` rebinds the run's history to a *normalised
copy* (consecutive requests merged, orphaned tool results dropped, dangling calls repaired) that
is routinely shorter. A cursor opened against the pre-normalisation length therefore sits past
where the run's own messages begin and skips everything behind it, in silence. Open it against
the list the sweep will index.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from akgentic.llm import ContextManager, EventSourcingCapability, LlmMessageEvent

class Recorder:
    def __init__(self) -> None:
        self.messages = []

    def notify_event(self, event: object) -> None:
        if isinstance(event, LlmMessageEvent):
            self.messages.append(event.message)

context = ContextManager()
recorder = Recorder()
context.subscribe(recorder)

# A bare pydantic-ai Agent — no ReactAgent anywhere.
agent = Agent(model=TestModel(), capabilities=[EventSourcingCapability(context)])
result = await agent.run("hello")

# Everything the run produced, persisted once, in run order — and every observer saw it.
assert context.messages == list(result.all_messages())
assert recorder.messages == context.messages
```

**A history processor must preserve message identity for messages it does not own.** That is a
contract, not advice: the persistence sweep depends on it. The sweep bounds itself on the last
message it recorded, located by **identity**, with a positional cursor as the fallback. A
processor that *shifts* history (a prepend) moves that message without changing which object it
is, and identity absorbs the shift; one that *rebuilds* history out of equal copies leaves
positions intact, and the cursor absorbs the rebuild. Two edits defeat both at once, because each
destroys the identity anchor **while** moving what sits behind it:

- **rebuild plus prepend** — e.g. `[deepcopy(m) for m in messages]` with a block prepended —
  re-persists an earlier run's `ModelResponse` as a copy, and emits a spurious extra
  `LlmUsageEvent`;
- **removing the anchor message itself** — an interior removal — silently drops later runs' own
  `ModelRequest`s.

Both are exactly the shape a **summarising or redacting** processor has. pydantic-ai's own
layered equivalent reaches for `run_id` as a third layer here; this one has no third layer, and
whether it owes one is an open decision, not a shipped guarantee. Until that is settled: prepend
to the list you were handed, do not rebuild it.

**The event API is unchanged** — across both the persistence decomposition and the later move of
the lifetime budget and compaction into capabilities. Those changed *where* the concerns live,
not what a run emits: the same seven event types (`LlmMessageEvent`, `ToolCallEvent`,
`ToolReturnEvent`, `LlmUsageEvent`, `LlmSystemPromptEvent`, `LlmContextCompactedEvent`,
`LlmContextClearedEvent`), the same payload shapes, the same per-message ordering
(`LlmMessageEvent` → tool events → `LlmUsageEvent`), the same run-id correlation. **No consumer
needs a schema change.** Two paths do emit *more* than they used to, both described below.

**Two behavioural differences, and both are the fix working.**

*The blind tail closes.* The hand-rolled loop this replaces dropped messages a run appended after
its last drain — a cancellation tail, or an end-of-run drain that never happened — **in silence**.
The closing sweep in `wrap_run`'s `finally` persists them, so a downstream consumer will see
`LlmMessageEvent`s on those paths that it never saw before. That is the loss being repaired.

*A failed or cancelled run can now emit `LlmSystemPromptEvent`.* The per-run system-prompt
recording moved into that same `finally`. Its previous call site sat inside the success path and
was unreachable once a run raised, so a run that failed recorded nothing even when it had rendered
a new prompt. Two consequences, before you assume this stream is byte-identical to the old one:

- an event can appear on a failure or cancellation path where none appeared before — subject to
  the unchanged hash dedup, so an unchanged rendering still emits nothing; and
- on a healed failure it lands **ahead of** the healing `LlmMessageEvent`, a relative position that
  did not previously exist.

Group a trace by `run_id`, which every event carries, rather than by arrival order.

### Hook timeline

The five internal capabilities, `PendingMessageDrainCapability` (auto-injected by pydantic-ai)
and any caller capability all hang off the **same** set of hooks. What separates them is *which*
hook and *which direction the chain is walked* — and neither is guessable from a signature. This
section is the map.

**One ordering rule covers every hook family:**

> `before_*` walks the list **forwards** — outermost first.
> `after_*`, `on_*_error` and the tail of every `wrap_*` walk it **backwards** — innermost first.

So the capability that sees a request **first** is the one that sees a result **last**. That
asymmetry is load-bearing, and it is the single most common source of surprise: a capability
mounted `outermost` gets the first word on the way in and the *last* word on the way out.

#### The hooks, by family

| Family | Type | Runs at | Input it is handed | What returning a value does |
|---|---|---|---|---|
| `before_run` | observe | once, run start | `ctx` | nothing — observe-only |
| `wrap_run` | wrap | around the whole run | `handler()` | short-circuit the run, or recover a raised error |
| `on_run_error` | error | run raised | the exception | **raise** to propagate, **return** an `AgentRunResult` to recover |
| `after_run` | transform | run produced a result | `AgentRunResult` | replaces the run's result |
| `before_node_run` | transform | each graph node | the node | replaces the node about to execute |
| `wrap_node_run` | wrap | around each node | `handler(node)` | retry the node, or redirect graph progression |
| `after_node_run` | transform | each node succeeded | next node **or** `End` | **replaces graph progression** — an `End` may become a node, and vice versa |
| `before_model_request` | transform | each model request | `ModelRequestContext` | replaces messages, settings, model, parameters |
| `wrap_model_request` | wrap | around the provider call | `handler(request_context)` | `ModelRetry` to retry; a `ModelResponse` to substitute |
| `after_model_request` | transform | each model response | `ModelResponse` | replaces the response; `ModelRetry` rejects it |
| `before_tool_validate` / `after_tool_validate` | transform | per tool call, arg parsing | raw args / validated args | rewrites arguments |
| `before_tool_execute` / `after_tool_execute` | transform | per tool call, execution | validated args / the tool's return | rewrites arguments or the result |
| `before_output_validate` / `after_output_validate` | transform | **structured output only** | raw / parsed output | rewrites the parsed value |
| `before_output_process` / `after_output_process` | transform | **every** output type | the output | rewrites the final value |

`wrap_*` hooks are the only ones that can decline to call the inner chain at all, and the only
place a `try`/`finally` observes a cancellation. `after_*` hooks are **not** called for a node
interrupted by cancellation.

#### A run, top to bottom

One run of `ReactAgent`, with two model requests and a tool call in between. Read it downwards.
`▼` is the chain walking forwards (outermost→innermost), `▲` backwards.

```
                                         │  who acts here, in the shipped stack
─────────────────────────────────────────┼──────────────────────────────────────────────────
 run starts                              │
   ▼ before_run                          │  MailboxCapability: clear run-local announce-once
   ▼ wrap_run — HEAD                     │  LifetimeBudget: refuse a spent agent (costs nothing
     LifetimeBudget                      │    — every inner capability is downstream of here)
     └ Compaction                        │  Compaction: fold history if input tokens crossed the
       └ EventSourcing                   │    armed threshold; write to ContextManager AND to the
         └ LimitRecovery                 │    run's own list, as one operation
           └ Healing                     │  EventSourcing: open the persistence cursor
             └ …yours                    │
                                         │
 ┌─ UserPromptNode ──────────────────────┤
 │   ▼ before_node_run                   │  EventSourcing: re-anchor the live history
 │   ▼ wrap_node_run ▲                   │
 │   ▲ after_node_run                    │
 └───────────────────────────────────────┤
                                         │
 ┌─ ModelRequestNode  (request #1) ──────┤
 │   ▼ before_node_run                   │
 │   │ ▼ before_model_request            │  PendingMessageDrain (outermost — FIRST): drain the
 │   │                                   │    'asap' queue into this request
 │   │                                   │  yours: inject/transform history (ProcessHistory)
 │   │                                   │  MailboxCapability (agent pkg): raise on a queued
 │   │                                   │    cancel, else ENQUEUE the arrival notice — which
 │   │                                   │    this step's drain has already been past, so it
 │   │                                   │    lands in request #2
 │   │ ▼ wrap_model_request → PROVIDER ▲ │
 │   │ ▲ after_model_request             │
 │   ▲ after_node_run                    │  EventSourcing: emit LlmMessageEvent + LlmUsageEvent
 └───────────────────────────────────────┤    (steady state — emission stays incremental)
                                         │
 ┌─ CallToolsNode ───────────────────────┤
 │   ▼ before_node_run                   │
 │   │  per tool call:                   │
 │   │   ▼ before_tool_validate          │
 │   │   ▼ wrap_tool_validate ▲          │
 │   │   ▲ after_tool_validate           │
 │   │   ▼ before_tool_execute           │
 │   │   ▼ wrap_tool_execute → TOOL ▲    │
 │   │   ▲ after_tool_execute            │  MailboxCapability: on a read_mailbox call, consume
 │   ▲ after_node_run                    │    the named message and enqueue its own rendering
 └───────────────────────────────────────┤  EventSourcing: ToolCallEvent / ToolReturnEvent
                                         │
 ┌─ ModelRequestNode  (request #2) ──────┤
 │   ▼ before_model_request              │  PendingMessageDrain: NOW the notice (and the
 │   │                                   │    absorbed message) reach the model
 │   │ ▼ wrap_model_request → PROVIDER ▲ │
 │   │ ▲ after_model_request             │
 └───────────────────────────────────────┤
                                         │
 ┌─ CallToolsNode  (final output) ───────┤
 │   ▼ before_output_validate            │  structured output only — parsing the payload
 │   ▲ after_output_validate             │
 │   ▼ before_output_process             │  every output type
 │   ▲ after_output_process              │
 │   node returns End(FinalResult)       │
 │   ▲ after_node_run   ← WALKED BACKWARDS, so the drain is LAST
 │       …yours                          │
 │       MailboxCapability               │  ← the only place to withdraw enqueued content
 │       PendingMessageDrain             │  if the queue is non-empty it DISCARDS the End and
 └───────────────────────────────────────┤    redirects into one more request (see below)
                                         │
   ▲ wrap_run — TAIL / finally           │  EventSourcing: closing sweep + system-prompt record
   ▲ after_run                           │  LifetimeBudget: fold what the run burned into the
 run ends                                │    lifetime total (in `finally`, so a FAILED run counts)

 on the error path instead:              │
   ▲ on_run_error  (innermost first)     │  Healing fires BEFORE LimitRecovery — it is later in
                                         │    the list, and this chain walks backwards. So a
                                         │    recovery policy reading the context sees a HEALED one
```

**Two consequences worth stating outright, because both have already cost a bug:**

- **An `'asap'` enqueue is always one step late.** `PendingMessageDrainCapability` is
  `outermost`, so its `before_model_request` has already run by the time an inner capability
  enqueues during the same step. The content lands in the *next* request.
- **A queued message at run end costs you the run's output.** When the graph returns
  `End(FinalResult)` with the queue non-empty, the drain's `after_node_run` **throws the `End`
  away** and returns a `ModelRequestNode` instead, so the run continues and produces a second,
  different final result. `run()` returns only that second one; the first exists in durable
  history and nowhere else. That is intended for content with no other delivery path — and wrong
  for content that has one, which must therefore withdraw itself from `ctx.pending_messages` in
  its own `after_node_run`, ahead of the drain's.

### Run-tier recovery

`LimitRecoveryCapability` answers one question, on one hook: when pydantic-ai raises
`UsageLimitExceeded` mid-run, does this turn degrade into an answer, or does the breach surface?
**The seam is a single method:**

```python
async def handle_limit_exceeded(
    self, ctx: RunContext[Any], *, error: UsageLimitExceeded
) -> ConclusionDecision | None: ...
```

- **`ConclusionDecision(reason=…)`** asks for one tool-free conclusion, started with `reason` as
  its prompt on top of the healed context. The default implementation returns
  `ConclusionDecision()`, whose `reason` is `DEFAULT_CONCLUSION_REASON` — **one string, used for
  every kind of run-tier breach.** There is no per-limit wording, and it is not exported from
  `akgentic.llm`: reach it as
  `from akgentic.llm.capabilities import DEFAULT_CONCLUSION_REASON` if you need to read it.
- **`None`** declines. The breach re-raises unchanged and reaches the caller as
  `RunUsageLimitError`, which is the exact pre-recovery contract — see the opt-out under
  [Telling the two tiers apart](#telling-the-two-tiers-apart).
- **Anything that is not a `UsageLimitExceeded` never reaches the seam** — including this
  package's own `AgentUsageLimitError`, which is a different class. The agent tier is terminal.

**Override it by subclassing, and mount the subclass through `limit_recovery=`:**

```python
from akgentic.llm import ConclusionDecision, LimitRecoveryCapability, ReactAgent

class HouseStyle(LimitRecoveryCapability):
    """Conclude with a deployment's own wording instead of the default prompt."""

    async def handle_limit_exceeded(self, ctx, *, error):
        return ConclusionDecision(reason="Budget spent — answer now with what you have.")

agent = ReactAgent(config=config, limit_recovery=HouseStyle())
```

**`limit_recovery=` is the only mount point.** Passing a subclass through `capabilities=` instead
mounts a *second* recovery capability **beside** the default rather than replacing it, and the
decision `run()` acts on is read back off the instance held under `limit_recovery=` — so the
override would be silently ignored. There is one such policy per agent.

Four more things worth knowing:

- **The capability only decides.** It records the decision and re-raises; it never runs anything.
  The conclusion is a *sibling run* driven by `ReactAgent`, so a recovery never nests a run inside
  a capability hook. It uses `on_run_error` and defines **no `wrap_run` at all** — deliberately:
  pydantic-ai gives error hooks their chance only once the exception has escaped the whole
  `wrap_run` chain, so a capability that caught the breach in its own `wrap_run` would stop
  `HealingCapability.on_run_error` from ever running and the conclusion would start from a context
  still carrying a dangling tool call.
- **A conclusion is never itself recovered.** It enters the shared run core with recovery off, so
  a breach *during* a conclusion raises instead of starting another one.
- **A rescued turn costs two units of the agent-lifetime run budget** — see
  [AgentUsageLimits](#agentusagelimits--reactagentconfigagent_usage_limits).
- **The event stream is unchanged by a rescue.** The outer run's events arrive under its own
  `run_id`, then the healing `ToolReturnPart`, then the conclusion's events under a *second*
  `run_id` — because the conclusion has always been an ordinary second run. No event type, shape
  or ordering changed; the frozen event API described above still applies unmodified.

**This seam is the whole of the degradation policy, deliberately.** A consumer does not implement
its own — `akgentic-agent` used to, and retired it. Whether to conclude, with what prompt, and
against which output type all live here, where the healed context and the caller's `output_type`
are; the consumer is left with a single `except UsageLimitError`. Two things the seam cannot
currently express are ADR-021 §Q1 and §Q2 — see
[What a consumer has to handle](#what-a-consumer-has-to-handle).

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
as `message_history` on every `Agent.run()` invocation, giving the LLM full conversation
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

# User-role turns added outside a run (operator actions, context-update blocks)
ctx.append_user_prompt("…")     # buffered before the first run, appended after it
ctx.drain_pending_user_prompts()  # → list[str]; ReactAgent.run folds these into the prompt

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

**Where the auto-trigger runs.** It is `CompactionCapability`'s `wrap_run` head — *inside* the
run, before the run reads its history, and once per run by construction rather than by a guard.
`ReactAgent` does not test the gate itself. The two manual paths are unchanged, and are
deliberately **not** capabilities: `compact()` and `clear_context()` run outside any agent run,
where a run-scoped hook has no purchase. `compact()` still reaches the same single fold site the
auto path does, only with no run in flight. And because `conclude_without_tools()` goes through
the same capability stack as `run()`, the gate fires for a tool-free conclusion exactly as it
does for a normal turn.

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
uv sync --all-extras
```

### Commands

```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=akgentic.llm --cov-fail-under=80

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

### CI Pipeline

Every pull request runs the full quality gate via GitHub Actions (`.github/workflows/ci.yml`):

CI checks out this repository standalone and resolves `akgentic-*` dependencies
from PyPI, so it runs the same repo-relative commands listed above.

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
    agent.py        # ReactAgent, ModelSwitchError, UserPrompt type alias; re-exports
                    #   UsageLimitError, RunUsageLimitError, AgentUsageLimitError and
                    #   RUN_LIMIT_HEALING_MESSAGE from capabilities/errors.py, so imports
                    #   written against their old home keep working
    capabilities/   # The run-loop capabilities, one module each
        __init__.py      # Re-exports; holds the whole composition/cursor module docstring
        budget.py        # LifetimeBudgetCapability
        compaction.py    # CompactionCapability
        errors.py        # UsageLimitError, RunUsageLimitError, AgentUsageLimitError,
                         #   RUN_LIMIT_HEALING_MESSAGE
        event_sourcing.py  # EventSourcingCapability
        healing.py       # HealingCapability
        limit_recovery.py  # LimitRecoveryCapability, ConclusionDecision,
                         #   DEFAULT_CONCLUSION_REASON
    compaction.py   # COMPACTION_STRATEGIES, SUMMARY_INSTRUCTIONS, CompactionStrategy,
                    #   CompactionResult, create_compaction()
    config.py       # ModelConfig, CompactionConfig, TokenUsageLimits, RunUsageLimits,
                    #   AgentUsageLimits, UsageLimits (deprecated), HttpClientConfig,
                    #   RuntimeConfig, ReactAgentConfig, _supports_native_output(),
                    #   model_roster_key(), normalize_model_roster(),
                    #   validate_unique_roster_keys(), validate_compaction_bounds()
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
    py.typed        # PEP 561 marker
    loadtest/       # Optional `loadtest` extra: token-free mock agent
        __init__.py
        mock_agent.py
        scenario.py
tests/              # Tests organised by module
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-llm/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
