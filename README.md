# akgentic-llm

LLM integration layer for Akgentic agent systems - clean abstraction for LLM providers with REACT pattern support.

## Features

- **REACT Pattern Support** - Powered by pydantic-ai
- **Multi-Provider Support** - OpenAI, Azure, Anthropic, Google, Mistral, NVIDIA
- **Context Management** - Checkpointing, rewind, compactification
- **Usage Limits** - Cost control and safety
- **HTTP Retry Logic** - Production-grade reliability
- **Token Counting** - Accurate tracking from provider responses
- **Dynamic Prompts** - Programmatic system prompt registry
- **Observer Pattern** - Real-time context updates

## Installation

```bash
# With uv (recommended)
uv add akgentic-llm

# With pip
pip install akgentic-llm
```

## Quick Start

```python
from akgentic.llm import ReactAgent, ModelConfig, ReactAgentConfig

# Configure agent
config = ReactAgentConfig(
    model=ModelConfig(provider="openai", model="gpt-4o")
)

# Create agent
agent = ReactAgent(config=config, deps_type=MyDeps)

# Run agent
result, messages = await agent.run("Hello, how are you?")
print(result)
```

## Configuration

### Basic Model Configuration

```python
from akgentic.llm import ModelConfig

# Minimal configuration
config = ModelConfig(
    provider="openai",
    model="gpt-4o"
)

# With temperature and token limits
config = ModelConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=2000,
    seed=42  # For reproducible outputs
)

# Reasoning models
config = ModelConfig(
    provider="openai",
    model="o1-preview",
    reasoning_effort="high"
)
```

### Usage Limits for Cost Control

```python
from akgentic.llm import UsageLimits

# Set limits to control costs
limits = UsageLimits(
    request_limit=10,              # Max 10 LLM requests
    total_tokens_limit=5000        # Max 5000 tokens total
)

# Granular control
limits = UsageLimits(
    request_limit=50,
    tool_calls_limit=20,           # Max tool calls per request
    input_tokens_limit=10000,      # Max input tokens
    output_tokens_limit=2000       # Max output tokens
)
```

### Complete Agent Configuration

```python
from akgentic.llm import ReactAgentConfig, ModelConfig, UsageLimits, AgentRuntimeConfig

# Full configuration with all options
config = ReactAgentConfig(
    model=ModelConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    ),
    usage_limits=UsageLimits(
        request_limit=10,
        total_tokens_limit=5000
    ),
    runtime=AgentRuntimeConfig(
        retries=5,
        end_strategy="exhaustive",
        timeout_seconds=120.0,
        max_http_retries=5
    ),
    system_prompts=[
        "You are a helpful assistant.",
        "Always provide detailed explanations."
    ]
)

# Use with agent
agent = ReactAgent(config=config, deps_type=MyDeps)
```

### Supported Providers

```python
# OpenAI
ModelConfig(provider="openai", model="gpt-4o")
ModelConfig(provider="openai", model="gpt-4o-mini")

# Anthropic
ModelConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
ModelConfig(provider="anthropic", model="claude-3-opus-20240229")

# Google
ModelConfig(provider="google-gla", model="gemini-1.5-pro")

# Azure OpenAI
ModelConfig(provider="azure", model="gpt-4o")

# Mistral
ModelConfig(provider="mistral", model="mistral-large-latest")

# NVIDIA
ModelConfig(provider="nvidia", model="meta/llama-3.1-8b-instruct")
```

## Documentation

See [architecture.md](../../_bmad-output/planning-artifacts/architecture.md#phase-2-module-architecture---akgentic-llm) for complete specifications.

## Development

```bash
# Install with dev dependencies
uv add --dev akgentic-llm

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```
