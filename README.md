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

### Standalone Package

```bash
# Clone and enter package directory
cd packages/akgentic-llm

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

uv pip install -e .
```

### Within Monorepo Workspace

If you're developing in the Akgentic Platform v2 monorepo:

```bash
# From workspace root
source .venv/bin/activate

# Package is already installed in editable mode via workspace
# Cross-package dependencies (akgentic-core) are automatically resolved
```

## Quick Start

```python
from akgentic.llm import ReactAgent, ModelConfig, ReactAgentConfig

# Configure agent
config = ReactAgentConfig(
    model_cfg=ModelConfig(provider="openai", model="gpt-4.1")
)

# Create agent
agent = ReactAgent(config=config)

# Run agent
result = agent.run_sync("Hello, how are you?")
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

### Runtime Configuration

```python
from akgentic.llm import RuntimeConfig, HttpClientConfig

# Default runtime settings
runtime = RuntimeConfig()

# Custom retry and execution strategy
runtime = RuntimeConfig(
    retries=5,                      # Retry tool failures 5 times
    end_strategy="exhaustive",      # Execute all tools vs. early stop
    parallel_tool_calls=True        # Enable concurrent tool execution
)
```

End Strategies:

- `"early"`: Stops after first successful tool result (fast path)
- `"exhaustive"`: Executes all tool calls even when result is available (complete data gathering)

```python
runtime = RuntimeConfig(
    retries=5,
    end_strategy="exhaustive",
    parallel_tool_calls=True
    http_client_config=HttpClientConfig(
        timeout_seconds=120.0,      # Max time for single LLM request
        max_retries=5,              # HTTP retry attempts
        backoff_multiplier=0.5,     # Exponential backoff multiplier
        backoff_max=60.0            # Max backoff delay
    )
```

`HttpClientConfig` allows fine-tuning of HTTP request behavior for LLM calls, including timeouts and retry logic.

### Complete agent configuration example

```python
from akgentic.llm import ReactAgentConfig, ModelConfig, UsageLimits, RuntimeConfig, HttpClientConfig

# Full configuration with all options
config = ReactAgentConfig(
    model_cfg=ModelConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    ),
    usage_limits=UsageLimits(
        request_limit=10,
        total_tokens_limit=5000
    ),
    runtime_cfg=RuntimeConfig(
        retries=5,
        end_strategy="exhaustive",
        http_client_config=HttpClientConfig(
            timeout_seconds=120.0,
            max_retries=5
        )
    )
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

### Standalone Package Development

```bash
# Clone and enter package directory
cd packages/akgentic-llm

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

### Monorepo Workspace Development

```bash
# From workspace root
source .venv/bin/activate

# Run tests
pytest packages/akgentic-llm/tests/

# Type checking
mypy packages/akgentic-llm/src/

# Linting
ruff check packages/akgentic-llm/src/
```
