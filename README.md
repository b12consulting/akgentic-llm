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
