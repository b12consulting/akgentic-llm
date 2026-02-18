"""LLM provider factory functions and HTTP client utilities.

This module provides factory functions for creating LLM models and
HTTP clients with production-ready retry logic.

The HTTP client uses pydantic-ai's ``AsyncTenacityTransport`` with
``wait_retry_after`` to implement exponential backoff with jitter and
automatic ``Retry-After`` header support for transient failures (429, 5xx).

Supported LLM Providers:
    - OpenAI (GPT-4, GPT-4o, o1 series)
    - Azure OpenAI Service
    - Anthropic (Claude 3.5 Sonnet, etc.)
    - Google Gemini (google-gla)
    - Mistral AI
    - NVIDIA NIM

Example:
    >>> from akgentic.llm import ModelConfig, create_model
    >>> config = ModelConfig(provider="openai", model="gpt-4o")
    >>> client = create_http_client()
    >>> model = create_model(config, client)
    >>>
    >>> # Use with pydantic-ai
    >>> from pydantic_ai import Agent
    >>> agent = Agent(model, system_prompt="You are helpful")
    >>> result = await agent.run("Hello!")
"""

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic_ai.models import Model
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.settings import ModelSettings
from tenacity import retry_if_exception, stop_after_attempt, wait_random_exponential

from .config import ModelConfig

if TYPE_CHECKING:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

logger = logging.getLogger(__name__)


_RETRYABLE_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True if the exception represents a retryable failure.

    Retries on:
    - HTTP 429 (rate limit) and 5xx (server errors)
    - Transient network errors: ConnectError, RemoteProtocolError, ReadError, WriteError

    Does NOT retry on 4xx client errors (except 429) or timeouts.

    Args:
        exc: The exception to evaluate.

    Returns:
        True if the request should be retried, False otherwise.
    """
    if isinstance(exc, _RETRYABLE_NETWORK_ERRORS):
        return True
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status = exc.response.status_code
    return status == 429 or 500 <= status < 600


def create_http_client(
    timeout_s: float = 120.0,
    max_attempts: int = 5,
    exp_multiplier: float = 0.5,
    exp_max_s: float = 60.0,
    retry_after_cap_s: float = 300.0,
) -> httpx.AsyncClient:
    """Create an async HTTP client with production-ready retry logic for LLM requests.

    Uses pydantic-ai's ``AsyncTenacityTransport`` with ``wait_retry_after`` to
    automatically retry transient failures using randomised exponential backoff
    while respecting ``Retry-After`` response headers.

    Retry Strategy:
        - Retries on HTTP 429 (rate limit) and 5xx (server errors)
        - Retries on transient network errors (ConnectError, RemoteProtocolError, ReadError, WriteError)
        - ``Retry-After`` header parsed and honoured (capped at ``retry_after_cap_s``)
        - Fallback: randomised exponential backoff with jitter
        - Fast-fail on 4xx errors other than 429 and on timeouts

    Timeout Behavior:
        All requests use ``timeout_s`` as a total per-request timeout.
        If the server does not respond within this duration the request
        raises ``httpx.TimeoutException``.

    Error Handling:
        - 429 / 5xx → ``httpx.HTTPStatusError`` raised internally, retried
          automatically until ``max_attempts`` is exhausted (then re-raised)
        - 4xx (non-429) → ``httpx.HTTPStatusError`` raised immediately
        - Timeout → ``httpx.TimeoutException`` raised immediately

    Args:
        timeout_s: Per-request timeout in seconds. Default: 120.0.
        max_attempts: Maximum total attempts (initial + retries). Default: 5.
        exp_multiplier: Base multiplier for randomised exponential backoff. Default: 0.5.
        exp_max_s: Maximum wait time between retries in seconds. Default: 60.0.
        retry_after_cap_s: Maximum seconds to honour a ``Retry-After`` header.
            Values above this cap are clamped. Default: 300.0.

    Returns:
        ``httpx.AsyncClient`` configured with retry transport and timeout.
        Use as an async context manager to ensure proper connection cleanup.

    Example:
        >>> client = create_http_client(max_attempts=5, timeout_s=60.0)
        >>> async with client:
        ...     response = await client.post(
        ...         "https://api.openai.com/v1/chat/completions",
        ...         json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        ...     )

    Note:
        The client must be used as an async context manager (``async with``) or
        explicitly closed via ``await client.aclose()`` to release connections.
    """
    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception(_is_retryable_http_error),
            wait=wait_retry_after(
                fallback_strategy=wait_random_exponential(
                    multiplier=exp_multiplier,
                    max=exp_max_s,
                ),
                max_wait=retry_after_cap_s,
            ),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        ),
        validate_response=lambda r: r.raise_for_status(),
    )

    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout_s,
    )


def _build_settings(config: ModelConfig) -> ModelSettings | None:
    """Build ModelSettings from ModelConfig, omitting None values.

    Args:
        config: LLM model configuration.

    Returns:
        ModelSettings instance if any parameters are set, else None.
    """
    kwargs: dict[str, Any] = {}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.seed is not None:
        kwargs["seed"] = config.seed
    return cast(ModelSettings, kwargs) if kwargs else None


def _build_openai_settings(config: ModelConfig) -> "OpenAIChatModelSettings | None":
    """Build OpenAIChatModelSettings from ModelConfig, including reasoning_effort.

    Delegates shared parameters (temperature, max_tokens, seed) to
    ``_build_settings`` and extends with OpenAI-specific fields.

    Args:
        config: LLM model configuration.

    Returns:
        OpenAIChatModelSettings instance if any parameters are set, else None.
    """
    from pydantic_ai.models.openai import OpenAIChatModelSettings  # noqa: PLC0415

    kwargs: dict[str, Any] = dict(cast(dict[str, Any], _build_settings(config) or {}))
    if config.reasoning_effort is not None:
        kwargs["openai_reasoning_effort"] = config.reasoning_effort
    return cast(OpenAIChatModelSettings, kwargs) if kwargs else None


def _create_openai_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> "OpenAIChatModel":
    """Create OpenAI chat model.

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured OpenAIChatModel instance.
    """
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415

    return OpenAIChatModel(
        model_name=config.model,
        provider=OpenAIProvider(http_client=http_client),
        settings=_build_openai_settings(config),
    )


def _create_azure_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> "OpenAIChatModel":
    """Create Azure OpenAI model.

    Uses the Azure OpenAI endpoint from ``AZURE_OPENAI_ENDPOINT`` env var.

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured OpenAIChatModel instance pointing at Azure endpoint.

    Raises:
        ValueError: If ``AZURE_OPENAI_ENDPOINT`` environment variable is not set.
    """
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415

    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not base_url:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure provider")
    return OpenAIChatModel(
        model_name=config.model,
        provider=OpenAIProvider(base_url=base_url, http_client=http_client),
        settings=_build_openai_settings(config),
    )


def _create_anthropic_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> "AnthropicModel":
    """Create Anthropic model.

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured AnthropicModel instance.
    """
    from pydantic_ai.models.anthropic import AnthropicModel  # noqa: PLC0415
    from pydantic_ai.providers.anthropic import AnthropicProvider  # noqa: PLC0415

    settings = _build_settings(config)
    return AnthropicModel(
        model_name=config.model,
        provider=AnthropicProvider(http_client=http_client),
        settings=settings,
    )


def _create_google_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> Model:
    """Create Google Gemini model.

    Imports ``GoogleModel`` and ``GoogleProvider`` lazily to avoid a hard
    dependency on ``google-genai`` at module load time (the package has a
    transitive dependency on ``mcp`` which may not be installed or may conflict
    in some environments).

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured GoogleModel instance.
    """
    from pydantic_ai.models.google import GoogleModel  # noqa: PLC0415
    from pydantic_ai.providers.google import GoogleProvider  # noqa: PLC0415

    settings = _build_settings(config)
    return GoogleModel(
        model_name=config.model,
        provider=GoogleProvider(http_client=http_client),
        settings=settings,
    )


def _create_mistral_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> "MistralModel":
    """Create Mistral model.

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured MistralModel instance.
    """
    from pydantic_ai.models.mistral import MistralModel  # noqa: PLC0415
    from pydantic_ai.providers.mistral import MistralProvider  # noqa: PLC0415

    settings = _build_settings(config)
    return MistralModel(
        model_name=config.model,
        provider=MistralProvider(http_client=http_client),
        settings=settings,
    )


def _create_nvidia_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient,
) -> "OpenAIChatModel":
    """Create NVIDIA NIM model.

    Uses NVIDIA's OpenAI-compatible API via the ``NVIDIA_BASE_URL`` env var
    (defaults to ``https://integrate.api.nvidia.com/v1``).

    Args:
        config: LLM model configuration.
        http_client: Async HTTP client with retry logic.

    Returns:
        Configured OpenAIChatModel instance pointing at NVIDIA NIM endpoint.
    """
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415

    settings = _build_openai_settings(config)
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return OpenAIChatModel(
        model_name=config.model,
        provider=OpenAIProvider(base_url=base_url, http_client=http_client),
        settings=settings,
    )


_PROVIDER_FACTORIES = {
    "openai": _create_openai_model,
    "azure": _create_azure_model,
    "anthropic": _create_anthropic_model,
    "google-gla": _create_google_model,
    "mistral": _create_mistral_model,
    "nvidia": _create_nvidia_model,
}


def create_model(
    config: ModelConfig,
    http_client: httpx.AsyncClient | None = None,
) -> Model:
    """Create pydantic-ai model from configuration.

    Factory function that instantiates the appropriate pydantic-ai model
    based on provider configuration. Supports provider-agnostic configuration
    for maximum flexibility.

    Supported Providers:
        - openai: OpenAI models (GPT-4, GPT-4o, o1, etc.)
        - azure: Azure OpenAI Service
        - anthropic: Anthropic models (Claude 3.5 Sonnet, etc.)
        - google-gla: Google Gemini models
        - mistral: Mistral AI models
        - nvidia: NVIDIA NIM models

    Args:
        config: LLM model configuration.
        http_client: Optional async HTTP client with retry logic. If None,
            creates a default client via ``create_http_client()``.

    Returns:
        Configured pydantic-ai Model instance.

    Raises:
        ValueError: If provider is not supported.

    Example:
        >>> from akgentic.llm import ModelConfig, create_model
        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> model = create_model(config)
        >>> # Use with pydantic-ai Agent
        >>> from pydantic_ai import Agent
        >>> agent = Agent(model, system_prompt="You are helpful")
        >>> result = await agent.run("Hello!")

    Note:
        The reasoning_effort parameter is only supported by OpenAI models
        (e.g., o1 series). It is mapped to ``openai_reasoning_effort`` in
        model settings and ignored for other providers.
    """
    if http_client is None:
        http_client = create_http_client()

    factory = _PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        supported = ", ".join(_PROVIDER_FACTORIES.keys())
        raise ValueError(f"Unsupported provider: {config.provider}. Supported providers: {supported}")

    return factory(config, http_client)
