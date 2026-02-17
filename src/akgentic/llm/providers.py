"""Provider factory with HTTP retry logic."""

# TODO: Implementation to be completed by Dev agent
# This is a skeleton file created by Architect

from typing import Any

import httpx

from .config import ModelConfig


def create_http_client(
    timeout_s: float = 120.0,
    max_attempts: int = 5,
    exp_multiplier: float = 0.5,
    exp_max_s: float = 60.0,
    retry_after_cap_s: float = 300.0,
) -> httpx.AsyncClient:
    """Create HTTP client with V1 retry logic.

    Retries on:
    - 429 (rate limit)
    - 5xx (server errors)

    Respects Retry-After header with exponential backoff fallback.

    Args:
        timeout_s: Request timeout in seconds
        max_attempts: Maximum retry attempts
        exp_multiplier: Exponential backoff multiplier
        exp_max_s: Maximum backoff time
        retry_after_cap_s: Maximum time to wait for Retry-After header

    Returns:
        Configured AsyncClient with retry transport

    TODO: Complete implementation following architecture.md specifications
    """
    raise NotImplementedError("create_http_client implementation pending")


def create_model(config: ModelConfig, http_client: httpx.AsyncClient) -> Any:
    """Create pydantic-ai model from configuration.

    Args:
        config: Model configuration
        http_client: HTTP client with retry logic

    Returns:
        pydantic-ai Model instance

    Raises:
        ValueError: If provider is not supported

    TODO: Complete implementation following architecture.md specifications
    """
    raise NotImplementedError("create_model implementation pending")
