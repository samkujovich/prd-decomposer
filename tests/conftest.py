"""Shared test fixtures for PRD Decomposer tests."""

import json
from unittest.mock import MagicMock

import pytest

import prd_decomposer.config as config_module
import prd_decomposer.server as server_module
from prd_decomposer.config import Settings
from prd_decomposer.server import RateLimiter


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singletons after each test to prevent state leakage."""
    yield
    server_module._client = None
    server_module._rate_limiter = None
    config_module._settings = None


@pytest.fixture
def mock_client_factory():
    """Factory fixture that creates a mock OpenAI client with a canned response."""

    def _make(response_data: dict) -> MagicMock:
        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(response_data)))],
            usage=mock_usage,
        )
        return mock_client

    return _make


@pytest.fixture
def fast_settings():
    """Settings with minimal retry delay for fast tests."""
    return Settings(initial_retry_delay=0.01)


@pytest.fixture
def permissive_rate_limiter():
    """Rate limiter that won't interfere with tests."""
    return RateLimiter(max_calls=9999, window_seconds=1)
