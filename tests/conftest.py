"""Shared test fixtures for PRD Decomposer tests."""

import json
from unittest.mock import MagicMock

import pytest

import prd_decomposer.config as config_module
import prd_decomposer.server as server_module
from prd_decomposer.log import correlation_id
from prd_decomposer.server import CircuitBreaker, RateLimiter


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singletons after each test to prevent state leakage."""
    yield
    server_module._client = None
    server_module._rate_limiter = None
    server_module._circuit_breaker = None
    config_module._settings = None
    correlation_id.set("")


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
def make_llm_response():
    """Factory to build a mock LLM response object."""

    def _make(data: dict, prompt_tokens=100, completion_tokens=50, total_tokens=150):
        mock_usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(data)))],
            usage=mock_usage,
        )

    return _make


@pytest.fixture
def permissive_rate_limiter():
    """Rate limiter that won't interfere with tests."""
    return RateLimiter(max_calls=9999, window_seconds=1)


@pytest.fixture
def permissive_circuit_breaker():
    """Circuit breaker that won't interfere with tests."""
    return CircuitBreaker(failure_threshold=9999, reset_timeout=1.0)


@pytest.fixture
def allowed_tmp_path(tmp_path):
    """Temporarily add tmp_path to ALLOWED_DIRECTORIES."""
    server_module.ALLOWED_DIRECTORIES.append(tmp_path)
    yield tmp_path
    server_module.ALLOWED_DIRECTORIES = [
        d for d in server_module.ALLOWED_DIRECTORIES if d != tmp_path
    ]


@pytest.fixture
def sample_requirement():
    """Minimal valid requirement dict."""
    return {
        "id": "REQ-001",
        "title": "Test",
        "description": "Test",
        "acceptance_criteria": [],
        "dependencies": [],
        "ambiguity_flags": [],
        "priority": "high",
    }


@pytest.fixture
def sample_input_requirements(sample_requirement):
    """Minimal valid StructuredRequirements dict."""
    return {
        "requirements": [sample_requirement],
        "summary": "Test",
        "source_hash": "12345678",
    }


@pytest.fixture
def sample_epic_response():
    """Minimal valid epic LLM response."""
    return {
        "epics": [
            {"title": "Epic", "description": "Desc", "stories": [], "labels": []}
        ]
    }
