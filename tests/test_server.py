"""Tests for MCP server tools with mocked LLM calls."""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from arcade_core.errors import FatalToolError
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from prd_decomposer.config import Settings
from prd_decomposer.log import correlation_id
from prd_decomposer.models import SizingRubric
from prd_decomposer.prompts import PROMPT_VERSION
from prd_decomposer.server import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    LLMError,
    RateLimiter,
    RateLimitExceededError,
    _analyze_prd_impl,
    _call_llm_with_retry,
    _decompose_to_tickets_impl,
    _is_path_allowed,
    analyze_prd,
    decompose_to_tickets,
    export_tickets,
    get_circuit_breaker,
    get_client,
    get_rate_limiter,
    health_check,
    read_file,
)


class TestReadFile:
    """Tests for the read_file tool."""

    def test_read_file_returns_content(self, allowed_tmp_path):
        """Verify read_file returns file contents."""
        test_file = allowed_tmp_path / "test.md"
        test_file.write_text("# Test PRD\n\nThis is a test.")

        result = read_file(str(test_file))

        assert result == "# Test PRD\n\nThis is a test."

    def test_read_file_nonexistent_raises_error(self):
        """Verify read_file raises FatalToolError for missing files within allowed dirs."""
        with pytest.raises(FatalToolError):
            read_file("nonexistent_file_in_cwd.md")

    @pytest.mark.parametrize("path", [
        "/etc/passwd",
        "../../../etc/passwd",
        pytest.param(os.path.expanduser("~/.ssh/id_rsa"), id="home_directory"),
    ], ids=["absolute_outside", "parent_traversal", "home_directory"])
    def test_read_file_blocked_paths(self, path):
        """Verify read_file blocks path traversal attempts."""
        with pytest.raises(FatalToolError):
            read_file(path)


class TestPathValidation:
    """Tests for path validation security."""

    def test_is_path_allowed_cwd(self):
        """Verify paths within cwd are allowed."""
        cwd = Path.cwd()
        assert _is_path_allowed(cwd / "somefile.md")

    def test_is_path_allowed_rejects_outside(self):
        """Verify paths outside allowed directories are rejected."""
        assert not _is_path_allowed(Path("/etc/passwd"))
        assert not _is_path_allowed(Path("/tmp/secret.txt"))

    def test_is_path_allowed_handles_invalid_path(self):
        """Verify _is_path_allowed handles paths that can't be resolved."""
        with patch.object(Path, "resolve", side_effect=OSError("Permission denied")):
            result = _is_path_allowed(Path("/some/path"))
            assert result is False


class TestReadFileEdgeCases:
    """Additional edge case tests for read_file."""

    def test_read_file_directory_raises_error(self, allowed_tmp_path):
        """Verify read_file raises error when path is a directory."""
        with pytest.raises(FatalToolError):
            read_file(str(allowed_tmp_path))

    def test_read_file_symlink_outside_allowed_blocked(self, allowed_tmp_path):
        """Verify symlink pointing outside allowed directories is rejected."""
        link = allowed_tmp_path / "sneaky.md"
        link.symlink_to("/etc/passwd")
        with pytest.raises(FatalToolError):
            read_file(str(link))

    def test_read_file_binary_file_raises_error(self, allowed_tmp_path):
        """Verify read_file raises FatalToolError for binary files."""
        binary_file = allowed_tmp_path / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\xfd")
        with pytest.raises(FatalToolError):
            read_file(str(binary_file))


class TestGetClient:
    """Tests for the lazy OpenAI client initialization."""

    def test_get_client_returns_openai_client(self):
        """Verify get_client returns an OpenAI client instance."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = get_client()
            assert client is mock_client
            mock_openai.assert_called_once()

    def test_get_client_reuses_existing_client(self):
        """Verify get_client returns cached client on subsequent calls."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client1 = get_client()
            client2 = get_client()

            assert client1 is client2
            mock_openai.assert_called_once()

    def test_get_client_thread_safe(self):
        """Verify concurrent get_client calls create only one OpenAI instance."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            results: list = []
            barrier = threading.Barrier(10)

            def worker():
                barrier.wait()
                results.append(get_client())

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 10
            assert all(r is results[0] for r in results)
            mock_openai.assert_called_once()


class TestLLMRetry:
    """Tests for LLM retry logic."""

    def test_call_llm_with_retry_success(self, mock_client_factory, permissive_rate_limiter):
        """Verify successful LLM call returns data and usage."""
        mock_client = mock_client_factory({"result": "success"})

        data, usage = _call_llm_with_retry(
            [{"role": "user", "content": "test"}],
            temperature=0.2,
            client=mock_client,
            rate_limiter=permissive_rate_limiter,
        )

        assert data == {"result": "success"}
        assert usage["total_tokens"] == 150

    def test_call_llm_with_retry_empty_response(self, permissive_rate_limiter):
        """Verify LLMError raised for empty response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )

        with pytest.raises(LLMError, match="empty response"):
            _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
            )

    def test_call_llm_with_retry_invalid_json(self, permissive_rate_limiter):
        """Verify LLMError raised for invalid JSON."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json"))]
        )

        with pytest.raises(LLMError, match="invalid JSON"):
            _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
            )

    def test_call_llm_with_retry_rate_limit_then_success(
        self, make_llm_response, permissive_rate_limiter
    ):
        """Verify retry on RateLimitError eventually succeeds."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded", response=MagicMock(status_code=429), body=None
            ),
            make_llm_response({"result": "success"}),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            data, _usage = _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
            )

        assert data == {"result": "success"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_connection_error_then_success(
        self, make_llm_response, permissive_rate_limiter
    ):
        """Verify retry on APIConnectionError eventually succeeds."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            make_llm_response({"result": "success"}),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            data, _usage = _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
            )

        assert data == {"result": "success"}

    def test_call_llm_with_retry_api_error_4xx_no_retry(self, permissive_rate_limiter):
        """Verify 4xx APIError raises immediately without retry."""
        mock_client = MagicMock()
        error = APIError(message="Bad request", request=MagicMock(), body=None)
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
            )

        assert mock_client.chat.completions.create.call_count == 1

    def test_call_llm_with_retry_api_error_no_status_code(
        self, make_llm_response, permissive_rate_limiter
    ):
        """Verify APIError without status_code attribute retries."""
        error = APIError(message="Unknown error", request=MagicMock(), body=None)
        if hasattr(error, "status_code"):
            delattr(error, "status_code")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            error,
            make_llm_response({"result": "success"}),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            data, _usage = _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
            )

        assert data == {"result": "success"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_api_error_5xx_retries(self, permissive_rate_limiter):
        """Verify 5xx APIError retries then fails."""
        mock_client = MagicMock()
        error = APIError(message="Server error", request=MagicMock(), body=None)
        error.status_code = 500
        mock_client.chat.completions.create.side_effect = error

        settings = Settings(max_retries=3, initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(LLMError, match="failed after 3 attempts"):
                _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                )

        assert mock_client.chat.completions.create.call_count == 3

    def test_call_llm_with_retry_all_retries_exhausted(self, permissive_rate_limiter):
        """Verify LLMError raised after all retries exhausted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(max_retries=2, initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(LLMError, match="failed after 2 attempts"):
                _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                )

    def test_call_llm_with_retry_timeout_then_success(
        self, make_llm_response, permissive_rate_limiter
    ):
        """Verify retry on APITimeoutError eventually succeeds."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            make_llm_response({"result": "success"}),
        ]

        settings = Settings(initial_retry_delay=0.01, llm_timeout=30.0)
        with patch("prd_decomposer.server.time.sleep"):
            data, _usage = _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
            )

        assert data == {"result": "success"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_timeout_all_retries_exhausted(self, permissive_rate_limiter):
        """Verify LLMError raised after all timeout retries exhausted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        settings = Settings(max_retries=2, initial_retry_delay=0.01, llm_timeout=30.0)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(LLMError, match="failed after 2 attempts"):
                _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                )

        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_uses_timeout_setting(self):
        """Verify timeout setting is passed to OpenAI API call."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        settings = Settings(llm_timeout=45.0)
        _call_llm_with_retry(
            [{"role": "user", "content": "test"}],
            temperature=0.2,
            client=mock_client,
            settings=settings,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["timeout"] == 45.0


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_allows_calls_within_limit(self):
        """Verify rate limiter allows calls within the configured limit."""
        limiter = RateLimiter(max_calls=5, window_seconds=60)

        for _ in range(5):
            limiter.check_and_record()

    def test_rate_limiter_blocks_calls_exceeding_limit(self):
        """Verify rate limiter blocks calls that exceed the limit."""
        limiter = RateLimiter(max_calls=3, window_seconds=60)

        for _ in range(3):
            limiter.check_and_record()

        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
            limiter.check_and_record()

    def test_rate_limiter_resets_after_window(self):
        """Verify rate limiter allows calls after window expires."""
        limiter = RateLimiter(max_calls=2, window_seconds=1)

        base_time = 1000.0
        with patch("prd_decomposer.server.time.time", return_value=base_time):
            limiter.check_and_record()
            limiter.check_and_record()

        with patch("prd_decomposer.server.time.time", return_value=base_time + 1.1):
            limiter.check_and_record()

    def test_rate_limiter_reset_clears_calls(self):
        """Verify reset() clears the call history."""
        limiter = RateLimiter(max_calls=2, window_seconds=60)

        limiter.check_and_record()
        limiter.check_and_record()

        limiter.reset()

        limiter.check_and_record()

    def test_call_llm_with_retry_respects_rate_limit(self):
        """Verify _call_llm_with_retry raises when rate limit exceeded."""
        limiter = RateLimiter(max_calls=1, window_seconds=60)
        limiter.check_and_record()

        mock_client = MagicMock()

        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
            _call_llm_with_retry(
                [{"role": "user", "content": "test"}],
                temperature=0.2,
                client=mock_client,
                rate_limiter=limiter,
            )

        mock_client.chat.completions.create.assert_not_called()


class TestAnalyzePrd:
    """Tests for the analyze_prd tool."""

    def test_analyze_prd_returns_structured_requirements(self, mock_client_factory):
        """Verify analyze_prd returns validated StructuredRequirements with metadata."""
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "User login",
                    "description": "Users must be able to log in",
                    "acceptance_criteria": ["Login form exists"],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high",
                }
            ],
            "summary": "Authentication system PRD",
            "source_hash": "abc12345",
        }
        mock_client = mock_client_factory(mock_response)

        result = _analyze_prd_impl("Test PRD content", client=mock_client)

        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["id"] == "REQ-001"
        assert result["summary"] == "Authentication system PRD"
        assert len(result["source_hash"]) == 8
        assert all(c in "0123456789abcdef" for c in result["source_hash"])
        assert "_metadata" in result
        assert "usage" in result["_metadata"]
        assert "prompt_version" in result["_metadata"]

    def test_analyze_prd_generates_source_hash(self, mock_client_factory):
        """Verify analyze_prd generates a hash from the input text."""
        mock_response = {"requirements": [], "summary": "Empty PRD", "source_hash": "ignored"}
        mock_client = mock_client_factory(mock_response)

        result1 = _analyze_prd_impl("PRD content A", client=mock_client)
        result2 = _analyze_prd_impl("PRD content B", client=mock_client)

        assert result1["source_hash"] != result2["source_hash"]

    def test_analyze_prd_with_ambiguity_flags(self, mock_client_factory):
        """Verify analyze_prd preserves ambiguity flags from LLM response."""
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Fast API",
                    "description": "The API should be fast",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [
                        {
                            "category": "vague_quantifier",
                            "issue": "'fast' without metrics",
                            "severity": "warning",
                            "suggested_action": "Define specific latency target",
                        }
                    ],
                    "priority": "high",
                }
            ],
            "summary": "Vague PRD",
            "source_hash": "12345678",
        }
        mock_client = mock_client_factory(mock_response)

        result = _analyze_prd_impl("The API should be fast", client=mock_client)

        flags = result["requirements"][0]["ambiguity_flags"]
        assert len(flags) == 1
        assert flags[0]["category"] == "vague_quantifier"
        assert flags[0]["issue"] == "'fast' without metrics"

    def test_analyze_prd_validates_llm_response(self, mock_client_factory):
        """Verify analyze_prd raises RuntimeError for invalid LLM response."""
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }
        mock_client = mock_client_factory(mock_response)

        with pytest.raises(RuntimeError, match="LLM returned invalid structure"):
            _analyze_prd_impl("Test PRD", client=mock_client)

    def test_analyze_prd_llm_error_propagates(self):
        """Verify analyze_prd raises RuntimeError when LLM call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to analyze PRD"):
                _analyze_prd_impl("Test PRD", client=mock_client, settings=settings)

    def test_analyze_prd_rejects_oversized_input(self):
        """Verify analyze_prd raises ValueError when PRD exceeds max length."""
        oversized_prd = "x" * 2000
        settings = Settings(max_prd_length=1000)
        mock_client = MagicMock()

        with pytest.raises(ValueError, match="exceeds maximum length"):
            _analyze_prd_impl(oversized_prd, client=mock_client, settings=settings)

    def test_analyze_prd_accepts_input_at_max_length(self, mock_client_factory):
        """Verify analyze_prd accepts PRD exactly at max length."""
        mock_response = {
            "requirements": [],
            "summary": "Empty PRD",
            "source_hash": "12345678",
        }
        mock_client = mock_client_factory(mock_response)

        prd_at_limit = "x" * 1000
        settings = Settings(max_prd_length=1000)

        result = _analyze_prd_impl(prd_at_limit, client=mock_client, settings=settings)
        assert "requirements" in result

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n\t"], ids=["empty", "whitespace", "newlines"])
    def test_analyze_prd_rejects_empty_input(self, empty_input):
        """Verify analyze_prd raises ValueError for empty/whitespace input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _analyze_prd_impl(empty_input, client=MagicMock())


class TestDecomposeToTickets:
    """Tests for the decompose_to_tickets tool."""

    def test_decompose_to_tickets_returns_ticket_collection(self, mock_client_factory):
        """Verify decompose_to_tickets returns validated TicketCollection."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "User login",
                    "description": "Users must log in",
                    "acceptance_criteria": ["Login works"],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high",
                }
            ],
            "summary": "Auth PRD",
            "source_hash": "abc12345",
        }

        mock_response = {
            "epics": [
                {
                    "title": "Authentication Epic",
                    "description": "All auth features",
                    "stories": [
                        {
                            "title": "Implement login endpoint",
                            "description": "Create POST /login",
                            "acceptance_criteria": ["Returns JWT"],
                            "size": "M",
                            "labels": ["backend", "auth"],
                            "requirement_ids": ["REQ-001"],
                        }
                    ],
                    "labels": ["auth"],
                }
            ]
        }
        mock_client = mock_client_factory(mock_response)

        result = _decompose_to_tickets_impl(input_requirements, client=mock_client)

        assert "epics" in result
        assert len(result["epics"]) == 1
        assert result["epics"][0]["title"] == "Authentication Epic"
        assert len(result["epics"][0]["stories"]) == 1
        assert result["epics"][0]["stories"][0]["size"] == "M"

    def test_decompose_to_tickets_adds_metadata(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify decompose_to_tickets adds generation metadata with usage."""
        mock_client = mock_client_factory(sample_epic_response)

        result = _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        assert "metadata" in result
        assert "generated_at" in result["metadata"]
        assert result["metadata"]["model"] == "gpt-4o"
        assert result["metadata"]["prompt_version"] == PROMPT_VERSION
        assert result["metadata"]["requirement_count"] == 1
        assert result["metadata"]["story_count"] == 0
        assert "usage" in result["metadata"]
        assert result["metadata"]["usage"]["total_tokens"] == 150

    def test_decompose_to_tickets_counts_stories(
        self, sample_input_requirements, mock_client_factory
    ):
        """Verify decompose_to_tickets correctly counts total stories."""
        mock_response = {
            "epics": [
                {
                    "title": "Epic 1",
                    "description": "Desc",
                    "stories": [
                        {
                            "title": "S1",
                            "description": "D",
                            "acceptance_criteria": [],
                            "size": "S",
                            "labels": [],
                            "requirement_ids": [],
                        },
                        {
                            "title": "S2",
                            "description": "D",
                            "acceptance_criteria": [],
                            "size": "S",
                            "labels": [],
                            "requirement_ids": [],
                        },
                    ],
                    "labels": [],
                },
                {
                    "title": "Epic 2",
                    "description": "Desc",
                    "stories": [
                        {
                            "title": "S3",
                            "description": "D",
                            "acceptance_criteria": [],
                            "size": "M",
                            "labels": [],
                            "requirement_ids": [],
                        }
                    ],
                    "labels": [],
                },
            ]
        }
        mock_client = mock_client_factory(mock_response)

        result = _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        assert result["metadata"]["story_count"] == 3

    def test_decompose_to_tickets_validates_input(self):
        """Verify decompose_to_tickets raises ValueError for invalid input."""
        invalid_input = {"requirements": [{"id": "REQ-001"}]}

        with pytest.raises(ValueError, match="Invalid requirements structure"):
            _decompose_to_tickets_impl(invalid_input, client=MagicMock())

    @pytest.mark.parametrize("bad_input,match", [
        (None, "cannot be empty"),
        ("", "cannot be empty"),
        ({}, "cannot be empty"),
    ], ids=["none", "empty_string", "empty_dict"])
    def test_decompose_to_tickets_rejects_empty_input(self, bad_input, match):
        """Verify decompose_to_tickets raises ValueError for empty inputs."""
        with pytest.raises(ValueError, match=match):
            _decompose_to_tickets_impl(bad_input, client=MagicMock())

    def test_decompose_to_tickets_strips_internal_metadata(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify decompose_to_tickets handles _metadata from analyze_prd."""
        sample_input_requirements["_metadata"] = {"prompt_version": "1.0.0", "usage": {}}
        mock_client = mock_client_factory(sample_epic_response)

        result = _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        assert "epics" in result

    def test_decompose_to_tickets_validates_llm_response(
        self, sample_input_requirements, mock_client_factory
    ):
        """Verify decompose_to_tickets raises RuntimeError for invalid LLM response."""
        mock_response = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "acceptance_criteria": [],
                            "size": "XL",  # Invalid - must be S/M/L
                            "labels": [],
                            "requirement_ids": [],
                        }
                    ],
                    "labels": [],
                }
            ]
        }
        mock_client = mock_client_factory(mock_response)

        with pytest.raises(RuntimeError, match="LLM returned invalid ticket structure"):
            _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

    def test_decompose_to_tickets_llm_missing_epics_key(
        self, sample_input_requirements, mock_client_factory
    ):
        """Verify decompose raises RuntimeError when LLM omits epics key."""
        mock_client = mock_client_factory({"some_other_key": []})

        with pytest.raises(RuntimeError, match="LLM returned invalid ticket structure"):
            _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

    def test_decompose_to_tickets_string_requirements(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify decompose_to_tickets handles string (JSON) requirements."""
        mock_client = mock_client_factory(sample_epic_response)

        result = _decompose_to_tickets_impl(
            json.dumps(sample_input_requirements), client=mock_client
        )

        assert "epics" in result

    def test_decompose_to_tickets_json_array_raises_error(self):
        """Verify decompose_to_tickets rejects JSON array strings."""
        with pytest.raises(ValueError, match="must be an object"):
            _decompose_to_tickets_impl("[1, 2, 3]", client=MagicMock())

    def test_decompose_to_tickets_invalid_json_string(self):
        """Verify decompose_to_tickets raises for invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _decompose_to_tickets_impl("not valid json {", client=MagicMock())

    def test_decompose_to_tickets_llm_error_propagates(self, sample_input_requirements):
        """Verify decompose_to_tickets raises RuntimeError when LLM call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to decompose"):
                _decompose_to_tickets_impl(
                    sample_input_requirements, client=mock_client, settings=settings
                )

    def test_decompose_with_default_sizing_rubric(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify default sizing rubric is used when none provided."""
        mock_client = mock_client_factory(sample_epic_response)

        # Capture the prompt sent to the LLM
        _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        # Check that the call was made
        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]

        # Verify default rubric text is in the prompt
        assert "Less than 1 day" in content
        assert "1-3 days" in content
        assert "3-5 days" in content

    def test_decompose_with_custom_sizing_rubric_model(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify custom SizingRubric model is used in prompt."""
        mock_client = mock_client_factory(sample_epic_response)

        from prd_decomposer.models import SizeDefinition

        custom_rubric = SizingRubric(
            small=SizeDefinition(label="S", duration="Up to 4 hours", scope="Single file", risk="Minimal"),
            medium=SizeDefinition(label="M", duration="1-2 days", scope="Few modules", risk="Low"),
            large=SizeDefinition(label="L", duration="1 week", scope="Cross-team", risk="High"),
        )

        _decompose_to_tickets_impl(
            sample_input_requirements, client=mock_client, sizing_rubric=custom_rubric
        )

        # Check that the custom rubric was used
        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]

        assert "Up to 4 hours" in content
        assert "Single file" in content
        assert "1 week" in content
        assert "Cross-team" in content

    def test_decompose_with_custom_rubric_string(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify raw string rubric is used as-is."""
        mock_client = mock_client_factory(sample_epic_response)

        custom_rubric_text = """   - S: Very quick task
   - M: A few hours to a day
   - L: Multiple days of work"""

        _decompose_to_tickets_impl(
            sample_input_requirements, client=mock_client, sizing_rubric=custom_rubric_text
        )

        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]

        assert "Very quick task" in content
        assert "Multiple days of work" in content

    def test_decompose_tool_accepts_sizing_rubric_json(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify decompose_to_tickets MCP tool parses sizing_rubric JSON."""
        mock_client = mock_client_factory(sample_epic_response)

        rubric_json = json.dumps({
            "small": {"label": "S", "duration": "4h", "scope": "tiny", "risk": "none"},
            "medium": {"label": "M", "duration": "2d", "scope": "small", "risk": "low"},
            "large": {"label": "L", "duration": "5d", "scope": "big", "risk": "high"},
        })

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric=rubric_json
            )

        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]

        assert "4h" in content or "tiny" in content  # Custom rubric was used

    def test_decompose_tool_invalid_sizing_rubric_raises(self, sample_input_requirements):
        """Verify decompose_to_tickets raises for invalid rubric JSON."""
        with pytest.raises(FatalToolError, match="Invalid sizing_rubric"):
            decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric="not valid json"
            )


class TestIntegrationPipeline:
    """Integration tests for the full analyze -> decompose pipeline."""

    def test_full_pipeline_mocked(self, make_llm_response):
        """Test the full pipeline with mocked LLM calls."""
        analyze_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "User authentication",
                    "description": "Users must log in securely",
                    "acceptance_criteria": ["Login form exists", "JWT issued"],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high",
                }
            ],
            "summary": "Auth system",
            "source_hash": "abc12345",
        }

        decompose_response = {
            "epics": [
                {
                    "title": "Authentication",
                    "description": "User auth features",
                    "stories": [
                        {
                            "title": "Implement login",
                            "description": "Create login endpoint",
                            "acceptance_criteria": ["Returns JWT"],
                            "size": "M",
                            "labels": ["backend"],
                            "requirement_ids": ["REQ-001"],
                        }
                    ],
                    "labels": ["auth"],
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            make_llm_response(analyze_response),
            make_llm_response(decompose_response),
        ]

        requirements = _analyze_prd_impl(
            "# Sample PRD\n\nUser auth required.", client=mock_client
        )

        assert "requirements" in requirements
        assert len(requirements["requirements"]) == 1
        assert requirements["requirements"][0]["id"] == "REQ-001"
        assert "_metadata" in requirements

        tickets = _decompose_to_tickets_impl(requirements, client=mock_client)

        assert "epics" in tickets
        assert len(tickets["epics"]) == 1
        assert tickets["epics"][0]["stories"][0]["requirement_ids"] == ["REQ-001"]
        # Verify story requirement_ids reference IDs from analyze output
        analyze_req_ids = {r["id"] for r in requirements["requirements"]}
        story_req_ids = set(tickets["epics"][0]["stories"][0]["requirement_ids"])
        assert story_req_ids <= analyze_req_ids
        assert "metadata" in tickets
        assert tickets["metadata"]["requirement_count"] == 1


class TestServerLogging:
    """Tests for structured logging in server tool implementations."""

    def test_analyze_prd_sets_correlation_id(self, caplog, mock_client_factory):
        """Verify _analyze_prd_impl sets a correlation ID."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = mock_client_factory(mock_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _analyze_prd_impl("Test PRD", client=mock_client)

        assert any("Starting PRD analysis" in r.message for r in caplog.records)

    def test_analyze_prd_logs_completion(
        self, caplog, sample_input_requirements, mock_client_factory
    ):
        """Verify _analyze_prd_impl logs completion with requirement count."""
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high",
                }
            ],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = mock_client_factory(mock_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _analyze_prd_impl("Test PRD", client=mock_client)

        messages = [r.message for r in caplog.records]
        assert any("PRD analysis complete" in m for m in messages)

    def test_decompose_logs_start_and_completion(
        self, caplog, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify _decompose_to_tickets_impl logs start and completion."""
        mock_client = mock_client_factory(sample_epic_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        messages = [r.message for r in caplog.records]
        assert any("Starting ticket decomposition" in m for m in messages)
        assert any("Ticket decomposition complete" in m for m in messages)

    def test_call_llm_with_retry_logs_retry_attempts(self, caplog, permissive_rate_limiter):
        """Verify _call_llm_with_retry logs retry attempts."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit", response=MagicMock(status_code=429), body=None
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
                usage=mock_usage,
            ),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            with patch("prd_decomposer.server.time.sleep"):
                _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                )

        messages = [r.message for r in caplog.records]
        assert any("Retrying LLM call" in m for m in messages)

    def test_analyze_prd_logs_error_on_failure(self, caplog):
        """Verify _analyze_prd_impl logs errors when LLM call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01)
        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            with patch("prd_decomposer.server.time.sleep"):
                with pytest.raises(RuntimeError):
                    _analyze_prd_impl("Test PRD", client=mock_client, settings=settings)

        messages = [r.message for r in caplog.records]
        assert any("PRD analysis failed" in m for m in messages)
        error_records = [r for r in caplog.records if "PRD analysis failed" in r.message]
        assert error_records[0].levelno == logging.ERROR

    def test_analyze_prd_correlation_id_format(self, mock_client_factory):
        """Verify correlation ID is an 8-char non-empty hex-like string."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = mock_client_factory(mock_response)

        _analyze_prd_impl("Test PRD", client=mock_client)

        cid = correlation_id.get()
        assert len(cid) == 8
        assert cid.strip() != ""


class TestCallLLMEdgeCases:
    """Edge-case tests for _call_llm_with_retry."""

    def test_call_llm_with_retry_no_usage(self, permissive_rate_limiter):
        """Verify _call_llm_with_retry returns empty usage when response.usage is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"result": "ok"}'))],
            usage=None,
        )

        data, usage = _call_llm_with_retry(
            [{"role": "user", "content": "test"}],
            temperature=0.2,
            client=mock_client,
            rate_limiter=permissive_rate_limiter,
        )

        assert data == {"result": "ok"}
        assert usage == {}

    def test_call_llm_with_retry_exponential_backoff_delays(self, permissive_rate_limiter):
        """Verify retry sleep delays follow exponential backoff with jitter."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(max_retries=4, initial_retry_delay=1.0)
        with patch("prd_decomposer.server.time.sleep") as mock_sleep:
            with pytest.raises(LLMError):
                _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                )

        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # Jitter: delay = base_delay * (0.5 + random()), so range is [0.5x, 1.5x]
        base_delays = [1.0, 2.0, 4.0]
        for actual, base in zip(delays, base_delays):
            assert base * 0.5 <= actual <= base * 1.5


class TestReadFileUTF8:
    """Tests for UTF-8 file reading."""

    def test_read_file_utf8_content(self, allowed_tmp_path):
        """Verify read_file correctly reads non-ASCII UTF-8 content."""
        test_file = allowed_tmp_path / "unicode.md"
        content = "PRD: données résumé 日本語"
        test_file.write_text(content, encoding="utf-8")

        result = read_file(str(test_file))

        assert result == content


class TestGetRateLimiterThreadSafety:
    """Tests for get_rate_limiter thread safety."""

    def test_get_rate_limiter_thread_safe(self):
        """Verify concurrent get_rate_limiter calls create only one instance."""
        results: list = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            results.append(get_rate_limiter())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is results[0] for r in results)


class TestMCPWrappers:
    """Tests for MCP tool wrapper functions (PE-4)."""

    def test_analyze_prd_wrapper_delegates_to_impl(self):
        """Verify analyze_prd delegates to _analyze_prd_impl."""
        with patch("prd_decomposer.server._analyze_prd_impl") as mock_impl:
            mock_impl.return_value = {"requirements": [], "summary": "Test", "source_hash": "abc"}
            result = analyze_prd("Test PRD")
            mock_impl.assert_called_once_with("Test PRD")
            assert result == mock_impl.return_value

    def test_decompose_to_tickets_wrapper_delegates_to_impl(self):
        """Verify decompose_to_tickets delegates to _decompose_to_tickets_impl."""
        with patch("prd_decomposer.server._decompose_to_tickets_impl") as mock_impl:
            mock_impl.return_value = {"epics": [], "metadata": {}}
            result = decompose_to_tickets('{"requirements": []}')
            mock_impl.assert_called_once()
            # Check first positional arg is the requirements JSON
            assert mock_impl.call_args[0][0] == '{"requirements": []}'
            # Check sizing_rubric defaults to None
            assert mock_impl.call_args[1].get("sizing_rubric") is None
            assert result == mock_impl.return_value


class TestCorrelationIDIsolation:
    """Tests for correlation ID isolation across threads (QA-2)."""

    def test_concurrent_analyze_prd_correlation_ids_isolated(self, make_llm_response):
        """Verify each thread gets a unique correlation ID."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }

        captured_ids: list[str] = []
        lock = threading.Lock()

        def worker():
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = make_llm_response(mock_response)
            _analyze_prd_impl("Test PRD", client=mock_client)
            cid = correlation_id.get()
            with lock:
                captured_ids.append(cid)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(captured_ids) == 5
        assert all(len(cid) == 8 for cid in captured_ids)
        assert len(set(captured_ids)) == 5  # All unique


class TestMetadataTimestamps:
    """Tests for ISO timestamp format in metadata (QA-9)."""

    def test_analyze_prd_timestamp_is_valid_iso(self, mock_client_factory):
        """Verify _metadata.analyzed_at is a valid ISO timestamp."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = mock_client_factory(mock_response)

        result = _analyze_prd_impl("Test PRD", client=mock_client)

        ts = result["_metadata"]["analyzed_at"]
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_decompose_timestamp_is_valid_iso(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify metadata.generated_at is a valid ISO timestamp."""
        mock_client = mock_client_factory(sample_epic_response)

        result = _decompose_to_tickets_impl(sample_input_requirements, client=mock_client)

        ts = result["metadata"]["generated_at"]
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None


class TestHealthCheck:
    """Tests for health_check tool."""

    def test_health_check_returns_healthy_when_api_connected(self):
        """Verify health_check returns healthy status when OpenAI API is connected."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[MagicMock(), MagicMock()])

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = health_check()

        assert result["status"] == "healthy"
        assert result["checks"]["openai_api"]["status"] == "connected"
        assert result["checks"]["openai_api"]["models_available"] == 2
        assert result["checks"]["rate_limiter"]["status"] == "ok"
        assert "version" in result
        assert "config" in result

    def test_health_check_returns_unhealthy_when_api_fails(self):
        """Verify health_check returns unhealthy status when OpenAI API fails."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("API connection failed")

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = health_check()

        assert result["status"] == "unhealthy"
        assert result["checks"]["openai_api"]["status"] == "error"
        assert "API connection failed" in result["checks"]["openai_api"]["error"]

    def test_health_check_includes_config_summary(self):
        """Verify health_check returns configuration summary."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = health_check()

        assert "config" in result
        assert "model" in result["config"]
        assert "max_retries" in result["config"]
        assert "llm_timeout" in result["config"]
        assert "max_prd_length" in result["config"]

    def test_health_check_includes_rate_limiter_status(self):
        """Verify health_check returns rate limiter status."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = health_check()

        assert "rate_limiter" in result["checks"]
        assert "current_calls" in result["checks"]["rate_limiter"]
        assert "max_calls" in result["checks"]["rate_limiter"]
        assert "window_seconds" in result["checks"]["rate_limiter"]

    def test_health_check_degraded_in_half_open_state(self):
        """Verify health_check returns degraded when circuit breaker is half-open."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])

        # Create a circuit breaker in half-open state
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        import time
        time.sleep(0.02)  # Wait for half-open

        with (
            patch("prd_decomposer.server.get_client", return_value=mock_client),
            patch("prd_decomposer.server.get_circuit_breaker", return_value=cb),
        ):
            result = health_check()

        # Both component and top-level should be degraded
        assert result["checks"]["circuit_breaker"]["status"] == "degraded"
        assert result["checks"]["circuit_breaker"]["state"] == "half_open"
        assert result["status"] == "degraded"

    def test_health_check_degraded_in_open_state(self):
        """Verify health_check returns degraded when circuit breaker is open."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])

        # Create a circuit breaker in open state
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60)
        cb.record_failure()  # Opens circuit

        with (
            patch("prd_decomposer.server.get_client", return_value=mock_client),
            patch("prd_decomposer.server.get_circuit_breaker", return_value=cb),
        ):
            result = health_check()

        # Both component and top-level should be degraded
        assert result["checks"]["circuit_breaker"]["status"] == "degraded"
        assert result["checks"]["circuit_breaker"]["state"] == "open"
        assert result["status"] == "degraded"


class TestExportTickets:
    """Tests for export_tickets tool."""

    @pytest.fixture
    def sample_tickets(self):
        """Sample ticket collection for export tests."""
        return {
            "epics": [
                {
                    "title": "User Authentication",
                    "description": "Implement secure user authentication",
                    "labels": ["auth", "security"],
                    "stories": [
                        {
                            "title": "Login endpoint",
                            "description": "Create POST /auth/login endpoint",
                            "acceptance_criteria": ["Returns JWT on success", "Returns 401 on failure"],
                            "size": "M",
                            "priority": "high",
                            "labels": ["backend", "api"],
                            "requirement_ids": ["REQ-001"],
                        },
                        {
                            "title": "Login form UI",
                            "description": "Build login form component",
                            "acceptance_criteria": ["Email validation", "Password field masked"],
                            "size": "S",
                            "priority": "medium",
                            "labels": ["frontend"],
                            "requirement_ids": ["REQ-001"],
                        },
                    ],
                }
            ],
            "metadata": {"story_count": 2},
        }

    def test_export_to_csv_returns_valid_csv(self, sample_tickets):
        """Verify export_tickets produces valid CSV."""
        result = export_tickets(json.dumps(sample_tickets), format="csv")

        assert "epic_title,story_title" in result
        assert "User Authentication" in result
        assert "Login endpoint" in result
        assert "Login form UI" in result

    def test_export_to_csv_includes_all_fields(self, sample_tickets):
        """Verify CSV export includes all expected fields."""
        result = export_tickets(json.dumps(sample_tickets), format="csv")
        lines = result.strip().split("\n")

        # Check header
        header = lines[0]
        assert "priority" in header
        assert "size" in header
        assert "requirement_ids" in header

        # Check data row has values
        data_row = lines[1]
        assert "high" in data_row
        assert "M" in data_row
        assert "REQ-001" in data_row

    def test_export_to_jira_returns_valid_json(self, sample_tickets):
        """Verify export_tickets produces valid Jira API payload."""
        result = export_tickets(json.dumps(sample_tickets), format="jira")
        parsed = json.loads(result)

        assert "issueUpdates" in parsed
        assert len(parsed["issueUpdates"]) == 3  # 1 epic + 2 stories

    def test_export_to_jira_maps_priority(self, sample_tickets):
        """Verify Jira export maps priority correctly."""
        result = export_tickets(json.dumps(sample_tickets), format="jira")
        parsed = json.loads(result)

        # Find a story issue (not epic)
        story_issues = [i for i in parsed["issueUpdates"] if i["fields"]["issuetype"]["name"] == "Story"]
        assert len(story_issues) == 2

        # High priority story should have "High" in Jira
        high_priority_story = story_issues[0]
        assert high_priority_story["fields"]["priority"]["name"] == "High"

    def test_export_to_jira_no_metadata_in_issues(self, sample_tickets):
        """Verify Jira issues don't contain _prd_decomposer_metadata (breaks Jira API)."""
        result = export_tickets(json.dumps(sample_tickets), format="jira")
        parsed = json.loads(result)

        # No issue should have _prd_decomposer_metadata
        for issue in parsed["issueUpdates"]:
            assert "_prd_decomposer_metadata" not in issue, \
                "Issue entries should not contain _prd_decomposer_metadata"

        # Payload should be clean - only issueUpdates key (Jira schema compliance)
        assert "_prd_decomposer_metadata" not in parsed, \
            "Jira payload should not contain _prd_decomposer_metadata (breaks Jira REST API)"
        assert list(parsed.keys()) == ["issueUpdates"], \
            "Jira payload should only contain issueUpdates key"

    def test_export_to_yaml_returns_valid_yaml(self, sample_tickets):
        """Verify export_tickets produces valid YAML-like output."""
        result = export_tickets(json.dumps(sample_tickets), format="yaml")

        assert "epics:" in result
        assert "title:" in result
        assert "stories:" in result
        assert "User Authentication" in result

    def test_export_invalid_format_raises(self, sample_tickets):
        """Verify export_tickets raises for invalid format."""
        with pytest.raises(FatalToolError, match="Unsupported format"):
            export_tickets(json.dumps(sample_tickets), format="xml")

    def test_export_invalid_json_raises(self):
        """Verify export_tickets raises for invalid JSON input."""
        with pytest.raises(FatalToolError, match="Invalid JSON"):
            export_tickets("not valid json", format="csv")

    def test_export_missing_epics_raises(self):
        """Verify export_tickets raises when epics key missing."""
        with pytest.raises(FatalToolError, match="missing 'epics'"):
            export_tickets('{"stories": []}', format="csv")

    def test_export_array_json_raises(self):
        """Verify export_tickets raises for JSON array input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('[{"title": "Epic"}]', format="csv")

    def test_export_string_json_raises(self):
        """Verify export_tickets raises for JSON string input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('"just a string"', format="csv")

    def test_export_number_json_raises(self):
        """Verify export_tickets raises for JSON number input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('123', format="csv")

    def test_export_epics_not_list_raises(self):
        """Verify export_tickets raises when epics is not a list."""
        with pytest.raises(FatalToolError, match="'epics' must be a list"):
            export_tickets('{"epics": "not a list"}', format="csv")

    def test_export_epics_contains_non_object_raises(self):
        """Verify export_tickets raises when epics contains non-objects."""
        with pytest.raises(FatalToolError, match=r"epics\[0\] must be an object"):
            export_tickets('{"epics": ["string instead of object"]}', format="csv")

    def test_export_epics_mixed_types_raises(self):
        """Verify export_tickets raises for mixed types in epics array."""
        tickets = {
            "epics": [
                {"title": "Valid Epic", "description": "D", "labels": [], "stories": []},
                123,  # Invalid - not an object
            ]
        }
        with pytest.raises(FatalToolError, match=r"epics\[1\] must be an object"):
            export_tickets(json.dumps(tickets), format="csv")

    def test_export_stories_not_list_raises(self):
        """Verify export_tickets raises when stories is not a list."""
        tickets = {
            "epics": [
                {"title": "Epic", "description": "D", "labels": [], "stories": "not a list"}
            ]
        }
        with pytest.raises(FatalToolError, match=r"epics\[0\].stories must be a list"):
            export_tickets(json.dumps(tickets), format="csv")

    def test_export_story_not_object_raises(self):
        """Verify export_tickets raises when story is not an object."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "D",
                    "labels": [],
                    "stories": ["not an object"],
                }
            ]
        }
        with pytest.raises(FatalToolError, match=r"epics\[0\].stories\[0\] must be an object"):
            export_tickets(json.dumps(tickets), format="csv")

    def test_export_mixed_story_types_raises(self):
        """Verify export_tickets raises for mixed types in stories array."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "D",
                    "labels": [],
                    "stories": [
                        {"title": "Valid Story", "description": "D", "size": "S"},
                        123,  # Invalid
                    ],
                }
            ]
        }
        with pytest.raises(FatalToolError, match=r"epics\[0\].stories\[1\] must be an object"):
            export_tickets(json.dumps(tickets), format="csv")


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_circuit_breaker_starts_closed(self):
        """Verify new circuit breaker is in closed state."""
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_circuit_breaker_opens_after_threshold(self):
        """Verify circuit opens after failure threshold exceeded."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"

    def test_circuit_breaker_blocks_when_open(self):
        """Verify open circuit breaker blocks requests."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure()  # Opens circuit

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.allow_request()
        assert exc_info.value.retry_after > 0

    def test_circuit_breaker_success_resets_count(self):
        """Verify success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_circuit_breaker_half_open_after_timeout(self):
        """Verify circuit transitions to half-open after reset timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        # Wait for reset timeout
        import time
        time.sleep(0.02)

        assert cb.state == "half_open"

    def test_circuit_breaker_half_open_allows_one_request(self):
        """Verify half-open state allows limited test requests."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01, half_open_max_calls=1)
        cb.record_failure()

        import time
        time.sleep(0.02)

        assert cb.allow_request() is True
        # Second request should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            cb.allow_request()

    def test_circuit_breaker_closes_on_half_open_success(self):
        """Verify successful call in half-open state closes circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        import time
        time.sleep(0.02)

        cb.allow_request()
        cb.record_success()

        assert cb.state == "closed"

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Verify failure in half-open state reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        import time
        time.sleep(0.02)

        cb.allow_request()
        cb.record_failure()

        assert cb.state == "open"

    def test_circuit_breaker_reset(self):
        """Verify reset() clears all state."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with LLM calls."""

    def test_call_llm_with_retry_respects_circuit_breaker(
        self, permissive_rate_limiter
    ):
        """Verify _call_llm_with_retry raises when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure()  # Open the circuit

        with pytest.raises(CircuitBreakerOpenError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=MagicMock(),
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

    def test_call_llm_with_retry_records_success(
        self, mock_client_factory, permissive_rate_limiter
    ):
        """Verify successful LLM call records success with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()  # Add one failure
        assert cb._failure_count == 1

        mock_client = mock_client_factory({"result": "ok"})
        _call_llm_with_retry(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            client=mock_client,
            rate_limiter=permissive_rate_limiter,
            circuit_breaker=cb,
        )

        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_call_llm_with_retry_records_failure(self, permissive_rate_limiter):
        """Verify failed LLM call records failure with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=5)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01, max_retries=1)

        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(LLMError):
                _call_llm_with_retry(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.5,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                    circuit_breaker=cb,
                )

        assert cb._failure_count == 1

    def test_health_check_includes_circuit_breaker(self):
        """Verify health_check includes circuit breaker status."""
        with patch("prd_decomposer.server.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.list.return_value = MagicMock(data=[])
            mock_get_client.return_value = mock_client

            result = health_check()

            assert "circuit_breaker" in result["checks"]
            assert result["checks"]["circuit_breaker"]["state"] == "closed"
            assert "circuit_breaker_failure_threshold" in result["config"]

    def test_get_circuit_breaker_returns_singleton(self):
        """Verify get_circuit_breaker returns same instance."""
        cb1 = get_circuit_breaker()
        cb2 = get_circuit_breaker()
        assert cb1 is cb2


class TestCircuitBreakerBugFixes:
    """Tests for specific circuit breaker bug fixes."""

    def test_half_open_failure_resets_slot_count(self):
        """Verify half-open failure properly resets _half_open_calls (bug fix #1)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        import time
        time.sleep(0.02)  # Wait for half-open

        # Take a half-open slot
        cb.allow_request()
        assert cb._half_open_calls == 1

        # Fail the probe - should reset _half_open_calls
        cb.record_failure()
        assert cb._half_open_calls == 0
        assert cb.state == "open"

    def test_release_half_open_slot(self):
        """Verify release_half_open_slot decrements counter (bug fix #2)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        import time
        time.sleep(0.02)

        cb.allow_request()
        assert cb._half_open_calls == 1

        cb.release_half_open_slot()
        assert cb._half_open_calls == 0

    def test_rate_limit_releases_half_open_slot(self, permissive_circuit_breaker):
        """Verify rate limit error releases half-open slot (bug fix #2)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        import time
        time.sleep(0.02)

        # Create a rate limiter that will fail on second call
        rate_limiter = RateLimiter(max_calls=1, window_seconds=60)
        rate_limiter.check_and_record()  # Use up the one allowed call

        with pytest.raises(RateLimitExceededError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=MagicMock(),
                rate_limiter=rate_limiter,
                circuit_breaker=cb,
            )

        # Half-open slot should have been released but NOT counted as failure
        assert cb._half_open_calls == 0
        # Still in half-open (rate limit is client-side, not upstream failure)
        assert cb.state == "half_open"

    def test_non_retryable_error_records_failure(self, permissive_rate_limiter):
        """Verify non-retryable LLM errors record circuit breaker failure."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that returns empty response (non-retryable error)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None  # Empty response
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="empty response"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Failure should be recorded
        assert cb._failure_count == 1

    def test_invalid_json_records_failure(self, permissive_rate_limiter):
        """Verify invalid JSON response records circuit breaker failure."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that returns invalid JSON
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json {"
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="invalid JSON"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Failure should be recorded
        assert cb._failure_count == 1

    def test_half_open_non_retryable_error_reopens_circuit(self, permissive_rate_limiter):
        """Verify non-retryable error in half-open state reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        import time
        time.sleep(0.02)  # Wait for half-open

        # Mock client that returns empty response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="empty response"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should be back to open state
        assert cb.state == "open"
        assert cb._half_open_calls == 0

    def test_4xx_error_does_not_count_as_failure(self, permissive_rate_limiter):
        """Verify 4xx client errors don't increment circuit breaker failure count."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that raises 4xx APIError
        mock_client = MagicMock()
        error = APIError(
            message="Bad request",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx should NOT count as failure (client error, not upstream)
        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_5xx_error_counts_as_failure(self, permissive_rate_limiter):
        """Verify 5xx server errors DO increment circuit breaker failure count."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)
        settings = Settings(max_retries=1)  # Single retry to speed up test

        # Mock client that raises 5xx APIError
        mock_client = MagicMock()
        error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 500
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="LLM call failed"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 5xx SHOULD count as failure (upstream error)
        assert cb._failure_count == 1

    def test_half_open_probe_single_attempt_only(self, permissive_rate_limiter):
        """Verify half-open probes only attempt once (no retries)."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        import time
        time.sleep(0.02)  # Wait for half-open
        assert cb.state == "half_open"

        settings = Settings(max_retries=3)  # Would normally retry 3 times

        # Mock client that always fails with retryable error
        mock_client = MagicMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="1 attempts"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should have only made 1 call, not 3
        assert mock_client.chat.completions.create.call_count == 1

    def test_half_open_detected_after_open_timeout(self, permissive_rate_limiter):
        """Verify half-open is detected even when state was 'open' before allow_request()."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        # State is "open" right now, but will transition to half_open on allow_request
        assert cb._state == "open"

        import time
        time.sleep(0.02)  # Wait for reset timeout

        settings = Settings(max_retries=3)

        # Mock client that always fails
        mock_client = MagicMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="1 attempts"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should detect half-open and limit to 1 attempt
        assert mock_client.chat.completions.create.call_count == 1

    def test_4xx_during_half_open_closes_circuit(self, permissive_rate_limiter):
        """Verify 4xx error during half-open probe closes circuit (upstream is responsive)."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        import time
        time.sleep(0.02)  # Wait for half-open
        assert cb.state == "half_open"

        settings = Settings(max_retries=1)

        # Mock client that raises 4xx error
        mock_client = MagicMock()
        error = APIError(
            message="Bad request - invalid input",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx indicates upstream is responsive, so circuit should close
        assert cb.state == "closed", "4xx during half-open should close circuit (upstream responsive)"


class TestJiraPriorityMapping:
    """Tests for Jira priority mapping robustness."""

    def test_map_priority_non_string_int(self):
        """Verify _map_priority_to_jira coerces int to string."""
        from prd_decomposer.server import _map_priority_to_jira
        # Integer priority should be coerced to string
        result = _map_priority_to_jira(1)
        assert result == "Medium"  # Falls through to default since "1" isn't a valid key

    def test_map_priority_non_string_none(self):
        """Verify _map_priority_to_jira handles None."""
        from prd_decomposer.server import _map_priority_to_jira
        result = _map_priority_to_jira(None)
        assert result == "Medium"

    def test_map_priority_non_string_bool(self):
        """Verify _map_priority_to_jira coerces bool to string."""
        from prd_decomposer.server import _map_priority_to_jira
        result = _map_priority_to_jira(True)
        assert result == "Medium"  # Falls through to default

    def test_map_priority_valid_strings(self):
        """Verify _map_priority_to_jira handles valid string priorities."""
        from prd_decomposer.server import _map_priority_to_jira
        assert _map_priority_to_jira("high") == "High"
        assert _map_priority_to_jira("HIGH") == "High"
        assert _map_priority_to_jira("medium") == "Medium"
        assert _map_priority_to_jira("low") == "Low"
        assert _map_priority_to_jira("unknown") == "Medium"


class TestNestedFieldValidation:
    """Tests for nested story field validation in export."""

    def test_validate_string_list_with_integers(self):
        """Verify _validate_string_list coerces integers to strings."""
        from prd_decomposer.server import _validate_string_list
        result = _validate_string_list([1, 2, 3], "labels", "story[0]")
        assert result == ["1", "2", "3"]

    def test_validate_string_list_with_none_items(self):
        """Verify _validate_string_list filters out None items."""
        from prd_decomposer.server import _validate_string_list
        result = _validate_string_list(["a", None, "b"], "labels", "story[0]")
        assert result == ["a", "b"]

    def test_validate_string_list_with_mixed_types(self):
        """Verify _validate_string_list handles mixed types."""
        from prd_decomposer.server import _validate_string_list
        result = _validate_string_list(["a", 1, True, "b"], "labels", "story[0]")
        assert result == ["a", "1", "True", "b"]

    def test_validate_string_list_raises_for_non_list(self):
        """Verify _validate_string_list raises for non-list value."""
        from prd_decomposer.server import _validate_string_list
        with pytest.raises(ValueError, match="must be a list"):
            _validate_string_list("not a list", "labels", "story[0]")

    def test_validate_string_list_handles_none(self):
        """Verify _validate_string_list handles None input."""
        from prd_decomposer.server import _validate_string_list
        result = _validate_string_list(None, "labels", "story[0]")
        assert result == []

    def test_export_with_integer_labels(self):
        """Verify export handles stories with integer labels."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": ["epic-label"],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "acceptance_criteria": ["AC1"],
                            "labels": [123, "valid-label"],  # Integer in labels
                            "size": "S",
                            "requirement_ids": ["REQ-001"],
                        }
                    ],
                }
            ]
        }
        # Should not crash - integers coerced to strings
        result = export_tickets(json.dumps(tickets), format="csv")
        assert "123" in result
        assert "valid-label" in result


class TestCircuitBreaker4xxInClosedState:
    """Tests for circuit breaker 4xx handling in closed state."""

    def test_4xx_in_closed_state_resets_failure_count(self, permissive_rate_limiter):
        """Verify 4xx in closed state resets failure streak (upstream responsive)."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        # First failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1

        # Now simulate 4xx error - should reset failure count
        mock_client = MagicMock()
        error = APIError(
            message="Bad request",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        settings = Settings(max_retries=1)
        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx should reset failure count (upstream is responsive)
        assert cb._failure_count == 0, "4xx should reset failure count"
        assert cb.state == "closed"

    def test_5xx_4xx_5xx_does_not_open_circuit(self, permissive_rate_limiter):
        """Verify 5xx/4xx/5xx sequence doesn't incorrectly open circuit."""
        from prd_decomposer.config import Settings
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)  # Opens after 2 failures

        settings = Settings(max_retries=1)

        # First failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1

        # 4xx - should reset count
        mock_client = MagicMock()
        error_4xx = APIError(message="Bad request", request=MagicMock(), body=None)
        error_4xx.status_code = 400
        mock_client.chat.completions.create.side_effect = error_4xx

        with pytest.raises(LLMError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )
        assert cb._failure_count == 0  # Reset by 4xx

        # Second failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1
        assert cb.state == "closed", "Should not open - only 1 consecutive failure"


class TestScalarFieldValidation:
    """Tests for scalar string field validation in export."""

    def test_validate_string_field_with_string(self):
        """Verify _validate_string_field returns strings unchanged."""
        from prd_decomposer.server import _validate_string_field
        result = _validate_string_field("hello", "title", "epic[0]")
        assert result == "hello"

    def test_validate_string_field_with_integer(self):
        """Verify _validate_string_field coerces integers to strings."""
        from prd_decomposer.server import _validate_string_field
        result = _validate_string_field(123, "title", "epic[0]")
        assert result == "123"

    def test_validate_string_field_with_bool(self):
        """Verify _validate_string_field coerces bools to strings."""
        from prd_decomposer.server import _validate_string_field
        result = _validate_string_field(True, "title", "epic[0]")
        assert result == "True"

    def test_validate_string_field_required_none_raises(self):
        """Verify _validate_string_field raises for required None field."""
        from prd_decomposer.server import _validate_string_field
        with pytest.raises(ValueError, match="is required"):
            _validate_string_field(None, "title", "epic[0]", required=True)

    def test_validate_string_field_optional_none_returns_empty(self):
        """Verify _validate_string_field returns empty string for optional None."""
        from prd_decomposer.server import _validate_string_field
        result = _validate_string_field(None, "description", "epic[0]", required=False)
        assert result == ""

    def test_export_with_integer_title_coerces(self):
        """Verify export coerces integer title to string."""
        tickets = {
            "epics": [
                {
                    "title": 12345,  # Integer instead of string
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "acceptance_criteria": [],
                            "labels": [],
                            "size": "S",
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        result = export_tickets(json.dumps(tickets), format="csv")
        assert "12345" in result  # Epic title coerced to string

    def test_export_with_integer_story_title_coerces(self):
        """Verify export coerces integer story title to string."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": 99999,  # Integer instead of string
                            "description": "Story desc",
                            "acceptance_criteria": [],
                            "labels": [],
                            "size": "S",
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        result = export_tickets(json.dumps(tickets), format="csv")
        assert "99999" in result

    def test_export_with_missing_title_raises(self):
        """Verify export raises for missing required title."""
        tickets = {
            "epics": [
                {
                    # Missing title
                    "description": "Desc",
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        with pytest.raises(FatalToolError, match="title is required"):
            export_tickets(json.dumps(tickets), format="csv")

    def test_export_with_null_description_allowed(self):
        """Verify export allows null description (optional field)."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": None,  # Null is allowed for optional field
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        # Should not raise - use YAML format which shows epic titles
        result = export_tickets(json.dumps(tickets), format="yaml")
        assert "Epic" in result

    def test_export_with_empty_title_raises(self):
        """Verify export raises for empty required title."""
        tickets = {
            "epics": [
                {
                    "title": "",  # Empty string should fail
                    "description": "Desc",
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        with pytest.raises(FatalToolError, match="cannot be empty"):
            export_tickets(json.dumps(tickets), format="csv")


class TestShutdownCircuitBreaker:
    """Tests for shutdown handling with circuit breaker."""

    def test_shutdown_does_not_trip_circuit_breaker(self, permissive_rate_limiter):
        """Verify shutdown-aborted requests don't count as failures."""
        from prd_decomposer.config import Settings
        from prd_decomposer.server import _shutdown_event

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60)

        mock_client = MagicMock()
        settings = Settings(max_retries=1)

        # Set shutdown flag
        _shutdown_event.set()

        try:
            with pytest.raises(LLMError, match="shutting down"):
                _call_llm_with_retry(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.5,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                    circuit_breaker=cb,
                )

            # Shutdown abort should NOT trip circuit breaker
            assert cb._failure_count == 0, "Shutdown should not count as failure"
            assert cb.state == "closed"
        finally:
            _shutdown_event.clear()


class TestYAMLSizePriorityEscaping:
    """Tests for YAML size/priority escaping."""

    def test_yaml_size_is_escaped(self):
        """Verify size field is properly escaped in YAML."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "acceptance_criteria": [],
                            "labels": [],
                            "size": 'S"pecial',  # Special chars in size
                            "priority": "medium",
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        result = export_tickets(json.dumps(tickets), format="yaml")
        # Size should be quoted and escaped
        assert 'size: "S\\"pecial"' in result

    def test_yaml_priority_is_escaped(self):
        """Verify priority field is properly escaped in YAML."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "acceptance_criteria": [],
                            "labels": [],
                            "size": "M",
                            "priority": "high\ninjection",  # Newline in priority
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        result = export_tickets(json.dumps(tickets), format="yaml")
        # Priority should be quoted and newline escaped
        assert 'priority: "high\\ninjection"' in result


class TestNullStoriesHandling:
    """Tests for null stories array handling in export."""

    def test_export_with_null_stories_csv(self):
        """Verify export handles null stories in CSV format."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": None,  # null instead of []
                }
            ]
        }
        # Should not crash - null normalized to []
        result = export_tickets(json.dumps(tickets), format="csv")
        assert "epic_title" in result  # Header should be present

    def test_export_with_null_stories_jira(self):
        """Verify export handles null stories in Jira format."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": None,  # null instead of []
                }
            ]
        }
        # Should not crash
        result = export_tickets(json.dumps(tickets), format="jira")
        parsed = json.loads(result)
        # Should have epic issue only
        assert len(parsed["issueUpdates"]) == 1

    def test_export_with_null_stories_yaml(self):
        """Verify export handles null stories in YAML format."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": None,  # null instead of []
                }
            ]
        }
        # Should not crash
        result = export_tickets(json.dumps(tickets), format="yaml")
        assert "stories: []" in result  # Normalized to empty list


class TestYAMLExportBugFix:
    """Test for YAML export bug fix."""

    def test_yaml_export_single_epics_key(self):
        """Verify YAML export only has one epics: key (bug fix #3)."""
        tickets = {
            "epics": [
                {"title": "Epic 1", "description": "Desc 1", "labels": [], "stories": []},
                {"title": "Epic 2", "description": "Desc 2", "labels": [], "stories": []},
            ]
        }
        from prd_decomposer.server import _export_to_yaml
        result = _export_to_yaml(tickets)

        # Count occurrences of "epics:" at start of line
        lines = result.split("\n")
        epics_count = sum(1 for line in lines if line.strip() == "epics:")
        assert epics_count == 1, f"Expected 1 'epics:' key, found {epics_count}"

    def test_yaml_export_escapes_quotes(self):
        """Verify YAML export escapes double quotes in text."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {
                    "title": 'Epic with "quotes" in title',
                    "description": 'Said "hello"',
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        result = _export_to_yaml(tickets)
        assert r'\"quotes\"' in result
        assert r'\"hello\"' in result

    def test_yaml_export_escapes_newlines(self):
        """Verify YAML export escapes newlines in text."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {
                    "title": "Multi-line",
                    "description": "Line one\nLine two",
                    "labels": [],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Has\nnewlines",
                            "size": "S",
                            "priority": "medium",
                            "labels": [],
                            "requirement_ids": [],
                            "acceptance_criteria": ["AC with\nnewline"],
                        }
                    ],
                }
            ]
        }
        result = _export_to_yaml(tickets)
        # Escaped newlines should appear as literal \n in the output
        assert "\\n" in result
        # No actual newlines within quoted strings (split on newlines, check values)
        for line in result.split("\n"):
            if "description:" in line:
                # The description value should not have raw newline
                assert "\nLine" not in line

    def test_yaml_export_escapes_backslashes(self):
        """Verify YAML export escapes backslashes in text."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {
                    "title": r"Path: C:\Users\test",
                    "description": r"Use \n for newline",
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        result = _export_to_yaml(tickets)
        # Backslashes should be doubled: \ → \\
        # Input has 2 backslashes (C:\Users\test), output should have 4 (C:\\Users\\test)
        assert "C:\\\\Users\\\\test" in result
        # Literal \n in input should become \\n (escaped backslash + n, not newline)
        assert "\\\\n" in result

    def test_escape_yaml_string_function(self):
        """Test the _escape_yaml_string helper directly."""
        from prd_decomposer.server import _escape_yaml_string

        # Double quotes
        assert _escape_yaml_string('say "hi"') == r'say \"hi\"'
        # Newlines
        assert _escape_yaml_string("a\nb") == r"a\nb"
        # Tabs
        assert _escape_yaml_string("a\tb") == r"a\tb"
        # Backslashes (escaped first)
        assert _escape_yaml_string(r"a\b") == r"a\\b"
        # Combined
        assert _escape_yaml_string('"\n\\') == r'\"' + r"\n" + r"\\"

    def test_yaml_flow_list_function(self):
        """Test the _yaml_flow_list helper directly."""
        from prd_decomposer.server import _yaml_flow_list

        # Empty list
        assert _yaml_flow_list([]) == "[]"
        # Simple items
        assert _yaml_flow_list(["a", "b"]) == '["a", "b"]'
        # Items with commas (would break naive join)
        assert _yaml_flow_list(["foo, bar", "baz"]) == '["foo, bar", "baz"]'
        # Items with colons
        assert _yaml_flow_list(["key: value"]) == '["key: value"]'
        # Items with quotes
        assert _yaml_flow_list(['say "hi"']) == r'["say \"hi\""]'

    def test_yaml_export_labels_with_special_chars(self):
        """Verify YAML export properly quotes labels with special characters."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": ["foo, bar", "key: value", "normal"],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "size": "S",
                            "priority": "medium",
                            "labels": ["has, comma"],
                            "requirement_ids": ["REQ: 001"],
                            "acceptance_criteria": [],
                        }
                    ],
                }
            ]
        }
        result = _export_to_yaml(tickets)
        # Each label should be individually quoted
        assert '"foo, bar"' in result
        assert '"key: value"' in result
        assert '"has, comma"' in result
        assert '"REQ: 001"' in result

    def test_yaml_export_empty_epics_uses_empty_list(self):
        """Verify YAML export uses [] for empty epics, not null."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {"epics": []}
        result = _export_to_yaml(tickets)
        assert "epics: []" in result
        # Should not have just "epics:" with nothing following
        assert "epics:\n" not in result

    def test_yaml_export_empty_stories_uses_empty_list(self):
        """Verify YAML export uses [] for empty stories, not null."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {"title": "Epic", "description": "Desc", "labels": [], "stories": []}
            ]
        }
        result = _export_to_yaml(tickets)
        assert "stories: []" in result
        # Should not have dangling "stories:" with no items
        lines = result.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "stories:":
                # If stories: is followed by an empty line or end, it's wrong
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                assert next_line.strip().startswith("-"), "stories: should have items or be []"

    def test_yaml_export_empty_acceptance_criteria_uses_empty_list(self):
        """Verify YAML export uses [] for empty acceptance_criteria."""
        from prd_decomposer.server import _export_to_yaml
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": "Story",
                            "description": "Desc",
                            "size": "S",
                            "priority": "medium",
                            "labels": [],
                            "requirement_ids": [],
                            "acceptance_criteria": [],
                        }
                    ],
                }
            ]
        }
        result = _export_to_yaml(tickets)
        assert "acceptance_criteria: []" in result


class TestSizingRubricBugFix:
    """Test for sizing rubric validation bug fix."""

    def test_sizing_rubric_without_labels(self):
        """Verify sizing rubric works without explicit labels (bug fix #4)."""
        # This format matches the documented API (no labels)
        rubric_data = {
            "small": {"duration": "4h", "scope": "tiny", "risk": "none"},
            "medium": {"duration": "2d", "scope": "small", "risk": "low"},
            "large": {"duration": "5d", "scope": "big", "risk": "high"},
        }
        rubric = SizingRubric(**rubric_data)

        # Labels should be auto-filled
        assert rubric.small.label == "S"
        assert rubric.medium.label == "M"
        assert rubric.large.label == "L"

    def test_decompose_with_documented_rubric_format(
        self, sample_input_requirements, mock_client_factory, sample_epic_response
    ):
        """Verify decompose_to_tickets accepts documented rubric format (bug fix #4)."""
        mock_client = mock_client_factory(sample_epic_response)

        # Format exactly as documented in tool annotation (no labels)
        rubric_json = json.dumps({
            "small": {"duration": "4h", "scope": "tiny", "risk": "none"},
            "medium": {"duration": "2d", "scope": "small", "risk": "low"},
            "large": {"duration": "5d", "scope": "big", "risk": "high"},
        })

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            # Should NOT raise "Invalid sizing_rubric" error
            result = decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric=rubric_json
            )

        assert "epics" in result

    def test_sizing_rubric_array_raises_error(self, sample_input_requirements):
        """Verify non-object sizing_rubric raises error, not TypeError."""
        from arcade_core.errors import FatalToolError

        # JSON array instead of object
        rubric_json = '["small", "medium", "large"]'

        # arcade_tdk wraps ValueError in FatalToolError
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric=rubric_json
            )

    def test_sizing_rubric_string_raises_error(self, sample_input_requirements):
        """Verify string sizing_rubric raises error, not TypeError."""
        from arcade_core.errors import FatalToolError

        # JSON string instead of object
        rubric_json = '"just a string"'

        with pytest.raises(FatalToolError, match="must be a JSON object"):
            decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric=rubric_json
            )

    def test_sizing_rubric_number_raises_error(self, sample_input_requirements):
        """Verify numeric sizing_rubric raises error, not TypeError."""
        from arcade_core.errors import FatalToolError

        # JSON number instead of object
        rubric_json = "123"

        with pytest.raises(FatalToolError, match="must be a JSON object"):
            decompose_to_tickets(
                json.dumps(sample_input_requirements), sizing_rubric=rubric_json
            )
