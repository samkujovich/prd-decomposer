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
from prd_decomposer.logging import correlation_id
from prd_decomposer.prompts import PROMPT_VERSION
from prd_decomposer.server import (
    LLMError,
    RateLimiter,
    RateLimitExceededError,
    _analyze_prd_impl,
    _call_llm_with_retry,
    _decompose_to_tickets_impl,
    _is_path_allowed,
    analyze_prd,
    decompose_to_tickets,
    get_client,
    get_rate_limiter,
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
            with pytest.raises(LLMError, match="failed after 3 retries"):
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
            with pytest.raises(LLMError, match="failed after 2 retries"):
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
            with pytest.raises(LLMError, match="failed after 2 retries"):
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
                    "ambiguity_flags": ["Vague quantifier: 'fast' without metrics"],
                    "priority": "high",
                }
            ],
            "summary": "Vague PRD",
            "source_hash": "12345678",
        }
        mock_client = mock_client_factory(mock_response)

        result = _analyze_prd_impl("The API should be fast", client=mock_client)

        assert result["requirements"][0]["ambiguity_flags"] == [
            "Vague quantifier: 'fast' without metrics"
        ]

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
        """Verify retry sleep delays follow exponential backoff pattern."""
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
        assert delays == [1.0, 2.0, 4.0]


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

    def test_analyze_prd_wrapper_delegates_to_impl(self, mock_client_factory):
        """Verify analyze_prd delegates to _analyze_prd_impl."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = mock_client_factory(mock_response)

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
            mock_impl.assert_called_once_with('{"requirements": []}')
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
