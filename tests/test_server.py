"""Tests for MCP server tools with mocked LLM calls."""

import json
import logging
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from arcade_core.errors import FatalToolError
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from prd_decomposer.config import Settings
from prd_decomposer.logging import correlation_id
from prd_decomposer.server import (
    LLMError,
    _analyze_prd_impl,
    _call_llm_with_retry,
    _decompose_to_tickets_impl,
    _is_path_allowed,
    get_client,
    read_file,
)


class TestReadFile:
    """Tests for the read_file tool."""

    def test_read_file_returns_content(self, tmp_path, monkeypatch):
        """Verify read_file returns file contents."""
        # Add tmp_path to allowed directories for this test
        import prd_decomposer.server as server_module

        original_allowed = server_module.ALLOWED_DIRECTORIES.copy()
        server_module.ALLOWED_DIRECTORIES.append(tmp_path)

        try:
            test_file = tmp_path / "test.md"
            test_file.write_text("# Test PRD\n\nThis is a test.")

            result = read_file(str(test_file))

            assert result == "# Test PRD\n\nThis is a test."
        finally:
            server_module.ALLOWED_DIRECTORIES = original_allowed

    def test_read_file_nonexistent_raises_error(self):
        """Verify read_file raises FatalToolError for missing files within allowed dirs."""
        # Use a path within cwd that doesn't exist
        with pytest.raises(FatalToolError):
            read_file("nonexistent_file_in_cwd.md")

    def test_read_file_path_traversal_blocked(self):
        """Verify read_file blocks path traversal attempts."""
        with pytest.raises(FatalToolError):
            read_file("/etc/passwd")

    def test_read_file_parent_traversal_blocked(self):
        """Verify read_file blocks parent directory traversal."""
        with pytest.raises(FatalToolError):
            read_file("../../../etc/passwd")

    def test_read_file_home_directory_blocked(self):
        """Verify read_file blocks home directory access."""
        with pytest.raises(FatalToolError):
            read_file(os.path.expanduser("~/.ssh/id_rsa"))


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
        # Create a mock path that raises OSError on resolve
        with patch.object(Path, "resolve", side_effect=OSError("Permission denied")):
            # Should return False, not raise
            result = _is_path_allowed(Path("/some/path"))
            assert result is False


class TestReadFileEdgeCases:
    """Additional edge case tests for read_file."""

    def test_read_file_directory_raises_error(self, tmp_path, monkeypatch):
        """Verify read_file raises error when path is a directory."""
        import prd_decomposer.server as server_module

        original_allowed = server_module.ALLOWED_DIRECTORIES.copy()
        server_module.ALLOWED_DIRECTORIES.append(tmp_path)

        try:
            # tmp_path is a directory, not a file
            with pytest.raises(FatalToolError):
                read_file(str(tmp_path))
        finally:
            server_module.ALLOWED_DIRECTORIES = original_allowed


class TestGetClient:
    """Tests for the lazy OpenAI client initialization."""

    def test_get_client_returns_openai_client(self):
        """Verify get_client returns an OpenAI client instance."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Reset the global _client to test initialization
            import prd_decomposer.server as server_module

            server_module._client = None

            client = get_client()
            assert client is mock_client
            mock_openai.assert_called_once()

    def test_get_client_reuses_existing_client(self):
        """Verify get_client returns cached client on subsequent calls."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            import prd_decomposer.server as server_module

            server_module._client = None

            client1 = get_client()
            client2 = get_client()

            assert client1 is client2
            # Only called once due to caching
            mock_openai.assert_called_once()

    def test_get_client_thread_safe(self):
        """Verify concurrent get_client calls create only one OpenAI instance."""
        with patch("prd_decomposer.server.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            import prd_decomposer.server as server_module

            server_module._client = None

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

            # All threads must get the exact same instance
            assert len(results) == 10
            assert all(r is results[0] for r in results)
            # OpenAI constructor called exactly once
            mock_openai.assert_called_once()


class TestLLMRetry:
    """Tests for LLM retry logic."""

    def test_call_llm_with_retry_success(self):
        """Verify successful LLM call returns data and usage."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            data, usage = _call_llm_with_retry(
                [{"role": "user", "content": "test"}], temperature=0.2
            )

        assert data == {"result": "success"}
        assert usage["total_tokens"] == 150

    def test_call_llm_with_retry_empty_response(self):
        """Verify LLMError raised for empty response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with pytest.raises(LLMError, match="empty response"):
                _call_llm_with_retry([{"role": "user", "content": "test"}], temperature=0.2)

    def test_call_llm_with_retry_invalid_json(self):
        """Verify LLMError raised for invalid JSON."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json"))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with pytest.raises(LLMError, match="invalid JSON"):
                _call_llm_with_retry([{"role": "user", "content": "test"}], temperature=0.2)

    def test_call_llm_with_retry_rate_limit_then_success(self):
        """Verify retry on RateLimitError eventually succeeds."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        # First call raises RateLimitError, second succeeds
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded", response=MagicMock(status_code=429), body=None
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
                usage=mock_usage,
            ),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):  # Skip actual sleep
                data, _usage = _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    settings=settings,
                )

        assert data == {"result": "success"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_connection_error_then_success(self):
        """Verify retry on APIConnectionError eventually succeeds."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
                usage=mock_usage,
            ),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                data, _usage = _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    settings=settings,
                )

        assert data == {"result": "success"}

    def test_call_llm_with_retry_api_error_4xx_no_retry(self):
        """Verify 4xx APIError raises immediately without retry."""
        mock_client = MagicMock()
        error = APIError(message="Bad request", request=MagicMock(), body=None)
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with pytest.raises(LLMError, match="OpenAI API error"):
                _call_llm_with_retry([{"role": "user", "content": "test"}], temperature=0.2)

        # Should only be called once (no retries for 4xx)
        assert mock_client.chat.completions.create.call_count == 1

    def test_call_llm_with_retry_api_error_no_status_code(self):
        """Verify APIError without status_code attribute retries."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        # Create error without status_code attribute
        error = APIError(message="Unknown error", request=MagicMock(), body=None)
        # Explicitly remove status_code if it exists
        if hasattr(error, "status_code"):
            delattr(error, "status_code")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            error,
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
                usage=mock_usage,
            ),
        ]

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                data, _usage = _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    settings=settings,
                )

        assert data == {"result": "success"}
        # Should retry (2 calls total)
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_api_error_5xx_retries(self):
        """Verify 5xx APIError retries then fails."""
        mock_client = MagicMock()
        error = APIError(message="Server error", request=MagicMock(), body=None)
        error.status_code = 500
        mock_client.chat.completions.create.side_effect = error

        settings = Settings(max_retries=3, initial_retry_delay=0.01)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                with pytest.raises(LLMError, match="failed after 3 retries"):
                    _call_llm_with_retry(
                        [{"role": "user", "content": "test"}],
                        temperature=0.2,
                        settings=settings,
                    )

        # Should be called 3 times (all retries exhausted)
        assert mock_client.chat.completions.create.call_count == 3

    def test_call_llm_with_retry_all_retries_exhausted(self):
        """Verify LLMError raised after all retries exhausted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(max_retries=2, initial_retry_delay=0.01)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                with pytest.raises(LLMError, match="failed after 2 retries"):
                    _call_llm_with_retry(
                        [{"role": "user", "content": "test"}],
                        temperature=0.2,
                        settings=settings,
                    )

    def test_call_llm_with_retry_timeout_then_success(self):
        """Verify retry on APITimeoutError eventually succeeds."""
        mock_response = {"result": "success"}
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        # First call raises timeout, second succeeds
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
                usage=mock_usage,
            ),
        ]

        settings = Settings(initial_retry_delay=0.01, llm_timeout=30.0)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                data, _usage = _call_llm_with_retry(
                    [{"role": "user", "content": "test"}],
                    temperature=0.2,
                    settings=settings,
                )

        assert data == {"result": "success"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_call_llm_with_retry_timeout_all_retries_exhausted(self):
        """Verify LLMError raised after all timeout retries exhausted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        settings = Settings(max_retries=2, initial_retry_delay=0.01, llm_timeout=30.0)
        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with patch("prd_decomposer.server.time.sleep"):
                with pytest.raises(LLMError, match="failed after 2 retries"):
                    _call_llm_with_retry(
                        [{"role": "user", "content": "test"}],
                        temperature=0.2,
                        settings=settings,
                    )

        # Should be called 2 times (all retries exhausted)
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

        # Verify timeout was passed to the API call
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["timeout"] == 45.0


class TestAnalyzePrd:
    """Tests for the analyze_prd tool."""

    def test_analyze_prd_returns_structured_requirements(self):
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

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _analyze_prd_impl("Test PRD content", client=mock_client)

        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["id"] == "REQ-001"
        assert result["summary"] == "Authentication system PRD"
        # source_hash is overwritten by the function
        assert len(result["source_hash"]) == 8
        # Verify metadata is included
        assert "_metadata" in result
        assert "usage" in result["_metadata"]
        assert "prompt_version" in result["_metadata"]

    def test_analyze_prd_generates_source_hash(self):
        """Verify analyze_prd generates a hash from the input text."""
        mock_response = {"requirements": [], "summary": "Empty PRD", "source_hash": "ignored"}

        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result1 = _analyze_prd_impl("PRD content A", client=mock_client)
        result2 = _analyze_prd_impl("PRD content B", client=mock_client)

        # Different inputs should produce different hashes
        assert result1["source_hash"] != result2["source_hash"]

    def test_analyze_prd_with_ambiguity_flags(self):
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

        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _analyze_prd_impl("The API should be fast", client=mock_client)

        assert result["requirements"][0]["ambiguity_flags"] == [
            "Vague quantifier: 'fast' without metrics"
        ]

    def test_analyze_prd_validates_llm_response(self):
        """Verify analyze_prd raises RuntimeError for invalid LLM response."""
        # Missing required 'priority' field
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    # Missing: acceptance_criteria, dependencies, ambiguity_flags, priority
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }

        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

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
        # Create PRD that exceeds the limit
        oversized_prd = "x" * 2000  # 2000 chars
        settings = Settings(max_prd_length=1000)  # Limit to 1000

        with pytest.raises(ValueError, match="exceeds maximum length"):
            _analyze_prd_impl(oversized_prd, settings=settings)

    def test_analyze_prd_accepts_input_at_max_length(self):
        """Verify analyze_prd accepts PRD exactly at max length."""
        mock_response = {
            "requirements": [],
            "summary": "Empty PRD",
            "source_hash": "12345678",
        }
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        # PRD exactly at limit should work
        prd_at_limit = "x" * 1000
        settings = Settings(max_prd_length=1000)

        result = _analyze_prd_impl(prd_at_limit, client=mock_client, settings=settings)
        assert "requirements" in result


class TestDecomposeToTickets:
    """Tests for the decompose_to_tickets tool."""

    def test_decompose_to_tickets_returns_ticket_collection(self):
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

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _decompose_to_tickets_impl(input_requirements, client=mock_client)

        assert "epics" in result
        assert len(result["epics"]) == 1
        assert result["epics"][0]["title"] == "Authentication Epic"
        assert len(result["epics"][0]["stories"]) == 1
        assert result["epics"][0]["stories"][0]["size"] == "M"

    def test_decompose_to_tickets_adds_metadata(self):
        """Verify decompose_to_tickets adds generation metadata with usage."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "low",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }

        mock_response = {
            "epics": [{"title": "Epic", "description": "Desc", "stories": [], "labels": []}]
        }

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _decompose_to_tickets_impl(input_requirements, client=mock_client)

        assert "metadata" in result
        assert "generated_at" in result["metadata"]
        assert result["metadata"]["model"] == "gpt-4o"
        assert result["metadata"]["prompt_version"] is not None
        assert result["metadata"]["requirement_count"] == 1
        assert result["metadata"]["story_count"] == 0
        assert "usage" in result["metadata"]
        assert result["metadata"]["usage"]["total_tokens"] == 150

    def test_decompose_to_tickets_counts_stories(self):
        """Verify decompose_to_tickets correctly counts total stories."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "medium",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }

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

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _decompose_to_tickets_impl(input_requirements, client=mock_client)

        assert result["metadata"]["story_count"] == 3

    def test_decompose_to_tickets_validates_input(self):
        """Verify decompose_to_tickets raises ValueError for invalid input."""
        # Invalid input - missing required fields
        invalid_input = {"requirements": [{"id": "REQ-001"}]}

        with pytest.raises(ValueError, match="Invalid requirements structure"):
            _decompose_to_tickets_impl(invalid_input, client=MagicMock())

    def test_decompose_to_tickets_empty_requirements_raises(self):
        """Verify decompose_to_tickets raises for empty requirements."""
        with pytest.raises(ValueError, match="Requirements cannot be empty"):
            _decompose_to_tickets_impl({}, client=MagicMock())

    def test_decompose_to_tickets_strips_internal_metadata(self):
        """Verify decompose_to_tickets handles _metadata from analyze_prd."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "low",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
            "_metadata": {"prompt_version": "1.0.0", "usage": {}},  # Should be stripped
        }

        mock_response = {
            "epics": [{"title": "Epic", "description": "Desc", "stories": [], "labels": []}]
        }

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        result = _decompose_to_tickets_impl(input_requirements, client=mock_client)

        # Should succeed - _metadata is stripped before validation
        assert "epics" in result

    def test_decompose_to_tickets_validates_llm_response(self):
        """Verify decompose_to_tickets raises RuntimeError for invalid LLM response."""
        input_requirements = {
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
            "source_hash": "12345678",
        }

        # Invalid response - story has invalid size
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

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        with pytest.raises(RuntimeError, match="LLM returned invalid ticket structure"):
            _decompose_to_tickets_impl(input_requirements, client=mock_client)

    def test_decompose_to_tickets_string_requirements(self):
        """Verify decompose_to_tickets handles string (JSON) requirements."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "low",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }

        mock_response = {
            "epics": [{"title": "Epic", "description": "Desc", "stories": [], "labels": []}]
        }

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=mock_usage,
        )

        # Pass as JSON string instead of dict
        result = _decompose_to_tickets_impl(json.dumps(input_requirements), client=mock_client)

        assert "epics" in result

    def test_decompose_to_tickets_invalid_json_string(self):
        """Verify decompose_to_tickets raises for invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            _decompose_to_tickets_impl("not valid json {", client=MagicMock())

    def test_decompose_to_tickets_llm_error_propagates(self):
        """Verify decompose_to_tickets raises RuntimeError when LLM call fails."""
        input_requirements = {
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
            "source_hash": "12345678",
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01)
        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to decompose"):
                _decompose_to_tickets_impl(
                    input_requirements, client=mock_client, settings=settings
                )


class TestIntegrationPipeline:
    """Integration tests for the full analyze -> decompose pipeline."""

    def test_full_pipeline_mocked(self):
        """Test the full pipeline with mocked LLM calls."""
        # Mock analyze_prd response
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

        # Mock decompose_to_tickets response
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

        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client = MagicMock()
        # First call returns analyze response, second returns decompose response
        mock_client.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(analyze_response)))],
                usage=mock_usage,
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=json.dumps(decompose_response)))],
                usage=mock_usage,
            ),
        ]

        # Step 1: Analyze with direct injection
        requirements = _analyze_prd_impl(
            "# Sample PRD\n\nUser auth required.", client=mock_client
        )

        assert "requirements" in requirements
        assert len(requirements["requirements"]) == 1
        assert "_metadata" in requirements

        # Step 2: Decompose with direct injection (passing requirements explicitly)
        tickets = _decompose_to_tickets_impl(requirements, client=mock_client)

        assert "epics" in tickets
        assert len(tickets["epics"]) == 1
        assert tickets["epics"][0]["stories"][0]["requirement_ids"] == ["REQ-001"]
        assert "metadata" in tickets
        assert tickets["metadata"]["requirement_count"] == 1


def _make_mock_client(response_data: dict) -> MagicMock:
    """Helper to create a mock OpenAI client with a canned response."""
    mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(response_data)))],
        usage=mock_usage,
    )
    return mock_client


class TestServerLogging:
    """Tests for structured logging in server tool implementations."""

    def test_analyze_prd_sets_correlation_id(self, caplog):
        """Verify _analyze_prd_impl sets a correlation ID."""
        mock_response = {
            "requirements": [],
            "summary": "Test",
            "source_hash": "ignored",
        }
        mock_client = _make_mock_client(mock_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _analyze_prd_impl("Test PRD", client=mock_client)

        # After the call, correlation_id should have been set (non-empty)
        # We check the log records for a non-empty correlation_id
        assert any("Starting PRD analysis" in r.message for r in caplog.records)

    def test_analyze_prd_logs_completion(self, caplog):
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
        mock_client = _make_mock_client(mock_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _analyze_prd_impl("Test PRD", client=mock_client)

        messages = [r.message for r in caplog.records]
        assert any("PRD analysis complete" in m for m in messages)

    def test_decompose_logs_start_and_completion(self, caplog):
        """Verify _decompose_to_tickets_impl logs start and completion."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "low",
                }
            ],
            "summary": "Test",
            "source_hash": "12345678",
        }
        mock_response = {
            "epics": [{"title": "Epic", "description": "Desc", "stories": [], "labels": []}]
        }
        mock_client = _make_mock_client(mock_response)

        with caplog.at_level(logging.DEBUG, logger="prd_decomposer"):
            _decompose_to_tickets_impl(input_requirements, client=mock_client)

        messages = [r.message for r in caplog.records]
        assert any("Starting ticket decomposition" in m for m in messages)
        assert any("Ticket decomposition complete" in m for m in messages)

    def test_call_llm_with_retry_logs_retry_attempts(self, caplog):
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
