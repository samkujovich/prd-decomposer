"""Tests for MCP server tools with mocked LLM calls."""

import json
from unittest.mock import MagicMock, patch

import pytest
from arcade_core.errors import FatalToolError
from pydantic import ValidationError

from prd_decomposer.server import analyze_prd, decompose_to_tickets, get_client


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


class TestAnalyzePrd:
    """Tests for the analyze_prd tool."""

    def test_analyze_prd_returns_structured_requirements(self):
        """Verify analyze_prd returns validated StructuredRequirements."""
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "User login",
                    "description": "Users must be able to log in",
                    "acceptance_criteria": ["Login form exists"],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high"
                }
            ],
            "summary": "Authentication system PRD",
            "source_hash": "abc12345"
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = analyze_prd("Test PRD content")

        assert "requirements" in result
        assert len(result["requirements"]) == 1
        assert result["requirements"][0]["id"] == "REQ-001"
        assert result["summary"] == "Authentication system PRD"
        # source_hash is overwritten by the function
        assert len(result["source_hash"]) == 8

    def test_analyze_prd_generates_source_hash(self):
        """Verify analyze_prd generates a hash from the input text."""
        mock_response = {
            "requirements": [],
            "summary": "Empty PRD",
            "source_hash": "ignored"
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result1 = analyze_prd("PRD content A")
            result2 = analyze_prd("PRD content B")

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
                    "priority": "high"
                }
            ],
            "summary": "Vague PRD",
            "source_hash": "12345678"
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = analyze_prd("The API should be fast")

        assert result["requirements"][0]["ambiguity_flags"] == [
            "Vague quantifier: 'fast' without metrics"
        ]

    def test_analyze_prd_validates_llm_response(self):
        """Verify analyze_prd raises FatalToolError for invalid LLM response."""
        # Missing required 'priority' field
        mock_response = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test"
                    # Missing: acceptance_criteria, dependencies, ambiguity_flags, priority
                }
            ],
            "summary": "Test",
            "source_hash": "12345678"
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with pytest.raises(FatalToolError):
                analyze_prd("Test PRD")


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
                    "priority": "high"
                }
            ],
            "summary": "Auth PRD",
            "source_hash": "abc12345"
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
                            "requirement_ids": ["REQ-001"]
                        }
                    ],
                    "labels": ["auth"]
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = decompose_to_tickets(input_requirements)

        assert "epics" in result
        assert len(result["epics"]) == 1
        assert result["epics"][0]["title"] == "Authentication Epic"
        assert len(result["epics"][0]["stories"]) == 1
        assert result["epics"][0]["stories"][0]["size"] == "M"

    def test_decompose_to_tickets_adds_metadata(self):
        """Verify decompose_to_tickets adds generation metadata."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "low"
                }
            ],
            "summary": "Test",
            "source_hash": "12345678"
        }

        mock_response = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "stories": [],
                    "labels": []
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = decompose_to_tickets(input_requirements)

        assert "metadata" in result
        assert "generated_at" in result["metadata"]
        assert result["metadata"]["model"] == "gpt-4o"
        assert result["metadata"]["requirement_count"] == 1
        assert result["metadata"]["story_count"] == 0

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
                    "priority": "medium"
                }
            ],
            "summary": "Test",
            "source_hash": "12345678"
        }

        mock_response = {
            "epics": [
                {
                    "title": "Epic 1",
                    "description": "Desc",
                    "stories": [
                        {"title": "S1", "description": "D", "acceptance_criteria": [],
                         "size": "S", "labels": [], "requirement_ids": []},
                        {"title": "S2", "description": "D", "acceptance_criteria": [],
                         "size": "S", "labels": [], "requirement_ids": []}
                    ],
                    "labels": []
                },
                {
                    "title": "Epic 2",
                    "description": "Desc",
                    "stories": [
                        {"title": "S3", "description": "D", "acceptance_criteria": [],
                         "size": "M", "labels": [], "requirement_ids": []}
                    ],
                    "labels": []
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            result = decompose_to_tickets(input_requirements)

        assert result["metadata"]["story_count"] == 3

    def test_decompose_to_tickets_validates_input(self):
        """Verify decompose_to_tickets raises FatalToolError for invalid input."""
        # Invalid input - missing required fields
        invalid_input = {"requirements": [{"id": "REQ-001"}]}

        with pytest.raises(FatalToolError):
            decompose_to_tickets(invalid_input)

    def test_decompose_to_tickets_validates_llm_response(self):
        """Verify decompose_to_tickets raises FatalToolError for invalid LLM response."""
        input_requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Test",
                    "acceptance_criteria": [],
                    "dependencies": [],
                    "ambiguity_flags": [],
                    "priority": "high"
                }
            ],
            "summary": "Test",
            "source_hash": "12345678"
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
                            "requirement_ids": []
                        }
                    ],
                    "labels": []
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )

        with patch("prd_decomposer.server.get_client", return_value=mock_client):
            with pytest.raises(FatalToolError):
                decompose_to_tickets(input_requirements)
