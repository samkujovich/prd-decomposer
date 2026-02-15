"""Integration tests requiring OPENAI_API_KEY.

Run with: OPENAI_API_KEY=sk-... uv run pytest tests/integration/ -v

These tests make real API calls and are skipped when OPENAI_API_KEY is not set.
They validate end-to-end functionality against the actual OpenAI API.
"""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Integration tests require OPENAI_API_KEY",
)


class TestAnalyzePrdRealAPI:
    """Integration tests for analyze_prd with real API."""

    def test_analyze_simple_prd(self):
        """Analyze a simple PRD with real OpenAI API."""
        from prd_decomposer.server import analyze_prd

        simple_prd = """# Feature: User Login

## Requirements
- Users can log in with email and password
- Failed login shows error message
- Successful login redirects to dashboard
"""

        result = analyze_prd(prd_text=simple_prd)

        assert "requirements" in result
        assert len(result["requirements"]) > 0
        assert "summary" in result
        assert "source_hash" in result
        assert "_metadata" in result

        # Verify requirement structure
        req = result["requirements"][0]
        assert "id" in req
        assert req["id"].startswith("REQ-")
        assert "title" in req
        assert "description" in req

    def test_analyze_prd_detects_ambiguities(self):
        """Verify LLM detects vague language as ambiguities."""
        from prd_decomposer.server import analyze_prd

        vague_prd = """# Feature: Dashboard

## Requirements
- The dashboard should be fast and user-friendly
- It needs to scale well as we grow
- Performance should be acceptable
"""

        result = analyze_prd(prd_text=vague_prd)

        assert "requirements" in result

        # Check if any requirement has ambiguity flags
        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        # Should detect at least one ambiguity (fast, user-friendly, scale, acceptable)
        assert len(all_flags) > 0


class TestDecomposeToTicketsRealAPI:
    """Integration tests for decompose_to_tickets with real API."""

    def test_decompose_creates_epics_and_stories(self):
        """Decompose requirements into epics and stories."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: Password Reset

## Requirements
- User can request password reset via email
- Reset link expires after 24 hours
- User must set new password meeting complexity requirements
"""

        # First analyze
        requirements = analyze_prd(prd_text=prd)

        # Then decompose
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        assert "epics" in tickets
        assert len(tickets["epics"]) > 0
        assert "metadata" in tickets

        # Verify epic structure
        epic = tickets["epics"][0]
        assert "title" in epic
        assert "description" in epic
        assert "stories" in epic
        assert len(epic["stories"]) > 0

        # Verify story structure
        story = epic["stories"][0]
        assert "title" in story
        assert "size" in story
        assert story["size"] in ("S", "M", "L")
        assert "requirement_ids" in story


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow_with_export(self):
        """Test complete workflow: analyze -> decompose -> export."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets, export_tickets

        prd = """# Feature: User Notifications

## Requirements
- Send email notifications for important events
- Allow users to configure notification preferences
- Support daily digest emails
"""

        # Step 1: Analyze
        requirements = analyze_prd(prd_text=prd)
        assert len(requirements["requirements"]) > 0

        # Step 2: Decompose
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))
        assert len(tickets["epics"]) > 0

        # Step 3: Export to different formats
        tickets_json = json.dumps(tickets)

        csv_output = export_tickets(tickets_json=tickets_json, format="csv")
        assert "epic_title" in csv_output
        assert "story_title" in csv_output

        yaml_output = export_tickets(tickets_json=tickets_json, format="yaml")
        assert "epics:" in yaml_output

        jira_output = export_tickets(tickets_json=tickets_json, format="jira")
        jira_data = json.loads(jira_output)
        assert "issueUpdates" in jira_data
