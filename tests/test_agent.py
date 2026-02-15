"""Tests for agent command parsing and session state."""

from agent.agent import extract_requirements_from_output, handle_command, parse_command
from agent.formatters import (
    format_analysis_summary,
    format_requirements_table,
    format_ticket_summary,
    format_tickets_hierarchy,
)
from agent.session_state import SessionState


class TestParseCommand:
    """Tests for parse_command function."""

    def test_parse_accept_command(self):
        """Parse 'accept N' command."""
        cmd, idx, arg = parse_command("accept 3")
        assert cmd == "accept"
        assert idx == 3
        assert arg is None

    def test_parse_dismiss_command(self):
        """Parse 'dismiss N' command."""
        cmd, idx, arg = parse_command("dismiss 2")
        assert cmd == "dismiss"
        assert idx == 2
        assert arg is None

    def test_parse_clarify_with_quotes(self):
        """Parse 'clarify N "text"' command."""
        cmd, idx, arg = parse_command('clarify 1 "Response time under 200ms"')
        assert cmd == "clarify"
        assert idx == 1
        assert arg == "Response time under 200ms"

    def test_parse_clarify_without_quotes(self):
        """Parse 'clarify N text' command."""
        cmd, idx, arg = parse_command("clarify 1 Response time under 200ms")
        assert cmd == "clarify"
        assert idx == 1
        assert arg == "Response time under 200ms"

    def test_parse_tickets_command(self):
        """Parse 'tickets' command."""
        cmd, idx, arg = parse_command("tickets")
        assert cmd == "tickets"
        assert idx is None
        assert arg is None

    def test_parse_tickets_aliases(self):
        """Parse ticket command aliases."""
        for alias in ("ticket", "decompose"):
            cmd, _, _ = parse_command(alias)
            assert cmd == "tickets"

    def test_parse_ambiguities_command(self):
        """Parse 'ambiguities' command."""
        cmd, idx, arg = parse_command("ambiguities")
        assert cmd == "ambiguities"
        assert idx is None
        assert arg is None

    def test_parse_ambiguities_aliases(self):
        """Parse ambiguities command aliases."""
        for alias in ("ambigs", "status"):
            cmd, _, _ = parse_command(alias)
            assert cmd == "ambiguities"

    def test_parse_regular_input(self):
        """Regular input returns None command."""
        cmd, idx, arg = parse_command("analyze samples/prd.md")
        assert cmd is None
        assert idx is None
        assert arg is None

    def test_parse_case_insensitive_accept(self):
        """Accept command is case-insensitive."""
        cmd, idx, _ = parse_command("ACCEPT 1")
        assert cmd == "accept"
        assert idx == 1

    def test_parse_clarify_case_insensitive_preserves_text(self):
        """Clarify command is case-insensitive but preserves text case."""
        cmd, idx, arg = parse_command('CLARIFY 1 "Response Time Under 200ms"')
        assert cmd == "clarify"
        assert idx == 1
        assert arg == "Response Time Under 200ms"  # Preserves original case


class TestExtractRequirements:
    """Tests for extract_requirements_from_output function."""

    def test_extract_from_json_block(self):
        """Extract requirements from markdown JSON block."""
        output = '''Here are the requirements:

```json
{"requirements": [{"id": "REQ-001", "title": "Test"}], "summary": "test"}
```

Let me know if you need changes.'''
        result = extract_requirements_from_output(output)
        assert result is not None
        assert "requirements" in result
        assert result["requirements"][0]["id"] == "REQ-001"

    def test_extract_no_requirements(self):
        """Return None when no requirements found."""
        output = "I'll help you analyze the PRD."
        result = extract_requirements_from_output(output)
        assert result is None

    def test_extract_nested_json(self):
        """Extract requirements with nested objects."""
        output = '''```json
{"requirements": [{"id": "REQ-001", "ambiguity_flags": [{"category": "vague", "issue": "fast"}]}], "summary": "test"}
```'''
        result = extract_requirements_from_output(output)
        assert result is not None
        assert result["requirements"][0]["ambiguity_flags"][0]["category"] == "vague"

    def test_extract_json_with_escaped_quotes(self):
        """Extract JSON containing escaped quotes."""
        output = '''```json
{"requirements": [{"id": "REQ-001", "title": "Test \\"quoted\\" text"}], "summary": "ok"}
```'''
        result = extract_requirements_from_output(output)
        assert result is not None
        assert "quoted" in result["requirements"][0]["title"]

    def test_extract_invalid_json(self):
        """Return None for invalid JSON."""
        output = '```json\n{not valid json}\n```'
        result = extract_requirements_from_output(output)
        assert result is None

    def test_extract_unbalanced_braces(self):
        """Return None for unbalanced braces."""
        output = '```json\n{"incomplete": [\n```'
        result = extract_requirements_from_output(output)
        assert result is None


class TestSessionState:
    """Tests for SessionState class."""

    def test_store_requirements(self):
        """Store requirements and reset decisions."""
        session = SessionState()
        session.accepted_ambiguities.add("old-id")

        requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "ambiguity_flags": [{"category": "vague_quantifier", "severity": "warning", "issue": "fast"}],
                }
            ]
        }
        session.store_requirements(requirements)

        assert session.current_requirements == requirements
        assert len(session.accepted_ambiguities) == 0  # Reset

    def test_get_ambiguities(self):
        """Get ambiguities from requirements."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "ambiguity_flags": [
                        {"category": "vague_quantifier", "severity": "warning", "issue": "fast", "suggested_action": "define SLA"},
                        {"category": "missing_criteria", "severity": "critical", "issue": "no AC", "suggested_action": "add AC"},
                    ],
                }
            ]
        })

        ambiguities = session.get_ambiguities()
        assert len(ambiguities) == 2
        assert ambiguities[0]["requirement_id"] == "REQ-001"
        assert ambiguities[0]["category"] == "vague_quantifier"

    def test_ambiguity_ids_are_unique(self):
        """Two ambiguities with same category get unique IDs."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "ambiguity_flags": [
                        {"category": "vague_quantifier", "severity": "warning", "issue": "fast", "suggested_action": "define"},
                        {"category": "vague_quantifier", "severity": "warning", "issue": "scalable", "suggested_action": "define"},
                    ],
                }
            ]
        })

        ambiguities = session.get_ambiguities()
        assert len(ambiguities) == 2
        # IDs should be unique even with same category
        ids = [a["id"] for a in ambiguities]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
        assert "REQ-001:vague_quantifier:0" in ids
        assert "REQ-001:vague_quantifier:1" in ids

    def test_accept_ambiguity(self):
        """Accept an ambiguity by index."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "ambiguity_flags": [{"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"}],
                }
            ]
        })

        result = session.accept_ambiguity(1)
        assert result is not None
        assert len(session.get_active_ambiguities()) == 0

    def test_accept_invalid_index(self):
        """Return None for invalid index."""
        session = SessionState()
        session.store_requirements({"requirements": []})

        result = session.accept_ambiguity(99)
        assert result is None

    def test_add_clarification(self):
        """Add clarification to a requirement."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "description": "Original description",
                    "ambiguity_flags": [{"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"}],
                }
            ]
        })

        result = session.add_clarification(1, "Response time under 200ms")
        assert result == "REQ-001"
        assert "REQ-001" in session.clarifications
        assert session.clarifications["REQ-001"] == "Response time under 200ms"

    def test_get_requirements_with_clarifications(self):
        """Get requirements with clarifications injected."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {"id": "REQ-001", "title": "Test", "description": "Original", "ambiguity_flags": []},
            ]
        })
        session.clarifications["REQ-001"] = "Clarification text"

        result = session.get_requirements_with_clarifications()
        assert "**Clarification:** Clarification text" in result["requirements"][0]["description"]

    def test_format_ambiguities_display(self):
        """Format ambiguities for display."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {
                    "id": "REQ-001",
                    "title": "Test",
                    "ambiguity_flags": [
                        {"category": "vague", "severity": "critical", "issue": "fast undefined", "suggested_action": "define SLA"},
                    ],
                }
            ]
        })

        display = session.format_ambiguities_display()
        assert "1 Ambiguity to Review" in display
        assert "🔴" in display  # critical emoji
        assert "REQ-001" in display
        assert "fast undefined" in display


class TestHandleCommand:
    """Tests for handle_command function."""

    def test_handle_ambiguities(self):
        """Handle ambiguities command."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {"id": "REQ-001", "title": "Test", "ambiguity_flags": [
                    {"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"}
                ]},
            ]
        })

        result = handle_command("ambiguities", None, None, session)
        assert "1 Ambiguity" in result

    def test_handle_accept(self):
        """Handle accept command."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {"id": "REQ-001", "title": "Test", "ambiguity_flags": [
                    {"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"}
                ]},
            ]
        })

        result = handle_command("accept", 1, None, session)
        assert "Accepted ambiguity #1" in result
        assert "0 ambiguities remaining" in result

    def test_handle_accept_no_index(self):
        """Handle accept without index."""
        session = SessionState()
        result = handle_command("accept", None, None, session)
        assert "Usage:" in result

    def test_handle_dismiss(self):
        """Handle dismiss command."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {"id": "REQ-001", "title": "Test", "ambiguity_flags": [
                    {"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"},
                    {"category": "missing", "severity": "critical", "issue": "no AC", "suggested_action": "add"},
                ]},
            ]
        })

        result = handle_command("dismiss", 1, None, session)
        assert "Dismissed ambiguity #1" in result
        assert "1 ambiguity remaining" in result  # singular

    def test_handle_dismiss_no_index(self):
        """Handle dismiss without index."""
        session = SessionState()
        result = handle_command("dismiss", None, None, session)
        assert "Usage:" in result

    def test_handle_clarify(self):
        """Handle clarify command."""
        session = SessionState()
        session.store_requirements({
            "requirements": [
                {"id": "REQ-001", "title": "Test", "ambiguity_flags": [
                    {"category": "vague", "severity": "warning", "issue": "test", "suggested_action": "fix"}
                ]},
            ]
        })

        result = handle_command("clarify", 1, "under 200ms", session)
        assert "Added clarification to REQ-001" in result
        assert "0 ambiguities remaining" in result


class TestFormatters:
    """Tests for output formatters."""

    def test_format_requirements_table(self):
        """Format requirements as markdown table."""
        requirements = {
            "requirements": [
                {"id": "REQ-001", "title": "User login", "priority": "high", "ambiguity_flags": [{"category": "vague"}]},
                {"id": "REQ-002", "title": "Dashboard", "priority": "medium", "ambiguity_flags": []},
            ]
        }
        result = format_requirements_table(requirements)

        assert "## Requirements" in result
        assert "| REQ-001 |" in result
        assert "| REQ-002 |" in result
        assert "| high |" in result
        assert "| 1 |" in result  # ambiguity count
        assert "| - |" in result  # no ambiguities

    def test_format_requirements_table_empty(self):
        """Format empty requirements."""
        result = format_requirements_table({"requirements": []})
        assert "No requirements found" in result

    def test_format_tickets_hierarchy(self):
        """Format tickets as tree hierarchy."""
        tickets = {
            "epics": [
                {
                    "title": "Authentication",
                    "stories": [
                        {"title": "Login form", "size": "S"},
                        {"title": "Password reset", "size": "M"},
                    ],
                },
                {
                    "title": "Dashboard",
                    "stories": [
                        {"title": "Charts", "size": "L"},
                    ],
                },
            ]
        }
        result = format_tickets_hierarchy(tickets)

        assert "## Tickets" in result
        assert "📦 **Authentication**" in result
        assert "📦 **Dashboard**" in result
        assert "🟢 [S] Login form" in result
        assert "🟡 [M] Password reset" in result
        assert "🔴 [L] Charts" in result
        assert "2 epics" in result
        assert "3 stories" in result

    def test_format_tickets_hierarchy_empty(self):
        """Format empty tickets."""
        result = format_tickets_hierarchy({"epics": []})
        assert "No tickets generated" in result

    def test_format_analysis_summary(self):
        """Format analysis summary with ambiguity counts."""
        requirements = {
            "requirements": [
                {
                    "id": "REQ-001",
                    "ambiguity_flags": [
                        {"severity": "critical"},
                        {"severity": "warning"},
                    ],
                },
                {
                    "id": "REQ-002",
                    "ambiguity_flags": [
                        {"severity": "suggestion"},
                    ],
                },
            ]
        }
        result = format_analysis_summary(requirements)

        assert "## Analysis Complete" in result
        assert "2 requirements" in result
        assert "3 ambiguities" in result
        assert "🔴 1 critical" in result
        assert "🟡 1 warning" in result
        assert "💡 1 suggestion" in result

    def test_format_analysis_summary_no_ambiguities(self):
        """Format analysis summary with no ambiguities."""
        requirements = {
            "requirements": [{"id": "REQ-001", "ambiguity_flags": []}]
        }
        result = format_analysis_summary(requirements)

        assert "✅ No ambiguities detected" in result

    def test_format_ticket_summary(self):
        """Format ticket summary with size breakdown."""
        tickets = {
            "epics": [
                {
                    "title": "Epic 1",
                    "stories": [
                        {"title": "S1", "size": "S"},
                        {"title": "S2", "size": "S"},
                        {"title": "S3", "size": "M"},
                    ],
                }
            ]
        }
        result = format_ticket_summary(tickets)

        assert "## Decomposition Complete" in result
        assert "1 epic" in result
        assert "3 stories" in result
        assert "🟢 2 Small" in result
        assert "🟡 1 Medium" in result
        assert "🔴 0 Large" in result
