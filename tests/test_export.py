"""Tests for export functionality (CSV, Jira, YAML)."""

import json

import pytest
from arcade_core.errors import FatalToolError

from prd_decomposer.export import _map_priority_to_jira
from prd_decomposer.server import export_tickets


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
                            "acceptance_criteria": [
                                "Returns JWT on success", "Returns 401 on failure"
                            ],
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
        result = export_tickets(json.dumps(sample_tickets), output_format="csv")

        assert "epic_title,story_title" in result
        assert "User Authentication" in result
        assert "Login endpoint" in result
        assert "Login form UI" in result

    def test_export_to_csv_includes_all_fields(self, sample_tickets):
        """Verify CSV export includes all expected fields."""
        result = export_tickets(json.dumps(sample_tickets), output_format="csv")
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
        result = export_tickets(json.dumps(sample_tickets), output_format="jira")
        parsed = json.loads(result)

        assert "issueUpdates" in parsed
        assert len(parsed["issueUpdates"]) == 3  # 1 epic + 2 stories

    def test_export_to_jira_maps_priority(self, sample_tickets):
        """Verify Jira export maps priority correctly."""
        result = export_tickets(json.dumps(sample_tickets), output_format="jira")
        parsed = json.loads(result)

        # Find a story issue (not epic)
        story_issues = [
            i for i in parsed["issueUpdates"]
            if i["fields"]["issuetype"]["name"] == "Story"
        ]
        assert len(story_issues) == 2

        # High priority story should have "High" in Jira
        high_priority_story = story_issues[0]
        assert high_priority_story["fields"]["priority"]["name"] == "High"

    def test_export_to_jira_no_metadata_in_issues(self, sample_tickets):
        """Verify Jira issues don't contain _prd_decomposer_metadata (breaks Jira API)."""
        result = export_tickets(json.dumps(sample_tickets), output_format="jira")
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
        result = export_tickets(json.dumps(sample_tickets), output_format="yaml")

        assert "epics:" in result
        assert "title:" in result
        assert "stories:" in result
        assert "User Authentication" in result

    def test_export_invalid_format_raises(self, sample_tickets):
        """Verify export_tickets raises for invalid format."""
        with pytest.raises(FatalToolError, match="Unsupported format"):
            export_tickets(json.dumps(sample_tickets), output_format="xml")

    def test_export_invalid_json_raises(self):
        """Verify export_tickets raises for invalid JSON input."""
        with pytest.raises(FatalToolError, match="Invalid JSON"):
            export_tickets("not valid json", output_format="csv")

    def test_export_missing_epics_raises(self):
        """Verify export_tickets raises when epics key missing."""
        with pytest.raises(FatalToolError, match=r"epics.*Field required"):
            export_tickets('{"stories": []}', output_format="csv")

    def test_export_array_json_raises(self):
        """Verify export_tickets raises for JSON array input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('[{"title": "Epic"}]', output_format="csv")

    def test_export_string_json_raises(self):
        """Verify export_tickets raises for JSON string input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('"just a string"', output_format="csv")

    def test_export_number_json_raises(self):
        """Verify export_tickets raises for JSON number input."""
        with pytest.raises(FatalToolError, match="must be a JSON object"):
            export_tickets('123', output_format="csv")

    def test_export_epics_not_list_raises(self):
        """Verify export_tickets raises when epics is not a list."""
        with pytest.raises(FatalToolError, match=r"epics.*Input should be a valid list"):
            export_tickets('{"epics": "not a list"}', output_format="csv")

    def test_export_epics_contains_non_object_raises(self):
        """Verify export_tickets raises when epics contains non-objects."""
        with pytest.raises(FatalToolError, match=r"epics\.0.*Input should be a valid dictionary"):
            export_tickets('{"epics": ["string instead of object"]}', output_format="csv")

    def test_export_epics_mixed_types_raises(self):
        """Verify export_tickets raises for mixed types in epics array."""
        tickets = {
            "epics": [
                {"title": "Valid Epic", "description": "D", "labels": [], "stories": []},
                123,  # Invalid - not an object
            ]
        }
        with pytest.raises(FatalToolError, match=r"epics\.1.*Input should be a valid dictionary"):
            export_tickets(json.dumps(tickets), output_format="csv")

    def test_export_stories_not_list_raises(self):
        """Verify export_tickets raises when stories is not a list."""
        tickets = {
            "epics": [
                {"title": "Epic", "description": "D", "labels": [], "stories": "not a list"}
            ]
        }
        with pytest.raises(
            FatalToolError, match=r"epics\.0\.stories.*Input should be a valid list"
        ):
            export_tickets(json.dumps(tickets), output_format="csv")

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
        with pytest.raises(
            FatalToolError,
            match=r"epics\.0\.stories\.0.*Input should be a valid dictionary"
        ):
            export_tickets(json.dumps(tickets), output_format="csv")

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
        with pytest.raises(
            FatalToolError,
            match=r"epics\.0\.stories\.1.*Input should be a valid dictionary"
        ):
            export_tickets(json.dumps(tickets), output_format="csv")


class TestJiraPriorityMapping:
    """Tests for Jira priority mapping."""

    def test_map_priority_valid_strings(self):
        """Verify _map_priority_to_jira maps valid priorities."""
        # Pydantic validates input as Literal["high", "medium", "low"]
        assert _map_priority_to_jira("high") == "High"
        assert _map_priority_to_jira("medium") == "Medium"
        assert _map_priority_to_jira("low") == "Low"


class TestNestedFieldValidation:
    """Tests for nested story field validation in export."""

    def test_export_with_integer_labels_raises(self):
        """Verify export rejects stories with integer labels (Pydantic validation)."""
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
                            # Integer in labels - rejected by Pydantic
                            "labels": [123, "valid-label"],
                            "size": "S",
                            "requirement_ids": ["REQ-001"],
                        }
                    ],
                }
            ]
        }
        # Pydantic rejects integer labels - they must be strings
        with pytest.raises(FatalToolError, match="Input should be a valid string"):
            export_tickets(json.dumps(tickets), output_format="csv")


class TestScalarFieldValidation:
    """Tests for scalar string field validation in export (via Pydantic)."""

    def test_export_with_integer_title_rejected(self):
        """Verify export rejects integer title (Pydantic requires string)."""
        tickets = {
            "epics": [
                {
                    "title": 12345,  # Integer instead of string - rejected by Pydantic
                    "description": "Desc",
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        with pytest.raises(FatalToolError, match="Input should be a valid string"):
            export_tickets(json.dumps(tickets), output_format="csv")

    def test_export_with_integer_story_title_rejected(self):
        """Verify export rejects integer story title (Pydantic requires string)."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [
                        {
                            "title": 99999,  # Integer instead of string - rejected
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
        with pytest.raises(FatalToolError, match="Input should be a valid string"):
            export_tickets(json.dumps(tickets), output_format="csv")

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
        with pytest.raises(FatalToolError, match=r"title.*Field required"):
            export_tickets(json.dumps(tickets), output_format="csv")

    def test_export_with_missing_description_uses_default(self):
        """Verify export accepts missing description (uses default empty string)."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    # description omitted - defaults to ""
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        # Should not raise - missing description uses default
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        assert "Epic" in result

    def test_export_with_empty_title_raises(self):
        """Verify export raises for empty required title."""
        tickets = {
            "epics": [
                {
                    "title": "",  # Empty string should fail (min_length=1)
                    "description": "Desc",
                    "labels": [],
                    "stories": [],
                }
            ]
        }
        with pytest.raises(
            FatalToolError, match=r"title.*String should have at least 1 character"
        ):
            export_tickets(json.dumps(tickets), output_format="csv")


class TestYAMLSizePriorityEscaping:
    """Tests for YAML size/priority validation."""

    def test_yaml_invalid_size_rejected(self):
        """Verify invalid size values are rejected by Pydantic validation."""
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
                            "size": 'invalid',  # Not S, M, or L
                            "priority": "medium",
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        # Pydantic validates size must be S, M, or L
        with pytest.raises(FatalToolError, match=r"size.*Input should be 'S', 'M' or 'L'"):
            export_tickets(json.dumps(tickets), output_format="yaml")

    def test_yaml_invalid_priority_rejected(self):
        """Verify invalid priority values are rejected by Pydantic validation."""
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
                            "priority": "invalid",  # Not high, medium, or low
                            "requirement_ids": [],
                        }
                    ],
                }
            ]
        }
        # Pydantic validates priority must be high, medium, or low
        with pytest.raises(FatalToolError, match=r"priority.*Input should be"):
            export_tickets(json.dumps(tickets), output_format="yaml")


class TestNullStoriesHandling:
    """Tests for null stories array handling in export (Pydantic validation)."""

    def test_export_with_null_stories_rejected(self):
        """Verify export rejects null stories (Pydantic requires list)."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": None,  # null is rejected by Pydantic
                }
            ]
        }
        # Pydantic rejects null for list fields
        with pytest.raises(FatalToolError, match=r"stories.*Input should be a valid list"):
            export_tickets(json.dumps(tickets), output_format="csv")

    def test_export_with_missing_stories_uses_default(self):
        """Verify export accepts missing stories (uses default empty list)."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    # stories key omitted - defaults to []
                }
            ]
        }
        # Missing stories should use default empty list
        result = export_tickets(json.dumps(tickets), output_format="csv")
        assert "epic_title" in result  # Header should be present

    def test_export_with_empty_stories_works(self):
        """Verify export handles empty stories array."""
        tickets = {
            "epics": [
                {
                    "title": "Epic",
                    "description": "Desc",
                    "labels": [],
                    "stories": [],  # Empty list is valid
                }
            ]
        }
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        assert "stories:" in result


class TestCSVAgentPromptExport:
    """Tests for CSV export with agent_prompt column."""

    def test_csv_export_includes_agent_prompt(self):
        """CSV export includes agent_prompt column."""
        tickets = {
            "epics": [{
                "title": "Epic",
                "description": "Desc",
                "stories": [{
                    "title": "Story",
                    "description": "Do thing",
                    "size": "M",
                    "acceptance_criteria": [],
                    "labels": [],
                    "requirement_ids": [],
                    "agent_context": {
                        "goal": "The why",
                        "exploration_paths": [],
                        "exploration_hints": [],
                        "known_patterns": [],
                        "verification_tests": [],
                        "self_check": [],
                    },
                }],
                "labels": [],
            }],
        }
        result = export_tickets(json.dumps(tickets), output_format="csv")

        assert "agent_prompt" in result
        assert "The why" in result


class TestYAMLExportWithPyyaml:
    """Tests for YAML export using pyyaml serialization."""

    def test_yaml_export_single_epics_key(self):
        """Verify YAML export only has one epics: key."""
        tickets = {
            "epics": [
                {"title": "Epic 1", "description": "Desc 1", "labels": [], "stories": []},
                {"title": "Epic 2", "description": "Desc 2", "labels": [], "stories": []},
            ]
        }
        result = export_tickets(json.dumps(tickets), output_format="yaml")

        # Count occurrences of "epics:" at start of line
        lines = result.split("\n")
        epics_count = sum(1 for line in lines if line.strip() == "epics:")
        assert epics_count == 1, f"Expected 1 'epics:' key, found {epics_count}"

    def test_yaml_export_handles_quotes(self):
        """Verify YAML export handles double quotes in text (pyyaml)."""
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
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        # pyyaml handles quoting - just verify the content is present
        assert "quotes" in result
        assert "hello" in result

    def test_yaml_export_handles_newlines(self):
        """Verify YAML export handles newlines in text (pyyaml)."""
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
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        # pyyaml uses block scalar or escaped newlines - just verify export works
        assert "Multi-line" in result
        assert "Story" in result

    def test_yaml_export_handles_special_chars_in_labels(self):
        """Verify YAML export handles special characters in labels (pyyaml)."""
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
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        # pyyaml quotes strings with special chars - verify content present
        assert "foo, bar" in result
        assert "key: value" in result
        assert "has, comma" in result
        assert "REQ: 001" in result

    def test_yaml_export_empty_epics(self):
        """Verify YAML export handles empty epics list."""
        tickets = {"epics": []}
        result = export_tickets(json.dumps(tickets), output_format="yaml")
        assert "epics:" in result
