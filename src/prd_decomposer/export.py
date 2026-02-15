"""Export ticket collections to various formats (CSV, Jira, YAML)."""

import csv
import io
import json
import logging
from datetime import UTC, datetime
from typing import Any

import yaml
from pydantic import ValidationError

from prd_decomposer.formatters import render_agent_prompt
from prd_decomposer.models import TicketCollection

logger = logging.getLogger("prd_decomposer")


def export_tickets(
    tickets_json: str,
    output_format: str = "csv",
    project_key: str = "PROJECT",
) -> str:
    """Export tickets to different formats for integration with external tools.

    Args:
        tickets_json: JSON string of ticket collection from decompose_to_tickets
        output_format: Export format - 'csv', 'jira', or 'yaml'
        project_key: Jira project key for issue creation (default: 'PROJECT')

    Returns:
        Exported content as a string.

    Supported formats:
    - csv: Flat CSV with one row per story (for spreadsheet import)
    - jira: Jira REST API bulk create payload (ready for POST to /rest/api/3/issue/bulk)
    - yaml: YAML format (for GitOps workflows)
    """
    # Parse and validate with Pydantic
    try:
        raw_data = json.loads(tickets_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tickets: {e}")

    if not isinstance(raw_data, dict):
        raise ValueError(
            f"Invalid ticket structure: must be a JSON object, got {type(raw_data).__name__}"
        )

    try:
        tickets = TicketCollection.model_validate(raw_data)
    except ValidationError as e:
        # Convert Pydantic errors to user-friendly messages
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        raise ValueError("Invalid ticket structure:\n" + "\n".join(errors)) from e

    format_lower = output_format.lower()

    if format_lower == "csv":
        return _export_to_csv(tickets)
    elif format_lower == "jira":
        return _export_to_jira(tickets, project_key)
    elif format_lower == "yaml":
        return _export_to_yaml(tickets)
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'csv', 'jira', or 'yaml'.")


def _export_to_csv(tickets: TicketCollection) -> str:
    """Export tickets to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "epic_title",
        "story_title",
        "story_description",
        "acceptance_criteria",
        "size",
        "priority",
        "labels",
        "requirement_ids",
        "agent_prompt",
    ])

    # Data rows
    for epic in tickets.epics:
        for story in epic.stories:
            # Convert Pydantic model to dict for render_agent_prompt
            story_dict = story.model_dump()
            writer.writerow([
                epic.title,
                story.title,
                story.description,
                "; ".join(story.acceptance_criteria),
                story.size,
                story.priority,
                ", ".join(story.labels),
                ", ".join(story.requirement_ids),
                render_agent_prompt(story_dict),
            ])

    return output.getvalue()


def _map_priority_to_jira(priority: str) -> str:
    """Map internal priority to Jira priority names.

    Input is validated by Pydantic as Literal["high", "medium", "low"].
    """
    return {"high": "High", "medium": "Medium", "low": "Low"}[priority]


def _export_to_jira(tickets: TicketCollection, project_key: str) -> str:
    """Export tickets to Jira REST API bulk create format.

    Returns JSON compatible with POST /rest/api/3/issue/bulk.
    Epic linking must be done post-creation using Jira's Epic Link API.
    """
    issues = []

    for epic in tickets.epics:
        # Create epic issue
        epic_issue = {
            "fields": {
                "project": {"key": project_key},
                "issuetype": {"name": "Epic"},
                "summary": epic.title,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": epic.description}],
                        }
                    ],
                },
                "labels": epic.labels,
            }
        }
        issues.append(epic_issue)

        # Create story issues
        for story in epic.stories:
            # Build description with acceptance criteria
            description_parts = [story.description]
            if story.acceptance_criteria:
                description_parts.append("\n\nAcceptance Criteria:")
                for ac in story.acceptance_criteria:
                    description_parts.append(f"- {ac}")

            story_issue = {
                "fields": {
                    "project": {"key": project_key},
                    "issuetype": {"name": "Story"},
                    "summary": story.title,
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {"type": "text", "text": "\n".join(description_parts)}
                                ],
                            }
                        ],
                    },
                    "labels": story.labels,
                    "priority": {"name": _map_priority_to_jira(story.priority)},
                }
            }
            issues.append(story_issue)

    return json.dumps({"issueUpdates": issues}, indent=2)


def _export_to_yaml(tickets: TicketCollection) -> str:
    """Export tickets to YAML format using pyyaml for spec-compliant output."""
    # Build output structure with metadata
    output: dict[str, Any] = {
        "_metadata": {
            "generator": "PRD Decomposer",
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "epics": [],
    }

    for epic in tickets.epics:
        epic_dict: dict[str, Any] = {
            "title": epic.title,
            "description": epic.description,
            "labels": epic.labels,
            "stories": [],
        }

        for story in epic.stories:
            story_dict = {
                "title": story.title,
                "description": story.description,
                "size": story.size,
                "priority": story.priority,
                "labels": story.labels,
                "requirement_ids": story.requirement_ids,
                "acceptance_criteria": story.acceptance_criteria,
            }
            epic_dict["stories"].append(story_dict)

        output["epics"].append(epic_dict)

    # Use pyyaml for proper YAML serialization
    # default_flow_style=False ensures block style for readability
    # allow_unicode=True handles international characters
    # sort_keys=False preserves field order
    return yaml.dump(
        output,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
