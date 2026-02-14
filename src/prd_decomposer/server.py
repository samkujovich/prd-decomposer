"""MCP server for PRD analysis and decomposition."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated

from arcade_mcp_server import MCPApp
from openai import OpenAI

from prd_decomposer.models import StructuredRequirements, TicketCollection
from prd_decomposer.prompts import ANALYZE_PRD_PROMPT, DECOMPOSE_TO_TICKETS_PROMPT

app = MCPApp(name="prd_decomposer", version="1.0.0")

# Lazy client initialization to avoid requiring API key at import time
_client = None


def get_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()  # Uses OPENAI_API_KEY env var
    return _client


@app.tool
def analyze_prd(
    prd_text: Annotated[str, "Raw PRD markdown text to analyze"]
) -> dict:
    """Analyze a PRD and extract structured requirements.

    Extracts requirements with IDs, acceptance criteria, dependencies,
    and flags ambiguous requirements (missing criteria or vague quantifiers).
    """
    # Generate source hash for traceability
    source_hash = hashlib.sha256(prd_text.encode()).hexdigest()[:8]

    # Call LLM
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": ANALYZE_PRD_PROMPT.format(prd_text=prd_text)
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # Lower temperature for more consistent output
    )

    # Parse and validate response
    data = json.loads(response.choices[0].message.content)

    # Ensure source_hash is set
    data["source_hash"] = source_hash

    # Validate with Pydantic
    validated = StructuredRequirements(**data)

    return validated.model_dump()


@app.tool
def decompose_to_tickets(
    requirements: Annotated[dict, "Structured requirements from analyze_prd"]
) -> dict:
    """Convert structured requirements into Jira-compatible epics and stories.

    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels. Output is ready for Jira import.
    """
    # Validate input
    validated_input = StructuredRequirements(**requirements)

    # Call LLM
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": DECOMPOSE_TO_TICKETS_PROMPT.format(
                    requirements_json=validated_input.model_dump_json(indent=2)
                )
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    # Parse and validate response
    data = json.loads(response.choices[0].message.content)

    # Add metadata if not present
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["metadata"]["model"] = "gpt-4o"
    data["metadata"]["requirement_count"] = len(validated_input.requirements)

    # Count stories
    story_count = sum(len(epic.get("stories", [])) for epic in data.get("epics", []))
    data["metadata"]["story_count"] = story_count

    # Validate with Pydantic
    validated = TicketCollection(**data)

    return validated.model_dump()


if __name__ == "__main__":
    app.run()
