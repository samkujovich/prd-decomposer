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
client = OpenAI()  # Uses OPENAI_API_KEY env var


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
    response = client.chat.completions.create(
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


if __name__ == "__main__":
    app.run()
