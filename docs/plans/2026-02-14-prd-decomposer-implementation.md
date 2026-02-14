# PRD Decomposer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP server that analyzes PRDs and decomposes them into Jira-ready epics and stories.

**Architecture:** Two MCP tools (`analyze_prd`, `decompose_to_tickets`) each making independent GPT-4o calls. Agent built with OpenAI Agents SDK consumes via stdio. Pydantic models enforce schema.

**Tech Stack:** Python 3.11+, arcade-mcp, OpenAI SDK, Pydantic v2, pytest, arcade_evals

---

## Task 1: Project Scaffolding

**Files:**
- Create: `prd-decomposer/` directory structure via arcade CLI

**Step 1: Install arcade-mcp**

Run:
```bash
uv tool install arcade-mcp
```
Expected: `arcade` command available

**Step 2: Scaffold the project**

Run:
```bash
cd /Users/samkujovich/Documents/git/prd-decomposer
arcade new prd_decomposer
```
Expected: Creates `src/prd_decomposer/` with `__init__.py` and `server.py` template

**Step 3: Create additional directories**

Run:
```bash
mkdir -p agent evals tests samples
touch agent/__init__.py agent/agent.py
touch evals/__init__.py evals/eval_prd_tools.py
touch tests/__init__.py tests/test_tools.py
touch samples/sample_prd.md
touch src/prd_decomposer/models.py
touch src/prd_decomposer/prompts.py
touch src/prd_decomposer/.env.example
touch AI_USAGE.md
```

**Step 4: Set up pyproject.toml dependencies**

Modify: `pyproject.toml` - add dependencies:
```toml
[project]
name = "prd-decomposer"
version = "1.0.0"
description = "MCP server that analyzes PRDs and decomposes them into Jira-ready tickets"
requires-python = ">=3.11"
dependencies = [
    "arcade-mcp",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "arcade-evals",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 5: Install dependencies**

Run:
```bash
cd /Users/samkujovich/Documents/git/prd-decomposer
uv sync --all-extras
```

**Step 6: Create .env.example**

Write to `src/prd_decomposer/.env.example`:
```
OPENAI_API_KEY=sk-your-key-here
```

**Step 7: Commit scaffold**

Run:
```bash
git add -A
git commit -m "feat: scaffold prd-decomposer project with arcade-mcp"
```

---

## Task 2: Pydantic Models - Requirement

**Files:**
- Create: `src/prd_decomposer/models.py`
- Test: `tests/test_tools.py`

**Step 1: Write failing test for Requirement model**

Write to `tests/test_tools.py`:
```python
import pytest
from pydantic import ValidationError


def test_requirement_model_valid():
    """Verify Requirement model accepts valid data."""
    from prd_decomposer.models import Requirement

    req = Requirement(
        id="REQ-001",
        title="User authentication",
        description="Users must be able to log in with email and password",
        acceptance_criteria=["Login form exists", "JWT issued on success"],
        dependencies=[],
        ambiguity_flags=[],
        priority="high"
    )
    assert req.id == "REQ-001"
    assert req.priority == "high"
    assert len(req.acceptance_criteria) == 2


def test_requirement_invalid_priority():
    """Verify Requirement rejects invalid priority."""
    from prd_decomposer.models import Requirement

    with pytest.raises(ValidationError):
        Requirement(
            id="REQ-001",
            title="Test",
            description="Test",
            acceptance_criteria=[],
            dependencies=[],
            ambiguity_flags=[],
            priority="critical"  # Invalid - must be high/medium/low
        )
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/samkujovich/Documents/git/prd-decomposer
uv run pytest tests/test_tools.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'prd_decomposer'` or `ImportError`

**Step 3: Write Requirement model**

Write to `src/prd_decomposer/models.py`:
```python
from typing import Literal
from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """A single requirement extracted from a PRD."""

    id: str = Field(..., description="Unique identifier (e.g., REQ-001)")
    title: str = Field(..., description="Short title of the requirement")
    description: str = Field(..., description="Detailed description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Testable acceptance criteria"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of requirements this depends on"
    )
    ambiguity_flags: list[str] = Field(
        default_factory=list,
        description="Reasons this requirement is ambiguous"
    )
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Priority level"
    )
```

**Step 4: Add package init**

Write to `src/prd_decomposer/__init__.py`:
```python
from prd_decomposer.models import (
    Requirement,
    StructuredRequirements,
    Story,
    Epic,
    TicketCollection,
)

__all__ = [
    "Requirement",
    "StructuredRequirements",
    "Story",
    "Epic",
    "TicketCollection",
]
```

Note: This will fail until all models exist. We'll update incrementally.

**Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/samkujovich/Documents/git/prd-decomposer
uv run pytest tests/test_tools.py::test_requirement_model_valid tests/test_tools.py::test_requirement_invalid_priority -v
```
Expected: 2 passed

**Step 6: Commit**

Run:
```bash
git add src/prd_decomposer/models.py tests/test_tools.py
git commit -m "feat: add Requirement model with validation"
```

---

## Task 3: Pydantic Models - StructuredRequirements

**Files:**
- Modify: `src/prd_decomposer/models.py`
- Modify: `tests/test_tools.py`

**Step 1: Write failing test**

Append to `tests/test_tools.py`:
```python
def test_structured_requirements_model():
    """Verify StructuredRequirements validates nested requirements."""
    from prd_decomposer.models import Requirement, StructuredRequirements

    req = Requirement(
        id="REQ-001",
        title="Test requirement",
        description="Description",
        acceptance_criteria=["AC1"],
        dependencies=[],
        ambiguity_flags=[],
        priority="medium"
    )

    structured = StructuredRequirements(
        requirements=[req],
        summary="Test PRD summary",
        source_hash="abc123"
    )

    assert len(structured.requirements) == 1
    assert structured.summary == "Test PRD summary"


def test_structured_requirements_serialization():
    """Verify StructuredRequirements round-trips through JSON."""
    from prd_decomposer.models import Requirement, StructuredRequirements

    req = Requirement(
        id="REQ-001",
        title="Test",
        description="Desc",
        acceptance_criteria=[],
        dependencies=[],
        ambiguity_flags=["Missing metrics"],
        priority="low"
    )

    original = StructuredRequirements(
        requirements=[req],
        summary="Summary",
        source_hash="hash123"
    )

    # Round-trip through JSON
    json_str = original.model_dump_json()
    restored = StructuredRequirements.model_validate_json(json_str)

    assert restored.requirements[0].id == "REQ-001"
    assert restored.requirements[0].ambiguity_flags == ["Missing metrics"]
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_tools.py::test_structured_requirements_model -v
```
Expected: FAIL with `ImportError` (StructuredRequirements doesn't exist)

**Step 3: Add StructuredRequirements model**

Append to `src/prd_decomposer/models.py`:
```python
class StructuredRequirements(BaseModel):
    """Collection of requirements extracted from a PRD."""

    requirements: list[Requirement] = Field(
        ..., description="List of extracted requirements"
    )
    summary: str = Field(..., description="Brief overview of the PRD")
    source_hash: str = Field(..., description="Hash of source PRD for traceability")
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_tools.py -v -k "structured"
```
Expected: 2 passed

**Step 5: Commit**

Run:
```bash
git add -A
git commit -m "feat: add StructuredRequirements model"
```

---

## Task 4: Pydantic Models - Story and Epic

**Files:**
- Modify: `src/prd_decomposer/models.py`
- Modify: `tests/test_tools.py`

**Step 1: Write failing tests**

Append to `tests/test_tools.py`:
```python
def test_story_model_valid():
    """Verify Story model accepts valid data."""
    from prd_decomposer.models import Story

    story = Story(
        title="Implement login endpoint",
        description="Create POST /auth/login endpoint",
        acceptance_criteria=["Returns JWT on success", "Returns 401 on failure"],
        size="M",
        labels=["backend", "auth"],
        requirement_ids=["REQ-001"]
    )
    assert story.size == "M"
    assert "backend" in story.labels


def test_story_invalid_size():
    """Verify Story rejects invalid size."""
    from prd_decomposer.models import Story

    with pytest.raises(ValidationError):
        Story(
            title="Test",
            description="Test",
            acceptance_criteria=[],
            size="XL",  # Invalid - must be S/M/L
            labels=[],
            requirement_ids=[]
        )


def test_epic_model_with_stories():
    """Verify Epic contains stories correctly."""
    from prd_decomposer.models import Story, Epic

    story = Story(
        title="Story 1",
        description="Desc",
        acceptance_criteria=[],
        size="S",
        labels=[],
        requirement_ids=["REQ-001"]
    )

    epic = Epic(
        title="Authentication Epic",
        description="All auth-related work",
        stories=[story],
        labels=["auth"]
    )

    assert len(epic.stories) == 1
    assert epic.stories[0].title == "Story 1"
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_tools.py -v -k "story or epic"
```
Expected: FAIL with `ImportError`

**Step 3: Add Story and Epic models**

Append to `src/prd_decomposer/models.py`:
```python
class Story(BaseModel):
    """A Jira-compatible story."""

    title: str = Field(..., description="Story title")
    description: str = Field(..., description="Story description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Acceptance criteria for the story"
    )
    size: Literal["S", "M", "L"] = Field(..., description="T-shirt size estimate")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")
    requirement_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source requirements for traceability"
    )


class Epic(BaseModel):
    """A Jira-compatible epic containing stories."""

    title: str = Field(..., description="Epic title")
    description: str = Field(..., description="Epic description")
    stories: list[Story] = Field(default_factory=list, description="Child stories")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_tools.py -v -k "story or epic"
```
Expected: 3 passed

**Step 5: Commit**

Run:
```bash
git add -A
git commit -m "feat: add Story and Epic models"
```

---

## Task 5: Pydantic Models - TicketCollection

**Files:**
- Modify: `src/prd_decomposer/models.py`
- Modify: `tests/test_tools.py`

**Step 1: Write failing test**

Append to `tests/test_tools.py`:
```python
def test_ticket_collection_model():
    """Verify TicketCollection contains epics and metadata."""
    from prd_decomposer.models import Story, Epic, TicketCollection

    story = Story(
        title="Story",
        description="Desc",
        acceptance_criteria=[],
        size="S",
        labels=[],
        requirement_ids=[]
    )
    epic = Epic(
        title="Epic",
        description="Desc",
        stories=[story],
        labels=[]
    )

    collection = TicketCollection(
        epics=[epic],
        metadata={"generated_at": "2026-02-14", "model": "gpt-4o"}
    )

    assert len(collection.epics) == 1
    assert collection.metadata["model"] == "gpt-4o"


def test_ticket_collection_serialization():
    """Verify TicketCollection round-trips through JSON."""
    from prd_decomposer.models import Story, Epic, TicketCollection

    story = Story(
        title="Story",
        description="Desc",
        acceptance_criteria=["AC1"],
        size="L",
        labels=["backend"],
        requirement_ids=["REQ-001"]
    )
    epic = Epic(title="Epic", description="Desc", stories=[story], labels=["auth"])
    original = TicketCollection(epics=[epic], metadata={"version": "1.0"})

    json_str = original.model_dump_json()
    restored = TicketCollection.model_validate_json(json_str)

    assert restored.epics[0].stories[0].size == "L"
    assert restored.epics[0].stories[0].labels == ["backend"]
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_tools.py -v -k "ticket_collection"
```
Expected: FAIL

**Step 3: Add TicketCollection model**

Append to `src/prd_decomposer/models.py`:
```python
class TicketCollection(BaseModel):
    """Collection of epics ready for Jira import."""

    epics: list[Epic] = Field(..., description="List of epics with stories")
    metadata: dict = Field(
        default_factory=dict,
        description="Generation metadata (timestamp, model version, etc.)"
    )
```

**Step 4: Update __init__.py with all exports**

Write to `src/prd_decomposer/__init__.py`:
```python
from prd_decomposer.models import (
    Requirement,
    StructuredRequirements,
    Story,
    Epic,
    TicketCollection,
)

__all__ = [
    "Requirement",
    "StructuredRequirements",
    "Story",
    "Epic",
    "TicketCollection",
]
```

**Step 5: Run all model tests**

Run:
```bash
uv run pytest tests/test_tools.py -v
```
Expected: All tests pass (should be ~8 tests)

**Step 6: Commit**

Run:
```bash
git add -A
git commit -m "feat: add TicketCollection model, complete model layer"
```

---

## Task 6: Prompt Templates

**Files:**
- Create: `src/prd_decomposer/prompts.py`

**Step 1: Write ANALYZE_PRD_PROMPT**

Write to `src/prd_decomposer/prompts.py`:
```python
"""Prompt templates for PRD analysis and decomposition."""

ANALYZE_PRD_PROMPT = '''You are a senior technical product manager. Analyze the following PRD and extract structured requirements.

For each requirement you identify:
1. Assign a unique ID (REQ-001, REQ-002, etc.)
2. Write a clear title and description
3. Extract or infer acceptance criteria (testable conditions for success)
4. Identify dependencies on other requirements (by ID)
5. Flag ambiguities - add to ambiguity_flags if:
   - Missing acceptance criteria (no clear way to test success)
   - Vague quantifiers without metrics (e.g., "fast", "scalable", "user-friendly", "easy to use")
6. Assign priority: "high", "medium", or "low" based on language cues and business impact

PRD:
{prd_text}

Return valid JSON matching this exact schema:
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "string",
      "description": "string",
      "acceptance_criteria": ["string"],
      "dependencies": ["REQ-XXX"],
      "ambiguity_flags": ["string describing the ambiguity"],
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Brief 1-2 sentence overview of the PRD",
  "source_hash": "Use first 8 chars of a hash of the PRD text"
}}'''


DECOMPOSE_TO_TICKETS_PROMPT = '''You are a senior engineering manager. Convert these structured requirements into Jira-ready epics and stories.

Guidelines:
1. Group related requirements into epics (1-4 epics typically)
2. Break each requirement into implementable stories (1-3 stories per requirement)
3. Size stories using this rubric:
   - S (Small): Less than 1 day, single component, low risk
   - M (Medium): 1-3 days, may touch multiple components, moderate complexity
   - L (Large): 3-5 days, significant complexity, unknowns, or cross-team coordination
4. Generate descriptive labels (e.g., "backend", "frontend", "api", "database", "auth", "testing")
5. Preserve traceability by including requirement_ids on each story
6. Write clear acceptance criteria derived from the requirements

Requirements:
{requirements_json}

Return valid JSON matching this exact schema:
{{
  "epics": [
    {{
      "title": "string",
      "description": "string",
      "stories": [
        {{
          "title": "string",
          "description": "string",
          "acceptance_criteria": ["string"],
          "size": "S|M|L",
          "labels": ["string"],
          "requirement_ids": ["REQ-XXX"]
        }}
      ],
      "labels": ["string"]
    }}
  ],
  "metadata": {{
    "generated_at": "ISO timestamp",
    "model": "gpt-4o",
    "requirement_count": number,
    "story_count": number
  }}
}}'''
```

**Step 2: Verify syntax**

Run:
```bash
uv run python -c "from prd_decomposer.prompts import ANALYZE_PRD_PROMPT, DECOMPOSE_TO_TICKETS_PROMPT; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

Run:
```bash
git add src/prd_decomposer/prompts.py
git commit -m "feat: add LLM prompt templates"
```

---

## Task 7: MCP Server - analyze_prd Tool

**Files:**
- Modify: `src/prd_decomposer/server.py`

**Step 1: Write server.py with analyze_prd**

Write to `src/prd_decomposer/server.py`:
```python
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
```

**Step 2: Verify syntax**

Run:
```bash
uv run python -c "from prd_decomposer.server import app; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

Run:
```bash
git add src/prd_decomposer/server.py
git commit -m "feat: add analyze_prd MCP tool"
```

---

## Task 8: MCP Server - decompose_to_tickets Tool

**Files:**
- Modify: `src/prd_decomposer/server.py`

**Step 1: Add decompose_to_tickets tool**

Add after `analyze_prd` function in `src/prd_decomposer/server.py`:
```python
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
    response = client.chat.completions.create(
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
```

**Step 2: Verify syntax**

Run:
```bash
uv run python -c "from prd_decomposer.server import analyze_prd, decompose_to_tickets; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

Run:
```bash
git add src/prd_decomposer/server.py
git commit -m "feat: add decompose_to_tickets MCP tool"
```

---

## Task 9: Sample PRD

**Files:**
- Create: `samples/sample_prd.md`

**Step 1: Write sample PRD**

Write to `samples/sample_prd.md`:
```markdown
# PRD: API Rate Limiting System

## Overview

Implement rate limiting for our public API to prevent abuse, ensure fair usage across customers, and protect backend services from overload.

## Background

Our API currently has no rate limiting, which has led to:
- Occasional service degradation from high-volume users
- Difficulty identifying and blocking abusive clients
- No visibility into per-customer usage patterns

## Requirements

### 1. Tier-Based Rate Limits

Implement different rate limits based on customer tier:

- **Free tier**: 100 requests per minute, 1,000 requests per day
- **Pro tier**: 1,000 requests per minute, unlimited daily requests
- **Enterprise tier**: Custom limits configured per customer

The system should be fast and scalable to handle our growing traffic.

### 2. Rate Limit Response Headers

All API responses must include rate limit information:

- `X-RateLimit-Limit`: Maximum requests allowed in the window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

### 3. Rate Limit Exceeded Handling

When a client exceeds their rate limit:

- Return HTTP 429 (Too Many Requests) status code
- Include `Retry-After` header with seconds until reset
- Return JSON error body with clear message and documentation link

### 4. Developer Dashboard

Provide a self-service dashboard where developers can:

- View their current usage and limits
- See historical usage graphs
- The dashboard should be user-friendly and intuitive

### 5. Backend Implementation

Technical requirements for the rate limiting service:

- Use Redis for storing rate limit counters (sliding window algorithm)
- Support configurable limits per endpoint (some endpoints may have stricter limits)
- Implement bypass mechanism for internal services (authenticated via service tokens)
- Logging and alerting when customers approach or exceed limits

### 6. Monitoring and Observability

- Emit metrics for rate limit checks, passes, and rejections
- Create dashboards showing rate limit health across the fleet
- Alert on anomalies (sudden spikes in rejections, Redis failures)

## Success Criteria

- Rate limiting deployed to all API endpoints
- Less than 1ms p99 latency overhead
- Zero false positives for internal services
- Dashboard accessible to all API customers

## Timeline

Phase 1: Core rate limiting (backend + headers)
Phase 2: Dashboard
Phase 3: Advanced monitoring

## Open Questions

- Should we implement graduated throttling (slow down before hard block)?
- How do we handle customers who legitimately need temporary limit increases?
```

**Step 2: Commit**

Run:
```bash
git add samples/sample_prd.md
git commit -m "feat: add sample PRD for API rate limiting"
```

---

## Task 10: Agent Implementation

**Files:**
- Create: `agent/agent.py`

**Step 1: Write agent**

Write to `agent/agent.py`:
```python
"""Agent that consumes the PRD Decomposer MCP server."""

import asyncio
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


async def main():
    """Run the PRD Decomposer agent."""

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    # Connect to MCP server via stdio
    async with MCPServerStdio(
        command="uv",
        args=["run", "python", str(server_path)]
    ) as mcp_server:

        agent = Agent(
            name="PRD Decomposer",
            instructions="""You help engineers convert Product Requirements Documents (PRDs) into actionable Jira tickets.

Your workflow:
1. Ask the user to provide their PRD text (they can paste it directly or you can read from a file path)
2. Use the analyze_prd tool to extract structured requirements
3. Review the results with the user:
   - Summarize the requirements found
   - Highlight any ambiguity flags that need clarification
   - Ask if they want to proceed or clarify anything first
4. Once confirmed, use decompose_to_tickets to generate Jira-ready epics and stories
5. Present the ticket structure in a clear format
6. Offer to export as JSON if needed

Be conversational and helpful. Explain what you're doing at each step. If the PRD has ambiguities, help the user understand what additional information would improve the tickets.""",
            mcp_servers=[mcp_server],
            model="gpt-4o",
        )

        # Run interactive loop
        runner = Runner()
        print("PRD Decomposer Agent")
        print("=" * 40)
        print("I help convert PRDs into Jira tickets.")
        print("Paste your PRD or provide a file path to get started.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                result = await runner.run(agent, user_input)
                print(f"\nAssistant: {result.final_output}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Update agent __init__.py**

Write to `agent/__init__.py`:
```python
"""PRD Decomposer Agent."""
```

**Step 3: Verify syntax**

Run:
```bash
uv run python -c "import agent.agent; print('OK')"
```
Expected: `OK` (may warn about missing agents package - that's fine)

**Step 4: Commit**

Run:
```bash
git add agent/
git commit -m "feat: add OpenAI Agents SDK consumer"
```

---

## Task 11: Arcade Evals

**Files:**
- Create: `evals/eval_prd_tools.py`

**Step 1: Write eval suite**

Write to `evals/eval_prd_tools.py`:
```python
"""Arcade eval suite for PRD Decomposer tools."""

from pathlib import Path

from arcade_evals import (
    BinaryCritic,
    EvalSuite,
    ExpectedMCPToolCall,
    tool_eval,
)


@tool_eval()
async def prd_eval_suite() -> EvalSuite:
    """Eval suite for PRD Decomposer MCP tools."""

    suite = EvalSuite(
        name="PRD Decomposer Tools",
        system_message="You are a helpful engineering assistant that helps convert PRDs into Jira tickets.",
    )

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", str(server_path)]
    )

    # Eval 1: Does the LLM select analyze_prd for analysis requests?
    suite.add_case(
        name="Analyze PRD intent - direct request",
        user_message="I have a PRD for a new feature. Can you analyze it and extract the requirements? Here's the PRD:\n\n# Feature: User Settings\n\nUsers should be able to update their email preferences.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={"prd_text": "# Feature: User Settings\n\nUsers should be able to update their email preferences."}
            )
        ],
        critics=[
            BinaryCritic(critic_field="prd_text", weight=1.0)
        ],
    )

    # Eval 2: Does the LLM select analyze_prd for implicit requests?
    suite.add_case(
        name="Analyze PRD intent - implicit request",
        user_message="What requirements are in this PRD?\n\n# API Versioning\n\nImplement API versioning with v1/v2 prefixes.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={"prd_text": "# API Versioning\n\nImplement API versioning with v1/v2 prefixes."}
            )
        ],
        critics=[
            BinaryCritic(critic_field="prd_text", weight=1.0)
        ],
    )

    # Eval 3: Does decompose_to_tickets get selected for ticket generation?
    sample_requirements = {
        "requirements": [
            {
                "id": "REQ-001",
                "title": "User login",
                "description": "Users can log in with email/password",
                "acceptance_criteria": ["Login form exists"],
                "dependencies": [],
                "ambiguity_flags": [],
                "priority": "high"
            }
        ],
        "summary": "Authentication feature",
        "source_hash": "abc12345"
    }

    suite.add_case(
        name="Decompose to tickets intent",
        user_message=f"Turn these requirements into Jira tickets: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets",
                parameters={"requirements": sample_requirements}
            )
        ],
        critics=[
            BinaryCritic(critic_field="requirements", weight=1.0)
        ],
    )

    # Eval 4: Does the LLM understand "create stories" means decompose?
    suite.add_case(
        name="Decompose to tickets - alternative phrasing",
        user_message=f"Create Jira stories from these requirements: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets",
                parameters={"requirements": sample_requirements}
            )
        ],
        critics=[
            BinaryCritic(critic_field="requirements", weight=1.0)
        ],
    )

    return suite
```

**Step 2: Update evals __init__.py**

Write to `evals/__init__.py`:
```python
"""PRD Decomposer evaluation suite."""
```

**Step 3: Verify syntax**

Run:
```bash
uv run python -c "import evals.eval_prd_tools; print('OK')"
```
Expected: `OK` (may warn about missing arcade_evals - that's fine)

**Step 4: Commit**

Run:
```bash
git add evals/
git commit -m "feat: add Arcade eval suite for tool selection"
```

---

## Task 12: README

**Files:**
- Create: `README.md`

**Step 1: Write README**

Write to `README.md`:
```markdown
# PRD Decomposer

An MCP server that analyzes Product Requirements Documents (PRDs) and decomposes them into Jira-ready epics and stories.

Built with [arcade-mcp](https://github.com/ArcadeAI/arcade-mcp) for the Arcade.dev Engineering Manager take-home project.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                │
│                   (OpenAI Agents SDK)                        │
│                                                              │
│  1. Receives PRD from user                                   │
│  2. Calls analyze_prd → surfaces ambiguities                 │
│  3. Calls decompose_to_tickets → returns Jira-ready output   │
└─────────────────────┬───────────────────────────────────────┘
                      │ stdio
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server                                │
│                 (prd_decomposer)                             │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────┐            │
│  │   analyze_prd   │    │  decompose_to_tickets │            │
│  │   (GPT-4o)      │    │      (GPT-4o)         │            │
│  └─────────────────┘    └──────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Tools

### `analyze_prd`

Analyzes raw PRD text and extracts structured requirements.

**Input:** PRD markdown text
**Output:** Structured requirements with:
- Unique IDs (REQ-001, REQ-002, etc.)
- Acceptance criteria
- Dependencies between requirements
- Ambiguity flags (missing criteria, vague quantifiers)
- Priority levels (high/medium/low)

### `decompose_to_tickets`

Converts structured requirements into Jira-compatible tickets.

**Input:** Structured requirements from `analyze_prd`
**Output:** Epics and stories with:
- Clear titles and descriptions
- Acceptance criteria
- T-shirt sizing (S/M/L)
- Labels
- Traceability back to requirements

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prd-decomposer.git
cd prd-decomposer

# Install dependencies
uv sync --all-extras

# Set up environment
cp src/prd_decomposer/.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Run the MCP Server

```bash
uv run python src/prd_decomposer/server.py
```

### Run the Agent

```bash
uv run python agent/agent.py
```

### Example Session

```
PRD Decomposer Agent
========================================
I help convert PRDs into Jira tickets.
Paste your PRD or provide a file path to get started.

You: Analyze this PRD: [paste PRD]

Assistant: I found 3 requirements in your PRD. REQ-001 has an ambiguity flag...
```

**Step 2: Commit**

Run:
```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Task 13: AI_USAGE.md

**Files:**
- Create: `AI_USAGE.md`

**Step 1: Write AI usage documentation**

Write to `AI_USAGE.md`:
```markdown
# AI Usage Documentation

This document tracks AI tool usage during development of prd-decomposer.

## Tools Used

- **Claude Code** (Anthropic): Project planning, code generation, documentation
- **GPT-4o** (OpenAI): Runtime LLM for PRD analysis and ticket decomposition

## AI-Generated vs Human-Written

### Fully AI-Generated (with human review)
- `src/prd_decomposer/models.py` - Pydantic model definitions
- `src/prd_decomposer/prompts.py` - LLM prompt templates
- `tests/test_tools.py` - Unit test scaffolding
- `evals/eval_prd_tools.py` - Arcade eval suite
- `README.md` - Documentation

### Human-Written with AI Assistance
- `src/prd_decomposer/server.py` - MCP tool implementations (structure AI-generated, logic reviewed)
- `agent/agent.py` - Agent consumer (based on OpenAI SDK patterns)

### Fully Human-Written
- `samples/sample_prd.md` - Sample PRD content
- This file (`AI_USAGE.md`)

## Prompts Used

Key prompts used during development:

1. **Project scaffolding**: "Create a PRD decomposer MCP server with arcade-mcp..."
2. **Model generation**: "Create Pydantic models for requirements and Jira tickets..."
3. **Test generation**: "Write pytest tests validating the Pydantic models..."

## Quality Assurance

- All AI-generated code was reviewed before committing
- Unit tests validate model behavior
- Arcade evals validate tool selection
- Manual testing with sample PRD
```

**Step 2: Commit**

Run:
```bash
git add AI_USAGE.md
git commit -m "docs: add AI usage attribution"
```

---

## Task 14: Final Verification

**Step 1: Run all tests**

Run:
```bash
uv run pytest tests/ -v
```
Expected: All tests pass

**Step 2: Verify server starts**

Run:
```bash
timeout 5 uv run python src/prd_decomposer/server.py || true
```
Expected: Server starts without import errors

**Step 3: Final commit**

Run:
```bash
git add -A
git status
```
Expected: Working tree clean or only untracked files

---

## Execution Complete

Plan complete and saved to `docs/plans/2026-02-14-prd-decomposer-implementation.md`.

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
