# PRD Decomposer Design

**Date:** 2026-02-14
**Status:** Approved

## Overview

An MCP server that solves the manual process of translating PRDs into Jira epics and stories. Built with `arcade-mcp`, consumed by an OpenAI Agents SDK agent.

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
│                 (arcade-mcp / prd_decomposer)                │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────┐            │
│  │   analyze_prd   │    │  decompose_to_tickets │            │
│  │                 │    │                       │            │
│  │  PRD text       │    │  StructuredRequirements│           │
│  │       ↓         │    │          ↓             │           │
│  │  GPT-4o call    │    │     GPT-4o call        │           │
│  │       ↓         │    │          ↓             │           │
│  │  Structured     │    │   TicketCollection     │           │
│  │  Requirements   │    │   (Epics + Stories)    │           │
│  └─────────────────┘    └──────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Two Independent LLM Calls
Each tool makes its own LLM call. Tools are independently usable and testable. The agent handles chaining.

### 2. T-Shirt Sizing: S, M, L
3-point scale. Sizing rubric in prompt:
- S: < 1 day, single component
- M: 1-3 days, may touch multiple components
- L: 3-5 days, significant complexity

### 3. Ambiguity Detection
Flag requirements with:
- Missing acceptance criteria
- Vague quantifiers ("fast", "scalable", "user-friendly" without metrics)

### 4. No External API Dependencies
Tools are the intelligence layer. Output is Jira-schema-compatible but doesn't call Jira APIs. Designed to compose with Arcade's existing Jira toolkit.

## Data Models

### Input/Intermediate

```python
class Requirement:
    id: str                      # REQ-001, REQ-002, etc.
    title: str
    description: str
    acceptance_criteria: list[str]
    dependencies: list[str]      # IDs of other requirements
    ambiguity_flags: list[str]   # Reasons flagged as ambiguous
    priority: Literal["high", "medium", "low"]

class StructuredRequirements:
    requirements: list[Requirement]
    summary: str
    source_hash: str             # For traceability
```

### Output (Jira-compatible)

```python
class Story:
    title: str
    description: str
    acceptance_criteria: list[str]
    size: Literal["S", "M", "L"]
    labels: list[str]
    requirement_ids: list[str]   # Traceability

class Epic:
    title: str
    description: str
    stories: list[Story]
    labels: list[str]

class TicketCollection:
    epics: list[Epic]
    metadata: dict               # Timestamp, model version, etc.
```

## Prompts

### analyze_prd
- Role: Senior technical product manager
- Task: Extract requirements with IDs, acceptance criteria, dependencies
- Flag ambiguities per detection rules
- Output: JSON matching StructuredRequirements schema

### decompose_to_tickets
- Role: Senior engineering manager
- Task: Group requirements into epics, break into stories
- Apply sizing rubric, generate labels
- Preserve traceability via requirement_ids
- Output: JSON matching TicketCollection schema

## Testing Strategy

### Unit Tests (pytest)
- Validate Pydantic models accept valid data
- Validate models reject invalid data (e.g., size="XL")
- Validate JSON round-trip serialization

### Evals (arcade_evals)
- Validate LLM selects `analyze_prd` for analysis requests
- Validate LLM selects `decompose_to_tickets` for decomposition requests
- Validate correct parameter passing

## Sample PRD

Developer tool feature: API Rate Limiting System
- Includes clear requirements (limits, headers, error handling)
- Includes intentional ambiguity ("fast and scalable")
- Spans multiple epics (user limits, DX, backend)

## Project Structure

```
prd-decomposer/
├── src/
│   └── prd_decomposer/
│       ├── __init__.py
│       ├── server.py          # MCP server + tool definitions
│       ├── models.py          # Pydantic models
│       ├── prompts.py         # LLM prompt templates
│       └── .env.example
├── agent/
│   └── agent.py               # OpenAI Agents SDK consumer
├── evals/
│   └── eval_prd_tools.py      # Arcade eval suite
├── tests/
│   └── test_tools.py          # Unit tests
├── samples/
│   └── sample_prd.md          # Example PRD
├── docs/
│   └── plans/
│       └── 2026-02-14-prd-decomposer-design.md
├── AI_USAGE.md
├── pyproject.toml
└── README.md
```

## Constraints

- 6-hour time cap
- Must use `arcade new` to scaffold
- Must include both tests and evals
- Document all AI usage
- Public GitHub repo with clean README
