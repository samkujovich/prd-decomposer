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

### Run Tests

```bash
# Unit tests
uv run pytest tests/ -v

# Arcade evals (requires OPENAI_API_KEY)
uv run arcade evals evals/eval_prd_tools.py
```

## Project Structure

```
prd-decomposer/
├── src/prd_decomposer/
│   ├── server.py      # MCP server + tool definitions
│   ├── models.py      # Pydantic models
│   └── prompts.py     # LLM prompt templates
├── agent/
│   └── agent.py       # OpenAI Agents SDK consumer
├── evals/
│   └── eval_prd_tools.py  # Arcade eval suite
├── tests/
│   └── test_tools.py  # Unit tests
├── samples/
│   └── sample_prd.md  # Example PRD
└── AI_USAGE.md        # AI attribution
```

## License

MIT
