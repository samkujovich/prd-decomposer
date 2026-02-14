# PRD Decomposer

An MCP server that analyzes Product Requirements Documents (PRDs) and decomposes them into Jira-ready epics and stories.

Built with [arcade-mcp](https://github.com/ArcadeAI/arcade-mcp).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                               │
│                   (OpenAI Agents SDK)                       │
│                                                             │
│  1. Receives PRD from user (file path or pasted text)       │
│  2. Calls read_file → gets PRD content                      │
│  3. Calls analyze_prd → surfaces ambiguities                │
│  4. Calls decompose_to_tickets → returns Jira-ready output  │
└─────────────────────┬───────────────────────────────────────┘
                      │ stdio
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server                               │
│                 (prd_decomposer)                            │
│                                                             │
│  ┌───────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ read_file │  │ analyze_prd │  │ decompose_to_tickets │  │
│  │           │  │  (GPT-4o)   │  │      (GPT-4o)        │  │
│  └───────────┘  └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Tools

### `read_file`

Reads a file from the filesystem.

**Input:** File path (relative or absolute)
**Output:** File contents as string

### `analyze_prd`

Analyzes raw PRD text and extracts structured requirements.

**Input:** PRD markdown text
**Output:** Structured requirements with:
- Unique IDs (REQ-001, REQ-002, etc.)
- Acceptance criteria
- Dependencies between requirements
- Ambiguity flags (missing criteria, vague quantifiers like "fast", "scalable", "user-friendly")
- Priority levels (high/medium/low)

### `decompose_to_tickets`

Converts structured requirements into Jira-compatible tickets.

**Input:** Structured requirements from `analyze_prd` (or uses cached result if not provided)
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
git clone https://github.com/samkujovich/prd-decomposer.git
cd prd-decomposer

# Install dependencies
uv sync --all-extras

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Interactive Agent

```bash
uv run python agent/agent.py
```

Then:
```
You: analyze samples/sample_prd_01_rate_limiting.md
# ... shows requirements and ambiguity flags ...

You: jira tickets
# ... generates epics and stories ...
```

### Batch Processing (All 10 Sample PRDs)

```bash
uv run python scripts/run_all_prds.py
```

Processes all sample PRDs and saves results to `outputs/`:
- `*_requirements.json` - extracted requirements
- `*_tickets.json` - Jira epics/stories
- `summary.json` - aggregated metrics

### Run the MCP Server Standalone

```bash
# stdio mode (default, used by the agent)
uv run python src/prd_decomposer/server.py

# HTTP mode (for remote/network access - advanced use)
# uv run python src/prd_decomposer/server.py http
```

> **Note**: The agent uses stdio transport by default. HTTP mode is supported by arcade-mcp for remote access scenarios but is not covered by the test suite.

### Run Tests

```bash
# Unit tests (27 tests, 94% coverage)
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=prd_decomposer --cov-report=term-missing
```

## Sample PRDs

10 diverse PRDs for testing different domains and ambiguity patterns:

| # | PRD | Ambiguity Pattern |
|---|-----|-------------------|
| 1 | API Rate Limiting | Vague: "scalable", "user-friendly" |
| 2 | User Onboarding | Clear requirements |
| 3 | E-commerce Checkout | Vague: "seamless", "quick" |
| 4 | Notification System | Clear requirements |
| 5 | Analytics Dashboard | Vague: "acceptable" metrics |
| 6 | File Upload Service | Clear technical specs |
| 7 | Search Feature | Vague: "relevant", "fast" |
| 8 | Subscription Billing | Clear business rules |
| 9 | Webhook System | Clear technical spec |
| 10 | Mobile Push | Vague: "significant", "good" |

## Project Structure

```
prd-decomposer/
├── src/prd_decomposer/
│   ├── __init__.py       # Public exports
│   ├── server.py         # MCP server + tool definitions
│   ├── models.py         # Pydantic models
│   └── prompts.py        # LLM prompt templates
├── agent/
│   └── agent.py          # OpenAI Agents SDK consumer
├── scripts/
│   └── run_all_prds.py   # Batch processing script
├── tests/
│   ├── test_tools.py     # Model tests
│   ├── test_server.py    # Server/tool tests (mocked LLM)
│   └── test_prompts.py   # Prompt template tests
├── samples/
│   └── sample_prd_*.md   # 10 sample PRDs
├── outputs/              # Generated JSON outputs (gitignored)
├── CLAUDE.md             # Project coding standards
└── AI_USAGE.md           # AI attribution
```

## License

MIT
