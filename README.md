# PRD Decomposer

An MCP server that analyzes Product Requirements Documents (PRDs) and decomposes them into Jira-ready epics and stories.

Built with [arcade-mcp](https://github.com/ArcadeAI/arcade-mcp).

## The Problem

Most engineering teams share the same broken workflow: a PM writes a PRD in Google Docs, an EM or tech lead manually translates it into Jira epics and stories, requirements change but tickets don't, and inevitably things get lost in the mix. Someone ends up spending hours playing human ETL—keeping docs, tickets, and stakeholder expectations in sync.

## The Solution

An MCP server that gives AI agents the ability to analyze a PRD, extract structured requirements, identify gaps and ambiguities, and decompose the whole thing into ready-to-create epics and stories—structured in a format that can feed directly into Jira. The agent handles the translation layer so humans don't have to.

## Current Scope / Non-Goals

**What this project does:**
- Extracts structured requirements from PRD text with ambiguity detection
- Generates Jira-compatible epics/stories with sizing, labels, and acceptance criteria
- Exports to CSV, Jira REST API format, and YAML for integration
- Provides AI agent context (`agent_context`) for downstream coding assistants

**What this project does NOT do (intentional non-goals for v1):**
- **No direct Jira integration** — Output is Jira-compatible JSON; actual ticket creation is left to [Arcade's Jira toolkit](https://docs.arcade.ai/tools/atlassian) or manual import
- **No document source integrations** — PRDs must be local files or pasted text; Google Docs/Notion/Confluence connectors are future work
- **No bidirectional sync** — This is a one-way PRD→tickets flow; tracking changes over time is out of scope
- **No multi-user/auth** — Designed for single-user CLI usage; no API authentication layer

See [Future Iterations](#future-iterations) for planned enhancements.

## Architecture

![Architecture Diagram](docs/diagrams/architecture.svg)

## Tools

### `read_file`

Reads a file from the filesystem (restricted to working directory for security).

**Input:** File path (relative to working directory)
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

**Input:** Structured requirements from `analyze_prd` (required), optional custom sizing rubric
**Output:** Epics and stories with:
- Clear titles and descriptions
- Acceptance criteria
- T-shirt sizing (S/M/L) with configurable rubric
- Labels
- Traceability back to requirements

### `export_tickets`

Exports ticket collections to different formats for integration with external tools.

**Input:** Ticket collection JSON, format (`csv`, `jira`, or `yaml`)
**Output:** Formatted string ready for import:
- **CSV**: Flat format with one row per story (spreadsheet import)
- **Jira**: REST API bulk create payload (ready for POST to `/rest/api/3/issue/bulk`)
- **YAML**: Structured format for GitOps workflows

### `health_check`

Checks service health and internal component status.

**Output:** Status information including:
- Service status (healthy/degraded/unhealthy)
- Circuit breaker state
- Rate limiter status
- Configuration summary

**Note:** Does not probe the OpenAI API directly; reflects circuit breaker state from recent LLM calls.

## Features

### Security
- Path traversal protection restricts file access to working directory
- Symlink resolution prevents bypass attacks
- Input length limits prevent resource exhaustion (`PRD_MAX_PRD_LENGTH`)
- Rate limiting prevents API quota exhaustion (`PRD_RATE_LIMIT_CALLS`, `PRD_RATE_LIMIT_WINDOW`)
- XML delimiters in prompts help LLMs distinguish instructions from user content

**Note on Prompt Injection:** Like all LLM-based tools, this system is susceptible to prompt injection attacks where malicious content in PRDs could attempt to manipulate LLM behavior. Mitigations include input length limits and structural delimiters, but these are not foolproof. Do not use with untrusted input in security-sensitive contexts.

### Reliability
- **Circuit breaker pattern** prevents cascading failures during upstream outages
  - Opens after consecutive failures, blocks calls during recovery
  - Half-open probes test recovery with single attempts (no retries)
  - 4xx client errors don't trip the breaker (only transient upstream failures)
- Automatic retry with exponential backoff for rate limits and connection errors
- Graceful shutdown handling (SIGTERM/SIGINT)
- Graceful error handling with clear error messages

### Observability
- Token usage tracking in metadata (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- Prompt versioning for traceability (`PROMPT_VERSION`)
- Generation timestamps and model info in outputs

### Cost Efficiency
Using GPT-4o (as of Feb 2026):
- Typical PRD analysis: ~$0.02-0.05
- Ticket decomposition: ~$0.03-0.08
- Full workflow: ~$0.05-0.15 per PRD
- Process 50 PRDs/month for under $10

### Configuration
Environment variables with `PRD_` prefix (via pydantic-settings):
- `PRD_OPENAI_MODEL` - Model to use (default: `gpt-4o`)
- `PRD_ANALYZE_TEMPERATURE` - Temperature for analysis (default: `0.2`)
- `PRD_DECOMPOSE_TEMPERATURE` - Temperature for decomposition (default: `0.3`)
- `PRD_MAX_RETRIES` - Max retry attempts, 1-10 (default: `3`)
- `PRD_INITIAL_RETRY_DELAY` - Initial retry delay in seconds, 0-60 (default: `1.0`)
- `PRD_LLM_TIMEOUT` - Timeout for LLM API calls in seconds, 1-300 (default: `60`)
- `PRD_MAX_PRD_LENGTH` - Maximum PRD text length in characters, 1000-500000 (default: `100000`)
- `PRD_RATE_LIMIT_CALLS` - Maximum LLM calls per window, 1-1000 (default: `60`)
- `PRD_RATE_LIMIT_WINDOW` - Rate limit window in seconds, 1-3600 (default: `60`)
- `PRD_CIRCUIT_BREAKER_FAILURE_THRESHOLD` - Failures before circuit opens, 1-20 (default: `5`)
- `PRD_CIRCUIT_BREAKER_RESET_TIMEOUT` - Seconds before half-open probe, 1-300 (default: `60`)

### AI-Executable Tickets

Stories include optional `agent_context` for AI coding assistants:

- **goal**: Why this work matters (the problem being solved)
- **exploration_paths**: Keywords to search in codebase
- **exploration_hints**: Specific files/modules to start with
- **known_patterns**: Libraries and conventions to follow
- **verification_tests**: Tests that should pass when done
- **self_check**: Questions to verify before completion

Use `prompt N` in the agent to get a copy-pasteable prompt for any story.

## Key Decisions

| Decision | Choice | Rationale | Alternative Considered |
|----------|--------|-----------|------------------------|
| **Language** | Python | Arcade's SDK, CLI, and eval framework are Python. Fighting the toolchain wastes time. | TypeScript (team uses it day-to-day, but Arcade tooling is Python-first) |
| **Data Validation** | Pydantic models | Self-documenting, validates inputs/outputs, plays well with Arcade's type annotations | Plain dicts (faster to write but harder to validate) |
| **LLM Provider** | OpenAI (gpt-4o) | Agent layer uses OpenAI Agents SDK; single provider keeps dependency surface clean | Anthropic Claude (would also work) |
| **Agent Framework** | OpenAI Agents SDK | Production experience from building multi-agent systems; less framework wrangling | LangGraph (Arcade has examples, but "any framework" was allowed) |
| **MCP Transport** | stdio | Arcade's default for local dev, simplest setup, what eval framework expects | HTTP (more production-realistic but adds complexity) |
| **External APIs** | None (no Jira/Google OAuth) | Tools are the intelligence layer; output is Jira-compatible schema that composes with Arcade's Jira toolkit | Real Jira/Google OAuth (risky time sink) |
| **Prompt Management** | Separate `prompts.py` | Keeps prompts testable, iterable, reviewable separately from tool logic | Inline prompts in tool functions |
| **Configuration** | pydantic-settings with `PRD_` prefix | Runtime config without code changes; environment variables are 12-factor compliant | Hardcoded values or config files |
| **Testability** | Dependency injection for LLM client | Tests can inject mock clients directly; no patching required | Global client with `@patch` decorators |
| **Prompt Quality** | Few-shot examples in prompts | Improves LLM output consistency for ambiguity detection and story decomposition | Zero-shot prompts (less consistent) |
| **MCP Tool Parameters** | JSON string for complex types | `dict` parameters weren't passed correctly by OpenAI agent; strings are more portable | `dict` type annotation (caused None values) |
| **Retry Config** | Bounded validation (1-10 retries, 0-60s delay) | Prevents invalid env var values from causing runtime failures | Unbounded (accepts any value) |
| **Path Security** | Symlink resolution + allowlist | Prevents path traversal attacks even through symlinked directories | Basic path checking (bypassable) |
| **Circuit Breaker** | Fail-fast after consecutive failures | Prevents cascading failures during upstream outages; gives APIs time to recover | No circuit breaker (keeps hammering failed APIs) |
| **Half-Open Probes** | Single attempt, no retries | Minimizes load during recovery; faster failure detection | Normal retry budget (adds unnecessary load) |
| **4xx Error Handling** | Don't trip circuit breaker | Client errors indicate bad requests, not upstream failures; avoids false positives | Count all errors (opens circuit unnecessarily) |
| **Export Formats** | CSV, Jira REST, YAML | Covers common integration needs: spreadsheets, direct API, GitOps | Single format (limits usefulness) |
| **Input Validation** | Deep nested validation | Catches malformed payloads early with clear error paths | Shallow validation (cryptic errors in exporters) |
| **YAML Escaping** | Quote all user content | Prevents YAML injection from special chars in titles/descriptions | Raw interpolation (broken YAML output) |

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
# Unit tests (310 tests, no API key required)
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=prd_decomposer --cov-report=term-missing
```

### Run Evals

Two evaluation suites validate quality (both require `OPENAI_API_KEY`):

```bash
# Tool Selection Evals - does the LLM pick the right tool?
uv run arcade evals evals/eval_prd_tools.py

# Output Quality Evals - are tool outputs correct?
uv run pytest evals/eval_output_quality.py -v
```

Output quality evals validate:
- Vague quantifiers ("fast", "scalable") flagged as ambiguities
- Clear PRDs with explicit criteria have no critical ambiguities
- Stories have valid `requirement_ids` linking to source requirements
- Epic count is reasonable (1-4 for single-feature PRD)
- All stories have valid T-shirt sizes (S/M/L)

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
│   ├── __init__.py        # Public exports
│   ├── server.py          # MCP server + tool definitions
│   ├── models.py          # Pydantic models (includes AgentContext)
│   ├── prompts.py         # LLM prompt templates
│   ├── formatters.py      # Prompt rendering for AI agents
│   ├── config.py          # Settings via environment variables
│   ├── log.py             # Structured JSON logging
│   ├── circuit_breaker.py # Circuit breaker + rate limiter
│   └── export.py          # CSV/Jira/YAML export functions
├── agent/
│   ├── agent.py          # OpenAI Agents SDK consumer
│   ├── session_state.py  # Agent session state management
│   └── formatters.py     # Re-exports from prd_decomposer.formatters
├── scripts/
│   └── run_all_prds.py   # Batch processing script
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_models.py         # Pydantic model tests
│   ├── test_server.py         # Server/tool tests (mocked LLM)
│   ├── test_circuit_breaker.py # Circuit breaker tests
│   ├── test_export.py         # Export format tests
│   ├── test_prompts.py        # Prompt template tests
│   ├── test_config.py         # Configuration tests
│   ├── test_logging.py        # Logging tests
│   ├── test_init.py           # Package init tests
│   └── integration/           # Real API tests (require OPENAI_API_KEY)
├── samples/
│   └── sample_prd_*.md   # 10 sample PRDs
├── outputs/              # Generated JSON outputs (gitignored)
├── CLAUDE.md             # Project coding standards
└── AI_USAGE.md           # AI attribution
```

## Future Iterations

### Document Source Integrations
Currently PRDs must be local markdown files. Future versions could pull directly from:
- **Google Docs** - Where most PRDs live in practice
- **Notion** - Popular for product teams
- **Confluence** - Enterprise standard

### Jira Integration
Output is currently Jira-compatible JSON. Next step: actual ticket creation via Atlassian API, composing with [Arcade's Jira toolkit](https://docs.arcade.ai/tools/atlassian).

### Auto-Commenting on Ambiguities
When the tool flags something as ambiguous, automatically leave a comment on the source document (Google Docs comment, Notion comment, Confluence inline comment) asking for clarification—closing the feedback loop without human copy-paste.

### Expanded Analysis Categories
Ambiguity detection is just the start. The same pattern could flag:
- **Incompleteness** - Missing sections (error handling, edge cases, rollback plan)
- **Edge case coverage** - Scenarios the PRD doesn't address
- **Design questions** - Technical decisions that need engineering input before implementation
- **UX concerns** - Flows that might frustrate users (too many steps, unclear affordances)
- **Security considerations** - Auth, data handling, PII that needs review

### Continuous Sync & Bidirectional Updates
Run agents continuously to keep PRDs and Jira in sync:
- **PRD → Jira**: When a PRD changes, automatically detect deltas and update/create tickets
- **Jira → PRD**: When ticket status changes (in progress, blocked, complete), reflect it back in the PRD
- **Real-time status dashboard**: Product and non-technical stakeholders see live implementation status tied to the high-level requirements doc—no more "what's the status of X?" meetings

## License

MIT
