# CLAUDE.md

## Project Overview

PRD Decomposer: An MCP server that analyzes Product Requirements Documents and decomposes them into Jira-ready epics and stories. Built with arcade-mcp, consumed by an OpenAI Agents SDK agent.

**Tech stack:** Python 3.11+, arcade-mcp, OpenAI SDK, Pydantic v2, pytest, arcade_evals
**Package manager:** uv (NOT pip)
**Build system:** hatchling

## Architecture

Two MCP tools (`analyze_prd`, `decompose_to_tickets`) each make independent GPT-4o calls.
Agent (OpenAI Agents SDK) consumes tools via stdio. Pydantic models enforce all schemas.

```
Agent (OpenAI Agents SDK) → stdio → MCP Server (arcade-mcp)
                                      ├── analyze_prd → GPT-4o → StructuredRequirements
                                      └── decompose_to_tickets → GPT-4o → TicketCollection
```

## Development Environment

- Always use `uv` for package management, never `pip`.
- Virtual environment: `.venv/` (already in .gitignore).
- Python 3.11+ required.

### Key Commands

```bash
uv sync --all-extras          # Install all dependencies including dev
uv run pytest tests/ -v       # Run unit tests
uv run python src/prd_decomposer/server.py          # Run MCP server (stdio)
uv run python src/prd_decomposer/server.py http      # Run MCP server (HTTP)
uv run python agent/agent.py  # Run the agent consumer
uv run arcade evals evals/eval_prd_tools.py          # Run arcade evals
```

### Environment Variables

- `OPENAI_API_KEY` — required for LLM calls in server tools.
- Never commit `.env` files or API keys.

## Python Coding Standards

### Style

- Follow PEP 8. Use 4-space indentation, no tabs.
- Maximum line length: 99 characters.
- Use double quotes for strings.
- Imports: standard library → third-party → local, separated by blank lines.
- No wildcard imports (`from module import *`).

### Type Annotations

- All function signatures must have full type annotations (parameters and return types).
- Use `Annotated[type, "description"]` for MCP tool parameters — arcade-mcp uses these as tool parameter descriptions.
- Use `Literal` for constrained string fields (e.g., `Literal["S", "M", "L"]`).
- Prefer `list[str]` over `List[str]` (modern Python syntax).

### Pydantic Models

- All data models live in `src/prd_decomposer/models.py`.
- Use `BaseModel` with `Field(...)` for required fields and `Field(default_factory=...)` for optional collections.
- Every `Field` must include a `description` parameter.
- Models are the single source of truth for data validation — never manually validate what Pydantic can enforce.
- Use `model_dump()` / `model_dump_json()` for serialization (not `.dict()` / `.json()` which are deprecated in Pydantic v2).

### Error Handling

- Let Pydantic `ValidationError` propagate — don't catch and re-raise with less information.
- In MCP tools: validate inputs with Pydantic models before passing to LLM calls.
- In LLM response parsing: use `json.loads()` then Pydantic validation. If the LLM returns invalid JSON, let the error surface clearly.

## MCP Server Patterns

### Tool Design

- Each tool is a standalone unit — independently callable and testable.
- Tools do NOT chain to each other. The agent handles orchestration.
- Every tool must have a docstring that clearly describes what it does — arcade-mcp uses this as the tool description for LLM consumption.
- Use `@app.tool` decorator. No custom tool names — the function name IS the tool name.

### LLM Integration

- Use `response_format={"type": "json_object"}` for all LLM calls to guarantee valid JSON.
- Use low temperature (0.2–0.3) for consistent, deterministic output.
- Always validate LLM responses against Pydantic models before returning.
- Return `model.model_dump()` (dict), not the Pydantic model instance — MCP serialization expects plain dicts.

### Prompts

- All prompt templates live in `src/prd_decomposer/prompts.py` as module-level constants.
- Use `.format()` string interpolation (not f-strings) so templates remain static constants.
- Double-brace `{{` / `}}` for literal braces in JSON schema examples within prompts.

## Testing

### Unit Tests (pytest)

- Tests live in `tests/`. Test files mirror source structure with `test_` prefix.
- Test both valid and invalid inputs for every Pydantic model.
- Test JSON round-trip serialization (`model_dump_json()` → `model_validate_json()`).
- Use plain `pytest.raises(ValidationError)` for rejection tests — no need for custom error messages.
- Tests must run without network access or API keys. Mock external calls.
- Run with: `uv run pytest tests/ -v`

### Arcade Evals

- Evals live in `evals/`. These validate that LLMs select the correct tool for a given user intent.
- Evals DO require API keys and network access — they are not part of the unit test suite.
- Each eval case needs: `user_message`, `expected_tool_calls`, and `critics`.
- Run with: `uv run arcade evals evals/eval_prd_tools.py`

### Quality Gates (must pass before committing)

1. `uv run pytest tests/ -v` — all unit tests pass.
2. No import errors: `uv run python -c "from prd_decomposer.server import app"` succeeds.
3. Pydantic models validate: no `Field` without `description`, no deprecated v1 methods.
4. Type annotations present on all public function signatures.

### Test-Driven Development

- Write failing tests FIRST, then implement to make them pass.
- Each model and tool should have at least: one happy-path test, one validation/rejection test.

## Project Structure

```
prd-decomposer/
├── src/prd_decomposer/       # MCP server package
│   ├── __init__.py            # Public exports
│   ├── server.py              # MCP app + tool definitions
│   ├── models.py              # All Pydantic models
│   └── prompts.py             # LLM prompt templates
├── agent/                     # OpenAI Agents SDK consumer
│   └── agent.py
├── evals/                     # Arcade eval suite (requires API keys)
│   └── eval_prd_tools.py
├── tests/                     # Unit tests (no network required)
│   └── test_tools.py
├── samples/                   # Example PRDs for testing
│   └── sample_prd.md
├── docs/plans/                # Design docs and implementation plans
├── pyproject.toml             # Project config (uv/hatchling)
└── AI_USAGE.md                # AI tool attribution
```

### Conventions

- Source code lives under `src/prd_decomposer/` — this is a proper Python package with `__init__.py` exports.
- Do NOT add new top-level directories without strong justification.
- Keep models in `models.py`, prompts in `prompts.py`, tools in `server.py` — resist splitting into many small files prematurely.
- Design docs go in `docs/plans/` with `YYYY-MM-DD-<topic>-<type>.md` naming.

### Files That Must Not Be Committed

- `.env` or any file containing API keys
- `.venv/` directory
- `__pycache__/` directories
- `*.pyc` files
