# Design: Configuration, Dependency Injection, and Few-Shot Prompts

**Date:** 2025-02-14
**Status:** Approved
**Scope:** Address Principal Engineer feedback on testability and configuration

## Problem Statement

The current implementation has three issues identified in code review:

1. **Global state** — `_client = None` pattern limits testability (requires patching)
2. **Hard-coded values** — Model name (`gpt-4o`) and temperatures appear inline
3. **Prompts lack examples** — No few-shot examples for LLM output consistency

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Configuration approach | Pydantic Settings | Type-safe, auto-loads from env vars, validates on startup |
| Dependency injection | Optional function params | Backward compatible, simple, no architectural change |
| Few-shot examples | Input/output pairs | Concrete examples improve LLM output consistency |
| LLM abstraction | Deferred | Not addressing vendor lock-in in this iteration |

## Design

### 1. Configuration (`config.py`)

New file with Pydantic Settings:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRD_")

    # LLM settings
    openai_model: str = "gpt-4o"
    analyze_temperature: float = 0.2
    decompose_temperature: float = 0.3

    # Retry settings
    max_retries: int = 3
    initial_retry_delay: float = 1.0

_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**Environment variable overrides:**
- `PRD_OPENAI_MODEL` — Change model (e.g., `gpt-4o-mini`)
- `PRD_ANALYZE_TEMPERATURE` — Adjust analysis creativity
- `PRD_DECOMPOSE_TEMPERATURE` — Adjust decomposition creativity
- `PRD_MAX_RETRIES` — Retry count for transient failures
- `PRD_INITIAL_RETRY_DELAY` — Base delay for exponential backoff

### 2. Dependency Injection

Modify functions to accept optional `client` and `settings` parameters:

```python
def _call_llm_with_retry(
    messages: list[dict],
    temperature: float,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> tuple[dict, dict]:
    client = client or get_client()
    settings = settings or get_settings()
    ...

@app.tool
def analyze_prd(
    prd_text: Annotated[str, "Raw PRD markdown text to analyze"],
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    client = client or get_client()
    settings = settings or get_settings()
    ...
```

**Benefits:**
- MCP tools remain backward compatible (no args required)
- Tests inject mocks directly: `analyze_prd("text", client=mock)`
- No more `patch("prd_decomposer.server.get_client", ...)`

### 3. Few-Shot Prompts

Add one input/output example to each prompt template:

**ANALYZE_PRD_PROMPT example section:**
```
## Example

**Input PRD:**
# Feature: Password Reset
Users must be able to reset their password via email.
The reset flow should be fast and user-friendly.

**Output:**
{
  "requirements": [{
    "id": "REQ-001",
    "title": "Email-based password reset",
    "description": "Users can request a password reset link...",
    "acceptance_criteria": ["Reset link sent within 30 seconds", ...],
    "dependencies": [],
    "ambiguity_flags": [
      "Vague quantifier: 'fast' - no specific latency target",
      "Vague quantifier: 'user-friendly' - no measurable criteria"
    ],
    "priority": "high"
  }],
  "summary": "Password reset feature via email"
}
```

**Key aspects:**
- Example demonstrates ambiguity detection (flags "fast", "user-friendly")
- Shows complete JSON structure
- One example per prompt (balances guidance vs token usage)

## File Changes

| File | Change |
|------|--------|
| `src/prd_decomposer/config.py` | NEW — Pydantic Settings class |
| `src/prd_decomposer/server.py` | MODIFY — Add client/settings params to functions |
| `src/prd_decomposer/prompts.py` | MODIFY — Add few-shot examples |
| `src/prd_decomposer/__init__.py` | MODIFY — Export Settings, get_settings |
| `pyproject.toml` | MODIFY — Add pydantic-settings dependency |
| `tests/test_server.py` | MODIFY — Simplify mocking (pass client directly) |
| `tests/test_config.py` | NEW — Test Settings loading and validation |

## Testing Strategy

1. **Config tests** — Verify env var loading, defaults, validation
2. **Update existing tests** — Pass mock client directly instead of patching
3. **Prompt tests** — Verify few-shot examples are formattable
4. **Integration** — Ensure MCP tools still work without explicit args

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    ...
    "pydantic-settings>=2.0.0",
]
```

## Out of Scope

- LLM provider abstraction (switching from OpenAI to Anthropic)
- Config file support (TOML/YAML) — env vars sufficient for now
- Multiple few-shot examples — one is enough to guide the model
