# Pre-Submission Polish Plan

**Date:** 2025-02-14
**Status:** Ready to execute
**Estimated Time:** ~1 hour total

## Context

Principal Engineer / CTO / CPO review identified 4 polish items before interview submission. All are quick fixes that elevate the project from "very good" to "production-grade."

---

## Task 1: Initialize Logging (5 minutes)

**Problem:** `log.py` provides `setup_logging()` but it's never called. Server uses Python's default logger instead of structured JSON logs.

**File:** `src/prd_decomposer/server.py`

**Fix:**
```python
# Near the top of server.py, after imports
from prd_decomposer.log import setup_logging

# After get_settings() is defined (around line 50), add:
# Initialize structured logging
setup_logging(get_settings())
```

**Verify:**
```bash
uv run python -c "from prd_decomposer.server import app; print('Logging initialized')"
```

---

## Task 2: Add health_check Test (20 minutes)

**Problem:** `health_check` tool exists but has no unit test coverage.

**File:** `tests/test_server.py`

**Add test class:**
```python
class TestHealthCheck:
    """Tests for health_check tool."""

    @pytest.mark.asyncio
    async def test_health_check_healthy_state(self, mock_client_factory, permissive_rate_limiter):
        """Health check returns healthy when all systems nominal."""
        mock_client = mock_client_factory({"status": "ok"})

        # Reset circuit breaker to closed state
        cb = get_circuit_breaker()
        cb._state = "closed"
        cb._failure_count = 0

        result = await health_check(
            client=mock_client,
            settings=Settings(),
            rate_limiter=permissive_rate_limiter,
            circuit_breaker=cb,
        )

        assert result["status"] == "healthy"
        assert result["circuit_breaker"]["state"] == "closed"
        assert "rate_limiter" in result

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_circuit_open(self, mock_client_factory, permissive_rate_limiter):
        """Health check returns degraded when circuit breaker is open."""
        mock_client = mock_client_factory({"status": "ok"})

        cb = get_circuit_breaker()
        cb._state = "open"

        result = await health_check(
            client=mock_client,
            settings=Settings(),
            rate_limiter=permissive_rate_limiter,
            circuit_breaker=cb,
        )

        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_half_open(self, mock_client_factory, permissive_rate_limiter):
        """Health check returns degraded when circuit breaker is half-open."""
        mock_client = mock_client_factory({"status": "ok"})

        cb = get_circuit_breaker()
        cb._state = "half_open"

        result = await health_check(
            client=mock_client,
            settings=Settings(),
            rate_limiter=permissive_rate_limiter,
            circuit_breaker=cb,
        )

        assert result["status"] == "degraded"
```

**Verify:**
```bash
uv run pytest tests/test_server.py::TestHealthCheck -v
```

---

## Task 3: Validate Sizing Rubric Early (15 minutes)

**Problem:** If malformed rubric JSON is passed to `decompose_to_tickets`, it goes to LLM without validation.

**File:** `src/prd_decomposer/server.py`

**Find:** `_decompose_to_tickets_impl` function

**Add validation after JSON parsing:**
```python
# After parsing sizing_rubric JSON (around line where rubric is used)
if sizing_rubric:
    if isinstance(sizing_rubric, str):
        try:
            sizing_rubric = json.loads(sizing_rubric)
        except json.JSONDecodeError as e:
            raise ValueError(f"sizing_rubric must be valid JSON: {e}")

    # Validate against SizingRubric model
    try:
        rubric = SizingRubric(**sizing_rubric)
    except ValidationError as e:
        raise ValueError(f"Invalid sizing_rubric structure: {e}")
else:
    rubric = SizingRubric()  # Use defaults
```

**Add test:**
```python
@pytest.mark.asyncio
async def test_decompose_rejects_invalid_sizing_rubric(self, mock_client_factory):
    """decompose_to_tickets validates sizing_rubric structure."""
    mock_client = mock_client_factory({})

    with pytest.raises(ValueError, match="Invalid sizing_rubric"):
        await decompose_to_tickets(
            requirements='{"requirements": [], "summary": "test", "source_hash": "abc"}',
            sizing_rubric='{"small": "not an object"}',  # Invalid structure
            client=mock_client,
        )
```

**Verify:**
```bash
uv run pytest tests/test_server.py -k "sizing_rubric" -v
```

---

## Task 4: Add Integration Test (Optional, 30 minutes)

**Problem:** All tests mock LLM responses. No validation against real API.

**File:** `tests/integration/test_real_api.py` (new file)

```python
"""Integration tests requiring OPENAI_API_KEY.

Run with: OPENAI_API_KEY=sk-... uv run pytest tests/integration/ -v
"""
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Integration tests require OPENAI_API_KEY"
)


@pytest.mark.asyncio
async def test_analyze_prd_real_api():
    """Analyze a simple PRD with real OpenAI API."""
    from prd_decomposer.server import analyze_prd

    simple_prd = """
    # Feature: User Login

    ## Requirements
    - Users can log in with email and password
    - Failed login shows error message
    - Successful login redirects to dashboard
    """

    result = await analyze_prd(prd_text=simple_prd)

    assert "requirements" in result
    assert len(result["requirements"]) > 0
    assert "summary" in result
    assert "source_hash" in result


@pytest.mark.asyncio
async def test_decompose_to_tickets_real_api():
    """Decompose requirements with real OpenAI API."""
    from prd_decomposer.server import analyze_prd, decompose_to_tickets
    import json

    simple_prd = """
    # Feature: Password Reset

    ## Requirements
    - User can request password reset via email
    - Reset link expires after 24 hours
    - User must set new password meeting complexity requirements
    """

    requirements = await analyze_prd(prd_text=simple_prd)
    tickets = await decompose_to_tickets(requirements=json.dumps(requirements))

    assert "epics" in tickets
    assert len(tickets["epics"]) > 0
    assert "metadata" in tickets
```

**Create directory:**
```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

**Verify:**
```bash
# Without API key (should skip):
uv run pytest tests/integration/ -v

# With API key (should run):
OPENAI_API_KEY=sk-... uv run pytest tests/integration/ -v
```

---

## Verification Checklist

After completing all tasks:

```bash
# Run full test suite
uv run pytest tests/ -v

# Check linting
uv run ruff check src/ tests/

# Check types
uv run mypy src/

# Verify server starts
uv run python -c "from prd_decomposer.server import app; print('OK')"

# Run evals (optional, requires API key)
uv run arcade evals evals/eval_prd_tools.py
```

**Expected results:**
- 233+ tests passing (230 existing + 3 new health_check + sizing_rubric)
- Ruff clean
- Mypy clean
- Server imports successfully

---

## Commit Message

```
fix: pre-submission polish (logging, health_check tests, rubric validation)

- Initialize structured logging at server startup
- Add unit tests for health_check tool (healthy, degraded states)
- Validate sizing_rubric against SizingRubric model before LLM call
- Add optional integration test suite (requires OPENAI_API_KEY)

Addresses Principal Engineer review feedback.
```

---

## Post-Submission Notes

These items were explicitly out of scope for 6-hour limit but worth noting for follow-up:

1. **Refactor server.py** - Split into resilience.py, export.py, tools.py when adding more features
2. **LLM provider abstraction** - Define Protocol for swappable OpenAI/Anthropic/local models
3. **Output quality evals** - Arcade evals for requirement extraction accuracy, not just tool selection
4. **Request deduplication** - Cache StructuredRequirements by source_hash
5. **Agent-executable tickets** - Tickets today are human-readable but not agent-optimized. Add explicit file paths, machine-verifiable completion criteria, dependency graphs, context pointers, and execution hints so agents can execute tickets directly (added to README Future Iterations)
