# Config, DI, and Few-Shot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Pydantic Settings for configuration, dependency injection for testability, and few-shot examples for LLM consistency.

**Architecture:** Incremental enhancement—add `config.py` with Settings, modify existing functions to accept optional `client`/`settings` params with defaults, enhance prompts with input/output examples.

**Tech Stack:** pydantic-settings, pytest, existing pydantic/openai stack

---

## Task 1: Add pydantic-settings Dependency

**Files:**
- Modify: `pyproject.toml:6-11`

**Step 1: Add dependency to pyproject.toml**

Edit `pyproject.toml` dependencies section:

```toml
dependencies = [
    "arcade-mcp",
    "openai>=1.0.0",
    "openai-agents>=0.0.17",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync --all-extras`
Expected: Dependencies install successfully

**Step 3: Verify import works**

Run: `uv run python -c "from pydantic_settings import BaseSettings; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pydantic-settings dependency"
```

---

## Task 2: Create Settings Class with Tests

**Files:**
- Create: `src/prd_decomposer/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test for Settings defaults**

Create `tests/test_config.py`:

```python
"""Tests for configuration settings."""

from prd_decomposer.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_has_default_model(self):
        """Verify Settings has default openai_model."""
        settings = Settings()
        assert settings.openai_model == "gpt-4o"

    def test_settings_has_default_temperatures(self):
        """Verify Settings has default temperatures."""
        settings = Settings()
        assert settings.analyze_temperature == 0.2
        assert settings.decompose_temperature == 0.3

    def test_settings_has_default_retry_config(self):
        """Verify Settings has default retry configuration."""
        settings = Settings()
        assert settings.max_retries == 3
        assert settings.initial_retry_delay == 1.0

    def test_settings_loads_from_env(self, monkeypatch):
        """Verify Settings loads from environment variables."""
        monkeypatch.setenv("PRD_OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("PRD_MAX_RETRIES", "5")

        settings = Settings()

        assert settings.openai_model == "gpt-4o-mini"
        assert settings.max_retries == 5


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self, monkeypatch):
        """Verify get_settings returns a Settings instance."""
        # Reset singleton for test isolation
        import prd_decomposer.config as config_module
        config_module._settings = None

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_caches_instance(self, monkeypatch):
        """Verify get_settings returns same instance on repeated calls."""
        import prd_decomposer.config as config_module
        config_module._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prd_decomposer.config'`

**Step 3: Write minimal implementation**

Create `src/prd_decomposer/config.py`:

```python
"""Configuration settings for PRD Decomposer."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="PRD_")

    # LLM settings
    openai_model: str = "gpt-4o"
    analyze_temperature: float = 0.2
    decompose_temperature: float = 0.3

    # Retry settings
    max_retries: int = 3
    initial_retry_delay: float = 1.0


# Singleton for convenience
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/prd_decomposer/config.py tests/test_config.py
git commit -m "feat: add Settings class with env var support"
```

---

## Task 3: Update __init__.py Exports

**Files:**
- Modify: `src/prd_decomposer/__init__.py`

**Step 1: Read current exports**

Run: `cat src/prd_decomposer/__init__.py`

**Step 2: Add config exports**

Update `src/prd_decomposer/__init__.py`:

```python
"""PRD Decomposer: MCP server for PRD analysis and ticket generation."""

from prd_decomposer.config import Settings, get_settings
from prd_decomposer.models import (
    Epic,
    Requirement,
    Story,
    StructuredRequirements,
    TicketCollection,
)

__all__ = [
    "Epic",
    "Requirement",
    "Settings",
    "Story",
    "StructuredRequirements",
    "TicketCollection",
    "get_settings",
]
```

**Step 3: Verify import works**

Run: `uv run python -c "from prd_decomposer import Settings, get_settings; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/prd_decomposer/__init__.py
git commit -m "feat: export Settings and get_settings from package"
```

---

## Task 4: Update _call_llm_with_retry for DI

**Files:**
- Modify: `src/prd_decomposer/server.py:89-159`
- Modify: `tests/test_server.py` (TestLLMRetry class)

**Step 1: Update _call_llm_with_retry signature**

Modify `src/prd_decomposer/server.py`. Update imports at top:

```python
from prd_decomposer.config import Settings, get_settings
```

Update `_call_llm_with_retry` function:

```python
def _call_llm_with_retry(
    messages: list[dict],
    temperature: float,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> tuple[dict, dict]:
    """Call LLM with exponential backoff retry.

    Returns:
        Tuple of (parsed_json_response, usage_metadata)

    Raises:
        LLMError: If all retries fail or response is invalid.
    """
    client = client or get_client()
    settings = settings or get_settings()

    last_error = None

    for attempt in range(settings.max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )

            # Extract content
            content = response.choices[0].message.content
            if not content:
                raise LLMError("LLM returned empty response")

            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(f"LLM returned invalid JSON: {e}")

            # Extract usage for cost tracking
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return data, usage

        except RateLimitError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

        except APIError as e:
            # Don't retry on 4xx errors (bad request, auth, etc.)
            status_code = getattr(e, "status_code", None)
            if status_code and 400 <= status_code < 500:
                raise LLMError(f"OpenAI API error: {e}")
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

    raise LLMError(f"LLM call failed after {settings.max_retries} retries: {last_error}")
```

**Step 2: Run existing tests to verify they still pass**

Run: `uv run pytest tests/test_server.py::TestLLMRetry -v`
Expected: All tests PASS (existing tests use patching, still works)

**Step 3: Commit**

```bash
git add src/prd_decomposer/server.py
git commit -m "refactor: add client/settings params to _call_llm_with_retry"
```

---

## Task 5: Update analyze_prd for DI

**Files:**
- Modify: `src/prd_decomposer/server.py:162-202`

**Step 1: Update analyze_prd signature and body**

```python
@app.tool
def analyze_prd(
    prd_text: Annotated[str, "Raw PRD markdown text to analyze"],
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    """Analyze a PRD and extract structured requirements.

    Extracts requirements with IDs, acceptance criteria, dependencies,
    and flags ambiguous requirements (missing criteria or vague quantifiers).

    Returns structured requirements with metadata including token usage.
    """
    client = client or get_client()
    settings = settings or get_settings()

    # Generate source hash for traceability
    source_hash = hashlib.sha256(prd_text.encode()).hexdigest()[:8]

    # Call LLM with retry
    try:
        data, usage = _call_llm_with_retry(
            messages=[{"role": "user", "content": ANALYZE_PRD_PROMPT.format(prd_text=prd_text)}],
            temperature=settings.analyze_temperature,
            client=client,
            settings=settings,
        )
    except LLMError as e:
        raise RuntimeError(f"Failed to analyze PRD: {e}")

    # Ensure source_hash is set
    data["source_hash"] = source_hash

    # Validate with Pydantic
    try:
        validated = StructuredRequirements(**data)
    except ValidationError as e:
        raise RuntimeError(f"LLM returned invalid structure: {e}")

    result = validated.model_dump()

    # Add metadata for observability
    result["_metadata"] = {
        "prompt_version": PROMPT_VERSION,
        "model": settings.openai_model,
        "usage": usage,
        "analyzed_at": datetime.now(UTC).isoformat(),
    }

    return result
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/test_server.py::TestAnalyzePrd -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/prd_decomposer/server.py
git commit -m "refactor: add client/settings params to analyze_prd"
```

---

## Task 6: Update decompose_to_tickets for DI

**Files:**
- Modify: `src/prd_decomposer/server.py:205-270`

**Step 1: Update decompose_to_tickets signature and body**

```python
@app.tool
def decompose_to_tickets(
    requirements: Annotated[dict, "Structured requirements from analyze_prd (required)"],
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    """Convert structured requirements into Jira-compatible epics and stories.

    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels. Output is ready for Jira import.

    Requires the requirements dict from analyze_prd to be passed explicitly.
    """
    client = client or get_client()
    settings = settings or get_settings()

    # Handle case where requirements might be passed as string
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in requirements: {e}")

    if not requirements:
        raise ValueError("Requirements cannot be empty. Run analyze_prd first.")

    # Strip internal metadata before validation
    requirements_clean = {k: v for k, v in requirements.items() if not k.startswith("_")}

    # Validate input
    try:
        validated_input = StructuredRequirements(**requirements_clean)
    except ValidationError as e:
        raise ValueError(f"Invalid requirements structure: {e}")

    # Call LLM with retry
    try:
        data, usage = _call_llm_with_retry(
            messages=[
                {
                    "role": "user",
                    "content": DECOMPOSE_TO_TICKETS_PROMPT.format(
                        requirements_json=validated_input.model_dump_json(indent=2)
                    ),
                }
            ],
            temperature=settings.decompose_temperature,
            client=client,
            settings=settings,
        )
    except LLMError as e:
        raise RuntimeError(f"Failed to decompose requirements: {e}")

    # Add metadata if not present
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["generated_at"] = datetime.now(UTC).isoformat()
    data["metadata"]["model"] = settings.openai_model
    data["metadata"]["prompt_version"] = PROMPT_VERSION
    data["metadata"]["requirement_count"] = len(validated_input.requirements)
    data["metadata"]["usage"] = usage

    # Count stories
    story_count = sum(len(epic.get("stories", [])) for epic in data.get("epics", []))
    data["metadata"]["story_count"] = story_count

    # Validate with Pydantic
    try:
        validated = TicketCollection(**data)
    except ValidationError as e:
        raise RuntimeError(f"LLM returned invalid ticket structure: {e}")

    return validated.model_dump()
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/test_server.py::TestDecomposeToTickets -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/prd_decomposer/server.py
git commit -m "refactor: add client/settings params to decompose_to_tickets"
```

---

## Task 7: Simplify Test Mocking

**Files:**
- Modify: `tests/test_server.py`

**Step 1: Update one test to use direct injection**

Update `test_analyze_prd_returns_structured_requirements` in `tests/test_server.py`:

```python
def test_analyze_prd_returns_structured_requirements(self):
    """Verify analyze_prd returns validated StructuredRequirements with metadata."""
    mock_response = {
        "requirements": [
            {
                "id": "REQ-001",
                "title": "User login",
                "description": "Users must be able to log in",
                "acceptance_criteria": ["Login form exists"],
                "dependencies": [],
                "ambiguity_flags": [],
                "priority": "high",
            }
        ],
        "summary": "Authentication system PRD",
        "source_hash": "abc12345",
    }

    mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
        usage=mock_usage,
    )

    # Direct injection - no patching needed
    result = analyze_prd("Test PRD content", client=mock_client)

    assert "requirements" in result
    assert len(result["requirements"]) == 1
    assert result["requirements"][0]["id"] == "REQ-001"
    assert result["summary"] == "Authentication system PRD"
    assert len(result["source_hash"]) == 8
    assert "_metadata" in result
    assert "usage" in result["_metadata"]
    assert "prompt_version" in result["_metadata"]
```

**Step 2: Run the updated test**

Run: `uv run pytest tests/test_server.py::TestAnalyzePrd::test_analyze_prd_returns_structured_requirements -v`
Expected: PASS

**Step 3: Update remaining tests similarly**

Update all tests in `TestAnalyzePrd`, `TestDecomposeToTickets`, and `TestIntegrationPipeline` to use direct client injection instead of `with patch(...)`.

Key change pattern:
```python
# Before
with patch("prd_decomposer.server.get_client", return_value=mock_client):
    result = analyze_prd("Test PRD")

# After
result = analyze_prd("Test PRD", client=mock_client)
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All 51+ tests PASS

**Step 5: Commit**

```bash
git add tests/test_server.py
git commit -m "refactor: simplify tests with direct client injection"
```

---

## Task 8: Add Few-Shot Example to ANALYZE_PRD_PROMPT

**Files:**
- Modify: `src/prd_decomposer/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write test for few-shot example presence**

Add to `tests/test_prompts.py`:

```python
def test_analyze_prd_prompt_has_example():
    """Verify ANALYZE_PRD_PROMPT contains a few-shot example."""
    assert "## Example" in ANALYZE_PRD_PROMPT
    assert "ambiguity_flags" in ANALYZE_PRD_PROMPT
    # Example should demonstrate ambiguity detection
    assert "Vague quantifier" in ANALYZE_PRD_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py::test_analyze_prd_prompt_has_example -v`
Expected: FAIL with `AssertionError`

**Step 3: Update ANALYZE_PRD_PROMPT with few-shot example**

Update `src/prd_decomposer/prompts.py`:

```python
"""Prompt templates for PRD analysis and decomposition."""

# Version for traceability - increment when prompts change
PROMPT_VERSION = "1.1.0"

ANALYZE_PRD_PROMPT = """You are a senior technical product manager. Analyze the following PRD and extract structured requirements.

For each requirement you identify:
1. Assign a unique ID (REQ-001, REQ-002, etc.)
2. Write a clear title and description
3. Extract or infer acceptance criteria (testable conditions for success)
4. Identify dependencies on other requirements (by ID)
5. Flag ambiguities - add to ambiguity_flags if:
   - Missing acceptance criteria (no clear way to test success)
   - Vague quantifiers without metrics (e.g., "fast", "scalable", "user-friendly", "easy to use")
6. Assign priority: "high", "medium", or "low" based on language cues and business impact

## Example

**Input PRD:**
# Feature: Password Reset
Users must be able to reset their password via email.
The reset flow should be fast and user-friendly.

**Output:**
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "Email-based password reset",
      "description": "Users can request a password reset link sent to their registered email address",
      "acceptance_criteria": [
        "User can request reset from login page",
        "Reset email sent within 30 seconds",
        "Reset link expires after 1 hour",
        "User can set new password via reset link"
      ],
      "dependencies": [],
      "ambiguity_flags": [
        "Vague quantifier: 'fast' - no specific latency requirement defined",
        "Vague quantifier: 'user-friendly' - no measurable UX criteria specified"
      ],
      "priority": "high"
    }}
  ],
  "summary": "Password reset feature allowing users to recover account access via email"
}}

---

Now analyze this PRD:
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
  "source_hash": "Will be set by the system"
}}"""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/prd_decomposer/prompts.py tests/test_prompts.py
git commit -m "feat: add few-shot example to ANALYZE_PRD_PROMPT"
```

---

## Task 9: Add Few-Shot Example to DECOMPOSE_TO_TICKETS_PROMPT

**Files:**
- Modify: `src/prd_decomposer/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write test for few-shot example presence**

Add to `tests/test_prompts.py`:

```python
def test_decompose_to_tickets_prompt_has_example():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT contains a few-shot example."""
    assert "## Example" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "epics" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "stories" in DECOMPOSE_TO_TICKETS_PROMPT
    # Example should show sizing
    assert '"size": "M"' in DECOMPOSE_TO_TICKETS_PROMPT or '"size": "S"' in DECOMPOSE_TO_TICKETS_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py::test_decompose_to_tickets_prompt_has_example -v`
Expected: FAIL

**Step 3: Update DECOMPOSE_TO_TICKETS_PROMPT with few-shot example**

Update in `src/prd_decomposer/prompts.py`:

```python
DECOMPOSE_TO_TICKETS_PROMPT = """You are a senior engineering manager. Convert these structured requirements into Jira-ready epics and stories.

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

## Example

**Input Requirements:**
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "Email-based password reset",
      "description": "Users can request a password reset link sent to their email",
      "acceptance_criteria": ["Reset email sent within 30 seconds", "Link expires after 1 hour"],
      "dependencies": [],
      "ambiguity_flags": [],
      "priority": "high"
    }}
  ],
  "summary": "Password reset feature"
}}

**Output:**
{{
  "epics": [
    {{
      "title": "Password Reset",
      "description": "Enable users to securely reset their passwords via email",
      "stories": [
        {{
          "title": "Create password reset request endpoint",
          "description": "Implement POST /auth/reset-password endpoint that validates email and sends reset link",
          "acceptance_criteria": [
            "Endpoint accepts email in request body",
            "Returns 200 for valid registered emails",
            "Returns 200 for unregistered emails (prevent enumeration)",
            "Triggers email send within 30 seconds"
          ],
          "size": "M",
          "labels": ["backend", "api", "auth"],
          "requirement_ids": ["REQ-001"]
        }},
        {{
          "title": "Implement password reset email template",
          "description": "Create email template with secure reset link and branding",
          "acceptance_criteria": [
            "Email contains secure one-time reset link",
            "Link expires after 1 hour",
            "Email follows brand guidelines"
          ],
          "size": "S",
          "labels": ["backend", "email"],
          "requirement_ids": ["REQ-001"]
        }},
        {{
          "title": "Build password reset form UI",
          "description": "Create frontend form for entering new password after clicking reset link",
          "acceptance_criteria": [
            "Form validates password strength",
            "Shows success/error states",
            "Redirects to login on success"
          ],
          "size": "M",
          "labels": ["frontend", "auth"],
          "requirement_ids": ["REQ-001"]
        }}
      ],
      "labels": ["auth", "security"]
    }}
  ],
  "metadata": {{
    "requirement_count": 1,
    "story_count": 3
  }}
}}

---

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
}}"""
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/prd_decomposer/prompts.py tests/test_prompts.py
git commit -m "feat: add few-shot example to DECOMPOSE_TO_TICKETS_PROMPT"
```

---

## Task 10: Final Verification

**Step 1: Run linting**

Run: `uv run ruff check src/ tests/`
Expected: No errors

**Step 2: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --cov=prd_decomposer --cov-report=term-missing`
Expected: All tests PASS, coverage >= 99%

**Step 3: Verify MCP server starts**

Run: `uv run python -c "from prd_decomposer.server import app; print('Server OK')"`
Expected: `Server OK`

**Step 4: Commit any final fixes**

If needed, commit any remaining fixes.

---

## Summary

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add pydantic-settings dependency | `build: add pydantic-settings dependency` |
| 2 | Create Settings class with tests | `feat: add Settings class with env var support` |
| 3 | Update __init__.py exports | `feat: export Settings and get_settings from package` |
| 4 | Update _call_llm_with_retry for DI | `refactor: add client/settings params to _call_llm_with_retry` |
| 5 | Update analyze_prd for DI | `refactor: add client/settings params to analyze_prd` |
| 6 | Update decompose_to_tickets for DI | `refactor: add client/settings params to decompose_to_tickets` |
| 7 | Simplify test mocking | `refactor: simplify tests with direct client injection` |
| 8 | Add few-shot to ANALYZE_PRD_PROMPT | `feat: add few-shot example to ANALYZE_PRD_PROMPT` |
| 9 | Add few-shot to DECOMPOSE_TO_TICKETS_PROMPT | `feat: add few-shot example to DECOMPOSE_TO_TICKETS_PROMPT` |
| 10 | Final verification | (verification only) |
