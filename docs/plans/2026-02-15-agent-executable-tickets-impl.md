# Agent-Executable Tickets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `agent_context` field to tickets so AI agents can execute them with structured exploration, patterns, and verification.

**Architecture:** Add `AgentContext` Pydantic model, update `Story` model, modify decomposition prompt to generate agent context, add `render_agent_prompt()` formatter, add `prompt N` CLI command.

**Tech Stack:** Python, Pydantic v2, pytest

---

## Task 1: Add AgentContext Model

**Files:**
- Modify: `src/prd_decomposer/models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
class TestAgentContext:
    """Tests for AgentContext model."""

    def test_agent_context_requires_goal(self):
        """AgentContext requires a goal field."""
        from prd_decomposer.models import AgentContext

        with pytest.raises(ValidationError):
            AgentContext()

    def test_agent_context_minimal(self):
        """AgentContext with only required goal field."""
        from prd_decomposer.models import AgentContext

        ctx = AgentContext(goal="Enable users to reset passwords")
        assert ctx.goal == "Enable users to reset passwords"
        assert ctx.exploration_paths == []
        assert ctx.exploration_hints == []
        assert ctx.known_patterns == []
        assert ctx.verification_tests == []
        assert ctx.self_check == []

    def test_agent_context_full(self):
        """AgentContext with all fields populated."""
        from prd_decomposer.models import AgentContext

        ctx = AgentContext(
            goal="Enable users to reset passwords",
            exploration_paths=["auth", "email"],
            exploration_hints=["src/auth/"],
            known_patterns=["Use JWT tokens"],
            verification_tests=["test_reset_flow"],
            self_check=["Is token secure?"],
        )
        assert len(ctx.exploration_paths) == 2
        assert "src/auth/" in ctx.exploration_hints
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::TestAgentContext -v`
Expected: FAIL with "cannot import name 'AgentContext'"

**Step 3: Write minimal implementation**

Add to `src/prd_decomposer/models.py` (after `AmbiguityFlag`, before `Requirement`):

```python
class AgentContext(BaseModel):
    """AI agent execution context for a story."""

    goal: str = Field(
        ...,
        min_length=1,
        description="The 'why' - what problem this solves and why it matters",
    )
    exploration_paths: list[str] = Field(
        default_factory=list,
        description="Keywords/concepts to search for during exploration",
    )
    exploration_hints: list[str] = Field(
        default_factory=list,
        description="Optional specific paths or files to start with if known",
    )
    known_patterns: list[str] = Field(
        default_factory=list,
        description="Libraries, patterns, or conventions to follow",
    )
    verification_tests: list[str] = Field(
        default_factory=list,
        description="Test names or patterns that should pass when done",
    )
    self_check: list[str] = Field(
        default_factory=list,
        description="Questions the agent should verify before marking complete",
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::TestAgentContext -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/prd_decomposer/models.py tests/test_models.py
git commit -m "feat: add AgentContext model for AI-executable tickets"
```

---

## Task 2: Update Story Model

**Files:**
- Modify: `src/prd_decomposer/models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py` in `TestStory` class:

```python
def test_story_with_agent_context(self):
    """Story can include optional agent_context."""
    from prd_decomposer.models import AgentContext, Story

    ctx = AgentContext(goal="Enable password reset")
    story = Story(
        title="Create reset endpoint",
        size="M",
        agent_context=ctx,
    )
    assert story.agent_context is not None
    assert story.agent_context.goal == "Enable password reset"

def test_story_without_agent_context(self):
    """Story works without agent_context (backward compatible)."""
    from prd_decomposer.models import Story

    story = Story(title="Create reset endpoint", size="M")
    assert story.agent_context is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::TestStory::test_story_with_agent_context -v`
Expected: FAIL with "unexpected keyword argument 'agent_context'"

**Step 3: Write minimal implementation**

Modify `Story` class in `src/prd_decomposer/models.py`:

```python
class Story(BaseModel):
    """A Jira-compatible story."""

    title: str = Field(..., min_length=1, description="Story title")
    description: str = Field(default="", description="Story description")
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criteria for the story"
    )
    size: Literal["S", "M", "L"] = Field(..., description="T-shirt size estimate")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium", description="Story priority"
    )
    labels: list[str] = Field(default_factory=list, description="Labels/tags")
    requirement_ids: list[str] = Field(
        default_factory=list, description="IDs of source requirements for traceability"
    )
    agent_context: AgentContext | None = Field(
        default=None, description="Optional AI agent execution context"
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::TestStory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/prd_decomposer/models.py tests/test_models.py
git commit -m "feat: add optional agent_context field to Story model"
```

---

## Task 3: Update Decomposition Prompt

**Files:**
- Modify: `src/prd_decomposer/prompts.py`
- Test: `tests/test_prompts.py`

**Step 1: Write the failing test**

Add to `tests/test_prompts.py`:

```python
def test_decompose_prompt_includes_agent_context_instructions():
    """Decomposition prompt instructs LLM to generate agent_context."""
    from prd_decomposer.prompts import DECOMPOSE_TO_TICKETS_PROMPT

    assert "agent_context" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "goal" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "exploration_paths" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "self_check" in DECOMPOSE_TO_TICKETS_PROMPT


def test_decompose_prompt_example_includes_agent_context():
    """Decomposition prompt example shows agent_context usage."""
    from prd_decomposer.prompts import DECOMPOSE_TO_TICKETS_PROMPT

    # Example should demonstrate agent_context structure
    assert '"goal":' in DECOMPOSE_TO_TICKETS_PROMPT
    assert '"exploration_paths":' in DECOMPOSE_TO_TICKETS_PROMPT
    assert '"verification_tests":' in DECOMPOSE_TO_TICKETS_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py::test_decompose_prompt_includes_agent_context_instructions -v`
Expected: FAIL with assertion error

**Step 3: Write minimal implementation**

Update `DECOMPOSE_TO_TICKETS_PROMPT` in `src/prd_decomposer/prompts.py`:

1. Add guideline 8 after existing guidelines:

```
8. For each story, include an agent_context object with:
   - goal: A clear statement of WHY this work matters (the problem being solved)
   - exploration_paths: Keywords/concepts an AI agent should search for to understand context
   - exploration_hints: Specific file paths or modules to start with (if inferrable from requirements)
   - known_patterns: Libraries, conventions, or existing code patterns to follow
   - verification_tests: Test names or commands to verify completion
   - self_check: Questions to validate edge cases, security, and correctness
```

2. Update the example output to include agent_context in each story:

```json
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
  "priority": "high",
  "labels": ["backend", "api", "auth"],
  "requirement_ids": ["REQ-001"],
  "agent_context": {{
    "goal": "Allow users who forgot their password to securely regain account access without contacting support",
    "exploration_paths": ["password reset", "authentication", "email sending", "token generation"],
    "exploration_hints": ["src/auth/", "src/email/"],
    "known_patterns": ["Use existing email service", "Follow JWT token pattern for reset tokens"],
    "verification_tests": ["test_password_reset_request", "test_reset_token_expiry"],
    "self_check": [
      "Does this prevent email enumeration attacks?",
      "Is the reset token cryptographically secure?",
      "What happens if the email service is down?"
    ]
  }}
}}
```

3. Update the schema section at the end to include agent_context field.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

**Step 5: Bump prompt version and commit**

Update `PROMPT_VERSION = "1.6.0"` in prompts.py.

```bash
git add src/prd_decomposer/prompts.py tests/test_prompts.py
git commit -m "feat: update decomposition prompt for agent_context generation"
```

---

## Task 4: Add Prompt Renderer

**Files:**
- Modify: `agent/formatters.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
from agent.formatters import render_agent_prompt


class TestRenderAgentPrompt:
    """Tests for render_agent_prompt function."""

    def test_render_with_full_agent_context(self):
        """Render story with complete agent_context."""
        story = {
            "title": "Create reset endpoint",
            "description": "Implement POST /auth/reset-password",
            "acceptance_criteria": ["Returns 200 for valid emails"],
            "agent_context": {
                "goal": "Enable password recovery",
                "exploration_paths": ["auth", "email"],
                "exploration_hints": ["src/auth/"],
                "known_patterns": ["Use JWT tokens"],
                "verification_tests": ["test_reset"],
                "self_check": ["Is token secure?"],
            },
        }
        result = render_agent_prompt(story)

        assert "## Goal" in result
        assert "Enable password recovery" in result
        assert "## Task" in result
        assert "Create reset endpoint" in result
        assert "## Before You Start" in result
        assert "Search for: `auth`" in result
        assert "Start with: `src/auth/`" in result
        assert "## Patterns & Libraries" in result
        assert "Use JWT tokens" in result
        assert "## Acceptance Criteria" in result
        assert "- [ ] Returns 200 for valid emails" in result
        assert "## Verification" in result
        assert "`test_reset`" in result
        assert "## Before Marking Done" in result
        assert "Is token secure?" in result

    def test_render_without_agent_context(self):
        """Render story without agent_context falls back to basic format."""
        story = {
            "title": "Simple task",
            "description": "Do the thing",
        }
        result = render_agent_prompt(story)

        assert "## Simple task" in result
        assert "Do the thing" in result
        assert "## Goal" not in result

    def test_render_with_minimal_agent_context(self):
        """Render story with only goal in agent_context."""
        story = {
            "title": "Task",
            "description": "Description",
            "agent_context": {
                "goal": "The why",
            },
        }
        result = render_agent_prompt(story)

        assert "## Goal" in result
        assert "The why" in result
        assert "## Before You Start" not in result  # No exploration paths
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::TestRenderAgentPrompt -v`
Expected: FAIL with "cannot import name 'render_agent_prompt'"

**Step 3: Write minimal implementation**

Add to `agent/formatters.py`:

```python
def render_agent_prompt(story: dict[str, Any]) -> str:
    """Render a story as a ready-to-paste prompt for AI agents.

    Args:
        story: Story dict, optionally containing agent_context

    Returns:
        Markdown-formatted prompt string
    """
    ctx = story.get("agent_context")
    if not ctx:
        # Fallback to basic format for stories without agent_context
        return f"## {story.get('title', 'Task')}\n\n{story.get('description', '')}"

    sections = []

    # Goal (the why)
    if ctx.get("goal"):
        sections.append(f"## Goal\n{ctx['goal']}")

    # What to build
    title = story.get("title", "Task")
    desc = story.get("description", "")
    sections.append(f"## Task\n{title}\n\n{desc}" if desc else f"## Task\n{title}")

    # Exploration phase
    paths = ctx.get("exploration_paths", [])
    hints = ctx.get("exploration_hints", [])
    if paths or hints:
        exploration = "## Before You Start\nExplore the codebase to understand:\n"
        for path in paths:
            exploration += f"- Search for: `{path}`\n"
        for hint in hints:
            exploration += f"- Start with: `{hint}`\n"
        sections.append(exploration.rstrip())

    # Patterns to follow
    patterns = ctx.get("known_patterns", [])
    if patterns:
        pattern_section = "## Patterns & Libraries\n"
        for p in patterns:
            pattern_section += f"- {p}\n"
        sections.append(pattern_section.rstrip())

    # Acceptance criteria
    criteria = story.get("acceptance_criteria", [])
    if criteria:
        ac = "## Acceptance Criteria\n"
        for criterion in criteria:
            ac += f"- [ ] {criterion}\n"
        sections.append(ac.rstrip())

    # Verification
    tests = ctx.get("verification_tests", [])
    if tests:
        verify = "## Verification\nTests that should pass:\n"
        for test in tests:
            verify += f"- `{test}`\n"
        sections.append(verify.rstrip())

    # Self-check
    checks = ctx.get("self_check", [])
    if checks:
        check = "## Before Marking Done\nVerify:\n"
        for q in checks:
            check += f"- {q}\n"
        sections.append(check.rstrip())

    return "\n\n".join(sections)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py::TestRenderAgentPrompt -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add agent/formatters.py tests/test_agent.py
git commit -m "feat: add render_agent_prompt for copy-paste prompts"
```

---

## Task 5: Add CLI Prompt Command

**Files:**
- Modify: `agent/agent.py`
- Modify: `agent/session_state.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_parse_prompt_command():
    """Parse 'prompt N' command."""
    cmd, idx, arg = parse_command("prompt 1")
    assert cmd == "prompt"
    assert idx == 1
    assert arg is None


def test_parse_prompt_aliases():
    """Parse prompt command aliases."""
    for alias in ("copy", "show"):
        cmd, idx, _ = parse_command(f"{alias} 2")
        assert cmd == "prompt"
        assert idx == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_parse_prompt_command -v`
Expected: FAIL (cmd is None, not "prompt")

**Step 3: Write minimal implementation**

3a. Update `parse_command` in `agent/agent.py` to handle prompt command:

```python
# Add to the command patterns section
elif cmd_lower in ("prompt", "copy", "show"):
    return ("prompt", idx, None)
```

3b. Add `current_tickets` to `SessionState` in `agent/session_state.py`:

```python
@dataclass
class SessionState:
    # ... existing fields ...
    current_tickets: dict[str, Any] | None = None
```

Add method:

```python
def store_tickets(self, tickets: dict[str, Any]) -> None:
    """Store tickets from decompose_to_tickets."""
    self.current_tickets = tickets

def get_story_by_index(self, index: int) -> dict[str, Any] | None:
    """Get a story by 1-based index across all epics."""
    if not self.current_tickets:
        return None

    story_num = 0
    for epic in self.current_tickets.get("epics", []):
        for story in epic.get("stories", []):
            story_num += 1
            if story_num == index:
                return story
    return None
```

3c. Update `handle_command` in `agent/agent.py`:

```python
elif command == "prompt":
    if index is None:
        return "Usage: prompt [n] - Show copy-paste prompt for story N"
    story = session.get_story_by_index(index)
    if story is None:
        return f"Story #{index} not found. Run 'tickets' first."
    from agent.formatters import render_agent_prompt
    return render_agent_prompt(story)
```

3d. Update main loop to store tickets when extracted.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent/agent.py agent/session_state.py tests/test_agent.py
git commit -m "feat: add prompt command for copy-paste agent prompts"
```

---

## Task 6: Update Export Formats

**Files:**
- Modify: `src/prd_decomposer/export.py`
- Test: `tests/test_export.py`

**Step 1: Write the failing test**

Add to `tests/test_export.py`:

```python
def test_csv_export_includes_agent_prompt():
    """CSV export includes agent_prompt column."""
    from prd_decomposer.export import export_to_csv

    tickets = {
        "epics": [{
            "title": "Epic",
            "description": "Desc",
            "stories": [{
                "title": "Story",
                "description": "Do thing",
                "size": "M",
                "acceptance_criteria": [],
                "labels": [],
                "requirement_ids": [],
                "agent_context": {
                    "goal": "The why",
                    "exploration_paths": [],
                    "exploration_hints": [],
                    "known_patterns": [],
                    "verification_tests": [],
                    "self_check": [],
                },
            }],
            "labels": [],
        }],
    }
    result = export_to_csv(tickets)

    assert "agent_prompt" in result
    assert "The why" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export.py::test_csv_export_includes_agent_prompt -v`
Expected: FAIL (agent_prompt not in output)

**Step 3: Write minimal implementation**

Update `export_to_csv` in `src/prd_decomposer/export.py`:

1. Import render function:
```python
from agent.formatters import render_agent_prompt
```

2. Add `agent_prompt` to CSV headers and row generation.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/prd_decomposer/export.py tests/test_export.py
git commit -m "feat: include agent_prompt in CSV export"
```

---

## Task 7: Full Integration Test

**Files:**
- Test: `tests/test_server.py`

**Step 1: Write the integration test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_decompose_generates_agent_context(self, mock_client_factory):
    """decompose_to_tickets generates agent_context for stories."""
    mock_response = {
        "epics": [{
            "title": "Test Epic",
            "description": "Desc",
            "stories": [{
                "title": "Test Story",
                "description": "Do thing",
                "size": "M",
                "acceptance_criteria": ["AC1"],
                "labels": ["backend"],
                "requirement_ids": ["REQ-001"],
                "agent_context": {
                    "goal": "Enable feature X",
                    "exploration_paths": ["feature"],
                    "exploration_hints": [],
                    "known_patterns": [],
                    "verification_tests": ["test_feature"],
                    "self_check": ["Does it work?"],
                },
            }],
            "labels": [],
        }],
    }
    mock_client = mock_client_factory(mock_response)

    result = await _decompose_to_tickets_impl(
        requirements={"requirements": [], "summary": "test", "source_hash": "abc"},
        client=mock_client,
    )

    story = result["epics"][0]["stories"][0]
    assert "agent_context" in story
    assert story["agent_context"]["goal"] == "Enable feature X"
```

**Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 3: Final commit**

```bash
git add tests/test_server.py
git commit -m "test: add integration test for agent_context generation"
```

---

## Task 8: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add documentation**

Add section under "Tools" or "Features":

```markdown
### AI-Executable Tickets

Stories include optional `agent_context` for AI coding assistants:

- **goal**: Why this work matters (the problem being solved)
- **exploration_paths**: Keywords to search in codebase
- **exploration_hints**: Specific files/modules to start with
- **known_patterns**: Libraries and conventions to follow
- **verification_tests**: Tests that should pass when done
- **self_check**: Questions to verify before completion

Use `prompt N` in the agent to get a copy-pasteable prompt for any story.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document agent_context and prompt command"
```

---

## Verification Checklist

After all tasks:

```bash
# All tests pass
uv run pytest tests/ -v

# Lint clean
uv run ruff check src/ agent/ tests/

# Type check
uv run mypy src/

# Server imports
uv run python -c "from prd_decomposer.models import AgentContext; print('OK')"
```

---

## Summary

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add AgentContext model | `feat: add AgentContext model` |
| 2 | Update Story model | `feat: add agent_context to Story` |
| 3 | Update decomposition prompt | `feat: update prompt for agent_context` |
| 4 | Add prompt renderer | `feat: add render_agent_prompt` |
| 5 | Add CLI prompt command | `feat: add prompt command` |
| 6 | Update CSV export | `feat: agent_prompt in CSV` |
| 7 | Integration test | `test: integration test` |
| 8 | Update README | `docs: document agent_context` |
