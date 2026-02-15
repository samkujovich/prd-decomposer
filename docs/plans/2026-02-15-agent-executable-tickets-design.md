# Agent-Executable Tickets Design

**Date:** 2026-02-15
**Status:** Approved
**Branch:** `feat/prompt-improvements`

## Problem

Current tickets are human-readable but not AI-agent optimized. When an engineer copies a ticket into Claude Code (or similar), the agent:

1. Doesn't know where to start (has to explore codebase first)
2. Makes wrong architectural decisions (doesn't follow existing patterns)
3. Doesn't know when it's done (acceptance criteria too vague)
4. Misses edge cases (no prompts for security, error handling, testing)

## Solution

Add an optional `agent_context` field to each Story that provides AI agents with structured guidance for exploration, implementation, and verification.

## Design Principles

1. **Two-phase execution**: Exploration first (understand context), then implementation
2. **Goal-driven**: Keep the "why" front and center so agent can self-check
3. **Discoverable patterns**: Agent finds most patterns from codebase; ticket provides hints
4. **Backward compatible**: Existing human-readable fields unchanged; `agent_context` is additive

## Target Workflow

1. Engineer sees ticket in Jira/project tracker
2. Copies ticket content (or runs `prompt N` command in agent CLI)
3. Pastes into Claude Code
4. Agent explores codebase based on `exploration_paths` and `exploration_hints`
5. Agent implements following `known_patterns`
6. Agent verifies using `verification_tests` and `self_check` questions
7. Work is done when all checks pass

## Data Model

### New: `AgentContext`

```python
class AgentContext(BaseModel):
    """AI agent execution context for a story."""

    goal: str = Field(
        ...,
        description="The 'why' - what problem this solves and why it matters"
    )
    exploration_paths: list[str] = Field(
        default_factory=list,
        description="Keywords/concepts to search for during exploration"
    )
    exploration_hints: list[str] = Field(
        default_factory=list,
        description="Optional specific paths or files to start with if known"
    )
    known_patterns: list[str] = Field(
        default_factory=list,
        description="Libraries, patterns, or conventions to follow"
    )
    verification_tests: list[str] = Field(
        default_factory=list,
        description="Test names or patterns that should pass when done"
    )
    self_check: list[str] = Field(
        default_factory=list,
        description="Questions the agent should verify before marking complete"
    )
```

### Updated: `Story`

```python
class Story(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[str]
    size: Literal["S", "M", "L"]
    priority: Literal["high", "medium", "low"]
    labels: list[str]
    requirement_ids: list[str]
    agent_context: AgentContext | None = None  # NEW
```

## Prompt Changes

Update `DECOMPOSE_TO_TICKETS_PROMPT` to instruct LLM to generate `agent_context`:

```
8. For each story, include an agent_context with:
   - goal: A clear statement of WHY this work matters (the problem being solved)
   - exploration_paths: Keywords/concepts an AI agent should search for
   - exploration_hints: Specific file paths or modules to start with (if inferrable)
   - known_patterns: Libraries, conventions, or existing code patterns to follow
   - verification_tests: Test names or commands to verify completion
   - self_check: Questions to validate edge cases, security, and correctness
```

### Example Output

```json
{
  "title": "Create password reset request endpoint",
  "description": "Implement POST /auth/reset-password endpoint...",
  "acceptance_criteria": [
    "Endpoint accepts email in request body",
    "Returns 200 for valid registered emails",
    "Returns 200 for unregistered emails (prevent enumeration)"
  ],
  "size": "M",
  "labels": ["backend", "api", "auth"],
  "requirement_ids": ["REQ-001"],
  "agent_context": {
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
  }
}
```

## Prompt Renderer

Utility function to render `agent_context` into copy-pasteable markdown:

```python
def render_agent_prompt(story: dict) -> str:
    """Render a story as a ready-to-paste prompt for AI agents."""
```

### Rendered Output Example

```markdown
## Goal
Allow users who forgot their password to securely regain account access without contacting support

## Task
Create password reset request endpoint

Implement POST /auth/reset-password endpoint that validates email and sends reset link

## Before You Start
Explore the codebase to understand:
- Search for: `password reset`
- Search for: `authentication`
- Search for: `email sending`
- Start with: `src/auth/`

## Patterns & Libraries
- Use existing email service
- Follow JWT token pattern for reset tokens

## Acceptance Criteria
- [ ] Endpoint accepts email in request body
- [ ] Returns 200 for valid registered emails
- [ ] Returns 200 for unregistered emails (prevent enumeration)

## Verification
Tests that should pass:
- `test_password_reset_request`
- `test_reset_token_expiry`

## Before Marking Done
Verify:
- Does this prevent email enumeration attacks?
- Is the reset token cryptographically secure?
- What happens if the email service is down?
```

## Integration Points

### Agent CLI (`agent/agent.py`)

Add `prompt N` command to display rendered prompt for story N:

```
You: tickets
[shows ticket hierarchy]

You: prompt 1
[renders agent prompt for story 1]
```

### Export Formats (`src/prd_decomposer/export.py`)

| Format | Handling |
|--------|----------|
| JSON | Include `agent_context` as-is |
| CSV | Add `agent_prompt` column with rendered text |
| YAML | Include `agent_context` nested structure |
| Jira | Map to description body with formatting |

### Batch Script

No changes needed; JSON output already includes full structure.

## Files to Modify

1. `src/prd_decomposer/models.py` - Add `AgentContext`, update `Story`
2. `src/prd_decomposer/prompts.py` - Update decomposition prompt + example
3. `src/prd_decomposer/export.py` - Handle `agent_context` in CSV/Jira exports
4. `agent/formatters.py` - Add `render_agent_prompt()` function
5. `agent/agent.py` - Add `prompt N` command
6. `tests/test_models.py` - Test new model
7. `tests/test_agent.py` - Test prompt command

## Success Criteria

- [ ] Stories include `agent_context` when generated
- [ ] `prompt N` command renders copy-pasteable prompt
- [ ] Existing ticket formats still work (backward compatible)
- [ ] Export formats include agent context appropriately
- [ ] All existing tests pass + new tests for agent_context
