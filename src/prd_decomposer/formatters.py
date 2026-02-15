"""Prompt formatters for AI agent consumption."""

from typing import Any


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
