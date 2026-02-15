"""Output formatters for the PRD Decomposer agent.

Provides formatted views of requirements, ambiguities, and tickets
for better readability in the terminal.
"""

from typing import Any


def format_requirements_table(requirements: dict[str, Any]) -> str:
    """Format requirements as a markdown table.

    Args:
        requirements: The analyze_prd result containing requirements

    Returns:
        Markdown table string
    """
    reqs = requirements.get("requirements", [])
    if not reqs:
        return "No requirements found."

    lines = [
        "## Requirements\n",
        "| ID | Title | Priority | Ambiguities |",
        "|:---|:------|:---------|:------------|",
    ]

    for req in reqs:
        req_id = req.get("id", "?")
        title = req.get("title", "Untitled")[:40]
        priority = req.get("priority", "medium")
        amb_count = len(req.get("ambiguity_flags", []))
        amb_display = str(amb_count) if amb_count > 0 else "-"
        lines.append(f"| {req_id} | {title} | {priority} | {amb_display} |")

    return "\n".join(lines)


def format_tickets_hierarchy(tickets: dict[str, Any]) -> str:
    """Format tickets as a tree hierarchy.

    Args:
        tickets: The decompose_to_tickets result

    Returns:
        Tree view string showing epics and stories
    """
    epics = tickets.get("epics", [])
    if not epics:
        return "No tickets generated."

    lines = ["## Tickets\n"]

    for i, epic in enumerate(epics, 1):
        epic_title = epic.get("title", "Untitled Epic")
        stories = epic.get("stories", [])
        is_last_epic = i == len(epics)

        # Epic line
        prefix = "└── " if is_last_epic else "├── "
        lines.append(f"{prefix}📦 **{epic_title}**")

        # Stories under this epic
        for j, story in enumerate(stories, 1):
            story_title = story.get("title", "Untitled Story")
            size = story.get("size", "?")
            is_last_story = j == len(stories)

            # Indentation depends on whether this is last epic
            indent = "    " if is_last_epic else "│   "
            story_prefix = "└── " if is_last_story else "├── "

            size_emoji = {"S": "🟢", "M": "🟡", "L": "🔴"}.get(size, "⚪")
            lines.append(f"{indent}{story_prefix}{size_emoji} [{size}] {story_title}")

    # Summary
    total_stories = sum(len(e.get("stories", [])) for e in epics)
    lines.append(f"\n**Summary:** {len(epics)} epic(s), {total_stories} story(ies)")

    return "\n".join(lines)


def format_analysis_summary(requirements: dict[str, Any]) -> str:
    """Format a brief summary of the analysis results.

    Args:
        requirements: The analyze_prd result

    Returns:
        Summary string with counts and warnings
    """
    reqs = requirements.get("requirements", [])

    total_reqs = len(reqs)
    total_ambiguities = sum(len(r.get("ambiguity_flags", [])) for r in reqs)

    # Count by severity
    critical = 0
    warning = 0
    suggestion = 0
    for req in reqs:
        for flag in req.get("ambiguity_flags", []):
            sev = flag.get("severity", "")
            if sev == "critical":
                critical += 1
            elif sev == "warning":
                warning += 1
            elif sev == "suggestion":
                suggestion += 1

    lines = ["## Analysis Complete\n"]
    lines.append(f"**{total_reqs} requirement(s)** extracted")

    if total_ambiguities > 0:
        parts = []
        if critical > 0:
            parts.append(f"🔴 {critical} critical")
        if warning > 0:
            parts.append(f"🟡 {warning} warning")
        if suggestion > 0:
            parts.append(f"💡 {suggestion} suggestion")

        lines.append(f"\n⚠️ **{total_ambiguities} ambiguity(ies)** found: {', '.join(parts)}")
    else:
        lines.append("\n✅ No ambiguities detected")

    return "\n".join(lines)


def format_ticket_summary(tickets: dict[str, Any]) -> str:
    """Format a brief summary of generated tickets.

    Args:
        tickets: The decompose_to_tickets result

    Returns:
        Summary string with counts
    """
    epics = tickets.get("epics", [])
    total_stories = sum(len(e.get("stories", [])) for e in epics)

    # Size breakdown
    sizes = {"S": 0, "M": 0, "L": 0}
    for epic in epics:
        for story in epic.get("stories", []):
            size = story.get("size", "M")
            if size in sizes:
                sizes[size] += 1

    lines = [
        "## Decomposition Complete\n",
        f"**{len(epics)} epic(s)** with **{total_stories} story(ies)**",
        f"\nSize breakdown: 🟢 {sizes['S']} Small, 🟡 {sizes['M']} Medium, 🔴 {sizes['L']} Large",
    ]

    return "\n".join(lines)
