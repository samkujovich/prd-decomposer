"""Session state management for the PRD Decomposer agent.

Tracks analyzed requirements and user decisions about ambiguities
within a single session.
"""

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """Tracks state across agent interactions within a session.

    Attributes:
        current_requirements: The most recent analyze_prd result, or None
        accepted_ambiguities: Set of ambiguity IDs the user has accepted
        dismissed_ambiguities: Set of ambiguity IDs the user has dismissed
        clarifications: Map of requirement ID -> clarification text
    """

    current_requirements: dict[str, Any] | None = None
    accepted_ambiguities: set[str] = field(default_factory=set)
    dismissed_ambiguities: set[str] = field(default_factory=set)
    clarifications: dict[str, str] = field(default_factory=dict)

    def store_requirements(self, requirements: dict[str, Any]) -> None:
        """Store requirements from analyze_prd and reset decisions."""
        self.current_requirements = requirements
        self.accepted_ambiguities.clear()
        self.dismissed_ambiguities.clear()
        self.clarifications.clear()

    def get_ambiguities(self) -> list[dict[str, Any]]:
        """Get all ambiguities from current requirements with unique IDs."""
        if not self.current_requirements:
            return []

        ambiguities = []
        # Track counts per (req_id, category) to ensure unique IDs
        category_counts: dict[str, int] = {}

        for req in self.current_requirements.get("requirements", []):
            req_id = req.get("id", "")
            for flag in req.get("ambiguity_flags", []):
                category = flag.get("category", "unknown")
                # Create unique key and increment counter
                key = f"{req_id}:{category}"
                count = category_counts.get(key, 0)
                category_counts[key] = count + 1
                # Include index in ID to handle duplicates
                amb_id = f"{req_id}:{category}:{count}"

                ambiguities.append({
                    "id": amb_id,
                    "requirement_id": req_id,
                    "requirement_title": req.get("title", ""),
                    "category": category,
                    "severity": flag.get("severity", ""),
                    "issue": flag.get("issue", ""),
                    "suggested_action": flag.get("suggested_action", ""),
                })
        return ambiguities

    def get_active_ambiguities(self) -> list[dict[str, Any]]:
        """Get ambiguities that haven't been accepted or dismissed."""
        return [
            amb for amb in self.get_ambiguities()
            if amb["id"] not in self.accepted_ambiguities
            and amb["id"] not in self.dismissed_ambiguities
        ]

    def accept_ambiguity(self, index: int) -> str | None:
        """Accept an ambiguity by its 1-based index.

        Returns the ambiguity ID if successful, None if invalid index.
        """
        active = self.get_active_ambiguities()
        if 1 <= index <= len(active):
            amb_id = active[index - 1]["id"]
            self.accepted_ambiguities.add(amb_id)
            return amb_id
        return None

    def dismiss_ambiguity(self, index: int) -> str | None:
        """Dismiss an ambiguity by its 1-based index.

        Returns the ambiguity ID if successful, None if invalid index.
        """
        active = self.get_active_ambiguities()
        if 1 <= index <= len(active):
            amb_id = active[index - 1]["id"]
            self.dismissed_ambiguities.add(amb_id)
            return amb_id
        return None

    def add_clarification(self, index: int, clarification: str) -> str | None:
        """Add clarification for an ambiguity by its 1-based index.

        Returns the requirement ID if successful, None if invalid index.
        """
        active = self.get_active_ambiguities()
        if 1 <= index <= len(active):
            req_id = active[index - 1]["requirement_id"]
            self.clarifications[req_id] = clarification
            # Also mark as accepted since it's been addressed
            self.accepted_ambiguities.add(active[index - 1]["id"])
            return req_id
        return None

    def get_requirements_with_clarifications(self) -> dict[str, Any] | None:
        """Get requirements with clarifications injected into descriptions.

        Note: Accepted/dismissed ambiguities are tracked for UI display but are
        NOT filtered from the requirements passed to ticket generation. This is
        intentional - ambiguities describe PRD quality issues, not whether the
        requirement should be implemented. A future enhancement could optionally
        filter dismissed ambiguities from the ambiguity_flags.
        """
        if not self.current_requirements:
            return None

        # Deep copy to avoid mutating original
        result = copy.deepcopy(self.current_requirements)

        for req in result.get("requirements", []):
            req_id = req.get("id", "")
            if req_id in self.clarifications:
                clarification = self.clarifications[req_id]
                req["description"] = (
                    f"{req.get('description', '')}\n\n"
                    f"**Clarification:** {clarification}"
                )

        return result

    def format_ambiguities_display(self) -> str:
        """Format active ambiguities for display."""
        active = self.get_active_ambiguities()
        if not active:
            return "No active ambiguities."

        count = len(active)
        noun = "Ambiguity" if count == 1 else "Ambiguities"
        lines = [f"## {count} {noun} to Review\n"]

        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡",
            "suggestion": "💡",
        }

        for i, amb in enumerate(active, 1):
            emoji = severity_emoji.get(amb["severity"], "⚪")
            lines.append(
                f"  [{i}] {emoji} {amb['severity']} | {amb['requirement_id']} | "
                f"{amb['issue']}"
            )
            lines.append(f"      → {amb['suggested_action']}")
            lines.append("")

        lines.append("Commands: accept [n], dismiss [n], clarify [n] \"text\", tickets")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all session state."""
        self.current_requirements = None
        self.accepted_ambiguities.clear()
        self.dismissed_ambiguities.clear()
        self.clarifications.clear()
