from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AmbiguityFlag(BaseModel):
    """Structured ambiguity flag with actionable guidance."""

    category: Literal[
        "missing_acceptance_criteria",
        "vague_quantifier",
        "undefined_term",
        "missing_details",
        "conflicting_requirements",
        "out_of_scope",
        "security_concern",
        "other",
    ] = Field(..., description="Type of ambiguity")
    issue: str = Field(..., min_length=1, description="Description of the ambiguity")
    severity: Literal["critical", "warning", "suggestion"] = Field(
        ..., description="How critical this issue is to resolve"
    )
    suggested_action: str = Field(
        ..., min_length=1, description="What the PRD author should do to resolve this"
    )


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


class Requirement(BaseModel):
    """A single requirement extracted from a PRD."""

    id: str = Field(..., min_length=1, description="Unique identifier (e.g., REQ-001)")
    title: str = Field(..., min_length=1, description="Short title of the requirement")
    description: str = Field(..., min_length=1, description="Detailed description")
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Testable acceptance criteria"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of requirements this depends on"
    )
    ambiguity_flags: list[AmbiguityFlag] = Field(
        default_factory=list, description="Structured ambiguity flags with actionable guidance"
    )
    priority: Literal["high", "medium", "low"] = Field(..., description="Priority level")


class StructuredRequirements(BaseModel):
    """Collection of requirements extracted from a PRD."""

    requirements: list[Requirement] = Field(..., description="List of extracted requirements")
    summary: str = Field(..., min_length=1, description="Brief overview of the PRD")
    source_hash: str = Field(..., min_length=1, description="Hash of source PRD for traceability")

    @model_validator(mode="after")
    def validate_dependencies(self) -> "StructuredRequirements":
        """Ensure all dependency IDs reference existing requirements."""
        valid_ids = {req.id for req in self.requirements}
        for req in self.requirements:
            for dep_id in req.dependencies:
                if dep_id not in valid_ids:
                    raise ValueError(
                        f"Requirement {req.id} depends on non-existent requirement {dep_id}"
                    )
        return self


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


class Epic(BaseModel):
    """A Jira-compatible epic containing stories."""

    title: str = Field(..., min_length=1, description="Epic title")
    description: str = Field(default="", description="Epic description")
    stories: list[Story] = Field(default_factory=list, description="Child stories")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")


class TicketCollection(BaseModel):
    """Collection of epics ready for Jira import."""

    epics: list[Epic] = Field(..., description="List of epics with stories")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata (timestamp, model version, etc.)"
    )


class SizeDefinition(BaseModel):
    """Definition of a single t-shirt size for story estimation."""

    duration: str = Field(..., min_length=1, description="Expected duration (e.g., 'Less than 1 day')")
    scope: str = Field(..., min_length=1, description="Scope description (e.g., 'Single component')")
    risk: str = Field(..., min_length=1, description="Risk level (e.g., 'Low risk')")


class SizingRubric(BaseModel):
    """Configurable rubric for t-shirt sizing stories."""

    small: SizeDefinition = Field(
        default_factory=lambda: SizeDefinition(
            duration="Less than 1 day",
            scope="Single component",
            risk="Low risk",
        ),
        description="Definition for Small stories",
    )
    medium: SizeDefinition = Field(
        default_factory=lambda: SizeDefinition(
            duration="1-3 days",
            scope="May touch multiple components",
            risk="Moderate complexity",
        ),
        description="Definition for Medium stories",
    )
    large: SizeDefinition = Field(
        default_factory=lambda: SizeDefinition(
            duration="3-5 days",
            scope="Significant complexity",
            risk="Unknowns or cross-team coordination",
        ),
        description="Definition for Large stories",
    )

    def to_prompt_text(self) -> str:
        """Format rubric as text for prompt injection."""
        return (
            f"   - S (Small): {self.small.duration}, {self.small.scope}, {self.small.risk}\n"
            f"   - M (Medium): {self.medium.duration}, {self.medium.scope}, {self.medium.risk}\n"
            f"   - L (Large): {self.large.duration}, {self.large.scope}, {self.large.risk}"
        )
