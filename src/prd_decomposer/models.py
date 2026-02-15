from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    ambiguity_flags: list[str] = Field(
        default_factory=list, description="Reasons this requirement is ambiguous"
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
    description: str = Field(..., min_length=1, description="Story description")
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
    description: str = Field(..., min_length=1, description="Epic description")
    stories: list[Story] = Field(default_factory=list, description="Child stories")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")


class TicketCollection(BaseModel):
    """Collection of epics ready for Jira import."""

    epics: list[Epic] = Field(..., description="List of epics with stories")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata (timestamp, model version, etc.)"
    )
