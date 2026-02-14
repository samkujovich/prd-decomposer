from typing import Literal
from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """A single requirement extracted from a PRD."""

    id: str = Field(..., description="Unique identifier (e.g., REQ-001)")
    title: str = Field(..., description="Short title of the requirement")
    description: str = Field(..., description="Detailed description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Testable acceptance criteria"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of requirements this depends on"
    )
    ambiguity_flags: list[str] = Field(
        default_factory=list,
        description="Reasons this requirement is ambiguous"
    )
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Priority level"
    )


class StructuredRequirements(BaseModel):
    """Collection of requirements extracted from a PRD."""

    requirements: list[Requirement] = Field(
        ..., description="List of extracted requirements"
    )
    summary: str = Field(..., description="Brief overview of the PRD")
    source_hash: str = Field(..., description="Hash of source PRD for traceability")


class Story(BaseModel):
    """A Jira-compatible story."""

    title: str = Field(..., description="Story title")
    description: str = Field(..., description="Story description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Acceptance criteria for the story"
    )
    size: Literal["S", "M", "L"] = Field(..., description="T-shirt size estimate")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")
    requirement_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source requirements for traceability"
    )


class Epic(BaseModel):
    """A Jira-compatible epic containing stories."""

    title: str = Field(..., description="Epic title")
    description: str = Field(..., description="Epic description")
    stories: list[Story] = Field(default_factory=list, description="Child stories")
    labels: list[str] = Field(default_factory=list, description="Labels/tags")


class TicketCollection(BaseModel):
    """Collection of epics ready for Jira import."""

    epics: list[Epic] = Field(..., description="List of epics with stories")
    metadata: dict = Field(
        default_factory=dict,
        description="Generation metadata (timestamp, model version, etc.)"
    )