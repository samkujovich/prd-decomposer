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
