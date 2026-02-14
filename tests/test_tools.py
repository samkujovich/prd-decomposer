import pytest
from pydantic import ValidationError

from prd_decomposer.models import (
    Epic,
    Requirement,
    Story,
    StructuredRequirements,
    TicketCollection,
)


def test_requirement_model_valid():
    """Verify Requirement model accepts valid data."""
    req = Requirement(
        id="REQ-001",
        title="User authentication",
        description="Users must be able to log in with email and password",
        acceptance_criteria=["Login form exists", "JWT issued on success"],
        dependencies=[],
        ambiguity_flags=[],
        priority="high",
    )
    assert req.id == "REQ-001"
    assert req.priority == "high"
    assert len(req.acceptance_criteria) == 2


def test_requirement_invalid_priority():
    """Verify Requirement rejects invalid priority."""
    with pytest.raises(ValidationError):
        Requirement(
            id="REQ-001",
            title="Test",
            description="Test",
            acceptance_criteria=[],
            dependencies=[],
            ambiguity_flags=[],
            priority="critical",  # Invalid - must be high/medium/low
        )


def test_structured_requirements_model():
    """Verify StructuredRequirements validates nested requirements."""
    req = Requirement(
        id="REQ-001",
        title="Test requirement",
        description="Description",
        acceptance_criteria=["AC1"],
        dependencies=[],
        ambiguity_flags=[],
        priority="medium",
    )

    structured = StructuredRequirements(
        requirements=[req], summary="Test PRD summary", source_hash="abc123"
    )

    assert len(structured.requirements) == 1
    assert structured.summary == "Test PRD summary"


def test_structured_requirements_serialization():
    """Verify StructuredRequirements round-trips through JSON."""
    req = Requirement(
        id="REQ-001",
        title="Test",
        description="Desc",
        acceptance_criteria=[],
        dependencies=[],
        ambiguity_flags=["Missing metrics"],
        priority="low",
    )

    original = StructuredRequirements(requirements=[req], summary="Summary", source_hash="hash123")

    # Round-trip through JSON
    json_str = original.model_dump_json()
    restored = StructuredRequirements.model_validate_json(json_str)

    assert restored.requirements[0].id == "REQ-001"
    assert restored.requirements[0].ambiguity_flags == ["Missing metrics"]


def test_story_model_valid():
    """Verify Story model accepts valid data."""
    story = Story(
        title="Implement login endpoint",
        description="Create POST /auth/login endpoint",
        acceptance_criteria=["Returns JWT on success", "Returns 401 on failure"],
        size="M",
        labels=["backend", "auth"],
        requirement_ids=["REQ-001"],
    )
    assert story.size == "M"
    assert "backend" in story.labels


def test_story_invalid_size():
    """Verify Story rejects invalid size."""
    with pytest.raises(ValidationError):
        Story(
            title="Test",
            description="Test",
            acceptance_criteria=[],
            size="XL",  # Invalid - must be S/M/L
            labels=[],
            requirement_ids=[],
        )


def test_epic_model_with_stories():
    """Verify Epic contains stories correctly."""
    story = Story(
        title="Story 1",
        description="Desc",
        acceptance_criteria=[],
        size="S",
        labels=[],
        requirement_ids=["REQ-001"],
    )

    epic = Epic(
        title="Authentication Epic",
        description="All auth-related work",
        stories=[story],
        labels=["auth"],
    )

    assert len(epic.stories) == 1
    assert epic.stories[0].title == "Story 1"


def test_ticket_collection_model():
    """Verify TicketCollection contains epics and metadata."""
    story = Story(
        title="Story",
        description="Desc",
        acceptance_criteria=[],
        size="S",
        labels=[],
        requirement_ids=[],
    )
    epic = Epic(title="Epic", description="Desc", stories=[story], labels=[])

    collection = TicketCollection(
        epics=[epic], metadata={"generated_at": "2026-02-14", "model": "gpt-4o"}
    )

    assert len(collection.epics) == 1
    assert collection.metadata["model"] == "gpt-4o"


def test_ticket_collection_serialization():
    """Verify TicketCollection round-trips through JSON."""
    story = Story(
        title="Story",
        description="Desc",
        acceptance_criteria=["AC1"],
        size="L",
        labels=["backend"],
        requirement_ids=["REQ-001"],
    )
    epic = Epic(title="Epic", description="Desc", stories=[story], labels=["auth"])
    original = TicketCollection(epics=[epic], metadata={"version": "1.0"})

    json_str = original.model_dump_json()
    restored = TicketCollection.model_validate_json(json_str)

    assert restored.epics[0].stories[0].size == "L"
    assert restored.epics[0].stories[0].labels == ["backend"]
