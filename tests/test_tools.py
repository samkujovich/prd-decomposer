import pytest
from pydantic import ValidationError


def test_requirement_model_valid():
    """Verify Requirement model accepts valid data."""
    from prd_decomposer.models import Requirement

    req = Requirement(
        id="REQ-001",
        title="User authentication",
        description="Users must be able to log in with email and password",
        acceptance_criteria=["Login form exists", "JWT issued on success"],
        dependencies=[],
        ambiguity_flags=[],
        priority="high"
    )
    assert req.id == "REQ-001"
    assert req.priority == "high"
    assert len(req.acceptance_criteria) == 2


def test_requirement_invalid_priority():
    """Verify Requirement rejects invalid priority."""
    from prd_decomposer.models import Requirement

    with pytest.raises(ValidationError):
        Requirement(
            id="REQ-001",
            title="Test",
            description="Test",
            acceptance_criteria=[],
            dependencies=[],
            ambiguity_flags=[],
            priority="critical"  # Invalid - must be high/medium/low
        )


def test_structured_requirements_model():
    """Verify StructuredRequirements validates nested requirements."""
    from prd_decomposer.models import Requirement, StructuredRequirements

    req = Requirement(
        id="REQ-001",
        title="Test requirement",
        description="Description",
        acceptance_criteria=["AC1"],
        dependencies=[],
        ambiguity_flags=[],
        priority="medium"
    )

    structured = StructuredRequirements(
        requirements=[req],
        summary="Test PRD summary",
        source_hash="abc123"
    )

    assert len(structured.requirements) == 1
    assert structured.summary == "Test PRD summary"


def test_structured_requirements_serialization():
    """Verify StructuredRequirements round-trips through JSON."""
    from prd_decomposer.models import Requirement, StructuredRequirements

    req = Requirement(
        id="REQ-001",
        title="Test",
        description="Desc",
        acceptance_criteria=[],
        dependencies=[],
        ambiguity_flags=["Missing metrics"],
        priority="low"
    )

    original = StructuredRequirements(
        requirements=[req],
        summary="Summary",
        source_hash="hash123"
    )

    # Round-trip through JSON
    json_str = original.model_dump_json()
    restored = StructuredRequirements.model_validate_json(json_str)

    assert restored.requirements[0].id == "REQ-001"
    assert restored.requirements[0].ambiguity_flags == ["Missing metrics"]