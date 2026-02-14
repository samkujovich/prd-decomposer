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
