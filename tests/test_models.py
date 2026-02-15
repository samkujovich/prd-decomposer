import pytest
from pydantic import ValidationError

from prd_decomposer.models import (
    AmbiguityFlag,
    Epic,
    Requirement,
    SizeDefinition,
    SizingRubric,
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
    flag = AmbiguityFlag(
        category="vague_quantifier",
        issue="Missing metrics",
        severity="warning",
        suggested_action="Add specific metrics",
    )
    req = Requirement(
        id="REQ-001",
        title="Test",
        description="Desc",
        acceptance_criteria=[],
        dependencies=[],
        ambiguity_flags=[flag],
        priority="low",
    )

    original = StructuredRequirements(requirements=[req], summary="Summary", source_hash="hash123")

    # Round-trip through JSON
    json_str = original.model_dump_json()
    restored = StructuredRequirements.model_validate_json(json_str)

    assert restored.requirements[0].id == "REQ-001"
    assert len(restored.requirements[0].ambiguity_flags) == 1
    assert restored.requirements[0].ambiguity_flags[0].issue == "Missing metrics"


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


def test_requirement_empty_id_rejected():
    """Verify empty string IDs are rejected by min_length constraint."""
    with pytest.raises(ValidationError):
        Requirement(id="", title="T", description="D", priority="high")


def test_structured_requirements_empty_list_accepted():
    """Document that empty requirements list is accepted."""
    sr = StructuredRequirements(requirements=[], summary="Empty", source_hash="abc")
    assert len(sr.requirements) == 0


def test_epic_empty_stories_accepted():
    """Document that an epic with no stories is accepted."""
    epic = Epic(title="E", description="D", stories=[], labels=[])
    assert len(epic.stories) == 0


def test_story_all_sizes_accepted():
    """Verify all valid t-shirt sizes are accepted."""
    for size in ("S", "M", "L"):
        story = Story(
            title="T", description="D", size=size,
            acceptance_criteria=[], labels=[], requirement_ids=[],
        )
        assert story.size == size


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


# SizingRubric tests


def test_size_definition_valid():
    """Verify SizeDefinition accepts valid data."""
    size_def = SizeDefinition(
        duration="Half day",
        scope="Single function",
        risk="Minimal",
    )
    assert size_def.duration == "Half day"
    assert size_def.scope == "Single function"


def test_size_definition_empty_duration_rejected():
    """Verify SizeDefinition rejects empty duration."""
    with pytest.raises(ValidationError):
        SizeDefinition(
            duration="",  # min_length=1
            scope="Single component",
            risk="Low",
        )


def test_sizing_rubric_default():
    """Verify SizingRubric has sensible defaults."""
    rubric = SizingRubric()
    assert "Less than 1 day" in rubric.small.duration
    assert "1-3 days" in rubric.medium.duration
    assert "3-5 days" in rubric.large.duration


def test_sizing_rubric_custom():
    """Verify SizingRubric accepts custom definitions."""
    rubric = SizingRubric(
        small=SizeDefinition(
            duration="Up to 4 hours",
            scope="Single file",
            risk="None",
        ),
        medium=SizeDefinition(
            duration="1-2 days",
            scope="Few files",
            risk="Low",
        ),
        large=SizeDefinition(
            duration="1 week",
            scope="Multiple modules",
            risk="Medium",
        ),
    )
    assert rubric.small.duration == "Up to 4 hours"
    assert rubric.large.duration == "1 week"


def test_sizing_rubric_to_prompt_text():
    """Verify SizingRubric formats correctly for prompt injection."""
    rubric = SizingRubric()
    prompt_text = rubric.to_prompt_text()

    assert "S (Small):" in prompt_text
    assert "M (Medium):" in prompt_text
    assert "L (Large):" in prompt_text
    assert "Less than 1 day" in prompt_text
    assert "1-3 days" in prompt_text


def test_sizing_rubric_json_roundtrip():
    """Verify SizingRubric survives JSON serialization."""
    original = SizingRubric(
        small=SizeDefinition(duration="4h", scope="tiny", risk="none"),
        medium=SizeDefinition(duration="2d", scope="small", risk="low"),
        large=SizeDefinition(duration="5d", scope="big", risk="high"),
    )
    json_str = original.model_dump_json()
    restored = SizingRubric.model_validate_json(json_str)

    assert restored.small.duration == "4h"
    assert restored.large.risk == "high"


# AmbiguityFlag tests


def test_ambiguity_flag_valid():
    """Verify AmbiguityFlag accepts valid data."""
    flag = AmbiguityFlag(
        category="vague_quantifier",
        issue="'fast' has no specific latency requirement",
        severity="warning",
        suggested_action="Define target latency, e.g., 'under 100ms'",
    )
    assert flag.category == "vague_quantifier"
    assert flag.severity == "warning"
    assert "Define target" in flag.suggested_action


def test_ambiguity_flag_all_categories():
    """Verify all category values are accepted."""
    categories = [
        "missing_acceptance_criteria",
        "vague_quantifier",
        "undefined_term",
        "missing_details",
        "conflicting_requirements",
        "out_of_scope",
        "security_concern",
        "other",
    ]
    for cat in categories:
        flag = AmbiguityFlag(
            category=cat,
            issue="Test issue",
            severity="warning",
            suggested_action="Fix it",
        )
        assert flag.category == cat


def test_ambiguity_flag_all_severities():
    """Verify all severity values are accepted."""
    for sev in ["critical", "warning", "suggestion"]:
        flag = AmbiguityFlag(
            category="other",
            issue="Test",
            severity=sev,
            suggested_action="Fix",
        )
        assert flag.severity == sev


def test_ambiguity_flag_invalid_category():
    """Verify AmbiguityFlag rejects invalid category."""
    with pytest.raises(ValidationError):
        AmbiguityFlag(
            category="unknown_category",
            issue="Test",
            severity="warning",
            suggested_action="Fix",
        )


def test_ambiguity_flag_invalid_severity():
    """Verify AmbiguityFlag rejects invalid severity."""
    with pytest.raises(ValidationError):
        AmbiguityFlag(
            category="other",
            issue="Test",
            severity="low",  # Invalid - must be critical/warning/suggestion
            suggested_action="Fix",
        )


def test_ambiguity_flag_empty_issue_rejected():
    """Verify AmbiguityFlag rejects empty issue."""
    with pytest.raises(ValidationError):
        AmbiguityFlag(
            category="other",
            issue="",  # min_length=1
            severity="warning",
            suggested_action="Fix it",
        )


def test_ambiguity_flag_json_roundtrip():
    """Verify AmbiguityFlag survives JSON serialization."""
    original = AmbiguityFlag(
        category="security_concern",
        issue="No rate limiting mentioned",
        severity="critical",
        suggested_action="Add rate limiting requirements to prevent DoS",
    )
    json_str = original.model_dump_json()
    restored = AmbiguityFlag.model_validate_json(json_str)

    assert restored.category == "security_concern"
    assert restored.severity == "critical"
    assert "DoS" in restored.suggested_action


# AgentContext tests


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_agent_context_requires_goal(self):
        """AgentContext requires a goal field."""
        from prd_decomposer.models import AgentContext

        with pytest.raises(ValidationError):
            AgentContext()

    def test_agent_context_minimal(self):
        """AgentContext with only required goal field."""
        from prd_decomposer.models import AgentContext

        ctx = AgentContext(goal="Enable users to reset passwords")
        assert ctx.goal == "Enable users to reset passwords"
        assert ctx.exploration_paths == []
        assert ctx.exploration_hints == []
        assert ctx.known_patterns == []
        assert ctx.verification_tests == []
        assert ctx.self_check == []

    def test_agent_context_full(self):
        """AgentContext with all fields populated."""
        from prd_decomposer.models import AgentContext

        ctx = AgentContext(
            goal="Enable users to reset passwords",
            exploration_paths=["auth", "email"],
            exploration_hints=["src/auth/"],
            known_patterns=["Use JWT tokens"],
            verification_tests=["test_reset_flow"],
            self_check=["Is token secure?"],
        )
        assert len(ctx.exploration_paths) == 2
        assert "src/auth/" in ctx.exploration_hints

    def test_agent_context_json_roundtrip(self):
        """Verify AgentContext survives JSON serialization."""
        from prd_decomposer.models import AgentContext

        original = AgentContext(
            goal="Enable password reset",
            exploration_paths=["auth", "email"],
            exploration_hints=["src/auth/"],
            known_patterns=["Use JWT tokens"],
            verification_tests=["test_reset_flow"],
            self_check=["Is token secure?"],
        )
        json_str = original.model_dump_json()
        restored = AgentContext.model_validate_json(json_str)

        assert restored.goal == original.goal
        assert restored.exploration_paths == original.exploration_paths
        assert restored.self_check == original.self_check
