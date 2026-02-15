"""Tests for prompt templates."""

from prd_decomposer.models import Epic, Requirement, Story
from prd_decomposer.prompts import (
    ANALYZE_PRD_PROMPT,
    DECOMPOSE_TO_TICKETS_PROMPT,
    DEFAULT_SIZING_RUBRIC,
    PROMPT_VERSION,
)


def test_prompt_version_exists():
    """Verify PROMPT_VERSION is defined and follows semver."""
    assert PROMPT_VERSION
    parts = PROMPT_VERSION.split(".")
    assert len(parts) == 3, "PROMPT_VERSION should be semver (x.y.z)"
    assert all(p.isdigit() for p in parts), "PROMPT_VERSION parts should be numeric"


def test_analyze_prd_prompt_exists():
    """Verify ANALYZE_PRD_PROMPT is defined and non-empty."""
    assert ANALYZE_PRD_PROMPT
    assert len(ANALYZE_PRD_PROMPT) > 100


def test_analyze_prd_prompt_has_placeholder():
    """Verify ANALYZE_PRD_PROMPT contains the prd_text placeholder."""
    assert "{prd_text}" in ANALYZE_PRD_PROMPT


def test_decompose_to_tickets_prompt_exists():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT is defined and non-empty."""
    assert DECOMPOSE_TO_TICKETS_PROMPT
    assert len(DECOMPOSE_TO_TICKETS_PROMPT) > 100


def test_decompose_to_tickets_prompt_has_placeholder():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT contains the requirements_json placeholder."""
    assert "{requirements_json}" in DECOMPOSE_TO_TICKETS_PROMPT


def test_decompose_to_tickets_prompt_has_sizing_rubric_placeholder():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT contains the sizing_rubric placeholder."""
    assert "{sizing_rubric}" in DECOMPOSE_TO_TICKETS_PROMPT


def test_default_sizing_rubric_has_all_sizes():
    """Verify DEFAULT_SIZING_RUBRIC defines S, M, and L."""
    assert "S (Small)" in DEFAULT_SIZING_RUBRIC or "- S:" in DEFAULT_SIZING_RUBRIC
    assert "M (Medium)" in DEFAULT_SIZING_RUBRIC or "- M:" in DEFAULT_SIZING_RUBRIC
    assert "L (Large)" in DEFAULT_SIZING_RUBRIC or "- L:" in DEFAULT_SIZING_RUBRIC


def test_prompts_are_formattable():
    """Verify prompts can be formatted with expected variables."""
    formatted_analyze = ANALYZE_PRD_PROMPT.format(prd_text="Test PRD content")
    assert "Test PRD content" in formatted_analyze

    formatted_decompose = DECOMPOSE_TO_TICKETS_PROMPT.format(
        requirements_json='{"requirements": []}',
        sizing_rubric=DEFAULT_SIZING_RUBRIC,
    )
    assert '{"requirements": []}' in formatted_decompose
    assert "S (Small)" in formatted_decompose


def test_analyze_prd_prompt_has_example():
    """Verify ANALYZE_PRD_PROMPT contains a few-shot example."""
    assert "## Example" in ANALYZE_PRD_PROMPT
    assert "ambiguity_flags" in ANALYZE_PRD_PROMPT
    # Example should demonstrate structured ambiguity detection
    assert "vague_quantifier" in ANALYZE_PRD_PROMPT
    assert "severity" in ANALYZE_PRD_PROMPT
    assert "suggested_action" in ANALYZE_PRD_PROMPT


def test_decompose_to_tickets_prompt_has_example():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT contains a few-shot example."""
    assert "## Example" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "epics" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "stories" in DECOMPOSE_TO_TICKETS_PROMPT
    # Example should show sizing
    assert '"size": "M"' in DECOMPOSE_TO_TICKETS_PROMPT or '"size": "S"' in DECOMPOSE_TO_TICKETS_PROMPT


def test_analyze_prd_prompt_has_xml_delimiters():
    """Verify ANALYZE_PRD_PROMPT wraps input with XML delimiters."""
    assert "<prd_document>" in ANALYZE_PRD_PROMPT
    assert "</prd_document>" in ANALYZE_PRD_PROMPT


def test_decompose_prompt_has_xml_delimiters():
    """Verify DECOMPOSE_TO_TICKETS_PROMPT wraps input with XML delimiters."""
    assert "<requirements_document>" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "</requirements_document>" in DECOMPOSE_TO_TICKETS_PROMPT


def test_prompts_handle_curly_braces_in_input():
    """Verify .format() doesn't choke on curly braces in user input."""
    tricky_input = 'The API returns {status: 200} and {"error": null}'
    formatted = ANALYZE_PRD_PROMPT.format(prd_text=tricky_input)
    assert tricky_input in formatted


def test_analyze_prd_prompt_schema_matches_model():
    """Verify all Requirement model fields appear in ANALYZE_PRD_PROMPT."""
    for field_name in Requirement.model_fields.keys():
        assert f'"{field_name}"' in ANALYZE_PRD_PROMPT, (
            f"Requirement field '{field_name}' not found in ANALYZE_PRD_PROMPT"
        )


def test_decompose_prompt_schema_matches_models():
    """Verify all Story and Epic model fields appear in DECOMPOSE_TO_TICKETS_PROMPT."""
    for field_name in Story.model_fields.keys():
        assert f'"{field_name}"' in DECOMPOSE_TO_TICKETS_PROMPT, (
            f"Story field '{field_name}' not found in DECOMPOSE_TO_TICKETS_PROMPT"
        )
    for field_name in Epic.model_fields.keys():
        assert f'"{field_name}"' in DECOMPOSE_TO_TICKETS_PROMPT, (
            f"Epic field '{field_name}' not found in DECOMPOSE_TO_TICKETS_PROMPT"
        )


def test_decompose_prompt_includes_agent_context_instructions():
    """Decomposition prompt instructs LLM to generate agent_context."""
    from prd_decomposer.prompts import DECOMPOSE_TO_TICKETS_PROMPT

    assert "agent_context" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "goal" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "exploration_paths" in DECOMPOSE_TO_TICKETS_PROMPT
    assert "self_check" in DECOMPOSE_TO_TICKETS_PROMPT


def test_decompose_prompt_example_includes_agent_context():
    """Decomposition prompt example shows agent_context usage."""
    from prd_decomposer.prompts import DECOMPOSE_TO_TICKETS_PROMPT

    # Example should demonstrate agent_context structure
    assert '"goal":' in DECOMPOSE_TO_TICKETS_PROMPT
    assert '"exploration_paths":' in DECOMPOSE_TO_TICKETS_PROMPT
    assert '"verification_tests":' in DECOMPOSE_TO_TICKETS_PROMPT
