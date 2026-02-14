"""Tests for prompt templates."""

from prd_decomposer.prompts import (
    ANALYZE_PRD_PROMPT,
    DECOMPOSE_TO_TICKETS_PROMPT,
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


def test_prompts_are_formattable():
    """Verify prompts can be formatted with expected variables."""
    formatted_analyze = ANALYZE_PRD_PROMPT.format(prd_text="Test PRD content")
    assert "Test PRD content" in formatted_analyze

    formatted_decompose = DECOMPOSE_TO_TICKETS_PROMPT.format(
        requirements_json='{"requirements": []}'
    )
    assert '{"requirements": []}' in formatted_decompose
