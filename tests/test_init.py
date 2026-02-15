"""Tests for prd_decomposer package __init__.py exports (QA-10)."""

import prd_decomposer

EXPECTED_EXPORTS = {
    "ANALYZE_PRD_PROMPT",
    "DECOMPOSE_TO_TICKETS_PROMPT",
    "PROMPT_VERSION",
    "Epic",
    "Requirement",
    "Settings",
    "Story",
    "StructuredRequirements",
    "TicketCollection",
    "get_settings",
}


def test_all_exports_match_expected():
    """Verify __all__ contains the exact expected set of names."""
    assert set(prd_decomposer.__all__) == EXPECTED_EXPORTS


def test_server_tools_not_in_all():
    """Verify server tools are intentionally excluded from __all__."""
    excluded = {"app", "analyze_prd", "decompose_to_tickets", "read_file"}
    for name in excluded:
        assert name not in prd_decomposer.__all__, (
            f"Server tool '{name}' should not be in __all__"
        )


def test_all_exports_are_importable():
    """Verify every name in __all__ is actually importable from the package."""
    for name in prd_decomposer.__all__:
        assert hasattr(prd_decomposer, name), (
            f"'{name}' is in __all__ but not importable from prd_decomposer"
        )
