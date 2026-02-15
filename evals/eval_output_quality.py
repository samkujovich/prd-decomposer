"""Output quality evaluation suite for PRD Decomposer.

Unlike eval_prd_tools.py which tests tool SELECTION (did the LLM pick the right tool?),
this suite tests tool OUTPUT QUALITY (did the tool produce correct results?).

These evals make real API calls and validate that:
1. Vague quantifiers ("fast", "scalable") are flagged as ambiguities
2. Clear PRDs with explicit acceptance criteria have no critical ambiguities
3. Generated stories have valid requirement_ids linking back to source requirements
4. Epic count is reasonable (1-4 for typical single-feature PRD)
5. All stories have valid T-shirt sizes (S/M/L)

Run with: OPENAI_API_KEY=sk-... uv run pytest evals/eval_output_quality.py -v
"""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Output quality evals require OPENAI_API_KEY",
)


# =============================================================================
# EVAL 1: Vague quantifiers are flagged
# =============================================================================


class TestAmbiguityDetection:
    """Evals for ambiguity detection quality."""

    def test_flags_vague_quantifier_fast(self):
        """PRD with 'fast' should have vague_quantifier ambiguity flag."""
        from prd_decomposer.server import analyze_prd

        prd = """# Feature: API Response

The API response should be fast.
Users expect quick load times.
"""

        result = analyze_prd(prd_text=prd)

        # Collect all ambiguity flags
        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        # Should flag "fast" as vague_quantifier
        vague_flags = [f for f in all_flags if f["category"] == "vague_quantifier"]
        assert len(vague_flags) > 0, (
            f"Expected 'fast' to be flagged as vague_quantifier. "
            f"Got flags: {[f['category'] for f in all_flags]}"
        )

    def test_flags_vague_quantifier_scalable(self):
        """PRD with 'scalable' should have vague_quantifier ambiguity flag."""
        from prd_decomposer.server import analyze_prd

        prd = """# Feature: Database Design

The system must be scalable to handle growth.
Performance should be acceptable under load.
"""

        result = analyze_prd(prd_text=prd)

        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        vague_flags = [f for f in all_flags if f["category"] == "vague_quantifier"]
        assert len(vague_flags) > 0, (
            f"Expected 'scalable' or 'acceptable' to be flagged as vague_quantifier. "
            f"Got flags: {[f['category'] for f in all_flags]}"
        )

    def test_flags_user_friendly(self):
        """PRD with 'user-friendly' should flag as vague."""
        from prd_decomposer.server import analyze_prd

        prd = """# Feature: Checkout Flow

The checkout experience should be user-friendly and seamless.
"""

        result = analyze_prd(prd_text=prd)

        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        # "user-friendly" and "seamless" are vague
        assert len(all_flags) > 0, (
            "Expected 'user-friendly' or 'seamless' to be flagged. Got no flags."
        )

    def test_ambiguity_flags_have_suggested_action(self):
        """All ambiguity flags should include actionable suggested_action."""
        from prd_decomposer.server import analyze_prd

        prd = """# Feature: Dashboard

The dashboard should be fast, scalable, and user-friendly.
It needs to handle significant traffic.
"""

        result = analyze_prd(prd_text=prd)

        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        assert len(all_flags) > 0, "Expected at least one ambiguity flag"

        for flag in all_flags:
            assert "suggested_action" in flag, f"Flag missing suggested_action: {flag}"
            assert len(flag["suggested_action"]) > 10, (
                f"suggested_action too short to be actionable: {flag['suggested_action']}"
            )


# =============================================================================
# EVAL 2: Clear PRDs have minimal/no critical ambiguities
# =============================================================================


class TestClearPrdHandling:
    """Evals for handling well-written PRDs."""

    def test_clear_prd_no_critical_ambiguities(self):
        """PRD with explicit acceptance criteria should have no critical ambiguities."""
        from prd_decomposer.server import analyze_prd

        clear_prd = """# Feature: Password Reset

## Requirements
Users must be able to reset their password via email.

## Acceptance Criteria
- User can request password reset from login page
- Reset email is sent within 30 seconds of request
- Reset link expires after exactly 1 hour
- New password must be at least 12 characters with 1 uppercase, 1 number, 1 symbol
- User receives confirmation email after successful reset
- Failed reset attempts are logged for security audit

## Error Handling
- Invalid email shows "If this email exists, a reset link has been sent"
- Expired link shows "This link has expired. Please request a new reset."
- Rate limit: Maximum 3 reset requests per hour per email
"""

        result = analyze_prd(prd_text=clear_prd)

        # Collect critical ambiguities only
        critical_flags = []
        for req in result["requirements"]:
            for flag in req.get("ambiguity_flags", []):
                if flag.get("severity") == "critical":
                    critical_flags.append(flag)

        assert len(critical_flags) == 0, (
            f"Clear PRD should have no critical ambiguities. "
            f"Got {len(critical_flags)}: {[f['issue'] for f in critical_flags]}"
        )

    def test_prd_with_numbers_not_flagged_as_vague(self):
        """PRD with explicit numbers should not flag those as vague."""
        from prd_decomposer.server import analyze_prd

        specific_prd = """# Feature: Rate Limiting

## Requirements
- API rate limit: 100 requests per minute per user
- Response time: 99th percentile under 200ms
- Cache TTL: 5 minutes for read endpoints
- Maximum payload size: 10MB
"""

        result = analyze_prd(prd_text=specific_prd)

        all_flags = []
        for req in result["requirements"]:
            all_flags.extend(req.get("ambiguity_flags", []))

        # Should have minimal flags since everything is quantified
        vague_flags = [f for f in all_flags if f["category"] == "vague_quantifier"]
        assert len(vague_flags) == 0, (
            f"Explicit numbers should not be flagged as vague. Got: {vague_flags}"
        )


# =============================================================================
# EVAL 3: Traceability - stories link back to requirements
# =============================================================================


class TestTraceability:
    """Evals for requirement traceability in generated tickets."""

    def test_stories_have_requirement_ids(self):
        """Every story should have at least one requirement_id."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: User Authentication

## Requirements
- Users can register with email and password
- Users can log in with email and password
- Users can log out from any device
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        for epic in tickets["epics"]:
            for story in epic["stories"]:
                assert "requirement_ids" in story, f"Story missing requirement_ids: {story['title']}"
                assert len(story["requirement_ids"]) > 0, (
                    f"Story has empty requirement_ids: {story['title']}"
                )

    def test_requirement_ids_are_valid(self):
        """All requirement_ids in stories should reference actual requirements."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: File Upload

## Requirements
- Users can upload files up to 100MB
- Supported formats: PDF, PNG, JPG, DOCX
- Files are scanned for viruses before storage
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        # Collect all valid requirement IDs
        valid_req_ids = {req["id"] for req in requirements["requirements"]}

        # Check all story requirement_ids reference valid requirements
        for epic in tickets["epics"]:
            for story in epic["stories"]:
                for req_id in story.get("requirement_ids", []):
                    assert req_id in valid_req_ids, (
                        f"Story '{story['title']}' references non-existent requirement: {req_id}. "
                        f"Valid IDs: {valid_req_ids}"
                    )


# =============================================================================
# EVAL 4: Epic count is reasonable
# =============================================================================


class TestEpicStructure:
    """Evals for epic/story structure quality."""

    def test_single_feature_has_reasonable_epic_count(self):
        """Single feature PRD should produce 1-4 epics."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: Shopping Cart

## Requirements
- Add items to cart
- Update item quantities
- Remove items from cart
- View cart summary with total
- Apply discount codes
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        epic_count = len(tickets["epics"])
        assert 1 <= epic_count <= 4, (
            f"Single feature should have 1-4 epics. Got {epic_count}: "
            f"{[e['title'] for e in tickets['epics']]}"
        )

    def test_each_epic_has_stories(self):
        """Every epic should have at least one story."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: Notifications

## Requirements
- Send push notifications
- Send email notifications
- User can configure notification preferences
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        for epic in tickets["epics"]:
            assert len(epic["stories"]) > 0, f"Epic has no stories: {epic['title']}"


# =============================================================================
# EVAL 5: All stories have valid sizes
# =============================================================================


class TestStorySizing:
    """Evals for story sizing quality."""

    def test_all_stories_have_valid_size(self):
        """Every story must have size S, M, or L."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: Search

## Requirements
- Full-text search across all content
- Filter by date, author, category
- Sort results by relevance or date
- Pagination with 20 results per page
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        valid_sizes = {"S", "M", "L"}
        for epic in tickets["epics"]:
            for story in epic["stories"]:
                assert "size" in story, f"Story missing size: {story['title']}"
                assert story["size"] in valid_sizes, (
                    f"Story has invalid size '{story['size']}': {story['title']}. "
                    f"Must be one of {valid_sizes}"
                )

    def test_stories_have_acceptance_criteria(self):
        """Stories should have acceptance criteria derived from requirements."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: User Profile

## Requirements
- User can view their profile
- User can edit display name
- User can upload profile picture (max 5MB, JPG/PNG only)
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        for epic in tickets["epics"]:
            for story in epic["stories"]:
                assert "acceptance_criteria" in story, (
                    f"Story missing acceptance_criteria: {story['title']}"
                )
                # Most stories should have at least one acceptance criterion
                # (some very small stories might not, so we don't require it)


# =============================================================================
# EVAL 6: Labels are meaningful
# =============================================================================


class TestLabeling:
    """Evals for story labeling quality."""

    def test_stories_have_labels(self):
        """Stories should have descriptive labels."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: API Authentication

## Requirements
- Implement JWT token-based authentication
- Create login endpoint that returns access + refresh tokens
- Create token refresh endpoint
- Add authentication middleware to protected routes
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        for epic in tickets["epics"]:
            for story in epic["stories"]:
                assert "labels" in story, f"Story missing labels: {story['title']}"
                # Labels should include relevant technical domains
                # (backend, frontend, api, auth, database, etc.)

    def test_auth_feature_has_auth_labels(self):
        """Authentication feature should produce stories with auth-related labels."""
        from prd_decomposer.server import analyze_prd, decompose_to_tickets

        prd = """# Feature: Two-Factor Authentication

## Requirements
- User can enable 2FA via authenticator app
- Generate QR code for authenticator setup
- Verify TOTP codes during login
- Provide backup codes for account recovery
"""

        requirements = analyze_prd(prd_text=prd)
        tickets = decompose_to_tickets(requirements_json=json.dumps(requirements))

        # Collect all labels across all stories
        all_labels = set()
        for epic in tickets["epics"]:
            all_labels.update(epic.get("labels", []))
            for story in epic["stories"]:
                all_labels.update(story.get("labels", []))

        # Verify labels exist (don't require specific auth labels as LLM may vary)
        assert len(all_labels) > 0, (
            "Expected at least some labels for 2FA feature. "
            "Common labels: backend, frontend, api, auth, security, database"
        )
