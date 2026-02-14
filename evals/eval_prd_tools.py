"""Arcade eval suite for PRD Decomposer tools.

This eval suite validates:
1. Tool selection intent - does the LLM pick the right tool?
2. Parameter passing - are parameters correctly extracted?
3. Workflow understanding - does the LLM understand the two-step process?
"""

from pathlib import Path

from arcade_evals import (
    BinaryCritic,
    EvalSuite,
    ExpectedMCPToolCall,
    SimilarityCritic,
    tool_eval,
)


@tool_eval()
async def prd_eval_suite() -> EvalSuite:
    """Eval suite for PRD Decomposer MCP tools."""

    suite = EvalSuite(
        name="PRD Decomposer Tools",
        system_message="""You are a helpful engineering assistant that helps convert PRDs into Jira tickets.

You have access to these tools:
- read_file: Read a file from the filesystem
- analyze_prd: Analyze PRD text and extract structured requirements
- decompose_to_tickets: Convert requirements into Jira epics and stories

Workflow:
1. If given a file path, use read_file first
2. Use analyze_prd to extract requirements from PRD text
3. Use decompose_to_tickets to create Jira tickets from requirements""",
    )

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    await suite.add_mcp_stdio_server(command=["uv", "run", "python", str(server_path)])

    # =========================================================================
    # TOOL SELECTION EVALS
    # =========================================================================

    # Eval 1: Does the LLM select analyze_prd for analysis requests?
    suite.add_case(
        name="Analyze PRD intent - direct request",
        user_message="I have a PRD for a new feature. Can you analyze it and extract the requirements? Here's the PRD:\n\n# Feature: User Settings\n\nUsers should be able to update their email preferences.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={
                    "prd_text": "# Feature: User Settings\n\nUsers should be able to update their email preferences."
                },
            )
        ],
        critics=[BinaryCritic(critic_field="prd_text", weight=1.0)],
    )

    # Eval 2: Does the LLM select analyze_prd for implicit requests?
    suite.add_case(
        name="Analyze PRD intent - implicit request",
        user_message="What requirements are in this PRD?\n\n# API Versioning\n\nImplement API versioning with v1/v2 prefixes.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={
                    "prd_text": "# API Versioning\n\nImplement API versioning with v1/v2 prefixes."
                },
            )
        ],
        critics=[BinaryCritic(critic_field="prd_text", weight=1.0)],
    )

    # Eval 3: Does decompose_to_tickets get selected for ticket generation?
    sample_requirements = {
        "requirements": [
            {
                "id": "REQ-001",
                "title": "User login",
                "description": "Users can log in with email/password",
                "acceptance_criteria": ["Login form exists"],
                "dependencies": [],
                "ambiguity_flags": [],
                "priority": "high",
            }
        ],
        "summary": "Authentication feature",
        "source_hash": "abc12345",
    }

    suite.add_case(
        name="Decompose to tickets intent",
        user_message=f"Turn these requirements into Jira tickets: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets", parameters={"requirements": sample_requirements}
            )
        ],
        critics=[BinaryCritic(critic_field="requirements", weight=1.0)],
    )

    # Eval 4: Does the LLM understand "create stories" means decompose?
    suite.add_case(
        name="Decompose to tickets - alternative phrasing",
        user_message=f"Create Jira stories from these requirements: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets", parameters={"requirements": sample_requirements}
            )
        ],
        critics=[BinaryCritic(critic_field="requirements", weight=1.0)],
    )

    # =========================================================================
    # FILE READ WORKFLOW EVALS
    # =========================================================================

    # Eval 5: Does the LLM use read_file when given a file path?
    suite.add_case(
        name="Read file before analysis",
        user_message="Analyze the PRD in samples/sample_prd_01_rate_limiting.md",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="read_file",
                parameters={"file_path": "samples/sample_prd_01_rate_limiting.md"},
            )
        ],
        critics=[
            SimilarityCritic(
                critic_field="file_path",
                weight=1.0,
                similarity_threshold=0.8,  # Allow minor path variations
            )
        ],
    )

    # Eval 6: Does the LLM recognize .md files as PRDs?
    suite.add_case(
        name="Recognize markdown file as PRD",
        user_message="Read and analyze docs/feature-spec.md for me",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="read_file", parameters={"file_path": "docs/feature-spec.md"}
            )
        ],
        critics=[SimilarityCritic(critic_field="file_path", weight=1.0, similarity_threshold=0.8)],
    )

    # =========================================================================
    # CONTENT EXTRACTION QUALITY EVALS
    # =========================================================================

    # Eval 7: Does analyze_prd extract meaningful content?
    prd_with_clear_requirement = """# Feature: Password Reset

## Requirements

Users must be able to reset their password via email.

### Acceptance Criteria
- User receives reset email within 30 seconds
- Reset link expires after 1 hour
- Password must meet complexity requirements"""

    suite.add_case(
        name="Extract requirements from structured PRD",
        user_message=f"Extract the requirements from this PRD:\n\n{prd_with_clear_requirement}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd", parameters={"prd_text": prd_with_clear_requirement}
            )
        ],
        critics=[
            SimilarityCritic(
                critic_field="prd_text",
                weight=1.0,
                similarity_threshold=0.9,  # Should capture most of the PRD
            )
        ],
    )

    # Eval 8: Does the LLM handle vague language appropriately?
    vague_prd = """# Feature: Dashboard

The dashboard should be fast and user-friendly.
It needs to scale well as we grow."""

    suite.add_case(
        name="Handle vague PRD language",
        user_message=f"What requirements can you extract from this? Flag any ambiguities:\n\n{vague_prd}",
        expected_tool_calls=[
            ExpectedMCPToolCall(tool_name="analyze_prd", parameters={"prd_text": vague_prd})
        ],
        critics=[BinaryCritic(critic_field="prd_text", weight=1.0)],
    )

    return suite
