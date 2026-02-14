"""Arcade eval suite for PRD Decomposer tools."""

from pathlib import Path

from arcade_evals import (
    BinaryCritic,
    EvalSuite,
    ExpectedMCPToolCall,
    tool_eval,
)


@tool_eval()
async def prd_eval_suite() -> EvalSuite:
    """Eval suite for PRD Decomposer MCP tools."""

    suite = EvalSuite(
        name="PRD Decomposer Tools",
        system_message="You are a helpful engineering assistant that helps convert PRDs into Jira tickets.",
    )

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    await suite.add_mcp_stdio_server(
        command=["uv", "run", "python", str(server_path)]
    )

    # Eval 1: Does the LLM select analyze_prd for analysis requests?
    suite.add_case(
        name="Analyze PRD intent - direct request",
        user_message="I have a PRD for a new feature. Can you analyze it and extract the requirements? Here's the PRD:\n\n# Feature: User Settings\n\nUsers should be able to update their email preferences.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={"prd_text": "# Feature: User Settings\n\nUsers should be able to update their email preferences."}
            )
        ],
        critics=[
            BinaryCritic(critic_field="prd_text", weight=1.0)
        ],
    )

    # Eval 2: Does the LLM select analyze_prd for implicit requests?
    suite.add_case(
        name="Analyze PRD intent - implicit request",
        user_message="What requirements are in this PRD?\n\n# API Versioning\n\nImplement API versioning with v1/v2 prefixes.",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="analyze_prd",
                parameters={"prd_text": "# API Versioning\n\nImplement API versioning with v1/v2 prefixes."}
            )
        ],
        critics=[
            BinaryCritic(critic_field="prd_text", weight=1.0)
        ],
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
                "priority": "high"
            }
        ],
        "summary": "Authentication feature",
        "source_hash": "abc12345"
    }

    suite.add_case(
        name="Decompose to tickets intent",
        user_message=f"Turn these requirements into Jira tickets: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets",
                parameters={"requirements": sample_requirements}
            )
        ],
        critics=[
            BinaryCritic(critic_field="requirements", weight=1.0)
        ],
    )

    # Eval 4: Does the LLM understand "create stories" means decompose?
    suite.add_case(
        name="Decompose to tickets - alternative phrasing",
        user_message=f"Create Jira stories from these requirements: {sample_requirements}",
        expected_tool_calls=[
            ExpectedMCPToolCall(
                tool_name="decompose_to_tickets",
                parameters={"requirements": sample_requirements}
            )
        ],
        critics=[
            BinaryCritic(critic_field="requirements", weight=1.0)
        ],
    )

    return suite
