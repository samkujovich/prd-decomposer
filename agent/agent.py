"""Agent that consumes the PRD Decomposer MCP server."""

import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams


async def main() -> None:
    """Run the PRD Decomposer agent."""

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    # Connect to MCP server via stdio
    async with MCPServerStdio(
        params=MCPServerStdioParams(
            command="uv",
            args=["run", "python", str(server_path)]
        ),
        client_session_timeout_seconds=120  # LLM calls can take 30+ seconds
    ) as mcp_server:

        agent = Agent(
            name="PRD Decomposer",
            instructions="""You help engineers convert Product Requirements Documents (PRDs) into actionable Jira tickets.

Your workflow:
1. Get the PRD content:
   - If the user provides a file path, use the read_file tool to read it first
   - If they paste text directly, use that
2. Use the analyze_prd tool to extract structured requirements
3. Review the results with the user:
   - Summarize the requirements found
   - Highlight any ambiguity flags that need clarification
   - Ask if they want to proceed or clarify anything first
4. Once confirmed, use decompose_to_tickets to generate Jira-ready epics and stories
5. Present the ticket structure in a clear format
6. Offer to export as JSON if needed

IMPORTANT: When a user mentions a file path (like "samples/sample_prd.md" or any .md/.txt file), ALWAYS use the read_file tool first to get its contents, then pass that content to analyze_prd.

Be conversational and helpful. Explain what you're doing at each step. If the PRD has ambiguities, help the user understand what additional information would improve the tickets.""",
            mcp_servers=[mcp_server],
            model="gpt-4o",
        )

        # Run interactive loop
        runner = Runner()
        print("PRD Decomposer Agent")
        print("=" * 40)
        print("I help convert PRDs into Jira tickets.")
        print("Paste your PRD or provide a file path to get started.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                result = await runner.run(agent, user_input)
                print(f"\nAssistant: {result.final_output}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
