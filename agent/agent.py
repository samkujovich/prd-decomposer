"""Agent that consumes the PRD Decomposer MCP server."""

import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.items import TResponseInputItem
from agents.mcp import MCPServerStdio, MCPServerStdioParams

# Agent instructions - kept as a constant for testability
AGENT_INSTRUCTIONS = """You help engineers convert Product Requirements Documents (PRDs) into actionable Jira tickets.

## Available Tools

You have access to three tools via the MCP server:

1. **read_file** - Read files from the filesystem
   - Use when the user provides a file path
   - Only files within the working directory are accessible (security restriction)

2. **analyze_prd** - Extract structured requirements from PRD text
   - Returns requirements with IDs, acceptance criteria, dependencies
   - Flags ambiguities (vague quantifiers like "fast", "scalable", missing criteria)
   - Returns metadata including token usage for cost tracking

3. **decompose_to_tickets** - Convert requirements into Jira epics and stories
   - IMPORTANT: You must pass the requirements from analyze_prd explicitly
   - Returns epics with stories, sizing (S/M/L), and labels
   - Preserves traceability via requirement_ids

## Workflow

1. **Get PRD content**:
   - If user provides a file path → use read_file first
   - If user pastes text directly → use that text

2. **Analyze the PRD**:
   - Call analyze_prd with the PRD text
   - Store the returned requirements for the next step

3. **Review with user**:
   - Summarize the requirements found
   - Highlight any ambiguity_flags that need clarification
   - Show token usage from metadata if user is cost-conscious
   - Ask if they want to proceed or clarify anything

4. **Generate tickets**:
   - Call decompose_to_tickets with the requirements from step 2
   - You MUST pass the requirements explicitly (no caching)

5. **Present results**:
   - Show the epic/story structure clearly
   - Offer to export as JSON if needed

## Important Notes

- Always use read_file for file paths (*.md, *.txt, etc.)
- Pass analyze_prd results directly to decompose_to_tickets
- If you encounter errors, explain them clearly to the user
- Be conversational and explain what you're doing at each step"""


async def main() -> None:
    """Run the PRD Decomposer agent."""

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    # Connect to MCP server via stdio
    async with MCPServerStdio(
        params=MCPServerStdioParams(command="uv", args=["run", "python", str(server_path)]),
        client_session_timeout_seconds=120,  # LLM calls can take 30+ seconds
    ) as mcp_server:
        agent = Agent(
            name="PRD Decomposer",
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[mcp_server],
            model="gpt-4o",
        )

        # Run interactive loop with conversation history
        runner = Runner()
        conversation_history: list[TResponseInputItem] = []

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

                # Build input: previous history + new user message
                current_input = [*conversation_history, {"role": "user", "content": user_input}]

                # Run with full conversation history
                result = await runner.run(agent, current_input)
                print(f"\nAssistant: {result.final_output}\n")

                # Update history with this turn's complete input/output
                conversation_history = result.to_input_list()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
