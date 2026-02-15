"""Agent that consumes the PRD Decomposer MCP server."""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from agents import Agent, Runner
from agents.items import TResponseInputItem
from agents.mcp import MCPServerStdio, MCPServerStdioParams

# Retry configuration for MCP server connection
MAX_CONNECTION_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRD Decomposer Agent - Convert PRDs into Jira tickets"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output for debugging"
    )
    return parser.parse_args()


async def connect_mcp_server_with_retry(
    server_path: Path,
    verbose: bool = False
) -> MCPServerStdio:
    """Connect to MCP server with retry logic and exponential backoff.

    Args:
        server_path: Path to the MCP server script
        verbose: Whether to print detailed connection info

    Returns:
        Connected MCPServerStdio instance

    Raises:
        ConnectionError: If all retry attempts fail
    """
    last_error: Exception | None = None
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
        try:
            if verbose:
                print(f"[DEBUG] Connection attempt {attempt}/{MAX_CONNECTION_RETRIES}...")

            mcp_server = MCPServerStdio(
                params=MCPServerStdioParams(
                    command="uv",
                    args=["run", "python", str(server_path)]
                ),
                client_session_timeout_seconds=120,  # LLM calls can take 30+ seconds
            )
            await mcp_server.__aenter__()

            if verbose:
                print("[DEBUG] MCP server connected successfully")

            return mcp_server

        except Exception as e:
            last_error = e
            if verbose:
                print(f"[DEBUG] Connection attempt {attempt} failed: {e}")

            if attempt < MAX_CONNECTION_RETRIES:
                print(f"Connection failed, retrying in {delay:.1f}s... ({attempt}/{MAX_CONNECTION_RETRIES})")
                await asyncio.sleep(delay)
                delay *= BACKOFF_MULTIPLIER

    raise ConnectionError(
        f"Failed to connect to MCP server after {MAX_CONNECTION_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


async def run_agent_turn(
    runner: Runner,
    agent: Agent,
    current_input: list[TResponseInputItem],
    verbose: bool = False
) -> str:
    """Run a single agent turn with timeout handling.

    Args:
        runner: The agent runner
        agent: The agent instance
        current_input: Conversation history + current message
        verbose: Whether to print detailed info

    Returns:
        The agent's response text

    Raises:
        asyncio.TimeoutError: If the request times out
        Exception: For other errors
    """
    try:
        # Wrap the run in a timeout - 90 seconds should be plenty
        result = await asyncio.wait_for(
            runner.run(agent, current_input),
            timeout=90.0
        )
        return result.final_output, result.to_input_list()

    except TimeoutError:
        raise TimeoutError(
            "Request timed out after 90 seconds. The LLM may be overloaded. "
            "Please try again."
        )


async def main() -> None:
    """Run the PRD Decomposer agent."""
    args = parse_args()
    verbose = args.verbose

    if verbose:
        print("[DEBUG] Verbose mode enabled")

    # Path to the MCP server
    server_path = Path(__file__).parent.parent / "src" / "prd_decomposer" / "server.py"

    if verbose:
        print(f"[DEBUG] MCP server path: {server_path}")

    # Connect to MCP server with retry logic
    try:
        mcp_server = await connect_mcp_server_with_retry(server_path, verbose)
    except ConnectionError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure OPENAI_API_KEY is set")
        print("  2. Run 'uv sync' to install dependencies")
        print("  3. Check that the server.py file exists")
        sys.exit(1)

    try:
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

                if verbose:
                    print(f"[DEBUG] Sending request with {len(conversation_history)} history items...")

                # Run with timeout handling
                print("Thinking...", end="", flush=True)
                final_output, new_history = await run_agent_turn(
                    runner, agent, current_input, verbose
                )
                print("\r" + " " * 12 + "\r", end="")  # Clear "Thinking..."

                print(f"\nAssistant: {final_output}\n")

                # Update history with this turn's complete input/output
                conversation_history = new_history

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

            except TimeoutError as e:
                print(f"\n\nTimeout: {e}\n")

            except asyncio.CancelledError:
                print("\n\nRequest cancelled.\n")

            except Exception as e:
                # Specific error handling with helpful messages
                error_msg = str(e)

                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    print("\n\nRate limited by OpenAI. Please wait a moment and try again.\n")
                elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    print("\n\nAuthentication error. Please check your OPENAI_API_KEY.\n")
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    print("\n\nNetwork error. Please check your internet connection.\n")
                else:
                    print(f"\n\nError: {e}\n")

                if verbose:
                    print("[DEBUG] Full traceback:")
                    traceback.print_exc()
                    print()

    finally:
        # Clean up MCP server connection
        try:
            await mcp_server.__aexit__(None, None, None)
            if verbose:
                print("[DEBUG] MCP server disconnected")
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    asyncio.run(main())
