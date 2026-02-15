"""Agent that consumes the PRD Decomposer MCP server."""

import argparse
import asyncio
import json
import re
import sys
import traceback
from pathlib import Path

from agents import Agent, Runner
from agents.items import TResponseInputItem
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from agent.session_state import SessionState

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


def parse_command(user_input: str) -> tuple[str | None, int | None, str | None]:
    """Parse special commands from user input.

    Returns:
        Tuple of (command, index, argument) where:
        - command: "accept", "dismiss", "clarify", "tickets", "ambiguities", or None
        - index: 1-based index for accept/dismiss/clarify, or None
        - argument: clarification text for "clarify", or None
    """
    stripped = user_input.strip().lower()

    # Simple commands
    if stripped in ("tickets", "ticket", "decompose"):
        return ("tickets", None, None)
    if stripped in ("ambiguities", "ambigs", "status"):
        return ("ambiguities", None, None)

    # accept N
    match = re.match(r"accept\s+(\d+)", stripped)
    if match:
        return ("accept", int(match.group(1)), None)

    # dismiss N
    match = re.match(r"dismiss\s+(\d+)", stripped)
    if match:
        return ("dismiss", int(match.group(1)), None)

    # clarify N "text" or clarify N text (case-insensitive command, preserve text)
    original = user_input.strip()
    match = re.match(r'clarify\s+(\d+)\s+"([^"]+)"', original, re.IGNORECASE)
    if match:
        return ("clarify", int(match.group(1)), match.group(2))
    match = re.match(r"clarify\s+(\d+)\s+(.+)", original, re.IGNORECASE)
    if match:
        return ("clarify", int(match.group(1)), match.group(2))

    return (None, None, None)


def handle_command(
    command: str,
    index: int | None,
    argument: str | None,
    session: SessionState,
) -> str:
    """Handle a special command and return the response to display.

    Args:
        command: The command name
        index: Optional 1-based index for the ambiguity
        argument: Optional argument (clarification text)
        session: The session state

    Returns:
        Response text to display to the user
    """
    if command == "ambiguities":
        return session.format_ambiguities_display()

    if command == "accept":
        if index is None:
            return "Usage: accept [n] - specify which ambiguity to accept"
        amb_id = session.accept_ambiguity(index)
        if amb_id:
            remaining = len(session.get_active_ambiguities())
            noun = "ambiguity" if remaining == 1 else "ambiguities"
            return f"Accepted ambiguity #{index}. {remaining} {noun} remaining."
        return f"Invalid index: {index}. Use 'ambiguities' to see the list."

    if command == "dismiss":
        if index is None:
            return "Usage: dismiss [n] - specify which ambiguity to dismiss"
        amb_id = session.dismiss_ambiguity(index)
        if amb_id:
            remaining = len(session.get_active_ambiguities())
            noun = "ambiguity" if remaining == 1 else "ambiguities"
            return f"Dismissed ambiguity #{index}. {remaining} {noun} remaining."
        return f"Invalid index: {index}. Use 'ambiguities' to see the list."

    if command == "clarify":
        if index is None or argument is None:
            return 'Usage: clarify [n] "clarification text"'
        req_id = session.add_clarification(index, argument)
        if req_id:
            remaining = len(session.get_active_ambiguities())
            noun = "ambiguity" if remaining == 1 else "ambiguities"
            return f"Added clarification to {req_id}. {remaining} {noun} remaining."
        return f"Invalid index: {index}. Use 'ambiguities' to see the list."

    return f"Unknown command: {command}"


def extract_requirements_from_output(output: str) -> dict | None:
    """Try to extract analyze_prd JSON result from agent output.

    The agent often includes the JSON in its response. This function
    attempts to find and parse it using brace-matching for robustness.
    """
    # Strategy 1: Look for JSON in markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*(\{)", re.IGNORECASE)
    for match in fence_pattern.finditer(output):
        start = match.start(1)
        json_str = _extract_balanced_braces(output, start)
        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "requirements" in data:
                    return data
            except json.JSONDecodeError:
                continue

    # Strategy 2: Look for {"requirements": anywhere in text
    req_pattern = re.compile(r'(\{"requirements"\s*:)', re.IGNORECASE)
    for match in req_pattern.finditer(output):
        start = match.start(1)
        json_str = _extract_balanced_braces(output, start)
        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "requirements" in data:
                    return data
            except json.JSONDecodeError:
                continue

    return None


def _extract_balanced_braces(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at the given position.

    Args:
        text: The full text to extract from
        start: Position of the opening brace

    Returns:
        The extracted JSON string, or None if braces don't balance
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # Unbalanced braces


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

        # Session state for tracking requirements and user decisions
        session = SessionState()

        print("PRD Decomposer Agent")
        print("=" * 40)
        print("I help convert PRDs into Jira tickets.")
        print("Paste your PRD or provide a file path to get started.")
        print("\nCommands: accept [n], dismiss [n], clarify [n] \"text\", tickets, ambiguities")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                # Check for special commands first
                command, index, argument = parse_command(user_input)

                if command == "tickets":
                    # Special handling: generate tickets with clarifications
                    if not session.current_requirements:
                        print("\nNo requirements loaded. Analyze a PRD first.\n")
                        continue

                    # Get requirements with clarifications injected
                    clarified = session.get_requirements_with_clarifications()
                    clarifications_note = ""
                    if session.clarifications:
                        clarifications_note = (
                            "\n\nNote: The following clarifications have been added:\n"
                            + "\n".join(
                                f"- {req_id}: {text}"
                                for req_id, text in session.clarifications.items()
                            )
                        )

                    # Send request to decompose with clarifications
                    decompose_request = (
                        f"Generate Jira tickets from these requirements. "
                        f"Call decompose_to_tickets with this JSON:\n\n"
                        f"```json\n{json.dumps(clarified)}\n```"
                        f"{clarifications_note}"
                    )
                    user_input = decompose_request

                elif command in ("accept", "dismiss", "clarify", "ambiguities"):
                    # Handle locally without LLM call
                    response = handle_command(command, index, argument, session)
                    print(f"\n{response}\n")
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

                # Try to extract and store requirements from analyze_prd results
                extracted = extract_requirements_from_output(final_output)
                if extracted:
                    session.store_requirements(extracted)
                    if verbose:
                        print(f"[DEBUG] Stored {len(extracted.get('requirements', []))} requirements")

                    # Show ambiguity summary if any
                    ambiguities = session.get_active_ambiguities()
                    if ambiguities:
                        print(f"\nAssistant: {final_output}")
                        print("\n" + "=" * 40)
                        print(session.format_ambiguities_display())
                        print("=" * 40 + "\n")
                    else:
                        print(f"\nAssistant: {final_output}\n")
                else:
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
