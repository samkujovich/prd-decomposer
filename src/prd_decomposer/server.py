"""MCP server for PRD analysis and decomposition."""

import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from arcade_mcp_server import MCPApp
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from prd_decomposer.config import Settings, get_settings
from prd_decomposer.models import StructuredRequirements, TicketCollection
from prd_decomposer.prompts import (
    ANALYZE_PRD_PROMPT,
    DECOMPOSE_TO_TICKETS_PROMPT,
    PROMPT_VERSION,
)

app = MCPApp(name="prd_decomposer", version="1.0.0")

# Lazy client initialization to avoid requiring API key at import time
# Thread-safe with double-checked locking
_client = None
_client_lock = threading.Lock()

# Allowed directories for file access (security: prevent path traversal)
# Defaults to current working directory, can be extended via environment
# Resolved to handle symlinked workspaces correctly
ALLOWED_DIRECTORIES: list[Path] = [
    Path.cwd().resolve(),
]


class LLMError(Exception):
    """Raised when LLM call fails after retries."""

    pass


def _is_path_allowed(path: Path) -> bool:
    """Check if a path is within allowed directories.

    Resolves symlinks and checks against allowed directory list.
    """
    try:
        resolved = path.resolve()
        return any(
            resolved == allowed or allowed in resolved.parents for allowed in ALLOWED_DIRECTORIES
        )
    except (OSError, ValueError):
        return False


@app.tool
def read_file(
    file_path: Annotated[str, "Path to the file to read (relative to working directory)"],
) -> str:
    """Read a file from the filesystem and return its contents.

    Use this to read PRD files before passing their content to analyze_prd.
    Files must be within the working directory for security.
    """
    path = Path(file_path)

    # Security: validate path is within allowed directories
    if not _is_path_allowed(path):
        raise PermissionError(
            f"Access denied: {file_path} is outside allowed directories. "
            f"Files must be within: {[str(d) for d in ALLOWED_DIRECTORIES]}"
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return path.read_text(encoding="utf-8")


def get_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI()  # Uses OPENAI_API_KEY env var
    return _client


def _call_llm_with_retry(
    messages: list[dict],
    temperature: float,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> tuple[dict, dict]:
    """Call LLM with exponential backoff retry.

    Returns:
        Tuple of (parsed_json_response, usage_metadata)

    Raises:
        LLMError: If all retries fail or response is invalid.
    """
    client = client or get_client()
    settings = settings or get_settings()

    last_error = None

    for attempt in range(settings.max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=settings.llm_timeout,
            )

            # Extract content
            content = response.choices[0].message.content
            if not content:
                raise LLMError("LLM returned empty response")

            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(f"LLM returned invalid JSON: {e}")

            # Extract usage for cost tracking
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return data, usage

        except RateLimitError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

        except APITimeoutError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

        except APIError as e:
            # Don't retry on 4xx errors (bad request, auth, etc.)
            status_code = getattr(e, "status_code", None)
            if status_code and 400 <= status_code < 500:
                raise LLMError(f"OpenAI API error: {e}")
            last_error = e
            if attempt < settings.max_retries - 1:
                delay = settings.initial_retry_delay * (2**attempt)
                time.sleep(delay)

    raise LLMError(f"LLM call failed after {settings.max_retries} retries: {last_error}")


def _analyze_prd_impl(
    prd_text: str,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    """Internal implementation of analyze_prd with DI support.

    Extracts requirements with IDs, acceptance criteria, dependencies,
    and flags ambiguous requirements (missing criteria or vague quantifiers).

    Returns structured requirements with metadata including token usage.
    """
    client = client or get_client()
    settings = settings or get_settings()

    # Generate source hash for traceability
    source_hash = hashlib.sha256(prd_text.encode()).hexdigest()[:8]

    # Call LLM with retry
    try:
        data, usage = _call_llm_with_retry(
            messages=[{"role": "user", "content": ANALYZE_PRD_PROMPT.format(prd_text=prd_text)}],
            temperature=settings.analyze_temperature,
            client=client,
            settings=settings,
        )
    except LLMError as e:
        raise RuntimeError(f"Failed to analyze PRD: {e}")

    # Ensure source_hash is set
    data["source_hash"] = source_hash

    # Validate with Pydantic
    try:
        validated = StructuredRequirements(**data)
    except ValidationError as e:
        raise RuntimeError(f"LLM returned invalid structure: {e}")

    result = validated.model_dump()

    # Add metadata for observability
    result["_metadata"] = {
        "prompt_version": PROMPT_VERSION,
        "model": settings.openai_model,
        "usage": usage,
        "analyzed_at": datetime.now(UTC).isoformat(),
    }

    return result


@app.tool
def analyze_prd(prd_text: Annotated[str, "Raw PRD markdown text to analyze"]) -> dict:
    """Analyze a PRD and extract structured requirements.

    Extracts requirements with IDs, acceptance criteria, dependencies,
    and flags ambiguous requirements (missing criteria or vague quantifiers).

    Returns structured requirements with metadata including token usage.
    """
    return _analyze_prd_impl(prd_text)


def _decompose_to_tickets_impl(
    requirements: dict,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> dict:
    """Internal implementation of decompose_to_tickets with DI support.

    Converts structured requirements into Jira-compatible epics and stories.
    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels.

    Returns ticket collection with metadata including token usage.
    """
    client = client or get_client()
    settings = settings or get_settings()

    # Check for None or empty input first
    if requirements is None or requirements == "":
        raise ValueError(
            "Requirements cannot be empty. Pass the JSON output from analyze_prd as a string."
        )

    # Handle case where requirements is passed as string (expected for MCP)
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in requirements: {e}")

    if not requirements:
        raise ValueError("Requirements cannot be empty. Run analyze_prd first.")

    # Strip internal metadata before validation
    requirements_clean = {k: v for k, v in requirements.items() if not k.startswith("_")}

    # Validate input
    try:
        validated_input = StructuredRequirements(**requirements_clean)
    except ValidationError as e:
        raise ValueError(f"Invalid requirements structure: {e}")

    # Call LLM with retry
    try:
        data, usage = _call_llm_with_retry(
            messages=[
                {
                    "role": "user",
                    "content": DECOMPOSE_TO_TICKETS_PROMPT.format(
                        requirements_json=validated_input.model_dump_json(indent=2)
                    ),
                }
            ],
            temperature=settings.decompose_temperature,
            client=client,
            settings=settings,
        )
    except LLMError as e:
        raise RuntimeError(f"Failed to decompose requirements: {e}")

    # Add metadata if not present
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["generated_at"] = datetime.now(UTC).isoformat()
    data["metadata"]["model"] = settings.openai_model
    data["metadata"]["prompt_version"] = PROMPT_VERSION
    data["metadata"]["requirement_count"] = len(validated_input.requirements)
    data["metadata"]["usage"] = usage

    # Count stories
    story_count = sum(len(epic.get("stories", [])) for epic in data.get("epics", []))
    data["metadata"]["story_count"] = story_count

    # Validate with Pydantic
    try:
        validated = TicketCollection(**data)
    except ValidationError as e:
        raise RuntimeError(f"LLM returned invalid ticket structure: {e}")

    return validated.model_dump()


@app.tool
def decompose_to_tickets(
    requirements_json: Annotated[str, "JSON string of the structured requirements from analyze_prd"],
) -> dict:
    """Convert structured requirements into Jira-compatible epics and stories.

    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels. Output is ready for Jira import.

    Pass the complete JSON output from analyze_prd as a string.
    """
    return _decompose_to_tickets_impl(requirements_json)


if __name__ == "__main__":
    app.run()
