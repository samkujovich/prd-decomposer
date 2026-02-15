"""MCP server for PRD analysis and decomposition."""

import atexit
import hashlib
import json
import logging
import random
import signal
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from arcade_mcp_server import MCPApp
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError

from prd_decomposer.config import Settings, get_settings
from prd_decomposer.log import correlation_id
from prd_decomposer.models import StructuredRequirements, TicketCollection
from prd_decomposer.prompts import (
    ANALYZE_PRD_PROMPT,
    DECOMPOSE_TO_TICKETS_PROMPT,
    PROMPT_VERSION,
)

logger = logging.getLogger("prd_decomposer")

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

# Graceful shutdown handling
_shutdown_event = threading.Event()


def _shutdown_handler(signum: int, frame: object) -> None:
    """Handle shutdown signals gracefully."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s, initiating graceful shutdown...", sig_name)
    _shutdown_event.set()


def _cleanup() -> None:
    """Cleanup resources on exit."""
    try:
        logger.info("Cleaning up resources...")
    except (ValueError, OSError):
        # Log stream may be closed during interpreter shutdown
        pass
    # Reset global singletons
    global _client, _rate_limiter
    _client = None
    _rate_limiter = None


# Register signal handlers and cleanup
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(_cleanup)


def is_shutting_down() -> bool:
    """Check if server is shutting down. Use to abort long-running operations."""
    return _shutdown_event.is_set()


class LLMError(Exception):
    """Raised when LLM call fails after retries."""

    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class RateLimiter:
    """Thread-safe in-memory rate limiter using sliding window.

    Tracks call timestamps and rejects calls that exceed the configured
    rate limit within the time window.
    """

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def check_and_record(self) -> None:
        """Check rate limit and record a call.

        Raises:
            RateLimitExceededError: If rate limit is exceeded.
        """
        now = time.time()

        with self._lock:
            # Remove calls outside the window
            cutoff = now - self.window_seconds
            self._calls = [t for t in self._calls if t > cutoff]

            if len(self._calls) >= self.max_calls:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s. "
                    f"Try again in {self._calls[0] + self.window_seconds - now:.1f}s."
                )

            self._calls.append(now)

    def reset(self) -> None:
        """Reset the rate limiter (mainly for testing)."""
        with self._lock:
            self._calls = []


# Global rate limiter instance (initialized lazily)
_rate_limiter: RateLimiter | None = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    """Get or create the global RateLimiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                settings = settings or get_settings()
                _rate_limiter = RateLimiter(
                    max_calls=settings.rate_limit_calls,
                    window_seconds=settings.rate_limit_window,
                )
    return _rate_limiter


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

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"File {file_path} is not a valid UTF-8 text file. "
            "PRD files must be plain text (Markdown, TXT, etc.)."
        )


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
    rate_limiter: RateLimiter | None = None,
) -> tuple[dict, dict]:
    """Call LLM with exponential backoff retry.

    Returns:
        Tuple of (parsed_json_response, usage_metadata)

    Raises:
        LLMError: If all retries fail or response is invalid.
        RateLimitExceededError: If rate limit is exceeded.
    """
    client = client or get_client()
    settings = settings or get_settings()
    rate_limiter = rate_limiter or get_rate_limiter(settings)

    # Check rate limit before making call
    rate_limiter.check_and_record()

    last_error: Exception | None = None

    for attempt in range(settings.max_retries):
        # Check for shutdown before attempting LLM call
        if is_shutting_down():
            raise LLMError("Server is shutting down, aborting LLM call")

        try:
            response = client.chat.completions.create(  # type: ignore[call-overload]
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
                base_delay = settings.initial_retry_delay * (2**attempt)
                delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                logger.warning(
                    "Retrying LLM call (attempt %d/%d) after RateLimitError, delay=%.1fs",
                    attempt + 1, settings.max_retries, delay,
                )
                time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                base_delay = settings.initial_retry_delay * (2**attempt)
                delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                logger.warning(
                    "Retrying LLM call (attempt %d/%d) after APIConnectionError, delay=%.1fs",
                    attempt + 1, settings.max_retries, delay,
                )
                time.sleep(delay)

        except APITimeoutError as e:
            last_error = e
            if attempt < settings.max_retries - 1:
                base_delay = settings.initial_retry_delay * (2**attempt)
                delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                logger.warning(
                    "Retrying LLM call (attempt %d/%d) after APITimeoutError, delay=%.1fs",
                    attempt + 1, settings.max_retries, delay,
                )
                time.sleep(delay)

        except APIError as e:
            # Don't retry on 4xx errors (bad request, auth, etc.)
            status_code = getattr(e, "status_code", None)
            if status_code and 400 <= status_code < 500:
                raise LLMError(f"OpenAI API error: {e}")
            last_error = e
            if attempt < settings.max_retries - 1:
                base_delay = settings.initial_retry_delay * (2**attempt)
                delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                logger.warning(
                    "Retrying LLM call (attempt %d/%d) after APIError, delay=%.1fs",
                    attempt + 1, settings.max_retries, delay,
                )
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

    # Validate empty/whitespace-only input
    if not prd_text or not prd_text.strip():
        raise ValueError("PRD text cannot be empty.")

    # Set correlation ID for this request
    request_id = str(uuid.uuid4())[:8]
    correlation_id.set(request_id)

    logger.info("Starting PRD analysis, prd_length=%d", len(prd_text))

    # Validate input length (prompt injection mitigation)
    if len(prd_text) > settings.max_prd_length:
        raise ValueError(
            f"PRD text exceeds maximum length of {settings.max_prd_length} characters "
            f"(got {len(prd_text)}). Set PRD_MAX_PRD_LENGTH to increase limit."
        )

    # Generate source hash for traceability
    source_hash = hashlib.sha256(prd_text.encode()).hexdigest()[:8]

    # Call LLM with retry
    start_time = time.monotonic()
    try:
        data, usage = _call_llm_with_retry(
            messages=[{"role": "user", "content": ANALYZE_PRD_PROMPT.format(prd_text=prd_text)}],
            temperature=settings.analyze_temperature,
            client=client,
            settings=settings,
        )
    except LLMError as e:
        elapsed = time.monotonic() - start_time
        logger.error("PRD analysis failed after %.2fs: %s", elapsed, e)
        raise RuntimeError(f"Failed to analyze PRD: {e}")

    elapsed = time.monotonic() - start_time

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

    logger.info(
        "PRD analysis complete, requirement_count=%d, elapsed=%.2fs, tokens=%s",
        len(result["requirements"]), elapsed, usage.get("total_tokens", "N/A"),
    )

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
    requirements: dict | str,
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

    # Set correlation ID for this request
    request_id = str(uuid.uuid4())[:8]
    correlation_id.set(request_id)

    logger.info("Starting ticket decomposition")

    # Check for None or empty input first
    if requirements is None or requirements == "":
        raise ValueError(
            "Requirements cannot be empty. Pass the JSON output from analyze_prd as a string."
        )

    # Handle case where requirements is passed as string (expected for MCP)
    if isinstance(requirements, str):
        try:
            parsed = json.loads(requirements)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in requirements: {e}")
        if not isinstance(parsed, dict):
            raise ValueError("Requirements JSON must be an object, not an array or primitive")
        requirements = parsed

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
    start_time = time.monotonic()
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
        elapsed = time.monotonic() - start_time
        logger.error("Ticket decomposition failed after %.2fs: %s", elapsed, e)
        raise RuntimeError(f"Failed to decompose requirements: {e}")

    elapsed = time.monotonic() - start_time

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

    logger.info(
        "Ticket decomposition complete, epic_count=%d, story_count=%d, elapsed=%.2fs, tokens=%s",
        len(data.get("epics", [])), story_count, elapsed, usage.get("total_tokens", "N/A"),
    )

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


@app.tool
def health_check() -> dict:
    """Check service health and OpenAI API connectivity.

    Returns status information including:
    - Service status (healthy/degraded/unhealthy)
    - OpenAI API connectivity
    - Configuration summary
    - Rate limiter status

    Use this to verify the service is operational before processing PRDs.
    """
    settings = get_settings()
    status = "healthy"
    checks: dict[str, dict] = {}

    # Check OpenAI API connectivity
    try:
        client = get_client()
        # Make a minimal API call to verify connectivity
        response = client.models.list()
        checks["openai_api"] = {
            "status": "connected",
            "models_available": len(response.data) if response.data else 0,
        }
    except Exception as e:
        status = "unhealthy"
        checks["openai_api"] = {
            "status": "error",
            "error": str(e),
        }

    # Check rate limiter status
    try:
        rate_limiter = get_rate_limiter(settings)
        with rate_limiter._lock:
            current_calls = len(rate_limiter._calls)
        checks["rate_limiter"] = {
            "status": "ok",
            "current_calls": current_calls,
            "max_calls": rate_limiter.max_calls,
            "window_seconds": rate_limiter.window_seconds,
        }
    except Exception as e:
        status = "degraded" if status == "healthy" else status
        checks["rate_limiter"] = {
            "status": "error",
            "error": str(e),
        }

    return {
        "status": status,
        "checks": checks,
        "config": {
            "model": settings.openai_model,
            "max_retries": settings.max_retries,
            "llm_timeout": settings.llm_timeout,
            "max_prd_length": settings.max_prd_length,
        },
        "version": PROMPT_VERSION,
    }


if __name__ == "__main__":
    app.run()
