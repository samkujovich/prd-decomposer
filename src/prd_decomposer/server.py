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

from prd_decomposer.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RateLimiter,
    RateLimitExceededError,
)
from prd_decomposer.config import Settings, get_settings

# Re-export for backwards compatibility - used by tests
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "RateLimitExceededError",
    "RateLimiter",
]
from prd_decomposer.export import export_tickets as _export_tickets_impl
from prd_decomposer.log import correlation_id, setup_logging
from prd_decomposer.models import SizingRubric, StructuredRequirements, TicketCollection
from prd_decomposer.prompts import (
    ANALYZE_PRD_PROMPT,
    DECOMPOSE_TO_TICKETS_PROMPT,
    DEFAULT_SIZING_RUBRIC,
    PROMPT_VERSION,
)

# Lazy logging initialization to avoid freezing env config at import time
_logging_initialized = False
_logging_lock = threading.Lock()


def _ensure_logging_initialized() -> None:
    """Initialize logging lazily on first use."""
    global _logging_initialized
    if _logging_initialized:
        return
    with _logging_lock:
        if not _logging_initialized:
            setup_logging(get_settings())
            _logging_initialized = True


# Get logger (lazy init happens on first log call via ensure function)
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
    # Reset global singletons - no logging here as stream may be closed
    global _client, _rate_limiter, _circuit_breaker, _logging_initialized
    _client = None
    _rate_limiter = None
    _circuit_breaker = None
    _logging_initialized = False  # Reset so logging can be re-initialized


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


# Global circuit breaker instance (initialized lazily)
_circuit_breaker: CircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker(settings: Settings | None = None) -> CircuitBreaker:
    """Get or create the global CircuitBreaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        with _circuit_breaker_lock:
            if _circuit_breaker is None:
                settings = settings or get_settings()
                _circuit_breaker = CircuitBreaker(
                    failure_threshold=settings.circuit_breaker_failure_threshold,
                    reset_timeout=settings.circuit_breaker_reset_timeout,
                )
    return _circuit_breaker


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


# Retryable errors tuple for DRY exception handling
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def _call_llm_with_retry(
    messages: list[dict],
    temperature: float,
    client: OpenAI | None = None,
    settings: Settings | None = None,
    rate_limiter: RateLimiter | None = None,
    circuit_breaker: CircuitBreaker | None = None,
) -> tuple[dict, dict]:
    """Call LLM with exponential backoff retry and circuit breaker.

    Returns:
        Tuple of (parsed_json_response, usage_metadata)

    Note: Initializes logging lazily on first call.

    Raises:
        LLMError: If all retries fail or response is invalid.
        RateLimitExceededError: If rate limit is exceeded.
        CircuitBreakerOpenError: If circuit breaker is open.
    """
    # Initialize logging lazily (avoids import-time side effects)
    _ensure_logging_initialized()

    client = client or get_client()
    settings = settings or get_settings()
    rate_limiter = rate_limiter or get_rate_limiter(settings)
    circuit_breaker = circuit_breaker or get_circuit_breaker(settings)

    # Check circuit breaker and get observed state atomically
    observed_state = circuit_breaker.allow_request()
    is_half_open_probe = observed_state == "half_open"

    # Track whether we've properly closed the circuit breaker state
    cb_resolved = False
    # Track whether we actually attempted an LLM call (vs failing on rate limit)
    llm_call_attempted = False
    # Track whether error is client-side (4xx) vs upstream failure
    is_client_error = False

    try:
        # Check rate limit before making call
        rate_limiter.check_and_record()

        last_error: Exception | None = None

        # In half-open state, limit to 1 attempt (no retries) to minimize load
        max_attempts = 1 if is_half_open_probe else settings.max_retries

        for attempt in range(max_attempts):
            # Check for shutdown BEFORE marking call as attempted
            # (shutdown aborts should not trip circuit breaker)
            if is_shutting_down():
                raise LLMError("Server is shutting down, aborting LLM call")

            llm_call_attempted = True  # Mark that we're attempting an LLM call

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

                # Record success with circuit breaker
                circuit_breaker.record_success()
                cb_resolved = True

                return data, usage

            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                    logger.warning(
                        "Retrying LLM call (attempt %d/%d) after %s, delay=%.1fs",
                        attempt + 1, max_attempts, type(e).__name__, delay,
                    )
                    time.sleep(delay)

            except APIError as e:
                # Check for 4xx client errors (don't retry, don't count as failure)
                status_code = getattr(e, "status_code", None)
                if status_code and 400 <= status_code < 500:
                    is_client_error = True
                    raise LLMError(f"OpenAI API error: {e}")

                # 5xx or unknown - treat as retryable
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())
                    logger.warning(
                        "Retrying LLM call (attempt %d/%d) after APIError, delay=%.1fs",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)

        # Record failure with circuit breaker (after all retries exhausted)
        circuit_breaker.record_failure()
        cb_resolved = True

        raise LLMError(f"LLM call failed after {max_attempts} attempts: {last_error}")

    finally:
        # Handle circuit breaker state if we exit without resolving
        if not cb_resolved:
            if llm_call_attempted and not is_client_error:
                # LLM call failed (transient upstream error) - count as failure
                # This ensures half-open probes that fail trigger reopening
                circuit_breaker.record_failure()
            elif is_client_error:
                # 4xx error: upstream IS responsive (just rejected our request)
                # In half-open: close the circuit since service is available
                # In closed: reset failure count so 5xx/4xx/5xx isn't 3 consecutive failures
                circuit_breaker.record_success()
            else:
                # Failed before LLM call (rate limit) - just release slot
                circuit_breaker.release_half_open_slot()


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

    # Validate input is not empty
    if not prd_text or not prd_text.strip():
        raise ValueError("PRD text cannot be empty or whitespace only")

    # Set correlation ID for this request
    request_id = str(uuid.uuid4())[:8]
    correlation_id.set(request_id)

    logger.info("Starting PRD analysis, prd_length=%d", len(prd_text))
    start_time = time.time()

    # Validate input length (security: prevent resource exhaustion)
    if len(prd_text) > settings.max_prd_length:
        raise ValueError(
            f"PRD text too long: {len(prd_text)} chars exceeds limit of {settings.max_prd_length}. "
            "Consider splitting into smaller documents."
        )

    # Generate source hash for traceability
    source_hash = hashlib.sha256(prd_text.encode()).hexdigest()[:16]

    messages = [
        {
            "role": "system",
            "content": ANALYZE_PRD_PROMPT.format(prd_text=prd_text),
        },
        {"role": "user", "content": f"Analyze the PRD above. Use source_hash: {source_hash}"},
    ]

    try:
        data, usage = _call_llm_with_retry(
            messages=messages, temperature=settings.analyze_temperature, client=client, settings=settings
        )
    except LLMError:
        logger.error("PRD analysis failed")
        raise

    # Validate response against Pydantic model
    try:
        validated = StructuredRequirements(**data)
    except ValidationError as e:
        logger.error("PRD analysis failed: invalid LLM response")
        raise LLMError(f"LLM returned invalid structure: {e}")

    elapsed = time.time() - start_time
    logger.info(
        "PRD analysis complete, requirements=%d, elapsed=%.2fs, tokens=%s",
        len(validated.requirements), elapsed, usage.get("total_tokens", "N/A"),
    )

    # Add metadata and override source_hash with computed value
    result = validated.model_dump()
    result["source_hash"] = source_hash[:8]  # Use computed hash, truncated to 8 chars
    result["_metadata"] = {
        "analyzed_at": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": settings.openai_model,
        **usage,
    }

    return result


@app.tool
def analyze_prd(
    prd_text: Annotated[str, "Raw PRD markdown/text content to analyze"],
) -> dict:
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
    sizing_rubric: str | SizingRubric | None = None,
) -> dict:
    """Internal implementation of decompose_to_tickets with DI support.

    Converts structured requirements into Jira-compatible epics and stories.
    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels.

    Args:
        requirements: Structured requirements JSON from analyze_prd
        client: OpenAI client (optional, for DI)
        settings: Settings instance (optional, for DI)
        sizing_rubric: Custom sizing rubric - can be a SizingRubric model or pre-formatted string

    Returns ticket collection with metadata including token usage.
    """
    client = client or get_client()
    settings = settings or get_settings()

    # Resolve sizing rubric
    if sizing_rubric is None:
        rubric_text = DEFAULT_SIZING_RUBRIC
    elif isinstance(sizing_rubric, SizingRubric):
        rubric_text = sizing_rubric.to_prompt_text()
    else:
        rubric_text = sizing_rubric

    # Set correlation ID for this request
    request_id = str(uuid.uuid4())[:8]
    correlation_id.set(request_id)

    logger.info("Starting ticket decomposition")

    # Check for None, empty string, or empty dict
    if requirements is None or requirements == "" or requirements == {}:
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

    # Validate input length (security: prevent resource exhaustion)
    requirements_json = json.dumps(requirements)
    if len(requirements_json) > settings.max_prd_length:
        raise ValueError(
            f"Requirements too long: {len(requirements_json)} chars exceeds limit of {settings.max_prd_length}. "
            "Consider processing in smaller batches."
        )

    start_time = time.time()

    messages = [
        {
            "role": "system",
            "content": DECOMPOSE_TO_TICKETS_PROMPT.format(
                sizing_rubric=rubric_text, requirements_json=requirements_json
            ),
        },
        {"role": "user", "content": "Process the requirements above and generate tickets."},
    ]

    data, usage = _call_llm_with_retry(
        messages=messages, temperature=settings.decompose_temperature, client=client, settings=settings
    )

    # Validate response against Pydantic model
    try:
        validated = TicketCollection(**data)
    except ValidationError as e:
        raise LLMError(f"LLM returned invalid structure: {e}")

    elapsed = time.time() - start_time
    story_count = sum(len(epic.stories) for epic in validated.epics)

    logger.info(
        "Ticket decomposition complete, epic_count=%d, story_count=%d, elapsed=%.2fs, tokens=%s",
        len(validated.epics), story_count, elapsed, usage.get("total_tokens", "N/A"),
    )

    # Count requirements from input
    requirement_count = len(requirements.get("requirements", []))

    # Add metadata to result
    result = validated.model_dump()
    result["metadata"] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": settings.openai_model,
        "requirement_count": requirement_count,
        "story_count": story_count,
        "usage": usage,
    }

    return result


@app.tool
def decompose_to_tickets(
    requirements_json: Annotated[str, "JSON string of the structured requirements from analyze_prd"],
    sizing_rubric: Annotated[
        str | None,
        "Optional custom sizing rubric JSON. Format: "
        '{"small": {"duration": "...", "scope": "...", "risk": "..."}, "medium": {...}, "large": {...}}'
    ] = None,
) -> dict:
    """Convert structured requirements into Jira-compatible epics and stories.

    Produces epics with child stories, acceptance criteria, t-shirt sizing (S/M/L),
    and labels. Output is ready for Jira import.

    Pass the complete JSON output from analyze_prd as a string.

    Optionally provide a custom sizing_rubric to customize how stories are sized.
    If not provided, uses the default rubric:
    - S: Less than 1 day, single component, low risk
    - M: 1-3 days, may touch multiple components, moderate complexity
    - L: 3-5 days, significant complexity, unknowns or cross-team coordination
    """
    # Parse custom rubric if provided
    rubric = None
    if sizing_rubric:
        try:
            rubric_data = json.loads(sizing_rubric)
            if not isinstance(rubric_data, dict):
                raise ValueError(
                    f"sizing_rubric must be a JSON object, got {type(rubric_data).__name__}"
                )
            rubric = SizingRubric(**rubric_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Invalid sizing_rubric: {e}")

    return _decompose_to_tickets_impl(requirements_json, sizing_rubric=rubric)


@app.tool
def export_tickets(
    tickets_json: Annotated[str, "JSON string of ticket collection from decompose_to_tickets"],
    output_format: Annotated[str, "Export format: 'csv', 'jira', or 'yaml'"] = "csv",
    project_key: Annotated[str, "Jira project key for issue creation"] = "PROJECT",
) -> str:
    """Export tickets to different formats for integration with external tools.

    Supported formats:
    - csv: Flat CSV with one row per story (for spreadsheet import)
    - jira: Jira REST API bulk create payload (ready for POST to /rest/api/3/issue/bulk)
    - yaml: YAML format (for GitOps workflows)

    Returns the exported content as a string.
    """
    return _export_tickets_impl(tickets_json, output_format=output_format, project_key=project_key)


@app.tool
def health_check() -> dict:
    """Check service health and dependencies.

    Returns status information including:
    - Service status (healthy/degraded/unhealthy)
    - OpenAI API connectivity (via simple request)
    - Circuit breaker state
    - Rate limiter status
    - Configuration summary
    """
    _ensure_logging_initialized()

    settings = get_settings()
    circuit_breaker = get_circuit_breaker(settings)
    rate_limiter = get_rate_limiter(settings)

    # Check circuit breaker state
    cb_state = circuit_breaker.state

    # Determine overall status
    if cb_state == "open":
        status = "degraded"
        status_message = "Circuit breaker open - upstream failures detected"
    elif cb_state == "half_open":
        status = "degraded"
        status_message = "Circuit breaker half-open - testing recovery"
    else:
        status = "healthy"
        status_message = "All systems operational"

    return {
        "status": status,
        "message": status_message,
        "circuit_breaker": {
            "state": cb_state,
            "failure_threshold": circuit_breaker.failure_threshold,
            "reset_timeout": circuit_breaker.reset_timeout,
        },
        "rate_limiter": {
            "max_calls": rate_limiter.max_calls,
            "window_seconds": rate_limiter.window_seconds,
        },
        "config": {
            "openai_model": settings.openai_model,
            "analyze_temperature": settings.analyze_temperature,
            "decompose_temperature": settings.decompose_temperature,
            "max_retries": settings.max_retries,
            "llm_timeout": settings.llm_timeout,
            "max_prd_length": settings.max_prd_length,
            "circuit_breaker_failure_threshold": settings.circuit_breaker_failure_threshold,
            "circuit_breaker_reset_timeout": settings.circuit_breaker_reset_timeout,
        },
        "version": PROMPT_VERSION,
    }


if __name__ == "__main__":
    # Initialize logging when running as CLI
    _ensure_logging_initialized()
    app.run()
