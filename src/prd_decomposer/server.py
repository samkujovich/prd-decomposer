"""MCP server for PRD analysis and decomposition."""

import atexit
import csv
import hashlib
import io
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
    global _client, _rate_limiter, _circuit_breaker
    _client = None
    _rate_limiter = None
    _circuit_breaker = None


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


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """Thread-safe circuit breaker for LLM calls.

    Prevents cascading failures by tracking consecutive errors and
    temporarily blocking calls when the failure threshold is exceeded.

    States:
    - CLOSED: Normal operation, calls allowed
    - OPEN: Blocking calls after threshold failures
    - HALF_OPEN: Testing with single call after reset timeout
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            reset_timeout: Seconds to wait before attempting half-open state
            half_open_max_calls: Max concurrent calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._get_state_unlocked()

    def _get_state_unlocked(self) -> str:
        """Get state without lock (caller must hold lock)."""
        if self._state == "open" and self._last_failure_time is not None:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                return "half_open"
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if request is allowed, False otherwise.

        Raises:
            CircuitBreakerOpenError: If circuit is open and blocking calls.
        """
        with self._lock:
            state = self._get_state_unlocked()

            if state == "closed":
                return True

            if state == "half_open":
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                # Too many half-open calls in progress
                retry_after = 1.0  # Short retry for half-open
                raise CircuitBreakerOpenError(
                    "Circuit breaker half-open, max test calls in progress",
                    retry_after=retry_after,
                )

            # state == "open"
            assert self._last_failure_time is not None
            retry_after = self.reset_timeout - (time.time() - self._last_failure_time)
            raise CircuitBreakerOpenError(
                f"Circuit breaker open due to {self._failure_count} consecutive failures. "
                f"Retry in {retry_after:.1f}s.",
                retry_after=max(0.0, retry_after),
            )

    def record_success(self) -> None:
        """Record a successful call. Closes circuit if half-open."""
        with self._lock:
            self._failure_count = 0
            self._half_open_calls = 0
            self._state = "closed"
            logger.debug("Circuit breaker: success recorded, state=closed")

    def record_failure(self) -> None:
        """Record a failed call. May open circuit if threshold exceeded."""
        with self._lock:
            # Check state BEFORE updating timestamp (otherwise half-open check fails)
            was_half_open = self._get_state_unlocked() == "half_open"

            self._failure_count += 1
            self._last_failure_time = time.time()

            # In half-open state, any failure reopens circuit
            if was_half_open:
                self._state = "open"
                self._half_open_calls = 0
                logger.warning(
                    "Circuit breaker: failure in half-open state, reopening circuit"
                )
                return

            # Check if threshold exceeded
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker: failure threshold reached (%d), opening circuit",
                    self._failure_count,
                )
            else:
                logger.debug(
                    "Circuit breaker: failure recorded (%d/%d)",
                    self._failure_count,
                    self.failure_threshold,
                )

    def release_half_open_slot(self) -> None:
        """Release a half-open slot without recording success/failure.

        Use this if an exception occurs before the actual call completes,
        to prevent slot leaks in half-open state.
        """
        with self._lock:
            if self._half_open_calls > 0:
                self._half_open_calls -= 1
                logger.debug("Circuit breaker: released half-open slot")

    def reset(self) -> None:
        """Reset circuit breaker state (for testing)."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


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

    # Check circuit breaker before attempting call
    circuit_breaker.allow_request()
    # Check state AFTER allow_request() - it may have transitioned open→half_open
    is_half_open_probe = circuit_breaker.state == "half_open"

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

            except RateLimitError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                    logger.warning(
                        "Retrying LLM call (attempt %d/%d) after RateLimitError, delay=%.1fs",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)

            except APIConnectionError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                    logger.warning(
                        "Retrying LLM call (attempt %d/%d) after APIConnectionError, delay=%.1fs",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)

            except APITimeoutError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
                    logger.warning(
                        "Retrying LLM call (attempt %d/%d) after APITimeoutError, delay=%.1fs",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)

            except APIError as e:
                # Don't retry on 4xx errors (bad request, auth, etc.)
                # Also don't count as circuit breaker failure (client error, not upstream)
                status_code = getattr(e, "status_code", None)
                if status_code and 400 <= status_code < 500:
                    is_client_error = True
                    raise LLMError(f"OpenAI API error: {e}")
                last_error = e
                if attempt < max_attempts - 1:
                    base_delay = settings.initial_retry_delay * (2**attempt)
                    delay = base_delay * (0.5 + random.random())  # Jitter: 0.5x to 1.5x
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
                        requirements_json=validated_input.model_dump_json(indent=2),
                        sizing_rubric=rubric_text,
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

    # Check circuit breaker status
    try:
        circuit_breaker = get_circuit_breaker(settings)
        cb_state = circuit_breaker.state
        checks["circuit_breaker"] = {
            "status": "ok" if cb_state == "closed" else "degraded",
            "state": cb_state,
            "failure_count": circuit_breaker._failure_count,
            "failure_threshold": circuit_breaker.failure_threshold,
            "reset_timeout": circuit_breaker.reset_timeout,
        }
        # Mark degraded for both open and half_open states (recovery in progress)
        if cb_state in ("open", "half_open"):
            status = "degraded" if status == "healthy" else status
    except Exception as e:
        status = "degraded" if status == "healthy" else status
        checks["circuit_breaker"] = {
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
            "circuit_breaker_failure_threshold": settings.circuit_breaker_failure_threshold,
            "circuit_breaker_reset_timeout": settings.circuit_breaker_reset_timeout,
        },
        "version": PROMPT_VERSION,
    }


@app.tool
def export_tickets(
    tickets_json: Annotated[str, "JSON string of ticket collection from decompose_to_tickets"],
    format: Annotated[str, "Export format: 'csv', 'jira', or 'yaml'"] = "csv",
) -> str:
    """Export tickets to different formats for integration with external tools.

    Supported formats:
    - csv: Flat CSV with one row per story (for spreadsheet import)
    - jira: Jira REST API bulk create payload (ready for POST to /rest/api/3/issue/bulk)
    - yaml: YAML format (for GitOps workflows)

    Returns the exported content as a string.
    """
    # Parse input
    try:
        tickets = json.loads(tickets_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tickets: {e}")

    if not isinstance(tickets, dict):
        raise ValueError(
            f"Invalid ticket structure: must be a JSON object, got {type(tickets).__name__}"
        )

    if "epics" not in tickets:
        raise ValueError("Invalid ticket structure: missing 'epics' key")

    epics = tickets["epics"]
    if not isinstance(epics, list):
        raise ValueError(
            f"Invalid ticket structure: 'epics' must be a list, got {type(epics).__name__}"
        )

    for i, epic in enumerate(epics):
        if not isinstance(epic, dict):
            raise ValueError(
                f"Invalid ticket structure: epics[{i}] must be an object, got {type(epic).__name__}"
            )

        # Validate epic scalar fields
        epic_path = f"epics[{i}]"
        epic["title"] = _validate_string_field(
            epic.get("title"), "title", epic_path, required=True
        )
        epic["description"] = _validate_string_field(
            epic.get("description"), "description", epic_path, required=False
        )

        # Validate and normalize stories (null -> empty list)
        stories = epic.get("stories")
        if stories is None:
            stories = []
            epic["stories"] = stories  # Normalize null to empty list
        if not isinstance(stories, list):
            raise ValueError(
                f"Invalid ticket structure: epics[{i}].stories must be a list, "
                f"got {type(stories).__name__}"
            )
        if stories:
            for j, story in enumerate(stories):
                if not isinstance(story, dict):
                    raise ValueError(
                        f"Invalid ticket structure: epics[{i}].stories[{j}] must be an object, "
                        f"got {type(story).__name__}"
                    )
                story_path = f"epics[{i}].stories[{j}]"

                # Validate scalar string fields
                story["title"] = _validate_string_field(
                    story.get("title"), "title", story_path, required=True
                )
                story["description"] = _validate_string_field(
                    story.get("description"), "description", story_path, required=False
                )

                # Validate and normalize list-of-string fields
                story["acceptance_criteria"] = _validate_string_list(
                    story.get("acceptance_criteria"), "acceptance_criteria", story_path
                )
                story["labels"] = _validate_string_list(
                    story.get("labels"), "labels", story_path
                )
                story["requirement_ids"] = _validate_string_list(
                    story.get("requirement_ids"), "requirement_ids", story_path
                )

        # Validate epic labels too
        epic["labels"] = _validate_string_list(
            epic.get("labels"), "labels", f"epics[{i}]"
        )

    format_lower = format.lower()

    if format_lower == "csv":
        return _export_to_csv(tickets)
    elif format_lower == "jira":
        return _export_to_jira(tickets)
    elif format_lower == "yaml":
        return _export_to_yaml(tickets)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'jira', or 'yaml'.")


def _export_to_csv(tickets: dict) -> str:
    """Export tickets to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "epic_title",
        "story_title",
        "story_description",
        "acceptance_criteria",
        "size",
        "priority",
        "labels",
        "requirement_ids",
    ])

    # Data rows
    for epic in tickets.get("epics", []):
        epic_title = epic.get("title", "")
        for story in epic.get("stories", []):
            writer.writerow([
                epic_title,
                story.get("title", ""),
                story.get("description", ""),
                "; ".join(story.get("acceptance_criteria", [])),
                story.get("size", ""),
                story.get("priority", "medium"),
                ", ".join(story.get("labels", [])),
                ", ".join(story.get("requirement_ids", [])),
            ])

    return output.getvalue()


def _export_to_jira(tickets: dict) -> str:
    """Export tickets to Jira REST API bulk create format.

    Returns JSON compatible with POST /rest/api/3/issue/bulk.
    Epic linking must be done post-creation using Jira's Epic Link API.
    """
    issues = []

    for epic in tickets.get("epics", []):
        # Create epic issue
        epic_issue = {
            "fields": {
                "project": {"key": "${PROJECT_KEY}"},
                "issuetype": {"name": "Epic"},
                "summary": epic.get("title", ""),
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": epic.get("description", "")}],
                        }
                    ],
                },
                "labels": epic.get("labels", []),
            }
        }
        issues.append(epic_issue)

        # Create story issues
        for story in epic.get("stories", []):
            # Build description with acceptance criteria
            description_parts = [story.get("description", "")]
            if story.get("acceptance_criteria"):
                description_parts.append("\n\nAcceptance Criteria:")
                for ac in story.get("acceptance_criteria", []):
                    description_parts.append(f"- {ac}")

            story_issue = {
                "fields": {
                    "project": {"key": "${PROJECT_KEY}"},
                    "issuetype": {"name": "Story"},
                    "summary": story.get("title", ""),
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {"type": "text", "text": "\n".join(description_parts)}
                                ],
                            }
                        ],
                    },
                    "labels": story.get("labels", []),
                    "priority": {"name": _map_priority_to_jira(story.get("priority", "medium"))},
                    # Custom field for t-shirt sizing (placeholder)
                    # "customfield_10001": story.get("size", "M"),
                },
            }
            issues.append(story_issue)

    # Return clean Jira bulk create payload
    # The payload is compatible with POST /rest/api/3/issue/bulk
    # Epic linking must be done post-creation using the Jira Epic Link API
    return json.dumps({"issueUpdates": issues}, indent=2)


def _map_priority_to_jira(priority: str | None) -> str:
    """Map internal priority to Jira priority names.

    Handles non-string priority values by coercing to string or defaulting.
    """
    # Handle non-string types
    if priority is None:
        return "Medium"
    if not isinstance(priority, str):
        # Coerce to string (handles int, float, bool, etc.)
        priority = str(priority)

    mapping = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return mapping.get(priority.lower(), "Medium")


def _validate_string_field(value: object, field_name: str, path: str, required: bool = True) -> str:
    """Validate that a value is a string.

    Returns the validated string, coercing non-strings if possible.
    Raises ValueError with clear path if validation fails for required fields.
    For required fields, both None and empty strings are rejected.
    """
    if value is None:
        if required:
            raise ValueError(f"Invalid ticket structure: {path}.{field_name} is required")
        return ""
    if not isinstance(value, str):
        # Coerce to string with warning in logs
        logger.warning(
            "Coercing %s.%s from %s to string",
            path, field_name, type(value).__name__
        )
        value = str(value)
    # Check for empty string after coercion
    if required and value == "":
        raise ValueError(f"Invalid ticket structure: {path}.{field_name} cannot be empty")
    return value


def _validate_string_list(value: object, field_name: str, path: str) -> list[str]:
    """Validate that a value is a list of strings.

    Returns the validated list, coercing non-string elements to strings.
    Raises ValueError with clear path if validation fails.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"Invalid ticket structure: {path}.{field_name} must be a list, "
            f"got {type(value).__name__}"
        )
    # Coerce each element to string
    result = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            if item is None:
                continue  # Skip null items
            # Coerce to string with warning in logs
            logger.warning(
                "Coercing %s.%s[%d] from %s to string",
                path, field_name, i, type(item).__name__
            )
            result.append(str(item))
        else:
            result.append(item)
    return result


def _escape_yaml_string(s: str) -> str:
    """Escape a string for use in double-quoted YAML scalars.

    Handles: backslashes, double quotes, newlines, tabs, carriage returns.
    """
    # Order matters: escape backslashes first
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("\r", "\\r")
    return s


def _yaml_flow_list(items: list[str]) -> str:
    """Format a list as YAML flow sequence with properly quoted elements.

    Each element is quoted to handle YAML-significant characters (commas, colons, etc).
    """
    if not items:
        return "[]"
    escaped = [f'"{_escape_yaml_string(item)}"' for item in items]
    return f"[{', '.join(escaped)}]"


def _export_to_yaml(tickets: dict) -> str:
    """Export tickets to YAML format."""
    # Simple YAML serialization without external dependency
    lines = ["# PRD Decomposer Export", f"# Generated: {datetime.now(UTC).isoformat()}", ""]

    epics = tickets.get("epics", [])
    # Always emit epics key for consistent schema
    lines.append("epics:")

    if not epics:
        # Empty list - use flow syntax for consistency
        lines[-1] = "epics: []"
    else:
        for i, epic in enumerate(epics):
            lines.append(f"  - title: \"{_escape_yaml_string(epic.get('title', ''))}\"")
            lines.append(f"    description: \"{_escape_yaml_string(epic.get('description', ''))}\"")
            lines.append(f"    labels: {_yaml_flow_list(epic.get('labels', []))}")

            stories = epic.get("stories", [])
            if not stories:
                # Empty stories - use flow syntax instead of block with no items
                lines.append("    stories: []")
            else:
                lines.append("    stories:")
                for story in stories:
                    lines.append(f"      - title: \"{_escape_yaml_string(story.get('title', ''))}\"")
                    lines.append(f"        description: \"{_escape_yaml_string(story.get('description', ''))}\"")
                    lines.append(f"        size: \"{_escape_yaml_string(str(story.get('size', 'M')))}\"")
                    lines.append(f"        priority: \"{_escape_yaml_string(str(story.get('priority', 'medium')))}\"")
                    lines.append(f"        labels: {_yaml_flow_list(story.get('labels', []))}")
                    lines.append(f"        requirement_ids: {_yaml_flow_list(story.get('requirement_ids', []))}")

                    ac_list = story.get("acceptance_criteria", [])
                    if ac_list:
                        lines.append("        acceptance_criteria:")
                        for ac in ac_list:
                            lines.append(f"          - \"{_escape_yaml_string(ac)}\"")
                    else:
                        lines.append("        acceptance_criteria: []")

            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Initialize logging when running as CLI
    _ensure_logging_initialized()
    app.run()
