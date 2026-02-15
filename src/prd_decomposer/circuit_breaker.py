"""Circuit breaker and rate limiting for upstream API calls."""

import logging
import threading
import time

logger = logging.getLogger("prd_decomposer")


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

    def allow_request(self) -> str:
        """Check if a request should be allowed and return the observed state.

        Returns:
            The observed state ("closed" or "half_open") if request is allowed.

        Raises:
            CircuitBreakerOpenError: If circuit is open and blocking calls.
        """
        with self._lock:
            state = self._get_state_unlocked()

            if state == "closed":
                return "closed"

            if state == "half_open":
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return "half_open"
                # Too many half-open calls in progress
                retry_after = 1.0  # Short retry for half-open
                raise CircuitBreakerOpenError(
                    "Circuit breaker half-open, max test calls in progress",
                    retry_after=retry_after,
                )

            # state == "open"
            if self._last_failure_time is None:
                raise RuntimeError("Circuit breaker in open state but no failure time recorded")
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
