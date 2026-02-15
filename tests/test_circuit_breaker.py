"""Tests for circuit breaker and rate limiter functionality."""

import time
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIError, RateLimitError

from prd_decomposer.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RateLimiter,
    RateLimitExceededError,
)
from prd_decomposer.config import Settings
from prd_decomposer.server import (
    LLMError,
    _call_llm_with_retry,
    _shutdown_event,
    get_circuit_breaker,
    health_check,
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_circuit_breaker_starts_closed(self):
        """Verify new circuit breaker is in closed state."""
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.allow_request() == "closed"

    def test_circuit_breaker_opens_after_threshold(self):
        """Verify circuit opens after failure threshold exceeded."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"

    def test_circuit_breaker_blocks_when_open(self):
        """Verify open circuit breaker blocks requests."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure()  # Opens circuit

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.allow_request()
        assert exc_info.value.retry_after > 0

    def test_circuit_breaker_success_resets_count(self):
        """Verify success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_circuit_breaker_half_open_after_timeout(self):
        """Verify circuit transitions to half-open after reset timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        # Wait for reset timeout
        time.sleep(0.02)

        assert cb.state == "half_open"

    def test_circuit_breaker_half_open_allows_one_request(self):
        """Verify half-open state allows limited test requests."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01, half_open_max_calls=1)
        cb.record_failure()

        time.sleep(0.02)

        assert cb.allow_request() == "half_open"
        # Second request should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            cb.allow_request()

    def test_circuit_breaker_closes_on_half_open_success(self):
        """Verify successful call in half-open state closes circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)

        cb.allow_request()
        cb.record_success()

        assert cb.state == "closed"

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Verify failure in half-open state reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)

        cb.allow_request()
        cb.record_failure()

        assert cb.state == "open"

    def test_circuit_breaker_reset(self):
        """Verify reset() clears all state."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with LLM calls."""

    def test_call_llm_with_retry_respects_circuit_breaker(
        self, permissive_rate_limiter
    ):
        """Verify _call_llm_with_retry raises when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure()  # Open the circuit

        with pytest.raises(CircuitBreakerOpenError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=MagicMock(),
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

    def test_call_llm_with_retry_records_success(
        self, mock_client_factory, permissive_rate_limiter
    ):
        """Verify successful LLM call records success with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()  # Add one failure
        assert cb._failure_count == 1

        mock_client = mock_client_factory({"result": "ok"})
        _call_llm_with_retry(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            client=mock_client,
            rate_limiter=permissive_rate_limiter,
            circuit_breaker=cb,
        )

        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_call_llm_with_retry_records_failure(self, permissive_rate_limiter):
        """Verify failed LLM call records failure with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=5)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit", response=MagicMock(status_code=429), body=None
        )

        settings = Settings(initial_retry_delay=0.01, max_retries=1)

        with patch("prd_decomposer.server.time.sleep"):
            with pytest.raises(LLMError):
                _call_llm_with_retry(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.5,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                    circuit_breaker=cb,
                )

        assert cb._failure_count == 1

    def test_health_check_includes_circuit_breaker(self):
        """Verify health_check includes circuit breaker status."""
        result = health_check()

        assert "circuit_breaker" in result
        assert result["circuit_breaker"]["state"] == "closed"
        assert "circuit_breaker_failure_threshold" in result["config"]

    def test_get_circuit_breaker_returns_singleton(self):
        """Verify get_circuit_breaker returns same instance."""
        cb1 = get_circuit_breaker()
        cb2 = get_circuit_breaker()
        assert cb1 is cb2


class TestCircuitBreakerBugFixes:
    """Tests for specific circuit breaker bug fixes."""

    def test_half_open_failure_resets_slot_count(self):
        """Verify half-open failure properly resets _half_open_calls (bug fix #1)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        time.sleep(0.02)  # Wait for half-open

        # Take a half-open slot
        cb.allow_request()
        assert cb._half_open_calls == 1

        # Fail the probe - should reset _half_open_calls
        cb.record_failure()
        assert cb._half_open_calls == 0
        assert cb.state == "open"

    def test_release_half_open_slot(self):
        """Verify release_half_open_slot decrements counter (bug fix #2)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)

        cb.allow_request()
        assert cb._half_open_calls == 1

        cb.release_half_open_slot()
        assert cb._half_open_calls == 0

    def test_rate_limit_releases_half_open_slot(self, permissive_circuit_breaker):
        """Verify rate limit error releases half-open slot (bug fix #2)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)

        # Create a rate limiter that will fail on second call
        rate_limiter = RateLimiter(max_calls=1, window_seconds=60)
        rate_limiter.check_and_record()  # Use up the one allowed call

        with pytest.raises(RateLimitExceededError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=MagicMock(),
                rate_limiter=rate_limiter,
                circuit_breaker=cb,
            )

        # Half-open slot should have been released but NOT counted as failure
        assert cb._half_open_calls == 0
        # Still in half-open (rate limit is client-side, not upstream failure)
        assert cb.state == "half_open"

    def test_non_retryable_error_records_failure(self, permissive_rate_limiter):
        """Verify non-retryable LLM errors record circuit breaker failure."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that returns empty response (non-retryable error)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None  # Empty response
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="empty response"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Failure should be recorded
        assert cb._failure_count == 1

    def test_invalid_json_records_failure(self, permissive_rate_limiter):
        """Verify invalid JSON response records circuit breaker failure."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that returns invalid JSON
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json {"
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="invalid JSON"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Failure should be recorded
        assert cb._failure_count == 1

    def test_half_open_non_retryable_error_reopens_circuit(self, permissive_rate_limiter):
        """Verify non-retryable error in half-open state reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        time.sleep(0.02)  # Wait for half-open

        # Mock client that returns empty response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(LLMError, match="empty response"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should be back to open state
        assert cb.state == "open"
        assert cb._half_open_calls == 0

    def test_4xx_error_does_not_count_as_failure(self, permissive_rate_limiter):
        """Verify 4xx client errors don't increment circuit breaker failure count."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

        # Mock client that raises 4xx APIError
        mock_client = MagicMock()
        error = APIError(
            message="Bad request",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx should NOT count as failure (client error, not upstream)
        assert cb._failure_count == 0
        assert cb.state == "closed"

    def test_5xx_error_counts_as_failure(self, permissive_rate_limiter):
        """Verify 5xx server errors DO increment circuit breaker failure count."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)
        settings = Settings(max_retries=1)  # Single retry to speed up test

        # Mock client that raises 5xx APIError
        mock_client = MagicMock()
        error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 500
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="LLM call failed"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 5xx SHOULD count as failure (upstream error)
        assert cb._failure_count == 1

    def test_half_open_probe_single_attempt_only(self, permissive_rate_limiter):
        """Verify half-open probes only attempt once (no retries)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        time.sleep(0.02)  # Wait for half-open
        assert cb.state == "half_open"

        settings = Settings(max_retries=3)  # Would normally retry 3 times

        # Mock client that always fails with retryable error
        mock_client = MagicMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="1 attempts"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should have only made 1 call, not 3
        assert mock_client.chat.completions.create.call_count == 1

    def test_half_open_detected_after_open_timeout(self, permissive_rate_limiter):
        """Verify half-open is detected even when state was 'open' before allow_request()."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        # State is "open" right now, but will transition to half_open on allow_request
        assert cb._state == "open"

        time.sleep(0.02)  # Wait for reset timeout

        settings = Settings(max_retries=3)

        # Mock client that always fails
        mock_client = MagicMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="1 attempts"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # Should detect half-open and limit to 1 attempt
        assert mock_client.chat.completions.create.call_count == 1

    def test_4xx_during_half_open_closes_circuit(self, permissive_rate_limiter):
        """Verify 4xx error during half-open probe closes circuit (upstream is responsive)."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)
        cb.record_failure()  # Opens circuit

        time.sleep(0.02)  # Wait for half-open
        assert cb.state == "half_open"

        settings = Settings(max_retries=1)

        # Mock client that raises 4xx error
        mock_client = MagicMock()
        error = APIError(
            message="Bad request - invalid input",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx indicates upstream is responsive, so circuit should close
        assert cb.state == "closed", "4xx during half-open should close circuit (upstream responsive)"


class TestCircuitBreaker4xxInClosedState:
    """Tests for circuit breaker 4xx handling in closed state."""

    def test_4xx_in_closed_state_resets_failure_count(self, permissive_rate_limiter):
        """Verify 4xx in closed state resets failure streak (upstream responsive)."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        # First failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1

        # Now simulate 4xx error - should reset failure count
        mock_client = MagicMock()
        error = APIError(
            message="Bad request",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400
        mock_client.chat.completions.create.side_effect = error

        settings = Settings(max_retries=1)
        with pytest.raises(LLMError, match="OpenAI API error"):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )

        # 4xx should reset failure count (upstream is responsive)
        assert cb._failure_count == 0, "4xx should reset failure count"
        assert cb.state == "closed"

    def test_5xx_4xx_5xx_does_not_open_circuit(self, permissive_rate_limiter):
        """Verify 5xx/4xx/5xx sequence doesn't incorrectly open circuit."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)  # Opens after 2 failures

        settings = Settings(max_retries=1)

        # First failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1

        # 4xx - should reset count
        mock_client = MagicMock()
        error_4xx = APIError(message="Bad request", request=MagicMock(), body=None)
        error_4xx.status_code = 400
        mock_client.chat.completions.create.side_effect = error_4xx

        with pytest.raises(LLMError):
            _call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                client=mock_client,
                settings=settings,
                rate_limiter=permissive_rate_limiter,
                circuit_breaker=cb,
            )
        assert cb._failure_count == 0  # Reset by 4xx

        # Second failure (5xx)
        cb.record_failure()
        assert cb._failure_count == 1
        assert cb.state == "closed", "Should not open - only 1 consecutive failure"


class TestShutdownCircuitBreaker:
    """Tests for shutdown handling with circuit breaker."""

    def test_shutdown_does_not_trip_circuit_breaker(self, permissive_rate_limiter):
        """Verify shutdown-aborted requests don't count as failures."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60)

        mock_client = MagicMock()
        settings = Settings(max_retries=1)

        # Set shutdown flag
        _shutdown_event.set()

        try:
            with pytest.raises(LLMError, match="shutting down"):
                _call_llm_with_retry(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.5,
                    client=mock_client,
                    settings=settings,
                    rate_limiter=permissive_rate_limiter,
                    circuit_breaker=cb,
                )

            # Shutdown abort should NOT trip circuit breaker
            assert cb._failure_count == 0, "Shutdown should not count as failure"
            assert cb.state == "closed"
        finally:
            _shutdown_event.clear()
