"""Tests for configuration settings."""

import threading

import pytest
from pydantic import ValidationError

from prd_decomposer.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_has_default_model(self):
        """Verify Settings has default openai_model."""
        settings = Settings()
        assert settings.openai_model == "gpt-4o"

    def test_settings_has_default_temperatures(self):
        """Verify Settings has default temperatures."""
        settings = Settings()
        assert settings.analyze_temperature == 0.2
        assert settings.decompose_temperature == 0.3

    def test_settings_has_default_retry_config(self):
        """Verify Settings has default retry configuration."""
        settings = Settings()
        assert settings.max_retries == 3
        assert settings.initial_retry_delay == 1.0

    def test_settings_loads_from_env(self, monkeypatch):
        """Verify Settings loads from environment variables."""
        monkeypatch.setenv("PRD_OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("PRD_MAX_RETRIES", "5")

        settings = Settings()

        assert settings.openai_model == "gpt-4o-mini"
        assert settings.max_retries == 5

    def test_settings_validates_temperature_from_env(self, monkeypatch):
        """Verify Settings loads float temperatures from env vars."""
        monkeypatch.setenv("PRD_ANALYZE_TEMPERATURE", "0.5")
        settings = Settings()
        assert settings.analyze_temperature == 0.5

    def test_settings_invalid_int_from_env_raises(self, monkeypatch):
        """Verify Settings raises for invalid integer env vars."""
        monkeypatch.setenv("PRD_MAX_RETRIES", "not-a-number")
        with pytest.raises(ValidationError):
            Settings()

    def test_settings_max_retries_minimum_bound(self):
        """Verify max_retries rejects values below 1."""
        with pytest.raises(ValidationError):
            Settings(max_retries=0)

    def test_settings_max_retries_maximum_bound(self):
        """Verify max_retries rejects values above 10."""
        with pytest.raises(ValidationError):
            Settings(max_retries=11)

    def test_settings_max_retries_valid_bounds(self):
        """Verify max_retries accepts values within bounds."""
        settings_min = Settings(max_retries=1)
        settings_max = Settings(max_retries=10)
        assert settings_min.max_retries == 1
        assert settings_max.max_retries == 10

    def test_settings_initial_retry_delay_minimum_bound(self):
        """Verify initial_retry_delay rejects zero and negative values."""
        with pytest.raises(ValidationError):
            Settings(initial_retry_delay=0)
        with pytest.raises(ValidationError):
            Settings(initial_retry_delay=-1.0)

    def test_settings_initial_retry_delay_maximum_bound(self):
        """Verify initial_retry_delay rejects values above 60."""
        with pytest.raises(ValidationError):
            Settings(initial_retry_delay=61)

    def test_settings_initial_retry_delay_valid_bounds(self):
        """Verify initial_retry_delay accepts values within bounds."""
        settings_min = Settings(initial_retry_delay=0.001)
        settings_max = Settings(initial_retry_delay=60)
        assert settings_min.initial_retry_delay == 0.001
        assert settings_max.initial_retry_delay == 60

    def test_settings_has_default_log_level(self):
        """Verify Settings has default log_level of INFO."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_settings_has_default_log_format(self):
        """Verify Settings has default log_format of json."""
        settings = Settings()
        assert settings.log_format == "json"

    def test_settings_log_level_from_env(self, monkeypatch):
        """Verify log_level loads from PRD_LOG_LEVEL env var."""
        monkeypatch.setenv("PRD_LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_settings_log_format_from_env(self, monkeypatch):
        """Verify log_format loads from PRD_LOG_FORMAT env var."""
        monkeypatch.setenv("PRD_LOG_FORMAT", "text")
        settings = Settings()
        assert settings.log_format == "text"

    def test_settings_log_level_rejects_invalid(self):
        """Verify log_level rejects values outside allowed set."""
        with pytest.raises(ValidationError):
            Settings(log_level="VERBOSE")

    def test_settings_log_format_rejects_invalid(self):
        """Verify log_format rejects values outside allowed set."""
        with pytest.raises(ValidationError):
            Settings(log_format="yaml")

    def test_settings_has_default_llm_timeout(self):
        """Verify Settings has default llm_timeout of 60 seconds."""
        settings = Settings()
        assert settings.llm_timeout == 60.0

    def test_settings_llm_timeout_from_env(self, monkeypatch):
        """Verify llm_timeout loads from environment variable."""
        monkeypatch.setenv("PRD_LLM_TIMEOUT", "120.0")
        settings = Settings()
        assert settings.llm_timeout == 120.0

    def test_settings_llm_timeout_minimum_bound(self):
        """Verify llm_timeout rejects zero and negative values."""
        with pytest.raises(ValidationError):
            Settings(llm_timeout=0)
        with pytest.raises(ValidationError):
            Settings(llm_timeout=-1.0)

    def test_settings_llm_timeout_maximum_bound(self):
        """Verify llm_timeout rejects values above 300."""
        with pytest.raises(ValidationError):
            Settings(llm_timeout=301)

    def test_settings_llm_timeout_valid_bounds(self):
        """Verify llm_timeout accepts values within bounds."""
        settings_min = Settings(llm_timeout=0.1)
        settings_max = Settings(llm_timeout=300)
        assert settings_min.llm_timeout == 0.1
        assert settings_max.llm_timeout == 300

    def test_settings_has_default_max_prd_length(self):
        """Verify Settings has default max_prd_length of 100000."""
        settings = Settings()
        assert settings.max_prd_length == 100000

    def test_settings_max_prd_length_from_env(self, monkeypatch):
        """Verify max_prd_length loads from environment variable."""
        monkeypatch.setenv("PRD_MAX_PRD_LENGTH", "50000")
        settings = Settings()
        assert settings.max_prd_length == 50000

    def test_settings_max_prd_length_minimum_bound(self):
        """Verify max_prd_length rejects values below 1000."""
        with pytest.raises(ValidationError):
            Settings(max_prd_length=999)

    def test_settings_max_prd_length_maximum_bound(self):
        """Verify max_prd_length rejects values above 500000."""
        with pytest.raises(ValidationError):
            Settings(max_prd_length=500001)

    def test_settings_max_prd_length_valid_bounds(self):
        """Verify max_prd_length accepts values within bounds."""
        settings_min = Settings(max_prd_length=1000)
        settings_max = Settings(max_prd_length=500000)
        assert settings_min.max_prd_length == 1000
        assert settings_max.max_prd_length == 500000

    def test_settings_has_default_rate_limit(self):
        """Verify Settings has default rate limit of 60 calls per 60 seconds."""
        settings = Settings()
        assert settings.rate_limit_calls == 60
        assert settings.rate_limit_window == 60

    def test_settings_rate_limit_from_env(self, monkeypatch):
        """Verify rate limit settings load from environment variables."""
        monkeypatch.setenv("PRD_RATE_LIMIT_CALLS", "100")
        monkeypatch.setenv("PRD_RATE_LIMIT_WINDOW", "120")
        settings = Settings()
        assert settings.rate_limit_calls == 100
        assert settings.rate_limit_window == 120

    def test_settings_rate_limit_calls_bounds(self):
        """Verify rate_limit_calls rejects values outside 1-1000."""
        with pytest.raises(ValidationError):
            Settings(rate_limit_calls=0)
        with pytest.raises(ValidationError):
            Settings(rate_limit_calls=1001)

    def test_settings_rate_limit_window_bounds(self):
        """Verify rate_limit_window rejects values outside 1-3600."""
        with pytest.raises(ValidationError):
            Settings(rate_limit_window=0)
        with pytest.raises(ValidationError):
            Settings(rate_limit_window=3601)

    def test_settings_rate_limit_valid_bounds(self):
        """Verify rate limit settings accept values within bounds."""
        settings = Settings(rate_limit_calls=1, rate_limit_window=1)
        assert settings.rate_limit_calls == 1
        assert settings.rate_limit_window == 1
        settings = Settings(rate_limit_calls=1000, rate_limit_window=3600)
        assert settings.rate_limit_calls == 1000
        assert settings.rate_limit_window == 3600


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self, monkeypatch):
        """Verify get_settings returns a Settings instance."""
        # Reset singleton for test isolation
        import prd_decomposer.config as config_module
        config_module._settings = None

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_caches_instance(self):
        """Verify get_settings returns same instance on repeated calls."""
        import prd_decomposer.config as config_module
        config_module._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_thread_safe(self):
        """Verify concurrent get_settings calls return the same instance."""
        import prd_decomposer.config as config_module
        config_module._settings = None

        results: list[Settings] = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()  # All threads start at once
            results.append(get_settings())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must get the exact same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)
