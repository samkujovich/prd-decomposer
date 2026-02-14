"""Tests for configuration settings."""

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
