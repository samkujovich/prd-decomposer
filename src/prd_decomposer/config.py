"""Configuration settings for PRD Decomposer."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="PRD_")

    # LLM settings
    openai_model: str = "gpt-4o"
    analyze_temperature: float = 0.2
    decompose_temperature: float = 0.3

    # Retry settings
    max_retries: int = 3
    initial_retry_delay: float = 1.0


# Singleton for convenience
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
