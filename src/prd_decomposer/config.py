"""Configuration settings for PRD Decomposer."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="PRD_")

    # LLM settings
    openai_model: str = Field(default="gpt-4o", description="OpenAI model identifier")
    analyze_temperature: float = Field(
        default=0.2, description="Temperature for analyze_prd tool"
    )
    decompose_temperature: float = Field(
        default=0.3, description="Temperature for decompose_to_tickets tool"
    )

    # Retry settings
    max_retries: int = Field(
        default=3, description="Maximum retry attempts for LLM calls"
    )
    initial_retry_delay: float = Field(
        default=1.0, description="Initial delay in seconds for retry backoff"
    )


# Singleton for convenience
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
