"""Configuration settings for PRD Decomposer."""

import threading
from typing import Literal

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
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for LLM calls (1-10)",
    )
    initial_retry_delay: float = Field(
        default=1.0,
        gt=0,
        le=60,
        description="Initial delay in seconds for retry backoff (0-60)",
    )

    # Timeout settings
    llm_timeout: float = Field(
        default=60.0,
        gt=0,
        le=300,
        description="Timeout in seconds for LLM API calls (1-300)",
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log format: json or text"
    )

    # Input validation settings
    max_prd_length: int = Field(
        default=100000,
        ge=1000,
        le=500000,
        description="Maximum PRD text length in characters (1000-500000)",
    )


# Singleton for convenience (thread-safe with double-checked locking)
_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Get or create Settings instance."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings
