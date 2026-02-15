"""Structured logging with correlation IDs for request tracing."""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

from prd_decomposer.config import Settings

# Correlation ID for request tracing across tool calls
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class StructuredFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
            "module": record.module,
            "function": record.funcName,
        })


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure logging based on settings.

    Returns the 'prd_decomposer' logger with appropriate handler and formatter.
    """
    logger = logging.getLogger("prd_decomposer")
    logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers to avoid duplicates on repeated calls
    logger.handlers.clear()

    handler = logging.StreamHandler()

    if settings.log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )

    logger.addHandler(handler)

    return logger
