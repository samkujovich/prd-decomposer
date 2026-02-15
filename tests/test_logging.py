"""Tests for structured logging with correlation IDs."""

import json
import logging

from prd_decomposer.config import Settings
from prd_decomposer.logging import (
    StructuredFormatter,
    correlation_id,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for the JSON log formatter."""

    def test_json_format_produces_valid_json(self):
        """Verify StructuredFormatter outputs valid JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_format_includes_correlation_id(self):
        """Verify StructuredFormatter includes correlation_id from context."""
        formatter = StructuredFormatter()
        token = correlation_id.set("req-abc123")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Correlated message",
                args=None,
                exc_info=None,
            )

            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed["correlation_id"] == "req-abc123"
        finally:
            correlation_id.reset(token)

    def test_json_format_includes_module_and_function(self):
        """Verify StructuredFormatter includes module and function name."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="server.py",
            lineno=42,
            msg="Warning",
            args=None,
            exc_info=None,
        )
        record.funcName = "analyze_prd"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["module"] == "server"
        assert parsed["function"] == "analyze_prd"

    def test_json_format_with_exception_info(self):
        """Verify StructuredFormatter handles exc_info without crashing."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=None,
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Error occurred"
        assert parsed["level"] == "ERROR"

    def test_json_format_handles_getMessage_failure(self):
        """Verify formatter returns valid JSON when getMessage() raises."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Value is %d",
            args=("not_an_int",),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "ERROR"
        assert "Value is %d" in parsed["message"]

    def test_json_format_empty_correlation_id_when_unset(self):
        """Verify correlation_id is empty string when not set."""
        formatter = StructuredFormatter()
        # Ensure no correlation_id is set in this context
        token = correlation_id.set("")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="No correlation",
                args=None,
                exc_info=None,
            )

            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed["correlation_id"] == ""
        finally:
            correlation_id.reset(token)


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_json_format(self):
        """Verify setup_logging configures JSON formatter."""
        settings = Settings(log_level="DEBUG", log_format="json")
        logger = setup_logging(settings)

        assert logger.level == logging.DEBUG
        # Should have at least one handler with StructuredFormatter
        assert any(
            isinstance(h.formatter, StructuredFormatter) for h in logger.handlers
        )

    def test_setup_logging_text_format(self):
        """Verify setup_logging configures text formatter for text format."""
        settings = Settings(log_level="WARNING", log_format="text")
        logger = setup_logging(settings)

        assert logger.level == logging.WARNING
        # Should have handlers but NOT with StructuredFormatter
        assert len(logger.handlers) > 0
        assert not any(
            isinstance(h.formatter, StructuredFormatter) for h in logger.handlers
        )

    def test_setup_logging_returns_named_logger(self):
        """Verify setup_logging returns logger named prd_decomposer."""
        settings = Settings()
        logger = setup_logging(settings)

        assert logger.name == "prd_decomposer"

    def test_setup_logging_clears_duplicate_handlers(self):
        """Verify calling setup_logging twice does not duplicate handlers."""
        settings = Settings(log_format="json")
        setup_logging(settings)
        logger = setup_logging(settings)

        assert len(logger.handlers) == 1
