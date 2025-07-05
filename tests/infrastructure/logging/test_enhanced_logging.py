"""Tests for the enhanced logging framework.

This module contains tests for the enhanced logging framework, including:
- Structured logging
- Log formatting
- Log enrichment
- Correlation ID tracking
- Execution time logging
- Function call logging
"""

import json
import logging
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from src.infrastructure.logging.enhanced import (
    StructuredLogger, JSONFormatter, DetailedFormatter, LogEnricher,
    LogEnricherFilter, CorrelationIdFilter, with_correlation_id,
    log_execution_time, log_function_call, setup_enhanced_logging,
    get_correlation_id, set_correlation_id, clear_correlation_id,
    structured_log, debug, info, warning, error, critical, LogContext
)


class TestStructuredLogging(unittest.TestCase):
    """Tests for structured logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.logger = StructuredLogger("test_logger")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def test_structured_logger(self):
        """Test that StructuredLogger correctly handles structured data."""
        self.logger.info("Test message", extra={"structured_data": {"key": "value"}})
        log_output = self.log_output.getvalue()
        self.assertIn("Test message", log_output)
        self.assertIn("key", log_output)
        self.assertIn("value", log_output)

    def test_json_formatter(self):
        """Test that JSONFormatter correctly formats log records as JSON."""
        formatter = JSONFormatter()
        self.handler.setFormatter(formatter)

        self.logger.info("Test JSON message", extra={"structured_data": {"key": "value"}})
        log_output = self.log_output.getvalue()

        # Verify that the output is valid JSON
        log_data = json.loads(log_output)
        self.assertEqual(log_data["message"], "Test JSON message")
        self.assertEqual(log_data["key"], "value")
        self.assertEqual(log_data["level"], "INFO")

    def test_detailed_formatter(self):
        """Test that DetailedFormatter correctly formats log records."""
        formatter = DetailedFormatter()
        self.handler.setFormatter(formatter)

        self.logger.info("Test detailed message", extra={"structured_data": {"key": "value"}})
        log_output = self.log_output.getvalue()

        self.assertIn("Test detailed message", log_output)
        self.assertIn("key=value", log_output)
        self.assertIn("INFO", log_output)
        self.assertIn("test_logger", log_output)


class TestLogEnrichment(unittest.TestCase):
    """Tests for log enrichment functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setFormatter(JSONFormatter())
        self.logger = logging.getLogger("test_enrichment")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def test_log_enricher(self):
        """Test that LogEnricher adds static context to logs."""
        enricher = LogEnricher({"app_name": "test_app", "environment": "test"})
        enricher_filter = LogEnricherFilter(enricher)
        self.logger.addFilter(enricher_filter)

        self.logger.info("Test enriched message")
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(log_data["app_name"], "test_app")
        self.assertEqual(log_data["environment"], "test")

    def test_log_enricher_with_dynamic_context(self):
        """Test that LogEnricher adds dynamic context to logs."""
        def get_dynamic_context():
            return {"request_id": "12345", "timestamp": "2023-01-01T00:00:00Z"}

        enricher = LogEnricher({"app_name": "test_app"}, get_dynamic_context)
        enricher_filter = LogEnricherFilter(enricher)
        self.logger.addFilter(enricher_filter)

        self.logger.info("Test dynamic enriched message")
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(log_data["app_name"], "test_app")
        self.assertEqual(log_data["request_id"], "12345")
        self.assertEqual(log_data["timestamp"], "2023-01-01T00:00:00Z")


class TestCorrelationId(unittest.TestCase):
    """Tests for correlation ID functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setFormatter(JSONFormatter())
        self.logger = logging.getLogger("test_correlation")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addFilter(CorrelationIdFilter())

        # Clear any existing correlation ID
        clear_correlation_id()

    def test_correlation_id_filter(self):
        """Test that CorrelationIdFilter adds correlation ID to logs."""
        set_correlation_id("test-correlation-id")

        self.logger.info("Test correlation message")
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(log_data["correlation_id"], "test-correlation-id")

    def test_with_correlation_id_decorator(self):
        """Test that with_correlation_id decorator sets correlation ID."""
        @with_correlation_id()
        def test_function():
            self.logger.info("Test decorator message")
            return get_correlation_id()

        correlation_id = test_function()
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertIsNotNone(correlation_id)
        self.assertEqual(log_data["correlation_id"], correlation_id)

    def test_with_correlation_id_decorator_with_custom_id(self):
        """Test that with_correlation_id decorator uses custom ID."""
        @with_correlation_id(correlation_id="custom-id")
        def test_function():
            self.logger.info("Test custom decorator message")
            return get_correlation_id()

        correlation_id = test_function()
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(correlation_id, "custom-id")
        self.assertEqual(log_data["correlation_id"], "custom-id")


class TestLoggingDecorators(unittest.TestCase):
    """Tests for logging decorators."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setFormatter(JSONFormatter())
        self.logger = logging.getLogger("test_decorators")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def test_log_execution_time(self):
        """Test that log_execution_time decorator logs execution time."""
        @log_execution_time(self.logger)
        def test_function(arg1, arg2=None):
            return arg1 + (arg2 or 0)

        result = test_function(1, arg2=2)
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(result, 3)
        self.assertEqual(log_data["function"], "test_function")
        self.assertIn("execution_time_ms", log_data)
        self.assertIn("Executed in", log_data["message"])

    def test_log_function_call(self):
        """Test that log_function_call decorator logs function calls."""
        @log_function_call(self.logger)
        def test_function(arg1, arg2=None):
            return arg1 + (arg2 or 0)

        result = test_function(1, arg2=2)
        log_output = self.log_output.getvalue()
        log_data = json.loads(log_output)

        self.assertEqual(result, 3)
        self.assertEqual(log_data["function"], "test_function")
        self.assertEqual(log_data["args"], [1])
        self.assertEqual(log_data["kwargs"], {"arg2": 2})
        self.assertEqual(log_data["result"], 3)


class TestSetupEnhancedLogging(unittest.TestCase):
    """Tests for setup_enhanced_logging function."""

    @patch("logging.getLogger")
    @patch("logging.StreamHandler")
    @patch("logging.FileHandler")
    def test_setup_enhanced_logging(self, mock_file_handler, mock_stream_handler, mock_get_logger):
        """Test that setup_enhanced_logging configures logging correctly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_stream_handler.return_value = MagicMock()
        mock_file_handler.return_value = MagicMock()

        # Test with default parameters
        setup_enhanced_logging()
        mock_logger.setLevel.assert_called_with(logging.INFO)
        mock_logger.addHandler.assert_called()
        mock_logger.addFilter.assert_called()

        # Test with custom parameters
        mock_logger.reset_mock()
        setup_enhanced_logging(
            json_output=True,
            log_file="test.log",
            level=logging.DEBUG,
            enable_correlation_id=True,
            static_context={"app": "test"}
        )
        mock_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_file_handler.assert_called_with("test.log")
        self.assertEqual(mock_logger.addHandler.call_count, 2)  # Stream and file handlers
        self.assertEqual(mock_logger.addFilter.call_count, 2)  # Enricher and correlation ID filters


class TestStructuredLogFunctions(unittest.TestCase):
    """Tests for structured log functions."""

    @patch("logging.getLogger")
    def test_structured_log_functions(self, mock_get_logger):
        """Test that structured log functions correctly log messages."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test structured_log
        structured_log(logging.INFO, "Test message", key="value")
        mock_logger.log.assert_called_with(
            logging.INFO, "Test message", extra={"structured_data": {"key": "value"}}
        )

        # Test debug
        debug("Debug message", key="value")
        mock_logger.debug.assert_called_with(
            "Debug message", extra={"structured_data": {"key": "value"}}
        )

        # Test info
        info("Info message", key="value")
        mock_logger.info.assert_called_with(
            "Info message", extra={"structured_data": {"key": "value"}}
        )

        # Test warning
        warning("Warning message", key="value")
        mock_logger.warning.assert_called_with(
            "Warning message", extra={"structured_data": {"key": "value"}}
        )

        # Test error
        error("Error message", key="value")
        mock_logger.error.assert_called_with(
            "Error message", extra={"structured_data": {"key": "value"}}
        )

        # Test critical
        critical("Critical message", key="value")
        mock_logger.critical.assert_called_with(
            "Critical message", extra={"structured_data": {"key": "value"}}
        )


class TestLogContext(unittest.TestCase):
    """Tests for LogContext context manager."""

    @patch("src.infrastructure.logging.enhanced._context_vars")
    def test_log_context(self, mock_context_vars):
        """Test that LogContext correctly manages context."""
        mock_context_vars.copy.return_value = {}

        with LogContext(key1="value1", key2="value2"):
            # Check that context was set
            self.assertEqual(mock_context_vars.__setitem__.call_count, 2)
            mock_context_vars.__setitem__.assert_any_call("key1", "value1")
            mock_context_vars.__setitem__.assert_any_call("key2", "value2")

        # Check that context was restored
        mock_context_vars.clear.assert_called_once()
        self.assertEqual(mock_context_vars.update.call_count, 1)


if __name__ == "__main__":
    unittest.main()