"""Tests for the error handling framework.

This module contains tests for the error handling framework, including:
- Base error classes
- Error severity and category classification
- Error reporting and logging
- Error handler functionality
- Retry mechanisms
- Fallback mechanisms
"""

import logging
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.infrastructure.error import (
    FridayError, SystemError, NetworkError, DatabaseError, APIError, DataError,
    AuthenticationError, ConfigurationError, RetryableError, RetryExhaustedError,
    ErrorSeverity, ErrorCategory, create_error_report, log_error, ErrorHandler,
    RetryStrategy, RetryPolicy, retry, RetryPolicies, RetryPolicyFactory,
    FallbackStrategy, fallback_to_default, fallback_to_alternative, CircuitBreaker,
    FallbackChain, with_fallback_chain
)
from src.infrastructure.error.error_codes import ErrorCode


class TestBaseErrorClasses(unittest.TestCase):
    """Tests for the base error classes."""

    def test_friday_error_initialization(self):
        """Test that FridayError initializes correctly."""
        error = FridayError(
            "Test error message",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            troubleshooting_guidance="Test guidance",
            context={"test_key": "test_value"},
            error_code=ErrorCode.SYSTEM_GENERAL_ERROR
        )

        self.assertEqual(str(error), "Test error message")
        self.assertEqual(error.severity, ErrorSeverity.MEDIUM)
        self.assertEqual(error.category, ErrorCategory.SYSTEM)
        self.assertEqual(error.troubleshooting_guidance, "Test guidance")
        self.assertEqual(error.context, {"test_key": "test_value"})
        self.assertEqual(error.error_code, ErrorCode.SYSTEM_GENERAL_ERROR)
        self.assertIsNotNone(error.timestamp)
        self.assertIsNotNone(error.stack_trace)
        self.assertIsNotNone(error.function_name)

    def test_friday_error_to_dict(self):
        """Test that FridayError.to_dict() returns the correct dictionary."""
        error = FridayError(
            "Test error message",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            troubleshooting_guidance="Test guidance",
            context={"test_key": "test_value"},
            error_code=ErrorCode.SYSTEM_GENERAL_ERROR
        )

        error_dict = error.to_dict()

        self.assertEqual(error_dict["error_type"], "FridayError")
        self.assertEqual(error_dict["message"], "Test error message")
        self.assertEqual(error_dict["severity"], "MEDIUM")
        self.assertEqual(error_dict["category"], "SYSTEM")
        self.assertEqual(error_dict["troubleshooting_guidance"], "Test guidance")
        self.assertEqual(error_dict["context"], {"test_key": "test_value"})
        self.assertEqual(error_dict["error_code"], ErrorCode.SYSTEM_GENERAL_ERROR.value)

    def test_specialized_error_classes(self):
        """Test that specialized error classes initialize correctly."""
        # Test SystemError
        system_error = SystemError("System error")
        self.assertEqual(system_error.category, ErrorCategory.SYSTEM)
        self.assertIsNotNone(system_error.troubleshooting_guidance)

        # Test NetworkError
        network_error = NetworkError("Network error")
        self.assertEqual(network_error.category, ErrorCategory.NETWORK)
        self.assertIsNotNone(network_error.troubleshooting_guidance)

        # Test DatabaseError
        db_error = DatabaseError("Database error")
        self.assertEqual(db_error.category, ErrorCategory.DATABASE)
        self.assertIsNotNone(db_error.troubleshooting_guidance)

        # Test APIError
        api_error = APIError("API error")
        self.assertEqual(api_error.category, ErrorCategory.API)
        self.assertIsNotNone(api_error.troubleshooting_guidance)

        # Test DataError
        data_error = DataError("Data error")
        self.assertEqual(data_error.category, ErrorCategory.DATA)
        self.assertIsNotNone(data_error.troubleshooting_guidance)

        # Test AuthenticationError
        auth_error = AuthenticationError("Authentication error")
        self.assertEqual(auth_error.category, ErrorCategory.AUTHENTICATION)
        self.assertIsNotNone(auth_error.troubleshooting_guidance)

        # Test ConfigurationError
        config_error = ConfigurationError("Configuration error")
        self.assertEqual(config_error.category, ErrorCategory.CONFIGURATION)
        self.assertIsNotNone(config_error.troubleshooting_guidance)

        # Test RetryableError
        retry_error = RetryableError("Retryable error")
        self.assertIsNotNone(retry_error.troubleshooting_guidance)

        # Test RetryExhaustedError
        retry_exhausted = RetryExhaustedError("Retry exhausted")
        self.assertEqual(retry_exhausted.category, ErrorCategory.RETRY)
        self.assertIsNotNone(retry_exhausted.troubleshooting_guidance)


class TestErrorReporting(unittest.TestCase):
    """Tests for error reporting functions."""

    def test_create_error_report_friday_error(self):
        """Test create_error_report with a FridayError."""
        error = FridayError("Test error")
        report = create_error_report(error)

        self.assertEqual(report["error_type"], "FridayError")
        self.assertEqual(report["message"], "Test error")
        self.assertIn("timestamp", report)

    def test_create_error_report_standard_exception(self):
        """Test create_error_report with a standard exception."""
        error = ValueError("Test value error")
        report = create_error_report(error)

        self.assertEqual(report["error_type"], "ValueError")
        self.assertEqual(report["message"], "Test value error")
        self.assertIn("timestamp", report)
        self.assertIn("stack_trace", report)

    @patch("logging.Logger.log")
    @patch("logging.Logger.debug")
    def test_log_error_friday_error(self, mock_debug, mock_log):
        """Test log_error with a FridayError."""
        error = FridayError("Test error", severity=ErrorSeverity.HIGH)
        logger = logging.getLogger("test")
        log_error(error, logger)

        # Check that the error was logged at the appropriate level
        mock_log.assert_called_once()
        # The first argument should be the log level (ERROR for HIGH severity)
        self.assertEqual(mock_log.call_args[0][0], logging.ERROR)

    @patch("logging.Logger.error")
    @patch("logging.Logger.debug")
    def test_log_error_standard_exception(self, mock_debug, mock_error):
        """Test log_error with a standard exception."""
        error = ValueError("Test value error")
        logger = logging.getLogger("test")
        log_error(error, logger)

        # Check that the error was logged
        mock_error.assert_called_once()
        self.assertIn("ValueError", mock_error.call_args[0][0])


class TestErrorHandler(unittest.TestCase):
    """Tests for the ErrorHandler class."""

    def test_error_handler_as_context_manager(self):
        """Test ErrorHandler as a context manager."""
        handler = MagicMock()
        logger = logging.getLogger("test")
        logger.error = MagicMock()

        # Test with no exception
        with ErrorHandler(ValueError, handler=handler, logger=logger):
            pass
        handler.assert_not_called()

        # Test with matching exception
        with self.assertRaises(ValueError):
            with ErrorHandler(ValueError, handler=handler, logger=logger, reraise=True):
                raise ValueError("Test error")
        handler.assert_called_once()

        # Test with non-matching exception
        handler.reset_mock()
        with self.assertRaises(TypeError):
            with ErrorHandler(ValueError, handler=handler, logger=logger):
                raise TypeError("Test error")
        handler.assert_not_called()

        # Test with suppressed exception
        handler.reset_mock()
        with ErrorHandler(ValueError, handler=handler, logger=logger, reraise=False):
            raise ValueError("Test error")
        handler.assert_called_once()

    def test_error_handler_as_decorator(self):
        """Test ErrorHandler as a decorator."""
        handler = MagicMock()
        logger = logging.getLogger("test")
        logger.error = MagicMock()

        # Test with no exception
        @ErrorHandler(ValueError, handler=handler, logger=logger)
        def func_no_error():
            return "success"

        self.assertEqual(func_no_error(), "success")
        handler.assert_not_called()

        # Test with matching exception
        @ErrorHandler(ValueError, handler=handler, logger=logger, reraise=True)
        def func_with_error():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            func_with_error()
        handler.assert_called_once()

        # Test with suppressed exception
        handler.reset_mock()
        @ErrorHandler(ValueError, handler=handler, logger=logger, reraise=False)
        def func_with_suppressed_error():
            raise ValueError("Test error")

        func_with_suppressed_error()
        handler.assert_called_once()

    def test_error_handler_reraise_as(self):
        """Test ErrorHandler with reraise_as option."""
        with self.assertRaises(DatabaseError):
            with ErrorHandler(ValueError, reraise=True, reraise_as=DatabaseError):
                raise ValueError("Test error")


class TestRetryMechanisms(unittest.TestCase):
    """Tests for retry mechanisms."""

    def test_retry_decorator_success(self):
        """Test retry decorator with successful execution."""
        mock_func = MagicMock(return_value="success")
        decorated_func = retry()(mock_func)

        result = decorated_func("arg1", kwarg1="value1")

        self.assertEqual(result, "success")
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_decorator_with_retries(self):
        """Test retry decorator with retries before success."""
        mock_func = MagicMock(side_effect=[ValueError("Error 1"), ValueError("Error 2"), "success"])
        decorated_func = retry(max_attempts=3)(mock_func)

        result = decorated_func()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)

    def test_retry_decorator_exhausted(self):
        """Test retry decorator with exhausted retries."""
        mock_func = MagicMock(side_effect=ValueError("Test error"))
        decorated_func = retry(max_attempts=3)(mock_func)

        with self.assertRaises(RetryExhaustedError):
            decorated_func()

        self.assertEqual(mock_func.call_count, 3)

    def test_retry_policies(self):
        """Test predefined retry policies."""
        # Test order submission policy
        order_policy = RetryPolicies.order_submission()
        self.assertEqual(order_policy.max_attempts, 3)
        self.assertEqual(order_policy.strategy, RetryStrategy.EXPONENTIAL_BACKOFF)

        # Test market data policy
        market_data_policy = RetryPolicies.market_data()
        self.assertEqual(market_data_policy.max_attempts, 5)
        self.assertEqual(market_data_policy.strategy, RetryStrategy.EXPONENTIAL_BACKOFF)

    def test_retry_policy_factory(self):
        """Test RetryPolicyFactory."""
        # Test creating from predefined type
        policy = RetryPolicyFactory.create_from_type("order_submission", max_attempts=5)
        self.assertEqual(policy.max_attempts, 5)  # Overridden value
        self.assertEqual(policy.strategy, RetryStrategy.EXPONENTIAL_BACKOFF)  # Default value

        # Test creating from config
        config = {
            "max_attempts": 4,
            "strategy": "fixed_delay",
            "base_delay": 2.0,
            "max_delay": 10.0,
            "timeout": 30.0
        }
        policy = RetryPolicyFactory.create_from_config(config)
        self.assertEqual(policy.max_attempts, 4)
        self.assertEqual(policy.strategy, RetryStrategy.FIXED_DELAY)
        self.assertEqual(policy.base_delay, 2.0)
        self.assertEqual(policy.max_delay, 10.0)
        self.assertEqual(policy.timeout, 30.0)


class TestFallbackMechanisms(unittest.TestCase):
    """Tests for fallback mechanisms."""

    def test_fallback_to_default(self):
        """Test fallback_to_default decorator."""
        @fallback_to_default(default_value="fallback")
        def func_with_error():
            raise ValueError("Test error")

        result = func_with_error()
        self.assertEqual(result, "fallback")

    def test_fallback_to_alternative(self):
        """Test fallback_to_alternative decorator."""
        def alternative_func(exception=None):
            return f"alternative: {str(exception)}"

        @fallback_to_alternative(alternative_func, pass_exception=True)
        def func_with_error():
            raise ValueError("Test error")

        result = func_with_error()
        self.assertEqual(result, "alternative: Test error")

    def test_circuit_breaker(self):
        """Test CircuitBreaker class."""
        # Create a circuit breaker with a low threshold for testing
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
            monitored_exceptions=ValueError
        )

        # Function that will fail
        mock_func = MagicMock(side_effect=ValueError("Test error"))
        decorated_func = cb(mock_func)

        # First call - should fail but circuit still closed
        with self.assertRaises(ValueError):
            decorated_func()

        # Second call - should fail and open the circuit
        with self.assertRaises(ValueError):
            decorated_func()

        # Third call - circuit is open, should raise CircuitBreakerError
        with self.assertRaises(Exception) as context:
            decorated_func()
        self.assertIn("Circuit breaker is open", str(context.exception))

    def test_fallback_chain(self):
        """Test FallbackChain class."""
        # Create functions for the chain
        primary = MagicMock(side_effect=ValueError("Primary error"))
        fallback1 = MagicMock(side_effect=ValueError("Fallback 1 error"))
        fallback2 = MagicMock(return_value="fallback 2 success")

        # Create the chain
        chain = FallbackChain(
            primary_func=primary,
            fallback_funcs=[fallback1, fallback2],
            exceptions_to_catch=ValueError
        )

        # Execute the chain
        result = chain.execute("arg1", kwarg1="value1")

        # Check the result
        self.assertEqual(result, "fallback 2 success")

        # Check that all functions were called with the same arguments
        primary.assert_called_once_with("arg1", kwarg1="value1")
        fallback1.assert_called_once_with("arg1", kwarg1="value1")
        fallback2.assert_called_once_with("arg1", kwarg1="value1")

    def test_with_fallback_chain_decorator(self):
        """Test with_fallback_chain decorator."""
        # Create fallback functions
        fallback1 = MagicMock(side_effect=ValueError("Fallback 1 error"))
        fallback2 = MagicMock(return_value="fallback 2 success")

        # Create decorated function
        @with_fallback_chain([fallback1, fallback2], exceptions_to_catch=ValueError)
        def primary_func(arg1, kwarg1=None):
            raise ValueError("Primary error")

        # Call the decorated function
        result = primary_func("arg1", kwarg1="value1")

        # Check the result
        self.assertEqual(result, "fallback 2 success")

        # Check that all functions were called with the same arguments
        fallback1.assert_called_once_with("arg1", kwarg1="value1")
        fallback2.assert_called_once_with("arg1", kwarg1="value1")


if __name__ == "__main__":
    unittest.main()