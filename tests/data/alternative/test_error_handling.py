"""Tests for the error handling module.

This module contains unit tests for the error handling utilities used in alternative data integration.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import time
from datetime import datetime, timedelta

# Import the module to test
from src.data.alternative.error_handling import (
    AlternativeDataError, DataSourceUnavailableError, 
    DataProcessingError, DataValidationError,
    retry, fallback_to_cache, log_execution_time,
    validate_data, handle_alternative_data_errors,
    create_error_report
)

class TestErrorHandlingDecorators(unittest.TestCase):
    """Test cases for the error handling decorators."""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function execution."""
        # Arrange
        mock_func = MagicMock(return_value="success")
        decorated_func = retry(max_retries=3, retry_delay=0.01)(mock_func)
        
        # Act
        result = decorated_func("arg1", kwarg1="value1")
        
        # Assert
        self.assertEqual(result, "success")
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_retry_decorator_failure_then_success(self):
        """Test retry decorator with initial failure then success."""
        # Arrange
        side_effects = [Exception("Error"), Exception("Error"), "success"]
        mock_func = MagicMock(side_effect=side_effects)
        decorated_func = retry(max_retries=3, retry_delay=0.01)(mock_func)
        
        # Act
        result = decorated_func()
        
        # Assert
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_retry_decorator_all_failures(self):
        """Test retry decorator with all attempts failing."""
        # Arrange
        mock_func = MagicMock(side_effect=Exception("Persistent error"))
        decorated_func = retry(max_retries=2, retry_delay=0.01)(mock_func)
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            decorated_func()
        
        self.assertEqual(str(context.exception), "Persistent error")
        self.assertEqual(mock_func.call_count, 3)  # Initial + 2 retries
    
    def test_retry_decorator_specific_exceptions(self):
        """Test retry decorator with specific exception types."""
        # Arrange
        mock_func = MagicMock(side_effect=[ValueError("Value error"), "success"])
        decorated_func = retry(max_retries=2, retry_delay=0.01, exceptions=(ValueError,))(mock_func)
        
        # Act
        result = decorated_func()
        
        # Assert
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
    
    def test_retry_decorator_unspecified_exception(self):
        """Test retry decorator with an exception type not specified for retry."""
        # Arrange
        mock_func = MagicMock(side_effect=TypeError("Type error"))
        decorated_func = retry(max_retries=2, retry_delay=0.01, exceptions=(ValueError,))(mock_func)
        
        # Act & Assert
        with self.assertRaises(TypeError):
            decorated_func()
        
        # Should not retry on TypeError
        mock_func.assert_called_once()
    
    @patch('src.data.alternative.error_handling.find_one')
    @patch('src.data.alternative.error_handling.update_one')
    def test_fallback_to_cache_success(self, mock_update, mock_find):
        """Test fallback_to_cache decorator with successful function execution."""
        # Arrange
        mock_func = MagicMock(return_value={"data": "fresh_data"})
        mock_query_func = MagicMock(return_value={"symbol": "AAPL"})
        decorated_func = fallback_to_cache("test_collection", mock_query_func)(mock_func)
        
        # Act
        result = decorated_func("AAPL")
        
        # Assert
        self.assertEqual(result, {"data": "fresh_data"})
        mock_func.assert_called_once_with("AAPL")
        mock_query_func.assert_called_once_with("AAPL")
        mock_update.assert_called_once()
        mock_find.assert_not_called()
    
    @patch('src.data.alternative.error_handling.find_one')
    @patch('src.data.alternative.error_handling.update_one')
    def test_fallback_to_cache_failure_with_cache(self, mock_update, mock_find):
        """Test fallback_to_cache decorator with function failure and valid cache."""
        # Arrange
        mock_func = MagicMock(side_effect=Exception("API error"))
        mock_query_func = MagicMock(return_value={"symbol": "AAPL"})
        mock_find.return_value = {
            "data": {"cached": "data"},
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        decorated_func = fallback_to_cache("test_collection", mock_query_func)(mock_func)
        
        # Act
        result = decorated_func("AAPL")
        
        # Assert
        self.assertEqual(result, {"cached": "data"})
        mock_func.assert_called_once_with("AAPL")
        mock_query_func.assert_called_with("AAPL")
        self.assertEqual(mock_query_func.call_count, 2)  # Once for potential cache, once for fallback
        mock_update.assert_not_called()
        mock_find.assert_called_once()
    
    @patch('src.data.alternative.error_handling.find_one')
    @patch('src.data.alternative.error_handling.update_one')
    def test_fallback_to_cache_failure_no_cache(self, mock_update, mock_find):
        """Test fallback_to_cache decorator with function failure and no valid cache."""
        # Arrange
        mock_func = MagicMock(side_effect=Exception("API error"))
        mock_query_func = MagicMock(return_value={"symbol": "AAPL"})
        mock_find.return_value = None  # No cache found
        decorated_func = fallback_to_cache("test_collection", mock_query_func)(mock_func)
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            decorated_func("AAPL")
        
        self.assertEqual(str(context.exception), "API error")
        mock_func.assert_called_once_with("AAPL")
        mock_query_func.assert_called_with("AAPL")
        mock_update.assert_not_called()
        mock_find.assert_called_once()
    
    @patch('src.data.alternative.error_handling.time.time')
    def test_log_execution_time(self, mock_time):
        """Test log_execution_time decorator."""
        # Arrange
        mock_time.side_effect = [100.0, 105.5]  # Start time, end time
        mock_func = MagicMock(return_value="result")
        decorated_func = log_execution_time(mock_func)
        
        # Act
        with patch('src.data.alternative.error_handling.logger.info') as mock_logger:
            result = decorated_func("arg1", kwarg1="value1")
        
        # Assert
        self.assertEqual(result, "result")
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
        mock_logger.assert_called_once()
        # Check that the log message contains the execution time
        self.assertIn("5.50", mock_logger.call_args[0][0])
    
    def test_validate_data_valid(self):
        """Test validate_data decorator with valid data."""
        # Arrange
        mock_func = MagicMock(return_value=[1, 2, 3])
        validation_func = lambda data: len(data) > 0
        decorated_func = validate_data(validation_func)(mock_func)
        
        # Act
        result = decorated_func()
        
        # Assert
        self.assertEqual(result, [1, 2, 3])
        mock_func.assert_called_once()
    
    def test_validate_data_invalid(self):
        """Test validate_data decorator with invalid data."""
        # Arrange
        mock_func = MagicMock(return_value=[])
        validation_func = lambda data: len(data) > 0
        decorated_func = validate_data(validation_func, "Empty data")(mock_func)
        
        # Act & Assert
        with self.assertRaises(DataValidationError) as context:
            decorated_func()
        
        self.assertIn("Empty data", str(context.exception))
        mock_func.assert_called_once()
    
    def test_handle_alternative_data_errors_success(self):
        """Test handle_alternative_data_errors decorator with successful function."""
        # Arrange
        mock_func = MagicMock(return_value="success")
        decorated_func = handle_alternative_data_errors(mock_func)
        
        # Act
        result = decorated_func()
        
        # Assert
        self.assertEqual(result, "success")
        mock_func.assert_called_once()
    
    def test_handle_alternative_data_errors_data_source_error(self):
        """Test handle_alternative_data_errors with DataSourceUnavailableError."""
        # Arrange
        mock_func = MagicMock(side_effect=DataSourceUnavailableError("API down"))
        decorated_func = handle_alternative_data_errors(mock_func)
        
        # Act & Assert
        with patch('src.data.alternative.error_handling.logger.error') as mock_logger:
            with self.assertRaises(DataSourceUnavailableError):
                decorated_func()
        
        mock_func.assert_called_once()
        mock_logger.assert_called_once()
    
    def test_handle_alternative_data_errors_processing_error(self):
        """Test handle_alternative_data_errors with DataProcessingError."""
        # Arrange
        mock_func = MagicMock(side_effect=DataProcessingError("Processing failed"))
        decorated_func = handle_alternative_data_errors(mock_func)
        
        # Act & Assert
        with patch('src.data.alternative.error_handling.logger.error') as mock_logger:
            with self.assertRaises(DataProcessingError):
                decorated_func()
        
        mock_func.assert_called_once()
        mock_logger.assert_called_once()
    
    def test_handle_alternative_data_errors_validation_error(self):
        """Test handle_alternative_data_errors with DataValidationError."""
        # Arrange
        mock_func = MagicMock(side_effect=DataValidationError("Invalid data"))
        decorated_func = handle_alternative_data_errors(mock_func)
        
        # Act & Assert
        with patch('src.data.alternative.error_handling.logger.error') as mock_logger:
            with self.assertRaises(DataValidationError):
                decorated_func()
        
        mock_func.assert_called_once()
        mock_logger.assert_called_once()
    
    def test_handle_alternative_data_errors_generic_error(self):
        """Test handle_alternative_data_errors with a generic exception."""
        # Arrange
        mock_func = MagicMock(side_effect=ValueError("Some value error"))
        decorated_func = handle_alternative_data_errors(mock_func)
        
        # Act & Assert
        with patch('src.data.alternative.error_handling.logger.error') as mock_logger:
            with self.assertRaises(AlternativeDataError):
                decorated_func()
        
        mock_func.assert_called_once()
        mock_logger.assert_called_once()


class TestErrorReporting(unittest.TestCase):
    """Test cases for error reporting functions."""
    
    @patch('src.data.alternative.error_handling.insert_one')
    def test_create_error_report(self, mock_insert):
        """Test creating an error report."""
        # Arrange
        error = ValueError("Test error")
        function_name = "test_function"
        args = ("arg1", "arg2")
        kwargs = {"kwarg1": "value1"}
        
        # Act
        with patch('src.data.alternative.error_handling.logger.info') as mock_logger_info:
            with patch('src.data.alternative.error_handling.logger.error') as mock_logger_error:
                report = create_error_report(error, function_name, args, kwargs)
        
        # Assert
        self.assertEqual(report['error_type'], "ValueError")
        self.assertEqual(report['error_message'], "Test error")
        self.assertEqual(report['function_name'], function_name)
        self.assertIn("arg1", report['args'])
        self.assertIn("kwarg1", report['kwargs'])
        mock_insert.assert_called_once()
        mock_logger_info.assert_called_once()
    
    @patch('src.data.alternative.error_handling.insert_one')
    def test_create_error_report_mongodb_error(self, mock_insert):
        """Test creating an error report when MongoDB insert fails."""
        # Arrange
        error = ValueError("Test error")
        function_name = "test_function"
        args = ("arg1", "arg2")
        kwargs = {"kwarg1": "value1"}
        mock_insert.side_effect = Exception("MongoDB error")
        
        # Act
        with patch('src.data.alternative.error_handling.logger.info') as mock_logger_info:
            with patch('src.data.alternative.error_handling.logger.error') as mock_logger_error:
                report = create_error_report(error, function_name, args, kwargs)
        
        # Assert
        self.assertEqual(report['error_type'], "ValueError")
        mock_insert.assert_called_once()
        mock_logger_info.assert_not_called()
        mock_logger_error.assert_called_once()


class TestExceptionClasses(unittest.TestCase):
    """Test cases for custom exception classes."""
    
    def test_alternative_data_error(self):
        """Test AlternativeDataError exception."""
        # Act
        error = AlternativeDataError("Test error")
        
        # Assert
        self.assertEqual(str(error), "Test error")
        self.assertIsInstance(error, Exception)
    
    def test_data_source_unavailable_error(self):
        """Test DataSourceUnavailableError exception."""
        # Act
        error = DataSourceUnavailableError("API unavailable")
        
        # Assert
        self.assertEqual(str(error), "API unavailable")
        self.assertIsInstance(error, AlternativeDataError)
    
    def test_data_processing_error(self):
        """Test DataProcessingError exception."""
        # Act
        error = DataProcessingError("Processing failed")
        
        # Assert
        self.assertEqual(str(error), "Processing failed")
        self.assertIsInstance(error, AlternativeDataError)
    
    def test_data_validation_error(self):
        """Test DataValidationError exception."""
        # Act
        error = DataValidationError("Invalid data")
        
        # Assert
        self.assertEqual(str(error), "Invalid data")
        self.assertIsInstance(error, AlternativeDataError)


if __name__ == '__main__':
    unittest.main()