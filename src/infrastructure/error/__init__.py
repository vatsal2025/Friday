

"""Error handling framework for the Friday AI Trading System.

This module provides a comprehensive error handling framework including:
- Base error classes with enhanced information
- Error severity and category classification
- Detailed error reporting and logging
- Error handler for consistent error management
"""

import inspect
import logging
import traceback
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, Set, TypeVar, cast

from src.infrastructure.error.error_codes import ErrorCode

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Classification of error severity levels."""
    CRITICAL = auto()  # System cannot function, immediate attention required
    HIGH = auto()      # Major functionality impacted, urgent attention needed
    MEDIUM = auto()    # Partial functionality impacted, attention needed soon
    LOW = auto()       # Minor issue, can be addressed in regular maintenance
    INFO = auto()      # Informational only, no immediate action required


class ErrorCategory(Enum):
    """Classification of error categories."""
    SYSTEM = auto()        # System-level errors (OS, hardware, etc.)
    NETWORK = auto()       # Network-related errors
    DATABASE = auto()      # Database-related errors
    API = auto()           # API-related errors
    DATA = auto()          # Data validation, parsing, or integrity errors
    VALIDATION = auto()    # Data validation errors
    AUTHENTICATION = auto() # Authentication and authorization errors
    CONFIGURATION = auto()  # Configuration-related errors
    RETRY = auto()         # Retry-related errors
    UNKNOWN = auto()       # Uncategorized errors


class FridayError(Exception):
    """Base exception class for all Friday AI Trading System errors.
    
    This class provides enhanced error information including:
    - Error severity level
    - Error category
    - Troubleshooting guidance
    - Contextual information
    - Original cause
    - Timestamp
    - Stack trace
    - Function name where the error occurred
    - Error code
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new FridayError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            category: The category of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.troubleshooting_guidance = troubleshooting_guidance
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        self.stack_trace = traceback.format_exc()
        self.function_name = self._get_caller_function_name()
        self.error_code = error_code
    
    def _get_caller_function_name(self) -> str:
        """Get the name of the function that raised the error.
        
        Returns:
            The name of the function that raised the error
        """
        stack = inspect.stack()
        # Skip the current function and its caller (the __init__ method)
        # to get to the actual caller that raised the error
        for frame_info in stack[2:]:
            if frame_info.function != '__init__':
                return f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}"
        return "Unknown function"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation.
        
        Returns:
            A dictionary containing all error information
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.name,
            'category': self.category.name,
            'troubleshooting_guidance': self.troubleshooting_guidance,
            'context': self.context,
            'cause': str(self.cause) if self.cause else None,
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'error_code': self.error_code.value if self.error_code else None
        }
    
    def log(self, logger: Optional[logging.Logger] = None) -> None:
        """Log the error with appropriate severity level.
        
        Args:
            logger: The logger to use. If None, the root logger is used.
        """
        logger = logger or logging.getLogger()
        
        # Map error severity to logging level
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.INFO: logging.DEBUG
        }.get(self.severity, logging.ERROR)
        
        # Create a structured log message
        error_dict = self.to_dict()
        error_code_str = f" [{error_dict['error_code']}]" if error_dict['error_code'] else ""
        
        # Log the basic error information
        logger.log(log_level, f"{error_dict['error_type']}{error_code_str}: {error_dict['message']}")
        
        # Log additional details at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Error details: {error_dict}")
            if self.cause:
                logger.debug(f"Caused by: {self.cause}", exc_info=self.cause)


class SystemError(FridayError):
    """Exception raised for system-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new SystemError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check system resources (CPU, memory, disk space). "
                "Verify system dependencies are installed and up to date. "
                "Check system logs for additional information."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.SYSTEM,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class NetworkError(FridayError):
    """Exception raised for network-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new NetworkError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check network connectivity. "
                "Verify DNS resolution is working. "
                "Check if the remote service is available. "
                "Verify firewall settings allow the connection."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.NETWORK,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class DatabaseError(FridayError):
    """Exception raised for database-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new DatabaseError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check database connection settings. "
                "Verify the database server is running. "
                "Check database logs for errors. "
                "Verify database schema is up to date."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.DATABASE,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class APIError(FridayError):
    """Exception raised for API-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new APIError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check API endpoint URL. "
                "Verify API credentials. "
                "Check API request parameters. "
                "Verify API service is available. "
                "Check for rate limiting or quota issues."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.API,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class DataError(FridayError):
    """Exception raised for data-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new DataError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check data format and structure. "
                "Verify data sources are available and providing valid data. "
                "Check for data corruption or inconsistency. "
                "Verify data validation rules."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.DATA,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class DataValidationError(FridayError):
    """Exception raised for data validation errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.LOW,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new DataValidationError.

        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Verify data format and constraints. "
                "Ensure all required fields are present and valid. "
                "Check for data type mismatches and violations."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.VALIDATION,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class AuthenticationError(FridayError):
    """Exception raised for authentication-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new AuthenticationError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Verify authentication credentials. "
                "Check if authentication token is expired. "
                "Verify user has necessary permissions. "
                "Check if account is locked or disabled."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.AUTHENTICATION,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class ConfigurationError(FridayError):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new ConfigurationError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check configuration file exists and is readable. "
                "Verify configuration values are valid. "
                "Check for missing required configuration. "
                "Verify environment variables are set correctly."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.CONFIGURATION,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class RetryableError(FridayError):
    """Exception that indicates an operation can be retried."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new RetryableError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            category: The category of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "This is a transient error that can be retried. "
                "Wait for a short period before retrying. "
                "If the error persists after multiple retries, check for underlying issues."
            )
        super().__init__(
            message,
            severity,
            category,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class RetryExhaustedError(FridayError):
    """Exception raised when retry attempts are exhausted."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new RetryExhaustedError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "All retry attempts have been exhausted. "
                "Check for underlying issues causing the operation to fail. "
                "Verify the retry policy is appropriate for the operation. "
                "Check logs for details on each retry attempt."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.RETRY,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class PermanentError(FridayError):
    """Exception raised for permanent errors that should not be retried."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new PermanentError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            category: The category of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "This is a permanent error that should not be retried. "
                "Check the underlying issue and fix it before attempting the operation again. "
                "Review error details and context for specific guidance."
            )
        super().__init__(
            message,
            severity,
            category,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


class DataSourceUnavailableError(FridayError):
    """Exception raised when a data source is unavailable.
    
    This error is typically used by circuit breakers and fallback mechanisms
    to indicate that a data source cannot be reached or is temporarily unavailable.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new DataSourceUnavailableError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Data source is temporarily unavailable. "
                "Check data source connectivity and status. "
                "Consider using fallback data sources or cached data. "
                "Wait for data source to recover or check for service disruptions."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.DATA,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )


def create_error_report(error: Exception) -> Dict[str, Any]:
    """Create a detailed error report from an exception.
    
    Args:
        error: The exception to create a report for
        
    Returns:
        A dictionary containing detailed error information
    """
    if isinstance(error, FridayError):
        # Use the built-in to_dict method for FridayError instances
        report = error.to_dict()
    else:
        # Create a basic report for standard exceptions
        report = {
            'error_type': error.__class__.__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'stack_trace': traceback.format_exc()
        }
    
    return report


def log_error(error: Exception, logger: Optional[logging.Logger] = None) -> None:
    """Log an error with appropriate severity and details.
    
    Args:
        error: The exception to log
        logger: The logger to use. If None, the root logger is used.
    """
    logger = logger or logging.getLogger()
    
    if isinstance(error, FridayError):
        # Use the built-in log method for FridayError instances
        error.log(logger)
    else:
        # Log standard exceptions as errors
        logger.error(f"{error.__class__.__name__}: {str(error)}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")


class ErrorHandler:
    """Context manager and decorator for handling errors consistently.
    
    This class can be used as a context manager or decorator to catch and handle
    specified error types with options for logging, custom handlers, and re-raising.
    """
    
    def __init__(
        self,
        error_types: Union[Type[Exception], List[Type[Exception]], None] = None,
        log_errors: bool = True,
        logger: Optional[logging.Logger] = None,
        handler: Optional[Callable[[Exception], Any]] = None,
        reraise: bool = True,
        reraise_as: Optional[Type[Exception]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize a new ErrorHandler.
        
        Args:
            error_types: The exception type(s) to catch. If None, catches all exceptions.
            log_errors: Whether to log caught errors
            logger: The logger to use. If None, the root logger is used.
            handler: A custom function to handle caught errors
            reraise: Whether to re-raise caught errors
            reraise_as: The exception type to re-raise as. If None, re-raises the original.
            context: Additional context to include in error logs
        """
        self.error_types = error_types or Exception
        self.log_errors = log_errors
        self.logger = logger or logging.getLogger()
        self.handler = handler
        self.reraise = reraise
        self.reraise_as = reraise_as
        self.context = context or {}
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and handle any exceptions.
        
        Args:
            exc_type: The type of the exception
            exc_val: The exception instance
            exc_tb: The exception traceback
            
        Returns:
            True if the exception was handled, False otherwise
        """
        if exc_type is None:
            return False
        
        if not isinstance(exc_val, self.error_types):
            return False
        
        # Add context to FridayError instances
        if isinstance(exc_val, FridayError) and self.context:
            exc_val.context.update(self.context)
        
        # Log the error if requested
        if self.log_errors:
            log_error(exc_val, self.logger)
        
        # Call the custom handler if provided
        if self.handler:
            self.handler(exc_val)
        
        # Re-raise the error if requested
        if self.reraise:
            if self.reraise_as and not isinstance(exc_val, self.reraise_as):
                if issubclass(self.reraise_as, FridayError):
                    raise self.reraise_as(
                        str(exc_val),
                        context=self.context,
                        cause=exc_val
                    )
                else:
                    raise self.reraise_as(str(exc_val)) from exc_val
            return False  # Re-raise the original exception
        
        return True  # Suppress the exception
    
    def __call__(self, func: F) -> F:
        """Use the ErrorHandler as a decorator.
        
        Args:
            func: The function to decorate
            
        Returns:
            The decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        
        return cast(F, wrapper)


# Import retry and fallback modules to make them available through the error package
from src.infrastructure.error.retry import (
    RetryStrategy, RetryResult, RetryPolicy, RetryHandler, retry,
    RetryPolicies, RetryPolicyFactory
)



class DataProcessingError(FridayError):
    """Exception raised for data processing errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        troubleshooting_guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        """Initialize a new DataProcessingError.
        
        Args:
            message: The error message
            severity: The severity level of the error
            troubleshooting_guidance: Guidance for troubleshooting the error
            context: Additional contextual information about the error
            cause: The original exception that caused this error
            error_code: The standardized error code
        """
        if troubleshooting_guidance is None:
            troubleshooting_guidance = (
                "Check data integrity and processing logic. "
                "Review data input and output for validity. "
                "Ensure data processing steps are correct."
            )
        super().__init__(
            message,
            severity,
            ErrorCategory.DATA,
            troubleshooting_guidance,
            context,
            cause,
            error_code
        )

from src.infrastructure.error.fallback import (
    FallbackStrategy, CircuitState, fallback_to_cache, fallback_to_default,
    fallback_to_alternative, CircuitBreaker, FallbackChain, with_fallback_chain
)

# Import data errors
from src.infrastructure.error.data_errors import (
    DataError, DataConnectionError, DataValidationError, 
    DataProcessingError, DataNotFoundError
)

# Initialize error code registry
from src.infrastructure.error.error_codes import register_standard_error_codes
register_standard_error_codes()

# Export all error classes and utilities
__all__ = [
    # Base classes
    'FridayError',
    'ErrorSeverity',
    'ErrorCategory',
    # Specific error types
    'SystemError',
    'NetworkError', 
    'DatabaseError',
    'APIError',
    'DataError',
    'DataConnectionError',
    'DataValidationError',
    'DataProcessingError',
    'DataNotFoundError',
    'AuthenticationError',
    'ConfigurationError',
    'RetryableError',
    'RetryExhaustedError',
    'PermanentError',
    'DataSourceUnavailableError',
    # Error handling utilities
    'ErrorHandler',
    'create_error_report',
    'log_error',
    # Retry functionality
    'RetryStrategy',
    'RetryResult',
    'RetryPolicy',
    'RetryHandler',
    'retry',
    'RetryPolicies',
    'RetryPolicyFactory',
    # Fallback functionality
    'FallbackStrategy',
    'CircuitState',
    'fallback_to_cache',
    'fallback_to_default',
    'fallback_to_alternative',
    'CircuitBreaker',
    'FallbackChain',
    'with_fallback_chain'
]
