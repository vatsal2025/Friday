"""Error code standardization for the Friday AI Trading System.

This module provides standardized error codes for different types of errors
in the system, ensuring consistent error handling and reporting.
"""

from enum import Enum, auto
from typing import Dict, Optional, Type


class ErrorCode(Enum):
    """Standardized error codes for the Friday AI Trading System.
    
    Error codes are organized by category with the following format:
    - CATEGORY_SUBCATEGORY_SPECIFIC_ERROR
    
    Categories include:
    - SYS: System errors
    - NET: Network errors
    - DB: Database errors
    - API: API errors
    - DATA: Data errors
    - AUTH: Authentication errors
    - CFG: Configuration errors
    - RETRY: Retry-related errors
    """
    # System errors (SYS)
    SYS_RESOURCE_CPU_LIMIT = "SYS-001"
    SYS_RESOURCE_MEMORY_LIMIT = "SYS-002"
    SYS_RESOURCE_DISK_SPACE = "SYS-003"
    SYS_PROCESS_CRASH = "SYS-004"
    SYS_DEPENDENCY_MISSING = "SYS-005"
    SYS_PERMISSION_DENIED = "SYS-006"
    SYS_TIMEOUT_OPERATION = "SYS-007"
    SYS_UNEXPECTED_ERROR = "SYS-999"
    
    # Network errors (NET)
    NET_CONNECTION_FAILED = "NET-001"
    NET_CONNECTION_TIMEOUT = "NET-002"
    NET_CONNECTION_RESET = "NET-003"
    NET_DNS_RESOLUTION = "NET-004"
    NET_SSL_CERTIFICATE = "NET-005"
    NET_SOCKET_ERROR = "NET-006"
    NET_PROXY_ERROR = "NET-007"
    NET_WEBSOCKET_DISCONNECT = "NET-008"
    
    # Database errors (DB)
    DB_CONNECTION_FAILED = "DB-001"
    DB_CONNECTION_TIMEOUT = "DB-002"
    DB_QUERY_TIMEOUT = "DB-003"
    DB_QUERY_SYNTAX = "DB-004"
    DB_CONSTRAINT_VIOLATION = "DB-005"
    DB_DEADLOCK = "DB-006"
    DB_POOL_EXHAUSTED = "DB-007"
    DB_TRANSACTION_FAILED = "DB-008"
    DB_SCHEMA_MISMATCH = "DB-009"
    
    # API errors (API)
    API_REQUEST_FAILED = "API-001"
    API_RESPONSE_INVALID = "API-002"
    API_RATE_LIMIT = "API-003"
    API_AUTHENTICATION = "API-004"
    API_AUTHORIZATION = "API-005"
    API_RESOURCE_NOT_FOUND = "API-006"
    API_METHOD_NOT_ALLOWED = "API-007"
    API_VALIDATION_ERROR = "API-008"
    API_SERVER_ERROR = "API-009"
    API_SERVICE_UNAVAILABLE = "API-010"
    
    # Data errors (DATA)
    DATA_VALIDATION_FAILED = "DATA-001"
    DATA_PARSING_ERROR = "DATA-002"
    DATA_INTEGRITY_ERROR = "DATA-003"
    DATA_MISSING_REQUIRED = "DATA-004"
    DATA_TYPE_MISMATCH = "DATA-005"
    DATA_RANGE_ERROR = "DATA-006"
    DATA_DUPLICATE_ENTRY = "DATA-007"
    DATA_INCONSISTENCY = "DATA-008"
    DATA_SERIALIZATION_ERROR = "DATA-009"
    
    # Authentication errors (AUTH)
    AUTH_INVALID_CREDENTIALS = "AUTH-001"
    AUTH_TOKEN_EXPIRED = "AUTH-002"
    AUTH_TOKEN_INVALID = "AUTH-003"
    AUTH_PERMISSION_DENIED = "AUTH-004"
    AUTH_ACCOUNT_LOCKED = "AUTH-005"
    AUTH_ACCOUNT_DISABLED = "AUTH-006"
    AUTH_MFA_REQUIRED = "AUTH-007"
    AUTH_SESSION_EXPIRED = "AUTH-008"
    
    # Configuration errors (CFG)
    CFG_MISSING_REQUIRED = "CFG-001"
    CFG_INVALID_VALUE = "CFG-002"
    CFG_TYPE_MISMATCH = "CFG-003"
    CFG_DEPENDENCY_CONFLICT = "CFG-004"
    CFG_FILE_NOT_FOUND = "CFG-005"
    CFG_PARSE_ERROR = "CFG-006"
    CFG_ENVIRONMENT_MISSING = "CFG-007"
    
    # Retry-related errors (RETRY)
    RETRY_ATTEMPTS_EXHAUSTED = "RETRY-001"
    RETRY_TIMEOUT_EXCEEDED = "RETRY-002"
    RETRY_CIRCUIT_OPEN = "RETRY-003"
    RETRY_PERMANENT_FAILURE = "RETRY-004"


class ErrorCodeRegistry:
    """Registry for mapping error codes to error classes and descriptions."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorCodeRegistry, cls).__new__(cls)
            cls._instance._error_code_to_class = {}
            cls._instance._error_code_to_description = {}
            cls._instance._class_to_error_code = {}
        return cls._instance
    
    def register_error_code(self, error_code: ErrorCode, error_class: Type[Exception], description: str) -> None:
        """Register an error code with its corresponding error class and description.
        
        Args:
            error_code: The error code to register
            error_class: The exception class associated with the error code
            description: A description of the error
        """
        self._error_code_to_class[error_code] = error_class
        self._error_code_to_description[error_code] = description
        self._class_to_error_code[error_class] = error_code
    
    def get_error_class(self, error_code: ErrorCode) -> Optional[Type[Exception]]:
        """Get the exception class associated with an error code.
        
        Args:
            error_code: The error code to look up
            
        Returns:
            The exception class associated with the error code, or None if not found
        """
        return self._error_code_to_class.get(error_code)
    
    def get_error_description(self, error_code: ErrorCode) -> Optional[str]:
        """Get the description associated with an error code.
        
        Args:
            error_code: The error code to look up
            
        Returns:
            The description associated with the error code, or None if not found
        """
        return self._error_code_to_description.get(error_code)
    
    def get_error_code(self, error_class: Type[Exception]) -> Optional[ErrorCode]:
        """Get the error code associated with an exception class.
        
        Args:
            error_class: The exception class to look up
            
        Returns:
            The error code associated with the exception class, or None if not found
        """
        return self._class_to_error_code.get(error_class)
    
    def get_all_error_codes(self) -> Dict[ErrorCode, Dict[str, object]]:
        """Get all registered error codes with their classes and descriptions.
        
        Returns:
            A dictionary mapping error codes to their classes and descriptions
        """
        result = {}
        for error_code in self._error_code_to_class.keys():
            result[error_code] = {
                'class': self._error_code_to_class.get(error_code),
                'description': self._error_code_to_description.get(error_code)
            }
        return result


def register_standard_error_codes():
    """Register standard error codes with their corresponding error classes and descriptions."""
    from src.infrastructure.error import (
        SystemError, NetworkError, DatabaseError, APIError, DataError,
        AuthenticationError, ConfigurationError, RetryExhaustedError
    )
    
    registry = ErrorCodeRegistry()
    
    # System errors
    registry.register_error_code(
        ErrorCode.SYS_RESOURCE_CPU_LIMIT,
        SystemError,
        "CPU resource limit exceeded"
    )
    registry.register_error_code(
        ErrorCode.SYS_RESOURCE_MEMORY_LIMIT,
        SystemError,
        "Memory resource limit exceeded"
    )
    registry.register_error_code(
        ErrorCode.SYS_RESOURCE_DISK_SPACE,
        SystemError,
        "Insufficient disk space"
    )
    registry.register_error_code(
        ErrorCode.SYS_PROCESS_CRASH,
        SystemError,
        "Process crashed unexpectedly"
    )
    registry.register_error_code(
        ErrorCode.SYS_DEPENDENCY_MISSING,
        SystemError,
        "Required system dependency is missing"
    )
    registry.register_error_code(
        ErrorCode.SYS_PERMISSION_DENIED,
        SystemError,
        "Permission denied for system operation"
    )
    registry.register_error_code(
        ErrorCode.SYS_TIMEOUT_OPERATION,
        SystemError,
        "System operation timed out"
    )
    registry.register_error_code(
        ErrorCode.SYS_UNEXPECTED_ERROR,
        SystemError,
        "Unexpected system error"
    )
    
    # Network errors
    registry.register_error_code(
        ErrorCode.NET_CONNECTION_FAILED,
        NetworkError,
        "Network connection failed"
    )
    registry.register_error_code(
        ErrorCode.NET_CONNECTION_TIMEOUT,
        NetworkError,
        "Network connection timed out"
    )
    registry.register_error_code(
        ErrorCode.NET_CONNECTION_RESET,
        NetworkError,
        "Network connection was reset"
    )
    registry.register_error_code(
        ErrorCode.NET_DNS_RESOLUTION,
        NetworkError,
        "DNS resolution failed"
    )
    registry.register_error_code(
        ErrorCode.NET_SSL_CERTIFICATE,
        NetworkError,
        "SSL certificate validation failed"
    )
    registry.register_error_code(
        ErrorCode.NET_SOCKET_ERROR,
        NetworkError,
        "Socket error occurred"
    )
    registry.register_error_code(
        ErrorCode.NET_PROXY_ERROR,
        NetworkError,
        "Proxy server error"
    )
    registry.register_error_code(
        ErrorCode.NET_WEBSOCKET_DISCONNECT,
        NetworkError,
        "WebSocket connection disconnected"
    )
    
    # Database errors
    registry.register_error_code(
        ErrorCode.DB_CONNECTION_FAILED,
        DatabaseError,
        "Database connection failed"
    )
    registry.register_error_code(
        ErrorCode.DB_CONNECTION_TIMEOUT,
        DatabaseError,
        "Database connection timed out"
    )
    registry.register_error_code(
        ErrorCode.DB_QUERY_TIMEOUT,
        DatabaseError,
        "Database query timed out"
    )
    registry.register_error_code(
        ErrorCode.DB_QUERY_SYNTAX,
        DatabaseError,
        "Database query syntax error"
    )
    registry.register_error_code(
        ErrorCode.DB_CONSTRAINT_VIOLATION,
        DatabaseError,
        "Database constraint violation"
    )
    registry.register_error_code(
        ErrorCode.DB_DEADLOCK,
        DatabaseError,
        "Database deadlock detected"
    )
    registry.register_error_code(
        ErrorCode.DB_POOL_EXHAUSTED,
        DatabaseError,
        "Database connection pool exhausted"
    )
    registry.register_error_code(
        ErrorCode.DB_TRANSACTION_FAILED,
        DatabaseError,
        "Database transaction failed"
    )
    registry.register_error_code(
        ErrorCode.DB_SCHEMA_MISMATCH,
        DatabaseError,
        "Database schema mismatch"
    )
    
    # API errors
    registry.register_error_code(
        ErrorCode.API_REQUEST_FAILED,
        APIError,
        "API request failed"
    )
    registry.register_error_code(
        ErrorCode.API_RESPONSE_INVALID,
        APIError,
        "Invalid API response"
    )
    registry.register_error_code(
        ErrorCode.API_RATE_LIMIT,
        APIError,
        "API rate limit exceeded"
    )
    registry.register_error_code(
        ErrorCode.API_AUTHENTICATION,
        APIError,
        "API authentication failed"
    )
    registry.register_error_code(
        ErrorCode.API_AUTHORIZATION,
        APIError,
        "API authorization failed"
    )
    registry.register_error_code(
        ErrorCode.API_RESOURCE_NOT_FOUND,
        APIError,
        "API resource not found"
    )
    registry.register_error_code(
        ErrorCode.API_METHOD_NOT_ALLOWED,
        APIError,
        "API method not allowed"
    )
    registry.register_error_code(
        ErrorCode.API_VALIDATION_ERROR,
        APIError,
        "API validation error"
    )
    registry.register_error_code(
        ErrorCode.API_SERVER_ERROR,
        APIError,
        "API server error"
    )
    registry.register_error_code(
        ErrorCode.API_SERVICE_UNAVAILABLE,
        APIError,
        "API service unavailable"
    )
    
    # Data errors
    registry.register_error_code(
        ErrorCode.DATA_VALIDATION_FAILED,
        DataError,
        "Data validation failed"
    )
    registry.register_error_code(
        ErrorCode.DATA_PARSING_ERROR,
        DataError,
        "Data parsing error"
    )
    registry.register_error_code(
        ErrorCode.DATA_INTEGRITY_ERROR,
        DataError,
        "Data integrity error"
    )
    registry.register_error_code(
        ErrorCode.DATA_MISSING_REQUIRED,
        DataError,
        "Missing required data"
    )
    registry.register_error_code(
        ErrorCode.DATA_TYPE_MISMATCH,
        DataError,
        "Data type mismatch"
    )
    registry.register_error_code(
        ErrorCode.DATA_RANGE_ERROR,
        DataError,
        "Data value out of range"
    )
    registry.register_error_code(
        ErrorCode.DATA_DUPLICATE_ENTRY,
        DataError,
        "Duplicate data entry"
    )
    registry.register_error_code(
        ErrorCode.DATA_INCONSISTENCY,
        DataError,
        "Data inconsistency detected"
    )
    registry.register_error_code(
        ErrorCode.DATA_SERIALIZATION_ERROR,
        DataError,
        "Data serialization error"
    )
    
    # Authentication errors
    registry.register_error_code(
        ErrorCode.AUTH_INVALID_CREDENTIALS,
        AuthenticationError,
        "Invalid authentication credentials"
    )
    registry.register_error_code(
        ErrorCode.AUTH_TOKEN_EXPIRED,
        AuthenticationError,
        "Authentication token expired"
    )
    registry.register_error_code(
        ErrorCode.AUTH_TOKEN_INVALID,
        AuthenticationError,
        "Invalid authentication token"
    )
    registry.register_error_code(
        ErrorCode.AUTH_PERMISSION_DENIED,
        AuthenticationError,
        "Permission denied"
    )
    registry.register_error_code(
        ErrorCode.AUTH_ACCOUNT_LOCKED,
        AuthenticationError,
        "Account locked"
    )
    registry.register_error_code(
        ErrorCode.AUTH_ACCOUNT_DISABLED,
        AuthenticationError,
        "Account disabled"
    )
    registry.register_error_code(
        ErrorCode.AUTH_MFA_REQUIRED,
        AuthenticationError,
        "Multi-factor authentication required"
    )
    registry.register_error_code(
        ErrorCode.AUTH_SESSION_EXPIRED,
        AuthenticationError,
        "Authentication session expired"
    )
    
    # Configuration errors
    registry.register_error_code(
        ErrorCode.CFG_MISSING_REQUIRED,
        ConfigurationError,
        "Missing required configuration"
    )
    registry.register_error_code(
        ErrorCode.CFG_INVALID_VALUE,
        ConfigurationError,
        "Invalid configuration value"
    )
    registry.register_error_code(
        ErrorCode.CFG_TYPE_MISMATCH,
        ConfigurationError,
        "Configuration type mismatch"
    )
    registry.register_error_code(
        ErrorCode.CFG_DEPENDENCY_CONFLICT,
        ConfigurationError,
        "Configuration dependency conflict"
    )
    registry.register_error_code(
        ErrorCode.CFG_FILE_NOT_FOUND,
        ConfigurationError,
        "Configuration file not found"
    )
    registry.register_error_code(
        ErrorCode.CFG_PARSE_ERROR,
        ConfigurationError,
        "Configuration parse error"
    )
    registry.register_error_code(
        ErrorCode.CFG_ENVIRONMENT_MISSING,
        ConfigurationError,
        "Required environment variable missing"
    )
    
    # Retry-related errors
    registry.register_error_code(
        ErrorCode.RETRY_ATTEMPTS_EXHAUSTED,
        RetryExhaustedError,
        "Retry attempts exhausted"
    )
    registry.register_error_code(
        ErrorCode.RETRY_TIMEOUT_EXCEEDED,
        RetryExhaustedError,
        "Retry timeout exceeded"
    )
    registry.register_error_code(
        ErrorCode.RETRY_CIRCUIT_OPEN,
        RetryExhaustedError,
        "Circuit breaker open, retry not attempted"
    )
    registry.register_error_code(
        ErrorCode.RETRY_PERMANENT_FAILURE,
        RetryExhaustedError,
        "Permanent failure detected, retry not attempted"
    )