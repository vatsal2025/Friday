"""Error classes for data-related operations."""

from typing import Any, Dict, List, Optional


class DataError(Exception):
    """Base class for all data-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DataConnectionError(DataError):
    """Error raised when there's an issue connecting to a data source."""

    def __init__(
        self,
        message: str,
        source: str,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, details)
        self.source = source
        self.retry_after = retry_after


class DataValidationError(DataError):
    """Error raised when data fails validation checks."""

    def __init__(
        self,
        message: str,
        validation_errors: List[Dict[str, Any]],
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.validation_errors = validation_errors


class DataProcessingError(DataError):
    """Error raised during data processing operations."""

    def __init__(
        self,
        message: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.operation = operation


class DataNotFoundError(DataError):
    """Error raised when requested data cannot be found."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_id: Any,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id