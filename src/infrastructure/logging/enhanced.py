"""Enhanced logging capabilities for the Friday AI Trading System.

This module provides enhanced logging capabilities including:
- Structured logging with additional context
- Log enrichment with system metrics
- Custom log formatters
- Log correlation for tracking request flows
"""

import functools
import inspect
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from src.infrastructure.logging import get_logger

# Type variable for the return type of the function being wrapped
T = TypeVar('T')

# Create logger
logger = get_logger(__name__)


class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord that supports structured logging."""
    
    def __init__(self, *args, **kwargs):
        """Initialize a StructuredLogRecord."""
        super().__init__(*args, **kwargs)
        self.structured_data = {}


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging."""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create a LogRecord with structured data support."""
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key == 'structured_data' and isinstance(extra[key], dict):
                    record.structured_data = extra[key]
                elif key in ['message', 'asctime'] or key in record.__dict__:
                    raise KeyError(f"Attempt to overwrite {key} in LogRecord")
                else:
                    record.__dict__[key] = extra[key]
        return record


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON."""
    
    def __init__(self, include_structured_data=True, **kwargs):
        """Initialize a JSONFormatter.
        
        Args:
            include_structured_data: Whether to include structured data in the output
            **kwargs: Additional fields to include in every log record
        """
        super().__init__()
        self.include_structured_data = include_structured_data
        self.additional_fields = kwargs
    
    def format(self, record):
        """Format a log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add structured data if present and enabled
        if self.include_structured_data and hasattr(record, 'structured_data') and record.structured_data:
            log_data['structured_data'] = record.structured_data
        
        # Add any additional fields from record.__dict__
        for key, value in record.__dict__.items():
            if key not in log_data and key not in ['args', 'exc_info', 'exc_text', 'stack_info', 'lineno', 
                                                 'funcName', 'created', 'msecs', 'relativeCreated', 
                                                 'levelname', 'levelno', 'pathname', 'filename', 
                                                 'module', 'name', 'thread', 'threadName', 
                                                 'processName', 'process', 'msg', 'structured_data']:
                log_data[key] = value
        
        # Add additional fields specified in constructor
        log_data.update(self.additional_fields)
        
        return json.dumps(log_data, default=str)


class DetailedFormatter(logging.Formatter):
    """Formatter that provides detailed log output."""
    
    def __init__(self, include_process_info=True, include_thread_info=True):
        """Initialize a DetailedFormatter.
        
        Args:
            include_process_info: Whether to include process information
            include_thread_info: Whether to include thread information
        """
        super().__init__()
        self.include_process_info = include_process_info
        self.include_thread_info = include_thread_info
    
    def format(self, record):
        """Format a log record with detailed information."""
        # Basic log format
        log_parts = [
            f"[{datetime.fromtimestamp(record.created).isoformat()}]",
            f"[{record.levelname}]",
            f"[{record.name}]"
        ]
        
        # Add process info if enabled
        if self.include_process_info:
            log_parts.append(f"[PID:{record.process}]")
        
        # Add thread info if enabled
        if self.include_thread_info:
            log_parts.append(f"[Thread:{record.threadName}]")
        
        # Add location info
        log_parts.append(f"[{record.module}.{record.funcName}:{record.lineno}]")
        
        # Add the message
        log_parts.append(record.getMessage())
        
        # Format the basic log message
        log_message = " ".join(log_parts)
        
        # Add structured data if present
        if hasattr(record, 'structured_data') and record.structured_data:
            structured_str = json.dumps(record.structured_data, indent=2, default=str)
            log_message += f"\nStructured Data: {structured_str}"
        
        # Add exception info if present
        if record.exc_info:
            log_message += f"\nException: {record.exc_info[0].__name__}: {record.exc_info[1]}"
            log_message += f"\nTraceback:\n{''.join(traceback.format_exception(*record.exc_info))}"
        
        return log_message


class LogEnricher:
    """Enriches log records with additional context."""
    
    def __init__(self):
        """Initialize a LogEnricher."""
        self.static_context = self._get_static_context()
    
    def _get_static_context(self) -> Dict[str, Any]:
        """Get static context information that doesn't change between log records.
        
        Returns:
            Dictionary with static context
        """
        return {
            'hostname': socket.gethostname(),
            'ip': socket.gethostbyname(socket.gethostname()),
            'os': platform.platform(),
            'python_version': sys.version,
            'process_id': os.getpid()
        }
    
    def _get_dynamic_context(self) -> Dict[str, Any]:
        """Get dynamic context information that may change between log records.
        
        Returns:
            Dictionary with dynamic context
        """
        return {
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            'timestamp': datetime.now().isoformat()
        }
    
    def enrich(self, record: logging.LogRecord) -> logging.LogRecord:
        """Enrich a log record with additional context.
        
        Args:
            record: LogRecord to enrich
            
        Returns:
            Enriched LogRecord
        """
        # Add static context
        for key, value in self.static_context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add dynamic context
        dynamic_context = self._get_dynamic_context()
        for key, value in dynamic_context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        return record


class LogEnricherFilter(logging.Filter):
    """Filter that enriches log records with additional context."""
    
    def __init__(self, name=''):
        """Initialize a LogEnricherFilter.
        
        Args:
            name: Name of the filter
        """
        super().__init__(name)
        self.enricher = LogEnricher()
    
    def filter(self, record):
        """Enrich a log record and allow it to be emitted.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            True to allow the record to be emitted
        """
        self.enricher.enrich(record)
        return True


class CorrelationIdFilter(logging.Filter):
    """Filter that adds a correlation ID to log records."""
    
    def __init__(self, name=''):
        """Initialize a CorrelationIdFilter.
        
        Args:
            name: Name of the filter
        """
        super().__init__(name)
        self._local = threading.local()
    
    def set_correlation_id(self, correlation_id):
        """Set the correlation ID for the current thread.
        
        Args:
            correlation_id: Correlation ID to set
        """
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self):
        """Get the correlation ID for the current thread.
        
        Returns:
            Correlation ID or None if not set
        """
        return getattr(self._local, 'correlation_id', None)
    
    def clear_correlation_id(self):
        """Clear the correlation ID for the current thread."""
        if hasattr(self._local, 'correlation_id'):
            del self._local.correlation_id
    
    def filter(self, record):
        """Add correlation ID to a log record and allow it to be emitted.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            True to allow the record to be emitted
        """
        correlation_id = self.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


def with_correlation_id(correlation_id=None):
    """Decorator that sets a correlation ID for the duration of a function call.
    
    Args:
        correlation_id: Correlation ID to use, or None to generate a new one
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the correlation ID filter
            correlation_filter = None
            for handler in logging.getLogger().handlers:
                for filter in handler.filters:
                    if isinstance(filter, CorrelationIdFilter):
                        correlation_filter = filter
                        break
                if correlation_filter:
                    break
            
            if correlation_filter:
                # Generate a correlation ID if not provided
                cid = correlation_id or f"{time.time()}-{os.getpid()}-{threading.get_ident()}"
                
                # Set the correlation ID
                previous_correlation_id = correlation_filter.get_correlation_id()
                correlation_filter.set_correlation_id(cid)
                
                try:
                    # Call the function
                    return func(*args, **kwargs)
                finally:
                    # Restore the previous correlation ID
                    if previous_correlation_id:
                        correlation_filter.set_correlation_id(previous_correlation_id)
                    else:
                        correlation_filter.clear_correlation_id()
            else:
                # No correlation filter found, just call the function
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_execution_time(logger_name=None, level=logging.INFO):
    """Decorator that logs the execution time of a function.
    
    Args:
        logger_name: Name of the logger to use, or None to use the module logger
        level: Logging level to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the logger
            log = logging.getLogger(logger_name or func.__module__)
            
            # Log the start of execution
            log.log(level, f"Starting execution of {func.__name__}")
            
            # Record the start time
            start_time = time.time()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Calculate the execution time
                execution_time = time.time() - start_time
                
                # Log the end of execution
                log.log(level, f"Finished execution of {func.__name__} in {execution_time:.6f} seconds")
                
                return result
            except Exception as e:
                # Calculate the execution time
                execution_time = time.time() - start_time
                
                # Log the error
                log.log(logging.ERROR, f"Error in {func.__name__} after {execution_time:.6f} seconds: {e}")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator


def log_function_call(logger_name=None, level=logging.DEBUG, log_args=True, log_result=True):
    """Decorator that logs function calls with arguments and results.
    
    Args:
        logger_name: Name of the logger to use, or None to use the module logger
        level: Logging level to use
        log_args: Whether to log function arguments
        log_result: Whether to log function results
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the logger
            log = logging.getLogger(logger_name or func.__module__)
            
            # Log the function call with arguments if enabled
            if log_args:
                # Format the arguments
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{key}={repr(value)}" for key, value in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                
                log.log(level, f"Calling {func.__name__}({all_args})")
            else:
                log.log(level, f"Calling {func.__name__}")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log the result if enabled
                if log_result:
                    log.log(level, f"{func.__name__} returned: {repr(result)}")
                else:
                    log.log(level, f"{func.__name__} completed successfully")
                
                return result
            except Exception as e:
                # Log the error
                log.log(logging.ERROR, f"Error in {func.__name__}: {e}")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator


def setup_enhanced_logging(
    json_output=False,
    log_file=None,
    level=logging.INFO,
    include_correlation_id=True,
    include_enrichment=True
):
    """Set up enhanced logging with structured logging support.
    
    Args:
        json_output: Whether to output logs as JSON
        log_file: Path to the log file, or None to log to console only
        level: Logging level to use
        include_correlation_id: Whether to include correlation IDs in logs
        include_enrichment: Whether to enrich logs with additional context
    """
    # Register the StructuredLogger class
    logging.setLoggerClass(StructuredLogger)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:  # Make a copy of the list
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    # Create formatter
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = DetailedFormatter()
    
    # Add filters and formatter to handlers
    for handler in handlers:
        # Add correlation ID filter if enabled
        if include_correlation_id:
            handler.addFilter(CorrelationIdFilter())
        
        # Add enrichment filter if enabled
        if include_enrichment:
            handler.addFilter(LogEnricherFilter())
        
        # Set formatter
        handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger.addHandler(handler)
    
    # Log that enhanced logging has been set up
    logger.info("Enhanced logging has been set up", extra={
        'structured_data': {
            'json_output': json_output,
            'log_file': log_file,
            'level': logging.getLevelName(level),
            'include_correlation_id': include_correlation_id,
            'include_enrichment': include_enrichment
        }
    })


def get_correlation_id():
    """Get the current correlation ID.
    
    Returns:
        Current correlation ID or None if not set
    """
    # Find the correlation ID filter
    for handler in logging.getLogger().handlers:
        for filter in handler.filters:
            if isinstance(filter, CorrelationIdFilter):
                return filter.get_correlation_id()
    
    return None


def set_correlation_id(correlation_id):
    """Set the correlation ID for the current thread.
    
    Args:
        correlation_id: Correlation ID to set
    """
    # Find the correlation ID filter
    for handler in logging.getLogger().handlers:
        for filter in handler.filters:
            if isinstance(filter, CorrelationIdFilter):
                filter.set_correlation_id(correlation_id)
                return
    
    # No correlation ID filter found, log a warning
    logger.warning("No CorrelationIdFilter found, correlation ID not set")


def clear_correlation_id():
    """Clear the correlation ID for the current thread."""
    # Find the correlation ID filter
    for handler in logging.getLogger().handlers:
        for filter in handler.filters:
            if isinstance(filter, CorrelationIdFilter):
                filter.clear_correlation_id()
                return
    
    # No correlation ID filter found, log a warning
    logger.warning("No CorrelationIdFilter found, correlation ID not cleared")


def structured_log(logger_name, level, message, **kwargs):
    """Log a structured message.
    
    Args:
        logger_name: Name of the logger to use
        level: Logging level to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    # Get the logger
    log = logging.getLogger(logger_name)
    
    # Log the message with structured data
    log.log(level, message, extra={'structured_data': kwargs})


def debug(logger_name, message, **kwargs):
    """Log a debug message with structured data.
    
    Args:
        logger_name: Name of the logger to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    structured_log(logger_name, logging.DEBUG, message, **kwargs)


def info(logger_name, message, **kwargs):
    """Log an info message with structured data.
    
    Args:
        logger_name: Name of the logger to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    structured_log(logger_name, logging.INFO, message, **kwargs)


def warning(logger_name, message, **kwargs):
    """Log a warning message with structured data.
    
    Args:
        logger_name: Name of the logger to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    structured_log(logger_name, logging.WARNING, message, **kwargs)


def error(logger_name, message, **kwargs):
    """Log an error message with structured data.
    
    Args:
        logger_name: Name of the logger to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    structured_log(logger_name, logging.ERROR, message, **kwargs)


def critical(logger_name, message, **kwargs):
    """Log a critical message with structured data.
    
    Args:
        logger_name: Name of the logger to use
        message: Log message
        **kwargs: Additional structured data to include in the log
    """
    structured_log(logger_name, logging.CRITICAL, message, **kwargs)


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **context):
        """Initialize a LogContext.
        
        Args:
            **context: Context data to add to logs
        """
        self.context = context
        self.previous_context = {}
    
    def __enter__(self):
        """Enter the context."""
        # Store the current context for each logger handler
        for handler in logging.getLogger().handlers:
            for filter in handler.filters:
                if isinstance(filter, LogEnricherFilter):
                    # Store the previous context
                    self.previous_context[filter] = {}
                    for key, value in self.context.items():
                        if hasattr(filter.enricher, key):
                            self.previous_context[filter][key] = getattr(filter.enricher, key)
                        setattr(filter.enricher, key, value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # Restore the previous context for each logger handler
        for handler in logging.getLogger().handlers:
            for filter in handler.filters:
                if isinstance(filter, LogEnricherFilter) and filter in self.previous_context:
                    # Restore the previous context
                    for key, value in self.previous_context[filter].items():
                        setattr(filter.enricher, key, value)
        
        return False  # Don't suppress exceptions