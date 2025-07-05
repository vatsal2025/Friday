"""Fallback mechanisms for handling failures.

This module provides fallback strategies for handling failures, including:
- Fallback to cache
- Fallback to default value
- Fallback to alternative function
- Circuit breaker pattern implementation
"""

import functools
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from pymongo import MongoClient
from pymongo.collection import Collection

from src.infrastructure.error import (
    FridayError, DataSourceUnavailableError, DataProcessingError, DataValidationError,
    ErrorSeverity, ErrorCategory, log_error
)
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Type variable for the return type of the function being wrapped
T = TypeVar('T')


class FallbackStrategy(Enum):
    """Strategies for fallback behavior."""
    CACHE = "cache"                # Fallback to cached data
    DEFAULT = "default"            # Fallback to a default value
    ALTERNATIVE = "alternative"    # Fallback to an alternative function
    NONE = "none"                  # No fallback (fail fast)


class CircuitState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = "closed"      # Normal operation, requests are allowed
    OPEN = "open"          # Circuit is open, requests are not allowed
    HALF_OPEN = "half_open"  # Testing if the service is back online


def fallback_to_cache(
    collection_name: str,
    query_func: Callable[..., Dict[str, Any]],
    expiry_time: int = 3600,
    mongo_uri: str = "mongodb://localhost:27017/",
    database_name: str = "friday_cache"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for falling back to cached data when a function fails.
    
    Args:
        collection_name: Name of the MongoDB collection to use for caching
        query_func: Function that returns a query to find the cached data
        expiry_time: Time in seconds after which cached data is considered expired
        mongo_uri: MongoDB connection URI
        database_name: Name of the MongoDB database to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get MongoDB collection
            client = MongoClient(mongo_uri)
            db = client[database_name]
            collection = db[collection_name]
            
            # Generate query for this function call
            query = query_func(*args, **kwargs)
            
            try:
                # Try to execute the function
                result = func(*args, **kwargs)
                
                # Cache the successful result
                cache_document = {
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now(),
                    "expiry": datetime.now() + timedelta(seconds=expiry_time)
                }
                
                # Use upsert to update or insert
                collection.update_one({"query": query}, {"$set": cache_document}, upsert=True)
                
                logger.debug(f"Cached result for {func.__name__} with query {query}")
                return result
            
            except (DataSourceUnavailableError, DataProcessingError, DataValidationError) as e:
                # Log the error
                log_error(e)
                
                # Try to get data from cache
                logger.info(f"Falling back to cache for {func.__name__} due to error: {e}")
                
                # Find valid cache entry (not expired)
                cache_entry = collection.find_one({
                    "query": query,
                    "expiry": {"$gt": datetime.now()}
                })
                
                if cache_entry:
                    logger.info(f"Using cached data for {func.__name__} from {cache_entry['timestamp']}")
                    return cast(T, cache_entry["result"])
                else:
                    logger.error(f"No valid cache entry found for {func.__name__} with query {query}")
                    raise
        
        return wrapper
    
    return decorator


def fallback_to_default(
    default_value: Any,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    log_error_before_fallback: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for falling back to a default value when a function fails.
    
    Args:
        default_value: Default value to return when the function fails
        exceptions: Exception type(s) to catch
        log_error_before_fallback: Whether to log the error before falling back
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error_before_fallback:
                    log_error(e)
                
                logger.info(f"Falling back to default value for {func.__name__} due to error: {e}")
                return cast(T, default_value)
        
        return wrapper
    
    return decorator


def fallback_to_alternative(
    alternative_func: Callable[..., T],
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    log_error_before_fallback: bool = True,
    pass_exception_to_alternative: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for falling back to an alternative function when a function fails.
    
    Args:
        alternative_func: Alternative function to call when the primary function fails
        exceptions: Exception type(s) to catch
        log_error_before_fallback: Whether to log the error before falling back
        pass_exception_to_alternative: Whether to pass the exception to the alternative function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error_before_fallback:
                    log_error(e)
                
                logger.info(f"Falling back to alternative function for {func.__name__} due to error: {e}")
                
                if pass_exception_to_alternative:
                    return alternative_func(*args, exception=e, **kwargs)
                else:
                    return alternative_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class CircuitBreaker(Generic[T]):
    """Implementation of the Circuit Breaker pattern.
    
    The Circuit Breaker pattern prevents a cascade of failures by failing fast
    when a service is unavailable. It has three states:
    - CLOSED: Normal operation, requests are allowed
    - OPEN: Circuit is open, requests are not allowed
    - HALF_OPEN: Testing if the service is back online
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        fallback_function: Optional[Callable[..., T]] = None,
        exceptions_to_monitor: Union[Type[Exception], List[Type[Exception]]] = Exception,
        exclude_exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
        window_size: int = 10
    ):
        """Initialize a CircuitBreaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before attempting to close the circuit
            fallback_function: Function to call when the circuit is open
            exceptions_to_monitor: Exception type(s) to monitor
            exclude_exceptions: Exception type(s) to exclude from monitoring
            window_size: Size of the sliding window for failure counting
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback_function = fallback_function
        self.exceptions_to_monitor = exceptions_to_monitor
        self.exclude_exceptions = exclude_exceptions or []
        self.window_size = window_size
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.success_count = 0
        
        # Sliding window of success/failure results (True for success, False for failure)
        self.results_window: List[bool] = []
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function or fallback
            
        Raises:
            Exception: If the circuit is open and no fallback is provided
        """
        # Check if the circuit is open
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(f"Circuit breaker for {func.__name__} transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                # Circuit is still open, use fallback or fail fast
                logger.warning(f"Circuit breaker for {func.__name__} is OPEN, failing fast")
                if self.fallback_function:
                    return self.fallback_function(*args, **kwargs)
                else:
                    raise DataSourceUnavailableError(
                        f"Circuit breaker for {func.__name__} is open",
                        severity=ErrorSeverity.HIGH,
                        troubleshooting_guidance=(
                            f"The circuit breaker for {func.__name__} is open due to multiple failures. "
                            f"Wait for the recovery timeout ({self.recovery_timeout}s) to elapse or "
                            f"check the underlying service for issues."
                        )
                    )
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Update state based on success
            self._on_success()
            
            return result
        
        except Exception as e:
            # Check if this exception should be monitored
            if self._should_monitor_exception(e):
                # Update state based on failure
                self._on_failure()
                
                # If in HALF_OPEN state, a single failure opens the circuit again
                if self.state == CircuitState.HALF_OPEN:
                    logger.warning(f"Circuit breaker for {func.__name__} transitioning from HALF_OPEN to OPEN due to failure")
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
            
            # Use fallback or re-raise
            if self.fallback_function:
                return self.fallback_function(*args, **kwargs)
            else:
                raise
    
    def _should_monitor_exception(self, exception: Exception) -> bool:
        """Determine if an exception should be monitored by the circuit breaker.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if the exception should be monitored, False otherwise
        """
        # Check if the exception is in the exclude list
        if self.exclude_exceptions:
            if isinstance(self.exclude_exceptions, list):
                for exc_type in self.exclude_exceptions:
                    if isinstance(exception, exc_type):
                        return False
            elif isinstance(exception, self.exclude_exceptions):
                return False
        
        # Check if the exception is in the monitor list
        if isinstance(self.exceptions_to_monitor, list):
            for exc_type in self.exceptions_to_monitor:
                if isinstance(exception, exc_type):
                    return True
            return False
        else:
            return isinstance(exception, self.exceptions_to_monitor)
    
    def _on_success(self) -> None:
        """Update state after a successful execution."""
        # Add success to the window
        self.results_window.append(True)
        if len(self.results_window) > self.window_size:
            self.results_window.pop(0)
        
        # Reset failure count based on window
        self.failures = self.window_size - sum(self.results_window)
        
        # If in HALF_OPEN state, increment success count
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # If enough successes, close the circuit
            if self.success_count >= 2:  # Require at least 2 consecutive successes
                logger.info(f"Circuit breaker transitioning from HALF_OPEN to CLOSED after {self.success_count} successes")
                self.state = CircuitState.CLOSED
                self.failures = 0
                self.results_window = [True] * min(self.window_size, self.success_count)
    
    def _on_failure(self) -> None:
        """Update state after a failed execution."""
        # Add failure to the window
        self.results_window.append(False)
        if len(self.results_window) > self.window_size:
            self.results_window.pop(0)
        
        # Update failure count based on window
        self.failures = self.window_size - sum(self.results_window)
        self.last_failure_time = time.time()
        
        # If enough failures, open the circuit
        if self.state == CircuitState.CLOSED and self.failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker transitioning from CLOSED to OPEN after {self.failures} failures")
            self.state = CircuitState.OPEN


class FallbackChain(Generic[T]):
    """Chain of fallback strategies to try in sequence."""
    
    def __init__(self, primary_function: Callable[..., T]):
        """Initialize a FallbackChain.
        
        Args:
            primary_function: Primary function to execute
        """
        self.primary_function = primary_function
        self.fallbacks: List[Callable[..., T]] = []
        self.exceptions_to_catch: List[Type[Exception]] = [Exception]
    
    def with_fallback(self, fallback_function: Callable[..., T]) -> 'FallbackChain[T]':
        """Add a fallback function to the chain.
        
        Args:
            fallback_function: Fallback function to add
            
        Returns:
            Self for method chaining
        """
        self.fallbacks.append(fallback_function)
        return self
    
    def on_exceptions(self, exceptions: Union[Type[Exception], List[Type[Exception]]]) -> 'FallbackChain[T]':
        """Set the exceptions to catch.
        
        Args:
            exceptions: Exception type(s) to catch
            
        Returns:
            Self for method chaining
        """
        if isinstance(exceptions, list):
            self.exceptions_to_catch = exceptions
        else:
            self.exceptions_to_catch = [exceptions]
        return self
    
    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Execute the function chain.
        
        Args:
            *args: Positional arguments for the functions
            **kwargs: Keyword arguments for the functions
            
        Returns:
            Result of the first successful function
            
        Raises:
            Exception: If all functions fail
        """
        # Try the primary function first
        try:
            return self.primary_function(*args, **kwargs)
        except tuple(self.exceptions_to_catch) as e:
            log_error(e)
            last_error = e
        
        # Try each fallback in sequence
        for i, fallback in enumerate(self.fallbacks):
            try:
                logger.info(f"Trying fallback {i+1}/{len(self.fallbacks)}")
                return fallback(*args, **kwargs)
            except tuple(self.exceptions_to_catch) as e:
                log_error(e)
                last_error = e
        
        # If all fallbacks fail, raise the last error
        logger.error("All fallbacks failed")
        raise last_error
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Execute the function chain.
        
        Args:
            *args: Positional arguments for the functions
            **kwargs: Keyword arguments for the functions
            
        Returns:
            Result of the first successful function
        """
        return self.execute(*args, **kwargs)


def with_fallback_chain(fallbacks: List[Callable[..., T]], exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for creating a fallback chain.
    
    Args:
        fallbacks: List of fallback functions to try in sequence
        exceptions: Exception type(s) to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            chain = FallbackChain(func).on_exceptions(exceptions)
            for fallback in fallbacks:
                chain.with_fallback(fallback)
            return chain.execute(*args, **kwargs)
        
        return wrapper
    
    return decorator