"""Error handling and fallback mechanisms for alternative data integration.

This module provides utilities for handling errors and implementing fallbacks
when alternative data sources are unavailable or return errors.
"""

import time
import functools
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, cast

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, find_one, update_one
)

# Create logger
logger = get_logger(__name__)

# Type variable for function return type
T = TypeVar('T')

class AlternativeDataError(Exception):
    """Base exception class for alternative data integration errors."""
    pass

class DataSourceUnavailableError(AlternativeDataError):
    """Exception raised when a data source is unavailable."""
    pass

class DataProcessingError(AlternativeDataError):
    """Exception raised when there's an error processing data."""
    pass

class DataValidationError(AlternativeDataError):
    """Exception raised when data fails validation."""
    pass

def retry(max_retries: int = 3, retry_delay: int = 5, 
         exceptions: tuple = (Exception,)) -> Callable:
    """Decorator for retrying a function on exception.
    
    Args:
        max_retries: Maximum number of retries.
        retry_delay: Delay between retries in seconds.
        exceptions: Tuple of exceptions to catch and retry on.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                        )
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            
            # This should never happen, but to satisfy the type checker
            raise RuntimeError(f"Unexpected error in retry decorator for {func.__name__}")
        
        return wrapper
    
    return decorator

def fallback_to_cache(collection_name: str, query_func: Callable[..., Dict[str, Any]], 
                     cache_expiry: int = 86400) -> Callable:
    """Decorator for falling back to cached data on exception.
    
    Args:
        collection_name: Name of the MongoDB collection to use for cache.
        query_func: Function that returns a query to find the cached data.
        cache_expiry: Cache expiry time in seconds.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                # Try to execute the original function
                result = func(*args, **kwargs)
                
                # Cache the successful result
                try:
                    # Get the query to find existing cache
                    query = query_func(*args, **kwargs)
                    
                    # Add cache metadata
                    cache_data = {
                        'data': result,
                        'cached_at': datetime.now().isoformat(),
                        'expires_at': (datetime.now() + timedelta(seconds=cache_expiry)).isoformat(),
                        'source': func.__name__
                    }
                    
                    # Update or insert cache
                    update_one(
                        collection_name,
                        query,
                        {'$set': cache_data},
                        upsert=True
                    )
                    
                    logger.debug(f"Cached result for {func.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to cache result for {func.__name__}: {str(e)}")
                
                return result
            
            except Exception as e:
                logger.warning(f"Error in {func.__name__}: {str(e)}. Falling back to cache...")
                
                try:
                    # Get the query to find existing cache
                    query = query_func(*args, **kwargs)
                    
                    # Add expiry condition
                    query['expires_at'] = {'$gte': datetime.now().isoformat()}
                    
                    # Find cached data
                    cached = find_one(collection_name, query)
                    
                    if cached and 'data' in cached:
                        logger.info(f"Using cached data for {func.__name__} from {cached.get('cached_at')}")
                        return cast(T, cached['data'])
                    else:
                        logger.error(f"No valid cache found for {func.__name__}")
                        raise
                
                except Exception as cache_error:
                    logger.error(f"Failed to retrieve cache for {func.__name__}: {str(cache_error)}")
                    raise e
        
        return wrapper
    
    return decorator

def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for logging function execution time.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

def validate_data(validation_func: Callable[[Any], bool], 
                error_message: str = "Data validation failed") -> Callable:
    """Decorator for validating data.
    
    Args:
        validation_func: Function that validates the result and returns a boolean.
        error_message: Error message to use when validation fails.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            
            if not validation_func(result):
                logger.error(f"{error_message} in {func.__name__}")
                raise DataValidationError(f"{error_message} in {func.__name__}")
            
            return result
        
        return wrapper
    
    return decorator

def handle_alternative_data_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for handling alternative data errors.
    
    This decorator logs errors and provides appropriate error handling
    for alternative data integration functions.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except DataSourceUnavailableError as e:
            logger.error(f"Data source unavailable in {func.__name__}: {str(e)}")
            # Re-raise to allow fallback mechanisms to work
            raise
        except DataProcessingError as e:
            logger.error(f"Data processing error in {func.__name__}: {str(e)}")
            # Re-raise to allow fallback mechanisms to work
            raise
        except DataValidationError as e:
            logger.error(f"Data validation error in {func.__name__}: {str(e)}")
            # Re-raise to allow fallback mechanisms to work
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            # Wrap in AlternativeDataError to standardize error handling
            raise AlternativeDataError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    
    return wrapper

def create_error_report(error: Exception, function_name: str, 
                      args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Create an error report for an exception.
    
    Args:
        error: The exception that was raised.
        function_name: Name of the function where the error occurred.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.
        
    Returns:
        Dictionary with error report details.
    """
    error_report = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'function_name': function_name,
        'timestamp': datetime.now().isoformat(),
        'args': str(args),
        'kwargs': str(kwargs)
    }
    
    # Store error report in MongoDB
    try:
        insert_one('error_reports', error_report)
        logger.info(f"Error report stored in MongoDB for {function_name}")
    except Exception as e:
        logger.error(f"Failed to store error report in MongoDB: {str(e)}")
    
    return error_report