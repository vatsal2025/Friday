"""Retry mechanism for handling transient failures.

This module provides a comprehensive retry framework including:
- Configurable retry strategies (fixed, linear, exponential, fibonacci, random)
- Customizable backoff and jitter
- Timeout handling
- Detailed logging of retry attempts
- Integration with the error handling framework
"""

import functools
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from src.infrastructure.error import (
    FridayError, RetryableError, PermanentError, RetryExhaustedError,
    ErrorSeverity, ErrorCategory, log_error
)
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Type variable for the return type of the function being retried
T = TypeVar('T')


class RetryStrategy(Enum):
    """Strategies for calculating retry delay."""
    FIXED = "fixed"                  # Fixed delay between retries
    LINEAR = "linear"                # Linearly increasing delay
    EXPONENTIAL = "exponential"      # Exponentially increasing delay
    FIBONACCI = "fibonacci"          # Fibonacci sequence for delay
    RANDOM = "random"                # Random delay within a range
    RANDOM_EXPONENTIAL = "random_exponential"  # Random delay with exponential backoff


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool                    # Whether the operation succeeded
    result: Any = None               # Result of the operation if successful
    attempts: int = 0                # Number of attempts made
    last_error: Optional[Exception] = None  # Last error encountered
    total_delay: float = 0.0         # Total time spent in delays between retries
    execution_time: float = 0.0      # Total execution time including retries


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3            # Maximum number of retry attempts
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL  # Retry delay strategy
    initial_delay: float = 1.0       # Initial delay in seconds
    max_delay: float = 60.0          # Maximum delay in seconds
    backoff_factor: float = 2.0      # Factor for backoff calculation
    jitter: float = 0.1              # Jitter factor (0.0 to 1.0) to randomize delay
    timeout: Optional[float] = None  # Overall timeout in seconds (None for no timeout)
    retryable_errors: List[Type[Exception]] = field(default_factory=list)  # Errors that should trigger retry
    permanent_errors: List[Type[Exception]] = field(default_factory=list)  # Errors that should not trigger retry
    retry_on_result: Optional[Callable[[Any], bool]] = None  # Function to determine if result should trigger retry
    on_retry: Optional[Callable[[int, Exception, float], None]] = None  # Callback on retry

    def __post_init__(self):
        """Validate and initialize the retry policy."""
        # If no retryable errors are specified, default to RetryableError
        if not self.retryable_errors:
            self.retryable_errors = [RetryableError]
        
        # If no permanent errors are specified, default to PermanentError
        if not self.permanent_errors:
            self.permanent_errors = [PermanentError]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the retry policy to a dictionary.
        
        Returns:
            Dictionary representation of the retry policy
        """
        result = {
            "max_attempts": self.max_attempts,
            "strategy": self.strategy.value,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "backoff_factor": self.backoff_factor,
            "jitter": self.jitter,
        }
        
        if self.timeout is not None:
            result["timeout"] = self.timeout
        
        if self.retryable_errors:
            result["retryable_errors"] = [e.__name__ for e in self.retryable_errors]
        
        if self.permanent_errors:
            result["permanent_errors"] = [e.__name__ for e in self.permanent_errors]
        
        # Note: retry_on_result and on_retry callbacks are not serialized
        
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryPolicy':
        """Create a retry policy from a dictionary.
        
        Args:
            data: Dictionary representation of a retry policy
            
        Returns:
            RetryPolicy instance
        """
        # Copy the data to avoid modifying the original
        policy_data = data.copy()
        
        # Convert strategy string to enum
        if "strategy" in policy_data and isinstance(policy_data["strategy"], str):
            try:
                policy_data["strategy"] = RetryStrategy(policy_data["strategy"])
            except ValueError:
                logger.warning(f"Invalid retry strategy: {policy_data['strategy']}. Using EXPONENTIAL.")
                policy_data["strategy"] = RetryStrategy.EXPONENTIAL
        
        # Note: retryable_errors and permanent_errors are not deserialized
        # They would need to be mapped from string names to actual exception classes
        if "retryable_errors" in policy_data:
            del policy_data["retryable_errors"]
        
        if "permanent_errors" in policy_data:
            del policy_data["permanent_errors"]
        
        return cls(**policy_data)


class RetryHandler:
    """Handler for executing functions with retry logic."""
    
    def __init__(self, policy: RetryPolicy):
        """Initialize a RetryHandler.
        
        Args:
            policy: Retry policy configuration
        """
        self.policy = policy
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate the delay for a retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        strategy = self.policy.strategy
        initial_delay = self.policy.initial_delay
        backoff_factor = self.policy.backoff_factor
        max_delay = self.policy.max_delay
        jitter = self.policy.jitter
        
        # Calculate base delay based on strategy
        if strategy == RetryStrategy.FIXED:
            delay = initial_delay
        elif strategy == RetryStrategy.LINEAR:
            delay = initial_delay * attempt
        elif strategy == RetryStrategy.EXPONENTIAL:
            delay = initial_delay * (backoff_factor ** (attempt - 1))
        elif strategy == RetryStrategy.FIBONACCI:
            # Calculate Fibonacci number (starting with 1, 1, 2, 3, 5, ...)
            fib = [1, 1]
            for i in range(2, attempt + 1):
                fib.append(fib[i-1] + fib[i-2])
            delay = initial_delay * fib[attempt]
        elif strategy == RetryStrategy.RANDOM:
            # Random delay between initial_delay and initial_delay * backoff_factor
            min_delay = initial_delay
            max_random_delay = initial_delay * backoff_factor
            delay = random.uniform(min_delay, max_random_delay)
        elif strategy == RetryStrategy.RANDOM_EXPONENTIAL:
            # Random delay between initial_delay and exponential backoff
            min_delay = initial_delay
            max_random_delay = initial_delay * (backoff_factor ** (attempt - 1))
            delay = random.uniform(min_delay, max_random_delay)
        else:
            # Default to exponential backoff
            delay = initial_delay * (backoff_factor ** (attempt - 1))
        
        # Apply jitter if specified
        if jitter > 0:
            jitter_amount = delay * jitter
            delay = random.uniform(delay - jitter_amount, delay + jitter_amount)
        
        # Cap at max_delay
        return min(delay, max_delay)
    
    def _should_retry(self, attempt: int, error: Optional[Exception] = None, result: Any = None) -> bool:
        """Determine if a retry should be attempted.
        
        Args:
            attempt: Current attempt number (1-based)
            error: Exception that was raised, if any
            result: Result of the function call, if no exception was raised
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        # Check if max attempts reached
        if attempt >= self.policy.max_attempts:
            return False
        
        # If an error occurred, check if it's retryable
        if error is not None:
            # Check if it's a permanent error (should not retry)
            for err_type in self.policy.permanent_errors:
                if isinstance(error, err_type):
                    return False
            
            # Check if it's a retryable error
            for err_type in self.policy.retryable_errors:
                if isinstance(error, err_type):
                    return True
            
            # If it's not explicitly retryable or permanent, don't retry
            return False
        
        # If no error, check if result should trigger retry
        if self.policy.retry_on_result is not None:
            return self.policy.retry_on_result(result)
        
        # No error and no retry_on_result function, no need to retry
        return False
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> RetryResult:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            RetryResult with the execution result or last error
        """
        start_time = time.time()
        attempt = 0
        total_delay = 0.0
        last_error = None
        
        # Calculate timeout deadline if specified
        deadline = None if self.policy.timeout is None else start_time + self.policy.timeout
        
        while True:
            attempt += 1
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if self._should_retry(attempt, result=result):
                    last_error = None  # No error, but retry based on result
                else:
                    # Success, no retry needed
                    return RetryResult(
                        success=True,
                        result=result,
                        attempts=attempt,
                        last_error=None,
                        total_delay=total_delay,
                        execution_time=time.time() - start_time
                    )
            except Exception as e:
                # Store the error
                last_error = e
                
                # Log the error
                log_error(e, include_stack_trace=False)
                
                # Check if we should retry
                if not self._should_retry(attempt, error=e):
                    # No retry, return failure
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempt,
                        last_error=e,
                        total_delay=total_delay,
                        execution_time=time.time() - start_time
                    )
            
            # Check if we've exceeded the timeout
            if deadline is not None and time.time() + self._calculate_delay(attempt) > deadline:
                logger.warning(f"Retry timeout exceeded after {attempt} attempts")
                return RetryResult(
                    success=False,
                    result=None,
                    attempts=attempt,
                    last_error=last_error or TimeoutError("Retry timeout exceeded"),
                    total_delay=total_delay,
                    execution_time=time.time() - start_time
                )
            
            # Calculate delay for next retry
            delay = self._calculate_delay(attempt)
            total_delay += delay
            
            # Call on_retry callback if provided
            if self.policy.on_retry is not None:
                self.policy.on_retry(attempt, last_error or Exception("Retry based on result"), delay)
            
            # Log retry attempt
            if last_error:
                logger.info(f"Retry attempt {attempt}/{self.policy.max_attempts} after error: {last_error}. "
                           f"Waiting {delay:.2f}s before next attempt.")
            else:
                logger.info(f"Retry attempt {attempt}/{self.policy.max_attempts} based on result. "
                           f"Waiting {delay:.2f}s before next attempt.")
            
            # Wait before retrying
            time.sleep(delay)


def retry(
    max_attempts: Optional[int] = None,
    strategy: Optional[Union[RetryStrategy, str]] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    jitter: Optional[float] = None,
    timeout: Optional[float] = None,
    retryable_errors: Optional[List[Type[Exception]]] = None,
    permanent_errors: Optional[List[Type[Exception]]] = None,
    retry_on_result: Optional[Callable[[Any], bool]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    policy: Optional[RetryPolicy] = None,
    raise_on_failure: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function on specified exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        strategy: Retry delay strategy
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor for backoff calculation
        jitter: Jitter factor (0.0 to 1.0) to randomize delay
        timeout: Overall timeout in seconds (None for no timeout)
        retryable_errors: Errors that should trigger retry
        permanent_errors: Errors that should not trigger retry
        retry_on_result: Function to determine if result should trigger retry
        on_retry: Callback on retry
        policy: Retry policy (overrides other parameters if provided)
        raise_on_failure: Whether to raise an exception if all retries fail
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create retry policy
            retry_policy = policy or RetryPolicy()
            
            # Override policy parameters if provided
            if max_attempts is not None:
                retry_policy.max_attempts = max_attempts
            
            if strategy is not None:
                if isinstance(strategy, str):
                    try:
                        retry_policy.strategy = RetryStrategy(strategy)
                    except ValueError:
                        logger.warning(f"Invalid retry strategy: {strategy}. Using {retry_policy.strategy}.")
                else:
                    retry_policy.strategy = strategy
            
            if initial_delay is not None:
                retry_policy.initial_delay = initial_delay
            
            if max_delay is not None:
                retry_policy.max_delay = max_delay
            
            if backoff_factor is not None:
                retry_policy.backoff_factor = backoff_factor
            
            if jitter is not None:
                retry_policy.jitter = jitter
            
            if timeout is not None:
                retry_policy.timeout = timeout
            
            if retryable_errors is not None:
                retry_policy.retryable_errors = retryable_errors
            
            if permanent_errors is not None:
                retry_policy.permanent_errors = permanent_errors
            
            if retry_on_result is not None:
                retry_policy.retry_on_result = retry_on_result
            
            if on_retry is not None:
                retry_policy.on_retry = on_retry
            
            # Execute with retry
            handler = RetryHandler(retry_policy)
            result = handler.execute(func, *args, **kwargs)
            
            if result.success:
                return cast(T, result.result)
            elif raise_on_failure:
                if result.last_error:
                    # Wrap the original error in a RetryExhaustedError
                    raise RetryExhaustedError(
                        original_error=result.last_error,
                        attempts=result.attempts,
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.TIMEOUT if isinstance(result.last_error, TimeoutError) else ErrorCategory.UNKNOWN
                    ) from result.last_error
                else:
                    # This should not happen, but just in case
                    raise RetryExhaustedError(
                        original_error=Exception("Unknown error during retry"),
                        attempts=result.attempts,
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.UNKNOWN
                    )
            else:
                # Return None if raise_on_failure is False
                return cast(T, None)
        
        return wrapper
    
    return decorator


# Predefined retry policies
class RetryPolicies:
    """Predefined retry policies for common scenarios."""
    
    @staticmethod
    def order_submission() -> RetryPolicy:
        """Retry policy for order submission.
        
        Returns:
            RetryPolicy for order submission
        """
        return RetryPolicy(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=0.5,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=0.2,
            timeout=30.0,
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )
    
    @staticmethod
    def order_cancellation() -> RetryPolicy:
        """Retry policy for order cancellation.
        
        Returns:
            RetryPolicy for order cancellation
        """
        return RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=0.2,
            max_delay=5.0,
            backoff_factor=2.0,
            jitter=0.1,
            timeout=15.0,
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )
    
    @staticmethod
    def market_data() -> RetryPolicy:
        """Retry policy for market data requests.
        
        Returns:
            RetryPolicy for market data requests
        """
        return RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.LINEAR,
            initial_delay=1.0,
            max_delay=5.0,
            backoff_factor=1.0,
            jitter=0.1,
            timeout=10.0,
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )
    
    @staticmethod
    def api_request() -> RetryPolicy:
        """Retry policy for general API requests.
        
        Returns:
            RetryPolicy for API requests
        """
        return RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=15.0,
            backoff_factor=2.0,
            jitter=0.2,
            timeout=30.0,
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )
    
    @staticmethod
    def database_operation() -> RetryPolicy:
        """Retry policy for database operations.
        
        Returns:
            RetryPolicy for database operations
        """
        return RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=0.5,
            max_delay=5.0,
            backoff_factor=2.0,
            jitter=0.1,
            timeout=10.0,
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )
    
    @staticmethod
    def data_processing() -> RetryPolicy:
        """Retry policy for data processing operations.
        
        Returns:
            RetryPolicy for data processing operations
        """
        return RetryPolicy(
            max_attempts=2,
            strategy=RetryStrategy.FIXED,
            initial_delay=2.0,
            max_delay=2.0,
            backoff_factor=1.0,
            jitter=0.0,
            timeout=None,  # No timeout for data processing
            retryable_errors=[
                RetryableError,
                ConnectionError,
                TimeoutError
            ]
        )


class RetryPolicyFactory:
    """Factory for creating retry policies."""
    
    @staticmethod
    def create_policy(
        policy_type: str,
        max_attempts: Optional[int] = None,
        strategy: Optional[Union[RetryStrategy, str]] = None,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        jitter: Optional[float] = None,
        timeout: Optional[float] = None,
        retryable_errors: Optional[List[Type[Exception]]] = None,
        permanent_errors: Optional[List[Type[Exception]]] = None
    ) -> RetryPolicy:
        """Create a retry policy based on a predefined type with optional overrides.
        
        Args:
            policy_type: Type of policy to create (order_submission, order_cancellation, market_data, api_request, database_operation, data_processing, custom)
            max_attempts: Override for max_attempts
            strategy: Override for strategy
            initial_delay: Override for initial_delay
            max_delay: Override for max_delay
            backoff_factor: Override for backoff_factor
            jitter: Override for jitter
            timeout: Override for timeout
            retryable_errors: Override for retryable_errors
            permanent_errors: Override for permanent_errors
            
        Returns:
            RetryPolicy instance
        """
        # Get base policy based on type
        if policy_type == "order_submission":
            policy = RetryPolicies.order_submission()
        elif policy_type == "order_cancellation":
            policy = RetryPolicies.order_cancellation()
        elif policy_type == "market_data":
            policy = RetryPolicies.market_data()
        elif policy_type == "api_request":
            policy = RetryPolicies.api_request()
        elif policy_type == "database_operation":
            policy = RetryPolicies.database_operation()
        elif policy_type == "data_processing":
            policy = RetryPolicies.data_processing()
        elif policy_type == "custom":
            policy = RetryPolicy()
        else:
            logger.warning(f"Unknown policy type: {policy_type}. Using custom policy.")
            policy = RetryPolicy()
        
        # Apply overrides if provided
        if max_attempts is not None:
            policy.max_attempts = max_attempts
        
        if strategy is not None:
            if isinstance(strategy, str):
                try:
                    policy.strategy = RetryStrategy(strategy)
                except ValueError:
                    logger.warning(f"Invalid retry strategy: {strategy}. Using {policy.strategy}.")
            else:
                policy.strategy = strategy
        
        if initial_delay is not None:
            policy.initial_delay = initial_delay
        
        if max_delay is not None:
            policy.max_delay = max_delay
        
        if backoff_factor is not None:
            policy.backoff_factor = backoff_factor
        
        if jitter is not None:
            policy.jitter = jitter
        
        if timeout is not None:
            policy.timeout = timeout
        
        if retryable_errors is not None:
            policy.retryable_errors = retryable_errors
        
        if permanent_errors is not None:
            policy.permanent_errors = permanent_errors
        
        return policy
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> RetryPolicy:
        """Create a retry policy from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RetryPolicy instance
        """
        # Extract policy type if specified
        policy_type = config.pop("policy_type", "custom")
        
        # Extract strategy if specified and convert to enum
        strategy = config.pop("strategy", None)
        if strategy is not None and isinstance(strategy, str):
            try:
                strategy = RetryStrategy(strategy)
            except ValueError:
                logger.warning(f"Invalid retry strategy: {strategy}. Using default.")
                strategy = None
        
        # Create policy with remaining config as overrides
        return RetryPolicyFactory.create_policy(policy_type, strategy=strategy, **config)