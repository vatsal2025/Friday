"""Retry Policy for Trading Engine.

This module provides configurable retry policies for handling transient failures
in order submissions, market data requests, and other operations that may fail temporarily.
"""

import time
import logging
import random
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)

# Generic type for function return value
T = TypeVar('T')


class RetryStrategy(Enum):
    """Strategies for retry backoff."""
    FIXED = "fixed"  # Fixed delay between retries
    LINEAR = "linear"  # Linearly increasing delay
    EXPONENTIAL = "exponential"  # Exponentially increasing delay
    FIBONACCI = "fibonacci"  # Fibonacci sequence delay
    RANDOM = "random"  # Random delay within a range
    RANDOM_EXPONENTIAL = "random_exponential"  # Random delay with exponential increase


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""
    pass


class PermanentError(Exception):
    """Base class for errors that should not trigger a retry."""
    pass


class RetryExhaustedError(Exception):
    """Error raised when all retry attempts have been exhausted."""
    def __init__(self, original_error: Exception, attempts: int):
        self.original_error = original_error
        self.attempts = attempts
        super().__init__(f"Retry exhausted after {attempts} attempts. Original error: {original_error}")


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation."""
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    last_delay: float = 0.0


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_factor: float = 2.0
    jitter: float = 0.1  # percentage of jitter to add
    timeout: Optional[float] = None  # total timeout in seconds
    retryable_errors: List[type] = field(default_factory=lambda: [RetryableError, ConnectionError, TimeoutError])
    permanent_errors: List[type] = field(default_factory=lambda: [PermanentError, ValueError, TypeError])
    retry_on_result: Optional[Callable[[Any], bool]] = None  # Function to determine if result should trigger retry
    on_retry: Optional[Callable[[int, Exception, float], None]] = None  # Callback on retry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the retry policy to a dictionary.
        
        Returns:
            Dictionary representation of the retry policy
        """
        return {
            "max_attempts": self.max_attempts,
            "strategy": self.strategy.value,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "backoff_factor": self.backoff_factor,
            "jitter": self.jitter,
            "timeout": self.timeout,
            "retryable_errors": [error.__name__ for error in self.retryable_errors],
            "permanent_errors": [error.__name__ for error in self.permanent_errors]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryPolicy':
        """Create a retry policy from a dictionary.
        
        Args:
            data: Dictionary representation of the retry policy
            
        Returns:
            RetryPolicy instance
        """
        # Handle strategy conversion
        strategy_str = data.get("strategy", RetryStrategy.EXPONENTIAL.value)
        strategy = RetryStrategy(strategy_str)
        
        # Create policy with basic parameters
        policy = cls(
            max_attempts=data.get("max_attempts", 3),
            strategy=strategy,
            initial_delay=data.get("initial_delay", 1.0),
            max_delay=data.get("max_delay", 60.0),
            backoff_factor=data.get("backoff_factor", 2.0),
            jitter=data.get("jitter", 0.1),
            timeout=data.get("timeout")
        )
        
        # Error classes need special handling - we can't reconstruct them from names
        # so we'll keep the defaults unless the caller provides actual error classes
        
        return policy


class RetryHandler:
    """Handles retry logic based on a retry policy."""
    def __init__(self, policy: RetryPolicy):
        """Initialize the retry handler.
        
        Args:
            policy: The retry policy to use
        """
        self.policy = policy
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate the delay for a retry attempt.
        
        Args:
            attempt: The current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return 0
        
        # Base delay calculation based on strategy
        if self.policy.strategy == RetryStrategy.FIXED:
            delay = self.policy.initial_delay
        
        elif self.policy.strategy == RetryStrategy.LINEAR:
            delay = self.policy.initial_delay * attempt
        
        elif self.policy.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.policy.initial_delay * (self.policy.backoff_factor ** (attempt - 1))
        
        elif self.policy.strategy == RetryStrategy.FIBONACCI:
            # Calculate Fibonacci number
            a, b = 1, 1
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = self.policy.initial_delay * a
        
        elif self.policy.strategy == RetryStrategy.RANDOM:
            min_delay = self.policy.initial_delay
            max_delay = self.policy.initial_delay * self.policy.backoff_factor * attempt
            delay = random.uniform(min_delay, max_delay)
        
        elif self.policy.strategy == RetryStrategy.RANDOM_EXPONENTIAL:
            base_delay = self.policy.initial_delay * (self.policy.backoff_factor ** (attempt - 1))
            min_delay = base_delay * (1 - self.policy.jitter)
            max_delay = base_delay * (1 + self.policy.jitter)
            delay = random.uniform(min_delay, max_delay)
        
        else:
            # Default to exponential
            delay = self.policy.initial_delay * (self.policy.backoff_factor ** (attempt - 1))
        
        # Apply jitter if not already applied
        if self.policy.strategy not in [RetryStrategy.RANDOM, RetryStrategy.RANDOM_EXPONENTIAL] and self.policy.jitter > 0:
            jitter_amount = delay * self.policy.jitter
            delay = random.uniform(delay - jitter_amount, delay + jitter_amount)
        
        # Cap at max delay
        return min(delay, self.policy.max_delay)
    
    def _should_retry(self, error: Optional[Exception], result: Any, attempt: int) -> bool:
        """Determine if a retry should be attempted.
        
        Args:
            error: The error that occurred, if any
            result: The result of the operation, if successful
            attempt: The current attempt number
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        # Check if max attempts reached
        if attempt >= self.policy.max_attempts:
            return False
        
        # If there was an error, check if it's retryable
        if error is not None:
            # Check if it's a permanent error
            for error_type in self.policy.permanent_errors:
                if isinstance(error, error_type):
                    return False
            
            # Check if it's a retryable error
            for error_type in self.policy.retryable_errors:
                if isinstance(error, error_type):
                    return True
            
            # Default to not retry if error type is not recognized
            return False
        
        # If there was no error but we have a retry_on_result function, use it
        if self.policy.retry_on_result is not None:
            return self.policy.retry_on_result(result)
        
        # No error and no retry_on_result function means success, no retry needed
        return False
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> RetryResult[T]:
        """Execute a function with retry logic.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            RetryResult containing the result or error
        """
        start_time = time.time()
        attempt = 0
        total_delay = 0.0
        last_delay = 0.0
        
        while True:
            attempt += 1
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check if we should retry based on the result
                if self._should_retry(None, result, attempt):
                    delay = self._calculate_delay(attempt)
                    last_delay = delay
                    total_delay += delay
                    
                    # Check timeout
                    if self.policy.timeout is not None and (time.time() - start_time + delay) > self.policy.timeout:
                        logger.warning(f"Retry timeout exceeded after {attempt} attempts")
                        return RetryResult(
                            success=True,  # We have a result, even if it would trigger a retry
                            result=result,
                            attempts=attempt,
                            total_delay=total_delay,
                            last_delay=last_delay
                        )
                    
                    # Call on_retry callback if provided
                    if self.policy.on_retry is not None:
                        try:
                            self.policy.on_retry(attempt, None, delay)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {callback_error}")
                    
                    logger.info(f"Retrying due to result condition (attempt {attempt}/{self.policy.max_attempts}, delay {delay:.2f}s)")
                    time.sleep(delay)
                    continue
                
                # Success without retry condition
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_delay=total_delay,
                    last_delay=last_delay
                )
            
            except Exception as error:
                # Check if we should retry
                if self._should_retry(error, None, attempt):
                    delay = self._calculate_delay(attempt)
                    last_delay = delay
                    total_delay += delay
                    
                    # Check timeout
                    if self.policy.timeout is not None and (time.time() - start_time + delay) > self.policy.timeout:
                        logger.warning(f"Retry timeout exceeded after {attempt} attempts")
                        return RetryResult(
                            success=False,
                            error=error,
                            attempts=attempt,
                            total_delay=total_delay,
                            last_delay=last_delay
                        )
                    
                    # Call on_retry callback if provided
                    if self.policy.on_retry is not None:
                        try:
                            self.policy.on_retry(attempt, error, delay)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {callback_error}")
                    
                    logger.warning(f"Retrying due to error: {error} (attempt {attempt}/{self.policy.max_attempts}, delay {delay:.2f}s)")
                    time.sleep(delay)
                    continue
                
                # Non-retryable error
                return RetryResult(
                    success=False,
                    error=error,
                    attempts=attempt,
                    total_delay=total_delay,
                    last_delay=last_delay
                )
        
        # This should never be reached
        return RetryResult(success=False, attempts=attempt, total_delay=total_delay, last_delay=last_delay)


def retry(policy: Optional[RetryPolicy] = None):
    """Decorator for applying retry logic to a function.
    
    Args:
        policy: The retry policy to use (default policy if None)
        
    Returns:
        Decorated function
    """
    if policy is None:
        policy = RetryPolicy()
    
    handler = RetryHandler(policy)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = handler.execute(func, *args, **kwargs)
            if not result.success and result.error is not None:
                raise RetryExhaustedError(result.error, result.attempts)
            return result.result
        return wrapper
    
    return decorator


# Predefined retry policies
def get_default_order_submission_policy() -> RetryPolicy:
    """Get the default retry policy for order submissions.
    
    Returns:
        RetryPolicy for order submissions
    """
    return RetryPolicy(
        max_attempts=5,
        strategy=RetryStrategy.EXPONENTIAL,
        initial_delay=0.5,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=0.2,
        timeout=120.0,  # 2 minutes total timeout
        retryable_errors=[
            RetryableError,
            ConnectionError,
            TimeoutError,
            # Add broker-specific retryable errors here
        ]
    )


def get_default_order_cancellation_policy() -> RetryPolicy:
    """Get the default retry policy for order cancellations.
    
    Returns:
        RetryPolicy for order cancellations
    """
    return RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL,
        initial_delay=0.2,
        max_delay=10.0,
        backoff_factor=2.0,
        jitter=0.1,
        timeout=30.0,  # 30 seconds total timeout
        retryable_errors=[
            RetryableError,
            ConnectionError,
            TimeoutError,
            # Add broker-specific retryable errors here
        ]
    )


def get_default_market_data_policy() -> RetryPolicy:
    """Get the default retry policy for market data requests.
    
    Returns:
        RetryPolicy for market data requests
    """
    return RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.LINEAR,
        initial_delay=0.1,
        max_delay=5.0,
        backoff_factor=1.0,
        jitter=0.1,
        timeout=10.0,  # 10 seconds total timeout
        retryable_errors=[
            RetryableError,
            ConnectionError,
            TimeoutError,
            # Add market data provider specific errors here
        ]
    )


def get_default_api_request_policy() -> RetryPolicy:
    """Get the default retry policy for general API requests.
    
    Returns:
        RetryPolicy for API requests
    """
    return RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL,
        initial_delay=0.5,
        max_delay=15.0,
        backoff_factor=2.0,
        jitter=0.1,
        timeout=30.0,  # 30 seconds total timeout
        retryable_errors=[
            RetryableError,
            ConnectionError,
            TimeoutError,
            # Add API-specific errors here
        ]
    )


class RetryPolicyFactory:
    """Factory for creating retry policies."""
    @staticmethod
    def create_policy(policy_type: str, **kwargs) -> RetryPolicy:
        """Create a retry policy based on a predefined type with optional overrides.
        
        Args:
            policy_type: Type of policy to create (order_submission, order_cancellation, market_data, api_request, custom)
            **kwargs: Override parameters for the policy
            
        Returns:
            RetryPolicy instance
        """
        # Get base policy
        if policy_type == "order_submission":
            policy = get_default_order_submission_policy()
        elif policy_type == "order_cancellation":
            policy = get_default_order_cancellation_policy()
        elif policy_type == "market_data":
            policy = get_default_market_data_policy()
        elif policy_type == "api_request":
            policy = get_default_api_request_policy()
        elif policy_type == "custom":
            policy = RetryPolicy()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(policy, key):
                # Handle special case for strategy
                if key == "strategy" and isinstance(value, str):
                    value = RetryStrategy(value)
                
                setattr(policy, key, value)
        
        return policy
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> RetryPolicy:
        """Create a retry policy from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RetryPolicy instance
        """
        policy_type = config.pop("policy_type", "custom")
        return RetryPolicyFactory.create_policy(policy_type, **config)


# Example usage of retry decorator
'''
def example_usage():
    # Define a function that might fail
    @retry(get_default_api_request_policy())
    def fetch_data(url):
        # Simulated API request that might fail
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Connection failed")
        return {"data": "success"}
    
    try:
        result = fetch_data("https://api.example.com/data")
        print(f"Success: {result}")
    except RetryExhaustedError as e:
        print(f"Failed after multiple attempts: {e}")
'''