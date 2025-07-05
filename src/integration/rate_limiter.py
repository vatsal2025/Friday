"""Rate limiting and throttling for external API requests.

This module provides utilities for rate limiting and throttling requests to external APIs
to prevent overloading external systems and avoid being blocked due to excessive requests.
"""

from typing import Dict, Optional, Callable, Any, List, Tuple, Union
import time
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError

# Create logger
logger = get_logger(__name__)


class RateLimitError(FridayError):
    """Exception raised when rate limits are exceeded."""
    pass


class RateLimitStrategy(Enum):
    """Strategies for handling rate limit exceeded situations."""
    FAIL = 'fail'  # Raise an exception when rate limit is exceeded
    WAIT = 'wait'  # Wait until the rate limit resets
    QUEUE = 'queue'  # Queue the request for later execution


class TokenBucket:
    """Token bucket algorithm implementation for rate limiting.
    
    The token bucket algorithm allows for bursts of requests up to a certain limit,
    while still maintaining a long-term rate limit.
    """
    
    def __init__(self, rate: float, capacity: int):
        """Initialize a token bucket.
        
        Args:
            rate: The rate at which tokens are added to the bucket (tokens per second).
            capacity: The maximum number of tokens the bucket can hold.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.RLock()
        
    def _refill(self):
        """Refill the bucket based on the time elapsed since the last refill."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
        
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket.
        
        Args:
            tokens: The number of tokens to consume.
            
        Returns:
            bool: True if tokens were consumed, False otherwise.
        """
        with self.lock:
            self._refill()
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False
            
    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available and consume them.
        
        Args:
            tokens: The number of tokens to consume.
            timeout: The maximum time to wait in seconds. If None, wait indefinitely.
            
        Returns:
            bool: True if tokens were consumed, False if timeout was reached.
        """
        start_time = time.time()
        while True:
            with self.lock:
                self._refill()
                if tokens <= self.tokens:
                    self.tokens -= tokens
                    return True
                    
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            # Wait a bit before trying again
            time.sleep(0.01)
            
    def get_wait_time(self, tokens: int = 1) -> float:
        """Calculate the time to wait for the requested tokens to become available.
        
        Args:
            tokens: The number of tokens needed.
            
        Returns:
            float: The time to wait in seconds.
        """
        with self.lock:
            self._refill()
            if tokens <= self.tokens:
                return 0.0
            additional_tokens_needed = tokens - self.tokens
            return additional_tokens_needed / self.rate


class SlidingWindowCounter:
    """Sliding window counter implementation for rate limiting.
    
    The sliding window counter tracks the number of requests in a sliding time window,
    providing more accurate rate limiting than fixed windows.
    """
    
    def __init__(self, window_size: int, max_requests: int):
        """Initialize a sliding window counter.
        
        Args:
            window_size: The size of the window in seconds.
            max_requests: The maximum number of requests allowed in the window.
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: List[float] = []
        self.lock = threading.RLock()
        
    def _cleanup(self):
        """Remove requests that are outside the current window."""
        now = time.time()
        cutoff = now - self.window_size
        self.requests = [t for t in self.requests if t > cutoff]
        
    def check(self) -> bool:
        """Check if a request can be made.
        
        Returns:
            bool: True if the request can be made, False otherwise.
        """
        with self.lock:
            self._cleanup()
            return len(self.requests) < self.max_requests
            
    def record(self):
        """Record a request."""
        with self.lock:
            self._cleanup()
            self.requests.append(time.time())
            
    def try_record(self) -> bool:
        """Try to record a request if it doesn't exceed the limit.
        
        Returns:
            bool: True if the request was recorded, False otherwise.
        """
        with self.lock:
            self._cleanup()
            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return True
            return False
            
    def get_wait_time(self) -> float:
        """Calculate the time to wait before the next request can be made.
        
        Returns:
            float: The time to wait in seconds.
        """
        with self.lock:
            self._cleanup()
            if len(self.requests) < self.max_requests:
                return 0.0
            oldest = min(self.requests)
            return (oldest + self.window_size) - time.time()


class RateLimiter:
    """Rate limiter for external API requests.
    
    This class provides rate limiting functionality for external API requests,
    supporting different rate limiting algorithms and strategies.
    """
    
    def __init__(self, system_id: str, strategy: RateLimitStrategy = RateLimitStrategy.WAIT):
        """Initialize a rate limiter.
        
        Args:
            system_id: The ID of the external system.
            strategy: The strategy to use when rate limits are exceeded.
        """
        self.system_id = system_id
        self.strategy = strategy
        self.limiters: Dict[str, Union[TokenBucket, SlidingWindowCounter]] = {}
        self.lock = threading.RLock()
        
    def add_token_bucket(self, name: str, rate: float, capacity: int):
        """Add a token bucket limiter.
        
        Args:
            name: The name of the limiter.
            rate: The rate at which tokens are added to the bucket (tokens per second).
            capacity: The maximum number of tokens the bucket can hold.
        """
        with self.lock:
            self.limiters[name] = TokenBucket(rate, capacity)
            
    def add_sliding_window(self, name: str, window_size: int, max_requests: int):
        """Add a sliding window limiter.
        
        Args:
            name: The name of the limiter.
            window_size: The size of the window in seconds.
            max_requests: The maximum number of requests allowed in the window.
        """
        with self.lock:
            self.limiters[name] = SlidingWindowCounter(window_size, max_requests)
            
    def check_limit(self, name: str) -> bool:
        """Check if a request can be made.
        
        Args:
            name: The name of the limiter to check.
            
        Returns:
            bool: True if the request can be made, False otherwise.
            
        Raises:
            RateLimitError: If the limiter does not exist.
        """
        with self.lock:
            limiter = self.limiters.get(name)
            if not limiter:
                raise RateLimitError(f"Rate limiter '{name}' does not exist")
                
            if isinstance(limiter, TokenBucket):
                return limiter.consume()
            elif isinstance(limiter, SlidingWindowCounter):
                return limiter.check()
            else:
                raise RateLimitError(f"Unknown limiter type: {type(limiter)}")
                
    def record_request(self, name: str):
        """Record a request.
        
        Args:
            name: The name of the limiter to record the request for.
            
        Raises:
            RateLimitError: If the limiter does not exist.
        """
        with self.lock:
            limiter = self.limiters.get(name)
            if not limiter:
                raise RateLimitError(f"Rate limiter '{name}' does not exist")
                
            if isinstance(limiter, SlidingWindowCounter):
                limiter.record()
            else:
                raise RateLimitError(f"Cannot record request for limiter type: {type(limiter)}")
                
    def try_acquire(self, name: str, tokens: int = 1) -> bool:
        """Try to acquire tokens or record a request.
        
        Args:
            name: The name of the limiter to use.
            tokens: The number of tokens to consume (only for token bucket).
            
        Returns:
            bool: True if the tokens were acquired or the request was recorded, False otherwise.
            
        Raises:
            RateLimitError: If the limiter does not exist.
        """
        with self.lock:
            limiter = self.limiters.get(name)
            if not limiter:
                raise RateLimitError(f"Rate limiter '{name}' does not exist")
                
            if isinstance(limiter, TokenBucket):
                return limiter.consume(tokens)
            elif isinstance(limiter, SlidingWindowCounter):
                return limiter.try_record()
            else:
                raise RateLimitError(f"Unknown limiter type: {type(limiter)}")
                
    def wait_for_limit(self, name: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for the rate limit to allow a request.
        
        Args:
            name: The name of the limiter to use.
            tokens: The number of tokens to consume (only for token bucket).
            timeout: The maximum time to wait in seconds. If None, wait indefinitely.
            
        Returns:
            bool: True if the tokens were acquired or the request was recorded, False if timeout was reached.
            
        Raises:
            RateLimitError: If the limiter does not exist.
        """
        with self.lock:
            limiter = self.limiters.get(name)
            if not limiter:
                raise RateLimitError(f"Rate limiter '{name}' does not exist")
                
        start_time = time.time()
        while True:
            # Try to acquire
            if self.try_acquire(name, tokens):
                return True
                
            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            # Wait before trying again
            wait_time = self.get_wait_time(name, tokens)
            time.sleep(min(wait_time, 0.1))  # Wait at most 100ms at a time
            
    def get_wait_time(self, name: str, tokens: int = 1) -> float:
        """Calculate the time to wait before the next request can be made.
        
        Args:
            name: The name of the limiter to use.
            tokens: The number of tokens needed (only for token bucket).
            
        Returns:
            float: The time to wait in seconds.
            
        Raises:
            RateLimitError: If the limiter does not exist.
        """
        with self.lock:
            limiter = self.limiters.get(name)
            if not limiter:
                raise RateLimitError(f"Rate limiter '{name}' does not exist")
                
            if isinstance(limiter, TokenBucket):
                return limiter.get_wait_time(tokens)
            elif isinstance(limiter, SlidingWindowCounter):
                return limiter.get_wait_time()
            else:
                raise RateLimitError(f"Unknown limiter type: {type(limiter)}")
                
    def limit(self, name: str, tokens: int = 1):
        """Decorator for rate limiting a function.
        
        Args:
            name: The name of the limiter to use.
            tokens: The number of tokens to consume (only for token bucket).
            
        Returns:
            Callable: The decorated function.
            
        Raises:
            RateLimitError: If the rate limit is exceeded and the strategy is FAIL.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check if the rate limit allows the request
                if self.strategy == RateLimitStrategy.FAIL:
                    if not self.try_acquire(name, tokens):
                        wait_time = self.get_wait_time(name, tokens)
                        raise RateLimitError(
                            f"Rate limit exceeded for '{name}'. Try again in {wait_time:.2f} seconds."
                        )
                elif self.strategy == RateLimitStrategy.WAIT:
                    self.wait_for_limit(name, tokens)
                elif self.strategy == RateLimitStrategy.QUEUE:
                    # For QUEUE strategy, we just wait for the limit
                    self.wait_for_limit(name, tokens)
                else:
                    raise RateLimitError(f"Unknown rate limit strategy: {self.strategy}")
                    
                # Execute the function
                return func(*args, **kwargs)
            return wrapper
        return decorator


class RateLimiterRegistry:
    """Registry for rate limiters.
    
    This class provides a central registry for rate limiters, allowing them to be
    accessed by system ID.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RateLimiterRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
            
    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._limiters: Dict[str, RateLimiter] = {}
                self._initialized = True
                
    def get_limiter(self, system_id: str) -> RateLimiter:
        """Get a rate limiter for a system.
        
        Args:
            system_id: The ID of the external system.
            
        Returns:
            RateLimiter: The rate limiter for the system.
        """
        with self._lock:
            if system_id not in self._limiters:
                self._limiters[system_id] = RateLimiter(system_id)
            return self._limiters[system_id]
            
    def register_limiter(self, limiter: RateLimiter):
        """Register a rate limiter.
        
        Args:
            limiter: The rate limiter to register.
        """
        with self._lock:
            self._limiters[limiter.system_id] = limiter
            
    def unregister_limiter(self, system_id: str):
        """Unregister a rate limiter.
        
        Args:
            system_id: The ID of the external system.
        """
        with self._lock:
            if system_id in self._limiters:
                del self._limiters[system_id]
                
    def get_all_limiters(self) -> Dict[str, RateLimiter]:
        """Get all registered rate limiters.
        
        Returns:
            Dict[str, RateLimiter]: A dictionary of all registered rate limiters.
        """
        with self._lock:
            return self._limiters.copy()


def get_rate_limiter(system_id: str) -> RateLimiter:
    """Get a rate limiter for a system.
    
    Args:
        system_id: The ID of the external system.
        
    Returns:
        RateLimiter: The rate limiter for the system.
    """
    registry = RateLimiterRegistry()
    return registry.get_limiter(system_id)


def configure_rate_limiter_from_config(system_id: str, config: Dict[str, Any]) -> RateLimiter:
    """Configure a rate limiter from a configuration dictionary.
    
    Args:
        system_id: The ID of the external system.
        config: The configuration dictionary.
        
    Returns:
        RateLimiter: The configured rate limiter.
        
    Raises:
        RateLimitError: If the configuration is invalid.
    """
    try:
        # Get the rate limiter strategy
        strategy_str = config.get('rate_limit_strategy', 'wait')
        try:
            strategy = RateLimitStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Invalid rate limit strategy: {strategy_str}. Using WAIT.")
            strategy = RateLimitStrategy.WAIT
            
        # Create the rate limiter
        limiter = RateLimiter(system_id, strategy)
        
        # Configure token bucket limiters
        token_buckets = config.get('token_buckets', [])
        for bucket in token_buckets:
            name = bucket.get('name')
            rate = bucket.get('rate')
            capacity = bucket.get('capacity')
            
            if not name or not rate or not capacity:
                logger.warning(f"Invalid token bucket configuration: {bucket}. Skipping.")
                continue
                
            limiter.add_token_bucket(name, float(rate), int(capacity))
            
        # Configure sliding window limiters
        sliding_windows = config.get('sliding_windows', [])
        for window in sliding_windows:
            name = window.get('name')
            window_size = window.get('window_size')
            max_requests = window.get('max_requests')
            
            if not name or not window_size or not max_requests:
                logger.warning(f"Invalid sliding window configuration: {window}. Skipping.")
                continue
                
            limiter.add_sliding_window(name, int(window_size), int(max_requests))
            
        # Register the limiter
        registry = RateLimiterRegistry()
        registry.register_limiter(limiter)
        
        return limiter
    except Exception as e:
        raise RateLimitError(f"Failed to configure rate limiter: {str(e)}") from e


def rate_limited(system_id: str, limiter_name: str, tokens: int = 1):
    """Decorator for rate limiting a function.
    
    Args:
        system_id: The ID of the external system.
        limiter_name: The name of the limiter to use.
        tokens: The number of tokens to consume (only for token bucket).
        
    Returns:
        Callable: The decorated function.
        
    Raises:
        RateLimitError: If the rate limit is exceeded and the strategy is FAIL.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(system_id)
            return limiter.limit(limiter_name, tokens)(func)(*args, **kwargs)
        return wrapper
    return decorator