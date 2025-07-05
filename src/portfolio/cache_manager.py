import logging
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Tuple
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

# Type variable for generic caching
T = TypeVar('T')

class CacheEntry(Generic[T]):
    """Class representing a cached entry with expiration."""
    
    def __init__(self, value: T, expiration: float):
        """Initialize a cache entry.
        
        Args:
            value: The value to cache
            expiration: Expiration time as Unix timestamp
        """
        self.value = value
        self.expiration = expiration
        self.created_at = time.time()
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired.
        
        Returns:
            True if expired, False otherwise
        """
        return time.time() > self.expiration
    
    def access(self) -> T:
        """Access the cached value and update access count.
        
        Returns:
            The cached value
        """
        self.access_count += 1
        return self.value

class CacheManager:
    """Cache manager for portfolio data and calculations.
    
    This class provides caching functionality for frequently accessed portfolio data
    and expensive calculations to improve performance.
    
    Features:
    - Time-based cache expiration
    - Manual cache invalidation
    - Cache statistics tracking
    - Decorator for easy function result caching
    """
    
    def __init__(self, default_ttl: float = 60.0):
        """Initialize the cache manager.
        
        Args:
            default_ttl: Default time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        
        logger.info(f"CacheManager initialized with default TTL of {default_ttl} seconds")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired():
                logger.debug(f"Cache entry expired for key: {key}")
                del self.cache[key]
                self.misses += 1
                return None
            else:
                self.hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry.access()
        else:
            self.misses += 1
            logger.debug(f"Cache miss for key: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        expiration = time.time() + ttl
        self.cache[key] = CacheEntry(value, expiration)
        logger.debug(f"Cached value for key: {key} with TTL: {ttl}s")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and invalidated, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            self.invalidations += 1
            logger.debug(f"Invalidated cache entry for key: {key}")
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: String pattern to match against cache keys
            
        Returns:
            Number of invalidated entries
        """
        keys_to_invalidate = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_invalidate:
            del self.cache[key]
            self.invalidations += 1
        
        logger.debug(f"Invalidated {len(keys_to_invalidate)} cache entries matching pattern: {pattern}")
        return len(keys_to_invalidate)
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of cleared entries
        """
        count = len(self.cache)
        self.cache.clear()
        self.invalidations += count
        logger.info(f"Cleared all {count} cache entries")
        return count
    
    def cleanup_expired(self) -> int:
        """Remove all expired cache entries.
        
        Returns:
            Number of removed entries
        """
        keys_to_remove = [k for k, v in self.cache.items() if v.is_expired()]
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.debug(f"Cleaned up {len(keys_to_remove)} expired cache entries")
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "hit_ratio": hit_ratio,
            "memory_usage_estimate": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes.
        
        Returns:
            Estimated memory usage in bytes
        """
        # This is a very rough estimate
        return sum(len(str(k)) + len(str(v.value)) * 2 for k, v in self.cache.items())

    def cached(self, ttl: Optional[float] = None):
        """Decorator for caching function results.
        
        Args:
            ttl: Time-to-live for the cached result in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create a cache key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Calculate result and cache it
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator