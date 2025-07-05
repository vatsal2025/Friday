"""Cache module for the Friday AI Trading System.

This module provides functions for caching data using Redis.
It includes support for various data types, namespaces, and advanced features like pub/sub.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, Callable

from src.infrastructure.cache.redis_cache import RedisCache
from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Global Redis client instance
_redis_client = None

# Global RedisCache instances by namespace
_redis_cache_instances = {}


def get_redis_client():
    """Get the Redis client instance.

    Returns:
        redis.Redis: The Redis client instance.
    """
    global _redis_client

    if _redis_client is None:
        try:
            import redis
            
            # Get Redis configuration
            redis_config = get_config("redis")
            
            # Create Redis client
            _redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                password=redis_config["password"],
                decode_responses=False,  # Let RedisCache handle decoding
                socket_timeout=redis_config["socket_timeout"],
                socket_connect_timeout=redis_config["socket_connect_timeout"],
                retry_on_timeout=redis_config["retry_on_timeout"],
            )
            
            # Test connection
            _redis_client.ping()
            logger.info("Connected to Redis server at %s:%s", redis_config["host"], redis_config["port"])
        except ImportError:
            logger.error("Redis package not installed. Install it with 'pip install redis'")
            raise
        except Exception as e:
            logger.error("Failed to connect to Redis server: %s", str(e))
            raise

    return _redis_client


def get_redis_cache(namespace: str = "", serializer: str = "json") -> RedisCache:
    """Get a RedisCache instance for the given namespace.

    Args:
        namespace: The namespace for the cache keys.
        serializer: The serializer to use for values ("json" or "pickle").

    Returns:
        RedisCache: The RedisCache instance.
    """
    global _redis_cache_instances

    # Create a unique key for this cache configuration
    cache_key = f"{namespace}:{serializer}"

    # Return existing instance if available
    if cache_key in _redis_cache_instances:
        return _redis_cache_instances[cache_key]

    # Create a new instance
    client = get_redis_client()
    cache = RedisCache(client, namespace, serializer)
    _redis_cache_instances[cache_key] = cache
    return cache


def set_value(key: str, value: Any, expiry: Optional[int] = None) -> bool:
    """Set a value in the cache using the default namespace.

    Args:
        key: The key to set.
        value: The value to set.
        expiry: The expiry time in seconds. If None, the value will not expire.

    Returns:
        bool: True if the value was set, False otherwise.
    """
    cache = get_redis_cache()
    return cache.set(key, value, expiry)


def get_value(key: str, default: Optional[Any] = None) -> Any:
    """Get a value from the cache using the default namespace.

    Args:
        key: The key to get.
        default: The default value to return if the key does not exist.

    Returns:
        The value from the cache, or the default value if the key does not exist.
    """
    cache = get_redis_cache()
    return cache.get(key, default)


def delete_value(key: str) -> bool:
    """Delete a value from the cache using the default namespace.

    Args:
        key: The key to delete.

    Returns:
        bool: True if the value was deleted, False otherwise.
    """
    cache = get_redis_cache()
    return cache.delete(key)


def exists(key: str) -> bool:
    """Check if a key exists in the cache using the default namespace.

    Args:
        key: The key to check.

    Returns:
        bool: True if the key exists, False otherwise.
    """
    cache = get_redis_cache()
    return cache.exists(key)


def expire(key: str, seconds: int) -> bool:
    """Set the expiry time for a key using the default namespace.

    Args:
        key: The key to set the expiry time for.
        seconds: The expiry time in seconds.

    Returns:
        bool: True if the expiry time was set, False otherwise.
    """
    cache = get_redis_cache()
    return cache.expire(key, seconds)


def ttl(key: str) -> int:
    """Get the time to live for a key using the default namespace.

    Args:
        key: The key to get the time to live for.

    Returns:
        int: The time to live in seconds, or -1 if the key has no expiry, or -2 if the key does not exist.
    """
    cache = get_redis_cache()
    return cache.ttl(key)


def incr(key: str, amount: int = 1) -> int:
    """Increment a key by the given amount using the default namespace.

    Args:
        key: The key to increment.
        amount: The amount to increment by.

    Returns:
        int: The new value, or -1 if there was an error.
    """
    cache = get_redis_cache()
    return cache.incr(key, amount)


def decr(key: str, amount: int = 1) -> int:
    """Decrement a key by the given amount using the default namespace.

    Args:
        key: The key to decrement.
        amount: The amount to decrement by.

    Returns:
        int: The new value, or -1 if there was an error.
    """
    cache = get_redis_cache()
    return cache.decr(key, amount)


def get_or_set(key: str, default_value_func: Callable[[], Any], expire: int = None) -> Any:
    """Get a value from the cache, or set it if it doesn't exist using the default namespace.

    Args:
        key: The key to get or set.
        default_value_func: A function that returns the default value to set if the key doesn't exist.
        expire: The expiry time in seconds.

    Returns:
        Any: The value from the cache, or the default value if the key was not found.
    """
    cache = get_redis_cache()
    return cache.get_or_set(key, default_value_func, expire)


def cache_decorator(key_prefix: str, expire: int = None):
    """Decorator for caching function results using the default namespace.

    Args:
        key_prefix: The prefix for the cache key.
        expire: The expiry time in seconds.

    Returns:
        Callable: The decorated function.
    """
    cache = get_redis_cache()
    return cache.cache_decorator(key_prefix, expire)