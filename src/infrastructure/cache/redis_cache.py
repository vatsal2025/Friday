"""Redis cache implementation for the Friday AI Trading System.

This module provides a Redis cache implementation for caching data in the Friday AI Trading System.
It includes support for various data types, namespaces, and advanced features like pub/sub.
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set

from redis import Redis
import warnings

from ..logging import get_logger

# Create logger
logger = get_logger(__name__)


class RedisCache:
    """Redis cache implementation.

    This class provides a Redis cache implementation for caching data in the Friday AI Trading System.
    It supports various data types, namespaces, and advanced features like pub/sub.
    """

    def __init__(self, redis_client: Optional[Redis] = None, namespace: str = "", serializer: str = "json"):
        """Initialize the Redis cache.

        Args:
            redis_client: The Redis client instance.
            namespace: The namespace for the cache keys.
            serializer: The serializer to use for values ("json" or "pickle").
        """
        if redis_client is None:
            warnings.warn("Redis client is not provided. Redis operations will be no-op.")
            self.redis_client = None
        else:
            self.redis_client = redis_client
        self.namespace = namespace
        self.serializer = serializer

    def _get_namespaced_key(self, key: str) -> str:
        """Get a namespaced key.

        Args:
            key: The key to namespace.

        Returns:
            str: The namespaced key.
        """
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set a value in the cache.

        Args:
            key: The key to set.
            value: The value to set.
            expiry: The expiry time in seconds. If None, the value will not expire.

        Returns:
            bool: True if the value was set, False otherwise.
        """
        try:
            if self.redis_client is None:
                return False

            namespaced_key = self._get_namespaced_key(key)
            self.redis_client.set(namespaced_key, value, ex=expiry)
            return True
        except Exception as e:
            logger.error("Failed to set cache value: %s", str(e))
            return False

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value from the cache.

        Args:
            key: The key to get.
            default: The default value to return if the key does not exist.

        Returns:
            The value from the cache, or the default value if the key does not exist.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            value = self.redis_client.get(namespaced_key)
            return value if value is not None else default
        except Exception as e:
            logger.error("Failed to get cache value: %s", str(e))
            return default

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: The key to delete.

        Returns:
            bool: True if the value was deleted, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            self.redis_client.delete(namespaced_key)
            return True
        except Exception as e:
            logger.error("Failed to delete cache value: %s", str(e))
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return bool(self.client.exists(namespaced_key))
        except Exception as e:
            logger.error("Failed to check if key exists: %s", str(e))
            return False

    def expire(self, key: str, expiry: int) -> bool:
        """Set an expiry time for a key.

        Args:
            key: The key to set the expiry for.
            expiry: The expiry time in seconds.

        Returns:
            bool: True if the expiry was set, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return bool(self.client.expire(namespaced_key, expiry))
        except Exception as e:
            logger.error("Failed to set expiry: %s", str(e))
            return False

    def ttl(self, key: str) -> int:
        """Get the time to live for a key.

        Args:
            key: The key to get the TTL for.

        Returns:
            int: The TTL in seconds, or -1 if the key does not exist or has no expiry.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.ttl(namespaced_key)
        except Exception as e:
            logger.error("Failed to get TTL: %s", str(e))
            return -1

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a value in the cache.

        Args:
            key: The key to increment.
            amount: The amount to increment by.

        Returns:
            int: The new value, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.incrby(namespaced_key, amount)
        except Exception as e:
            logger.error("Failed to increment value: %s", str(e))
            return -1

    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a value in the cache.

        Args:
            key: The key to decrement.
            amount: The amount to decrement by.

        Returns:
            int: The new value, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.decrby(namespaced_key, amount)
        except Exception as e:
            logger.error("Failed to decrement value: %s", str(e))
            return -1

    # Hash operations
    def hset(self, key: str, field: str, value: Any) -> bool:
        """Set a hash field in the cache.

        Args:
            key: The key of the hash.
            field: The field to set.
            value: The value to set.

        Returns:
            bool: True if the field was set, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            self.client.hset(namespaced_key, field, value)
            return True
        except Exception as e:
            logger.error("Failed to set hash field: %s", str(e))
            return False

    def hget(self, key: str, field: str, default: Optional[Any] = None) -> Any:
        """Get a hash field from the cache.

        Args:
            key: The key of the hash.
            field: The field to get.
            default: The default value to return if the field does not exist.

        Returns:
            The value of the field, or the default value if the field does not exist.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            value = self.client.hget(namespaced_key, field)
            return value if value is not None else default
        except Exception as e:
            logger.error("Failed to get hash field: %s", str(e))
            return default

    def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all fields and values from a hash.

        Args:
            key: The key of the hash.

        Returns:
            Dict[str, Any]: A dictionary of fields and values.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.hgetall(namespaced_key)
        except Exception as e:
            logger.error("Failed to get all hash fields: %s", str(e))
            return {}

    def hdel(self, key: str, field: str) -> bool:
        """Delete a hash field from the cache.

        Args:
            key: The key of the hash.
            field: The field to delete.

        Returns:
            bool: True if the field was deleted, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return bool(self.client.hdel(namespaced_key, field))
        except Exception as e:
            logger.error("Failed to delete hash field: %s", str(e))
            return False

    def hexists(self, key: str, field: str) -> bool:
        """Check if a hash field exists in the cache.

        Args:
            key: The key of the hash.
            field: The field to check.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return bool(self.client.hexists(namespaced_key, field))
        except Exception as e:
            logger.error("Failed to check if hash field exists: %s", str(e))
            return False

    # List operations
    def lpush(self, key: str, *values: Any) -> int:
        """Push values to the head of a list.

        Args:
            key: The key of the list.
            *values: The values to push.

        Returns:
            int: The length of the list after the push, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.lpush(namespaced_key, *values)
        except Exception as e:
            logger.error("Failed to push to list: %s", str(e))
            return -1

    def rpush(self, key: str, *values: Any) -> int:
        """Push values to the tail of a list.

        Args:
            key: The key of the list.
            *values: The values to push.

        Returns:
            int: The length of the list after the push, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.rpush(namespaced_key, *values)
        except Exception as e:
            logger.error("Failed to push to list: %s", str(e))
            return -1

    def lpop(self, key: str) -> Any:
        """Pop a value from the head of a list.

        Args:
            key: The key of the list.

        Returns:
            The popped value, or None if the list is empty or the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.lpop(namespaced_key)
        except Exception as e:
            logger.error("Failed to pop from list: %s", str(e))
            return None

    def rpop(self, key: str) -> Any:
        """Pop a value from the tail of a list.

        Args:
            key: The key of the list.

        Returns:
            The popped value, or None if the list is empty or the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.rpop(namespaced_key)
        except Exception as e:
            logger.error("Failed to pop from list: %s", str(e))
            return None

    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get a range of values from a list.

        Args:
            key: The key of the list.
            start: The start index.
            end: The end index.

        Returns:
            List[Any]: The range of values.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.lrange(namespaced_key, start, end)
        except Exception as e:
            logger.error("Failed to get range from list: %s", str(e))
            return []

    def llen(self, key: str) -> int:
        """Get the length of a list.

        Args:
            key: The key of the list.

        Returns:
            int: The length of the list, or 0 if the list does not exist or the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.llen(namespaced_key)
        except Exception as e:
            logger.error("Failed to get list length: %s", str(e))
            return 0

    # Set operations
    def sadd(self, key: str, *values: Any) -> int:
        """Add values to a set.

        Args:
            key: The key of the set.
            *values: The values to add.

        Returns:
            int: The number of values added to the set, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.sadd(namespaced_key, *values)
        except Exception as e:
            logger.error("Failed to add to set: %s", str(e))
            return -1

    def smembers(self, key: str) -> Set[Any]:
        """Get all members of a set.

        Args:
            key: The key of the set.

        Returns:
            Set[Any]: The members of the set.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.smembers(namespaced_key)
        except Exception as e:
            logger.error("Failed to get set members: %s", str(e))
            return set()

    def srem(self, key: str, *values: Any) -> int:
        """Remove values from a set.

        Args:
            key: The key of the set.
            *values: The values to remove.

        Returns:
            int: The number of values removed from the set, or -1 if the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.srem(namespaced_key, *values)
        except Exception as e:
            logger.error("Failed to remove from set: %s", str(e))
            return -1

    def sismember(self, key: str, value: Any) -> bool:
        """Check if a value is a member of a set.

        Args:
            key: The key of the set.
            value: The value to check.

        Returns:
            bool: True if the value is a member of the set, False otherwise.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return bool(self.client.sismember(namespaced_key, value))
        except Exception as e:
            logger.error("Failed to check set membership: %s", str(e))
            return False

    def scard(self, key: str) -> int:
        """Get the number of members in a set.

        Args:
            key: The key of the set.

        Returns:
            int: The number of members in the set, or 0 if the set does not exist or the operation failed.
        """
        try:
            namespaced_key = self._get_namespaced_key(key)
            return self.client.scard(namespaced_key)
        except Exception as e:
            logger.error("Failed to get set cardinality: %s", str(e))
            return 0

    # Serialization methods
    def set_json(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set a JSON value in the cache.

        Args:
            key: The key to set.
            value: The value to set (will be JSON serialized).
            expiry: The expiry time in seconds. If None, the value will not expire.

        Returns:
            bool: True if the value was set, False otherwise.
        """
        try:
            serialized = json.dumps(value)
            return self.set(key, serialized, expiry)
        except Exception as e:
            logger.error("Failed to set JSON value: %s", str(e))
            return False

    def get_json(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a JSON value from the cache.

        Args:
            key: The key to get.
            default: The default value to return if the key does not exist.

        Returns:
            The deserialized JSON value, or the default value if the key does not exist.
        """
        try:
            value = self.get(key)
            if value is None:
                return default
            return json.loads(value)
        except Exception as e:
            logger.error("Failed to get JSON value: %s", str(e))
            return default

    def set_pickle(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set a pickled value in the cache.

        Args:
            key: The key to set.
            value: The value to set (will be pickled).
            expiry: The expiry time in seconds. If None, the value will not expire.

        Returns:
            bool: True if the value was set, False otherwise.
        """
        try:
            serialized = pickle.dumps(value)
            return self.set(key, serialized, expiry)
        except Exception as e:
            logger.error("Failed to set pickled value: %s", str(e))
            return False

    def get_pickle(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a pickled value from the cache.

        Args:
            key: The key to get.
            default: The default value to return if the key does not exist.

        Returns:
            The unpickled value, or the default value if the key does not exist.
        """
        try:
            value = self.get(key)
            if value is None:
                return default
            return pickle.loads(value)
        except Exception as e:
            logger.error("Failed to get pickled value: %s", str(e))
            return default

    # Utility methods
    def flush(self) -> bool:
        """Flush all keys in the current namespace.

        Returns:
            bool: True if the keys were flushed, False otherwise.
        """
        try:
            if not self.namespace:
                logger.warning("Flushing all keys in Redis is not allowed. Use a namespace.")
                return False

            pattern = f"{self.namespace}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error("Failed to flush keys: %s", str(e))
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern.

        Args:
            pattern: The pattern to match.

        Returns:
            List[str]: The matching keys.
        """
        try:
            if self.namespace:
                pattern = f"{self.namespace}:{pattern}"
            keys = self.client.keys(pattern)
            if self.namespace:
                # Remove namespace prefix from keys
                prefix_len = len(self.namespace) + 1  # +1 for the colon
                return [key[prefix_len:] for key in keys]
            return keys
        except Exception as e:
            logger.error("Failed to get keys: %s", str(e))
            return []
