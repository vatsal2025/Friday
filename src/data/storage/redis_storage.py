"""Redis storage module for the Friday AI Trading System.

This module provides a class for storing and retrieving data from Redis.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union

from src.data.storage.data_storage import DataStorage, StorageError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Try to import redis, but don't fail if it's not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not available. RedisStorage will not be functional.")


class RedisStorage(DataStorage):
    """Class for storing and retrieving data from Redis.
    
    This class provides methods for storing and retrieving data from Redis,
    including support for various data types and serialization formats.
    """
    
    def __init__(self, config=None, host="localhost", port=6379, db=0, password=None, **kwargs):
        """Initialize a Redis storage backend.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password.
            **kwargs: Additional arguments to pass to the Redis client.
            
        Raises:
            StorageError: If Redis is not available.
        """
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise StorageError("Redis package not available. Please install it with 'pip install redis'.")
        
        try:
            # Get Redis configuration from config if available
            if self.config is not None:
                redis_config = self.config.get("redis", {})
                host = redis_config.get("host", host)
                port = redis_config.get("port", port)
                db = redis_config.get("db", db)
                password = redis_config.get("password", password)
            
            # Initialize Redis client
            self.client = redis.Redis(host=host, port=port, db=db, password=password, **kwargs)
            
            # Test connection
            self.client.ping()
            
            logger.info(f"Connected to Redis at {host}:{port}, db={db}")
        
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            raise StorageError(f"Error connecting to Redis: {str(e)}")
    
    def store(self, key: str, data: Any, expiry: Optional[int] = None, 
             serialization: str = "json") -> bool:
        """Store data in Redis.
        
        Args:
            key: The key to store the data under.
            data: The data to store.
            expiry: The expiry time in seconds. If None, the data will not expire.
            serialization: The serialization format to use. Options: "json", "pickle", "string".
            
        Returns:
            True if the data was stored successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during storage.
        """
        try:
            # Serialize data
            if serialization == "json":
                serialized_data = json.dumps(data)
            elif serialization == "pickle":
                serialized_data = pickle.dumps(data)
            elif serialization == "string":
                serialized_data = str(data)
            else:
                raise StorageError(f"Invalid serialization format: {serialization}. Options: 'json', 'pickle', 'string'.")
            
            # Store data
            self.client.set(key, serialized_data, ex=expiry)
            
            logger.debug(f"Stored data under key '{key}' with serialization '{serialization}'")
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing data in Redis: {str(e)}")
            raise StorageError(f"Error storing data in Redis: {str(e)}")
    
    def retrieve(self, key: str, serialization: str = "json") -> Any:
        """Retrieve data from Redis.
        
        Args:
            key: The key to retrieve the data from.
            serialization: The serialization format to use. Options: "json", "pickle", "string".
            
        Returns:
            The retrieved data, or None if the key does not exist.
            
        Raises:
            StorageError: If an error occurs during retrieval.
        """
        try:
            # Retrieve data
            data = self.client.get(key)
            
            # Return None if key does not exist
            if data is None:
                return None
            
            # Deserialize data
            if serialization == "json":
                return json.loads(data)
            elif serialization == "pickle":
                return pickle.loads(data)
            elif serialization == "string":
                return data.decode("utf-8")
            else:
                raise StorageError(f"Invalid serialization format: {serialization}. Options: 'json', 'pickle', 'string'.")
        
        except Exception as e:
            logger.error(f"Error retrieving data from Redis: {str(e)}")
            raise StorageError(f"Error retrieving data from Redis: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """Delete data from Redis.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the data was deleted successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during deletion.
        """
        try:
            # Delete data
            result = self.client.delete(key)
            
            logger.debug(f"Deleted key '{key}' from Redis")
            
            return result > 0
        
        except Exception as e:
            logger.error(f"Error deleting data from Redis: {str(e)}")
            raise StorageError(f"Error deleting data from Redis: {str(e)}")
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key exists, False otherwise.
            
        Raises:
            StorageError: If an error occurs during the check.
        """
        try:
            # Check if key exists
            return self.client.exists(key) > 0
        
        except Exception as e:
            logger.error(f"Error checking if key exists in Redis: {str(e)}")
            raise StorageError(f"Error checking if key exists in Redis: {str(e)}")
    
    def expire(self, key: str, expiry: int) -> bool:
        """Set an expiry time for a key in Redis.
        
        Args:
            key: The key to set the expiry time for.
            expiry: The expiry time in seconds.
            
        Returns:
            True if the expiry time was set successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during the operation.
        """
        try:
            # Set expiry time
            return self.client.expire(key, expiry)
        
        except Exception as e:
            logger.error(f"Error setting expiry time for key in Redis: {str(e)}")
            raise StorageError(f"Error setting expiry time for key in Redis: {str(e)}")
    
    def close(self):
        """Close the Redis connection.
        
        Raises:
            StorageError: If an error occurs during closure.
        """
        try:
            # Close connection
            self.client.close()
            
            logger.info("Closed Redis connection")
        
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
            raise StorageError(f"Error closing Redis connection: {str(e)}")