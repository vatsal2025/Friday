"""Redis initialization module for the Friday AI Trading System.

This module provides functions for initializing Redis with the required data structures.
"""

import json
from typing import Dict, List, Any, Optional

from src.infrastructure.cache import get_redis_cache
from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


def initialize_redis() -> bool:
    """Initialize Redis with the required data structures.

    This function sets up the required Redis data structures for the Friday AI Trading System.
    It creates namespaces for different types of data and initializes any required keys.

    Returns:
        bool: True if Redis was initialized successfully, False otherwise.
    """
    try:
        # Get Redis configuration
        redis_config = get_config("redis")
        
        # Initialize market data cache
        market_data_cache = get_redis_cache(namespace="market_data")
        
        # Initialize model cache
        model_cache = get_redis_cache(namespace="models")
        
        # Initialize trading cache
        trading_cache = get_redis_cache(namespace="trading")
        
        # Initialize user cache
        user_cache = get_redis_cache(namespace="users")
        
        # Initialize system cache
        system_cache = get_redis_cache(namespace="system")
        
        # Set system status
        system_cache.set("status", "initialized")
        system_cache.set("version", "1.0.0")
        
        # Initialize rate limiting
        rate_limit_cache = get_redis_cache(namespace="rate_limit")
        
        # Initialize session cache
        session_cache = get_redis_cache(namespace="sessions")
        
        # Initialize notification cache
        notification_cache = get_redis_cache(namespace="notifications")
        
        # Initialize job queue
        job_queue_cache = get_redis_cache(namespace="job_queue")
        
        logger.info("Redis initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Redis: {e}")
        return False


def setup_market_data_cache() -> bool:
    """Set up the market data cache.

    This function sets up the market data cache with the required data structures.

    Returns:
        bool: True if the market data cache was set up successfully, False otherwise.
    """
    try:
        cache = get_redis_cache(namespace="market_data")
        
        # Create hash for tracking available symbols
        if not cache.exists("available_symbols"):
            cache.hset("available_symbols", "last_updated", "")
        
        # Create sorted sets for each timeframe
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        for timeframe in timeframes:
            # Create a sorted set for tracking data freshness by symbol
            cache.delete(f"freshness:{timeframe}")
        
        logger.info("Market data cache set up successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up market data cache: {e}")
        return False


def setup_model_cache() -> bool:
    """Set up the model cache.

    This function sets up the model cache with the required data structures.

    Returns:
        bool: True if the model cache was set up successfully, False otherwise.
    """
    try:
        cache = get_redis_cache(namespace="models")
        
        # Create hash for tracking available models
        if not cache.exists("available_models"):
            cache.hset("available_models", "last_updated", "")
        
        # Create hash for model metadata
        if not cache.exists("model_metadata"):
            cache.hset("model_metadata", "last_updated", "")
        
        logger.info("Model cache set up successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up model cache: {e}")
        return False


def setup_trading_cache() -> bool:
    """Set up the trading cache.

    This function sets up the trading cache with the required data structures.

    Returns:
        bool: True if the trading cache was set up successfully, False otherwise.
    """
    try:
        cache = get_redis_cache(namespace="trading")
        
        # Create hash for active orders
        if not cache.exists("active_orders"):
            cache.hset("active_orders", "last_updated", "")
        
        # Create hash for positions
        if not cache.exists("positions"):
            cache.hset("positions", "last_updated", "")
        
        # Create sorted set for signals
        cache.delete("signals")
        
        logger.info("Trading cache set up successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up trading cache: {e}")
        return False


def clear_all_caches() -> bool:
    """Clear all Redis caches.

    This function clears all Redis caches used by the Friday AI Trading System.
    It should be used with caution, as it will delete all cached data.

    Returns:
        bool: True if all caches were cleared successfully, False otherwise.
    """
    try:
        namespaces = [
            "market_data", 
            "models", 
            "trading", 
            "users", 
            "system",
            "rate_limit",
            "sessions",
            "notifications",
            "job_queue"
        ]
        
        for namespace in namespaces:
            cache = get_redis_cache(namespace=namespace)
            cache.clear_namespace()
        
        logger.info("All Redis caches cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing Redis caches: {e}")
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the Redis caches.

    This function returns statistics about the Redis caches used by the Friday AI Trading System.

    Returns:
        Dict[str, Any]: A dictionary containing statistics about the Redis caches.
    """
    try:
        stats = {}
        namespaces = [
            "market_data", 
            "models", 
            "trading", 
            "users", 
            "system",
            "rate_limit",
            "sessions",
            "notifications",
            "job_queue"
        ]
        
        for namespace in namespaces:
            cache = get_redis_cache(namespace=namespace)
            keys = cache.scan_keys()
            stats[namespace] = {
                "key_count": len(keys),
                "keys": keys[:10] + ["..."] if len(keys) > 10 else keys
            }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting Redis cache statistics: {e}")
        return {"error": str(e)}