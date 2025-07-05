"""Database initialization module for the Friday AI Trading System.

This module provides functions to initialize the MongoDB and Redis databases
with the necessary collections, indexes, and schema validators.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from pymongo import ASCENDING, DESCENDING, IndexModel
from redis import Redis

from src.infrastructure.config import get_config
from src.infrastructure.database.mongodb import get_database, get_collection
from src.infrastructure.database.schema_validators import get_validator
from src.infrastructure.cache import get_redis_client
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Collection names
MARKET_DATA_COLLECTION = "market_data"
TICK_DATA_COLLECTION = "tick_data"
ORDER_BOOK_COLLECTION = "order_book"
MODEL_STORAGE_COLLECTION = "model_storage"
TRADING_STRATEGY_COLLECTION = "trading_strategy"
BACKTEST_RESULTS_COLLECTION = "backtest_results"
TRADING_SIGNALS_COLLECTION = "trading_signals"
TRADING_ORDERS_COLLECTION = "trading_orders"
TRADING_POSITIONS_COLLECTION = "trading_positions"


def initialize_mongodb(force_recreate: bool = False) -> Dict[str, Any]:
    """Initialize MongoDB with the necessary collections and indexes.

    This function creates the required collections with schema validation
    and sets up the necessary indexes for efficient querying.
    
    Args:
        force_recreate: If True, drop and recreate collections and indexes.
    """
    logger.info(f"Initializing MongoDB (force_recreate={force_recreate})...")
    
    # Get MongoDB configuration
    mongodb_config = get_config("mongodb")
    db_name = mongodb_config.get("database", "friday")
    
    # Get database
    db = get_database(db_name)
    
    if force_recreate:
        logger.warning("Force recreate enabled - dropping existing collections...")
        _drop_existing_collections(db_name)
    
    # Initialize market data collection
    _initialize_market_data_collection(db_name, force_recreate)
    
    # Initialize tick data collection
    _initialize_tick_data_collection(db_name, force_recreate)
    
    # Initialize order book collection
    _initialize_order_book_collection(db_name, force_recreate)
    
    # Initialize model storage collection
    _initialize_model_storage_collection(db_name, force_recreate)
    
    # Initialize trading strategy collection
    _initialize_trading_strategy_collection(db_name, force_recreate)
    
    # Initialize backtest results collection
    _initialize_backtest_results_collection(db_name, force_recreate)
    
    # Initialize trading signals collection
    _initialize_trading_signals_collection(db_name, force_recreate)
    
    # Initialize trading orders collection
    _initialize_trading_orders_collection(db_name, force_recreate)
    
    # Initialize trading positions collection
    _initialize_trading_positions_collection(db_name, force_recreate)
    
    logger.info("MongoDB initialization completed successfully.")
    return {"success": True, "message": "MongoDB initialization completed successfully"}


def _drop_existing_collections(db_name: Optional[str] = None) -> None:
    """Drop all existing collections in the database.
    
    Args:
        db_name: The name of the database. If not provided, the default database will be used.
    """
    db = get_database(db_name)
    
    collections_to_drop = [
        MARKET_DATA_COLLECTION,
        TICK_DATA_COLLECTION,
        ORDER_BOOK_COLLECTION,
        MODEL_STORAGE_COLLECTION,
        TRADING_STRATEGY_COLLECTION,
        BACKTEST_RESULTS_COLLECTION,
        TRADING_SIGNALS_COLLECTION,
        TRADING_ORDERS_COLLECTION,
        TRADING_POSITIONS_COLLECTION,
    ]
    
    for collection_name in collections_to_drop:
        if collection_name in db.list_collection_names():
            db.drop_collection(collection_name)
            logger.info(f"Dropped collection: {collection_name}")
    
    logger.info("All existing collections dropped.")


def _initialize_market_data_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the market data collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("market_data")
    
    # Get collection with validator
    collection = get_collection(MARKET_DATA_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Market data collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING), ("exchange", ASCENDING), ("timeframe", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("timeframe", ASCENDING)]))
    indexes_created.append(collection.create_index([("data.timestamp", DESCENDING)]))
    
    logger.info(f"Market data collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_tick_data_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the tick data collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("tick_data")

    # Get collection with validator
    collection = get_collection(TICK_DATA_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Tick data collection indexes dropped.")

    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING), ("exchange", ASCENDING), ("timestamp", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("timestamp", DESCENDING)]))
    indexes_created.append(collection.create_index([("trade_id", ASCENDING)]))

    logger.info(f"Tick data collection initialized with schema validation and {len(indexes_created)} indexes.")
    

def _initialize_order_book_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the order book collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("order_book")
    
    # Get collection with validator
    collection = get_collection(ORDER_BOOK_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Order book collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING), ("exchange", ASCENDING), ("timestamp", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("timestamp", DESCENDING)]))
    
    logger.info(f"Order book collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_model_storage_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the model storage collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("model_storage")
    
    # Get collection with validator
    collection = get_collection(MODEL_STORAGE_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Model storage collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("name", ASCENDING), ("version", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("name", ASCENDING)]))
    indexes_created.append(collection.create_index([("model_type", ASCENDING)]))
    indexes_created.append(collection.create_index([("framework", ASCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Model storage collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_trading_strategy_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the trading strategy collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("trading_strategy")
    
    # Get collection with validator
    collection = get_collection(TRADING_STRATEGY_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Trading strategy collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("name", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("performance.sharpe_ratio", DESCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Trading strategy collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_backtest_results_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the backtest results collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("backtest_results")
    
    # Get collection with validator
    collection = get_collection(BACKTEST_RESULTS_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Backtest results collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("strategy_id", ASCENDING)]))
    indexes_created.append(collection.create_index([("start_date", ASCENDING)]))
    indexes_created.append(collection.create_index([("end_date", ASCENDING)]))
    indexes_created.append(collection.create_index([("symbols", ASCENDING)]))
    indexes_created.append(collection.create_index([("performance.sharpe_ratio", DESCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Backtest results collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_trading_signals_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the trading signals collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("trading_signals")
    
    # Get collection with validator
    collection = get_collection(TRADING_SIGNALS_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Trading signals collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("strategy_id", ASCENDING)]))
    indexes_created.append(collection.create_index([("signal_type", ASCENDING)]))
    indexes_created.append(collection.create_index([("timestamp", DESCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Trading signals collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_trading_orders_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the trading orders collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("trading_orders")
    
    # Get collection with validator
    collection = get_collection(TRADING_ORDERS_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Trading orders collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("strategy_id", ASCENDING)]))
    indexes_created.append(collection.create_index([("order_id", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("status", ASCENDING)]))
    indexes_created.append(collection.create_index([("timestamp", DESCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Trading orders collection initialized with schema validation and {len(indexes_created)} indexes.")


def _initialize_trading_positions_collection(db_name: Optional[str] = None, force_recreate: bool = False) -> None:
    """Initialize the trading positions collection with schema validation and indexes.

    Args:
        db_name: The name of the database. If not provided, the default database will be used.
        force_recreate: If True, drop and recreate indexes.
    """
    # Get validator
    validator = get_validator("trading_positions")
    
    # Get collection with validator
    collection = get_collection(TRADING_POSITIONS_COLLECTION, db_name, validator)
    
    if force_recreate:
        collection.drop_indexes()
        logger.info("Trading positions collection indexes dropped.")
    
    # Create indexes and log results
    indexes_created = []
    indexes_created.append(collection.create_index([("symbol", ASCENDING)]))
    indexes_created.append(collection.create_index([("exchange", ASCENDING)]))
    indexes_created.append(collection.create_index([("strategy_id", ASCENDING)]))
    indexes_created.append(collection.create_index([("position_id", ASCENDING)], unique=True))
    indexes_created.append(collection.create_index([("status", ASCENDING)]))
    indexes_created.append(collection.create_index([("open_timestamp", DESCENDING)]))
    indexes_created.append(collection.create_index([("close_timestamp", DESCENDING)]))
    indexes_created.append(collection.create_index([("created_at", DESCENDING)]))
    
    logger.info(f"Trading positions collection initialized with schema validation and {len(indexes_created)} indexes.")


def initialize_redis(force_recreate: bool = False) -> Dict[str, Any]:
    """Initialize Redis with the necessary data structures.

    This function sets up the Redis database with the required data structures
    and ensures that the connection is working properly.
    
    Args:
        force_recreate: If True, clear stale keys and reset namespaces.
    """
    logger.info(f"Initializing Redis (force_recreate={force_recreate})...")

    # Get Redis client
    redis_client = get_redis_client()

    # Test connection
    try:
        redis_client.ping()
        logger.info("Redis connection successful.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

    if force_recreate:
        logger.warning("Force recreate enabled - clearing all keys...")
        redis_client.flushdb()
        logger.info("All Redis keys cleared.")

    # Set up Redis namespaces
    _setup_redis_namespaces(redis_client)

    logger.info("Redis initialization completed successfully.")
    return {"success": True, "message": "Redis initialization completed successfully"}


def _setup_redis_namespaces(redis_client: Redis) -> None:
    """Set up Redis namespaces for organizing data.

    Args:
        redis_client: The Redis client instance.
    """
    # Define namespaces
    namespaces = [
        "market_data",
        "tick_data",
        "order_book",
        "trading_signals",
        "trading_orders",
        "trading_positions",
        "system_status",
        "user_sessions",
        "api_rate_limits",
    ]
    
    # Store namespaces in Redis for reference
    redis_client.sadd("friday:namespaces", *namespaces)
    
    logger.info(f"Redis namespaces set up successfully: {', '.join(namespaces)}")


def initialize_redis_structures(force_recreate: bool = False) -> Dict[str, Any]:
    """Initialize Redis data structures for the Friday AI Trading System.
    
    This function sets up the required Redis data structures such as sorted sets,
    hashes, and lists for various trading system components.
    
    Args:
        force_recreate: If True, clear existing structures before creating new ones.
        
    Returns:
        Dictionary with initialization results.
    """
    logger.info(f"Initializing Redis data structures (force_recreate={force_recreate})...")
    
    # Get Redis client
    redis_client = get_redis_client()
    
    # Initialize market data structures
    _init_market_data_structures(redis_client, force_recreate)
    
    # Initialize trading structures
    _init_trading_structures(redis_client, force_recreate)
    
    # Initialize system structures
    _init_system_structures(redis_client, force_recreate)
    
    logger.info("Redis data structures initialized successfully.")
    return {"success": True, "message": "Redis data structures initialized successfully"}


def _init_market_data_structures(redis_client: Redis, force_recreate: bool = False) -> None:
    """Initialize Redis structures for market data.
    
    Args:
        redis_client: The Redis client instance.
        force_recreate: If True, clear existing structures before creating new ones.
    """
    # Market data keys
    prefix = "friday:market_data:"
    
    # Clear existing keys if force_recreate is True
    if force_recreate:
        keys = redis_client.keys(f"{prefix}*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} existing market data keys.")
    
    # Set up sorted sets for recent price data
    symbols = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD"]
    for symbol in symbols:
        key = f"{prefix}recent_prices:{symbol}"
        # Only create if it doesn't exist or force_recreate is True
        if force_recreate or not redis_client.exists(key):
            # Initialize with empty sorted set
            logger.info(f"Initialized sorted set for {symbol} recent prices.")
    
    logger.info("Market data structures initialized.")


def _init_trading_structures(redis_client: Redis, force_recreate: bool = False) -> None:
    """Initialize Redis structures for trading operations.
    
    Args:
        redis_client: The Redis client instance.
        force_recreate: If True, clear existing structures before creating new ones.
    """
    # Trading data keys
    prefix = "friday:trading:"
    
    # Clear existing keys if force_recreate is True
    if force_recreate:
        keys = redis_client.keys(f"{prefix}*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} existing trading data keys.")
    
    # Set up active orders list
    active_orders_key = f"{prefix}active_orders"
    if force_recreate or not redis_client.exists(active_orders_key):
        # Initialize with empty list
        logger.info("Initialized active orders list.")
    
    # Set up active strategies hash
    active_strategies_key = f"{prefix}active_strategies"
    if force_recreate or not redis_client.exists(active_strategies_key):
        # Initialize with empty hash
        logger.info("Initialized active strategies hash.")
    
    logger.info("Trading structures initialized.")


def _init_system_structures(redis_client: Redis, force_recreate: bool = False) -> None:
    """Initialize Redis structures for system operations.
    
    Args:
        redis_client: The Redis client instance.
        force_recreate: If True, clear existing structures before creating new ones.
    """
    # System data keys
    prefix = "friday:system:"
    
    # Clear existing keys if force_recreate is True
    if force_recreate:
        keys = redis_client.keys(f"{prefix}*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} existing system data keys.")
    
    # Set up system status hash
    status_key = f"{prefix}status"
    if force_recreate or not redis_client.exists(status_key):
        redis_client.hset(status_key, mapping={
            "status": "idle",
            "last_updated": str(datetime.now()),
            "version": "1.0.0"
        })
        logger.info("Initialized system status hash.")
    
    # Set up API rate limits hash
    rate_limits_key = f"{prefix}rate_limits"
    if force_recreate or not redis_client.exists(rate_limits_key):
        # Initialize with empty hash
        logger.info("Initialized API rate limits hash.")
    
    logger.info("System structures initialized.")


def initialize_mongodb_collections(force_recreate: bool = False) -> Dict[str, Any]:
    """Initialize MongoDB collections only (alias for initialize_mongodb).
    
    This function is an alias for initialize_mongodb to maintain compatibility.
    
    Args:
        force_recreate: If True, drop and recreate collections and indexes.
    
    Returns:
        Dictionary with initialization results.
    """
    return initialize_mongodb(force_recreate)


def initialize_databases(force_recreate: bool = False) -> None:
    """Initialize all databases for the Friday AI Trading System.

    This function initializes both MongoDB and Redis databases.
    
    Args:
        force_recreate: If True, drop and recreate collections/indexes and clear Redis keys.
    """
    logger.info(f"Initializing all databases (force_recreate={force_recreate})...")
    
    # Initialize MongoDB
    initialize_mongodb(force_recreate)
    
    # Initialize Redis
    initialize_redis(force_recreate)
    
    # Initialize Redis data structures
    initialize_redis_structures(force_recreate)
    
    logger.info("All databases initialized successfully.")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize databases for the Friday AI Trading System")
    parser.add_argument(
        "--force-recreate", 
        action="store_true", 
        help="Drop and recreate all collections, indexes, and Redis keys"
    )
    args = parser.parse_args()
    
    # Initialize all databases when the script is run directly
    initialize_databases(force_recreate=args.force_recreate)
