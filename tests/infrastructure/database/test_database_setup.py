"""Tests for database setup and functionality.

This module contains tests for MongoDB and Redis database setup and functionality.
"""

import os
import sys
import unittest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.infrastructure.config import get_config
from src.infrastructure.database.mongodb import get_mongo_client, get_database, get_collection
from src.infrastructure.database.initialize_db import (
    initialize_mongodb_collections,
    initialize_redis_structures
)
from src.infrastructure.database.verify_db import verify_mongodb_connection, verify_redis_connection
from src.infrastructure.cache import get_redis_client, get_redis_cache
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class TestDatabaseSetup(unittest.TestCase):
    """Test case for database setup and functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Initialize MongoDB collections with test flag
        cls.mongo_result = initialize_mongodb_collections(force_recreate=True, test_mode=True)
        
        # Initialize Redis structures with test flag
        cls.redis_result = initialize_redis_structures(force_recreate=True, test_mode=True)
        
        # Get MongoDB database
        cls.db = get_database()
        
        # Get Redis client
        cls.redis = get_redis_client()
        
        # Get Redis cache
        cls.redis_cache = get_redis_cache(namespace="test")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up test collections
        if hasattr(cls, 'db'):
            for collection_name in cls.db.list_collection_names():
                if collection_name.startswith("test_"):
                    cls.db.drop_collection(collection_name)
        
        # Clean up test Redis keys
        if hasattr(cls, 'redis'):
            test_keys = cls.redis.keys("test:*")
            if test_keys:
                cls.redis.delete(*test_keys)
    
    def test_mongodb_connection(self):
        """Test MongoDB connection."""
        success, results = verify_mongodb_connection()
        self.assertTrue(success, f"MongoDB connection failed: {results.get('error')}")
        self.assertTrue(results["connection"], "MongoDB connection check failed")
        self.assertTrue(results["authentication"], "MongoDB authentication check failed")
        self.assertTrue(results["write_test"], "MongoDB write test failed")
        self.assertTrue(results["read_test"], "MongoDB read test failed")
        self.assertTrue(results["delete_test"], "MongoDB delete test failed")
    
    def test_redis_connection(self):
        """Test Redis connection."""
        success, results = verify_redis_connection()
        self.assertTrue(success, f"Redis connection failed: {results.get('error')}")
        self.assertTrue(results["connection"], "Redis connection check failed")
        self.assertTrue(results["write_test"], "Redis write test failed")
        self.assertTrue(results["read_test"], "Redis read test failed")
        self.assertTrue(results["delete_test"], "Redis delete test failed")
    
    def test_mongodb_collections(self):
        """Test MongoDB collections were created."""
        # Check that collections exist
        collections = self.db.list_collection_names()
        expected_collections = [
            "market_data",
            "tick_data",
            "order_book",
            "model_storage",
            "trading_strategies",
            "backtest_results",
            "trading_signals",
            "orders",
            "positions"
        ]
        
        for collection_name in expected_collections:
            self.assertIn(collection_name, collections, f"Collection {collection_name} not found")
    
    def test_mongodb_indexes(self):
        """Test MongoDB indexes were created."""
        # Check indexes on market_data collection
        market_data_indexes = self.db["market_data"].index_information()
        self.assertIn("symbol_1_exchange_1_timeframe_1", market_data_indexes, "Market data compound index not found")
        
        # Check indexes on model_storage collection
        model_indexes = self.db["model_storage"].index_information()
        self.assertIn("name_1_version_1", model_indexes, "Model storage compound index not found")
        
        # Check indexes on trading_strategies collection
        strategy_indexes = self.db["trading_strategies"].index_information()
        self.assertIn("name_1", strategy_indexes, "Trading strategy name index not found")
    
    def test_mongodb_schema_validation(self):
        """Test MongoDB schema validation."""
        # Test market data schema validation
        market_data_collection = self.db["market_data"]
        
        # Valid document should insert successfully
        valid_doc = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "timeframe": "1h",
            "data": [
                {
                    "timestamp": datetime.now(),
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 1000000
                }
            ],
            "last_updated": datetime.now()
        }
        
        result = market_data_collection.insert_one(valid_doc)
        self.assertTrue(result.acknowledged, "Valid market data document insertion failed")
        
        # Invalid document should still insert but with validation warning
        # (since validationAction is set to "warn")
        invalid_doc = {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            # Missing timeframe field
            "data": [
                {
                    "timestamp": datetime.now(),
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    # Missing volume field
                }
            ]
            # Missing last_updated field
        }
        
        # This should still insert because validationAction is "warn"
        result = market_data_collection.insert_one(invalid_doc)
        self.assertTrue(result.acknowledged, "Invalid document insertion with warn validation failed")
        
        # Clean up test documents
        market_data_collection.delete_many({"_id": {"$in": [valid_doc["_id"], invalid_doc["_id"]]}})
    
    def test_redis_basic_operations(self):
        """Test Redis basic operations."""
        # Test string operations
        test_key = "test:string:key"
        test_value = "test_value"
        
        # Set value
        self.redis.set(test_key, test_value)
        
        # Get value
        retrieved_value = self.redis.get(test_key)
        self.assertEqual(retrieved_value, test_value.encode(), "Redis string get failed")
        
        # Delete value
        self.redis.delete(test_key)
        self.assertIsNone(self.redis.get(test_key), "Redis delete failed")
    
    def test_redis_cache_operations(self):
        """Test Redis cache operations."""
        # Test cache operations
        test_key = "cache:key"
        test_value = {"name": "test", "value": 123}
        
        # Set value
        self.redis_cache.set(test_key, test_value)
        
        # Get value
        retrieved_value = self.redis_cache.get(test_key)
        self.assertEqual(retrieved_value, test_value, "Redis cache get failed")
        
        # Test expiry
        self.redis_cache.set(test_key, test_value, expire=1)  # 1 second expiry
        time.sleep(1.5)  # Wait for expiry
        self.assertIsNone(self.redis_cache.get(test_key), "Redis cache expiry failed")
        
        # Test hash operations
        hash_key = "cache:hash"
        hash_value = {"field1": "value1", "field2": "value2"}
        
        # Set hash
        self.redis_cache.hset(hash_key, mapping=hash_value)
        
        # Get hash
        retrieved_hash = self.redis_cache.hgetall(hash_key)
        self.assertEqual(retrieved_hash, hash_value, "Redis cache hash operations failed")
        
        # Test list operations
        list_key = "cache:list"
        list_values = ["item1", "item2", "item3"]
        
        # Push to list
        for value in list_values:
            self.redis_cache.rpush(list_key, value)
        
        # Get list range
        retrieved_list = self.redis_cache.lrange(list_key, 0, -1)
        self.assertEqual(retrieved_list, list_values, "Redis cache list operations failed")
        
        # Clean up test keys
        self.redis_cache.delete(test_key)
        self.redis_cache.delete(hash_key)
        self.redis_cache.delete(list_key)
    
    def test_market_data_operations(self):
        """Test market data operations."""
        # Get market data collection
        market_data_collection = self.db["market_data"]
        
        # Create test market data
        test_data = {
            "symbol": "TEST",
            "exchange": "TEST_EXCHANGE",
            "timeframe": "1h",
            "data": [
                {
                    "timestamp": datetime.now() - timedelta(hours=2),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000
                },
                {
                    "timestamp": datetime.now() - timedelta(hours=1),
                    "open": 102.0,
                    "high": 107.0,
                    "low": 101.0,
                    "close": 106.0,
                    "volume": 1200
                }
            ],
            "last_updated": datetime.now()
        }
        
        # Insert test data
        result = market_data_collection.insert_one(test_data)
        self.assertTrue(result.acknowledged, "Market data insertion failed")
        
        # Query test data
        query = {
            "symbol": "TEST",
            "exchange": "TEST_EXCHANGE",
            "timeframe": "1h"
        }
        retrieved_data = market_data_collection.find_one(query)
        
        self.assertIsNotNone(retrieved_data, "Market data query failed")
        self.assertEqual(retrieved_data["symbol"], test_data["symbol"], "Market data symbol mismatch")
        self.assertEqual(len(retrieved_data["data"]), len(test_data["data"]), "Market data length mismatch")
        
        # Update test data
        new_candle = {
            "timestamp": datetime.now(),
            "open": 106.0,
            "high": 110.0,
            "low": 104.0,
            "close": 108.0,
            "volume": 1500
        }
        
        update_result = market_data_collection.update_one(
            query,
            {
                "$push": {"data": new_candle},
                "$set": {"last_updated": datetime.now()}
            }
        )
        
        self.assertTrue(update_result.acknowledged, "Market data update failed")
        self.assertEqual(update_result.modified_count, 1, "Market data update count mismatch")
        
        # Verify update
        updated_data = market_data_collection.find_one(query)
        self.assertEqual(len(updated_data["data"]), len(test_data["data"]) + 1, "Market data update verification failed")
        
        # Clean up test data
        delete_result = market_data_collection.delete_one({"_id": result.inserted_id})
        self.assertTrue(delete_result.acknowledged, "Market data deletion failed")
    
    def test_model_storage_operations(self):
        """Test model storage operations."""
        # Get model storage collection
        model_collection = self.db["model_storage"]
        
        # Create test model data
        test_model = {
            "name": "TestModel",
            "description": "Test model for unit testing",
            "type": "classification",
            "framework": "tensorflow",
            "version": "1.0.0",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "optimizer": "adam"
            },
            "storage_path": "/models/test_model.h5",
            "is_active": True
        }
        
        # Insert test model
        result = model_collection.insert_one(test_model)
        self.assertTrue(result.acknowledged, "Model storage insertion failed")
        
        # Query test model
        query = {
            "name": "TestModel",
            "version": "1.0.0"
        }
        retrieved_model = model_collection.find_one(query)
        
        self.assertIsNotNone(retrieved_model, "Model storage query failed")
        self.assertEqual(retrieved_model["name"], test_model["name"], "Model name mismatch")
        self.assertEqual(retrieved_model["version"], test_model["version"], "Model version mismatch")
        
        # Update test model
        update_result = model_collection.update_one(
            query,
            {
                "$set": {
                    "metrics.accuracy": 0.87,
                    "updated_at": datetime.now()
                }
            }
        )
        
        self.assertTrue(update_result.acknowledged, "Model storage update failed")
        self.assertEqual(update_result.modified_count, 1, "Model storage update count mismatch")
        
        # Verify update
        updated_model = model_collection.find_one(query)
        self.assertEqual(updated_model["metrics"]["accuracy"], 0.87, "Model storage update verification failed")
        
        # Clean up test model
        delete_result = model_collection.delete_one({"_id": result.inserted_id})
        self.assertTrue(delete_result.acknowledged, "Model storage deletion failed")


if __name__ == "__main__":
    unittest.main()