"""Database verification module for the Friday AI Trading System.

This module provides functions for verifying the database connections and configurations.
"""

import time
from typing import Dict, List, Any, Optional, Tuple

from src.infrastructure.cache import get_redis_client
from src.infrastructure.database.mongodb import get_mongo_client, get_database
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


def verify_mongodb_connection() -> Tuple[bool, Dict[str, Any]]:
    """Verify the MongoDB connection.

    This function verifies that the MongoDB connection is working correctly.
    It checks the connection, authentication, and basic operations.

    Returns:
        Tuple[bool, Dict[str, Any]]: A tuple containing a boolean indicating success or failure,
            and a dictionary with detailed results.
    """
    results = {
        "connection": False,
        "authentication": False,
        "write_test": False,
        "read_test": False,
        "delete_test": False,
        "latency_ms": None,
        "server_info": None,
        "error": None
    }
    
    try:
        # Test connection
        start_time = time.time()
        client = get_mongo_client()
        results["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Get server info
        results["server_info"] = client.server_info()
        results["connection"] = True
        
        # Test authentication by listing databases
        client.list_database_names()
        results["authentication"] = True
        
        # Test write operation
        db = get_database("test_db")
        collection = db["test_collection"]
        test_doc = {"test": "data", "timestamp": time.time()}
        insert_result = collection.insert_one(test_doc)
        results["write_test"] = insert_result.acknowledged
        
        # Test read operation
        read_doc = collection.find_one({"_id": insert_result.inserted_id})
        results["read_test"] = read_doc is not None
        
        # Test delete operation
        delete_result = collection.delete_one({"_id": insert_result.inserted_id})
        results["delete_test"] = delete_result.acknowledged
        
        # Clean up test collection
        db.drop_collection("test_collection")
        
        logger.info("MongoDB connection verified successfully")
        return all([results["connection"], results["authentication"], 
                   results["write_test"], results["read_test"], 
                   results["delete_test"]]), results
    except Exception as e:
        error_msg = f"Error verifying MongoDB connection: {str(e)}"
        logger.error(error_msg)
        results["error"] = error_msg
        return False, results


def verify_redis_connection() -> Tuple[bool, Dict[str, Any]]:
    """Verify the Redis connection.

    This function verifies that the Redis connection is working correctly.
    It checks the connection and basic operations.

    Returns:
        Tuple[bool, Dict[str, Any]]: A tuple containing a boolean indicating success or failure,
            and a dictionary with detailed results.
    """
    results = {
        "connection": False,
        "write_test": False,
        "read_test": False,
        "delete_test": False,
        "latency_ms": None,
        "server_info": None,
        "error": None
    }
    
    try:
        # Test connection
        start_time = time.time()
        client = get_redis_client()
        results["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Get server info
        results["server_info"] = client.info()
        results["connection"] = True
        
        # Test write operation
        test_key = "test:verify:key"
        test_value = f"test_data_{time.time()}"
        write_result = client.set(test_key, test_value)
        results["write_test"] = write_result
        
        # Test read operation
        read_value = client.get(test_key)
        results["read_test"] = read_value == test_value
        
        # Test delete operation
        delete_result = client.delete(test_key)
        results["delete_test"] = delete_result > 0
        
        logger.info("Redis connection verified successfully")
        return all([results["connection"], results["write_test"], 
                   results["read_test"], results["delete_test"]]), results
    except Exception as e:
        error_msg = f"Error verifying Redis connection: {str(e)}"
        logger.error(error_msg)
        results["error"] = error_msg
        return False, results


def verify_all_database_connections() -> Dict[str, Any]:
    """Verify all database connections.

    This function verifies all database connections used by the Friday AI Trading System.

    Returns:
        Dict[str, Any]: A dictionary containing the verification results for each database.
    """
    results = {}
    
    # Verify MongoDB
    mongo_success, mongo_results = verify_mongodb_connection()
    results["mongodb"] = {
        "success": mongo_success,
        "details": mongo_results
    }
    
    # Verify Redis (optional)
    try:
        redis_success, redis_results = verify_redis_connection()
        results["redis"] = {
            "success": redis_success,
            "details": redis_results
        }
    except Exception as e:
        logger.warning(f"Redis verification failed: {str(e)}. Continuing without Redis.")
        results["redis"] = {
            "success": False,
            "details": {"error": str(e)},
            "optional": True
        }
        redis_success = True  # Treat as success since it's optional
    
    # Overall success (Redis is optional)
    results["overall_success"] = mongo_success
    
    return results


def run_database_diagnostics() -> Dict[str, Any]:
    """Run diagnostics on the databases.

    This function runs diagnostics on the databases used by the Friday AI Trading System.
    It checks for potential issues and provides recommendations.

    Returns:
        Dict[str, Any]: A dictionary containing the diagnostic results.
    """
    results = {
        "mongodb": {},
        "redis": {},
        "recommendations": []
    }
    
    # MongoDB diagnostics
    try:
        client = get_mongo_client()
        db = get_database()
        
        # Check server status
        server_status = client.admin.command("serverStatus")
        results["mongodb"]["server_status"] = {
            "connections": server_status.get("connections", {}),
            "mem": server_status.get("mem", {}),
            "uptime": server_status.get("uptime", 0),
            "version": server_status.get("version", "unknown")
        }
        
        # Check database stats
        db_stats = db.command("dbStats")
        results["mongodb"]["db_stats"] = {
            "collections": db_stats.get("collections", 0),
            "objects": db_stats.get("objects", 0),
            "data_size": db_stats.get("dataSize", 0),
            "storage_size": db_stats.get("storageSize", 0),
            "indexes": db_stats.get("indexes", 0),
            "index_size": db_stats.get("indexSize", 0)
        }
        
        # Check collection stats
        results["mongodb"]["collections"] = {}
        for collection_name in db.list_collection_names():
            coll_stats = db.command("collStats", collection_name)
            results["mongodb"]["collections"][collection_name] = {
                "count": coll_stats.get("count", 0),
                "size": coll_stats.get("size", 0),
                "storage_size": coll_stats.get("storageSize", 0),
                "index_count": len(coll_stats.get("indexSizes", {})),
                "index_size": coll_stats.get("totalIndexSize", 0)
            }
        
        # Add recommendations based on MongoDB diagnostics
        if server_status.get("connections", {}).get("current", 0) > 100:
            results["recommendations"].append(
                "High number of MongoDB connections. Consider implementing connection pooling."
            )
        
        if db_stats.get("dataSize", 0) > 1_000_000_000:  # 1 GB
            results["recommendations"].append(
                "Large MongoDB database size. Consider implementing data archiving or sharding."
            )
    except Exception as e:
        results["mongodb"]["error"] = str(e)
        results["recommendations"].append(
            f"Error running MongoDB diagnostics: {str(e)}. Check MongoDB configuration."
        )
    
    # Redis diagnostics
    try:
        client = get_redis_client()
        
        # Get Redis info
        info = client.info()
        results["redis"]["info"] = {
            "redis_version": info.get("redis_version", "unknown"),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
            "total_connections_received": info.get("total_connections_received", 0),
            "total_commands_processed": info.get("total_commands_processed", 0)
        }
        
        # Get Redis memory stats
        memory_stats = client.memory_stats() if hasattr(client, "memory_stats") else {}
        results["redis"]["memory_stats"] = {
            "total": memory_stats.get("total.allocated", 0),
            "peak": memory_stats.get("peak.allocated", 0),
            "startup": memory_stats.get("startup.allocated", 0),
            "keys": memory_stats.get("db.0", {}).get("keys", 0) if "db.0" in memory_stats else 0
        }
        
        # Get Redis key stats
        db_size = client.dbsize()
        results["redis"]["key_stats"] = {
            "total_keys": db_size,
            "expires": info.get("expired_keys", 0),
            "evicted": info.get("evicted_keys", 0)
        }
        
        # Add recommendations based on Redis diagnostics
        if info.get("connected_clients", 0) > 100:
            results["recommendations"].append(
                "High number of Redis connections. Consider implementing connection pooling."
            )
        
        if info.get("used_memory", 0) > 1_000_000_000:  # 1 GB
            results["recommendations"].append(
                "High Redis memory usage. Consider implementing key expiration or using Redis Cluster."
            )
        
        if info.get("evicted_keys", 0) > 0:
            results["recommendations"].append(
                "Redis keys are being evicted. Consider increasing Redis memory or optimizing key usage."
            )
    except Exception as e:
        results["redis"]["error"] = str(e)
        results["recommendations"].append(
            f"Error running Redis diagnostics: {str(e)}. Check Redis configuration."
        )
    
    return results


if __name__ == "__main__":
    # Run verification when script is executed directly
    verification_results = verify_all_database_connections()
    print("Database Verification Results:")
    print(f"Overall Success: {verification_results['overall_success']}")
    print(f"MongoDB Success: {verification_results['mongodb']['success']}")
    print(f"Redis Success: {verification_results['redis']['success']}")
    
    if not verification_results["overall_success"]:
        print("\nErrors:")
        if not verification_results["mongodb"]["success"]:
            print(f"MongoDB Error: {verification_results['mongodb']['details'].get('error')}")
        if not verification_results["redis"]["success"]:
            print(f"Redis Error: {verification_results['redis']['details'].get('error')}")
    
    # Run diagnostics
    print("\nRunning Database Diagnostics...")
    diagnostic_results = run_database_diagnostics()
    
    if diagnostic_results["recommendations"]:
        print("\nRecommendations:")
        for i, recommendation in enumerate(diagnostic_results["recommendations"], 1):
            print(f"{i}. {recommendation}")