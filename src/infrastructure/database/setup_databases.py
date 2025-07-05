"""Database setup script for the Friday AI Trading System.

This script initializes all required database components, including MongoDB collections
and Redis data structures.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional

from src.infrastructure.config import get_config
from src.infrastructure.database.mongodb import get_mongo_client, get_database
from src.infrastructure.database.initialize_db import (
    initialize_mongodb,
    initialize_redis
)
from src.infrastructure.database.verify_db import verify_all_database_connections
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


def setup_all_databases(force_recreate: bool = False) -> Dict[str, Any]:
    """Set up all required databases for the Friday AI Trading System.

    This function initializes MongoDB collections and Redis data structures.

    Args:
        force_recreate (bool): Whether to force recreation of collections and indexes.

    Returns:
        Dict[str, Any]: A dictionary containing the setup results.
    """
    results = {
        "mongodb": None,
        "redis": None,
        "verification": None,
        "overall_success": False
    }
    
    try:
        # Initialize MongoDB collections
        logger.info("Initializing MongoDB collections...")
        mongo_result = initialize_mongodb(force_recreate=force_recreate)
        results["mongodb"] = mongo_result
        
        # Initialize Redis structures (optional)
        logger.info("Initializing Redis structures...")
        try:
            redis_result = initialize_redis(force_recreate=force_recreate)
            results["redis"] = redis_result
        except Exception as e:
            logger.warning(f"Redis initialization failed: {str(e)}. Continuing without Redis.")
            results["redis"] = {"success": False, "error": str(e), "optional": True}
        
        # Verify database connections
        logger.info("Verifying database connections...")
        verification_result = verify_all_database_connections()
        results["verification"] = verification_result
        
        # Set overall success (Redis is optional)
        redis_success = results["redis"].get("success", False) or results["redis"].get("optional", False)
        results["overall_success"] = (
            mongo_result.get("success", False) and 
            redis_success and 
            verification_result.get("overall_success", False)
        )
        
        if results["overall_success"]:
            logger.info("Database setup completed successfully")
        else:
            logger.error("Database setup failed")
        
        return results
    except Exception as e:
        logger.error(f"Error setting up databases: {str(e)}")
        results["error"] = str(e)
        return results


def create_test_data(sample_size: int = 10) -> Dict[str, Any]:
    """Create test data in the databases.

    This function creates sample data for testing purposes.

    Args:
        sample_size (int): Number of sample records to create.

    Returns:
        Dict[str, Any]: A dictionary containing the results.
    """
    from datetime import datetime, timedelta
    import random
    import uuid
    
    results = {
        "market_data": None,
        "models": None,
        "trading": None,
        "overall_success": False
    }
    
    try:
        # Get MongoDB database
        db = get_database()
        
        # Create sample market data
        market_data_collection = db["market_data"]
        market_data_samples = []
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        exchanges = ["NYSE", "NASDAQ"]
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Generate sample OHLCV data
        for symbol in symbols[:min(sample_size, len(symbols))]:
            for timeframe in timeframes[:min(3, len(timeframes))]:
                # Create a series of candles
                candles = []
                base_time = datetime.now() - timedelta(days=30)
                base_price = random.uniform(100, 1000)
                
                for i in range(30):  # 30 days of data
                    timestamp = base_time + timedelta(days=i)
                    price_change = random.uniform(-5, 5)
                    open_price = base_price + price_change
                    high_price = open_price + random.uniform(0, 3)
                    low_price = open_price - random.uniform(0, 3)
                    close_price = random.uniform(low_price, high_price)
                    volume = random.randint(1000, 10000)
                    
                    candle = {
                        "timestamp": timestamp,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume
                    }
                    candles.append(candle)
                    base_price = close_price
                
                # Create market data document
                market_data_doc = {
                    "symbol": symbol,
                    "exchange": random.choice(exchanges),
                    "timeframe": timeframe,
                    "data": candles,
                    "last_updated": datetime.now()
                }
                
                market_data_samples.append(market_data_doc)
        
        # Insert market data
        if market_data_samples:
            market_data_collection.insert_many(market_data_samples)
            results["market_data"] = {
                "success": True,
                "count": len(market_data_samples)
            }
        
        # Create sample model data
        model_collection = db["model_storage"]
        model_samples = []
        
        model_types = ["classification", "regression", "time_series"]
        frameworks = ["tensorflow", "pytorch", "scikit-learn"]
        
        for i in range(min(sample_size, 5)):
            # Create model document
            model_doc = {
                "name": f"Model_{i+1}",
                "description": f"Sample model {i+1} for testing",
                "type": random.choice(model_types),
                "framework": random.choice(frameworks),
                "version": "1.0.0",
                "created_at": datetime.now() - timedelta(days=random.randint(1, 30)),
                "updated_at": datetime.now(),
                "metrics": {
                    "accuracy": random.uniform(0.7, 0.99),
                    "precision": random.uniform(0.7, 0.99),
                    "recall": random.uniform(0.7, 0.99),
                    "f1_score": random.uniform(0.7, 0.99)
                },
                "parameters": {
                    "learning_rate": random.uniform(0.001, 0.1),
                    "batch_size": random.choice([16, 32, 64, 128]),
                    "epochs": random.randint(10, 100),
                    "optimizer": random.choice(["adam", "sgd", "rmsprop"])
                },
                "storage_path": f"/models/model_{i+1}.pkl",
                "is_active": random.choice([True, False])
            }
            
            model_samples.append(model_doc)
        
        # Insert model data
        if model_samples:
            model_collection.insert_many(model_samples)
            results["models"] = {
                "success": True,
                "count": len(model_samples)
            }
        
        # Create sample trading data
        trading_strategy_collection = db["trading_strategy"]
        strategy_samples = []
        
        strategy_types = ["trend_following", "mean_reversion", "breakout", "statistical_arbitrage"]
        risk_levels = ["low", "medium", "high"]
        
        for i in range(min(sample_size, 5)):
            # Create strategy document
            strategy_doc = {
                "name": f"Strategy_{i+1}",
                "description": f"Sample trading strategy {i+1} for testing",
                "type": random.choice(strategy_types),
                "risk_level": random.choice(risk_levels),
                "parameters": {
                    "entry_threshold": random.uniform(0.1, 0.5),
                    "exit_threshold": random.uniform(0.1, 0.5),
                    "stop_loss": random.uniform(0.02, 0.1),
                    "take_profit": random.uniform(0.02, 0.1)
                },
                "symbols": random.sample(symbols, random.randint(1, len(symbols))),
                "timeframes": random.sample(timeframes, random.randint(1, len(timeframes))),
                "created_at": datetime.now() - timedelta(days=random.randint(1, 30)),
                "updated_at": datetime.now(),
                "is_active": random.choice([True, False]),
                "performance": {
                    "win_rate": random.uniform(0.4, 0.7),
                    "profit_factor": random.uniform(1.1, 2.0),
                    "sharpe_ratio": random.uniform(0.5, 2.0),
                    "max_drawdown": random.uniform(0.05, 0.2)
                }
            }
            
            strategy_samples.append(strategy_doc)
        
        # Insert strategy data
        if strategy_samples:
            trading_strategy_collection.insert_many(strategy_samples)
            results["trading"] = {
                "success": True,
                "count": len(strategy_samples)
            }
        
        # Set overall success
        results["overall_success"] = all([
            results["market_data"].get("success", False) if results["market_data"] else False,
            results["models"].get("success", False) if results["models"] else False,
            results["trading"].get("success", False) if results["trading"] else False
        ])
        
        if results["overall_success"]:
            logger.info("Test data created successfully")
        else:
            logger.warning("Some test data creation failed")
        
        return results
    except Exception as e:
        logger.error(f"Error creating test data: {str(e)}")
        results["error"] = str(e)
        return results


def main():
    """Main function to set up databases and optionally create test data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up databases for the Friday AI Trading System")
    parser.add_argument("--force", action="store_true", help="Force recreation of collections and indexes")
    parser.add_argument("--test-data", action="store_true", help="Create test data")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of sample records to create")
    
    args = parser.parse_args()
    
    # Set up databases
    print("Setting up databases...")
    setup_result = setup_all_databases(force_recreate=args.force)
    
    if setup_result["overall_success"]:
        print("Database setup completed successfully")
        
        # Create test data if requested
        if args.test_data:
            print("Creating test data...")
            test_data_result = create_test_data(sample_size=args.sample_size)
            
            if test_data_result["overall_success"]:
                print("Test data created successfully")
            else:
                print("Some test data creation failed")
                if "error" in test_data_result:
                    print(f"Error: {test_data_result['error']}")
    else:
        print("Database setup failed")
        if "error" in setup_result:
            print(f"Error: {setup_result['error']}")


if __name__ == "__main__":
    main()