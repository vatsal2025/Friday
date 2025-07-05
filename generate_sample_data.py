#!/usr/bin/env python
"""
Friday AI Trading System - Sample Data Generator

This script generates sample data for the Friday AI Trading System.
It creates sample market data, model data, trading strategies, and backtest results.
"""

import sys
import os
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import required modules
try:
    import pymongo
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("Error: pymongo module not found. Please install it using: pip install pymongo")
    sys.exit(1)

try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError:
    print("Error: redis module not found. Please install it using: pip install redis")
    sys.exit(1)

# Import configuration
try:
    from unified_config import MONGODB_CONFIG, REDIS_CONFIG
    from src.infrastructure.database.mongodb import get_database, get_collection
    from src.infrastructure.cache.redis_cache import RedisCache
except ImportError:
    print("Error: Required modules not found. Make sure you're running this script from the project root.")
    sys.exit(1)


def generate_ohlcv_data(symbol: str, exchange: str, timeframe: str, 
                       start_date: datetime.datetime, end_date: datetime.datetime) -> List[Dict[str, Any]]:
    """Generate sample OHLCV data for a symbol.

    Args:
        symbol: The trading symbol (e.g., 'AAPL', 'BTC-USD')
        exchange: The exchange name (e.g., 'NYSE', 'NASDAQ', 'Binance')
        timeframe: The timeframe (e.g., '1m', '5m', '1h', '1d')
        start_date: The start date for the data
        end_date: The end date for the data

    Returns:
        A list of OHLCV data points
    """
    # Determine the time delta based on the timeframe
    if timeframe == '1m':
        delta = datetime.timedelta(minutes=1)
    elif timeframe == '5m':
        delta = datetime.timedelta(minutes=5)
    elif timeframe == '15m':
        delta = datetime.timedelta(minutes=15)
    elif timeframe == '30m':
        delta = datetime.timedelta(minutes=30)
    elif timeframe == '1h':
        delta = datetime.timedelta(hours=1)
    elif timeframe == '4h':
        delta = datetime.timedelta(hours=4)
    elif timeframe == '1d':
        delta = datetime.timedelta(days=1)
    else:
        delta = datetime.timedelta(days=1)
    
    # Generate timestamps
    current_date = start_date
    timestamps = []
    while current_date <= end_date:
        timestamps.append(current_date)
        current_date += delta
    
    # Generate random price data
    base_price = random.uniform(50, 500)
    volatility = random.uniform(0.01, 0.05)
    
    data = []
    for i, timestamp in enumerate(timestamps):
        if i == 0:
            open_price = base_price
            close_price = base_price * (1 + random.uniform(-volatility, volatility))
        else:
            open_price = data[i-1]['close']
            close_price = open_price * (1 + random.uniform(-volatility, volatility))
        
        high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))
        volume = random.uniform(1000, 100000)
        
        data.append({
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'created_at': datetime.datetime.utcnow(),
            'updated_at': datetime.datetime.utcnow()
        })
    
    return data


def generate_model_data(num_models: int = 5) -> List[Dict[str, Any]]:
    """Generate sample model data.

    Args:
        num_models: The number of models to generate

    Returns:
        A list of model data
    """
    model_types = ['classification', 'regression', 'time_series', 'reinforcement_learning']
    frameworks = ['tensorflow', 'pytorch', 'scikit-learn', 'xgboost', 'lightgbm']
    
    models = []
    for i in range(num_models):
        model_type = random.choice(model_types)
        framework = random.choice(frameworks)
        version = f"0.{random.randint(1, 9)}.{random.randint(0, 9)}"
        
        # Generate random metrics
        metrics = {
            'accuracy': random.uniform(0.7, 0.99) if model_type == 'classification' else None,
            'precision': random.uniform(0.7, 0.99) if model_type == 'classification' else None,
            'recall': random.uniform(0.7, 0.99) if model_type == 'classification' else None,
            'f1_score': random.uniform(0.7, 0.99) if model_type == 'classification' else None,
            'mse': random.uniform(0.01, 0.2) if model_type == 'regression' or model_type == 'time_series' else None,
            'mae': random.uniform(0.01, 0.2) if model_type == 'regression' or model_type == 'time_series' else None,
            'r2': random.uniform(0.7, 0.99) if model_type == 'regression' or model_type == 'time_series' else None,
            'sharpe_ratio': random.uniform(0.5, 3.0) if model_type == 'reinforcement_learning' else None,
            'sortino_ratio': random.uniform(0.5, 3.0) if model_type == 'reinforcement_learning' else None,
            'max_drawdown': random.uniform(0.05, 0.3) if model_type == 'reinforcement_learning' else None
        }
        
        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        # Generate random parameters
        parameters = {
            'learning_rate': random.uniform(0.001, 0.1),
            'batch_size': random.choice([16, 32, 64, 128, 256]),
            'epochs': random.randint(10, 100),
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop', 'adagrad']),
            'layers': random.randint(1, 5),
            'neurons': random.choice([32, 64, 128, 256, 512]),
            'dropout': random.uniform(0.1, 0.5),
            'activation': random.choice(['relu', 'sigmoid', 'tanh', 'leaky_relu']),
        }
        
        model = {
            'name': f"model_{i+1}",
            'model_type': model_type,
            'framework': framework,
            'version': version,
            'description': f"Sample {model_type} model using {framework}",
            'metrics': metrics,
            'parameters': parameters,
            'created_at': datetime.datetime.utcnow(),
            'updated_at': datetime.datetime.utcnow()
        }
        
        models.append(model)
    
    return models


def generate_trading_strategies(num_strategies: int = 3) -> List[Dict[str, Any]]:
    """Generate sample trading strategies.

    Args:
        num_strategies: The number of strategies to generate

    Returns:
        A list of trading strategy data
    """
    strategy_types = ['trend_following', 'mean_reversion', 'breakout', 'statistical_arbitrage', 'machine_learning']
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    strategies = []
    for i in range(num_strategies):
        strategy_type = random.choice(strategy_types)
        
        # Generate random performance metrics
        performance = {
            'sharpe_ratio': random.uniform(0.5, 3.0),
            'sortino_ratio': random.uniform(0.5, 3.0),
            'max_drawdown': random.uniform(0.05, 0.3),
            'win_rate': random.uniform(0.4, 0.7),
            'profit_factor': random.uniform(1.1, 2.5),
            'average_profit': random.uniform(0.5, 2.0),
            'average_loss': random.uniform(0.5, 1.0),
            'expectancy': random.uniform(0.1, 0.5),
            'trades_per_day': random.uniform(1, 10)
        }
        
        # Generate random parameters
        parameters = {
            'entry_threshold': random.uniform(0.01, 0.05),
            'exit_threshold': random.uniform(0.01, 0.05),
            'stop_loss': random.uniform(0.01, 0.05),
            'take_profit': random.uniform(0.02, 0.1),
            'lookback_period': random.randint(10, 100),
            'timeframe': random.choice(timeframes),
            'risk_per_trade': random.uniform(0.01, 0.05),
            'max_open_positions': random.randint(1, 10),
            'position_sizing': random.choice(['fixed', 'percent_risk', 'kelly']),
        }
        
        strategy = {
            'name': f"strategy_{i+1}",
            'strategy_type': strategy_type,
            'description': f"Sample {strategy_type} strategy",
            'performance': performance,
            'parameters': parameters,
            'created_at': datetime.datetime.utcnow(),
            'updated_at': datetime.datetime.utcnow()
        }
        
        strategies.append(strategy)
    
    return strategies


def generate_backtest_results(strategies: List[Dict[str, Any]], symbols: List[str]) -> List[Dict[str, Any]]:
    """Generate sample backtest results.

    Args:
        strategies: The list of trading strategies
        symbols: The list of symbols

    Returns:
        A list of backtest result data
    """
    results = []
    
    for strategy in strategies:
        # Generate 2-3 backtest results for each strategy
        num_backtests = random.randint(2, 3)
        
        for _ in range(num_backtests):
            # Select random symbols for this backtest
            backtest_symbols = random.sample(symbols, random.randint(1, min(3, len(symbols))))
            
            # Generate random date range
            end_date = datetime.datetime.utcnow() - datetime.timedelta(days=random.randint(1, 30))
            start_date = end_date - datetime.timedelta(days=random.randint(30, 180))
            
            # Generate random performance metrics
            performance = {
                'sharpe_ratio': random.uniform(0.5, 3.0),
                'sortino_ratio': random.uniform(0.5, 3.0),
                'max_drawdown': random.uniform(0.05, 0.3),
                'win_rate': random.uniform(0.4, 0.7),
                'profit_factor': random.uniform(1.1, 2.5),
                'total_return': random.uniform(-0.1, 0.5),
                'annualized_return': random.uniform(-0.1, 0.5),
                'num_trades': random.randint(10, 100),
                'average_trade_duration': random.uniform(1, 10),
            }
            
            # Generate random equity curve
            days = (end_date - start_date).days
            dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
            
            initial_equity = 10000
            returns = np.random.normal(0.001, 0.02, days)
            equity_curve = [initial_equity]
            
            for ret in returns:
                equity_curve.append(equity_curve[-1] * (1 + ret))
            
            equity_curve_data = [
                {'date': date.strftime('%Y-%m-%d'), 'equity': equity}
                for date, equity in zip(dates, equity_curve[1:])
            ]
            
            result = {
                'strategy_id': strategy['name'],
                'strategy_type': strategy['strategy_type'],
                'symbols': backtest_symbols,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_equity,
                'final_capital': equity_curve[-1],
                'performance': performance,
                'equity_curve': equity_curve_data,
                'parameters': strategy['parameters'],
                'created_at': datetime.datetime.utcnow()
            }
            
            results.append(result)
    
    return results


def insert_sample_data(force_recreate: bool = False) -> Dict[str, Any]:
    """Insert sample data into MongoDB collections.

    Args:
        force_recreate: Whether to drop existing collections and recreate them

    Returns:
        A dictionary with the results of the operation
    """
    try:
        # Get database
        db_name = MONGODB_CONFIG.get("database", "friday_trading")
        db = get_database(db_name)
        
        # Drop collections if force_recreate is True
        if force_recreate:
            db.drop_collection("market_data")
            db.drop_collection("model_storage")
            db.drop_collection("trading_strategy")
            db.drop_collection("backtest_results")
        
        # Get collections
        market_data_collection = get_collection("market_data", db_name)
        model_storage_collection = get_collection("model_storage", db_name)
        trading_strategy_collection = get_collection("trading_strategy", db_name)
        backtest_results_collection = get_collection("backtest_results", db_name)
        
        # Generate sample data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD']
        exchanges = ['NYSE', 'NASDAQ', 'Binance', 'Coinbase']
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        # Generate market data
        market_data = []
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=30)  # 30 days of data
        
        for symbol in symbols[:5]:  # Use first 5 symbols for stocks
            for timeframe in timeframes:
                data = generate_ohlcv_data(symbol, 'NASDAQ', timeframe, start_date, end_date)
                market_data.extend(data)
        
        for symbol in symbols[5:]:  # Use last 2 symbols for crypto
            for timeframe in timeframes:
                data = generate_ohlcv_data(symbol, 'Binance', timeframe, start_date, end_date)
                market_data.extend(data)
        
        # Insert market data
        if market_data:
            market_data_collection.insert_many(market_data)
            print(f"Inserted {len(market_data)} market data records")
        
        # Generate and insert model data
        models = generate_model_data(num_models=5)
        if models:
            model_storage_collection.insert_many(models)
            print(f"Inserted {len(models)} model records")
        
        # Generate and insert trading strategies
        strategies = generate_trading_strategies(num_strategies=3)
        if strategies:
            trading_strategy_collection.insert_many(strategies)
            print(f"Inserted {len(strategies)} trading strategy records")
        
        # Generate and insert backtest results
        backtest_results = generate_backtest_results(strategies, symbols)
        if backtest_results:
            backtest_results_collection.insert_many(backtest_results)
            print(f"Inserted {len(backtest_results)} backtest result records")
        
        # Update Redis cache with available symbols and timeframes
        redis_cache = RedisCache()
        
        # Store available symbols
        redis_cache.set('market_data:available_symbols', symbols)
        
        # Store available timeframes
        redis_cache.set('market_data:available_timeframes', timeframes)
        
        # Store last update timestamp for each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"market_data:last_update:{symbol}:{timeframe}"
                redis_cache.set(key, end_date.timestamp())
        
        return {
            "success": True,
            "market_data_count": len(market_data),
            "model_count": len(models),
            "strategy_count": len(strategies),
            "backtest_count": len(backtest_results)
        }
    
    except Exception as e:
        print(f"Error inserting sample data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description="Generate sample data for Friday AI Trading System")
    parser.add_argument("--force", action="store_true", help="Force recreation of collections")
    
    args = parser.parse_args()
    
    print("====================================================")
    print("Friday AI Trading System - Sample Data Generator")
    print("====================================================")
    print()
    
    # Insert sample data
    result = insert_sample_data(force_recreate=args.force)
    
    if result.get("success", False):
        print("\nSample data generated successfully:")
        print(f"- Market data records: {result.get('market_data_count', 0)}")
        print(f"- Model records: {result.get('model_count', 0)}")
        print(f"- Trading strategy records: {result.get('strategy_count', 0)}")
        print(f"- Backtest result records: {result.get('backtest_count', 0)}")
        print("\nYou can now run the Friday AI Trading System.")
        print("Run 'python run_friday.py --all' to start the system.")
    else:
        print(f"\nError generating sample data: {result.get('error', 'Unknown error')}")
        print("Make sure MongoDB and Redis are running.")
        print("Run 'python verify_databases.py' to check database connections.")


if __name__ == "__main__":
    main()