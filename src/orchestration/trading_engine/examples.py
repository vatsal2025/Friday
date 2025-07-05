"""Examples of using the Trading Engine.

This module provides examples of how to use the trading engine components.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import random

from src.infrastructure.event import EventSystem
from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine import (
    TradingEngineIntegrator,
    ModelTradingBridgeIntegration,
    create_trading_engine,
    create_model_trading_bridge,
    TradingEngineConfig,
    get_default_config
)

# Configure logger
logger = get_logger(__name__)


def example_process_prediction():
    """Example of processing a model prediction."""
    # Create event system
    event_system = EventSystem()
    
    # Create trading engine
    trading_engine = create_trading_engine(event_system)
    
    # Create model trading bridge
    bridge = create_model_trading_bridge(trading_engine, event_system)
    
    # Example prediction
    prediction = {
        "symbol": "AAPL",
        "prediction_type": "price_movement",
        "value": 0.75,  # Positive value indicates upward movement
        "horizon": "1d",
        "timestamp": datetime.now().isoformat(),
        "model_id": "example_model",
        "confidence": 0.85
    }
    
    # Process prediction
    signal = bridge.process_prediction(prediction, prediction["confidence"])
    
    if signal:
        logger.info(f"Generated signal: {signal}")
    else:
        logger.info("No signal generated")


def example_execute_order():
    """Example of executing an order."""
    # Create event system
    event_system = EventSystem()
    
    # Create trading engine
    trading_engine = create_trading_engine(event_system)
    
    # Example order parameters
    order_params = {
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "side": "buy",
        "order_type": "limit",
        "time_in_force": "day"
    }
    
    # Execute order with immediate execution strategy
    result = trading_engine.execute_order(order_params, "immediate")
    
    logger.info(f"Execution result: {result}")


def example_continuous_trading():
    """Example of continuous trading."""
    # Create event system
    event_system = EventSystem()
    
    # Create trading engine with custom configuration
    config = get_default_config()
    config.signal_config.min_signal_strength = 0.3  # Lower threshold for testing
    
    trading_engine = create_trading_engine(event_system)
    
    # Create model trading bridge
    bridge = create_model_trading_bridge(trading_engine, event_system)
    
    # Start continuous trading with 10-second interval
    trading_engine.start_continuous_trading(interval_seconds=10.0)
    
    try:
        # Simulate model predictions every few seconds
        for i in range(5):
            # Example prediction with random values
            prediction = {
                "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "AMZN"]),
                "prediction_type": "price_movement",
                "value": random.uniform(-1.0, 1.0),
                "horizon": random.choice(["1h", "4h", "1d"]),
                "timestamp": datetime.now().isoformat(),
                "model_id": "example_model",
                "confidence": random.uniform(0.6, 0.95)
            }
            
            # Emit model prediction event
            event_system.emit(
                event_type="model_prediction",
                data=prediction,
                source="example_script"
            )
            
            logger.info(f"Emitted prediction for {prediction['symbol']} with value {prediction['value']:.2f}")
            
            # Wait a few seconds
            time.sleep(3)
        
        # Wait for processing to complete
        logger.info("Waiting for processing to complete...")
        time.sleep(5)
        
    finally:
        # Stop continuous trading
        trading_engine.stop_continuous_trading()
        logger.info("Stopped continuous trading")


def example_trade_lifecycle():
    """Example of trade lifecycle management."""
    # Create event system
    event_system = EventSystem()
    
    # Create trading engine
    trading_engine = create_trading_engine(event_system)
    
    # Example order parameters
    order_params = {
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "side": "buy",
        "order_type": "limit",
        "time_in_force": "day"
    }
    
    # Execute order
    execution_result = trading_engine.execute_order(order_params, "immediate")
    trade_id = execution_result["execution_id"]
    
    # Get trade
    trade = trading_engine.get_trade(trade_id)
    logger.info(f"Trade: {trade}")
    
    # Generate trade summary
    summary = trading_engine.generate_trade_summary(trade_id)
    logger.info(f"Trade summary: {summary}")


def example_event_handling():
    """Example of event handling."""
    # Create event system
    event_system = EventSystem()
    
    # Create trading engine
    trading_engine = create_trading_engine(event_system)
    
    # Register custom event handler
    def handle_order_filled(event):
        logger.info(f"Order filled: {event.data}")
    
    event_system.register_handler(
        callback=handle_order_filled,
        event_types=["order_filled"]
    )
    
    # Example order parameters
    order_params = {
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "side": "buy",
        "order_type": "limit",
        "time_in_force": "day"
    }
    
    # Execute order
    trading_engine.execute_order(order_params, "immediate")
    
    # Wait for event processing
    logger.info("Waiting for event processing...")
    time.sleep(2)


def run_all_examples():
    """Run all examples."""
    logger.info("Running example_process_prediction()")
    example_process_prediction()
    
    logger.info("\nRunning example_execute_order()")
    example_execute_order()
    
    logger.info("\nRunning example_continuous_trading()")
    example_continuous_trading()
    
    logger.info("\nRunning example_trade_lifecycle()")
    example_trade_lifecycle()
    
    logger.info("\nRunning example_event_handling()")
    example_event_handling()


if __name__ == "__main__":
    run_all_examples()