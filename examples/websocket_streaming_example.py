"""Example script demonstrating WebSocket streaming functionality.

This script shows how to use the WebSocket streaming implementation
to connect to a WebSocket market data source and process real-time updates.
"""

import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from src.data.acquisition.websocket_stream_factory import WebSocketStreamFactory
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger, setup_logging
from src.infrastructure.event import Event, EventSystem

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Create event system
event_system = EventSystem()

# Create config manager
config = ConfigManager()

# Example WebSocket URL and authentication parameters
WEBSOCKET_URL = "wss://example.com/marketdata/ws"
AUTH_PARAMS = {
    "api_key": "your_api_key_here"
}

# Example symbols and timeframes to subscribe to
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
TIMEFRAMES = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]

# Flag to control the main loop
running = True


def handle_market_data_tick(event: Event) -> None:
    """Handle market data tick event.

    Args:
        event: The event object.
    """
    data = event.data
    symbol = data.get("symbol")
    market_data = data.get("data")
    timestamp = data.get("timestamp")
    
    logger.info(f"Tick: {symbol} - Price: {market_data.get('price')} - Time: {timestamp}")


def handle_market_data_bar(event: Event) -> None:
    """Handle market data bar event.

    Args:
        event: The event object.
    """
    data = event.data
    symbol = data.get("symbol")
    timeframe = data.get("timeframe")
    market_data = data.get("data")
    timestamp = data.get("timestamp")
    
    logger.info(f"Bar: {symbol} ({timeframe}) - O: {market_data.get('open')} H: {market_data.get('high')} "
               f"L: {market_data.get('low')} C: {market_data.get('close')} - Time: {timestamp}")


def handle_market_data_connected(event: Event) -> None:
    """Handle market data connected event.

    Args:
        event: The event object.
    """
    logger.info("Market data source connected")


def handle_market_data_disconnected(event: Event) -> None:
    """Handle market data disconnected event.

    Args:
        event: The event object.
    """
    logger.info("Market data source disconnected")


def handle_market_data_error(event: Event) -> None:
    """Handle market data error event.

    Args:
        event: The event object.
    """
    error = event.data.get("error", "Unknown error")
    logger.error(f"Market data error: {error}")


def signal_handler(sig, frame) -> None:
    """Handle signals to gracefully shut down.

    Args:
        sig: The signal number.
        frame: The current stack frame.
    """
    global running
    logger.info("Shutting down...")
    running = False


def main() -> None:
    """Main function to run the example."""
    global running
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register event handlers
    event_system.subscribe("market_data.tick", handle_market_data_tick)
    event_system.subscribe("market_data.bar", handle_market_data_bar)
    event_system.subscribe("market_data.connected", handle_market_data_connected)
    event_system.subscribe("market_data.disconnected", handle_market_data_disconnected)
    event_system.subscribe("market_data.error", handle_market_data_error)
    
    try:
        # Create WebSocket stream factory
        factory = WebSocketStreamFactory(config=config, event_system=event_system)
        
        # Create complete WebSocket streaming stack
        connector = factory.create_complete_stack(
            url=WEBSOCKET_URL,
            auth_params=AUTH_PARAMS
        )
        
        # Start the connector
        if not connector.start():
            logger.error("Failed to start connector")
            return
            
        # Subscribe to symbols and timeframes
        for symbol in SYMBOLS:
            connector.subscribe(symbol, TIMEFRAMES)
            
        # Main loop
        logger.info("WebSocket streaming example running. Press Ctrl+C to exit.")
        while running:
            # Check connection status
            status = connector.get_status()
            logger.debug(f"Connection status: {status}")
            
            # Sleep to avoid CPU hogging
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in WebSocket streaming example: {str(e)}")
    finally:
        # Stop the connector
        if 'connector' in locals():
            connector.stop()
            
        logger.info("WebSocket streaming example stopped")


if __name__ == "__main__":
    main()