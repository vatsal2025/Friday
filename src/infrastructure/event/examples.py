"""Examples of using the event system in the Friday AI Trading System.

This module provides practical examples of how to use the event system
for various scenarios in the application.
"""

import time
from typing import Any, Dict, List, Optional

from src.infrastructure.event.event_system import Event, EventSystem
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


def basic_usage_example():
    """Basic example of using the event system."""
    # Initialize the event system
    event_system = EventSystem(max_queue_size=1000, enable_persistence=True)
    
    # Start the event system
    event_system.start()
    
    # Define an event handler function
    def handle_data_event(event: Event):
        logger.info(f"Received data event: {event.event_type}")
        logger.info(f"Event data: {event.data}")
    
    # Register the handler for specific event types
    event_system.register_handler(
        callback=handle_data_event,
        event_types=["data_received", "data_processed"]
    )
    
    # Emit an event
    event_system.emit(
        event_type="data_received",
        data={"source": "market_data_feed", "symbol": "AAPL", "price": 150.25},
        source="market_data_service"
    )
    
    # Allow time for event processing
    time.sleep(1)
    
    # Emit another event
    event_system.emit(
        event_type="data_processed",
        data={"symbol": "AAPL", "indicators": {"sma": 148.75, "rsi": 62}},
        source="data_processor"
    )
    
    # Allow time for event processing
    time.sleep(1)
    
    # Stop the event system
    event_system.stop()


def filtered_handler_example():
    """Example of using a filtered event handler."""
    # Initialize the event system
    event_system = EventSystem()
    
    # Start the event system
    event_system.start()
    
    # Define a filter function
    def price_threshold_filter(event: Event) -> bool:
        # Only process events where the price is above 100
        if event.event_type == "trade_signal" and "price" in event.data:
            return event.data["price"] > 100
        return True
    
    # Define an event handler function
    def handle_trade_signal(event: Event):
        logger.info(f"Processing high-value trade signal: {event.data}")
    
    # Register the handler with a filter
    event_system.register_handler(
        callback=handle_trade_signal,
        event_types=["trade_signal"],
        filter_func=price_threshold_filter
    )
    
    # Emit events with different prices
    event_system.emit(
        event_type="trade_signal",
        data={"symbol": "AAPL", "action": "BUY", "price": 150.25},
        source="trading_strategy"
    )
    
    event_system.emit(
        event_type="trade_signal",
        data={"symbol": "XYZ", "action": "BUY", "price": 75.50},
        source="trading_strategy"
    )
    
    # Allow time for event processing
    time.sleep(1)
    
    # Stop the event system
    event_system.stop()


def event_store_example():
    """Example of using the event store for event persistence and retrieval."""
    # Initialize the event system with persistence enabled
    event_system = EventSystem(enable_persistence=True)
    
    # Start the event system
    event_system.start()
    
    # Emit several events of different types
    for i in range(5):
        event_system.emit(
            event_type="system_status",
            data={"status": "healthy", "memory_usage": 50 + i * 5},
            source="system_monitor"
        )
    
    for i in range(3):
        event_system.emit(
            event_type="trade_executed",
            data={
                "trade_id": f"T{1000 + i}",
                "symbol": "AAPL",
                "quantity": 10,
                "price": 150.25 + i
            },
            source="trading_engine"
        )
    
    # Allow time for event processing and storage
    time.sleep(1)
    
    # Retrieve events from the store
    if event_system.event_store:
        # Get all events (limited to 10)
        all_events = event_system.event_store.get_events(limit=10)
        logger.info(f"Retrieved {len(all_events)} events from the store")
        
        # Get events of a specific type
        trade_events = event_system.event_store.get_events(event_type="trade_executed")
        logger.info(f"Retrieved {len(trade_events)} trade events")
        
        # Display the trade events
        for event in trade_events:
            logger.info(f"Trade event: {event.to_dict()}")
    
    # Stop the event system
    event_system.stop()


def error_handling_example():
    """Example of error handling in event handlers."""
    # Initialize the event system
    event_system = EventSystem()
    
    # Start the event system
    event_system.start()
    
    # Define a handler that might raise an exception
    def problematic_handler(event: Event):
        logger.info(f"Processing event: {event.event_type}")
        if event.data.get("trigger_error", False):
            raise ValueError("This is a simulated error in the event handler")
        logger.info("Event processed successfully")
    
    # Register the handler
    event_system.register_handler(
        callback=problematic_handler,
        event_types=["test_event"]
    )
    
    # Emit a normal event
    event_system.emit(
        event_type="test_event",
        data={"message": "This is a normal event"},
        source="test"
    )
    
    # Emit an event that will trigger an error
    event_system.emit(
        event_type="test_event",
        data={"message": "This will cause an error", "trigger_error": True},
        source="test"
    )
    
    # Emit another normal event to show processing continues
    event_system.emit(
        event_type="test_event",
        data={"message": "This event should still be processed"},
        source="test"
    )
    
    # Allow time for event processing
    time.sleep(1)
    
    # Stop the event system
    event_system.stop()


def multiple_handlers_example():
    """Example of using multiple handlers for the same event type."""
    # Initialize the event system
    event_system = EventSystem()
    
    # Start the event system
    event_system.start()
    
    # Define multiple handlers for the same event type
    def logging_handler(event: Event):
        logger.info(f"Logging event: {event.event_type} - {event.data}")
    
    def analytics_handler(event: Event):
        logger.info(f"Analytics processing for event: {event.event_type}")
        # Simulate analytics processing
        if event.event_type == "user_action":
            logger.info(f"User {event.data.get('user_id')} performed action: {event.data.get('action')}")
    
    def notification_handler(event: Event):
        logger.info(f"Sending notification for event: {event.event_type}")
        # Simulate sending a notification
        if event.data.get('importance', '') == 'high':
            logger.info("Sending high-priority notification!")
    
    # Register all handlers
    event_system.register_handler(
        callback=logging_handler,
        event_types=["user_action"]  # This handler only cares about user actions
    )
    
    event_system.register_handler(
        callback=analytics_handler,
        event_types=["user_action", "system_event"]  # This handler processes multiple event types
    )
    
    event_system.register_handler(
        callback=notification_handler  # This handler processes all events
    )
    
    # Emit an event that will be handled by all handlers
    event_system.emit(
        event_type="user_action",
        data={
            "user_id": "user123",
            "action": "login",
            "timestamp": time.time(),
            "importance": "high"
        },
        source="auth_service"
    )
    
    # Emit an event that will be handled by some handlers
    event_system.emit(
        event_type="system_event",
        data={
            "component": "database",
            "status": "connected",
            "importance": "medium"
        },
        source="system_monitor"
    )
    
    # Allow time for event processing
    time.sleep(1)
    
    # Stop the event system
    event_system.stop()


if __name__ == "__main__":
    # Run the examples
    print("\n=== Basic Usage Example ===")
    basic_usage_example()
    
    print("\n=== Filtered Handler Example ===")
    filtered_handler_example()
    
    print("\n=== Event Store Example ===")
    event_store_example()
    
    print("\n=== Error Handling Example ===")
    error_handling_example()
    
    print("\n=== Multiple Handlers Example ===")
    multiple_handlers_example()