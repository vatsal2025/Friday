"""Examples for using the notification system.

This module provides examples for using the notification system.
"""

import time
from typing import Dict, Any

from src.infrastructure.event import EventSystem
from src.services.notification.notification_service import NotificationService


def send_direct_notifications_example() -> None:
    """Example of sending notifications directly."""
    # Create a notification service
    notification_service = NotificationService()
    notification_service.start()

    # Send a trade notification
    trade_data = {
        "symbol": "RELIANCE",
        "quantity": 10,
        "price": 2500.50,
        "order_type": "MARKET",
        "action": "BUY",
        "status": "EXECUTED"
    }
    notification_service.send_trade_notification(trade_data)

    # Send an alert notification
    alert_data = {
        "type": "PRICE",
        "symbol": "INFY",
        "price": 1500.75,
        "condition": ">",
        "threshold": 1500.00,
        "message": "Infosys crossed the threshold price"
    }
    notification_service.send_alert_notification(alert_data)

    # Send a system notification
    system_data = {
        "type": "ERROR",
        "severity": "HIGH",
        "message": "Database connection failed",
        "details": {
            "error_code": "DB_CONN_001",
            "timestamp": time.time(),
            "service": "data_service"
        }
    }
    notification_service.send_system_notification(system_data)

    # Stop the notification service
    notification_service.stop()


def event_based_notifications_example() -> None:
    """Example of sending notifications through events."""
    # Create an event system
    event_system = EventSystem()
    event_system.start()

    # Create a notification service with the event system
    notification_service = NotificationService(event_system)
    notification_service.start()

    # Emit a trade event
    trade_data = {
        "symbol": "RELIANCE",
        "quantity": 10,
        "price": 2500.50,
        "order_type": "MARKET",
        "action": "BUY"
    }
    notification_service.emit_trade_event("executed", trade_data)

    # Emit an alert event
    alert_data = {
        "symbol": "INFY",
        "price": 1500.75,
        "condition": ">",
        "threshold": 1500.00,
        "message": "Infosys crossed the threshold price"
    }
    notification_service.emit_alert_event("price", alert_data)

    # Emit a system event
    system_data = {
        "severity": "HIGH",
        "message": "Database connection failed",
        "details": {
            "error_code": "DB_CONN_001",
            "timestamp": time.time(),
            "service": "data_service"
        }
    }
    notification_service.emit_system_event("error", system_data)

    # Wait for events to be processed
    time.sleep(1)

    # Stop the notification service and event system
    notification_service.stop()
    event_system.stop()


def custom_event_handler_example() -> None:
    """Example of using a custom event handler with the notification system."""
    from src.infrastructure.event import EventHandler, Event

    # Create a custom event handler
    class CustomTradeHandler(EventHandler):
        def handles(self, event_type: str) -> bool:
            return event_type.startswith("trade.")

        def handle(self, event: Event) -> None:
            print(f"Custom handler processing {event.event_type} event")
            print(f"Trade data: {event.data}")

            # Forward to notification service for notification
            if hasattr(event, "data") and isinstance(event.data, dict):
                notification_service.send_trade_notification(event.data)

    # Create an event system
    event_system = EventSystem()
    
    # Create a notification service with the event system
    notification_service = NotificationService(event_system)
    
    # Create and register the custom handler
    custom_handler = CustomTradeHandler()
    event_system.register_handler(custom_handler)
    
    # Start the services
    event_system.start()
    notification_service.start()

    # Emit a trade event
    trade_data = {
        "symbol": "RELIANCE",
        "quantity": 10,
        "price": 2500.50,
        "order_type": "MARKET",
        "action": "BUY",
        "status": "EXECUTED"
    }
    event_system.emit("trade.executed", trade_data)

    # Wait for events to be processed
    time.sleep(1)

    # Stop the services
    notification_service.stop()
    event_system.stop()


if __name__ == "__main__":
    print("Running direct notifications example...")
    send_direct_notifications_example()
    
    print("\nRunning event-based notifications example...")
    event_based_notifications_example()
    
    print("\nRunning custom event handler example...")
    custom_event_handler_example()