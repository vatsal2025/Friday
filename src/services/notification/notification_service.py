"""Notification service for the Friday AI Trading System.

This module provides a class for managing notifications and integrating with the event system.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem, Event
from src.services.notification.notification_manager import NotificationManager

# Create logger
logger = get_logger(__name__)


class NotificationService:
    """Notification service.

    This class provides methods for sending notifications and integrates with the event system.

    Attributes:
        notification_manager: The notification manager.
        event_system: The event system.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the notification service.

        Args:
            event_system: The event system to use. If None, a new one will be created.
        """
        self.notification_manager = NotificationManager()
        self.event_system = event_system or EventSystem()

        # Register the notification manager with the event system
        self.event_system.register_handler(self.notification_manager)

    def start(self) -> None:
        """Start the notification service."""
        logger.info("Starting notification service")
        if not self.event_system.is_running():
            self.event_system.start()

    def stop(self) -> None:
        """Stop the notification service."""
        logger.info("Stopping notification service")
        # We don't stop the event system here as it might be used by other services

    def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send a trade notification.

        Args:
            trade_data: The trade data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        return self.notification_manager.send_notification("trade", trade_data)

    def send_alert_notification(self, alert_data: Dict[str, Any]) -> bool:
        """Send an alert notification.

        Args:
            alert_data: The alert data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        return self.notification_manager.send_notification("alert", alert_data)

    def send_system_notification(self, system_data: Dict[str, Any]) -> bool:
        """Send a system notification.

        Args:
            system_data: The system data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        return self.notification_manager.send_notification("system", system_data)

    def emit_trade_event(self, event_type: str, trade_data: Dict[str, Any]) -> None:
        """Emit a trade event.

        Args:
            event_type: The event type (trade.executed, trade.filled, trade.cancelled, trade.rejected).
            trade_data: The trade data.
        """
        if not event_type.startswith("trade."):
            event_type = f"trade.{event_type}"

        self.event_system.emit(event_type, trade_data)

    def emit_alert_event(self, event_type: str, alert_data: Dict[str, Any]) -> None:
        """Emit an alert event.

        Args:
            event_type: The event type (alert.price, alert.technical, alert.news).
            alert_data: The alert data.
        """
        if not event_type.startswith("alert."):
            event_type = f"alert.{event_type}"

        self.event_system.emit(event_type, alert_data)

    def emit_system_event(self, event_type: str, system_data: Dict[str, Any]) -> None:
        """Emit a system event.

        Args:
            event_type: The event type (system.error, system.warning, system.info).
            system_data: The system data.
        """
        if not event_type.startswith("system."):
            event_type = f"system.{event_type}"

        self.event_system.emit(event_type, system_data)