"""Notification manager for the Friday AI Trading System.

This module provides a class for managing notifications across multiple channels.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventHandler, Event
from src.services.notification.email_channel import EmailChannel
from src.services.notification.telegram_channel import TelegramChannel

# Create logger
logger = get_logger(__name__)


class NotificationManager(EventHandler):
    """Notification manager.

    This class manages notifications across multiple channels.

    Attributes:
        config: The notification configuration.
        enabled: Whether notifications are enabled.
        channels: The notification channels.
        email_channel: The email notification channel.
        telegram_channel: The Telegram notification channel.
    """

    def __init__(self):
        """Initialize the notification manager."""
        super().__init__()
        self.config = get_config("NOTIFICATION_CONFIG", {})
        self.enabled = self.config.get("enabled", True)
        self.channels = self.config.get("channels", [])

        # Initialize channels
        self.email_channel = None
        self.telegram_channel = None

        if "email" in self.channels:
            self.email_channel = EmailChannel()

        if "telegram" in self.channels:
            self.telegram_channel = TelegramChannel()

    def handles(self, event_type: str) -> bool:
        """Check if this handler handles the given event type.

        Args:
            event_type: The event type to check.

        Returns:
            bool: Whether this handler handles the given event type.
        """
        notification_event_types = [
            "trade.executed",
            "trade.filled",
            "trade.cancelled",
            "trade.rejected",
            "alert.price",
            "alert.technical",
            "alert.news",
            "system.error",
            "system.warning",
            "system.info"
        ]
        return event_type in notification_event_types

    def should_handle(self, event: Event) -> bool:
        """Check if this handler should handle the given event.

        Args:
            event: The event to check.

        Returns:
            bool: Whether this handler should handle the given event.
        """
        if not self.enabled:
            return False

        # Check if the event type is handled
        if not self.handles(event.event_type):
            return False

        # Check if the event has a notification flag
        if hasattr(event, "data") and isinstance(event.data, dict):
            if event.data.get("notify", True) is False:
                return False

        return True

    def handle(self, event: Event) -> None:
        """Handle the given event.

        Args:
            event: The event to handle.
        """
        if not self.should_handle(event):
            return

        try:
            # Determine the notification type based on the event type
            if event.event_type.startswith("trade."):
                self._handle_trade_notification(event)
            elif event.event_type.startswith("alert."):
                self._handle_alert_notification(event)
            elif event.event_type.startswith("system."):
                self._handle_system_notification(event)
        except Exception as e:
            logger.error(f"Error handling notification for event {event.event_type}: {str(e)}")

    def _handle_trade_notification(self, event: Event) -> None:
        """Handle a trade notification.

        Args:
            event: The trade event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"Trade event {event.event_type} has no data")
            return

        # Prepare trade data
        trade_data = event.data.copy()
        
        # Add action based on event type
        if event.event_type == "trade.executed":
            trade_data["action"] = "executed"
            trade_data["status"] = "executed"
        elif event.event_type == "trade.filled":
            trade_data["action"] = "filled"
            trade_data["status"] = "filled"
        elif event.event_type == "trade.cancelled":
            trade_data["action"] = "cancelled"
            trade_data["status"] = "cancelled"
        elif event.event_type == "trade.rejected":
            trade_data["action"] = "rejected"
            trade_data["status"] = "rejected"

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_trade_notification(trade_data)
            self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_trade_notification(trade_data)
            self.telegram_channel.send(formatted)

    def _handle_alert_notification(self, event: Event) -> None:
        """Handle an alert notification.

        Args:
            event: The alert event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"Alert event {event.event_type} has no data")
            return

        # Prepare alert data
        alert_data = event.data.copy()
        
        # Add type based on event type
        if event.event_type == "alert.price":
            alert_data["type"] = "price"
        elif event.event_type == "alert.technical":
            alert_data["type"] = "technical"
        elif event.event_type == "alert.news":
            alert_data["type"] = "news"

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_alert_notification(alert_data)
            self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_alert_notification(alert_data)
            self.telegram_channel.send(formatted)

    def _handle_system_notification(self, event: Event) -> None:
        """Handle a system notification.

        Args:
            event: The system event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"System event {event.event_type} has no data")
            return

        # Prepare system data
        system_data = event.data.copy()
        
        # Add type and severity based on event type
        if event.event_type == "system.error":
            system_data["type"] = "error"
            system_data["severity"] = "error"
        elif event.event_type == "system.warning":
            system_data["type"] = "warning"
            system_data["severity"] = "warning"
        elif event.event_type == "system.info":
            system_data["type"] = "info"
            system_data["severity"] = "info"

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_system_notification(system_data)
            self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_system_notification(system_data)
            self.telegram_channel.send(formatted)

    def send_notification(self, notification_type: str, data: Dict[str, Any]) -> bool:
        """Send a notification.

        Args:
            notification_type: The notification type (trade, alert, system).
            data: The notification data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        if not self.enabled:
            logger.warning("Notifications are disabled")
            return False

        try:
            if notification_type == "trade":
                return self._send_trade_notification(data)
            elif notification_type == "alert":
                return self._send_alert_notification(data)
            elif notification_type == "system":
                return self._send_system_notification(data)
            else:
                logger.warning(f"Unknown notification type: {notification_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending {notification_type} notification: {str(e)}")
            return False

    def _send_trade_notification(self, data: Dict[str, Any]) -> bool:
        """Send a trade notification.

        Args:
            data: The trade data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        results = []

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_trade_notification(data)
            result = self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )
            results.append(result)

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_trade_notification(data)
            result = self.telegram_channel.send(formatted)
            results.append(result)

        return any(results) if results else False

    def _send_alert_notification(self, data: Dict[str, Any]) -> bool:
        """Send an alert notification.

        Args:
            data: The alert data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        results = []

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_alert_notification(data)
            result = self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )
            results.append(result)

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_alert_notification(data)
            result = self.telegram_channel.send(formatted)
            results.append(result)

        return any(results) if results else False

    def _send_system_notification(self, data: Dict[str, Any]) -> bool:
        """Send a system notification.

        Args:
            data: The system data.

        Returns:
            bool: Whether the notification was sent successfully.
        """
        results = []

        # Send email notification
        if self.email_channel and "email" in self.channels:
            formatted = self.email_channel.format_system_notification(data)
            result = self.email_channel.send(
                subject=formatted["subject"],
                message=formatted["message"],
                html_message=formatted["html_message"]
            )
            results.append(result)

        # Send Telegram notification
        if self.telegram_channel and "telegram" in self.channels:
            formatted = self.telegram_channel.format_system_notification(data)
            result = self.telegram_channel.send(formatted)
            results.append(result)

        return any(results) if results else False