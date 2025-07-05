"""Notification service package for the Friday AI Trading System.

This package provides classes for sending notifications through various channels.
"""

from src.services.notification.email_channel import EmailChannel
from src.services.notification.telegram_channel import TelegramChannel
from src.services.notification.notification_manager import NotificationManager
from src.services.notification.notification_service import NotificationService

__all__ = [
    'EmailChannel',
    'TelegramChannel',
    'NotificationManager',
    'NotificationService'
]