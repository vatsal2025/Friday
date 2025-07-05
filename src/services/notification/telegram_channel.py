"""Telegram notification channel for the Friday AI Trading System.

This module provides a class for sending Telegram notifications.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
import aiohttp

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class TelegramChannel:
    """Telegram notification channel.

    This class provides methods for sending Telegram notifications.

    Attributes:
        config: The Telegram configuration.
        enabled: Whether the channel is enabled.
        bot_token: The Telegram bot token.
        chat_ids: The Telegram chat IDs to send messages to.
        api_url: The Telegram API URL.
    """

    def __init__(self):
        """Initialize the Telegram channel."""
        self.config = get_config("NOTIFICATION_CONFIG", {}).get("telegram", {})
        self.enabled = self.config.get("enabled", False)
        self.bot_token = self.config.get("bot_token", "")
        self.chat_ids = self.config.get("chat_ids", [])
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the Telegram configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.enabled:
            return

        if not self.bot_token:
            logger.warning("Telegram bot token not configured, Telegram notifications will be disabled")
            self.enabled = False
            return

        if not self.chat_ids:
            logger.warning("Telegram chat IDs not configured, Telegram notifications will be disabled")
            self.enabled = False
            return

    async def _send_message(self, chat_id: Union[int, str], message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to a Telegram chat.

        Args:
            chat_id: The chat ID to send the message to.
            message: The message to send.
            parse_mode: The parse mode to use. Defaults to "HTML".

        Returns:
            bool: Whether the message was sent successfully.
        """
        if not self.enabled:
            return False

        try:
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Telegram message sent to chat ID {chat_id}")
                        return True
                    else:
                        response_json = await response.json()
                        logger.error(f"Error sending Telegram message: {response_json}")
                        return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    async def _send_to_all_chats(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to all configured chat IDs.

        Args:
            message: The message to send.
            parse_mode: The parse mode to use. Defaults to "HTML".

        Returns:
            bool: Whether the message was sent successfully to all chats.
        """
        if not self.enabled:
            logger.warning("Telegram notifications are disabled")
            return False

        results = []
        for chat_id in self.chat_ids:
            result = await self._send_message(chat_id, message, parse_mode)
            results.append(result)

        return all(results)

    def send(self, message: str) -> bool:
        """Send a Telegram notification.

        Args:
            message: The message to send.

        Returns:
            bool: Whether the message was sent successfully.
        """
        if not self.enabled:
            logger.warning("Telegram notifications are disabled")
            return False

        # Create event loop if not exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async function
        return loop.run_until_complete(self._send_to_all_chats(message))

    def format_trade_notification(self, trade_data: Dict[str, Any]) -> str:
        """Format a trade notification.

        Args:
            trade_data: The trade data.

        Returns:
            str: The formatted notification message.
        """
        action = trade_data.get("action", "")
        symbol = trade_data.get("symbol", "")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        order_type = trade_data.get("order_type", "")
        status = trade_data.get("status", "")

        message = f"<b>🔔 Trade {action.upper()} - {status.upper()}</b>\n\n"
        message += f"<b>Symbol:</b> {symbol}\n"
        message += f"<b>Quantity:</b> {quantity}\n"
        message += f"<b>Price:</b> {price}\n"
        message += f"<b>Order Type:</b> {order_type}\n"
        message += f"<b>Status:</b> {status}\n"

        return message

    def format_alert_notification(self, alert_data: Dict[str, Any]) -> str:
        """Format an alert notification.

        Args:
            alert_data: The alert data.

        Returns:
            str: The formatted notification message.
        """
        alert_type = alert_data.get("type", "")
        symbol = alert_data.get("symbol", "")
        price = alert_data.get("price", 0)
        condition = alert_data.get("condition", "")
        threshold = alert_data.get("threshold", 0)
        message_text = alert_data.get("message", "")

        message = f"<b>⚠️ Alert {alert_type.upper()}</b>\n\n"
        message += f"<b>Symbol:</b> {symbol}\n"
        message += f"<b>Current Price:</b> {price}\n"
        message += f"<b>Condition:</b> {condition} {threshold}\n"
        message += f"<b>Message:</b> {message_text}\n"

        return message

    def format_system_notification(self, system_data: Dict[str, Any]) -> str:
        """Format a system notification.

        Args:
            system_data: The system data.

        Returns:
            str: The formatted notification message.
        """
        event_type = system_data.get("type", "")
        severity = system_data.get("severity", "info").upper()
        message_text = system_data.get("message", "")
        details = system_data.get("details", {})

        # Choose emoji based on severity
        emoji = "ℹ️"
        if severity == "WARNING":
            emoji = "⚠️"
        elif severity == "ERROR":
            emoji = "❌"
        elif severity == "CRITICAL":
            emoji = "🚨"

        message = f"<b>{emoji} System {severity}: {event_type}</b>\n\n"
        message += f"<b>Message:</b> {message_text}\n\n"
        
        if details:
            message += "<b>Details:</b>\n"
            for key, value in details.items():
                message += f"<b>{key}:</b> {value}\n"

        return message