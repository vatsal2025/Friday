"""Email notification channel for the Friday AI Trading System.

This module provides a class for sending email notifications.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any, Union

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class EmailChannel:
    """Email notification channel.

    This class provides methods for sending email notifications.

    Attributes:
        config: The email configuration.
        enabled: Whether the channel is enabled.
        smtp_server: The SMTP server address.
        smtp_port: The SMTP server port.
        smtp_username: The SMTP username.
        smtp_password: The SMTP password.
        use_tls: Whether to use TLS.
        sender_email: The sender email address.
        recipient_emails: The recipient email addresses.
    """

    def __init__(self):
        """Initialize the email channel."""
        self.config = get_config("NOTIFICATION_CONFIG", {}).get("email", {})
        self.enabled = self.config.get("enabled", False)
        self.smtp_server = self.config.get("smtp_server", "")
        self.smtp_port = self.config.get("smtp_port", 587)
        self.smtp_username = self.config.get("smtp_username", "")
        self.smtp_password = self.config.get("smtp_password", "")
        self.use_tls = self.config.get("use_tls", True)
        self.sender_email = self.config.get("sender_email", "")
        self.recipient_emails = self.config.get("recipient_emails", [])

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the email configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.enabled:
            return

        if not self.smtp_server:
            logger.warning("SMTP server not configured, email notifications will be disabled")
            self.enabled = False
            return

        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured, email notifications will be disabled")
            self.enabled = False
            return

        if not self.sender_email:
            logger.warning("Sender email not configured, email notifications will be disabled")
            self.enabled = False
            return

        if not self.recipient_emails:
            logger.warning("Recipient emails not configured, email notifications will be disabled")
            self.enabled = False
            return

    def send(self, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """Send an email notification.

        Args:
            subject: The email subject.
            message: The email message (plain text).
            html_message: The email message (HTML). Defaults to None.

        Returns:
            bool: Whether the email was sent successfully.
        """
        if not self.enabled:
            logger.warning("Email notifications are disabled")
            return False

        try:
            # Create message container
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.recipient_emails)

            # Attach parts
            part1 = MIMEText(message, 'plain')
            msg.attach(part1)

            if html_message:
                part2 = MIMEText(html_message, 'html')
                msg.attach(part2)

            # Connect to server and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            
            if self.use_tls:
                server.starttls()
                server.ehlo()
            
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.sender_email, self.recipient_emails, msg.as_string())
            server.quit()

            logger.info(f"Email sent to {', '.join(self.recipient_emails)}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False

    def format_trade_notification(self, trade_data: Dict[str, Any]) -> Dict[str, str]:
        """Format a trade notification.

        Args:
            trade_data: The trade data.

        Returns:
            Dict[str, str]: The formatted notification with subject and message.
        """
        action = trade_data.get("action", "")
        symbol = trade_data.get("symbol", "")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        order_type = trade_data.get("order_type", "")
        status = trade_data.get("status", "")

        subject = f"Trade {action.upper()}: {symbol} - {status.upper()}"

        message = f"Trade {action.upper()} - {status.upper()}\n\n"
        message += f"Symbol: {symbol}\n"
        message += f"Quantity: {quantity}\n"
        message += f"Price: {price}\n"
        message += f"Order Type: {order_type}\n"
        message += f"Status: {status}\n"

        html_message = f"<h2>Trade {action.upper()} - {status.upper()}</h2>"
        html_message += f"<p><strong>Symbol:</strong> {symbol}</p>"
        html_message += f"<p><strong>Quantity:</strong> {quantity}</p>"
        html_message += f"<p><strong>Price:</strong> {price}</p>"
        html_message += f"<p><strong>Order Type:</strong> {order_type}</p>"
        html_message += f"<p><strong>Status:</strong> {status}</p>"

        return {
            "subject": subject,
            "message": message,
            "html_message": html_message
        }

    def format_alert_notification(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """Format an alert notification.

        Args:
            alert_data: The alert data.

        Returns:
            Dict[str, str]: The formatted notification with subject and message.
        """
        alert_type = alert_data.get("type", "")
        symbol = alert_data.get("symbol", "")
        price = alert_data.get("price", 0)
        condition = alert_data.get("condition", "")
        threshold = alert_data.get("threshold", 0)
        message_text = alert_data.get("message", "")

        subject = f"Alert {alert_type.upper()}: {symbol}"

        message = f"Alert {alert_type.upper()}\n\n"
        message += f"Symbol: {symbol}\n"
        message += f"Current Price: {price}\n"
        message += f"Condition: {condition} {threshold}\n"
        message += f"Message: {message_text}\n"

        html_message = f"<h2>Alert {alert_type.upper()}</h2>"
        html_message += f"<p><strong>Symbol:</strong> {symbol}</p>"
        html_message += f"<p><strong>Current Price:</strong> {price}</p>"
        html_message += f"<p><strong>Condition:</strong> {condition} {threshold}</p>"
        html_message += f"<p><strong>Message:</strong> {message_text}</p>"

        return {
            "subject": subject,
            "message": message,
            "html_message": html_message
        }

    def format_system_notification(self, system_data: Dict[str, Any]) -> Dict[str, str]:
        """Format a system notification.

        Args:
            system_data: The system data.

        Returns:
            Dict[str, str]: The formatted notification with subject and message.
        """
        event_type = system_data.get("type", "")
        severity = system_data.get("severity", "info").upper()
        message_text = system_data.get("message", "")
        details = system_data.get("details", {})

        subject = f"System {severity}: {event_type}"

        message = f"System {severity}: {event_type}\n\n"
        message += f"Message: {message_text}\n\n"
        
        if details:
            message += "Details:\n"
            for key, value in details.items():
                message += f"{key}: {value}\n"

        html_message = f"<h2>System {severity}: {event_type}</h2>"
        html_message += f"<p><strong>Message:</strong> {message_text}</p>"
        
        if details:
            html_message += "<h3>Details:</h3>"
            html_message += "<ul>"
            for key, value in details.items():
                html_message += f"<li><strong>{key}:</strong> {value}</li>"
            html_message += "</ul>"

        return {
            "subject": subject,
            "message": message,
            "html_message": html_message
        }