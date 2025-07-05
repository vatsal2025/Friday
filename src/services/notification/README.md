# Friday AI Trading System - Notification Service

The Notification Service provides a flexible and extensible system for sending notifications through various channels. It integrates with the Event System to allow event-driven notifications.

## Features

- Multiple notification channels (Email, Telegram)
- Event-driven notifications
- Direct notification sending
- Configurable through unified configuration
- Integration with the Event System

## Components

### NotificationService

The main entry point for the notification system. It provides methods for sending notifications and emitting events.

```python
from src.services.notification import NotificationService

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

# Stop the notification service
notification_service.stop()
```

### NotificationManager

Manages notifications across multiple channels and handles events from the Event System.

### EmailChannel

Sends notifications through email using SMTP.

### TelegramChannel

Sends notifications through Telegram using the Telegram Bot API.

## Configuration

The notification system can be configured through the unified configuration system. Here's an example configuration:

```python
# Notification Configuration
NOTIFICATION_CONFIG = {
    "enabled": True,
    "channels": ["email", "telegram"],
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "your-email@gmail.com",
        "smtp_password": "your-app-password",  # Use app password for Gmail
        "use_tls": True,
        "sender_email": "your-email@gmail.com",
        "recipient_emails": ["recipient1@example.com", "recipient2@example.com"]
    },
    "telegram": {
        "enabled": True,
        "bot_token": "your-telegram-bot-token",
        "chat_ids": ["your-chat-id-1", "your-chat-id-2"]
    }
}
```

## Event Types

The notification system handles the following event types:

### Trade Events

- `trade.executed`: A trade has been executed
- `trade.filled`: A trade has been filled
- `trade.cancelled`: A trade has been cancelled
- `trade.rejected`: A trade has been rejected

### Alert Events

- `alert.price`: A price alert has been triggered
- `alert.technical`: A technical indicator alert has been triggered
- `alert.news`: A news alert has been triggered

### System Events

- `system.error`: A system error has occurred
- `system.warning`: A system warning has occurred
- `system.info`: A system information event has occurred

## Examples

See the `examples.py` file for examples of how to use the notification system.

## Integration with Other Services

The notification system can be integrated with other services by:

1. Using the NotificationService directly to send notifications
2. Emitting events that the NotificationManager will handle
3. Creating custom event handlers that use the NotificationService

## Adding New Notification Channels

To add a new notification channel:

1. Create a new channel class (e.g., `SlackChannel`)
2. Implement the required methods (e.g., `send`, `format_trade_notification`, etc.)
3. Update the NotificationManager to use the new channel
4. Update the configuration schema to include the new channel

## Security Considerations

- Store sensitive configuration (SMTP passwords, API tokens) securely
- Use environment variables or a secure configuration store
- Use TLS for email communication
- Validate and sanitize notification data before sending