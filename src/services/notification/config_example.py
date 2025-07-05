"""Configuration example for the notification system.

This module provides an example of how to configure the notification system.
"""

from src.infrastructure.config import get_config, set_config


def configure_notification_system() -> None:
    """Configure the notification system.

    This function demonstrates how to configure the notification system
    using the configuration manager.
    """
    # Get the current notification configuration
    current_config = get_config("NOTIFICATION_CONFIG", {})
    print("Current notification configuration:")
    print(current_config)

    # Set up a new notification configuration
    notification_config = {
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

    # Set the notification configuration
    set_config("NOTIFICATION_CONFIG", notification_config)
    print("\nNew notification configuration set.")

    # Get the updated notification configuration
    updated_config = get_config("NOTIFICATION_CONFIG", {})
    print("\nUpdated notification configuration:")
    print(updated_config)


def configure_notification_system_in_unified_config() -> None:
    """Configure the notification system in the unified configuration.

    This function demonstrates how to configure the notification system
    in the unified configuration file.
    """
    print("To configure the notification system in the unified configuration file,")
    print("add the following section to your unified_config.py file:")
    print("""
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
    """)


def configure_notification_system_in_env_file() -> None:
    """Configure the notification system in the environment file.

    This function demonstrates how to configure the notification system
    in the .env file.
    """
    print("To configure the notification system in the .env file,")
    print("add the following variables to your .env file:")
    print("""
# Notification Configuration
NOTIFICATION_ENABLED=true
NOTIFICATION_CHANNELS=email,telegram

# Email Configuration
EMAIL_ENABLED=true
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USERNAME=your-email@gmail.com
EMAIL_SMTP_PASSWORD=your-app-password
EMAIL_USE_TLS=true
EMAIL_SENDER=your-email@gmail.com
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com

# Telegram Configuration
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_IDS=your-chat-id-1,your-chat-id-2
    """)


if __name__ == "__main__":
    print("Running notification system configuration example...\n")
    configure_notification_system()
    
    print("\n" + "-" * 80 + "\n")
    configure_notification_system_in_unified_config()
    
    print("\n" + "-" * 80 + "\n")
    configure_notification_system_in_env_file()