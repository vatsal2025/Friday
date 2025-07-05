"""Logging configuration for the Friday AI Trading System.

This module provides functions to set up and configure logging for the application.
"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Dict, Optional

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log directory
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage/logs"))


def setup_logging(
    log_level: Optional[int] = None, log_file: Optional[str] = None
) -> None:
    """Set up logging configuration for the application.

    Args:
        log_level: The log level to use. Defaults to DEFAULT_LOG_LEVEL.
        log_file: The log file to use. If None, a default log file will be created.

    Returns:
        None
    """
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL

    # Create log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Generate default log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"friday_{timestamp}.log")

    # Configure logging
    logging_config = get_logging_config(log_level, log_file)
    logging.config.dictConfig(logging_config)

    # Log that logging has been set up
    logging.info("Logging configured with level %s to %s", logging.getLevelName(log_level), log_file)


def get_logging_config(log_level: int, log_file: str) -> Dict:
    """Get logging configuration dictionary.

    Args:
        log_level: The log level to use.
        log_file: The log file to use.

    Returns:
        Dict: Logging configuration dictionary.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": DEFAULT_LOG_FORMAT,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": True,
            },
            "src": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            # Add specific loggers for different components if needed
            "src.application": {
                "level": log_level,
                "propagate": True,
            },
            "src.orchestration": {
                "level": log_level,
                "propagate": True,
            },
            "src.services": {
                "level": log_level,
                "propagate": True,
            },
            "src.data": {
                "level": log_level,
                "propagate": True,
            },
            "src.infrastructure": {
                "level": log_level,
                "propagate": True,
            },
        },
    }


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)