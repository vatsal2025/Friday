"""Configuration management for the Friday AI Trading System.

This module provides functions and classes to load, validate, and access configuration settings.
"""

import os
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


def get_env_var(key: str, default: Optional[Any] = None, required: bool = False) -> Any:
    """Get an environment variable.

    Args:
        key: The name of the environment variable.
        default: The default value to return if the environment variable is not set.
        required: Whether the environment variable is required.

    Returns:
        The value of the environment variable, or the default value if not set.

    Raises:
        ValueError: If the environment variable is required but not set.
    """
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Environment variable {key} is required but not set.")
    return value


# This is a duplicate function, removed
    """Load configuration from environment variables and other sources.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    # Load configuration from unified_config.py
    try:
        from unified_config import (
            ZERODHA_CONFIG,
            MCP_CONFIG,
            KNOWLEDGE_CONFIG,
            DATABASE_CONFIG,
            REDIS_CONFIG,
            MONGODB_CONFIG,
            API_CONFIG,
            BROKER_CONFIG,
        )
        
        return {
            'zerodha': ZERODHA_CONFIG,
            'mcp': MCP_CONFIG,
            'knowledge': KNOWLEDGE_CONFIG,
            'database': DATABASE_CONFIG,
            'redis': REDIS_CONFIG,
            'mongodb': MONGODB_CONFIG,
            'API_CONFIG': API_CONFIG,
            'broker': BROKER_CONFIG,
        }
    except ImportError as e:
        print(f"Warning: Could not import from unified_config: {e}")
        return {}


class ConfigManager:
    """Configuration manager for the Friday AI Trading System.
    
    This class provides methods to load, validate, and access configuration settings
    from various sources including environment variables, JSON files, and Python modules.
    
    It implements a singleton pattern to ensure only one instance exists throughout the application.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Create a new instance of ConfigManager or return the existing one."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the ConfigManager.
        
        Args:
            config_path: Path to a JSON configuration file (optional)
        """
        # Only initialize once
        if getattr(self, '_initialized', False):
            return
        
        self._config = {}
        self._config_path = config_path
        
        # Load configuration
        self._load_config()
        
        self._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from all sources."""
        # Load from unified_config.py
        self._load_from_unified_config()
        
        # Load from JSON file if provided
        if self._config_path:
            self._load_from_json(self._config_path)
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_unified_config(self) -> None:
        """Load configuration from unified_config.py."""
        try:
            from unified_config import (
                ZERODHA_CONFIG,
                MCP_CONFIG,
                KNOWLEDGE_CONFIG,
                DATABASE_CONFIG,
                REDIS_CONFIG,
                MONGODB_CONFIG,
                API_CONFIG,
                BROKER_CONFIG,
            )
            
            self._config.update({
                'zerodha': ZERODHA_CONFIG,
                'mcp': MCP_CONFIG,
                'knowledge': KNOWLEDGE_CONFIG,
                'database': DATABASE_CONFIG,
                'redis': REDIS_CONFIG,
                'mongodb': MONGODB_CONFIG,
                'API_CONFIG': API_CONFIG,
                'broker': BROKER_CONFIG,
            })
        except ImportError as e:
            print(f"Warning: Could not import from unified_config: {e}")
    
    def _load_from_json(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._config.update(config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load configuration from {config_path}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Example: Load API keys from environment variables
        for key in ['ZERODHA_API_KEY', 'ZERODHA_API_SECRET']:
            value = get_env_var(key)
            if value:
                if 'zerodha' not in self._config:
                    self._config['zerodha'] = {}
                self._config['zerodha'][key.lower().replace('zerodha_', '')] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key (can be nested using dot notation, e.g., 'zerodha.api_key')
            default: The default value to return if the key is not found
            
        Returns:
            The configuration value, or the default value if not found
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key (can be nested using dot notation, e.g., 'zerodha.api_key')
            value: The value to set
        """
        keys = key.split('.')
        config = self._config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.
        
        Returns:
            The configuration dictionary
        """
        return self._config.copy()

def load_config() -> Dict[str, Any]:
    """Load configuration from unified_config.py and environment variables.
    
    Returns:
        The configuration dictionary
    """
    # Import configuration from unified_config.py
    try:
        from unified_config import (
            ZERODHA_CONFIG,
            MCP_CONFIG,
            KNOWLEDGE_CONFIG,
            DATABASE_CONFIG,
            REDIS_CONFIG,
            MONGODB_CONFIG,
            API_CONFIG,
            BROKER_CONFIG,
        )
        
        # Create a configuration dictionary
        config = {
            "zerodha": ZERODHA_CONFIG,
            "mcp": MCP_CONFIG,
            "knowledge": KNOWLEDGE_CONFIG,
            "database": DATABASE_CONFIG,
            "redis": REDIS_CONFIG,
            "mongodb": MONGODB_CONFIG,
            "API_CONFIG": API_CONFIG,
            "broker": BROKER_CONFIG,
            # Add other configuration sections as needed
        }
    except ImportError as e:
        print(f"Warning: Could not import from unified_config: {e}")
        config = {}

    # Override with environment variables if set
    # Zerodha API key and secret
    zerodha_api_key = get_env_var("ZERODHA_API_KEY")
    if zerodha_api_key:
        config["zerodha"]["api_key"] = zerodha_api_key

    zerodha_api_secret = get_env_var("ZERODHA_API_SECRET")
    if zerodha_api_secret:
        config["zerodha"]["api_secret"] = zerodha_api_secret

    # Add other environment variable overrides as needed

    return config


# Load configuration on module import
CONFIG = load_config()


def get_config(section: Optional[str] = None, key: Optional[str] = None) -> Any:
    """Get configuration value.

    Args:
        section: The configuration section to get. If None, returns the entire config.
        key: The configuration key to get. If None, returns the entire section.

    Returns:
        The configuration value, section, or entire config.

    Raises:
        KeyError: If the section or key does not exist.
    """
    if section is None:
        return CONFIG

    if section not in CONFIG:
        raise KeyError(f"Configuration section '{section}' does not exist.")

    if key is None:
        return CONFIG[section]

    if key not in CONFIG[section]:
        raise KeyError(
            f"Configuration key '{key}' does not exist in section '{section}'."
        )

    return CONFIG[section][key]


def update_config(section: str, key: str, value: Any) -> None:
    """Update a configuration value.

    Args:
        section: The configuration section to update.
        key: The configuration key to update.
        value: The new value.

    Raises:
        KeyError: If the section does not exist.
    """
    if section not in CONFIG:
        raise KeyError(f"Configuration section '{section}' does not exist.")

    CONFIG[section][key] = value