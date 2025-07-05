"""Configuration module for external system integration.

This module provides utilities for loading and validating external system configurations.
"""

from typing import Any, Dict, List, Optional, Union, Type, Set
import os
import json
import yaml
import logging
from pathlib import Path

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.config import ConfigManager
from src.integration.external_system_registry import SystemType

# Create logger
logger = get_logger(__name__)


class IntegrationConfigError(FridayError):
    """Exception raised for errors in the integration configuration."""
    pass


class ExternalSystemConfig:
    """Class for loading and validating external system configurations."""

    # Required fields for all external system configurations
    REQUIRED_FIELDS = {
        'system_id', 'name', 'system_type', 'api_url', 'auth_type'
    }

    # Required fields for specific system types
    SYSTEM_TYPE_REQUIRED_FIELDS = {
        SystemType.BROKER: {'order_endpoints', 'account_endpoints'},
        SystemType.MARKET_DATA: {'data_endpoints', 'subscription_endpoints'},
        SystemType.FINANCIAL_DATA: {'data_endpoints'},
        SystemType.CUSTOM: set()
    }

    # Required fields for specific authentication types
    AUTH_TYPE_REQUIRED_FIELDS = {
        'api_key': {'api_key', 'api_secret'},
        'basic': {'username', 'password'},
        'oauth': {'client_id', 'client_secret', 'token_url'},
        'hmac': {'api_key', 'api_secret'},
        'jwt': {'jwt_secret', 'jwt_algorithm'},
        'custom': set()
    }

    @classmethod
    def load_from_config_manager(cls, config_manager: ConfigManager) -> List[Dict[str, Any]]:
        """Load external system configurations from a ConfigManager.

        Args:
            config_manager: The ConfigManager to load configurations from.

        Returns:
            List[Dict[str, Any]]: A list of external system configurations.

        Raises:
            IntegrationConfigError: If the configuration is invalid.
        """
        try:
            # Get the external systems configuration section
            external_systems = config_manager.get('external_systems', [])
            
            if not external_systems:
                logger.warning("No external systems found in configuration")
                return []
                
            # Validate each system configuration
            validated_configs = []
            for system_config in external_systems:
                validated_config = cls.validate_config(system_config)
                validated_configs.append(validated_config)
                
            return validated_configs
        except Exception as e:
            raise IntegrationConfigError(f"Failed to load external system configurations: {str(e)}") from e

    @classmethod
    def load_configs_from_directory(cls, config_dir: str) -> List[Dict[str, Any]]:
        """Load external system configurations from a directory.

        Args:
            config_dir: The directory containing configuration files.

        Returns:
            List[Dict[str, Any]]: A list of external system configurations.

        Raises:
            IntegrationConfigError: If the configuration is invalid or the directory does not exist.
        """
        try:
            config_path = Path(config_dir)
            if not config_path.exists() or not config_path.is_dir():
                raise IntegrationConfigError(f"Configuration directory does not exist: {config_dir}")
                
            # Get all JSON and YAML files in the directory
            config_files = list(config_path.glob('*.json')) + list(config_path.glob('*.yaml')) + list(config_path.glob('*.yml'))
            
            if not config_files:
                logger.warning(f"No configuration files found in {config_dir}")
                return []
                
            # Load and validate each configuration file
            validated_configs = []
            for config_file in config_files:
                config = cls.load_config_from_file(str(config_file))
                validated_config = cls.validate_config(config)
                validated_configs.append(validated_config)
                
            return validated_configs
        except Exception as e:
            raise IntegrationConfigError(f"Failed to load external system configurations from directory: {str(e)}") from e

    @staticmethod
    def load_config_from_file(file_path: str) -> Dict[str, Any]:
        """Load a configuration from a file.

        Args:
            file_path: The path to the configuration file.

        Returns:
            Dict[str, Any]: The configuration.

        Raises:
            IntegrationConfigError: If the file does not exist or is invalid.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                raise IntegrationConfigError(f"Configuration file does not exist: {file_path}")
                
            # Load the configuration based on the file extension
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() == '.json':
                    config = json.load(f)
                elif file_path.suffix.lower() in ('.yaml', '.yml'):
                    config = yaml.safe_load(f)
                else:
                    raise IntegrationConfigError(f"Unsupported configuration file format: {file_path.suffix}")
                    
            return config
        except Exception as e:
            raise IntegrationConfigError(f"Failed to load configuration from file {file_path}: {str(e)}") from e

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an external system configuration.

        Args:
            config: The configuration to validate.

        Returns:
            Dict[str, Any]: The validated configuration.

        Raises:
            IntegrationConfigError: If the configuration is invalid.
        """
        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(config.keys())
        if missing_fields:
            raise IntegrationConfigError(f"Missing required fields in configuration: {missing_fields}")
            
        # Validate system type
        system_type = config.get('system_type')
        try:
            system_type_enum = SystemType(system_type)
        except ValueError:
            valid_types = [t.value for t in SystemType]
            raise IntegrationConfigError(f"Invalid system type: {system_type}. Valid types: {valid_types}")
            
        # Check system type specific required fields
        system_type_required = cls.SYSTEM_TYPE_REQUIRED_FIELDS.get(system_type_enum, set())
        missing_system_fields = system_type_required - set(config.keys())
        if missing_system_fields:
            raise IntegrationConfigError(
                f"Missing required fields for system type {system_type}: {missing_system_fields}"
            )
            
        # Validate authentication type
        auth_type = config.get('auth_type')
        if auth_type not in cls.AUTH_TYPE_REQUIRED_FIELDS:
            valid_auth_types = list(cls.AUTH_TYPE_REQUIRED_FIELDS.keys())
            raise IntegrationConfigError(f"Invalid auth type: {auth_type}. Valid types: {valid_auth_types}")
            
        # Check auth type specific required fields
        auth_required = cls.AUTH_TYPE_REQUIRED_FIELDS.get(auth_type, set())
        auth_config = config.get('auth_config', {})
        missing_auth_fields = auth_required - set(auth_config.keys())
        if missing_auth_fields:
            raise IntegrationConfigError(
                f"Missing required fields for auth type {auth_type}: {missing_auth_fields}"
            )
            
        return config


def load_external_system_configs(config_manager: Optional[ConfigManager] = None, 
                                config_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load external system configurations.

    Args:
        config_manager: The ConfigManager to load configurations from.
        config_dir: The directory containing configuration files.

    Returns:
        List[Dict[str, Any]]: A list of external system configurations.

    Raises:
        ValueError: If neither config_manager nor config_dir is provided.
        IntegrationConfigError: If the configuration is invalid.
    """
    if config_manager is None and config_dir is None:
        raise ValueError("Either config_manager or config_dir must be provided")
        
    if config_manager is not None:
        return ExternalSystemConfig.load_from_config_manager(config_manager)
    else:
        return ExternalSystemConfig.load_configs_from_directory(config_dir)