"""Mock Service Configuration for the Friday AI Trading System.

This module provides utilities for loading and validating mock service configurations.
"""

from typing import Any, Dict, List, Optional, Union, Type
import json
import os
import logging
from enum import Enum

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.config import ConfigManager
from src.integration.external_system_registry import SystemType

# Create logger
logger = get_logger(__name__)


class MockConfigError(FridayError):
    """Exception raised for mock configuration errors."""
    pass


class MockServiceConfig:
    """Utility class for loading and validating mock service configurations.

    This class provides methods for loading mock service configurations from files or dictionaries,
    validating the configurations, and creating mock services from the configurations.
    """

    @staticmethod
    def load_config_from_file(file_path: str) -> Dict[str, Any]:
        """Load a mock service configuration from a file.

        Args:
            file_path: Path to the configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.

        Raises:
            MockConfigError: If the file cannot be loaded or is not valid JSON.
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                
            # Validate the configuration
            MockServiceConfig.validate_config(config)
            
            return config
        except json.JSONDecodeError as e:
            raise MockConfigError(f"Failed to parse configuration file '{file_path}': {str(e)}")
        except FileNotFoundError:
            raise MockConfigError(f"Configuration file '{file_path}' not found")
        except Exception as e:
            raise MockConfigError(f"Failed to load configuration file '{file_path}': {str(e)}")

    @staticmethod
    def load_configs_from_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
        """Load all mock service configurations from a directory.

        Args:
            directory_path: Path to the directory containing configuration files.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of loaded configurations, keyed by service ID.

        Raises:
            MockConfigError: If the directory cannot be accessed.
        """
        try:
            configs = {}
            
            for filename in os.listdir(directory_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(directory_path, filename)
                    try:
                        config = MockServiceConfig.load_config_from_file(file_path)
                        service_id = config.get('service_id')
                        
                        if not service_id:
                            logger.warning(f"Skipping configuration file '{file_path}': Missing service_id")
                            continue
                            
                        configs[service_id] = config
                    except MockConfigError as e:
                        logger.warning(f"Skipping configuration file '{file_path}': {str(e)}")
                        
            return configs
        except Exception as e:
            raise MockConfigError(f"Failed to load configurations from directory '{directory_path}': {str(e)}")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate a mock service configuration.

        Args:
            config: The configuration to validate.

        Raises:
            MockConfigError: If the configuration is invalid.
        """
        # Check required fields
        required_fields = ['service_id', 'service_type', 'name']
        for field in required_fields:
            if field not in config:
                raise MockConfigError(f"Missing required field '{field}' in configuration")
                
        # Validate service type
        service_type = config['service_type']
        try:
            if isinstance(service_type, str):
                SystemType[service_type.upper()]
            elif isinstance(service_type, int):
                SystemType(service_type)
            else:
                raise ValueError(f"Invalid service type: {service_type}")
        except (KeyError, ValueError):
            valid_types = [t.name for t in SystemType]
            raise MockConfigError(
                f"Invalid service type '{service_type}' in configuration. "
                f"Valid types are: {', '.join(valid_types)}"
            )
            
        # Validate authentication configuration if present
        auth_config = config.get('authentication', {})
        if auth_config:
            if 'type' not in auth_config:
                raise MockConfigError("Missing 'type' field in authentication configuration")
                
            auth_type = auth_config['type']
            if auth_type == 'api_key':
                if 'valid_keys' not in auth_config:
                    raise MockConfigError("Missing 'valid_keys' field in API key authentication configuration")
            elif auth_type == 'basic':
                if 'valid_credentials' not in auth_config:
                    raise MockConfigError("Missing 'valid_credentials' field in basic authentication configuration")
            elif auth_type == 'oauth':
                if 'client_ids' not in auth_config:
                    raise MockConfigError("Missing 'client_ids' field in OAuth authentication configuration")
            elif auth_type == 'token':
                if 'valid_tokens' not in auth_config:
                    raise MockConfigError("Missing 'valid_tokens' field in token authentication configuration")
                    
        # Validate behavior configuration if present
        behavior_config = config.get('behavior', {})
        if behavior_config:
            for field in ['error_rate', 'timeout_rate', 'rate_limit_rate', 'maintenance_rate', 'latency']:
                if field in behavior_config and not isinstance(behavior_config[field], (int, float)):
                    raise MockConfigError(f"Invalid '{field}' in behavior configuration: must be a number")
                    
            if 'endpoints' in behavior_config and not isinstance(behavior_config['endpoints'], dict):
                raise MockConfigError("Invalid 'endpoints' in behavior configuration: must be a dictionary")

    @staticmethod
    def create_mock_service_from_config(config: Dict[str, Any]) -> str:
        """Create a mock service from a configuration.

        Args:
            config: The configuration to create the service from.

        Returns:
            str: The ID of the created mock service.

        Raises:
            MockConfigError: If the service cannot be created.
        """
        try:
            # Validate the configuration
            MockServiceConfig.validate_config(config)
            
            # Get service details
            service_id = config['service_id']
            service_type_str = config['service_type']
            
            # Convert service type string to enum
            if isinstance(service_type_str, str):
                service_type = SystemType[service_type_str.upper()]
            else:
                service_type = SystemType(service_type_str)
                
            # Create the service
            from src.integration.mock.mock_service import create_mock_service
            return create_mock_service(service_id, service_type, config)
        except Exception as e:
            raise MockConfigError(f"Failed to create mock service: {str(e)}")

    @staticmethod
    def create_mock_services_from_configs(configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Create mock services from configurations.

        Args:
            configs: Dictionary of configurations, keyed by service ID.

        Returns:
            List[str]: List of created service IDs.
        """
        service_ids = []
        
        for service_id, config in configs.items():
            try:
                created_id = MockServiceConfig.create_mock_service_from_config(config)
                service_ids.append(created_id)
            except MockConfigError as e:
                logger.warning(f"Failed to create mock service '{service_id}': {str(e)}")
                
        return service_ids

    @staticmethod
    def load_from_config_manager(config_manager: ConfigManager, section: str = 'mock_services') -> Dict[str, Dict[str, Any]]:
        """Load mock service configurations from a ConfigManager.

        Args:
            config_manager: The ConfigManager to load configurations from.
            section: The section in the configuration containing mock service configurations.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of loaded configurations, keyed by service ID.

        Raises:
            MockConfigError: If the configurations cannot be loaded.
        """
        try:
            configs = {}
            
            # Get the mock services section
            mock_services = config_manager.get_section(section)
            if not mock_services:
                return configs
                
            # Process each service configuration
            for service_id, service_config in mock_services.items():
                if isinstance(service_config, dict):
                    # Ensure service_id is in the configuration
                    if 'service_id' not in service_config:
                        service_config['service_id'] = service_id
                        
                    try:
                        # Validate the configuration
                        MockServiceConfig.validate_config(service_config)
                        configs[service_id] = service_config
                    except MockConfigError as e:
                        logger.warning(f"Skipping mock service '{service_id}': {str(e)}")
                        
            return configs
        except Exception as e:
            raise MockConfigError(f"Failed to load mock service configurations from ConfigManager: {str(e)}")


# Convenience functions for working with mock service configurations

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """Load a mock service configuration from a file.

    Args:
        file_path: Path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration.
    """
    return MockServiceConfig.load_config_from_file(file_path)


def load_configs_from_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """Load all mock service configurations from a directory.

    Args:
        directory_path: Path to the directory containing configuration files.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of loaded configurations, keyed by service ID.
    """
    return MockServiceConfig.load_configs_from_directory(directory_path)


def create_mock_service_from_config(config: Dict[str, Any]) -> str:
    """Create a mock service from a configuration.

    Args:
        config: The configuration to create the service from.

    Returns:
        str: The ID of the created mock service.
    """
    return MockServiceConfig.create_mock_service_from_config(config)


def create_mock_services_from_configs(configs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Create mock services from configurations.

    Args:
        configs: Dictionary of configurations, keyed by service ID.

    Returns:
        List[str]: List of created service IDs.
    """
    return MockServiceConfig.create_mock_services_from_configs(configs)