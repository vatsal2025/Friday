"""Mock services for external system integration testing.

This package provides mock implementations of external systems for development and testing.
"""

from typing import Any, Dict, List, Optional, Union, Type
import os
import logging

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.config import ConfigManager
from src.integration.external_system_registry import SystemType

# Create logger
logger = get_logger(__name__)


def initialize_mock_services(config_manager: Optional[ConfigManager] = None, config_dir: Optional[str] = None) -> List[str]:
    """Initialize mock services from configuration.

    This function initializes mock services from either a ConfigManager or a directory of configuration files.

    Args:
        config_manager: The ConfigManager to load configurations from.
        config_dir: The directory containing configuration files.

    Returns:
        List[str]: The IDs of the created services.

    Raises:
        ValueError: If neither config_manager nor config_dir is provided.
    """
    if config_manager is None and config_dir is None:
        raise ValueError("Either config_manager or config_dir must be provided")
        
    # Import here to avoid circular imports
    from src.integration.mock.mock_config import MockServiceConfig
    
    if config_manager is not None:
        # Load configurations from ConfigManager
        configs = MockServiceConfig.load_from_config_manager(config_manager)
    else:
        # Load configurations from directory
        configs = MockServiceConfig.load_configs_from_directory(config_dir)
        
    # Create services from configurations
    return MockServiceConfig.create_mock_services_from_configs(configs)


def create_mock_broker(service_id: str, name: str, config: Dict[str, Any]) -> str:
    """Create a mock broker service.

    Args:
        service_id: Unique identifier for the mock service.
        name: Human-readable name of the mock service.
        config: Configuration for the mock service.

    Returns:
        str: The ID of the created mock service.
    """
    # Import here to avoid circular imports
    from src.integration.mock.mock_service import create_mock_service
    
    # Ensure service_id and name are in the configuration
    config['service_id'] = service_id
    config['name'] = name
    
    return create_mock_service(service_id, SystemType.BROKER, config)


def create_mock_market_data(service_id: str, name: str, config: Dict[str, Any]) -> str:
    """Create a mock market data service.

    Args:
        service_id: Unique identifier for the mock service.
        name: Human-readable name of the mock service.
        config: Configuration for the mock service.

    Returns:
        str: The ID of the created mock service.
    """
    # Import here to avoid circular imports
    from src.integration.mock.mock_service import create_mock_service
    
    # Ensure service_id and name are in the configuration
    config['service_id'] = service_id
    config['name'] = name
    
    return create_mock_service(service_id, SystemType.MARKET_DATA, config)


def create_mock_financial_data(service_id: str, name: str, config: Dict[str, Any]) -> str:
    """Create a mock financial data service.

    Args:
        service_id: Unique identifier for the mock service.
        name: Human-readable name of the mock service.
        config: Configuration for the mock service.

    Returns:
        str: The ID of the created mock service.
    """
    # Import here to avoid circular imports
    from src.integration.mock.mock_service import create_mock_service
    
    # Ensure service_id and name are in the configuration
    config['service_id'] = service_id
    config['name'] = name
    
    return create_mock_service(service_id, SystemType.FINANCIAL_DATA, config)


def get_mock_service(service_id: str) -> Any:
    """Get a mock service by ID.

    Args:
        service_id: The ID of the service to get.

    Returns:
        Any: The mock service.

    Raises:
        Exception: If the service is not found.
    """
    # Import here to avoid circular imports
    from src.integration.mock.mock_registry import get_service
    
    return get_service(service_id)


def send_mock_request(service_id: str, endpoint: str, params: Dict[str, Any]) -> Any:
    """Send a request to a mock service.

    Args:
        service_id: The ID of the service to send the request to.
        endpoint: The endpoint to call.
        params: Parameters for the request.

    Returns:
        Any: The response from the endpoint handler.

    Raises:
        Exception: If the service is not found or the request fails.
    """
    # Import here to avoid circular imports
    from src.integration.mock.mock_registry import handle_request
    
    return handle_request(service_id, endpoint, params)


# Export classes and functions
from src.integration.mock.mock_service import (
    MockServiceError, MockResponseType, MockService,
    MockBrokerService, MockMarketDataService, MockFinancialDataService,
    MockServiceFactory, create_mock_service
)

from src.integration.mock.mock_registry import (
    MockRegistryError, MockServiceRegistry,
    register_service, unregister_service, get_service, handle_request
)

from src.integration.mock.mock_config import (
    MockConfigError, MockServiceConfig,
    load_config_from_file, load_configs_from_directory,
    create_mock_service_from_config, create_mock_services_from_configs
)

# Make the API router available
from src.integration.mock.mock_api import router as mock_api_router