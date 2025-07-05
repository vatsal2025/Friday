"""Mock Service Registry for the Friday AI Trading System.

This module provides a registry for managing mock services for development and testing.
"""

from typing import Any, Dict, List, Optional, Union, Type
import threading
import time
import json
import logging
from enum import Enum

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.integration.external_system_registry import SystemType, SystemStatus

# Create logger
logger = get_logger(__name__)


class MockRegistryError(FridayError):
    """Exception raised for mock registry errors."""
    pass


class MockServiceRegistry:
    """Registry for mock services.

    This class provides a registry for managing mock services for development and testing.

    Attributes:
        _services: Dictionary of registered mock services.
        _lock: Lock for thread-safe access to the registry.
    """

    _services = {}
    _lock = threading.RLock()

    @classmethod
    def register_service(cls, service: Any) -> None:
        """Register a mock service with the registry.

        Args:
            service: The mock service to register.

        Raises:
            MockRegistryError: If a service with the same ID is already registered.
        """
        with cls._lock:
            if service.service_id in cls._services:
                raise MockRegistryError(f"Service with ID '{service.service_id}' is already registered")
                
            cls._services[service.service_id] = service
            logger.info(f"Registered mock service '{service.service_id}' of type {service.service_type.name}")

    @classmethod
    def unregister_service(cls, service_id: str) -> None:
        """Unregister a mock service from the registry.

        Args:
            service_id: The ID of the service to unregister.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            # Stop the service if it's running
            service = cls._services[service_id]
            if service.is_running:
                service.stop()
                
            del cls._services[service_id]
            logger.info(f"Unregistered mock service '{service_id}'")

    @classmethod
    def get_service(cls, service_id: str) -> Any:
        """Get a mock service from the registry.

        Args:
            service_id: The ID of the service to get.

        Returns:
            Any: The mock service.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            return cls._services[service_id]

    @classmethod
    def get_services(cls, service_type: Optional[SystemType] = None) -> List[Any]:
        """Get all mock services of a specific type from the registry.

        Args:
            service_type: The type of services to get. If None, get all services.

        Returns:
            List[Any]: List of mock services.
        """
        with cls._lock:
            if service_type is None:
                return list(cls._services.values())
                
            return [service for service in cls._services.values() if service.service_type == service_type]

    @classmethod
    def start_service(cls, service_id: str) -> bool:
        """Start a mock service.

        Args:
            service_id: The ID of the service to start.

        Returns:
            bool: True if the service was started, False if it was already running.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            return cls._services[service_id].start()

    @classmethod
    def stop_service(cls, service_id: str) -> bool:
        """Stop a mock service.

        Args:
            service_id: The ID of the service to stop.

        Returns:
            bool: True if the service was stopped, False if it was not running.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            return cls._services[service_id].stop()

    @classmethod
    def start_all_services(cls, service_type: Optional[SystemType] = None) -> int:
        """Start all mock services of a specific type.

        Args:
            service_type: The type of services to start. If None, start all services.

        Returns:
            int: The number of services started.
        """
        with cls._lock:
            services = cls.get_services(service_type)
            started = 0
            
            for service in services:
                if service.start():
                    started += 1
                    
            return started

    @classmethod
    def stop_all_services(cls, service_type: Optional[SystemType] = None) -> int:
        """Stop all mock services of a specific type.

        Args:
            service_type: The type of services to stop. If None, stop all services.

        Returns:
            int: The number of services stopped.
        """
        with cls._lock:
            services = cls.get_services(service_type)
            stopped = 0
            
            for service in services:
                if service.stop():
                    stopped += 1
                    
            return stopped

    @classmethod
    def handle_request(cls, service_id: str, endpoint: str, params: Dict[str, Any]) -> Any:
        """Handle a request to a mock service.

        Args:
            service_id: The ID of the service to handle the request.
            endpoint: The endpoint to call.
            params: Parameters for the request.

        Returns:
            Any: The response from the endpoint handler.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            return cls._services[service_id].handle_request(endpoint, params)

    @classmethod
    def get_service_status(cls, service_id: str) -> Dict[str, Any]:
        """Get the status of a mock service.

        Args:
            service_id: The ID of the service to get the status of.

        Returns:
            Dict[str, Any]: The status of the mock service.

        Raises:
            MockRegistryError: If the service is not registered.
        """
        with cls._lock:
            if service_id not in cls._services:
                raise MockRegistryError(f"Service with ID '{service_id}' is not registered")
                
            service = cls._services[service_id]
            
            return {
                "service_id": service.service_id,
                "name": service.name,
                "type": service.service_type.name,
                "status": "running" if service.is_running else "stopped",
                "endpoints": list(service.endpoints.keys())
            }

    @classmethod
    def get_all_service_statuses(cls, service_type: Optional[SystemType] = None) -> List[Dict[str, Any]]:
        """Get the status of all mock services of a specific type.

        Args:
            service_type: The type of services to get the status of. If None, get the status of all services.

        Returns:
            List[Dict[str, Any]]: List of service statuses.
        """
        with cls._lock:
            services = cls.get_services(service_type)
            
            return [
                {
                    "service_id": service.service_id,
                    "name": service.name,
                    "type": service.service_type.name,
                    "status": "running" if service.is_running else "stopped",
                    "endpoints": list(service.endpoints.keys())
                }
                for service in services
            ]

    @classmethod
    def clear_registry(cls) -> int:
        """Clear the registry of all mock services.

        Returns:
            int: The number of services cleared.
        """
        with cls._lock:
            # Stop all services
            for service in cls._services.values():
                if service.is_running:
                    service.stop()
                    
            count = len(cls._services)
            cls._services.clear()
            logger.info(f"Cleared {count} mock services from registry")
            return count


# Convenience functions for working with the registry

def register_service(service: Any) -> None:
    """Register a mock service with the registry.

    Args:
        service: The mock service to register.
    """
    MockServiceRegistry.register_service(service)


def unregister_service(service_id: str) -> None:
    """Unregister a mock service from the registry.

    Args:
        service_id: The ID of the service to unregister.
    """
    MockServiceRegistry.unregister_service(service_id)


def get_service(service_id: str) -> Any:
    """Get a mock service from the registry.

    Args:
        service_id: The ID of the service to get.

    Returns:
        Any: The mock service.
    """
    return MockServiceRegistry.get_service(service_id)


def handle_request(service_id: str, endpoint: str, params: Dict[str, Any]) -> Any:
    """Handle a request to a mock service.

    Args:
        service_id: The ID of the service to handle the request.
        endpoint: The endpoint to call.
        params: Parameters for the request.

    Returns:
        Any: The response from the endpoint handler.
    """
    return MockServiceRegistry.handle_request(service_id, endpoint, params)