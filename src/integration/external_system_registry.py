"""External System Registry for the Friday AI Trading System.

This module provides the ExternalSystemRegistry class for managing and tracking
external system integrations, including their configuration, status, and metrics.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
from enum import Enum, auto
import time
import threading
import logging
from datetime import datetime

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.error import FridayError, ErrorSeverity, ErrorCode
from src.integration.external_api_client import ExternalApiClient, ApiProtocol
from src.integration.auth_manager import AuthManager, AuthManagerFactory, AuthType

# Create logger
logger = get_logger(__name__)


class SystemStatus(Enum):
    """Enum for external system status."""
    UNKNOWN = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    RATE_LIMITED = auto()
    MAINTENANCE = auto()


class SystemType(Enum):
    """Enum for external system types."""
    BROKER = auto()
    MARKET_DATA = auto()
    FINANCIAL_DATA = auto()
    ANALYTICS = auto()
    NOTIFICATION = auto()
    STORAGE = auto()
    OTHER = auto()


class ExternalSystemInfo:
    """Class for storing information about an external system.

    Attributes:
        system_id: Unique identifier for the external system.
        name: Human-readable name of the external system.
        system_type: Type of the external system.
        status: Current status of the external system.
        client: API client for the external system.
        auth_manager: Authentication manager for the external system.
        config: Configuration for the external system.
        metrics: Metrics for the external system.
        last_status_change: Timestamp of the last status change.
        error_count: Number of errors encountered.
        last_error: Last error encountered.
    """

    def __init__(
        self,
        system_id: str,
        name: str,
        system_type: SystemType,
        client: Optional[ExternalApiClient] = None,
        auth_manager: Optional[AuthManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize external system information.

        Args:
            system_id: Unique identifier for the external system.
            name: Human-readable name of the external system.
            system_type: Type of the external system.
            client: API client for the external system.
            auth_manager: Authentication manager for the external system.
            config: Configuration for the external system.
        """
        self.system_id = system_id
        self.name = name
        self.system_type = system_type
        self.status = SystemStatus.UNKNOWN
        self.client = client
        self.auth_manager = auth_manager
        self.config = config or {}
        self.metrics: Dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "latency": {
                "total": 0.0,
                "count": 0,
                "average": 0.0,
                "min": float('inf'),
                "max": 0.0
            },
            "rate_limits": {
                "hits": 0,
                "last_hit": None
            },
            "last_request": None,
            "uptime": {
                "connected_at": None,
                "total_uptime": 0.0,
                "total_downtime": 0.0,
                "availability": 100.0
            }
        }
        self.last_status_change = time.time()
        self.error_count = 0
        self.last_error: Optional[Exception] = None

    def update_status(self, status: SystemStatus) -> None:
        """Update the status of the external system.

        Args:
            status: The new status of the external system.
        """
        if self.status != status:
            old_status = self.status
            self.status = status
            now = time.time()
            duration = now - self.last_status_change
            self.last_status_change = now
            
            # Update uptime metrics
            if old_status == SystemStatus.CONNECTED:
                self.metrics["uptime"]["total_uptime"] += duration
            elif old_status in (SystemStatus.DISCONNECTED, SystemStatus.ERROR, SystemStatus.RATE_LIMITED):
                self.metrics["uptime"]["total_downtime"] += duration
                
            if status == SystemStatus.CONNECTED:
                self.metrics["uptime"]["connected_at"] = now
                
            # Calculate availability percentage
            total_time = self.metrics["uptime"]["total_uptime"] + self.metrics["uptime"]["total_downtime"]
            if total_time > 0:
                self.metrics["uptime"]["availability"] = (self.metrics["uptime"]["total_uptime"] / total_time) * 100.0
                
            logger.info(f"External system '{self.system_id}' status changed from {old_status.name} to {status.name}")

    def record_request(self) -> None:
        """Record a request to the external system."""
        self.metrics["requests"] += 1
        self.metrics["last_request"] = time.time()

    def record_error(self, error: Exception) -> None:
        """Record an error from the external system.

        Args:
            error: The error that occurred.
        """
        self.metrics["errors"] += 1
        self.error_count += 1
        self.last_error = error
        
        # Check if this is a rate limit error
        if hasattr(error, "status_code") and getattr(error, "status_code") == 429:
            self.metrics["rate_limits"]["hits"] += 1
            self.metrics["rate_limits"]["last_hit"] = time.time()
            self.update_status(SystemStatus.RATE_LIMITED)
        else:
            # Only update status to ERROR if not already in a more specific error state
            if self.status not in (SystemStatus.RATE_LIMITED, SystemStatus.MAINTENANCE):
                self.update_status(SystemStatus.ERROR)

    def record_latency(self, latency: float) -> None:
        """Record the latency of a request to the external system.

        Args:
            latency: The latency of the request in seconds.
        """
        self.metrics["latency"]["total"] += latency
        self.metrics["latency"]["count"] += 1
        self.metrics["latency"]["min"] = min(self.metrics["latency"]["min"], latency)
        self.metrics["latency"]["max"] = max(self.metrics["latency"]["max"], latency)
        self.metrics["latency"]["average"] = self.metrics["latency"]["total"] / self.metrics["latency"]["count"]

    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report for the external system.

        Returns:
            Dict[str, Any]: A status report containing system information and metrics.
        """
        client_status = self.client.get_status() if self.client else {"connected": False}
        
        return {
            "system_id": self.system_id,
            "name": self.name,
            "type": self.system_type.name,
            "status": self.status.name,
            "client": client_status,
            "metrics": self.metrics,
            "last_status_change": datetime.fromtimestamp(self.last_status_change).isoformat(),
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None
        }


class ExternalSystemRegistry:
    """Registry for managing external system integrations.

    This class provides methods for registering, configuring, and monitoring
    external systems, as well as creating and managing API clients for them.

    Attributes:
        config_manager: Configuration manager for accessing system configs.
        systems: Dictionary of registered external systems.
        auth_managers: Dictionary of authentication managers for external systems.
        _lock: Lock for thread-safe access to the registry.
    """

    def __init__(self, config_manager: ConfigManager):
        """Initialize the external system registry.

        Args:
            config_manager: Configuration manager for accessing system configs.
        """
        self.config_manager = config_manager
        self.systems: Dict[str, ExternalSystemInfo] = {}
        self.auth_managers: Dict[str, AuthManager] = {}
        self._lock = threading.RLock()

    def register_system(
        self,
        system_id: str,
        name: str,
        system_type: SystemType,
        config: Optional[Dict[str, Any]] = None
    ) -> ExternalSystemInfo:
        """Register an external system with the registry.

        Args:
            system_id: Unique identifier for the external system.
            name: Human-readable name of the external system.
            system_type: Type of the external system.
            config: Configuration for the external system.

        Returns:
            ExternalSystemInfo: Information about the registered system.

        Raises:
            ValueError: If a system with the same ID is already registered.
        """
        with self._lock:
            if system_id in self.systems:
                raise ValueError(f"External system '{system_id}' is already registered")
                
            # Get configuration from config manager if not provided
            if config is None:
                external_systems = self.config_manager.get("external_systems", {})
                config = external_systems.get(system_id, {})
                
            # Create system info
            system_info = ExternalSystemInfo(
                system_id=system_id,
                name=name,
                system_type=system_type,
                config=config
            )
            
            self.systems[system_id] = system_info
            logger.info(f"Registered external system '{system_id}' of type {system_type.name}")
            
            return system_info

    def unregister_system(self, system_id: str) -> bool:
        """Unregister an external system from the registry.

        Args:
            system_id: The ID of the external system to unregister.

        Returns:
            bool: True if the system was unregistered, False if it was not found.
        """
        with self._lock:
            if system_id not in self.systems:
                logger.warning(f"External system '{system_id}' is not registered")
                return False
                
            # Disconnect client if connected
            system_info = self.systems[system_id]
            if system_info.client and system_info.client.is_connected():
                system_info.client.disconnect()
                
            # Remove system from registry
            del self.systems[system_id]
            
            # Remove auth manager if exists
            if system_id in self.auth_managers:
                del self.auth_managers[system_id]
                
            logger.info(f"Unregistered external system '{system_id}'")
            return True

    def get_system(self, system_id: str) -> Optional[ExternalSystemInfo]:
        """Get information about a registered external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Optional[ExternalSystemInfo]: Information about the system, or None if not found.
        """
        return self.systems.get(system_id)

    def get_systems_by_type(self, system_type: SystemType) -> List[ExternalSystemInfo]:
        """Get all registered external systems of a specific type.

        Args:
            system_type: The type of external systems to retrieve.

        Returns:
            List[ExternalSystemInfo]: A list of external systems of the specified type.
        """
        return [system for system in self.systems.values() if system.system_type == system_type]

    def get_all_systems(self) -> List[ExternalSystemInfo]:
        """Get all registered external systems.

        Returns:
            List[ExternalSystemInfo]: A list of all registered external systems.
        """
        return list(self.systems.values())

    def create_client(
        self,
        system_id: str,
        client_class: Type[ExternalApiClient],
        auth_manager: Optional[AuthManager] = None
    ) -> ExternalApiClient:
        """Create an API client for an external system.

        Args:
            system_id: The ID of the external system.
            client_class: The class of API client to create.
            auth_manager: Authentication manager for the client.

        Returns:
            ExternalApiClient: The created API client.

        Raises:
            ValueError: If the system is not registered.
        """
        with self._lock:
            system_info = self.get_system(system_id)
            if not system_info:
                raise ValueError(f"External system '{system_id}' is not registered")
                
            # Get auth manager if not provided
            if auth_manager is None:
                if system_id in self.auth_managers:
                    auth_manager = self.auth_managers[system_id]
                else:
                    auth_manager = AuthManagerFactory.create_auth_manager_for_system(
                        system_id, self.config_manager)
                    self.auth_managers[system_id] = auth_manager
                    
            # Create client
            client = client_class(system_id, system_info.config, auth_manager)
            
            # Update system info
            system_info.client = client
            system_info.auth_manager = auth_manager
            
            logger.info(f"Created API client for external system '{system_id}'")
            return client

    def connect_system(self, system_id: str) -> bool:
        """Connect to an external system.

        Args:
            system_id: The ID of the external system to connect to.

        Returns:
            bool: True if the connection was successful, False otherwise.

        Raises:
            ValueError: If the system is not registered or has no client.
        """
        system_info = self.get_system(system_id)
        if not system_info:
            raise ValueError(f"External system '{system_id}' is not registered")
            
        if not system_info.client:
            raise ValueError(f"External system '{system_id}' has no API client")
            
        # Connect to the system
        success = system_info.client.connect()
        
        # Update status
        if success:
            system_info.update_status(SystemStatus.CONNECTED)
        else:
            system_info.update_status(SystemStatus.DISCONNECTED)
            
        return success

    def disconnect_system(self, system_id: str) -> bool:
        """Disconnect from an external system.

        Args:
            system_id: The ID of the external system to disconnect from.

        Returns:
            bool: True if the disconnection was successful, False otherwise.

        Raises:
            ValueError: If the system is not registered or has no client.
        """
        system_info = self.get_system(system_id)
        if not system_info:
            raise ValueError(f"External system '{system_id}' is not registered")
            
        if not system_info.client:
            raise ValueError(f"External system '{system_id}' has no API client")
            
        # Disconnect from the system
        success = system_info.client.disconnect()
        
        # Update status
        if success:
            system_info.update_status(SystemStatus.DISCONNECTED)
            
        return success

    def check_system_status(self, system_id: str) -> SystemStatus:
        """Check the status of an external system.

        Args:
            system_id: The ID of the external system to check.

        Returns:
            SystemStatus: The current status of the system.

        Raises:
            ValueError: If the system is not registered.
        """
        system_info = self.get_system(system_id)
        if not system_info:
            raise ValueError(f"External system '{system_id}' is not registered")
            
        # If there's no client, status is UNKNOWN
        if not system_info.client:
            return SystemStatus.UNKNOWN
            
        # Check if the client is connected
        if system_info.client.is_connected():
            system_info.update_status(SystemStatus.CONNECTED)
        else:
            system_info.update_status(SystemStatus.DISCONNECTED)
            
        return system_info.status

    def get_system_status_report(self, system_id: str) -> Dict[str, Any]:
        """Get a status report for an external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, Any]: A status report for the system.

        Raises:
            ValueError: If the system is not registered.
        """
        system_info = self.get_system(system_id)
        if not system_info:
            raise ValueError(f"External system '{system_id}' is not registered")
            
        return system_info.get_status_report()

    def get_all_status_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get status reports for all registered external systems.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of status reports, keyed by system ID.
        """
        return {system_id: system_info.get_status_report()
                for system_id, system_info in self.systems.items()}

    def load_systems_from_config(self) -> int:
        """Load external systems from the configuration.

        Returns:
            int: The number of systems loaded.
        """
        with self._lock:
            # Get external systems configuration
            external_systems = self.config_manager.get("external_systems", {})
            count = 0
            
            # Register each system
            for system_id, config in external_systems.items():
                if system_id in self.systems:
                    continue  # Skip already registered systems
                    
                system_type_str = config.get("type", "OTHER").upper()
                try:
                    system_type = SystemType[system_type_str]
                except KeyError:
                    logger.warning(f"Unknown system type '{system_type_str}' for system '{system_id}', using OTHER")
                    system_type = SystemType.OTHER
                    
                name = config.get("name", system_id)
                
                self.register_system(system_id, name, system_type, config)
                count += 1
                
            logger.info(f"Loaded {count} external systems from configuration")
            return count

    def initialize_all_systems(self) -> Dict[str, bool]:
        """Initialize all registered external systems.

        This method creates API clients and connects to all registered systems.

        Returns:
            Dict[str, bool]: A dictionary of initialization results, keyed by system ID.
        """
        results = {}
        
        for system_id, system_info in self.systems.items():
            try:
                # Determine client class based on protocol
                from src.integration.external_api_client import (
                    RestApiClient, WebSocketApiClient, GraphQLApiClient
                )
                
                protocol_str = system_info.config.get("connection", {}).get("protocol", "REST").upper()
                try:
                    protocol = ApiProtocol[protocol_str]
                except KeyError:
                    logger.warning(f"Unknown protocol '{protocol_str}' for system '{system_id}', using REST")
                    protocol = ApiProtocol.REST
                    
                if protocol == ApiProtocol.REST:
                    client_class = RestApiClient
                elif protocol == ApiProtocol.WEBSOCKET:
                    client_class = WebSocketApiClient
                elif protocol == ApiProtocol.GRAPHQL:
                    client_class = GraphQLApiClient
                else:
                    logger.warning(f"Unsupported protocol '{protocol.name}' for system '{system_id}', using REST")
                    client_class = RestApiClient
                    
                # Create client
                self.create_client(system_id, client_class)
                
                # Connect to system if auto_connect is enabled
                auto_connect = system_info.config.get("connection", {}).get("auto_connect", False)
                if auto_connect:
                    success = self.connect_system(system_id)
                    results[system_id] = success
                else:
                    results[system_id] = True  # Client created successfully
                    
            except Exception as e:
                logger.error(f"Failed to initialize external system '{system_id}': {str(e)}")
                system_info.record_error(e)
                results[system_id] = False
                
        return results