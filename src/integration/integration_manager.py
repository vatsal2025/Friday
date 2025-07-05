"""Integration Manager for the Friday AI Trading System.

This module provides the IntegrationManager class for coordinating external system
integrations, data import/export operations, and integration monitoring.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
import threading
import time
from datetime import datetime
import json
import os
import logging

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.error import FridayError, ErrorSeverity, ErrorCode
from src.infrastructure.event import EventSystem, EventType
from src.infrastructure.security import SecureCredentialManager
from src.integration.external_system_registry import (
    ExternalSystemRegistry, ExternalSystemInfo, SystemType, SystemStatus
)
from src.integration.external_api_client import ExternalApiClient, ApiProtocol
from src.integration.auth_manager import AuthManager, AuthType
from src.data.integration.data_orchestrator import DataOrchestrator
from src.data.integration.data_pipeline import DataPipeline

# Create logger
logger = get_logger(__name__)


class IntegrationError(FridayError):
    """Exception raised for integration-related errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
                 error_code: ErrorCode = ErrorCode.INTEGRATION_ERROR):
        super().__init__(message, severity, error_code)


class DataDirection(Enum):
    """Enum for data transfer direction."""
    IMPORT = auto()
    EXPORT = auto()
    BIDIRECTIONAL = auto()


class DataFormat(Enum):
    """Enum for data formats."""
    JSON = auto()
    CSV = auto()
    XML = auto()
    EXCEL = auto()
    PARQUET = auto()
    CUSTOM = auto()


class IntegrationManager:
    """Manager for coordinating external system integrations.

    This class provides methods for managing external system integrations,
    coordinating data import/export operations, and monitoring integration status.

    Attributes:
        config_manager: Configuration manager for accessing system configs.
        event_system: Event system for publishing integration events.
        system_registry: Registry of external systems.
        data_orchestrator: Orchestrator for data pipelines.
        secure_credential_manager: Manager for secure credentials.
        _lock: Lock for thread-safe access to the manager.
        _monitor_thread: Thread for monitoring integration status.
        _monitor_interval: Interval for monitoring integration status.
        _monitor_running: Flag indicating if monitoring is running.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        event_system: EventSystem,
        secure_credential_manager: Optional[SecureCredentialManager] = None,
        data_orchestrator: Optional[DataOrchestrator] = None
    ):
        """Initialize the integration manager.

        Args:
            config_manager: Configuration manager for accessing system configs.
            event_system: Event system for publishing integration events.
            secure_credential_manager: Manager for secure credentials.
            data_orchestrator: Orchestrator for data pipelines.
        """
        self.config_manager = config_manager
        self.event_system = event_system
        self.system_registry = ExternalSystemRegistry(config_manager)
        self.secure_credential_manager = secure_credential_manager
        
        # Create data orchestrator if not provided
        if data_orchestrator is None:
            self.data_orchestrator = DataOrchestrator(
                name="integration_orchestrator",
                config_manager=config_manager,
                event_system=event_system
            )
        else:
            self.data_orchestrator = data_orchestrator
            
        # Initialize monitoring
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._monitor_interval = config_manager.get(
            "integration.monitoring.interval", 60)  # Default: 60 seconds
        self._monitor_running = False
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Integration Manager initialized")

    def _register_event_handlers(self) -> None:
        """Register event handlers for integration events."""
        self.event_system.register_handler(
            EventType.EXTERNAL_SYSTEM_CONNECTED,
            self._handle_system_connected
        )
        self.event_system.register_handler(
            EventType.EXTERNAL_SYSTEM_DISCONNECTED,
            self._handle_system_disconnected
        )
        self.event_system.register_handler(
            EventType.EXTERNAL_SYSTEM_ERROR,
            self._handle_system_error
        )
        self.event_system.register_handler(
            EventType.DATA_IMPORT_REQUESTED,
            self._handle_data_import_request
        )
        self.event_system.register_handler(
            EventType.DATA_EXPORT_REQUESTED,
            self._handle_data_export_request
        )

    def initialize(self) -> bool:
        """Initialize the integration manager and load external systems.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Load external systems from configuration
            num_systems = self.system_registry.load_systems_from_config()
            logger.info(f"Loaded {num_systems} external systems from configuration")
            
            # Initialize external systems
            results = self.system_registry.initialize_all_systems()
            success_count = sum(1 for result in results.values() if result)
            logger.info(f"Initialized {success_count}/{len(results)} external systems")
            
            # Start monitoring if enabled
            if self.config_manager.get("integration.monitoring.enabled", True):
                self.start_monitoring()
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Integration Manager: {str(e)}")
            return False

    def shutdown(self) -> None:
        """Shutdown the integration manager and disconnect from external systems."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Disconnect from all external systems
        for system_id in self.system_registry.get_all_systems():
            try:
                self.system_registry.disconnect_system(system_id.system_id)
            except Exception as e:
                logger.warning(f"Error disconnecting from system '{system_id.system_id}': {str(e)}")
                
        logger.info("Integration Manager shutdown complete")

    def start_monitoring(self) -> bool:
        """Start monitoring external system integrations.

        Returns:
            bool: True if monitoring was started, False if it was already running.
        """
        with self._lock:
            if self._monitor_running:
                logger.warning("Integration monitoring is already running")
                return False
                
            self._monitor_running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_systems,
                name="IntegrationMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info("Started integration monitoring")
            return True

    def stop_monitoring(self) -> bool:
        """Stop monitoring external system integrations.

        Returns:
            bool: True if monitoring was stopped, False if it was not running.
        """
        with self._lock:
            if not self._monitor_running:
                logger.warning("Integration monitoring is not running")
                return False
                
            self._monitor_running = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
                
            logger.info("Stopped integration monitoring")
            return True

    def _monitor_systems(self) -> None:
        """Monitor external system integrations.

        This method runs in a separate thread and periodically checks the status
        of all registered external systems, updating their status and publishing
        events as needed.
        """
        logger.info("Integration monitoring thread started")
        
        while self._monitor_running:
            try:
                # Check status of all systems
                for system_info in self.system_registry.get_all_systems():
                    try:
                        # Skip systems without clients
                        if not system_info.client:
                            continue
                            
                        # Get current status
                        current_status = system_info.status
                        
                        # Check new status
                        new_status = self.system_registry.check_system_status(system_info.system_id)
                        
                        # If status changed, publish event
                        if current_status != new_status:
                            if new_status == SystemStatus.CONNECTED:
                                self.event_system.publish(
                                    EventType.EXTERNAL_SYSTEM_CONNECTED,
                                    {
                                        "system_id": system_info.system_id,
                                        "name": system_info.name,
                                        "type": system_info.system_type.name
                                    }
                                )
                            elif new_status == SystemStatus.DISCONNECTED:
                                self.event_system.publish(
                                    EventType.EXTERNAL_SYSTEM_DISCONNECTED,
                                    {
                                        "system_id": system_info.system_id,
                                        "name": system_info.name,
                                        "type": system_info.system_type.name
                                    }
                                )
                            elif new_status in (SystemStatus.ERROR, SystemStatus.RATE_LIMITED):
                                self.event_system.publish(
                                    EventType.EXTERNAL_SYSTEM_ERROR,
                                    {
                                        "system_id": system_info.system_id,
                                        "name": system_info.name,
                                        "type": system_info.system_type.name,
                                        "status": new_status.name,
                                        "error": str(system_info.last_error) if system_info.last_error else "Unknown error"
                                    }
                                )
                    except Exception as e:
                        logger.error(f"Error monitoring system '{system_info.system_id}': {str(e)}")
                        
                # Sleep for monitoring interval
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in integration monitoring thread: {str(e)}")
                time.sleep(5.0)  # Sleep briefly before retrying
                
        logger.info("Integration monitoring thread stopped")

    def _handle_system_connected(self, event_data: Dict[str, Any]) -> None:
        """Handle external system connected event.

        Args:
            event_data: Event data containing system information.
        """
        system_id = event_data.get("system_id")
        logger.info(f"External system '{system_id}' connected")

    def _handle_system_disconnected(self, event_data: Dict[str, Any]) -> None:
        """Handle external system disconnected event.

        Args:
            event_data: Event data containing system information.
        """
        system_id = event_data.get("system_id")
        logger.info(f"External system '{system_id}' disconnected")

    def _handle_system_error(self, event_data: Dict[str, Any]) -> None:
        """Handle external system error event.

        Args:
            event_data: Event data containing system information and error details.
        """
        system_id = event_data.get("system_id")
        error = event_data.get("error")
        logger.error(f"External system '{system_id}' error: {error}")

    def _handle_data_import_request(self, event_data: Dict[str, Any]) -> None:
        """Handle data import request event.

        Args:
            event_data: Event data containing import request details.
        """
        system_id = event_data.get("system_id")
        data_type = event_data.get("data_type")
        params = event_data.get("params", {})
        
        logger.info(f"Data import requested from system '{system_id}' for data type '{data_type}'")
        
        try:
            self.import_data(system_id, data_type, params)
            
            # Publish success event
            self.event_system.publish(
                EventType.DATA_IMPORT_COMPLETED,
                {
                    "system_id": system_id,
                    "data_type": data_type,
                    "success": True
                }
            )
        except Exception as e:
            logger.error(f"Error importing data from system '{system_id}': {str(e)}")
            
            # Publish failure event
            self.event_system.publish(
                EventType.DATA_IMPORT_COMPLETED,
                {
                    "system_id": system_id,
                    "data_type": data_type,
                    "success": False,
                    "error": str(e)
                }
            )

    def _handle_data_export_request(self, event_data: Dict[str, Any]) -> None:
        """Handle data export request event.

        Args:
            event_data: Event data containing export request details.
        """
        system_id = event_data.get("system_id")
        data_type = event_data.get("data_type")
        data = event_data.get("data")
        params = event_data.get("params", {})
        
        logger.info(f"Data export requested to system '{system_id}' for data type '{data_type}'")
        
        try:
            self.export_data(system_id, data_type, data, params)
            
            # Publish success event
            self.event_system.publish(
                EventType.DATA_EXPORT_COMPLETED,
                {
                    "system_id": system_id,
                    "data_type": data_type,
                    "success": True
                }
            )
        except Exception as e:
            logger.error(f"Error exporting data to system '{system_id}': {str(e)}")
            
            # Publish failure event
            self.event_system.publish(
                EventType.DATA_EXPORT_COMPLETED,
                {
                    "system_id": system_id,
                    "data_type": data_type,
                    "success": False,
                    "error": str(e)
                }
            )

    def connect_system(self, system_id: str) -> bool:
        """Connect to an external system.

        Args:
            system_id: The ID of the external system to connect to.

        Returns:
            bool: True if the connection was successful, False otherwise.

        Raises:
            IntegrationError: If the system is not registered or has no client.
        """
        try:
            return self.system_registry.connect_system(system_id)
        except ValueError as e:
            raise IntegrationError(str(e))

    def disconnect_system(self, system_id: str) -> bool:
        """Disconnect from an external system.

        Args:
            system_id: The ID of the external system to disconnect from.

        Returns:
            bool: True if the disconnection was successful, False otherwise.

        Raises:
            IntegrationError: If the system is not registered or has no client.
        """
        try:
            return self.system_registry.disconnect_system(system_id)
        except ValueError as e:
            raise IntegrationError(str(e))

    def get_system_status(self, system_id: str) -> Dict[str, Any]:
        """Get the status of an external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, Any]: A status report for the system.

        Raises:
            IntegrationError: If the system is not registered.
        """
        try:
            return self.system_registry.get_system_status_report(system_id)
        except ValueError as e:
            raise IntegrationError(str(e))

    def get_all_system_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all external systems.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of status reports, keyed by system ID.
        """
        return self.system_registry.get_all_status_reports()

    def import_data(
        self,
        system_id: str,
        data_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Import data from an external system.

        Args:
            system_id: The ID of the external system to import data from.
            data_type: The type of data to import.
            params: Additional parameters for the import operation.

        Returns:
            Any: The imported data.

        Raises:
            IntegrationError: If the system is not registered, has no client,
                or the import operation fails.
        """
        params = params or {}
        system_info = self.system_registry.get_system(system_id)
        
        if not system_info:
            raise IntegrationError(f"External system '{system_id}' is not registered")
            
        if not system_info.client:
            raise IntegrationError(f"External system '{system_id}' has no API client")
            
        if not system_info.client.is_connected():
            raise IntegrationError(f"External system '{system_id}' is not connected")
            
        # Record request
        system_info.record_request()
        
        # Start timing for latency measurement
        start_time = time.time()
        
        try:
            # Import data from the external system
            data = system_info.client.import_data(data_type, params)
            
            # Record latency
            latency = time.time() - start_time
            system_info.record_latency(latency)
            
            # Create and execute data pipeline if needed
            if params.get("process_data", False):
                pipeline_name = f"{system_id}_{data_type}_import_pipeline"
                pipeline_config = params.get("pipeline_config", {})
                
                # Create pipeline if it doesn't exist
                if not self.data_orchestrator.has_pipeline(pipeline_name):
                    pipeline = self._create_import_pipeline(system_id, data_type, pipeline_config)
                    self.data_orchestrator.add_pipeline(pipeline)
                    
                # Execute pipeline
                result = self.data_orchestrator.execute_pipeline(
                    pipeline_name,
                    input_data={"raw_data": data, "params": params}
                )
                
                return result
            else:
                return data
        except Exception as e:
            # Record error
            system_info.record_error(e)
            
            # Raise integration error
            raise IntegrationError(f"Error importing data from system '{system_id}': {str(e)}")

    def export_data(
        self,
        system_id: str,
        data_type: str,
        data: Any,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Export data to an external system.

        Args:
            system_id: The ID of the external system to export data to.
            data_type: The type of data to export.
            data: The data to export.
            params: Additional parameters for the export operation.

        Returns:
            bool: True if the export was successful, False otherwise.

        Raises:
            IntegrationError: If the system is not registered, has no client,
                or the export operation fails.
        """
        params = params or {}
        system_info = self.system_registry.get_system(system_id)
        
        if not system_info:
            raise IntegrationError(f"External system '{system_id}' is not registered")
            
        if not system_info.client:
            raise IntegrationError(f"External system '{system_id}' has no API client")
            
        if not system_info.client.is_connected():
            raise IntegrationError(f"External system '{system_id}' is not connected")
            
        # Record request
        system_info.record_request()
        
        # Start timing for latency measurement
        start_time = time.time()
        
        try:
            # Process data before export if needed
            if params.get("process_data", False):
                pipeline_name = f"{system_id}_{data_type}_export_pipeline"
                pipeline_config = params.get("pipeline_config", {})
                
                # Create pipeline if it doesn't exist
                if not self.data_orchestrator.has_pipeline(pipeline_name):
                    pipeline = self._create_export_pipeline(system_id, data_type, pipeline_config)
                    self.data_orchestrator.add_pipeline(pipeline)
                    
                # Execute pipeline
                processed_data = self.data_orchestrator.execute_pipeline(
                    pipeline_name,
                    input_data={"raw_data": data, "params": params}
                )
                
                # Export processed data
                result = system_info.client.export_data(data_type, processed_data, params)
            else:
                # Export raw data
                result = system_info.client.export_data(data_type, data, params)
                
            # Record latency
            latency = time.time() - start_time
            system_info.record_latency(latency)
            
            return result
        except Exception as e:
            # Record error
            system_info.record_error(e)
            
            # Raise integration error
            raise IntegrationError(f"Error exporting data to system '{system_id}': {str(e)}")

    def _create_import_pipeline(self, system_id: str, data_type: str,
                               config: Dict[str, Any]) -> DataPipeline:
        """Create a data pipeline for importing data from an external system.

        Args:
            system_id: The ID of the external system.
            data_type: The type of data being imported.
            config: Configuration for the pipeline.

        Returns:
            DataPipeline: The created data pipeline.
        """
        # Create pipeline with appropriate processors based on data type and config
        pipeline_name = f"{system_id}_{data_type}_import_pipeline"
        pipeline = DataPipeline(pipeline_name)
        
        # Add processors based on data type and config
        # This is a simplified example - actual implementation would be more complex
        if data_type == "market_data":
            from src.data.processing.market_data_processor import MarketDataProcessor
            pipeline.add_processor(MarketDataProcessor())
        elif data_type == "portfolio_data":
            from src.data.processing.portfolio_data_processor import PortfolioDataProcessor
            pipeline.add_processor(PortfolioDataProcessor())
        elif data_type == "order_data":
            from src.data.processing.order_data_processor import OrderDataProcessor
            pipeline.add_processor(OrderDataProcessor())
            
        # Add storage if specified
        if config.get("store_data", False):
            storage_type = config.get("storage_type", "default")
            if storage_type == "mongodb":
                from src.data.storage.mongodb_storage import MongoDBStorage
                pipeline.add_storage(MongoDBStorage())
            elif storage_type == "sql":
                from src.data.storage.sql_storage import SQLStorage
                pipeline.add_storage(SQLStorage())
            else:
                from src.data.storage.data_storage import DataStorage
                pipeline.add_storage(DataStorage())
                
        return pipeline

    def _create_export_pipeline(self, system_id: str, data_type: str,
                              config: Dict[str, Any]) -> DataPipeline:
        """Create a data pipeline for exporting data to an external system.

        Args:
            system_id: The ID of the external system.
            data_type: The type of data being exported.
            config: Configuration for the pipeline.

        Returns:
            DataPipeline: The created data pipeline.
        """
        # Create pipeline with appropriate processors based on data type and config
        pipeline_name = f"{system_id}_{data_type}_export_pipeline"
        pipeline = DataPipeline(pipeline_name)
        
        # Add processors based on data type and config
        # This is a simplified example - actual implementation would be more complex
        if data_type == "order_data":
            from src.data.processing.order_formatter import OrderFormatter
            pipeline.add_processor(OrderFormatter())
        elif data_type == "portfolio_data":
            from src.data.processing.portfolio_formatter import PortfolioFormatter
            pipeline.add_processor(PortfolioFormatter())
            
        return pipeline

    def register_custom_system(
        self,
        system_id: str,
        name: str,
        system_type: SystemType,
        config: Dict[str, Any],
        client_class: Type[ExternalApiClient],
        auth_type: AuthType
    ) -> ExternalSystemInfo:
        """Register a custom external system.

        Args:
            system_id: Unique identifier for the external system.
            name: Human-readable name of the external system.
            system_type: Type of the external system.
            config: Configuration for the external system.
            client_class: Class of API client to use.
            auth_type: Type of authentication to use.

        Returns:
            ExternalSystemInfo: Information about the registered system.

        Raises:
            IntegrationError: If the system is already registered or registration fails.
        """
        try:
            # Register system
            system_info = self.system_registry.register_system(
                system_id, name, system_type, config)
                
            # Create auth manager
            auth_manager = AuthManagerFactory.create_auth_manager(
                auth_type, system_id, config.get("authentication", {}))
                
            # Create client
            client = self.system_registry.create_client(
                system_id, client_class, auth_manager)
                
            return system_info
        except ValueError as e:
            raise IntegrationError(str(e))
        except Exception as e:
            raise IntegrationError(f"Error registering custom system '{system_id}': {str(e)}")

    def create_mock_service(
        self,
        system_id: str,
        system_type: SystemType,
        mock_config: Dict[str, Any]
    ) -> str:
        """Create a mock service for development and testing.

        Args:
            system_id: Unique identifier for the mock service.
            system_type: Type of the mock service.
            mock_config: Configuration for the mock service.

        Returns:
            str: The ID of the created mock service.

        Raises:
            IntegrationError: If the mock service creation fails.
        """
        try:
            # Import mock service classes
            from src.integration.mock.mock_service import create_mock_service
            
            # Create mock service
            mock_id = create_mock_service(system_id, system_type, mock_config)
            
            logger.info(f"Created mock service '{mock_id}' of type {system_type.name}")
            return mock_id
        except Exception as e:
            raise IntegrationError(f"Error creating mock service '{system_id}': {str(e)}")

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics for all external system integrations.

        Returns:
            Dict[str, Any]: A dictionary of integration metrics.
        """
        metrics = {
            "systems": {
                "total": len(self.system_registry.get_all_systems()),
                "connected": sum(1 for system in self.system_registry.get_all_systems()
                               if system.status == SystemStatus.CONNECTED),
                "disconnected": sum(1 for system in self.system_registry.get_all_systems()
                                  if system.status == SystemStatus.DISCONNECTED),
                "error": sum(1 for system in self.system_registry.get_all_systems()
                           if system.status in (SystemStatus.ERROR, SystemStatus.RATE_LIMITED))
            },
            "requests": {
                "total": sum(system.metrics["requests"] for system in self.system_registry.get_all_systems()),
                "errors": sum(system.metrics["errors"] for system in self.system_registry.get_all_systems())
            },
            "latency": {
                "average": self._calculate_average_latency()
            },
            "uptime": {
                "average": self._calculate_average_availability()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics

    def _calculate_average_latency(self) -> float:
        """Calculate the average latency across all external systems.

        Returns:
            float: The average latency in seconds.
        """
        systems = self.system_registry.get_all_systems()
        if not systems:
            return 0.0
            
        total_latency = 0.0
        count = 0
        
        for system in systems:
            if system.metrics["latency"]["count"] > 0:
                total_latency += system.metrics["latency"]["average"]
                count += 1
                
        return total_latency / count if count > 0 else 0.0

    def _calculate_average_availability(self) -> float:
        """Calculate the average availability across all external systems.

        Returns:
            float: The average availability as a percentage.
        """
        systems = self.system_registry.get_all_systems()
        if not systems:
            return 100.0
            
        total_availability = 0.0
        count = 0
        
        for system in systems:
            if system.metrics["uptime"]["total_uptime"] + system.metrics["uptime"]["total_downtime"] > 0:
                total_availability += system.metrics["uptime"]["availability"]
                count += 1
                
        return total_availability / count if count > 0 else 100.0