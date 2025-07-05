"""Data Adapter Factory for the Friday AI Trading System.

This module provides a factory for creating data source adapters
for the portfolio system.
"""

from typing import Dict, Optional, Type, Any

from src.data.acquisition.data_fetcher import DataSourceAdapter, DataSourceType
from src.data.acquisition.adapters.alpha_vantage_adapter import AlphaVantageAdapter
from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter
from src.data.acquisition.adapters.polygon_adapter import PolygonAdapter
from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.adapters.yahoo_finance_adapter import YahooFinanceAdapter
from src.data.acquisition.adapters.zerodha_data_adapter import ZerodhaDataAdapter
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem

# Create logger
logger = get_logger(__name__)


class DataAdapterFactory:
    """Factory for creating data source adapters.

    This class simplifies the creation of data source adapters for the portfolio system.
    It provides methods for registering adapter classes and creating adapter instances.
    
    Attributes:
        config: Configuration manager.
        event_system: The event system for publishing events.
    """

    # Dictionary to store registered adapter classes by source type and name
    _adapters: Dict[DataSourceType, Dict[str, Type[DataSourceAdapter]]] = {}

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None
    ):
        """Initialize the data adapter factory.

        Args:
            config: Configuration manager. If None, a new one will be created.
            event_system: The event system for publishing events. If None, a new one will be created.
        """
        self.config = config or ConfigManager()
        self.event_system = event_system or EventSystem()

    @classmethod
    def register_adapter(cls, source_type: DataSourceType, adapter_class: Type[DataSourceAdapter], name: Optional[str] = None) -> None:
        """Register an adapter class for a data source type.

        Args:
            source_type: The data source type.
            adapter_class: The adapter class to register.
            name: Optional name for the adapter. If None, the class name will be used.
        """
        # Initialize the dictionary for this source type if it doesn't exist
        if source_type not in cls._adapters:
            cls._adapters[source_type] = {}
            
        # Use the class name if no name is provided
        adapter_name = name or adapter_class.__name__
        
        # Register the adapter class
        cls._adapters[source_type][adapter_name] = adapter_class
        logger.info(f"Registered adapter class {adapter_class.__name__} as '{adapter_name}' for source type {source_type.value}")

    def create_adapter(self, source_type: DataSourceType, adapter_name: Optional[str] = None, **kwargs) -> DataSourceAdapter:
        """Create a data source adapter.

        Args:
            source_type: The data source type.
            adapter_name: The name of the adapter to create. If None and only one adapter is registered
                         for the source type, that adapter will be used. Otherwise, an error will be raised.
            **kwargs: Additional arguments for the adapter constructor.

        Returns:
            DataSourceAdapter: The created data source adapter.

        Raises:
            ValueError: If the data source type is not registered or if adapter_name is not provided
                       when multiple adapters are registered for the source type.
        """
        if source_type not in self._adapters:
            raise ValueError(f"No adapters registered for source type: {source_type.value}")

        adapters = self._adapters[source_type]
        
        if not adapters:
            raise ValueError(f"No adapters registered for source type: {source_type.value}")
            
        # If adapter_name is provided, use that specific adapter
        if adapter_name:
            if adapter_name not in adapters:
                raise ValueError(f"No adapter named '{adapter_name}' registered for source type: {source_type.value}")
            adapter_class = adapters[adapter_name]
        # If only one adapter is registered, use that
        elif len(adapters) == 1:
            adapter_class = list(adapters.values())[0]
        # If multiple adapters are registered and no name is provided, raise an error
        else:
            available_adapters = ", ".join(adapters.keys())
            raise ValueError(f"Multiple adapters registered for source type {source_type.value}. "
                           f"Please specify one of: {available_adapters}")
        
        # Add event_system to kwargs if not provided
        if 'event_system' not in kwargs:
            kwargs['event_system'] = self.event_system
            
        # Create the adapter instance
        adapter = adapter_class(**kwargs)
        logger.info(f"Created adapter of type {adapter_class.__name__} for source type {source_type.value}")
        return adapter


# Register built-in adapters
DataAdapterFactory.register_adapter(DataSourceType.MARKET_DATA_PROVIDER, AlphaVantageAdapter, "alpha_vantage")
DataAdapterFactory.register_adapter(DataSourceType.MARKET_DATA_PROVIDER, YahooFinanceAdapter, "yahoo_finance")
DataAdapterFactory.register_adapter(DataSourceType.BROKER_API, ZerodhaDataAdapter, "zerodha")
DataAdapterFactory.register_adapter(DataSourceType.REST_API, PolygonAdapter, "polygon")
DataAdapterFactory.register_adapter(DataSourceType.MARKET_DATA_PROVIDER, FinancialDataAdapter, "financial_data")