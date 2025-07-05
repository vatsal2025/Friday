"""WebSocket Stream Factory for the Friday AI Trading System.

This module provides a factory for creating WebSocket-based data streams
and connectors for the portfolio system.
"""

from typing import Dict, List, Optional, Union, Any

from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.websocket_data_stream import WebSocketDataStream
from src.data.acquisition.streaming_market_data_connector import StreamingMarketDataConnector
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem

# Create logger
logger = get_logger(__name__)


class WebSocketStreamFactory:
    """Factory for creating WebSocket-based data streams and connectors.

    This class simplifies the creation of WebSocket adapters, streams, and connectors
    for the portfolio system.
    
    Attributes:
        config: Configuration manager.
        event_system: The event system for publishing events.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None
    ):
        """Initialize the WebSocket stream factory.

        Args:
            config: Configuration manager. If None, a new one will be created.
            event_system: The event system for publishing events. If None, a new one will be created.
        """
        self.config = config or ConfigManager()
        self.event_system = event_system or EventSystem()

    def create_adapter(
        self,
        url: str,
        auth_params: Optional[Dict[str, Any]] = None,
        available_symbols: Optional[List[str]] = None,
        available_timeframes: Optional[List[DataTimeframe]] = None
    ) -> WebSocketAdapter:
        """Create a WebSocket adapter.

        Args:
            url: The WebSocket endpoint URL.
            auth_params: Authentication parameters for the WebSocket connection.
            available_symbols: List of available symbols. If None, will be populated on connect.
            available_timeframes: List of available timeframes. If None, will use defaults.

        Returns:
            WebSocketAdapter: The created WebSocket adapter.
        """
        return WebSocketAdapter(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes,
            event_system=self.event_system,
            config=self.config
        )

    def create_stream(
        self,
        adapter: WebSocketAdapter
    ) -> WebSocketDataStream:
        """Create a WebSocket data stream.

        Args:
            adapter: The WebSocket adapter instance.

        Returns:
            WebSocketDataStream: The created WebSocket data stream.
        """
        return WebSocketDataStream(
            adapter=adapter,
            event_system=self.event_system,
            config=self.config
        )

    def create_connector(
        self,
        stream: WebSocketDataStream
    ) -> StreamingMarketDataConnector:
        """Create a streaming market data connector.

        Args:
            stream: The WebSocket data stream instance.

        Returns:
            StreamingMarketDataConnector: The created streaming market data connector.
        """
        return StreamingMarketDataConnector(
            stream=stream,
            event_system=self.event_system,
            config=self.config
        )

    def create_complete_stack(
        self,
        url: str,
        auth_params: Optional[Dict[str, Any]] = None,
        available_symbols: Optional[List[str]] = None,
        available_timeframes: Optional[List[DataTimeframe]] = None
    ) -> StreamingMarketDataConnector:
        """Create a complete WebSocket streaming stack.

        This method creates an adapter, stream, and connector in one step.

        Args:
            url: The WebSocket endpoint URL.
            auth_params: Authentication parameters for the WebSocket connection.
            available_symbols: List of available symbols. If None, will be populated on connect.
            available_timeframes: List of available timeframes. If None, will use defaults.

        Returns:
            StreamingMarketDataConnector: The created streaming market data connector.
        """
        adapter = self.create_adapter(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes
        )
        
        stream = self.create_stream(adapter=adapter)
        
        connector = self.create_connector(stream=stream)
        
        return connector