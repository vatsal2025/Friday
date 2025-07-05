"""Unit tests for the WebSocketStreamFactory class."""

import unittest
from unittest.mock import MagicMock, patch

from src.data.acquisition.websocket_stream_factory import WebSocketStreamFactory
from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.websocket_data_stream import WebSocketDataStream
from src.data.acquisition.streaming_market_data_connector import StreamingMarketDataConnector
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.event import EventSystem
from src.infrastructure.config import ConfigManager


class TestWebSocketStreamFactory(unittest.TestCase):
    """Test cases for the WebSocketStreamFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_system = MagicMock(spec=EventSystem)
        self.config = MagicMock(spec=ConfigManager)
        
        # Create the factory
        self.factory = WebSocketStreamFactory(
            config=self.config,
            event_system=self.event_system
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.factory.config, self.config)
        self.assertEqual(self.factory.event_system, self.event_system)

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch('src.data.acquisition.websocket_stream_factory.ConfigManager') as mock_config_manager, \
             patch('src.data.acquisition.websocket_stream_factory.EventSystem') as mock_event_system:
            # Create mock instances
            mock_config = MagicMock()
            mock_event = MagicMock()
            mock_config_manager.return_value = mock_config
            mock_event_system.return_value = mock_event
            
            # Create factory with default parameters
            factory = WebSocketStreamFactory()
            
            # Assertions
            self.assertEqual(factory.config, mock_config)
            self.assertEqual(factory.event_system, mock_event)
            mock_config_manager.assert_called_once()
            mock_event_system.assert_called_once()

    @patch('src.data.acquisition.websocket_stream_factory.WebSocketAdapter')
    def test_create_adapter(self, mock_adapter_class):
        """Test the create_adapter method."""
        # Setup
        mock_adapter = MagicMock(spec=WebSocketAdapter)
        mock_adapter_class.return_value = mock_adapter
        url = "wss://test.example.com/ws"
        auth_params = {"api_key": "test_key"}
        available_symbols = ["AAPL", "MSFT"]
        available_timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Call create_adapter
        adapter = self.factory.create_adapter(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes
        )
        
        # Assertions
        self.assertEqual(adapter, mock_adapter)
        mock_adapter_class.assert_called_once_with(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes,
            event_system=self.event_system,
            config=self.config
        )

    @patch('src.data.acquisition.websocket_stream_factory.WebSocketDataStream')
    def test_create_stream(self, mock_stream_class):
        """Test the create_stream method."""
        # Setup
        mock_adapter = MagicMock(spec=WebSocketAdapter)
        mock_stream = MagicMock(spec=WebSocketDataStream)
        mock_stream_class.return_value = mock_stream
        
        # Call create_stream
        stream = self.factory.create_stream(adapter=mock_adapter)
        
        # Assertions
        self.assertEqual(stream, mock_stream)
        mock_stream_class.assert_called_once_with(
            adapter=mock_adapter,
            event_system=self.event_system,
            config=self.config
        )

    @patch('src.data.acquisition.websocket_stream_factory.StreamingMarketDataConnector')
    def test_create_connector(self, mock_connector_class):
        """Test the create_connector method."""
        # Setup
        mock_stream = MagicMock(spec=WebSocketDataStream)
        mock_connector = MagicMock(spec=StreamingMarketDataConnector)
        mock_connector_class.return_value = mock_connector
        
        # Call create_connector
        connector = self.factory.create_connector(stream=mock_stream)
        
        # Assertions
        self.assertEqual(connector, mock_connector)
        mock_connector_class.assert_called_once_with(
            stream=mock_stream,
            event_system=self.event_system,
            config=self.config
        )

    def test_create_complete_stack(self):
        """Test the create_complete_stack method."""
        # Setup
        mock_adapter = MagicMock(spec=WebSocketAdapter)
        mock_stream = MagicMock(spec=WebSocketDataStream)
        mock_connector = MagicMock(spec=StreamingMarketDataConnector)
        
        # Mock the factory methods
        self.factory.create_adapter = MagicMock(return_value=mock_adapter)
        self.factory.create_stream = MagicMock(return_value=mock_stream)
        self.factory.create_connector = MagicMock(return_value=mock_connector)
        
        url = "wss://test.example.com/ws"
        auth_params = {"api_key": "test_key"}
        available_symbols = ["AAPL", "MSFT"]
        available_timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Call create_complete_stack
        connector = self.factory.create_complete_stack(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes
        )
        
        # Assertions
        self.assertEqual(connector, mock_connector)
        self.factory.create_adapter.assert_called_once_with(
            url=url,
            auth_params=auth_params,
            available_symbols=available_symbols,
            available_timeframes=available_timeframes
        )
        self.factory.create_stream.assert_called_once_with(adapter=mock_adapter)
        self.factory.create_connector.assert_called_once_with(stream=mock_stream)


if __name__ == '__main__':
    unittest.main()