"""Unit tests for the WebSocketAdapter class."""

import unittest
from unittest.mock import MagicMock, patch
import json
import queue
from datetime import datetime

from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.data_fetcher import DataTimeframe, DataSourceType, DataConnectionError
from src.infrastructure.event import EventSystem
from src.infrastructure.config import ConfigManager


class TestWebSocketAdapter(unittest.TestCase):
    """Test cases for the WebSocketAdapter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_system = MagicMock(spec=EventSystem)
        self.config = MagicMock(spec=ConfigManager)
        self.url = "wss://test.example.com/ws"
        self.auth_params = {"api_key": "test_key"}
        
        # Create the adapter
        self.adapter = WebSocketAdapter(
            url=self.url,
            auth_params=self.auth_params,
            event_system=self.event_system,
            config=self.config
        )
        
        # Mock the websocket client
        self.mock_ws = MagicMock()
        self.adapter._ws = self.mock_ws

    @patch('websocket.WebSocketApp')
    def test_connect(self, mock_websocket_app):
        """Test the connect method."""
        # Setup mock
        mock_websocket_app.return_value = self.mock_ws
        self.mock_ws.run_forever.return_value = None
        
        # Call connect
        result = self.adapter.connect()
        
        # Assertions
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_connected())
        mock_websocket_app.assert_called_once()
        
    def test_disconnect(self):
        """Test the disconnect method."""
        # Setup
        self.adapter._connected = True
        
        # Call disconnect
        self.adapter.disconnect()
        
        # Assertions
        self.assertFalse(self.adapter.is_connected())
        self.mock_ws.close.assert_called_once()

    def test_is_connected(self):
        """Test the is_connected method."""
        # Test when not connected
        self.adapter._connected = False
        self.assertFalse(self.adapter.is_connected())
        
        # Test when connected
        self.adapter._connected = True
        self.assertTrue(self.adapter.is_connected())

    def test_fetch_data(self):
        """Test the fetch_data method."""
        # Setup
        self.adapter._connected = True
        test_data = {
            "symbol": "AAPL",
            "data": {
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add data to the queue
        self.adapter._data_queue.put(test_data)
        
        # Call fetch_data
        result = self.adapter.fetch_data("AAPL", DataTimeframe.ONE_MINUTE)
        
        # Assertions
        self.assertEqual(result, test_data)

    def test_fetch_data_not_connected(self):
        """Test fetch_data when not connected."""
        # Setup
        self.adapter._connected = False
        
        # Assertions
        with self.assertRaises(DataConnectionError):
            self.adapter.fetch_data("AAPL", DataTimeframe.ONE_MINUTE)

    def test_fetch_data_empty_queue(self):
        """Test fetch_data with an empty queue."""
        # Setup
        self.adapter._connected = True
        
        # Call fetch_data with timeout
        result = self.adapter.fetch_data("AAPL", DataTimeframe.ONE_MINUTE, timeout=0.1)
        
        # Assertions
        self.assertIsNone(result)

    def test_subscribe(self):
        """Test the subscribe method."""
        # Setup
        self.adapter._connected = True
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        
        # Call subscribe
        self.adapter.subscribe(symbol, timeframe)
        
        # Assertions
        self.mock_ws.send.assert_called_once()
        sent_data = json.loads(self.mock_ws.send.call_args[0][0])
        self.assertEqual(sent_data["action"], "subscribe")
        self.assertEqual(sent_data["symbol"], symbol)
        self.assertEqual(sent_data["timeframe"], timeframe.value)

    def test_unsubscribe(self):
        """Test the unsubscribe method."""
        # Setup
        self.adapter._connected = True
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        
        # Call unsubscribe
        self.adapter.unsubscribe(symbol, timeframe)
        
        # Assertions
        self.mock_ws.send.assert_called_once()
        sent_data = json.loads(self.mock_ws.send.call_args[0][0])
        self.assertEqual(sent_data["action"], "unsubscribe")
        self.assertEqual(sent_data["symbol"], symbol)
        self.assertEqual(sent_data["timeframe"], timeframe.value)

    def test_on_message(self):
        """Test the _on_message method."""
        # Setup
        test_message = json.dumps({
            "type": "data",
            "symbol": "AAPL",
            "data": {
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # Call _on_message
        self.adapter._on_message(self.mock_ws, test_message)
        
        # Assertions
        self.assertEqual(self.adapter._data_queue.qsize(), 1)
        
    def test_on_error(self):
        """Test the _on_error method."""
        # Setup
        error = Exception("Test error")
        
        # Call _on_error
        self.adapter._on_error(self.mock_ws, error)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "websocket.error")

    def test_on_close(self):
        """Test the _on_close method."""
        # Setup
        self.adapter._connected = True
        
        # Call _on_close
        self.adapter._on_close(self.mock_ws, 1000, "Normal closure")
        
        # Assertions
        self.assertFalse(self.adapter._connected)
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "websocket.disconnected")

    def test_on_open(self):
        """Test the _on_open method."""
        # Call _on_open
        self.adapter._on_open(self.mock_ws)
        
        # Assertions
        self.assertTrue(self.adapter._connected)
        self.mock_ws.send.assert_called_once()
        sent_data = json.loads(self.mock_ws.send.call_args[0][0])
        self.assertEqual(sent_data["action"], "authenticate")
        self.assertEqual(sent_data["api_key"], self.auth_params["api_key"])
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "websocket.connected")


if __name__ == '__main__':
    unittest.main()