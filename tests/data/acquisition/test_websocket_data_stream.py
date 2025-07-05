"""Unit tests for the WebSocketDataStream class."""

import unittest
from unittest.mock import MagicMock, patch
import threading
from datetime import datetime, timedelta

from src.data.acquisition.websocket_data_stream import WebSocketDataStream
from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.real_time_data_stream import StreamStatus, StreamEvent
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.config import ConfigManager


class TestWebSocketDataStream(unittest.TestCase):
    """Test cases for the WebSocketDataStream class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_system = MagicMock(spec=EventSystem)
        self.config = MagicMock(spec=ConfigManager)
        
        # Mock the WebSocketAdapter
        self.adapter = MagicMock(spec=WebSocketAdapter)
        self.adapter.connect.return_value = True
        self.adapter.disconnect.return_value = None
        self.adapter.is_connected.return_value = True
        
        # Create the stream
        self.stream = WebSocketDataStream(
            adapter=self.adapter,
            event_system=self.event_system,
            config=self.config
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.stream._adapter, self.adapter)
        self.assertEqual(self.stream._event_system, self.event_system)
        self.assertEqual(self.stream._config, self.config)
        self.assertEqual(self.stream._status, StreamStatus.DISCONNECTED)

    def test_connect(self):
        """Test the connect method."""
        # Call connect
        result = self.stream.connect()
        
        # Assertions
        self.assertTrue(result)
        self.adapter.connect.assert_called_once()
        self.assertEqual(self.stream._status, StreamStatus.CONNECTED)

    def test_connect_failure(self):
        """Test connect method when adapter fails to connect."""
        # Setup
        self.adapter.connect.return_value = False
        
        # Call connect
        result = self.stream.connect()
        
        # Assertions
        self.assertFalse(result)
        self.adapter.connect.assert_called_once()
        self.assertEqual(self.stream._status, StreamStatus.ERROR)

    def test_disconnect(self):
        """Test the disconnect method."""
        # Setup
        self.stream._status = StreamStatus.CONNECTED
        
        # Call disconnect
        self.stream.disconnect()
        
        # Assertions
        self.adapter.disconnect.assert_called_once()
        self.assertEqual(self.stream._status, StreamStatus.DISCONNECTED)

    def test_is_connected(self):
        """Test the is_connected method."""
        # Test when connected
        self.stream._status = StreamStatus.CONNECTED
        self.assertTrue(self.stream.is_connected())
        
        # Test when not connected
        self.stream._status = StreamStatus.DISCONNECTED
        self.assertFalse(self.stream.is_connected())

    @patch('threading.Thread')
    def test_start(self, mock_thread):
        """Test the start method."""
        # Setup
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Call start
        result = self.stream.start()
        
        # Assertions
        self.assertTrue(result)
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        self.assertEqual(self.stream._status, StreamStatus.STREAMING)

    def test_stop(self):
        """Test the stop method."""
        # Setup
        self.stream._status = StreamStatus.STREAMING
        self.stream._stop_event = threading.Event()
        self.stream._worker_thread = MagicMock()
        
        # Call stop
        self.stream.stop()
        
        # Assertions
        self.assertTrue(self.stream._stop_event.is_set())
        self.assertEqual(self.stream._status, StreamStatus.DISCONNECTED)

    def test_subscribe(self):
        """Test the subscribe method."""
        # Setup
        symbol = "AAPL"
        timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Call subscribe
        self.stream.subscribe(symbol, timeframes)
        
        # Assertions
        self.assertEqual(len(self.adapter.subscribe.call_args_list), 2)
        self.adapter.subscribe.assert_any_call(symbol, DataTimeframe.ONE_MINUTE)
        self.adapter.subscribe.assert_any_call(symbol, DataTimeframe.FIVE_MINUTES)
        self.assertIn(symbol, self.stream._subscribed_symbols)
        for timeframe in timeframes:
            self.assertIn(timeframe, self.stream._subscribed_timeframes)
            self.assertIn(symbol, self.stream._data_buffers[timeframe])

    def test_unsubscribe(self):
        """Test the unsubscribe method."""
        # Setup
        symbol = "AAPL"
        timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Add to subscribed lists and buffers
        self.stream._subscribed_symbols.add(symbol)
        for timeframe in timeframes:
            self.stream._subscribed_timeframes.add(timeframe)
            self.stream._data_buffers[timeframe] = {symbol: []}
        
        # Call unsubscribe
        self.stream.unsubscribe(symbol, timeframes)
        
        # Assertions
        self.assertEqual(len(self.adapter.unsubscribe.call_args_list), 2)
        self.adapter.unsubscribe.assert_any_call(symbol, DataTimeframe.ONE_MINUTE)
        self.adapter.unsubscribe.assert_any_call(symbol, DataTimeframe.FIVE_MINUTES)
        self.assertNotIn(symbol, self.stream._subscribed_symbols)
        for timeframe in timeframes:
            self.assertNotIn(symbol, self.stream._data_buffers[timeframe])

    def test_get_latest_data(self):
        """Test the get_latest_data method."""
        # Setup
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        test_data = [
            {"timestamp": datetime.now() - timedelta(minutes=2), "data": {"close": 150}},
            {"timestamp": datetime.now() - timedelta(minutes=1), "data": {"close": 155}},
            {"timestamp": datetime.now(), "data": {"close": 160}}
        ]
        
        # Add data to buffer
        self.stream._data_buffers[timeframe] = {symbol: test_data}
        
        # Call get_latest_data
        result = self.stream.get_latest_data(symbol, timeframe)
        
        # Assertions
        self.assertEqual(result, test_data[-1])

    def test_get_latest_data_empty(self):
        """Test get_latest_data with empty buffer."""
        # Setup
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        
        # Add empty buffer
        self.stream._data_buffers[timeframe] = {symbol: []}
        
        # Call get_latest_data
        result = self.stream.get_latest_data(symbol, timeframe)
        
        # Assertions
        self.assertIsNone(result)

    def test_get_latest_data_no_symbol(self):
        """Test get_latest_data with non-existent symbol."""
        # Setup
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        
        # Add buffer without the symbol
        self.stream._data_buffers[timeframe] = {"MSFT": []}
        
        # Call get_latest_data
        result = self.stream.get_latest_data(symbol, timeframe)
        
        # Assertions
        self.assertIsNone(result)

    def test_process_stream_data(self):
        """Test the _process_stream_data method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            "timestamp": datetime.now().isoformat(),
            "timeframe": DataTimeframe.ONE_MINUTE.value
        }
        
        # Add symbol to subscribed list
        self.stream._subscribed_symbols.add("AAPL")
        self.stream._subscribed_timeframes.add(DataTimeframe.ONE_MINUTE)
        self.stream._data_buffers[DataTimeframe.ONE_MINUTE] = {"AAPL": []}
        
        # Call _process_stream_data
        self.stream._process_stream_data(test_data)
        
        # Assertions
        self.assertEqual(len(self.stream._data_buffers[DataTimeframe.ONE_MINUTE]["AAPL"]), 1)
        self.event_system.emit.assert_called_once()

    def test_handle_websocket_connected(self):
        """Test the _handle_websocket_connected method."""
        # Setup
        event = Event("websocket.connected", {})
        
        # Call _handle_websocket_connected
        self.stream._handle_websocket_connected(event)
        
        # Assertions
        self.assertEqual(self.stream._status, StreamStatus.CONNECTED)
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "stream.event")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data["event"], StreamEvent.CONNECTED)

    def test_handle_websocket_disconnected(self):
        """Test the _handle_websocket_disconnected method."""
        # Setup
        event = Event("websocket.disconnected", {})
        
        # Call _handle_websocket_disconnected
        self.stream._handle_websocket_disconnected(event)
        
        # Assertions
        self.assertEqual(self.stream._status, StreamStatus.DISCONNECTED)
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "stream.event")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data["event"], StreamEvent.DISCONNECTED)

    def test_handle_websocket_error(self):
        """Test the _handle_websocket_error method."""
        # Setup
        error_msg = "Test error"
        event = Event("websocket.error", {"error": error_msg})
        
        # Call _handle_websocket_error
        self.stream._handle_websocket_error(event)
        
        # Assertions
        self.assertEqual(self.stream._status, StreamStatus.ERROR)
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "stream.event")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data["event"], StreamEvent.ERROR)
        self.assertEqual(event_data["error"], error_msg)

    def test_handle_websocket_data(self):
        """Test the _handle_websocket_data method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            "timestamp": datetime.now().isoformat(),
            "timeframe": DataTimeframe.ONE_MINUTE.value
        }
        event = Event("websocket.data", test_data)
        
        # Add symbol to subscribed list
        self.stream._subscribed_symbols.add("AAPL")
        self.stream._subscribed_timeframes.add(DataTimeframe.ONE_MINUTE)
        self.stream._data_buffers[DataTimeframe.ONE_MINUTE] = {"AAPL": []}
        
        # Call _handle_websocket_data
        self.stream._handle_websocket_data(event)
        
        # Assertions
        self.assertEqual(len(self.stream._data_buffers[DataTimeframe.ONE_MINUTE]["AAPL"]), 1)


if __name__ == '__main__':
    unittest.main()