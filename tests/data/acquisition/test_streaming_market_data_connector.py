"""Unit tests for the StreamingMarketDataConnector class."""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.data.acquisition.streaming_market_data_connector import StreamingMarketDataConnector
from src.data.acquisition.websocket_data_stream import WebSocketDataStream
from src.data.acquisition.real_time_data_stream import StreamStatus, StreamEvent
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.config import ConfigManager


class TestStreamingMarketDataConnector(unittest.TestCase):
    """Test cases for the StreamingMarketDataConnector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_system = MagicMock(spec=EventSystem)
        self.config = MagicMock(spec=ConfigManager)
        
        # Mock the WebSocketDataStream
        self.stream = MagicMock(spec=WebSocketDataStream)
        self.stream.connect.return_value = True
        self.stream.disconnect.return_value = None
        self.stream.is_connected.return_value = True
        self.stream.start.return_value = True
        self.stream.stop.return_value = None
        self.stream.get_status.return_value = StreamStatus.CONNECTED
        
        # Create the connector
        self.connector = StreamingMarketDataConnector(
            stream=self.stream,
            event_system=self.event_system,
            config=self.config
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.connector._stream, self.stream)
        self.assertEqual(self.connector._event_system, self.event_system)
        self.assertEqual(self.connector._config, self.config)
        self.assertFalse(self.connector._running)
        self.assertEqual(self.connector._throttle_interval, 0.0)
        self.assertEqual(self.connector._backpressure_threshold, 0)
        self.assertEqual(self.connector._symbol_priorities, {})

    def test_start(self):
        """Test the start method."""
        # Call start
        result = self.connector.start()
        
        # Assertions
        self.assertTrue(result)
        self.stream.connect.assert_called_once()
        self.stream.start.assert_called_once()
        self.assertTrue(self.connector._running)

    def test_start_failure(self):
        """Test start method when stream fails to start."""
        # Setup
        self.stream.start.return_value = False
        
        # Call start
        result = self.connector.start()
        
        # Assertions
        self.assertFalse(result)
        self.stream.connect.assert_called_once()
        self.stream.start.assert_called_once()
        self.assertFalse(self.connector._running)

    def test_stop(self):
        """Test the stop method."""
        # Setup
        self.connector._running = True
        
        # Call stop
        self.connector.stop()
        
        # Assertions
        self.stream.stop.assert_called_once()
        self.stream.disconnect.assert_called_once()
        self.assertFalse(self.connector._running)

    def test_subscribe(self):
        """Test the subscribe method."""
        # Setup
        symbol = "AAPL"
        timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Call subscribe
        self.connector.subscribe(symbol, timeframes)
        
        # Assertions
        self.stream.subscribe.assert_called_once_with(symbol, timeframes)

    def test_unsubscribe(self):
        """Test the unsubscribe method."""
        # Setup
        symbol = "AAPL"
        timeframes = [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES]
        
        # Call unsubscribe
        self.connector.unsubscribe(symbol, timeframes)
        
        # Assertions
        self.stream.unsubscribe.assert_called_once_with(symbol, timeframes)

    def test_get_latest_data(self):
        """Test the get_latest_data method."""
        # Setup
        symbol = "AAPL"
        timeframe = DataTimeframe.ONE_MINUTE
        test_data = {"timestamp": datetime.now(), "data": {"close": 150}}
        self.stream.get_latest_data.return_value = test_data
        
        # Call get_latest_data
        result = self.connector.get_latest_data(symbol, timeframe)
        
        # Assertions
        self.assertEqual(result, test_data)
        self.stream.get_latest_data.assert_called_once_with(symbol, timeframe)

    def test_get_status(self):
        """Test the get_status method."""
        # Call get_status
        result = self.connector.get_status()
        
        # Assertions
        self.assertEqual(result, StreamStatus.CONNECTED)
        self.stream.get_status.assert_called_once()

    def test_get_subscribed_symbols(self):
        """Test the get_subscribed_symbols method."""
        # Setup
        test_symbols = {"AAPL", "MSFT"}
        self.stream.get_subscribed_symbols.return_value = test_symbols
        
        # Call get_subscribed_symbols
        result = self.connector.get_subscribed_symbols()
        
        # Assertions
        self.assertEqual(result, test_symbols)
        self.stream.get_subscribed_symbols.assert_called_once()

    def test_get_subscribed_timeframes(self):
        """Test the get_subscribed_timeframes method."""
        # Setup
        test_timeframes = {DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES}
        self.stream.get_subscribed_timeframes.return_value = test_timeframes
        
        # Call get_subscribed_timeframes
        result = self.connector.get_subscribed_timeframes()
        
        # Assertions
        self.assertEqual(result, test_timeframes)
        self.stream.get_subscribed_timeframes.assert_called_once()

    def test_set_throttle_interval(self):
        """Test the set_throttle_interval method."""
        # Setup
        interval = 0.5
        
        # Call set_throttle_interval
        self.connector.set_throttle_interval(interval)
        
        # Assertions
        self.assertEqual(self.connector._throttle_interval, interval)

    def test_set_backpressure_threshold(self):
        """Test the set_backpressure_threshold method."""
        # Setup
        threshold = 100
        
        # Call set_backpressure_threshold
        self.connector.set_backpressure_threshold(threshold)
        
        # Assertions
        self.assertEqual(self.connector._backpressure_threshold, threshold)

    def test_set_symbol_priority(self):
        """Test the set_symbol_priority method."""
        # Setup
        symbol = "AAPL"
        priority = 10
        
        # Call set_symbol_priority
        self.connector.set_symbol_priority(symbol, priority)
        
        # Assertions
        self.assertEqual(self.connector._symbol_priorities[symbol], priority)

    def test_handle_stream_tick(self):
        """Test the _handle_stream_tick method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {"price": 150.0},
            "timestamp": datetime.now()
        }
        event = Event("stream.tick", test_data)
        
        # Call _handle_stream_tick
        self.connector._handle_stream_tick(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.tick")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_bar(self):
        """Test the _handle_stream_bar method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "timeframe": DataTimeframe.ONE_MINUTE.value,
            "data": {
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            "timestamp": datetime.now()
        }
        event = Event("stream.bar", test_data)
        
        # Call _handle_stream_bar
        self.connector._handle_stream_bar(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.bar")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_trade(self):
        """Test the _handle_stream_trade method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {
                "price": 150.0,
                "volume": 100,
                "side": "buy"
            },
            "timestamp": datetime.now()
        }
        event = Event("stream.trade", test_data)
        
        # Call _handle_stream_trade
        self.connector._handle_stream_trade(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.trade")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_quote(self):
        """Test the _handle_stream_quote method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {
                "bid": 149.5,
                "ask": 150.5,
                "bid_size": 100,
                "ask_size": 200
            },
            "timestamp": datetime.now()
        }
        event = Event("stream.quote", test_data)
        
        # Call _handle_stream_quote
        self.connector._handle_stream_quote(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.quote")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_orderbook(self):
        """Test the _handle_stream_orderbook method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {
                "bids": [{"price": 149.5, "size": 100}, {"price": 149.0, "size": 200}],
                "asks": [{"price": 150.5, "size": 150}, {"price": 151.0, "size": 250}]
            },
            "timestamp": datetime.now()
        }
        event = Event("stream.orderbook", test_data)
        
        # Call _handle_stream_orderbook
        self.connector._handle_stream_orderbook(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.orderbook")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_custom(self):
        """Test the _handle_stream_custom method."""
        # Setup
        test_data = {
            "symbol": "AAPL",
            "data": {"custom_field": "custom_value"},
            "timestamp": datetime.now()
        }
        event = Event("stream.custom", test_data)
        
        # Call _handle_stream_custom
        self.connector._handle_stream_custom(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.custom")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data, test_data)

    def test_handle_stream_event(self):
        """Test the _handle_stream_event method."""
        # Setup
        test_data = {
            "event": StreamEvent.CONNECTED,
            "timestamp": datetime.now()
        }
        event = Event("stream.event", test_data)
        
        # Call _handle_stream_event
        self.connector._handle_stream_event(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.connected")

    def test_handle_stream_event_error(self):
        """Test the _handle_stream_event method with error event."""
        # Setup
        error_msg = "Test error"
        test_data = {
            "event": StreamEvent.ERROR,
            "error": error_msg,
            "timestamp": datetime.now()
        }
        event = Event("stream.event", test_data)
        
        # Call _handle_stream_event
        self.connector._handle_stream_event(event)
        
        # Assertions
        self.event_system.emit.assert_called_once()
        event_name = self.event_system.emit.call_args[0][0]
        self.assertEqual(event_name, "market_data.error")
        event_data = self.event_system.emit.call_args[0][1]
        self.assertEqual(event_data["error"], error_msg)

    @patch('time.time')
    def test_should_throttle(self, mock_time):
        """Test the _should_throttle method."""
        # Setup
        self.connector._throttle_interval = 0.5
        self.connector._last_update_time = {"AAPL": 100.0}
        mock_time.return_value = 100.4  # Less than throttle interval
        
        # Call _should_throttle
        result = self.connector._should_throttle("AAPL")
        
        # Assertions
        self.assertTrue(result)
        
        # Change time to be greater than throttle interval
        mock_time.return_value = 100.6
        
        # Call _should_throttle again
        result = self.connector._should_throttle("AAPL")
        
        # Assertions
        self.assertFalse(result)

    def test_should_apply_backpressure(self):
        """Test the _should_apply_backpressure method."""
        # Setup
        self.connector._backpressure_threshold = 5
        self.connector._event_count = 10  # Greater than threshold
        
        # Call _should_apply_backpressure
        result = self.connector._should_apply_backpressure()
        
        # Assertions
        self.assertTrue(result)
        
        # Change event count to be less than threshold
        self.connector._event_count = 3
        
        # Call _should_apply_backpressure again
        result = self.connector._should_apply_backpressure()
        
        # Assertions
        self.assertFalse(result)

    def test_get_symbol_priority(self):
        """Test the _get_symbol_priority method."""
        # Setup
        self.connector._symbol_priorities = {"AAPL": 10, "MSFT": 5}
        
        # Call _get_symbol_priority for existing symbol
        result = self.connector._get_symbol_priority("AAPL")
        
        # Assertions
        self.assertEqual(result, 10)
        
        # Call _get_symbol_priority for non-existing symbol
        result = self.connector._get_symbol_priority("GOOGL")
        
        # Assertions
        self.assertEqual(result, 1)  # Default priority


if __name__ == '__main__':
    unittest.main()