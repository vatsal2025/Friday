"""Tests for the Zerodha data adapter.

This module contains tests for the ZerodhaDataAdapter class.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pandas as pd

from src.data.acquisition.adapters.zerodha_data_adapter import ZerodhaDataAdapter
from src.data.acquisition.data_fetcher import DataTimeframe, DataConnectionError, DataValidationError


class TestZerodhaDataAdapter(unittest.TestCase):
    """Tests for the ZerodhaDataAdapter class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a mock KiteConnect instance
        self.mock_kite_patcher = patch('src.data.acquisition.adapters.zerodha_data_adapter.KiteConnect')
        self.mock_kite_class = self.mock_kite_patcher.start()
        self.mock_kite = MagicMock()
        self.mock_kite_class.return_value = self.mock_kite

        # Create a mock KiteTicker instance
        self.mock_ticker_patcher = patch('src.data.acquisition.adapters.zerodha_data_adapter.KiteTicker')
        self.mock_ticker_class = self.mock_ticker_patcher.start()
        self.mock_ticker = MagicMock()
        self.mock_ticker_class.return_value = self.mock_ticker

        # Create a mock ConfigManager instance
        self.mock_config_patcher = patch('src.data.acquisition.adapters.zerodha_data_adapter.ConfigManager')
        self.mock_config_class = self.mock_config_patcher.start()
        self.mock_config = MagicMock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'zerodha.api_key': 'test_api_key',
            'zerodha.api_secret': 'test_api_secret'
        }.get(key, default)
        self.mock_config_class.return_value = self.mock_config

        # Create a mock EventSystem instance
        self.mock_event_system = MagicMock()

        # Create the adapter
        self.adapter = ZerodhaDataAdapter(event_system=self.mock_event_system)

    def tearDown(self):
        """Tear down the test environment."""
        self.mock_kite_patcher.stop()
        self.mock_ticker_patcher.stop()
        self.mock_config_patcher.stop()

    def test_init(self):
        """Test initialization of the adapter."""
        self.assertEqual(self.adapter.api_key, 'test_api_key')
        self.assertEqual(self.adapter.api_secret, 'test_api_secret')
        self.assertFalse(self.adapter.connected)
        self.assertFalse(self.adapter.authenticated)
        self.assertIsNone(self.adapter.access_token)
        self.assertFalse(self.adapter.ticker_running)
        self.assertEqual(self.adapter.event_system, self.mock_event_system)

    def test_connect(self):
        """Test connecting to Zerodha."""
        # Test successful connection
        self.mock_kite.margins.return_value = {'equity': {'available': {'cash': 10000}}}
        result = self.adapter.connect()
        self.assertTrue(result)
        self.assertTrue(self.adapter.connected)
        self.mock_kite.margins.assert_called_once()

        # Test connection failure
        self.adapter.connected = False
        self.mock_kite.margins.side_effect = Exception('Connection error')
        with self.assertRaises(DataConnectionError):
            self.adapter.connect()
        self.assertFalse(self.adapter.connected)

    def test_disconnect(self):
        """Test disconnecting from Zerodha."""
        # Set up the adapter as connected
        self.adapter.connected = True
        self.adapter.ticker = self.mock_ticker
        self.adapter.ticker_running = True

        # Test successful disconnection
        result = self.adapter.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.adapter.connected)
        self.assertFalse(self.adapter.authenticated)
        self.assertIsNone(self.adapter.access_token)
        self.mock_ticker.close.assert_called_once()

        # Test disconnection when not connected
        self.adapter.connected = False
        self.mock_ticker.reset_mock()
        result = self.adapter.disconnect()
        self.assertTrue(result)
        self.mock_ticker.close.assert_not_called()

    def test_is_connected(self):
        """Test checking if connected to Zerodha."""
        self.adapter.connected = True
        self.assertTrue(self.adapter.is_connected())

        self.adapter.connected = False
        self.assertFalse(self.adapter.is_connected())

    def test_authenticate(self):
        """Test authenticating with Zerodha."""
        # Test successful authentication
        self.mock_kite.generate_session.return_value = {'access_token': 'test_access_token'}
        self.mock_kite.instruments.return_value = [
            {'tradingsymbol': 'RELIANCE', 'instrument_token': 123456},
            {'tradingsymbol': 'INFY', 'instrument_token': 654321}
        ]

        result = self.adapter.authenticate('test_request_token')
        self.assertTrue(result)
        self.assertTrue(self.adapter.authenticated)
        self.assertEqual(self.adapter.access_token, 'test_access_token')
        self.mock_kite.generate_session.assert_called_once_with('test_request_token', api_secret='test_api_secret')
        self.mock_kite.set_access_token.assert_called_once_with('test_access_token')
        self.mock_ticker_class.assert_called_once_with(api_key='test_api_key', access_token='test_access_token')

        # Check that instrument tokens were loaded
        self.assertEqual(self.adapter.instrument_token_map, {'RELIANCE': 123456, 'INFY': 654321})
        self.assertEqual(self.adapter.symbol_token_map, {123456: 'RELIANCE', 654321: 'INFY'})

        # Test authentication failure
        self.adapter.authenticated = False
        self.mock_kite.generate_session.side_effect = Exception('Authentication error')
        with self.assertRaises(DataConnectionError):
            self.adapter.authenticate('test_request_token')
        self.assertFalse(self.adapter.authenticated)

    def test_start_ticker(self):
        """Test starting the ticker."""
        # Set up the adapter as authenticated
        self.adapter.authenticated = True
        self.adapter.access_token = 'test_access_token'

        # Test successful ticker start
        result = self.adapter.start_ticker()
        self.assertIsNone(result)  # Method returns None
        self.assertTrue(self.adapter.ticker_running)
        self.mock_ticker.connect.assert_called_once_with(threaded=True)

        # Test ticker start when already running
        self.mock_ticker.reset_mock()
        result = self.adapter.start_ticker()
        self.assertIsNone(result)
        self.mock_ticker.connect.assert_not_called()

        # Test ticker start when not authenticated
        self.adapter.authenticated = False
        self.adapter.ticker_running = False
        with self.assertRaises(DataConnectionError):
            self.adapter.start_ticker()

    def test_stop_ticker(self):
        """Test stopping the ticker."""
        # Set up the adapter with ticker running
        self.adapter.ticker = self.mock_ticker
        self.adapter.ticker_running = True

        # Test successful ticker stop
        result = self.adapter.stop_ticker()
        self.assertIsNone(result)  # Method returns None
        self.assertFalse(self.adapter.ticker_running)
        self.mock_ticker.close.assert_called_once()

        # Test ticker stop when not running
        self.mock_ticker.reset_mock()
        self.adapter.ticker_running = False
        result = self.adapter.stop_ticker()
        self.assertIsNone(result)
        self.mock_ticker.close.assert_not_called()

    def test_subscribe_symbols(self):
        """Test subscribing to symbols."""
        # Set up the adapter as authenticated with ticker running
        self.adapter.authenticated = True
        self.adapter.ticker = self.mock_ticker
        self.adapter.ticker_running = True
        self.adapter.instrument_token_map = {'RELIANCE': 123456, 'INFY': 654321}

        # Test subscribing to symbols
        self.adapter.subscribe_symbols(['RELIANCE', 'INFY'])
        self.assertEqual(self.adapter.subscribed_symbols, {'RELIANCE', 'INFY'})
        self.mock_ticker.subscribe.assert_called_once_with([123456, 654321])

        # Test subscribing when not authenticated
        self.adapter.authenticated = False
        with self.assertRaises(DataConnectionError):
            self.adapter.subscribe_symbols(['TCS'])

    def test_unsubscribe_symbols(self):
        """Test unsubscribing from symbols."""
        # Set up the adapter with subscribed symbols
        self.adapter.ticker = self.mock_ticker
        self.adapter.ticker_running = True
        self.adapter.instrument_token_map = {'RELIANCE': 123456, 'INFY': 654321, 'TCS': 789012}
        self.adapter.subscribed_symbols = {'RELIANCE', 'INFY', 'TCS'}

        # Test unsubscribing from symbols
        self.adapter.unsubscribe_symbols(['RELIANCE', 'INFY'])
        self.assertEqual(self.adapter.subscribed_symbols, {'TCS'})
        self.mock_ticker.unsubscribe.assert_called_once_with([123456, 654321])

    def test_fetch_data(self):
        """Test fetching data from Zerodha."""
        # Set up the adapter as authenticated
        self.adapter.connected = True
        self.adapter.authenticated = True
        self.adapter.instrument_token_map = {'RELIANCE': 123456}

        # Mock historical data response
        historical_data = [
            {
                'date': '2023-01-01T09:15:00+05:30',
                'open': 2500.0,
                'high': 2550.0,
                'low': 2480.0,
                'close': 2520.0,
                'volume': 1000000
            },
            {
                'date': '2023-01-01T09:16:00+05:30',
                'open': 2520.0,
                'high': 2540.0,
                'low': 2510.0,
                'close': 2530.0,
                'volume': 900000
            }
        ]
        self.mock_kite.historical_data.return_value = historical_data

        # Test fetching data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        result = self.adapter.fetch_data('RELIANCE', DataTimeframe.ONE_MINUTE, start_date, end_date)

        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.mock_kite.historical_data.assert_called_once_with(
            instrument_token=123456,
            from_date=start_date,
            to_date=end_date,
            interval='minute',
            continuous=False
        )

        # Test fetching data for unknown symbol
        with self.assertRaises(DataValidationError):
            self.adapter.fetch_data('UNKNOWN', DataTimeframe.ONE_MINUTE)

        # Test fetching data when not authenticated
        self.adapter.authenticated = False
        with self.assertRaises(DataConnectionError):
            self.adapter.fetch_data('RELIANCE', DataTimeframe.ONE_MINUTE)

    def test_get_tick_data(self):
        """Test getting tick data from the buffer."""
        # Set up the adapter with tick data in buffer
        now = datetime.now()
        self.adapter.data_buffer = {
            'RELIANCE': [
                {
                    'timestamp': now - timedelta(seconds=2),
                    'open': 2500.0,
                    'high': 2550.0,
                    'low': 2480.0,
                    'close': 2520.0,
                    'volume': 1000000
                },
                {
                    'timestamp': now - timedelta(seconds=1),
                    'open': 2520.0,
                    'high': 2540.0,
                    'low': 2510.0,
                    'close': 2530.0,
                    'volume': 900000
                }
            ]
        }

        # Test getting tick data
        result = self.adapter._get_tick_data('RELIANCE', None, None, None)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

        # Test getting tick data with limit
        result = self.adapter._get_tick_data('RELIANCE', None, None, 1)
        self.assertEqual(len(result), 1)

        # Test getting tick data for unknown symbol
        result = self.adapter._get_tick_data('UNKNOWN', None, None, None)
        self.assertTrue(result.empty)

    def test_get_symbols(self):
        """Test getting available symbols from Zerodha."""
        # Set up the adapter as authenticated with instrument tokens
        self.adapter.connected = True
        self.adapter.authenticated = True
        self.adapter.instrument_token_map = {'RELIANCE': 123456, 'INFY': 654321}

        # Test getting symbols from cache
        result = self.adapter.get_symbols()
        self.assertEqual(result, ['RELIANCE', 'INFY'])

        # Test getting symbols when not authenticated
        self.adapter.authenticated = False
        with self.assertRaises(DataConnectionError):
            self.adapter.get_symbols()

    def test_get_timeframes(self):
        """Test getting available timeframes from Zerodha."""
        result = self.adapter.get_timeframes()
        self.assertEqual(result, [
            DataTimeframe.ONE_MINUTE,
            DataTimeframe.FIVE_MINUTES,
            DataTimeframe.FIFTEEN_MINUTES,
            DataTimeframe.THIRTY_MINUTES,
            DataTimeframe.ONE_HOUR,
            DataTimeframe.ONE_DAY,
            DataTimeframe.ONE_WEEK,
            DataTimeframe.ONE_MONTH
        ])


if __name__ == '__main__':
    unittest.main()