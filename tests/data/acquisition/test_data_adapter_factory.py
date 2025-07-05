"""Tests for the DataAdapterFactory class.

This module contains tests for the DataAdapterFactory class and demonstrates
how to use it with HistoricalDataFetcher.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.acquisition.data_adapter_factory import DataAdapterFactory
from src.data.acquisition.data_fetcher import DataSourceType, DataTimeframe
from src.data.acquisition.historical_data_fetcher import HistoricalDataFetcher
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem


class TestDataAdapterFactory(unittest.TestCase):
    """Tests for the DataAdapterFactory class."""

    def setUp(self):
        """Set up the test environment."""
        self.config = ConfigManager()
        self.event_system = EventSystem()
        self.factory = DataAdapterFactory(self.config, self.event_system)

    def test_create_adapter_yahoo_finance(self):
        """Test creating a Yahoo Finance adapter."""
        # Create a Yahoo Finance adapter
        adapter = self.factory.create_adapter(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter_name="yahoo_finance"
        )
        
        # Verify the adapter type
        from src.data.acquisition.adapters.yahoo_finance_adapter import YahooFinanceAdapter
        self.assertIsInstance(adapter, YahooFinanceAdapter)

    def test_create_adapter_alpha_vantage(self):
        """Test creating an Alpha Vantage adapter."""
        # Create an Alpha Vantage adapter with API key
        adapter = self.factory.create_adapter(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter_name="alpha_vantage",
            api_key="demo"
        )
        
        # Verify the adapter type
        from src.data.acquisition.adapters.alpha_vantage_adapter import AlphaVantageAdapter
        self.assertIsInstance(adapter, AlphaVantageAdapter)
        self.assertEqual(adapter.api_key, "demo")

    def test_create_adapter_zerodha(self):
        """Test creating a Zerodha adapter."""
        # Create a Zerodha adapter with API key and secret
        adapter = self.factory.create_adapter(
            source_type=DataSourceType.BROKER_API,
            adapter_name="zerodha",
            api_key="demo_key",
            api_secret="demo_secret"
        )
        
        # Verify the adapter type
        from src.data.acquisition.adapters.zerodha_data_adapter import ZerodhaDataAdapter
        self.assertIsInstance(adapter, ZerodhaDataAdapter)
        self.assertEqual(adapter.api_key, "demo_key")
        self.assertEqual(adapter.api_secret, "demo_secret")

    def test_create_adapter_default(self):
        """Test creating an adapter without specifying a name."""
        # Mock the _adapters dictionary to have only one adapter for REST_API
        original_adapters = DataAdapterFactory._adapters.copy()
        try:
            # Create a temporary state with only one adapter for REST_API
            DataAdapterFactory._adapters = {
                DataSourceType.REST_API: {"polygon": DataAdapterFactory._adapters[DataSourceType.REST_API]["polygon"]}
            }
            
            # Create an adapter without specifying a name
            adapter = self.factory.create_adapter(
                source_type=DataSourceType.REST_API,
                api_key="demo"
            )
            
            # Verify the adapter type
            from src.data.acquisition.adapters.polygon_adapter import PolygonAdapter
            self.assertIsInstance(adapter, PolygonAdapter)
            self.assertEqual(adapter.api_key, "demo")
        finally:
            # Restore the original adapters
            DataAdapterFactory._adapters = original_adapters

    def test_create_adapter_with_invalid_source_type(self):
        """Test creating an adapter with an invalid source type."""
        # Try to create an adapter with an invalid source type
        with self.assertRaises(ValueError):
            self.factory.create_adapter(source_type=DataSourceType.CSV_FILE)

    def test_create_adapter_with_invalid_name(self):
        """Test creating an adapter with an invalid name."""
        # Try to create an adapter with an invalid name
        with self.assertRaises(ValueError):
            self.factory.create_adapter(
                source_type=DataSourceType.MARKET_DATA_PROVIDER,
                adapter_name="invalid_adapter"
            )

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""
        # Create a mock adapter class
        mock_adapter_class = MagicMock()
        
        # Register the mock adapter
        DataAdapterFactory.register_adapter(
            source_type=DataSourceType.CUSTOM,
            adapter_class=mock_adapter_class,
            name="mock_adapter"
        )
        
        # Verify the adapter was registered
        self.assertIn(DataSourceType.CUSTOM, DataAdapterFactory._adapters)
        self.assertIn("mock_adapter", DataAdapterFactory._adapters[DataSourceType.CUSTOM])
        self.assertEqual(DataAdapterFactory._adapters[DataSourceType.CUSTOM]["mock_adapter"], mock_adapter_class)


class TestDataAdapterFactoryWithHistoricalDataFetcher(unittest.TestCase):
    """Tests for using DataAdapterFactory with HistoricalDataFetcher."""

    def setUp(self):
        """Set up the test environment."""
        self.config = ConfigManager()
        self.event_system = EventSystem()
        self.factory = DataAdapterFactory(self.config, self.event_system)

    @patch('src.data.acquisition.adapters.yahoo_finance_adapter.yfinance.download')
    def test_historical_data_fetcher_with_yahoo_finance(self, mock_download):
        """Test using HistoricalDataFetcher with a Yahoo Finance adapter."""
        # Mock the yfinance.download method
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        mock_download.return_value = mock_data
        
        # Create a Yahoo Finance adapter using the factory
        adapter = self.factory.create_adapter(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter_name="yahoo_finance"
        )
        
        # Create a HistoricalDataFetcher with the adapter
        fetcher = HistoricalDataFetcher(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter=adapter,
            cache_enabled=True,
            config=self.config
        )
        
        # Connect to the data source
        fetcher.connect()
        
        # Fetch historical data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)
        data = fetcher.fetch_data(
            symbol="AAPL",
            timeframe=DataTimeframe.ONE_DAY,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify the data
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        
        # Verify the cache
        cache_key = f"AAPL_{DataTimeframe.ONE_DAY.value}"
        self.assertIn(cache_key, fetcher.cache)
        self.assertIs(data, fetcher.cache[cache_key])

    @patch('src.data.acquisition.adapters.alpha_vantage_adapter.requests.get')
    def test_historical_data_fetcher_with_alpha_vantage(self, mock_get):
        """Test using HistoricalDataFetcher with an Alpha Vantage adapter."""
        # Mock the requests.get method
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'Meta Data': {
                '1. Information': 'Daily Prices',
                '2. Symbol': 'AAPL',
                '3. Last Refreshed': '2023-01-03',
                '4. Output Size': 'Compact',
                '5. Time Zone': 'US/Eastern'
            },
            'Time Series (Daily)': {
                '2023-01-03': {
                    '1. open': '102.0',
                    '2. high': '107.0',
                    '3. low': '97.0',
                    '4. close': '105.0',
                    '5. volume': '1200'
                },
                '2023-01-02': {
                    '1. open': '101.0',
                    '2. high': '106.0',
                    '3. low': '96.0',
                    '4. close': '104.0',
                    '5. volume': '1100'
                },
                '2023-01-01': {
                    '1. open': '100.0',
                    '2. high': '105.0',
                    '3. low': '95.0',
                    '4. close': '103.0',
                    '5. volume': '1000'
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Create an Alpha Vantage adapter using the factory
        adapter = self.factory.create_adapter(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter_name="alpha_vantage",
            api_key="demo"
        )
        
        # Create a HistoricalDataFetcher with the adapter
        fetcher = HistoricalDataFetcher(
            source_type=DataSourceType.MARKET_DATA_PROVIDER,
            adapter=adapter,
            cache_enabled=True,
            config=self.config
        )
        
        # Connect to the data source
        fetcher.connect()
        
        # Fetch historical data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)
        data = fetcher.fetch_data(
            symbol="AAPL",
            timeframe=DataTimeframe.ONE_DAY,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify the data
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        
        # Verify the cache
        cache_key = f"AAPL_{DataTimeframe.ONE_DAY.value}"
        self.assertIn(cache_key, fetcher.cache)
        self.assertIs(data, fetcher.cache[cache_key])


if __name__ == "__main__":
    unittest.main()