"""Tests for the Financial Data Service.

This module contains tests for the Financial Data Service, which provides access to
financial data, including market data, company information, financial statements, and news articles.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

from src.integration.services.financial_data_service import FinancialDataService
from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter, FinancialDataType
from src.data.acquisition.data_fetcher import DataTimeframe


class TestFinancialDataService(unittest.TestCase):
    """Tests for the Financial Data Service."""

    def setUp(self):
        """Set up the test environment."""
        # Create a mock adapter
        self.mock_adapter = MagicMock(spec=FinancialDataAdapter)
        
        # Configure the mock adapter
        self.mock_adapter.connect.return_value = True
        self.mock_adapter.disconnect.return_value = True
        self.mock_adapter.is_connected.return_value = True
        
        # Create the service with the mock adapter
        self.service = FinancialDataService(adapter=self.mock_adapter)

    def test_start(self):
        """Test starting the service."""
        # Test starting the service
        result = self.service.start()
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the adapter was connected
        self.mock_adapter.connect.assert_called_once()

    def test_stop(self):
        """Test stopping the service."""
        # Test stopping the service
        result = self.service.stop()
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify the adapter was disconnected
        self.mock_adapter.disconnect.assert_called_once()

    def test_authenticate(self):
        """Test authenticating with the service."""
        # Configure the mock adapter
        self.mock_adapter.authenticate.return_value = {
            "token": "test_token",
            "expires_at": 1234567890,
            "status": "success"
        }
        
        # Test authenticating with the service
        result = self.service.authenticate("test_api_key")
        
        # Verify the result
        self.assertEqual(result["token"], "test_token")
        self.assertEqual(result["status"], "success")
        
        # Verify the adapter was called
        self.mock_adapter.authenticate.assert_called_once_with("test_api_key")

    def test_get_companies(self):
        """Test getting company information."""
        # Configure the mock adapter
        self.mock_adapter.get_company_info.return_value = {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NASDAQ",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
        }
        
        # Test getting company information
        result = self.service.get_companies(["AAPL"])
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")
        self.assertEqual(result[0]["name"], "Apple Inc.")
        
        # Verify the adapter was called
        self.mock_adapter.get_company_info.assert_called_once_with("AAPL")

    def test_get_financials(self):
        """Test getting financial statements."""
        # Configure the mock adapter
        self.mock_adapter.get_financial_statements.return_value = {
            "symbol": "AAPL",
            "period_type": "quarterly",
            "statements": [
                {
                    "date": "2023-03-31",
                    "revenue": 94836000000,
                    "net_income": 24160000000,
                    "eps": 1.52,
                    "total_assets": 335033000000,
                    "total_liabilities": 234242000000,
                    "total_equity": 100791000000
                }
            ]
        }
        
        # Test getting financial statements
        result = self.service.get_financials("AAPL", "quarterly")
        
        # Verify the result
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["period_type"], "quarterly")
        self.assertEqual(len(result["statements"]), 1)
        self.assertEqual(result["statements"][0]["revenue"], 94836000000)
        
        # Verify the adapter was called
        self.mock_adapter.get_financial_statements.assert_called_once_with("AAPL", "quarterly")

    def test_get_news(self):
        """Test getting news articles."""
        # Configure the mock adapter
        self.mock_adapter.get_news.return_value = [
            {
                "symbol": "AAPL",
                "title": "Apple Announces New iPhone",
                "summary": "Apple Inc. announced the new iPhone 15 today.",
                "url": "https://example.com/news/apple-iphone-15",
                "published_at": "2023-09-12T10:00:00Z"
            }
        ]
        
        # Test getting news articles
        result = self.service.get_news("AAPL", 1)
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")
        self.assertEqual(result[0]["title"], "Apple Announces New iPhone")
        
        # Verify the adapter was called
        self.mock_adapter.get_news.assert_called_once_with("AAPL", 1)

    def test_get_market_data(self):
        """Test getting market data."""
        # Create a sample DataFrame
        df = pd.DataFrame({
            "timestamp": [datetime(2023, 9, 12, 10, 0), datetime(2023, 9, 12, 11, 0)],
            "open": [180.0, 181.0],
            "high": [182.0, 183.0],
            "low": [179.0, 180.0],
            "close": [181.0, 182.0],
            "volume": [1000000, 1100000]
        })
        
        # Configure the mock adapter
        self.mock_adapter.fetch_data.return_value = df
        
        # Test getting market data
        result = self.service.get_market_data(
            "AAPL",
            DataTimeframe.ONE_HOUR,
            datetime(2023, 9, 12),
            datetime(2023, 9, 13),
            10
        )
        
        # Verify the result
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["timeframe"], DataTimeframe.ONE_HOUR.value)
        self.assertEqual(len(result["data"]), 2)
        self.assertEqual(result["data"][0]["open"], 180.0)
        
        # Verify the adapter was called
        self.mock_adapter.fetch_data.assert_called_once_with(
            "AAPL",
            DataTimeframe.ONE_HOUR,
            datetime(2023, 9, 12),
            datetime(2023, 9, 13),
            10
        )

    def test_get_market_data_with_string_timeframe(self):
        """Test getting market data with a string timeframe."""
        # Create a sample DataFrame
        df = pd.DataFrame({
            "timestamp": [datetime(2023, 9, 12, 10, 0), datetime(2023, 9, 12, 11, 0)],
            "open": [180.0, 181.0],
            "high": [182.0, 183.0],
            "low": [179.0, 180.0],
            "close": [181.0, 182.0],
            "volume": [1000000, 1100000]
        })
        
        # Configure the mock adapter
        self.mock_adapter.fetch_data.return_value = df
        
        # Test getting market data with a string timeframe
        result = self.service.get_market_data(
            "AAPL",
            "ONE_HOUR",
            "2023-09-12",
            "2023-09-13",
            10
        )
        
        # Verify the result
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["timeframe"], DataTimeframe.ONE_HOUR.value)
        self.assertEqual(len(result["data"]), 2)
        
        # Verify the adapter was called with the correct timeframe
        self.mock_adapter.fetch_data.assert_called_once()
        args, _ = self.mock_adapter.fetch_data.call_args
        self.assertEqual(args[1], DataTimeframe.ONE_HOUR)


if __name__ == "__main__":
    unittest.main()