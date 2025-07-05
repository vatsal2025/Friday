"""Tests for the Polygon.io Adapter.

This module contains tests for the Polygon.io Adapter, which provides access to
financial data from the Polygon.io API.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd
import requests

from src.data.acquisition.adapters.polygon_adapter import PolygonAdapter
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError


class TestPolygonAdapter(unittest.TestCase):
    """Tests for the Polygon.io Adapter."""

    def setUp(self):
        """Set up the test environment."""
        # Create the adapter with a test API key
        self.adapter = PolygonAdapter(api_key="test_api_key")
        
        # Mock the requests.get method
        self.patcher = patch("requests.get")
        self.mock_get = self.patcher.start()
        
        # Configure the mock response for the connection test
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"ticker": "AAPL"}]}
        self.mock_get.return_value = mock_response

    def tearDown(self):
        """Clean up after the tests."""
        self.patcher.stop()

    def test_connect(self):
        """Test connecting to the Polygon.io API."""
        # Test connecting to the API
        result = self.adapter.connect()
        
        # Verify the result
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_connected())
        
        # Verify the API call was made
        self.mock_get.assert_called_once()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])

    def test_connect_failure(self):
        """Test connecting to the Polygon.io API with a failure."""
        # Configure the mock response for a failed connection
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        self.mock_get.return_value = mock_response
        
        # Test connecting to the API
        with self.assertRaises(DataConnectionError):
            self.adapter.connect()
        
        # Verify the API call was made
        self.mock_get.assert_called_once()

    def test_disconnect(self):
        """Test disconnecting from the Polygon.io API."""
        # Connect first
        self.adapter.connect()
        
        # Test disconnecting
        result = self.adapter.disconnect()
        
        # Verify the result
        self.assertTrue(result)
        self.assertFalse(self.adapter.is_connected())

    def test_fetch_data(self):
        """Test fetching market data from the Polygon.io API."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the data fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {
                    "t": 1631448000000,  # 2021-09-12 10:00:00
                    "o": 180.0,
                    "h": 182.0,
                    "l": 179.0,
                    "c": 181.0,
                    "v": 1000000,
                    "vw": 180.5,
                    "n": 5000
                },
                {
                    "t": 1631451600000,  # 2021-09-12 11:00:00
                    "o": 181.0,
                    "h": 183.0,
                    "l": 180.0,
                    "c": 182.0,
                    "v": 1100000,
                    "vw": 181.5,
                    "n": 5500
                }
            ]
        }
        self.mock_get.return_value = mock_response
        
        # Test fetching data
        df = self.adapter.fetch_data(
            "AAPL",
            DataTimeframe.ONE_HOUR,
            datetime(2021, 9, 12),
            datetime(2021, 9, 13),
            10
        )
        
        # Verify the result
        self.assertEqual(len(df), 2)
        self.assertEqual(df["open"].iloc[0], 180.0)
        self.assertEqual(df["high"].iloc[0], 182.0)
        self.assertEqual(df["low"].iloc[0], 179.0)
        self.assertEqual(df["close"].iloc[0], 181.0)
        self.assertEqual(df["volume"].iloc[0], 1000000)
        
        # Verify the API call was made
        self.mock_get.assert_called()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])
        self.assertIn("AAPL", args[0])
        self.assertIn("1/hour", args[0])  # 1 hour timeframe

    def test_fetch_data_failure(self):
        """Test fetching market data with a failure."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for a failed data fetch
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        self.mock_get.return_value = mock_response
        
        # Test fetching data
        with self.assertRaises(DataConnectionError):
            self.adapter.fetch_data(
                "AAPL",
                DataTimeframe.ONE_HOUR,
                datetime(2021, 9, 12),
                datetime(2021, 9, 13),
                10
            )
        
        # Verify the API call was made
        self.mock_get.assert_called()

    def test_fetch_data_invalid_response(self):
        """Test fetching market data with an invalid response."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response with an invalid format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ERROR", "error": "Some error"}
        self.mock_get.return_value = mock_response
        
        # Test fetching data
        with self.assertRaises(DataValidationError):
            self.adapter.fetch_data(
                "AAPL",
                DataTimeframe.ONE_HOUR,
                datetime(2021, 9, 12),
                datetime(2021, 9, 13),
                10
            )
        
        # Verify the API call was made
        self.mock_get.assert_called()

    def test_get_symbols(self):
        """Test getting a list of available symbols."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the symbols fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"},
                {"ticker": "GOOGL"}
            ],
            "next_url": None
        }
        self.mock_get.return_value = mock_response
        
        # Test getting symbols
        symbols = self.adapter.get_symbols()
        
        # Verify the result
        self.assertEqual(len(symbols), 3)
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        self.assertIn("GOOGL", symbols)
        
        # Verify the API call was made
        self.mock_get.assert_called()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])
        self.assertIn("reference/tickers", args[0])

    def test_get_symbols_with_pagination(self):
        """Test getting a list of available symbols with pagination."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the first page
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "results": [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"}
            ],
            "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc123"
        }
        
        # Configure the mock response for the second page
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "results": [
                {"ticker": "GOOGL"},
                {"ticker": "AMZN"}
            ],
            "next_url": None
        }
        
        # Set up the mock to return different responses for different calls
        self.mock_get.side_effect = [first_response, second_response]
        
        # Test getting symbols
        symbols = self.adapter.get_symbols()
        
        # Verify the result
        self.assertEqual(len(symbols), 4)
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        self.assertIn("GOOGL", symbols)
        self.assertIn("AMZN", symbols)
        
        # Verify the API calls were made
        self.assertEqual(self.mock_get.call_count, 2)

    def test_get_company_info(self):
        """Test getting company information."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the company info fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
                "primary_exchange": "NASDAQ",
                "sic_description": "Electronic Computers",
                "homepage_url": "https://www.apple.com",
                "market_cap": 2500000000000,
                "total_employees": 154000,
                "locale": "US",
                "address": {
                    "address1": "One Apple Park Way",
                    "city": "Cupertino",
                    "state": "CA",
                    "postal_code": "95014"
                },
                "phone_number": "(408) 996-1010"
            }
        }
        self.mock_get.return_value = mock_response
        
        # Test getting company info
        company = self.adapter.get_company_info("AAPL")
        
        # Verify the result
        self.assertEqual(company["symbol"], "AAPL")
        self.assertEqual(company["name"], "Apple Inc.")
        self.assertEqual(company["exchange"], "NASDAQ")
        self.assertEqual(company["industry"], "Electronic Computers")
        self.assertEqual(company["website"], "https://www.apple.com")
        self.assertEqual(company["market_cap"], 2500000000000)
        self.assertEqual(company["employees"], 154000)
        self.assertEqual(company["country"], "US")
        self.assertEqual(company["address"], "One Apple Park Way")
        self.assertEqual(company["city"], "Cupertino")
        self.assertEqual(company["state"], "CA")
        self.assertEqual(company["zip"], "95014")
        self.assertEqual(company["phone"], "(408) 996-1010")
        
        # Verify the API call was made
        self.mock_get.assert_called()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])
        self.assertIn("reference/tickers/AAPL", args[0])

    def test_get_financial_statements(self):
        """Test getting financial statements."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the financial statements fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "filing_date": "2023-03-31",
                    "start_date": "2023-01-01",
                    "financials": {
                        "income_statement": {
                            "revenue": {"value": 94836000000},
                            "net_income_loss": {"value": 24160000000},
                            "basic_earnings_per_share": {"value": 1.52}
                        },
                        "balance_sheet": {
                            "assets": {"value": 335033000000},
                            "liabilities": {"value": 234242000000}
                        },
                        "cash_flow_statement": {
                            "net_cash_flow_from_operating_activities": {"value": 28600000000},
                            "net_cash_flow_from_investing_activities": {"value": -8600000000},
                            "net_cash_flow_from_financing_activities": {"value": -24200000000},
                            "free_cash_flow": {"value": 20000000000}
                        }
                    }
                }
            ]
        }
        self.mock_get.return_value = mock_response
        
        # Test getting financial statements
        financials = self.adapter.get_financial_statements("AAPL", "quarterly")
        
        # Verify the result
        self.assertEqual(financials["symbol"], "AAPL")
        self.assertEqual(financials["period_type"], "quarterly")
        self.assertEqual(len(financials["statements"]), 1)
        self.assertEqual(financials["statements"][0]["date"], "2023-03-31")
        self.assertEqual(financials["statements"][0]["revenue"], 94836000000)
        self.assertEqual(financials["statements"][0]["net_income"], 24160000000)
        self.assertEqual(financials["statements"][0]["eps"], 1.52)
        self.assertEqual(financials["statements"][0]["total_assets"], 335033000000)
        self.assertEqual(financials["statements"][0]["total_liabilities"], 234242000000)
        self.assertEqual(financials["statements"][0]["total_equity"], 335033000000 - 234242000000)
        self.assertEqual(financials["statements"][0]["operating_cash_flow"], 28600000000)
        self.assertEqual(financials["statements"][0]["investing_cash_flow"], -8600000000)
        self.assertEqual(financials["statements"][0]["financing_cash_flow"], -24200000000)
        self.assertEqual(financials["statements"][0]["free_cash_flow"], 20000000000)
        
        # Verify the API call was made
        self.mock_get.assert_called()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])
        self.assertIn("reference/financials/AAPL", args[0])
        self.assertIn("timeframe=Q", args[0])  # Quarterly

    def test_get_news(self):
        """Test getting news articles."""
        # Connect first
        self.adapter.connect()
        
        # Configure the mock response for the news fetch
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Apple Announces New iPhone",
                    "description": "Apple Inc. announced the new iPhone 15 today.",
                    "article_url": "https://example.com/news/apple-iphone-15",
                    "published_utc": "2023-09-12T10:00:00Z",
                    "tickers": ["AAPL"]
                }
            ]
        }
        self.mock_get.return_value = mock_response
        
        # Test getting news
        news = self.adapter.get_news("AAPL", 1)
        
        # Verify the result
        self.assertEqual(len(news), 1)
        self.assertEqual(news[0]["symbol"], "AAPL")
        self.assertEqual(news[0]["title"], "Apple Announces New iPhone")
        self.assertEqual(news[0]["summary"], "Apple Inc. announced the new iPhone 15 today.")
        self.assertEqual(news[0]["url"], "https://example.com/news/apple-iphone-15")
        self.assertEqual(news[0]["published_at"], "2023-09-12T10:00:00Z")
        
        # Verify the API call was made
        self.mock_get.assert_called()
        args, _ = self.mock_get.call_args
        self.assertIn("api.polygon.io", args[0])
        self.assertIn("apiKey=test_api_key", args[0])
        self.assertIn("reference/news", args[0])
        self.assertIn("ticker=AAPL", args[0])


if __name__ == "__main__":
    unittest.main()