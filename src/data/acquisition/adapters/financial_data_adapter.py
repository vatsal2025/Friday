"""Financial Data Adapter for the Friday AI Trading System.

This module provides an adapter that implements the DataSourceAdapter interface
for financial data providers, offering access to company information, financial statements,
and news articles in addition to market data.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import json
import logging
import pandas as pd
import requests
from abc import ABC, abstractmethod

from src.data.acquisition.data_fetcher import DataSourceAdapter, DataTimeframe
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError

# Set up logging
logger = logging.getLogger(__name__)


class FinancialDataType(Enum):
    """Types of financial data that can be retrieved."""
    COMPANY_INFO = "company_info"  # Basic company information
    FINANCIAL_STATEMENTS = "financial_statements"  # Financial statements (income, balance sheet, cash flow)
    NEWS = "news"  # News articles
    MARKET_DATA = "market_data"  # Market data (OHLCV)
    FUNDAMENTALS = "fundamentals"  # Fundamental data (P/E, EPS, etc.)
    ECONOMIC_INDICATORS = "economic_indicators"  # Economic indicators (GDP, inflation, etc.)


class FinancialDataAdapter(DataSourceAdapter):
    """Adapter for financial data providers that implements the DataSourceAdapter interface.

    This class adapts financial data providers to the DataSourceAdapter interface,
    allowing the system to fetch financial data through the common data API.
    It extends the standard market data capabilities with financial information.
    
    Attributes:
        api_key: The API key for the financial data provider.
        base_url: The base URL for the financial data provider's API.
        connected: Whether the adapter is connected to the financial data provider.
        rate_limit_tracker: Dictionary tracking API calls for rate limiting.
        available_timeframes: List of available timeframes.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the financial data adapter.

        Args:
            api_key: The API key for the financial data provider. If None, it will be loaded from config.
            base_url: The base URL for the financial data provider's API. If None, it will be loaded from config.
        """
        self.config = ConfigManager()
        self.api_key = api_key or self.config.get("financial_data.api_key")
        self.base_url = base_url or self.config.get("financial_data.base_url")
        self.connected = False
        self.rate_limit_tracker = {}
        
        # Define available timeframes
        self.available_timeframes = [
            DataTimeframe.ONE_MINUTE,
            DataTimeframe.FIVE_MINUTES,
            DataTimeframe.FIFTEEN_MINUTES,
            DataTimeframe.THIRTY_MINUTES,
            DataTimeframe.ONE_HOUR,
            DataTimeframe.FOUR_HOURS,
            DataTimeframe.ONE_DAY,
            DataTimeframe.ONE_WEEK,
            DataTimeframe.ONE_MONTH
        ]

    def connect(self) -> bool:
        """Connect to the financial data provider.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        try:
            # Test connection by making a simple API call
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/status", headers=headers)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Successfully connected to financial data provider")
                return True
            else:
                error_msg = f"Failed to connect to financial data provider: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Failed to connect to financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from the financial data provider.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        # No actual disconnection needed for REST API
        self.connected = False
        logger.info("Disconnected from financial data provider")
        return True

    def is_connected(self) -> bool:
        """Check if connected to the financial data provider.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connected

    def _check_rate_limit(self, endpoint: str) -> None:
        """Check if rate limit has been reached for the given endpoint.

        Args:
            endpoint: The API endpoint to check rate limit for.

        Raises:
            DataConnectionError: If rate limit has been reached.
        """
        # Implementation would depend on the specific API's rate limiting policies
        # This is a placeholder for actual rate limit checking logic
        pass

    def fetch_data(self, 
                  symbol: str, 
                  timeframe: DataTimeframe, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None, 
                  limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch market data from the financial data provider.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to the data source fails.
            DataValidationError: If the fetched data is invalid.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check if timeframe is supported
            if timeframe not in self.available_timeframes:
                raise DataValidationError(f"Unsupported timeframe: {timeframe.value}")
            
            # Check rate limit
            self._check_rate_limit("market_data")
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "interval": timeframe.value
            }
            
            if start_date:
                params["start_date"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["end_date"] = end_date.strftime("%Y-%m-%d")
            if limit:
                params["limit"] = limit
            
            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/market-data", headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch data: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["data"])
            
            # Apply limit if specified
            if limit is not None and not df.empty:
                df = df.tail(limit)
            
            # Ensure standard column names
            if not df.empty:
                df = df.rename(columns={
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })
                
                # Keep only OHLCV columns
                columns_to_keep = ["timestamp", "open", "high", "low", "close", "volume"]
                df = df[[col for col in columns_to_keep if col in df.columns]]
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to fetch data from financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_symbols(self) -> List[str]:
        """Get available symbols from the financial data provider.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check rate limit
            self._check_rate_limit("symbols")
            
            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/symbols", headers=headers)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch symbols: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse response
            data = response.json()
            
            # Extract symbols
            symbols = [item["symbol"] for item in data["data"]]
            
            return symbols
            
        except Exception as e:
            error_msg = f"Failed to fetch symbols from financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from the financial data provider.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes

    # Additional methods specific to financial data

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for a specific symbol.

        Args:
            symbol: The symbol to get company information for.

        Returns:
            Dict[str, Any]: Company information.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check rate limit
            self._check_rate_limit("company_info")
            
            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/company/{symbol}", headers=headers)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch company info: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse response
            data = response.json()
            
            return data["data"]
            
        except Exception as e:
            error_msg = f"Failed to fetch company info from financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_financial_statements(self, symbol: str, period_type: str = "quarterly") -> Dict[str, Any]:
        """Get financial statements for a specific symbol.

        Args:
            symbol: The symbol to get financial statements for.
            period_type: The period type (quarterly or annual).

        Returns:
            Dict[str, Any]: Financial statements.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check rate limit
            self._check_rate_limit("financial_statements")
            
            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            params = {"period_type": period_type}
            response = requests.get(f"{self.base_url}/financials/{symbol}", headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch financial statements: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse response
            data = response.json()
            
            return data["data"]
            
        except Exception as e:
            error_msg = f"Failed to fetch financial statements from financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news articles for a specific symbol or general market news.

        Args:
            symbol: The symbol to get news for. If None, get general market news.
            limit: The maximum number of news articles to return.

        Returns:
            List[Dict[str, Any]]: List of news articles.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check rate limit
            self._check_rate_limit("news")
            
            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            params = {"limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            response = requests.get(f"{self.base_url}/news", headers=headers, params=params)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch news: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse response
            data = response.json()
            
            return data["data"]
            
        except Exception as e:
            error_msg = f"Failed to fetch news from financial data provider: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)