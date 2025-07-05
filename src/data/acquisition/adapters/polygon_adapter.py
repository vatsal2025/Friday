"""Polygon.io Adapter for the Friday AI Trading System.

This module provides an adapter for the Polygon.io API, which offers financial data
including market data, company information, financial statements, and news articles.
"""

from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests

from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter, FinancialDataType
from src.data.acquisition.data_fetcher import DataSourceAdapter, DataTimeframe
from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError
from src.infrastructure.config.config_manager import ConfigManager

# Set up logging
logger = logging.getLogger(__name__)


class PolygonAdapter(FinancialDataAdapter):
    """Adapter for the Polygon.io API.

    This adapter provides access to financial data from Polygon.io, including market data,
    company information, financial statements, and news articles.
    
    Attributes:
        api_key: The API key for the Polygon.io API.
        base_url: The base URL for the Polygon.io API.
        is_connected_flag: Flag indicating whether the adapter is connected.
        last_api_call: The timestamp of the last API call.
        api_calls_minute: The number of API calls made in the current minute.
        api_calls_day: The number of API calls made in the current day.
        max_calls_minute: The maximum number of API calls allowed per minute.
        max_calls_day: The maximum number of API calls allowed per day.
    """

    # Mapping from DataTimeframe to Polygon.io timespan
    TIMEFRAME_MAP = {
        DataTimeframe.TICK: "minute",  # Polygon doesn't have tick data, use minute as fallback
        DataTimeframe.ONE_MINUTE: "minute",
        DataTimeframe.FIVE_MINUTES: "minute",
        DataTimeframe.FIFTEEN_MINUTES: "minute",
        DataTimeframe.THIRTY_MINUTES: "minute",
        DataTimeframe.ONE_HOUR: "hour",
        DataTimeframe.FOUR_HOURS: "hour",
        DataTimeframe.ONE_DAY: "day",
        DataTimeframe.ONE_WEEK: "week",
        DataTimeframe.ONE_MONTH: "month"
    }

    # Mapping from DataTimeframe to Polygon.io multiplier
    MULTIPLIER_MAP = {
        DataTimeframe.TICK: 1,  # Polygon doesn't have tick data, use 1 minute as fallback
        DataTimeframe.ONE_MINUTE: 1,
        DataTimeframe.FIVE_MINUTES: 5,
        DataTimeframe.FIFTEEN_MINUTES: 15,
        DataTimeframe.THIRTY_MINUTES: 30,
        DataTimeframe.ONE_HOUR: 1,
        DataTimeframe.FOUR_HOURS: 4,
        DataTimeframe.ONE_DAY: 1,
        DataTimeframe.ONE_WEEK: 1,
        DataTimeframe.ONE_MONTH: 1
    }

    # API limits for different subscription tiers
    API_LIMITS = {
        "basic": {"minute": 5, "day": 1000},
        "starter": {"minute": 10, "day": 2000},
        "developer": {"minute": 20, "day": 10000},
        "advanced": {"minute": 100, "day": 100000},
        "enterprise": {"minute": 200, "day": 200000}
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Polygon.io adapter.

        Args:
            api_key: The API key for the Polygon.io API. If None, it will be loaded from config.
        """
        super().__init__()
        
        config = ConfigManager()
        self.api_key = api_key or config.get("polygon.api_key")
        self.base_url = "https://api.polygon.io"
        self.is_connected_flag = False
        
        # Rate limiting
        self.last_api_call = 0
        self.api_calls_minute = 0
        self.api_calls_day = 0
        self.minute_reset = datetime.now()
        self.day_reset = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Set API limits based on subscription tier
        subscription_tier = config.get("polygon.subscription_tier", "basic").lower()
        if subscription_tier not in self.API_LIMITS:
            subscription_tier = "basic"
            logger.warning(f"Unknown subscription tier: {subscription_tier}. Using 'basic' limits.")
        
        self.max_calls_minute = self.API_LIMITS[subscription_tier]["minute"]
        self.max_calls_day = self.API_LIMITS[subscription_tier]["day"]
        
        # Available timeframes
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
        """Connect to the Polygon.io API.

        Returns:
            bool: True if the connection was successful, False otherwise.

        Raises:
            DataConnectionError: If the connection fails.
        """
        try:
            # Test the connection by making a simple API call
            url = f"{self.base_url}/v3/reference/tickers?active=true&limit=1&apiKey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                self.is_connected_flag = True
                logger.info("Successfully connected to Polygon.io API")
                return True
            else:
                error_msg = f"Failed to connect to Polygon.io API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Failed to connect to Polygon.io API: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from the Polygon.io API.

        Returns:
            bool: True if the disconnection was successful, False otherwise.
        """
        # No actual disconnection needed for REST API
        self.is_connected_flag = False
        logger.info("Disconnected from Polygon.io API")
        return True

    def is_connected(self) -> bool:
        """Check if the adapter is connected to the Polygon.io API.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.is_connected_flag

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits for the Polygon.io API.

        Raises:
            Exception: If the rate limit is exceeded.
        """
        current_time = datetime.now()
        
        # Reset counters if a new minute or day has started
        if current_time - self.minute_reset >= timedelta(minutes=1):
            self.api_calls_minute = 0
            self.minute_reset = current_time
        
        if current_time - self.day_reset >= timedelta(days=1):
            self.api_calls_day = 0
            self.day_reset = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if we've exceeded the rate limits
        if self.api_calls_minute >= self.max_calls_minute:
            wait_time = 60 - (current_time - self.minute_reset).seconds
            logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds.")
            time.sleep(wait_time)
            self.api_calls_minute = 0
            self.minute_reset = datetime.now()
        
        if self.api_calls_day >= self.max_calls_day:
            raise Exception("Daily API call limit exceeded")
        
        # Increment counters
        self.api_calls_minute += 1
        self.api_calls_day += 1
        
        # Add a small delay between API calls to avoid hitting the rate limit
        elapsed = time.time() - self.last_api_call
        if elapsed < 0.2:  # Minimum 200ms between calls
            time.sleep(0.2 - elapsed)
        
        self.last_api_call = time.time()

    def fetch_data(self, 
                  symbol: str, 
                  timeframe: DataTimeframe, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None, 
                  limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch market data from Polygon.io.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data.

        Raises:
            DataConnectionError: If the connection fails.
            DataValidationError: If the data validation fails.
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()
        
        # Check rate limit
        self._check_rate_limit()
        
        # Validate timeframe
        if timeframe not in self.available_timeframes:
            raise DataValidationError(f"Timeframe {timeframe} not supported by Polygon.io adapter")
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Default to 1 year of data
            start_date = end_date - timedelta(days=365)
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Get the appropriate timespan and multiplier
        timespan = self.TIMEFRAME_MAP[timeframe]
        multiplier = self.MULTIPLIER_MAP[timeframe]
        
        # Build the URL
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}?apiKey={self.api_key}"
        
        # Add limit if provided
        if limit:
            url += f"&limit={limit}"
        
        try:
            # Make the API call
            response = requests.get(url)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch data from Polygon.io: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse the response
            data = response.json()
            
            if data.get("status") != "OK" or "results" not in data:
                error_msg = f"Invalid response from Polygon.io: {data}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Create DataFrame
            results = data["results"]
            if not results:
                logger.warning(f"No data returned for {symbol} with timeframe {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # Rename columns to standard format
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions"
            })
            
            # Convert timestamp from milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Apply limit if provided
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            return df
        except Exception as e:
            error_msg = f"Failed to fetch data from Polygon.io: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_symbols(self) -> List[str]:
        """Get a list of available symbols from Polygon.io.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If the connection fails.
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()
        
        # Check rate limit
        self._check_rate_limit()
        
        try:
            # Build the URL for active tickers
            url = f"{self.base_url}/v3/reference/tickers?active=true&limit=1000&apiKey={self.api_key}"
            
            # Make the API call
            response = requests.get(url)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch symbols from Polygon.io: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse the response
            data = response.json()
            
            if "results" not in data:
                error_msg = f"Invalid response from Polygon.io: {data}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Extract symbols
            symbols = [ticker["ticker"] for ticker in data["results"]]
            
            # If there are more pages, fetch them
            next_url = data.get("next_url")
            while next_url:
                # Check rate limit
                self._check_rate_limit()
                
                # Make the API call
                next_url_with_key = f"{next_url}&apiKey={self.api_key}"
                response = requests.get(next_url_with_key)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch next page of symbols: {response.status_code} - {response.text}")
                    break
                
                # Parse the response
                data = response.json()
                
                if "results" not in data:
                    logger.warning(f"Invalid response for next page: {data}")
                    break
                
                # Extract symbols
                symbols.extend([ticker["ticker"] for ticker in data["results"]])
                
                # Update next_url
                next_url = data.get("next_url")
            
            return symbols
        except Exception as e:
            error_msg = f"Failed to fetch symbols from Polygon.io: {str(e)}"
            logger.error(error_msg)
            
            # Return a fallback list of common symbols
            logger.info("Using fallback list of common symbols")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get a list of available timeframes.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for the specified symbol.

        Args:
            symbol: The symbol to get company information for.

        Returns:
            Dict[str, Any]: Company information.

        Raises:
            DataConnectionError: If the connection fails.
            DataValidationError: If the data validation fails.
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()
        
        # Check rate limit
        self._check_rate_limit()
        
        try:
            # Build the URL for ticker details
            url = f"{self.base_url}/v3/reference/tickers/{symbol}?apiKey={self.api_key}"
            
            # Make the API call
            response = requests.get(url)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch company info from Polygon.io: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse the response
            data = response.json()
            
            if "results" not in data:
                error_msg = f"Invalid response from Polygon.io: {data}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Extract company info
            company = data["results"]
            
            # Format the response to match the expected structure
            return {
                "symbol": company["ticker"],
                "name": company["name"],
                "description": company.get("description", ""),
                "exchange": company.get("primary_exchange", ""),
                "industry": company.get("sic_description", ""),
                "sector": "",  # Polygon doesn't provide sector directly
                "website": company.get("homepage_url", ""),
                "market_cap": company.get("market_cap", 0),
                "employees": company.get("total_employees", 0),
                "country": company.get("locale", ""),
                "address": company.get("address", {}).get("address1", ""),
                "city": company.get("address", {}).get("city", ""),
                "state": company.get("address", {}).get("state", ""),
                "zip": company.get("address", {}).get("postal_code", ""),
                "phone": company.get("phone_number", "")
            }
        except Exception as e:
            error_msg = f"Failed to fetch company info from Polygon.io: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_financial_statements(self, symbol: str, period_type: str = "quarterly") -> Dict[str, Any]:
        """Get financial statements for the specified symbol.

        Args:
            symbol: The symbol to get financial statements for.
            period_type: The period type (quarterly or annual).

        Returns:
            Dict[str, Any]: Financial statements.

        Raises:
            DataConnectionError: If the connection fails.
            DataValidationError: If the data validation fails.
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()
        
        # Check rate limit
        self._check_rate_limit()
        
        # Validate period_type
        if period_type not in ["quarterly", "annual"]:
            raise DataValidationError(f"Invalid period type: {period_type}. Must be 'quarterly' or 'annual'.")
        
        # Map period_type to Polygon.io timeframe
        timeframe = "Q" if period_type == "quarterly" else "A"
        
        try:
            # Build the URL for financial statements
            url = f"{self.base_url}/v2/reference/financials/{symbol}?timeframe={timeframe}&limit=4&apiKey={self.api_key}"
            
            # Make the API call
            response = requests.get(url)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch financial statements from Polygon.io: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse the response
            data = response.json()
            
            if "results" not in data:
                error_msg = f"Invalid response from Polygon.io: {data}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Extract financial statements
            statements = data["results"]
            
            # Format the response to match the expected structure
            financials = {
                "symbol": symbol,
                "period_type": period_type,
                "statements": []
            }
            
            for statement in statements:
                # Extract the date
                date = statement.get("filing_date", statement.get("start_date", ""))
                
                # Extract financial metrics
                financials_data = statement.get("financials", {})
                income_statement = financials_data.get("income_statement", {})
                balance_sheet = financials_data.get("balance_sheet", {})
                cash_flow_statement = financials_data.get("cash_flow_statement", {})
                
                # Calculate derived metrics
                revenue = income_statement.get("revenue", {}).get("value", 0)
                net_income = income_statement.get("net_income_loss", {}).get("value", 0)
                total_assets = balance_sheet.get("assets", {}).get("value", 0)
                total_liabilities = balance_sheet.get("liabilities", {}).get("value", 0)
                
                # Add to statements list
                financials["statements"].append({
                    "date": date,
                    "revenue": revenue,
                    "net_income": net_income,
                    "eps": income_statement.get("basic_earnings_per_share", {}).get("value", 0),
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "total_equity": total_assets - total_liabilities,
                    "operating_cash_flow": cash_flow_statement.get("net_cash_flow_from_operating_activities", {}).get("value", 0),
                    "investing_cash_flow": cash_flow_statement.get("net_cash_flow_from_investing_activities", {}).get("value", 0),
                    "financing_cash_flow": cash_flow_statement.get("net_cash_flow_from_financing_activities", {}).get("value", 0),
                    "free_cash_flow": cash_flow_statement.get("free_cash_flow", {}).get("value", 0)
                })
            
            return financials
        except Exception as e:
            error_msg = f"Failed to fetch financial statements from Polygon.io: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news articles for the specified symbol or general market news.

        Args:
            symbol: The symbol to get news for. If None, get general market news.
            limit: The maximum number of news articles to return.

        Returns:
            List[Dict[str, Any]]: List of news articles.

        Raises:
            DataConnectionError: If the connection fails.
            DataValidationError: If the data validation fails.
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()
        
        # Check rate limit
        self._check_rate_limit()
        
        try:
            # Build the URL for news
            url = f"{self.base_url}/v2/reference/news?limit={limit}&apiKey={self.api_key}"
            
            # Add ticker filter if symbol is provided
            if symbol:
                url += f"&ticker={symbol}"
            
            # Make the API call
            response = requests.get(url)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch news from Polygon.io: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise DataConnectionError(error_msg)
            
            # Parse the response
            data = response.json()
            
            if "results" not in data:
                error_msg = f"Invalid response from Polygon.io: {data}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Extract news articles
            articles = data["results"]
            
            # Format the response to match the expected structure
            news = []
            for article in articles:
                news.append({
                    "symbol": article.get("tickers", [symbol])[0] if article.get("tickers") else symbol,
                    "title": article.get("title", ""),
                    "summary": article.get("description", ""),
                    "url": article.get("article_url", ""),
                    "published_at": article.get("published_utc", "")
                })
            
            return news
        except Exception as e:
            error_msg = f"Failed to fetch news from Polygon.io: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)