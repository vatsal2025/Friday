"""Alpha Vantage adapter for the Friday AI Trading System.

This module provides an adapter that implements the DataSourceAdapter interface
for Alpha Vantage's API.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time
import os

import pandas as pd
import requests

from src.data.acquisition.data_fetcher import (
    DataSourceAdapter,
    DataTimeframe,
    DataConnectionError,
    DataValidationError
)
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Mapping from DataTimeframe to Alpha Vantage interval
TIMEFRAME_MAP = {
    DataTimeframe.ONE_MINUTE: "1min",
    DataTimeframe.FIVE_MINUTES: "5min",
    DataTimeframe.FIFTEEN_MINUTES: "15min",
    DataTimeframe.THIRTY_MINUTES: "30min",
    DataTimeframe.ONE_HOUR: "60min",
    DataTimeframe.ONE_DAY: "daily",
    DataTimeframe.ONE_WEEK: "weekly",
    DataTimeframe.ONE_MONTH: "monthly"
}

# Alpha Vantage API limits
API_LIMITS = {
    "standard": {"calls_per_minute": 5, "calls_per_day": 500},
    "premium": {"calls_per_minute": 75, "calls_per_day": 5000},
    "enterprise": {"calls_per_minute": 300, "calls_per_day": 20000}
}


class AlphaVantageAdapter(DataSourceAdapter):
    """Adapter for Alpha Vantage that implements the DataSourceAdapter interface.

    This class adapts the Alpha Vantage API to the DataSourceAdapter interface,
    allowing the system to fetch data from Alpha Vantage through the common data API.
    
    Attributes:
        api_key: The Alpha Vantage API key.
        base_url: The base URL for Alpha Vantage API.
        connected: Whether the adapter is connected to Alpha Vantage.
        rate_limit_tracker: Dictionary tracking API calls for rate limiting.
        plan_tier: The Alpha Vantage plan tier (standard, premium, enterprise).
        available_timeframes: List of available timeframes.
    """

    def __init__(self, api_key: Optional[str] = None, plan_tier: str = "standard"):
        """Initialize the Alpha Vantage adapter.

        Args:
            api_key: The Alpha Vantage API key. If None, it will be loaded from config.
            plan_tier: The Alpha Vantage plan tier. Defaults to "standard".
        """
        self.config = ConfigManager()
        self.api_key = api_key or self.config.get("alpha_vantage.api_key")
        self.base_url = "https://www.alphavantage.co/query"
        self.connected = False
        self.plan_tier = plan_tier.lower()
        
        # Initialize rate limit tracker
        self.rate_limit_tracker = {
            "minute": {
                "calls": 0,
                "reset_time": datetime.now() + timedelta(minutes=1)
            },
            "day": {
                "calls": 0,
                "reset_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            }
        }
        
        self.available_timeframes = [
            DataTimeframe.ONE_MINUTE,
            DataTimeframe.FIVE_MINUTES,
            DataTimeframe.FIFTEEN_MINUTES,
            DataTimeframe.THIRTY_MINUTES,
            DataTimeframe.ONE_HOUR,
            DataTimeframe.ONE_DAY,
            DataTimeframe.ONE_WEEK,
            DataTimeframe.ONE_MONTH
        ]

    def connect(self) -> bool:
        """Connect to Alpha Vantage.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        if not self.api_key:
            error_msg = "Alpha Vantage API key is not provided"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

        try:
            # Test connection by fetching a small amount of data
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "MSFT",
                "interval": "1min",
                "outputsize": "compact",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Error Message" in data:
                error_msg = f"Failed to connect to Alpha Vantage: {data['Error Message']}"
                logger.error(error_msg)
                self.connected = False
                return False
                
            if "Note" in data and "API call frequency" in data["Note"]:
                logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
            self.connected = True
            logger.info("Connected to Alpha Vantage successfully")
            return True
        except Exception as e:
            self.connected = False
            error_msg = f"Failed to connect to Alpha Vantage: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from Alpha Vantage.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        # No actual disconnection needed for REST API
        self.connected = False
        logger.info("Disconnected from Alpha Vantage")
        return True

    def is_connected(self) -> bool:
        """Check if connected to Alpha Vantage.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connected

    def _check_rate_limit(self) -> None:
        """Check and handle rate limiting for Alpha Vantage API.

        Raises:
            DataConnectionError: If rate limit is exceeded.
        """
        now = datetime.now()
        
        # Check minute rate limit
        minute_tracker = self.rate_limit_tracker["minute"]
        if now >= minute_tracker["reset_time"]:
            minute_tracker["calls"] = 0
            minute_tracker["reset_time"] = now + timedelta(minutes=1)
        
        # Check day rate limit
        day_tracker = self.rate_limit_tracker["day"]
        if now >= day_tracker["reset_time"]:
            day_tracker["calls"] = 0
            day_tracker["reset_time"] = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Get rate limits based on plan tier
        limits = API_LIMITS.get(self.plan_tier, API_LIMITS["standard"])
        
        # Check if we've exceeded the rate limits
        if minute_tracker["calls"] >= limits["calls_per_minute"]:
            wait_time = (minute_tracker["reset_time"] - now).total_seconds()
            error_msg = f"Alpha Vantage minute rate limit exceeded. Try again in {wait_time:.0f} seconds."
            logger.warning(error_msg)
            raise DataConnectionError(error_msg)
            
        if day_tracker["calls"] >= limits["calls_per_day"]:
            wait_time = (day_tracker["reset_time"] - now).total_seconds()
            error_msg = f"Alpha Vantage daily rate limit exceeded. Try again in {wait_time/3600:.1f} hours."
            logger.warning(error_msg)
            raise DataConnectionError(error_msg)
        
        # Increment the call counters
        minute_tracker["calls"] += 1
        day_tracker["calls"] += 1

    def _parse_intraday_data(self, data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """Parse intraday data from Alpha Vantage response.

        Args:
            data: The response data from Alpha Vantage.
            symbol: The symbol for the data.

        Returns:
            pd.DataFrame: The parsed data as a pandas DataFrame.

        Raises:
            DataValidationError: If the data cannot be parsed.
        """
        try:
            # Get the time series data
            meta_data = data.get("Meta Data", {})
            interval = meta_data.get("4. Interval", "")
            time_series_key = f"Time Series ({interval})"
            
            if time_series_key not in data:
                raise DataValidationError(f"Invalid data format: Missing {time_series_key}")
                
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col])
            df["volume"] = pd.to_numeric(df["volume"], downcast="integer")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            df = df.rename(columns={"index": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            raise DataValidationError(f"Failed to parse intraday data: {str(e)}")

    def _parse_daily_data(self, data: Dict[str, Any], symbol: str, adjusted: bool = True) -> pd.DataFrame:
        """Parse daily data from Alpha Vantage response.

        Args:
            data: The response data from Alpha Vantage.
            symbol: The symbol for the data.
            adjusted: Whether the data is adjusted for splits and dividends.

        Returns:
            pd.DataFrame: The parsed data as a pandas DataFrame.

        Raises:
            DataValidationError: If the data cannot be parsed.
        """
        try:
            # Get the time series data
            time_series_key = "Time Series (Daily)" if not adjusted else "Time Series (Daily) Adjusted"
            
            if time_series_key not in data:
                raise DataValidationError(f"Invalid data format: Missing {time_series_key}")
                
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns based on whether it's adjusted or not
            if adjusted:
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                    "7. dividend amount": "dividend",
                    "8. split coefficient": "split"
                })
            else:
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                })
            
            # Convert types
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            if "adjusted_close" in df.columns:
                df["adjusted_close"] = pd.to_numeric(df["adjusted_close"])
                
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], downcast="integer")
                
            # Reset index to make timestamp a column
            df = df.reset_index()
            df = df.rename(columns={"index": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            raise DataValidationError(f"Failed to parse daily data: {str(e)}")

    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to Alpha Vantage fails.
            DataValidationError: If the fetched data is invalid.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Map timeframe to Alpha Vantage interval
            if timeframe not in TIMEFRAME_MAP:
                raise DataValidationError(f"Unsupported timeframe: {timeframe.value}")
            
            interval = TIMEFRAME_MAP[timeframe]
            
            # Check rate limit
            self._check_rate_limit()
            
            # Determine function and parameters based on timeframe
            params = {"apikey": self.api_key, "symbol": symbol}
            
            if timeframe in [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES, 
                           DataTimeframe.FIFTEEN_MINUTES, DataTimeframe.THIRTY_MINUTES, 
                           DataTimeframe.ONE_HOUR]:
                # Intraday data
                params["function"] = "TIME_SERIES_INTRADAY"
                params["interval"] = interval
                params["outputsize"] = "full"  # Get full data, we'll filter by date later
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    raise DataConnectionError(f"Alpha Vantage API error: {data['Error Message']}")
                    
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
                df = self._parse_intraday_data(data, symbol)
                
            elif timeframe == DataTimeframe.ONE_DAY:
                # Daily data
                params["function"] = "TIME_SERIES_DAILY_ADJUSTED"
                params["outputsize"] = "full"  # Get full data, we'll filter by date later
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    raise DataConnectionError(f"Alpha Vantage API error: {data['Error Message']}")
                    
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
                df = self._parse_daily_data(data, symbol, adjusted=True)
                
            elif timeframe == DataTimeframe.ONE_WEEK:
                # Weekly data
                params["function"] = "TIME_SERIES_WEEKLY_ADJUSTED"
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    raise DataConnectionError(f"Alpha Vantage API error: {data['Error Message']}")
                    
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
                # Parse similar to daily data
                time_series_key = "Weekly Adjusted Time Series"
                if time_series_key not in data:
                    raise DataValidationError(f"Invalid data format: Missing {time_series_key}")
                    
                time_series = data[time_series_key]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                    "7. dividend amount": "dividend"
                })
                
                # Convert types and prepare DataFrame
                for col in ["open", "high", "low", "close", "adjusted_close"]:
                    df[col] = pd.to_numeric(df[col])
                df["volume"] = pd.to_numeric(df["volume"], downcast="integer")
                
                df = df.reset_index()
                df = df.rename(columns={"index": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                
            elif timeframe == DataTimeframe.ONE_MONTH:
                # Monthly data
                params["function"] = "TIME_SERIES_MONTHLY_ADJUSTED"
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    raise DataConnectionError(f"Alpha Vantage API error: {data['Error Message']}")
                    
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
                # Parse similar to daily data
                time_series_key = "Monthly Adjusted Time Series"
                if time_series_key not in data:
                    raise DataValidationError(f"Invalid data format: Missing {time_series_key}")
                    
                time_series = data[time_series_key]
                df = pd.DataFrame.from_dict(time_series, orient="index")
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                    "7. dividend amount": "dividend"
                })
                
                # Convert types and prepare DataFrame
                for col in ["open", "high", "low", "close", "adjusted_close"]:
                    df[col] = pd.to_numeric(df[col])
                df["volume"] = pd.to_numeric(df["volume"], downcast="integer")
                
                df = df.reset_index()
                df = df.rename(columns={"index": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            
            # Filter by date range if specified
            if start_date is not None:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
                
            if end_date is not None:
                df = df[df["timestamp"] <= pd.Timestamp(end_date)]
            
            # Apply limit if specified
            if limit is not None and not df.empty:
                df = df.tail(limit)
            
            # Keep only OHLCV columns (and adjusted_close if available)
            columns_to_keep = ["timestamp", "open", "high", "low", "close", "volume"]
            if "adjusted_close" in df.columns:
                columns_to_keep.append("adjusted_close")
                
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            return df
            
        except (DataConnectionError, DataValidationError) as e:
            raise e
        except Exception as e:
            error_msg = f"Failed to fetch data from Alpha Vantage: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_symbols(self) -> List[str]:
        """Get available symbols from Alpha Vantage.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to Alpha Vantage fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Check rate limit
            self._check_rate_limit()
            
            # Alpha Vantage provides a list of supported symbols
            params = {
                "function": "LISTING_STATUS",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            
            # The response is a CSV file
            if response.status_code == 200:
                # Create a temporary file to store the CSV
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                # Read the CSV file
                try:
                    df = pd.read_csv(temp_file_path)
                    symbols = df["symbol"].tolist()
                    
                    # Clean up the temporary file
                    os.remove(temp_file_path)
                    
                    return symbols
                except Exception as e:
                    # Clean up the temporary file
                    os.remove(temp_file_path)
                    raise e
            else:
                # If we can't get the full list, return a small list of common symbols
                return [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
                    "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "DIS"
                ]
                
        except Exception as e:
            logger.warning(f"Failed to get symbols from Alpha Vantage: {str(e)}")
            # Return a small list of common symbols as fallback
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
                "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "DIS"
            ]

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from Alpha Vantage.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes