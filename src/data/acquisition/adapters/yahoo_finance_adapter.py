"""Yahoo Finance adapter for the Friday AI Trading System.

This module provides an adapter that implements the DataSourceAdapter interface
for Yahoo Finance's API using the yfinance library.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time

import pandas as pd
import yfinance as yf

from src.data.acquisition.data_fetcher import (
    DataSourceAdapter,
    DataTimeframe,
    DataConnectionError,
    DataValidationError
)
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Mapping from DataTimeframe to yfinance interval
TIMEFRAME_MAP = {
    DataTimeframe.ONE_MINUTE: "1m",
    DataTimeframe.FIVE_MINUTES: "5m",
    DataTimeframe.FIFTEEN_MINUTES: "15m",
    DataTimeframe.THIRTY_MINUTES: "30m",
    DataTimeframe.ONE_HOUR: "1h",
    DataTimeframe.FOUR_HOURS: "4h",
    DataTimeframe.ONE_DAY: "1d",
    DataTimeframe.ONE_WEEK: "1wk",
    DataTimeframe.ONE_MONTH: "1mo"
}

# Yahoo Finance API limits
API_LIMITS = {
    "1m": {"period": "7d", "max_calls_per_hour": 2000},
    "5m": {"period": "60d", "max_calls_per_hour": 500},
    "15m": {"period": "60d", "max_calls_per_hour": 500},
    "30m": {"period": "60d", "max_calls_per_hour": 500},
    "1h": {"period": "730d", "max_calls_per_hour": 500},
    "4h": {"period": "730d", "max_calls_per_hour": 500},
    "1d": {"period": "max", "max_calls_per_hour": 500},
    "1wk": {"period": "max", "max_calls_per_hour": 500},
    "1mo": {"period": "max", "max_calls_per_hour": 500}
}


class YahooFinanceAdapter(DataSourceAdapter):
    """Adapter for Yahoo Finance that implements the DataSourceAdapter interface.

    This class adapts the yfinance library to the DataSourceAdapter interface,
    allowing the system to fetch data from Yahoo Finance through the common data API.
    
    Attributes:
        connected: Whether the adapter is connected to Yahoo Finance.
        rate_limit_tracker: Dictionary tracking API calls for rate limiting.
        available_symbols: Cache of available symbols.
        available_timeframes: List of available timeframes.
    """

    def __init__(self):
        """Initialize the Yahoo Finance adapter."""
        self.connected = False
        self.rate_limit_tracker = {}
        self.available_symbols = None
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
        
        # Initialize rate limit tracker
        for interval, limits in API_LIMITS.items():
            self.rate_limit_tracker[interval] = {
                "calls": 0,
                "reset_time": datetime.now() + timedelta(hours=1)
            }

    def connect(self) -> bool:
        """Connect to Yahoo Finance.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        try:
            # Test connection by fetching a small amount of data
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            
            if test_data.empty:
                logger.warning("Failed to connect to Yahoo Finance: No data returned")
                self.connected = False
                return False
                
            self.connected = True
            logger.info("Connected to Yahoo Finance successfully")
            return True
        except Exception as e:
            self.connected = False
            error_msg = f"Failed to connect to Yahoo Finance: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from Yahoo Finance.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        # No actual disconnection needed for yfinance
        self.connected = False
        logger.info("Disconnected from Yahoo Finance")
        return True

    def is_connected(self) -> bool:
        """Check if connected to Yahoo Finance.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connected

    def _check_rate_limit(self, interval: str) -> None:
        """Check and handle rate limiting for Yahoo Finance API.

        Args:
            interval: The data interval being requested.

        Raises:
            DataConnectionError: If rate limit is exceeded.
        """
        now = datetime.now()
        tracker = self.rate_limit_tracker.get(interval, {
            "calls": 0,
            "reset_time": now + timedelta(hours=1)
        })
        
        # Reset counter if the hour has passed
        if now >= tracker["reset_time"]:
            tracker["calls"] = 0
            tracker["reset_time"] = now + timedelta(hours=1)
        
        # Check if we've exceeded the rate limit
        max_calls = API_LIMITS.get(interval, {}).get("max_calls_per_hour", 500)
        if tracker["calls"] >= max_calls:
            wait_time = (tracker["reset_time"] - now).total_seconds()
            error_msg = f"Yahoo Finance rate limit exceeded for {interval}. Try again in {wait_time:.0f} seconds."
            logger.warning(error_msg)
            raise DataConnectionError(error_msg)
        
        # Increment the call counter
        tracker["calls"] += 1
        self.rate_limit_tracker[interval] = tracker

    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to Yahoo Finance fails.
            DataValidationError: If the fetched data is invalid.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        try:
            # Map timeframe to yfinance interval
            if timeframe not in TIMEFRAME_MAP:
                raise DataValidationError(f"Unsupported timeframe: {timeframe.value}")
            
            interval = TIMEFRAME_MAP[timeframe]
            
            # Check rate limit
            self._check_rate_limit(interval)
            
            # Determine period or date range
            period = None
            if start_date is None and end_date is None:
                # Use period if no dates specified
                period = API_LIMITS.get(interval, {}).get("period", "1mo")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            data = ticker.history(
                period=period,
                interval=interval,
                start=start_date,
                end=end_date,
                auto_adjust=True
            )
            
            # Apply limit if specified
            if limit is not None and not data.empty:
                data = data.tail(limit)
            
            # Rename columns to standard format
            if not data.empty:
                data = data.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                
                # Reset index to make Date a column
                data = data.reset_index()
                data = data.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
                
                # Keep only OHLCV columns
                columns_to_keep = ["timestamp", "open", "high", "low", "close", "volume"]
                data = data[[col for col in columns_to_keep if col in data.columns]]
            
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data from Yahoo Finance: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def get_symbols(self) -> List[str]:
        """Get available symbols from Yahoo Finance.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to Yahoo Finance fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e

        # Yahoo Finance doesn't provide a direct way to get all symbols
        # This would typically return a predefined list or fetch from a source
        # For now, we'll return a small list of common symbols
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
            "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "DIS"
        ]

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from Yahoo Finance.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes