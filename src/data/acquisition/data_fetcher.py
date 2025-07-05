"""Base data fetcher module for the Friday AI Trading System.

This module provides the base DataFetcher class and related components.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


class DataConnectionError(Exception):
    """Exception raised for data connection errors."""

    pass


class DataSourceType(Enum):
    """Enum for data source types."""

    BROKER_API = "broker_api"
    MARKET_DATA_PROVIDER = "market_data_provider"
    CSV_FILE = "csv_file"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    CUSTOM = "custom"


class DataTimeframe(Enum):
    """Enum for data timeframes."""

    TICK = "tick"
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class DataSourceAdapter(ABC):
    """Interface for data source adapters.

    This class defines the interface that all data source adapters must implement.
    Data source adapters are responsible for connecting to specific data sources
    and fetching data in a standardized format.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the data source.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the data source.

        Returns:
            bool: True if connected, False otherwise.
        """
        pass

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from the data source.

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
        pass

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get available symbols from the data source.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        pass

    @abstractmethod
    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from the data source.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        pass


class DataFetcher(ABC):
    """Base class for data fetchers.

    This class provides the base functionality for fetching data from various sources.
    It handles connection management, data validation, and error handling.

    Attributes:
        source_type: The type of the data source.
        adapter: The data source adapter.
        connected: Whether the fetcher is connected to the data source.
    """

    def __init__(self, source_type: DataSourceType, adapter: DataSourceAdapter):
        """Initialize a data fetcher.

        Args:
            source_type: The type of the data source.
            adapter: The data source adapter.
        """
        self.source_type = source_type
        self.adapter = adapter
        self.connected = False

    def connect(self) -> bool:
        """Connect to the data source.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        try:
            self.connected = self.adapter.connect()
            if self.connected:
                logger.info(f"Connected to {self.source_type.value} data source")
            else:
                logger.warning(f"Failed to connect to {self.source_type.value} data source")
            return self.connected
        except Exception as e:
            logger.error(f"Error connecting to {self.source_type.value} data source: {str(e)}")
            self.connected = False
            raise DataConnectionError(f"Failed to connect to data source: {str(e)}") from e

    def disconnect(self) -> bool:
        """Disconnect from the data source.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        try:
            result = self.adapter.disconnect()
            if result:
                logger.info(f"Disconnected from {self.source_type.value} data source")
                self.connected = False
            else:
                logger.warning(f"Failed to disconnect from {self.source_type.value} data source")
            return result
        except Exception as e:
            logger.error(f"Error disconnecting from {self.source_type.value} data source: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to the data source.

        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            self.connected = self.adapter.is_connected()
            return self.connected
        except Exception as e:
            logger.error(f"Error checking connection to {self.source_type.value} data source: {str(e)}")
            self.connected = False
            return False

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data.

        Args:
            data: The data to validate.

        Returns:
            bool: True if data is valid, False otherwise.

        Raises:
            DataValidationError: If the data is invalid.
        """
        if data is None or data.empty:
            raise DataValidationError("Data is empty")

        # Check for required columns based on data type
        if self._is_ohlcv_data(data):
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")

        # Check for NaN values
        if data.isna().any().any():
            logger.warning("Data contains NaN values")

        return True

    def _is_ohlcv_data(self, data: pd.DataFrame) -> bool:
        """Check if data is OHLCV data.

        Args:
            data: The data to check.

        Returns:
            bool: True if data is OHLCV data, False otherwise.
        """
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        return all(col.lower() in [c.lower() for c in data.columns] for col in ohlcv_columns)

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from the data source.

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
        pass

    def get_symbols(self) -> List[str]:
        """Get available symbols from the data source.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            self.connect()

        try:
            symbols = self.adapter.get_symbols()
            logger.debug(f"Retrieved {len(symbols)} symbols from {self.source_type.value} data source")
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols from {self.source_type.value} data source: {str(e)}")
            raise DataConnectionError(f"Failed to get symbols: {str(e)}") from e

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from the data source.

        Returns:
            List[DataTimeframe]: List of available timeframes.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        if not self.is_connected():
            self.connect()

        try:
            timeframes = self.adapter.get_timeframes()
            logger.debug(f"Retrieved {len(timeframes)} timeframes from {self.source_type.value} data source")
            return timeframes
        except Exception as e:
            logger.error(f"Error getting timeframes from {self.source_type.value} data source: {str(e)}")
            raise DataConnectionError(f"Failed to get timeframes: {str(e)}") from e