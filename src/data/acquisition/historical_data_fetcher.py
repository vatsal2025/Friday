"""Historical data fetcher module for the Friday AI Trading System.

This module provides the HistoricalDataFetcher class for fetching historical market data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.data.acquisition.data_fetcher import (
    DataConnectionError,
    DataFetcher,
    DataSourceAdapter,
    DataSourceType,
    DataTimeframe,
    DataValidationError,
)
from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class HistoricalDataFetcher(DataFetcher):
    """Historical data fetcher for market data.

    This class provides functionality for fetching historical market data from various sources.
    It supports data caching, batch fetching, and data validation.

    Attributes:
        source_type: The type of the data source.
        adapter: The data source adapter.
        connected: Whether the fetcher is connected to the data source.
        cache_enabled: Whether data caching is enabled.
        cache: The data cache.
        config: The configuration manager.
    """

    def __init__(
        self,
        source_type: DataSourceType,
        adapter: DataSourceAdapter,
        cache_enabled: bool = True,
        config: Optional[ConfigManager] = None,
    ):
        """Initialize a historical data fetcher.

        Args:
            source_type: The type of the data source.
            adapter: The data source adapter.
            cache_enabled: Whether to enable data caching. Defaults to True.
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(source_type, adapter)
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, pd.DataFrame] = {}
        self.config = config or ConfigManager()
        
        # Load configuration
        self.batch_size = self.config.get("data.historical.batch_size", 1000)
        self.max_retries = self.config.get("data.historical.max_retries", 3)
        self.retry_delay = self.config.get("data.historical.retry_delay", 5)  # seconds
        self.show_progress = self.config.get("data.historical.show_progress", True)

    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical data from the data source.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.
            force_refresh: Whether to force refresh the cache. Defaults to False.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to the data source fails.
            DataValidationError: If the fetched data is invalid.
        """
        # Check if data is in cache and cache is enabled
        cache_key = f"{symbol}_{timeframe.value}_{start_date}_{end_date}_{limit}"
        if self.cache_enabled and not force_refresh and cache_key in self.cache:
            logger.debug(f"Using cached data for {symbol} {timeframe.value}")
            return self.cache[cache_key].copy()

        # Ensure connection
        if not self.is_connected():
            self.connect()

        # Set default end_date to now if not provided
        if end_date is None:
            end_date = datetime.now()

        # Set default start_date if not provided
        if start_date is None and limit is None:
            # Default to 1 year of data if neither start_date nor limit is provided
            start_date = end_date - timedelta(days=365)
        
        try:
            # For large date ranges, fetch data in batches to avoid timeouts or memory issues
            if self._should_use_batching(start_date, end_date, timeframe):
                data = self._fetch_data_in_batches(symbol, timeframe, start_date, end_date, limit)
            else:
                # Fetch data directly
                data = self._fetch_with_retry(symbol, timeframe, start_date, end_date, limit)
            
            # Validate the data
            self.validate_data(data)
            
            # Cache the data if caching is enabled
            if self.cache_enabled:
                self.cache[cache_key] = data.copy()
            
            return data
        
        except DataConnectionError as e:
            logger.error(f"Connection error while fetching data for {symbol}: {str(e)}")
            raise
        except DataValidationError as e:
            logger.error(f"Data validation error for {symbol}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
            raise DataConnectionError(f"Failed to fetch data: {str(e)}") from e

    def _should_use_batching(self, start_date: datetime, end_date: datetime, timeframe: DataTimeframe) -> bool:
        """Determine if batching should be used based on the date range and timeframe.

        Args:
            start_date: The start date.
            end_date: The end date.
            timeframe: The timeframe.

        Returns:
            bool: True if batching should be used, False otherwise.
        """
        if start_date is None or end_date is None:
            return False
            
        # Calculate the expected number of data points
        delta = end_date - start_date
        
        # Estimate number of data points based on timeframe
        if timeframe == DataTimeframe.ONE_MINUTE:
            # Trading hours per day (approx 8 hours) * 60 minutes
            points_per_day = 8 * 60
        elif timeframe == DataTimeframe.FIVE_MINUTES:
            points_per_day = 8 * 12
        elif timeframe == DataTimeframe.FIFTEEN_MINUTES:
            points_per_day = 8 * 4
        elif timeframe == DataTimeframe.THIRTY_MINUTES:
            points_per_day = 8 * 2
        elif timeframe == DataTimeframe.ONE_HOUR:
            points_per_day = 8
        elif timeframe == DataTimeframe.FOUR_HOURS:
            points_per_day = 2
        elif timeframe == DataTimeframe.ONE_DAY:
            points_per_day = 1
        elif timeframe == DataTimeframe.ONE_WEEK:
            points_per_day = 1/7
        elif timeframe == DataTimeframe.ONE_MONTH:
            points_per_day = 1/30
        else:  # TICK data or unknown
            # For tick data, always use batching
            return True
            
        # Estimate total points
        total_points = delta.days * points_per_day
        
        # Use batching if the estimated number of points exceeds the batch size
        return total_points > self.batch_size

    def _fetch_data_in_batches(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data in batches to handle large date ranges.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data.
            end_date: The end date for the data.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to the data source fails.
            DataValidationError: If the fetched data is invalid.
        """
        # Calculate batch periods based on timeframe
        batch_periods = self._calculate_batch_periods(timeframe, start_date, end_date)
        
        all_data = []
        
        # Create progress bar if enabled
        batch_iter = tqdm(batch_periods) if self.show_progress else batch_periods
        
        for batch_start, batch_end in batch_iter:
            if self.show_progress:
                batch_iter.set_description(f"Fetching {symbol} {timeframe.value}")
                
            try:
                batch_data = self._fetch_with_retry(symbol, timeframe, batch_start, batch_end, None)
                if not batch_data.empty:
                    all_data.append(batch_data)
            except DataConnectionError as e:
                logger.warning(f"Error fetching batch {batch_start} to {batch_end}: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        
        if not all_data:
            raise DataConnectionError(f"Failed to fetch any data for {symbol}")
        
        # Combine all batches
        combined_data = pd.concat(all_data)
        
        # Remove duplicates that might occur at batch boundaries
        combined_data = combined_data.drop_duplicates()
        
        # Sort by timestamp
        if 'timestamp' in combined_data.columns:
            combined_data = combined_data.sort_values('timestamp')
        
        # Apply limit if specified
        if limit is not None and len(combined_data) > limit:
            combined_data = combined_data.tail(limit)
        
        return combined_data

    def _calculate_batch_periods(
        self, timeframe: DataTimeframe, start_date: datetime, end_date: datetime
    ) -> List[tuple]:
        """Calculate batch periods based on timeframe.

        Args:
            timeframe: The timeframe of the data.
            start_date: The start date for the data.
            end_date: The end date for the data.

        Returns:
            List[tuple]: List of (batch_start, batch_end) tuples.
        """
        batch_periods = []
        current_start = start_date
        
        # Determine batch delta based on timeframe
        if timeframe == DataTimeframe.ONE_MINUTE:
            batch_delta = timedelta(days=1)  # 1 day batches for 1-minute data
        elif timeframe == DataTimeframe.FIVE_MINUTES:
            batch_delta = timedelta(days=5)  # 5 day batches for 5-minute data
        elif timeframe == DataTimeframe.FIFTEEN_MINUTES:
            batch_delta = timedelta(days=15)  # 15 day batches for 15-minute data
        elif timeframe == DataTimeframe.THIRTY_MINUTES:
            batch_delta = timedelta(days=30)  # 30 day batches for 30-minute data
        elif timeframe == DataTimeframe.ONE_HOUR:
            batch_delta = timedelta(days=60)  # 60 day batches for 1-hour data
        elif timeframe == DataTimeframe.FOUR_HOURS:
            batch_delta = timedelta(days=120)  # 120 day batches for 4-hour data
        elif timeframe == DataTimeframe.ONE_DAY:
            batch_delta = timedelta(days=365)  # 1 year batches for daily data
        else:  # Weekly, monthly, or tick data
            batch_delta = timedelta(days=365 * 2)  # 2 year batches for weekly/monthly data
        
        while current_start < end_date:
            batch_end = min(current_start + batch_delta, end_date)
            batch_periods.append((current_start, batch_end))
            current_start = batch_end
        
        return batch_periods

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Fetch data with retry logic.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data.
            end_date: The end date for the data.
            limit: The maximum number of data points to fetch.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If all retries fail.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                data = self.adapter.fetch_data(symbol, timeframe, start_date, end_date, limit)
                return data
            except Exception as e:
                last_exception = e
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    import time
                    time.sleep(self.retry_delay)
        
        # All retries failed
        raise DataConnectionError(f"Failed to fetch data after {self.max_retries} attempts: {str(last_exception)}") from last_exception

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: The symbols to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data.

        Raises:
            DataConnectionError: If connection to the data source fails.
        """
        result = {}
        errors = {}
        
        # Create progress bar if enabled
        symbol_iter = tqdm(symbols) if self.show_progress else symbols
        
        for symbol in symbol_iter:
            if self.show_progress:
                symbol_iter.set_description(f"Fetching {symbol}")
                
            try:
                data = self.fetch_data(symbol, timeframe, start_date, end_date, limit)
                result[symbol] = data
            except (DataConnectionError, DataValidationError) as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                errors[symbol] = str(e)
        
        if not result and errors:
            # All fetches failed
            error_msg = "; ".join([f"{s}: {e}" for s, e in errors.items()])
            raise DataConnectionError(f"Failed to fetch data for any symbol: {error_msg}")
        
        if errors:
            # Some fetches failed, log warning
            logger.warning(f"Failed to fetch data for {len(errors)} out of {len(symbols)} symbols")
        
        return result

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[DataTimeframe] = None) -> None:
        """Clear the data cache.

        Args:
            symbol: The symbol to clear cache for. If None, clear for all symbols.
            timeframe: The timeframe to clear cache for. If None, clear for all timeframes.
        """
        if not self.cache_enabled:
            return
            
        if symbol is None and timeframe is None:
            # Clear entire cache
            self.cache = {}
            logger.debug("Cleared entire data cache")
        else:
            # Clear specific entries
            keys_to_remove = []
            for key in self.cache.keys():
                parts = key.split('_')
                if len(parts) >= 2:
                    key_symbol = parts[0]
                    key_timeframe = parts[1]
                    
                    if (symbol is None or key_symbol == symbol) and \
                       (timeframe is None or key_timeframe == timeframe.value):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                
            logger.debug(f"Cleared {len(keys_to_remove)} cache entries")