"""Market data provider module for the Friday AI Trading System.

This module provides a class for retrieving market data from various sources.
"""

from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data.providers.data_provider import DataProvider, ProviderError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class MarketDataProvider(DataProvider):
    """Class for retrieving market data from various sources.
    
    This class provides methods for retrieving market data such as
    price data, volume data, and other market indicators.
    """
    
    def __init__(self, config=None):
        """Initialize a market data provider.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)
        
        # Initialize API clients or connections here
        self.api_clients = {}
        
        # Get API keys from config if available
        if self.config is not None:
            self.api_keys = self.config.get("api_keys", {})
        else:
            self.api_keys = {}
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime = None,
               interval: str = "1d", source: str = "default", **kwargs) -> pd.DataFrame:
        """Get market data for a symbol.
        
        Args:
            symbol: The symbol to get data for.
            start_date: The start date for the data.
            end_date: The end date for the data. If None, the current date is used.
            interval: The interval for the data (e.g., "1m", "5m", "1h", "1d").
            source: The data source to use.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the market data.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        try:
            # Use current date if end_date is None
            if end_date is None:
                end_date = datetime.now()
            
            # Get data from the specified source
            if source == "default":
                # Use the default data source
                data = self._get_data_from_default_source(symbol, start_date, end_date, interval, **kwargs)
            else:
                # Use the specified data source
                data = self._get_data_from_source(source, symbol, start_date, end_date, interval, **kwargs)
            
            logger.info(f"Retrieved market data for {symbol} from {start_date} to {end_date} with interval {interval}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            raise ProviderError(f"Error retrieving market data: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate market data.
        
        Args:
            data: The market data to validate.
            
        Returns:
            True if the data is valid, False otherwise.
        """
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning("Market data is not a pandas DataFrame")
            return False
        
        # Check if data is empty
        if data.empty:
            logger.warning("Market data is empty")
            return False
        
        # Check if required columns are present
        required_columns = ["open", "high", "low", "close"]
        for column in required_columns:
            if column not in data.columns:
                logger.warning(f"Market data is missing required column: {column}")
                return False
        
        # Check for missing values in required columns
        for column in required_columns:
            if data[column].isnull().any():
                logger.warning(f"Market data has missing values in column: {column}")
                return False
        
        return True
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data.
        
        Args:
            data: The market data to process.
            
        Returns:
            The processed market data.
        """
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "date" in data.columns:
                data.set_index("date", inplace=True)
            elif "datetime" in data.columns:
                data.set_index("datetime", inplace=True)
            elif "timestamp" in data.columns:
                data.set_index("timestamp", inplace=True)
        
        # Sort by index
        data.sort_index(inplace=True)
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep="first")]
        
        return data
    
    def get_latest_price(self, symbol: str, source: str = "default") -> float:
        """Get the latest price for a symbol.
        
        Args:
            symbol: The symbol to get the latest price for.
            source: The data source to use.
            
        Returns:
            The latest price for the symbol.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        try:
            # Get data for the last day
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            data = self.get_data(symbol, start_date, end_date, interval="1m", source=source)
            
            # Get the latest price
            latest_price = data["close"].iloc[-1]
            
            logger.info(f"Retrieved latest price for {symbol}: {latest_price}")
            
            return latest_price
        
        except Exception as e:
            logger.error(f"Error retrieving latest price: {str(e)}")
            raise ProviderError(f"Error retrieving latest price: {str(e)}")
    
    def get_historical_prices(self, symbol: str, lookback_days: int = 30,
                            interval: str = "1d", source: str = "default") -> pd.DataFrame:
        """Get historical prices for a symbol.
        
        Args:
            symbol: The symbol to get historical prices for.
            lookback_days: The number of days to look back.
            interval: The interval for the data (e.g., "1m", "5m", "1h", "1d").
            source: The data source to use.
            
        Returns:
            A pandas DataFrame containing the historical prices.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        try:
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get historical data
            data = self.get_data(symbol, start_date, end_date, interval=interval, source=source)
            
            logger.info(f"Retrieved historical prices for {symbol} for the last {lookback_days} days")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving historical prices: {str(e)}")
            raise ProviderError(f"Error retrieving historical prices: {str(e)}")
    
    def _get_data_from_default_source(self, symbol: str, start_date: datetime,
                                    end_date: datetime, interval: str, **kwargs) -> pd.DataFrame:
        """Get data from the default data source.
        
        This method should be implemented by subclasses to retrieve data
        from their default data source.
        
        Args:
            symbol: The symbol to get data for.
            start_date: The start date for the data.
            end_date: The end date for the data.
            interval: The interval for the data.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the market data.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("_get_data_from_default_source must be implemented by subclasses")
    
    def _get_data_from_source(self, source: str, symbol: str, start_date: datetime,
                           end_date: datetime, interval: str, **kwargs) -> pd.DataFrame:
        """Get data from a specific data source.
        
        This method should be implemented by subclasses to retrieve data
        from a specific data source.
        
        Args:
            source: The data source to use.
            symbol: The symbol to get data for.
            start_date: The start date for the data.
            end_date: The end date for the data.
            interval: The interval for the data.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the market data.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("_get_data_from_source must be implemented by subclasses")
    
    def close(self):
        """Close the market data provider.
        
        This method closes any open connections or API clients.
        """
        # Close API clients or connections here
        for client_name, client in self.api_clients.items():
            try:
                if hasattr(client, "close"):
                    client.close()
                    logger.debug(f"Closed API client: {client_name}")
            except Exception as e:
                logger.warning(f"Error closing API client {client_name}: {str(e)}")
        
        super().close()