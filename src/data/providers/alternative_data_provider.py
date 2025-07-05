"""Alternative data provider module for the Friday AI Trading System.

This module provides a class for retrieving alternative data from various sources.
"""

from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data.providers.data_provider import DataProvider, ProviderError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class AlternativeDataProvider(DataProvider):
    """Class for retrieving alternative data from various sources.
    
    This class provides methods for retrieving alternative data such as
    sentiment data, social media data, and other non-traditional market indicators.
    """
    
    def __init__(self, config=None):
        """Initialize an alternative data provider.
        
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
    
    def get_data(self, data_type: str, start_date: datetime, end_date: datetime = None,
               source: str = "default", **kwargs) -> pd.DataFrame:
        """Get alternative data.
        
        Args:
            data_type: The type of alternative data to retrieve.
            start_date: The start date for the data.
            end_date: The end date for the data. If None, the current date is used.
            source: The data source to use.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the alternative data.
            
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
                data = self._get_data_from_default_source(data_type, start_date, end_date, **kwargs)
            else:
                # Use the specified data source
                data = self._get_data_from_source(source, data_type, start_date, end_date, **kwargs)
            
            logger.info(f"Retrieved {data_type} data from {start_date} to {end_date}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving alternative data: {str(e)}")
            raise ProviderError(f"Error retrieving alternative data: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate alternative data.
        
        Args:
            data: The alternative data to validate.
            
        Returns:
            True if the data is valid, False otherwise.
        """
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning("Alternative data is not a pandas DataFrame")
            return False
        
        # Check if data is empty
        if data.empty:
            logger.warning("Alternative data is empty")
            return False
        
        # Check if required columns are present (depends on data type)
        # This should be implemented by subclasses for specific data types
        
        return True
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process alternative data.
        
        Args:
            data: The alternative data to process.
            
        Returns:
            The processed alternative data.
        """
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure datetime index if possible
        if not isinstance(data.index, pd.DatetimeIndex):
            if "date" in data.columns:
                data.set_index("date", inplace=True)
            elif "datetime" in data.columns:
                data.set_index("datetime", inplace=True)
            elif "timestamp" in data.columns:
                data.set_index("timestamp", inplace=True)
        
        # Sort by index if it's a DatetimeIndex
        if isinstance(data.index, pd.DatetimeIndex):
            data.sort_index(inplace=True)
            
            # Remove duplicates
            data = data[~data.index.duplicated(keep="first")]
        
        return data
    
    def get_sentiment_data(self, symbol: str, start_date: datetime, end_date: datetime = None,
                         source: str = "default") -> pd.DataFrame:
        """Get sentiment data for a symbol.
        
        Args:
            symbol: The symbol to get sentiment data for.
            start_date: The start date for the data.
            end_date: The end date for the data. If None, the current date is used.
            source: The data source to use.
            
        Returns:
            A pandas DataFrame containing the sentiment data.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        try:
            # Get sentiment data
            data = self.get_data("sentiment", start_date, end_date, source=source, symbol=symbol)
            
            logger.info(f"Retrieved sentiment data for {symbol} from {start_date} to {end_date}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            raise ProviderError(f"Error retrieving sentiment data: {str(e)}")
    
    def get_social_media_data(self, symbol: str, start_date: datetime, end_date: datetime = None,
                            source: str = "default", platform: str = "all") -> pd.DataFrame:
        """Get social media data for a symbol.
        
        Args:
            symbol: The symbol to get social media data for.
            start_date: The start date for the data.
            end_date: The end date for the data. If None, the current date is used.
            source: The data source to use.
            platform: The social media platform to get data from (e.g., "twitter", "reddit").
                      If "all", data from all available platforms is retrieved.
            
        Returns:
            A pandas DataFrame containing the social media data.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        try:
            # Get social media data
            data = self.get_data("social_media", start_date, end_date, source=source,
                               symbol=symbol, platform=platform)
            
            logger.info(f"Retrieved social media data for {symbol} from {start_date} to {end_date}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving social media data: {str(e)}")
            raise ProviderError(f"Error retrieving social media data: {str(e)}")
    
    def _get_data_from_default_source(self, data_type: str, start_date: datetime,
                                    end_date: datetime, **kwargs) -> pd.DataFrame:
        """Get data from the default data source.
        
        This method should be implemented by subclasses to retrieve data
        from their default data source.
        
        Args:
            data_type: The type of alternative data to retrieve.
            start_date: The start date for the data.
            end_date: The end date for the data.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the alternative data.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("_get_data_from_default_source must be implemented by subclasses")
    
    def _get_data_from_source(self, source: str, data_type: str, start_date: datetime,
                           end_date: datetime, **kwargs) -> pd.DataFrame:
        """Get data from a specific data source.
        
        This method should be implemented by subclasses to retrieve data
        from a specific data source.
        
        Args:
            source: The data source to use.
            data_type: The type of alternative data to retrieve.
            start_date: The start date for the data.
            end_date: The end date for the data.
            **kwargs: Additional arguments to pass to the data source.
            
        Returns:
            A pandas DataFrame containing the alternative data.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("_get_data_from_source must be implemented by subclasses")
    
    def close(self):
        """Close the alternative data provider.
        
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