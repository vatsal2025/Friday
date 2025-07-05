"""Base data provider module for the Friday AI Trading System.

This module provides a base class for data providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class ProviderError(Exception):
    """Exception raised for errors in data providers."""
    pass


class DataProvider(ABC):
    """Abstract base class for data providers.
    
    This class defines the interface for data providers, which are responsible
    for retrieving data from various sources.
    """
    
    def __init__(self, config=None):
        """Initialize a data provider.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def get_data(self, *args, **kwargs) -> Any:
        """Get data from the provider.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The retrieved data.
            
        Raises:
            ProviderError: If an error occurs during data retrieval.
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate data from the provider.
        
        Args:
            data: The data to validate.
            
        Returns:
            True if the data is valid, False otherwise.
            
        Raises:
            ProviderError: If an error occurs during validation.
        """
        pass
    
    def process_data(self, data: Any) -> Any:
        """Process data from the provider.
        
        This method can be overridden by subclasses to perform additional
        processing on the data before returning it.
        
        Args:
            data: The data to process.
            
        Returns:
            The processed data.
            
        Raises:
            ProviderError: If an error occurs during processing.
        """
        return data
    
    def get_and_process_data(self, *args, **kwargs) -> Any:
        """Get and process data from the provider.
        
        This method combines the get_data and process_data methods.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The processed data.
            
        Raises:
            ProviderError: If an error occurs during data retrieval or processing.
        """
        data = self.get_data(*args, **kwargs)
        
        if not self.validate_data(data):
            raise ProviderError("Invalid data received from provider")
        
        return self.process_data(data)
    
    def close(self):
        """Close the provider.
        
        This method can be overridden by subclasses to perform cleanup
        when the provider is no longer needed.
        """
        logger.debug(f"Closed {self.__class__.__name__}")
        pass