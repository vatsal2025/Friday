"""CSV storage module for the Friday AI Trading System.

This module provides a class for storing and retrieving data from CSV files.
"""

import os
import csv
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from src.data.storage.data_storage import DataStorage, StorageError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class CSVStorage(DataStorage):
    """Class for storing and retrieving data from CSV files.
    
    This class provides methods for storing and retrieving data from CSV files,
    with support for various data formats and operations.
    """
    
    def __init__(self, config=None, base_dir=None):
        """Initialize a CSV storage backend.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
            base_dir: Base directory for CSV files. If None, the current directory will be used.
            
        Raises:
            StorageError: If the base directory does not exist and cannot be created.
        """
        super().__init__(config)
        
        try:
            # Get base directory from config if available
            if self.config is not None:
                csv_config = self.config.get("csv", {})
                base_dir = csv_config.get("base_dir", base_dir)
            
            # Use current directory if base_dir is None
            self.base_dir = base_dir or os.getcwd()
            
            # Create base directory if it doesn't exist
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
                logger.info(f"Created base directory: {self.base_dir}")
            
            logger.info(f"Initialized CSV storage with base directory: {self.base_dir}")
        
        except Exception as e:
            logger.error(f"Error initializing CSV storage: {str(e)}")
            raise StorageError(f"Error initializing CSV storage: {str(e)}")
    
    def _get_file_path(self, key: str) -> str:
        """Get the file path for a key.
        
        Args:
            key: The key to get the file path for.
            
        Returns:
            The file path for the key.
        """
        # Ensure key has .csv extension
        if not key.endswith(".csv"):
            key = f"{key}.csv"
        
        return os.path.join(self.base_dir, key)
    
    def store(self, key: str, data: Any, **kwargs) -> bool:
        """Store data in a CSV file.
        
        Args:
            key: The key (filename) to store the data under.
            data: The data to store. Can be a pandas DataFrame, a list of dictionaries,
                  or a list of lists.
            **kwargs: Additional arguments to pass to the pandas to_csv method.
            
        Returns:
            True if the data was stored successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during storage.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Store data
            data.to_csv(file_path, **kwargs)
            
            logger.debug(f"Stored data in CSV file: {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing data in CSV file: {str(e)}")
            raise StorageError(f"Error storing data in CSV file: {str(e)}")
    
    def retrieve(self, key: str, **kwargs) -> pd.DataFrame:
        """Retrieve data from a CSV file.
        
        Args:
            key: The key (filename) to retrieve the data from.
            **kwargs: Additional arguments to pass to the pandas read_csv method.
            
        Returns:
            The retrieved data as a pandas DataFrame.
            
        Raises:
            StorageError: If an error occurs during retrieval or the file does not exist.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"CSV file does not exist: {file_path}")
                return None
            
            # Retrieve data
            data = pd.read_csv(file_path, **kwargs)
            
            logger.debug(f"Retrieved data from CSV file: {file_path}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving data from CSV file: {str(e)}")
            raise StorageError(f"Error retrieving data from CSV file: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """Delete a CSV file.
        
        Args:
            key: The key (filename) to delete.
            
        Returns:
            True if the file was deleted successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during deletion.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"CSV file does not exist: {file_path}")
                return False
            
            # Delete file
            os.remove(file_path)
            
            logger.debug(f"Deleted CSV file: {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting CSV file: {str(e)}")
            raise StorageError(f"Error deleting CSV file: {str(e)}")
    
    def exists(self, key: str) -> bool:
        """Check if a CSV file exists.
        
        Args:
            key: The key (filename) to check.
            
        Returns:
            True if the file exists, False otherwise.
            
        Raises:
            StorageError: If an error occurs during the check.
        """
        try:
            file_path = self._get_file_path(key)
            
            return os.path.exists(file_path)
        
        except Exception as e:
            logger.error(f"Error checking if CSV file exists: {str(e)}")
            raise StorageError(f"Error checking if CSV file exists: {str(e)}")
    
    def list_files(self, pattern: str = "*.csv") -> List[str]:
        """List CSV files in the base directory.
        
        Args:
            pattern: The pattern to match files against.
            
        Returns:
            A list of file names matching the pattern.
            
        Raises:
            StorageError: If an error occurs during listing.
        """
        try:
            import glob
            
            # Get list of files
            files = glob.glob(os.path.join(self.base_dir, pattern))
            
            # Extract file names without path
            file_names = [os.path.basename(file) for file in files]
            
            logger.debug(f"Listed {len(file_names)} CSV files")
            
            return file_names
        
        except Exception as e:
            logger.error(f"Error listing CSV files: {str(e)}")
            raise StorageError(f"Error listing CSV files: {str(e)}")
    
    def append(self, key: str, data: Any, **kwargs) -> bool:
        """Append data to a CSV file.
        
        Args:
            key: The key (filename) to append the data to.
            data: The data to append. Can be a pandas DataFrame, a list of dictionaries,
                  or a list of lists.
            **kwargs: Additional arguments to pass to the pandas to_csv method.
            
        Returns:
            True if the data was appended successfully, False otherwise.
            
        Raises:
            StorageError: If an error occurs during appending.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Set mode to append
            kwargs["mode"] = "a"
            
            # Don't write header if file exists
            if os.path.exists(file_path):
                kwargs["header"] = False
            
            # Append data
            data.to_csv(file_path, **kwargs)
            
            logger.debug(f"Appended data to CSV file: {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error appending data to CSV file: {str(e)}")
            raise StorageError(f"Error appending data to CSV file: {str(e)}")
    
    def close(self):
        """Close the CSV storage.
        
        This method is a no-op for CSV storage, as there is no connection to close.
        """
        logger.debug("CSV storage does not require closing")
        pass