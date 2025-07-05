"""Parquet storage module for the Friday AI Trading System.

This module provides a class for storing and retrieving data from Parquet files.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union

from src.data.storage.data_storage import DataStorage, StorageError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Try to import pandas and pyarrow, but don't fail if they're not available
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("Pandas or PyArrow package not available. ParquetStorage will not be functional.")


class ParquetStorage(DataStorage):
    """Class for storing and retrieving data from Parquet files.
    
    This class provides methods for storing and retrieving data from Parquet files,
    with support for various data formats and operations.
    """
    
    def __init__(self, config=None, base_dir=None):
        """Initialize a Parquet storage backend.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
            base_dir: Base directory for Parquet files. If None, the current directory will be used.
            
        Raises:
            StorageError: If the base directory does not exist and cannot be created,
                          or if pandas or pyarrow is not available.
        """
        super().__init__(config)
        
        if not PARQUET_AVAILABLE:
            raise StorageError("Pandas or PyArrow package not available. Please install them with 'pip install pandas pyarrow'.")
        
        try:
            # Get base directory from config if available
            if self.config is not None:
                parquet_config = self.config.get("parquet", {})
                base_dir = parquet_config.get("base_dir", base_dir)
            
            # Use current directory if base_dir is None
            self.base_dir = base_dir or os.getcwd()
            
            # Create base directory if it doesn't exist
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
                logger.info(f"Created base directory: {self.base_dir}")
            
            logger.info(f"Initialized Parquet storage with base directory: {self.base_dir}")
        
        except Exception as e:
            logger.error(f"Error initializing Parquet storage: {str(e)}")
            raise StorageError(f"Error initializing Parquet storage: {str(e)}")
    
    def _get_file_path(self, key: str) -> str:
        """Get the file path for a key.
        
        Args:
            key: The key to get the file path for.
            
        Returns:
            The file path for the key.
        """
        # Ensure key has .parquet extension
        if not key.endswith(".parquet"):
            key = f"{key}.parquet"
        
        return os.path.join(self.base_dir, key)
    
    def store(self, key: str, data: Any, **kwargs) -> bool:
        """Store data in a Parquet file.
        
        Args:
            key: The key (filename) to store the data under.
            data: The data to store. Must be a pandas DataFrame.
            **kwargs: Additional arguments to pass to the pandas to_parquet method.
            
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
            data.to_parquet(file_path, **kwargs)
            
            logger.debug(f"Stored data in Parquet file: {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing data in Parquet file: {str(e)}")
            raise StorageError(f"Error storing data in Parquet file: {str(e)}")
    
    def retrieve(self, key: str, columns=None, filters=None, **kwargs) -> pd.DataFrame:
        """Retrieve data from a Parquet file.
        
        Args:
            key: The key (filename) to retrieve the data from.
            columns: List of columns to load. If None, all columns are loaded.
            filters: Filters to apply when loading the data.
            **kwargs: Additional arguments to pass to the pandas read_parquet method.
            
        Returns:
            The retrieved data as a pandas DataFrame.
            
        Raises:
            StorageError: If an error occurs during retrieval or the file does not exist.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Parquet file does not exist: {file_path}")
                return None
            
            # Retrieve data
            data = pd.read_parquet(file_path, columns=columns, filters=filters, **kwargs)
            
            logger.debug(f"Retrieved data from Parquet file: {file_path}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving data from Parquet file: {str(e)}")
            raise StorageError(f"Error retrieving data from Parquet file: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """Delete a Parquet file.
        
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
                logger.warning(f"Parquet file does not exist: {file_path}")
                return False
            
            # Delete file
            os.remove(file_path)
            
            logger.debug(f"Deleted Parquet file: {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting Parquet file: {str(e)}")
            raise StorageError(f"Error deleting Parquet file: {str(e)}")
    
    def exists(self, key: str) -> bool:
        """Check if a Parquet file exists.
        
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
            logger.error(f"Error checking if Parquet file exists: {str(e)}")
            raise StorageError(f"Error checking if Parquet file exists: {str(e)}")
    
    def list_files(self, pattern: str = "*.parquet") -> List[str]:
        """List Parquet files in the base directory.
        
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
            
            logger.debug(f"Listed {len(file_names)} Parquet files")
            
            return file_names
        
        except Exception as e:
            logger.error(f"Error listing Parquet files: {str(e)}")
            raise StorageError(f"Error listing Parquet files: {str(e)}")
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a Parquet file.
        
        Args:
            key: The key (filename) to get metadata for.
            
        Returns:
            A dictionary containing metadata for the file.
            
        Raises:
            StorageError: If an error occurs during metadata retrieval or the file does not exist.
        """
        try:
            file_path = self._get_file_path(key)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Parquet file does not exist: {file_path}")
                return None
            
            # Get metadata
            parquet_file = pq.ParquetFile(file_path)
            metadata = {
                "num_rows": parquet_file.metadata.num_rows,
                "num_columns": len(parquet_file.schema.names),
                "columns": parquet_file.schema.names,
                "created_by": parquet_file.metadata.created_by,
                "num_row_groups": parquet_file.metadata.num_row_groups,
            }
            
            logger.debug(f"Retrieved metadata for Parquet file: {file_path}")
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error retrieving metadata for Parquet file: {str(e)}")
            raise StorageError(f"Error retrieving metadata for Parquet file: {str(e)}")
    
    def append(self, key: str, data: pd.DataFrame) -> bool:
        """Append data to a Parquet file.
        
        Args:
            key: The key (filename) to append the data to.
            data: The data to append. Must be a pandas DataFrame.
            
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
            
            # If file doesn't exist, just store the data
            if not os.path.exists(file_path):
                return self.store(key, data)
            
            # Read existing data
            existing_data = self.retrieve(key)
            
            # Append new data
            combined_data = pd.concat([existing_data, data], ignore_index=True)
            
            # Store combined data
            return self.store(key, combined_data)
        
        except Exception as e:
            logger.error(f"Error appending data to Parquet file: {str(e)}")
            raise StorageError(f"Error appending data to Parquet file: {str(e)}")
    
    def close(self):
        """Close the Parquet storage.
        
        This method is a no-op for Parquet storage, as there is no connection to close.
        """
        logger.debug("Parquet storage does not require closing")
        pass
