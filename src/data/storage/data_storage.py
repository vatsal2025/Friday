"""Base data storage module for the Friday AI Trading System.

This module provides the base DataStorage class that defines the interface
for all storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class StorageError(Exception):
    """Exception raised for errors in the storage module."""
    pass


class DataStorage(ABC):
    """Abstract base class for data storage.

    This class defines the interface for all storage implementations.
    Concrete implementations must inherit from this class and implement
    the abstract methods.

    Attributes:
        config: Configuration manager.
        metadata: Dictionary for storing metadata about the storage operations.
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize a data storage instance.

        Args:
            config: Configuration manager. If None, a new one will be created.
        """
        self.config = config or ConfigManager()
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def connect(self) -> None:
        """Connect to the storage backend.

        Raises:
            StorageError: If connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the storage backend.

        Raises:
            StorageError: If disconnection fails.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the storage backend.

        Returns:
            bool: True if connected, False otherwise.
        """
        pass

    @abstractmethod
    def store_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = True,
        **kwargs
    ) -> bool:
        """Store data in the storage backend.

        Args:
            data: The data to store.
            table_name: The name of the table or collection to store the data in.
            if_exists: What to do if the table exists ('fail', 'replace', or 'append').
            index: Whether to store the index.
            **kwargs: Additional keyword arguments for the specific storage backend.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If storing fails.
        """
        pass

    @abstractmethod
    def retrieve_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        condition: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data from the storage backend.

        Args:
            table_name: The name of the table or collection to retrieve data from.
            columns: List of columns to retrieve. If None, all columns are retrieved.
            condition: Condition for filtering the data (e.g., "date > '2021-01-01'").
            limit: Maximum number of rows to retrieve.
            **kwargs: Additional keyword arguments for the specific storage backend.

        Returns:
            pd.DataFrame: The retrieved data.

        Raises:
            StorageError: If retrieval fails.
        """
        pass

    @abstractmethod
    def delete_data(
        self,
        table_name: str,
        condition: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete data from the storage backend.

        Args:
            table_name: The name of the table or collection to delete data from.
            condition: Condition for filtering the data to delete.
            **kwargs: Additional keyword arguments for the specific storage backend.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If deletion fails.
        """
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if a table or collection exists.

        Args:
            table_name: The name of the table or collection to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        """List all tables or collections in the storage backend.

        Returns:
            List[str]: List of table or collection names.

        Raises:
            StorageError: If listing fails.
        """
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table or collection.

        Args:
            table_name: The name of the table or collection.

        Returns:
            Dict[str, Any]: Dictionary with table information.

        Raises:
            StorageError: If getting information fails.
        """
        pass

    @abstractmethod
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute a custom query on the storage backend.

        Args:
            query: The query to execute.
            params: Parameters for the query.
            **kwargs: Additional keyword arguments for the specific storage backend.

        Returns:
            Any: The result of the query.

        Raises:
            StorageError: If query execution fails.
        """
        pass

    def _handle_error(self, operation: str, error: Exception) -> None:
        """Handle an error by logging it and raising a StorageError.

        Args:
            operation: The operation that failed.
            error: The exception that was raised.

        Raises:
            StorageError: Always raised with details about the original error.
        """
        error_msg = f"Error during {operation}: {str(error)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise StorageError(error_msg) from error

    def _prepare_data_for_storage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for storage by handling non-serializable types.

        Args:
            data: The data to prepare.

        Returns:
            pd.DataFrame: The prepared data.
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()

        # Convert numpy and pandas types to Python native types
        for col in result.columns:
            # Handle numpy numeric types
            if np.issubdtype(result[col].dtype, np.number):
                result[col] = result[col].astype(float)
            
            # Handle datetime types
            elif np.issubdtype(result[col].dtype, np.datetime64):
                result[col] = result[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle complex objects by converting to JSON strings
            elif result[col].dtype == 'object':
                result[col] = result[col].apply(
                    lambda x: json.dumps(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
                )

        return result

    def _record_operation_metadata(self, operation: str, **kwargs) -> None:
        """Record metadata about a storage operation.

        Args:
            operation: The operation being performed.
            **kwargs: Additional metadata to record.
        """
        timestamp = datetime.now().isoformat()
        
        if "operations" not in self.metadata:
            self.metadata["operations"] = []
            
        operation_metadata = {
            "operation": operation,
            "timestamp": timestamp,
            **kwargs
        }
        
        self.metadata["operations"].append(operation_metadata)