"""Storage module for the Friday AI Trading System.

This module provides classes for storing and retrieving data from various storage backends.
"""

from src.data.storage.data_storage import DataStorage, StorageError
from src.data.storage.sql_storage import SQLStorage
from src.data.storage.mongodb_storage import MongoDBStorage
from src.data.storage.redis_storage import RedisStorage
from src.data.storage.csv_storage import CSVStorage
from src.data.storage.parquet_storage import ParquetStorage
from src.data.storage.local_parquet_storage import LocalParquetStorage
from src.data.storage.storage_factory import (
    DataStorageFactory, 
    get_storage_factory, 
    get_default_storage, 
    get_training_storage,
    create_storage
)
from src.data.storage.retrieval_utils import (
    DataRetrievalUtils,
    DataRetrievalError,
    get_training_data,
    get_feature_matrix
)

__all__ = [
    "DataStorage",
    "StorageError",
    "SQLStorage",
    "MongoDBStorage",
    "RedisStorage",
    "CSVStorage",
    "ParquetStorage",
    "LocalParquetStorage",
    "DataStorageFactory",
    "get_storage_factory",
    "get_default_storage",
    "get_training_storage",
    "create_storage",
    "DataRetrievalUtils",
    "DataRetrievalError",
    "get_training_data",
    "get_feature_matrix",
]
