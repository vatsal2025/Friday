"""Data module for the Friday AI Trading System.

This module provides components for data acquisition, processing, and storage.
"""

from src.data.acquisition import (
    DataFetcher,
    HistoricalDataFetcher,
    RealTimeDataStream,
    MarketCalendar,
    DataSourceAdapter,
    DataValidationError,
    DataConnectionError,
)

from src.data.processing import (
    DataProcessor,
    DataCleaner,
    DataTransformer,
    TimeframeManager,
    FeatureEngineer,
    DataValidator,
    AnomalyDetector,
)

from src.data.storage import (
    DataStorage,
    SQLStorage,
    MongoDBStorage,
    RedisStorage,
    CSVStorage,
    ParquetStorage,
)

__all__ = [
    # Acquisition
    "DataFetcher",
    "HistoricalDataFetcher",
    "RealTimeDataStream",
    "MarketCalendar",
    "DataSourceAdapter",
    "DataValidationError",
    "DataConnectionError",
    
    # Processing
    "DataProcessor",
    "DataCleaner",
    "DataTransformer",
    "TimeframeManager",
    "FeatureEngineer",
    "DataValidator",
    "AnomalyDetector",
    
    # Storage
    "DataStorage",
    "SQLStorage",
    "MongoDBStorage",
    "RedisStorage",
    "CSVStorage",
    "ParquetStorage",
]