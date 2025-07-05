"""Data processing module for the Friday AI Trading System.

This module provides components for data processing, cleaning, validation,
and transformation of market data.
"""

from src.data.processing.data_processor import (
    DataProcessor,
    ProcessingStep,
    DataValidationError,
    DataProcessingError,
)
from src.data.processing.data_cleaner import (
    DataCleaner,
    CleaningStrategy,
    OutlierDetectionMethod,
)
from src.data.processing.feature_engineering import (
    FeatureEngineer,
    FeatureSet,
    FeatureCategory,
)
from src.data.processing.multi_timeframe import (
    MultiTimeframeProcessor,
    TimeframeConverter,
    TimeframeAlignment,
)

from src.data.data_transformer import (
    DataTransformer,
    TransformationType,
)

from src.data.processing.timeframe_manager import TimeframeManager
from src.data.processing.data_validator import DataValidator, ValidationRule
from src.data.processing.anomaly_detector import AnomalyDetector, AnomalyType

__all__ = [
    "DataProcessor",
    "ProcessingStep",
    "DataValidationError",
    "DataProcessingError",
    "DataCleaner",
    "CleaningStrategy",
    "OutlierDetectionMethod",
    "FeatureEngineer",
    "FeatureSet",
    "FeatureCategory",
    "MultiTimeframeProcessor",
    "TimeframeConverter",
    "TimeframeAlignment",
    "DataTransformer",
    "TransformationType",
    "TimeframeManager",
    "DataValidator",
    "ValidationRule",
    "AnomalyDetector",
    "AnomalyType",
]