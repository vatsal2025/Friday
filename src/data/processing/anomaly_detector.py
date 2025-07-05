"""Anomaly detection module for the Friday AI Trading System.

This module provides classes for detecting anomalies in data, including outliers,
change points, and other anomalies.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from src.infrastructure.logging import get_logger
from src.data.processing.data_processor import DataProcessor, ProcessingStep

# Create logger
logger = get_logger(__name__)


class AnomalyType(Enum):
    """Enumeration of anomaly types."""
    OUTLIER = "outlier"
    CHANGE_POINT = "change_point"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    TREND_ANOMALY = "trend_anomaly"
    CUSTOM = "custom"


class AnomalyDetector(DataProcessor):
    """Class for detecting anomalies in data.
    
    This class provides methods for detecting anomalies in data, including outliers,
    change points, and other anomalies.
    """
    
    def __init__(self, config=None):
        """Initialize an anomaly detector.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)
        
        # Register default processing steps
        self.add_processing_step(ProcessingStep.ANOMALY_DETECTION, self.detect_anomalies)
        
        # Store anomaly detection parameters
        self.detection_params = {}
    
    def detect_anomalies(self, data: pd.DataFrame, columns: Optional[List[str]] = None, 
                        method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies in data.
        
        Args:
            data: The input data.
            columns: List of columns to check for anomalies. If None, all numeric columns are checked.
            method: The method to use for anomaly detection. Options: "zscore", "iqr", "mad".
            threshold: The threshold to use for anomaly detection.
            
        Returns:
            pd.DataFrame: A DataFrame with the same shape as the input, where True indicates an anomaly.
            
        Raises:
            ValueError: If an invalid method is specified.
        """
        # Make a copy of the data
        result = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Detect anomalies in each column
        for col in columns:
            if col in data.columns:
                if method == "zscore":
                    result[col] = self._detect_zscore_anomalies(data[col], threshold)
                elif method == "iqr":
                    result[col] = self._detect_iqr_anomalies(data[col], threshold)
                elif method == "mad":
                    result[col] = self._detect_mad_anomalies(data[col], threshold)
                else:
                    raise ValueError(f"Invalid method: {method}. Options: 'zscore', 'iqr', 'mad'.")
        
        return result
    
    def _detect_zscore_anomalies(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect anomalies using the Z-score method.
        
        Args:
            series: The input series.
            threshold: The threshold to use for anomaly detection.
            
        Returns:
            pd.Series: A series with the same shape as the input, where True indicates an anomaly.
        """
        # Calculate Z-scores
        mean = series.mean()
        std = series.std()
        
        # Avoid division by zero
        if std == 0:
            return pd.Series(False, index=series.index)
        
        z_scores = (series - mean) / std
        
        # Detect anomalies
        return z_scores.abs() > threshold
    
    def _detect_iqr_anomalies(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect anomalies using the IQR method.
        
        Args:
            series: The input series.
            threshold: The threshold to use for anomaly detection.
            
        Returns:
            pd.Series: A series with the same shape as the input, where True indicates an anomaly.
        """
        # Calculate IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # Avoid division by zero
        if iqr == 0:
            return pd.Series(False, index=series.index)
        
        # Detect anomalies
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_mad_anomalies(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect anomalies using the MAD method.
        
        Args:
            series: The input series.
            threshold: The threshold to use for anomaly detection.
            
        Returns:
            pd.Series: A series with the same shape as the input, where True indicates an anomaly.
        """
        # Calculate MAD
        median = series.median()
        mad = (series - median).abs().median()
        
        # Avoid division by zero
        if mad == 0:
            return pd.Series(False, index=series.index)
        
        # Detect anomalies
        return ((series - median).abs() / mad) > threshold
    
    def detect_change_points(self, series: pd.Series, window: int = 10, threshold: float = 3.0) -> pd.Series:
        """Detect change points in a time series.
        
        Args:
            series: The input time series.
            window: The window size to use for change point detection.
            threshold: The threshold to use for change point detection.
            
        Returns:
            pd.Series: A series with the same shape as the input, where True indicates a change point.
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Calculate Z-scores
        z_scores = (series - rolling_mean) / rolling_std
        
        # Detect change points
        return z_scores.abs() > threshold