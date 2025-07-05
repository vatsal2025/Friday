"""Data cleaner module for the Friday AI Trading System.

This module provides the DataCleaner class and related components for
cleaning market data.
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import traceback
import pytz

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.data.processing.data_processor import DataProcessor, ProcessingStep, DataProcessingError

# Create logger
logger = get_logger(__name__)


class CleaningStrategy(Enum):
    """Enum for data cleaning strategies."""

    DROP = auto()  # Drop rows or columns with issues
    FILL_MEAN = auto()  # Fill missing values with mean
    FILL_MEDIAN = auto()  # Fill missing values with median
    FILL_MODE = auto()  # Fill missing values with mode
    FILL_CONSTANT = auto()  # Fill missing values with a constant
    FILL_INTERPOLATE = auto()  # Fill missing values with interpolation
    FILL_FORWARD = auto()  # Fill missing values with forward fill (last observation carried forward)
    FILL_BACKWARD = auto()  # Fill missing values with backward fill (next observation carried backward)
    WINSORIZE = auto()  # Cap outliers at specified percentiles
    CLIP = auto()  # Clip values to specified min/max
    CUSTOM = auto()  # Custom cleaning strategy


class OutlierDetectionMethod(Enum):
    """Enum for outlier detection methods."""

    Z_SCORE = auto()  # Z-score method
    IQR = auto()  # Interquartile range method
    PERCENTILE = auto()  # Percentile method
    MAD = auto()  # Median absolute deviation method
    ISOLATION_FOREST = auto()  # Isolation Forest algorithm
    LOCAL_OUTLIER_FACTOR = auto()  # Local Outlier Factor algorithm
    DBSCAN = auto()  # DBSCAN clustering algorithm
    CUSTOM = auto()  # Custom outlier detection method


class DataCleaner(DataProcessor):
    """Class for cleaning market data.

    This class provides methods for handling missing values, outliers,
    duplicates, and other data cleaning operations.

    Attributes:
        config: Configuration manager.
        missing_value_strategies: Dictionary mapping column names to cleaning strategies.
        outlier_strategies: Dictionary mapping column names to outlier handling strategies.
        default_missing_strategy: Default strategy for handling missing values.
        default_outlier_strategy: Default strategy for handling outliers.
        outlier_detection_method: Method used for outlier detection.
        outlier_threshold: Threshold for outlier detection.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        default_missing_strategy: CleaningStrategy = CleaningStrategy.FILL_FORWARD,
        default_outlier_strategy: CleaningStrategy = CleaningStrategy.WINSORIZE,
        outlier_detection_method: OutlierDetectionMethod = OutlierDetectionMethod.IQR,
        outlier_threshold: float = 1.5,
    ):
        """Initialize a data cleaner.

        Args:
            config: Configuration manager. If None, a new one will be created.
            default_missing_strategy: Default strategy for handling missing values.
            default_outlier_strategy: Default strategy for handling outliers.
            outlier_detection_method: Method used for outlier detection.
            outlier_threshold: Threshold for outlier detection.
        """
        super().__init__(config)
        self.missing_value_strategies: Dict[str, CleaningStrategy] = {}
        self.outlier_strategies: Dict[str, CleaningStrategy] = {}
        self.outlier_detection_methods: Dict[str, OutlierDetectionMethod] = {}
        self.outlier_thresholds: Dict[str, float] = {}
        self.missing_fill_values: Dict[str, Any] = {}  # For FILL_CONSTANT strategy
        self.default_missing_strategy = default_missing_strategy
        self.default_outlier_strategy = default_outlier_strategy
        self.outlier_detection_method = outlier_detection_method
        self.outlier_threshold = outlier_threshold
        self.z_score_threshold = 5.0  # Threshold for extreme outlier z-score replacement

        # Register default cleaning steps
        self.add_processing_step(ProcessingStep.CLEANING, self.handle_missing_values)
        self.add_processing_step(ProcessingStep.CLEANING, self.handle_outliers)
        self.add_processing_step(ProcessingStep.CLEANING, self.handle_duplicates)

    def set_missing_value_strategy(self, column: str, strategy: CleaningStrategy) -> None:
        """Set the missing value handling strategy for a specific column.

        Args:
            column: The column name.
            strategy: The cleaning strategy to use.
        """
        self.missing_value_strategies[column] = strategy

    def set_outlier_strategy(self, column: str, strategy: CleaningStrategy) -> None:
        """Set the outlier handling strategy for a specific column.

        Args:
            column: The column name.
            strategy: The cleaning strategy to use.
        """
        self.outlier_strategies[column] = strategy
    
    def set_outlier_detection_method(self, column: str, method: OutlierDetectionMethod) -> None:
        """Set the outlier detection method for a specific column.

        Args:
            column: The column name.
            method: The outlier detection method to use.
        """
        self.outlier_detection_methods[column] = method
    
    def set_outlier_threshold(self, column: str, threshold: float) -> None:
        """Set the outlier threshold for a specific column.

        Args:
            column: The column name.
            threshold: The threshold value for outlier detection.
        """
        self.outlier_thresholds[column] = threshold
    
    def set_missing_fill_value(self, column: str, value: Any) -> None:
        """Set the fill value for FILL_CONSTANT strategy for a specific column.

        Args:
            column: The column name.
            value: The value to use for filling missing values.
        """
        self.missing_fill_values[column] = value

    def handle_missing_values(
        self, data: pd.DataFrame, fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """Handle missing values in the data.

        Args:
            data: The data to clean.
            fill_value: Value to use for constant fill strategy.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Make a copy of the input data
            cleaned_data = data.copy()

            # Get columns with missing values
            columns_with_missing = cleaned_data.columns[cleaned_data.isna().any()].tolist()

            # Record missing value statistics before cleaning
            missing_stats_before = cleaned_data.isna().sum().to_dict()
            self.metadata["missing_values_before"] = missing_stats_before

            # Process each column with missing values
            for column in columns_with_missing:
                # Get strategy for this column, or use default
                strategy = self.missing_value_strategies.get(
                    column, self.default_missing_strategy
                )

                # Apply the appropriate strategy
                if strategy == CleaningStrategy.DROP:
                    # Drop rows with missing values in this column
                    cleaned_data = cleaned_data.dropna(subset=[column])

                elif strategy == CleaningStrategy.FILL_MEAN:
                    # Fill missing values with column mean
                    if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mean())
                    else:
                        logger.warning(f"Cannot fill non-numeric column {column} with mean")

                elif strategy == CleaningStrategy.FILL_MEDIAN:
                    # Fill missing values with column median
                    if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())
                    else:
                        logger.warning(f"Cannot fill non-numeric column {column} with median")

                elif strategy == CleaningStrategy.FILL_MODE:
                    # Fill missing values with column mode
                    mode_value = cleaned_data[column].mode().iloc[0] if not cleaned_data[column].mode().empty else None
                    if mode_value is not None:
                        cleaned_data[column] = cleaned_data[column].fillna(mode_value)

                elif strategy == CleaningStrategy.FILL_CONSTANT:
                    # Fill missing values with a constant (column-specific or parameter)
                    column_fill_value = self.missing_fill_values.get(column, fill_value)
                    if column_fill_value is not None:
                        cleaned_data[column] = cleaned_data[column].fillna(column_fill_value)
                    else:
                        logger.warning(f"No fill value provided for constant fill strategy on column {column}")

                elif strategy == CleaningStrategy.FILL_INTERPOLATE:
                    # Fill missing values with interpolation
                    if pd.api.types.is_numeric_dtype(cleaned_data[column]):
                        cleaned_data[column] = cleaned_data[column].interpolate(method="linear")
                        # Handle edge cases (start/end of series)
                        cleaned_data[column] = cleaned_data[column].bfill().ffill()
                    else:
                        logger.warning(f"Cannot interpolate non-numeric column {column}")

                elif strategy == CleaningStrategy.FILL_FORWARD:
                    # Fill missing values with forward fill
                    cleaned_data[column] = cleaned_data[column].ffill()
                    # Handle case where first values are NaN
                    if cleaned_data[column].isna().any():
                        cleaned_data[column] = cleaned_data[column].bfill()

                elif strategy == CleaningStrategy.FILL_BACKWARD:
                    # Fill missing values with backward fill
                    cleaned_data[column] = cleaned_data[column].bfill()
                    # Handle case where last values are NaN
                    if cleaned_data[column].isna().any():
                        cleaned_data[column] = cleaned_data[column].ffill()

                elif strategy == CleaningStrategy.CUSTOM:
                    # Custom strategy should be implemented by subclasses
                    logger.warning(f"Custom missing value strategy for column {column} not implemented")

            # Record missing value statistics after cleaning
            missing_stats_after = cleaned_data.isna().sum().to_dict()
            self.metadata["missing_values_after"] = missing_stats_after

            return cleaned_data

        except Exception as e:
            error_msg = f"Error handling missing values: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def handle_outliers(
        self, data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle outliers in the data.

        Args:
            data: The data to clean.
            columns: Optional list of columns to check for outliers.
                If None, all numeric columns will be checked.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Make a copy of the input data
            cleaned_data = data.copy()

            # If no columns specified, use all numeric columns
            if columns is None:
                columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()

            # Record outlier statistics before cleaning
            outlier_stats_before = {}

            # Process each column
            for column in columns:
                if column not in cleaned_data.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(cleaned_data[column]):
                    continue

                # Get strategy for this column, or use default
                strategy = self.outlier_strategies.get(
                    column, self.default_outlier_strategy
                )

                # Detect outliers based on the specified method
                outlier_mask = self._detect_outliers(cleaned_data[column], column)
                outlier_count = outlier_mask.sum()
                outlier_stats_before[column] = outlier_count

                # Apply the appropriate strategy
                if strategy == CleaningStrategy.DROP:
                    # Drop rows with outliers in this column
                    cleaned_data = cleaned_data[~outlier_mask]

                elif strategy == CleaningStrategy.WINSORIZE:
                    # Cap outliers at specified percentiles using column-specific settings
                    detection_method = self.outlier_detection_methods.get(column, self.outlier_detection_method)
                    threshold = self.outlier_thresholds.get(column, self.outlier_threshold)
                    
                    if detection_method == OutlierDetectionMethod.PERCENTILE:
                        # Using percentiles for detection - threshold is the percentile level
                        lower_bound = cleaned_data[column].quantile(threshold)
                        upper_bound = cleaned_data[column].quantile(1.0 - threshold)
                    elif detection_method == OutlierDetectionMethod.IQR:
                        # Using IQR for detection
                        q1 = cleaned_data[column].quantile(0.25)
                        q3 = cleaned_data[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - (threshold * iqr)
                        upper_bound = q3 + (threshold * iqr)
                    else:
                        # Default percentiles (1% and 99%)
                        lower_bound = cleaned_data[column].quantile(0.01)
                        upper_bound = cleaned_data[column].quantile(0.99)

                    # Apply winsorization
                    cleaned_data[column] = cleaned_data[column].clip(lower=lower_bound, upper=upper_bound)

                elif strategy == CleaningStrategy.CLIP:
                    # Clip values to specified min/max
                    # This is similar to winsorization but allows for custom bounds
                    # Default to 3 standard deviations from the mean
                    mean = cleaned_data[column].mean()
                    std = cleaned_data[column].std()
                    lower_bound = mean - (3 * std)
                    upper_bound = mean + (3 * std)
                    cleaned_data[column] = cleaned_data[column].clip(lower=lower_bound, upper=upper_bound)

                elif strategy == CleaningStrategy.FILL_MEAN:
                    # Replace outliers with column mean
                    mean_value = cleaned_data.loc[~outlier_mask, column].mean()
                    cleaned_data.loc[outlier_mask, column] = mean_value

                elif strategy == CleaningStrategy.FILL_MEDIAN:
                    # Replace outliers with column median
                    median_value = cleaned_data.loc[~outlier_mask, column].median()
                    cleaned_data.loc[outlier_mask, column] = median_value

                elif strategy == CleaningStrategy.FILL_INTERPOLATE:
                    # Replace outliers with interpolated values
                    # First, store original values
                    original_values = cleaned_data[column].copy()
                    # Set outliers to NaN
                    cleaned_data.loc[outlier_mask, column] = np.nan
                    # Interpolate
                    cleaned_data[column] = cleaned_data[column].interpolate(method="linear")
                    # Handle edge cases
                    if cleaned_data[column].isna().any():
                        cleaned_data[column] = cleaned_data[column].ffill().bfill()
                    # If still have NaNs, restore original values for those positions
                    still_na = cleaned_data[column].isna()
                    if still_na.any():
                        cleaned_data.loc[still_na, column] = original_values.loc[still_na]

                elif strategy == CleaningStrategy.CUSTOM:
                    # Custom strategy should be implemented by subclasses
                    logger.warning(f"Custom outlier strategy for column {column} not implemented")

            # Record outlier statistics in metadata
            self.metadata["outliers_before"] = outlier_stats_before

            return cleaned_data

        except Exception as e:
            error_msg = f"Error handling outliers: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def _detect_outliers(self, series: pd.Series, column: Optional[str] = None) -> pd.Series:
        """Detect outliers in a series based on the specified method.

        Args:
            series: The series to check for outliers.
            column: Optional column name for column-specific settings.

        Returns:
            pd.Series: Boolean mask where True indicates an outlier.

        Raises:
            DataProcessingError: If an error occurs during outlier detection.
        """
        try:
            # Get column-specific settings or use defaults
            detection_method = self.outlier_detection_methods.get(column, self.outlier_detection_method)
            threshold = self.outlier_thresholds.get(column, self.outlier_threshold)
            
            if detection_method == OutlierDetectionMethod.Z_SCORE:
                # Z-score method
                z_scores = np.abs((series - series.mean()) / series.std())
                return z_scores > threshold

            elif detection_method == OutlierDetectionMethod.IQR:
                # Interquartile range method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                return (series < lower_bound) | (series > upper_bound)

            elif detection_method == OutlierDetectionMethod.PERCENTILE:
                # Percentile method - threshold represents the percentile level (e.g., 0.005 = 0.5%)
                lower_bound = series.quantile(threshold)  
                upper_bound = series.quantile(1.0 - threshold)  
                return (series < lower_bound) | (series > upper_bound)

            elif detection_method == OutlierDetectionMethod.MAD:
                # Median absolute deviation method
                median = series.median()
                mad = np.median(np.abs(series - median))
                return np.abs(series - median) > (threshold * mad)

            elif detection_method in [
                OutlierDetectionMethod.ISOLATION_FOREST,
                OutlierDetectionMethod.LOCAL_OUTLIER_FACTOR,
                OutlierDetectionMethod.DBSCAN,
            ]:
                # These methods require scikit-learn and are more complex
                # They should be implemented in a subclass or extension
                logger.warning(f"{detection_method.name} not implemented, falling back to IQR")
                # Fall back to IQR method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                return (series < lower_bound) | (series > upper_bound)

            elif detection_method == OutlierDetectionMethod.CUSTOM:
                # Custom method should be implemented by subclasses
                logger.warning("Custom outlier detection method not implemented, falling back to IQR")
                # Fall back to IQR method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                return (series < lower_bound) | (series > upper_bound)

            else:
                # Unknown method
                logger.warning(f"Unknown outlier detection method: {detection_method.name}")
                # Fall back to IQR method
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                return (series < lower_bound) | (series > upper_bound)

        except Exception as e:
            error_msg = f"Error detecting outliers: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def handle_duplicates(self, data: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle duplicate rows in the data.

        Args:
            data: The data to clean.
            subset: Optional list of column names to check for duplicates.
                If None, all columns will be used.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Make a copy of the input data
            cleaned_data = data.copy()

            # Check for duplicates
            duplicate_mask = cleaned_data.duplicated(subset=subset, keep="first")
            duplicate_count = duplicate_mask.sum()

            # Record duplicate statistics in metadata
            self.metadata["duplicates_removed"] = duplicate_count

            # Remove duplicates if any
            if duplicate_count > 0:
                logger.info(f"Removing {duplicate_count} duplicate rows")
                cleaned_data = cleaned_data[~duplicate_mask]

            return cleaned_data

        except Exception as e:
            error_msg = f"Error handling duplicates: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def fill_missing_timestamps(
        self, data: pd.DataFrame, freq: str, method: str = "ffill"
    ) -> pd.DataFrame:
        """Fill missing timestamps in time series data.

        Args:
            data: The data to process.
            freq: Frequency string (e.g., '1D', '1H', '1min').
            method: Method to fill values ('ffill', 'bfill', 'interpolate').

        Returns:
            pd.DataFrame: The processed data with complete timestamps.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Ensure the index is a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                raise DataProcessingError("Data index must be a DatetimeIndex")

            # Create a complete timestamp index
            full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)

            # Reindex the data
            reindexed_data = data.reindex(full_index)

            # Fill missing values based on the specified method
            if method == "ffill":
                filled_data = reindexed_data.ffill()
                # Handle case where first values are NaN
                if filled_data.isna().any().any():
                    filled_data = filled_data.bfill()
            elif method == "bfill":
                filled_data = reindexed_data.bfill()
                # Handle case where last values are NaN
                if filled_data.isna().any().any():
                    filled_data = filled_data.ffill()
            elif method == "interpolate":
                filled_data = reindexed_data.interpolate(method="time")
                # Handle edge cases
                if filled_data.isna().any().any():
                    filled_data = filled_data.ffill().bfill()
            else:
                raise DataProcessingError(f"Unknown fill method: {method}")

            # Record statistics in metadata
            added_timestamps = len(full_index) - len(data.index)
            self.metadata["timestamps_added"] = added_timestamps

            return filled_data

        except Exception as e:
            error_msg = f"Error filling missing timestamps: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def normalize_column(
        self, data: pd.DataFrame, column: str, method: str = "z-score"
    ) -> pd.DataFrame:
        """Normalize a column in the data.

        Args:
            data: The data to process.
            column: The column to normalize.
            method: Normalization method ('z-score', 'min-max', 'robust').

        Returns:
            pd.DataFrame: The processed data with normalized column.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Make a copy of the input data
            processed_data = data.copy()

            # Check if column exists and is numeric
            if column not in processed_data.columns:
                raise DataProcessingError(f"Column {column} not found in data")

            if not pd.api.types.is_numeric_dtype(processed_data[column]):
                raise DataProcessingError(f"Column {column} is not numeric")

            # Apply normalization based on the specified method
            if method == "z-score":
                # Z-score normalization (mean=0, std=1)
                mean = processed_data[column].mean()
                std = processed_data[column].std()
                if std == 0:
                    logger.warning(f"Standard deviation is zero for column {column}, skipping normalization")
                    return processed_data
                processed_data[column] = (processed_data[column] - mean) / std

            elif method == "min-max":
                # Min-max normalization (range [0, 1])
                min_val = processed_data[column].min()
                max_val = processed_data[column].max()
                if max_val == min_val:
                    logger.warning(f"Max equals min for column {column}, skipping normalization")
                    return processed_data
                processed_data[column] = (processed_data[column] - min_val) / (max_val - min_val)

            elif method == "robust":
                # Robust normalization using median and IQR
                median = processed_data[column].median()
                q1 = processed_data[column].quantile(0.25)
                q3 = processed_data[column].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    logger.warning(f"IQR is zero for column {column}, skipping normalization")
                    return processed_data
                processed_data[column] = (processed_data[column] - median) / iqr

            else:
                raise DataProcessingError(f"Unknown normalization method: {method}")

            # Record normalization in metadata
            if "normalized_columns" not in self.metadata:
                self.metadata["normalized_columns"] = {}
            self.metadata["normalized_columns"][column] = method

            return processed_data

        except Exception as e:
            error_msg = f"Error normalizing column {column}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def fill_timestamp_gaps(
        self, data: pd.DataFrame, freq: Optional[str] = None, method: str = "ffill"
    ) -> pd.DataFrame:
        """Fill missing timestamps in time series data with auto-detection.

        Args:
            data: The data to process.
            freq: Frequency string (e.g., '1D', '1H', '1min'). If None, will auto-detect.
            method: Method to fill values ('ffill', 'bfill', 'interpolate').

        Returns:
            pd.DataFrame: The processed data with complete timestamps.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Ensure the index is a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                raise DataProcessingError("Data index must be a DatetimeIndex")

            # Auto-detect frequency if not provided
            if freq is None:
                freq = pd.infer_freq(data.index)
                if freq is None:
                    # If auto-detection fails, try to infer from most common diff
                    time_diffs = data.index.to_series().diff().dropna()
                    most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else None
                    
                    if most_common_diff is not None:
                        # Convert to frequency string
                        if most_common_diff == pd.Timedelta(days=1):
                            freq = 'D'
                        elif most_common_diff == pd.Timedelta(hours=1):
                            freq = 'H'
                        elif most_common_diff == pd.Timedelta(minutes=1):
                            freq = '1min'
                        elif most_common_diff == pd.Timedelta(minutes=5):
                            freq = '5min'
                        elif most_common_diff == pd.Timedelta(minutes=15):
                            freq = '15min'
                        elif most_common_diff == pd.Timedelta(hours=4):
                            freq = '4H'
                        else:
                            # Default to most common difference as seconds
                            freq = f"{int(most_common_diff.total_seconds())}s"
                    else:
                        logger.warning("Could not auto-detect frequency, skipping timestamp gap filling")
                        return data
                        
                logger.info(f"Auto-detected frequency: {freq}")

            # Use the existing fill_missing_timestamps method
            return self.fill_missing_timestamps(data, freq, method)

        except Exception as e:
            error_msg = f"Error filling timestamp gaps: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def _handle_extreme_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers using z-score replacement.
        
        This method identifies values with z-scores greater than the threshold
        (default 5.0) and replaces them with the nearest non-extreme value.
        
        Args:
            data: The data to process.
            
        Returns:
            pd.DataFrame: The processed data with extreme outliers replaced.
            
        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            # Make a copy of the input data
            cleaned_data = data.copy()
            
            # Get numeric columns
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
            
            extreme_outliers_count = 0
            
            for column in numeric_columns:
                if column not in cleaned_data.columns:
                    continue
                    
                # Calculate z-scores
                mean = cleaned_data[column].mean()
                std = cleaned_data[column].std()
                
                if std == 0:
                    continue  # Skip columns with no variation
                    
                z_scores = np.abs((cleaned_data[column] - mean) / std)
                
                # Identify extreme outliers
                extreme_mask = z_scores > self.z_score_threshold
                extreme_count = extreme_mask.sum()
                
                if extreme_count > 0:
                    logger.info(f"Found {extreme_count} extreme outliers in column {column} (z-score > {self.z_score_threshold})")
                    extreme_outliers_count += extreme_count
                    
                    # Replace extreme outliers with median of non-extreme values
                    non_extreme_values = cleaned_data.loc[~extreme_mask, column]
                    if not non_extreme_values.empty:
                        replacement_value = non_extreme_values.median()
                        cleaned_data.loc[extreme_mask, column] = replacement_value
                    else:
                        # If all values are extreme, keep the original column unchanged
                        logger.warning(f"All values in column {column} are extreme outliers, keeping original values")
            
            # Record statistics in metadata
            self.metadata["extreme_outliers_replaced"] = extreme_outliers_count
            
            return cleaned_data
            
        except Exception as e:
            error_msg = f"Error handling extreme outliers: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def apply_corporate_actions(
        self,
        data: pd.DataFrame,
        corporate_actions_file: str,
        symbol: str,
        preserve_original_close: bool = True
    ) -> pd.DataFrame:
        """Apply corporate actions adjustments to OHLCV data.
        
        This is a convenience method that loads corporate actions from a file
        and applies them to the data.
        
        Args:
            data: DataFrame with OHLCV data
            corporate_actions_file: Path to CSV file with corporate actions
            symbol: Symbol to filter corporate actions for
            preserve_original_close: If True, preserves original close prices
        
        Returns:
            pd.DataFrame: Data adjusted for corporate actions
        
        Raises:
            DataProcessingError: If an error occurs during processing
        """
        try:
            # Load corporate actions from file
            corporate_actions = load_corporate_actions_from_file(corporate_actions_file)
            
            # Apply adjustments
            adjusted_data, metadata = adjust_for_corporate_actions(
                data=data,
                corporate_actions=corporate_actions,
                symbol=symbol,
                preserve_original_close=preserve_original_close
            )
            
            # Store metadata
            self.metadata.update({
                'corporate_actions_metadata': metadata,
                'corporate_actions_file': corporate_actions_file
            })
            
            return adjusted_data
            
        except Exception as e:
            error_msg = f"Error applying corporate actions: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def normalize_timezones(
        self,
        data: pd.DataFrame,
        source_timezone: Optional[str] = None,
        target_timezone: str = 'UTC',
        preserve_metadata: bool = True
    ) -> pd.DataFrame:
        """Normalize all datetime columns to UTC timezone.
        
        This is a convenience method that applies timezone normalization
        to the data.
        
        Args:
            data: DataFrame containing datetime data
            source_timezone: Source timezone (e.g., 'US/Eastern')
            target_timezone: Target timezone (default 'UTC')
            preserve_metadata: If True, stores timezone info in metadata
        
        Returns:
            pd.DataFrame: Data with normalized timezone
        
        Raises:
            DataProcessingError: If an error occurs during processing
        """
        try:
            # Apply timezone normalization
            normalized_data, metadata = normalize_timezone(
                data=data,
                source_timezone=source_timezone,
                target_timezone=target_timezone,
                preserve_metadata=preserve_metadata
            )
            
            # Store metadata
            self.metadata.update({
                'timezone_normalization_metadata': metadata
            })
            
            return normalized_data
            
        except Exception as e:
            error_msg = f"Error normalizing timezones: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)


def build_default_market_cleaner() -> DataCleaner:
    """Build and return a default market data cleaner with robust market-specific settings.
    
    This factory function creates a DataCleaner instance configured with:
    - Column-specific missing value strategies:
      * volume: FILL_CONSTANT with 0 (for non-trading periods)
      * open/high/low/close: FILL_FORWARD (preserve last known price)
      * Default: FILL_FORWARD for other columns
    - Outlier strategies with winsorization at 0.5% percentiles:
      * close/high/low: WINSORIZE with PERCENTILE method (0.5% bounds)
      * volume: WINSORIZE with IQR method (handle extreme volume spikes)
      * Default: WINSORIZE for other columns
    - Z-score replacement option for extreme outliers (>5 sigma)
    - Duplicate removal enabled
    - Timestamp gap filling with auto-frequency detection
    
    Returns:
        DataCleaner: Configured cleaner instance for market data
    """
    # Create enhanced cleaner with PERCENTILE detection for more precise outlier handling
    cleaner = DataCleaner(
        default_missing_strategy=CleaningStrategy.FILL_FORWARD,
        default_outlier_strategy=CleaningStrategy.WINSORIZE,
        outlier_detection_method=OutlierDetectionMethod.PERCENTILE,
        outlier_threshold=0.005  # 0.5% percentiles for winsorization
    )
    
    # Configure column-specific missing value strategies
    # Volume should be 0 during non-trading periods (weekends, holidays, after hours)
    cleaner.set_missing_value_strategy('volume', CleaningStrategy.FILL_CONSTANT)
    cleaner.set_missing_fill_value('volume', 0.0)  # Set volume to 0 for non-trading periods
    
    # Price columns should use forward fill to preserve last known price
    for price_col in ['open', 'high', 'low', 'close']:
        cleaner.set_missing_value_strategy(price_col, CleaningStrategy.FILL_FORWARD)
    
    # Configure outlier strategies with 0.5% winsorization for price columns
    for price_col in ['close', 'high', 'low', 'open']:
        cleaner.set_outlier_strategy(price_col, CleaningStrategy.WINSORIZE)
        cleaner.set_outlier_detection_method(price_col, OutlierDetectionMethod.PERCENTILE)
        cleaner.set_outlier_threshold(price_col, 0.005)  # 0.5% percentiles
    
    # Volume outliers use IQR method (better for handling volume spikes)
    cleaner.set_outlier_strategy('volume', CleaningStrategy.WINSORIZE)
    cleaner.set_outlier_detection_method('volume', OutlierDetectionMethod.IQR)
    cleaner.set_outlier_threshold('volume', 1.5)
    
    # Add z-score replacement for extreme outliers (>5 sigma) as a final cleanup step
    cleaner.add_processing_step(ProcessingStep.CLEANING, cleaner._handle_extreme_outliers)
    
    # Enable timestamp gap filling with auto-frequency detection
    cleaner.add_processing_step(ProcessingStep.CLEANING, cleaner.fill_timestamp_gaps)
    
    return cleaner


class CorporateAction:
    """Data class for corporate actions."""
    
    def __init__(
        self,
        symbol: str,
        date: datetime,
        action_type: str,
        factor: float,
        dividend_amount: Optional[float] = None,
        currency: str = "USD",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.symbol = symbol
        self.date = date if isinstance(date, datetime) else pd.to_datetime(date)
        self.action_type = action_type.lower()  # 'split', 'dividend', 'spinoff', etc.
        self.factor = factor
        self.dividend_amount = dividend_amount
        self.currency = currency
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"CorporateAction(symbol={self.symbol}, date={self.date}, type={self.action_type}, factor={self.factor})"


def load_corporate_actions_from_file(file_path: str) -> List[CorporateAction]:
    """Load corporate actions from a CSV file.
    
    Expected CSV format:
    symbol,date,action_type,factor,dividend_amount,currency
    AAPL,2022-08-29,split,4.0,,USD
    AAPL,2022-02-11,dividend,,0.22,USD
    
    Args:
        file_path: Path to the CSV file containing corporate actions
        
    Returns:
        List[CorporateAction]: List of corporate actions
        
    Raises:
        DataProcessingError: If file cannot be loaded or parsed
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['symbol', 'date', 'action_type', 'factor']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataProcessingError(f"Missing required columns in corporate actions file: {missing_columns}")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Fill optional columns with defaults
        df['dividend_amount'] = df.get('dividend_amount', None)
        df['currency'] = df.get('currency', 'USD')
        
        # Create CorporateAction objects
        actions = []
        for _, row in df.iterrows():
            action = CorporateAction(
                symbol=row['symbol'],
                date=row['date'],
                action_type=row['action_type'],
                factor=row['factor'],
                dividend_amount=row['dividend_amount'] if pd.notna(row['dividend_amount']) else None,
                currency=row['currency']
            )
            actions.append(action)
        
        logger.info(f"Loaded {len(actions)} corporate actions from {file_path}")
        return actions
        
    except Exception as e:
        error_msg = f"Error loading corporate actions from file {file_path}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise DataProcessingError(error_msg)


def adjust_for_corporate_actions(
    data: pd.DataFrame,
    corporate_actions: List[CorporateAction],
    symbol: str,
    preserve_original_close: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Adjust OHLCV data for corporate actions (splits and dividends).
    
    This function applies corporate action adjustments to historical OHLCV data
    to ensure price continuity. The adjustments are applied in reverse chronological
    order (most recent first) to maintain accuracy.
    
    Args:
        data: DataFrame with OHLCV data. Index should be DatetimeIndex.
              Expected columns: open, high, low, close, volume
        corporate_actions: List of CorporateAction objects to apply
        symbol: Symbol to filter corporate actions for
        preserve_original_close: If True, preserves original close prices and
                               adjusts historical data. If False, adjusts all data.
    
    Returns:
        Tuple containing:
        - Adjusted DataFrame
        - Dictionary with adjustment metadata
    
    Raises:
        DataProcessingError: If an error occurs during adjustment
    """
    try:
        # Validate input data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataProcessingError("Data index must be a DatetimeIndex")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataProcessingError(f"Missing required OHLCV columns: {missing_columns}")
        
        # Make a copy of the input data
        adjusted_data = data.copy()
        
        # Filter corporate actions for the symbol
        symbol_actions = [action for action in corporate_actions if action.symbol.upper() == symbol.upper()]
        
        if not symbol_actions:
            logger.info(f"No corporate actions found for symbol {symbol}")
            return adjusted_data, {"adjustments_applied": 0}
        
        # Sort actions by date in reverse chronological order (most recent first)
        symbol_actions.sort(key=lambda x: x.date, reverse=True)
        
        # Track adjustments for metadata
        adjustments_applied = []
        cumulative_split_factor = 1.0
        cumulative_dividend_adjustment = 0.0
        
        # Apply adjustments in reverse chronological order
        for action in symbol_actions:
            # Only adjust data before the action date
            mask = adjusted_data.index < action.date
            
            if not mask.any():
                logger.info(f"No data before {action.date} for action {action}")
                continue
            
            if action.action_type == 'split':
                # Stock split adjustment
                split_factor = action.factor
                
                # Adjust prices (divide by split factor)
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    adjusted_data.loc[mask, col] = adjusted_data.loc[mask, col] / split_factor
                
                # Adjust volume (multiply by split factor)
                adjusted_data.loc[mask, 'volume'] = adjusted_data.loc[mask, 'volume'] * split_factor
                
                cumulative_split_factor *= split_factor
                
                adjustment_info = {
                    'type': 'split',
                    'date': action.date,
                    'factor': split_factor,
                    'records_affected': mask.sum()
                }
                adjustments_applied.append(adjustment_info)
                
                logger.info(f"Applied {split_factor}:1 split adjustment for {symbol} on {action.date}, affected {mask.sum()} records")
            
            elif action.action_type == 'dividend':
                # Dividend adjustment
                if action.dividend_amount is None:
                    logger.warning(f"Dividend action for {symbol} on {action.date} has no dividend amount, skipping")
                    continue
                
                dividend_amount = action.dividend_amount
                
                # For dividend adjustments, we need the close price on the ex-dividend date
                # to calculate the adjustment factor
                ex_div_date_mask = adjusted_data.index == action.date
                if not ex_div_date_mask.any():
                    # If exact date not found, find the closest trading day after
                    future_dates = adjusted_data.index[adjusted_data.index > action.date]
                    if future_dates.empty:
                        logger.warning(f"No trading data found on or after dividend date {action.date} for {symbol}")
                        continue
                    ex_div_close = adjusted_data.loc[future_dates[0], 'close']
                else:
                    ex_div_close = adjusted_data.loc[ex_div_date_mask, 'close'].iloc[0]
                
                # Calculate dividend adjustment factor
                dividend_factor = (ex_div_close - dividend_amount) / ex_div_close
                
                # Adjust prices before the ex-dividend date
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    adjusted_data.loc[mask, col] = adjusted_data.loc[mask, col] * dividend_factor
                
                # Volume is not typically adjusted for dividends
                cumulative_dividend_adjustment += dividend_amount
                
                adjustment_info = {
                    'type': 'dividend',
                    'date': action.date,
                    'amount': dividend_amount,
                    'factor': dividend_factor,
                    'ex_div_close': ex_div_close,
                    'records_affected': mask.sum()
                }
                adjustments_applied.append(adjustment_info)
                
                logger.info(f"Applied ${dividend_amount} dividend adjustment for {symbol} on {action.date}, factor: {dividend_factor:.6f}, affected {mask.sum()} records")
            
            else:
                logger.warning(f"Unknown corporate action type '{action.action_type}' for {symbol} on {action.date}, skipping")
        
        # Prepare metadata
        metadata = {
            'symbol': symbol,
            'adjustments_applied': len(adjustments_applied),
            'adjustment_details': adjustments_applied,
            'cumulative_split_factor': cumulative_split_factor,
            'cumulative_dividend_adjustment': cumulative_dividend_adjustment,
            'preserve_original_close': preserve_original_close,
            'total_records': len(adjusted_data),
            'date_range': (adjusted_data.index.min(), adjusted_data.index.max())
        }
        
        logger.info(f"Corporate action adjustment completed for {symbol}: {len(adjustments_applied)} adjustments applied")
        return adjusted_data, metadata
        
    except Exception as e:
        error_msg = f"Error adjusting for corporate actions: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise DataProcessingError(error_msg)


def normalize_timezone(
    data: pd.DataFrame,
    source_timezone: Optional[str] = None,
    target_timezone: str = 'UTC',
    preserve_metadata: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize all datetime columns to a target timezone (default UTC).
    
    This function converts datetime columns from source timezone to target timezone,
    preserving all non-datetime data and optionally maintaining original timezone 
    information in metadata.
    
    Args:
        data: DataFrame containing datetime data
        source_timezone: Source timezone (e.g., 'US/Eastern', 'Europe/London').
                        If None, assumes data is timezone-naive and in local time
        target_timezone: Target timezone (default 'UTC')
        preserve_metadata: If True, stores original timezone info in metadata
    
    Returns:
        Tuple containing:
        - DataFrame with normalized datetime columns
        - Dictionary with timezone conversion metadata
    
    Raises:
        DataProcessingError: If an error occurs during timezone conversion
    """
    try:
        # Make a copy of the input data
        normalized_data = data.copy()
        
        # Track timezone conversions for metadata
        conversions_applied = []
        datetime_columns = []
        
        # Get timezone objects
        target_tz = pytz.timezone(target_timezone)
        source_tz = pytz.timezone(source_timezone) if source_timezone else None
        
        # Process DataFrame index if it's a DatetimeIndex
        if isinstance(normalized_data.index, pd.DatetimeIndex):
            original_tz = normalized_data.index.tz
            
            if normalized_data.index.tz is None:
                # Timezone-naive index
                if source_timezone:
                    # Localize to source timezone first, then convert to target
                    normalized_data.index = normalized_data.index.tz_localize(source_tz)
                    normalized_data.index = normalized_data.index.tz_convert(target_tz)
                else:
                    # Assume UTC if no source timezone specified
                    normalized_data.index = normalized_data.index.tz_localize(target_tz)
            else:
                # Already timezone-aware, just convert
                normalized_data.index = normalized_data.index.tz_convert(target_tz)
            
            conversion_info = {
                'column': 'index',
                'original_timezone': str(original_tz),
                'target_timezone': target_timezone,
                'records_converted': len(normalized_data.index)
            }
            conversions_applied.append(conversion_info)
            
            logger.info(f"Converted index timezone from {original_tz} to {target_timezone} for {len(normalized_data.index)} records")
        
        # Process datetime columns
        for column in normalized_data.columns:
            if pd.api.types.is_datetime64_any_dtype(normalized_data[column]):
                datetime_columns.append(column)
                original_tz = getattr(normalized_data[column].dtype, 'tz', None)
                
                # Convert the datetime column
                if normalized_data[column].dt.tz is None:
                    # Timezone-naive column
                    if source_timezone:
                        # Localize to source timezone first, then convert to target
                        normalized_data[column] = normalized_data[column].dt.tz_localize(source_tz)
                        normalized_data[column] = normalized_data[column].dt.tz_convert(target_tz)
                    else:
                        # Assume UTC if no source timezone specified
                        normalized_data[column] = normalized_data[column].dt.tz_localize(target_tz)
                else:
                    # Already timezone-aware, just convert
                    normalized_data[column] = normalized_data[column].dt.tz_convert(target_tz)
                
                conversion_info = {
                    'column': column,
                    'original_timezone': str(original_tz),
                    'target_timezone': target_timezone,
                    'records_converted': len(normalized_data[column].dropna())
                }
                conversions_applied.append(conversion_info)
                
                logger.info(f"Converted column '{column}' timezone from {original_tz} to {target_timezone} for {len(normalized_data[column].dropna())} records")
        
        # Prepare metadata
        metadata = {
            'source_timezone': source_timezone,
            'target_timezone': target_timezone,
            'datetime_columns_converted': datetime_columns,
            'conversions_applied': len(conversions_applied),
            'conversion_details': conversions_applied if preserve_metadata else [],
            'total_records': len(normalized_data),
            'preserve_metadata': preserve_metadata
        }
        
        if not conversions_applied:
            logger.info("No datetime columns found for timezone conversion")
        else:
            logger.info(f"Timezone normalization completed: {len(conversions_applied)} datetime fields converted to {target_timezone}")
        
        return normalized_data, metadata
        
    except Exception as e:
        error_msg = f"Error normalizing timezone: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise DataProcessingError(error_msg)
