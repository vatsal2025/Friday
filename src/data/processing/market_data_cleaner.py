"""Market Data Cleaner module for the Friday AI Trading System.

This module provides the MarketDataCleaner class with capabilities to:
- Handle duplicate rows using timestamp and symbol.
- Detect and cap extreme outliers using configurable z-score and IQR methods.
- Detect and correct bad numeric casts.
- Fill gaps using forward and back-fill strategies.
- Log cleaning operations and metrics.
- Integrate DataPipeline events for metric tracking.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem, Event
from src.data.processing.data_cleaner import (
    DataCleaner,
    CleaningStrategy,
    OutlierDetectionMethod,
    DataProcessingError
)

# Create logger
logger = get_logger(__name__)


class MarketDataCleaner(DataCleaner):
    """Market-specific data cleaner that extends DataCleaner.
    
    This class provides specialized cleaning methods for market data including:
    - Duplicate removal based on timestamp and symbol
    - Extreme outlier detection and capping using z-score or IQR methods  
    - Bad numeric cast detection and correction
    - Gap filling using forward-fill then back-fill
    - Comprehensive logging and metric tracking
    - Integration with DataPipeline events
    
    Attributes:
        symbol_column: Name of the symbol/ticker column for duplicate detection
        timestamp_column: Name of the timestamp column for duplicate detection
        numeric_columns: List of columns that should be numeric (float64)
        z_score_threshold: Threshold for z-score outlier detection
        iqr_multiplier: Multiplier for IQR outlier detection
        event_system: EventSystem for emitting cleaning metrics
        cleaning_metrics: Dictionary storing detailed cleaning metrics
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None,
        symbol_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        numeric_columns: Optional[List[str]] = None,
        z_score_threshold: Optional[float] = None,
        iqr_multiplier: Optional[float] = None,
        **kwargs
    ):
        """Initialize a MarketDataCleaner.
        
        Args:
            config: Configuration manager
            event_system: EventSystem for emitting cleaning metrics
            symbol_column: Name of symbol column for duplicate detection
            timestamp_column: Name of timestamp column for duplicate detection  
            numeric_columns: List of columns that should be numeric
            z_score_threshold: Threshold for z-score outlier detection
            iqr_multiplier: Multiplier for IQR outlier detection
            **kwargs: Additional arguments passed to parent DataCleaner
        """
        super().__init__(config=config, **kwargs)
        
        # Setup EventSystem for metrics tracking
        self.event_system = event_system
        
        # Load configuration from ConfigManager with defaults
        self.config = config or ConfigManager()
        
        # Get data cleaning configuration from ConfigManager
        cleaning_config = self.config.get('data.cleaning', {})
        
        # Set attributes with config fallbacks
        self.symbol_column = symbol_column or cleaning_config.get('symbol_column', "symbol")
        self.timestamp_column = timestamp_column or cleaning_config.get('timestamp_column', "timestamp")
        self.numeric_columns = numeric_columns or cleaning_config.get('numeric_columns', [
            'open', 'high', 'low', 'close', 'volume', 'price'
        ])
        self.z_score_threshold = z_score_threshold or cleaning_config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = iqr_multiplier or cleaning_config.get('iqr_multiplier', 1.5)
        
        # Additional thresholds from config
        self.max_outlier_percentage = cleaning_config.get('max_outlier_percentage', 0.05)
        self.min_numeric_cast_success_rate = cleaning_config.get('min_numeric_cast_success_rate', 0.95)
        self.gap_fill_max_consecutive = cleaning_config.get('gap_fill_max_consecutive', 5)
        self.enable_detailed_logging = cleaning_config.get('enable_detailed_logging', True)
        
        # Initialize cleaning metrics tracking
        self.cleaning_metrics = {
            'duplicates_removed': 0,
            'outliers_capped': 0,
            'bad_casts_corrected': 0,
            'gaps_filled': 0,
            'rows_modified': 0,
            'total_processing_time': 0.0,
            'stage_times': {},
            'data_quality_score': 0.0
        }
        
        # Override default strategies for market data
        self.default_missing_strategy = CleaningStrategy.FILL_FORWARD
        self.default_outlier_strategy = CleaningStrategy.WINSORIZE
        
        # Clear existing processing steps and add market-specific ones
        self.processing_steps = {}
        self._setup_market_processing_steps()
    
    def _setup_market_processing_steps(self) -> None:
        """Setup market-specific processing steps in order."""
        from src.data.processing.data_processor import ProcessingStep
        
        # Order is important for market data cleaning
        self.add_processing_step(ProcessingStep.CLEANING, self.detect_and_fix_bad_numeric_casts)
        self.add_processing_step(ProcessingStep.CLEANING, self.coerce_data_types)
        self.add_processing_step(ProcessingStep.CLEANING, self.remove_market_duplicates)
        self.add_processing_step(ProcessingStep.CLEANING, self.cap_outliers_market)
        self.add_processing_step(ProcessingStep.CLEANING, self.fill_gaps_market)
    
    def detect_and_fix_bad_numeric_casts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and fix bad numeric casts in the data.
        
        This method identifies values that should be numeric but can't be converted,
        and attempts to clean them or mark them for removal.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with corrected numeric values
            
        Raises:
            DataProcessingError: If too many bad casts are detected
        """
        try:
            import time
            stage_start = time.time()
            
            processed_data = data.copy()
            bad_cast_stats = {}
            total_bad_casts = 0
            
            for column in self.numeric_columns:
                if column not in processed_data.columns:
                    continue
                    
                original_col = processed_data[column].copy()
                
                # Count non-null values for success rate calculation
                non_null_count = original_col.notna().sum()
                if non_null_count == 0:
                    continue
                
                # Attempt to convert to numeric, identifying problematic values
                numeric_col = pd.to_numeric(original_col, errors='coerce')
                
                # Find values that became NaN after conversion (bad casts)
                bad_cast_mask = original_col.notna() & numeric_col.isna()
                bad_cast_count = bad_cast_mask.sum()
                
                if bad_cast_count > 0:
                    # Try to clean common bad formats
                    cleaned_values = self._clean_numeric_strings(original_col[bad_cast_mask])
                    
                    # Apply cleaned values back to the column
                    processed_data.loc[bad_cast_mask, column] = cleaned_values
                    
                    # Recheck conversion success
                    final_numeric = pd.to_numeric(processed_data[column], errors='coerce')
                    remaining_bad = processed_data[column].notna() & final_numeric.isna()
                    remaining_count = remaining_bad.sum()
                    
                    corrected_count = bad_cast_count - remaining_count
                    
                    bad_cast_stats[column] = {
                        'original_bad_casts': bad_cast_count,
                        'corrected': corrected_count,
                        'remaining_bad': remaining_count,
                        'success_rate': (non_null_count - remaining_count) / non_null_count
                    }
                    
                    total_bad_casts += remaining_count
                    
                    if self.enable_detailed_logging:
                        logger.info(f"Column '{column}': Fixed {corrected_count}/{bad_cast_count} bad numeric casts")
                        
                    # Check if success rate is acceptable
                    if bad_cast_stats[column]['success_rate'] < self.min_numeric_cast_success_rate:
                        logger.warning(f"Column '{column}' has low numeric cast success rate: {bad_cast_stats[column]['success_rate']:.2%}")
            
            # Update metrics
            stage_time = time.time() - stage_start
            self.cleaning_metrics['bad_casts_corrected'] = sum(stats['corrected'] for stats in bad_cast_stats.values())
            self.cleaning_metrics['stage_times']['bad_cast_detection'] = stage_time
            
            # Store detailed stats in metadata
            self.metadata['bad_cast_detection'] = bad_cast_stats
            self.metadata['total_remaining_bad_casts'] = total_bad_casts
            
            # Emit event with metrics
            self._emit_cleaning_event(
                'data.cleaning.bad_casts_detected',
                {
                    'total_corrected': self.cleaning_metrics['bad_casts_corrected'],
                    'total_remaining': total_bad_casts,
                    'column_stats': bad_cast_stats,
                    'processing_time': stage_time
                }
            )
            
            if self.enable_detailed_logging and total_bad_casts > 0:
                logger.warning(f"After cleaning: {total_bad_casts} bad numeric values remain across all columns")
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error during bad numeric cast detection: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def _clean_numeric_strings(self, series: pd.Series) -> pd.Series:
        """Clean common bad numeric string formats.
        
        Args:
            series: Series with bad numeric strings
            
        Returns:
            Series with cleaned numeric values
        """
        cleaned = series.copy()
        
        # Remove common non-numeric characters
        if cleaned.dtype == 'object':
            # Remove commas, dollar signs, percent signs, whitespace
            cleaned = cleaned.astype(str).str.replace(r'[,$%\s]', '', regex=True)
            
            # Handle parentheses (negative numbers)
            negative_mask = cleaned.str.contains(r'\(.*\)', na=False)
            cleaned.loc[negative_mask] = '-' + cleaned.loc[negative_mask].str.replace(r'[\(\)]', '', regex=True)
            
            # Replace empty strings with NaN
            cleaned = cleaned.replace('', np.nan)
            
            # Try to convert to numeric again
            cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        return cleaned
    
    def coerce_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric columns to float64 type.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with proper data types
            
        Raises:
            DataProcessingError: If type coercion fails
        """
        try:
            processed_data = data.copy()
            coercion_log = {}
            
            for column in self.numeric_columns:
                if column in processed_data.columns:
                    original_dtype = processed_data[column].dtype
                    
                    # Attempt to convert to numeric, coercing errors to NaN
                    processed_data[column] = pd.to_numeric(
                        processed_data[column], 
                        errors='coerce'
                    ).astype('float64')
                    
                    new_dtype = processed_data[column].dtype
                    coercion_log[column] = {
                        'original_dtype': str(original_dtype),
                        'new_dtype': str(new_dtype),
                        'na_introduced': processed_data[column].isna().sum() - data[column].isna().sum()
                    }
                    
                    logger.debug(f"Coerced column '{column}' from {original_dtype} to {new_dtype}")
            
            # Store coercion metadata
            self.metadata['type_coercion'] = coercion_log
            total_na_introduced = sum(log['na_introduced'] for log in coercion_log.values())
            self.metadata['na_values_introduced_by_coercion'] = total_na_introduced
            
            if total_na_introduced > 0:
                logger.warning(f"Type coercion introduced {total_na_introduced} NaN values")
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error during type coercion: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def remove_market_duplicates(
        self, 
        data: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Remove duplicates based on timestamp and symbol.
        
        Args:
            data: Input DataFrame
            subset: Optional list of columns to use for duplicate detection
                   If None, uses timestamp_column and symbol_column
                   
        Returns:
            DataFrame with duplicates removed
            
        Raises:
            DataProcessingError: If duplicate removal fails
        """
        try:
            import time
            stage_start = time.time()
            
            processed_data = data.copy()
            original_shape = processed_data.shape
            
            # Determine columns for duplicate detection
            if subset is None:
                duplicate_cols = []
                if self.timestamp_column in processed_data.columns:
                    duplicate_cols.append(self.timestamp_column)
                elif isinstance(processed_data.index, pd.DatetimeIndex):
                    # Use index as timestamp if no timestamp column
                    processed_data = processed_data.reset_index()
                    duplicate_cols.append('index')
                
                if self.symbol_column in processed_data.columns:
                    duplicate_cols.append(self.symbol_column)
                    
                subset = duplicate_cols if duplicate_cols else None
            
            # Check for duplicates
            if subset:
                duplicate_mask = processed_data.duplicated(subset=subset, keep='first')
            else:
                # Fall back to checking all columns
                duplicate_mask = processed_data.duplicated(keep='first')
                
            duplicate_count = duplicate_mask.sum()
            
            # Remove duplicates
            if duplicate_count > 0:
                if self.enable_detailed_logging:
                    logger.info(f"Removing {duplicate_count} duplicate rows based on {subset}")
                processed_data = processed_data[~duplicate_mask]
                
                # Reset index if we added one
                if 'index' in processed_data.columns and 'index' in (subset or []):
                    processed_data = processed_data.set_index('index')
            
            # Update metrics
            stage_time = time.time() - stage_start
            self.cleaning_metrics['duplicates_removed'] = duplicate_count
            self.cleaning_metrics['rows_modified'] += duplicate_count
            self.cleaning_metrics['stage_times']['duplicate_removal'] = stage_time
            
            # Store metadata
            self.metadata['market_duplicates_removed'] = duplicate_count
            self.metadata['duplicate_detection_columns'] = subset
            self.metadata['duplicate_removal_rate'] = duplicate_count / original_shape[0] if original_shape[0] > 0 else 0
            
            # Emit event with metrics
            self._emit_cleaning_event(
                'data.cleaning.duplicates_removed',
                {
                    'duplicates_removed': duplicate_count,
                    'original_rows': original_shape[0],
                    'remaining_rows': processed_data.shape[0],
                    'removal_rate': self.metadata['duplicate_removal_rate'],
                    'detection_columns': subset,
                    'processing_time': stage_time
                }
            )
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error removing market duplicates: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def fill_gaps_market(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in market data using forward-fill then back-fill strategies.
        
        This method handles missing values and data gaps specifically for time-series
        market data, ensuring proper temporal continuity.
        
        Args:
            data: Input DataFrame with potential gaps
            
        Returns:
            DataFrame with gaps filled
            
        Raises:
            DataProcessingError: If gap filling fails
        """
        try:
            import time
            stage_start = time.time()
            
            processed_data = data.copy()
            
            # Record missing values before filling
            missing_before = processed_data.isna().sum().to_dict()
            total_missing_before = sum(missing_before.values())
            
            # Sort by timestamp if available for proper forward/back fill
            if self.timestamp_column in processed_data.columns:
                processed_data = processed_data.sort_values(self.timestamp_column)
            elif isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data = processed_data.sort_index()
            
            # Track gaps filled by method
            gaps_stats = {
                'forward_fill': 0,
                'back_fill': 0,
                'consecutive_gaps_detected': 0,
                'max_consecutive_gap': 0
            }
            
            # Apply forward fill first
            after_ffill = processed_data.ffill()
            forward_filled_count = (processed_data.isna() & after_ffill.notna()).sum().sum()
            gaps_stats['forward_fill'] = forward_filled_count
            
            # Apply back fill for remaining NaNs
            final_data = after_ffill.bfill()
            back_filled_count = (after_ffill.isna() & final_data.notna()).sum().sum()
            gaps_stats['back_fill'] = back_filled_count
            
            # Detect consecutive gaps (before filling)
            for column in self.numeric_columns:
                if column in processed_data.columns:
                    na_mask = processed_data[column].isna()
                    if na_mask.any():
                        # Find consecutive NaN sequences
                        consecutive_groups = (na_mask != na_mask.shift()).cumsum()[na_mask]
                        if not consecutive_groups.empty:
                            gap_lengths = consecutive_groups.value_counts()
                            gaps_stats['consecutive_gaps_detected'] += len(gap_lengths)
                            gaps_stats['max_consecutive_gap'] = max(
                                gaps_stats['max_consecutive_gap'],
                                gap_lengths.max() if not gap_lengths.empty else 0
                            )
                            
                            # Warn about large consecutive gaps
                            large_gaps = gap_lengths[gap_lengths > self.gap_fill_max_consecutive]
                            if not large_gaps.empty:
                                logger.warning(f"Column '{column}': Found {len(large_gaps)} consecutive gap sequences longer than {self.gap_fill_max_consecutive}")
            
            # Record missing values after filling
            missing_after = final_data.isna().sum().to_dict()
            total_missing_after = sum(missing_after.values())
            
            # Update metrics
            stage_time = time.time() - stage_start
            total_filled = total_missing_before - total_missing_after
            self.cleaning_metrics['gaps_filled'] = total_filled
            self.cleaning_metrics['stage_times']['gap_filling'] = stage_time
            
            # Store detailed stats in metadata
            self.metadata['gap_filling'] = {
                'missing_before': missing_before,
                'missing_after': missing_after,
                'total_gaps_filled': total_filled,
                'fill_methods_used': gaps_stats,
                'fill_success_rate': (total_missing_before - total_missing_after) / total_missing_before if total_missing_before > 0 else 1.0
            }
            
            # Emit event with metrics
            self._emit_cleaning_event(
                'data.cleaning.gaps_filled',
                {
                    'total_filled': total_filled,
                    'forward_fill_count': gaps_stats['forward_fill'],
                    'back_fill_count': gaps_stats['back_fill'],
                    'consecutive_gaps': gaps_stats['consecutive_gaps_detected'],
                    'max_consecutive_gap': gaps_stats['max_consecutive_gap'],
                    'fill_success_rate': self.metadata['gap_filling']['fill_success_rate'],
                    'processing_time': stage_time
                }
            )
            
            if self.enable_detailed_logging and total_filled > 0:
                logger.info(f"Filled {total_filled} gaps using forward-fill ({gaps_stats['forward_fill']}) and back-fill ({gaps_stats['back_fill']})")
            
            # Log remaining missing values
            if total_missing_after > 0:
                logger.warning(f"{total_missing_after} missing values remain after gap filling")
            
            return final_data
            
        except Exception as e:
            error_msg = f"Error during gap filling: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def impute_missing_values_market(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using forward-fill then back-fill.
        
        This method applies forward-fill first to carry forward the last known value,
        then applies back-fill for any remaining NaN values at the beginning of series.
        
        Args:
            data: Input DataFrame with missing values
            
        Returns:
            DataFrame with missing values imputed
            
        Raises:
            DataProcessingError: If imputation fails
        """
        try:
            processed_data = data.copy()
            
            # Record missing values before imputation
            missing_before = processed_data.isna().sum().to_dict()
            self.metadata['missing_values_before_market_imputation'] = missing_before
            
            # Sort by timestamp if available for proper forward/back fill
            if self.timestamp_column in processed_data.columns:
                processed_data = processed_data.sort_values(self.timestamp_column)
            elif isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data = processed_data.sort_index()
            
            # Apply forward fill then back fill
            processed_data = processed_data.ffill()  # Forward fill
            processed_data = processed_data.bfill()  # Back fill for remaining NaNs
            
            # Record missing values after imputation
            missing_after = processed_data.isna().sum().to_dict()
            self.metadata['missing_values_after_market_imputation'] = missing_after
            
            # Calculate imputation statistics
            total_imputed = sum(missing_before.values()) - sum(missing_after.values())
            self.metadata['values_imputed_market'] = total_imputed
            
            if total_imputed > 0:
                logger.info(f"Imputed {total_imputed} missing values using forward-fill then back-fill")
            
            # Log remaining missing values
            remaining_missing = sum(missing_after.values())
            if remaining_missing > 0:
                logger.warning(f"{remaining_missing} missing values remain after market imputation")
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error during market missing value imputation: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def detect_outliers_zscore(
        self, 
        series: pd.Series, 
        threshold: Optional[float] = None
    ) -> pd.Series:
        """Detect outliers using z-score method.
        
        Args:
            series: Input series to check for outliers
            threshold: Z-score threshold (default: self.z_score_threshold)
            
        Returns:
            Boolean series where True indicates outlier
        """
        threshold = threshold or self.z_score_threshold
        
        # Calculate z-scores
        mean = series.mean()
        std = series.std()
        
        if std == 0 or not np.isfinite(std):
            # No variation in data or invalid std, no outliers
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        # Handle cases where z_scores might be infinite due to extreme values
        z_scores = z_scores.replace([np.inf, -np.inf], threshold + 1)
        return z_scores > threshold
    
    def detect_outliers_iqr(
        self, 
        series: pd.Series, 
        multiplier: Optional[float] = None
    ) -> pd.Series:
        """Detect outliers using IQR method.
        
        Args:
            series: Input series to check for outliers
            multiplier: IQR multiplier (default: self.iqr_multiplier)
            
        Returns:
            Boolean series where True indicates outlier
        """
        multiplier = multiplier or self.iqr_multiplier
        
        # Calculate IQR bounds
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        return (series < lower_bound) | (series > upper_bound)
    
    def cap_outliers_market(
        self, 
        data: pd.DataFrame,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Cap extreme outliers using z-score or IQR method.
        
        Args:
            data: Input DataFrame
            method: Method to use ('zscore' or 'iqr'). If None, uses config default.
            columns: Columns to process (default: numeric_columns)
            
        Returns:
            DataFrame with outliers capped
            
        Raises:
            DataProcessingError: If outlier capping fails or too many outliers detected
        """
        try:
            import time
            stage_start = time.time()
            
            processed_data = data.copy()
            original_shape = processed_data.shape
            
            # Get method from config if not specified
            if method is None:
                method = self.config.get('data.cleaning.outlier_method', 'iqr')
            
            # Determine columns to process
            if columns is None:
                columns = [col for col in self.numeric_columns if col in processed_data.columns]
            
            outlier_stats = {}
            total_outliers_detected = 0
            
            for column in columns:
                if column not in processed_data.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(processed_data[column]):
                    if self.enable_detailed_logging:
                        logger.warning(f"Skipping non-numeric column {column} for outlier capping")
                    continue
                
                series = processed_data[column].dropna()
                if len(series) == 0:
                    continue
                
                # Detect outliers based on method
                if method.lower() == "zscore":
                    outlier_mask = self.detect_outliers_zscore(series)
                elif method.lower() == "iqr":
                    outlier_mask = self.detect_outliers_iqr(series)
                else:
                    logger.error(f"Unknown outlier detection method: {method}")
                    raise DataProcessingError(f"Unknown outlier detection method: {method}")
                
                outlier_count = outlier_mask.sum()
                outlier_percentage = (outlier_count / len(series)) * 100
                
                outlier_stats[column] = {
                    'method': method,
                    'outliers_detected': outlier_count,
                    'outlier_percentage': outlier_percentage,
                    'total_values': len(series)
                }
                
                total_outliers_detected += outlier_count
                
                # Check if outlier percentage is too high
                if outlier_percentage > (self.max_outlier_percentage * 100):
                    logger.warning(f"Column '{column}': High outlier rate {outlier_percentage:.2f}% (threshold: {self.max_outlier_percentage * 100:.2f}%)")
                
                if outlier_count > 0:
                    if self.enable_detailed_logging:
                        logger.info(f"Detected {outlier_count} outliers ({outlier_percentage:.2f}%) in column '{column}' using {method} method")
                    
                    # Cap outliers based on method
                    if method.lower() == "zscore":
                        # For z-score method, use more robust bounds for extreme values
                        # Filter out the detected outliers to calculate bounds from non-outliers
                        non_outlier_series = series[~outlier_mask]
                        if len(non_outlier_series) > 0:
                            mean = non_outlier_series.mean()
                            std = non_outlier_series.std()
                            if std > 0:
                                lower_cap = mean - (self.z_score_threshold * std)
                                upper_cap = mean + (self.z_score_threshold * std)
                            else:
                                # Fallback for no variation in non-outliers
                                lower_cap = upper_cap = mean
                        else:
                            # All values are outliers, use percentile approach
                            lower_cap = series.quantile(0.01)
                            upper_cap = series.quantile(0.99)
                    else:  # IQR method
                        # Cap at IQR bounds
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr = q3 - q1
                        lower_cap = q1 - (self.iqr_multiplier * iqr)
                        upper_cap = q3 + (self.iqr_multiplier * iqr)
                    
                    # Apply capping
                    processed_data[column] = processed_data[column].clip(
                        lower=lower_cap, 
                        upper=upper_cap
                    )
                    
                    outlier_stats[column].update({
                        'lower_cap': lower_cap,
                        'upper_cap': upper_cap,
                        'values_capped_low': (series < lower_cap).sum(),
                        'values_capped_high': (series > upper_cap).sum()
                    })
            
            # Update metrics
            stage_time = time.time() - stage_start
            self.cleaning_metrics['outliers_capped'] = total_outliers_detected
            self.cleaning_metrics['rows_modified'] += total_outliers_detected
            self.cleaning_metrics['stage_times']['outlier_capping'] = stage_time
            
            # Store outlier statistics
            self.metadata['outlier_capping_stats'] = outlier_stats
            self.metadata['total_outliers_capped'] = total_outliers_detected
            self.metadata['outlier_detection_method'] = method
            overall_outlier_rate = total_outliers_detected / (original_shape[0] * len(columns)) if original_shape[0] > 0 and columns else 0
            self.metadata['overall_outlier_rate'] = overall_outlier_rate
            
            # Emit event with metrics
            self._emit_cleaning_event(
                'data.cleaning.outliers_capped',
                {
                    'total_outliers_capped': total_outliers_detected,
                    'detection_method': method,
                    'overall_outlier_rate': overall_outlier_rate,
                    'columns_processed': len(columns),
                    'column_stats': outlier_stats,
                    'processing_time': stage_time
                }
            )
            
            if self.enable_detailed_logging and total_outliers_detected > 0:
                logger.info(f"Capped {total_outliers_detected} total outliers using {method} method (rate: {overall_outlier_rate:.2%})")
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error capping outliers: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)
    
    def _emit_cleaning_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Emit a cleaning event with metrics to the EventSystem.
        
        Args:
            event_type: Type of the cleaning event
            event_data: Data to include in the event
        """
        if self.event_system:
            try:
                self.event_system.emit(
                    event_type=event_type,
                    data={
                        'cleaning_stage': event_type.split('.')[-1],
                        'timestamp': datetime.now().isoformat(),
                        'cleaner_config': {
                            'z_score_threshold': self.z_score_threshold,
                            'iqr_multiplier': self.iqr_multiplier,
                            'max_outlier_percentage': self.max_outlier_percentage,
                            'enable_detailed_logging': self.enable_detailed_logging
                        },
                        **event_data
                    },
                    source="market_data_cleaner"
                )
            except Exception as e:
                logger.warning(f"Failed to emit cleaning event '{event_type}': {str(e)}")
    
    def _calculate_data_quality_score(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> float:
        """Calculate a data quality score based on cleaning operations.
        
        Args:
            original_data: Original DataFrame before cleaning
            cleaned_data: Cleaned DataFrame after processing
            
        Returns:
            Data quality score between 0.0 and 1.0
        """
        try:
            if original_data.empty:
                return 1.0
            
            # Calculate various quality metrics
            original_size = original_data.shape[0] * original_data.shape[1]
            cleaned_size = cleaned_data.shape[0] * cleaned_data.shape[1]
            
            # Penalize for data loss (rows removed)
            data_retention_score = cleaned_size / original_size if original_size > 0 else 1.0
            
            # Reward for fixing bad values
            bad_casts_corrected = self.cleaning_metrics.get('bad_casts_corrected', 0)
            gaps_filled = self.cleaning_metrics.get('gaps_filled', 0)
            improvement_score = min(1.0, (bad_casts_corrected + gaps_filled) / (original_size * 0.1))
            
            # Penalize for high outlier rates
            outlier_rate = self.metadata.get('overall_outlier_rate', 0)
            outlier_penalty = max(0.0, 1.0 - (outlier_rate / self.max_outlier_percentage))
            
            # Calculate composite score
            quality_score = (
                data_retention_score * 0.5 +  # 50% weight on data retention
                improvement_score * 0.3 +      # 30% weight on improvements
                outlier_penalty * 0.2          # 20% weight on outlier management
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate data quality score: {str(e)}")
            return 0.5  # Default neutral score
    
    def clean_market_data(
        self,
        data: pd.DataFrame,
        outlier_method: Optional[str] = None,
        columns_to_clean: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Complete market data cleaning pipeline with enhanced metrics.
        
        This method runs the full cleaning process:
        1. Bad numeric cast detection and correction
        2. Type coercion to float64
        3. Duplicate removal based on timestamp & symbol
        4. Outlier capping using specified method
        5. Gap filling using forward-fill then back-fill
        
        Args:
            data: Input DataFrame to clean
            outlier_method: Method for outlier detection ('zscore' or 'iqr')
            columns_to_clean: Specific columns to clean (default: all numeric columns)
            
        Returns:
            Cleaned DataFrame with comprehensive metrics
            
        Raises:
            DataProcessingError: If cleaning process fails
        """
        try:
            import time
            pipeline_start = time.time()
            
            if self.enable_detailed_logging:
                logger.info("Starting enhanced market data cleaning pipeline")
            
            # Reset metrics for this run
            self.cleaning_metrics = {
                'duplicates_removed': 0,
                'outliers_capped': 0,
                'bad_casts_corrected': 0,
                'gaps_filled': 0,
                'rows_modified': 0,
                'total_processing_time': 0.0,
                'stage_times': {},
                'data_quality_score': 0.0
            }
            
            # Store original data reference for quality calculation
            original_data = data.copy()
            original_shape = data.shape
            
            # Use the base class process_data method which will run all registered steps
            cleaned_data = self.process_data(data)
            
            # Calculate final metrics
            pipeline_time = time.time() - pipeline_start
            final_shape = cleaned_data.shape
            
            # Update total processing time
            self.cleaning_metrics['total_processing_time'] = pipeline_time
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(original_data, cleaned_data)
            self.cleaning_metrics['data_quality_score'] = quality_score
            
            # Store comprehensive cleaning summary
            self.metadata['cleaning_summary'] = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'rows_removed': original_shape[0] - final_shape[0],
                'columns_processed': len(self.numeric_columns),
                'outlier_method_used': outlier_method or self.config.get('data.cleaning.outlier_method', 'iqr'),
                'processing_timestamp': datetime.now().isoformat(),
                'total_processing_time': pipeline_time,
                'data_quality_score': quality_score
            }
            
            # Emit comprehensive pipeline completion event
            self._emit_cleaning_event(
                'data.cleaning.pipeline_completed',
                {
                    'original_shape': original_shape,
                    'final_shape': final_shape,
                    'cleaning_metrics': self.cleaning_metrics,
                    'processing_summary': self.metadata['cleaning_summary'],
                    'success': True
                }
            )
            
            if self.enable_detailed_logging:
                logger.info(f"Market data cleaning completed successfully. Shape: {original_shape} -> {final_shape}")
                logger.info(f"Quality score: {quality_score:.3f}, Processing time: {pipeline_time:.2f}s")
                logger.info(f"Operations performed - Duplicates: {self.cleaning_metrics['duplicates_removed']}, "
                           f"Outliers: {self.cleaning_metrics['outliers_capped']}, "
                           f"Bad casts: {self.cleaning_metrics['bad_casts_corrected']}, "
                           f"Gaps filled: {self.cleaning_metrics['gaps_filled']}")
            
            return cleaned_data
            
        except Exception as e:
            error_msg = f"Error in market data cleaning pipeline: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Emit error event
            self._emit_cleaning_event(
                'data.cleaning.pipeline_failed',
                {
                    'error_message': error_msg,
                    'partial_metrics': self.cleaning_metrics,
                    'success': False
                }
            )
            
            raise DataProcessingError(error_msg)
    
    def clean_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clean data using the market data cleaning pipeline.
        
        This method provides the standard interface expected by the DataPipeline.
        It delegates to the clean_market_data method.
        
        Args:
            data: Input DataFrame to clean
            **kwargs: Additional keyword arguments (outlier_method, columns_to_clean, etc.)
            
        Returns:
            Cleaned DataFrame
        """
        return self.clean_market_data(data, **kwargs)
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report with enhanced metrics.
        
        Returns:
            Dictionary containing detailed cleaning statistics and metadata
        """
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'cleaner_config': {
                'symbol_column': self.symbol_column,
                'timestamp_column': self.timestamp_column,
                'numeric_columns': self.numeric_columns,
                'z_score_threshold': self.z_score_threshold,
                'iqr_multiplier': self.iqr_multiplier,
                'max_outlier_percentage': self.max_outlier_percentage,
                'min_numeric_cast_success_rate': self.min_numeric_cast_success_rate,
                'gap_fill_max_consecutive': self.gap_fill_max_consecutive,
                'enable_detailed_logging': self.enable_detailed_logging
            },
            'cleaning_metrics': self.cleaning_metrics.copy(),
            'metadata': self.metadata.copy()
        }
        
        # Enhanced operations summary
        operations = {
            'duplicates_removed': self.cleaning_metrics.get('duplicates_removed', 0),
            'outliers_capped': self.cleaning_metrics.get('outliers_capped', 0),
            'bad_casts_corrected': self.cleaning_metrics.get('bad_casts_corrected', 0),
            'gaps_filled': self.cleaning_metrics.get('gaps_filled', 0),
            'total_rows_modified': self.cleaning_metrics.get('rows_modified', 0),
            'na_from_coercion': self.metadata.get('na_values_introduced_by_coercion', 0)
        }
        report['operations_summary'] = operations
        
        # Performance summary
        performance = {
            'total_processing_time': self.cleaning_metrics.get('total_processing_time', 0.0),
            'stage_times': self.cleaning_metrics.get('stage_times', {}),
            'data_quality_score': self.cleaning_metrics.get('data_quality_score', 0.0)
        }
        report['performance_summary'] = performance
        
        # Quality assessment
        quality_assessment = {
            'overall_quality_score': self.cleaning_metrics.get('data_quality_score', 0.0),
            'outlier_rate': self.metadata.get('overall_outlier_rate', 0.0),
            'duplicate_removal_rate': self.metadata.get('duplicate_removal_rate', 0.0),
            'gap_fill_success_rate': self.metadata.get('gap_filling', {}).get('fill_success_rate', 1.0)
        }
        report['quality_assessment'] = quality_assessment
        
        return report


def build_market_data_cleaner(
    config: Optional[ConfigManager] = None,
    event_system: Optional[EventSystem] = None,
    symbol_column: Optional[str] = None,
    timestamp_column: Optional[str] = None, 
    numeric_columns: Optional[List[str]] = None,
    outlier_method: Optional[str] = None,
    z_score_threshold: Optional[float] = None,
    iqr_multiplier: Optional[float] = None,
    enable_detailed_logging: Optional[bool] = None
) -> MarketDataCleaner:
    """Factory function to build a configured MarketDataCleaner with ConfigManager integration.
    
    Args:
        config: ConfigManager instance (will create default if None)
        event_system: EventSystem for emitting cleaning metrics
        symbol_column: Name of symbol column (uses config default if None)
        timestamp_column: Name of timestamp column (uses config default if None)
        numeric_columns: List of numeric columns to process (uses config default if None)
        outlier_method: Default outlier detection method (uses config default if None)
        z_score_threshold: Z-score threshold for outlier detection (uses config default if None)
        iqr_multiplier: IQR multiplier for outlier detection (uses config default if None)
        enable_detailed_logging: Enable detailed logging (uses config default if None)
        
    Returns:
        Configured MarketDataCleaner instance
    """
    # Create config if not provided
    if config is None:
        config = ConfigManager()
    
    # Build cleaner with all parameters
    cleaner = MarketDataCleaner(
        config=config,
        event_system=event_system,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        numeric_columns=numeric_columns,
        z_score_threshold=z_score_threshold,
        iqr_multiplier=iqr_multiplier
    )
    
    # Set detailed logging if specified
    if enable_detailed_logging is not None:
        cleaner.enable_detailed_logging = enable_detailed_logging
    
    method = outlier_method or config.get('data.cleaning.outlier_method', 'iqr')
    logger.info(f"Built enhanced MarketDataCleaner with {method} outlier detection and event integration")
    
    return cleaner
