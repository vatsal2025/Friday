"""Data Retrieval Utilities for the Friday AI Trading System.

This module provides utilities for retrieving and preparing data for
downstream model training and analysis workflows.
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.storage.data_storage import DataStorage, StorageError
from src.data.storage.storage_factory import get_default_storage, get_training_storage
from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class DataRetrievalError(Exception):
    """Exception raised for errors in data retrieval operations."""
    pass


class DataRetrievalUtils:
    """Utilities for retrieving and preparing data for model training.
    
    This class provides methods for efficient data loading, filtering,
    preprocessing, and splitting for machine learning workflows.
    
    Attributes:
        storage: Data storage instance.
        config: Configuration manager.
        cache: In-memory cache for frequently accessed data.
    """
    
    def __init__(
        self, 
        storage: Optional[DataStorage] = None,
        config: Optional[ConfigManager] = None
    ):
        """Initialize DataRetrievalUtils.
        
        Args:
            storage: Data storage instance. If None, uses default storage.
            config: Configuration manager instance.
        """
        self.storage = storage or get_training_storage()
        self.config = config or ConfigManager()
        
        # Load retrieval configuration
        retrieval_config = self.config.get("data.storage.retrieval", {})
        self.batch_size = retrieval_config.get("batch_size", 10000)
        self.parallel_loading = retrieval_config.get("parallel_loading", True)
        self.max_workers = retrieval_config.get("max_workers", 4)
        self.cache_enabled = retrieval_config.get("cache_enabled", True)
        self.cache_ttl = retrieval_config.get("cache_ttl", 3600)
        self.prefetch_enabled = retrieval_config.get("prefetch_enabled", True)
        self.optimize_for_ml = retrieval_config.get("optimize_for_ml", True)
        
        # Initialize cache
        self.cache = {} if self.cache_enabled else None
        self.cache_timestamps = {} if self.cache_enabled else None
        
        logger.info("Initialized DataRetrievalUtils")
    
    def get_training_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data",
        features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Get training data with automatic train/validation/test splits.
        
        Args:
            symbols: Symbol or list of symbols to retrieve.
            start_date: Start date for data retrieval (YYYY-MM-DD).
            end_date: End date for data retrieval (YYYY-MM-DD).
            table_name: Name of the table to retrieve from.
            features: List of feature columns to include.
            target_column: Name of the target column for supervised learning.
            test_size: Proportion of data to use for testing.
            validation_size: Proportion of data to use for validation.
            random_state: Random state for reproducible splits.
            **kwargs: Additional arguments for data retrieval.
            
        Returns:
            Dict containing 'train', 'validation', 'test' DataFrames.
        """
        try:
            # Retrieve raw data
            data = self.get_symbol_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                table_name=table_name,
                **kwargs
            )
            
            if data.empty:
                raise DataRetrievalError("No data retrieved for specified criteria")
            
            # Filter features if specified
            if features:
                available_features = [col for col in features if col in data.columns]
                if target_column and target_column not in available_features:
                    available_features.append(target_column)
                
                if len(available_features) != len(features):
                    missing_features = set(features) - set(available_features)
                    logger.warning(f"Missing features: {missing_features}")
                
                data = data[available_features]
            
            # Prepare data for ML if enabled
            if self.optimize_for_ml:
                data = self._prepare_for_ml(data, target_column)
            
            # Create splits
            train_data, temp_data = train_test_split(
                data, 
                test_size=(test_size + validation_size), 
                random_state=random_state,
                shuffle=False  # Preserve time series order
            )
            
            if validation_size > 0:
                # Calculate validation size relative to temp_data
                val_size_relative = validation_size / (test_size + validation_size)
                validation_data, test_data = train_test_split(
                    temp_data,
                    test_size=(1 - val_size_relative),
                    random_state=random_state,
                    shuffle=False
                )
            else:
                validation_data = pd.DataFrame()
                test_data = temp_data
            
            result = {
                'train': train_data,
                'validation': validation_data,
                'test': test_data,
                'full': data
            }
            
            # Log split information
            logger.info(f"Data splits - Train: {len(train_data)}, "
                       f"Validation: {len(validation_data)}, Test: {len(test_data)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            raise DataRetrievalError(f"Error getting training data: {str(e)}")
    
    def get_symbol_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data",
        columns: Optional[List[str]] = None,
        use_cache: bool = None,
        **kwargs
    ) -> pd.DataFrame:
        """Get data for specified symbols and date range.
        
        Args:
            symbols: Symbol or list of symbols to retrieve.
            start_date: Start date for data retrieval (YYYY-MM-DD).
            end_date: End date for data retrieval (YYYY-MM-DD).
            table_name: Name of the table to retrieve from.
            columns: List of columns to retrieve.
            use_cache: Whether to use cache. If None, uses instance setting.
            **kwargs: Additional arguments for data retrieval.
            
        Returns:
            pd.DataFrame: Retrieved data.
        """
        try:
            # Normalize symbols to list
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Check cache if enabled
            use_cache = use_cache if use_cache is not None else self.cache_enabled
            cache_key = self._get_cache_key(symbols, start_date, end_date, table_name, columns)
            
            if use_cache and self._is_cache_valid(cache_key):
                logger.debug(f"Using cached data for key: {cache_key}")
                return self.cache[cache_key].copy()
            
            # Retrieve data
            if self.parallel_loading and len(symbols) > 1:
                data = self._parallel_retrieve_symbols(
                    symbols, start_date, end_date, table_name, columns, **kwargs
                )
            else:
                data = self._sequential_retrieve_symbols(
                    symbols, start_date, end_date, table_name, columns, **kwargs
                )
            
            # Apply date filtering if needed
            if start_date or end_date:
                data = self._filter_by_date(data, start_date, end_date)
            
            # Cache result if enabled
            if use_cache and not data.empty:
                self._cache_data(cache_key, data)
            
            logger.info(f"Retrieved {len(data)} rows for {len(symbols)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving symbol data: {str(e)}")
            raise DataRetrievalError(f"Error retrieving symbol data: {str(e)}")
    
    def get_batch_iterator(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data",
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Get an iterator for processing data in batches.
        
        Args:
            symbols: Symbol or list of symbols to retrieve.
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            table_name: Name of the table to retrieve from.
            batch_size: Size of each batch. If None, uses instance setting.
            **kwargs: Additional arguments for data retrieval.
            
        Yields:
            pd.DataFrame: Batches of data.
        """
        batch_size = batch_size or self.batch_size
        
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for symbol in symbols:
            try:
                # Get all data for the symbol
                symbol_data = self.storage.retrieve_data(
                    table_name=table_name,
                    symbol=symbol,
                    **kwargs
                )
                
                if symbol_data.empty:
                    continue
                
                # Apply date filtering
                if start_date or end_date:
                    symbol_data = self._filter_by_date(symbol_data, start_date, end_date)
                
                # Yield batches
                for i in range(0, len(symbol_data), batch_size):
                    batch = symbol_data.iloc[i:i + batch_size]
                    yield batch
                    
            except Exception as e:
                logger.warning(f"Error processing symbol {symbol}: {str(e)}")
                continue
    
    def get_feature_matrix(
        self,
        symbols: Union[str, List[str]],
        feature_columns: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data",
        normalize: bool = True,
        scaler_type: str = "standard",
        handle_missing: str = "drop",
        **kwargs
    ) -> Tuple[np.ndarray, List[str], Optional[object]]:
        """Get feature matrix ready for machine learning.
        
        Args:
            symbols: Symbol or list of symbols to retrieve.
            feature_columns: List of feature column names.
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            table_name: Name of the table to retrieve from.
            normalize: Whether to normalize the features.
            scaler_type: Type of scaler ("standard", "minmax").
            handle_missing: How to handle missing values ("drop", "fill").
            **kwargs: Additional arguments for data retrieval.
            
        Returns:
            Tuple of (feature_matrix, feature_names, scaler).
        """
        try:
            # Get data
            data = self.get_symbol_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                table_name=table_name,
                columns=feature_columns,
                **kwargs
            )
            
            if data.empty:
                raise DataRetrievalError("No data available for feature extraction")
            
            # Filter to feature columns only
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) != len(feature_columns):
                missing_features = set(feature_columns) - set(available_features)
                logger.warning(f"Missing feature columns: {missing_features}")
            
            feature_data = data[available_features]
            
            # Handle missing values
            if handle_missing == "drop":
                feature_data = feature_data.dropna()
            elif handle_missing == "fill":
                feature_data = feature_data.fillna(method='forward').fillna(method='backward')
            
            # Convert to numpy array
            feature_matrix = feature_data.values
            
            # Normalize if requested
            scaler = None
            if normalize:
                if scaler_type == "standard":
                    scaler = StandardScaler()
                elif scaler_type == "minmax":
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaler type: {scaler_type}")
                
                feature_matrix = scaler.fit_transform(feature_matrix)
            
            logger.info(f"Created feature matrix: {feature_matrix.shape}")
            return feature_matrix, available_features, scaler
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            raise DataRetrievalError(f"Error creating feature matrix: {str(e)}")
    
    def get_time_series_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data",
        sequence_length: int = 60,
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get time series data formatted for sequence models.
        
        Args:
            symbol: Symbol to retrieve.
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            table_name: Name of the table to retrieve from.
            sequence_length: Length of input sequences.
            target_column: Column to use as target.
            feature_columns: Columns to use as features.
            **kwargs: Additional arguments for data retrieval.
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets.
        """
        try:
            # Get data for single symbol
            data = self.get_symbol_data(
                symbols=symbol,
                start_date=start_date,
                end_date=end_date,
                table_name=table_name,
                **kwargs
            )
            
            if len(data) < sequence_length + 1:
                raise DataRetrievalError(f"Insufficient data for sequence length {sequence_length}")
            
            # Select feature columns
            if feature_columns:
                feature_data = data[feature_columns]
            else:
                # Use numeric columns excluding target
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numeric_cols:
                    numeric_cols.remove(target_column)
                feature_data = data[numeric_cols]
            
            # Get target data
            target_data = data[target_column].values
            
            # Create sequences
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(feature_data.iloc[i:i + sequence_length].values)
                y.append(target_data[i + sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created time series data: X{X.shape}, y{y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating time series data: {str(e)}")
            raise DataRetrievalError(f"Error creating time series data: {str(e)}")
    
    def _parallel_retrieve_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        table_name: str,
        columns: Optional[List[str]],
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data for multiple symbols in parallel."""
        data_frames = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(
                    self._retrieve_single_symbol,
                    symbol, start_date, end_date, table_name, columns, **kwargs
                ): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_data = future.result()
                    if not symbol_data.empty:
                        data_frames.append(symbol_data)
                except Exception as e:
                    logger.warning(f"Error retrieving data for {symbol}: {str(e)}")
        
        # Combine all data
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _sequential_retrieve_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        table_name: str,
        columns: Optional[List[str]],
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data for multiple symbols sequentially."""
        data_frames = []
        
        for symbol in symbols:
            try:
                symbol_data = self._retrieve_single_symbol(
                    symbol, start_date, end_date, table_name, columns, **kwargs
                )
                if not symbol_data.empty:
                    data_frames.append(symbol_data)
            except Exception as e:
                logger.warning(f"Error retrieving data for {symbol}: {str(e)}")
        
        # Combine all data
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _retrieve_single_symbol(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        table_name: str,
        columns: Optional[List[str]],
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data for a single symbol."""
        return self.storage.retrieve_data(
            table_name=table_name,
            columns=columns,
            symbol=symbol,
            **kwargs
        )
    
    def _filter_by_date(
        self,
        data: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter data by date range."""
        if data.empty:
            return data
        
        # Find date column
        date_columns = ['date', 'timestamp', 'datetime']
        date_col = None
        
        for col in date_columns:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No date column found for filtering")
            return data
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Apply filters
        filtered_data = data.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data[date_col] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data[date_col] <= end_dt]
        
        return filtered_data
    
    def _prepare_for_ml(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare data for machine learning."""
        prepared_data = data.copy()
        
        # Handle missing values
        numeric_columns = prepared_data.select_dtypes(include=[np.number]).columns
        prepared_data[numeric_columns] = prepared_data[numeric_columns].fillna(
            prepared_data[numeric_columns].mean()
        )
        
        # Handle categorical columns
        categorical_columns = prepared_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:  # Don't encode target if categorical
                prepared_data[col] = prepared_data[col].fillna('Unknown')
        
        # Remove rows with missing target values
        if target_column and target_column in prepared_data.columns:
            prepared_data = prepared_data.dropna(subset=[target_column])
        
        return prepared_data
    
    def _get_cache_key(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        table_name: str,
        columns: Optional[List[str]]
    ) -> str:
        """Generate cache key for data request."""
        key_parts = [
            "_".join(sorted(symbols)),
            start_date or "None",
            end_date or "None",
            table_name,
            "_".join(sorted(columns)) if columns else "None"
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if not self.cache_enabled or cache_key not in self.cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now().timestamp() - self.cache_timestamps[cache_key]
        return cache_age < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp."""
        if not self.cache_enabled:
            return
        
        self.cache[cache_key] = data.copy()
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        if self.cache_enabled:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cleared data retrieval cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        total_entries = len(self.cache)
        total_memory_mb = sum(
            data.memory_usage(deep=True).sum() for data in self.cache.values()
        ) / (1024 * 1024)
        
        return {
            "cache_enabled": True,
            "total_entries": total_entries,
            "total_memory_mb": round(total_memory_mb, 2),
            "cache_ttl": self.cache_ttl
        }


# Convenience functions
def get_training_data(
    symbols: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """Get training data with automatic splits.
    
    Args:
        symbols: Symbol or list of symbols to retrieve.
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        **kwargs: Additional arguments for DataRetrievalUtils.get_training_data.
        
    Returns:
        Dict containing train/validation/test splits.
    """
    utils = DataRetrievalUtils()
    return utils.get_training_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def get_feature_matrix(
    symbols: Union[str, List[str]],
    feature_columns: List[str],
    **kwargs
) -> Tuple[np.ndarray, List[str], Optional[object]]:
    """Get feature matrix for machine learning.
    
    Args:
        symbols: Symbol or list of symbols to retrieve.
        feature_columns: List of feature column names.
        **kwargs: Additional arguments for DataRetrievalUtils.get_feature_matrix.
        
    Returns:
        Tuple of (feature_matrix, feature_names, scaler).
    """
    utils = DataRetrievalUtils()
    return utils.get_feature_matrix(
        symbols=symbols,
        feature_columns=feature_columns,
        **kwargs
    )
