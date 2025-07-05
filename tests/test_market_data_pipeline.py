#!/usr/bin/env python3
"""Comprehensive test script for the Friday AI Trading System Market Data Pipeline.

This test script provides comprehensive testing for the market data pipeline including:
1. Sample real market CSV (AAPL 1-min for one week)
2. Controlled error injection for negative prices, duplicates, NaNs
3. Testing both strict and warn-only modes
4. Verification of generated features with known values
5. Complete integration testing

Run with:
    pytest tests/test_market_data_pipeline.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, time
import tempfile
import shutil
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pipelines.market_data_pipeline import MarketDataPipeline
from src.data.integration.data_pipeline import PipelineError
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class TestMarketDataPipeline:
    """Comprehensive test suite for the Market Data Pipeline."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method called before each test."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data_path = Path(self.temp_dir) / "sample_aapl_data.csv"
        self.processed_data_dir = Path(self.temp_dir) / "processed"
        self.processed_data_dir.mkdir(exist_ok=True)
        
        # Create sample AAPL data
        self.sample_data = self._create_sample_aapl_data()
        self.sample_data.to_csv(self.sample_data_path, index=False)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_aapl_data(self) -> pd.DataFrame:
        """Create realistic sample AAPL 1-minute data for one week.
        
        Returns:
            pd.DataFrame: Sample market data with OHLCV columns
        """
        # Create one week of 1-minute data during market hours (9:30 AM - 4:00 PM EST)
        start_date = datetime(2024, 1, 2, 9, 30)  # Tuesday, Jan 2, 2024, 9:30 AM
        
        # Generate timestamps for 5 trading days (Mon-Fri), 6.5 hours per day, 1-minute intervals
        timestamps = []
        current_date = start_date
        
        for day in range(5):  # 5 trading days
            day_start = current_date + timedelta(days=day)
            day_start = day_start.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Generate timestamps for trading hours: 9:30 AM to 4:00 PM (390 minutes)
            for minute in range(390):
                timestamp = day_start + timedelta(minutes=minute)
                timestamps.append(timestamp)
        
        n_rows = len(timestamps)
        
        # Generate realistic AAPL price movements
        np.random.seed(42)  # For reproducible results
        
        # Base price around $150 (typical AAPL price range)
        base_price = 150.0
        
        # Generate price changes with some volatility and trending
        # Use random walk with slight upward bias
        price_changes = np.random.normal(0.001, 0.005, n_rows)  # Small upward bias with volatility
        price_multipliers = np.exp(np.cumsum(price_changes))
        
        # Calculate closing prices
        closes = base_price * price_multipliers
        
        # Generate realistic OHLC from closes
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        # Add some intraday volatility
        intraday_vol = np.random.normal(0, 0.002, n_rows)
        highs = np.maximum(opens, closes) * (1 + np.abs(intraday_vol))
        lows = np.minimum(opens, closes) * (1 - np.abs(intraday_vol))
        
        # Ensure OHLC relationships are valid
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generate realistic volumes (higher at open/close, lower mid-day)
        base_volume = 1000000
        volume_pattern = np.sin(np.linspace(0, np.pi, 390))  # U-shaped pattern within each day
        volumes = []
        
        for day in range(5):
            start_idx = day * 390
            end_idx = (day + 1) * 390
            day_volumes = base_volume * (0.5 + 0.5 * volume_pattern) * np.random.lognormal(0, 0.3, 390)
            volumes.extend(day_volumes)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': 'AAPL',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return data

    def _inject_negative_prices(self, data: pd.DataFrame, num_errors: int = 3) -> pd.DataFrame:
        """Inject negative price errors into the dataset.
        
        Args:
            data: Original DataFrame
            num_errors: Number of negative price errors to inject
            
        Returns:
            pd.DataFrame: Data with negative price errors
        """
        corrupted_data = data.copy()
        
        # Randomly select rows to corrupt
        error_indices = np.random.choice(len(data), size=num_errors, replace=False)
        
        for idx in error_indices:
            # Randomly choose which price column to corrupt
            price_col = np.random.choice(['open', 'high', 'low', 'close'])
            corrupted_data.loc[idx, price_col] = -abs(corrupted_data.loc[idx, price_col])
        
        return corrupted_data

    def _inject_duplicate_timestamps(self, data: pd.DataFrame, num_duplicates: int = 2) -> pd.DataFrame:
        """Inject duplicate timestamp errors into the dataset.
        
        Args:
            data: Original DataFrame
            num_duplicates: Number of duplicate timestamp errors to inject
            
        Returns:
            pd.DataFrame: Data with duplicate timestamps
        """
        corrupted_data = data.copy()
        
        # Select random indices to duplicate
        dup_indices = np.random.choice(len(data) - 1, size=num_duplicates, replace=False)
        
        for idx in dup_indices:
            # Set the next row's timestamp to be the same as current row
            corrupted_data.loc[idx + 1, 'timestamp'] = corrupted_data.loc[idx, 'timestamp']
        
        return corrupted_data

    def _inject_nan_values(self, data: pd.DataFrame, num_nans: int = 5) -> pd.DataFrame:
        """Inject NaN values into the dataset.
        
        Args:
            data: Original DataFrame
            num_nans: Number of NaN values to inject
            
        Returns:
            pd.DataFrame: Data with NaN values
        """
        corrupted_data = data.copy()
        
        # Randomly select rows and columns to corrupt with NaN
        error_indices = np.random.choice(len(data), size=num_nans, replace=False)
        
        for idx in error_indices:
            # Randomly choose which column to corrupt
            col = np.random.choice(['open', 'high', 'low', 'close', 'volume'])
            corrupted_data.loc[idx, col] = np.nan
        
        return corrupted_data

    def _create_all_error_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create dataset with all types of errors for comprehensive testing.
        
        Args:
            data: Original DataFrame
            
        Returns:
            pd.DataFrame: Data with all error types injected
        """
        corrupted_data = data.copy()
        
        # Inject different types of errors
        corrupted_data = self._inject_negative_prices(corrupted_data, num_errors=2)
        corrupted_data = self._inject_duplicate_timestamps(corrupted_data, num_duplicates=2)
        corrupted_data = self._inject_nan_values(corrupted_data, num_nans=3)
        
        return corrupted_data

    def test_load_sample_real_market_csv(self):
        """Test loading sample real market CSV data."""
        # Verify sample data was created correctly
        assert self.sample_data_path.exists()
        
        # Load and verify the data
        loaded_data = pd.read_csv(self.sample_data_path)
        
        # Check basic properties
        assert len(loaded_data) > 0
        assert set(loaded_data.columns) == {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        assert loaded_data['symbol'].unique().tolist() == ['AAPL']
        assert len(loaded_data) == 5 * 390  # 5 days * 390 minutes per day
        
        # Verify price relationships
        assert (loaded_data['high'] >= loaded_data['open']).all()
        assert (loaded_data['high'] >= loaded_data['close']).all()
        assert (loaded_data['low'] <= loaded_data['open']).all()
        assert (loaded_data['low'] <= loaded_data['close']).all()
        assert (loaded_data['volume'] > 0).all()

    def test_pipeline_strict_mode_with_clean_data(self):
        """Test pipeline in strict mode with clean data - should succeed."""
        # Create pipeline in strict mode
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False  # Disable storage for faster testing
        )
        
        # Process clean data - should succeed
        result = pipeline.process_data(self.sample_data)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        
        # Should have original columns plus generated features
        original_cols = set(self.sample_data.columns)
        result_cols = set(result.columns)
        assert original_cols.issubset(result_cols)
        
        # Should have some generated features
        feature_cols = result_cols - original_cols
        assert len(feature_cols) > 0
        
        # Check that pipeline metadata indicates success
        metadata = pipeline.get_pipeline_metadata()
        assert metadata['last_run_status'] == 'success'

    def test_pipeline_strict_mode_with_negative_prices(self):
        """Test pipeline in strict mode with negative prices - should raise PipelineError."""
        # Create corrupted data with negative prices
        corrupted_data = self._inject_negative_prices(self.sample_data, num_errors=3)
        
        # Create pipeline in strict mode
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process corrupted data - should raise PipelineError
        with pytest.raises(PipelineError) as exc_info:
            pipeline.process_data(corrupted_data)
        
        # Verify the error is related to validation
        assert "validation" in str(exc_info.value).lower() or "negative" in str(exc_info.value).lower()
        
        # Check that pipeline metadata indicates error
        metadata = pipeline.get_pipeline_metadata()
        assert metadata['last_run_status'] == 'error'

    def test_pipeline_strict_mode_with_duplicates(self):
        """Test pipeline in strict mode with duplicate timestamps - should raise PipelineError."""
        # Create corrupted data with duplicate timestamps
        corrupted_data = self._inject_duplicate_timestamps(self.sample_data, num_duplicates=2)
        
        # Create pipeline in strict mode
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process corrupted data - should raise PipelineError
        with pytest.raises(PipelineError) as exc_info:
            pipeline.process_data(corrupted_data)
        
        # Verify the error is related to validation
        assert "validation" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()

    def test_pipeline_strict_mode_with_nans(self):
        """Test pipeline in strict mode with NaN values - should raise PipelineError."""
        # Create corrupted data with NaN values
        corrupted_data = self._inject_nan_values(self.sample_data, num_nans=5)
        
        # Create pipeline in strict mode
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process corrupted data - should raise PipelineError
        with pytest.raises(PipelineError) as exc_info:
            pipeline.process_data(corrupted_data)
        
        # Verify the error is related to validation
        assert "validation" in str(exc_info.value).lower() or "nan" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()

    def test_pipeline_warn_only_mode_with_negative_prices(self):
        """Test pipeline in warn-only mode with negative prices - should return cleaned DataFrame."""
        # Create corrupted data with negative prices
        corrupted_data = self._inject_negative_prices(self.sample_data, num_errors=3)
        
        # Create pipeline in warn-only mode
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process corrupted data - should succeed with warnings
        result = pipeline.process_data(corrupted_data)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should return cleaned data
        
        # Should have expected columns
        expected_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        assert expected_cols.issubset(set(result.columns))
        
        # Check that pipeline completed successfully despite validation warnings
        metadata = pipeline.get_pipeline_metadata()
        assert metadata['last_run_status'] == 'success'

    def test_pipeline_warn_only_mode_with_all_errors(self):
        """Test pipeline in warn-only mode with all error types - should return cleaned DataFrame."""
        # Create corrupted data with all error types
        corrupted_data = self._create_all_error_types(self.sample_data)
        
        # Create pipeline in warn-only mode
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process corrupted data - should succeed with warnings
        result = pipeline.process_data(corrupted_data)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should return cleaned data
        
        # Verify expected shape and columns
        expected_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        assert expected_cols.issubset(set(result.columns))
        
        # Should have generated features
        feature_cols = set(result.columns) - expected_cols
        assert len(feature_cols) > 0
        
        # Check that pipeline completed successfully
        metadata = pipeline.get_pipeline_metadata()
        assert metadata['last_run_status'] == 'success'

    def test_generated_features_match_known_values(self):
        """Test that generated features match known calculated values for specific rows."""
        # Create simple test data with known values
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=20, freq='1min'),
            'symbol': 'AAPL',
            'open': [100.0] * 20,
            'high': [101.0] * 20,
            'low': [99.0] * 20,
            'close': [100.0] * 20,  # Constant price for simple calculations
            'volume': [1000000.0] * 20
        })
        
        # Create pipeline with specific features enabled
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False,
            enable_all_features=False  # We'll enable specific features
        )
        
        # Enable only specific feature sets for testing
        pipeline.pipeline.stages[2]['component'].enable_feature_set("price_derived")
        pipeline.pipeline.stages[2]['component'].enable_feature_set("moving_averages")
        
        # Process data
        result = pipeline.process_data(test_data)
        
        # Verify specific feature calculations
        
        # Test typical_price: (high + low + close) / 3
        expected_typical_price = (101.0 + 99.0 + 100.0) / 3
        assert np.allclose(result['typical_price'], expected_typical_price, rtol=1e-10)
        
        # Test price_avg: (open + high + low + close) / 4
        expected_price_avg = (100.0 + 101.0 + 99.0 + 100.0) / 4
        assert np.allclose(result['price_avg'], expected_price_avg, rtol=1e-10)
        
        # Test SMA calculations (for constant price, SMA should equal the price)
        # Skip NaN values in the beginning due to rolling window
        sma_5_values = result['sma_5'].dropna()
        assert np.allclose(sma_5_values, 100.0, rtol=1e-10)
        
        sma_10_values = result['sma_10'].dropna()
        assert np.allclose(sma_10_values, 100.0, rtol=1e-10)
        
        # Test EMA calculations (for constant price, EMA should equal the price)
        ema_5_values = result['ema_5'].dropna()
        assert np.allclose(ema_5_values, 100.0, rtol=1e-10)

    def test_pipeline_with_varying_prices_features(self):
        """Test feature generation with varying prices to verify calculations."""
        # Create test data with specific price pattern
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=50, freq='1min'),
            'symbol': 'AAPL',
            'open': np.arange(100, 150, 1),  # Increasing prices
            'high': np.arange(101, 151, 1),
            'low': np.arange(99, 149, 1),
            'close': np.arange(100, 150, 1),
            'volume': [1000000.0] * 50
        })
        
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process data
        result = pipeline.process_data(test_data)
        
        # Verify features exist and have reasonable values
        assert 'typical_price' in result.columns
        assert 'price_avg' in result.columns
        assert 'sma_5' in result.columns
        assert 'ema_5' in result.columns
        
        # Check that features are not all NaN
        assert not result['typical_price'].isna().all()
        assert not result['price_avg'].isna().all()
        
        # For increasing prices, moving averages should be less than current price (lagging)
        # Test this for the last few rows where we have enough data
        last_close = result['close'].iloc[-1]
        last_sma_5 = result['sma_5'].iloc[-1]
        assert last_sma_5 < last_close  # SMA should lag behind increasing prices

    def test_pipeline_expected_shape_and_columns_warn_only(self):
        """Test that warn-only mode returns DataFrame with expected shape and columns."""
        # Create data with some errors
        corrupted_data = self._inject_negative_prices(self.sample_data, num_errors=2)
        
        # Create pipeline in warn-only mode
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process data
        result = pipeline.process_data(corrupted_data)
        
        # Verify shape (number of rows should be positive)
        assert len(result) > 0
        assert len(result.columns) > len(corrupted_data.columns)  # Should have additional features
        
        # Verify required columns exist
        required_columns = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        assert required_columns.issubset(set(result.columns))
        
        # Verify feature columns were added
        feature_columns = set(result.columns) - required_columns
        assert len(feature_columns) > 0
        
        # Common expected feature columns
        expected_features = [
            'typical_price', 'price_avg', 'sma_5', 'sma_10', 'ema_5', 'ema_10'
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_pipeline_file_processing(self):
        """Test processing data from file."""
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process file
        loaded_data = pipeline.process_file(str(self.sample_data_path))
        
        # Verify data was loaded correctly
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(self.sample_data)
        assert set(loaded_data.columns) == set(self.sample_data.columns)

    def test_pipeline_summary_metadata(self):
        """Test pipeline summary and metadata functionality."""
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Process data
        result = pipeline.process_data(self.sample_data)
        
        # Get summary
        summary = pipeline.get_pipeline_summary()
        
        # Verify summary contains expected keys
        expected_keys = [
            'pipeline_name', 'total_stages', 'stage_names', 'warn_only_mode',
            'output_directory', 'enable_all_features', 'last_run_status'
        ]
        for key in expected_keys:
            assert key in summary
        
        # Verify summary values
        assert summary['warn_only_mode'] is True
        assert summary['last_run_status'] == 'success'
        assert summary['total_stages'] > 0
        assert len(summary['stage_names']) == summary['total_stages']

    def test_error_handling_file_not_found(self):
        """Test error handling when input file doesn't exist."""
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Try to process non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process_file("non_existent_file.csv")

    def test_missing_required_columns(self):
        """Test error handling when required columns are missing."""
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min'),
            'symbol': 'AAPL',
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            # Missing 'low', 'close', 'volume'
        })
        
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=False,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError) as exc_info:
            pipeline.process_data(incomplete_data)
        
        assert "missing required columns" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_full_pipeline_integration_with_storage(self):
        """Integration test for the full pipeline including storage."""
        # Create pipeline with storage enabled
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=True
        )
        
        # Process sample data
        result = pipeline.process_data(self.sample_data)
        
        # Verify processing succeeded
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Verify storage directory exists (storage stage should create it)
        assert self.processed_data_dir.exists()
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        assert summary['last_run_status'] == 'success'

    def test_performance_benchmarking(self):
        """Test pipeline performance with larger dataset."""
        import time
        
        # Create larger dataset for performance testing
        large_timestamps = pd.date_range('2024-01-01 09:30:00', periods=10000, freq='1min')
        large_data = pd.DataFrame({
            'timestamp': large_timestamps,
            'symbol': 'AAPL',
            'open': np.random.normal(150, 5, 10000),
            'high': np.random.normal(151, 5, 10000),
            'low': np.random.normal(149, 5, 10000),
            'close': np.random.normal(150, 5, 10000),
            'volume': np.random.lognormal(13, 0.5, 10000)
        })
        
        # Ensure OHLC relationships are valid
        large_data['high'] = np.maximum(large_data['high'], 
                                       np.maximum(large_data['open'], large_data['close']))
        large_data['low'] = np.minimum(large_data['low'], 
                                      np.minimum(large_data['open'], large_data['close']))
        
        # Create pipeline
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir=str(self.processed_data_dir),
            enable_storage=False
        )
        
        # Time the processing
        start_time = time.time()
        result = pipeline.process_data(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_data)
        
        # Log performance metrics
        logger.info(f"Processed {len(large_data)} rows in {processing_time:.2f} seconds")
        logger.info(f"Processing rate: {len(large_data)/processing_time:.0f} rows/second")
        
        # Basic performance assertion (should process at least 1000 rows/second)
        assert len(large_data)/processing_time > 1000


if __name__ == "__main__":
    # Run tests with pytest when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
