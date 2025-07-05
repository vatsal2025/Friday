"""Unit tests for MarketDataCleaner class.

This module contains comprehensive tests for the MarketDataCleaner class,
including edge cases and error conditions.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data.processing.market_data_cleaner import (
    MarketDataCleaner,
    build_market_data_cleaner,
    DataProcessingError
)


class TestMarketDataCleaner:
    """Test cases for MarketDataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample market data with various issues
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        # Sample data with deliberate issues
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'open': [100.0, 101.0, np.nan, 103.0, '104.5', 105.0, 106.0, np.nan, 108.0, 109.0,
                    200.0, 201.0, np.nan, 203.0, '204.5', 205.0, 206.0, np.nan, 208.0, 209.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                    210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
                   190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                     202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0],
            'volume': [1000, 1100, 1200, np.nan, 1400, 1500, 1600, 1700, 1800, 1900,
                      2000, 2100, 2200, np.nan, 2400, 2500, 2600, 2700, 2800, 2900],
            'price': [101.0, 102.0, 103.0, 50000.0, 105.0, 106.0, 107.0, -1000.0, 109.0, 110.0,
                     201.0, 202.0, 203.0, 60000.0, 205.0, 206.0, 207.0, -2000.0, 209.0, 210.0]
        })
        
        # Add some duplicate rows
        duplicate_row = self.sample_data.iloc[5].copy()
        self.sample_data = pd.concat([self.sample_data, duplicate_row.to_frame().T], ignore_index=True)
        
        # Create edge case datasets
        self.empty_data = pd.DataFrame()
        
        self.single_row_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['TEST'],
            'open': [100.0],
            'close': [101.0],
            'volume': [1000]
        })
        
        self.all_missing_data = pd.DataFrame({
            'timestamp': dates[:5],
            'symbol': ['TEST'] * 5,
            'open': [np.nan] * 5,
            'close': [np.nan] * 5,
            'volume': [np.nan] * 5
        })
        
        self.all_identical_data = pd.DataFrame({
            'timestamp': dates[:5],
            'symbol': ['TEST'] * 5,
            'open': [100.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })
        
        # Data with extreme outliers
        self.extreme_outliers_data = pd.DataFrame({
            'timestamp': dates[:10],
            'symbol': ['TEST'] * 10,
            'price': [100, 101, 102, 1e10, 104, 105, -1e10, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1e15, 1400, 1500, 0, 1700, 1800, 1900]
        })
        
        # Data with non-numeric values
        self.mixed_types_data = pd.DataFrame({
            'timestamp': dates[:5],
            'symbol': ['TEST'] * 5,
            'open': [100.0, 'invalid', 102.0, '103.5', np.nan],
            'close': ['100.5', 101.0, 'bad_data', 103.0, 104.0],
            'volume': [1000, 1100, 'text', 1300, 1400]
        })
    
    def test_init_default_parameters(self):
        """Test MarketDataCleaner initialization with default parameters."""
        cleaner = MarketDataCleaner()
        
        assert cleaner.symbol_column == "symbol"
        assert cleaner.timestamp_column == "timestamp"
        assert cleaner.numeric_columns == ['open', 'high', 'low', 'close', 'volume', 'price']
        assert cleaner.z_score_threshold == 3.0
        assert cleaner.iqr_multiplier == 1.5
    
    def test_init_custom_parameters(self):
        """Test MarketDataCleaner initialization with custom parameters."""
        custom_columns = ['price', 'vol']
        cleaner = MarketDataCleaner(
            symbol_column="ticker",
            timestamp_column="date",
            numeric_columns=custom_columns,
            z_score_threshold=2.5,
            iqr_multiplier=2.0
        )
        
        assert cleaner.symbol_column == "ticker"
        assert cleaner.timestamp_column == "date"
        assert cleaner.numeric_columns == custom_columns
        assert cleaner.z_score_threshold == 2.5
        assert cleaner.iqr_multiplier == 2.0
    
    def test_coerce_data_types_basic(self):
        """Test basic data type coercion functionality."""
        cleaner = MarketDataCleaner()
        result = cleaner.coerce_data_types(self.mixed_types_data)
        
        # Check that numeric columns are float64
        for col in ['open', 'close', 'volume']:
            if col in result.columns:
                assert result[col].dtype == 'float64'
        
        # Check metadata tracking
        assert 'type_coercion' in cleaner.metadata
        assert 'na_values_introduced_by_coercion' in cleaner.metadata
    
    def test_coerce_data_types_with_string_numbers(self):
        """Test type coercion with string representations of numbers."""
        data = pd.DataFrame({
            'open': ['100.5', '101.2', '102.8'],
            'close': ['100.0', '101.0', '102.0'],
            'volume': ['1000', '1100', '1200']
        })
        
        cleaner = MarketDataCleaner()
        result = cleaner.coerce_data_types(data)
        
        # Should successfully convert string numbers
        assert result['open'].dtype == 'float64'
        assert result['close'].dtype == 'float64'
        assert result['volume'].dtype == 'float64'
        
        # Values should be preserved
        assert result['open'].iloc[0] == 100.5
        assert result['volume'].iloc[0] == 1000.0
    
    def test_coerce_data_types_invalid_data(self):
        """Test type coercion with invalid data that becomes NaN."""
        data = pd.DataFrame({
            'open': ['100.5', 'invalid', '102.8'],
            'volume': ['1000', 'bad_data', '1200']
        })
        
        cleaner = MarketDataCleaner()
        result = cleaner.coerce_data_types(data)
        
        # Invalid data should become NaN
        assert pd.isna(result['open'].iloc[1])
        assert pd.isna(result['volume'].iloc[1])
        
        # Should track NaN introduction
        assert cleaner.metadata['na_values_introduced_by_coercion'] == 2
    
    def test_remove_market_duplicates_basic(self):
        """Test basic duplicate removal functionality."""
        cleaner = MarketDataCleaner()
        result = cleaner.remove_market_duplicates(self.sample_data)
        
        # Should have removed the duplicate
        assert len(result) < len(self.sample_data)
        assert 'market_duplicates_removed' in cleaner.metadata
        assert cleaner.metadata['market_duplicates_removed'] > 0
    
    def test_remove_market_duplicates_with_datetime_index(self):
        """Test duplicate removal with datetime index."""
        data = self.sample_data.copy()
        data = data.set_index('timestamp')
        
        # Add a duplicate with same index and symbol
        duplicate_row = data.iloc[5].copy()
        data = pd.concat([data, duplicate_row.to_frame().T])
        
        cleaner = MarketDataCleaner()
        result = cleaner.remove_market_duplicates(data)
        
        # Should detect and remove duplicates
        assert len(result) < len(data)
        assert 'market_duplicates_removed' in cleaner.metadata
    
    def test_remove_market_duplicates_custom_subset(self):
        """Test duplicate removal with custom subset of columns."""
        cleaner = MarketDataCleaner()
        result = cleaner.remove_market_duplicates(
            self.sample_data, 
            subset=['symbol', 'open']
        )
        
        # Should use custom subset for duplicate detection
        assert 'duplicate_detection_columns' in cleaner.metadata
        assert cleaner.metadata['duplicate_detection_columns'] == ['symbol', 'open']
    
    def test_remove_market_duplicates_no_duplicates(self):
        """Test duplicate removal when no duplicates exist."""
        data = self.sample_data.drop_duplicates()
        cleaner = MarketDataCleaner()
        result = cleaner.remove_market_duplicates(data)
        
        # Should not remove any rows
        assert len(result) == len(data)
        assert cleaner.metadata['market_duplicates_removed'] == 0
    
    def test_impute_missing_values_market_basic(self):
        """Test basic missing value imputation."""
        cleaner = MarketDataCleaner()
        result = cleaner.impute_missing_values_market(self.sample_data)
        
        # Should have no missing values after imputation
        assert not result.isna().any().any()
        
        # Should track imputation metadata
        assert 'missing_values_before_market_imputation' in cleaner.metadata
        assert 'missing_values_after_market_imputation' in cleaner.metadata
        assert 'values_imputed_market' in cleaner.metadata
    
    def test_impute_missing_values_forward_fill_logic(self):
        """Test that forward-fill logic works correctly."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'symbol': ['TEST'] * 5,
            'price': [100.0, np.nan, np.nan, 103.0, np.nan]
        })
        
        cleaner = MarketDataCleaner()
        result = cleaner.impute_missing_values_market(data)
        
        # Forward fill should carry 100.0 to positions 1,2 and 103.0 to position 4
        assert result['price'].iloc[1] == 100.0
        assert result['price'].iloc[2] == 100.0
        assert result['price'].iloc[4] == 103.0
    
    def test_impute_missing_values_backfill_at_start(self):
        """Test back-fill for missing values at the beginning."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'symbol': ['TEST'] * 5,
            'price': [np.nan, np.nan, 100.0, 101.0, 102.0]
        })
        
        cleaner = MarketDataCleaner()
        result = cleaner.impute_missing_values_market(data)
        
        # Back fill should fill first two values with 100.0
        assert result['price'].iloc[0] == 100.0
        assert result['price'].iloc[1] == 100.0
    
    def test_impute_missing_values_all_missing(self):
        """Test imputation when all values are missing."""
        cleaner = MarketDataCleaner()
        result = cleaner.impute_missing_values_market(self.all_missing_data)
        
        # Should still have missing values if all were missing
        assert result.isna().any().any()
        
        # Should track that no imputation was possible
        remaining_missing = sum(cleaner.metadata['missing_values_after_market_imputation'].values())
        assert remaining_missing > 0
    
    def test_detect_outliers_zscore_basic(self):
        """Test z-score outlier detection."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is clear outlier
        cleaner = MarketDataCleaner(z_score_threshold=2.0)
        
        outliers = cleaner.detect_outliers_zscore(series)
        
        # Should detect the extreme value
        assert outliers.iloc[-1] == True  # 100 should be outlier
        assert outliers.iloc[0] == False  # 1 should not be outlier
    
    def test_detect_outliers_zscore_no_variation(self):
        """Test z-score detection with no variation in data."""
        series = pd.Series([100.0] * 10)  # All identical values
        cleaner = MarketDataCleaner()
        
        outliers = cleaner.detect_outliers_zscore(series)
        
        # Should detect no outliers when no variation
        assert not outliers.any()
    
    def test_detect_outliers_zscore_custom_threshold(self):
        """Test z-score detection with custom threshold."""
        series = pd.Series([1, 2, 3, 4, 5, 10])
        cleaner = MarketDataCleaner()
        
        # More strict threshold
        outliers_strict = cleaner.detect_outliers_zscore(series, threshold=1.5)
        # Less strict threshold  
        outliers_lenient = cleaner.detect_outliers_zscore(series, threshold=3.0)
        
        # Strict should detect more outliers
        assert outliers_strict.sum() >= outliers_lenient.sum()
    
    def test_detect_outliers_iqr_basic(self):
        """Test IQR outlier detection."""
        # Create data with clear outliers
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is outlier
        cleaner = MarketDataCleaner(iqr_multiplier=1.5)
        
        outliers = cleaner.detect_outliers_iqr(series)
        
        # Should detect the extreme value
        assert outliers.iloc[-1] == True  # 100 should be outlier
    
    def test_detect_outliers_iqr_custom_multiplier(self):
        """Test IQR detection with custom multiplier."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
        cleaner = MarketDataCleaner()
        
        # More strict multiplier
        outliers_strict = cleaner.detect_outliers_iqr(series, multiplier=1.0)
        # Less strict multiplier
        outliers_lenient = cleaner.detect_outliers_iqr(series, multiplier=3.0)
        
        # Strict should detect more outliers
        assert outliers_strict.sum() >= outliers_lenient.sum()
    
    def test_cap_outliers_market_zscore(self):
        """Test outlier capping using z-score method."""
        cleaner = MarketDataCleaner(z_score_threshold=2.0)
        result = cleaner.cap_outliers_market(
            self.extreme_outliers_data, 
            method="zscore"
        )
        
        # Extreme values should be capped
        original_max = self.extreme_outliers_data['price'].max()
        original_min = self.extreme_outliers_data['price'].min()
        capped_max = result['price'].max()
        capped_min = result['price'].min()
        
        assert capped_max < original_max
        assert capped_min > original_min
        
        # Should track outlier statistics
        assert 'outlier_capping_stats' in cleaner.metadata
        assert 'total_outliers_capped' in cleaner.metadata
    
    def test_cap_outliers_market_iqr(self):
        """Test outlier capping using IQR method."""
        cleaner = MarketDataCleaner(iqr_multiplier=1.5)
        result = cleaner.cap_outliers_market(
            self.extreme_outliers_data, 
            method="iqr"
        )
        
        # Extreme values should be capped
        original_max = self.extreme_outliers_data['price'].max()
        original_min = self.extreme_outliers_data['price'].min()
        capped_max = result['price'].max()
        capped_min = result['price'].min()
        
        assert capped_max < original_max
        assert capped_min > original_min
    
    def test_cap_outliers_market_custom_columns(self):
        """Test outlier capping with custom column list."""
        cleaner = MarketDataCleaner()
        result = cleaner.cap_outliers_market(
            self.extreme_outliers_data,
            method="iqr",
            columns=['price']  # Only process price column
        )
        
        # Should only process specified columns
        stats = cleaner.metadata['outlier_capping_stats']
        assert 'price' in stats
        # Volume should not be processed if not specified
        if 'volume' not in ['price']:
            assert stats.get('volume', {}).get('outliers_detected', 0) == 0
    
    def test_cap_outliers_market_invalid_method(self):
        """Test outlier capping with invalid method."""
        # Create data that will definitely be processed (numeric data)
        numeric_data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 104.0]
        })
        
        cleaner = MarketDataCleaner()
        
        with pytest.raises(DataProcessingError, match="Unknown outlier detection method"):
            cleaner.cap_outliers_market(numeric_data, method="invalid")
    
    def test_cap_outliers_market_non_numeric_columns(self):
        """Test outlier capping skips non-numeric columns."""
        data = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'price': [100, 200, 300]
        })
        
        cleaner = MarketDataCleaner(numeric_columns=['text_col', 'price'])
        
        # Should not raise error, just skip non-numeric
        result = cleaner.cap_outliers_market(data)
        
        # text_col should be unchanged
        assert (result['text_col'] == data['text_col']).all()
    
    def test_clean_market_data_full_pipeline(self):
        """Test the complete market data cleaning pipeline."""
        cleaner = MarketDataCleaner()
        result = cleaner.clean_market_data(self.sample_data)
        
        # Should have no missing values
        assert not result.isna().any().any()
        
        # Should have proper data types
        for col in ['open', 'close', 'volume']:
            if col in result.columns:
                assert result[col].dtype == 'float64'
        
        # Should have fewer or equal rows (duplicates removed)
        assert len(result) <= len(self.sample_data)
        
        # Should track all operations in metadata
        expected_keys = [
            'type_coercion',
            'market_duplicates_removed', 
            'missing_values_before_market_imputation',
            'missing_values_after_market_imputation',
            'outlier_capping_stats',
            'cleaning_summary'
        ]
        for key in expected_keys:
            assert key in cleaner.metadata
    
    def test_clean_market_data_different_outlier_methods(self):
        """Test cleaning with different outlier detection methods."""
        cleaner = MarketDataCleaner()
        
        # Test with z-score
        result_zscore = cleaner.clean_market_data(
            self.extreme_outliers_data, 
            outlier_method="zscore"
        )
        
        # Reset cleaner for IQR test
        cleaner = MarketDataCleaner()
        result_iqr = cleaner.clean_market_data(
            self.extreme_outliers_data, 
            outlier_method="iqr"
        )
        
        # Both should reduce extreme values
        original_range = (self.extreme_outliers_data['price'].max() - 
                         self.extreme_outliers_data['price'].min())
        zscore_range = result_zscore['price'].max() - result_zscore['price'].min()
        iqr_range = result_iqr['price'].max() - result_iqr['price'].min()
        
        assert zscore_range < original_range
        assert iqr_range < original_range
    
    def test_get_cleaning_report(self):
        """Test cleaning report generation."""
        cleaner = MarketDataCleaner()
        cleaner.clean_market_data(self.sample_data)
        
        report = cleaner.get_cleaning_report()
        
        # Should contain expected sections
        assert 'cleaning_timestamp' in report
        assert 'cleaner_config' in report
        assert 'metadata' in report
        assert 'operations_summary' in report
        
        # Config should match cleaner settings
        config = report['cleaner_config']
        assert config['symbol_column'] == cleaner.symbol_column
        assert config['timestamp_column'] == cleaner.timestamp_column
        assert config['numeric_columns'] == cleaner.numeric_columns
        
        # Operations summary should have counts
        ops = report['operations_summary']
        assert 'duplicates_removed' in ops
        assert 'values_imputed' in ops
        assert 'outliers_capped' in ops
        assert 'na_from_coercion' in ops
    
    def test_edge_case_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        cleaner = MarketDataCleaner()
        result = cleaner.coerce_data_types(self.empty_data)
        
        # Should return empty DataFrame without error
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_edge_case_single_row(self):
        """Test handling of single row DataFrame."""
        cleaner = MarketDataCleaner()
        result = cleaner.clean_market_data(self.single_row_data)
        
        # Should handle single row without error
        assert len(result) == 1
        assert not result.isna().any().any()
    
    def test_edge_case_all_identical_values(self):
        """Test handling of data with all identical values."""
        cleaner = MarketDataCleaner()
        result = cleaner.cap_outliers_market(self.all_identical_data)
        
        # Should not detect any outliers
        assert cleaner.metadata['total_outliers_capped'] == 0
        
        # Data should be unchanged
        assert (result['open'] == self.all_identical_data['open']).all()
    
    def test_edge_case_extreme_outliers(self):
        """Test handling of extremely large outliers."""
        data = pd.DataFrame({
            'price': [100, 101, 102, 50000, 104, 105, -5000, 107]  # More realistic but still extreme outliers
        })
        
        cleaner = MarketDataCleaner(z_score_threshold=2.0)  # More sensitive threshold
        result = cleaner.cap_outliers_market(data, method="zscore")
        
        # Should handle extreme values without error
        assert np.isfinite(result['price']).all()
        
        # The extreme values should be detected and capped
        original_max = data['price'].max()
        original_min = data['price'].min()
        result_max = result['price'].max()
        result_min = result['price'].min()
        
        # Values should be capped (reduced from original extremes)
        assert result_max < original_max
        assert result_min > original_min
        
        # Should track outliers in metadata
        assert 'outlier_capping_stats' in cleaner.metadata
        assert cleaner.metadata['total_outliers_capped'] > 0
    
    def test_error_handling_processing_failure(self):
        """Test error handling when processing fails."""
        cleaner = MarketDataCleaner()
        
        # Mock the process_data method to raise an exception (which is called by clean_market_data)
        with patch.object(cleaner, 'process_data', side_effect=Exception("Test error")):
            with pytest.raises(DataProcessingError, match="Error in market data cleaning pipeline"):
                cleaner.clean_market_data(self.sample_data)


class TestMarketDataCleanerFactory:
    """Test cases for the build_market_data_cleaner factory function."""
    
    def test_build_market_data_cleaner_defaults(self):
        """Test factory function with default parameters."""
        cleaner = build_market_data_cleaner()
        
        assert isinstance(cleaner, MarketDataCleaner)
        assert cleaner.symbol_column == "symbol"
        assert cleaner.timestamp_column == "timestamp"
        assert cleaner.z_score_threshold == 3.0
        assert cleaner.iqr_multiplier == 1.5
    
    def test_build_market_data_cleaner_custom_params(self):
        """Test factory function with custom parameters."""
        custom_columns = ['price', 'vol', 'adj_close']
        cleaner = build_market_data_cleaner(
            symbol_column="ticker",
            timestamp_column="date",
            numeric_columns=custom_columns,
            outlier_method="zscore",
            z_score_threshold=2.5,
            iqr_multiplier=2.0
        )
        
        assert cleaner.symbol_column == "ticker"
        assert cleaner.timestamp_column == "date"
        assert cleaner.numeric_columns == custom_columns
        assert cleaner.z_score_threshold == 2.5
        assert cleaner.iqr_multiplier == 2.0


class TestMarketDataCleanerIntegration:
    """Integration tests for MarketDataCleaner with realistic market data scenarios."""
    
    def setup_method(self):
        """Set up realistic market data scenarios."""
        # Realistic intraday market data with gaps
        dates = pd.date_range(
            start='2023-01-01 09:30:00', 
            end='2023-01-01 16:00:00', 
            freq='1min'
        )
        
        # Remove some timestamps to simulate gaps (lunch break, system issues)
        gaps = [100, 101, 102, 200, 201, 250, 251, 252, 253]  # Various gap patterns
        clean_dates = dates.delete(gaps)
        
        self.intraday_data = pd.DataFrame({
            'timestamp': clean_dates,
            'symbol': ['SPY'] * len(clean_dates),
            'open': np.random.normal(400, 5, len(clean_dates)),
            'high': np.random.normal(405, 5, len(clean_dates)),
            'low': np.random.normal(395, 5, len(clean_dates)),
            'close': np.random.normal(400, 5, len(clean_dates)),
            'volume': np.random.lognormal(10, 1, len(clean_dates))
        })
        
        # Add some realistic market anomalies
        # Price spike (news event)
        self.intraday_data.loc[50, 'high'] = 450
        self.intraday_data.loc[50, 'close'] = 445
        
        # Volume spike (institutional trade)
        self.intraday_data.loc[75, 'volume'] = 1e7
        
        # Data quality issues
        self.intraday_data.loc[25, 'open'] = np.nan  # Missing open
        self.intraday_data.loc[125, 'volume'] = np.nan  # Missing volume
        
        # Add some duplicate entries (system error)
        duplicate_rows = self.intraday_data.iloc[30:32].copy()
        self.intraday_data = pd.concat([self.intraday_data, duplicate_rows], ignore_index=True)
        
        # Multi-symbol daily data
        daily_dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        daily_data_list = []
        for symbol in symbols:
            symbol_data = pd.DataFrame({
                'timestamp': daily_dates,
                'symbol': [symbol] * len(daily_dates),
                'open': np.random.normal(150, 20, len(daily_dates)),
                'high': np.random.normal(155, 20, len(daily_dates)),
                'low': np.random.normal(145, 20, len(daily_dates)),
                'close': np.random.normal(150, 20, len(daily_dates)),
                'volume': np.random.lognormal(15, 0.5, len(daily_dates))
            })
            daily_data_list.append(symbol_data)
        
        self.multi_symbol_data = pd.concat(daily_data_list, ignore_index=True)
        
        # Add some symbol-specific issues
        # TSLA has volatile data with outliers
        tsla_mask = self.multi_symbol_data['symbol'] == 'TSLA'
        self.multi_symbol_data.loc[tsla_mask.idxmax() + 5, 'close'] = 1000  # Price spike
        self.multi_symbol_data.loc[tsla_mask.idxmax() + 10, 'volume'] = 1e9  # Volume spike
        
        # AAPL has some missing data
        aapl_mask = self.multi_symbol_data['symbol'] == 'AAPL'
        aapl_indices = self.multi_symbol_data[aapl_mask].index
        self.multi_symbol_data.loc[aapl_indices[5], 'open'] = np.nan
        self.multi_symbol_data.loc[aapl_indices[15], 'volume'] = np.nan
    
    def test_realistic_intraday_cleaning(self):
        """Test cleaning realistic intraday market data."""
        cleaner = build_market_data_cleaner(
            symbol_column='symbol',
            timestamp_column='timestamp',
            z_score_threshold=3.0,
            iqr_multiplier=1.5
        )
        
        result = cleaner.clean_market_data(self.intraday_data, outlier_method="iqr")
        
        # Should handle all issues
        assert not result.isna().any().any()  # No missing values
        assert len(result) <= len(self.intraday_data)  # Duplicates removed
        
        # Price and volume spikes should be controlled
        original_price_range = (self.intraday_data['close'].max() - 
                               self.intraday_data['close'].min())
        cleaned_price_range = result['close'].max() - result['close'].min()
        assert cleaned_price_range <= original_price_range
        
        # Should have proper data types
        assert all(result[col].dtype == 'float64' 
                  for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Get cleaning report
        report = cleaner.get_cleaning_report()
        assert report['operations_summary']['duplicates_removed'] > 0
        assert report['operations_summary']['values_imputed'] > 0
        assert report['operations_summary']['outliers_capped'] > 0
    
    def test_multi_symbol_daily_cleaning(self):
        """Test cleaning multi-symbol daily market data."""
        cleaner = build_market_data_cleaner(
            outlier_method="zscore",
            z_score_threshold=2.5
        )
        
        result = cleaner.clean_market_data(self.multi_symbol_data, outlier_method="zscore")
        
        # Should preserve all symbols
        original_symbols = set(self.multi_symbol_data['symbol'].unique())
        cleaned_symbols = set(result['symbol'].unique())
        assert original_symbols == cleaned_symbols
        
        # Each symbol should have clean data
        for symbol in original_symbols:
            symbol_data = result[result['symbol'] == symbol]
            assert not symbol_data.isna().any().any()
            assert len(symbol_data) > 0
        
        # TSLA outliers should be controlled
        tsla_data = result[result['symbol'] == 'TSLA']
        original_tsla = self.multi_symbol_data[self.multi_symbol_data['symbol'] == 'TSLA']
        
        assert tsla_data['close'].max() < original_tsla['close'].max()
        assert tsla_data['volume'].max() < original_tsla['volume'].max()
    
    def test_performance_with_large_dataset(self):
        """Test performance and memory efficiency with larger dataset."""
        # Create larger dataset
        large_dates = pd.date_range('2023-01-01', '2023-12-31', freq='1min')
        large_data = pd.DataFrame({
            'timestamp': large_dates,
            'symbol': ['SPY'] * len(large_dates),
            'open': np.random.normal(400, 10, len(large_dates)),
            'close': np.random.normal(400, 10, len(large_dates)),
            'volume': np.random.lognormal(12, 1, len(large_dates))
        })
        
        # Add some issues
        large_data.loc[::1000, 'open'] = np.nan  # Sparse missing values
        large_data.loc[::5000, 'close'] = large_data['close'] * 10  # Outliers
        
        cleaner = build_market_data_cleaner()
        
        # Should complete without memory issues
        result = cleaner.clean_market_data(large_data)
        
        assert len(result) == len(large_data)  # No duplicates in this case
        assert not result.isna().any().any()
        
        # Performance metadata should be tracked
        report = cleaner.get_cleaning_report()
        assert 'cleaning_summary' in report['metadata']
    
    def test_edge_case_weekend_gaps(self):
        """Test handling of weekend gaps in daily data."""
        # Create weekday-only data with weekend gaps
        business_dates = pd.bdate_range('2023-01-01', '2023-01-31')
        weekend_gap_data = pd.DataFrame({
            'timestamp': business_dates,
            'symbol': ['SPY'] * len(business_dates),
            'open': np.random.normal(400, 5, len(business_dates)),
            'close': np.random.normal(400, 5, len(business_dates)),
            'volume': np.random.lognormal(10, 0.5, len(business_dates))
        })
        
        cleaner = build_market_data_cleaner()
        result = cleaner.clean_market_data(weekend_gap_data)
        
        # Should handle weekend gaps without issues
        assert len(result) == len(weekend_gap_data)
        assert not result.isna().any().any()
    
    def test_mixed_frequency_data_robustness(self):
        """Test robustness with mixed frequency data."""
        # Mix of daily and intraday data (simulating data feed issues)
        daily_part = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', '2023-01-05', freq='D'),
            'symbol': ['TEST'] * 5,
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        intraday_part = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-06 09:30', '2023-01-06 16:00', freq='1H'),
            'symbol': ['TEST'] * 7,
            'price': [105, 106, 107, 108, 109, 110, 111],
            'volume': [500, 600, 700, 800, 900, 1000, 1100]
        })
        
        mixed_data = pd.concat([daily_part, intraday_part], ignore_index=True)
        
        cleaner = build_market_data_cleaner()
        result = cleaner.clean_market_data(mixed_data)
        
        # Should handle mixed frequencies
        assert len(result) == len(mixed_data)
        assert not result.isna().any().any()
        assert all(result[col].dtype == 'float64' for col in ['price', 'volume'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
