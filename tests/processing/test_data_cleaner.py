import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from src.data.processing.data_cleaner import (
    DataCleaner, 
    CleaningStrategy, 
    OutlierDetectionMethod,
    build_default_market_cleaner
)


class TestDataCleaner:
    """Test cases for DataCleaner class and factory functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample corrupted market data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        
        # Create corrupted data with various issues
        self.corrupted_data = pd.DataFrame({
            'open': [100.0, 105.0, np.nan, 98.0, 102.0, 99.0, 101.0, np.nan, 103.0, 97.0],
            'high': [110.0, 108.0, 104.0, 105.0, 108.0, 106.0, 107.0, 105.0, 109.0, 102.0],
            'low': [95.0, 102.0, 96.0, 93.0, 97.0, 94.0, 96.0, 98.0, 99.0, 92.0],
            'close': [105.0, 104.0, 100.0, 97.0, 105.0, 100.0, 102.0, 101.0, 105.0, 95.0],
            'volume': [1000.0, 1500.0, 800.0, np.nan, 1200.0, 900.0, 1100.0, 1300.0, 1000.0, 850.0],
            # Add outliers
            'outlier_column': [100.0, 105.0, 1000.0, 98.0, 102.0, 99.0, 101.0, -500.0, 103.0, 97.0]
        }, index=dates)
        
        # Add duplicate rows
        duplicate_row = self.corrupted_data.iloc[2].copy()
        self.corrupted_data = pd.concat([self.corrupted_data, duplicate_row.to_frame().T])
        
        # Create data with timestamp gaps
        self.gapped_data = self.corrupted_data.copy()
        # Remove some timestamps to create gaps
        self.gapped_data = self.gapped_data.drop(self.gapped_data.index[3:5])  # Remove 2 days
        
        # Create intraday data with gaps
        intraday_dates = pd.date_range(start='2023-01-01 09:00', periods=100, freq='5min')
        # Remove some timestamps to create gaps
        intraday_dates = intraday_dates.delete([10, 11, 12, 30, 31, 32, 33])  # Create gaps
        
        self.intraday_gapped_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, len(intraday_dates)),
            'high': np.random.uniform(105, 115, len(intraday_dates)),
            'low': np.random.uniform(95, 105, len(intraday_dates)),
            'close': np.random.uniform(100, 110, len(intraday_dates)),
            'volume': np.random.uniform(500, 2000, len(intraday_dates))
        }, index=intraday_dates)

    def test_build_default_market_cleaner(self):
        """Test the build_default_market_cleaner factory function."""
        # Create the default cleaner
        cleaner = build_default_market_cleaner()
        
        # Assert the cleaner was created correctly
        assert isinstance(cleaner, DataCleaner)
        assert cleaner.default_missing_strategy == CleaningStrategy.FILL_FORWARD
        assert cleaner.default_outlier_strategy == CleaningStrategy.WINSORIZE
        assert cleaner.outlier_detection_method == OutlierDetectionMethod.IQR
        assert cleaner.outlier_threshold == 1.5
        
        # Check that outlier strategy is set for all columns
        assert '*' in cleaner.outlier_strategies
        assert cleaner.outlier_strategies['*'] == CleaningStrategy.WINSORIZE

    def test_default_cleaner_on_corrupted_data(self):
        """Test the default cleaner on corrupted sample data."""
        # Create the default cleaner
        cleaner = build_default_market_cleaner()
        
        # Process the corrupted data
        cleaned_data = cleaner.process_data(self.corrupted_data)
        
        # Assert missing values are handled (FILL_FORWARD strategy)
        assert not cleaned_data.isna().any().any(), "Should have no missing values after cleaning"
        
        # Assert outliers are handled (WINSORIZE strategy with IQR)
        original_outlier_range = self.corrupted_data['outlier_column'].max() - self.corrupted_data['outlier_column'].min()
        cleaned_outlier_range = cleaned_data['outlier_column'].max() - cleaned_data['outlier_column'].min()
        assert cleaned_outlier_range < original_outlier_range, "Outlier range should be reduced"
        
        # Assert duplicates are removed
        assert not cleaned_data.duplicated().any(), "Should have no duplicates after cleaning"
        
        # Assert data integrity (should have fewer or equal rows due to duplicate removal)
        assert len(cleaned_data) <= len(self.corrupted_data), "Should have same or fewer rows after cleaning"

    def test_missing_value_handling(self):
        """Test missing value handling with different strategies."""
        cleaner = DataCleaner(default_missing_strategy=CleaningStrategy.FILL_FORWARD)
        
        # Test FILL_FORWARD (default in factory)
        cleaned_data = cleaner.handle_missing_values(self.corrupted_data)
        assert not cleaned_data['open'].isna().any(), "Open column should have no NaN values"
        assert not cleaned_data['volume'].isna().any(), "Volume column should have no NaN values"
        
        # Test FILL_MEAN strategy
        cleaner.set_missing_value_strategy('open', CleaningStrategy.FILL_MEAN)
        cleaned_data = cleaner.handle_missing_values(self.corrupted_data)
        assert not cleaned_data['open'].isna().any(), "Open column should have no NaN values with mean fill"
        
        # Test FILL_INTERPOLATE strategy
        cleaner.set_missing_value_strategy('volume', CleaningStrategy.FILL_INTERPOLATE)
        cleaned_data = cleaner.handle_missing_values(self.corrupted_data)
        assert not cleaned_data['volume'].isna().any(), "Volume column should have no NaN values with interpolation"

    def test_outlier_detection_iqr(self):
        """Test outlier detection using IQR method with threshold 1.5."""
        cleaner = DataCleaner(
            outlier_detection_method=OutlierDetectionMethod.IQR,
            outlier_threshold=1.5
        )
        
        # Detect outliers in the outlier_column
        outlier_mask = cleaner._detect_outliers(self.corrupted_data['outlier_column'])
        
        # Should detect the extreme values (1000.0 and -500.0)
        assert outlier_mask.sum() >= 2, "Should detect at least 2 outliers"
        
        # Verify the extreme values are detected as outliers
        extreme_indices = self.corrupted_data[self.corrupted_data['outlier_column'].isin([1000.0, -500.0])].index
        for idx in extreme_indices:
            assert outlier_mask.loc[idx].item(), f"Extreme value at index {idx} should be detected as outlier"

    def test_winsorization(self):
        """Test winsorization of outliers."""
        cleaner = DataCleaner(
            default_outlier_strategy=CleaningStrategy.WINSORIZE,
            outlier_detection_method=OutlierDetectionMethod.IQR,
            outlier_threshold=1.5
        )
        
        # Apply winsorization
        cleaned_data = cleaner.handle_outliers(self.corrupted_data, ['outlier_column'])
        
        # Check that extreme values are capped
        original_max = self.corrupted_data['outlier_column'].max()
        original_min = self.corrupted_data['outlier_column'].min()
        cleaned_max = cleaned_data['outlier_column'].max()
        cleaned_min = cleaned_data['outlier_column'].min()
        
        assert cleaned_max < original_max, "Maximum should be reduced after winsorization"
        assert cleaned_min > original_min, "Minimum should be increased after winsorization"

    def test_duplicate_removal(self):
        """Test duplicate row removal."""
        cleaner = DataCleaner()
        
        # Count duplicates before cleaning
        duplicates_before = self.corrupted_data.duplicated().sum()
        assert duplicates_before > 0, "Test data should contain duplicates"
        
        # Remove duplicates
        cleaned_data = cleaner.handle_duplicates(self.corrupted_data)
        
        # Check that duplicates are removed
        duplicates_after = cleaned_data.duplicated().sum()
        assert duplicates_after == 0, "Should have no duplicates after cleaning"
        assert len(cleaned_data) == len(self.corrupted_data) - duplicates_before, "Should have fewer rows after duplicate removal"

    def test_timestamp_gap_filling_auto_detection(self):
        """Test timestamp gap filling with auto-detection."""
        cleaner = DataCleaner()
        
        # Remove duplicates first to test gap filling properly
        deduplicated_data = self.gapped_data.drop_duplicates()
        
        # Test daily frequency auto-detection
        filled_data = cleaner.fill_timestamp_gaps(deduplicated_data)
        
        # Should fill the gaps
        expected_length = (self.gapped_data.index.max() - self.gapped_data.index.min()).days + 1
        assert len(filled_data) >= len(self.gapped_data), "Should have more or equal rows after gap filling"
        
        # Check that there are no more gaps in the index
        freq_detected = pd.infer_freq(filled_data.index)
        if freq_detected:
            # If frequency is detected, create expected complete index
            expected_index = pd.date_range(start=filled_data.index.min(), end=filled_data.index.max(), freq=freq_detected)
            assert len(filled_data.index) == len(expected_index), "Should have complete timestamp coverage"

    def test_intraday_timestamp_gap_filling(self):
        """Test timestamp gap filling for intraday data."""
        cleaner = DataCleaner()
        
        # Test intraday frequency auto-detection (5min intervals)
        filled_data = cleaner.fill_timestamp_gaps(self.intraday_gapped_data)
        
        # Should fill the gaps
        assert len(filled_data) >= len(self.intraday_gapped_data), "Should have more or equal rows after gap filling"
        
        # Values should be filled using forward fill (default method)
        assert not filled_data.isna().any().any(), "Should have no NaN values after gap filling"

    def test_frequency_auto_detection_fallback(self):
        """Test frequency auto-detection fallback mechanisms."""
        cleaner = DataCleaner()
        
        # Create irregular timestamp data
        irregular_dates = [
            datetime(2023, 1, 1, 9, 0),
            datetime(2023, 1, 1, 9, 5),
            datetime(2023, 1, 1, 9, 10),
            datetime(2023, 1, 1, 9, 20),  # Gap
            datetime(2023, 1, 1, 9, 25),
            datetime(2023, 1, 1, 9, 30),
        ]
        
        irregular_data = pd.DataFrame({
            'price': [100, 101, 102, 103, 104, 105]
        }, index=pd.DatetimeIndex(irregular_dates))
        
        # Should still work with fallback detection
        filled_data = cleaner.fill_timestamp_gaps(irregular_data)
        
        # Should have at least the original data
        assert len(filled_data) >= len(irregular_data), "Should have at least original data length"

    def test_end_to_end_cleaning_pipeline(self):
        """Test the complete cleaning pipeline end-to-end."""
        # Create the default market cleaner
        cleaner = build_default_market_cleaner()
        
        # Process the corrupted data through the complete pipeline
        cleaned_data = cleaner.process_data(self.corrupted_data)
        
        # Comprehensive assertions
        
        # 1. No missing values
        assert not cleaned_data.isna().any().any(), "Should have no missing values"
        
        # 2. No duplicates
        assert not cleaned_data.duplicated().any(), "Should have no duplicates"
        
        # 3. Outliers handled (range should be reduced)
        for col in ['outlier_column']:
            if col in cleaned_data.columns:
                original_range = self.corrupted_data[col].max() - self.corrupted_data[col].min()
                cleaned_range = cleaned_data[col].max() - cleaned_data[col].min()
                assert cleaned_range <= original_range, f"Range should be reduced or same for {col}"
        
        # 4. Data integrity preserved for normal values
        normal_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in normal_cols:
            if col in cleaned_data.columns:
                # Most values should be preserved (except for filled missing values)
                original_non_null = self.corrupted_data[col].dropna()
                # Check that non-extreme values are mostly preserved
                cleaned_values = cleaned_data[col]
                common_values = set(original_non_null) & set(cleaned_values)
                assert len(common_values) > 0, f"Should preserve some original values in {col}"
        
        # 5. Index should be datetime
        assert isinstance(cleaned_data.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"
        
        # 6. Metadata should be populated
        assert 'missing_values_before' in cleaner.metadata, "Should track missing values before cleaning"
        assert 'missing_values_after' in cleaner.metadata, "Should track missing values after cleaning"
        assert 'duplicates_removed' in cleaner.metadata, "Should track duplicates removed"

    def test_cleaner_with_custom_strategies(self):
        """Test cleaner with custom strategies per column."""
        # Create cleaner with custom strategies
        cleaner = build_default_market_cleaner()
        
        # Set custom strategies for specific columns
        cleaner.set_missing_value_strategy('volume', CleaningStrategy.FILL_MEAN)
        cleaner.set_outlier_strategy('outlier_column', CleaningStrategy.CLIP)
        
        # Process data
        cleaned_data = cleaner.process_data(self.corrupted_data)
        
        # Should still work and apply custom strategies
        assert not cleaned_data.isna().any().any(), "Should handle missing values with custom strategies"
        assert not cleaned_data.duplicated().any(), "Should handle duplicates"

    def test_metadata_tracking(self):
        """Test that cleaner properly tracks metadata."""
        cleaner = build_default_market_cleaner()
        
        # Process data and check metadata
        cleaned_data = cleaner.process_data(self.corrupted_data)
        
        # Check metadata exists and has expected keys
        expected_keys = ['missing_values_before', 'missing_values_after', 'duplicates_removed']
        for key in expected_keys:
            assert key in cleaner.metadata, f"Metadata should contain {key}"
        
        # Check that missing values were tracked correctly
        missing_before = cleaner.metadata['missing_values_before']
        missing_after = cleaner.metadata['missing_values_after']
        
        assert isinstance(missing_before, dict), "Missing values before should be a dict"
        assert isinstance(missing_after, dict), "Missing values after should be a dict"
        
        # Should have reduced missing values
        total_missing_before = sum(missing_before.values())
        total_missing_after = sum(missing_after.values())
        assert total_missing_after <= total_missing_before, "Should have reduced missing values"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        cleaner = build_default_market_cleaner()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = cleaner.handle_duplicates(empty_df)
        assert len(result) == 0, "Should handle empty DataFrame"
        
        # Test with single row
        single_row = self.corrupted_data.iloc[:1].copy()
        result = cleaner.handle_missing_values(single_row)
        assert len(result) == 1, "Should handle single row DataFrame"
        
        # Test with no missing values
        clean_data = self.corrupted_data.dropna()
        result = cleaner.handle_missing_values(clean_data)
        assert len(result) == len(clean_data), "Should handle data with no missing values"


class TestEnhancedMarketCleaner:
    """Test cases for enhanced market cleaner with robust strategies."""
    
    def setup_method(self):
        """Set up test fixtures for enhanced cleaner tests."""
        # Create irregular market data with various challenging scenarios
        irregular_dates = [
            datetime(2023, 1, 1, 9, 0),   # Weekend start
            datetime(2023, 1, 2, 9, 5),   # Monday
            datetime(2023, 1, 2, 9, 10),  # Regular interval
            datetime(2023, 1, 2, 9, 25),  # Gap (missing 15 min)
            datetime(2023, 1, 2, 9, 30),  # Resume
            datetime(2023, 1, 3, 9, 0),   # Next day
            datetime(2023, 1, 3, 9, 5),   # Normal
            datetime(2023, 1, 6, 9, 0),   # Weekend gap (Fri-Mon)
        ]
        
        self.irregular_market_data = pd.DataFrame({
            'open': [100.0, 101.0, np.nan, 103.0, 102.0, 104.0, np.nan, 105.0],
            'high': [105.0, 106.0, 108.0, 109.0, 107.0, 110.0, 111.0, 112.0],
            'low': [98.0, 99.0, 100.0, 101.0, 100.0, 102.0, 103.0, 104.0],
            'close': [101.0, 102.0, 103.0, 102.0, 104.0, 105.0, 106.0, 107.0],
            'volume': [1000.0, np.nan, 0.0, 2000.0, 1500.0, np.nan, 1800.0, 0.0],  # Include zeros and NaNs
            # Add extreme outliers
            'outlier_prices': [100.0, 200.0, 50000.0, 102.0, 0.01, 104.0, 105.0, -1000.0],  # Extreme values
        }, index=pd.DatetimeIndex(irregular_dates))
        
        # Create high-frequency data with gaps
        hf_dates = pd.date_range(start='2023-01-01 09:00', periods=50, freq='1min')
        # Remove some minutes to create gaps
        hf_dates = hf_dates.delete([5, 6, 7, 15, 16, 25, 26, 27, 28])  # Various gap sizes
        
        self.hf_irregular_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, len(hf_dates)),
            'high': np.random.uniform(105, 115, len(hf_dates)),
            'low': np.random.uniform(95, 105, len(hf_dates)),
            'close': np.random.uniform(100, 110, len(hf_dates)),
            'volume': np.random.uniform(0, 2000, len(hf_dates))  # Include some zeros
        }, index=hf_dates)
        
        # Add some extreme outliers
        self.hf_irregular_data.loc[self.hf_irregular_data.index[10], 'close'] = 50000  # Extreme high
        self.hf_irregular_data.loc[self.hf_irregular_data.index[20], 'volume'] = 100000  # Volume spike
        self.hf_irregular_data.loc[self.hf_irregular_data.index[30], 'open'] = 0.001  # Extreme low
    
    def test_enhanced_market_cleaner_factory(self):
        """Test the enhanced build_default_market_cleaner factory function."""
        cleaner = build_default_market_cleaner()
        
        # Verify factory configuration
        assert isinstance(cleaner, DataCleaner)
        assert cleaner.default_missing_strategy == CleaningStrategy.FILL_FORWARD
        assert cleaner.default_outlier_strategy == CleaningStrategy.WINSORIZE
        assert cleaner.outlier_detection_method == OutlierDetectionMethod.PERCENTILE
        assert cleaner.outlier_threshold == 0.005
        
        # Check column-specific missing value strategies
        assert cleaner.missing_value_strategies['volume'] == CleaningStrategy.FILL_CONSTANT
        assert cleaner.missing_fill_values['volume'] == 0.0
        
        for price_col in ['open', 'high', 'low', 'close']:
            assert cleaner.missing_value_strategies[price_col] == CleaningStrategy.FILL_FORWARD
        
        # Check column-specific outlier strategies
        for price_col in ['close', 'high', 'low', 'open']:
            assert cleaner.outlier_strategies[price_col] == CleaningStrategy.WINSORIZE
            assert cleaner.outlier_detection_methods[price_col] == OutlierDetectionMethod.PERCENTILE
            assert cleaner.outlier_thresholds[price_col] == 0.005
        
        # Check volume outlier strategy (IQR method)
        assert cleaner.outlier_strategies['volume'] == CleaningStrategy.WINSORIZE
        assert cleaner.outlier_detection_methods['volume'] == OutlierDetectionMethod.IQR
        assert cleaner.outlier_thresholds['volume'] == 1.5
    
    def test_column_specific_missing_value_handling(self):
        """Test column-specific missing value strategies."""
        cleaner = build_default_market_cleaner()
        
        # Process data with volume set to 0 for missing values
        cleaned_data = cleaner.handle_missing_values(self.irregular_market_data)
        
        # Volume missing values should be filled with 0
        assert not cleaned_data['volume'].isna().any(), "Volume should have no NaN values"
        # Check that NaN volumes became 0
        original_volume_na_mask = self.irregular_market_data['volume'].isna()
        assert (cleaned_data.loc[original_volume_na_mask, 'volume'] == 0.0).all(), "Missing volume should be filled with 0"
        
        # Price columns should use forward fill
        assert not cleaned_data['open'].isna().any(), "Open should have no NaN values after forward fill"
        
        # Check that forward fill was applied correctly
        original_open_na_indices = self.irregular_market_data['open'].isna()
        if original_open_na_indices.any():
            # Check that the filled values match the previous non-NaN value
            for idx in self.irregular_market_data.index[original_open_na_indices]:
                idx_pos = list(self.irregular_market_data.index).index(idx)
                if idx_pos > 0:
                    previous_valid_idx = self.irregular_market_data.index[idx_pos - 1]
                    expected_value = self.irregular_market_data.loc[previous_valid_idx, 'open']
                    actual_value = cleaned_data.loc[idx, 'open']
                    assert actual_value == expected_value, f"Forward fill not applied correctly at {idx}"
    
    def test_percentile_based_outlier_detection(self):
        """Test 0.5% percentile-based outlier detection for price columns."""
        cleaner = build_default_market_cleaner()
        
        # Test outlier detection on outlier_prices column
        outlier_mask = cleaner._detect_outliers(self.irregular_market_data['outlier_prices'], 'close')
        
        # With 0.5% percentiles, extreme values should be detected
        # Should detect extreme values (50000, -1000) as most extreme
        extreme_values = [50000.0, -1000.0]  # Focus on most extreme values
        for extreme_val in extreme_values:
            extreme_indices = self.irregular_market_data[self.irregular_market_data['outlier_prices'] == extreme_val].index
            if not extreme_indices.empty:
                assert outlier_mask.loc[extreme_indices[0]], f"Extreme value {extreme_val} should be detected as outlier"
        
        # Verify that at least some outliers are detected
        assert outlier_mask.sum() > 0, "Should detect at least some outliers with percentile method"
    
    def test_iqr_based_volume_outlier_detection(self):
        """Test IQR-based outlier detection for volume."""
        cleaner = build_default_market_cleaner()
        
        # Create test data with clear volume outliers using more extreme values
        test_volumes = pd.Series([100, 150, 200, 120, 180, 150, 130, 160, 50000, 100000])  # Last two are extreme outliers
        outlier_mask = cleaner._detect_outliers(test_volumes, 'volume')
        
        # Should detect the extreme volume spikes (50000, 100000)
        assert outlier_mask.iloc[-2], "Volume spike 50000 should be detected"
        assert outlier_mask.iloc[-1], "Volume spike 100000 should be detected"
        
        # Should detect at least the extreme outliers
        assert outlier_mask.sum() >= 2, "Should detect at least 2 extreme outliers"
    
    def test_winsorization_with_percentile_method(self):
        """Test winsorization using percentile method with 0.5% bounds."""
        cleaner = build_default_market_cleaner()
        
        # Apply outlier handling to price columns
        cleaned_data = cleaner.handle_outliers(self.irregular_market_data, ['outlier_prices'])
        
        # Check that extreme values were capped
        original_max = self.irregular_market_data['outlier_prices'].max()
        original_min = self.irregular_market_data['outlier_prices'].min()
        cleaned_max = cleaned_data['outlier_prices'].max()
        cleaned_min = cleaned_data['outlier_prices'].min()
        
        assert cleaned_max < original_max, "Maximum should be reduced after winsorization"
        assert cleaned_min > original_min, "Minimum should be increased after winsorization"
        
        # Values should be capped at 0.5% and 99.5% percentiles
        expected_upper = self.irregular_market_data['outlier_prices'].quantile(0.995)
        expected_lower = self.irregular_market_data['outlier_prices'].quantile(0.005)
        
        assert cleaned_max <= expected_upper, "Max should not exceed 99.5th percentile"
        assert cleaned_min >= expected_lower, "Min should not be below 0.5th percentile"
    
    def test_z_score_extreme_outlier_replacement(self):
        """Test z-score replacement for extreme outliers (>5 sigma)."""
        cleaner = build_default_market_cleaner()
        
        # Create data with clear normal distribution and extreme outliers
        # Using a very controlled set for reliable z-score calculation
        normal_values = [100.0] * 100  # 100 identical values (mean=100, std≈0)
        # Add some tiny variation to avoid std=0
        normal_values[:10] = [100.1, 100.2, 100.3, 100.4, 100.5, 99.9, 99.8, 99.7, 99.6, 99.5]
        extreme_values = [1000.0, -500.0]  # These will definitely be >5 sigma
        all_values = normal_values + extreme_values
        
        test_data = pd.DataFrame({
            'price': all_values
        })
        
        # Apply extreme outlier handling
        cleaned_data = cleaner._handle_extreme_outliers(test_data)
        
        # The z-score method should identify and replace extreme outliers
        # Check that the functionality works (extreme values should be replaced)
        original_max = test_data['price'].max()
        original_min = test_data['price'].min()
        cleaned_max = cleaned_data['price'].max()
        cleaned_min = cleaned_data['price'].min()
        
        # Extreme values should be reduced/replaced
        assert cleaned_max < original_max, "Maximum should be reduced after extreme outlier replacement"
        assert cleaned_min > original_min, "Minimum should be increased after extreme outlier replacement"
        
        # Check that normal values around 100 are preserved
        normal_vals_preserved = cleaned_data['price'][(cleaned_data['price'] >= 99.0) & (cleaned_data['price'] <= 101.0)]
        assert len(normal_vals_preserved) >= 90, "Most normal values should be preserved"
        
        # Check metadata tracks the operation
        assert "extreme_outliers_replaced" in cleaner.metadata, "Should track extreme outliers in metadata"
    
    def test_auto_frequency_detection_daily(self):
        """Test auto-frequency detection for daily data."""
        cleaner = DataCleaner()
        
        # Create daily data with gaps
        daily_dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        daily_dates = daily_dates.delete([3, 4, 7])  # Remove some days
        
        daily_data = pd.DataFrame({
            'price': range(len(daily_dates))
        }, index=daily_dates)
        
        # Test auto-frequency detection
        filled_data = cleaner.fill_timestamp_gaps(daily_data)
        
        # Should have filled the gaps
        assert len(filled_data) > len(daily_data), "Should have filled gaps in daily data"
        
        # Check that frequency was detected as daily
        expected_length = (daily_data.index.max() - daily_data.index.min()).days + 1
        assert len(filled_data) == expected_length, "Should have complete daily coverage"
    
    def test_auto_frequency_detection_intraday(self):
        """Test auto-frequency detection for intraday data."""
        cleaner = DataCleaner()
        
        # Use the high-frequency irregular data
        filled_data = cleaner.fill_timestamp_gaps(self.hf_irregular_data)
        
        # Should have filled the gaps
        assert len(filled_data) >= len(self.hf_irregular_data), "Should have filled gaps in intraday data"
        
        # Values should be filled (no NaN)
        assert not filled_data.isna().any().any(), "Should have no NaN values after gap filling"
    
    def test_auto_frequency_detection_fallback(self):
        """Test auto-frequency detection fallback for highly irregular data."""
        cleaner = DataCleaner()
        
        # Create highly irregular data
        very_irregular_dates = [
            datetime(2023, 1, 1, 9, 0),
            datetime(2023, 1, 1, 11, 23),
            datetime(2023, 1, 2, 14, 45),
            datetime(2023, 1, 5, 8, 12),
            datetime(2023, 1, 8, 16, 30),
        ]
        
        very_irregular_data = pd.DataFrame({
            'price': [100, 101, 102, 103, 104]
        }, index=pd.DatetimeIndex(very_irregular_dates))
        
        # Should still work without errors (may not fill gaps if detection fails)
        filled_data = cleaner.fill_timestamp_gaps(very_irregular_data)
        
        # At minimum, should return the original data unchanged
        assert len(filled_data) >= len(very_irregular_data), "Should at least preserve original data"
        assert filled_data.index.equals(very_irregular_data.index) or len(filled_data) > len(very_irregular_data), \
               "Should either preserve original index or add timestamps"
    
    def test_end_to_end_enhanced_cleaning(self):
        """Test complete enhanced cleaning pipeline end-to-end."""
        cleaner = build_default_market_cleaner()
        
        # Process the irregular market data through the complete pipeline
        cleaned_data = cleaner.process_data(self.irregular_market_data)
        
        # Comprehensive checks
        
        # 1. No missing values
        assert not cleaned_data.isna().any().any(), "Should have no missing values"
        
        # 2. Volume missing values filled with 0
        # Note: After gap filling, we need to check the original timestamps that existed
        original_timestamps = self.irregular_market_data.index
        original_volume_na_mask = self.irregular_market_data['volume'].isna()
        if original_volume_na_mask.any():
            # Find the original timestamps that had NaN volume
            original_na_timestamps = original_timestamps[original_volume_na_mask]
            # Check that those timestamps now have volume = 0
            for ts in original_na_timestamps:
                if ts in cleaned_data.index:
                    assert cleaned_data.loc[ts, 'volume'] == 0.0, \
                           f"Missing volume at {ts} should be filled with 0"
        
        # 3. Price outliers winsorized at 0.5% percentiles
        for col in ['outlier_prices']:
            if col in cleaned_data.columns:
                original_range = self.irregular_market_data[col].max() - self.irregular_market_data[col].min()
                cleaned_range = cleaned_data[col].max() - cleaned_data[col].min()
                assert cleaned_range < original_range, f"Range should be reduced for {col}"
        
        # 4. Index should remain DatetimeIndex
        assert isinstance(cleaned_data.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"
        
        # 5. Metadata should track all operations
        expected_metadata_keys = [
            'missing_values_before', 'missing_values_after', 
            'duplicates_removed', 'outliers_before',
            'extreme_outliers_replaced'
        ]
        for key in expected_metadata_keys:
            assert key in cleaner.metadata, f"Should track {key} in metadata"
        
        # 6. Data integrity for normal values
        normal_price_cols = ['open', 'high', 'low', 'close']
        for col in normal_price_cols:
            if col in cleaned_data.columns:
                # Most reasonable values should be preserved
                original_non_null = self.irregular_market_data[col].dropna()
                reasonable_values = original_non_null[(original_non_null > 50) & (original_non_null < 200)]
                if not reasonable_values.empty:
                    # Check that reasonable values are preserved
                    cleaned_values = cleaned_data[col]
                    preserved_values = set(reasonable_values) & set(cleaned_values)
                    assert len(preserved_values) > 0, f"Should preserve reasonable values in {col}"
    
    def test_high_frequency_data_processing(self):
        """Test processing of high-frequency data with various challenges."""
        cleaner = build_default_market_cleaner()
        
        # Process high-frequency irregular data
        cleaned_data = cleaner.process_data(self.hf_irregular_data)
        
        # Should handle high-frequency data without issues
        assert not cleaned_data.isna().any().any(), "Should have no missing values"
        assert len(cleaned_data) >= len(self.hf_irregular_data), "Should have same or more rows after processing"
        
        # Extreme outliers should be handled (check original timestamps that exist)
        original_extreme_close_timestamps = self.hf_irregular_data[self.hf_irregular_data['close'] > 1000].index
        if not original_extreme_close_timestamps.empty:
            # Check that extreme values at original timestamps were reduced
            for ts in original_extreme_close_timestamps:
                if ts in cleaned_data.index:
                    original_val = self.hf_irregular_data.loc[ts, 'close']
                    cleaned_val = cleaned_data.loc[ts, 'close']
                    assert cleaned_val < original_val, \
                           f"Extreme close value at {ts} should be reduced from {original_val} to {cleaned_val}"
        
        # Volume spikes should be handled (check original timestamps that exist)
        original_extreme_volume_timestamps = self.hf_irregular_data[self.hf_irregular_data['volume'] > 10000].index
        if not original_extreme_volume_timestamps.empty:
            # Check that extreme values at original timestamps were reduced
            for ts in original_extreme_volume_timestamps:
                if ts in cleaned_data.index:
                    original_val = self.hf_irregular_data.loc[ts, 'volume']
                    cleaned_val = cleaned_data.loc[ts, 'volume']
                    assert cleaned_val < original_val, \
                           f"Extreme volume value at {ts} should be reduced from {original_val} to {cleaned_val}"


if __name__ == "__main__":
    pytest.main([__file__])
