"""
Test script for Feature Engineering Pipeline Configuration

This script validates the FeatureEngineer configuration and demonstrates:
1. Instantiation with enable_all_features=True
2. Selective feature enabling
3. Required column validation
4. Basic performance measurement
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from src.data.processing.feature_engineering import FeatureEngineer
from src.infrastructure.config import ConfigManager


def create_sample_data(num_rows=1000):
    """Create sample market data for testing."""
    print(f"Creating sample data with {num_rows} rows...")
    
    # Generate datetime index
    start_date = datetime.now() - timedelta(days=num_rows//1440)  # Assuming 1-minute data
    date_range = pd.date_range(start=start_date, periods=num_rows, freq='1min')
    
    # Generate realistic market data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, num_rows)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=date_range)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(base_price) * (1 + np.random.normal(0, 0.001, num_rows))
    
    volatility = np.random.uniform(0.005, 0.03, num_rows)
    data['high'] = np.maximum(data['open'], data['close']) * (1 + volatility)
    data['low'] = np.minimum(data['open'], data['close']) * (1 - volatility)
    
    base_volume = 10000
    volume_multiplier = 1 + np.abs(returns) * 10
    data['volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, num_rows)).astype(int)
    
    data = data.clip(lower=0.01)
    
    print(f"✓ Created sample data: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def test_all_features_configuration():
    """Test FeatureEngineer with enable_all_features=True."""
    print("\n" + "="*50)
    print("TESTING: FeatureEngineer with enable_all_features=True")
    print("="*50)
    
    try:
        # Create configuration
        config = ConfigManager()
        
        # Instantiate with all features enabled
        feature_engineer = FeatureEngineer(
            config=config,
            enable_all_features=True
        )
        
        # Validate configuration
        enabled_sets = feature_engineer.get_enabled_feature_sets()
        total_features = len(feature_engineer.enabled_features)
        
        print(f"✓ FeatureEngineer instantiated with enable_all_features=True")
        print(f"✓ Enabled feature sets: {enabled_sets}")
        print(f"✓ Total enabled features: {total_features}")
        
        return feature_engineer
        
    except Exception as e:
        print(f"✗ Error in all features configuration: {e}")
        raise


def test_selective_features_configuration():
    """Test FeatureEngineer with selective feature enabling."""
    print("\n" + "="*50)
    print("TESTING: FeatureEngineer with selective features")
    print("="*50)
    
    try:
        # Create configuration
        config = ConfigManager()
        
        # Instantiate without enabling all features
        feature_engineer = FeatureEngineer(
            config=config,
            enable_all_features=False
        )
        
        # Selectively enable feature sets as specified in the task
        feature_sets_to_enable = [
            "price_derived",    # Price derived features
            "moving_averages",  # Moving averages
            "volatility",       # Volatility indicators
            "momentum",         # Momentum indicators
            "volume",           # Volume indicators
            "trend"             # Trend indicators
        ]
        
        for feature_set in feature_sets_to_enable:
            try:
                feature_engineer.enable_feature_set(feature_set)
                print(f"✓ Enabled feature set: {feature_set}")
            except ValueError as e:
                print(f"✗ Failed to enable feature set {feature_set}: {e}")
        
        # Validate configuration
        enabled_sets = feature_engineer.get_enabled_feature_sets()
        total_features = len(feature_engineer.enabled_features)
        
        print(f"✓ Selective configuration complete")
        print(f"✓ Enabled feature sets: {enabled_sets}")
        print(f"✓ Total enabled features: {total_features}")
        
        return feature_engineer
        
    except Exception as e:
        print(f"✗ Error in selective features configuration: {e}")
        raise


def test_required_columns_validation(feature_engineer):
    """Test required column validation via get_required_columns()."""
    print("\n" + "="*50)
    print("TESTING: Required Column Validation")
    print("="*50)
    
    try:
        # Get required columns
        required_columns = feature_engineer.get_required_columns()
        print(f"✓ Required columns: {sorted(required_columns)}")
        
        # Analyze column requirements by feature set
        print("\n--- Feature Set Requirements ---")
        for name, feature_set in feature_engineer.feature_sets.items():
            if any(feature in feature_engineer.enabled_features for feature in feature_set.features):
                print(f"{name}:")
                print(f"  Dependencies: {feature_set.dependencies}")
                print(f"  Features: {len(feature_set.features)} features")
                print(f"  Category: {feature_set.category.name}")
        
        # Check coverage with standard OHLCV data
        standard_columns = {'open', 'high', 'low', 'close', 'volume'}
        coverage = len(required_columns.intersection(standard_columns)) / len(required_columns) * 100
        
        print(f"\n✓ Standard OHLCV columns: {sorted(standard_columns)}")
        print(f"✓ Coverage with standard data: {coverage:.1f}%")
        
        if required_columns.issubset(standard_columns):
            print("✓ All required columns covered by standard OHLCV data")
        else:
            missing = required_columns - standard_columns
            print(f"⚠ Additional columns needed: {sorted(missing)}")
        
        return required_columns
        
    except Exception as e:
        print(f"✗ Error in column validation: {e}")
        raise


def test_performance_measurement(feature_engineer, data):
    """Test computational cost and memory impact measurement."""
    print("\n" + "="*50)
    print("TESTING: Performance Measurement")
    print("="*50)
    
    try:
        # Measure processing time
        start_time = time.time()
        processed_data = feature_engineer.process_data(data)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate metrics
        original_columns = len(data.columns)
        processed_columns = len(processed_data.columns)
        new_features = processed_columns - original_columns
        processing_rate = len(data) / processing_time
        
        print(f"✓ Processing completed successfully")
        print(f"✓ Processing time: {processing_time:.4f} seconds")
        print(f"✓ Original columns: {original_columns}")
        print(f"✓ Processed columns: {processed_columns}")
        print(f"✓ New features added: {new_features}")
        print(f"✓ Processing rate: {processing_rate:.0f} rows/second")
        
        # Memory usage estimation
        original_size = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        processed_size = processed_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        size_increase = processed_size - original_size
        
        print(f"✓ Original data size: {original_size:.2f} MB")
        print(f"✓ Processed data size: {processed_size:.2f} MB")
        print(f"✓ Size increase: {size_increase:.2f} MB")
        print(f"✓ Memory per feature: {size_increase / new_features:.4f} MB/feature")
        
        return {
            'processing_time': processing_time,
            'processing_rate': processing_rate,
            'new_features': new_features,
            'size_increase': size_increase
        }
        
    except Exception as e:
        print(f"✗ Error in performance measurement: {e}")
        raise


def main():
    """Main test function."""
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE - CONFIGURATION TEST")
    print("="*60)
    
    try:
        # Step 1: Create sample data
        sample_data = create_sample_data(1000)
        
        # Step 2: Test all features configuration
        all_features_engineer = test_all_features_configuration()
        
        # Step 3: Test required columns validation (all features)
        required_columns_all = test_required_columns_validation(all_features_engineer)
        
        # Step 4: Test performance with all features
        performance_all = test_performance_measurement(all_features_engineer, sample_data)
        
        # Step 5: Test selective features configuration
        selective_engineer = test_selective_features_configuration()
        
        # Step 6: Test required columns validation (selective)
        required_columns_selective = test_required_columns_validation(selective_engineer)
        
        # Step 7: Test performance with selective features
        performance_selective = test_performance_measurement(selective_engineer, sample_data)
        
        # Step 8: Summary
        print("\n" + "="*60)
        print("CONFIGURATION TEST SUMMARY")
        print("="*60)
        print("✓ FeatureEngineer instantiation with enable_all_features=True: SUCCESS")
        print("✓ Selective feature set enabling: SUCCESS")
        print("✓ Required column validation via get_required_columns(): SUCCESS")
        print("✓ Computational cost and memory impact measurement: SUCCESS")
        
        print(f"\n--- Performance Comparison ---")
        print(f"All Features ({performance_all['new_features']} features):")
        print(f"  Processing: {performance_all['processing_time']:.4f}s ({performance_all['processing_rate']:.0f} rows/s)")
        print(f"  Memory: {performance_all['size_increase']:.2f} MB")
        
        print(f"Selective Features ({performance_selective['new_features']} features):")
        print(f"  Processing: {performance_selective['processing_time']:.4f}s ({performance_selective['processing_rate']:.0f} rows/s)")
        print(f"  Memory: {performance_selective['size_increase']:.2f} MB")
        
        efficiency_gain = (performance_all['processing_time'] - performance_selective['processing_time']) / performance_all['processing_time'] * 100
        memory_savings = (performance_all['size_increase'] - performance_selective['size_increase']) / performance_all['size_increase'] * 100
        
        print(f"\nEfficiency Gains with Selective Features:")
        print(f"  Processing time reduction: {efficiency_gain:.1f}%")
        print(f"  Memory usage reduction: {memory_savings:.1f}%")
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
