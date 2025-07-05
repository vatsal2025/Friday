#!/usr/bin/env python3
"""
Enhanced Market Data Cleaner Example

This example demonstrates the refined MarketDataCleaner with:
- ConfigManager integration for threshold management
- EventSystem integration for cleaning metrics
- Enhanced duplicate removal, outlier detection, and gap filling
- Comprehensive logging and reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.config import ConfigManager
from infrastructure.event import EventSystem
from data.processing.market_data_cleaner import build_market_data_cleaner


def create_sample_dirty_data():
    """Create sample market data with various data quality issues."""
    
    # Create base data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            # Generate base OHLCV data
            base_price = 100 + np.random.normal(0, 10)
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': base_price + np.random.normal(0, 1),
                'high': base_price + abs(np.random.normal(0, 2)),
                'low': base_price - abs(np.random.normal(0, 2)),
                'close': base_price + np.random.normal(0, 1),
                'volume': int(abs(np.random.normal(1000000, 200000)))
            })
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    
    # 1. Add duplicates (5% of data)
    duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Add bad numeric values (corrupted strings)
    bad_value_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[bad_value_indices[:len(bad_value_indices)//2], 'close'] = ['$123.45', '(67.89)', '100,000', '$invalid', 'N/A'][:len(bad_value_indices)//2]
    df.loc[bad_value_indices[len(bad_value_indices)//2:], 'volume'] = ['1,000,000', '(500,000)', 'N/A', 'null', '2.5M'][:len(bad_value_indices[len(bad_value_indices)//2:])]
    
    # 3. Add extreme outliers (z-score > 4)
    outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[outlier_indices, 'high'] = df['high'].mean() + 6 * df['high'].std()  # Extreme outliers
    
    # 4. Add missing values/gaps
    gap_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[gap_indices, ['open', 'close']] = np.nan
    
    print(f"Created sample dirty data with {len(df)} rows")
    print(f"- Duplicates introduced: {len(duplicate_indices)}")
    print(f"- Bad values introduced: {len(bad_value_indices)}")
    print(f"- Outliers introduced: {len(outlier_indices)}")
    print(f"- Gaps introduced: {len(gap_indices)}")
    
    return df


def setup_event_listener(event_system):
    """Set up event listeners to capture cleaning metrics."""
    
    cleaning_events = []
    
    def capture_cleaning_event(event):
        """Capture and store cleaning events."""
        cleaning_events.append({
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'data': event.data
        })
        print(f"📊 Event: {event.event_type} - {event.data.get('cleaning_stage', 'unknown')}")
        
        # Print key metrics
        if 'total_corrected' in event.data:
            print(f"   └─ Bad casts corrected: {event.data['total_corrected']}")
        if 'duplicates_removed' in event.data:
            print(f"   └─ Duplicates removed: {event.data['duplicates_removed']}")
        if 'total_outliers_capped' in event.data:
            print(f"   └─ Outliers capped: {event.data['total_outliers_capped']}")
        if 'total_filled' in event.data:
            print(f"   └─ Gaps filled: {event.data['total_filled']}")
        if 'data_quality_score' in event.data:
            print(f"   └─ Quality score: {event.data['data_quality_score']:.3f}")
    
    # Register event listeners for all cleaning events
    event_system.subscribe('data.cleaning.*', capture_cleaning_event)
    
    return cleaning_events


def main():
    """Main demonstration function."""
    
    print("🧹 Enhanced Market Data Cleaner Example")
    print("=" * 50)
    
    # 1. Setup configuration with custom cleaning thresholds
    print("\n1. Setting up configuration...")
    config = ConfigManager()
    
    # Override some cleaning configuration for demonstration
    config.set('data.cleaning.z_score_threshold', 2.5)  # More sensitive outlier detection
    config.set('data.cleaning.iqr_multiplier', 1.8)
    config.set('data.cleaning.max_outlier_percentage', 0.10)  # Allow higher outlier rate
    config.set('data.cleaning.enable_detailed_logging', True)
    
    print("✅ Configuration loaded with custom cleaning thresholds")
    
    # 2. Setup event system
    print("\n2. Setting up event system...")
    event_system = EventSystem()
    cleaning_events = setup_event_listener(event_system)
    print("✅ Event system configured with cleaning listeners")
    
    # 3. Create sample dirty data
    print("\n3. Creating sample dirty data...")
    dirty_data = create_sample_dirty_data()
    print(f"✅ Sample data created: {dirty_data.shape}")
    print(f"   Data types: {dirty_data.dtypes.to_dict()}")
    print(f"   Missing values: {dirty_data.isnull().sum().sum()}")
    
    # 4. Build enhanced cleaner
    print("\n4. Building enhanced MarketDataCleaner...")
    cleaner = build_market_data_cleaner(
        config=config,
        event_system=event_system,
        enable_detailed_logging=True
    )
    print("✅ Enhanced MarketDataCleaner built successfully")
    
    # 5. Clean the data
    print("\n5. Running enhanced cleaning pipeline...")
    print("-" * 30)
    
    try:
        cleaned_data = cleaner.clean_market_data(dirty_data)
        print("-" * 30)
        print(f"✅ Cleaning completed successfully!")
        print(f"   Original shape: {dirty_data.shape}")
        print(f"   Cleaned shape: {cleaned_data.shape}")
        print(f"   Rows removed: {dirty_data.shape[0] - cleaned_data.shape[0]}")
        
    except Exception as e:
        print(f"❌ Cleaning failed: {str(e)}")
        return
    
    # 6. Generate comprehensive cleaning report
    print("\n6. Generating cleaning report...")
    report = cleaner.get_cleaning_report()
    
    print("\n📋 CLEANING REPORT")
    print("=" * 50)
    
    # Operations summary
    ops = report['operations_summary']
    print(f"Operations Performed:")
    print(f"  • Duplicates removed: {ops['duplicates_removed']}")
    print(f"  • Bad casts corrected: {ops['bad_casts_corrected']}")
    print(f"  • Outliers capped: {ops['outliers_capped']}")
    print(f"  • Gaps filled: {ops['gaps_filled']}")
    print(f"  • Total rows modified: {ops['total_rows_modified']}")
    
    # Performance summary
    perf = report['performance_summary']
    print(f"\nPerformance Metrics:")
    print(f"  • Total processing time: {perf['total_processing_time']:.3f}s")
    print(f"  • Data quality score: {perf['data_quality_score']:.3f}")
    
    # Stage times
    if perf['stage_times']:
        print(f"  • Stage breakdown:")
        for stage, time_taken in perf['stage_times'].items():
            print(f"    - {stage}: {time_taken:.3f}s")
    
    # Quality assessment
    quality = report['quality_assessment']
    print(f"\nQuality Assessment:")
    print(f"  • Overall quality score: {quality['overall_quality_score']:.3f}")
    print(f"  • Outlier rate: {quality['outlier_rate']:.2%}")
    print(f"  • Duplicate removal rate: {quality['duplicate_removal_rate']:.2%}")
    print(f"  • Gap fill success rate: {quality['gap_fill_success_rate']:.2%}")
    
    # 7. Show captured events
    print(f"\n7. Event System Integration")
    print("=" * 50)
    print(f"Total cleaning events captured: {len(cleaning_events)}")
    
    for i, event in enumerate(cleaning_events, 1):
        print(f"\nEvent {i}: {event['event_type']}")
        print(f"  Timestamp: {event['timestamp']}")
        if 'processing_time' in event['data']:
            print(f"  Processing time: {event['data']['processing_time']:.3f}s")
    
    # 8. Data validation
    print(f"\n8. Final Data Validation")
    print("=" * 50)
    
    # Check data types
    print("Data Types After Cleaning:")
    for col, dtype in cleaned_data.dtypes.items():
        print(f"  • {col}: {dtype}")
    
    # Check for remaining issues
    remaining_nulls = cleaned_data.isnull().sum().sum()
    duplicate_check = cleaned_data.duplicated().sum()
    
    print(f"\nRemaining Data Issues:")
    print(f"  • Missing values: {remaining_nulls}")
    print(f"  • Duplicates: {duplicate_check}")
    
    # Sample of cleaned data
    print(f"\nSample of Cleaned Data (first 5 rows):")
    print(cleaned_data.head().to_string())
    
    print(f"\n🎉 Enhanced Market Data Cleaner demonstration completed successfully!")
    print(f"📊 Data quality improved from unknown to {quality['overall_quality_score']:.3f}")
    
    return cleaned_data, report, cleaning_events


if __name__ == "__main__":
    # Run the example
    try:
        cleaned_data, report, events = main()
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
