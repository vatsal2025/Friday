#!/usr/bin/env python3
"""Example script demonstrating corporate actions adjustment and timezone normalization.

This script shows how to use the new data cleaning features:
1. Loading and applying corporate actions to adjust OHLCV data for splits and dividends
2. Normalizing datetime columns to UTC timezone while preserving metadata

Usage:
    python corporate_actions_timezone_example.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os

# Import the data cleaner and related functions
from src.data.processing.data_cleaner import (
    DataCleaner,
    build_default_market_cleaner,
    CorporateAction,
    load_corporate_actions_from_file,
    adjust_for_corporate_actions,
    normalize_timezone
)


def create_sample_ohlcv_data():
    """Create sample OHLCV data for demonstration."""
    # Create date range for 100 trading days
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=100, freq='D')
    
    # Create sample OHLCV data
    np.random.seed(42)  # For reproducible results
    
    base_price = 100.0
    data = []
    
    for i, date in enumerate(dates):
        # Generate realistic OHLCV data with some randomness
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
        
        # Calculate prices
        if i == 0:
            close = base_price
        else:
            close = data[-1]['close'] * (1 + daily_return)
        
        # OHLC based on close with some intraday volatility
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        
        # Volume
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def create_sample_corporate_actions_file():
    """Create a sample corporate actions CSV file."""
    corporate_actions_data = [
        {
            'symbol': 'AAPL',
            'date': '2023-02-15',
            'action_type': 'split',
            'factor': 2.0,
            'dividend_amount': None,
            'currency': 'USD'
        },
        {
            'symbol': 'AAPL',
            'date': '2023-01-31',
            'action_type': 'dividend',
            'factor': 1.0,
            'dividend_amount': 0.50,
            'currency': 'USD'
        },
        {
            'symbol': 'MSFT',
            'date': '2023-03-10',
            'action_type': 'dividend',
            'factor': 1.0,
            'dividend_amount': 0.68,
            'currency': 'USD'
        }
    ]
    
    df = pd.DataFrame(corporate_actions_data)
    file_path = 'sample_corporate_actions.csv'
    df.to_csv(file_path, index=False)
    print(f"Created sample corporate actions file: {file_path}")
    return file_path


def create_sample_timezone_data():
    """Create sample data with timezone-aware datetime columns."""
    # Create data in different timezones
    eastern_tz = pytz.timezone('US/Eastern')
    london_tz = pytz.timezone('Europe/London')
    
    # Create timestamps in Eastern timezone
    start_date = datetime(2023, 6, 1, 9, 30)  # 9:30 AM Eastern
    eastern_times = pd.date_range(
        start=eastern_tz.localize(start_date),
        periods=50,
        freq='H'
    )
    
    # Create some sample market data with timezone-aware timestamps
    data = {
        'price': np.random.uniform(90, 110, len(eastern_times)),
        'volume': np.random.randint(1000, 10000, len(eastern_times)),
        'trade_time': eastern_times,
        'settlement_time': eastern_times + timedelta(days=2)
    }
    
    df = pd.DataFrame(data, index=eastern_times)
    return df


def demonstrate_corporate_actions():
    """Demonstrate corporate actions adjustment functionality."""
    print("\n" + "="*60)
    print("CORPORATE ACTIONS ADJUSTMENT DEMONSTRATION")
    print("="*60)
    
    # Create sample OHLCV data
    print("1. Creating sample OHLCV data...")
    ohlcv_data = create_sample_ohlcv_data()
    print(f"   Created {len(ohlcv_data)} days of OHLCV data")
    print(f"   Date range: {ohlcv_data.index[0].date()} to {ohlcv_data.index[-1].date()}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    
    # Create sample corporate actions file
    print("\n2. Creating sample corporate actions file...")
    corporate_actions_file = create_sample_corporate_actions_file()
    
    # Load corporate actions
    print("\n3. Loading corporate actions...")
    corporate_actions = load_corporate_actions_from_file(corporate_actions_file)
    print(f"   Loaded {len(corporate_actions)} corporate actions:")
    for action in corporate_actions:
        print(f"   - {action}")
    
    # Apply corporate actions to AAPL data
    print("\n4. Applying corporate actions to AAPL data...")
    symbol = 'AAPL'
    
    # Show data before adjustment
    print(f"\n   BEFORE ADJUSTMENT (first 5 rows):")
    print(f"   {ohlcv_data.head().round(2)}")
    
    # Apply adjustments
    adjusted_data, metadata = adjust_for_corporate_actions(
        data=ohlcv_data,
        corporate_actions=corporate_actions,
        symbol=symbol,
        preserve_original_close=True
    )
    
    # Show data after adjustment
    print(f"\n   AFTER ADJUSTMENT (first 5 rows):")
    print(f"   {adjusted_data.head().round(2)}")
    
    # Show adjustment metadata
    print(f"\n   ADJUSTMENT METADATA:")
    print(f"   - Symbol: {metadata['symbol']}")
    print(f"   - Adjustments applied: {metadata['adjustments_applied']}")
    print(f"   - Cumulative split factor: {metadata['cumulative_split_factor']}")
    print(f"   - Cumulative dividend adjustment: ${metadata['cumulative_dividend_adjustment']:.2f}")
    
    for detail in metadata['adjustment_details']:
        if detail['type'] == 'split':
            print(f"   - Split: {detail['factor']}:1 on {detail['date'].date()}, affected {detail['records_affected']} records")
        elif detail['type'] == 'dividend':
            print(f"   - Dividend: ${detail['amount']:.2f} on {detail['date'].date()}, factor: {detail['factor']:.6f}, affected {detail['records_affected']} records")
    
    # Demonstrate using DataCleaner convenience method
    print("\n5. Using DataCleaner convenience method...")
    cleaner = build_default_market_cleaner()
    adjusted_data_cleaner = cleaner.apply_corporate_actions(
        data=ohlcv_data,
        corporate_actions_file=corporate_actions_file,
        symbol=symbol
    )
    
    print(f"   DataCleaner metadata keys: {list(cleaner.metadata.keys())}")
    
    # Cleanup
    os.remove(corporate_actions_file)
    print(f"\n   Cleaned up sample file: {corporate_actions_file}")


def demonstrate_timezone_normalization():
    """Demonstrate timezone normalization functionality."""
    print("\n" + "="*60)
    print("TIMEZONE NORMALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create sample timezone-aware data
    print("1. Creating sample timezone-aware data...")
    timezone_data = create_sample_timezone_data()
    print(f"   Created {len(timezone_data)} records with timezone-aware timestamps")
    print(f"   Original timezone: {timezone_data.index.tz}")
    print(f"   Date range: {timezone_data.index[0]} to {timezone_data.index[-1]}")
    
    # Show sample of original data
    print(f"\n   BEFORE NORMALIZATION (first 3 rows):")
    print(f"   Index timezone: {timezone_data.index.tz}")
    print(f"   Trade time timezone: {timezone_data['trade_time'].dt.tz}")
    print(timezone_data.head(3))
    
    # Apply timezone normalization
    print("\n2. Applying timezone normalization to UTC...")
    normalized_data, metadata = normalize_timezone(
        data=timezone_data,
        source_timezone='US/Eastern',  # Explicit source timezone
        target_timezone='UTC',
        preserve_metadata=True
    )
    
    # Show sample of normalized data
    print(f"\n   AFTER NORMALIZATION (first 3 rows):")
    print(f"   Index timezone: {normalized_data.index.tz}")
    print(f"   Trade time timezone: {normalized_data['trade_time'].dt.tz}")
    print(normalized_data.head(3))
    
    # Show normalization metadata
    print(f"\n   NORMALIZATION METADATA:")
    print(f"   - Source timezone: {metadata['source_timezone']}")
    print(f"   - Target timezone: {metadata['target_timezone']}")
    print(f"   - Datetime columns converted: {metadata['datetime_columns_converted']}")
    print(f"   - Conversions applied: {metadata['conversions_applied']}")
    print(f"   - Total records: {metadata['total_records']}")
    
    for detail in metadata['conversion_details']:
        print(f"   - Converted {detail['column']}: {detail['original_timezone']} -> {detail['target_timezone']} ({detail['records_converted']} records)")
    
    # Demonstrate using DataCleaner convenience method
    print("\n3. Using DataCleaner convenience method...")
    cleaner = build_default_market_cleaner()
    normalized_data_cleaner = cleaner.normalize_timezones(
        data=timezone_data,
        source_timezone='US/Eastern',
        target_timezone='UTC'
    )
    
    print(f"   DataCleaner metadata keys: {list(cleaner.metadata.keys())}")
    
    # Demonstrate timezone-naive data normalization
    print("\n4. Normalizing timezone-naive data...")
    naive_data = timezone_data.copy()
    naive_data.index = naive_data.index.tz_localize(None)  # Remove timezone info
    naive_data['trade_time'] = naive_data['trade_time'].dt.tz_localize(None)
    naive_data['settlement_time'] = naive_data['settlement_time'].dt.tz_localize(None)
    
    print(f"   Naive data index timezone: {naive_data.index.tz}")
    
    normalized_naive, naive_metadata = normalize_timezone(
        data=naive_data,
        source_timezone=None,  # No source timezone specified
        target_timezone='UTC'
    )
    
    print(f"   Normalized naive data index timezone: {normalized_naive.index.tz}")
    print(f"   Conversions applied: {naive_metadata['conversions_applied']}")


def demonstrate_integrated_processing():
    """Demonstrate integrated processing with both corporate actions and timezone normalization."""
    print("\n" + "="*60)
    print("INTEGRATED PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create sample data with timezone and corporate actions
    print("1. Creating comprehensive test scenario...")
    
    # Create timezone-aware OHLCV data
    eastern_tz = pytz.timezone('US/Eastern')
    start_date = datetime(2023, 1, 1, 9, 30)
    dates = pd.date_range(
        start=eastern_tz.localize(start_date),
        periods=50,
        freq='D'
    )
    
    # Create OHLCV data with timezone
    np.random.seed(42)
    ohlcv_data = pd.DataFrame({
        'open': np.random.uniform(95, 105, len(dates)),
        'high': np.random.uniform(100, 110, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(95, 105, len(dates)),
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    print(f"   Created timezone-aware OHLCV data: {len(ohlcv_data)} records")
    print(f"   Original timezone: {ohlcv_data.index.tz}")
    
    # Create corporate actions
    corporate_actions_file = create_sample_corporate_actions_file()
    
    # Create enhanced cleaner with both features
    print("\n2. Setting up DataCleaner with integrated processing...")
    cleaner = build_default_market_cleaner()
    
    # Add timezone normalization as a processing step
    def timezone_normalization_step(data):
        return cleaner.normalize_timezones(
            data=data,
            source_timezone='US/Eastern',
            target_timezone='UTC'
        )
    
    def corporate_actions_step(data):
        return cleaner.apply_corporate_actions(
            data=data,
            corporate_actions_file=corporate_actions_file,
            symbol='AAPL'
        )
    
    # Add these as custom processing steps
    from src.data.processing.data_processor import ProcessingStep
    cleaner.add_processing_step(ProcessingStep.NORMALIZATION, timezone_normalization_step)
    cleaner.add_processing_step(ProcessingStep.NORMALIZATION, corporate_actions_step)
    
    # Process the data
    print("\n3. Processing data with integrated pipeline...")
    processed_data = cleaner.process_data(ohlcv_data)
    
    print(f"\n   PROCESSING RESULTS:")
    print(f"   - Input records: {len(ohlcv_data)}")
    print(f"   - Output records: {len(processed_data)}")
    print(f"   - Input timezone: {ohlcv_data.index.tz}")
    print(f"   - Output timezone: {processed_data.index.tz}")
    
    # Show comprehensive metadata
    print(f"\n   COMPREHENSIVE METADATA:")
    for key, value in cleaner.metadata.items():
        if isinstance(value, dict):
            print(f"   - {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list) and len(subvalue) > 3:
                    print(f"     * {subkey}: [{len(subvalue)} items]")
                else:
                    print(f"     * {subkey}: {subvalue}")
        else:
            print(f"   - {key}: {value}")
    
    # Cleanup
    os.remove(corporate_actions_file)
    print(f"\n   Cleaned up sample file: {corporate_actions_file}")


def main():
    """Main function to run all demonstrations."""
    print("FRIDAY AI TRADING SYSTEM")
    print("Corporate Actions & Timezone Normalization Demo")
    print("=" * 60)
    
    try:
        # Demonstrate corporate actions
        demonstrate_corporate_actions()
        
        # Demonstrate timezone normalization
        demonstrate_timezone_normalization()
        
        # Demonstrate integrated processing
        demonstrate_integrated_processing()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey features demonstrated:")
        print("✓ Corporate actions adjustment (splits & dividends)")
        print("✓ Timezone normalization with metadata preservation")
        print("✓ Integration with existing DataCleaner pipeline")
        print("✓ Comprehensive error handling and logging")
        print("✓ Flexible configuration options")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
