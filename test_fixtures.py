"""Test fixtures for market data validation tests."""

import pandas as pd
from datetime import datetime, time, timedelta
import numpy as np

# Test constants
VALID_SYMBOL_WHITELIST = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
TRADING_HOURS_START = time(9, 30)  # 9:30 AM
TRADING_HOURS_END = time(16, 0)    # 4:00 PM
DEFAULT_MAX_GAP_MINUTES = 5


def create_valid_ohlcv_data(symbol='AAPL', periods=10):
    """Create valid OHLCV data for testing."""
    base_time = datetime(2023, 1, 1, 10, 0)  # Start at 10:00 AM
    timestamps = [base_time + timedelta(minutes=i) for i in range(periods)]
    
    # Create realistic price data
    open_prices = [100.0 + i * 0.5 for i in range(periods)]
    high_prices = [price + 1.0 for price in open_prices]
    low_prices = [price - 0.5 for price in open_prices]
    close_prices = [price + 0.25 for price in open_prices]
    volumes = [1000 + i * 100 for i in range(periods)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })


def create_missing_columns_data():
    """Create data with missing OHLCV columns."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        # Missing 'high', 'low', 'close', 'volume' columns
    })


def create_non_monotonic_timestamps():
    """Create data with non-monotonic timestamps."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=1),
        base_time + timedelta(minutes=3),  # Jump ahead
        base_time + timedelta(minutes=2),  # Back in time - non-monotonic
        base_time + timedelta(minutes=4)
    ]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_duplicate_timestamps():
    """Create data with duplicate timestamps."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=1),
        base_time + timedelta(minutes=1),  # Duplicate
        base_time + timedelta(minutes=2),
        base_time + timedelta(minutes=3)
    ]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 100.6, 101.0, 101.5],
        'high': [101.0, 101.5, 101.6, 102.0, 102.5],
        'low': [99.5, 100.0, 100.1, 100.5, 101.0],
        'close': [100.25, 100.75, 100.85, 101.25, 101.75],
        'volume': [1000, 1100, 1150, 1200, 1300]
    })


def create_large_timestamp_gaps():
    """Create data with large timestamp gaps."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=1),
        base_time + timedelta(minutes=2),
        base_time + timedelta(minutes=12),  # 10-minute gap
        base_time + timedelta(minutes=13)
    ]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_negative_prices():
    """Create data with negative prices."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, -101.0, 101.5, 102.0],  # Negative open price
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, -101.5, 101.0, 101.5],    # Negative low price
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_invalid_price_bounds():
    """Create data with invalid price bounds (high < low, etc.)."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [99.0, 101.5, 102.0, 102.5, 103.0],    # High < open in first row
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_high_less_than_low():
    """Create data where high price is less than low price."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [99.0, 101.5, 100.0, 102.5, 103.0],    # High < low in positions 0 and 2
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_negative_volume():
    """Create data with negative volume."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, -1100, 1200, 1300, 1400]  # Negative volume
    })


def create_invalid_symbol_data():
    """Create data with invalid symbols."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'INVALID_SYMBOL',  # Not in whitelist
        'open': [100.0, 100.5, 101.0, 101.5, 102.0],
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


def create_outside_trading_hours():
    """Create data with timestamps outside trading hours."""
    base_date = datetime(2023, 1, 1)
    timestamps = [
        datetime.combine(base_date.date(), time(8, 0)),   # Before market open
        datetime.combine(base_date.date(), time(10, 0)),  # During market hours
        datetime.combine(base_date.date(), time(17, 0)),  # After market close
        datetime.combine(base_date.date(), time(18, 0)),  # After market close
    ]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': [100.0, 100.5, 101.0, 101.5],
        'high': [101.0, 101.5, 102.0, 102.5],
        'low': [99.5, 100.0, 100.5, 101.0],
        'close': [100.25, 100.75, 101.25, 101.75],
        'volume': [1000, 1100, 1200, 1300]
    })


def create_invalid_data_types():
    """Create data with invalid data types."""
    base_time = datetime(2023, 1, 1, 10, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'AAPL',
        'open': ['100.0', '100.5', '101.0', '101.5', '102.0'],  # String instead of float
        'high': [101.0, 101.5, 102.0, 102.5, 103.0],
        'low': [99.5, 100.0, 100.5, 101.0, 101.5],
        'close': [100.25, 100.75, 101.25, 101.75, 102.25],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
