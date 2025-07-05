"""Test fixtures for market data validation tests."""

import pandas as pd
from datetime import datetime, time
from typing import Dict, Any

def create_valid_ohlcv_data() -> pd.DataFrame:
    """Create a valid OHLCV dataset for testing."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:32:00',
            '2024-01-01 09:33:00',
            '2024-01-01 09:34:00'
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0, 151.5],
        'high': [151.5, 152.0, 152.0, 153.0, 152.5],  # Ensure high >= all other prices
        'low': [149.5, 150.0, 149.8, 151.0, 150.5],
        'close': [151.0, 150.5, 152.0, 151.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0, 110000.0]
    })

def create_missing_columns_data() -> pd.DataFrame:
    """Create dataset with missing OHLCV columns."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'open': [150.0, 151.0],
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        # Missing 'close' and 'volume' columns
    })

def create_non_monotonic_timestamps() -> pd.DataFrame:
    """Create dataset with non-monotonic timestamps."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:32:00',  # Jump forward
            '2024-01-01 09:31:00',  # Jump backward - breaks monotonicity
            '2024-01-01 09:33:00',
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0],
        'high': [151.5, 152.0, 151.0, 153.0],
        'low': [149.5, 150.0, 149.8, 151.0],
        'close': [151.0, 150.5, 152.0, 151.5],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0]
    })

def create_duplicate_timestamps() -> pd.DataFrame:
    """Create dataset with duplicate timestamps."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:31:00',  # Duplicate
            '2024-01-01 09:32:00',
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0],
        'high': [151.5, 152.0, 151.0, 153.0],
        'low': [149.5, 150.0, 149.8, 151.0],
        'close': [151.0, 150.5, 152.0, 151.5],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0]
    })

def create_large_timestamp_gaps() -> pd.DataFrame:
    """Create dataset with large timestamp gaps."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:41:00',  # 10-minute gap (larger than default 5-minute threshold)
            '2024-01-01 09:42:00',
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0],
        'high': [151.5, 152.0, 151.0, 153.0],
        'low': [149.5, 150.0, 149.8, 151.0],
        'close': [151.0, 150.5, 152.0, 151.5],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0]
    })

def create_negative_prices() -> pd.DataFrame:
    """Create dataset with negative prices."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:32:00',
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, -150.5],  # Negative open price
        'high': [151.5, 152.0, 151.0],
        'low': [149.5, 150.0, 149.8],
        'close': [151.0, 150.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0]
    })

def create_invalid_price_bounds() -> pd.DataFrame:
    """Create dataset with invalid price bounds (high < low, etc.)."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:32:00',
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5],
        'high': [149.0, 152.0, 151.0],  # First high < open
        'low': [149.5, 150.0, 149.8],
        'close': [151.0, 150.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0]
    })

def create_high_less_than_low() -> pd.DataFrame:
    """Create dataset where high < low."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
        ]),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [149.0, 150.0],  # High < Low
        'low': [149.5, 150.5],
        'close': [151.0, 150.5],
        'volume': [100000.0, 85000.0]
    })

def create_negative_volume() -> pd.DataFrame:
    """Create dataset with negative volume."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
        ]),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        'close': [151.0, 150.5],
        'volume': [100000.0, -85000.0]  # Negative volume
    })

def create_invalid_symbol_data() -> pd.DataFrame:
    """Create dataset with symbols not in whitelist."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:32:00',
        ]),
        'symbol': ['AAPL', 'MSFT', 'INVALID_SYMBOL'],  # INVALID_SYMBOL not in whitelist
        'open': [150.0, 151.0, 150.5],
        'high': [151.5, 152.0, 151.0],
        'low': [149.5, 150.0, 149.8],
        'close': [151.0, 150.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0]
    })

def create_outside_trading_hours() -> pd.DataFrame:
    """Create dataset with timestamps outside trading hours."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 08:30:00',  # Before trading hours (assuming 9:30-16:00)
            '2024-01-01 09:31:00',  # During trading hours
            '2024-01-01 17:00:00',  # After trading hours
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5],
        'high': [151.5, 152.0, 151.0],
        'low': [149.5, 150.0, 149.8],
        'close': [151.0, 150.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0]
    })

def create_invalid_data_types() -> pd.DataFrame:
    """Create dataset with invalid data types."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
        ]),
        'symbol': ['AAPL', 'AAPL'],
        'open': ['150.0', '151.0'],  # String instead of float
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        'close': [151.0, 150.5],
        'volume': [100000.0, 85000.0]
    })

# Test configuration constants
VALID_SYMBOL_WHITELIST = {'AAPL', 'MSFT', 'GOOGL', 'TSLA'}
TRADING_HOURS_START = time(9, 30)  # 9:30 AM
TRADING_HOURS_END = time(16, 0)    # 4:00 PM
DEFAULT_MAX_GAP_MINUTES = 5
