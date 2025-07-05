# Market Data Validation Implementation

This document describes the comprehensive market data validation enhancements implemented for the Friday AI Trading System.

## Overview

The `build_default_market_validator()` function has been enhanced with comprehensive validation rules for market data integrity, including timestamp validation, price bounds checking, symbol whitelisting, and trading hours validation.

## Enhanced Validation Rules

### 1. Timestamp Validation Rules

#### Monotonicity Check
- **Rule Name**: `timestamp_monotonicity`
- **Purpose**: Ensures timestamps are in monotonic increasing order
- **Error Message**: "Timestamps are not in monotonic increasing order"

#### Duplicate Timestamp Detection
- **Rule Name**: `no_duplicate_timestamps`
- **Purpose**: Detects duplicate timestamps in the data
- **Error Message**: "Duplicate timestamps found in data"

#### Gap Detection
- **Rule Name**: `timestamp_gap_detection`
- **Purpose**: Identifies gaps in timestamps that exceed a configurable threshold
- **Configuration**: `max_timestamp_gap_minutes` parameter (default: 5 minutes)
- **Error Message**: "Timestamp gaps exceeding {X} minutes detected"

### 2. Price Validation Rules

#### Negative Price Detection
- **Rule Name**: `no_negative_prices`
- **Purpose**: Ensures no negative prices in OHLC data
- **Error Message**: "Negative prices detected in OHLC data"

#### High Price Bounds
- **Rule Name**: `high_price_bounds`
- **Purpose**: Validates that high ≥ open, close, and low
- **Error Message**: "High price is less than open, close, or low price"

#### Low Price Bounds
- **Rule Name**: `low_price_bounds`
- **Purpose**: Validates that low ≤ open, close, and high
- **Error Message**: "Low price is greater than open, close, or high price"

#### High-Low Consistency
- **Rule Name**: `high_low_consistency`
- **Purpose**: Ensures high ≥ low (original rule)
- **Error Message**: "'high' column has values less than 'low' column"

### 3. Volume Validation

#### Non-Negative Volume
- **Rule Name**: `non_negative_volume`
- **Purpose**: Ensures volume is non-negative
- **Error Message**: "Volume has negative values"

### 4. Symbol Whitelist Validation

#### Symbol Whitelist Check
- **Rule Name**: `symbol_whitelist` (only added if whitelist provided)
- **Purpose**: Validates that all symbols are in the approved whitelist
- **Configuration**: `symbol_whitelist` parameter (Set of allowed symbols)
- **Error Message**: "Symbols not in whitelist detected. Allowed symbols: {whitelist}"

### 5. Trading Hours Validation

#### Trading Hours Check
- **Rule Name**: `trading_hours` (only added if both start/end times provided)
- **Purpose**: Ensures data timestamps are within trading hours
- **Configuration**: 
  - `trading_hours_start`: Start time (e.g., time(9, 30))
  - `trading_hours_end`: End time (e.g., time(16, 0))
- **Error Message**: "Data outside trading hours detected. Trading hours: {start} - {end}"

### 6. Data Type Validation

#### OHLCV Type Check
- **Rule Name**: `ohlcv_types`
- **Purpose**: Ensures OHLCV columns are of float type
- **Error Message**: "OHLCV columns must be of type float"

## Configuration Options

The `build_default_market_validator()` function accepts the following optional parameters:

```python
def build_default_market_validator(
    symbol_whitelist: Optional[Set[str]] = None,
    trading_hours_start: Optional[time] = None,
    trading_hours_end: Optional[time] = None,
    max_timestamp_gap_minutes: int = 5
) -> DataValidator:
```

### Parameters

- **symbol_whitelist**: Set of allowed symbols. If provided, only these symbols are allowed.
- **trading_hours_start**: Start time for trading hours validation.
- **trading_hours_end**: End time for trading hours validation.
- **max_timestamp_gap_minutes**: Maximum allowed gap between consecutive timestamps in minutes.

## Usage Examples

### Basic Usage
```python
from src.data.processing.data_validator import build_default_market_validator

# Create validator with default settings
validator = build_default_market_validator()
is_valid, errors = validator.validate(data)
```

### Advanced Configuration
```python
from datetime import time

# Create validator with comprehensive configuration
validator = build_default_market_validator(
    symbol_whitelist={'AAPL', 'MSFT', 'GOOGL', 'TSLA'},
    trading_hours_start=time(9, 30),  # 9:30 AM
    trading_hours_end=time(16, 0),    # 4:00 PM
    max_timestamp_gap_minutes=10      # 10-minute tolerance
)

is_valid, errors = validator.validate(data)
```

### Selective Rule Validation
```python
# Validate only specific rules
is_valid, errors = validator.validate(
    data, 
    rules=['timestamp_monotonicity', 'no_negative_prices']
)
```

## Test Coverage

Comprehensive unit tests have been implemented covering:

### Basic Validation Tests
- Valid OHLCV data validation
- Missing column detection
- Invalid data type detection

### Timestamp Validation Tests
- Monotonicity validation (valid/invalid)
- Duplicate timestamp detection
- Gap detection with custom thresholds
- Graceful handling of missing timestamp columns

### Price Validation Tests
- Negative price detection
- High/low price bounds validation
- High-low consistency validation

### Volume Validation Tests
- Non-negative volume validation
- Zero volume edge case

### Symbol Whitelist Tests
- Valid/invalid symbol validation
- Graceful handling when no whitelist configured
- Missing symbol column handling

### Trading Hours Tests
- Valid/invalid trading hours validation
- Boundary condition testing
- Partial configuration handling

### Integration Tests
- Full validation with all rules
- Multiple validation failures
- Selective rule validation
- Empty DataFrame handling
- Single-row data validation

### Edge Cases
- Zero volume validation
- Equal OHLC prices
- Minimal timestamp differences
- Boundary trading hours

## Error Handling

The validation system includes robust error handling:

- **Type Safety**: All numeric comparisons are protected against type mismatches
- **Missing Columns**: Graceful handling when expected columns are missing
- **Data Conversion**: Automatic conversion to numeric types with error handling
- **Empty Data**: Proper handling of empty DataFrames

## File Structure

```
tests/data/validation/
├── __init__.py               # Package init
├── test_fixtures.py          # Test data fixtures
├── test_market_validator.py  # Comprehensive unit tests
└── README.md                # This documentation
```

## Performance

The validation system is designed for efficiency:
- Rules are executed only when relevant columns exist
- Early termination on rule failures
- Minimal data copying and transformation
- Configurable rule selection for performance optimization

## Future Enhancements

Potential areas for future enhancement:
- Statistical outlier detection
- Inter-symbol validation rules
- Real-time validation streaming
- Custom validation rule plugins
- Performance profiling and optimization
