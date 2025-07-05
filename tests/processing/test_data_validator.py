import pandas as pd
import pytest
from datetime import time
from src.data.processing.data_validator import build_default_market_validator


def test_build_default_market_validator():
    # Create a sample DataFrame
    data = {
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    # Create the validator
    validator = build_default_market_validator()

    # Validate the DataFrame
    is_valid, error_messages = validator.validate(df)

    # Assert that the DataFrame is valid and there are no error messages
    assert is_valid
    assert len(error_messages) == 0

    # Test with missing OHLCV column
    df_missing_volume = df.drop(columns=["volume"])
    is_valid, error_messages = validator.validate(df_missing_volume)

    # Assert that validation fails and the appropriate error message is returned
    assert not is_valid
    assert "Missing OHLCV columns" in error_messages

    # Test with volume negative
    df_negative_volume = df.copy()
    df_negative_volume["volume"][0] = -10.0
    is_valid, error_messages = validator.validate(df_negative_volume)

    # Assert that validation fails and the appropriate error message is returned
    assert not is_valid
    assert "Volume has negative values" in error_messages


def test_enhanced_market_validator_with_timestamps():
    """Test the enhanced market validator with timestamp validation."""
    # Create data with timestamps
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 09:30:00",
            "2024-01-01 09:31:00",
            "2024-01-01 09:32:00"
        ]),
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    # Create validator with all enhanced features
    validator = build_default_market_validator(
        symbol_whitelist={"AAPL", "MSFT"},
        trading_hours_start=time(9, 30),
        trading_hours_end=time(16, 0),
        max_timestamp_gap_minutes=5
    )

    # Validate the DataFrame
    is_valid, error_messages = validator.validate(df)

    # Should pass all validations
    assert is_valid, f"Enhanced validation should pass. Errors: {error_messages}"
    assert len(error_messages) == 0


def test_negative_price_validation():
    """Test validation of negative prices."""
    data = {
        "open": [-100.0, 105.0, 102.0],  # Negative open price
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator()
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("Negative prices detected" in msg for msg in error_messages)


def test_price_bounds_validation():
    """Test validation of price bounds (high >= open/close/low)."""
    data = {
        "open": [100.0, 105.0, 102.0],
        "high": [95.0, 108.0, 105.0],   # First high < open
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator()
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("High price is less than" in msg for msg in error_messages)


def test_symbol_whitelist_validation():
    """Test symbol whitelist validation."""
    data = {
        "symbol": ["AAPL", "INVALID", "MSFT"],  # INVALID not in whitelist
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator(symbol_whitelist={"AAPL", "MSFT"})
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("not in whitelist" in msg for msg in error_messages)


def test_trading_hours_validation():
    """Test trading hours validation."""
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 08:00:00",  # Before trading hours
            "2024-01-01 10:00:00",  # During trading hours
            "2024-01-01 17:00:00"   # After trading hours
        ]),
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator(
        trading_hours_start=time(9, 30),
        trading_hours_end=time(16, 0)
    )
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("outside trading hours" in msg for msg in error_messages)


def test_timestamp_monotonicity_validation():
    """Test timestamp monotonicity validation."""
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 09:30:00",
            "2024-01-01 09:32:00",  # Jump forward
            "2024-01-01 09:31:00"   # Jump backward - not monotonic
        ]),
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator()
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("monotonic increasing order" in msg for msg in error_messages)


def test_duplicate_timestamps_validation():
    """Test duplicate timestamp validation."""
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 09:30:00",
            "2024-01-01 09:31:00",
            "2024-01-01 09:31:00"   # Duplicate
        ]),
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator()
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("Duplicate timestamps" in msg for msg in error_messages)


def test_timestamp_gap_validation():
    """Test timestamp gap detection."""
    data = {
        "timestamp": pd.to_datetime([
            "2024-01-01 09:30:00",
            "2024-01-01 09:31:00",
            "2024-01-01 09:41:00"   # 10-minute gap (exceeds default 5-minute threshold)
        ]),
        "open": [100.0, 105.0, 102.0],
        "high": [110.0, 108.0, 105.0],
        "low": [95.0, 102.0, 101.0],
        "close": [105.0, 104.0, 102.0],
        "volume": [200.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)

    validator = build_default_market_validator(max_timestamp_gap_minutes=5)
    is_valid, error_messages = validator.validate(df)

    assert not is_valid
    assert any("gaps exceeding" in msg for msg in error_messages)


if __name__ == "__main__":
    pytest.main()

