"""Market Data Validation Rules Module

This module provides comprehensive validation rules for market data including:
- OHLCV column presence & types validation
- No-negative price/volume checks  
- High ≥ Open/Close ≥ Low consistency validation
- Timestamp uniqueness & monotonicity checks
- Trading-hours window validation (09:30–16:00 configurable)

The validation rules are designed to work with the DataValidator framework and can be
registered individually or as a complete set through build_comprehensive_market_validator.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Set
from datetime import time, datetime

from src.infrastructure.logging import get_logger
from src.data.processing.data_validator import DataValidator, ValidationRule

# Create logger
logger = get_logger(__name__)


class MarketDataValidationRules:
    """Collection of comprehensive market data validation rules.
    
    This class organizes validation rules into logical categories and provides
    factory methods for creating validators with different rule combinations.
    """
    
    @staticmethod
    def create_ohlcv_column_presence_rule() -> ValidationRule:
        """Create rule to validate presence of required OHLCV columns.
        
        Returns:
            ValidationRule: Rule that checks for open, high, low, close, volume columns
        """
        def check_ohlcv_columns(df: pd.DataFrame) -> bool:
            required_columns = {"open", "high", "low", "close", "volume"}
            return required_columns.issubset(set(df.columns))
        
        return ValidationRule(
            name="ohlcv_columns_presence",
            validation_func=check_ohlcv_columns,
            error_message="Missing required OHLCV columns. Required: open, high, low, close, volume"
        )
    
    @staticmethod
    def create_ohlcv_data_types_rule(strict_mode: bool = True) -> ValidationRule:
        """Create rule to validate OHLCV column data types.
        
        Args:
            strict_mode: If True, requires exact float dtype. If False, allows numeric convertible types.
            
        Returns:
            ValidationRule: Rule that validates OHLCV columns are numeric/float types
        """
        def check_ohlcv_types(df: pd.DataFrame) -> bool:
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            
            for col in ohlcv_columns:
                if col not in df.columns:
                    continue  # Skip missing columns (handled by presence rule)
                
                if strict_mode:
                    # Strict mode: must be float dtype
                    if not pd.api.types.is_float_dtype(df[col]):
                        logger.debug(f"Column '{col}' has dtype {df[col].dtype}, expected float")
                        return False
                else:
                    # Flexible mode: must be numeric or convertible to numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        logger.debug(f"Column '{col}' cannot be converted to numeric")
                        return False
            
            return True
        
        mode_desc = "exact float" if strict_mode else "numeric convertible"
        return ValidationRule(
            name="ohlcv_data_types",
            validation_func=check_ohlcv_types,
            error_message=f"OHLCV columns must be of {mode_desc} type"
        )
    
    @staticmethod
    def create_no_negative_prices_rule() -> ValidationRule:
        """Create rule to validate no negative prices in OHLC data.
        
        Returns:
            ValidationRule: Rule that checks for non-negative price values
        """
        def check_no_negative_prices(df: pd.DataFrame) -> bool:
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                if col not in df.columns:
                    continue  # Skip missing columns
                
                try:
                    # Convert to numeric and check for negative values
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if (numeric_col < 0).any():
                        logger.debug(f"Negative values detected in column '{col}'")
                        return False
                    
                    # Check for NaN values that might indicate conversion issues
                    if numeric_col.isna().any() and not df[col].isna().any():
                        logger.debug(f"Non-numeric values detected in column '{col}' during validation")
                        return False
                        
                except (TypeError, ValueError) as e:
                    logger.debug(f"Error validating column '{col}': {e}")
                    return False
            
            return True
        
        return ValidationRule(
            name="no_negative_prices",
            validation_func=check_no_negative_prices,
            error_message="Negative prices detected in OHLC data"
        )
    
    @staticmethod
    def create_no_negative_volume_rule() -> ValidationRule:
        """Create rule to validate no negative volume values.
        
        Returns:
            ValidationRule: Rule that checks for non-negative volume values
        """
        def check_no_negative_volume(df: pd.DataFrame) -> bool:
            if 'volume' not in df.columns:
                return True  # Skip if volume column missing
            
            try:
                # Convert to numeric and check for negative values
                numeric_volume = pd.to_numeric(df['volume'], errors='coerce')
                
                # Check for negative values
                if (numeric_volume < 0).any():
                    logger.debug("Negative values detected in volume column")
                    return False
                
                # Check for NaN values that might indicate conversion issues
                if numeric_volume.isna().any() and not df['volume'].isna().any():
                    logger.debug("Non-numeric values detected in volume column during validation")
                    return False
                    
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating volume column: {e}")
                return False
        
        return ValidationRule(
            name="no_negative_volume",
            validation_func=check_no_negative_volume,
            error_message="Volume cannot contain negative values"
        )
    
    @staticmethod
    def create_no_zero_volume_rule() -> ValidationRule:
        """Create rule to validate no zero volume values.
        
        Returns:
            ValidationRule: Rule that checks for positive volume values
        """
        def check_no_zero_volume(df: pd.DataFrame) -> bool:
            if 'volume' not in df.columns:
                return True  # Skip if volume column missing
            
            try:
                # Convert to numeric and check for zero values
                numeric_volume = pd.to_numeric(df['volume'], errors='coerce')
                
                # Check for zero values
                if (numeric_volume == 0).any():
                    logger.debug("Zero values detected in volume column")
                    return False
                
                # Check for NaN values that might indicate conversion issues
                if numeric_volume.isna().any() and not df['volume'].isna().any():
                    logger.debug("Non-numeric values detected in volume column during validation")
                    return False
                    
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating volume column: {e}")
                return False
        
        return ValidationRule(
            name="no_zero_volume",
            validation_func=check_no_zero_volume,
            error_message="Volume cannot contain zero values"
        )
    
    @staticmethod
    def create_price_bounds_rule(min_price: float = 0.01, max_price: float = 10000.0) -> ValidationRule:
        """Create rule to validate price bounds.
        
        Args:
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            
        Returns:
            ValidationRule: Rule that validates price bounds
        """
        def check_price_bounds(df: pd.DataFrame) -> bool:
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                if col not in df.columns:
                    continue  # Skip missing columns
                
                try:
                    # Convert to numeric and check bounds
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for values outside bounds
                    if (numeric_col < min_price).any() or (numeric_col > max_price).any():
                        logger.debug(f"Values outside bounds [{min_price}, {max_price}] detected in column '{col}'")
                        return False
                    
                    # Check for NaN values that might indicate conversion issues
                    if numeric_col.isna().any() and not df[col].isna().any():
                        logger.debug(f"Non-numeric values detected in column '{col}' during validation")
                        return False
                        
                except (TypeError, ValueError) as e:
                    logger.debug(f"Error validating price bounds for column '{col}': {e}")
                    return False
            
            return True
        
        return ValidationRule(
            name="price_bounds",
            validation_func=check_price_bounds,
            error_message=f"Price values must be between {min_price} and {max_price}"
        )
    
    @staticmethod
    def create_high_price_bounds_rule() -> ValidationRule:
        """Create rule to validate High >= Open, Close, Low.
        
        Returns:
            ValidationRule: Rule that validates high price bounds consistency
        """
        def check_high_bounds(df: pd.DataFrame) -> bool:
            required_cols = ['open', 'high', 'low', 'close']
            
            # Check if all required columns exist
            if not all(col in df.columns for col in required_cols):
                return True  # Skip if columns missing (handled by presence rule)
            
            try:
                # Convert to numeric to handle mixed types
                high = pd.to_numeric(df['high'], errors='coerce')
                open_val = pd.to_numeric(df['open'], errors='coerce')
                close_val = pd.to_numeric(df['close'], errors='coerce')
                low = pd.to_numeric(df['low'], errors='coerce')
                
                # Check for any conversion failures
                if any(col.isna().any() for col in [high, open_val, close_val, low]):
                    # Only fail if original data wasn't NaN
                    original_na_mask = df[required_cols].isna()
                    converted_na_mask = pd.DataFrame({
                        'high': high.isna(), 'open': open_val.isna(),
                        'close': close_val.isna(), 'low': low.isna()
                    })
                    
                    if not (original_na_mask == converted_na_mask).all().all():
                        logger.debug("Data conversion issues detected in OHLC columns")
                        return False
                
                # High should be >= open, close, and low
                high_ge_open = (high >= open_val) | (high.isna() | open_val.isna())
                high_ge_close = (high >= close_val) | (high.isna() | close_val.isna())
                high_ge_low = (high >= low) | (high.isna() | low.isna())
                
                violations = ~(high_ge_open & high_ge_close & high_ge_low)
                
                if violations.any():
                    logger.debug(f"High price bound violations at rows: {violations[violations].index.tolist()}")
                    return False
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating high price bounds: {e}")
                return False
        
        return ValidationRule(
            name="high_price_bounds",
            validation_func=check_high_bounds,
            error_message="High price must be greater than or equal to open, close, and low prices"
        )
    
    @staticmethod
    def create_low_price_bounds_rule() -> ValidationRule:
        """Create rule to validate Low <= Open, Close, High.
        
        Returns:
            ValidationRule: Rule that validates low price bounds consistency
        """
        def check_low_bounds(df: pd.DataFrame) -> bool:
            required_cols = ['open', 'high', 'low', 'close']
            
            # Check if all required columns exist
            if not all(col in df.columns for col in required_cols):
                return True  # Skip if columns missing (handled by presence rule)
            
            try:
                # Convert to numeric to handle mixed types
                high = pd.to_numeric(df['high'], errors='coerce')
                open_val = pd.to_numeric(df['open'], errors='coerce')
                close_val = pd.to_numeric(df['close'], errors='coerce')
                low = pd.to_numeric(df['low'], errors='coerce')
                
                # Check for any conversion failures
                if any(col.isna().any() for col in [high, open_val, close_val, low]):
                    # Only fail if original data wasn't NaN
                    original_na_mask = df[required_cols].isna()
                    converted_na_mask = pd.DataFrame({
                        'high': high.isna(), 'open': open_val.isna(),
                        'close': close_val.isna(), 'low': low.isna()
                    })
                    
                    if not (original_na_mask == converted_na_mask).all().all():
                        logger.debug("Data conversion issues detected in OHLC columns")
                        return False
                
                # Low should be <= open, close, and high
                low_le_open = (low <= open_val) | (low.isna() | open_val.isna())
                low_le_close = (low <= close_val) | (low.isna() | close_val.isna())
                low_le_high = (low <= high) | (low.isna() | high.isna())
                
                violations = ~(low_le_open & low_le_close & low_le_high)
                
                if violations.any():
                    logger.debug(f"Low price bound violations at rows: {violations[violations].index.tolist()}")
                    return False
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating low price bounds: {e}")
                return False
        
        return ValidationRule(
            name="low_price_bounds",
            validation_func=check_low_bounds,
            error_message="Low price must be less than or equal to open, close, and high prices"
        )
    
    @staticmethod
    def create_high_low_consistency_rule() -> ValidationRule:
        """Create rule to validate High >= Low consistency.
        
        Returns:
            ValidationRule: Rule that validates high >= low consistency
        """
        def check_high_low_consistency(df: pd.DataFrame) -> bool:
            if not all(col in df.columns for col in ['high', 'low']):
                return True  # Skip if columns missing
            
            try:
                high = pd.to_numeric(df['high'], errors='coerce')
                low = pd.to_numeric(df['low'], errors='coerce')
                
                # Check for conversion issues
                if (high.isna().any() and not df['high'].isna().any()) or \
                   (low.isna().any() and not df['low'].isna().any()):
                    logger.debug("Data conversion issues detected in high/low columns")
                    return False
                
                # High should be >= low (allowing for NaN values)
                valid_comparison = (high >= low) | (high.isna() | low.isna())
                violations = ~valid_comparison
                
                if violations.any():
                    logger.debug(f"High < Low violations at rows: {violations[violations].index.tolist()}")
                    return False
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating high/low consistency: {e}")
                return False
        
        return ValidationRule(
            name="high_low_consistency",
            validation_func=check_high_low_consistency,
            error_message="High price must be greater than or equal to low price"
        )
    
    @staticmethod
    def create_timestamp_uniqueness_rule() -> ValidationRule:
        """Create rule to validate timestamp uniqueness.
        
        Returns:
            ValidationRule: Rule that checks for duplicate timestamps
        """
        def check_timestamp_uniqueness(df: pd.DataFrame) -> bool:
            if 'timestamp' not in df.columns:
                return True  # Skip if no timestamp column
            
            # Check for duplicate timestamps
            duplicates = df['timestamp'].duplicated()
            
            if duplicates.any():
                duplicate_indices = duplicates[duplicates].index.tolist()
                logger.debug(f"Duplicate timestamps found at rows: {duplicate_indices}")
                return False
            
            return True
        
        return ValidationRule(
            name="timestamp_uniqueness",
            validation_func=check_timestamp_uniqueness,
            error_message="Duplicate timestamps detected in data"
        )
    
    @staticmethod
    def create_timestamp_monotonicity_rule() -> ValidationRule:
        """Create rule to validate timestamp monotonicity.
        
        Returns:
            ValidationRule: Rule that checks for monotonic increasing timestamps
        """
        def check_timestamp_monotonicity(df: pd.DataFrame) -> bool:
            if 'timestamp' not in df.columns:
                return True  # Skip if no timestamp column
            
            # Check if timestamps are monotonic increasing
            is_monotonic = df['timestamp'].is_monotonic_increasing
            
            if not is_monotonic:
                logger.debug("Timestamps are not in monotonic increasing order")
                
                # Log specific violations for debugging
                timestamps = pd.to_datetime(df['timestamp'])
                diffs = timestamps.diff()
                violations = diffs[diffs < pd.Timedelta(0)]
                if not violations.empty:
                    logger.debug(f"Timestamp violations at rows: {violations.index.tolist()}")
            
            return is_monotonic
        
        return ValidationRule(
            name="timestamp_monotonicity",
            validation_func=check_timestamp_monotonicity,
            error_message="Timestamps must be in monotonic increasing order"
        )
    
    @staticmethod
    def create_timestamp_gap_detection_rule(max_gap_minutes: int = 5) -> ValidationRule:
        """Create rule to detect large gaps in timestamps.
        
        Args:
            max_gap_minutes: Maximum allowed gap between consecutive timestamps in minutes
            
        Returns:
            ValidationRule: Rule that checks for timestamp gaps exceeding threshold
        """
        def check_timestamp_gaps(df: pd.DataFrame) -> bool:
            if 'timestamp' not in df.columns or len(df) < 2:
                return True  # Skip if no timestamp column or insufficient data
            
            try:
                # Convert timestamps to datetime if they aren't already
                timestamps = pd.to_datetime(df['timestamp'])
                
                # Calculate time differences between consecutive timestamps
                time_diffs = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
                
                # Check if any gap exceeds the maximum allowed
                large_gaps = time_diffs > max_gap_minutes
                
                if large_gaps.any():
                    gap_indices = large_gaps[large_gaps].index.tolist()
                    gap_values = time_diffs[large_gaps].tolist()
                    logger.debug(f"Large timestamp gaps at rows {gap_indices}: {gap_values} minutes")
                    return False
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating timestamp gaps: {e}")
                return False
        
        return ValidationRule(
            name="timestamp_gap_detection",
            validation_func=check_timestamp_gaps,
            error_message=f"Timestamp gaps exceeding {max_gap_minutes} minutes detected"
        )
    
    @staticmethod
    def create_trading_hours_rule(start_time: time, end_time: time) -> ValidationRule:
        """Create rule to validate data falls within trading hours.
        
        Args:
            start_time: Trading day start time (e.g., time(9, 30))
            end_time: Trading day end time (e.g., time(16, 0))
            
        Returns:
            ValidationRule: Rule that validates timestamps fall within trading hours
        """
        def check_trading_hours(df: pd.DataFrame) -> bool:
            if 'timestamp' not in df.columns:
                return True  # Skip if no timestamp column
            
            try:
                # Convert timestamps to datetime if they aren't already
                timestamps = pd.to_datetime(df['timestamp'])
                
                # Extract time component
                times = timestamps.dt.time
                
                # Check if all times are within trading hours
                within_hours = (times >= start_time) & (times <= end_time)
                
                if not within_hours.all():
                    violations = ~within_hours
                    violation_indices = violations[violations].index.tolist()
                    violation_times = times[violations].tolist()
                    logger.debug(f"Trading hours violations at rows {violation_indices}: {violation_times}")
                    return False
                
                return True
                
            except (TypeError, ValueError) as e:
                logger.debug(f"Error validating trading hours: {e}")
                return False
        
        return ValidationRule(
            name="trading_hours",
            validation_func=check_trading_hours,
            error_message=f"Data outside trading hours detected. Trading hours: {start_time} - {end_time}"
        )
    
    @staticmethod
    def create_symbol_whitelist_rule(allowed_symbols: Set[str]) -> ValidationRule:
        """Create rule to validate symbols against whitelist.
        
        Args:
            allowed_symbols: Set of allowed symbol strings
            
        Returns:
            ValidationRule: Rule that validates symbols are in whitelist
        """
        def check_symbol_whitelist(df: pd.DataFrame) -> bool:
            if 'symbol' not in df.columns:
                return True  # Skip if no symbol column
            
            # Check if all symbols are in whitelist
            invalid_symbols = ~df['symbol'].isin(allowed_symbols)
            
            if invalid_symbols.any():
                invalid_symbol_list = df.loc[invalid_symbols, 'symbol'].unique().tolist()
                logger.debug(f"Invalid symbols detected: {invalid_symbol_list}")
                return False
            
            return True
        
        return ValidationRule(
            name="symbol_whitelist",
            validation_func=check_symbol_whitelist,
            error_message=f"Symbols not in whitelist detected. Allowed symbols: {allowed_symbols}"
        )


def build_comprehensive_market_validator(
    symbol_whitelist: Optional[Set[str]] = None,
    trading_hours_start: Optional[time] = None,
    trading_hours_end: Optional[time] = None,
    max_timestamp_gap_minutes: int = 5,
    strict_type_validation: bool = True
) -> DataValidator:
    """Build a comprehensive market data validator with all validation rules.
    
    This function creates a DataValidator with all market data validation rules
    including OHLCV validation, price consistency, timestamp validation, and
    trading hours enforcement.
    
    Args:
        symbol_whitelist: Optional set of allowed symbols
        trading_hours_start: Optional start time for trading hours (e.g., time(9, 30))
        trading_hours_end: Optional end time for trading hours (e.g., time(16, 0))
        max_timestamp_gap_minutes: Maximum allowed gap between timestamps in minutes
        strict_type_validation: If True, requires exact float types for OHLCV columns
        
    Returns:
        DataValidator: Configured validator with comprehensive market data rules
    """
    validator = DataValidator()
    rules = MarketDataValidationRules()
    
    # === CORE OHLCV VALIDATION RULES ===
    
    # Column presence validation
    validator.add_validation_rule(rules.create_ohlcv_column_presence_rule())
    
    # Data type validation
    validator.add_validation_rule(rules.create_ohlcv_data_types_rule(strict_mode=strict_type_validation))
    
    # === PRICE VALIDATION RULES ===
    
    # No negative prices
    validator.add_validation_rule(rules.create_no_negative_prices_rule())
    
    # No negative volume
    validator.add_validation_rule(rules.create_no_negative_volume_rule())
    
    # Price bounds consistency
    validator.add_validation_rule(rules.create_high_price_bounds_rule())
    validator.add_validation_rule(rules.create_low_price_bounds_rule())
    validator.add_validation_rule(rules.create_high_low_consistency_rule())
    
    # === TIMESTAMP VALIDATION RULES ===
    
    # Timestamp uniqueness and monotonicity
    validator.add_validation_rule(rules.create_timestamp_uniqueness_rule())
    validator.add_validation_rule(rules.create_timestamp_monotonicity_rule())
    
    # Timestamp gap detection
    validator.add_validation_rule(rules.create_timestamp_gap_detection_rule(max_gap_minutes=max_timestamp_gap_minutes))
    
    # === OPTIONAL VALIDATION RULES ===
    
    # Symbol whitelist validation (if provided)
    if symbol_whitelist is not None:
        validator.add_validation_rule(rules.create_symbol_whitelist_rule(allowed_symbols=symbol_whitelist))
    
    # Trading hours validation (if both start and end times provided)
    if trading_hours_start is not None and trading_hours_end is not None:
        validator.add_validation_rule(rules.create_trading_hours_rule(start_time=trading_hours_start, end_time=trading_hours_end))
    
    logger.info(f"Created comprehensive market validator with {len(validator.validation_rules)} rules")
    
    return validator


def build_basic_market_validator() -> DataValidator:
    """Build a basic market data validator with essential rules only.
    
    Returns:
        DataValidator: Configured validator with basic market data rules
    """
    validator = DataValidator()
    rules = MarketDataValidationRules()
    
    # Add only essential validation rules
    validator.add_validation_rule(rules.create_ohlcv_column_presence_rule())
    validator.add_validation_rule(rules.create_ohlcv_data_types_rule(strict_mode=False))
    validator.add_validation_rule(rules.create_no_negative_prices_rule())
    validator.add_validation_rule(rules.create_no_negative_volume_rule())
    validator.add_validation_rule(rules.create_high_low_consistency_rule())
    
    logger.info(f"Created basic market validator with {len(validator.validation_rules)} rules")
    
    return validator


def build_timestamp_focused_validator(max_gap_minutes: int = 5) -> DataValidator:
    """Build a validator focused on timestamp validation rules.
    
    Args:
        max_gap_minutes: Maximum allowed gap between timestamps in minutes
        
    Returns:
        DataValidator: Configured validator focused on timestamp validation
    """
    validator = DataValidator()
    rules = MarketDataValidationRules()
    
    # Add timestamp-focused validation rules
    validator.add_validation_rule(rules.create_timestamp_uniqueness_rule())
    validator.add_validation_rule(rules.create_timestamp_monotonicity_rule())
    validator.add_validation_rule(rules.create_timestamp_gap_detection_rule(max_gap_minutes=max_gap_minutes))
    
    logger.info(f"Created timestamp-focused validator with {len(validator.validation_rules)} rules")
    
    return validator


def register_all_market_validation_rules(validator: DataValidator, 
                                        symbol_whitelist: Optional[Set[str]] = None,
                                        trading_hours_start: Optional[time] = None,
                                        trading_hours_end: Optional[time] = None,
                                        max_timestamp_gap_minutes: int = 5,
                                        strict_type_validation: bool = True) -> None:
    """Register all market validation rules to an existing DataValidator.
    
    Args:
        validator: Existing DataValidator instance to add rules to
        symbol_whitelist: Optional set of allowed symbols
        trading_hours_start: Optional start time for trading hours
        trading_hours_end: Optional end time for trading hours
        max_timestamp_gap_minutes: Maximum allowed gap between timestamps in minutes
        strict_type_validation: If True, requires exact float types for OHLCV columns
    """
    rules = MarketDataValidationRules()
    
    # Register all core rules
    rule_methods = [
        rules.create_ohlcv_column_presence_rule,
        lambda: rules.create_ohlcv_data_types_rule(strict_mode=strict_type_validation),
        rules.create_no_negative_prices_rule,
        rules.create_no_negative_volume_rule,
        rules.create_high_price_bounds_rule,
        rules.create_low_price_bounds_rule,
        rules.create_high_low_consistency_rule,
        rules.create_timestamp_uniqueness_rule,
        rules.create_timestamp_monotonicity_rule,
        lambda: rules.create_timestamp_gap_detection_rule(max_gap_minutes=max_timestamp_gap_minutes),
    ]
    
    # Register core rules
    for rule_method in rule_methods:
        validator.add_validation_rule(rule_method())
    
    # Register optional rules
    if symbol_whitelist is not None:
        validator.add_validation_rule(rules.create_symbol_whitelist_rule(allowed_symbols=symbol_whitelist))
    
    if trading_hours_start is not None and trading_hours_end is not None:
        validator.add_validation_rule(rules.create_trading_hours_rule(start_time=trading_hours_start, end_time=trading_hours_end))
    
    logger.info(f"Registered {len(validator.validation_rules)} total market validation rules")


# Default trading hours for major markets
DEFAULT_TRADING_HOURS_START = time(9, 30)   # 9:30 AM
DEFAULT_TRADING_HOURS_END = time(16, 0)     # 4:00 PM

# Common symbol whitelists for major exchanges
NYSE_NASDAQ_SYMBOLS = {
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
    'INTC', 'CSCO', 'PEP', 'AVGO', 'TXN', 'QCOM', 'COST', 'TMUS', 'HON', 'AMAT'
}

# Example usage and factory functions for common configurations
def build_us_market_validator(include_afterhours: bool = False, 
                             symbol_whitelist: Optional[Set[str]] = None) -> DataValidator:
    """Build validator configured for US market standards.
    
    Args:
        include_afterhours: If False, restricts to regular trading hours (9:30-16:00)
        symbol_whitelist: Optional symbol whitelist (defaults to major NYSE/NASDAQ symbols)
        
    Returns:
        DataValidator: US market configured validator
    """
    if symbol_whitelist is None:
        symbol_whitelist = NYSE_NASDAQ_SYMBOLS
    
    if include_afterhours:
        # No trading hours restriction for after-hours trading
        return build_comprehensive_market_validator(
            symbol_whitelist=symbol_whitelist,
            max_timestamp_gap_minutes=5
        )
    else:
        # Regular trading hours only
        return build_comprehensive_market_validator(
            symbol_whitelist=symbol_whitelist,
            trading_hours_start=DEFAULT_TRADING_HOURS_START,
            trading_hours_end=DEFAULT_TRADING_HOURS_END,
            max_timestamp_gap_minutes=5
        )
