"""Data validation module for the Friday AI Trading System.

This module provides classes for validating data, including checking for missing values,
data types, and other validation rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Set
from datetime import time, datetime

from src.infrastructure.logging import get_logger
from src.data.processing.data_processor import DataProcessor, ProcessingStep, DataValidationError

# Create logger
logger = get_logger(__name__)


class ValidationRule:
    """Class representing a validation rule.
    
    This class provides a way to define validation rules for data.
    """
    
    def __init__(self, name: str, validation_func: Callable, error_message: str):
        """Initialize a validation rule.
        
        Args:
            name: The name of the validation rule.
            validation_func: The function to use for validation.
            error_message: The error message to display if validation fails.
        """
        self.name = name
        self.validation_func = validation_func
        self.error_message = error_message
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data using the validation function.
        
        Args:
            data: The data to validate.
            
        Returns:
            True if validation passes, False otherwise.
        """
        return self.validation_func(data)


class DataValidator(DataProcessor):
    """Class for validating data.
    
    This class provides methods for validating data, including checking for missing values,
    data types, and other validation rules.
    """
    
    def __init__(self, config=None):
        """Initialize a data validator.
        
        Args:
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)
        
        # Register default processing steps
        self.add_processing_step(ProcessingStep.VALIDATION, self.validate)
        
        # Store validation rules
        self.validation_rules = {}
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule.
        
        Args:
            rule: The validation rule to add.
        """
        self.validation_rules[rule.name] = rule
    
    def validate(self, data: pd.DataFrame, rules: Optional[List[str]] = None, warn_only: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate data using the specified rules.
        
        Args:
            data: The data to validate.
            rules: List of rule names to use for validation. If None, all rules are used.
            warn_only: If True, log validation failures as warnings but don't fail validation.
            
        Returns:
            A tuple containing:
            - bool: Whether validation passed
            - List[str]: Error messages
            - Dict[str, Any]: Detailed validation metrics
        """
        from datetime import datetime
        import time
        
        # If rules not specified, use all rules
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        # Initialize validation metrics
        start_time = time.time()
        validation_metrics = {
            "start_time": datetime.now().isoformat(),
            "data_shape": data.shape,
            "data_size_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "rules_tested": len(rules),
            "rules_passed": 0,
            "rules_failed": 0,
            "rule_results": {},
            "error_messages": [],
            "warnings": [],
            "total_duration_seconds": 0,
            "warn_only_mode": warn_only
        }
        
        # Validate data using each rule
        validation_passed = True
        error_messages = []
        
        for rule_name in rules:
            if rule_name in self.validation_rules:
                rule = self.validation_rules[rule_name]
                rule_start_time = time.time()
                
                try:
                    rule_passed = rule.validate(data)
                    rule_duration = time.time() - rule_start_time
                    
                    validation_metrics["rule_results"][rule_name] = {
                        "passed": rule_passed,
                        "duration_seconds": rule_duration,
                        "error_message": rule.error_message if not rule_passed else None
                    }
                    
                    if rule_passed:
                        validation_metrics["rules_passed"] += 1
                        logger.debug(f"Validation rule '{rule_name}' passed in {rule_duration:.4f}s")
                    else:
                        validation_metrics["rules_failed"] += 1
                        if warn_only:
                            validation_metrics["warnings"].append(rule.error_message)
                            logger.warning(f"Validation rule '{rule_name}' failed (warn-only mode): {rule.error_message}")
                        else:
                            validation_passed = False
                            error_messages.append(rule.error_message)
                            validation_metrics["error_messages"].append(rule.error_message)
                            logger.error(f"Validation rule '{rule_name}' failed: {rule.error_message}")
                            
                except Exception as e:
                    rule_duration = time.time() - rule_start_time
                    error_msg = f"Error executing validation rule '{rule_name}': {str(e)}"
                    
                    validation_metrics["rules_failed"] += 1
                    validation_metrics["rule_results"][rule_name] = {
                        "passed": False,
                        "duration_seconds": rule_duration,
                        "error_message": error_msg,
                        "exception": str(e)
                    }
                    
                    if warn_only:
                        validation_metrics["warnings"].append(error_msg)
                        logger.warning(f"Validation rule '{rule_name}' error (warn-only mode): {error_msg}")
                    else:
                        validation_passed = False
                        error_messages.append(error_msg)
                        validation_metrics["error_messages"].append(error_msg)
                        logger.error(error_msg)
        
        # Finalize metrics
        validation_metrics["total_duration_seconds"] = time.time() - start_time
        validation_metrics["end_time"] = datetime.now().isoformat()
        validation_metrics["validation_passed"] = validation_passed
        validation_metrics["success_rate"] = validation_metrics["rules_passed"] / max(1, len(rules))
        
        # In warn_only mode, always return True for validation_passed
        if warn_only:
            validation_passed = True
            
        logger.info(
            f"Validation completed: {validation_metrics['rules_passed']}/{len(rules)} rules passed "
            f"(success rate: {validation_metrics['success_rate']:.2%}) in {validation_metrics['total_duration_seconds']:.4f}s"
        )
        
        return validation_passed, error_messages, validation_metrics
    
    def check_missing_values(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Check for missing values in data.
        
        Args:
            data: The data to check.
            columns: List of columns to check. If None, all columns are checked.
            
        Returns:
            A tuple containing a boolean indicating whether validation passed and a list of error messages.
        """
        # If columns not specified, use all columns
        if columns is None:
            columns = data.columns.tolist()
        
        # Check for missing values in each column
        validation_passed = True
        error_messages = []
        
        for col in columns:
            if col in data.columns and data[col].isnull().any():
                validation_passed = False
                error_messages.append(f"Column '{col}' contains missing values.")
        
        return validation_passed, error_messages
    
    def check_data_types(self, data: pd.DataFrame, type_map: Dict[str, type]) -> Tuple[bool, List[str]]:
        """Check data types in data.
        
        Args:
            data: The data to check.
            type_map: A dictionary mapping column names to expected types.
            
        Returns:
            A tuple containing a boolean indicating whether validation passed and a list of error messages.
        """
        # Check data types for each column in the type map
        validation_passed = True
        error_messages = []
        
        for col, expected_type in type_map.items():
            if col in data.columns:
                # Check if column has the expected type
                if expected_type == float:
                    if not pd.api.types.is_float_dtype(data[col]):
                        validation_passed = False
                        error_messages.append(f"Column '{col}' should be of type float.")
                elif expected_type == int:
                    if not pd.api.types.is_integer_dtype(data[col]):
                        validation_passed = False
                        error_messages.append(f"Column '{col}' should be of type int.")
                elif expected_type == str:
                    if not pd.api.types.is_string_dtype(data[col]):
                        validation_passed = False
                        error_messages.append(f"Column '{col}' should be of type str.")
                elif expected_type == bool:
                    if not pd.api.types.is_bool_dtype(data[col]):
                        validation_passed = False
                        error_messages.append(f"Column '{col}' should be of type bool.")
                elif expected_type == pd.Timestamp:
                    if not pd.api.types.is_datetime64_dtype(data[col]):
                        validation_passed = False
                        error_messages.append(f"Column '{col}' should be of type datetime.")
        
        return validation_passed, error_messages


def build_default_market_validator(symbol_whitelist: Optional[Set[str]] = None,
                                   trading_hours_start: Optional[time] = None,
                                   trading_hours_end: Optional[time] = None,
                                   max_timestamp_gap_minutes: int = 5) -> DataValidator:
    """Build and return a default market validator with comprehensive OHLCV validation rules.
    
    DEPRECATED: Use build_comprehensive_market_validator from market_validation_rules module instead.
    This function is maintained for backward compatibility.
    
    Args:
        symbol_whitelist: Optional set of allowed symbols. If provided, only these symbols are allowed.
        trading_hours_start: Optional start time for trading hours validation (e.g., time(9, 30)).
        trading_hours_end: Optional end time for trading hours validation (e.g., time(16, 0)).
        max_timestamp_gap_minutes: Maximum allowed gap between consecutive timestamps in minutes.
    
    Returns:
        DataValidator: Configured validator with comprehensive market data rules.
    """
    # Import here to avoid circular imports
    try:
        from src.data.processing.market_validation_rules import build_comprehensive_market_validator
        logger.info("Using comprehensive market validator from market_validation_rules module")
        return build_comprehensive_market_validator(
            symbol_whitelist=symbol_whitelist,
            trading_hours_start=trading_hours_start,
            trading_hours_end=trading_hours_end,
            max_timestamp_gap_minutes=max_timestamp_gap_minutes,
            strict_type_validation=False  # Use non-strict mode for backward compatibility
        )
    except ImportError:
        logger.warning("Could not import market_validation_rules, falling back to legacy implementation")
        # Fall back to legacy implementation for backward compatibility
        return _build_legacy_market_validator(
            symbol_whitelist=symbol_whitelist,
            trading_hours_start=trading_hours_start,
            trading_hours_end=trading_hours_end,
            max_timestamp_gap_minutes=max_timestamp_gap_minutes
        )


def _build_legacy_market_validator(symbol_whitelist: Optional[Set[str]] = None,
                                  trading_hours_start: Optional[time] = None,
                                  trading_hours_end: Optional[time] = None,
                                  max_timestamp_gap_minutes: int = 5) -> DataValidator:
    """Legacy implementation of market validator for backward compatibility.
    
    Args:
        symbol_whitelist: Optional set of allowed symbols. If provided, only these symbols are allowed.
        trading_hours_start: Optional start time for trading hours validation (e.g., time(9, 30)).
        trading_hours_end: Optional end time for trading hours validation (e.g., time(16, 0)).
        max_timestamp_gap_minutes: Maximum allowed gap between consecutive timestamps in minutes.
    
    Returns:
        DataValidator: Configured validator with comprehensive market data rules.
    """
    validator = DataValidator()
    
    # OHLCV Columns Check
    validator.add_validation_rule(ValidationRule(
        "ohlcv_columns",
        lambda df: set(["open","high","low","close","volume"]).issubset(df.columns),
        "Missing OHLCV columns"
    ))
    
    # === TIMESTAMP VALIDATION RULES ===
    
    # Timestamp Monotonicity Check
    def check_timestamp_monotonicity(df: pd.DataFrame) -> bool:
        if 'timestamp' not in df.columns:
            return True  # Skip if no timestamp column
        return df['timestamp'].is_monotonic_increasing
    
    validator.add_validation_rule(ValidationRule(
        "timestamp_monotonicity",
        check_timestamp_monotonicity,
        "Timestamps are not in monotonic increasing order"
    ))
    
    # Duplicate Timestamp Check
    def check_duplicate_timestamps(df: pd.DataFrame) -> bool:
        if 'timestamp' not in df.columns:
            return True  # Skip if no timestamp column
        return not df['timestamp'].duplicated().any()
    
    validator.add_validation_rule(ValidationRule(
        "no_duplicate_timestamps",
        check_duplicate_timestamps,
        "Duplicate timestamps found in data"
    ))
    
    # Timestamp Gap Detection
    def check_timestamp_gaps(df: pd.DataFrame) -> bool:
        if 'timestamp' not in df.columns or len(df) < 2:
            return True  # Skip if no timestamp column or insufficient data
        
        # Convert timestamps to datetime if they aren't already
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Calculate time differences between consecutive timestamps
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
        
        # Check if any gap exceeds the maximum allowed
        return not (time_diffs > max_timestamp_gap_minutes).any()
    
    validator.add_validation_rule(ValidationRule(
        "timestamp_gap_detection",
        check_timestamp_gaps,
        f"Timestamp gaps exceeding {max_timestamp_gap_minutes} minutes detected"
    ))
    
    # === PRICE VALIDATION RULES ===
    
    # No Negative Prices Check
    def check_no_negative_prices(df: pd.DataFrame) -> bool:
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                try:
                    # Try to convert to numeric and check for negative values
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if (numeric_col < 0).any():
                        return False
                except (TypeError, ValueError):
                    # If conversion fails, consider it invalid
                    return False
        return True
    
    validator.add_validation_rule(ValidationRule(
        "no_negative_prices",
        check_no_negative_prices,
        "Negative prices detected in OHLC data"
    ))
    
    # High >= Open/Close/Low Check
    def check_high_bounds(df: pd.DataFrame) -> bool:
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return True  # Skip if columns missing
        
        try:
            # Convert to numeric to handle mixed types
            high = pd.to_numeric(df['high'], errors='coerce')
            open_val = pd.to_numeric(df['open'], errors='coerce')
            close_val = pd.to_numeric(df['close'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            
            # High should be >= open, close, and low
            return (high >= open_val).all() and \
                   (high >= close_val).all() and \
                   (high >= low).all()
        except (TypeError, ValueError):
            return False
    
    validator.add_validation_rule(ValidationRule(
        "high_price_bounds",
        check_high_bounds,
        "High price is less than open, close, or low price"
    ))
    
    # Low <= Open/Close/High Check
    def check_low_bounds(df: pd.DataFrame) -> bool:
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return True  # Skip if columns missing
        
        try:
            # Convert to numeric to handle mixed types
            high = pd.to_numeric(df['high'], errors='coerce')
            open_val = pd.to_numeric(df['open'], errors='coerce')
            close_val = pd.to_numeric(df['close'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            
            # Low should be <= open, close, and high
            return (low <= open_val).all() and \
                   (low <= close_val).all() and \
                   (low <= high).all()
        except (TypeError, ValueError):
            return False
    
    validator.add_validation_rule(ValidationRule(
        "low_price_bounds",
        check_low_bounds,
        "Low price is greater than open, close, or high price"
    ))
    
    # High >= Low Check (original rule)
    def check_high_low_consistency(df: pd.DataFrame) -> bool:
        if "high" not in df.columns or "low" not in df.columns:
            return True  # Skip if columns missing
        try:
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            return (high >= low).all()
        except (TypeError, ValueError):
            return False
    
    validator.add_validation_rule(ValidationRule(
        "high_low_consistency",
        check_high_low_consistency,
        "'high' column has values less than 'low' column"
    ))
    
    # Volume >= 0 Check
    validator.add_validation_rule(ValidationRule(
        "non_negative_volume",
        lambda df: ("volume" in df.columns) and (df["volume"] >= 0).all(),
        "Volume has negative values"
    ))
    
    # === SYMBOL WHITELIST VALIDATION ===
    
    if symbol_whitelist is not None:
        def check_symbol_whitelist(df: pd.DataFrame) -> bool:
            if 'symbol' not in df.columns:
                return True  # Skip if no symbol column
            return df['symbol'].isin(symbol_whitelist).all()
        
        validator.add_validation_rule(ValidationRule(
            "symbol_whitelist",
            check_symbol_whitelist,
            f"Symbols not in whitelist detected. Allowed symbols: {symbol_whitelist}"
        ))
    
    # === TRADING HOURS VALIDATION ===
    
    if trading_hours_start is not None and trading_hours_end is not None:
        def check_trading_hours(df: pd.DataFrame) -> bool:
            if 'timestamp' not in df.columns:
                return True  # Skip if no timestamp column
            
            # Convert timestamps to datetime if they aren't already
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Extract time component
            times = timestamps.dt.time
            
            # Check if all times are within trading hours
            return ((times >= trading_hours_start) & (times <= trading_hours_end)).all()
        
        validator.add_validation_rule(ValidationRule(
            "trading_hours",
            check_trading_hours,
            f"Data outside trading hours detected. Trading hours: {trading_hours_start} - {trading_hours_end}"
        ))
    
    # Type checks (example for OHLCV assuming they are all float)
    validator.add_validation_rule(ValidationRule(
        "ohlcv_types",
        lambda df: all([
            c in df.columns and pd.api.types.is_float_dtype(df[c]) for c in ["open", "high", "low", "close", "volume"]
        ]),
        "OHLCV columns must be of type float"
    ))
    
    return validator
