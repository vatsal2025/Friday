"""Enhanced unit tests for comprehensive market data validation rules.

This test suite covers all validation rules implemented in Step 2:
- Timestamp gaps, negative/zero volumes, price bounds, symbol whitelist
- Trading hours, dtype consistency, and warn-only toggle functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta
from typing import Set

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data.processing.market_validation_rules import (
    build_comprehensive_market_validator,
    build_basic_market_validator,
    build_us_market_validator,
    MarketDataValidationRules,
    NYSE_NASDAQ_SYMBOLS,
    DEFAULT_TRADING_HOURS_START,
    DEFAULT_TRADING_HOURS_END
)
from src.infrastructure.config.config_manager import ConfigurationManager


class TestTimestampGapValidation:
    """Test timestamp gap detection with various gap thresholds."""
    
    def test_valid_timestamp_gaps(self):
        """Test that small timestamp gaps pass validation."""
        validator = build_comprehensive_market_validator(max_timestamp_gap_minutes=5)
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',
                '2024-01-01 09:33:00',  # 3 minute gap - should pass
                '2024-01-01 09:36:00',  # 3 minute gap - should pass
                '2024-01-01 09:40:00'   # 4 minute gap - should pass
            ]),
            'symbol': ['AAPL'] * 4,
            'open': [150.0] * 4,
            'high': [151.0] * 4,
            'low': [149.0] * 4,
            'close': [150.5] * 4,
            'volume': [100000.0] * 4
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["timestamp_gap_detection"])
        assert is_valid, f"Valid timestamp gaps should pass. Errors: {errors}"
    
    def test_invalid_timestamp_gaps(self):
        """Test that large timestamp gaps fail validation."""
        validator = build_comprehensive_market_validator(max_timestamp_gap_minutes=5)
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',
                '2024-01-01 09:37:00',  # 7 minute gap - should fail
                '2024-01-01 09:40:00'
            ]),
            'symbol': ['AAPL'] * 3,
            'open': [150.0] * 3,
            'high': [151.0] * 3,
            'low': [149.0] * 3,
            'close': [150.5] * 3,
            'volume': [100000.0] * 3
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["timestamp_gap_detection"])
        assert not is_valid, "Large timestamp gaps should fail validation"
        assert any("gaps exceeding" in error for error in errors)
    
    def test_custom_gap_tolerance(self):
        """Test custom gap tolerance settings."""
        # Create validator with 10-minute tolerance
        validator = build_comprehensive_market_validator(max_timestamp_gap_minutes=10)
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',
                '2024-01-01 09:38:00',  # 8 minute gap - should pass with 10 min tolerance
                '2024-01-01 09:40:00'
            ]),
            'symbol': ['AAPL'] * 3,
            'open': [150.0] * 3,
            'high': [151.0] * 3,
            'low': [149.0] * 3,
            'close': [150.5] * 3,
            'volume': [100000.0] * 3
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["timestamp_gap_detection"])
        assert is_valid, f"8-minute gap should pass with 10-minute tolerance. Errors: {errors}"


class TestVolumeValidation:
    """Test volume-related validation rules."""
    
    def test_negative_volume_detection(self):
        """Test detection of negative volume values."""
        validator = build_comprehensive_market_validator()
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0] * 2,
            'high': [151.0] * 2,
            'low': [149.0] * 2,
            'close': [150.5] * 2,
            'volume': [100000.0, -50000.0]  # Negative volume
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["no_negative_volume"])
        assert not is_valid, "Negative volume should fail validation"
        assert any("negative values" in error.lower() for error in errors)
    
    def test_zero_volume_handling(self):
        """Test handling of zero volume values (should pass for basic rule)."""
        validator = build_comprehensive_market_validator()
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0] * 2,
            'high': [151.0] * 2,
            'low': [149.0] * 2,
            'close': [150.5] * 2,
            'volume': [100000.0, 0.0]  # Zero volume
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["no_negative_volume"])
        assert is_valid, f"Zero volume should pass negative volume check. Errors: {errors}"
    
    def test_zero_volume_rejection(self):
        """Test optional zero volume rejection rule."""
        rules = MarketDataValidationRules()
        validator = build_comprehensive_market_validator()
        
        # Add the no_zero_volume rule
        validator.add_validation_rule(rules.create_no_zero_volume_rule())
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0] * 2,
            'high': [151.0] * 2,
            'low': [149.0] * 2,
            'close': [150.5] * 2,
            'volume': [100000.0, 0.0]  # Zero volume
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["no_zero_volume"])
        assert not is_valid, "Zero volume should fail when no_zero_volume rule is active"
        assert any("zero values" in error.lower() for error in errors)


class TestPriceBoundsValidation:
    """Test price bounds validation rules."""
    
    def test_valid_price_bounds(self):
        """Test that prices within bounds pass validation."""
        rules = MarketDataValidationRules()
        validator = build_comprehensive_market_validator()
        
        # Add price bounds rule with default bounds (0.01 to 10000.0)
        validator.add_validation_rule(rules.create_price_bounds_rule())
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["price_bounds"])
        assert is_valid, f"Valid price bounds should pass. Errors: {errors}"
    
    def test_invalid_price_bounds_too_low(self):
        """Test detection of prices below minimum bound."""
        rules = MarketDataValidationRules()
        validator = build_comprehensive_market_validator()
        
        # Add price bounds rule with minimum 1.0
        validator.add_validation_rule(rules.create_price_bounds_rule(min_price=1.0, max_price=10000.0))
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, 0.50],  # Below minimum
            'high': [151.0, 0.60],
            'low': [149.0, 0.45],
            'close': [150.5, 0.55],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["price_bounds"])
        assert not is_valid, "Prices below minimum should fail validation"
        assert any("between 1.0 and 10000.0" in error for error in errors)
    
    def test_invalid_price_bounds_too_high(self):
        """Test detection of prices above maximum bound."""
        rules = MarketDataValidationRules()
        validator = build_comprehensive_market_validator()
        
        # Add price bounds rule with maximum 1000.0
        validator.add_validation_rule(rules.create_price_bounds_rule(min_price=0.01, max_price=1000.0))
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, 1500.0],  # Above maximum
            'high': [151.0, 1600.0],
            'low': [149.0, 1400.0],
            'close': [150.5, 1550.0],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["price_bounds"])
        assert not is_valid, "Prices above maximum should fail validation"
        assert any("between 0.01 and 1000.0" in error for error in errors)


class TestSymbolWhitelistValidation:
    """Test symbol whitelist validation."""
    
    def test_valid_symbols(self):
        """Test that whitelisted symbols pass validation."""
        whitelist = {'AAPL', 'MSFT', 'GOOGL'}
        validator = build_comprehensive_market_validator(symbol_whitelist=whitelist)
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL', 'MSFT'],
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["symbol_whitelist"])
        assert is_valid, f"Whitelisted symbols should pass. Errors: {errors}"
    
    def test_invalid_symbols(self):
        """Test that non-whitelisted symbols fail validation."""
        whitelist = {'AAPL', 'MSFT', 'GOOGL'}
        validator = build_comprehensive_market_validator(symbol_whitelist=whitelist)
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL', 'INVALID_SYMBOL'],
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["symbol_whitelist"])
        assert not is_valid, "Non-whitelisted symbols should fail validation"
        assert any("not in whitelist" in error for error in errors)


class TestTradingHoursValidation:
    """Test trading hours validation."""
    
    def test_valid_trading_hours(self):
        """Test that data within trading hours passes validation."""
        start_time = time(9, 30)
        end_time = time(16, 0)
        validator = build_comprehensive_market_validator(
            trading_hours_start=start_time,
            trading_hours_end=end_time
        )
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',  # At start
                '2024-01-01 12:00:00',  # Middle
                '2024-01-01 16:00:00'   # At end
            ]),
            'symbol': ['AAPL'] * 3,
            'open': [150.0] * 3,
            'high': [151.0] * 3,
            'low': [149.0] * 3,
            'close': [150.5] * 3,
            'volume': [100000.0] * 3
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["trading_hours"])
        assert is_valid, f"Data within trading hours should pass. Errors: {errors}"
    
    def test_invalid_trading_hours(self):
        """Test that data outside trading hours fails validation."""
        start_time = time(9, 30)
        end_time = time(16, 0)
        validator = build_comprehensive_market_validator(
            trading_hours_start=start_time,
            trading_hours_end=end_time
        )
        
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 08:30:00',  # Before start
                '2024-01-01 12:00:00',  # Valid
                '2024-01-01 17:00:00'   # After end
            ]),
            'symbol': ['AAPL'] * 3,
            'open': [150.0] * 3,
            'high': [151.0] * 3,
            'low': [149.0] * 3,
            'close': [150.5] * 3,
            'volume': [100000.0] * 3
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["trading_hours"])
        assert not is_valid, "Data outside trading hours should fail validation"
        assert any("outside trading hours" in error for error in errors)


class TestDataTypeConsistency:
    """Test data type validation rules."""
    
    def test_strict_dtype_validation(self):
        """Test strict data type validation."""
        validator = build_comprehensive_market_validator(strict_type_validation=True)
        
        # Create data with float types
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["ohlcv_data_types"])
        assert is_valid, f"Float data types should pass strict validation. Errors: {errors}"
    
    def test_flexible_dtype_validation(self):
        """Test flexible data type validation."""
        validator = build_comprehensive_market_validator(strict_type_validation=False)
        
        # Create data with mixed numeric types
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150, 151],  # Integers
            'high': [151.0, 152.0],  # Floats
            'low': ['149.0', '150.0'],  # String numbers
            'close': [150.5, 151.5],
            'volume': [100000, 85000]
        })
        
        is_valid, errors, metrics = validator.validate(data, rules=["ohlcv_data_types"])
        assert is_valid, f"Mixed numeric types should pass flexible validation. Errors: {errors}"


class TestWarnOnlyMode:
    """Test warn-only toggle functionality for real-time streams."""
    
    def test_warn_only_mode_basic(self):
        """Test that warn-only mode passes validation but logs warnings."""
        validator = build_comprehensive_market_validator()
        
        # Create data with validation issues
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, -151.0],  # Negative price
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, -85000.0]  # Negative volume
        })
        
        # Test with warn_only=False (default)
        is_valid_strict, errors_strict, metrics_strict = validator.validate(data, warn_only=False)
        assert not is_valid_strict, "Validation should fail in strict mode"
        assert len(errors_strict) > 0, "Should have error messages in strict mode"
        
        # Test with warn_only=True
        is_valid_warn, errors_warn, metrics_warn = validator.validate(data, warn_only=True)
        assert is_valid_warn, "Validation should pass in warn-only mode"
        assert len(errors_warn) == 0, "Should have no error messages in warn-only mode"
        assert len(metrics_warn.get('warnings', [])) > 0, "Should have warnings in warn-only mode"
    
    def test_warn_only_mode_real_time_scenario(self):
        """Test warn-only mode for real-time streaming scenario."""
        validator = build_comprehensive_market_validator(
            symbol_whitelist={'AAPL', 'MSFT'},
            trading_hours_start=time(9, 30),
            trading_hours_end=time(16, 0),
            max_timestamp_gap_minutes=5
        )
        
        # Simulate real-time data with minor issues
        real_time_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 08:30:00',  # Outside trading hours
                '2024-01-01 09:31:00'
            ]),
            'symbol': ['AAPL', 'AAPL'],
            'open': [150.0, 151.0],
            'high': [151.0, 152.0],
            'low': [149.0, 150.0],
            'close': [150.5, 151.5],
            'volume': [100000.0, 85000.0]
        })
        
        is_valid, errors, metrics = validator.validate(real_time_data, warn_only=True)
        
        assert is_valid, "Real-time data should pass in warn-only mode"
        assert metrics['warn_only_mode'] is True, "Metrics should indicate warn-only mode"
        assert len(metrics.get('warnings', [])) > 0, "Should have warnings for trading hours violation"


class TestConfigManagerIntegration:
    """Test integration with ConfigManager for rule parameters."""
    
    def test_config_manager_rule_parameters(self):
        """Test that rule parameters can be loaded from ConfigManager."""
        # Get config instance
        config = ConfigurationManager.get_instance()
        
        # Set validation rule parameters
        config.set('validation.max_timestamp_gap_minutes', 10)
        config.set('validation.min_price_bound', 0.50)
        config.set('validation.max_price_bound', 5000.0)
        config.set('validation.strict_type_validation', True)
        config.set('validation.symbol_whitelist', ['AAPL', 'MSFT', 'GOOGL'])
        config.set('validation.trading_hours_start', '09:30')
        config.set('validation.trading_hours_end', '16:00')
        
        # Retrieve parameters from config
        max_gap = config.get('validation.max_timestamp_gap_minutes', 5)
        min_price = config.get('validation.min_price_bound', 0.01)
        max_price = config.get('validation.max_price_bound', 10000.0)
        strict_types = config.get('validation.strict_type_validation', False)
        symbols = config.get('validation.symbol_whitelist', None)
        
        # Convert time strings to time objects
        start_time_str = config.get('validation.trading_hours_start', None)
        end_time_str = config.get('validation.trading_hours_end', None)
        start_time = time(9, 30) if start_time_str == '09:30' else None
        end_time = time(16, 0) if end_time_str == '16:00' else None
        
        # Create validator with config parameters
        validator = build_comprehensive_market_validator(
            symbol_whitelist=set(symbols) if symbols else None,
            trading_hours_start=start_time,
            trading_hours_end=end_time,
            max_timestamp_gap_minutes=max_gap,
            strict_type_validation=strict_types
        )
        
        # Verify configuration was applied
        assert len(validator.validation_rules) > 0
        assert 'symbol_whitelist' in validator.validation_rules
        assert 'trading_hours' in validator.validation_rules
    
    def test_config_validation_schema(self):
        """Test configuration validation schema for market validation."""
        config = ConfigurationManager.get_instance()
        
        # Register validation schema
        validation_schema = {
            'max_timestamp_gap_minutes': {
                'type': int,
                'min': 1,
                'max': 60,
                'required': False
            },
            'min_price_bound': {
                'type': float,
                'min': 0.001,
                'max': 1.0,
                'required': False
            },
            'max_price_bound': {
                'type': float,
                'min': 100.0,
                'max': 100000.0,
                'required': False
            },
            'strict_type_validation': {
                'type': bool,
                'required': False
            }
        }
        
        config.register_validator('validation', validation_schema)
        
        # Clear any existing validation config to avoid conflicts
        config.delete('validation.symbol_whitelist', environment=None)
        config.delete('validation.trading_hours_start', environment=None)
        config.delete('validation.trading_hours_end', environment=None)
        
        # Test valid configuration
        config.set('validation.max_timestamp_gap_minutes', 10)
        config.set('validation.min_price_bound', 0.01)
        config.set('validation.max_price_bound', 1000.0)
        config.set('validation.strict_type_validation', True)
        
        errors = config.validate('validation')
        assert len(errors) == 0, f"Valid configuration should pass validation. Errors: {errors}"
        
        # Test invalid configuration
        config.set('validation.max_timestamp_gap_minutes', 100)  # Above max
        config.set('validation.min_price_bound', 2.0)  # Above max
        
        errors = config.validate('validation')
        assert len(errors) > 0, "Invalid configuration should fail validation"


class TestComprehensiveIntegration:
    """Test complete integration of all validation rules."""
    
    def test_fully_comprehensive_validation(self):
        """Test all validation rules working together."""
        # Create comprehensive validator with all options
        validator = build_comprehensive_market_validator(
            symbol_whitelist={'AAPL', 'MSFT', 'GOOGL', 'TSLA'},
            trading_hours_start=time(9, 30),
            trading_hours_end=time(16, 0),
            max_timestamp_gap_minutes=5,
            strict_type_validation=True
        )
        
        # Add additional rules
        rules = MarketDataValidationRules()
        validator.add_validation_rule(rules.create_price_bounds_rule(min_price=1.0, max_price=1000.0))
        
        # Create valid data that passes all rules
        valid_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',
                '2024-01-01 09:33:00',
                '2024-01-01 09:36:00'
            ]),
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'open': [150.0, 250.0, 350.0],
            'high': [151.0, 251.0, 351.0],
            'low': [149.0, 249.0, 349.0],
            'close': [150.5, 250.5, 350.5],
            'volume': [100000.0, 85000.0, 120000.0]
        })
        
        is_valid, errors, metrics = validator.validate(valid_data)
        assert is_valid, f"Valid comprehensive data should pass all rules. Errors: {errors}"
        assert metrics['success_rate'] == 1.0, "Success rate should be 100%"
    
    def test_multiple_rule_failures(self):
        """Test data that fails multiple validation rules."""
        validator = build_comprehensive_market_validator(
            symbol_whitelist={'AAPL', 'MSFT'},
            trading_hours_start=time(9, 30),
            trading_hours_end=time(16, 0),
            max_timestamp_gap_minutes=5
        )
        
        # Create data with multiple issues
        invalid_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 08:30:00',  # Outside trading hours
                '2024-01-01 09:45:00'   # Large gap (75 minutes)
            ]),
            'symbol': ['INVALID', 'AAPL'],  # Invalid symbol
            'open': [150.0, -151.0],  # Negative price
            'high': [149.0, 152.0],  # High < Open for first row
            'low': [149.5, 150.0],  # Low > High for first row
            'close': [150.5, 151.5],
            'volume': [-100000.0, 85000.0]  # Negative volume
        })
        
        is_valid, errors, metrics = validator.validate(invalid_data)
        
        assert not is_valid, "Data with multiple issues should fail validation"
        assert len(errors) >= 5, f"Should have multiple error messages. Got: {len(errors)} errors"
        assert metrics['success_rate'] < 0.5, "Success rate should be low due to multiple failures"
    
    def test_performance_metrics(self):
        """Test that validation provides comprehensive performance metrics."""
        validator = build_comprehensive_market_validator()
        
        # Create reasonably sized dataset
        n_rows = 1000
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=n_rows, freq='1min'),
            'symbol': ['AAPL'] * n_rows,
            'open': np.random.uniform(140, 160, n_rows),
            'high': np.random.uniform(145, 165, n_rows),
            'low': np.random.uniform(135, 155, n_rows),
            'close': np.random.uniform(140, 160, n_rows),
            'volume': np.random.randint(50000, 200000, n_rows)
        })
        
        # Ensure price consistency
        for i in range(n_rows):
            data.loc[i, 'high'] = max(data.loc[i, ['open', 'high', 'low', 'close']])
            data.loc[i, 'low'] = min(data.loc[i, ['open', 'high', 'low', 'close']])
        
        is_valid, errors, metrics = validator.validate(data)
        
        # Check metrics completeness
        required_metrics = [
            'start_time', 'end_time', 'data_shape', 'data_size_mb',
            'rules_tested', 'rules_passed', 'rules_failed', 'rule_results',
            'total_duration_seconds', 'success_rate', 'validation_passed'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        assert metrics['data_shape'][0] == n_rows, "Data shape should record correct number of rows"
        assert metrics['data_shape'][1] >= 6, "Data shape should have at least 6 columns"
        assert metrics['total_duration_seconds'] > 0, "Duration should be positive"
        assert 0 <= metrics['success_rate'] <= 1, "Success rate should be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
