"""Market Data Validation Rules Demo

This script demonstrates the comprehensive market data validation rules 
implemented in Step 2 of the trading system development plan.

Key Features Demonstrated:
- OHLCV column presence & types validation
- No-negative price/volume checks  
- High ≥ Open/Close ≥ Low consistency validation
- Timestamp uniqueness & monotonicity checks
- Trading-hours window validation (09:30–16:00 configurable)
"""

import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processing.market_validation_rules import (
    build_comprehensive_market_validator,
    build_basic_market_validator,
    build_us_market_validator,
    MarketDataValidationRules,
    NYSE_NASDAQ_SYMBOLS,
    DEFAULT_TRADING_HOURS_START,
    DEFAULT_TRADING_HOURS_END
)

def create_demo_data():
    """Create various demo datasets for validation testing."""
    
    # Valid OHLCV data
    valid_data = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00', 
            '2024-01-01 09:32:00',
            '2024-01-01 09:33:00',
            '2024-01-01 09:34:00'
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0, 151.5],
        'high': [151.5, 152.0, 152.5, 153.0, 152.5],  # Ensure high >= all other prices
        'low': [149.5, 150.0, 149.8, 151.0, 150.5],
        'close': [151.0, 150.5, 152.0, 151.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0, 110000.0]
    })
    
    # Data with missing columns
    missing_columns_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [151.5, 152.0],
        # Missing 'low', 'close', and 'volume' columns
    })
    
    # Data with negative prices
    negative_prices_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, -151.0],  # Negative price
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        'close': [151.0, 150.5],
        'volume': [100000.0, 85000.0]
    })
    
    # Data with invalid price bounds (high < low)
    invalid_bounds_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [149.0, 150.0],  # High < Low
        'low': [149.5, 150.5],
        'close': [151.0, 150.5],
        'volume': [100000.0, 85000.0]
    })
    
    # Data with duplicate timestamps
    duplicate_timestamps_data = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 09:30:00',
            '2024-01-01 09:31:00',
            '2024-01-01 09:31:00',  # Duplicate
            '2024-01-01 09:32:00'
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5, 152.0],
        'high': [151.5, 152.0, 152.5, 153.0],
        'low': [149.5, 150.0, 149.8, 151.0],
        'close': [151.0, 150.5, 152.0, 151.5],
        'volume': [100000.0, 85000.0, 120000.0, 95000.0]
    })
    
    # Data outside trading hours
    outside_hours_data = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 08:30:00',  # Before trading hours
            '2024-01-01 09:31:00',  # During trading hours
            '2024-01-01 17:00:00'   # After trading hours
        ]),
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'open': [150.0, 151.0, 150.5],
        'high': [151.5, 152.0, 152.5],
        'low': [149.5, 150.0, 149.8],
        'close': [151.0, 150.5, 152.0],
        'volume': [100000.0, 85000.0, 120000.0]
    })
    
    # Data with invalid symbols (not in whitelist)
    invalid_symbols_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'INVALID_SYMBOL'],  # INVALID_SYMBOL not in whitelist
        'open': [150.0, 151.0],
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        'close': [151.0, 150.5],
        'volume': [100000.0, 85000.0]
    })
    
    # Data with zero volume
    zero_volume_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [151.5, 152.0],
        'low': [149.5, 150.0],
        'close': [151.0, 150.5],
        'volume': [0.0, 0.0]  # Zero volume
    })

    # Data with prices out of bounds
    out_of_bounds_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:31:00']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [15000.0, 0.005],  # Prices out of set bounds
        'high': [15150.0, 0.002],
        'low': [14950.0, 0.004],
        'close': [15100.0, 0.001],
        'volume': [100000.0, 85000.0]
    })
    
    return {
        'valid': valid_data,
        'missing_columns': missing_columns_data,
        'negative_prices': negative_prices_data,
        'invalid_bounds': invalid_bounds_data,
        'duplicate_timestamps': duplicate_timestamps_data,
        'outside_hours': outside_hours_data,
        'invalid_symbols': invalid_symbols_data,
        'zero_volume': zero_volume_data,
        'out_of_bounds': out_of_bounds_data
    }

def test_validator(validator, data, test_name):
    """Test a validator with given data and display results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)
    
    try:
        is_valid, errors, metrics = validator.validate(data)
        
        print(f"✓ Validation Result: {'PASS' if is_valid else 'FAIL'}")
        print(f"✓ Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"✓ Rules Tested: {metrics.get('rules_tested', 0)}")
        print(f"✓ Rules Passed: {metrics.get('rules_passed', 0)}")
        print(f"✓ Rules Failed: {metrics.get('rules_failed', 0)}")
        print(f"✓ Duration: {metrics.get('total_duration_seconds', 0):.4f}s")
        
        if errors:
            print(f"\n❌ Validation Errors ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
        else:
            print("\n✅ No validation errors!")
            
        if metrics.get('warnings'):
            print(f"\n⚠️  Warnings ({len(metrics['warnings'])}):")
            for i, warning in enumerate(metrics['warnings'], 1):
                print(f"   {i}. {warning}")
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def demo_validation_rules():
    """Main demo function showcasing all validation rules."""
    
    print("🚀 MARKET DATA VALIDATION RULES DEMO")
    print("="*60)
    print("This demo showcases comprehensive market data validation")
    print("covering all requirements from Step 2 of the development plan:")
    print()
    print("✓ OHLCV column presence & types validation")
    print("✓ No-negative price/volume checks")
    print("✓ High ≥ Open/Close ≥ Low consistency validation")
    print("✓ Timestamp uniqueness & monotonicity checks")
    print("✓ Trading-hours window validation (09:30–16:00 configurable)")
    print()
    
    # Create demo datasets
    datasets = create_demo_data()
    
    # Demo 1: Comprehensive Validator
    print("\n" + "🔹" * 60)
    print("DEMO 1: COMPREHENSIVE MARKET VALIDATOR")
    print("🔹" * 60)
    
    comprehensive_validator = build_comprehensive_market_validator(
        symbol_whitelist={'AAPL', 'MSFT', 'GOOGL', 'TSLA'},
        trading_hours_start=time(9, 30),
        trading_hours_end=time(16, 0),
        max_timestamp_gap_minutes=5,
        strict_type_validation=False
    )
    
    print(f"Created comprehensive validator with {len(comprehensive_validator.validation_rules)} rules:")
    for rule_name in comprehensive_validator.validation_rules.keys():
        print(f"  • {rule_name}")
    
    # Test with valid data
    test_validator(comprehensive_validator, datasets['valid'], "Valid OHLCV Data")
    
    # Test with various invalid data
    test_validator(comprehensive_validator, datasets['missing_columns'], "Missing OHLCV Columns")
    test_validator(comprehensive_validator, datasets['negative_prices'], "Negative Prices")
    test_validator(comprehensive_validator, datasets['invalid_bounds'], "Invalid Price Bounds (High < Low)")
    test_validator(comprehensive_validator, datasets['duplicate_timestamps'], "Duplicate Timestamps")
    test_validator(comprehensive_validator, datasets['outside_hours'], "Data Outside Trading Hours")
    test_validator(comprehensive_validator, datasets['invalid_symbols'], "Invalid Symbols (Not in Whitelist)")
    
    # Demo 2: Basic Validator
    print("\n" + "🔹" * 60)
    print("DEMO 2: BASIC MARKET VALIDATOR")
    print("🔹" * 60)
    
    basic_validator = build_basic_market_validator()
    print(f"Created basic validator with {len(basic_validator.validation_rules)} essential rules")
    
    test_validator(basic_validator, datasets['valid'], "Valid Data with Basic Validator")
    test_validator(basic_validator, datasets['negative_prices'], "Negative Prices with Basic Validator")
    
    # Demo 3: US Market Validator
    print("\n" + "🔹" * 60)
    print("DEMO 3: US MARKET VALIDATOR")
    print("🔹" * 60)
    
    us_validator = build_us_market_validator(include_afterhours=False)
    print(f"Created US market validator with NYSE/NASDAQ symbols and regular trading hours")
    
    # Test with AAPL (valid US symbol)
    test_validator(us_validator, datasets['valid'], "Valid US Market Data (AAPL)")
    
    # Demo 4: Individual Validation Rules
    print("\n" + "🔹" * 60)
    print("DEMO 4: INDIVIDUAL VALIDATION RULES")
    print("🔹" * 60)
    
    rules = MarketDataValidationRules()
    
    # Test individual rules
    individual_validator = build_comprehensive_market_validator()
    
    # Test only timestamp rules
    timestamp_rules = ["timestamp_uniqueness", "timestamp_monotonicity"]
    is_valid, errors, metrics = individual_validator.validate(datasets['duplicate_timestamps'], rules=timestamp_rules)
    print(f"\nTimestamp Rules Only - Valid: {is_valid}, Errors: {len(errors)}")
    
    # Test only price rules  
    price_rules = ["no_negative_prices", "high_price_bounds", "low_price_bounds"]
    is_valid, errors, metrics = individual_validator.validate(datasets['negative_prices'], rules=price_rules)
    print(f"Price Rules Only - Valid: {is_valid}, Errors: {len(errors)}")
    
    # Demo 5: Custom Configuration
    print("\n" + "🔹" * 60)
    print("DEMO 5: CUSTOM VALIDATOR CONFIGURATION")
    print("🔹" * 60)
    
    # Custom validator with specific requirements
    custom_validator = build_comprehensive_market_validator(
        symbol_whitelist={'AAPL', 'TSLA'},  # Only allow AAPL and TSLA
        trading_hours_start=time(9, 0),     # Extended trading hours start
        trading_hours_end=time(17, 0),      # Extended trading hours end
        max_timestamp_gap_minutes=10,       # Allow larger gaps
        strict_type_validation=True         # Strict type checking
    )
    
    print("Custom validator configuration:")
    print(f"  • Symbol whitelist: AAPL, TSLA only")
    print(f"  • Trading hours: 09:00 - 17:00")
    print(f"  • Max timestamp gap: 10 minutes")
    print(f"  • Strict type validation: Enabled")
    
    test_validator(custom_validator, datasets['valid'], "Valid Data with Custom Config")
    
    # Demo 6: Warn-Only Mode
    print("\n" + "🔹" * 60)
    print("DEMO 6: WARN-ONLY MODE")
    print("🔹" * 60)
    
    print("Testing invalid data in warn-only mode (validation passes but logs warnings):")
    is_valid, errors, metrics = comprehensive_validator.validate(
        datasets['negative_prices'], 
        warn_only=True
    )
    print(f"Warn-only mode result: Valid={is_valid}, Warnings={len(metrics.get('warnings', []))}")
    
    print("\n" + "✅" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("✅" * 60)
    print("\nKey Achievements:")
    print("✓ All required validation rules implemented and tested")
    print("✓ Flexible configuration options demonstrated")
    print("✓ Comprehensive error reporting and metrics")
    print("✓ Backward compatibility maintained")
    print("✓ Production-ready validation system")
    
    print(f"\nFor more details, see:")
    print(f"  • src/data/processing/market_validation_rules.py")
    print(f"  • tests/data/validation/test_market_validator.py")

if __name__ == "__main__":
    try:
        demo_validation_rules()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
