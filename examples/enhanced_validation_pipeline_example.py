#!/usr/bin/env python3
"""
Enhanced Data Validation Pipeline Example

This example demonstrates the integration of the enhanced DataValidator
into DataPipeline with detailed metrics emission and warn_only mode.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import sys
import os

# Add src to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from data.integration.data_pipeline import DataPipeline, PipelineError
from data.processing.data_validator import DataValidator, ValidationRule, build_default_market_validator
from infrastructure.event.event_system import EventSystem
from infrastructure.logging import get_logger

# Set up logging
logger = get_logger(__name__)


def create_sample_market_data(rows: int = 1000, add_errors: bool = False) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = pd.date_range(start_time, periods=rows, freq='1min')
    
    # Create base price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.5, rows)
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Calculate OHLC with some random variation
        open_price = price + np.random.normal(0, 0.1)
        close_price = price + np.random.normal(0, 0.1)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'symbol': 'AAPL',
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    if add_errors:
        # Introduce some validation errors for testing
        # Negative price
        df.loc[10, 'low'] = -5.0
        
        # High < Low
        df.loc[20, 'high'] = 50.0
        df.loc[20, 'low'] = 60.0
        
        # Negative volume
        df.loc[30, 'volume'] = -100
        
        # Duplicate timestamp
        df.loc[40, 'timestamp'] = df.loc[39, 'timestamp']
        
        # Missing value
        df.loc[50, 'close'] = np.nan
    
    return df


def create_event_handlers(event_system: EventSystem):
    """Create event handlers to monitor validation metrics."""
    
    def validation_metrics_handler(event):
        """Handle validation metrics events."""
        data = event.data
        metrics = data['validation_metrics']
        
        logger.info(f"=== Validation Metrics for {data['stage_name']} ===")
        logger.info(f"Pipeline: {data['pipeline_name']} (Run: {data['run_id']})")
        logger.info(f"Data shape: {metrics['data_shape']}")
        logger.info(f"Data size: {metrics['data_size_mb']:.2f} MB")
        logger.info(f"Rules tested: {metrics['rules_tested']}")
        logger.info(f"Rules passed: {metrics['rules_passed']}")
        logger.info(f"Rules failed: {metrics['rules_failed']}")
        logger.info(f"Success rate: {metrics['success_rate']:.2%}")
        logger.info(f"Total duration: {metrics['total_duration_seconds']:.4f}s")
        logger.info(f"Warn-only mode: {metrics['warn_only_mode']}")
        
        if metrics.get('warnings'):
            logger.info(f"Warnings ({len(metrics['warnings'])}): {metrics['warnings']}")
        
        if metrics.get('error_messages'):
            logger.info(f"Errors ({len(metrics['error_messages'])}): {metrics['error_messages']}")
        
        # Print detailed rule results
        logger.info("Rule-by-rule results:")
        for rule_name, result in metrics['rule_results'].items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            duration = result['duration_seconds']
            logger.info(f"  {rule_name}: {status} ({duration:.4f}s)")
            if not result['passed'] and result.get('error_message'):
                logger.info(f"    Error: {result['error_message']}")
    
    def validation_failed_handler(event):
        """Handle validation failure events."""
        data = event.data
        logger.error(f"=== Validation Failed in {data['stage_name']} ===")
        logger.error(f"Pipeline: {data['pipeline_name']} (Run: {data['run_id']})")
        logger.error(f"Errors: {data['error_messages']}")
    
    # Register event handlers
    event_system.register_handler(
        validation_metrics_handler,
        event_types=["pipeline.validation.metrics"]
    )
    
    event_system.register_handler(
        validation_failed_handler,
        event_types=["pipeline.validation.failed"]
    )


def example_1_basic_enhanced_validation():
    """Example 1: Basic enhanced validation with clean data."""
    logger.info("=== Example 1: Basic Enhanced Validation ===")
    
    # Create event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers
    create_event_handlers(event_system)
    
    try:
        # Create clean sample data
        data = create_sample_market_data(rows=100, add_errors=False)
        logger.info(f"Created clean sample data with shape: {data.shape}")
        
        # Create pipeline with enhanced validation
        pipeline = DataPipeline(
            name="basic_validation_pipeline",
            event_system=event_system
        )
        
        # Add enhanced validation stage with all default rules
        pipeline.add_enhanced_validation_stage(
            stage_name="market_data_validation",
            warn_only=False
        )
        
        # Execute pipeline
        result = pipeline.execute(input_data=data)
        logger.info("✓ Pipeline executed successfully!")
        logger.info(f"Result shape: {result.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        event_system.stop()


def example_2_validation_with_errors_strict_mode():
    """Example 2: Validation with errors in strict mode (should fail)."""
    logger.info("\n=== Example 2: Validation with Errors (Strict Mode) ===")
    
    # Create event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers
    create_event_handlers(event_system)
    
    try:
        # Create data with validation errors
        data = create_sample_market_data(rows=100, add_errors=True)
        logger.info(f"Created sample data with errors, shape: {data.shape}")
        
        # Create pipeline with enhanced validation
        pipeline = DataPipeline(
            name="strict_validation_pipeline",
            event_system=event_system
        )
        
        # Add enhanced validation stage in strict mode
        pipeline.add_enhanced_validation_stage(
            stage_name="strict_market_validation",
            warn_only=False  # Strict mode - should fail on errors
        )
        
        # Execute pipeline (should fail)
        result = pipeline.execute(input_data=data)
        logger.error("Pipeline should have failed but didn't!")
        
    except PipelineError as e:
        logger.info(f"✓ Pipeline correctly failed in strict mode: {e}")
        logger.info(f"  Stage: {e.stage}")
        if hasattr(e, 'details') and e.details:
            details = e.details
            if isinstance(details, dict) and 'validation_metrics' in details:
                metrics = details['validation_metrics']
                logger.info(f"  Failed rules: {metrics['rules_failed']}/{metrics['rules_tested']}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        event_system.stop()


def example_3_validation_with_errors_warn_only_mode():
    """Example 3: Validation with errors in warn-only mode (should continue)."""
    logger.info("\n=== Example 3: Validation with Errors (Warn-Only Mode) ===")
    
    # Create event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers
    create_event_handlers(event_system)
    
    try:
        # Create data with validation errors
        data = create_sample_market_data(rows=100, add_errors=True)
        logger.info(f"Created sample data with errors, shape: {data.shape}")
        
        # Create pipeline with enhanced validation
        pipeline = DataPipeline(
            name="warn_only_validation_pipeline",
            event_system=event_system
        )
        
        # Add enhanced validation stage in warn-only mode
        pipeline.add_enhanced_validation_stage(
            stage_name="warn_only_market_validation",
            warn_only=True  # Warn-only mode - should continue despite errors
        )
        
        # Execute pipeline (should succeed despite errors)
        result = pipeline.execute(input_data=data)
        logger.info("✓ Pipeline completed successfully in warn-only mode!")
        logger.info(f"Result shape: {result.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed unexpectedly: {e}")
    finally:
        event_system.stop()


def example_4_custom_validation_rules():
    """Example 4: Custom validation rules with specific requirements."""
    logger.info("\n=== Example 4: Custom Validation Rules ===")
    
    # Create event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers
    create_event_handlers(event_system)
    
    try:
        # Create sample data
        data = create_sample_market_data(rows=100, add_errors=False)
        logger.info(f"Created sample data, shape: {data.shape}")
        
        # Create pipeline with enhanced validation
        pipeline = DataPipeline(
            name="custom_validation_pipeline",
            event_system=event_system
        )
        
        # Add enhanced validation stage with specific rules only
        # Test only basic price validation rules
        pipeline.add_enhanced_validation_stage(
            stage_name="custom_market_validation",
            validation_rules=[
                "ohlcv_columns",
                "no_negative_prices", 
                "high_low_consistency",
                "non_negative_volume"
            ],
            warn_only=False
        )
        
        # Execute pipeline
        result = pipeline.execute(input_data=data)
        logger.info("✓ Pipeline with custom rules executed successfully!")
        logger.info(f"Result shape: {result.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        event_system.stop()


def example_5_trading_hours_validation():
    """Example 5: Trading hours validation."""
    logger.info("\n=== Example 5: Trading Hours Validation ===")
    
    # Create event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers
    create_event_handlers(event_system)
    
    try:
        # Create sample data
        data = create_sample_market_data(rows=100, add_errors=False)
        logger.info(f"Created sample data, shape: {data.shape}")
        
        # Create pipeline with enhanced validation including trading hours
        pipeline = DataPipeline(
            name="trading_hours_validation_pipeline",
            event_system=event_system
        )
        
        # Add enhanced validation stage with trading hours constraints
        pipeline.add_enhanced_validation_stage(
            stage_name="trading_hours_validation",
            trading_hours_start=time(9, 30),  # 9:30 AM
            trading_hours_end=time(16, 0),    # 4:00 PM
            warn_only=True  # Use warn-only since our sample data might be outside trading hours
        )
        
        # Execute pipeline
        result = pipeline.execute(input_data=data)
        logger.info("✓ Pipeline with trading hours validation completed!")
        logger.info(f"Result shape: {result.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        event_system.stop()


def main():
    """Run all examples."""
    logger.info("Starting Enhanced Data Validation Pipeline Examples")
    logger.info("=" * 60)
    
    # Run examples
    example_1_basic_enhanced_validation()
    example_2_validation_with_errors_strict_mode()
    example_3_validation_with_errors_warn_only_mode()
    example_4_custom_validation_rules()
    example_5_trading_hours_validation()
    
    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")


if __name__ == "__main__":
    main()
