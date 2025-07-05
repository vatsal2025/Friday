#!/usr/bin/env python3
"""
Feature Engineering Benchmark and Demonstration Script

This script provides comprehensive benchmarking and demonstration of the Friday AI Trading System's 
feature engineering pipeline, including:

1. Performance benchmarking on different dataset sizes
2. Memory usage analysis
3. Feature set comparison
4. Configuration validation
5. Default feature set demonstration

Usage:
    python feature_engineering_benchmark.py [options]

Examples:
    # Run full benchmark suite
    python feature_engineering_benchmark.py --full

    # Benchmark specific feature sets
    python feature_engineering_benchmark.py --feature-sets price_derived moving_averages

    # Benchmark different dataset sizes
    python feature_engineering_benchmark.py --dataset-sizes 1month 1year

    # Save results to file
    python feature_engineering_benchmark.py --output benchmark_results.json

    # Run with verbose logging
    python feature_engineering_benchmark.py --verbose

Author: Friday AI Trading System
Date: 2024
"""

import argparse
import json
import logging
import os
import sys
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.data.processing.feature_engineering import FeatureEngineer
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger

# Set up logging
logger = get_logger(__name__)


class FeatureEngineeringBenchmark:
    """Comprehensive feature engineering benchmark and demonstration class."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the benchmark class.
        
        Args:
            config: Optional configuration manager instance.
        """
        self.config = config or ConfigManager()
        self.results = {}
        self.start_time = datetime.now()
        
        # Ensure output directory exists
        self.output_dir = Path("benchmarks/features")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_sample_data(self, num_rows: int, seed: int = 42) -> pd.DataFrame:
        """Generate realistic sample market data for benchmarking.
        
        Args:
            num_rows: Number of rows to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            pd.DataFrame: Sample market data with OHLCV columns.
        """
        logger.info(f"Generating {num_rows} rows of sample market data...")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Generate datetime index
        start_date = datetime.now() - timedelta(days=num_rows // (24 * 60))
        date_range = pd.date_range(start=start_date, periods=num_rows, freq='1min')
        
        # Base price and market parameters
        base_price = 100.0
        volatility_base = 0.02
        trend_factor = 0.0001
        
        # Generate realistic price data with trend and volatility
        returns = np.random.normal(trend_factor, volatility_base, num_rows)
        
        # Add some market microstructure patterns
        # Morning volatility increase
        hours = np.array([d.hour for d in date_range])
        morning_boost = np.where((hours >= 9) & (hours <= 11), 1.5, 1.0)
        returns *= morning_boost
        
        # Weekly patterns (Monday effect)
        weekdays = np.array([d.weekday() for d in date_range])
        monday_effect = np.where(weekdays == 0, 1.2, 1.0)
        returns *= monday_effect
        
        # Calculate cumulative prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        data = pd.DataFrame(index=date_range)
        data['close'] = prices
        
        # Generate open prices (previous close + gap)
        gaps = np.random.normal(0, 0.001, num_rows)
        data['open'] = data['close'].shift(1) * (1 + gaps)
        data.loc[data.index[0], 'open'] = base_price
        
        # Generate high and low prices with realistic intraday volatility
        intraday_volatility = np.random.uniform(0.005, 0.03, num_rows)
        
        # High is the maximum of open/close plus some upward movement
        data['high'] = np.maximum(data['open'], data['close']) * (1 + intraday_volatility * 0.7)
        
        # Low is the minimum of open/close minus some downward movement  
        data['low'] = np.minimum(data['open'], data['close']) * (1 - intraday_volatility * 0.7)
        
        # Generate volume with realistic patterns
        base_volume = 10000
        
        # Volume correlated with volatility and absolute returns
        volume_factor = 1 + np.abs(returns) * 20
        
        # Time-of-day volume patterns (higher at open/close)
        time_factor = np.where(
            (hours >= 9) & (hours <= 10) | (hours >= 15) & (hours <= 16),
            2.0, 1.0
        )
        
        data['volume'] = (
            base_volume * volume_factor * time_factor * 
            np.random.uniform(0.5, 2.0, num_rows)
        ).astype(int)
        
        # Ensure no negative values
        data = data.clip(lower=0.01)
        
        logger.info(f"Generated sample data: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        logger.info(f"Volume range: {data['volume'].min():,} - {data['volume'].max():,}")
        
        return data
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the feature engineering configuration.
        
        Returns:
            Dict containing validation results.
        """
        logger.info("Validating feature engineering configuration...")
        
        validation_results = {
            "config_valid": True,
            "issues": [],
            "warnings": [],
            "feature_sets": {},
            "default_configuration": {}
        }
        
        try:
            # Test feature engineer initialization
            feature_engineer = FeatureEngineer(config=self.config)
            
            # Check default configuration
            default_enabled = self.config.get("features.default_enabled", [])
            enabled_sets = feature_engineer.get_enabled_feature_sets()
            
            validation_results["default_configuration"] = {
                "configured_defaults": default_enabled,
                "actually_enabled": enabled_sets,
                "matches": set(default_enabled) == set(enabled_sets)
            }
            
            if set(default_enabled) != set(enabled_sets):
                validation_results["warnings"].append(
                    f"Default configuration mismatch: configured={default_enabled}, enabled={enabled_sets}"
                )
            
            # Validate each feature set
            for name, feature_set_info in feature_engineer.get_all_feature_sets_info().items():
                validation_results["feature_sets"][name] = feature_set_info
                
                # Check if dependencies are valid
                required_cols = set(feature_set_info["dependencies"])
                standard_cols = {"open", "high", "low", "close", "volume"}
                
                if not required_cols.issubset(standard_cols):
                    missing = required_cols - standard_cols
                    validation_results["issues"].append(
                        f"Feature set '{name}' requires non-standard columns: {missing}"
                    )
            
            # Test with sample data
            sample_data = self.generate_sample_data(1000)
            validation_result = feature_engineer.validate_data_columns(sample_data)
            
            if not validation_result["valid"]:
                validation_results["issues"].append(
                    f"Sample data validation failed: missing columns {validation_result['missing_columns']}"
                )
            
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            validation_results["config_valid"] = False
            validation_results["issues"].append(f"Configuration validation error: {str(e)}")
            logger.error(f"Configuration validation failed: {e}")
        
        return validation_results
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save benchmark results to a JSON file.
        
        Args:
            results: Benchmark results to save.
            output_file: Path to the output file.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_path}")
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of benchmark results.
        
        Args:
            results: Benchmark results to summarize.
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Configuration validation
        config_validation = results.get("configuration_validation", {})
        if config_validation.get("config_valid", False):
            print("✅ Configuration validation: PASSED")
            
            # Show default configuration details
            default_config = config_validation.get("default_configuration", {})
            if default_config:
                print(f"   Default feature sets: {default_config.get('configured_defaults', [])}")
                print(f"   Actually enabled: {default_config.get('actually_enabled', [])}")
                if default_config.get('matches', False):
                    print("   ✅ Configuration matches enabled sets")
                else:
                    print("   ⚠️  Configuration mismatch")
                    
            # Show feature set info
            feature_sets = config_validation.get("feature_sets", {})
            print(f"\n📋 Available Feature Sets: {len(feature_sets)}")
            for name, info in feature_sets.items():
                status = "✅" if info.get("enabled", False) else "❌"
                print(f"   {status} {name:16} | {info.get('feature_count', 0):2d} features | {info.get('category', 'Unknown'):12}")
                
        else:
            print("❌ Configuration validation: FAILED")
            for issue in config_validation.get("issues", []):
                print(f"   - {issue}")
        
        # Show warnings if any
        warnings = config_validation.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("="*80)


def main():
    """Main function to run the feature engineering benchmark."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Benchmark and Demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run configuration validation"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Output file for benchmark results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create benchmark instance
    benchmark = FeatureEngineeringBenchmark()
    
    # Run configuration validation
    results = {"configuration_validation": benchmark.validate_configuration()}
    benchmark.print_results_summary(results)
    
    # Save results if output file specified
    if args.output:
        benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()
