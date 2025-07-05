"""
Feature Engineering Pipeline Configuration

This script demonstrates the configuration of the FeatureEngineer class with:
1. Instantiation with enable_all_features=True
2. Selective feature enablement
3. Required column validation
4. Computational cost and memory impact documentation

Author: Friday AI Trading System
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
import time
import psutil
import os
from datetime import datetime, timedelta

from src.data.processing.feature_engineering import FeatureEngineer, FeatureCategory
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

class FeatureEngineeringPipelineConfig:
    """Configuration and demonstration class for Feature Engineering Pipeline."""
    
    def __init__(self):
        """Initialize the pipeline configuration."""
        self.config = ConfigManager()
        self.feature_engineer = None
        self.performance_metrics = {}
        
    def configure_with_all_features(self) -> FeatureEngineer:
        """
        Configure FeatureEngineer with enable_all_features=True.
        
        Returns:
            FeatureEngineer: Configured feature engineer instance.
        """
        logger.info("=== Configuring FeatureEngineer with enable_all_features=True ===")
        
        # Instantiate with all features enabled
        self.feature_engineer = FeatureEngineer(
            config=self.config,
            enable_all_features=True
        )
        
        # Log configuration details
        enabled_sets = self.feature_engineer.get_enabled_feature_sets()
        logger.info(f"Enabled feature sets: {enabled_sets}")
        logger.info(f"Total enabled features: {len(self.feature_engineer.enabled_features)}")
        
        return self.feature_engineer
        
    def configure_selective_features(self) -> FeatureEngineer:
        """
        Configure FeatureEngineer with selective feature sets.
        
        Returns:
            FeatureEngineer: Configured feature engineer instance.
        """
        logger.info("=== Configuring FeatureEngineer with selective features ===")
        
        # Instantiate without enabling all features
        self.feature_engineer = FeatureEngineer(
            config=self.config,
            enable_all_features=False
        )
        
        # Selectively enable feature sets as specified in the task
        feature_sets_to_enable = [
            "price_derived",    # Price derived features
            "moving_averages",  # Moving averages
            "volatility",       # Volatility indicators
            "momentum",         # Momentum indicators
            "volume",           # Volume indicators
            "trend"             # Trend indicators
        ]
        
        for feature_set in feature_sets_to_enable:
            try:
                self.feature_engineer.enable_feature_set(feature_set)
                logger.info(f"✓ Enabled feature set: {feature_set}")
            except ValueError as e:
                logger.error(f"✗ Failed to enable feature set {feature_set}: {e}")
        
        # Log configuration details
        enabled_sets = self.feature_engineer.get_enabled_feature_sets()
        logger.info(f"Enabled feature sets: {enabled_sets}")
        logger.info(f"Total enabled features: {len(self.feature_engineer.enabled_features)}")
        
        return self.feature_engineer
        
    def validate_required_columns(self) -> Dict[str, Set[str]]:
        """
        Validate required column coverage via get_required_columns().
        
        Returns:
            Dict[str, Set[str]]: Dictionary containing required columns and coverage info.
        """
        logger.info("=== Validating Required Column Coverage ===")
        
        if not self.feature_engineer:
            raise ValueError("Feature engineer not configured. Call configure_with_all_features() or configure_selective_features() first.")
        
        # Get required columns
        required_columns = self.feature_engineer.get_required_columns()
        logger.info(f"Required columns for enabled features: {sorted(required_columns)}")
        
        # Analyze column requirements by feature set
        column_requirements = {}
        for name, feature_set in self.feature_engineer.feature_sets.items():
            if any(feature in self.feature_engineer.enabled_features for feature in feature_set.features):
                column_requirements[name] = {
                    'features': feature_set.features,
                    'dependencies': feature_set.dependencies,
                    'category': feature_set.category.name,
                    'description': feature_set.description
                }
        
        # Log detailed requirements
        logger.info("\n=== Feature Set Requirements ===")
        for name, requirements in column_requirements.items():
            logger.info(f"\nFeature Set: {name}")
            logger.info(f"  Category: {requirements['category']}")
            logger.info(f"  Description: {requirements['description']}")
            logger.info(f"  Features: {requirements['features']}")
            logger.info(f"  Required Columns: {requirements['dependencies']}")
        
        # Standard market data columns
        standard_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_standard = required_columns - standard_columns
        extra_standard = standard_columns - required_columns
        
        logger.info(f"\n=== Column Coverage Analysis ===")
        logger.info(f"Standard OHLCV columns: {sorted(standard_columns)}")
        logger.info(f"Additional required columns: {sorted(missing_standard) if missing_standard else 'None'}")
        logger.info(f"Unused standard columns: {sorted(extra_standard) if extra_standard else 'None'}")
        
        return {
            'required_columns': required_columns,
            'column_requirements': column_requirements,
            'standard_columns': standard_columns,
            'missing_standard': missing_standard,
            'extra_standard': extra_standard
        }
        
    def generate_sample_data(self, num_rows: int = 1000) -> pd.DataFrame:
        """
        Generate sample market data for testing.
        
        Args:
            num_rows: Number of rows to generate.
            
        Returns:
            pd.DataFrame: Sample market data.
        """
        logger.info(f"Generating sample data with {num_rows} rows...")
        
        # Generate datetime index
        start_date = datetime.now() - timedelta(days=num_rows)
        date_range = pd.date_range(start=start_date, periods=num_rows, freq='1min')
        
        # Generate realistic market data
        np.random.seed(42)  # For reproducibility
        
        # Base price
        base_price = 100.0
        
        # Generate price data with trend and volatility
        returns = np.random.normal(0.0001, 0.02, num_rows)  # Small positive trend with volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=date_range)
        
        # Close prices
        data['close'] = prices
        
        # Open prices (close of previous period with small gap)
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.001, num_rows))
        data['open'].iloc[0] = base_price
        
        # High and low prices
        volatility = np.random.uniform(0.005, 0.03, num_rows)
        data['high'] = np.maximum(data['open'], data['close']) * (1 + volatility)
        data['low'] = np.minimum(data['open'], data['close']) * (1 - volatility)
        
        # Volume (correlated with price volatility)
        base_volume = 10000
        volume_multiplier = 1 + np.abs(returns) * 10  # Higher volume during high volatility
        data['volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, num_rows)).astype(int)
        
        # Ensure no negative prices
        data = data.clip(lower=0.01)
        
        logger.info(f"Generated sample data: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        return data
        
    def measure_computational_cost(self, data: pd.DataFrame) -> Dict:
        """
        Measure computational cost and memory impact of feature generation.
        
        Args:
            data: Input market data.
            
        Returns:
            Dict: Performance metrics.
        """
        logger.info("=== Measuring Computational Cost and Memory Impact ===")
        
        if not self.feature_engineer:
            raise ValueError("Feature engineer not configured.")
        
        # Measure memory before processing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure processing time
        start_time = time.time()
        
        # Process the data
        processed_data = self.feature_engineer.process_data(data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Calculate data size metrics
        original_size = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        processed_size = processed_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        size_increase = processed_size - original_size
        
        # Calculate feature metrics
        original_columns = len(data.columns)
        processed_columns = len(processed_data.columns)
        new_features = processed_columns - original_columns
        
        # Store performance metrics
        self.performance_metrics = {
            'processing_time_seconds': processing_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_increase,
            'original_data_size_mb': original_size,
            'processed_data_size_mb': processed_size,
            'data_size_increase_mb': size_increase,
            'original_columns': original_columns,
            'processed_columns': processed_columns,
            'new_features_count': new_features,
            'data_rows': len(data),
            'enabled_features': list(self.feature_engineer.enabled_features),
            'enabled_feature_sets': self.feature_engineer.get_enabled_feature_sets()
        }
        
        # Log performance metrics
        logger.info(f"\n=== Performance Metrics ===")
        logger.info(f"Processing Time: {processing_time:.4f} seconds")
        logger.info(f"Memory Usage: {memory_before:.2f} MB → {memory_after:.2f} MB (+{memory_increase:.2f} MB)")
        logger.info(f"Data Size: {original_size:.2f} MB → {processed_size:.2f} MB (+{size_increase:.2f} MB)")
        logger.info(f"Columns: {original_columns} → {processed_columns} (+{new_features} features)")
        logger.info(f"Processing Rate: {len(data) / processing_time:.0f} rows/second")
        logger.info(f"Memory per Feature: {memory_increase / new_features:.4f} MB/feature")
        
        return self.performance_metrics
        
    def document_feature_sets(self) -> None:
        """Document all available feature sets and their characteristics."""
        logger.info("=== Feature Set Documentation ===")
        
        if not self.feature_engineer:
            raise ValueError("Feature engineer not configured.")
        
        # Document each feature set
        for name, feature_set in self.feature_engineer.feature_sets.items():
            enabled = all(feature in self.feature_engineer.enabled_features for feature in feature_set.features)
            status = "✓ ENABLED" if enabled else "✗ DISABLED"
            
            logger.info(f"\n{status} Feature Set: {name}")
            logger.info(f"  Category: {feature_set.category.name}")
            logger.info(f"  Description: {feature_set.description}")
            logger.info(f"  Features ({len(feature_set.features)}): {', '.join(feature_set.features)}")
            logger.info(f"  Dependencies: {', '.join(feature_set.dependencies)}")
            
            # Estimate computational complexity
            complexity = self._estimate_complexity(feature_set)
            logger.info(f"  Estimated Complexity: {complexity}")
            
    def _estimate_complexity(self, feature_set) -> str:
        """Estimate computational complexity of a feature set."""
        feature_count = len(feature_set.features)
        
        # Complexity estimation based on feature types
        if feature_set.category == FeatureCategory.PRICE:
            return "Low (O(n) - simple arithmetic operations)"
        elif feature_set.category == FeatureCategory.TREND:
            if "moving_averages" in feature_set.name:
                return "Medium (O(n*k) - rolling calculations)"
            else:
                return "High (O(n²) - complex trend calculations)"
        elif feature_set.category == FeatureCategory.VOLATILITY:
            return "Medium-High (O(n*k) - rolling statistics with multiple series)"
        elif feature_set.category == FeatureCategory.MOMENTUM:
            return "Medium (O(n*k) - exponential smoothing and rolling calculations)"
        elif feature_set.category == FeatureCategory.VOLUME:
            return "Medium (O(n*k) - volume-weighted calculations)"
        else:
            return "Variable (depends on specific implementations)"
            
    def run_full_demonstration(self) -> None:
        """Run a complete demonstration of the feature engineering pipeline."""
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE DEMONSTRATION")
        logger.info("=" * 60)
        
        try:
            # Step 1: Configure with all features
            logger.info("\n" + "=" * 40)
            logger.info("STEP 1: Configure with All Features")
            logger.info("=" * 40)
            self.configure_with_all_features()
            
            # Step 2: Validate required columns
            logger.info("\n" + "=" * 40)
            logger.info("STEP 2: Validate Required Columns")
            logger.info("=" * 40)
            column_info = self.validate_required_columns()
            
            # Step 3: Generate sample data
            logger.info("\n" + "=" * 40)
            logger.info("STEP 3: Generate Sample Data")
            logger.info("=" * 40)
            sample_data = self.generate_sample_data(num_rows=5000)
            
            # Step 4: Measure computational cost
            logger.info("\n" + "=" * 40)
            logger.info("STEP 4: Measure Computational Cost")
            logger.info("=" * 40)
            performance = self.measure_computational_cost(sample_data)
            
            # Step 5: Document feature sets
            logger.info("\n" + "=" * 40)
            logger.info("STEP 5: Document Feature Sets")
            logger.info("=" * 40)
            self.document_feature_sets()
            
            # Step 6: Demonstrate selective configuration
            logger.info("\n" + "=" * 40)
            logger.info("STEP 6: Selective Feature Configuration")
            logger.info("=" * 40)
            self.configure_selective_features()
            self.validate_required_columns()
            
            # Final summary
            logger.info("\n" + "=" * 40)
            logger.info("DEMONSTRATION COMPLETE")
            logger.info("=" * 40)
            logger.info("✓ FeatureEngineer successfully configured with enable_all_features=True")
            logger.info("✓ Selective feature enabling demonstrated")
            logger.info("✓ Required column validation completed")
            logger.info("✓ Computational cost and memory impact documented")
            logger.info("✓ All feature sets documented with complexity estimates")
            
        except Exception as e:
            logger.error(f"Error during demonstration: {e}")
            raise


def main():
    """Main function to run the feature engineering pipeline configuration."""
    config = FeatureEngineeringPipelineConfig()
    config.run_full_demonstration()


if __name__ == "__main__":
    main()
