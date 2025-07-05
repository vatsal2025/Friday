#!/usr/bin/env python3
"""Example usage of the Market Data Pipeline."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from pipelines.market_data_pipeline import MarketDataPipeline, load_sample_data

def create_example_data():
    """Create example market data file."""
    print("Creating example market data...")
    
    # Generate more realistic market data
    np.random.seed(123)
    n_days = 5
    n_minutes_per_day = 390  # Trading day is 6.5 hours
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    all_data = []
    
    for symbol in symbols:
        base_price = np.random.uniform(100, 300)
        
        for day in range(n_days):
            # Create intraday timestamps
            start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=day) + pd.Timedelta(hours=9, minutes=30)
            times = pd.date_range(start_time, periods=n_minutes_per_day, freq='1min')
            
            # Generate price movements with realistic patterns
            returns = np.random.normal(0, 0.001, n_minutes_per_day)
            # Add some autocorrelation
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLC
            opens = np.concatenate([[base_price], prices[:-1]])
            closes = prices
            highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0005, n_minutes_per_day)))
            lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0005, n_minutes_per_day)))
            
            # Generate volume with realistic patterns
            base_volume = np.random.uniform(10000, 50000)
            volume_multiplier = 1 + 0.5 * np.abs(returns)  # Higher volume on larger moves
            volumes = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, n_minutes_per_day)).astype(int)
            
            day_data = pd.DataFrame({
                'timestamp': times,
                'symbol': symbol,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            all_data.append(day_data)
            base_price = closes[-1]  # Carry price to next day
    
    # Combine all data
    market_data = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    market_data.to_csv('example_market_data.csv', index=False)
    print(f"Created example_market_data.csv with {len(market_data)} rows")
    print(f"Data range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    print(f"Symbols: {market_data['symbol'].unique()}")
    
    return market_data

def demonstrate_pipeline():
    """Demonstrate the full pipeline functionality."""
    print("\n" + "="*60)
    print("MARKET DATA PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Create example data
    data = create_example_data()
    
    # Example 1: Full pipeline with all features
    print(f"\n1. FULL PIPELINE WITH ALL FEATURES")
    print("-" * 40)
    
    pipeline_full = MarketDataPipeline(
        warn_only=True,
        output_dir='demo_output/full_features',
        enable_all_features=True,
        enable_storage=False  # Skip storage for demo
    )
    
    processed_data_full = pipeline_full.process_data(data.copy())
    print(f"Original columns: {len(data.columns)}")
    print(f"With all features: {len(processed_data_full.columns)}")
    print(f"Features added: {len(processed_data_full.columns) - len(data.columns)}")
    
    # Example 2: Minimal pipeline
    print(f"\n2. MINIMAL PIPELINE (CLEANING ONLY)")
    print("-" * 40)
    
    pipeline_minimal = MarketDataPipeline(
        warn_only=True,
        output_dir='demo_output/minimal',
        enable_all_features=False,
        enable_storage=False
    )
    
    processed_data_minimal = pipeline_minimal.process_data(data.copy())
    print(f"Original columns: {len(data.columns)}")
    print(f"With minimal features: {len(processed_data_minimal.columns)}")
    print(f"Features added: {len(processed_data_minimal.columns) - len(data.columns)}")
    
    # Example 3: Show some sample features
    print(f"\n3. SAMPLE TECHNICAL INDICATORS")
    print("-" * 40)
    
    sample_data = processed_data_full[processed_data_full['symbol'] == 'AAPL'].head(10)
    feature_cols = ['close', 'sma_5', 'sma_20', 'rsi_14', 'bollinger_upper', 'bollinger_lower']
    available_cols = [col for col in feature_cols if col in sample_data.columns]
    
    print("Sample technical indicators for AAPL:")
    print(sample_data[['timestamp'] + available_cols].to_string(index=False))
    
    # Example 4: Performance metrics
    print(f"\n4. PIPELINE PERFORMANCE METRICS")
    print("-" * 40)
    
    metadata = pipeline_full.get_pipeline_metadata()
    print(f"Processing time: {metadata.get('last_run_duration', 0):.4f} seconds")
    print(f"Rows per second: {len(data) / metadata.get('last_run_duration', 1):.0f}")
    
    summary = pipeline_full.get_pipeline_summary()
    print(f"Pipeline stages: {summary['total_stages']}")
    for stage_name in summary['stage_names']:
        print(f"  - {stage_name}")
    
    print(f"\n5. CLI USAGE EXAMPLES")
    print("-" * 40)
    print("# Process a CSV file with all features:")
    print("python -m pipelines.market_data_pipeline --input example_market_data.csv --outdir storage/data/processed")
    print("")
    print("# Process with validation warnings only:")
    print("python -m pipelines.market_data_pipeline --input example_market_data.csv --outdir storage/data/processed --warn-only")
    print("")
    print("# Process without feature engineering:")
    print("python -m pipelines.market_data_pipeline --input example_market_data.csv --outdir storage/data/processed --no-features")
    print("")
    print("# Process without storage (for testing):")
    print("python -m pipelines.market_data_pipeline --input example_market_data.csv --outdir storage/data/processed --no-storage")
    print("")
    print("# Use sample data for testing:")
    print("python -m pipelines.market_data_pipeline --sample --outdir test_output")
    
    print(f"\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    demonstrate_pipeline()
