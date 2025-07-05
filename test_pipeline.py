#!/usr/bin/env python3
"""Test script for the market data pipeline."""

from pipelines.market_data_pipeline import MarketDataPipeline, load_sample_data
import pandas as pd

def test_pipeline():
    """Test the pipeline setup and functionality."""
    print('Testing Market Data Pipeline...')
    
    # Load sample data
    data = load_sample_data()
    print(f'Sample data columns: {list(data.columns)}')
    print(f'Sample data shape: {data.shape}')
    print(f'First few rows:')
    print(data.head())
    
    # Create pipeline
    pipeline = MarketDataPipeline(
        warn_only=True,
        output_dir='test_output',
        enable_all_features=False
    )
    
    print(f'\nPipeline stages: {len(pipeline.pipeline.stages)}')
    for stage in pipeline.pipeline.stages:
        print(f'  - {stage["name"]} ({stage["type"].name})')
    
    # Get pipeline summary
    summary = pipeline.get_pipeline_summary()
    print('\nPipeline Summary:')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    
    print('\nPipeline setup successful!')

if __name__ == "__main__":
    test_pipeline()
