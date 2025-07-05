#!/usr/bin/env python3
"""Comprehensive test for the market data pipeline."""

import sys
import traceback
from pathlib import Path
from pipelines.market_data_pipeline import MarketDataPipeline, load_sample_data

def test_full_pipeline():
    """Test the complete pipeline functionality."""
    try:
        print("=" * 60)
        print("COMPREHENSIVE MARKET DATA PIPELINE TEST")
        print("=" * 60)
        
        # Step 1: Load sample data
        print("\n1. Loading sample data...")
        data = load_sample_data()
        print(f"   Sample data shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   First few rows:")
        print(data.head(3).to_string())
        
        # Step 2: Create pipeline
        print("\n2. Creating pipeline...")
        pipeline = MarketDataPipeline(
            warn_only=True,
            output_dir='test_full_output',
            enable_all_features=False  # Use limited features for faster testing
        )
        print(f"   Pipeline created with {len(pipeline.pipeline.stages)} stages")
        
        # Step 3: Show pipeline configuration
        print("\n3. Pipeline configuration:")
        summary = pipeline.get_pipeline_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Step 4: Process data
        print("\n4. Processing data through pipeline...")
        processed_data = pipeline.process_data(data)
        
        print(f"   Processing completed!")
        print(f"   Original shape: {data.shape}")
        print(f"   Processed shape: {processed_data.shape}")
        print(f"   New columns added: {len(processed_data.columns) - len(data.columns)}")
        print(f"   Final columns: {list(processed_data.columns)}")
        
        # Step 5: Show sample processed data
        print("\n5. Sample processed data:")
        print(processed_data.head(3).to_string())
        
        # Step 6: Show pipeline metadata
        print("\n6. Pipeline execution metadata:")
        metadata = pipeline.get_pipeline_metadata()
        print(f"   Last run status: {metadata.get('last_run_status')}")
        print(f"   Last run duration: {metadata.get('last_run_duration'):.4f}s")
        print(f"   Total runs: {len(metadata.get('runs', []))}")
        
        # Step 7: Show feature details
        print("\n7. Feature engineering results:")
        original_cols = set(data.columns)
        new_cols = set(processed_data.columns) - original_cols
        if new_cols:
            print(f"   Added features ({len(new_cols)}):")
            for col in sorted(new_cols):
                print(f"     - {col}")
        else:
            print("   No new features added")
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
