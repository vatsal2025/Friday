"""Enhanced Data Storage Procedures Example for Friday AI Trading System.

This example demonstrates the new data storage procedures including:
- LocalParquetStorage as default with partitioning validation
- Database options (MongoDB, PostgreSQL) behind DataStorage interface
- Automatic directory creation, file rotation, and metadata logging
- Retrieval utilities for downstream model training
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.storage import (
    get_default_storage,
    get_training_storage,
    create_storage,
    DataStorageFactory,
    DataRetrievalUtils,
    get_training_data,
    get_feature_matrix
)
from src.infrastructure.config import ConfigManager


def create_sample_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'], days=30):
    """Create sample market data for testing."""
    data_frames = []
    
    for symbol in symbols:
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days,
            freq='D'
        )
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
        base_price = np.random.uniform(100, 300)
        
        data = {
            'symbol': symbol,
            'date': dates,
            'open': base_price + np.random.normal(0, 5, len(dates)).cumsum(),
            'high': np.nan,  # Will be calculated
            'low': np.nan,   # Will be calculated
            'close': np.nan, # Will be calculated
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }
        
        # Calculate OHLC relationships
        df = pd.DataFrame(data)
        df['close'] = df['open'] + np.random.normal(0, 2, len(df))
        df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0, 3, len(df))
        df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0, 3, len(df))
        
        data_frames.append(df)
    
    return pd.concat(data_frames, ignore_index=True)


def demonstrate_default_storage():
    """Demonstrate LocalParquetStorage as default storage."""
    print("\n" + "="*60)
    print("DEMONSTRATING DEFAULT STORAGE (LocalParquetStorage)")
    print("="*60)
    
    # Get default storage
    storage = get_default_storage()
    print(f"✓ Default storage type: {type(storage).__name__}")
    
    # Create sample data
    market_data = create_sample_market_data()
    print(f"✓ Created sample data: {len(market_data)} rows")
    
    # Store data with automatic partitioning
    print("\n📁 Storing data with automatic partitioning...")
    success = storage.store_data(market_data, "market_data")
    print(f"✓ Data stored successfully: {success}")
    
    # Show partition structure
    if hasattr(storage, 'get_partition_info'):
        print("\n🗂️  Partition structure:")
        partition_info = storage.get_partition_info()
        for symbol, dates in partition_info.items():
            print(f"  {symbol}:")
            for date, info in dates.items():
                print(f"    {date}: {info['file_count']} files, tables: {info['tables']}")
    
    # Test retrieval
    print("\n📖 Testing data retrieval...")
    retrieved_data = storage.retrieve_data("market_data")
    print(f"✓ Retrieved {len(retrieved_data)} rows")
    
    # Test specific partition retrieval
    if hasattr(storage, 'retrieve_data'):
        aapl_data = storage.retrieve_data("market_data", symbol="AAPL")
        print(f"✓ Retrieved {len(aapl_data)} rows for AAPL")
    
    return storage


def demonstrate_storage_factory():
    """Demonstrate the DataStorageFactory."""
    print("\n" + "="*60)
    print("DEMONSTRATING STORAGE FACTORY")
    print("="*60)
    
    # Create factory
    factory = DataStorageFactory()
    print(f"✓ Created storage factory")
    
    # List supported types
    supported_types = factory.list_supported_types()
    print(f"✓ Supported storage types: {supported_types}")
    
    # Validate configuration
    print("\n🔍 Validating storage configurations...")
    validation_results = factory.validate_storage_config()
    
    for storage_type, result in validation_results.items():
        status = "✓" if result['valid'] else "✗"
        print(f"  {status} {storage_type}: {'Valid' if result['valid'] else 'Issues found'}")
        
        if not result['valid'] and 'issues' in result:
            for issue in result['issues'][:2]:  # Show first 2 issues
                print(f"    - {issue}")
    
    # Create different storage types
    print("\n🏗️  Creating different storage instances...")
    
    try:
        parquet_storage = factory.create_storage("local_parquet")
        print(f"✓ Created LocalParquetStorage")
    except Exception as e:
        print(f"✗ Failed to create LocalParquetStorage: {e}")
    
    try:
        # This might fail if MongoDB is not configured/available
        mongodb_storage = factory.create_storage("mongodb")
        print(f"✓ Created MongoDBStorage")
    except Exception as e:
        print(f"⚠ Could not create MongoDBStorage: {e}")
    
    return factory


def demonstrate_file_rotation_and_metadata():
    """Demonstrate file rotation and metadata logging."""
    print("\n" + "="*60)
    print("DEMONSTRATING FILE ROTATION & METADATA LOGGING")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create LocalParquetStorage with custom config
        config = ConfigManager()
        
        # Set up configuration for demonstration
        test_config = {
            'data': {
                'storage': {
                    'local_parquet': {
                        'base_dir': temp_dir
                    },
                    'file_rotation': {
                        'enabled': True,
                        'strategy': 'size_based',
                        'max_file_size_mb': 0.001,  # Very small for demo
                        'compress_old_files': True
                    },
                    'metadata_logging': {
                        'enabled': True,
                        'log_operations': True,
                        'log_performance': True,
                        'detailed_logging': True
                    },
                    'auto_directory_creation': {
                        'enabled': True,
                        'create_parents': True
                    }
                }
            }
        }
        
        # Update config (this is for demonstration - in real usage, config would be loaded from file)
        for key, value in test_config.items():
            config.set(key, value)
        
        # Create storage with custom config
        storage = create_storage("local_parquet", config=config, base_dir=temp_dir)
        print("✓ Created storage with enhanced configuration")
        
        # Store data multiple times to trigger rotation
        print("\n📁 Storing data to trigger file rotation...")
        for i in range(3):
            sample_data = create_sample_market_data(symbols=['TEST'], days=100)
            success = storage.store_data(sample_data, f"test_data_{i}")
            print(f"✓ Stored batch {i+1}")
        
        # Check directory structure
        print("\n🗂️  Directory structure:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Check metadata
        metadata_dir = Path(temp_dir) / '.metadata'
        if metadata_dir.exists():
            print(f"\n📊 Metadata files found:")
            for metadata_file in metadata_dir.glob("*"):
                print(f"  {metadata_file.name}")
                
                if metadata_file.suffix == '.jsonl' and metadata_file.stat().st_size < 1000:
                    print(f"    Sample content:")
                    with open(metadata_file, 'r') as f:
                        lines = f.readlines()[:2]  # Show first 2 lines
                        for line in lines:
                            print(f"      {line.strip()}")


def demonstrate_retrieval_utilities():
    """Demonstrate retrieval utilities for model training."""
    print("\n" + "="*60)
    print("DEMONSTRATING RETRIEVAL UTILITIES")
    print("="*60)
    
    # Create and store sample data
    storage = get_training_storage()
    market_data = create_sample_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'], days=100)
    
    # Add some feature columns for ML demonstration
    market_data['sma_10'] = market_data.groupby('symbol')['close'].rolling(10).mean().reset_index(0, drop=True)
    market_data['sma_20'] = market_data.groupby('symbol')['close'].rolling(20).mean().reset_index(0, drop=True)
    market_data['rsi'] = np.random.uniform(20, 80, len(market_data))  # Mock RSI
    market_data['returns'] = market_data.groupby('symbol')['close'].pct_change()
    
    # Store the enhanced data
    storage.store_data(market_data, "enhanced_market_data")
    print(f"✓ Stored enhanced market data: {len(market_data)} rows")
    
    # Initialize retrieval utilities
    retrieval_utils = DataRetrievalUtils(storage=storage)
    print("✓ Initialized DataRetrievalUtils")
    
    # Demonstrate training data splits
    print("\n🎯 Getting training data with automatic splits...")
    training_data = retrieval_utils.get_training_data(
        symbols=['AAPL', 'MSFT'],
        table_name="enhanced_market_data",
        features=['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi'],
        target_column='returns',
        test_size=0.2,
        validation_size=0.1
    )
    
    print(f"✓ Training splits created:")
    for split_name, split_data in training_data.items():
        if not split_data.empty:
            print(f"  {split_name}: {len(split_data)} rows")
    
    # Demonstrate feature matrix creation
    print("\n🧮 Creating feature matrix...")
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_10', 'sma_20', 'rsi']
    
    try:
        X, feature_names, scaler = retrieval_utils.get_feature_matrix(
            symbols=['AAPL'],
            feature_columns=feature_columns,
            table_name="enhanced_market_data",
            normalize=True,
            handle_missing='drop'
        )
        
        print(f"✓ Feature matrix created: {X.shape}")
        print(f"✓ Features: {feature_names}")
        print(f"✓ Scaler type: {type(scaler).__name__}")
        
    except Exception as e:
        print(f"⚠ Feature matrix creation had issues: {e}")
    
    # Demonstrate time series data
    print("\n📈 Creating time series data...")
    try:
        X_seq, y_seq = retrieval_utils.get_time_series_data(
            symbol='AAPL',
            table_name="enhanced_market_data",
            sequence_length=10,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'volume']
        )
        
        print(f"✓ Time series data created: X{X_seq.shape}, y{y_seq.shape}")
        
    except Exception as e:
        print(f"⚠ Time series creation had issues: {e}")
    
    # Demonstrate batch iterator
    print("\n🔄 Testing batch iterator...")
    batch_count = 0
    total_rows = 0
    
    for batch in retrieval_utils.get_batch_iterator(
        symbols=['AAPL', 'MSFT'],
        table_name="enhanced_market_data",
        batch_size=20
    ):
        batch_count += 1
        total_rows += len(batch)
        if batch_count <= 3:  # Show first 3 batches
            print(f"  Batch {batch_count}: {len(batch)} rows")
    
    print(f"✓ Processed {batch_count} batches, {total_rows} total rows")
    
    # Show cache statistics
    cache_stats = retrieval_utils.get_cache_stats()
    print(f"\n💾 Cache statistics: {cache_stats}")


def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n" + "="*60)
    print("DEMONSTRATING CONVENIENCE FUNCTIONS")
    print("="*60)
    
    # Ensure we have data
    storage = get_default_storage()
    market_data = create_sample_market_data(symbols=['CONVENIENCE_TEST'], days=50)
    market_data['returns'] = market_data['close'].pct_change()
    storage.store_data(market_data, "convenience_test_data")
    
    # Use convenience function for training data
    print("📚 Using get_training_data convenience function...")
    try:
        splits = get_training_data(
            symbols=['CONVENIENCE_TEST'],
            table_name="convenience_test_data",
            features=['open', 'high', 'low', 'close', 'volume'],
            target_column='returns'
        )
        
        print(f"✓ Got training data splits:")
        for name, data in splits.items():
            if not data.empty:
                print(f"  {name}: {len(data)} rows")
    
    except Exception as e:
        print(f"⚠ Convenience function had issues: {e}")
    
    # Use convenience function for feature matrix
    print("\n🧮 Using get_feature_matrix convenience function...")
    try:
        X, features, scaler = get_feature_matrix(
            symbols=['CONVENIENCE_TEST'],
            feature_columns=['open', 'high', 'low', 'close', 'volume'],
            table_name="convenience_test_data"
        )
        
        print(f"✓ Got feature matrix: {X.shape}")
        print(f"✓ Feature names: {features}")
        
    except Exception as e:
        print(f"⚠ Feature matrix convenience function had issues: {e}")


def main():
    """Run the comprehensive data storage demonstration."""
    print("🚀 Friday AI Trading System - Enhanced Data Storage Procedures Demo")
    print("=" * 80)
    
    try:
        # 1. Demonstrate default storage (LocalParquetStorage)
        default_storage = demonstrate_default_storage()
        
        # 2. Demonstrate storage factory
        factory = demonstrate_storage_factory()
        
        # 3. Demonstrate file rotation and metadata logging
        demonstrate_file_rotation_and_metadata()
        
        # 4. Demonstrate retrieval utilities
        demonstrate_retrieval_utilities()
        
        # 5. Demonstrate convenience functions
        demonstrate_convenience_functions()
        
        print("\n" + "="*80)
        print("🎉 All demonstrations completed successfully!")
        print("="*80)
        
        print("\n📋 Summary of Features Demonstrated:")
        print("✓ LocalParquetStorage as default with symbol/date partitioning")
        print("✓ DataStorage interface supporting multiple backends")
        print("✓ Automatic directory creation with proper permissions")
        print("✓ File rotation based on size/time/count strategies")
        print("✓ Comprehensive metadata logging and performance tracking")
        print("✓ Retrieval utilities optimized for model training")
        print("✓ Automatic train/validation/test data splits")
        print("✓ Feature matrix generation with normalization")
        print("✓ Time series data formatting for sequence models")
        print("✓ Batch processing for large datasets")
        print("✓ Caching for improved performance")
        print("✓ Convenience functions for common operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
