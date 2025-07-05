"""Example usage of LocalParquetStorage for the Friday AI Trading System.

This example demonstrates how to use the LocalParquetStorage class to store
and retrieve market data with automatic partitioning by symbol and date.
"""

import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.storage.local_parquet_storage import LocalParquetStorage


def main():
    """Demonstrate LocalParquetStorage functionality."""
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize LocalParquetStorage
        storage = LocalParquetStorage(base_dir=temp_dir)
        print("✓ LocalParquetStorage initialized")
        
        # Create sample market data with multiple symbols and dates
        market_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
            'date': pd.to_datetime([
                '2024-01-01', '2024-01-02', 
                '2024-01-01', '2024-01-02',
                '2024-01-01', '2024-01-02'
            ]),
            'open': [150.0, 152.0, 300.0, 302.0, 2800.0, 2820.0],
            'high': [155.0, 157.0, 305.0, 307.0, 2850.0, 2870.0],
            'low': [149.0, 151.0, 299.0, 301.0, 2790.0, 2810.0],
            'close': [154.0, 156.0, 304.0, 306.0, 2840.0, 2860.0],
            'volume': [1000000, 1200000, 800000, 850000, 500000, 520000]
        })
        
        print(f"Sample data shape: {market_data.shape}")
        print(f"Symbols: {market_data['symbol'].unique()}")
        print(f"Date range: {market_data['date'].min()} to {market_data['date'].max()}")
        
        # Store data (will be automatically partitioned by symbol/date)
        print("\n📁 Storing market data...")
        success = storage.store_data(market_data, "market_data")
        print(f"✓ Data stored successfully: {success}")
        
        # Show partition structure
        print("\n🗂️  Partition structure:")
        partition_info = storage.get_partition_info()
        for symbol, dates in partition_info.items():
            print(f"  {symbol}:")
            for date, info in dates.items():
                print(f"    {date}: {info['file_count']} files, tables: {info['tables']}")
        
        # Retrieve all data
        print("\n📖 Retrieving all data...")
        all_data = storage.retrieve_data("market_data")
        print(f"Retrieved {len(all_data)} rows")
        
        # Retrieve data for specific symbol and date
        print("\n🔍 Retrieving AAPL data for 2024-01-01...")
        aapl_data = storage.retrieve_data("market_data", symbol="AAPL", date="2024-01-01")
        print(f"Retrieved {len(aapl_data)} rows for AAPL on 2024-01-01")
        print(aapl_data)
        
        # Test append mode
        print("\n➕ Testing append mode...")
        additional_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [pd.to_datetime('2024-01-01')],  # Same date as existing data
            'open': [158.0],
            'high': [162.0],
            'low': [157.0],
            'close': [161.0],
            'volume': [1500000]
        })
        
        storage.store_data(additional_data, "market_data", if_exists="append")
        print("✓ Additional data appended")
        
        # Check updated data
        aapl_updated = storage.retrieve_data("market_data", symbol="AAPL", date="2024-01-01")
        print(f"Updated AAPL data for 2024-01-01: {len(aapl_updated)} rows")
        
        # Test replace mode
        print("\n🔄 Testing replace mode...")
        replacement_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [pd.to_datetime('2024-01-01')],
            'open': [160.0],
            'high': [165.0],
            'low': [159.0],
            'close': [164.0],
            'volume': [2000000]
        })
        
        storage.store_data(replacement_data, "market_data", if_exists="replace")
        print("✓ Data replaced")
        
        # Check replaced data
        aapl_replaced = storage.retrieve_data("market_data", symbol="AAPL", date="2024-01-01")
        print(f"Replaced AAPL data for 2024-01-01: {len(aapl_replaced)} rows")
        print(aapl_replaced)
        
        # Get table information
        print("\n📊 Table information:")
        table_info = storage.get_table_info("market_data")
        print(f"Total rows: {table_info['total_rows']}")
        print(f"Total files: {table_info['total_files']}")
        print(f"Columns: {table_info['columns']}")
        
        # Get specific partition info
        partition_info = storage.get_table_info("market_data", symbol="AAPL", date="2024-01-01")
        print(f"\nAAPL 2024-01-01 partition:")
        print(f"  Rows: {partition_info['num_rows']}")
        print(f"  Row count (metadata): {partition_info['row_count']}")
        print(f"  Hash: {partition_info['hash'][:16]}...")
        
        # Test data integrity
        print("\n🔒 Testing data integrity...")
        retrieved_again = storage.retrieve_data("market_data", symbol="AAPL", date="2024-01-01")
        original_hash = storage._calculate_data_hash(replacement_data)
        retrieved_hash = storage._calculate_data_hash(retrieved_again)
        
        print(f"Original hash:  {original_hash}")
        print(f"Retrieved hash: {retrieved_hash}")
        print(f"✓ Data integrity verified: {original_hash == retrieved_hash}")
        
        # List all tables
        print(f"\n📋 Available tables: {storage.list_tables()}")
        
        # Check if table exists
        print(f"Table 'market_data' exists: {storage.table_exists('market_data')}")
        print(f"Table 'nonexistent' exists: {storage.table_exists('nonexistent')}")
        
        print(f"\n🎉 LocalParquetStorage example completed successfully!")


if __name__ == "__main__":
    main()
