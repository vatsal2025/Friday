"""Integration tests for LocalParquetStorage.

This module contains tests to verify the LocalParquetStorage implementation
including partitioning, metadata tracking, and read/write cycle integrity.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
import json

from src.data.storage.local_parquet_storage import LocalParquetStorage, StorageError


class TestLocalParquetStorage:
    """Test class for LocalParquetStorage."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create a LocalParquetStorage instance for testing."""
        return LocalParquetStorage(base_dir=temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        return pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [150.0, 152.0, 151.0, 153.0, 154.0],
            'high': [155.0, 157.0, 156.0, 158.0, 159.0],
            'low': [149.0, 151.0, 150.0, 152.0, 153.0],
            'close': [154.0, 153.0, 155.0, 157.0, 158.0],
            'volume': [1000000, 1200000, 1100000, 1300000, 1400000]
        })
    
    @pytest.fixture
    def multi_symbol_data(self):
        """Create sample data with multiple symbols and dates."""
        data = []
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range('2024-01-01', periods=3)
        
        for symbol in symbols:
            for date_val in dates:
                data.append({
                    'symbol': symbol,
                    'date': date_val,
                    'open': 150.0,
                    'high': 155.0,
                    'low': 149.0,
                    'close': 154.0,
                    'volume': 1000000
                })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, temp_dir):
        """Test LocalParquetStorage initialization."""
        storage = LocalParquetStorage(base_dir=temp_dir)
        
        assert storage.base_dir == Path(temp_dir)
        assert storage.is_connected()
        assert storage.base_dir.exists()
    
    def test_store_and_retrieve_data(self, storage, sample_data):
        """Test basic store and retrieve functionality."""
        table_name = "market_data"
        
        # Store data
        result = storage.store_data(sample_data, table_name)
        assert result is True
        
        # Retrieve data
        retrieved_data = storage.retrieve_data(table_name)
        
        # Verify data integrity
        assert len(retrieved_data) == len(sample_data)
        assert list(retrieved_data.columns) == list(sample_data.columns)
        
        # Check specific values
        pd.testing.assert_frame_equal(
            retrieved_data.reset_index(drop=True),
            sample_data.reset_index(drop=True),
            check_dtype=False
        )
    
    def test_partitioning_by_symbol_and_date(self, storage, temp_dir):
        """Test that data is correctly partitioned by symbol and date."""
        # Create data for different symbols and dates
        data1 = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [pd.Timestamp('2024-01-01')],
            'price': [150.0]
        })
        
        data2 = pd.DataFrame({
            'symbol': ['MSFT'],
            'date': [pd.Timestamp('2024-01-02')],
            'price': [200.0]
        })
        
        # Store data
        storage.store_data(data1, "test_table")
        storage.store_data(data2, "test_table")
        
        # Check partition structure
        base_path = Path(temp_dir)
        
        # AAPL partition
        aapl_path = base_path / "AAPL" / "2024-01-01" / "test_table.parquet"
        assert aapl_path.exists()
        
        # MSFT partition
        msft_path = base_path / "MSFT" / "2024-01-02" / "test_table.parquet"
        assert msft_path.exists()
        
        # Verify metadata files exist
        assert aapl_path.with_suffix('.json').exists()
        assert msft_path.with_suffix('.json').exists()
    
    def test_metadata_recording(self, storage, sample_data):
        """Test that metadata is correctly recorded."""
        table_name = "market_data"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Check metadata file exists for the first partition
        symbol = sample_data['symbol'].iloc[0]
        date_str = sample_data['date'].iloc[0].strftime('%Y-%m-%d')
        
        metadata_path = storage.base_dir / symbol / date_str / f"{table_name}.json"
        assert metadata_path.exists()
        
        # Load and verify metadata for the first partition
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # First partition should only have 1 row (first date)
        assert metadata['row_count'] == 1
        assert 'hash' in metadata
        assert metadata['columns'] == list(sample_data.columns)
        assert metadata['symbol'] == symbol
        assert 'created_at' in metadata
    
    def test_append_mode(self, storage, sample_data):
        """Test append mode functionality."""
        table_name = "market_data"
        
        # Store initial data
        storage.store_data(sample_data, table_name, if_exists="append")
        
        # Create additional data for the same symbol/date
        additional_data = pd.DataFrame({
            'symbol': ['AAPL'] * 2,
            'date': [sample_data['date'].iloc[0]] * 2,  # Same date as first row
            'open': [160.0, 161.0],
            'high': [165.0, 166.0],
            'low': [159.0, 160.0],
            'close': [164.0, 165.0],
            'volume': [1500000, 1600000]
        })
        
        # Append additional data
        storage.store_data(additional_data, table_name, if_exists="append")
        
        # Retrieve all data
        retrieved_data = storage.retrieve_data(table_name)
        
        # Should have original + additional data
        expected_length = len(sample_data) + len(additional_data)
        assert len(retrieved_data) == expected_length
    
    def test_replace_mode(self, storage, sample_data):
        """Test replace mode functionality."""
        table_name = "market_data"
        
        # Store initial data
        storage.store_data(sample_data, table_name, if_exists="append")
        
        # Create replacement data for the same date as the first row
        replacement_data = pd.DataFrame({
            'symbol': ['AAPL'] * 2,
            'date': [sample_data['date'].iloc[0]] * 2,  # Same date as first row
            'open': [170.0, 171.0],
            'high': [175.0, 176.0],
            'low': [169.0, 170.0],
            'close': [174.0, 175.0],
            'volume': [2000000, 2100000]
        })
        
        # Replace data (only affects the specific partition)
        storage.store_data(replacement_data, table_name, if_exists="replace")
        
        # Retrieve data for the specific partition
        symbol = sample_data['symbol'].iloc[0]
        date_str = sample_data['date'].iloc[0].strftime('%Y-%m-%d')
        partition_data = storage.retrieve_data(table_name, symbol=symbol, date=date_str)
        
        # Should only have replacement data for this partition
        assert len(partition_data) == len(replacement_data)
        pd.testing.assert_frame_equal(
            partition_data.reset_index(drop=True),
            replacement_data.reset_index(drop=True),
            check_dtype=False
        )
        
        # Other partitions should still exist
        all_data = storage.retrieve_data(table_name)
        # Should have replacement data (2 rows) + other partitions (4 rows)
        assert len(all_data) == len(replacement_data) + (len(sample_data) - 1)
    
    def test_fail_mode(self, storage, sample_data):
        """Test fail mode functionality."""
        table_name = "market_data"
        
        # Store initial data
        storage.store_data(sample_data, table_name, if_exists="append")
        
        # Try to store again with fail mode
        with pytest.raises(StorageError, match="already exists"):
            storage.store_data(sample_data, table_name, if_exists="fail")
    
    def test_retrieve_with_partition_filters(self, storage, multi_symbol_data):
        """Test retrieving data with symbol and date filters."""
        table_name = "multi_data"
        
        # Store multi-symbol data
        storage.store_data(multi_symbol_data, table_name)
        
        # Retrieve specific symbol and date
        filtered_data = storage.retrieve_data(
            table_name,
            symbol="AAPL",
            date="2024-01-01"
        )
        
        # Should only have one row for AAPL on 2024-01-01
        assert len(filtered_data) == 1
        assert filtered_data['symbol'].iloc[0] == "AAPL"
        assert filtered_data['date'].iloc[0] == pd.Timestamp('2024-01-01')
    
    def test_table_exists(self, storage, sample_data):
        """Test table existence checking."""
        table_name = "market_data"
        
        # Initially should not exist
        assert not storage.table_exists(table_name)
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Now should exist
        assert storage.table_exists(table_name)
        
        # Test with specific partition
        symbol = sample_data['symbol'].iloc[0]
        date_str = sample_data['date'].iloc[0].strftime('%Y-%m-%d')
        assert storage.table_exists(table_name, symbol=symbol, date=date_str)
        
        # Test with non-existent partition
        assert not storage.table_exists(table_name, symbol="NONEXISTENT", date="2024-01-01")
    
    def test_list_tables(self, storage, multi_symbol_data):
        """Test listing tables."""
        # Initially empty
        assert storage.list_tables() == []
        
        # Store data
        storage.store_data(multi_symbol_data, "table1")
        storage.store_data(multi_symbol_data, "table2")
        
        # Should list both tables
        tables = storage.list_tables()
        assert "table1" in tables
        assert "table2" in tables
        
        # Test filtering by symbol
        aapl_tables = storage.list_tables(symbol="AAPL")
        assert "table1" in aapl_tables
        assert "table2" in aapl_tables
    
    def test_get_table_info(self, storage, sample_data):
        """Test getting table information."""
        table_name = "market_data"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Get info for specific partition (first date only has 1 row)
        symbol = sample_data['symbol'].iloc[0]
        date_str = sample_data['date'].iloc[0].strftime('%Y-%m-%d')
        
        info = storage.get_table_info(table_name, symbol=symbol, date=date_str)
        
        # Should only have 1 row for this specific date partition
        assert info['num_rows'] == 1
        # Parquet file has +1 column for the index
        assert info['num_columns'] == len(sample_data.columns) + 1
        assert 'row_count' in info  # From custom metadata
        assert 'hash' in info      # From custom metadata
        
        # Get aggregated info
        agg_info = storage.get_table_info(table_name)
        assert agg_info['total_rows'] == len(sample_data)
        assert agg_info['total_files'] == 5  # 5 dates = 5 partitions
        assert agg_info['partitioned'] is True
    
    def test_delete_data(self, storage, multi_symbol_data):
        """Test data deletion."""
        table_name = "test_data"
        
        # Store data
        storage.store_data(multi_symbol_data, table_name)
        
        # Delete specific partition
        result = storage.delete_data(table_name, symbol="AAPL", date="2024-01-01")
        assert result is True
        
        # Verify deletion
        assert not storage.table_exists(table_name, symbol="AAPL", date="2024-01-01")
        
        # Other partitions should still exist
        assert storage.table_exists(table_name, symbol="MSFT", date="2024-01-01")
    
    def test_data_integrity_hash_verification(self, storage, sample_data):
        """Test data integrity through hash verification."""
        table_name = "integrity_test"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Retrieve data
        retrieved_data = storage.retrieve_data(table_name)
        
        # Calculate hash of original and retrieved data
        original_hash = storage._calculate_data_hash(sample_data)
        retrieved_hash = storage._calculate_data_hash(retrieved_data)
        
        # Hashes should match, confirming data integrity
        assert original_hash == retrieved_hash
    
    def test_read_write_cycle_integrity(self, storage, sample_data):
        """Test complete read/write cycle integrity."""
        table_name = "cycle_test"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Retrieve data
        retrieved_data = storage.retrieve_data(table_name)
        
        # Store retrieved data as new table
        storage.store_data(retrieved_data, f"{table_name}_copy")
        
        # Retrieve copy
        copy_data = storage.retrieve_data(f"{table_name}_copy")
        
        # All three datasets should be identical
        pd.testing.assert_frame_equal(
            sample_data.reset_index(drop=True),
            retrieved_data.reset_index(drop=True),
            check_dtype=False
        )
        
        pd.testing.assert_frame_equal(
            retrieved_data.reset_index(drop=True),
            copy_data.reset_index(drop=True),
            check_dtype=False
        )
    
    def test_error_handling_missing_columns(self, storage):
        """Test error handling for missing required columns."""
        # Data without symbol column
        invalid_data = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'price': [150.0]
        })
        
        with pytest.raises(StorageError, match="must contain 'symbol' column"):
            storage.store_data(invalid_data, "test_table")
        
        # Data without date column
        invalid_data2 = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [150.0]
        })
        
        with pytest.raises(StorageError, match="must contain a date column"):
            storage.store_data(invalid_data2, "test_table")
    
    def test_error_handling_empty_dataframe(self, storage):
        """Test error handling for empty DataFrame."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(StorageError, match="Cannot store empty DataFrame"):
            storage.store_data(empty_data, "test_table")
    
    def test_error_handling_non_dataframe(self, storage):
        """Test error handling for non-DataFrame input."""
        invalid_data = [1, 2, 3]
        
        with pytest.raises(StorageError, match="Data must be a pandas DataFrame"):
            storage.store_data(invalid_data, "test_table")
    
    def test_get_partition_info(self, storage, multi_symbol_data):
        """Test getting partition information."""
        table_name = "partition_test"
        
        # Store data
        storage.store_data(multi_symbol_data, table_name)
        
        # Get partition info
        partition_info = storage.get_partition_info()
        
        # Should have info for each symbol
        assert "AAPL" in partition_info
        assert "MSFT" in partition_info
        assert "GOOGL" in partition_info
        
        # Each symbol should have dates
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            symbol_info = partition_info[symbol]
            assert "2024-01-01" in symbol_info
            assert "2024-01-02" in symbol_info
            assert "2024-01-03" in symbol_info
            
            # Each date should have file info
            for date_key in symbol_info:
                date_info = symbol_info[date_key]
                assert date_info["file_count"] == 1
                assert table_name in date_info["tables"]
    
    def test_column_filtering(self, storage, sample_data):
        """Test retrieving specific columns."""
        table_name = "column_test"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Retrieve only specific columns
        retrieved_data = storage.retrieve_data(table_name, columns=['symbol', 'close'])
        
        # Should only have specified columns
        assert list(retrieved_data.columns) == ['symbol', 'close']
        assert len(retrieved_data) == len(sample_data)
    
    def test_limit_functionality(self, storage, sample_data):
        """Test limit functionality in retrieval."""
        table_name = "limit_test"
        
        # Store data
        storage.store_data(sample_data, table_name)
        
        # Retrieve with limit
        retrieved_data = storage.retrieve_data(table_name, limit=3)
        
        # Should only have limited number of rows
        assert len(retrieved_data) == 3
        assert list(retrieved_data.columns) == list(sample_data.columns)


@pytest.mark.integration
class TestLocalParquetStorageIntegration:
    """Integration tests for LocalParquetStorage with real-world scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_dir):
        """Create a LocalParquetStorage instance for testing."""
        return LocalParquetStorage(base_dir=temp_dir)
    
    def test_large_dataset_performance(self, storage):
        """Test with a larger dataset to verify performance."""
        # Create larger dataset (1000 rows)
        large_data = pd.DataFrame({
            'symbol': ['AAPL'] * 1000,
            'date': pd.date_range('2024-01-01', periods=1000),
            'open': [150.0 + i * 0.1 for i in range(1000)],
            'high': [155.0 + i * 0.1 for i in range(1000)],
            'low': [149.0 + i * 0.1 for i in range(1000)],
            'close': [154.0 + i * 0.1 for i in range(1000)],
            'volume': [1000000 + i * 1000 for i in range(1000)]
        })
        
        table_name = "large_dataset"
        
        # Store data
        result = storage.store_data(large_data, table_name)
        assert result is True
        
        # Retrieve data
        retrieved_data = storage.retrieve_data(table_name)
        
        # Verify integrity
        assert len(retrieved_data) == 1000
        pd.testing.assert_frame_equal(
            large_data.reset_index(drop=True),
            retrieved_data.reset_index(drop=True),
            check_dtype=False
        )
    
    def test_concurrent_operations_simulation(self, storage):
        """Test simulation of concurrent operations."""
        # Simulate multiple tables being created
        tables = []
        
        for i in range(5):
            data = pd.DataFrame({
                'symbol': [f'STOCK{i}'] * 10,
                'date': pd.date_range('2024-01-01', periods=10),
                'price': [100.0 + i] * 10
            })
            
            table_name = f"table_{i}"
            storage.store_data(data, table_name)
            tables.append(table_name)
        
        # Verify all tables exist and contain correct data
        for i, table_name in enumerate(tables):
            assert storage.table_exists(table_name)
            data = storage.retrieve_data(table_name)
            assert len(data) == 10
            assert data['symbol'].iloc[0] == f'STOCK{i}'
    
    def test_metadata_consistency_across_operations(self, storage):
        """Test that metadata remains consistent across multiple operations."""
        table_name = "metadata_test"
        
        # Initial data with single date to ensure single partition
        initial_data = pd.DataFrame({
            'symbol': ['TEST'] * 5,
            'date': [pd.Timestamp('2024-01-01')] * 5,  # All same date
            'value': range(5)
        })
        
        # Store initial data
        storage.store_data(initial_data, table_name)
        
        # Get initial metadata
        symbol = initial_data['symbol'].iloc[0]
        date_str = initial_data['date'].iloc[0].strftime('%Y-%m-%d')
        metadata_path = storage.base_dir / symbol / date_str / f"{table_name}.json"
        
        with open(metadata_path, 'r') as f:
            initial_metadata = json.load(f)
        
        # Append more data to the same partition
        additional_data = pd.DataFrame({
            'symbol': ['TEST'] * 3,
            'date': [initial_data['date'].iloc[0]] * 3,  # Same date
            'value': range(5, 8)
        })
        
        storage.store_data(additional_data, table_name, if_exists="append")
        
        # Check updated metadata
        with open(metadata_path, 'r') as f:
            updated_metadata = json.load(f)
        
        # Row count should be updated (5 + 3 = 8)
        assert updated_metadata['row_count'] == 8
        assert updated_metadata['hash'] != initial_metadata['hash']  # Hash should change
        assert updated_metadata['created_at'] != initial_metadata['created_at']  # Timestamp should change
