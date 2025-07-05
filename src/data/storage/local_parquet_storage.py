"""Local Parquet storage module for the Friday AI Trading System.

This module provides a LocalParquetStorage class that stores data in partitioned
Parquet files by symbol/date with metadata tracking.
"""

import os
import json
import hashlib
import shutil
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import threading
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.storage.data_storage import DataStorage, StorageError
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class LocalParquetStorage(DataStorage):
    """Local Parquet storage with partitioning by symbol/date and metadata tracking.
    
    This class provides methods for storing and retrieving data from partitioned
    Parquet files, with support for append/overwrite modes and metadata recording.
    """
    
    def __init__(self, config=None, base_dir=None):
        """Initialize LocalParquetStorage.
        
        Args:
            config: Configuration manager.
            base_dir: Base directory for partitioned Parquet files.
            
        Raises:
            StorageError: If the base directory cannot be created.
        """
        super().__init__(config)
        
        try:
            # Get configuration settings
            if self.config is not None:
                storage_config = self.config.get("data.storage", {})
                parquet_config = storage_config.get("local_parquet", {})
                base_dir = base_dir or parquet_config.get("base_dir")
                
                # Load configuration settings
                self.auto_create_dirs = storage_config.get("auto_directory_creation", {}).get("enabled", True)
                self.dir_permissions = storage_config.get("auto_directory_creation", {}).get("permissions", "755")
                self.create_parents = storage_config.get("auto_directory_creation", {}).get("create_parents", True)
                
                # File rotation settings
                rotation_config = storage_config.get("file_rotation", {})
                self.file_rotation_enabled = rotation_config.get("enabled", True)
                self.rotation_strategy = rotation_config.get("strategy", "size_based")
                self.max_file_size_mb = rotation_config.get("max_file_size_mb", 100)
                self.max_age_days = rotation_config.get("max_age_days", 30)
                self.max_files = rotation_config.get("max_files", 1000)
                self.compress_old_files = rotation_config.get("compress_old_files", True)
                
                # Metadata logging settings
                metadata_config = storage_config.get("metadata_logging", {})
                self.metadata_logging_enabled = metadata_config.get("enabled", True)
                self.log_operations = metadata_config.get("log_operations", True)
                self.log_performance = metadata_config.get("log_performance", True)
                self.log_errors = metadata_config.get("log_errors", True)
                self.metadata_retention_days = metadata_config.get("metadata_retention_days", 90)
                self.detailed_logging = metadata_config.get("detailed_logging", False)
                
                # Partitioning settings
                self.partition_by = parquet_config.get("partition_by", ["symbol", "date"])
                self.validate_partitioning = parquet_config.get("validate_partitioning", True)
                self.metadata_enabled = parquet_config.get("metadata_enabled", True)
            else:
                # Default settings
                self.auto_create_dirs = True
                self.dir_permissions = "755"
                self.create_parents = True
                self.file_rotation_enabled = True
                self.rotation_strategy = "size_based"
                self.max_file_size_mb = 100
                self.max_age_days = 30
                self.max_files = 1000
                self.compress_old_files = True
                self.metadata_logging_enabled = True
                self.log_operations = True
                self.log_performance = True
                self.log_errors = True
                self.metadata_retention_days = 90
                self.detailed_logging = False
                self.partition_by = ["symbol", "date"]
                self.validate_partitioning = True
                self.metadata_enabled = True
            
            # Use current directory if base_dir is None
            self.base_dir = Path(base_dir or "data/market_data")
            
            # Create base directory if it doesn't exist
            self._create_directory(self.base_dir)
            
            # Initialize metadata directory
            self.metadata_dir = self.base_dir / ".metadata"
            self._create_directory(self.metadata_dir)
            
            # Initialize thread lock for thread safety
            self._lock = threading.RLock()
            
            # Initialize operation statistics
            self.operation_stats = defaultdict(lambda: {
                'count': 0,
                'total_time': 0.0,
                'last_operation': None,
                'errors': 0
            })
            
            logger.info(f"Initialized LocalParquetStorage with base directory: {self.base_dir}")
            
            # Log configuration summary
            if self.detailed_logging:
                config_summary = {
                    'base_dir': str(self.base_dir),
                    'auto_create_dirs': self.auto_create_dirs,
                    'file_rotation_enabled': self.file_rotation_enabled,
                    'metadata_logging_enabled': self.metadata_logging_enabled,
                    'partition_by': self.partition_by
                }
                logger.debug(f"LocalParquetStorage configuration: {config_summary}")
        
        except Exception as e:
            logger.error(f"Error initializing LocalParquetStorage: {str(e)}")
            raise StorageError(f"Error initializing LocalParquetStorage: {str(e)}")
    
    def connect(self) -> None:
        """Connect to the storage backend (no-op for file storage)."""
        pass
    
    def disconnect(self) -> None:
        """Disconnect from the storage backend (no-op for file storage)."""
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to the storage backend."""
        return True
    
    def _create_directory(self, path: Path) -> None:
        """Create directory with proper permissions if auto_create_dirs is enabled.
        
        Args:
            path: Path to create.
        """
        if not self.auto_create_dirs:
            return
            
        try:
            path.mkdir(parents=self.create_parents, exist_ok=True)
            
            # Set permissions if on Unix-like system
            if hasattr(os, 'chmod') and self.dir_permissions:
                try:
                    os.chmod(path, int(self.dir_permissions, 8))
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not set permissions {self.dir_permissions} on {path}: {e}")
                    
            logger.debug(f"Created directory: {path}")
            
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise StorageError(f"Failed to create directory {path}: {e}")
    
    def _should_rotate_file(self, file_path: Path) -> bool:
        """Check if a file should be rotated based on rotation strategy.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            bool: True if file should be rotated.
        """
        if not self.file_rotation_enabled or not file_path.exists():
            return False
            
        try:
            if self.rotation_strategy == "size_based":
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                return file_size_mb >= self.max_file_size_mb
                
            elif self.rotation_strategy == "time_based":
                file_age_days = (datetime.now().timestamp() - file_path.stat().st_mtime) / (24 * 3600)
                return file_age_days >= self.max_age_days
                
            elif self.rotation_strategy == "count_based":
                partition_dir = file_path.parent
                file_count = len(list(partition_dir.glob("*.parquet")))
                return file_count >= self.max_files
                
        except Exception as e:
            logger.warning(f"Error checking file rotation for {file_path}: {e}")
            
        return False
    
    def _rotate_file(self, file_path: Path) -> None:
        """Rotate a file by moving it to archived location and optionally compressing.
        
        Args:
            file_path: Path to the file to rotate.
        """
        try:
            archive_dir = file_path.parent / "archived"
            self._create_directory(archive_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            archived_path = archive_dir / archived_name
            
            # Move the file
            shutil.move(str(file_path), str(archived_path))
            
            # Compress if enabled
            if self.compress_old_files:
                compressed_path = archived_path.with_suffix(archived_path.suffix + ".gz")
                
                with open(archived_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                archived_path.unlink()
                archived_path = compressed_path
            
            # Also move metadata file if it exists
            metadata_path = file_path.with_suffix('.json')
            if metadata_path.exists():
                archived_metadata = archive_dir / f"{file_path.stem}_{timestamp}.json"
                if self.compress_old_files:
                    archived_metadata = archived_metadata.with_suffix(".json.gz")
                    with open(metadata_path, 'rb') as f_in:
                        with gzip.open(archived_metadata, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.move(str(metadata_path), str(archived_metadata))
                
                metadata_path.unlink(missing_ok=True)
            
            logger.info(f"Rotated file {file_path} to {archived_path}")
            
        except Exception as e:
            logger.error(f"Failed to rotate file {file_path}: {e}")
            # Don't raise exception as this shouldn't prevent data storage
    
    def _log_operation_metadata(self, operation: str, **kwargs) -> None:
        """Log detailed operation metadata.
        
        Args:
            operation: Operation name.
            **kwargs: Additional metadata to log.
        """
        if not self.metadata_logging_enabled:
            return
            
        try:
            with self._lock:
                timestamp = datetime.now()
                
                # Update operation statistics
                if self.log_performance:
                    start_time = kwargs.get('start_time')
                    end_time = kwargs.get('end_time', timestamp)
                    
                    if start_time:
                        operation_time = (end_time - start_time).total_seconds()
                        self.operation_stats[operation]['total_time'] += operation_time
                        kwargs['operation_time'] = operation_time
                        kwargs['avg_operation_time'] = (
                            self.operation_stats[operation]['total_time'] / 
                            max(1, self.operation_stats[operation]['count'])
                        )
                
                self.operation_stats[operation]['count'] += 1
                self.operation_stats[operation]['last_operation'] = timestamp
                
                if kwargs.get('error'):
                    self.operation_stats[operation]['errors'] += 1
                
                # Log operation metadata
                if self.log_operations:
                    metadata_entry = {
                        'timestamp': timestamp.isoformat(),
                        'operation': operation,
                        'metadata': kwargs,
                        'stats': dict(self.operation_stats[operation]) if self.log_performance else None
                    }
                    
                    # Write to metadata log file
                    metadata_file = self.metadata_dir / f"operations_{timestamp.strftime('%Y%m')}.jsonl"
                    
                    with open(metadata_file, 'a') as f:
                        f.write(json.dumps(metadata_entry, default=str) + '\n')
                    
                    if self.detailed_logging:
                        logger.debug(f"Logged operation metadata: {operation}")
                        
        except Exception as e:
            if self.log_errors:
                logger.error(f"Failed to log operation metadata: {e}")
    
    def _cleanup_old_metadata(self) -> None:
        """Clean up old metadata files based on retention policy."""
        if not self.metadata_logging_enabled or not self.metadata_dir.exists():
            return
            
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=self.metadata_retention_days)
            
            for metadata_file in self.metadata_dir.glob("operations_*.jsonl"):
                try:
                    file_date = datetime.strptime(metadata_file.stem.split('_')[1], '%Y%m')
                    if file_date < cutoff_date:
                        metadata_file.unlink()
                        logger.debug(f"Cleaned up old metadata file: {metadata_file}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse metadata file date {metadata_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old metadata: {e}")
    
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate MD5 hash of the DataFrame.
        
        Args:
            data: DataFrame to hash.
            
        Returns:
            MD5 hash string.
        """
        return hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()
    
    def _save_metadata(self, file_path: Path, data: pd.DataFrame) -> None:
        """Save metadata alongside the parquet file.
        
        Args:
            file_path: Path to the parquet file.
            data: DataFrame that was stored.
        """
        metadata = {
            'row_count': len(data),
            'hash': self._calculate_data_hash(data),
            'columns': list(data.columns),
            'created_at': datetime.now().isoformat(),
            'symbol': data['symbol'].iloc[0] if 'symbol' in data.columns else None,
            'date_range': {
                'start': str(data.index.min()) if hasattr(data.index, 'min') else None,
                'end': str(data.index.max()) if hasattr(data.index, 'max') else None
            }
        }
        
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved metadata to {metadata_path}")
    
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata for a parquet file.
        
        Args:
            file_path: Path to the parquet file.
            
        Returns:
            Metadata dictionary or None if not found.
        """
        metadata_path = file_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def store_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = True,
        **kwargs
    ) -> bool:
        """Store data in partitioned Parquet files.
        
        Args:
            data: DataFrame to store.
            table_name: Name of the table/file.
            if_exists: What to do if data exists ('append', 'replace', 'fail').
            index: Whether to store the index.
            **kwargs: Additional arguments for to_parquet.
            
        Returns:
            True if successful.
            
        Raises:
            StorageError: If storage fails.
        """
        start_time = datetime.now()
        operation_metadata = {
            'table_name': table_name,
            'if_exists': if_exists,
            'index': index,
            'row_count': len(data) if not data.empty else 0,
            'start_time': start_time
        }
        
        try:
            if not isinstance(data, pd.DataFrame):
                raise StorageError("Data must be a pandas DataFrame")
            
            if data.empty:
                raise StorageError("Cannot store empty DataFrame")
            
            # Validate partitioning columns if enabled
            if self.validate_partitioning:
                required_cols = self.partition_by.copy()
                if 'symbol' in required_cols and 'symbol' not in data.columns:
                    raise StorageError("DataFrame must contain 'symbol' column for partitioning")
                
                # Find date column
                date_col = None
                if 'date' in required_cols:
                    for col in ['date', 'timestamp', 'datetime']:
                        if col in data.columns:
                            date_col = col
                            break
                    
                    if date_col is None:
                        raise StorageError("DataFrame must contain a date column ('date', 'timestamp', or 'datetime')")
            else:
                # Fallback for backward compatibility
                if 'symbol' not in data.columns:
                    raise StorageError("DataFrame must contain 'symbol' column for partitioning")
                
                date_col = None
                for col in ['date', 'timestamp', 'datetime']:
                    if col in data.columns:
                        date_col = col
                        break
                
                if date_col is None:
                    raise StorageError("DataFrame must contain a date column ('date', 'timestamp', or 'datetime')")
            
            # Group data by symbol and date for partitioning
            # Convert date to string for grouping
            data_copy = data.copy()
            
            # Ensure timestamp columns are properly formatted for Parquet
            for col in data_copy.columns:
                if col in ['timestamp', 'datetime', 'date'] or pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                    data_copy[col] = pd.to_datetime(data_copy[col])
                    # Remove timezone info if present for PyArrow compatibility
                    if hasattr(data_copy[col].dtype, 'tz') and data_copy[col].dtype.tz is not None:
                        data_copy[col] = data_copy[col].dt.tz_localize(None)
            
            if pd.api.types.is_datetime64_any_dtype(data_copy[date_col]):
                data_copy['partition_date'] = data_copy[date_col].dt.strftime('%Y-%m-%d')
            else:
                data_copy['partition_date'] = pd.to_datetime(data_copy[date_col]).dt.strftime('%Y-%m-%d')
            
            # Group by symbol and partition_date
            grouped = data_copy.groupby(['symbol', 'partition_date'])
            
            all_successful = True
            partitions_created = 0
            total_files_rotated = 0
            
            for (symbol, date_str), group_data in grouped:
                # Remove the partition_date column before saving
                group_data = group_data.drop('partition_date', axis=1)
                
                # Create partition directory
                partition_dir = self.base_dir / symbol / date_str
                self._create_directory(partition_dir)
                file_path = partition_dir / f"{table_name}.parquet"
                
                # Check if file rotation is needed before storing
                if self._should_rotate_file(file_path):
                    self._rotate_file(file_path)
                    total_files_rotated += 1
                
                # Handle different modes
                if if_exists == "fail" and file_path.exists():
                    raise StorageError(f"File {file_path} already exists and if_exists='fail'")
                
                elif if_exists == "replace":
                    # Remove existing file if it exists
                    if file_path.exists():
                        file_path.unlink()
                        # Also remove metadata file
                        metadata_path = file_path.with_suffix('.json')
                        if metadata_path.exists():
                            metadata_path.unlink()
                
                elif if_exists == "append" and file_path.exists():
                    # Load existing data and append
                    existing_data = pd.read_parquet(file_path)
                    group_data = pd.concat([existing_data, group_data], ignore_index=True)
                
                # Store the data
                group_data.to_parquet(file_path, index=index, **kwargs)
                
                # Save metadata if enabled
                if self.metadata_enabled:
                    self._save_metadata(file_path, group_data)
                
                partitions_created += 1
                logger.debug(f"Stored {len(group_data)} rows to {file_path}")
            
            # Update operation metadata
            end_time = datetime.now()
            operation_metadata.update({
                'partitions_created': partitions_created,
                'files_rotated': total_files_rotated,
                'end_time': end_time,
                'success': True
            })
            
            # Record operation metadata
            self._record_operation_metadata(
                "store_data",
                table_name=table_name,
                row_count=len(data),
                if_exists=if_exists,
                partitions_created=partitions_created
            )
            
            # Log enhanced operation metadata
            self._log_operation_metadata("store_data", **operation_metadata)
            
            # Cleanup old metadata periodically
            if partitions_created > 0:
                self._cleanup_old_metadata()
            
            logger.info(f"Stored {len(data)} rows across {partitions_created} partitions (rotated {total_files_rotated} files)")
            return all_successful
            
        except Exception as e:
            # Log error metadata
            end_time = datetime.now()
            operation_metadata.update({
                'end_time': end_time,
                'error': str(e),
                'success': False
            })
            self._log_operation_metadata("store_data", **operation_metadata)
            
            self._handle_error("store_data", e)
    
    def retrieve_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        condition: Optional[str] = None,
        limit: Optional[int] = None,
        symbol: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data from partitioned Parquet files.
        
        Args:
            table_name: Name of the table/file.
            columns: List of columns to retrieve.
            condition: Filter condition (not implemented for file storage).
            limit: Maximum number of rows to retrieve.
            symbol: Specific symbol to retrieve (for partition filtering).
            date: Specific date to retrieve (for partition filtering).
            **kwargs: Additional arguments for read_parquet.
            
        Returns:
            Retrieved DataFrame.
            
        Raises:
            StorageError: If retrieval fails.
        """
        try:
            if symbol and date:
                # Direct partition access
                partition_dir = self.base_dir / symbol / date
                file_path = partition_dir / f"{table_name}.parquet"
                
                if not file_path.exists():
                    logger.warning(f"File {file_path} does not exist")
                    return pd.DataFrame()
                
                data = pd.read_parquet(file_path, columns=columns, **kwargs)
            
            else:
                # Search across all partitions
                data_frames = []
                
                for symbol_dir in self.base_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for date_dir in symbol_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        file_path = date_dir / f"{table_name}.parquet"
                        if file_path.exists():
                            df = pd.read_parquet(file_path, columns=columns, **kwargs)
                            data_frames.append(df)
                
                if not data_frames:
                    logger.warning(f"No data found for table {table_name}")
                    return pd.DataFrame()
                
                data = pd.concat(data_frames, ignore_index=True)
            
            # Apply limit if specified
            if limit:
                data = data.head(limit)
            
            logger.info(f"Retrieved {len(data)} rows from {table_name}")
            return data
            
        except Exception as e:
            self._handle_error("retrieve_data", e)
    
    def delete_data(
        self,
        table_name: str,
        condition: Optional[str] = None,
        symbol: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete data from partitioned Parquet files.
        
        Args:
            table_name: Name of the table/file.
            condition: Filter condition (not implemented).
            symbol: Specific symbol partition to delete.
            date: Specific date partition to delete.
            **kwargs: Additional arguments.
            
        Returns:
            True if successful.
            
        Raises:
            StorageError: If deletion fails.
        """
        try:
            deleted = False
            
            if symbol and date:
                # Delete specific partition
                partition_dir = self.base_dir / symbol / date
                file_path = partition_dir / f"{table_name}.parquet"
                metadata_path = file_path.with_suffix('.json')
                
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
                
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove empty directories
                if partition_dir.exists() and not any(partition_dir.iterdir()):
                    partition_dir.rmdir()
                
                symbol_dir = self.base_dir / symbol
                if symbol_dir.exists() and not any(symbol_dir.iterdir()):
                    symbol_dir.rmdir()
            
            else:
                # Delete all partitions for the table
                for symbol_dir in self.base_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for date_dir in symbol_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        file_path = date_dir / f"{table_name}.parquet"
                        metadata_path = file_path.with_suffix('.json')
                        
                        if file_path.exists():
                            file_path.unlink()
                            deleted = True
                        
                        if metadata_path.exists():
                            metadata_path.unlink()
            
            logger.info(f"Deleted data for table {table_name}")
            return deleted
            
        except Exception as e:
            self._handle_error("delete_data", e)
    
    def table_exists(self, table_name: str, symbol: Optional[str] = None, date: Optional[str] = None) -> bool:
        """Check if a table exists in any partition.
        
        Args:
            table_name: Name of the table/file.
            symbol: Specific symbol to check.
            date: Specific date to check.
            
        Returns:
            True if table exists.
        """
        try:
            if symbol and date:
                partition_dir = self.base_dir / symbol / date
                file_path = partition_dir / f"{table_name}.parquet"
                return file_path.exists()
            
            else:
                # Check across all partitions
                for symbol_dir in self.base_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for date_dir in symbol_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        file_path = date_dir / f"{table_name}.parquet"
                        if file_path.exists():
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False
    
    def list_tables(self, symbol: Optional[str] = None, date: Optional[str] = None) -> List[str]:
        """List all tables in the storage.
        
        Args:
            symbol: Filter by specific symbol.
            date: Filter by specific date.
            
        Returns:
            List of table names.
            
        Raises:
            StorageError: If listing fails.
        """
        try:
            tables = set()
            
            if symbol and date:
                partition_dir = self.base_dir / symbol / date
                if partition_dir.exists():
                    for file_path in partition_dir.glob("*.parquet"):
                        tables.add(file_path.stem)
            
            else:
                for symbol_dir in self.base_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    if symbol and symbol_dir.name != symbol:
                        continue
                    
                    for date_dir in symbol_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        if date and date_dir.name != date:
                            continue
                        
                        for file_path in date_dir.glob("*.parquet"):
                            tables.add(file_path.stem)
            
            return sorted(list(tables))
            
        except Exception as e:
            self._handle_error("list_tables", e)
    
    def get_table_info(
        self, 
        table_name: str, 
        symbol: Optional[str] = None, 
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get information about a table.
        
        Args:
            table_name: Name of the table.
            symbol: Specific symbol partition.
            date: Specific date partition.
            
        Returns:
            Dictionary with table information.
            
        Raises:
            StorageError: If getting information fails.
        """
        try:
            if symbol and date:
                partition_dir = self.base_dir / symbol / date
                file_path = partition_dir / f"{table_name}.parquet"
                
                if not file_path.exists():
                    raise StorageError(f"Table {table_name} not found in partition {symbol}/{date}")
                
                # Get parquet metadata
                parquet_file = pq.ParquetFile(file_path)
                parquet_info = {
                    "num_rows": parquet_file.metadata.num_rows,
                    "num_columns": len(parquet_file.schema.names),
                    "columns": parquet_file.schema.names,
                    "file_size": file_path.stat().st_size,
                    "file_path": str(file_path)
                }
                
                # Get custom metadata
                custom_metadata = self._load_metadata(file_path)
                if custom_metadata:
                    parquet_info.update(custom_metadata)
                
                return parquet_info
            
            else:
                # Aggregate info across all partitions
                total_rows = 0
                total_files = 0
                total_size = 0
                all_columns = set()
                
                for symbol_dir in self.base_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for date_dir in symbol_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        
                        file_path = date_dir / f"{table_name}.parquet"
                        if file_path.exists():
                            parquet_file = pq.ParquetFile(file_path)
                            total_rows += parquet_file.metadata.num_rows
                            total_files += 1
                            total_size += file_path.stat().st_size
                            all_columns.update(parquet_file.schema.names)
                
                return {
                    "total_rows": total_rows,
                    "total_files": total_files,
                    "total_size": total_size,
                    "columns": sorted(list(all_columns)),
                    "partitioned": True
                }
            
        except Exception as e:
            self._handle_error("get_table_info", e)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute a custom query (not implemented for file storage).
        
        Args:
            query: Query string.
            params: Query parameters.
            **kwargs: Additional arguments.
            
        Raises:
            StorageError: Always, as file storage doesn't support queries.
        """
        raise StorageError("Custom queries not supported for LocalParquetStorage")
    
    def get_partition_info(self) -> Dict[str, Any]:
        """Get information about all partitions.
        
        Returns:
            Dictionary with partition information.
        """
        try:
            partitions = {}
            
            for symbol_dir in self.base_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                
                symbol = symbol_dir.name
                partitions[symbol] = {}
                
                for date_dir in symbol_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    date = date_dir.name
                    files = list(date_dir.glob("*.parquet"))
                    
                    partitions[symbol][date] = {
                        "file_count": len(files),
                        "tables": [f.stem for f in files]
                    }
            
            return partitions
            
        except Exception as e:
            logger.error(f"Error getting partition info: {str(e)}")
            return {}
