"""Enhanced Model Backup and Recovery for Friday AI Trading System.

This module provides advanced functionality for creating and managing model backups,
including incremental backups, compression, verification, and automatic scheduling.
"""

import os
import json
import shutil
import datetime
import time
import threading
import hashlib
import zlib
import difflib
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path

from src.infrastructure.logging import get_logger
from src.services.model.model_backup import ModelBackupManager, ModelBackupError
from src.infrastructure.security.digital_signatures import ModelSigner

# Create logger
logger = get_logger(__name__)


class EnhancedModelBackupError(Exception):
    """Exception raised for errors in the enhanced model backup operations."""
    pass


class BackupVerificationError(Exception):
    """Exception raised when backup verification fails."""
    pass


class EnhancedModelBackupManager(ModelBackupManager):
    """Enhanced Model Backup Manager with advanced features.

    This class extends the base ModelBackupManager with additional features:
    - Incremental backups for large models
    - Backup compression to save storage space
    - Backup verification mechanism
    - Automatic backup scheduling

    Attributes:
        backup_dir: Directory where backups are stored.
        max_backups: Maximum number of backups to keep per model.
        compression_level: Compression level (0-9, None for no compression).
        verify_backups: Whether to verify backups after creation.
        incremental_threshold: File size threshold for incremental backups (bytes).
        keys_dir: Directory for storing signature keys.
    """

    def __init__(self, 
                 backup_dir: str, 
                 max_backups: int = 5,
                 compression_level: Optional[int] = 6,
                 verify_backups: bool = True,
                 incremental_threshold: int = 10 * 1024 * 1024,  # 10 MB
                 keys_dir: Optional[str] = None):
        """Initialize the enhanced model backup manager.

        Args:
            backup_dir: Directory where backups are stored.
            max_backups: Maximum number of backups to keep per model.
            compression_level: Compression level (0-9, None for no compression).
            verify_backups: Whether to verify backups after creation.
            incremental_threshold: File size threshold for incremental backups (bytes).
            keys_dir: Directory for storing signature keys.
        """
        super().__init__(backup_dir, max_backups)
        
        # Validate compression level
        if compression_level is not None and (compression_level < 0 or compression_level > 9):
            raise ValueError(f"Compression level must be between 0 and 9, got {compression_level}")
        
        self.compression_level = compression_level
        self.verify_backups = verify_backups
        self.incremental_threshold = incremental_threshold
        self.model_signer = ModelSigner(keys_dir)
        self.scheduled_backups = {}
        
        logger.info(f"Initialized EnhancedModelBackupManager with compression level: {compression_level}")
        logger.info(f"Backup verification: {verify_backups}")
        logger.info(f"Incremental backup threshold: {incremental_threshold} bytes")

    def create_backup(self, model_path: str, model_name: str, version: str) -> str:
        """Create a backup of a model, using incremental backup for large models.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            str: Path to the backup file.

        Raises:
            EnhancedModelBackupError: If the backup operation fails.
        """
        try:
            # Ensure model file exists
            if not os.path.exists(model_path):
                raise EnhancedModelBackupError(f"Model file not found: {model_path}")
            
            # Create model backup directory if it doesn't exist
            model_backup_dir = os.path.join(self.backup_dir, model_name)
            os.makedirs(model_backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{model_name}_v{version}_{timestamp}.backup"
            backup_path = os.path.join(model_backup_dir, backup_filename)
            
            # Check if we should use incremental backup
            file_size = os.path.getsize(model_path)
            previous_backups = self.list_backups(model_name)
            
            if file_size > self.incremental_threshold and previous_backups:
                # Use incremental backup
                backup_path = self._create_incremental_backup(model_path, model_name, version, timestamp, previous_backups)
            else:
                # Use full backup with optional compression
                backup_path = self._create_full_backup(model_path, backup_path)
            
            # Create metadata file
            metadata = {
                "model_name": model_name,
                "version": version,
                "original_path": model_path,
                "backup_time": timestamp,
                "backup_path": backup_path,
                "backup_type": "incremental" if file_size > self.incremental_threshold and previous_backups else "full",
                "compressed": self.compression_level is not None,
                "compression_level": self.compression_level,
                "file_size": file_size,
                "file_hash": self._calculate_file_hash(model_path)
            }
            
            metadata_path = f"{backup_path}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Verify backup if enabled
            if self.verify_backups:
                if not self.verify_backup(backup_path):
                    # If verification fails, delete the backup and raise an error
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    raise BackupVerificationError(f"Backup verification failed for {backup_path}")
            
            # Prune old backups if necessary
            self._prune_backups(model_name)
            
            logger.info(f"Created {'incremental' if metadata['backup_type'] == 'incremental' else 'full'} backup of model {model_name} v{version} at {backup_path}")
            return backup_path
            
        except Exception as e:
            error_msg = f"Failed to create backup of model {model_name} v{version}: {str(e)}"
            logger.error(error_msg)
            raise EnhancedModelBackupError(error_msg)

    def _create_full_backup(self, model_path: str, backup_path: str) -> str:
        """Create a full backup of a model with optional compression.

        Args:
            model_path: Path to the model file.
            backup_path: Path where the backup will be stored.

        Returns:
            str: Path to the backup file.
        """
        if self.compression_level is not None:
            # Create compressed backup
            compressed_backup_path = f"{backup_path}.zlib"
            with open(model_path, 'rb') as src_file, open(compressed_backup_path, 'wb') as dest_file:
                # Read the source file in chunks to avoid loading large files into memory
                compressor = zlib.compressobj(self.compression_level)
                for chunk in iter(lambda: src_file.read(4096), b""):
                    compressed_chunk = compressor.compress(chunk)
                    if compressed_chunk:
                        dest_file.write(compressed_chunk)
                # Ensure any remaining compressed data is written
                dest_file.write(compressor.flush())
            
            logger.info(f"Created compressed backup at {compressed_backup_path}")
            return compressed_backup_path
        else:
            # Create uncompressed backup
            shutil.copy2(model_path, backup_path)
            logger.info(f"Created uncompressed backup at {backup_path}")
            return backup_path

    def _create_incremental_backup(self, 
                                  model_path: str, 
                                  model_name: str, 
                                  version: str, 
                                  timestamp: str,
                                  previous_backups: List[Dict[str, Any]]) -> str:
        """Create an incremental backup based on the most recent previous backup.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model.
            version: Version of the model.
            timestamp: Timestamp for the backup.
            previous_backups: List of previous backup metadata.

        Returns:
            str: Path to the incremental backup file.
        """
        # Sort backups by time (newest first)
        previous_backups.sort(key=lambda x: x["backup_time"], reverse=True)
        
        # Get the most recent backup
        base_backup = previous_backups[0]
        base_backup_path = base_backup["backup_path"]
        
        # Generate incremental backup filename
        model_backup_dir = os.path.join(self.backup_dir, model_name)
        incremental_filename = f"{model_name}_v{version}_{timestamp}.incremental"
        incremental_path = os.path.join(model_backup_dir, incremental_filename)
        
        # Create temporary files for uncompressed content if needed
        with tempfile.NamedTemporaryFile(delete=False) as temp_current_file, \
             tempfile.NamedTemporaryFile(delete=False) as temp_base_file:
            
            temp_current_path = temp_current_file.name
            temp_base_path = temp_base_file.name
            
            # If base backup is compressed, decompress it
            if base_backup.get("compressed", False):
                self._decompress_file(base_backup_path, temp_base_path)
                base_content_path = temp_base_path
            else:
                base_content_path = base_backup_path
            
            # Read current model file
            with open(model_path, 'rb') as f:
                current_content = f.read()
            
            # Write current content to temp file
            with open(temp_current_path, 'wb') as f:
                f.write(current_content)
            
            # Generate diff between base and current
            with open(base_content_path, 'rb') as f:
                base_content = f.read()
            
            # Create binary diff
            diff = self._create_binary_diff(base_content, current_content)
            
            # Write diff to incremental backup file with optional compression
            if self.compression_level is not None:
                compressed_path = f"{incremental_path}.zlib"
                with open(compressed_path, 'wb') as f:
                    compressor = zlib.compressobj(self.compression_level)
                    compressed_data = compressor.compress(diff)
                    f.write(compressed_data)
                    f.write(compressor.flush())
                
                # Create reference file to base backup
                ref_path = f"{incremental_path}.ref"
                with open(ref_path, 'w') as f:
                    f.write(base_backup_path)
                
                logger.info(f"Created compressed incremental backup at {compressed_path}")
                
                # Clean up temp files
                os.unlink(temp_current_path)
                os.unlink(temp_base_path)
                
                return compressed_path
            else:
                # Write uncompressed diff
                with open(incremental_path, 'wb') as f:
                    f.write(diff)
                
                # Create reference file to base backup
                ref_path = f"{incremental_path}.ref"
                with open(ref_path, 'w') as f:
                    f.write(base_backup_path)
                
                logger.info(f"Created uncompressed incremental backup at {incremental_path}")
                
                # Clean up temp files
                os.unlink(temp_current_path)
                os.unlink(temp_base_path)
                
                return incremental_path

    def _create_binary_diff(self, base_content: bytes, current_content: bytes) -> bytes:
        """Create a binary diff between two files.

        Args:
            base_content: Content of the base file.
            current_content: Content of the current file.

        Returns:
            bytes: Binary diff data.
        """
        # Convert binary data to lines of hex strings for difflib
        def to_hex_lines(binary_data):
            hex_str = binary_data.hex()
            return [hex_str[i:i+32] for i in range(0, len(hex_str), 32)]
        
        base_lines = to_hex_lines(base_content)
        current_lines = to_hex_lines(current_content)
        
        # Generate unified diff
        diff = difflib.unified_diff(base_lines, current_lines, n=2)
        diff_text = '\n'.join(diff)
        
        # Return as bytes
        return diff_text.encode('utf-8')

    def rollback(self, model_name: str, version: Optional[str] = None, timestamp: Optional[str] = None) -> str:
        """Roll back to a previous version of a model, handling incremental backups.

        Args:
            model_name: Name of the model.
            version: Version to roll back to. If None, uses the most recent backup.
            timestamp: Specific backup timestamp to roll back to. If provided, takes precedence over version.

        Returns:
            str: Path to the restored model file.

        Raises:
            EnhancedModelBackupError: If the rollback operation fails.
        """
        try:
            # Get list of available backups
            backups = self.list_backups(model_name)
            if not backups:
                raise EnhancedModelBackupError(f"No backups found for model {model_name}")
            
            # Find the backup to restore
            backup_to_restore = self._find_backup_to_restore(backups, version, timestamp)
            
            # Restore the backup
            backup_path = backup_to_restore["backup_path"]
            original_path = backup_to_restore["original_path"]
            
            # Create a backup of the current model before restoring
            if os.path.exists(original_path):
                self._create_pre_rollback_backup(original_path, model_name, backup_to_restore)
            
            # Check if this is an incremental backup
            if backup_to_restore.get("backup_type") == "incremental":
                # Restore from incremental backup
                self._restore_from_incremental(backup_path, original_path, backup_to_restore)
            else:
                # Restore from full backup
                self._restore_from_full(backup_path, original_path, backup_to_restore)
            
            logger.info(
                f"Rolled back model {model_name} to version {backup_to_restore.get('version', 'unknown')} "
                f"from backup {backup_path}"
            )
            
            return original_path
        
        except Exception as e:
            error_msg = f"Failed to roll back model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise EnhancedModelBackupError(error_msg)

    def _find_backup_to_restore(self, 
                               backups: List[Dict[str, Any]], 
                               version: Optional[str], 
                               timestamp: Optional[str]) -> Dict[str, Any]:
        """Find the backup to restore based on version or timestamp.

        Args:
            backups: List of backup metadata.
            version: Version to roll back to.
            timestamp: Specific backup timestamp to roll back to.

        Returns:
            Dict[str, Any]: Backup metadata for the backup to restore.

        Raises:
            EnhancedModelBackupError: If no matching backup is found.
        """
        backup_to_restore = None
        
        if timestamp:
            # Find backup with matching timestamp
            for backup in backups:
                if timestamp in backup["backup_time"]:
                    backup_to_restore = backup
                    break
            
            if not backup_to_restore:
                raise EnhancedModelBackupError(f"No backup found with timestamp {timestamp}")
        
        elif version:
            # Find most recent backup with matching version
            matching_backups = [b for b in backups if b["version"] == version]
            if not matching_backups:
                raise EnhancedModelBackupError(f"No backup found with version {version}")
            
            # Sort by backup time (descending) and take the most recent
            matching_backups.sort(key=lambda x: x["backup_time"], reverse=True)
            backup_to_restore = matching_backups[0]
        
        else:
            # Use the most recent backup
            backups.sort(key=lambda x: x["backup_time"], reverse=True)
            backup_to_restore = backups[0]
        
        return backup_to_restore

    def _create_pre_rollback_backup(self, 
                                   original_path: str, 
                                   model_name: str, 
                                   backup_to_restore: Dict[str, Any]) -> None:
        """Create a backup of the current model before rolling back.

        Args:
            original_path: Path to the current model file.
            model_name: Name of the model.
            backup_to_restore: Metadata of the backup being restored.
        """
        current_version = "unknown"
        if "version" in backup_to_restore:
            current_version = backup_to_restore["version"]
        
        # Create backup directory for current model if it doesn't exist
        current_backup_dir = os.path.join(os.path.dirname(original_path), "pre_rollback_backup")
        os.makedirs(current_backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_backup_path = os.path.join(
            current_backup_dir, 
            f"{model_name}_pre_rollback_{timestamp}.backup"
        )
        
        # Copy current model to backup location
        shutil.copy2(original_path, current_backup_path)
        logger.info(f"Created pre-rollback backup at {current_backup_path}")

    def _restore_from_full(self, 
                          backup_path: str, 
                          original_path: str, 
                          backup_metadata: Dict[str, Any]) -> None:
        """Restore a model from a full backup.

        Args:
            backup_path: Path to the backup file.
            original_path: Path where the model will be restored.
            backup_metadata: Metadata of the backup.
        """
        # Check if backup is compressed
        if backup_metadata.get("compressed", False):
            # Decompress backup to original path
            self._decompress_file(backup_path, original_path)
        else:
            # Copy backup to original location
            shutil.copy2(backup_path, original_path)

    def _restore_from_incremental(self, 
                                 backup_path: str, 
                                 original_path: str, 
                                 backup_metadata: Dict[str, Any]) -> None:
        """Restore a model from an incremental backup.

        Args:
            backup_path: Path to the incremental backup file.
            original_path: Path where the model will be restored.
            backup_metadata: Metadata of the backup.
        """
        # Get reference to base backup
        ref_path = f"{backup_path}.ref"
        if not os.path.exists(ref_path):
            raise EnhancedModelBackupError(f"Reference file not found for incremental backup: {ref_path}")
        
        with open(ref_path, 'r') as f:
            base_backup_path = f.read().strip()
        
        if not os.path.exists(base_backup_path):
            raise EnhancedModelBackupError(f"Base backup not found: {base_backup_path}")
        
        # Get base backup metadata
        base_metadata_path = f"{base_backup_path}.meta"
        if not os.path.exists(base_metadata_path):
            raise EnhancedModelBackupError(f"Base backup metadata not found: {base_metadata_path}")
        
        with open(base_metadata_path, 'r') as f:
            base_metadata = json.load(f)
        
        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_base_file, \
             tempfile.NamedTemporaryFile(delete=False) as temp_diff_file, \
             tempfile.NamedTemporaryFile(delete=False) as temp_result_file:
            
            temp_base_path = temp_base_file.name
            temp_diff_path = temp_diff_file.name
            temp_result_path = temp_result_file.name
            
            # Restore base backup to temp file
            if base_metadata.get("compressed", False):
                self._decompress_file(base_backup_path, temp_base_path)
            else:
                shutil.copy2(base_backup_path, temp_base_path)
            
            # Get diff content
            if backup_metadata.get("compressed", False):
                self._decompress_file(backup_path, temp_diff_path)
            else:
                shutil.copy2(backup_path, temp_diff_path)
            
            # Apply diff to base content
            with open(temp_base_path, 'rb') as f:
                base_content = f.read()
            
            with open(temp_diff_path, 'rb') as f:
                diff_content = f.read()
            
            # Apply binary diff
            result_content = self._apply_binary_diff(base_content, diff_content)
            
            # Write result to temp file
            with open(temp_result_path, 'wb') as f:
                f.write(result_content)
            
            # Copy result to original path
            shutil.copy2(temp_result_path, original_path)
            
            # Clean up temp files
            os.unlink(temp_base_path)
            os.unlink(temp_diff_path)
            os.unlink(temp_result_path)

    def _apply_binary_diff(self, base_content: bytes, diff_content: bytes) -> bytes:
        """Apply a binary diff to base content.

        Args:
            base_content: Content of the base file.
            diff_content: Diff content.

        Returns:
            bytes: Resulting content after applying the diff.
        """
        # Convert binary data to lines of hex strings
        def to_hex_lines(binary_data):
            hex_str = binary_data.hex()
            return [hex_str[i:i+32] for i in range(0, len(hex_str), 32)]
        
        # Convert hex lines back to binary
        def from_hex_lines(hex_lines):
            hex_str = ''.join(hex_lines)
            return bytes.fromhex(hex_str)
        
        # Parse the unified diff
        base_lines = to_hex_lines(base_content)
        diff_text = diff_content.decode('utf-8')
        
        # Apply the diff using difflib
        patched_lines = base_lines.copy()
        current_line = 0
        
        for line in diff_text.splitlines():
            if line.startswith('---') or line.startswith('+++'):
                continue
            elif line.startswith('@@'):
                # Parse the hunk header
                parts = line.split()
                ranges = parts[1].split(',')  # -start,count
                start = int(ranges[0][1:])  # Remove the '-'
                current_line = start - 1  # 0-based indexing
            elif line.startswith('-'):
                # Remove line
                if current_line < len(patched_lines) and patched_lines[current_line] == line[1:]:
                    patched_lines.pop(current_line)
                else:
                    logger.warning(f"Diff mismatch at line {current_line}: expected '{patched_lines[current_line]}', got '{line[1:]}'. Skipping.")
                    current_line += 1
            elif line.startswith('+'):
                # Add line
                patched_lines.insert(current_line, line[1:])
                current_line += 1
            else:
                # Context line
                current_line += 1
        
        # Convert back to binary
        return from_hex_lines(patched_lines)

    def _decompress_file(self, compressed_path: str, output_path: str) -> None:
        """Decompress a file compressed with zlib.

        Args:
            compressed_path: Path to the compressed file.
            output_path: Path where the decompressed file will be written.
        """
        with open(compressed_path, 'rb') as src_file, open(output_path, 'wb') as dest_file:
            decompressor = zlib.decompressobj()
            for chunk in iter(lambda: src_file.read(4096), b""):
                decompressed_chunk = decompressor.decompress(chunk)
                if decompressed_chunk:
                    dest_file.write(decompressed_chunk)
            # Ensure any remaining decompressed data is written
            dest_file.write(decompressor.flush())

    def verify_backup(self, backup_path: str) -> bool:
        """Verify the integrity of a backup.

        Args:
            backup_path: Path to the backup file.

        Returns:
            bool: True if the backup is valid, False otherwise.
        """
        try:
            # Get backup metadata
            metadata_path = f"{backup_path}.meta"
            if not os.path.exists(metadata_path):
                logger.error(f"Backup metadata not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if backup file exists
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # For incremental backups, verify the base backup exists
            if metadata.get("backup_type") == "incremental":
                ref_path = f"{backup_path}.ref"
                if not os.path.exists(ref_path):
                    logger.error(f"Reference file not found for incremental backup: {ref_path}")
                    return False
                
                with open(ref_path, 'r') as f:
                    base_backup_path = f.read().strip()
                
                if not os.path.exists(base_backup_path):
                    logger.error(f"Base backup not found: {base_backup_path}")
                    return False
                
                # Verify base backup
                if not self.verify_backup(base_backup_path):
                    logger.error(f"Base backup verification failed: {base_backup_path}")
                    return False
            
            # For full backups, verify by restoring to a temporary file and checking hash
            else:
                # Create a temporary file for verification
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Restore backup to temp file
                    if metadata.get("compressed", False):
                        self._decompress_file(backup_path, temp_path)
                    else:
                        shutil.copy2(backup_path, temp_path)
                    
                    # Calculate hash of restored file
                    restored_hash = self._calculate_file_hash(temp_path)
                    
                    # Compare with original hash
                    original_hash = metadata.get("file_hash")
                    if original_hash and restored_hash != original_hash:
                        logger.error(f"Hash mismatch for backup {backup_path}. Expected {original_hash}, got {restored_hash}")
                        return False
                    
                    logger.info(f"Backup verification successful for {backup_path}")
                    return True
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Error verifying backup {backup_path}: {str(e)}")
            return False

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate the SHA-256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            str: The hexadecimal digest of the hash.
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()

    def schedule_backups(self, 
                        model_paths: Dict[str, Dict[str, str]], 
                        interval_hours: int = 24, 
                        callback: Optional[Callable[[str, bool, Optional[str]], None]] = None) -> None:
        """Schedule regular backups for multiple models.

        Args:
            model_paths: Dictionary mapping model names to dictionaries containing 'path' and 'version'.
            interval_hours: The interval between backups in hours.
            callback: Optional callback function to call after each backup with parameters:
                      model_name, success, error_message.
        """
        def backup_job():
            while True:
                for model_name, info in model_paths.items():
                    try:
                        model_path = info["path"]
                        version = info["version"]
                        
                        # Skip if model file doesn't exist
                        if not os.path.exists(model_path):
                            logger.warning(f"Scheduled backup skipped: Model file not found: {model_path}")
                            if callback:
                                callback(model_name, False, f"Model file not found: {model_path}")
                            continue
                        
                        # Create backup
                        self.create_backup(model_path, model_name, version)
                        logger.info(f"Scheduled backup completed successfully for model {model_name}")
                        
                        if callback:
                            callback(model_name, True, None)
                    
                    except Exception as e:
                        error_msg = f"Scheduled backup failed for model {model_name}: {str(e)}"
                        logger.error(error_msg)
                        
                        if callback:
                            callback(model_name, False, error_msg)
                
                # Sleep until the next backup
                time.sleep(interval_hours * 3600)
        
        # Stop existing backup thread for these models if it exists
        self.stop_scheduled_backups(list(model_paths.keys()))
        
        # Start the backup thread
        backup_thread = threading.Thread(target=backup_job, daemon=True)
        backup_thread.start()
        
        # Store thread reference for each model
        for model_name in model_paths.keys():
            self.scheduled_backups[model_name] = backup_thread
        
        logger.info(f"Scheduled backups started for {len(model_paths)} models with interval of {interval_hours} hours")

    def stop_scheduled_backups(self, model_names: Optional[List[str]] = None) -> None:
        """Stop scheduled backups for specified models or all models.

        Args:
            model_names: List of model names to stop backups for. If None, stops all scheduled backups.
        """
        # If no model names provided, stop all scheduled backups
        if model_names is None:
            model_names = list(self.scheduled_backups.keys())
        
        # Remove models from scheduled backups
        for model_name in model_names:
            if model_name in self.scheduled_backups:
                del self.scheduled_backups[model_name]
        
        logger.info(f"Stopped scheduled backups for {len(model_names)} models")

    def get_backup_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about backups.

        Args:
            model_name: Name of the model. If None, gets stats for all models.

        Returns:
            Dict[str, Any]: Statistics about backups.
        """
        stats = {
            "total_backups": 0,
            "total_size": 0,
            "compressed_size": 0,
            "uncompressed_size": 0,
            "full_backups": 0,
            "incremental_backups": 0,
            "models": {}
        }
        
        # Get list of models
        if model_name:
            models = [model_name]
        else:
            models = [d for d in os.listdir(self.backup_dir) 
                     if os.path.isdir(os.path.join(self.backup_dir, d))]
        
        # Collect stats for each model
        for model in models:
            model_backup_dir = os.path.join(self.backup_dir, model)
            if not os.path.exists(model_backup_dir):
                continue
            
            model_stats = {
                "total_backups": 0,
                "total_size": 0,
                "compressed_size": 0,
                "uncompressed_size": 0,
                "full_backups": 0,
                "incremental_backups": 0,
                "latest_backup": None,
                "backups": []
            }
            
            # Get all backup metadata files
            metadata_files = [f for f in os.listdir(model_backup_dir) if f.endswith(".meta")]
            
            for metadata_file in metadata_files:
                metadata_path = os.path.join(model_backup_dir, metadata_file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    backup_path = metadata.get("backup_path")
                    if not backup_path or not os.path.exists(backup_path):
                        continue
                    
                    backup_size = os.path.getsize(backup_path)
                    backup_type = metadata.get("backup_type", "full")
                    compressed = metadata.get("compressed", False)
                    
                    # Update model stats
                    model_stats["total_backups"] += 1
                    model_stats["total_size"] += backup_size
                    
                    if compressed:
                        model_stats["compressed_size"] += backup_size
                    else:
                        model_stats["uncompressed_size"] += backup_size
                    
                    if backup_type == "incremental":
                        model_stats["incremental_backups"] += 1
                    else:
                        model_stats["full_backups"] += 1
                    
                    # Add backup info
                    backup_info = {
                        "version": metadata.get("version", "unknown"),
                        "backup_time": metadata.get("backup_time", ""),
                        "backup_type": backup_type,
                        "compressed": compressed,
                        "size": backup_size,
                        "path": backup_path
                    }
                    
                    model_stats["backups"].append(backup_info)
                    
                    # Update latest backup
                    if model_stats["latest_backup"] is None or \
                       backup_info["backup_time"] > model_stats["latest_backup"]["backup_time"]:
                        model_stats["latest_backup"] = backup_info
                
                except Exception as e:
                    logger.error(f"Error reading backup metadata {metadata_path}: {str(e)}")
            
            # Sort backups by time (newest first)
            model_stats["backups"].sort(key=lambda x: x["backup_time"], reverse=True)
            
            # Add model stats to overall stats
            stats["total_backups"] += model_stats["total_backups"]
            stats["total_size"] += model_stats["total_size"]
            stats["compressed_size"] += model_stats["compressed_size"]
            stats["uncompressed_size"] += model_stats["uncompressed_size"]
            stats["full_backups"] += model_stats["full_backups"]
            stats["incremental_backups"] += model_stats["incremental_backups"]
            stats["models"][model] = model_stats
        
        return stats


# Create a singleton instance of the enhanced model backup manager with default settings
# This will be initialized with proper settings when the application starts
enhanced_model_backup_manager = EnhancedModelBackupManager(
    os.path.join(os.path.dirname(__file__), "backups"),
    max_backups=10,
    compression_level=6,
    verify_backups=True,
    incremental_threshold=10 * 1024 * 1024  # 10 MB
)