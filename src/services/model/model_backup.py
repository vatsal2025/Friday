"""Model Backup and Recovery for Friday AI Trading System.

This module provides functionality for creating and managing model backups.

DEPRECATION NOTICE: This module is being replaced by the enhanced_model_backup.py module,
which provides additional features such as incremental backups, compression, verification,
and automatic scheduling. Please consider using EnhancedModelBackupManager instead.
"""

import os
import json
import shutil
import datetime
from typing import Dict, List, Optional, Any
import warnings

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Show deprecation warning
warnings.warn(
    "ModelBackupManager is deprecated and will be removed in a future version. "
    "Please use EnhancedModelBackupManager from src.services.model.enhanced_model_backup instead.",
    DeprecationWarning,
    stacklevel=2
)


class ModelBackupError(Exception):
    """Exception raised for errors in the model backup operations."""
    pass


class ModelBackupManager:
    """Model Backup Manager for creating and managing model backups.
    
    DEPRECATED: Please use EnhancedModelBackupManager from src.services.model.enhanced_model_backup instead.
    
    Attributes:
        backup_dir: Directory where backups are stored.
        max_backups: Maximum number of backups to keep per model.
    """

    def __init__(self, backup_dir: str, max_backups: int = 5):
        """Initialize the model backup manager.

        Args:
            backup_dir: Directory where backups are stored.
            max_backups: Maximum number of backups to keep per model.
        """
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelBackupManager with backup directory: {backup_dir}")
        logger.info(f"Maximum backups per model: {max_backups}")

    def create_backup(self, model_path: str, model_name: str, version: str) -> str:
        """Create a backup of a model.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            str: Path to the backup file.

        Raises:
            ModelBackupError: If the backup operation fails.
        """
        try:
            # Ensure model file exists
            if not os.path.exists(model_path):
                raise ModelBackupError(f"Model file not found: {model_path}")
            
            # Create model backup directory if it doesn't exist
            model_backup_dir = os.path.join(self.backup_dir, model_name)
            os.makedirs(model_backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{model_name}_v{version}_{timestamp}.backup"
            backup_path = os.path.join(model_backup_dir, backup_filename)
            
            # Copy model file to backup location
            shutil.copy2(model_path, backup_path)
            
            # Create metadata file
            metadata = {
                "model_name": model_name,
                "version": version,
                "original_path": model_path,
                "backup_time": timestamp,
                "backup_path": backup_path
            }
            
            metadata_path = f"{backup_path}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Prune old backups if necessary
            self._prune_backups(model_name)
            
            logger.info(f"Created backup of model {model_name} v{version} at {backup_path}")
            return backup_path
            
        except Exception as e:
            error_msg = f"Failed to create backup of model {model_name} v{version}: {str(e)}"
            logger.error(error_msg)
            raise ModelBackupError(error_msg)

    def rollback(self, model_name: str, version: Optional[str] = None, timestamp: Optional[str] = None) -> str:
        """Roll back to a previous version of a model.

        Args:
            model_name: Name of the model.
            version: Version to roll back to. If None, uses the most recent backup.
            timestamp: Specific backup timestamp to roll back to. If provided, takes precedence over version.

        Returns:
            str: Path to the restored model file.

        Raises:
            ModelBackupError: If the rollback operation fails.
        """
        try:
            # Get list of available backups
            backups = self.list_backups(model_name)
            if not backups:
                raise ModelBackupError(f"No backups found for model {model_name}")
            
            # Find the backup to restore
            backup_to_restore = None
            
            if timestamp:
                # Find backup with matching timestamp
                for backup in backups:
                    if timestamp in backup["backup_time"]:
                        backup_to_restore = backup
                        break
                
                if not backup_to_restore:
                    raise ModelBackupError(f"No backup found with timestamp {timestamp}")
            
            elif version:
                # Find most recent backup with matching version
                matching_backups = [b for b in backups if b["version"] == version]
                if not matching_backups:
                    raise ModelBackupError(f"No backup found with version {version}")
                
                # Sort by backup time (descending) and take the most recent
                matching_backups.sort(key=lambda x: x["backup_time"], reverse=True)
                backup_to_restore = matching_backups[0]
            
            else:
                # Use the most recent backup
                backups.sort(key=lambda x: x["backup_time"], reverse=True)
                backup_to_restore = backups[0]
            
            # Restore the backup
            backup_path = backup_to_restore["backup_path"]
            original_path = backup_to_restore["original_path"]
            
            # Create a backup of the current model before restoring
            if os.path.exists(original_path):
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
            
            # Copy backup to original location
            shutil.copy2(backup_path, original_path)
            
            logger.info(
                f"Rolled back model {model_name} to version {backup_to_restore.get('version', 'unknown')} "
                f"from backup {backup_path}"
            )
            
            return original_path
        
        except Exception as e:
            error_msg = f"Failed to roll back model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ModelBackupError(error_msg)

    def list_backups(self, model_name: str) -> List[Dict[str, Any]]:
        """List all backups for a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[Dict[str, Any]]: List of backup metadata.
        """
        model_backup_dir = os.path.join(self.backup_dir, model_name)
        if not os.path.exists(model_backup_dir):
            return []
        
        # Get all backup metadata files
        metadata_files = [f for f in os.listdir(model_backup_dir) if f.endswith(".meta")]
        
        backups = []
        for metadata_file in metadata_files:
            metadata_path = os.path.join(model_backup_dir, metadata_file)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if backup file exists
                backup_path = metadata.get("backup_path")
                if backup_path and os.path.exists(backup_path):
                    backups.append(metadata)
            
            except Exception as e:
                logger.error(f"Error reading backup metadata {metadata_path}: {str(e)}")
        
        # Sort backups by time (newest first)
        backups.sort(key=lambda x: x["backup_time"], reverse=True)
        
        return backups

    def _prune_backups(self, model_name: str) -> None:
        """Remove old backups exceeding the maximum limit.

        Args:
            model_name: Name of the model.
        """
        backups = self.list_backups(model_name)
        
        # If we have more backups than the maximum allowed, remove the oldest ones
        if len(backups) > self.max_backups:
            # Sort backups by time (oldest first)
            backups.sort(key=lambda x: x["backup_time"])
            
            # Remove the oldest backups
            for backup in backups[:len(backups) - self.max_backups]:
                backup_path = backup["backup_path"]
                metadata_path = f"{backup_path}.meta"
                
                try:
                    # Remove backup file
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    
                    # Remove metadata file
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    logger.info(f"Pruned old backup: {backup_path}")
                
                except Exception as e:
                    logger.error(f"Error pruning backup {backup_path}: {str(e)}")

    def get_backup_info(self, model_name: str, version: Optional[str] = None, timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a specific backup.

        Args:
            model_name: Name of the model.
            version: Version of the backup. If None, uses the most recent backup.
            timestamp: Specific backup timestamp. If provided, takes precedence over version.

        Returns:
            Optional[Dict[str, Any]]: Backup metadata, or None if not found.
        """
        backups = self.list_backups(model_name)
        if not backups:
            return None
        
        # Find the specific backup
        if timestamp:
            # Find backup with matching timestamp
            for backup in backups:
                if timestamp in backup["backup_time"]:
                    return backup
            return None
        
        elif version:
            # Find most recent backup with matching version
            matching_backups = [b for b in backups if b["version"] == version]
            if not matching_backups:
                return None
            
            # Sort by backup time (descending) and take the most recent
            matching_backups.sort(key=lambda x: x["backup_time"], reverse=True)
            return matching_backups[0]
        
        else:
            # Use the most recent backup
            backups.sort(key=lambda x: x["backup_time"], reverse=True)
            return backups[0]


# Create a singleton instance of the model backup manager
# This will be initialized with proper settings when the application starts
model_backup_manager = ModelBackupManager(
    os.path.join(os.path.dirname(__file__), "backups"),
    max_backups=5
)