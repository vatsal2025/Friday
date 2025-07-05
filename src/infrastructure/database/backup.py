"""Backup and recovery utilities for the Friday AI Trading System.

This module provides functions for backing up and restoring databases and other data.
"""

import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.database import backup_database as sql_backup_database
from src.infrastructure.database import restore_database as sql_restore_database
from src.infrastructure.database.mongodb import backup_database as mongo_backup_database
from src.infrastructure.database.mongodb import restore_database as mongo_restore_database

# Create logger
logger = get_logger(__name__)


def get_backup_directory() -> str:
    """Get the backup directory from configuration.

    Returns:
        str: The backup directory path.
    """
    backup_dir = get_config("system", "backup_directory")
    if not backup_dir:
        # Default to a 'backups' directory in the project root
        backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "backups")
    
    # Create the directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    return backup_dir


def backup_all(include_sql: bool = True, include_mongo: bool = True, 
               include_files: bool = True) -> Dict[str, str]:
    """Backup all databases and data.

    Args:
        include_sql: Whether to include SQL database backup.
        include_mongo: Whether to include MongoDB backup.
        include_files: Whether to include file backup.

    Returns:
        Dict[str, str]: A dictionary of backup paths for each component.
    """
    backup_paths = {}
    backup_dir = get_backup_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a timestamped directory for this backup
    backup_timestamp_dir = os.path.join(backup_dir, timestamp)
    os.makedirs(backup_timestamp_dir, exist_ok=True)
    
    # Backup SQL database
    if include_sql:
        try:
            sql_backup_path = sql_backup_database(backup_timestamp_dir)
            backup_paths["sql"] = sql_backup_path
            logger.info("SQL database backup completed: %s", sql_backup_path)
        except Exception as e:
            logger.error("SQL database backup failed: %s", str(e))
    
    # Backup MongoDB
    if include_mongo:
        try:
            mongo_backup_path = mongo_backup_database(backup_timestamp_dir)
            backup_paths["mongo"] = mongo_backup_path
            logger.info("MongoDB backup completed: %s", mongo_backup_path)
        except Exception as e:
            logger.error("MongoDB backup failed: %s", str(e))
    
    # Backup files
    if include_files:
        try:
            files_backup_path = backup_files(backup_timestamp_dir)
            backup_paths["files"] = files_backup_path
            logger.info("Files backup completed: %s", files_backup_path)
        except Exception as e:
            logger.error("Files backup failed: %s", str(e))
    
    return backup_paths


def restore_all(backup_timestamp: str, include_sql: bool = True, 
                include_mongo: bool = True, include_files: bool = True) -> bool:
    """Restore all databases and data from a backup.

    Args:
        backup_timestamp: The timestamp of the backup to restore.
        include_sql: Whether to include SQL database restore.
        include_mongo: Whether to include MongoDB restore.
        include_files: Whether to include file restore.

    Returns:
        bool: True if all specified components were restored successfully, False otherwise.
    """
    success = True
    backup_dir = get_backup_directory()
    backup_timestamp_dir = os.path.join(backup_dir, backup_timestamp)
    
    if not os.path.exists(backup_timestamp_dir):
        logger.error("Backup directory not found: %s", backup_timestamp_dir)
        return False
    
    # Restore SQL database
    if include_sql:
        try:
            # Find the SQL backup file
            sql_backup_path = None
            for item in os.listdir(backup_timestamp_dir):
                if item.endswith(".sql") or item.endswith(".db") or item.endswith(".sqlite"):
                    sql_backup_path = os.path.join(backup_timestamp_dir, item)
                    break
            
            if sql_backup_path:
                sql_restore_database(sql_backup_path)
                logger.info("SQL database restored from: %s", sql_backup_path)
            else:
                logger.warning("No SQL backup found in: %s", backup_timestamp_dir)
                success = False
        except Exception as e:
            logger.error("SQL database restore failed: %s", str(e))
            success = False
    
    # Restore MongoDB
    if include_mongo:
        try:
            # Find the MongoDB backup directory
            mongo_backup_path = None
            for item in os.listdir(backup_timestamp_dir):
                mongo_dir = os.path.join(backup_timestamp_dir, item)
                if os.path.isdir(mongo_dir) and item.startswith("friday_"):
                    mongo_backup_path = mongo_dir
                    break
            
            if mongo_backup_path:
                mongo_restore_database(mongo_backup_path)
                logger.info("MongoDB restored from: %s", mongo_backup_path)
            else:
                logger.warning("No MongoDB backup found in: %s", backup_timestamp_dir)
                success = False
        except Exception as e:
            logger.error("MongoDB restore failed: %s", str(e))
            success = False
    
    # Restore files
    if include_files:
        try:
            # Find the files backup directory
            files_backup_path = os.path.join(backup_timestamp_dir, "files")
            if os.path.exists(files_backup_path):
                restore_files(files_backup_path)
                logger.info("Files restored from: %s", files_backup_path)
            else:
                logger.warning("No files backup found in: %s", backup_timestamp_dir)
                success = False
        except Exception as e:
            logger.error("Files restore failed: %s", str(e))
            success = False
    
    return success


def list_backups() -> List[Dict[str, str]]:
    """List all available backups.

    Returns:
        List[Dict[str, str]]: A list of backup information dictionaries.
    """
    backup_dir = get_backup_directory()
    backups = []
    
    if not os.path.exists(backup_dir):
        return backups
    
    for item in os.listdir(backup_dir):
        backup_path = os.path.join(backup_dir, item)
        if os.path.isdir(backup_path) and item.isdigit() or "_" in item:
            # This looks like a timestamp directory
            try:
                # Parse the timestamp
                timestamp = datetime.strptime(item, "%Y%m%d_%H%M%S")
                
                # Check what's in this backup
                components = []
                for component in os.listdir(backup_path):
                    if component.endswith(".sql") or component.endswith(".db") or component.endswith(".sqlite"):
                        components.append("sql")
                    elif os.path.isdir(os.path.join(backup_path, component)) and component.startswith("friday_"):
                        components.append("mongo")
                    elif component == "files":
                        components.append("files")
                
                backups.append({
                    "timestamp": item,
                    "datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "path": backup_path,
                    "components": components
                })
            except ValueError:
                # Not a valid timestamp format, skip
                continue
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return backups


def backup_files(backup_dir: str) -> str:
    """Backup important files.

    Args:
        backup_dir: The directory to store the backup.

    Returns:
        str: The path to the backup directory.
    """
    # Create files backup directory
    files_backup_dir = os.path.join(backup_dir, "files")
    os.makedirs(files_backup_dir, exist_ok=True)
    
    # Get files to backup from configuration
    files_to_backup = get_config("system", "files_to_backup") or []
    
    # Add default important files if not specified
    if not files_to_backup:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Default files to backup
        files_to_backup = [
            os.path.join(project_root, ".env"),
            os.path.join(project_root, "src", "infrastructure", "config", "unified_config.py"),
            # Add more default files here
        ]
    
    # Copy each file to the backup directory
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            # Create the directory structure in the backup
            rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            backup_file_path = os.path.join(files_backup_dir, rel_path)
            os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, backup_file_path)
            logger.debug("Backed up file: %s", file_path)
    
    return files_backup_dir


def restore_files(backup_dir: str) -> None:
    """Restore files from a backup.

    Args:
        backup_dir: The directory containing the backup.
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Walk through the backup directory and restore files
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            # Get the relative path from the backup directory
            backup_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(backup_file_path, backup_dir)
            
            # Construct the destination path
            dest_path = os.path.join(project_root, rel_path)
            
            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(backup_file_path, dest_path)
            logger.debug("Restored file: %s", dest_path)


def schedule_backups(interval_hours: int = 24) -> None:
    """Schedule regular backups.

    Args:
        interval_hours: The interval between backups in hours.
    """
    import threading
    
    def backup_job():
        while True:
            try:
                backup_all()
                logger.info("Scheduled backup completed successfully")
            except Exception as e:
                logger.error("Scheduled backup failed: %s", str(e))
            
            # Sleep until the next backup
            time.sleep(interval_hours * 3600)
    
    # Start the backup thread
    backup_thread = threading.Thread(target=backup_job, daemon=True)
    backup_thread.start()
    logger.info("Scheduled backups started with interval of %d hours", interval_hours)


def cleanup_old_backups(max_backups: int = 10) -> None:
    """Clean up old backups, keeping only the specified number of most recent backups.

    Args:
        max_backups: The maximum number of backups to keep.
    """
    backups = list_backups()
    
    # If we have more backups than the maximum, delete the oldest ones
    if len(backups) > max_backups:
        for backup in backups[max_backups:]:
            try:
                shutil.rmtree(backup["path"])
                logger.info("Deleted old backup: %s", backup["timestamp"])
            except Exception as e:
                logger.error("Failed to delete backup %s: %s", backup["timestamp"], str(e))