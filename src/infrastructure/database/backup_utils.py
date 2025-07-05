"""Database backup utilities for the Friday AI Trading System.

This module provides functions for backing up and restoring MongoDB databases and Redis data.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from src.infrastructure.config import get_config
from src.infrastructure.cache import get_redis_client
from src.infrastructure.database.mongodb import get_mongo_client, get_database
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Get configurations
MONGODB_CONFIG = get_config("mongodb")
REDIS_CONFIG = get_config("redis")

# Default backup directory
DEFAULT_BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                  "storage", "backups")


def ensure_backup_dir(backup_dir: Optional[str] = None) -> str:
    """Ensure the backup directory exists.

    Args:
        backup_dir (Optional[str]): The backup directory path. If None, uses the default.

    Returns:
        str: The absolute path to the backup directory.
    """
    if backup_dir is None:
        backup_dir = DEFAULT_BACKUP_DIR
    
    # Create the backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    return os.path.abspath(backup_dir)


def generate_backup_filename(prefix: str, extension: str) -> str:
    """Generate a backup filename with timestamp.

    Args:
        prefix (str): The prefix for the backup filename.
        extension (str): The file extension.

    Returns:
        str: The generated backup filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def backup_mongodb(backup_dir: Optional[str] = None, 
                  database_name: Optional[str] = None,
                  collections: Optional[List[str]] = None,
                  compress: bool = True) -> Dict[str, Any]:
    """Backup MongoDB database using mongodump.

    Args:
        backup_dir (Optional[str]): The directory to store the backup. If None, uses the default.
        database_name (Optional[str]): The name of the database to backup. If None, uses the configured database.
        collections (Optional[List[str]]): List of collections to backup. If None, backs up all collections.
        compress (bool): Whether to compress the backup.

    Returns:
        Dict[str, Any]: A dictionary containing the backup results.
    """
    try:
        # Ensure backup directory exists
        backup_dir = ensure_backup_dir(backup_dir)
        
        # Get database name from config if not provided
        if database_name is None:
            database_name = MONGODB_CONFIG.get("database", "friday")
        
        # Generate backup filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mongodb_{database_name}_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        # Create backup directory
        os.makedirs(backup_path, exist_ok=True)
        
        # Build mongodump command
        cmd = ["mongodump", "--db", database_name, "--out", backup_path]
        
        # Add host and port if configured
        host = MONGODB_CONFIG.get("host")
        port = MONGODB_CONFIG.get("port")
        if host:
            cmd.extend(["--host", host])
        if port:
            cmd.extend(["--port", str(port)])
        
        # Add username and password if configured
        username = MONGODB_CONFIG.get("username")
        password = MONGODB_CONFIG.get("password")
        if username and password:
            cmd.extend(["--username", username, "--password", password, "--authenticationDatabase", "admin"])
        
        # Add collections if specified
        if collections:
            for collection in collections:
                cmd.extend(["--collection", collection])
        
        # Execute mongodump command
        logger.info(f"Starting MongoDB backup to {backup_path}")
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if process.returncode != 0:
            logger.error(f"MongoDB backup failed: {process.stderr}")
            return {
                "success": False,
                "error": process.stderr,
                "backup_path": None,
                "duration_seconds": time.time() - start_time
            }
        
        # Compress backup if requested
        archive_path = None
        if compress:
            archive_name = f"{backup_name}.tar.gz"
            archive_path = os.path.join(backup_dir, archive_name)
            logger.info(f"Compressing backup to {archive_path}")
            
            # Create tar.gz archive
            shutil.make_archive(
                os.path.join(backup_dir, backup_name),
                'gztar',
                backup_dir,
                backup_name
            )
            
            # Remove the uncompressed backup directory
            shutil.rmtree(backup_path)
            backup_path = archive_path
        
        duration = time.time() - start_time
        logger.info(f"MongoDB backup completed in {duration:.2f} seconds")
        
        return {
            "success": True,
            "backup_path": backup_path,
            "database": database_name,
            "collections": collections,
            "compressed": compress,
            "timestamp": timestamp,
            "duration_seconds": duration
        }
    except Exception as e:
        logger.error(f"Error during MongoDB backup: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "backup_path": None,
            "duration_seconds": 0
        }


def restore_mongodb(backup_path: str, 
                   database_name: Optional[str] = None,
                   collections: Optional[List[str]] = None,
                   drop: bool = False) -> Dict[str, Any]:
    """Restore MongoDB database from a backup.

    Args:
        backup_path (str): The path to the backup file or directory.
        database_name (Optional[str]): The name of the database to restore to. If None, uses the original name.
        collections (Optional[List[str]]): List of collections to restore. If None, restores all collections.
        drop (bool): Whether to drop the collections before restoring.

    Returns:
        Dict[str, Any]: A dictionary containing the restore results.
    """
    try:
        # Check if backup path exists
        if not os.path.exists(backup_path):
            return {
                "success": False,
                "error": f"Backup path does not exist: {backup_path}",
                "duration_seconds": 0
            }
        
        # Handle compressed backup
        is_compressed = backup_path.endswith(".tar.gz")
        temp_dir = None
        
        if is_compressed:
            # Create temporary directory for extraction
            temp_dir = os.path.join(os.path.dirname(backup_path), "temp_restore_" + str(int(time.time())))
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract archive
            logger.info(f"Extracting backup archive to {temp_dir}")
            shutil.unpack_archive(backup_path, temp_dir)
            
            # Find the extracted directory
            extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if not extracted_dirs:
                return {
                    "success": False,
                    "error": "No directories found in the extracted backup",
                    "duration_seconds": 0
                }
            
            backup_path = os.path.join(temp_dir, extracted_dirs[0])
        
        # Get database name from backup directory if not provided
        if database_name is None:
            # Extract database name from backup directory name
            backup_dir_name = os.path.basename(backup_path)
            parts = backup_dir_name.split("_")
            if len(parts) >= 2:
                database_name = parts[1]
            else:
                database_name = MONGODB_CONFIG.get("database", "friday")
        
        # Build mongorestore command
        cmd = ["mongorestore", "--db", database_name]
        
        # Add host and port if configured
        host = MONGODB_CONFIG.get("host")
        port = MONGODB_CONFIG.get("port")
        if host:
            cmd.extend(["--host", host])
        if port:
            cmd.extend(["--port", str(port)])
        
        # Add username and password if configured
        username = MONGODB_CONFIG.get("username")
        password = MONGODB_CONFIG.get("password")
        if username and password:
            cmd.extend(["--username", username, "--password", password, "--authenticationDatabase", "admin"])
        
        # Add drop option if specified
        if drop:
            cmd.append("--drop")
        
        # Add collections if specified
        if collections:
            for collection in collections:
                collection_path = os.path.join(backup_path, database_name, f"{collection}.bson")
                if os.path.exists(collection_path):
                    cmd.extend(["--collection", collection, collection_path])
        else:
            # Add backup path
            cmd.append(os.path.join(backup_path, database_name))
        
        # Execute mongorestore command
        logger.info(f"Starting MongoDB restore from {backup_path}")
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Clean up temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        if process.returncode != 0:
            logger.error(f"MongoDB restore failed: {process.stderr}")
            return {
                "success": False,
                "error": process.stderr,
                "duration_seconds": time.time() - start_time
            }
        
        duration = time.time() - start_time
        logger.info(f"MongoDB restore completed in {duration:.2f} seconds")
        
        return {
            "success": True,
            "database": database_name,
            "collections": collections,
            "drop": drop,
            "duration_seconds": duration
        }
    except Exception as e:
        logger.error(f"Error during MongoDB restore: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": 0
        }


def backup_redis(backup_dir: Optional[str] = None,
                namespace_pattern: Optional[str] = None,
                rdb_backup: bool = True) -> Dict[str, Any]:
    """Backup Redis data.

    This function backs up Redis data either by saving the RDB file or by dumping keys and values.

    Args:
        backup_dir (Optional[str]): The directory to store the backup. If None, uses the default.
        namespace_pattern (Optional[str]): Pattern to filter keys by namespace (e.g., "market:*").
        rdb_backup (bool): Whether to attempt an RDB file backup (requires Redis server access).

    Returns:
        Dict[str, Any]: A dictionary containing the backup results.
    """
    try:
        # Ensure backup directory exists
        backup_dir = ensure_backup_dir(backup_dir)
        
        # Generate backup filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"redis_{timestamp}"
        
        # Get Redis client
        client = get_redis_client()
        
        # Try RDB backup if requested
        if rdb_backup:
            try:
                # Save RDB file
                save_result = client.save()
                
                # Get RDB filename from Redis config
                config = client.config_get("dbfilename")
                rdb_filename = config.get("dbfilename", "dump.rdb")
                
                # Get RDB directory from Redis config
                config = client.config_get("dir")
                rdb_dir = config.get("dir", "/var/lib/redis")
                
                # Full path to RDB file
                rdb_path = os.path.join(rdb_dir, rdb_filename)
                
                # Check if we have access to the RDB file
                if os.path.exists(rdb_path) and os.access(rdb_path, os.R_OK):
                    # Copy RDB file to backup directory
                    backup_rdb_path = os.path.join(backup_dir, f"{backup_name}.rdb")
                    shutil.copy2(rdb_path, backup_rdb_path)
                    
                    logger.info(f"Redis RDB backup completed: {backup_rdb_path}")
                    return {
                        "success": True,
                        "backup_path": backup_rdb_path,
                        "backup_type": "rdb",
                        "timestamp": timestamp
                    }
                else:
                    logger.warning(f"Cannot access Redis RDB file at {rdb_path}. Falling back to key dump.")
            except Exception as e:
                logger.warning(f"RDB backup failed: {str(e)}. Falling back to key dump.")
        
        # Fallback to key dump
        backup_json_path = os.path.join(backup_dir, f"{backup_name}.json")
        
        # Get all keys matching the pattern
        pattern = namespace_pattern if namespace_pattern else "*"
        keys = client.keys(pattern)
        
        # Initialize backup data structure
        backup_data = {}
        
        # Process each key
        for key in keys:
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            
            # Get key type
            key_type = client.type(key).decode("utf-8") if isinstance(client.type(key), bytes) else client.type(key)
            
            # Get TTL
            ttl = client.ttl(key)
            
            # Get value based on key type
            if key_type == "string":
                value = client.get(key)
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        # If it's not UTF-8 decodable, store as base64
                        import base64
                        value = {"__binary": base64.b64encode(value).decode("ascii")}
            elif key_type == "list":
                value = client.lrange(key, 0, -1)
                value = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value]
            elif key_type == "set":
                value = client.smembers(key)
                value = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value]
            elif key_type == "zset":
                value = client.zrange(key, 0, -1, withscores=True)
                value = [(v[0].decode("utf-8") if isinstance(v[0], bytes) else v[0], v[1]) for v in value]
            elif key_type == "hash":
                value = client.hgetall(key)
                value = {k.decode("utf-8") if isinstance(k, bytes) else k: 
                         v.decode("utf-8") if isinstance(v, bytes) else v 
                         for k, v in value.items()}
            else:
                value = None
            
            # Store key data
            backup_data[key_str] = {
                "type": key_type,
                "ttl": ttl,
                "value": value
            }
        
        # Write backup data to JSON file
        with open(backup_json_path, "w") as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Redis key dump backup completed: {backup_json_path}")
        return {
            "success": True,
            "backup_path": backup_json_path,
            "backup_type": "json",
            "key_count": len(keys),
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error during Redis backup: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "backup_path": None
        }


def restore_redis(backup_path: str,
                 namespace_filter: Optional[str] = None,
                 clear_existing: bool = False) -> Dict[str, Any]:
    """Restore Redis data from a backup.

    Args:
        backup_path (str): The path to the backup file.
        namespace_filter (Optional[str]): Pattern to filter keys by namespace (e.g., "market:*").
        clear_existing (bool): Whether to clear existing keys in the namespace before restoring.

    Returns:
        Dict[str, Any]: A dictionary containing the restore results.
    """
    try:
        # Check if backup path exists
        if not os.path.exists(backup_path):
            return {
                "success": False,
                "error": f"Backup path does not exist: {backup_path}",
                "restored_keys": 0
            }
        
        # Get Redis client
        client = get_redis_client()
        
        # Handle RDB backup
        if backup_path.endswith(".rdb"):
            logger.error("Direct RDB restore is not supported. Please use Redis server tools to restore RDB file.")
            return {
                "success": False,
                "error": "Direct RDB restore is not supported. Please use Redis server tools to restore RDB file.",
                "restored_keys": 0
            }
        
        # Handle JSON backup
        if backup_path.endswith(".json"):
            # Load backup data
            with open(backup_path, "r") as f:
                backup_data = json.load(f)
            
            # Clear existing keys if requested
            if clear_existing and namespace_filter:
                keys_to_delete = client.keys(namespace_filter)
                if keys_to_delete:
                    client.delete(*keys_to_delete)
                    logger.info(f"Cleared {len(keys_to_delete)} existing keys matching pattern: {namespace_filter}")
            
            # Restore keys
            restored_count = 0
            for key, data in backup_data.items():
                # Skip keys that don't match the namespace filter
                if namespace_filter and not key.startswith(namespace_filter.rstrip("*")):
                    continue
                
                key_type = data["type"]
                value = data["value"]
                ttl = data["ttl"]
                
                # Restore based on key type
                if key_type == "string":
                    # Handle binary data
                    if isinstance(value, dict) and "__binary" in value:
                        import base64
                        binary_value = base64.b64decode(value["__binary"])
                        client.set(key, binary_value)
                    else:
                        client.set(key, value)
                elif key_type == "list":
                    # Delete existing key first
                    client.delete(key)
                    if value:  # Only push if there are items
                        client.rpush(key, *value)
                elif key_type == "set":
                    # Delete existing key first
                    client.delete(key)
                    if value:  # Only add if there are items
                        client.sadd(key, *value)
                elif key_type == "zset":
                    # Delete existing key first
                    client.delete(key)
                    if value:  # Only add if there are items
                        for member, score in value:
                            client.zadd(key, {member: score})
                elif key_type == "hash":
                    # Delete existing key first
                    client.delete(key)
                    if value:  # Only set if there are items
                        client.hset(key, mapping=value)
                
                # Set TTL if it's not -1 (no expiry)
                if ttl > 0:
                    client.expire(key, ttl)
                
                restored_count += 1
            
            logger.info(f"Redis restore completed: {restored_count} keys restored")
            return {
                "success": True,
                "restored_keys": restored_count,
                "backup_type": "json"
            }
        
        # Unsupported backup format
        return {
            "success": False,
            "error": f"Unsupported backup format: {os.path.basename(backup_path)}",
            "restored_keys": 0
        }
    except Exception as e:
        logger.error(f"Error during Redis restore: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "restored_keys": 0
        }


def schedule_backup(backup_type: str, 
                   schedule: str, 
                   retention_days: int = 30,
                   backup_dir: Optional[str] = None,
                   **backup_options) -> Dict[str, Any]:
    """Schedule a database backup using the system's scheduler.

    Args:
        backup_type (str): The type of backup ("mongodb" or "redis").
        schedule (str): The schedule in cron format (e.g., "0 0 * * *" for daily at midnight).
        retention_days (int): Number of days to keep backups.
        backup_dir (Optional[str]): The directory to store backups.
        **backup_options: Additional options for the specific backup type.

    Returns:
        Dict[str, Any]: A dictionary containing the scheduling results.
    """
    try:
        # Ensure backup directory exists
        backup_dir = ensure_backup_dir(backup_dir)
        
        # Get the path to the current script
        current_script = os.path.abspath(__file__)
        
        # Get the path to the Python interpreter
        python_interpreter = sys.executable
        
        # Create a backup script
        script_content = f"""#!/usr/bin/env python
"""
        
        if backup_type.lower() == "mongodb":
            script_content += f"""
import sys
sys.path.append('{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script))))}')  # Add project root to path

from src.infrastructure.database.backup_utils import backup_mongodb, cleanup_old_backups

# Run MongoDB backup
backup_options = {backup_options}
backup_result = backup_mongodb(backup_dir='{backup_dir}', **backup_options)

# Clean up old backups
cleanup_old_backups('{backup_dir}', retention_days={retention_days}, backup_type='mongodb')
"""
        elif backup_type.lower() == "redis":
            script_content += f"""
import sys
sys.path.append('{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script))))}')  # Add project root to path

from src.infrastructure.database.backup_utils import backup_redis, cleanup_old_backups

# Run Redis backup
backup_options = {backup_options}
backup_result = backup_redis(backup_dir='{backup_dir}', **backup_options)

# Clean up old backups
cleanup_old_backups('{backup_dir}', retention_days={retention_days}, backup_type='redis')
"""
        else:
            return {
                "success": False,
                "error": f"Unsupported backup type: {backup_type}"
            }
        
        # Write the backup script to a file
        script_path = os.path.join(backup_dir, f"scheduled_{backup_type}_backup.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Create a cron job (Linux/macOS) or scheduled task (Windows)
        if os.name == "posix":  # Linux/macOS
            # Create a temporary file for crontab
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            
            try:
                # Get existing crontab
                process = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
                existing_crontab = process.stdout if process.returncode == 0 else ""
                
                # Add new cron job
                cron_job = f"{schedule} {python_interpreter} {script_path}\n"
                
                # Check if job already exists
                if script_path in existing_crontab:
                    # Update existing job
                    new_crontab = ""
                    for line in existing_crontab.splitlines():
                        if script_path in line:
                            new_crontab += cron_job
                        else:
                            new_crontab += line + "\n"
                else:
                    # Add new job
                    new_crontab = existing_crontab + cron_job
                
                # Write to temporary file
                temp_file.write(new_crontab.encode())
                temp_file.flush()
                
                # Install new crontab
                subprocess.run(["crontab", temp_file.name], check=True)
                
                logger.info(f"Scheduled {backup_type} backup: {schedule}")
                return {
                    "success": True,
                    "backup_type": backup_type,
                    "schedule": schedule,
                    "script_path": script_path,
                    "scheduler": "crontab"
                }
            finally:
                # Clean up temporary file
                temp_file.close()
                os.unlink(temp_file.name)
        
        elif os.name == "nt":  # Windows
            # Convert cron schedule to Windows schedule
            schedule_parts = schedule.split()
            if len(schedule_parts) != 5:
                return {
                    "success": False,
                    "error": "Invalid cron schedule format"
                }
            
            minute, hour, day, month, weekday = schedule_parts
            
            # Create a scheduled task
            task_name = f"FridayAI_{backup_type}_Backup"
            
            # Build schtasks command
            cmd = ["schtasks", "/create", "/tn", task_name, "/tr", f"\"{python_interpreter}\" \"{script_path}\"", "/sc"]
            
            # Determine schedule type
            if minute != "*" and hour != "*" and day != "*" and month != "*" and weekday != "*":
                # Specific time
                cmd.extend(["once", "/st", f"{hour.zfill(2)}:{minute.zfill(2)}", "/sd", f"{month}/{day}/{datetime.now().year}"])
            elif minute != "*" and hour != "*" and day == "*" and month == "*" and weekday == "*":
                # Daily
                cmd.extend(["daily", "/st", f"{hour.zfill(2)}:{minute.zfill(2)}"])
            elif minute != "*" and hour != "*" and day == "*" and month == "*" and weekday != "*":
                # Weekly
                cmd.extend(["weekly", "/st", f"{hour.zfill(2)}:{minute.zfill(2)}", "/d", weekday])
            elif minute != "*" and hour != "*" and day != "*" and month == "*" and weekday == "*":
                # Monthly
                cmd.extend(["monthly", "/st", f"{hour.zfill(2)}:{minute.zfill(2)}", "/d", day])
            else:
                # Default to daily
                cmd.extend(["daily", "/st", f"{hour.zfill(2)}:{minute.zfill(2)}"])
            
            # Force overwrite if task exists
            cmd.extend(["/f"])
            
            # Execute schtasks command
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if process.returncode != 0:
                logger.error(f"Failed to schedule {backup_type} backup: {process.stderr}")
                return {
                    "success": False,
                    "error": process.stderr
                }
            
            logger.info(f"Scheduled {backup_type} backup: {task_name}")
            return {
                "success": True,
                "backup_type": backup_type,
                "schedule": schedule,
                "script_path": script_path,
                "scheduler": "schtasks",
                "task_name": task_name
            }
        
        else:
            return {
                "success": False,
                "error": f"Unsupported operating system: {os.name}"
            }
    except Exception as e:
        logger.error(f"Error scheduling {backup_type} backup: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def cleanup_old_backups(backup_dir: str, retention_days: int = 30, backup_type: Optional[str] = None) -> Dict[str, Any]:
    """Clean up old backups based on retention policy.

    Args:
        backup_dir (str): The directory containing backups.
        retention_days (int): Number of days to keep backups.
        backup_type (Optional[str]): Type of backups to clean up ("mongodb", "redis", or None for all).

    Returns:
        Dict[str, Any]: A dictionary containing the cleanup results.
    """
    try:
        # Ensure backup directory exists
        if not os.path.exists(backup_dir):
            return {
                "success": False,
                "error": f"Backup directory does not exist: {backup_dir}",
                "deleted_count": 0
            }
        
        # Calculate cutoff date
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        # Get all backup files
        backup_files = []
        for root, _, files in os.walk(backup_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip if not a backup file
                if backup_type == "mongodb" and not (file.startswith("mongodb_") or "mongodb" in root):
                    continue
                if backup_type == "redis" and not (file.startswith("redis_") or "redis" in root):
                    continue
                
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)
                
                # Add to list if older than cutoff
                if file_mtime < cutoff_time:
                    backup_files.append((file_path, file_mtime))
        
        # Delete old backup files
        deleted_count = 0
        for file_path, _ in backup_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                deleted_count += 1
                logger.info(f"Deleted old backup: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting backup {file_path}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old backups")
        return {
            "success": True,
            "deleted_count": deleted_count,
            "retention_days": retention_days,
            "backup_type": backup_type
        }
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "deleted_count": 0
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backup_utils.py [backup|restore|cleanup] [options]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "backup":
        if len(sys.argv) < 3:
            print("Usage: python backup_utils.py backup [mongodb|redis] [options]")
            sys.exit(1)
        
        backup_type = sys.argv[2].lower()
        
        if backup_type == "mongodb":
            result = backup_mongodb()
        elif backup_type == "redis":
            result = backup_redis()
        else:
            print(f"Unsupported backup type: {backup_type}")
            sys.exit(1)
        
        if result["success"]:
            print(f"Backup completed successfully: {result['backup_path']}")
        else:
            print(f"Backup failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif command == "restore":
        if len(sys.argv) < 4:
            print("Usage: python backup_utils.py restore [mongodb|redis] <backup_path> [options]")
            sys.exit(1)
        
        restore_type = sys.argv[2].lower()
        backup_path = sys.argv[3]
        
        if restore_type == "mongodb":
            result = restore_mongodb(backup_path)
        elif restore_type == "redis":
            result = restore_redis(backup_path)
        else:
            print(f"Unsupported restore type: {restore_type}")
            sys.exit(1)
        
        if result["success"]:
            print("Restore completed successfully")
        else:
            print(f"Restore failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif command == "cleanup":
        if len(sys.argv) < 3:
            print("Usage: python backup_utils.py cleanup <backup_dir> [retention_days] [backup_type]")
            sys.exit(1)
        
        backup_dir = sys.argv[2]
        retention_days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        backup_type = sys.argv[4] if len(sys.argv) > 4 else None
        
        result = cleanup_old_backups(backup_dir, retention_days, backup_type)
        
        if result["success"]:
            print(f"Cleanup completed successfully: {result['deleted_count']} backups deleted")
        else:
            print(f"Cleanup failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    else:
        print(f"Unsupported command: {command}")
        print("Usage: python backup_utils.py [backup|restore|cleanup] [options]")
        sys.exit(1)