# Enhanced Model Backup and Recovery System

## Overview

The Enhanced Model Backup and Recovery System provides robust functionality for creating, managing, and restoring backups of machine learning models in the Friday AI Trading System. This system builds upon the existing model backup capabilities with advanced features designed to improve efficiency, reliability, and storage utilization.

## Key Features

### 1. Incremental Backup Support

For large models, the system now supports incremental backups, which store only the differences between the current model and the previous backup. This significantly reduces storage requirements and backup time for large models that change frequently but have relatively small modifications between versions.

- **Automatic Detection**: The system automatically determines whether to use full or incremental backups based on the model size and the existence of previous backups.
- **Configurable Threshold**: You can configure the size threshold at which incremental backups are used.

### 2. Automatic Backup Scheduling

The system now supports automatic scheduled backups, ensuring that your models are regularly backed up without manual intervention.

- **Configurable Intervals**: Set the frequency of backups (e.g., hourly, daily, weekly).
- **Multiple Model Support**: Schedule backups for multiple models with different configurations.
- **Callback Notifications**: Receive notifications when backups succeed or fail.

### 3. Backup Compression

All backups can now be compressed to save storage space.

- **Configurable Compression Level**: Choose the compression level (0-9) to balance between compression ratio and performance.
- **Transparent Decompression**: Compressed backups are automatically decompressed during restoration.

### 4. Backup Verification

The system now includes a verification mechanism to ensure the integrity of backups.

- **Hash-based Verification**: Each backup is verified using SHA-256 hashes to ensure data integrity.
- **Automatic Verification**: Backups are automatically verified after creation.
- **Manual Verification**: Verify any backup at any time to ensure it's still valid.

## Usage Examples

### Basic Usage

```python
from src.services.model.enhanced_model_backup import EnhancedModelBackupManager

# Initialize the backup manager
backup_manager = EnhancedModelBackupManager(
    backup_dir="/path/to/backups",
    max_backups=5,
    compression_level=6,  # Medium compression level
    verify_backups=True,
    incremental_threshold=10 * 1024 * 1024  # 10 MB
)

# Create a backup
backup_path = backup_manager.create_backup(
    model_path="/path/to/model.joblib",
    model_name="trading_model",
    version="1.0"
)

# List available backups
backups = backup_manager.list_backups("trading_model")

# Roll back to a specific version
restored_path = backup_manager.rollback(
    model_name="trading_model",
    version="1.0"
)
```

### Scheduling Automatic Backups

```python
# Define callback function
def backup_callback(model_name, success, error_message):
    if success:
        print(f"Backup of {model_name} completed successfully")
    else:
        print(f"Backup of {model_name} failed: {error_message}")

# Schedule daily backups
backup_manager.schedule_backups(
    model_paths={
        "trading_model": {
            "path": "/path/to/model.joblib",
            "version": "1.0"
        },
        "prediction_model": {
            "path": "/path/to/prediction_model.joblib",
            "version": "2.3"
        }
    },
    interval_hours=24,  # Daily backups
    callback=backup_callback
)

# Stop scheduled backups for specific models
backup_manager.stop_scheduled_backups(["trading_model"])

# Stop all scheduled backups
backup_manager.stop_scheduled_backups()
```

### Getting Backup Statistics

```python
# Get statistics for all models
stats = backup_manager.get_backup_stats()
print(f"Total backups: {stats['total_backups']}")
print(f"Total size: {stats['total_size'] / (1024 * 1024):.2f} MB")

# Get statistics for a specific model
model_stats = backup_manager.get_backup_stats("trading_model")
print(f"Latest backup: {model_stats['models']['trading_model']['latest_backup']['backup_time']}")
```

## Configuration Options

### EnhancedModelBackupManager

- **backup_dir** (str): Directory where backups are stored.
- **max_backups** (int, default=5): Maximum number of backups to keep per model.
- **compression_level** (int or None, default=6): Compression level (0-9, None for no compression).
- **verify_backups** (bool, default=True): Whether to verify backups after creation.
- **incremental_threshold** (int, default=10MB): File size threshold for incremental backups (bytes).
- **keys_dir** (str, optional): Directory for storing signature keys.

## Best Practices

1. **Regular Backups**: Schedule regular backups for all critical models.
2. **Verification**: Always enable backup verification to ensure data integrity.
3. **Compression Level**: Use compression level 6 for a good balance between compression ratio and performance.
4. **Incremental Backups**: For large models (>10MB), use incremental backups to save storage space.
5. **Backup Rotation**: Configure `max_backups` based on your retention policy requirements.
6. **Pre-Deployment Backups**: Always create a backup before deploying a new model version.
7. **Test Restores**: Periodically test the restore process to ensure backups are valid.

## Error Handling

The system includes comprehensive error handling with specific exception types:

- **EnhancedModelBackupError**: General errors related to backup operations.
- **BackupVerificationError**: Errors that occur during backup verification.

All errors are logged with detailed information to help diagnose and resolve issues.

## Integration with Existing Systems

The enhanced backup system is fully compatible with the existing model serialization and versioning systems. It extends the base `ModelBackupManager` class, so it can be used as a drop-in replacement with additional features.

## Performance Considerations

- **Compression**: Higher compression levels provide better space savings but take longer to process.
- **Incremental Backups**: Significantly reduce backup size and time for large models with small changes.
- **Verification**: Adds a small overhead but provides important data integrity guarantees.
- **Scheduled Backups**: Run in a separate thread to avoid blocking the main application.

## Future Enhancements

- Cloud storage integration for off-site backups
- Differential backup support (storing changes from the original version)
- Backup encryption for sensitive models
- Distributed backup support for very large models