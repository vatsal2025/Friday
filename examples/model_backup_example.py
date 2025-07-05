"""Example script demonstrating the use of the enhanced model backup manager.

This script shows how to use the EnhancedModelBackupManager to create backups,
schedule automatic backups, perform rollbacks, and verify backups.
"""

import os
import time
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.services.model.enhanced_model_backup import EnhancedModelBackupManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_model(model_path, n_estimators=10):
    """Create and save a sample model for demonstration."""
    # Create a simple random forest model
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Created sample model with {n_estimators} estimators at {model_path}")
    return model


def modify_model(model_path, n_estimators=20):
    """Modify the existing model to simulate changes."""
    # Load the model
    model = joblib.load(model_path)
    
    # Modify the model by training on new data
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    model.n_estimators = n_estimators
    model.fit(X, y)
    
    # Save the modified model
    joblib.dump(model, model_path)
    logger.info(f"Modified model with {n_estimators} estimators at {model_path}")
    return model


def backup_callback(model_name, success, error_message):
    """Callback function for scheduled backups."""
    if success:
        logger.info(f"Scheduled backup completed successfully for model {model_name}")
    else:
        logger.error(f"Scheduled backup failed for model {model_name}: {error_message}")


def main():
    """Main function demonstrating the use of the enhanced model backup manager."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    backups_dir = os.path.join(base_dir, "model_backups")
    model_path = os.path.join(models_dir, "trading_model.joblib")
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)
    
    # Create a sample model
    model = create_sample_model(model_path, n_estimators=10)
    
    # Initialize the enhanced model backup manager
    backup_manager = EnhancedModelBackupManager(
        backup_dir=backups_dir,
        max_backups=5,
        compression_level=6,  # Medium compression level
        verify_backups=True,
        incremental_threshold=10 * 1024 * 1024  # 10 MB
    )
    
    # Create an initial backup
    logger.info("Creating initial backup...")
    backup_path = backup_manager.create_backup(
        model_path=model_path,
        model_name="trading_model",
        version="1.0"
    )
    logger.info(f"Initial backup created at {backup_path}")
    
    # Modify the model and create another backup
    logger.info("Modifying the model...")
    model = modify_model(model_path, n_estimators=20)
    
    logger.info("Creating backup of modified model...")
    backup_path = backup_manager.create_backup(
        model_path=model_path,
        model_name="trading_model",
        version="1.1"
    )
    logger.info(f"Backup of modified model created at {backup_path}")
    
    # List available backups
    logger.info("Listing available backups...")
    backups = backup_manager.list_backups("trading_model")
    for backup in backups:
        logger.info(f"Backup: {backup['version']} - {backup['backup_time']}")
    
    # Get backup statistics
    logger.info("Getting backup statistics...")
    stats = backup_manager.get_backup_stats("trading_model")
    logger.info(f"Total backups: {stats['total_backups']}")
    logger.info(f"Total size: {stats['total_size'] / 1024:.2f} KB")
    logger.info(f"Compressed size: {stats['compressed_size'] / 1024:.2f} KB")
    
    # Demonstrate rollback
    logger.info("Rolling back to version 1.0...")
    restored_path = backup_manager.rollback(
        model_name="trading_model",
        version="1.0"
    )
    logger.info(f"Rolled back to version 1.0 at {restored_path}")
    
    # Verify the rollback was successful
    restored_model = joblib.load(restored_path)
    logger.info(f"Restored model has {restored_model.n_estimators} estimators (should be 10)")
    
    # Schedule automatic backups
    logger.info("Scheduling automatic backups...")
    backup_manager.schedule_backups(
        model_paths={
            "trading_model": {
                "path": model_path,
                "version": "1.0"
            }
        },
        interval_hours=24,  # Daily backups
        callback=backup_callback
    )
    logger.info("Automatic backups scheduled")
    
    # In a real application, the program would continue running
    # For this example, we'll just wait a few seconds to simulate
    logger.info("Waiting for scheduled backup to run (in a real application, this would run in the background)...")
    time.sleep(5)  # In a real application, this would be unnecessary
    
    # Stop scheduled backups
    logger.info("Stopping scheduled backups...")
    backup_manager.stop_scheduled_backups(["trading_model"])
    logger.info("Scheduled backups stopped")
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()