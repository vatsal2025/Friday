import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import argparse
import logging

# Try to import config, fall back to defaults
try:
    from src.infrastructure.config.unified_config import KNOWLEDGE_CONFIG
    MAX_BACKUPS = KNOWLEDGE_CONFIG.get('max_backups', 10)
except ImportError:
    MAX_BACKUPS = 10

# Configuration
BACKUP_DIR = Path('storage/backups')
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Helper functions
def rotate_backups(directory: Path, max_backups: int):
    backups = sorted(directory.iterdir(), key=os.path.getmtime, reverse=True)
    for old_backup in backups[max_backups:]:
        if old_backup.is_dir():
            shutil.rmtree(old_backup)
        else:
            old_backup.unlink()

def backup_mongo():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / 'mongo' / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'mongodump',
        '--out', str(backup_path)
    ]
    subprocess.run(cmd, check=True)

    # Compress backup
    shutil.make_archive(str(backup_path), 'gztar', root_dir=backup_path)
    shutil.rmtree(backup_path)

    rotate_backups(BACKUP_DIR / 'mongo', MAX_BACKUPS)


def backup_redis():
    backup_path = BACKUP_DIR / 'redis'
    backup_path.mkdir(parents=True, exist_ok=True)

    # Using redis-cli --rdb
    rdb_backup_path = backup_path / 'dump.rdb'
    subprocess.run(['redis-cli', 'save'], check=True)

    redis_dir = subprocess.run(['redis-cli', 'config', 'get', 'dir'],
                                capture_output=True, text=True).stdout.splitlines()[1]
    shutil.move(os.path.join(redis_dir, 'dump.rdb'), rdb_backup_path)

    rotate_backups(backup_path, MAX_BACKUPS)


def main():
    if len(sys.argv) < 2:
        print("Usage: backup_databases.py [mongo|redis]")
        sys.exit(1)

    db_type = sys.argv[1].lower()
    if db_type == "mongo":
        backup_mongo()
    elif db_type == "redis":
        backup_redis()
    else:
        print("Invalid option. Choose 'mongo' or 'redis'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

