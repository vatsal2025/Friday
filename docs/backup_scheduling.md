# Database Backup Scheduling Guide

## Overview
This guide provides instructions for scheduling automated database backups for the Friday AI Trading System.

## Backup Script Usage

### Basic Usage
```bash
python backup_databases.py mongo    # Backup MongoDB
python backup_databases.py redis    # Backup Redis (if available)
```

### Features
- ✅ MongoDB backup using `mongodump` with gzip compression
- ✅ Automatic rotation based on `KNOWLEDGE_CONFIG["max_backups"]` (default: 10)
- ✅ Timestamped backup directories
- ✅ Error handling and logging

## Scheduling Options

### Linux/macOS - Crontab

#### Daily Backup at 2 AM
```bash
# Edit crontab
crontab -e

# Add this line for daily MongoDB backup at 2 AM
0 2 * * * cd /path/to/friday && python backup_databases.py mongo >> /var/log/friday_backup.log 2>&1
```

#### Weekly Backup on Sundays at 3 AM
```bash
0 3 * * 0 cd /path/to/friday && python backup_databases.py mongo >> /var/log/friday_backup.log 2>&1
```

### Windows - Task Scheduler

#### PowerShell Command to Create Daily Task
```powershell
$Action = New-ScheduledTaskAction -Execute "python" -Argument "backup_databases.py mongo" -WorkingDirectory "E:\Friday"
$Trigger = New-ScheduledTaskTrigger -Daily -At 2AM
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Friday MongoDB Backup" -Action $Action -Trigger $Trigger -Settings $Settings
```

#### Manual Task Creation via GUI
1. Open Task Scheduler (`taskschd.msc`)
2. Click "Create Basic Task"
3. Name: "Friday MongoDB Backup"
4. Trigger: Daily at 2:00 AM
5. Action: Start a program
   - Program: `python`
   - Arguments: `backup_databases.py mongo`
   - Start in: `E:\Friday`

### Docker Environment

#### Docker Compose with Backup Service
```yaml
version: '3.8'
services:
  friday-backup:
    build: .
    volumes:
      - ./storage/backups:/app/storage/backups
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
    command: >
      sh -c "echo '$BACKUP_SCHEDULE python backup_databases.py mongo' | crontab - && crond -f"
```

## Integration with MasterOrchestrator

### Scheduler Integration Example
```python
from src.orchestration.scheduler import MasterScheduler

scheduler = MasterScheduler()

# Schedule daily MongoDB backup
scheduler.add_job(
    func=lambda: subprocess.run(['python', 'backup_databases.py', 'mongo']),
    trigger='cron',
    hour=2,
    minute=0,
    id='mongodb_backup',
    name='Daily MongoDB Backup'
)
```

## Backup Storage Structure
```
storage/backups/
├── mongo/
│   ├── 20241228_020000.tar.gz
│   ├── 20241227_020000.tar.gz
│   └── ...
└── redis/  # If Redis is used
    ├── 20241228_020000.rdb
    └── ...
```

## Monitoring and Alerts

### Log Monitoring
- Backup logs are stored in the specified log files
- Monitor for "SUCCESS" or "ERROR" messages
- Set up log rotation to prevent disk space issues

### Recommended Monitoring
1. Disk space monitoring for backup directory
2. Backup job success/failure notifications
3. Backup file integrity checks
4. Automated restoration testing (monthly)

## Production Considerations

### Backup Verification
```bash
# Test MongoDB backup restoration
mongorestore --drop /path/to/backup/directory

# Verify backup integrity
mongodump --quiet --out /tmp/test_backup && echo "Backup integrity OK"
```

### Security
- Ensure backup directories have proper permissions (750 or 700)
- Consider encrypting backups for sensitive data
- Use secure transfer methods for off-site backups

### Performance
- Schedule backups during low-usage periods
- Monitor backup impact on system performance
- Consider using MongoDB replica sets for zero-downtime backups

## Troubleshooting

### Common Issues
1. **Permission denied**: Ensure the script has write access to backup directory
2. **mongodump not found**: Add MongoDB bin directory to PATH
3. **Disk space**: Monitor and clean up old backups automatically
4. **Network timeouts**: Increase MongoDB connection timeout settings

### Recovery Testing
Test backup recovery regularly:
```bash
# Create test database from backup
mongorestore --nsFrom="friday.*" --nsTo="friday_test.*" /path/to/backup
```
