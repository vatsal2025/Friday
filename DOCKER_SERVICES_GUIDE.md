# Friday AI Trading System - Docker Services Guide

This guide explains how to set up and manage MongoDB and Redis services for the Friday AI Trading System using Docker.

## Overview

The Friday AI Trading System uses:
- **MongoDB 6.0.17** for persistent data storage
- **Redis 7.2.5** for caching and session management

Both services are containerized using Docker and configured with production-ready settings, health checks, and persistent data volumes.

## Prerequisites

1. **Docker Desktop** installed and running
2. **Docker Compose** available (included with Docker Desktop)
3. **Python 3.7+** for health check utilities

### Installation Links
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Docker Desktop for macOS](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Engine for Linux](https://docs.docker.com/engine/install/)

## Quick Start

### 1. Setup Environment
```bash
# Windows
manage_docker_services.bat setup

# Linux/macOS
chmod +x manage_docker_services.sh
./manage_docker_services.sh setup
```

### 2. Check Status
```bash
# Windows
manage_docker_services.bat status

# Linux/macOS
./manage_docker_services.sh status
```

## Service Configuration

### MongoDB Configuration
- **Image**: `mongo:6.0.17`
- **Port**: `27017` (mapped from container to host)
- **Database**: `friday`
- **Authentication**: Enabled with admin user
- **Data Volume**: `./storage/data/mongodb:/data/db`
- **Config Volume**: `./storage/data/mongodb/configdb:/data/configdb`

### Redis Configuration
- **Image**: `redis:7.2.5-alpine`
- **Port**: `6379` (mapped from container to host)
- **Memory Limit**: `512MB`
- **Persistence**: AOF (Append Only File) enabled
- **Data Volume**: `./storage/data/redis:/data`
- **Eviction Policy**: `allkeys-lru`

## Available Commands

### Windows (manage_docker_services.bat)
```cmd
manage_docker_services.bat [COMMAND]
```

### Linux/macOS (manage_docker_services.sh)
```bash
./manage_docker_services.sh [COMMAND]
```

### Commands:

| Command | Description |
|---------|-------------|
| `start` | Start MongoDB and Redis services |
| `stop` | Stop MongoDB and Redis services |
| `restart` | Restart MongoDB and Redis services |
| `status` | Show status of services |
| `logs` | Show logs from services |
| `clean` | Stop and remove containers, networks, and volumes |
| `setup` | Create required directories and start services |
| `health` | Check health status of services |
| `help` | Show help message |

## Health Checks

Both services include comprehensive health checks:

### MongoDB Health Check
- **Command**: `mongosh --eval "db.adminCommand('ping')"`
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 5
- **Start Period**: 40 seconds

### Redis Health Check
- **Command**: `redis-cli ping`
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 5
- **Start Period**: 10 seconds

## Service Readiness Utility

The `wait_for_services.py` script provides automated health checking:

```bash
# Wait for all services (default timeout: 300s)
python wait_for_services.py

# Wait for specific services
python wait_for_services.py mongodb redis

# Custom timeout
python wait_for_services.py --timeout 60

# Verbose output
python wait_for_services.py --verbose
```

## Data Persistence

### Directory Structure
```
storage/
├── data/
│   ├── mongodb/           # MongoDB data files
│   │   └── configdb/      # MongoDB configuration files
│   └── redis/             # Redis data files
```

### Backup Recommendations
1. **MongoDB**: Use `mongodump` for regular backups
2. **Redis**: Copy AOF files or use `BGSAVE` command
3. **Volume Backup**: Backup entire `storage/data/` directory

## Connection Details

### MongoDB Connection
```python
# From unified_config.py
MONGODB_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "db_name": "friday",
    "username": "admin",
    "password": "friday_mongo_password"
}

# Connection string
mongodb://admin:friday_mongo_password@localhost:27017/friday
```

### Redis Connection
```python
# From unified_config.py
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None
}

# Connection string
redis://localhost:6379/0
```

## Production Considerations

### Security
1. **Change default passwords** in production
2. **Enable authentication** for both services
3. **Use TLS/SSL** for connections
4. **Restrict network access** using firewall rules

### Performance
1. **MongoDB**: Consider replica sets for high availability
2. **Redis**: Monitor memory usage and adjust limits
3. **Volumes**: Use dedicated storage volumes for production

### Monitoring
1. Monitor container health status
2. Set up log aggregation
3. Monitor resource usage (CPU, memory, disk)
4. Set up alerts for service failures

## Troubleshooting

### Common Issues

#### Docker Not Running
```
Error: Docker is not installed or not running
```
**Solution**: Start Docker Desktop or Docker daemon

#### Port Already in Use
```
Error: Port 27017 is already in use
```
**Solution**: Stop existing MongoDB instances or change port mapping

#### Permission Denied
```
Error: Permission denied accessing storage directory
```
**Solution**: Fix directory permissions
```bash
# Linux/macOS
chmod -R 755 storage/data/

# Windows (Run as Administrator)
icacls storage\data /grant Everyone:F /T
```

#### Health Check Failures
```
Health check failing for MongoDB/Redis
```
**Solution**: 
1. Check container logs: `docker logs friday_mongodb`
2. Verify port accessibility
3. Check disk space for data volumes

### Viewing Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mongodb
docker-compose logs -f redis

# Using management script
./manage_docker_services.sh logs mongodb
```

### Manual Container Management
```bash
# View running containers
docker ps

# Check container health
docker inspect friday_mongodb --format="{{.State.Health.Status}}"
docker inspect friday_redis --format="{{.State.Health.Status}}"

# Access container shell
docker exec -it friday_mongodb mongosh
docker exec -it friday_redis redis-cli
```

## Integration with Friday System

### Configuration Updates
The services are configured to work with the existing `unified_config.py` settings. No changes to the configuration file are required.

### Startup Integration
Add the following to your main startup script:

```python
# Example integration
import subprocess
import sys

def wait_for_services():
    """Wait for Docker services to be ready."""
    try:
        result = subprocess.run(
            ["python", "wait_for_services.py"], 
            timeout=300, 
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        print("Services not ready")
        return False
    except subprocess.TimeoutExpired:
        print("Timeout waiting for services")
        return False

# In your main startup function
if not wait_for_services():
    print("Cannot start system - services not ready")
    sys.exit(1)
```

## Environment Variables

You can customize the setup using environment variables:

```bash
# Docker Compose project name
export COMPOSE_PROJECT_NAME=friday

# Data directory location
export FRIDAY_DATA_DIR=./storage/data

# MongoDB password
export MONGO_PASSWORD=your_secure_password

# Redis memory limit
export REDIS_MEMORY_LIMIT=1024mb
```

## Support

For issues related to Docker services:

1. Check this guide for troubleshooting steps
2. Review container logs for error messages
3. Verify Docker and docker-compose versions
4. Ensure sufficient disk space and memory
5. Check network connectivity and port availability

For service-specific issues:
- **MongoDB**: [MongoDB Documentation](https://docs.mongodb.com/)
- **Redis**: [Redis Documentation](https://redis.io/documentation)
- **Docker**: [Docker Documentation](https://docs.docker.com/)
