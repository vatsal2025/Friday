# Friday AI Trading System - Production Setup Guide

This guide provides detailed instructions for setting up the Friday AI Trading System in a production environment. It covers installation, configuration, database setup, security considerations, and deployment best practices.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Database Setup](#database-setup)
4. [Configuration](#configuration)
5. [Security Considerations](#security-considerations)
6. [Deployment](#deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ SSD recommended
- **Network**: Stable internet connection with low latency

### Software Requirements

- **Operating System**: Ubuntu 20.04 LTS or later (recommended), Windows Server 2019+, or macOS 12+
- **Python**: 3.10 or later
- **MongoDB**: 5.0 or later
- **Redis**: 6.2 or later
- **Nginx**: 1.20 or later (for production API deployment)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/friday-ai-trading.git
cd friday-ai-trading
```

### 2. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

Alternatively, use the provided installation script:

```bash
python install_dependencies.py
```

### 3. Verify Installation

```bash
python verify_environment.py
```

## Database Setup

### MongoDB Setup

#### Installation

**Ubuntu/Debian:**

```bash
# Import MongoDB public GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -

# Create list file for MongoDB
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list

# Update package database and install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod
```

**Windows:**

Download and install MongoDB from the [official website](https://www.mongodb.com/try/download/community).

#### Security Configuration

1. Create admin user:

```javascript
use admin
db.createUser(
  {
    user: "adminUser",
    pwd: "securePassword",
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
```

2. Enable authentication in MongoDB configuration file (`/etc/mongod.conf` on Linux):

```yaml
security:
  authorization: enabled
```

3. Restart MongoDB service:

```bash
sudo systemctl restart mongod
```

### Redis Setup

#### Installation

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Windows:**

Download and install Redis from [Redis for Windows](https://github.com/microsoftarchive/redis/releases).

#### Security Configuration

1. Set password in Redis configuration file (`/etc/redis/redis.conf` on Linux):

```
requirepass secureRedisPassword
```

2. Disable dangerous commands:

```
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
rename-command SHUTDOWN ""
```

3. Restart Redis service:

```bash
sudo systemctl restart redis-server
```

### Verify Database Setup

Use the provided verification script:

```bash
python verify_databases.py
```

## Configuration

### Update Configuration File

Edit `unified_config.py` to set appropriate values for your production environment:

```python
# MongoDB Configuration
MONGODB_CONFIG = {
    "host": "your-mongodb-host",
    "port": 27017,
    "database": "friday_trading",
    "username": "your-mongodb-username",
    "password": "your-mongodb-password",
    "auth_source": "admin",
    "ssl": True,
    "ssl_cert_reqs": "CERT_REQUIRED",
    "ssl_ca_certs": "/path/to/ca.pem"
}

# Redis Configuration
REDIS_CONFIG = {
    "host": "your-redis-host",
    "port": 6379,
    "password": "your-redis-password",
    "ssl": True,
    "ssl_cert_reqs": "CERT_REQUIRED",
    "ssl_ca_certs": "/path/to/ca.pem"
}

# API Server Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "log_level": "info",
    "ssl_keyfile": "/path/to/key.pem",
    "ssl_certfile": "/path/to/cert.pem"
}
```

### Environment Variables

For sensitive information, use environment variables instead of hardcoding values in the configuration file:

```bash
export MONGODB_USERNAME="your-mongodb-username"
export MONGODB_PASSWORD="your-mongodb-password"
export REDIS_PASSWORD="your-redis-password"
export API_SECRET_KEY="your-api-secret-key"
```

Update `unified_config.py` to use environment variables:

```python
import os

MONGODB_CONFIG = {
    # ...
    "username": os.environ.get("MONGODB_USERNAME"),
    "password": os.environ.get("MONGODB_PASSWORD"),
    # ...
}
```

## Security Considerations

### Network Security

1. **Firewall Configuration**:
   - Allow only necessary ports (27017 for MongoDB, 6379 for Redis, 8000 for API)
   - Restrict access to trusted IP addresses

2. **VPN or Private Network**:
   - Consider running databases in a private network
   - Use VPN for secure remote access

### Data Security

1. **Encryption**:
   - Enable TLS/SSL for all connections
   - Use encrypted storage for sensitive data

2. **Access Control**:
   - Implement role-based access control
   - Use strong, unique passwords
   - Regularly rotate credentials

3. **Secrets Management**:
   - Use a secrets management solution (e.g., HashiCorp Vault, AWS Secrets Manager)
   - Never store secrets in code or version control

### Application Security

1. **API Security**:
   - Implement rate limiting
   - Use JWT or OAuth for authentication
   - Validate all inputs

2. **Dependency Security**:
   - Regularly update dependencies
   - Use a dependency scanning tool

## Deployment

### Option 1: Systemd Service (Linux)

1. Create a systemd service file `/etc/systemd/system/friday-trading.service`:

```ini
[Unit]
Description=Friday AI Trading System
After=network.target mongodb.service redis.service

[Service]
User=friday
WorkingDirectory=/path/to/friday-ai-trading
Environment="PATH=/path/to/friday-ai-trading/venv/bin"
ExecStart=/path/to/friday-ai-trading/venv/bin/python run_friday.py --all
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

2. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable friday-trading.service
sudo systemctl start friday-trading.service
```

### Option 2: Docker Deployment

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_friday.py", "--all"]
```

2. Create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  mongodb:
    image: mongo:5.0
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGODB_USERNAME}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGODB_PASSWORD}
    ports:
      - "27017:27017"

  redis:
    image: redis:6.2
    volumes:
      - redis_data:/data
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"

  friday-trading:
    build: .
    depends_on:
      - mongodb
      - redis
    environment:
      - MONGODB_USERNAME=${MONGODB_USERNAME}
      - MONGODB_PASSWORD=${MONGODB_PASSWORD}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - API_SECRET_KEY=${API_SECRET_KEY}
    ports:
      - "8000:8000"

volumes:
  mongodb_data:
  redis_data:
```

3. Build and run with Docker Compose:

```bash
docker-compose up -d
```

### Option 3: Kubernetes Deployment

For large-scale deployments, consider using Kubernetes. Create the necessary manifests for deployments, services, and persistent volumes.

## Monitoring and Maintenance

### Logging

1. **Centralized Logging**:
   - Configure logging to send logs to a centralized system (e.g., ELK Stack, Graylog)
   - Set appropriate log levels and rotation policies

2. **Log Analysis**:
   - Regularly review logs for errors and anomalies
   - Set up alerts for critical errors

### Monitoring

1. **System Monitoring**:
   - Monitor CPU, memory, disk, and network usage
   - Use tools like Prometheus, Grafana, or cloud provider monitoring services

2. **Application Monitoring**:
   - Monitor API response times and error rates
   - Track database performance and connection pools

3. **Alerting**:
   - Set up alerts for system and application issues
   - Configure on-call rotations for critical alerts

### Backup and Recovery

1. **Database Backups**:
   - Set up regular MongoDB backups
   - Test backup restoration procedures

2. **Disaster Recovery**:
   - Document disaster recovery procedures
   - Regularly test recovery scenarios

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Check network connectivity
   - Verify credentials and authentication settings
   - Check firewall rules

2. **API Server Issues**:
   - Check logs for errors
   - Verify port availability
   - Check for dependency conflicts

3. **MCP Server Issues**:
   - Check logs for errors
   - Verify memory and storage availability
   - Check network connectivity

### Diagnostic Tools

1. **Database Diagnostics**:
   - Use MongoDB Compass for MongoDB diagnostics
   - Use Redis CLI for Redis diagnostics

2. **System Diagnostics**:
   - Use `htop`, `iotop`, and `netstat` for system diagnostics
   - Use `journalctl` to view systemd service logs

3. **Application Diagnostics**:
   - Use the provided verification scripts
   - Check application logs