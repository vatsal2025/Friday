# Friday AI Trading System - Troubleshooting Guide

This document provides solutions for common issues you might encounter when setting up and running the Friday AI Trading System.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Connection Issues](#database-connection-issues)
3. [MCP Server Issues](#mcp-server-issues)
4. [API Server Issues](#api-server-issues)
5. [Trading Issues](#trading-issues)
6. [Performance Issues](#performance-issues)
7. [Data Issues](#data-issues)
8. [Common Error Messages](#common-error-messages)
9. [Getting Help](#getting-help)

## Installation Issues

### Python Version Compatibility

**Issue**: Error messages about incompatible Python version.

**Solution**:
- Ensure you have Python 3.10 or later installed
- Check your Python version with `python --version` or `python3 --version`
- If needed, install the correct Python version from [python.org](https://www.python.org/downloads/)

```bash
# Verify Python version
python --version

# Run the Python verification script
python verify_python.py
```

### Package Installation Failures

**Issue**: Errors when installing required packages.

**Solution**:
- Ensure you have the latest pip version: `python -m pip install --upgrade pip`
- Try installing packages one by one to identify problematic packages
- Check for system dependencies (e.g., C++ build tools on Windows)
- Use the provided installation script: `python install_dependencies.py`

```bash
# Update pip
python -m pip install --upgrade pip

# Install dependencies with verbose output
python -m pip install -r requirements.txt -v
```

### Virtual Environment Issues

**Issue**: Problems with virtual environment creation or activation.

**Solution**:
- Ensure you have the venv module installed
- Try creating a new virtual environment in a different location
- Check for permission issues in the directory

```bash
# Create a new virtual environment
python -m venv new_venv

# Activate on Windows
new_venv\Scripts\activate

# Activate on Linux/macOS
source new_venv/bin/activate
```

## Database Connection Issues

### MongoDB Connection Failures

**Issue**: Cannot connect to MongoDB.

**Solution**:
- Verify MongoDB is installed and running
- Check MongoDB connection settings in `unified_config.py`
- Ensure MongoDB is listening on the configured host and port
- Check for authentication issues if MongoDB requires authentication

```bash
# Check if MongoDB is running
# Windows
net start MongoDB

# Linux/macOS
systemctl status mongod

# Start MongoDB manually if needed
mongod --dbpath /data/db
```

### Redis Connection Failures

**Issue**: Cannot connect to Redis.

**Solution**:
- Verify Redis is installed and running
- Check Redis connection settings in `unified_config.py`
- Ensure Redis is listening on the configured host and port
- Check for authentication issues if Redis requires a password

```bash
# Check if Redis is running
# Windows
net start Redis

# Linux/macOS
systemctl status redis

# Start Redis manually if needed
redis-server
```

### Database Initialization Failures

**Issue**: Errors when initializing databases.

**Solution**:
- Check database connection settings
- Ensure you have the necessary permissions to create databases and collections
- Try running the database setup script with force recreation: `python start_friday.py --init-db --force-recreate`

## MCP Server Issues

### MCP Server Startup Failures

**Issue**: MCP servers fail to start.

**Solution**:
- Check MCP server configuration in `unified_config.py`
- Verify that the required ports are available and not in use by other applications
- Check for error messages in the console or log files
- Ensure Redis is running (required for MCP servers)

```bash
# Check if ports are in use
# Windows
netstat -ano | findstr 8001
netstat -ano | findstr 8002

# Linux/macOS
netstat -ano | grep 8001
netstat -ano | grep 8002
```

### Memory MCP Server Issues

**Issue**: Memory MCP server not functioning correctly.

**Solution**:
- Check Redis connection (Memory MCP server uses Redis for storage)
- Verify Memory MCP server configuration
- Restart the Memory MCP server
- Check for error messages in the logs

### Sequential Thinking MCP Server Issues

**Issue**: Sequential Thinking MCP server not functioning correctly.

**Solution**:
- Check Redis connection
- Verify Sequential Thinking MCP server configuration
- Restart the Sequential Thinking MCP server
- Check for error messages in the logs

## API Server Issues

### API Server Startup Failures

**Issue**: API server fails to start.

**Solution**:
- Check API server configuration in `unified_config.py`
- Verify that the required port is available and not in use by other applications
- Check for error messages in the console or log files
- Ensure MongoDB and Redis are running (required for the API server)

```bash
# Check if port is in use
# Windows
netstat -ano | findstr 8000

# Linux/macOS
netstat -ano | grep 8000
```

### API Endpoint Errors

**Issue**: API endpoints return errors.

**Solution**:
- Check API server logs for error messages
- Verify that the required databases and collections exist
- Check authentication settings if authentication is enabled
- Ensure MCP servers are running if the API endpoint requires them

## Trading Issues

### Strategy Execution Failures

**Issue**: Trading strategies fail to execute.

**Solution**:
- Check broker configuration in `unified_config.py`
- Verify that you have provided valid API keys for your broker
- Check for error messages in the logs
- Ensure the strategy is properly configured

### Order Placement Failures

**Issue**: Orders fail to be placed with the broker.

**Solution**:
- Check broker connection settings
- Verify that you have sufficient funds in your account
- Check for any broker-specific restrictions or limitations
- Ensure the order parameters are valid

### Position Sizing Issues

**Issue**: Incorrect position sizes.

**Solution**:
- Check risk management settings in `unified_config.py`
- Verify the position sizing calculation in the code
- Ensure you have provided valid account balance information
- Check for any overrides in the strategy configuration

## Performance Issues

### Slow System Performance

**Issue**: The system runs slowly.

**Solution**:
- Check system resource usage (CPU, memory, disk I/O)
- Optimize database queries and indexes
- Consider increasing cache usage
- Check for any resource-intensive operations in your code

```bash
# Check system resource usage
# Windows
Task Manager

# Linux/macOS
top
htop
```

### Memory Leaks

**Issue**: The system uses increasing amounts of memory over time.

**Solution**:
- Check for memory leaks in your code
- Ensure objects are properly garbage collected
- Consider implementing periodic restarts for long-running processes
- Monitor memory usage over time

### Database Performance Issues

**Issue**: Slow database operations.

**Solution**:
- Optimize MongoDB indexes
- Check for slow queries in the MongoDB logs
- Consider increasing cache usage for frequently accessed data
- Optimize data models and query patterns

## Data Issues

### Missing Market Data

**Issue**: Market data is missing or incomplete.

**Solution**:
- Check market data sources and API keys
- Verify that the data collection process is running
- Check for any rate limits or restrictions from data providers
- Consider using alternative data sources

### Data Synchronization Issues

**Issue**: Data is not synchronized between components.

**Solution**:
- Check cache invalidation settings
- Verify that all components are using the same data sources
- Ensure data is properly persisted to the database
- Check for any race conditions in the code

### Data Quality Issues

**Issue**: Poor data quality (outliers, missing values, etc.).

**Solution**:
- Implement data validation and cleaning procedures
- Check for any issues with data sources
- Consider implementing data quality monitoring
- Adjust outlier detection and handling settings

## Common Error Messages

### "MongoDB connection failed"

**Cause**: MongoDB is not running or connection settings are incorrect.

**Solution**:
- Start MongoDB if it's not running
- Check MongoDB connection settings in `unified_config.py`
- Verify that MongoDB is listening on the configured host and port

### "Redis connection failed"

**Cause**: Redis is not running or connection settings are incorrect.

**Solution**:
- Start Redis if it's not running
- Check Redis connection settings in `unified_config.py`
- Verify that Redis is listening on the configured host and port

### "MCP server failed to start"

**Cause**: MCP server configuration issues or port conflicts.

**Solution**:
- Check MCP server configuration in `unified_config.py`
- Verify that the required ports are available
- Check for error messages in the logs

### "API server failed to start"

**Cause**: API server configuration issues or port conflicts.

**Solution**:
- Check API server configuration in `unified_config.py`
- Verify that the required port is available
- Check for error messages in the logs

### "Broker authentication failed"

**Cause**: Invalid broker API keys or authentication issues.

**Solution**:
- Check broker API keys in `unified_config.py` or environment variables
- Verify that your broker account is active and in good standing
- Check for any IP restrictions or security settings on your broker account

## Getting Help

If you encounter issues not covered in this guide, you can get help through the following channels:

- **GitHub Issues**: Submit an issue on the GitHub repository
- **Documentation**: Check the comprehensive documentation in the project directory
- **Community Forums**: Join the community forums for discussion and support
- **Contact the Developers**: Reach out to the development team directly

### Reporting Issues

When reporting issues, please include the following information:

- Detailed description of the issue
- Steps to reproduce the issue
- Error messages and stack traces
- System information (OS, Python version, etc.)
- Logs and other relevant output

### Logs

Log files can be found in the `logs` directory. These logs can be invaluable for diagnosing issues.

```bash
# View the most recent log file
# Windows
type logs\friday.log

# Linux/macOS
cat logs/friday.log
```

---

For more information, refer to the `README.md`, `SETUP_INSTRUCTIONS.md`, and other documentation files in the project root directory.