# Friday AI Trading System - System Verification Guide

This document provides instructions for verifying and checking the status of your Friday AI Trading System installation. These tools help ensure that all components are properly configured and running.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Environment Verification](#python-environment-verification)
3. [System Status Check](#system-status-check)
4. [Starting the System](#starting-the-system)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

Before using these verification tools, ensure you have:

- Python 3.10 or later installed
- MongoDB installed and configured
- Redis installed and configured
- All dependencies installed (see `requirements.txt`)

## Python Environment Verification

The Friday AI Trading System requires Python 3.10 or later and several specific packages. You can verify your Python environment using the provided scripts.

### Windows

```bash
verify_python.bat
```

### Linux/macOS

```bash
./verify_python.sh
```

### Manual Verification

You can also run the Python verification script directly:

```bash
python verify_python.py
```

This script checks:

- Python version (must be 3.10+)
- Required packages installation status
- Optional packages installation status
- Virtual environment status
- System information

If any required packages are missing, the script will provide instructions for installing them.

## System Status Check

The system status check tool verifies that all components of the Friday AI Trading System are running correctly.

```bash
python check_system_status.py
```

This script checks the status of:

- MongoDB connection
- Redis connection
- MCP servers (Memory and Sequential Thinking)
- API server

The script provides detailed information about each component, including:

- Connection status
- Host and port information
- Database and collection existence
- Key counts and data availability
- Error messages (if any)

A summary table is displayed at the end, showing the overall status of each component.

## Starting the System

Once you've verified that your environment meets all requirements, you can start the Friday AI Trading System using the provided scripts.

### Windows

```bash
start_friday.bat
```

### Linux/macOS

```bash
./start_friday.sh
```

### Manual Start

You can also use the Python script directly:

```bash
python start_friday.py --all
```

The start script provides several options:

- `--init-db`: Initialize databases
- `--force-recreate`: Force recreation of databases and create test data
- `--start-mcp`: Start MCP servers
- `--start-api`: Start API server
- `--all`: Start all components

## Troubleshooting

### MongoDB Issues

If MongoDB is not running or cannot be connected to:

1. Verify MongoDB is installed: `mongod --version`
2. Check if MongoDB service is running:
   - Windows: `net start MongoDB`
   - Linux/macOS: `sudo systemctl status mongod`
3. Start MongoDB manually:
   - Windows: `mongod --dbpath C:\data\db`
   - Linux/macOS: `mongod --dbpath /data/db`

### Redis Issues

If Redis is not running or cannot be connected to:

1. Verify Redis is installed: `redis-server --version`
2. Check if Redis service is running:
   - Windows: `net start Redis`
   - Linux/macOS: `sudo systemctl status redis`
3. Start Redis manually: `redis-server`

### Python Environment Issues

If you encounter Python environment issues:

1. Verify Python version: `python --version` or `python3 --version`
2. Create a new virtual environment:
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on Linux/macOS
   source venv/bin/activate
   ```
3. Install required packages: `pip install -r requirements.txt`

### MCP Server Issues

If MCP servers are not starting or running correctly:

1. Check for port conflicts: `netstat -ano | findstr 8001` (Windows) or `netstat -ano | grep 8001` (Linux/macOS)
2. Verify the configuration in `unified_config.py`
3. Start MCP servers manually: `python src/mcp_servers.py`
4. Check logs for error messages

### API Server Issues

If the API server is not starting or running correctly:

1. Check for port conflicts: `netstat -ano | findstr 8000` (Windows) or `netstat -ano | grep 8000` (Linux/macOS)
2. Verify the configuration in `unified_config.py`
3. Start API server manually: `python -m uvicorn src.application.api.main:app --host 0.0.0.0 --port 8000`
4. Check logs for error messages

---

For more detailed information, refer to the `README.md` and `SETUP_INSTRUCTIONS.md` files in the project root directory.