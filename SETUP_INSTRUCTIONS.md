# Friday AI Trading System - Setup Instructions

## Prerequisites

### Required Software

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure Python is added to your PATH during installation

2. **MongoDB**
   - Download from [mongodb.com](https://www.mongodb.com/try/download/community)
   - Install and start the MongoDB service
   - Default port: 27017

3. **Redis**
   - **Windows**: Download from [Redis for Windows](https://github.com/microsoftarchive/redis/releases)
   - **Linux/macOS**: Install using package manager (apt, brew, etc.)
   - Start the Redis service
   - Default port: 6379

### Python Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Database Setup

### MongoDB Configuration

1. Create a MongoDB database named `friday_trading` (or use the name specified in `unified_config.py`)
2. No additional configuration is needed as the system will create all necessary collections

### Redis Configuration

1. Ensure Redis is running with default configuration
2. No additional configuration is needed as the system will create all necessary data structures

## Running the System

### Option 1: Using the Launcher Scripts

#### Windows

Run the batch script:

```
run_friday.bat
```

Select the desired option from the menu:
1. Initialize databases
2. Start MCP servers
3. Start API server
4. Run all components

#### Linux/macOS

Make the script executable and run it:

```bash
chmod +x run_friday.sh
./run_friday.sh
```

Select the desired option from the menu.

### Option 2: Using Python Directly

Run the Python script with appropriate arguments:

```bash
python run_friday.py --all  # Run all components
```

Or run specific components:

```bash
python run_friday.py --init-db  # Initialize databases
python run_friday.py --start-mcp  # Start MCP servers
python run_friday.py --start-api  # Start API server
```

Use `--force-recreate` to force recreation of database collections:

```bash
python run_friday.py --init-db --force-recreate
```

## Verifying the Setup

1. **Check MongoDB Collections**
   - Connect to MongoDB using a client like MongoDB Compass
   - Verify that all collections are created in the `friday_trading` database

2. **Check Redis Data Structures**
   - Use Redis CLI or a GUI client like Redis Desktop Manager
   - Verify that all namespaces and data structures are created

3. **Check API Server**
   - Open a web browser and navigate to `http://localhost:8000/health`
   - You should see a response indicating the API server is running

4. **Check MCP Servers**
   - The system will verify MCP server connections during startup
   - Check the logs for any connection errors

## Troubleshooting

### MongoDB Issues

- Ensure MongoDB service is running
- Check MongoDB connection string in `unified_config.py`
- Verify MongoDB port (default: 27017) is not blocked by firewall

### Redis Issues

- Ensure Redis service is running
- Check Redis connection settings in `unified_config.py`
- Verify Redis port (default: 6379) is not blocked by firewall

### MCP Server Issues

- Check MCP server logs for errors
- Verify MCP server configuration in `unified_config.py`
- Ensure required ports are not blocked by firewall

### API Server Issues

- Check API server logs for errors
- Verify API server configuration
- Ensure port 8000 is not blocked by firewall or used by another application

## Production Deployment Considerations

1. **Security**
   - Enable authentication for MongoDB and Redis
   - Use SSL/TLS for all connections
   - Set up proper firewall rules

2. **Performance**
   - Configure MongoDB and Redis for optimal performance
   - Consider using replica sets for MongoDB
   - Consider using Redis Cluster for high availability

3. **Monitoring**
   - Set up monitoring for all components
   - Configure alerts for critical errors
   - Implement logging to external systems

4. **Backup**
   - Set up regular backups for MongoDB
   - Configure Redis persistence
   - Implement disaster recovery procedures