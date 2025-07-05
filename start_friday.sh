#!/bin/bash

echo "===================================================="
echo "Friday AI Trading System - Startup Script"
echo "===================================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in the PATH."
    echo "Please install Python 3.10 or later and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or later is required."
    echo "Current version: $PYTHON_VERSION"
    echo "Please install Python 3.10 or later and try again."
    exit 1
fi

echo "Using Python version: $PYTHON_VERSION"
echo

# Check if MongoDB is running
echo "Checking if MongoDB is running..."
if ! python3 -c "import pymongo; pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000).admin.command('ping')" &> /dev/null; then
    echo "Warning: MongoDB is not running."
    echo
    
    read -p "Do you want to start MongoDB now? (y/n): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "Starting MongoDB..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start mongod
            if [ $? -ne 0 ]; then
                echo "Failed to start MongoDB as a service."
                echo "Please start MongoDB manually and try again."
                echo "You can start MongoDB by running: mongod --dbpath /data/db"
                exit 1
            fi
        else
            echo "systemctl not found. Trying to start MongoDB directly..."
            mongod --dbpath /data/db &
            if [ $? -ne 0 ]; then
                echo "Failed to start MongoDB directly."
                echo "Please start MongoDB manually and try again."
                echo "You can start MongoDB by running: mongod --dbpath /data/db"
                exit 1
            fi
        fi
        echo "MongoDB started successfully."
    else
        echo "MongoDB is required for the Friday AI Trading System to function properly."
        echo "Please start MongoDB manually and try again."
        exit 1
    fi
else
    echo "MongoDB is running."
fi

# Check if Redis is running
echo "Checking if Redis is running..."
if ! python3 -c "import redis; redis.Redis(host='localhost', port=6379, socket_timeout=2).ping()" &> /dev/null; then
    echo "Warning: Redis is not running."
    echo
    
    read -p "Do you want to start Redis now? (y/n): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "Starting Redis..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis
            if [ $? -ne 0 ]; then
                echo "Failed to start Redis as a service."
                echo "Please start Redis manually and try again."
                echo "You can start Redis by running: redis-server"
                exit 1
            fi
        else
            echo "systemctl not found. Trying to start Redis directly..."
            redis-server &
            if [ $? -ne 0 ]; then
                echo "Failed to start Redis directly."
                echo "Please start Redis manually and try again."
                echo "You can start Redis by running: redis-server"
                exit 1
            fi
        fi
        echo "Redis started successfully."
    else
        echo "Redis is required for the Friday AI Trading System to function properly."
        echo "Please start Redis manually and try again."
        exit 1
    fi
else
    echo "Redis is running."
fi

echo
echo "What would you like to do?"
echo "1. Initialize databases"
echo "2. Start MCP servers"
echo "3. Start API server"
echo "4. Start all components"
echo "5. Exit"
echo

read -p "Enter your choice (1-5): " CHOICE

case $CHOICE in
    1)
        echo
        read -p "Do you want to force recreation of databases and create test data? (y/n): " recreate
        if [[ "$recreate" =~ ^[Yy]$ ]]; then
            echo "Initializing databases with test data..."
            python3 start_friday.py --init-db --force-recreate
        else
            echo "Initializing databases..."
            python3 start_friday.py --init-db
        fi
        ;;
    2)
        echo "Starting MCP servers..."
        python3 start_friday.py --start-mcp &
        ;;
    3)
        echo "Starting API server..."
        python3 start_friday.py --start-api &
        ;;
    4)
        echo "Starting all components..."
        python3 start_friday.py --all
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        exit 1
        ;;
esac

echo
echo "Done."