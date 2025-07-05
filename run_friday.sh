#!/bin/bash
# Friday AI Trading System Launcher
# This shell script runs the Friday AI Trading System

echo "==================================================="
echo "Friday AI Trading System Launcher"
echo "==================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Check if MongoDB is running
echo "Checking if MongoDB is running..."
if ! nc -z localhost 27017 &> /dev/null; then
    echo "Warning: Cannot connect to MongoDB on port 27017."
    echo "Please make sure MongoDB is installed and running."
    read -p "Do you want to continue anyway? (y/n): " continue
    if [[ "$continue" != "y" ]]; then
        exit 1
    fi
fi

# Check if Redis is running
echo "Checking if Redis is running..."
if ! nc -z localhost 6379 &> /dev/null; then
    echo "Warning: Cannot connect to Redis on port 6379."
    echo "Please make sure Redis is installed and running."
    read -p "Do you want to continue anyway? (y/n): " continue
    if [[ "$continue" != "y" ]]; then
        exit 1
    fi
fi

# Display menu
echo ""
echo "Please select an option:"
echo "1. Initialize databases"
echo "2. Start MCP servers"
echo "3. Start API server"
echo "4. Run all components"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " option

case $option in
    1)
        echo "Initializing databases..."
        python3 run_friday.py --init-db
        ;;
    2)
        echo "Starting MCP servers..."
        python3 run_friday.py --start-mcp
        ;;
    3)
        echo "Starting API server..."
        python3 run_friday.py --start-api
        ;;
    4)
        echo "Running all components..."
        python3 run_friday.py --all
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please try again."
        exit 1
        ;;
esac

echo ""
echo "Operation completed."