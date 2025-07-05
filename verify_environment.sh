#!/bin/bash

echo "Friday AI Trading System - Environment Verification"
echo "==================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[WARNING] Virtual environment not found."
    echo "Run setup_environment.sh first to set up the environment."
    exit 1
fi

# Make script executable
chmod +x verify_environment.py

# Activate virtual environment and run verification
source venv/bin/activate
python3 verify_environment.py
RESULT=$?

# Deactivate virtual environment
deactivate

if [ $RESULT -ne 0 ]; then
    echo "[ERROR] Environment verification failed."
    exit 1
fi

echo ""
echo "Environment verification completed successfully."
echo ""