#!/bin/bash

echo "===================================================="
echo "Friday AI Trading System - Python Environment Check"
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

echo "Using Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or later is required."
    echo "Current version: $PYTHON_VERSION"
    echo "Please install Python 3.10 or later and try again."
    exit 1
fi

echo "Python version check: PASSED"
echo

# Run the Python verification script
python3 verify_python.py