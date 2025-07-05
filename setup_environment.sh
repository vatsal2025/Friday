#!/bin/bash

echo "Friday AI Trading System - Environment Setup"
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "[ERROR] Python 3.10+ is required, but $PYTHON_VERSION is installed."
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Make script executable
chmod +x setup_environment.py

# Run the setup script
python3 setup_environment.py

if [ $? -ne 0 ]; then
    echo "[ERROR] Environment setup failed."
    exit 1
fi

echo ""
echo "To activate the virtual environment, run: source venv/bin/activate"
echo ""