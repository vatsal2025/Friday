#!/bin/bash

echo "Friday AI Trading System - Dependency Installation"
echo "============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please make sure venv module is available."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

echo ""
echo "Required dependencies installed successfully."
echo ""

# Optional dependencies menu
echo "Would you like to install optional dependencies?"
echo "1. Install TA-Lib (Technical Analysis Library)"
echo "2. Install Alpaca Trade API"
echo "3. Install CCXT (Cryptocurrency Exchange Trading Library)"
echo "4. Install MCP Server and Client"
echo "5. Install all optional dependencies"
echo "6. Skip optional dependencies"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Installing TA-Lib..."
        # Note: TA-Lib requires system dependencies
        echo "Note: TA-Lib may require additional system dependencies."
        echo "For Ubuntu/Debian: sudo apt-get install build-essential ta-lib"
        echo "For macOS: brew install ta-lib"
        pip install ta-lib
        ;;
    2)
        echo "Installing Alpaca Trade API..."
        pip install alpaca-trade-api
        ;;
    3)
        echo "Installing CCXT..."
        pip install ccxt
        ;;
    4)
        echo "Installing MCP Server and Client..."
        pip install mcp-server mcp-client
        ;;
    5)
        echo "Installing all optional dependencies..."
        echo "Note: TA-Lib may require additional system dependencies."
        echo "For Ubuntu/Debian: sudo apt-get install build-essential ta-lib"
        echo "For macOS: brew install ta-lib"
        pip install ta-lib alpaca-trade-api ccxt mcp-server mcp-client
        ;;
    6)
        echo "Skipping optional dependencies."
        ;;
    *)
        echo "Invalid choice. Skipping optional dependencies."
        ;;
esac

echo ""
echo "Installation completed successfully."
echo "To activate the virtual environment in the future, run: source venv/bin/activate"
echo ""

# Make the script executable
chmod +x start_mcp_servers.sh 2>/dev/null