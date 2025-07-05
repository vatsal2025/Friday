# Friday AI Trading System - Environment Setup

This document provides instructions for setting up the development environment for the Friday AI Trading System.

## Prerequisites

- Python 3.10 or higher
- Git (for version control)
- Internet connection (for downloading dependencies)

## Setup Instructions

### Windows

1. Clone the repository (if you haven't already):
   ```
   git clone <repository-url>
   cd Friday
   ```

2. Run the setup script:
   ```
   setup_environment.bat
   ```

3. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

### macOS/Linux

1. Clone the repository (if you haven't already):
   ```
   git clone <repository-url>
   cd Friday
   ```

2. Make the setup script executable:
   ```
   chmod +x setup_environment.sh
   ```

3. Run the setup script:
   ```
   ./setup_environment.sh
   ```

4. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

## Environment Configuration

After running the setup script, a `.env` file will be created in the project root directory.

**Note:** API keys are not required for Phase 1 Task 1. You can configure them later when needed for other phases of the project.

## Manual Installation of Required Dependencies

Some dependencies need to be installed manually:

### TA-Lib (Required)

TA-Lib is a technical analysis library that is required for this project and needs special installation steps:

#### Windows

1. Download the appropriate wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
2. Install it using pip:
   ```
   pip install <downloaded-wheel-file>
   ```

#### macOS

```
brew install ta-lib
pip install ta-lib
```

#### Linux

```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

## Optional Dependencies

The following dependencies are optional and can be installed as needed for your specific use case:

```
pip install alpaca-trade-api ccxt mcp-server mcp-client
```

These packages are commented out in the requirements.txt file and can be installed manually when needed.

## Verifying the Installation

**Note:** TA-Lib is required for the verification to pass. Make sure to install it following the instructions above before running the verification script.

To verify that the environment is set up correctly:

1. Ensure the virtual environment is activated.
2. Run the verification script:
   ```
   python -c "import numpy, pandas, sklearn, matplotlib, yaml, requests, dotenv, fastapi, uvicorn, websocket, sqlalchemy, redis, pymongo, pydantic, aiohttp, asyncio, httpx; print('All imports successful!')"
   ```

## Troubleshooting

### Common Issues

1. **Python version not recognized**
   - Ensure Python 3.10+ is installed and in your PATH
   - Try using `python3` instead of `python` on macOS/Linux

2. **Dependency installation failures**
   - Check your internet connection
   - Try installing the problematic package individually
   - For system-level dependencies, you may need administrator privileges

3. **Virtual environment issues**
   - If the virtual environment creation fails, try creating it manually:
     ```
     python -m venv venv
     ```

### Getting Help

If you encounter issues not covered here, please:

1. Check the project documentation
2. Search for the error message online
3. Contact the development team for assistance

## Next Steps

After setting up your environment, you can proceed to the next tasks in Phase 1:

1. MCP Server Configuration
2. Database Setup
3. Storage Directory Configuration
4. Core Infrastructure Activation
5. Integration & Communication Setup