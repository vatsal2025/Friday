@echo off
echo Friday AI Trading System - MCP Servers Startup
echo =============================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

:: Check if MCP server modules are installed
python -c "import mcp_server, mcp_client" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: MCP Server and Client modules are not installed.
    echo Please run install_deps.bat first and select option 4 or 5 to install MCP components.
    pause
    exit /b 1
)

:: Check if the required directories exist
if not exist "storage\memory" (
    echo Creating memory storage directory...
    mkdir "storage\memory"
)

:: Check if the MCP servers script exists
if not exist "./src/mcp_servers.py" (
    echo Error: MCP servers script not found at ./src/mcp_servers.py
    echo Please ensure the project is correctly set up.
    pause
    exit /b 1
)

echo Starting MCP Servers...

:: Start the MCP servers with error handling
python ./src/mcp_servers.py start
if %errorlevel% neq 0 (
    echo Error: Failed to start MCP servers.
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)

:: Check if servers are running
timeout /t 3 /nobreak >nul
python -c "from mcp_client import MemoryClient, SequentialThinkingClient; print('Memory Server Status:', MemoryClient().is_available()); print('Sequential Thinking Server Status:', SequentialThinkingClient().is_available())" >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Unable to verify if MCP servers are running.
    echo The servers may have started but verification failed.
) else (
    echo MCP servers started successfully.
    echo Memory and Sequential Thinking capabilities are now available.
)

echo.
echo Press any key to exit this window. The servers will continue running in the background.
echo To stop the servers, run: python ./src/mcp_servers.py stop
echo.
pause
