@echo off
echo Friday AI Trading System - MCP Servers Shutdown
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

:: Check if the MCP servers script exists
if not exist "./src/mcp_servers.py" (
    echo Error: MCP servers script not found at ./src/mcp_servers.py
    echo Please ensure the project is correctly set up.
    pause
    exit /b 1
)

echo Stopping MCP Servers...

:: Stop the MCP servers with error handling
python ./src/mcp_servers.py stop
if %errorlevel% neq 0 (
    echo Error: Failed to stop MCP servers.
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)

:: Verify servers are stopped
timeout /t 2 /nobreak >nul
python -c "from mcp_client import MemoryClient, SequentialThinkingClient; print('Memory Server Status:', not MemoryClient().is_available()); print('Sequential Thinking Server Status:', not SequentialThinkingClient().is_available())" >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Unable to verify if MCP servers are stopped.
    echo The servers may have stopped but verification failed.
) else (
    echo MCP servers stopped successfully.
)

echo.
echo To start the servers again, run: start_mcp_servers.bat
echo.
pause