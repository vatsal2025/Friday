@echo off
REM Friday AI Trading System Launcher
REM This batch script runs the Friday AI Trading System

echo ===================================================
echo Friday AI Trading System Launcher
echo ===================================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher and try again.
    exit /b 1
)

REM Check if MongoDB is running
echo Checking if MongoDB is running...
ping -n 1 localhost >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Cannot ping localhost. MongoDB might not be running.
    echo Please make sure MongoDB is installed and running.
    set /p continue=Do you want to continue anyway? (y/n): 
    if /i "%continue%" NEQ "y" exit /b 1
)

REM Check if Redis is running
echo Checking if Redis is running...
ping -n 1 localhost >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Cannot ping localhost. Redis might not be running.
    echo Please make sure Redis is installed and running.
    set /p continue=Do you want to continue anyway? (y/n): 
    if /i "%continue%" NEQ "y" exit /b 1
)

REM Display menu
echo.
echo Please select an option:
echo 1. Initialize databases
echo 2. Start MCP servers
echo 3. Start API server
echo 4. Run all components
echo 5. Exit
echo.

set /p option=Enter your choice (1-5): 

if "%option%"=="1" (
    echo Initializing databases...
    python run_friday.py --init-db
) else if "%option%"=="2" (
    echo Starting MCP servers...
    python run_friday.py --start-mcp
) else if "%option%"=="3" (
    echo Starting API server...
    python run_friday.py --start-api
) else if "%option%"=="4" (
    echo Running all components...
    python run_friday.py --all
) else if "%option%"=="5" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid option. Please try again.
    exit /b 1
)

echo.
echo Operation completed.