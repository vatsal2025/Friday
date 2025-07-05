@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo Friday AI Trading System - Startup Script
echo ====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.10 or later and try again.
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%V in ('python --version 2^>^&1') do (
    set PYTHON_VERSION=%%V
)

echo Using Python version: %PYTHON_VERSION%
echo.

:: Check if MongoDB is running
echo Checking if MongoDB is running...
python -c "import pymongo; pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000).admin.command('ping')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: MongoDB is not running.
    echo.
    
    choice /C YN /M "Do you want to start MongoDB now"
    if !ERRORLEVEL! EQU 1 (
        echo Starting MongoDB...
        net start MongoDB >nul 2>&1
        if !ERRORLEVEL! NEQ 0 (
            echo Failed to start MongoDB as a service.
            echo Please start MongoDB manually and try again.
            echo You can start MongoDB by running: mongod --dbpath C:\data\db
            exit /b 1
        )
        echo MongoDB started successfully.
    ) else (
        echo MongoDB is required for the Friday AI Trading System to function properly.
        echo Please start MongoDB manually and try again.
        exit /b 1
    )
) else (
    echo MongoDB is running.
)

:: Check if Redis is running
echo Checking if Redis is running...
python -c "import redis; redis.Redis(host='localhost', port=6379, socket_timeout=2).ping()" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Redis is not running.
    echo.
    
    choice /C YN /M "Do you want to start Redis now"
    if !ERRORLEVEL! EQU 1 (
        echo Starting Redis...
        net start Redis >nul 2>&1
        if !ERRORLEVEL! NEQ 0 (
            echo Failed to start Redis as a service.
            echo Please start Redis manually and try again.
            echo You can start Redis by running: redis-server
            exit /b 1
        )
        echo Redis started successfully.
    ) else (
        echo Redis is required for the Friday AI Trading System to function properly.
        echo Please start Redis manually and try again.
        exit /b 1
    )
) else (
    echo Redis is running.
)

echo.
echo What would you like to do?
echo 1. Initialize databases
echo 2. Start MCP servers
echo 3. Start API server
echo 4. Start all components
echo 5. Exit
echo.

set /p CHOICE=Enter your choice (1-5): 

if "%CHOICE%"=="1" (
    echo.
    choice /C YN /M "Do you want to force recreation of databases and create test data"
    if !ERRORLEVEL! EQU 1 (
        echo Initializing databases with test data...
        python start_friday.py --init-db --force-recreate
    ) else (
        echo Initializing databases...
        python start_friday.py --init-db
    )
) else if "%CHOICE%"=="2" (
    echo Starting MCP servers...
    start "MCP Servers" python start_friday.py --start-mcp
) else if "%CHOICE%"=="3" (
    echo Starting API server...
    start "API Server" python start_friday.py --start-api
) else if "%CHOICE%"=="4" (
    echo Starting all components...
    python start_friday.py --all
) else if "%CHOICE%"=="5" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    exit /b 1
)

echo.
echo Done.

endlocal