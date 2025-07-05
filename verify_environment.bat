@echo off
echo Friday AI Trading System - Environment Verification
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher and try again.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv\ (
    echo [WARNING] Virtual environment not found.
    echo Run setup_environment.bat first to set up the environment.
    pause
    exit /b 1
)

:: Activate virtual environment and run verification
call venv\Scripts\activate
python verify_environment.py
set RESULT=%ERRORLEVEL%

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat

if %RESULT% NEQ 0 (
    echo [ERROR] Environment verification failed.
    pause
    exit /b 1
)

echo.
echo Environment verification completed successfully.
echo.

pause