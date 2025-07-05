@echo off
echo Friday AI Trading System - Environment Setup
echo =============================================

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher and try again.
    pause
    exit /b 1
)

:: Run the setup script
python setup_environment.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment setup failed.
    pause
    exit /b 1
)

echo.
echo To activate the virtual environment, run: venv\Scripts\activate
echo.

pause