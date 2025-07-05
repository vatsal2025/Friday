@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo Friday AI Trading System - Python Environment Check
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

:: Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

:: Check if Python version meets requirements
if %PYTHON_MAJOR% LSS 3 (
    echo Error: Python 3.10 or later is required.
    echo Current version: %PYTHON_VERSION%
    echo Please install Python 3.10 or later and try again.
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 10 (
        echo Error: Python 3.10 or later is required.
        echo Current version: %PYTHON_VERSION%
        echo Please install Python 3.10 or later and try again.
        exit /b 1
    )
)

echo Python version check: PASSED
echo.

:: Run the Python verification script
python verify_python.py

endlocal