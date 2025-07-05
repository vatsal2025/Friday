@echo off
REM Friday AI Trading System - Docker Services Management Script
REM This script manages MongoDB and Redis Docker containers

setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_FILE=docker-compose.yml
set PROJECT_NAME=friday

REM Colors for output (using echo with color codes)
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

REM Function to print colored output
:print_message
echo %~1
goto :eof

:show_help
echo.
echo %BLUE%Friday AI Trading System - Docker Services Management%RESET%
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   start     Start MongoDB and Redis services
echo   stop      Stop MongoDB and Redis services
echo   restart   Restart MongoDB and Redis services
echo   status    Show status of services
echo   logs      Show logs from services
echo   clean     Stop and remove containers, networks, and volumes
echo   setup     Create required directories and start services
echo   health    Check health status of services
echo   help      Show this help message
echo.
goto :eof

:check_docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: Docker is not installed or not running%RESET%
    echo Please install Docker Desktop and ensure it's running
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED%Error: docker-compose is not installed%RESET%
    echo Please install docker-compose
    exit /b 1
)
goto :eof

:create_directories
echo %BLUE%Creating required directories...%RESET%
if not exist "storage\data\mongodb" mkdir storage\data\mongodb
if not exist "storage\data\redis" mkdir storage\data\redis
if not exist "storage\data\mongodb\configdb" mkdir storage\data\mongodb\configdb
echo %GREEN%Directories created successfully%RESET%
goto :eof

:start_services
echo %BLUE%Starting Friday Docker services...%RESET%
docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d
if errorlevel 1 (
    echo %RED%Failed to start services%RESET%
    exit /b 1
)
echo %GREEN%Services started successfully%RESET%
echo.
echo %YELLOW%Waiting for services to be ready...%RESET%
python wait_for_services.py
goto :eof

:stop_services
echo %BLUE%Stopping Friday Docker services...%RESET%
docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down
if errorlevel 1 (
    echo %RED%Failed to stop services%RESET%
    exit /b 1
)
echo %GREEN%Services stopped successfully%RESET%
goto :eof

:restart_services
echo %BLUE%Restarting Friday Docker services...%RESET%
call :stop_services
timeout /t 3 /nobreak >nul
call :start_services
goto :eof

:show_status
echo %BLUE%Friday Docker Services Status:%RESET%
echo.
docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% ps
echo.
echo %BLUE%Container Health Status:%RESET%
docker inspect friday_mongodb --format="MongoDB: {{.State.Health.Status}}" 2>nul
docker inspect friday_redis --format="Redis: {{.State.Health.Status}}" 2>nul
goto :eof

:show_logs
echo %BLUE%Friday Docker Services Logs:%RESET%
if "%2"=="" (
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% logs -f
) else (
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% logs -f %2
)
goto :eof

:clean_services
echo %YELLOW%This will stop and remove all Friday Docker containers, networks, and volumes%RESET%
set /p confirm="Are you sure? (y/N): "
if /i "!confirm!"=="y" (
    echo %BLUE%Cleaning up Friday Docker environment...%RESET%
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down -v --remove-orphans
    echo %GREEN%Cleanup completed%RESET%
) else (
    echo %YELLOW%Cleanup cancelled%RESET%
)
goto :eof

:setup_environment
echo %BLUE%Setting up Friday Docker environment...%RESET%
call :create_directories
call :start_services
echo %GREEN%Setup completed successfully%RESET%
goto :eof

:check_health
echo %BLUE%Checking health of Friday services...%RESET%
python wait_for_services.py --timeout 30
goto :eof

REM Main script logic
if "%1"=="" goto show_help

call :check_docker

REM Command processing
if /i "%1"=="start" goto start_services
if /i "%1"=="stop" goto stop_services
if /i "%1"=="restart" goto restart_services
if /i "%1"=="status" goto show_status
if /i "%1"=="logs" goto show_logs
if /i "%1"=="clean" goto clean_services
if /i "%1"=="setup" goto setup_environment
if /i "%1"=="health" goto check_health
if /i "%1"=="help" goto show_help

echo %RED%Unknown command: %1%RESET%
echo Use '%0 help' for usage information
exit /b 1
