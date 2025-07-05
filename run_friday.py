#!/usr/bin/env python
"""
Friday AI Trading System Launcher

This script initializes and runs the Friday AI Trading System, making it fully functional
and production-ready. It performs the following steps:
1. Verifies the environment and dependencies
2. Initializes MongoDB and Redis databases
3. Starts MCP servers
4. Launches the API server

Usage:
    python run_friday.py [--init-db] [--start-mcp] [--start-api] [--all]
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.infrastructure.logging import get_logger, setup_logging
from src.infrastructure.database.verify_db import verify_all_database_connections
from src.infrastructure.database.setup_databases import setup_all_databases
from src.infrastructure.cache.initialize_redis import initialize_redis

# Set up logging
setup_logging()
logger = get_logger(__name__)


def verify_environment():
    """Verify that the environment is properly set up."""
    logger.info("Verifying environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        logger.error("Python 3.10+ is required.")
        return False
    
    # Check if required modules are installed
    required_modules = [
        "pymongo", "redis", "fastapi", "uvicorn", "numpy", "pandas", 
        "scikit-learn", "mcp-server", "mcp-client"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.error("Please install the required modules using: pip install -r requirements.txt")
        return False
    
    logger.info("Environment verification completed successfully.")
    return True


def initialize_databases(force_recreate=False):
    """Initialize MongoDB and Redis databases."""
    logger.info("Initializing databases...")
    
    # Verify database connections
    logger.info("Verifying database connections...")
    db_verification = verify_all_database_connections()
    
    if not db_verification.get("overall_success", False):
        logger.error("Database connection verification failed.")
        logger.error("Please make sure MongoDB and Redis are running.")
        return False
    
    # Initialize Redis
    logger.info("Initializing Redis...")
    redis_initialized = initialize_redis()
    if not redis_initialized:
        logger.error("Redis initialization failed.")
        return False
    
    # Set up all databases
    logger.info("Setting up databases...")
    setup_result = setup_all_databases(force_recreate=force_recreate)
    
    if not setup_result.get("overall_success", False):
        logger.error("Database setup failed.")
        return False
    
    logger.info("Database initialization completed successfully.")
    return True


def start_mcp_servers():
    """Start the MCP servers."""
    logger.info("Starting MCP servers...")
    
    # Create memory directory if it doesn't exist
    memory_dir = Path("./storage/memory")
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    # Start MCP servers
    if sys.platform == "win32":
        subprocess.Popen(["start_mcp_servers.bat"], shell=True)
    else:
        subprocess.Popen(["./start_mcp_servers.sh"], shell=True)
    
    # Wait for servers to start
    logger.info("Waiting for MCP servers to start...")
    time.sleep(5)
    
    # Check if servers are running
    from src.mcp_client import MemoryClient
    
    try:
        memory_client = MemoryClient()
        if memory_client.is_available():
            logger.info("MCP servers started successfully.")
            return True
        else:
            logger.error("Failed to connect to MCP servers.")
            return False
    except Exception as e:
        logger.error(f"Error checking MCP servers: {str(e)}")
        return False


def start_api_server():
    """Start the API server."""
    logger.info("Starting API server...")
    
    # Import the API server module
    from src.application.api.main import run_api_server
    
    # Start the API server
    run_api_server()
    
    return True


def main():
    """Main function to run the Friday AI Trading System."""
    parser = argparse.ArgumentParser(description="Friday AI Trading System Launcher")
    parser.add_argument("--init-db", action="store_true", help="Initialize databases")
    parser.add_argument("--start-mcp", action="store_true", help="Start MCP servers")
    parser.add_argument("--start-api", action="store_true", help="Start API server")
    parser.add_argument("--all", action="store_true", help="Run all components")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of database collections")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not (args.init_db or args.start_mcp or args.start_api or args.all):
        parser.print_help()
        return
    
    # Verify environment
    if not verify_environment():
        return
    
    # Initialize databases
    if args.init_db or args.all:
        if not initialize_databases(force_recreate=args.force_recreate):
            return
    
    # Start MCP servers
    if args.start_mcp or args.all:
        if not start_mcp_servers():
            return
    
    # Start API server
    if args.start_api or args.all:
        if not start_api_server():
            return


if __name__ == "__main__":
    main()