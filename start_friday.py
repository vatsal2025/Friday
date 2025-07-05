#!/usr/bin/env python
"""
Friday AI Trading System - Startup Script

This script starts all components of the Friday AI Trading System,
including MongoDB, Redis, MCP servers, and the API server.
"""

import sys
import os
import time
import argparse
import subprocess
import platform
from typing import Dict, Any, List, Tuple, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import required modules
try:
    import pymongo
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("Error: pymongo module not found. Please install it using: pip install pymongo")
    sys.exit(1)

try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError:
    print("Error: redis module not found. Please install it using: pip install redis")
    sys.exit(1)

# Import configuration
try:
    from unified_config import MONGODB_CONFIG, REDIS_CONFIG, MCP_CONFIG
except ImportError:
    print("Error: unified_config.py not found or missing required configurations.")
    sys.exit(1)

# Import database setup functions
try:
    from src.infrastructure.database.setup_databases import setup_all_databases, create_test_data
except ImportError:
    print("Warning: Database setup modules not found. Database initialization will be skipped.")
    setup_all_databases = None
    create_test_data = None


def check_mongodb_running() -> bool:
    """Check if MongoDB is running.

    Returns:
        bool: True if MongoDB is running, False otherwise.
    """
    try:
        # Get MongoDB connection parameters
        host = MONGODB_CONFIG.get("host", "localhost")
        port = MONGODB_CONFIG.get("port", 27017)
        username = MONGODB_CONFIG.get("username")
        password = MONGODB_CONFIG.get("password")
        auth_source = MONGODB_CONFIG.get("auth_source", "admin")
        
        # Create connection string
        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/{auth_source}"
        else:
            connection_string = f"mongodb://{host}:{port}/"
        
        # Try to connect with a short timeout
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')  # This will raise an exception if the server is not available
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return False
    except Exception:
        return False


def check_redis_running() -> bool:
    """Check if Redis is running.

    Returns:
        bool: True if Redis is running, False otherwise.
    """
    try:
        # Get Redis connection parameters
        host = REDIS_CONFIG.get("host", "localhost")
        port = REDIS_CONFIG.get("port", 6379)
        password = REDIS_CONFIG.get("password")
        db = REDIS_CONFIG.get("db", 0)
        
        # Try to connect
        r = redis.Redis(host=host, port=port, password=password, db=db, socket_timeout=2)
        r.ping()  # This will raise an exception if the server is not available
        return True
    except (RedisConnectionError, ConnectionRefusedError):
        return False
    except Exception:
        return False


def start_mongodb() -> bool:
    """Start MongoDB if it's not already running.

    Returns:
        bool: True if MongoDB is running after this function, False otherwise.
    """
    if check_mongodb_running():
        print("MongoDB is already running.")
        return True
    
    print("Starting MongoDB...")
    
    # Determine the operating system
    system = platform.system()
    
    if system == "Windows":
        try:
            # Try to start MongoDB as a service
            subprocess.run(["net", "start", "MongoDB"], check=False, capture_output=True)
            time.sleep(2)  # Wait for MongoDB to start
            
            if check_mongodb_running():
                print("MongoDB started successfully.")
                return True
            
            # If service start failed, try to start MongoDB directly
            print("Starting MongoDB directly...")
            subprocess.Popen(["mongod", "--dbpath", "C:\\data\\db"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL, 
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            time.sleep(5)  # Wait for MongoDB to start
            
            if check_mongodb_running():
                print("MongoDB started successfully.")
                return True
            else:
                print("Failed to start MongoDB. Please start it manually.")
                print("You can start MongoDB by running: mongod --dbpath C:\\data\\db")
                return False
        except Exception as e:
            print(f"Error starting MongoDB: {e}")
            print("Please start MongoDB manually.")
            print("You can start MongoDB by running: mongod --dbpath C:\\data\\db")
            return False
    elif system == "Linux" or system == "Darwin":  # Linux or macOS
        try:
            # Try to start MongoDB as a service
            subprocess.run(["sudo", "systemctl", "start", "mongod"], check=False, capture_output=True)
            time.sleep(2)  # Wait for MongoDB to start
            
            if check_mongodb_running():
                print("MongoDB started successfully.")
                return True
            
            # If service start failed, try to start MongoDB directly
            print("Starting MongoDB directly...")
            subprocess.Popen(["mongod", "--dbpath", "/data/db"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(5)  # Wait for MongoDB to start
            
            if check_mongodb_running():
                print("MongoDB started successfully.")
                return True
            else:
                print("Failed to start MongoDB. Please start it manually.")
                print("You can start MongoDB by running: mongod --dbpath /data/db")
                return False
        except Exception as e:
            print(f"Error starting MongoDB: {e}")
            print("Please start MongoDB manually.")
            print("You can start MongoDB by running: mongod --dbpath /data/db")
            return False
    else:
        print(f"Unsupported operating system: {system}")
        print("Please start MongoDB manually.")
        return False


def start_redis() -> bool:
    """Start Redis if it's not already running.

    Returns:
        bool: True if Redis is running after this function, False otherwise.
    """
    if check_redis_running():
        print("Redis is already running.")
        return True
    
    print("Starting Redis...")
    
    # Determine the operating system
    system = platform.system()
    
    if system == "Windows":
        try:
            # Try to start Redis as a service
            subprocess.run(["net", "start", "Redis"], check=False, capture_output=True)
            time.sleep(2)  # Wait for Redis to start
            
            if check_redis_running():
                print("Redis started successfully.")
                return True
            
            # If service start failed, try to start Redis directly
            print("Starting Redis directly...")
            subprocess.Popen(["redis-server"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL, 
                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            time.sleep(2)  # Wait for Redis to start
            
            if check_redis_running():
                print("Redis started successfully.")
                return True
            else:
                print("Failed to start Redis. Please start it manually.")
                print("You can start Redis by running: redis-server")
                return False
        except Exception as e:
            print(f"Error starting Redis: {e}")
            print("Please start Redis manually.")
            print("You can start Redis by running: redis-server")
            return False
    elif system == "Linux" or system == "Darwin":  # Linux or macOS
        try:
            # Try to start Redis as a service
            subprocess.run(["sudo", "systemctl", "start", "redis"], check=False, capture_output=True)
            time.sleep(2)  # Wait for Redis to start
            
            if check_redis_running():
                print("Redis started successfully.")
                return True
            
            # If service start failed, try to start Redis directly
            print("Starting Redis directly...")
            subprocess.Popen(["redis-server"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(2)  # Wait for Redis to start
            
            if check_redis_running():
                print("Redis started successfully.")
                return True
            else:
                print("Failed to start Redis. Please start it manually.")
                print("You can start Redis by running: redis-server")
                return False
        except Exception as e:
            print(f"Error starting Redis: {e}")
            print("Please start Redis manually.")
            print("You can start Redis by running: redis-server")
            return False
    else:
        print(f"Unsupported operating system: {system}")
        print("Please start Redis manually.")
        return False


def initialize_databases(force_recreate: bool = False) -> bool:
    """Initialize MongoDB and Redis databases.

    Args:
        force_recreate: Whether to force recreation of databases.

    Returns:
        bool: True if databases were initialized successfully, False otherwise.
    """
    if setup_all_databases is None:
        print("Database setup modules not found. Skipping database initialization.")
        return False
    
    try:
        print("Initializing databases...")
        setup_all_databases(force_recreate=force_recreate)
        print("Databases initialized successfully.")
        
        # Create test data if requested
        if force_recreate:
            print("Creating test data...")
            create_test_data()
            print("Test data created successfully.")
        
        return True
    except Exception as e:
        print(f"Error initializing databases: {e}")
        return False


def start_mcp_servers() -> Tuple[bool, Optional[subprocess.Popen]]:
    """Start MCP servers.

    Returns:
        Tuple[bool, Optional[subprocess.Popen]]: A tuple containing a boolean indicating
            whether the MCP servers were started successfully, and the process object
            if the servers were started successfully, None otherwise.
    """
    print("Starting MCP servers...")
    
    # Determine the operating system
    system = platform.system()
    
    try:
        if system == "Windows":
            # Start MCP servers using the Python script
            process = subprocess.Popen(
                [sys.executable, os.path.join("src", "mcp_servers.py")],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Linux or macOS
            # Start MCP servers using the Python script
            process = subprocess.Popen(
                [sys.executable, os.path.join("src", "mcp_servers.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        
        # Wait a bit for the servers to start
        time.sleep(5)
        
        print("MCP servers started successfully.")
        return True, process
    except Exception as e:
        print(f"Error starting MCP servers: {e}")
        return False, None


def start_api_server() -> Tuple[bool, Optional[subprocess.Popen]]:
    """Start the API server.

    Returns:
        Tuple[bool, Optional[subprocess.Popen]]: A tuple containing a boolean indicating
            whether the API server was started successfully, and the process object
            if the server was started successfully, None otherwise.
    """
    print("Starting API server...")
    
    # Determine the operating system
    system = platform.system()
    
    try:
        if system == "Windows":
            # Start API server using uvicorn
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "src.application.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Linux or macOS
            # Start API server using uvicorn
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "src.application.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        print("API server started successfully.")
        return True, process
    except Exception as e:
        print(f"Error starting API server: {e}")
        return False, None


def main():
    """Main function to start all components."""
    parser = argparse.ArgumentParser(description="Friday AI Trading System - Startup Script")
    parser.add_argument("--init-db", action="store_true", help="Initialize databases")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of databases")
    parser.add_argument("--start-mcp", action="store_true", help="Start MCP servers")
    parser.add_argument("--start-api", action="store_true", help="Start API server")
    parser.add_argument("--all", action="store_true", help="Start all components")
    args = parser.parse_args()
    
    # If no arguments are provided, start all components
    if not (args.init_db or args.start_mcp or args.start_api or args.all):
        args.all = True
    
    print("====================================================")
    print("Friday AI Trading System - Startup Script")
    print("====================================================")
    print()
    
    # Start MongoDB and Redis if needed
    mongodb_running = check_mongodb_running()
    redis_running = check_redis_running()
    
    if not mongodb_running:
        mongodb_running = start_mongodb()
        if not mongodb_running:
            print("MongoDB is required for the Friday AI Trading System to function properly.")
            print("Please start MongoDB manually and try again.")
            return 1
    
    if not redis_running:
        redis_running = start_redis()
        if not redis_running:
            print("Redis is required for the Friday AI Trading System to function properly.")
            print("Please start Redis manually and try again.")
            return 1
    
    # Initialize databases if requested
    if args.init_db or args.all:
        if not initialize_databases(force_recreate=args.force_recreate):
            print("Failed to initialize databases.")
            return 1
    
    # Start MCP servers if requested
    mcp_process = None
    if args.start_mcp or args.all:
        mcp_success, mcp_process = start_mcp_servers()
        if not mcp_success:
            print("Failed to start MCP servers.")
            return 1
    
    # Start API server if requested
    api_process = None
    if args.start_api or args.all:
        api_success, api_process = start_api_server()
        if not api_success:
            print("Failed to start API server.")
            # Kill MCP servers if they were started
            if mcp_process is not None:
                mcp_process.terminate()
            return 1
    
    print("\nFriday AI Trading System started successfully.")
    print("Press Ctrl+C to stop all components.")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all components...")
        
        # Stop API server if it was started
        if api_process is not None:
            api_process.terminate()
            print("API server stopped.")
        
        # Stop MCP servers if they were started
        if mcp_process is not None:
            mcp_process.terminate()
            print("MCP servers stopped.")
        
        print("All components stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())