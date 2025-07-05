#!/usr/bin/env python
"""
Friday AI Trading System - Database Verification Script

This script verifies that MongoDB and Redis are installed and running.
It also checks the connection to the databases and provides guidance on how to install
and start the databases if they are not running.
"""

import sys
import os
import subprocess
import platform
from typing import Dict, Any, Tuple, List

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
    from unified_config import MONGODB_CONFIG, REDIS_CONFIG
except ImportError:
    print("Error: unified_config.py not found or missing required configurations.")
    sys.exit(1)


def check_mongodb_running() -> Tuple[bool, str]:
    """Check if MongoDB is running.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating if MongoDB is running
                          and a message with additional information.
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
        
        return True, f"MongoDB is running on {host}:{port}"
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return False, f"MongoDB is not running or not accessible: {str(e)}"
    except Exception as e:
        return False, f"Error checking MongoDB: {str(e)}"


def check_redis_running() -> Tuple[bool, str]:
    """Check if Redis is running.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating if Redis is running
                          and a message with additional information.
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
        
        return True, f"Redis is running on {host}:{port}"
    except RedisConnectionError as e:
        return False, f"Redis is not running or not accessible: {str(e)}"
    except Exception as e:
        return False, f"Error checking Redis: {str(e)}"


def get_installation_instructions() -> Dict[str, List[str]]:
    """Get installation instructions for MongoDB and Redis based on the operating system.

    Returns:
        Dict[str, List[str]]: A dictionary containing installation instructions for MongoDB and Redis.
    """
    system = platform.system().lower()
    instructions = {
        "mongodb": [],
        "redis": []
    }
    
    if system == "windows":
        instructions["mongodb"] = [
            "1. Download MongoDB Community Server from https://www.mongodb.com/try/download/community",
            "2. Run the installer and follow the instructions",
            "3. Start MongoDB service:",
            "   - Open Services (services.msc)",
            "   - Find 'MongoDB Server' and start it",
            "   - Or run: net start MongoDB"
        ]
        instructions["redis"] = [
            "1. Download Redis for Windows from https://github.com/microsoftarchive/redis/releases",
            "2. Run the installer and follow the instructions",
            "3. Start Redis service:",
            "   - Open Services (services.msc)",
            "   - Find 'Redis' and start it",
            "   - Or run: net start Redis"
        ]
    elif system == "darwin":  # macOS
        instructions["mongodb"] = [
            "1. Install MongoDB using Homebrew:",
            "   brew tap mongodb/brew",
            "   brew install mongodb-community",
            "2. Start MongoDB service:",
            "   brew services start mongodb-community"
        ]
        instructions["redis"] = [
            "1. Install Redis using Homebrew:",
            "   brew install redis",
            "2. Start Redis service:",
            "   brew services start redis"
        ]
    else:  # Linux
        instructions["mongodb"] = [
            "1. Install MongoDB using package manager:",
            "   Ubuntu/Debian:",
            "   - sudo apt update",
            "   - sudo apt install -y mongodb",
            "   CentOS/RHEL:",
            "   - sudo yum install -y mongodb-org",
            "2. Start MongoDB service:",
            "   sudo systemctl start mongod"
        ]
        instructions["redis"] = [
            "1. Install Redis using package manager:",
            "   Ubuntu/Debian:",
            "   - sudo apt update",
            "   - sudo apt install -y redis-server",
            "   CentOS/RHEL:",
            "   - sudo yum install -y redis",
            "2. Start Redis service:",
            "   sudo systemctl start redis"
        ]
    
    return instructions


def main():
    """Main function to check if MongoDB and Redis are running."""
    print("====================================================")
    print("Friday AI Trading System - Database Verification")
    print("====================================================")
    print()
    
    # Check MongoDB
    mongodb_running, mongodb_message = check_mongodb_running()
    print(f"MongoDB: {'✓' if mongodb_running else '✗'} {mongodb_message}")
    
    # Check Redis
    redis_running, redis_message = check_redis_running()
    print(f"Redis: {'✓' if redis_running else '✗'} {redis_message}")
    print()
    
    # If either database is not running, show installation instructions
    if not mongodb_running or not redis_running:
        instructions = get_installation_instructions()
        
        if not mongodb_running:
            print("MongoDB Installation Instructions:")
            for step in instructions["mongodb"]:
                print(step)
            print()
        
        if not redis_running:
            print("Redis Installation Instructions:")
            for step in instructions["redis"]:
                print(step)
            print()
        
        print("After installing and starting the databases, run this script again to verify.")
        sys.exit(1)
    else:
        print("All databases are running. You can proceed with the setup.")
        print("Run 'python run_friday.py --all' to start the system.")
        sys.exit(0)


if __name__ == "__main__":
    main()