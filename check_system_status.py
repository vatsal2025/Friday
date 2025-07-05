#!/usr/bin/env python
"""
Friday AI Trading System - System Status Check

This script checks the status of all components of the Friday AI Trading System,
including MongoDB, Redis, MCP servers, and the API server.
"""

import sys
import os
import time
import socket
import subprocess
import requests
from typing import Dict, Any, List, Tuple

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


def check_mongodb_status() -> Dict[str, Any]:
    """Check the status of MongoDB.

    Returns:
        Dict[str, Any]: A dictionary containing the status of MongoDB.
    """
    try:
        # Get MongoDB connection parameters
        host = MONGODB_CONFIG.get("host", "localhost")
        port = MONGODB_CONFIG.get("port", 27017)
        username = MONGODB_CONFIG.get("username")
        password = MONGODB_CONFIG.get("password")
        auth_source = MONGODB_CONFIG.get("auth_source", "admin")
        db_name = MONGODB_CONFIG.get("database", "friday_trading")
        
        # Create connection string
        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/{auth_source}"
        else:
            connection_string = f"mongodb://{host}:{port}/"
        
        # Try to connect with a short timeout
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')  # This will raise an exception if the server is not available
        
        # Check if the database exists
        database_names = client.list_database_names()
        database_exists = db_name in database_names
        
        # Check if required collections exist
        collections = []
        if database_exists:
            db = client[db_name]
            collection_names = db.list_collection_names()
            collections = [
                {"name": "market_data", "exists": "market_data" in collection_names},
                {"name": "model_storage", "exists": "model_storage" in collection_names},
                {"name": "trading_strategy", "exists": "trading_strategy" in collection_names},
                {"name": "backtest_results", "exists": "backtest_results" in collection_names},
            ]
        
        return {
            "status": "running",
            "host": host,
            "port": port,
            "database_exists": database_exists,
            "collections": collections,
            "error": None
        }
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return {
            "status": "error",
            "host": MONGODB_CONFIG.get("host", "localhost"),
            "port": MONGODB_CONFIG.get("port", 27017),
            "database_exists": False,
            "collections": [],
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "host": MONGODB_CONFIG.get("host", "localhost"),
            "port": MONGODB_CONFIG.get("port", 27017),
            "database_exists": False,
            "collections": [],
            "error": str(e)
        }


def check_redis_status() -> Dict[str, Any]:
    """Check the status of Redis.

    Returns:
        Dict[str, Any]: A dictionary containing the status of Redis.
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
        
        # Check if required keys exist
        keys = r.keys("*")
        key_count = len(keys)
        
        # Check if market data namespace exists
        market_data_keys = r.keys("market_data:*")
        market_data_exists = len(market_data_keys) > 0
        
        return {
            "status": "running",
            "host": host,
            "port": port,
            "key_count": key_count,
            "market_data_exists": market_data_exists,
            "error": None
        }
    except RedisConnectionError as e:
        return {
            "status": "error",
            "host": REDIS_CONFIG.get("host", "localhost"),
            "port": REDIS_CONFIG.get("port", 6379),
            "key_count": 0,
            "market_data_exists": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "host": REDIS_CONFIG.get("host", "localhost"),
            "port": REDIS_CONFIG.get("port", 6379),
            "key_count": 0,
            "market_data_exists": False,
            "error": str(e)
        }


def check_mcp_servers_status() -> Dict[str, Any]:
    """Check the status of MCP servers.

    Returns:
        Dict[str, Any]: A dictionary containing the status of MCP servers.
    """
    try:
        # Get MCP server configuration
        memory_config = MCP_CONFIG.get("MEMORY", {})
        sequential_thinking_config = MCP_CONFIG.get("SEQUENTIAL_THINKING", {})
        
        memory_host = memory_config.get("host", "localhost")
        memory_port = memory_config.get("port", 8001)
        sequential_thinking_host = sequential_thinking_config.get("host", "localhost")
        sequential_thinking_port = sequential_thinking_config.get("port", 8002)
        
        # Check if ports are open
        memory_running = check_port_open(memory_host, memory_port)
        sequential_thinking_running = check_port_open(sequential_thinking_host, sequential_thinking_port)
        
        # Try to import MCP client
        mcp_client_available = False
        try:
            from src.mcp_client import MemoryClient
            memory_client = MemoryClient()
            mcp_client_available = memory_client.is_available()
        except ImportError:
            pass
        except Exception:
            pass
        
        return {
            "memory_server": {
                "status": "running" if memory_running else "not_running",
                "host": memory_host,
                "port": memory_port
            },
            "sequential_thinking_server": {
                "status": "running" if sequential_thinking_running else "not_running",
                "host": sequential_thinking_host,
                "port": sequential_thinking_port
            },
            "mcp_client_available": mcp_client_available,
            "error": None
        }
    except Exception as e:
        return {
            "memory_server": {
                "status": "error",
                "host": "unknown",
                "port": 0
            },
            "sequential_thinking_server": {
                "status": "error",
                "host": "unknown",
                "port": 0
            },
            "mcp_client_available": False,
            "error": str(e)
        }


def check_api_server_status() -> Dict[str, Any]:
    """Check the status of the API server.

    Returns:
        Dict[str, Any]: A dictionary containing the status of the API server.
    """
    try:
        # Try to connect to the API server
        response = requests.get("http://localhost:8000/health", timeout=2)
        
        if response.status_code == 200:
            # Try to get version
            try:
                version_response = requests.get("http://localhost:8000/version", timeout=2)
                version = version_response.json().get("version", "unknown") if version_response.status_code == 200 else "unknown"
            except Exception:
                version = "unknown"
            
            return {
                "status": "running",
                "host": "localhost",
                "port": 8000,
                "version": version,
                "error": None
            }
        else:
            return {
                "status": "error",
                "host": "localhost",
                "port": 8000,
                "version": "unknown",
                "error": f"API server returned status code {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "not_running",
            "host": "localhost",
            "port": 8000,
            "version": "unknown",
            "error": "Connection refused"
        }
    except Exception as e:
        return {
            "status": "error",
            "host": "localhost",
            "port": 8000,
            "version": "unknown",
            "error": str(e)
        }


def check_port_open(host: str, port: int) -> bool:
    """Check if a port is open.

    Args:
        host: The host to check.
        port: The port to check.

    Returns:
        bool: True if the port is open, False otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def print_status_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    """Print a status table.

    Args:
        title: The title of the table.
        headers: The headers of the table.
        rows: The rows of the table.
    """
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    # Calculate table width
    table_width = sum(col_widths) + len(headers) * 3 + 1
    
    # Print title
    print(f"\n{title}")
    print("-" * table_width)
    
    # Print headers
    header_row = ""
    for i, header in enumerate(headers):
        header_row += f"| {header:{col_widths[i]}} "
    header_row += "|"
    print(header_row)
    print("-" * table_width)
    
    # Print rows
    for row in rows:
        row_str = ""
        for i, cell in enumerate(row):
            row_str += f"| {cell:{col_widths[i]}} "
        row_str += "|"
        print(row_str)
    
    print("-" * table_width)


def main():
    """Main function to check the status of all components."""
    print("====================================================")
    print("Friday AI Trading System - System Status Check")
    print("====================================================")
    print()
    
    # Check MongoDB status
    print("Checking MongoDB status...")
    mongodb_status = check_mongodb_status()
    
    # Check Redis status
    print("Checking Redis status...")
    redis_status = check_redis_status()
    
    # Check MCP servers status
    print("Checking MCP servers status...")
    mcp_servers_status = check_mcp_servers_status()
    
    # Check API server status
    print("Checking API server status...")
    api_server_status = check_api_server_status()
    
    # Print MongoDB status
    print("\nMongoDB Status:")
    print(f"  Status: {mongodb_status['status']}")
    print(f"  Host: {mongodb_status['host']}")
    print(f"  Port: {mongodb_status['port']}")
    print(f"  Database exists: {mongodb_status['database_exists']}")
    if mongodb_status['error']:
        print(f"  Error: {mongodb_status['error']}")
    
    if mongodb_status['collections']:
        print("\nMongoDB Collections:")
        for collection in mongodb_status['collections']:
            print(f"  {collection['name']}: {'✓' if collection['exists'] else '✗'}")
    
    # Print Redis status
    print("\nRedis Status:")
    print(f"  Status: {redis_status['status']}")
    print(f"  Host: {redis_status['host']}")
    print(f"  Port: {redis_status['port']}")
    print(f"  Key count: {redis_status['key_count']}")
    print(f"  Market data exists: {redis_status['market_data_exists']}")
    if redis_status['error']:
        print(f"  Error: {redis_status['error']}")
    
    # Print MCP servers status
    print("\nMCP Servers Status:")
    print(f"  Memory Server: {mcp_servers_status['memory_server']['status']}")
    print(f"    Host: {mcp_servers_status['memory_server']['host']}")
    print(f"    Port: {mcp_servers_status['memory_server']['port']}")
    print(f"  Sequential Thinking Server: {mcp_servers_status['sequential_thinking_server']['status']}")
    print(f"    Host: {mcp_servers_status['sequential_thinking_server']['host']}")
    print(f"    Port: {mcp_servers_status['sequential_thinking_server']['port']}")
    print(f"  MCP Client Available: {mcp_servers_status['mcp_client_available']}")
    if mcp_servers_status['error']:
        print(f"  Error: {mcp_servers_status['error']}")
    
    # Print API server status
    print("\nAPI Server Status:")
    print(f"  Status: {api_server_status['status']}")
    print(f"  Host: {api_server_status['host']}")
    print(f"  Port: {api_server_status['port']}")
    print(f"  Version: {api_server_status['version']}")
    if api_server_status['error']:
        print(f"  Error: {api_server_status['error']}")
    
    # Print summary table
    headers = ["Component", "Status", "Details"]
    rows = [
        ["MongoDB", mongodb_status['status'], f"{mongodb_status['host']}:{mongodb_status['port']}"],
        ["Redis", redis_status['status'], f"{redis_status['host']}:{redis_status['port']}"],
        ["Memory MCP", mcp_servers_status['memory_server']['status'], f"{mcp_servers_status['memory_server']['host']}:{mcp_servers_status['memory_server']['port']}"],
        ["Sequential MCP", mcp_servers_status['sequential_thinking_server']['status'], f"{mcp_servers_status['sequential_thinking_server']['host']}:{mcp_servers_status['sequential_thinking_server']['port']}"],
        ["API Server", api_server_status['status'], f"{api_server_status['host']}:{api_server_status['port']}"],
    ]
    
    print_status_table("System Status Summary", headers, rows)
    
    # Print overall status
    all_running = (
        mongodb_status['status'] == "running" and
        redis_status['status'] == "running" and
        mcp_servers_status['memory_server']['status'] == "running" and
        mcp_servers_status['sequential_thinking_server']['status'] == "running" and
        api_server_status['status'] == "running"
    )
    
    print("\nOverall System Status:")
    if all_running:
        print("  ✓ All components are running.")
        print("  The Friday AI Trading System is fully operational.")
    else:
        print("  ✗ Some components are not running.")
        print("  The Friday AI Trading System is not fully operational.")
        print("  Please check the status of each component and fix any issues.")
    
    # Return exit code based on overall status
    return 0 if all_running else 1


if __name__ == "__main__":
    sys.exit(main())