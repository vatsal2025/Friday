"""MCP (Multi-Chain Protocol) integration for the Friday AI Trading System.

This module provides functions to interact with MCP servers and clients.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Global MCP server processes
_mcp_server_processes: Dict[str, subprocess.Popen] = {}


def get_mcp_servers() -> List[Dict[str, Any]]:
    """Get the list of configured MCP servers.

    Returns:
        List[Dict[str, Any]]: The list of MCP server configurations.
    """
    mcp_config = get_config("mcp")
    return mcp_config["servers"]


def get_mcp_server_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get an MCP server configuration by name.

    Args:
        name: The name of the MCP server.

    Returns:
        Optional[Dict[str, Any]]: The MCP server configuration, or None if not found.
    """
    servers = get_mcp_servers()
    for server in servers:
        if server["name"] == name:
            return server
    return None


def is_mcp_server_running(name: str) -> bool:
    """Check if an MCP server is running.

    Args:
        name: The name of the MCP server.

    Returns:
        bool: True if the MCP server is running, False otherwise.
    """
    return name in _mcp_server_processes and _mcp_server_processes[name].poll() is None


def start_mcp_server(name: str) -> Tuple[bool, str]:
    """Start an MCP server.

    Args:
        name: The name of the MCP server.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
    """
    if is_mcp_server_running(name):
        return True, f"MCP server {name} is already running."

    server_config = get_mcp_server_by_name(name)
    if server_config is None:
        return False, f"MCP server {name} not found in configuration."

    if not server_config.get("enabled", True):
        return False, f"MCP server {name} is disabled in configuration."

    try:
        # Start the MCP server as a subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", "mcp.server", "--name", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Store the process
        _mcp_server_processes[name] = process

        # Wait a bit for the server to start
        time.sleep(2)

        # Check if the process is still running
        if process.poll() is None:
            logger.info("Started MCP server %s", name)
            return True, f"MCP server {name} started successfully."
        else:
            # Process exited, get the error message
            stdout, stderr = process.communicate()
            error_message = stderr or stdout
            logger.error("Failed to start MCP server %s: %s", name, error_message)
            return False, f"Failed to start MCP server {name}: {error_message}"

    except Exception as e:
        logger.error("Error starting MCP server %s: %s", name, str(e))
        return False, f"Error starting MCP server {name}: {str(e)}"


def stop_mcp_server(name: str) -> Tuple[bool, str]:
    """Stop an MCP server.

    Args:
        name: The name of the MCP server.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
    """
    if not is_mcp_server_running(name):
        return True, f"MCP server {name} is not running."

    try:
        # Get the process
        process = _mcp_server_processes[name]

        # Terminate the process
        process.terminate()

        # Wait for the process to terminate
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate gracefully
            process.kill()
            process.wait()

        # Remove the process from the dictionary
        del _mcp_server_processes[name]

        logger.info("Stopped MCP server %s", name)
        return True, f"MCP server {name} stopped successfully."

    except Exception as e:
        logger.error("Error stopping MCP server %s: %s", name, str(e))
        return False, f"Error stopping MCP server {name}: {str(e)}"


def get_mcp_server_status(name: str) -> Dict[str, Any]:
    """Get the status of an MCP server.

    Args:
        name: The name of the MCP server.

    Returns:
        Dict[str, Any]: The status of the MCP server.
    """
    server_config = get_mcp_server_by_name(name)
    if server_config is None:
        return {
            "name": name,
            "running": False,
            "enabled": False,
            "error": "MCP server not found in configuration.",
        }

    running = is_mcp_server_running(name)

    return {
        "name": name,
        "running": running,
        "enabled": server_config.get("enabled", True),
        "auto_start": server_config.get("auto_start", False),
        "pid": _mcp_server_processes[name].pid if running else None,
    }


def get_all_mcp_server_status() -> List[Dict[str, Any]]:
    """Get the status of all MCP servers.

    Returns:
        List[Dict[str, Any]]: The status of all MCP servers.
    """
    servers = get_mcp_servers()
    return [get_mcp_server_status(server["name"]) for server in servers]


def start_all_mcp_servers() -> List[Dict[str, Any]]:
    """Start all enabled MCP servers.

    Returns:
        List[Dict[str, Any]]: The status of all MCP servers after starting.
    """
    servers = get_mcp_servers()
    results = []

    for server in servers:
        name = server["name"]
        if server.get("enabled", True) and server.get("auto_start", False):
            success, message = start_mcp_server(name)
            results.append({
                "name": name,
                "success": success,
                "message": message,
                **get_mcp_server_status(name),
            })
        else:
            results.append({
                "name": name,
                "success": True,
                "message": f"MCP server {name} is not configured for auto-start.",
                **get_mcp_server_status(name),
            })

    return results


def stop_all_mcp_servers() -> List[Dict[str, Any]]:
    """Stop all running MCP servers.

    Returns:
        List[Dict[str, Any]]: The status of all MCP servers after stopping.
    """
    results = []

    for name in list(_mcp_server_processes.keys()):
        success, message = stop_mcp_server(name)
        results.append({
            "name": name,
            "success": success,
            "message": message,
            **get_mcp_server_status(name),
        })

    return results


def call_mcp_tool(
    server_name: str, tool_name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Call an MCP tool.

    Args:
        server_name: The name of the MCP server.
        tool_name: The name of the MCP tool.
        args: The arguments for the MCP tool.

    Returns:
        Dict[str, Any]: The result of the MCP tool call.

    Raises:
        ValueError: If the MCP server is not running.
        RuntimeError: If the MCP tool call fails.
    """
    if not is_mcp_server_running(server_name):
        # Try to start the server
        success, message = start_mcp_server(server_name)
        if not success:
            raise ValueError(f"MCP server {server_name} is not running and could not be started: {message}")

    try:
        # Import the MCP client module
        from mcp.client import MCPClient

        # Get the MCP configuration
        mcp_config = get_config("mcp")
        host = mcp_config["server_host"]
        port = mcp_config["server_port"]

        # Create an MCP client
        client = MCPClient(host=host, port=port)

        # Call the MCP tool
        result = client.call_tool(server_name, tool_name, args)

        logger.info(
            "Called MCP tool %s.%s with args %s", server_name, tool_name, args
        )
        return result

    except Exception as e:
        logger.error(
            "Error calling MCP tool %s.%s: %s", server_name, tool_name, str(e)
        )
        raise RuntimeError(f"Error calling MCP tool {server_name}.{tool_name}: {str(e)}")


def get_mcp_server_tools(server_name: str) -> List[Dict[str, Any]]:
    """Get the list of tools provided by an MCP server.

    Args:
        server_name: The name of the MCP server.

    Returns:
        List[Dict[str, Any]]: The list of MCP tools.

    Raises:
        ValueError: If the MCP server is not running.
        RuntimeError: If the MCP server tools cannot be retrieved.
    """
    if not is_mcp_server_running(server_name):
        # Try to start the server
        success, message = start_mcp_server(server_name)
        if not success:
            raise ValueError(f"MCP server {server_name} is not running and could not be started: {message}")

    try:
        # Import the MCP client module
        from mcp.client import MCPClient

        # Get the MCP configuration
        mcp_config = get_config("mcp")
        host = mcp_config["server_host"]
        port = mcp_config["server_port"]

        # Create an MCP client
        client = MCPClient(host=host, port=port)

        # Get the MCP server tools
        tools = client.get_server_tools(server_name)

        logger.info("Retrieved tools for MCP server %s", server_name)
        return tools

    except Exception as e:
        logger.error(
            "Error retrieving tools for MCP server %s: %s", server_name, str(e)
        )
        raise RuntimeError(f"Error retrieving tools for MCP server {server_name}: {str(e)}")


def get_all_mcp_server_tools() -> Dict[str, List[Dict[str, Any]]]:
    """Get the list of tools provided by all MCP servers.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping server names to tool lists.
    """
    servers = get_mcp_servers()
    results = {}

    for server in servers:
        name = server["name"]
        if server.get("enabled", True):
            try:
                results[name] = get_mcp_server_tools(name)
            except Exception as e:
                logger.error(
                    "Error retrieving tools for MCP server %s: %s", name, str(e)
                )
                results[name] = []

    return results


def initialize_mcp() -> None:
    """Initialize the MCP integration.

    This function starts all MCP servers that are configured for auto-start.

    Returns:
        None
    """
    logger.info("Initializing MCP integration")
    start_all_mcp_servers()


def cleanup_mcp() -> None:
    """Clean up the MCP integration.

    This function stops all running MCP servers.

    Returns:
        None
    """
    logger.info("Cleaning up MCP integration")
    stop_all_mcp_servers()