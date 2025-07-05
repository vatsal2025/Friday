"""Example script demonstrating how to use the MCP server integration.

This script shows how to start MCP servers, call MCP tools, and stop MCP servers.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.config import get_config
from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.mcp import (
    get_mcp_servers,
    start_mcp_server,
    stop_mcp_server,
    get_mcp_server_status,
    call_mcp_tool,
    get_mcp_server_tools,
)


def main():
    """Main entry point for the MCP server example.

    Returns:
        None
    """
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MCP server example")
    parser.add_argument(
        "--server", default="example", help="Name of the MCP server to use"
    )
    args = parser.parse_args()

    # Get the MCP server name
    server_name = args.server

    # Get the list of MCP servers
    logger.info("Getting the list of MCP servers")
    servers = get_mcp_servers()
    logger.info("MCP servers: %s", [server["name"] for server in servers])

    # Start the MCP server
    logger.info("Starting MCP server %s", server_name)
    success, message = start_mcp_server(server_name)
    if not success:
        logger.error("Failed to start MCP server %s: %s", server_name, message)
        return

    # Get the MCP server status
    logger.info("Getting MCP server status for %s", server_name)
    status = get_mcp_server_status(server_name)
    logger.info("MCP server status: %s", status)

    # Get the MCP server tools
    logger.info("Getting MCP server tools for %s", server_name)
    try:
        tools = get_mcp_server_tools(server_name)
        logger.info("MCP server tools: %s", [tool["name"] for tool in tools])

        # Call the echo tool
        logger.info("Calling the echo tool")
        result = call_mcp_tool(server_name, "echo", {"message": "Hello, MCP!"})
        logger.info("Echo result: %s", result)

        # Call the add tool
        logger.info("Calling the add tool")
        result = call_mcp_tool(server_name, "add", {"a": 2, "b": 3})
        logger.info("Add result: %s", result)

        # Call the get_system_info tool
        logger.info("Calling the get_system_info tool")
        result = call_mcp_tool(server_name, "get_system_info", {})
        logger.info("System info: CPU cores=%d, Memory used=%.2f%%",
                   result["cpu"]["logical_cores"], result["memory"]["percent"])

    except Exception as e:
        logger.error("Error calling MCP tools: %s", str(e))

    # Stop the MCP server
    logger.info("Stopping MCP server %s", server_name)
    success, message = stop_mcp_server(server_name)
    if not success:
        logger.error("Failed to stop MCP server %s: %s", server_name, message)
        return

    logger.info("MCP server example completed successfully")


if __name__ == "__main__":
    main()