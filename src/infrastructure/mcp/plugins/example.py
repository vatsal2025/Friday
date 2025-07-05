"""Example MCP plugin for the Friday AI Trading System.

This module demonstrates how to implement MCP tools.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


def echo(message: str) -> Dict[str, Any]:
    """Echo a message back to the caller.

    Args:
        message: The message to echo.

    Returns:
        Dict[str, Any]: A dictionary containing the echoed message and timestamp.
    """
    logger.info("Echo: %s", message)
    return {
        "message": message,
        "timestamp": time.time(),
    }


def add(a: float, b: float) -> Dict[str, Any]:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        Dict[str, Any]: A dictionary containing the result of the addition.
    """
    logger.info("Add: %f + %f", a, b)
    return {
        "result": a + b,
    }


def get_system_info() -> Dict[str, Any]:
    """Get system information.

    Returns:
        Dict[str, Any]: A dictionary containing system information.
    """
    import platform
    import psutil

    logger.info("Getting system information")

    # Get CPU information
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "usage_percent": psutil.cpu_percent(interval=1),
        "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
    }

    # Get memory information
    memory_info = psutil.virtual_memory()
    memory = {
        "total": memory_info.total,
        "available": memory_info.available,
        "used": memory_info.used,
        "percent": memory_info.percent,
    }

    # Get disk information
    disk_info = psutil.disk_usage("/")
    disk = {
        "total": disk_info.total,
        "used": disk_info.used,
        "free": disk_info.free,
        "percent": disk_info.percent,
    }

    # Get network information
    network = psutil.net_io_counters()
    network_info = {
        "bytes_sent": network.bytes_sent,
        "bytes_recv": network.bytes_recv,
        "packets_sent": network.packets_sent,
        "packets_recv": network.packets_recv,
    }

    # Get platform information
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    return {
        "cpu": cpu_info,
        "memory": memory,
        "disk": disk,
        "network": network_info,
        "platform": platform_info,
        "timestamp": time.time(),
    }


def register_tools(server_name: str, tool_registry: Any) -> None:
    """Register MCP tools.

    Args:
        server_name: The name of the MCP server.
        tool_registry: The MCP tool registry.

    Returns:
        None
    """
    # Register the echo tool
    tool_registry.register_tool(
        server_name,
        "echo",
        echo,
        "Echo a message back to the caller",
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo",
                },
            },
            "required": ["message"],
        },
    )

    # Register the add tool
    tool_registry.register_tool(
        server_name,
        "add",
        add,
        "Add two numbers",
        {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number",
                },
                "b": {
                    "type": "number",
                    "description": "The second number",
                },
            },
            "required": ["a", "b"],
        },
    )

    # Register the get_system_info tool
    tool_registry.register_tool(
        server_name,
        "get_system_info",
        get_system_info,
        "Get system information",
        {
            "type": "object",
            "properties": {},
            "required": [],
        },
    )

    logger.info("Registered MCP tools for server %s", server_name)