"""MCP (Multi-Chain Protocol) client implementation for the Friday AI Trading System.

This module provides the client-side implementation of the MCP protocol.
"""

import json
import requests
from typing import Any, Dict, List, Optional, Union

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class MCPClient:
    """MCP client implementation."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize the MCP client.

        Args:
            host: The host of the MCP server.
            port: The port of the MCP server.
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the MCP server.

        Args:
            data: The request data.

        Returns:
            Dict[str, Any]: The response data.

        Raises:
            RuntimeError: If the request fails.
        """
        try:
            response = requests.post(self.base_url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error making request to MCP server: %s", str(e))
            raise RuntimeError(f"Error making request to MCP server: {str(e)}")

    def call_tool(
        self, server_name: str, tool_name: str, args: Dict[str, Any]
    ) -> Any:
        """Call an MCP tool.

        Args:
            server_name: The name of the MCP server.
            tool_name: The name of the MCP tool.
            args: The arguments for the MCP tool.

        Returns:
            Any: The result of the MCP tool call.

        Raises:
            RuntimeError: If the MCP tool call fails.
        """
        data = {
            "action": "call_tool",
            "server_name": server_name,
            "tool_name": tool_name,
            "args": args,
        }

        try:
            response = self._make_request(data)
            if "error" in response:
                raise RuntimeError(response["error"])
            return response.get("result")
        except Exception as e:
            logger.error(
                "Error calling MCP tool %s.%s: %s", server_name, tool_name, str(e)
            )
            raise RuntimeError(f"Error calling MCP tool {server_name}.{tool_name}: {str(e)}")

    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Get the list of tools provided by an MCP server.

        Args:
            server_name: The name of the MCP server.

        Returns:
            List[Dict[str, Any]]: The list of MCP tools.

        Raises:
            RuntimeError: If the MCP server tools cannot be retrieved.
        """
        data = {
            "action": "get_server_tools",
            "server_name": server_name,
        }

        try:
            response = self._make_request(data)
            if "error" in response:
                raise RuntimeError(response["error"])
            return response.get("tools", [])
        except Exception as e:
            logger.error(
                "Error retrieving tools for MCP server %s: %s", server_name, str(e)
            )
            raise RuntimeError(f"Error retrieving tools for MCP server {server_name}: {str(e)}")

    def get_all_server_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the list of tools provided by all MCP servers.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary mapping server names to tool lists.

        Raises:
            RuntimeError: If the MCP server tools cannot be retrieved.
        """
        data = {
            "action": "get_all_server_tools",
        }

        try:
            response = self._make_request(data)
            if "error" in response:
                raise RuntimeError(response["error"])
            return response.get("tools", {})
        except Exception as e:
            logger.error("Error retrieving tools for all MCP servers: %s", str(e))
            raise RuntimeError(f"Error retrieving tools for all MCP servers: {str(e)}")