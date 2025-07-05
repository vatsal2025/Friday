"""MCP (Multi-Chain Protocol) server implementation for the Friday AI Trading System.

This module provides the server-side implementation of the MCP protocol.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class MCPToolRegistry:
    """Registry for MCP tools."""

    def __init__(self):
        """Initialize the MCP tool registry."""
        self.tools: Dict[str, Dict[str, Callable]] = {}
        self.tool_descriptions: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        tool_func: Callable,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        """Register an MCP tool.

        Args:
            server_name: The name of the MCP server.
            tool_name: The name of the MCP tool.
            tool_func: The function implementing the MCP tool.
            description: The description of the MCP tool.
            input_schema: The JSON schema for the MCP tool input.

        Returns:
            None
        """
        if server_name not in self.tools:
            self.tools[server_name] = {}
            self.tool_descriptions[server_name] = {}

        self.tools[server_name][tool_name] = tool_func
        self.tool_descriptions[server_name][tool_name] = {
            "name": tool_name,
            "description": description,
            "inputSchema": input_schema,
        }

    def get_tool(self, server_name: str, tool_name: str) -> Optional[Callable]:
        """Get an MCP tool.

        Args:
            server_name: The name of the MCP server.
            tool_name: The name of the MCP tool.

        Returns:
            Optional[Callable]: The MCP tool function, or None if not found.
        """
        if server_name not in self.tools:
            return None

        return self.tools[server_name].get(tool_name)

    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Get the list of tools provided by an MCP server.

        Args:
            server_name: The name of the MCP server.

        Returns:
            List[Dict[str, Any]]: The list of MCP tools.
        """
        if server_name not in self.tool_descriptions:
            return []

        return list(self.tool_descriptions[server_name].values())

    def get_all_server_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the list of tools provided by all MCP servers.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary mapping server names to tool lists.
        """
        return {
            server_name: list(tools.values())
            for server_name, tools in self.tool_descriptions.items()
        }


# Create a global tool registry
tool_registry = MCPToolRegistry()


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCP server."""

    def _send_response(self, status_code: int, data: Any) -> None:
        """Send an HTTP response.

        Args:
            status_code: The HTTP status code.
            data: The response data.

        Returns:
            None
        """
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _handle_call_tool(self, data: Dict[str, Any]) -> None:
        """Handle a call_tool request.

        Args:
            data: The request data.

        Returns:
            None
        """
        server_name = data.get("server_name")
        tool_name = data.get("tool_name")
        args = data.get("args", {})

        if not server_name or not tool_name:
            self._send_response(
                400, {"error": "Missing server_name or tool_name in request"}
            )
            return

        tool_func = tool_registry.get_tool(server_name, tool_name)
        if tool_func is None:
            self._send_response(
                404, {"error": f"Tool {server_name}.{tool_name} not found"}
            )
            return

        try:
            result = tool_func(**args)
            self._send_response(200, {"result": result})
        except Exception as e:
            logger.error(
                "Error executing tool %s.%s: %s", server_name, tool_name, str(e)
            )
            self._send_response(
                500, {"error": f"Error executing tool {server_name}.{tool_name}: {str(e)}"}
            )

    def _handle_get_server_tools(self, data: Dict[str, Any]) -> None:
        """Handle a get_server_tools request.

        Args:
            data: The request data.

        Returns:
            None
        """
        server_name = data.get("server_name")

        if not server_name:
            self._send_response(400, {"error": "Missing server_name in request"})
            return

        tools = tool_registry.get_server_tools(server_name)
        self._send_response(200, {"tools": tools})

    def _handle_get_all_server_tools(self, data: Dict[str, Any]) -> None:
        """Handle a get_all_server_tools request.

        Args:
            data: The request data.

        Returns:
            None
        """
        tools = tool_registry.get_all_server_tools()
        self._send_response(200, {"tools": tools})

    def do_POST(self) -> None:
        """Handle a POST request.

        Returns:
            None
        """
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(post_data)
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON in request body"})
            return

        action = data.get("action")
        if not action:
            self._send_response(400, {"error": "Missing action in request"})
            return

        if action == "call_tool":
            self._handle_call_tool(data)
        elif action == "get_server_tools":
            self._handle_get_server_tools(data)
        elif action == "get_all_server_tools":
            self._handle_get_all_server_tools(data)
        else:
            self._send_response(400, {"error": f"Unknown action: {action}"})


class MCPServer:
    """MCP server implementation."""

    def __init__(self, name: str, host: str, port: int):
        """Initialize the MCP server.

        Args:
            name: The name of the MCP server.
            host: The host to bind to.
            port: The port to bind to.
        """
        self.name = name
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self) -> None:
        """Start the MCP server.

        Returns:
            None

        Raises:
            RuntimeError: If the server fails to start.
        """
        if self.running:
            return

        try:
            self.server = HTTPServer((self.host, self.port), MCPRequestHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.running = True
            logger.info(
                "Started MCP server %s on %s:%d", self.name, self.host, self.port
            )
        except Exception as e:
            logger.error("Error starting MCP server %s: %s", self.name, str(e))
            raise RuntimeError(f"Error starting MCP server {self.name}: {str(e)}")

    def stop(self) -> None:
        """Stop the MCP server.

        Returns:
            None
        """
        if not self.running:
            return

        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

        if self.server_thread:
            self.server_thread.join(timeout=5)
            self.server_thread = None

        self.running = False
        logger.info("Stopped MCP server %s", self.name)


def load_mcp_server_plugins(server_name: str) -> None:
    """Load MCP server plugins.

    Args:
        server_name: The name of the MCP server.

    Returns:
        None
    """
    # Get the MCP server configuration
    server_config = None
    mcp_config = get_config("mcp")
    for server in mcp_config["servers"]:
        if server["name"] == server_name:
            server_config = server
            break

    if server_config is None:
        logger.error("MCP server %s not found in configuration", server_name)
        return

    # Load the plugins
    plugins = server_config.get("plugins", [])
    for plugin in plugins:
        try:
            # Import the plugin module
            module_name = plugin["module"]
            module = __import__(module_name, fromlist=["register_tools"])

            # Call the register_tools function
            if hasattr(module, "register_tools"):
                module.register_tools(server_name, tool_registry)
                logger.info(
                    "Loaded MCP plugin %s for server %s", module_name, server_name
                )
            else:
                logger.warning(
                    "MCP plugin %s does not have a register_tools function", module_name
                )

        except Exception as e:
            logger.error(
                "Error loading MCP plugin %s for server %s: %s",
                plugin["module"],
                server_name,
                str(e),
            )


def run_mcp_server(name: str) -> None:
    """Run an MCP server.

    Args:
        name: The name of the MCP server.

    Returns:
        None
    """
    # Get the MCP configuration
    mcp_config = get_config("mcp")
    host = mcp_config["server_host"]
    port = mcp_config["server_port"]

    # Load the MCP server plugins
    load_mcp_server_plugins(name)

    # Create and start the MCP server
    server = MCPServer(name, host, port)

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received signal %d, shutting down MCP server %s", sig, name)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the server
    try:
        server.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down MCP server %s", name)
        server.stop()
    except Exception as e:
        logger.error("Error running MCP server %s: %s", name, str(e))
        server.stop()
        sys.exit(1)


def main():
    """Main entry point for the MCP server.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="MCP server")
    parser.add_argument("--name", required=True, help="Name of the MCP server")
    args = parser.parse_args()

    run_mcp_server(args.name)


if __name__ == "__main__":
    main()