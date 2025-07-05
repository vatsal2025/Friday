# MCP (Multi-Chain Protocol) Integration

This module provides integration with MCP (Multi-Chain Protocol) servers and clients for the Friday AI Trading System.

## Overview

MCP (Multi-Chain Protocol) is a protocol for communication between AI systems and external tools or services. It allows the Friday AI Trading System to extend its capabilities by integrating with various tools and services through a standardized interface.

## Components

### MCP Server

The MCP server provides a HTTP-based interface for clients to call tools. It manages a registry of tools and handles requests from clients.

### MCP Client

The MCP client provides a Python interface for calling tools on an MCP server. It handles the communication with the server and provides a simple API for calling tools.

### MCP Tools

MCP tools are functions that can be called by clients through the MCP server. They can perform various tasks, such as retrieving data, performing calculations, or interacting with external services.

### MCP Plugins

MCP plugins are Python modules that register tools with the MCP server. They provide a way to extend the functionality of the MCP server without modifying the core code.

## Usage

### Starting an MCP Server

To start an MCP server, you can use the `mcp_servers.py` script:

```bash
python src/mcp_servers.py start --server example
```

This will start the MCP server with the name "example".

### Calling MCP Tools

To call an MCP tool, you can use the `call_mcp_tool` function:

```python
from src.infrastructure.mcp import call_mcp_tool

result = call_mcp_tool("example", "echo", {"message": "Hello, MCP!"})
print(result)
```

This will call the "echo" tool on the "example" MCP server with the message "Hello, MCP!".

### Creating MCP Plugins

To create an MCP plugin, you need to create a Python module with a `register_tools` function that registers tools with the MCP server:

```python
def echo(message: str):
    return {"message": message}

def register_tools(server_name: str, tool_registry: Any):
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
```

Then, you need to add the plugin to the MCP server configuration in `unified_config.py`:

```python
"mcp": {
    "servers": [
        {
            "name": "example",
            "enabled": True,
            "auto_start": True,
            "plugins": [
                {"module": "src.infrastructure.mcp.plugins.example"},
            ],
        },
    ],
    "server_host": "localhost",
    "server_port": 8000,
},
```

## Examples

See the `examples/mcp_server_example.py` script for a complete example of how to use the MCP integration.

## Configuration

The MCP integration is configured in the `unified_config.py` file. The configuration includes the list of MCP servers, their plugins, and the server host and port.

## API Reference

### MCP Module

- `get_mcp_servers()`: Get the list of configured MCP servers.
- `get_mcp_server_by_name(name)`: Get an MCP server configuration by name.
- `is_mcp_server_running(name)`: Check if an MCP server is running.
- `start_mcp_server(name)`: Start an MCP server.
- `stop_mcp_server(name)`: Stop an MCP server.
- `get_mcp_server_status(name)`: Get the status of an MCP server.
- `get_all_mcp_server_status()`: Get the status of all MCP servers.
- `start_all_mcp_servers()`: Start all enabled MCP servers.
- `stop_all_mcp_servers()`: Stop all running MCP servers.
- `call_mcp_tool(server_name, tool_name, args)`: Call an MCP tool.
- `get_mcp_server_tools(server_name)`: Get the list of tools provided by an MCP server.
- `get_all_mcp_server_tools()`: Get the list of tools provided by all MCP servers.
- `initialize_mcp()`: Initialize the MCP integration.
- `cleanup_mcp()`: Clean up the MCP integration.

### MCP Server

- `MCPServer`: MCP server implementation.
- `MCPToolRegistry`: Registry for MCP tools.
- `run_mcp_server(name)`: Run an MCP server.

### MCP Client

- `MCPClient`: MCP client implementation.
- `MCPClient.call_tool(server_name, tool_name, args)`: Call an MCP tool.
- `MCPClient.get_server_tools(server_name)`: Get the list of tools provided by an MCP server.
- `MCPClient.get_all_server_tools()`: Get the list of tools provided by all MCP servers.