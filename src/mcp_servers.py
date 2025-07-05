#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Server Setup Script for Friday AI Trading System

This script provides utility functions to start and manage Model Context Protocol (MCP) servers
for memory and sequential thinking capabilities. These servers enable the system to maintain
context over time and perform step-by-step reasoning for complex trading decisions.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the configuration settings
from unified_config import MCP_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_servers.log')
    ]
)
logger = logging.getLogger('mcp_servers')

# Global variables to track server processes
memory_server_process = None
sequential_thinking_server_process = None


def start_memory_server() -> Optional[subprocess.Popen]:
    """Start the Memory MCP server.

    Returns:
        Optional[subprocess.Popen]: The process object of the started server, or None if disabled or error occurs
    """
    if not MCP_CONFIG['memory']['enabled']:
        logger.info("Memory MCP server is disabled in configuration")
        return None

    try:
        host = MCP_CONFIG['memory']['host']
        port = MCP_CONFIG['memory']['port']
        storage_path = MCP_CONFIG['memory']['persistence']['storage_path']

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)

        cmd = [
            'python', '-m', 'mcp_server.memory_server',
            '--host', host,
            '--port', str(port),
            '--storage-path', storage_path,
            '--log-level', MCP_CONFIG['common']['log_level']
        ]

        logger.info(f"Starting Memory MCP server on {host}:{port}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a bit to ensure server starts
        time.sleep(2)

        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            logger.error(f"Memory MCP server failed to start: {stderr}")
            return None

        logger.info("Memory MCP server started successfully")
        return process

    except Exception as e:
        logger.error(f"Error starting Memory MCP server: {str(e)}")
        return None


def start_sequential_thinking_server() -> Optional[subprocess.Popen]:
    """Start the Sequential Thinking MCP server.

    Returns:
        Optional[subprocess.Popen]: The process object of the started server, or None if disabled or error occurs
    """
    if not MCP_CONFIG['sequential_thinking']['enabled']:
        logger.info("Sequential Thinking MCP server is disabled in configuration")
        return None

    try:
        host = MCP_CONFIG['sequential_thinking']['host']
        port = MCP_CONFIG['sequential_thinking']['port']
        max_steps = MCP_CONFIG['sequential_thinking']['parameters']['max_steps']
        timeout = MCP_CONFIG['sequential_thinking']['parameters']['timeout']

        cmd = [
            'python', '-m', 'mcp_server.sequential_thinking_server',
            '--host', host,
            '--port', str(port),
            '--max-steps', str(max_steps),
            '--timeout', str(timeout),
            '--log-level', MCP_CONFIG['common']['log_level']
        ]

        if MCP_CONFIG['sequential_thinking']['parameters']['detailed_output']:
            cmd.append('--detailed-output')

        logger.info(f"Starting Sequential Thinking MCP server on {host}:{port}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a bit to ensure server starts
        time.sleep(2)

        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            logger.error(f"Sequential Thinking MCP server failed to start: {stderr}")
            return None

        logger.info("Sequential Thinking MCP server started successfully")
        return process

    except Exception as e:
        logger.error(f"Error starting Sequential Thinking MCP server: {str(e)}")
        return None


def start_all_servers() -> Dict[str, Optional[subprocess.Popen]]:
    """Start all MCP servers.

    Returns:
        Dict[str, Optional[subprocess.Popen]]: Dictionary of server processes
    """
    global memory_server_process, sequential_thinking_server_process

    memory_server_process = start_memory_server()
    sequential_thinking_server_process = start_sequential_thinking_server()

    return {
        'memory': memory_server_process,
        'sequential_thinking': sequential_thinking_server_process
    }


def stop_servers(server_processes: Dict[str, Optional[subprocess.Popen]]) -> None:
    """Stop all running MCP servers.

    Args:
        server_processes (Dict[str, Optional[subprocess.Popen]]): Dictionary of server processes
    """
    for name, process in server_processes.items():
        if process is not None and process.poll() is None:
            logger.info(f"Stopping {name} MCP server")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} MCP server did not respond to termination signal, killing process")
                process.kill()

    logger.info("All MCP servers stopped")


def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down servers."""
    logger.info("Received shutdown signal, stopping MCP servers...")
    stop_servers({
        'memory': memory_server_process,
        'sequential_thinking': sequential_thinking_server_process
    })
    sys.exit(0)


def check_server_status(server_processes: Dict[str, Optional[subprocess.Popen]]) -> Dict[str, str]:
    """Check the status of all MCP servers.

    Args:
        server_processes (Dict[str, Optional[subprocess.Popen]]): Dictionary of server processes

    Returns:
        Dict[str, str]: Status of each server
    """
    status = {}
    for name, process in server_processes.items():
        if process is None:
            status[name] = "Disabled" if MCP_CONFIG.get(name, {}).get('enabled', False) is False else "Not running"
        elif process.poll() is None:
            status[name] = "Running"
        else:
            status[name] = f"Crashed (exit code {process.returncode})"

    return status


def main():
    """Main function to handle command-line arguments and start/stop servers."""
    parser = argparse.ArgumentParser(description="MCP Server Manager for Friday AI Trading System")
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], help='Action to perform')
    parser.add_argument('--memory-only', action='store_true', help='Only manage memory server')
    parser.add_argument('--thinking-only', action='store_true', help='Only manage sequential thinking server')

    args = parser.parse_args()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.action == 'start':
        if args.memory_only:
            processes = {'memory': start_memory_server()}
        elif args.thinking_only:
            processes = {'sequential_thinking': start_sequential_thinking_server()}
        else:
            processes = start_all_servers()

        # Keep the script running to maintain the server processes
        try:
            while True:
                time.sleep(10)
                # Check if any server crashed and restart if configured to do so
                for name, process in processes.items():
                    if process is not None and process.poll() is not None and MCP_CONFIG['common']['auto_restart']:
                        logger.warning(f"{name} MCP server crashed, restarting...")
                        if name == 'memory':
                            processes[name] = start_memory_server()
                        elif name == 'sequential_thinking':
                            processes[name] = start_sequential_thinking_server()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping servers...")
            stop_servers(processes)

    elif args.action == 'stop':
        # Create a dictionary of current running processes
        processes = {}
        if not args.thinking_only:
            processes['memory'] = memory_server_process
        if not args.memory_only:
            processes['sequential_thinking'] = sequential_thinking_server_process

        stop_servers(processes)

    elif args.action == 'restart':
        # First stop any running servers
        processes = {}
        if not args.thinking_only:
            processes['memory'] = memory_server_process
        if not args.memory_only:
            processes['sequential_thinking'] = sequential_thinking_server_process

        stop_servers(processes)

        # Then start the servers
        if args.memory_only:
            processes = {'memory': start_memory_server()}
        elif args.thinking_only:
            processes = {'sequential_thinking': start_sequential_thinking_server()}
        else:
            processes = start_all_servers()

        # Keep the script running
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping servers...")
            stop_servers(processes)

    elif args.action == 'status':
        # Create dictionary of existing processes or None
        processes = {}
        if not args.thinking_only:
            processes['memory'] = memory_server_process
        if not args.memory_only:
            processes['sequential_thinking'] = sequential_thinking_server_process

        status = check_server_status(processes)

        # Print status
        print("MCP Server Status:")
        for name, state in status.items():
            print(f"- {name.capitalize()}: {state}")
            if state == "Running":
                if name == "memory":
                    print(f"  URL: http://{MCP_CONFIG['memory']['host']}:{MCP_CONFIG['memory']['port']}")
                elif name == "sequential_thinking":
                    print(f"  URL: http://{MCP_CONFIG['sequential_thinking']['host']}:{MCP_CONFIG['sequential_thinking']['port']}")


if __name__ == "__main__":
    main()
