#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Client Module for Friday AI Trading System

This module provides client interfaces to interact with the Memory and Sequential Thinking
MCP servers, enabling the trading system to maintain context and perform step-by-step reasoning.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
import requests
from aiohttp import ClientSession
import asyncio

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
        logging.FileHandler('mcp_client.log')
    ]
)
logger = logging.getLogger('mcp_client')


class MemoryClient:
    """Client for interacting with the Memory MCP server."""

    def __init__(self):
        """Initialize the Memory Client with configuration settings."""
        if not MCP_CONFIG['memory']['enabled']:
            logger.warning("Memory MCP server is disabled in configuration")

        self.base_url = f"http://{MCP_CONFIG['memory']['host']}:{MCP_CONFIG['memory']['port']}"
        self.timeout = MCP_CONFIG['common']['request_timeout']

    def is_available(self) -> bool:
        """Check if the Memory MCP server is available.

        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking Memory MCP server status: {str(e)}")
            return False

    def get_memory(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Get memory items that match the query.

        Args:
            query (Optional[str]): Optional search query to filter memory items

        Returns:
            Dict[str, Any]: Retrieved memory items or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['memory']['endpoints']['get']}"
            params = {}
            if query:
                params['query'] = query

            response = requests.get(endpoint, params=params, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error retrieving memory: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception retrieving memory: {str(e)}")
            return {'error': "Exception", 'message': str(e)}

    def search_memory(self, query: str) -> Dict[str, Any]:
        """Search memory with semantic similarity.

        Args:
            query (str): Search query for semantic similarity search

        Returns:
            Dict[str, Any]: Search results or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['memory']['endpoints']['search']}"
            response = requests.post(
                endpoint,
                json={'query': query},
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error searching memory: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception searching memory: {str(e)}")
            return {'error': "Exception", 'message': str(e)}

    def update_memory(self, memory_item: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update an item in memory.

        Args:
            memory_item (Dict[str, Any]): Memory item to add or update

        Returns:
            Dict[str, Any]: Operation result or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['memory']['endpoints']['update']}"
            response = requests.post(
                endpoint,
                json=memory_item,
                timeout=self.timeout
            )

            if response.status_code in (200, 201):
                return response.json()
            else:
                logger.error(f"Error updating memory: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception updating memory: {str(e)}")
            return {'error': "Exception", 'message': str(e)}

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete an item from memory.

        Args:
            memory_id (str): ID of the memory item to delete

        Returns:
            Dict[str, Any]: Operation result or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['memory']['endpoints']['delete']}"
            response = requests.post(
                endpoint,
                json={'id': memory_id},
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error deleting memory: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception deleting memory: {str(e)}")
            return {'error': "Exception", 'message': str(e)}


class SequentialThinkingClient:
    """Client for interacting with the Sequential Thinking MCP server."""

    def __init__(self):
        """Initialize the Sequential Thinking Client with configuration settings."""
        if not MCP_CONFIG['sequential_thinking']['enabled']:
            logger.warning("Sequential Thinking MCP server is disabled in configuration")

        self.base_url = f"http://{MCP_CONFIG['sequential_thinking']['host']}:{MCP_CONFIG['sequential_thinking']['port']}"
        self.timeout = MCP_CONFIG['common']['request_timeout']

    def is_available(self) -> bool:
        """Check if the Sequential Thinking MCP server is available.

        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking Sequential Thinking MCP server status: {str(e)}")
            return False

    def start_thinking(self, problem: str, context: Optional[Dict[str, Any]] = None,
                      max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Start a sequential thinking process.

        Args:
            problem (str): The problem to solve or analyze
            context (Optional[Dict[str, Any]]): Additional context information
            max_steps (Optional[int]): Override the default maximum number of thinking steps

        Returns:
            Dict[str, Any]: Thinking process result or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['sequential_thinking']['endpoints']['think']}"
            payload = {
                'problem': problem
            }

            if context:
                payload['context'] = context

            if max_steps:
                payload['max_steps'] = max_steps

            response = requests.post(
                endpoint,
                json=payload,
                timeout=MCP_CONFIG['sequential_thinking']['parameters']['timeout'] + 10  # Add buffer time
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error in sequential thinking: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception in sequential thinking: {str(e)}")
            return {'error': "Exception", 'message': str(e)}

    async def start_thinking_async(self, problem: str, context: Optional[Dict[str, Any]] = None,
                                 max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Start a sequential thinking process asynchronously.

        Args:
            problem (str): The problem to solve or analyze
            context (Optional[Dict[str, Any]]): Additional context information
            max_steps (Optional[int]): Override the default maximum number of thinking steps

        Returns:
            Dict[str, Any]: Thinking process result or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['sequential_thinking']['endpoints']['think']}"
            payload = {
                'problem': problem
            }

            if context:
                payload['context'] = context

            if max_steps:
                payload['max_steps'] = max_steps

            timeout = MCP_CONFIG['sequential_thinking']['parameters']['timeout'] + 10  # Add buffer time

            async with ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Error in async sequential thinking: {response.status} - {error_text}")
                        return {'error': f"Status code: {response.status}", 'message': error_text}
        except Exception as e:
            logger.error(f"Exception in async sequential thinking: {str(e)}")
            return {'error': "Exception", 'message': str(e)}

    def get_thinking_history(self) -> Dict[str, Any]:
        """Get history of past thinking processes.

        Returns:
            Dict[str, Any]: Thinking history or error information
        """
        try:
            endpoint = f"{self.base_url}{MCP_CONFIG['sequential_thinking']['endpoints']['history']}"
            response = requests.get(endpoint, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting thinking history: {response.status_code} - {response.text}")
                return {'error': f"Status code: {response.status_code}", 'message': response.text}
        except Exception as e:
            logger.error(f"Exception getting thinking history: {str(e)}")
            return {'error': "Exception", 'message': str(e)}


# Example usage
def main():
    """Example usage of the MCP clients."""
    # Memory client example
    memory_client = MemoryClient()
    if memory_client.is_available():
        print("Memory server is available")

        # Add a memory item
        memory_item = {
            'content': 'The market showed significant volatility after Fed announcement',
            'metadata': {
                'category': 'market_observation',
                'timestamp': time.time(),
                'importance': 'high'
            }
        }
        result = memory_client.update_memory(memory_item)
        print(f"Memory update result: {json.dumps(result, indent=2)}")

        # Search memory
        search_result = memory_client.search_memory("market volatility")
        print(f"Memory search result: {json.dumps(search_result, indent=2)}")
    else:
        print("Memory server is not available")

    # Sequential thinking client example
    thinking_client = SequentialThinkingClient()
    if thinking_client.is_available():
        print("Sequential thinking server is available")

        # Start a thinking process
        problem = "Analyze the potential market impact of a 0.25% Federal Reserve interest rate hike"
        context = {
            'current_market_state': 'bullish',
            'recent_events': ['Fed meeting scheduled', 'Inflation report released'],
            'asset_classes': ['stocks', 'bonds', 'forex']
        }

        result = thinking_client.start_thinking(problem, context)
        print(f"Sequential thinking result: {json.dumps(result, indent=2)}")
    else:
        print("Sequential thinking server is not available")


if __name__ == "__main__":
    main()
