#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Server Test Script for Friday AI Trading System

This script tests the connectivity to the Memory and Sequential Thinking MCP servers
and validates their basic functionality.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MCP clients
from src.mcp_client import MemoryClient, SequentialThinkingClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_test.log')
    ]
)
logger = logging.getLogger('mcp_test')


def test_memory_server() -> bool:
    """Test the Memory MCP server functionality.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n=== Testing Memory MCP Server ===")
    memory_client = MemoryClient()

    # Test server availability
    if not memory_client.is_available():
        print("❌ Memory server is not available")
        return False

    print("✅ Memory server is available")

    # Test adding a memory item
    test_item = {
        'content': 'Test memory item for validation',
        'metadata': {
            'category': 'test',
            'timestamp': time.time(),
            'test_id': 'mem_test_001'
        }
    }

    add_result = memory_client.update_memory(test_item)
    if 'error' in add_result:
        print(f"❌ Failed to add memory item: {add_result['message']}")
        return False

    print("✅ Successfully added memory item")
    item_id = add_result.get('id')

    # Test retrieving memory
    get_result = memory_client.get_memory()
    if 'error' in get_result:
        print(f"❌ Failed to retrieve memory: {get_result['message']}")
        return False

    if 'items' not in get_result or len(get_result['items']) == 0:
        print("❌ No memory items found")
        return False

    print(f"✅ Successfully retrieved {len(get_result['items'])} memory items")

    # Test searching memory
    search_result = memory_client.search_memory('test validation')
    if 'error' in search_result:
        print(f"❌ Failed to search memory: {search_result['message']}")
        return False

    if 'items' not in search_result or len(search_result['items']) == 0:
        print("❌ No matching memory items found in search")
        return False

    print(f"✅ Successfully searched memory and found {len(search_result['items'])} matching items")

    # Test deleting memory if an ID was returned
    if item_id:
        delete_result = memory_client.delete_memory(item_id)
        if 'error' in delete_result:
            print(f"❌ Failed to delete memory item: {delete_result['message']}")
            return False

        print("✅ Successfully deleted memory item")

    print("✅ All memory server tests passed")
    return True


def test_sequential_thinking_server() -> bool:
    """Test the Sequential Thinking MCP server functionality.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n=== Testing Sequential Thinking MCP Server ===")
    thinking_client = SequentialThinkingClient()

    # Test server availability
    if not thinking_client.is_available():
        print("❌ Sequential Thinking server is not available")
        return False

    print("✅ Sequential Thinking server is available")

    # Test a simple thinking process
    problem = "Calculate the optimal position size for a trade with $10,000 capital, " \
              "2% risk per trade, and a stop loss of 5%"

    context = {
        'capital': 10000,
        'risk_percentage': 0.02,
        'stop_loss_percentage': 0.05
    }

    print("Starting sequential thinking process...")

    result = thinking_client.start_thinking(problem, context, max_steps=5)
    if 'error' in result:
        print(f"❌ Failed to complete thinking process: {result['message']}")
        return False

    if 'steps' not in result or len(result['steps']) == 0:
        print("❌ No thinking steps returned")
        return False

    print(f"✅ Sequential thinking completed with {len(result['steps'])} steps")
    print(f"✅ Conclusion: {result.get('conclusion', 'No conclusion available')}")

    # Test getting thinking history
    history_result = thinking_client.get_thinking_history()
    if 'error' in history_result:
        print(f"❌ Failed to get thinking history: {history_result['message']}")
        return False

    if 'history' not in history_result:
        print("❌ No thinking history found")
        return False

    print(f"✅ Successfully retrieved thinking history with {len(history_result['history'])} records")
    print("✅ All sequential thinking server tests passed")
    return True


def main():
    """Main function to run the MCP server tests."""
    parser = argparse.ArgumentParser(description="Test MCP servers for Friday AI Trading System")
    parser.add_argument('--memory-only', action='store_true', help='Test only the memory server')
    parser.add_argument('--thinking-only', action='store_true', help='Test only the sequential thinking server')

    args = parser.parse_args()

    # Track test results
    results = {}

    try:
        # Test memory server
        if not args.thinking_only:
            memory_result = test_memory_server()
            results['memory'] = memory_result

        # Test sequential thinking server
        if not args.memory_only:
            thinking_result = test_sequential_thinking_server()
            results['sequential_thinking'] = thinking_result

        # Print summary
        print("\n=== Test Results Summary ===")
        all_passed = all(results.values())

        for server, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{server.capitalize()}: {status}")

        if all_passed:
            print("\n✅ All MCP server tests passed successfully!")
            return 0
        else:
            print("\n❌ Some MCP server tests failed. See details above.")
            return 1

    except Exception as e:
        logger.error(f"Error during MCP server testing: {str(e)}")
        print(f"\n❌ Test execution error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
