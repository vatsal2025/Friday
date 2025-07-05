#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Integration Example for Friday AI Trading System

This script demonstrates how to use the Memory and Sequential Thinking MCP servers
in the Friday AI Trading System.
"""

import os
import sys
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pprint import pprint

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
        logging.FileHandler('mcp_example.log')
    ]
)
logger = logging.getLogger('mcp_example')


def generate_sample_market_data() -> pd.DataFrame:
    """Generate sample market data for demonstration.

    Returns:
        pd.DataFrame: Sample market data
    """
    # Create date range for the last 30 days
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate random price data
    np.random.seed(42)  # For reproducibility
    close_prices = np.random.normal(100, 2, size=len(dates))

    # Create a DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * np.random.uniform(0.99, 1.01, size=len(dates)),
        'High': close_prices * np.random.uniform(1.01, 1.03, size=len(dates)),
        'Low': close_prices * np.random.uniform(0.97, 0.99, size=len(dates)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, size=len(dates))
    })

    return df


def store_technical_analysis_in_memory(memory_client: MemoryClient, market_data: pd.DataFrame) -> None:
    """Store technical analysis results in memory.

    Args:
        memory_client (MemoryClient): Memory client instance
        market_data (pd.DataFrame): Market data
    """
    # Calculate simple moving averages
    market_data['SMA_5'] = market_data['Close'].rolling(window=5).mean()
    market_data['SMA_20'] = market_data['Close'].rolling(window=20).mean()

    # Calculate RSI (simplified)
    delta = market_data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    market_data['RSI'] = 100 - (100 / (1 + rs))

    # Get the latest data point
    latest_data = market_data.iloc[-1].to_dict()

    # Store the technical analysis in memory
    memory_item = {
        'content': f"Technical Analysis as of {latest_data['Date']}: "
                  f"Close: ${latest_data['Close']:.2f}, "
                  f"SMA(5): ${latest_data['SMA_5']:.2f}, "
                  f"SMA(20): ${latest_data['SMA_20']:.2f}, "
                  f"RSI: {latest_data['RSI']:.2f}",
        'metadata': {
            'category': 'technical_analysis',
            'timestamp': time.time(),
            'metrics': {
                'close': float(latest_data['Close']),
                'sma5': float(latest_data['SMA_5']),
                'sma20': float(latest_data['SMA_20']),
                'rsi': float(latest_data['RSI'])
            }
        }
    }

    result = memory_client.update_memory(memory_item)
    logger.info(f"Stored technical analysis in memory: {result}")

    # Store the SMA crossover signal if it exists
    current_sma5 = market_data['SMA_5'].iloc[-1]
    current_sma20 = market_data['SMA_20'].iloc[-1]
    prev_sma5 = market_data['SMA_5'].iloc[-2]
    prev_sma20 = market_data['SMA_20'].iloc[-2]

    if prev_sma5 < prev_sma20 and current_sma5 > current_sma20:
        # Bullish crossover (5-day SMA crosses above 20-day SMA)
        signal_item = {
            'content': f"BULLISH SIGNAL: 5-day SMA crossed above 20-day SMA on {latest_data['Date']}",
            'metadata': {
                'category': 'trading_signal',
                'signal_type': 'sma_crossover',
                'direction': 'bullish',
                'timestamp': time.time(),
                'importance': 'high'
            }
        }
        result = memory_client.update_memory(signal_item)
        logger.info(f"Stored bullish crossover signal in memory: {result}")

    elif prev_sma5 > prev_sma20 and current_sma5 < current_sma20:
        # Bearish crossover (5-day SMA crosses below 20-day SMA)
        signal_item = {
            'content': f"BEARISH SIGNAL: 5-day SMA crossed below 20-day SMA on {latest_data['Date']}",
            'metadata': {
                'category': 'trading_signal',
                'signal_type': 'sma_crossover',
                'direction': 'bearish',
                'timestamp': time.time(),
                'importance': 'high'
            }
        }
        result = memory_client.update_memory(signal_item)
        logger.info(f"Stored bearish crossover signal in memory: {result}")


def analyze_market_with_sequential_thinking(thinking_client: SequentialThinkingClient,
                                          market_data: pd.DataFrame,
                                          memory_client: MemoryClient) -> Dict[str, Any]:
    """Use sequential thinking to analyze market data and make trading decision.

    Args:
        thinking_client (SequentialThinkingClient): Sequential thinking client instance
        market_data (pd.DataFrame): Market data
        memory_client (MemoryClient): Memory client instance

    Returns:
        Dict[str, Any]: Analysis result
    """
    # Get relevant memory items for context
    memory_items = memory_client.get_memory()

    # Prepare the context with market data and memory items
    latest_data = market_data.iloc[-1].to_dict()
    market_change = ((market_data['Close'].iloc[-1] / market_data['Close'].iloc[-2]) - 1) * 100

    context = {
        'latest_market_data': {
            'date': str(latest_data['Date']),
            'close': float(latest_data['Close']),
            'open': float(latest_data['Open']),
            'high': float(latest_data['High']),
            'low': float(latest_data['Low']),
            'volume': int(latest_data['Volume']),
            'daily_change_percent': float(market_change)
        },
        'recent_memory': memory_items.get('items', [])[:5] if 'items' in memory_items else []
    }

    # Define the problem to analyze
    problem = (
        "Analyze the current market conditions and provide a trading recommendation. "
        "Consider technical indicators, recent price action, and any significant signals. "
        "Develop a step-by-step reasoning process to reach a conclusion about "
        "whether to enter a long position, enter a short position, or remain neutral."
    )

    # Start the sequential thinking process
    result = thinking_client.start_thinking(problem, context)

    # Store the analysis result in memory
    if 'error' not in result:
        analysis_item = {
            'content': f"Trading Analysis Conclusion: {result.get('conclusion', 'No conclusion reached')}",
            'metadata': {
                'category': 'trading_analysis',
                'timestamp': time.time(),
                'thinking_process': result.get('steps', []),
                'recommendation': result.get('recommendation', 'neutral')
            }
        }
        memory_client.update_memory(analysis_item)

    return result


def main():
    """Main function to demonstrate MCP integration."""
    # Initialize MCP clients
    memory_client = MemoryClient()
    thinking_client = SequentialThinkingClient()

    # Check if MCP servers are available
    if not memory_client.is_available():
        logger.error("Memory MCP server is not available. Make sure it's running.")
        print("Memory MCP server is not available. Run the following command to start it:")
        print("python src/mcp_servers.py start --memory-only")
        return

    if not thinking_client.is_available():
        logger.error("Sequential Thinking MCP server is not available. Make sure it's running.")
        print("Sequential Thinking MCP server is not available. Run the following command to start it:")
        print("python src/mcp_servers.py start --thinking-only")
        return

    logger.info("MCP servers are available. Starting demonstration...")

    # Generate sample market data
    market_data = generate_sample_market_data()
    logger.info(f"Generated sample market data with {len(market_data)} entries")

    # Store technical analysis in memory
    store_technical_analysis_in_memory(memory_client, market_data)

    # Store some additional market context in memory
    market_context = {
        'content': "Market sentiment is cautiously optimistic with expectations of continued growth in tech sector",
        'metadata': {
            'category': 'market_sentiment',
            'timestamp': time.time(),
            'source': 'analyst_consensus'
        }
    }
    memory_client.update_memory(market_context)

    economic_data = {
        'content': "Recent economic data shows inflation easing to 3.1%, slightly below expectations",
        'metadata': {
            'category': 'economic_data',
            'timestamp': time.time(),
            'source': 'economic_report'
        }
    }
    memory_client.update_memory(economic_data)

    # Use sequential thinking to analyze the market
    print("\nAnalyzing market with Sequential Thinking...")
    analysis_result = analyze_market_with_sequential_thinking(thinking_client, market_data, memory_client)

    # Print the result
    print("\nSequential Thinking Analysis Result:")
    print(f"Problem: {analysis_result.get('problem', 'N/A')}")
    print("\nThinking Steps:")
    for i, step in enumerate(analysis_result.get('steps', [])):
        print(f"\nStep {i+1}:")
        print(step)

    print("\nConclusion:")
    print(analysis_result.get('conclusion', 'No conclusion reached'))

    # Retrieve stored memories
    print("\nRetrieving trading signals from memory:")
    signals = memory_client.search_memory("trading signal")
    if 'items' in signals:
        for item in signals['items']:
            print(f"- {item['content']}")

    logger.info("MCP Integration demonstration completed")


if __name__ == "__main__":
    main()
