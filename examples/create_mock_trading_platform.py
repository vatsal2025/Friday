#!/usr/bin/env python
"""
Example script for creating and using mock trading platform services.

This script demonstrates how to create mock broker, market data, and financial data services,
and how to interact with them using the mock service API.
"""

import json
import time
import logging
from typing import Dict, Any

# Import mock service modules
from src.infrastructure.logging import get_logger
from src.integration.external_system_registry import SystemType
from src.integration.mock import (
    create_mock_broker,
    create_mock_market_data,
    create_mock_financial_data,
    get_mock_service,
    send_mock_request,
    MockServiceConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def load_config(file_path: str) -> Dict[str, Any]:
    """Load a configuration from a JSON file.

    Args:
        file_path: Path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def create_mock_trading_platform():
    """Create mock broker, market data, and financial data services."""
    try:
        # Load configurations
        broker_config = load_config('examples/mock_broker_config.json')
        market_data_config = load_config('examples/mock_market_data_config.json')
        financial_data_config = load_config('examples/mock_financial_data_config.json')

        # Create mock services
        broker_id = create_mock_broker(
            service_id=broker_config['service_id'],
            name=broker_config['name'],
            config=broker_config
        )
        logger.info(f"Created mock broker service with ID: {broker_id}")

        market_data_id = create_mock_market_data(
            service_id=market_data_config['service_id'],
            name=market_data_config['name'],
            config=market_data_config
        )
        logger.info(f"Created mock market data service with ID: {market_data_id}")
        
        financial_data_id = create_mock_financial_data(
            service_id=financial_data_config['service_id'],
            name=financial_data_config['name'],
            config=financial_data_config
        )
        logger.info(f"Created mock financial data service with ID: {financial_data_id}")

        return broker_id, market_data_id, financial_data_id
    except Exception as e:
        logger.error(f"Failed to create mock trading platform: {e}")
        raise


def authenticate_with_broker(broker_id: str):
    """Authenticate with the mock broker service.

    Args:
        broker_id: The ID of the broker service.

    Returns:
        Dict[str, Any]: The authentication result.
    """
    auth_params = {
        "username": "demo",
        "password": "password"
    }
    
    result = send_mock_request(broker_id, "authenticate", auth_params)
    logger.info(f"Broker authentication result: {result}")
    return result


def authenticate_with_market_data(market_data_id: str):
    """Authenticate with the mock market data service.

    Args:
        market_data_id: The ID of the market data service.

    Returns:
        Dict[str, Any]: The authentication result.
    """
    auth_params = {
        "api_key": "demo_api_key"
    }
    
    result = send_mock_request(market_data_id, "authenticate", auth_params)
    logger.info(f"Market data authentication result: {result}")
    return result


def authenticate_with_financial_data(financial_data_id: str):
    """Authenticate with the mock financial data service.

    Args:
        financial_data_id: The ID of the financial data service.

    Returns:
        Dict[str, Any]: The authentication result.
    """
    auth_params = {
        "api_key": "demo_api_key"
    }
    
    result = send_mock_request(financial_data_id, "authenticate", auth_params)
    logger.info(f"Financial data authentication result: {result}")
    return result


def get_broker_accounts(broker_id: str):
    """Get accounts from the mock broker service.

    Args:
        broker_id: The ID of the broker service.

    Returns:
        Dict[str, Any]: The accounts.
    """
    result = send_mock_request(broker_id, "get_accounts", {})
    logger.info(f"Broker accounts: {result}")
    return result


def get_broker_positions(broker_id: str, account_id: str):
    """Get positions from the mock broker service.

    Args:
        broker_id: The ID of the broker service.
        account_id: The ID of the account.

    Returns:
        Dict[str, Any]: The positions.
    """
    params = {"account_id": account_id}
    result = send_mock_request(broker_id, "get_positions", params)
    logger.info(f"Broker positions for account {account_id}: {result}")
    return result


def place_order(broker_id: str, account_id: str, symbol: str, side: str, quantity: int, order_type: str, price: float = None):
    """Place an order with the mock broker service.

    Args:
        broker_id: The ID of the broker service.
        account_id: The ID of the account.
        symbol: The symbol to trade.
        side: The side of the order (buy or sell).
        quantity: The quantity to trade.
        order_type: The type of order (market or limit).
        price: The price for limit orders.

    Returns:
        Dict[str, Any]: The order result.
    """
    params = {
        "account_id": account_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type
    }
    
    if price is not None:
        params["price"] = price
        
    result = send_mock_request(broker_id, "place_order", params)
    logger.info(f"Order placement result: {result}")
    return result


def get_market_data_quote(market_data_id: str, symbol: str):
    """Get a quote from the mock market data service.

    Args:
        market_data_id: The ID of the market data service.
        symbol: The symbol to get a quote for.

    Returns:
        Dict[str, Any]: The quote.
    """
    params = {"symbol": symbol}
    result = send_mock_request(market_data_id, "get_quote", params)
    logger.info(f"Quote for {symbol}: {result}")
    return result


def get_market_data_bars(market_data_id: str, symbol: str, timeframe: str = "1d", limit: int = 10):
    """Get bars from the mock market data service.

    Args:
        market_data_id: The ID of the market data service.
        symbol: The symbol to get bars for.
        timeframe: The timeframe of the bars.
        limit: The maximum number of bars to return.

    Returns:
        Dict[str, Any]: The bars.
    """
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit
    }
    
    result = send_mock_request(market_data_id, "get_bars", params)
    logger.info(f"Bars for {symbol} ({timeframe}): {result}")
    return result


def get_company_info(financial_data_id: str, symbol: str):
    """Get company information from the mock financial data service.

    Args:
        financial_data_id: The ID of the financial data service.
        symbol: The symbol to get company information for.

    Returns:
        Dict[str, Any]: The company information.
    """
    params = {"symbol": symbol}
    result = send_mock_request(financial_data_id, "get_companies", params)
    logger.info(f"Company info for {symbol}: {result}")
    return result


def get_financial_data(financial_data_id: str, symbol: str, period_type: str = "quarterly"):
    """Get financial data from the mock financial data service.

    Args:
        financial_data_id: The ID of the financial data service.
        symbol: The symbol to get financial data for.
        period_type: The period type (quarterly or annual).

    Returns:
        Dict[str, Any]: The financial data.
    """
    params = {
        "symbol": symbol,
        "period_type": period_type
    }
    
    result = send_mock_request(financial_data_id, "get_financials", params)
    logger.info(f"Financial data for {symbol} ({period_type}): {result}")
    return result


def get_news(financial_data_id: str, symbol: str = None, limit: int = 5):
    """Get news from the mock financial data service.

    Args:
        financial_data_id: The ID of the financial data service.
        symbol: The symbol to get news for (optional).
        limit: The maximum number of news items to return.

    Returns:
        Dict[str, Any]: The news items.
    """
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
        
    result = send_mock_request(financial_data_id, "get_news", params)
    logger.info(f"News for {symbol if symbol else 'all symbols'}: {result}")
    return result


def demo_trading_workflow(broker_id: str, market_data_id: str, financial_data_id: str):
    """Demonstrate a simple trading workflow using the mock services.

    Args:
        broker_id: The ID of the broker service.
        market_data_id: The ID of the market data service.
        financial_data_id: The ID of the financial data service.
    """
    try:
        # Authenticate with services
        broker_auth = authenticate_with_broker(broker_id)
        market_data_auth = authenticate_with_market_data(market_data_id)
        financial_data_auth = authenticate_with_financial_data(financial_data_id)
        
        if not broker_auth.get("success") or not market_data_auth.get("success") or not financial_data_auth.get("success"):
            logger.error("Authentication failed")
            return
            
        # Get account information
        account_id = broker_auth.get("account_id")
        accounts = get_broker_accounts(broker_id)
        positions = get_broker_positions(broker_id, account_id)
        
        # Get market data
        symbol = "AAPL"
        quote = get_market_data_quote(market_data_id, symbol)
        bars = get_market_data_bars(market_data_id, symbol)
        
        # Get financial data
        company_info = get_company_info(financial_data_id, symbol)
        financials = get_financial_data(financial_data_id, symbol, "quarterly")
        news = get_news(financial_data_id, symbol, 2)
        
        # Log financial information
        logger.info(f"Company info: {company_info}")
        logger.info(f"Financial data: {financials}")
        logger.info(f"Recent news: {news}")
        
        # Place an order based on market data
        if quote.get("price"):
            price = quote.get("price")
            # Place a limit buy order 1% below current price
            limit_price = price * 0.99
            order_result = place_order(
                broker_id=broker_id,
                account_id=account_id,
                symbol=symbol,
                side="buy",
                quantity=10,
                order_type="limit",
                price=limit_price
            )
            
            # Wait for order processing
            time.sleep(3)
            
            # Get updated positions
            updated_positions = get_broker_positions(broker_id, account_id)
            logger.info(f"Updated positions: {updated_positions}")
    except Exception as e:
        logger.error(f"Error in trading workflow: {e}")


def main():
    """Main function to run the example."""
    try:
        # Create mock trading platform
        broker_id, market_data_id, financial_data_id = create_mock_trading_platform()
        
        # Run demo trading workflow
        demo_trading_workflow(broker_id, market_data_id, financial_data_id)
        
        logger.info("Mock trading platform example completed successfully")
    except Exception as e:
        logger.error(f"Mock trading platform example failed: {e}")


if __name__ == "__main__":
    main()