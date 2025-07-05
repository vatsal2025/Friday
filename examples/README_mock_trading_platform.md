# Mock Trading Platform

This directory contains example configurations and code for creating a mock trading platform using the Friday framework's mock service capabilities. The mock trading platform simulates broker, market data, and financial data services for development and testing purposes.

## Overview

The mock trading platform consists of three main components:

1. **Mock Broker Service** - Simulates a brokerage service with accounts, positions, and order execution capabilities.
2. **Mock Market Data Service** - Provides market data such as quotes and historical bars for various symbols.
3. **Mock Financial Data Service** - Offers company information, financial statements, and news articles.

## Configuration Files

The platform uses the following configuration files:

- `mock_broker_config.json` - Configuration for the mock broker service
- `mock_market_data_config.json` - Configuration for the mock market data service
- `mock_financial_data_config.json` - Configuration for the mock financial data service

These configuration files define the behavior, authentication requirements, and pre-populated data for each service.

## Example Script

The `create_mock_trading_platform.py` script demonstrates how to:

1. Load the configuration files
2. Create the mock services
3. Authenticate with each service
4. Retrieve data from each service
5. Execute a simple trading workflow

## Usage

To use the mock trading platform:

1. Ensure the Friday framework is properly installed and configured
2. Run the example script:

```bash
python examples/create_mock_trading_platform.py
```

## Customization

You can customize the mock trading platform by modifying the configuration files:

- Add or modify symbols, accounts, positions, and orders
- Adjust behavior settings like error rates and latency
- Change authentication requirements

## Integration

To integrate the mock trading platform into your application:

```python
from src.integration.mock import (
    create_mock_broker,
    create_mock_market_data,
    create_mock_financial_data,
    send_mock_request
)

# Create the services using your configurations
broker_id = create_mock_broker(service_id="my_broker", name="My Broker", config=broker_config)
market_data_id = create_mock_market_data(service_id="my_market_data", name="My Market Data", config=market_data_config)
financial_data_id = create_mock_financial_data(service_id="my_financial_data", name="My Financial Data", config=financial_data_config)

# Authenticate and interact with the services
auth_result = send_mock_request(broker_id, "authenticate", {"username": "demo", "password": "password"})
accounts = send_mock_request(broker_id, "get_accounts", {})
```

## Service Endpoints

### Broker Service
- `authenticate` - Authenticate with username/password
- `get_accounts` - Get all accounts
- `get_positions` - Get positions for an account
- `get_orders` - Get orders for an account
- `place_order` - Place a new order
- `cancel_order` - Cancel an existing order

### Market Data Service
- `authenticate` - Authenticate with API key
- `get_symbols` - Get available symbols
- `get_quote` - Get current quote for a symbol
- `get_bars` - Get historical bars for a symbol
- `subscribe` - Subscribe to real-time updates
- `unsubscribe` - Unsubscribe from updates

### Financial Data Service
- `authenticate` - Authenticate with API key
- `get_companies` - Get company information
- `get_financials` - Get financial statements
- `get_news` - Get news articles

## Error Handling

The mock services simulate real-world behavior including errors, timeouts, and rate limiting based on the configuration settings. Your application should handle these scenarios appropriately.

## Limitations

- The mock services use pre-populated data and do not connect to external data sources
- Real-time data updates are simulated and not based on actual market movements
- Order execution is simulated with random success/failure rates