# Friday AI Trading System - API Examples

This document provides practical examples of how to interact with the Friday AI Trading System API using Python and JavaScript. These examples demonstrate common operations such as authentication, retrieving market data, placing trades, and managing portfolios.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Authentication](#authentication)
3. [System Status](#system-status)
4. [Market Data](#market-data)
5. [Trading](#trading)
6. [Portfolio Management](#portfolio-management)
7. [Models and Predictions](#models-and-predictions)
8. [Backtesting](#backtesting)
9. [WebSocket Examples](#websocket-examples)
10. [Error Handling](#error-handling)

## Prerequisites

### Python

For Python examples, you'll need the following packages:

```bash
pip install requests websockets pandas matplotlib
```

### JavaScript

For JavaScript examples, you'll need the following packages:

```bash
npm install axios ws chart.js moment
```

## Authentication

### Python Example

```python
import requests
import json

def get_auth_token(api_url, username, password):
    """Authenticate with the Friday API and get a JWT token."""
    auth_endpoint = f"{api_url}/auth/token"
    
    payload = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(auth_endpoint, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        token_data = response.json()
        return token_data["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Authentication error: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
USERNAME = "your_username"
PASSWORD = "your_password"

token = get_auth_token(API_URL, USERNAME, PASSWORD)

if token:
    print(f"Authentication successful. Token: {token[:10]}...")
    
    # Store the token for subsequent requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
else:
    print("Authentication failed.")
```

### JavaScript Example

```javascript
const axios = require('axios');

async function getAuthToken(apiUrl, username, password) {
    const authEndpoint = `${apiUrl}/auth/token`;
    
    const payload = {
        username: username,
        password: password
    };
    
    try {
        const response = await axios.post(authEndpoint, payload);
        return response.data.access_token;
    } catch (error) {
        console.error('Authentication error:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const USERNAME = 'your_username';
const PASSWORD = 'your_password';

(async () => {
    const token = await getAuthToken(API_URL, USERNAME, PASSWORD);
    
    if (token) {
        console.log(`Authentication successful. Token: ${token.substring(0, 10)}...`);
        
        // Store the token for subsequent requests
        const headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    } else {
        console.log('Authentication failed.');
    }
})();
```

## System Status

### Python Example

```python
import requests

def check_system_status(api_url, headers):
    """Check the status of the Friday AI Trading System."""
    status_endpoint = f"{api_url}/system/status"
    
    try:
        response = requests.get(status_endpoint, headers=headers)
        response.raise_for_status()
        
        status_data = response.json()
        return status_data
    except requests.exceptions.RequestException as e:
        print(f"Error checking system status: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

status = check_system_status(API_URL, HEADERS)

if status:
    print("System Status:")
    print(f"  API Server: {status['api_server']}")
    print(f"  MongoDB: {status['mongodb']}")
    print(f"  Redis: {status['redis']}")
    print(f"  MCP Servers:")
    for mcp_server, mcp_status in status['mcp_servers'].items():
        print(f"    {mcp_server}: {mcp_status}")
else:
    print("Failed to retrieve system status.")
```

### JavaScript Example

```javascript
const axios = require('axios');

async function checkSystemStatus(apiUrl, headers) {
    const statusEndpoint = `${apiUrl}/system/status`;
    
    try {
        const response = await axios.get(statusEndpoint, { headers });
        return response.data;
    } catch (error) {
        console.error('Error checking system status:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const HEADERS = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

(async () => {
    const status = await checkSystemStatus(API_URL, HEADERS);
    
    if (status) {
        console.log('System Status:');
        console.log(`  API Server: ${status.api_server}`);
        console.log(`  MongoDB: ${status.mongodb}`);
        console.log(`  Redis: ${status.redis}`);
        console.log(`  MCP Servers:`);
        for (const [mcpServer, mcpStatus] of Object.entries(status.mcp_servers)) {
            console.log(`    ${mcpServer}: ${mcpStatus}`);
        }
    } else {
        console.log('Failed to retrieve system status.');
    }
})();
```

## Market Data

### Python Example: Fetching Historical Data

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_historical_data(api_url, headers, symbol, interval, start_date, end_date):
    """Fetch historical market data for a symbol."""
    historical_endpoint = f"{api_url}/market/historical"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date
    }
    
    try:
        response = requests.get(historical_endpoint, headers=headers, params=params)
        response.raise_for_status()
        
        historical_data = response.json()
        return historical_data["data"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get data for the last 30 days
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

historical_data = get_historical_data(
    API_URL, 
    HEADERS, 
    "AAPL", 
    "1d", 
    start_date, 
    end_date
)

if historical_data:
    # Convert to pandas DataFrame
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f"AAPL Close Price ({start_date} to {end_date})")
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Retrieved {len(df)} data points for AAPL")
else:
    print("Failed to retrieve historical data.")
```

### JavaScript Example: Fetching Real-time Data

```javascript
const axios = require('axios');
const WebSocket = require('ws');

async function subscribeToRealTimeData(apiUrl, token, symbols) {
    // First, get the WebSocket URL from the API
    const wsEndpointUrl = `${apiUrl}/market/websocket-info`;
    
    try {
        const response = await axios.get(wsEndpointUrl, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        const wsUrl = response.data.websocket_url;
        
        // Connect to the WebSocket
        const ws = new WebSocket(wsUrl);
        
        ws.on('open', () => {
            console.log('WebSocket connection established');
            
            // Subscribe to real-time data for the specified symbols
            const subscribeMessage = {
                action: 'subscribe',
                channel: 'market_data',
                symbols: symbols
            };
            
            ws.send(JSON.stringify(subscribeMessage));
        });
        
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            console.log('Received real-time data:', message);
            
            // Process the real-time data as needed
            if (message.channel === 'market_data') {
                const { symbol, price, timestamp, volume } = message.data;
                console.log(`${symbol}: $${price} (Volume: ${volume}) at ${new Date(timestamp).toLocaleTimeString()}`);
            }
        });
        
        ws.on('error', (error) => {
            console.error('WebSocket error:', error.message);
        });
        
        ws.on('close', () => {
            console.log('WebSocket connection closed');
        });
        
        // Return the WebSocket instance for later use
        return ws;
    } catch (error) {
        console.error('Error setting up real-time data subscription:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const TOKEN = 'your_jwt_token';
const SYMBOLS = ['AAPL', 'MSFT', 'GOOGL'];

(async () => {
    const ws = await subscribeToRealTimeData(API_URL, TOKEN, SYMBOLS);
    
    // To unsubscribe and close the connection after 5 minutes
    setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const unsubscribeMessage = {
                action: 'unsubscribe',
                channel: 'market_data',
                symbols: SYMBOLS
            };
            
            ws.send(JSON.stringify(unsubscribeMessage));
            ws.close();
            console.log('Unsubscribed and closed WebSocket connection');
        }
    }, 5 * 60 * 1000);
})();
```

## Trading

### Python Example: Placing an Order

```python
import requests

def place_order(api_url, headers, symbol, order_type, side, quantity, price=None):
    """Place a trading order."""
    order_endpoint = f"{api_url}/trading/orders"
    
    payload = {
        "symbol": symbol,
        "order_type": order_type,  # "market", "limit", "stop", "stop_limit"
        "side": side,  # "buy" or "sell"
        "quantity": quantity
    }
    
    # Add price for limit and stop_limit orders
    if order_type in ["limit", "stop_limit"] and price is not None:
        payload["price"] = price
    
    try:
        response = requests.post(order_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        order_data = response.json()
        return order_data
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Place a market buy order
order = place_order(
    API_URL,
    HEADERS,
    "AAPL",
    "market",
    "buy",
    10
)

if order:
    print("Order placed successfully:")
    print(f"  Order ID: {order['order_id']}")
    print(f"  Symbol: {order['symbol']}")
    print(f"  Type: {order['order_type']}")
    print(f"  Side: {order['side']}")
    print(f"  Quantity: {order['quantity']}")
    print(f"  Status: {order['status']}")

except Exception as e:
    print(f"Error: {e}")
```

### JavaScript Example: Robust Error Handling

```javascript
class FridayAPIClient {
    /**
     * A client for the Friday AI Trading System API with robust error handling.
     */
    constructor(apiUrl, options = {}) {
        this.apiUrl = apiUrl;
        this.username = options.username;
        this.password = options.password;
        this.token = options.token;
        this.maxRetries = options.maxRetries || 3;
        this.retryDelay = options.retryDelay || 1000;
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (this.token) {
            this.headers['Authorization'] = `Bearer ${this.token}`;
        }
    }
    
    /**
     * Authenticate with the API and get a JWT token.
     */
    async authenticate() {
        if (!this.username || !this.password) {
            throw new Error('Username and password are required for authentication');
        }
        
        const authEndpoint = `${this.apiUrl}/auth/token`;
        
        const payload = {
            username: this.username,
            password: this.password
        };
        
        try {
            const response = await this._makeRequest('POST', authEndpoint, { json: payload });
            this.token = response.access_token;
            this.headers['Authorization'] = `Bearer ${this.token}`;
            return true;
        } catch (error) {
            console.error('Authentication error:', error.message);
            return false;
        }
    }
    
    /**
     * Make a request to the API with retry logic and error handling.
     */
    async _makeRequest(method, endpoint, { params, json } = {}, retryCount = 0) {
        const axios = require('axios');
        
        try {
            let response;
            
            const config = {
                method,
                url: endpoint,
                headers: this.headers
            };
            
            if (params) {
                config.params = params;
            }
            
            if (json) {
                config.data = json;
            }
            
            response = await axios(config);
            return response.data;
        } catch (error) {
            // Handle authentication errors
            if (error.response && error.response.status === 401) {
                // Token expired, try to re-authenticate
                if (this.username && this.password && retryCount < this.maxRetries) {
                    console.log('Token expired, re-authenticating...');
                    await this.authenticate();
                    return this._makeRequest(method, endpoint, { params, json }, retryCount + 1);
                } else {
                    throw new Error('Authentication failed: Invalid credentials or token expired');
                }
            }
            
            // Handle timeout errors
            if (error.code === 'ECONNABORTED' && retryCount < this.maxRetries) {
                console.log(`Request timed out, retrying (${retryCount + 1}/${this.maxRetries})...`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * Math.pow(2, retryCount)));
                return this._makeRequest(method, endpoint, { params, json }, retryCount + 1);
            }
            
            // Handle connection errors
            if ((error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') && retryCount < this.maxRetries) {
                console.log(`Connection error, retrying (${retryCount + 1}/${this.maxRetries})...`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * Math.pow(2, retryCount)));
                return this._makeRequest(method, endpoint, { params, json }, retryCount + 1);
            }
            
            // Handle server errors (5xx)
            if (error.response && error.response.status >= 500 && error.response.status < 600 && retryCount < this.maxRetries) {
                console.log(`Server error ${error.response.status}, retrying (${retryCount + 1}/${this.maxRetries})...`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * Math.pow(2, retryCount)));
                return this._makeRequest(method, endpoint, { params, json }, retryCount + 1);
            }
            
            // Handle other errors
            let errorMessage = error.message;
            
            if (error.response && error.response.data) {
                if (error.response.data.detail) {
                    errorMessage = error.response.data.detail;
                } else if (typeof error.response.data === 'string') {
                    errorMessage = error.response.data;
                }
            }
            
            throw new Error(`API error (${error.response ? error.response.status : 'unknown'}): ${errorMessage}`);
        }
    }
    
    /**
     * Get the system status.
     */
    async getSystemStatus() {
        const statusEndpoint = `${this.apiUrl}/system/status`;
        return this._makeRequest('GET', statusEndpoint);
    }
    
    /**
     * Get historical market data.
     */
    async getHistoricalData(symbol, interval, startDate, endDate) {
        const historicalEndpoint = `${this.apiUrl}/market/historical`;
        
        const params = {
            symbol,
            interval,
            start_date: startDate,
            end_date: endDate
        };
        
        const response = await this._makeRequest('GET', historicalEndpoint, { params });
        return response.data;
    }
    
    /**
     * Place a trading order.
     */
    async placeOrder(symbol, orderType, side, quantity, price = null) {
        const orderEndpoint = `${this.apiUrl}/trading/orders`;
        
        const payload = {
            symbol,
            order_type: orderType,
            side,
            quantity
        };
        
        if (['limit', 'stop_limit'].includes(orderType) && price !== null) {
            payload.price = price;
        }
        
        return this._makeRequest('POST', orderEndpoint, { json: payload });
    }
    
    /**
     * Get the portfolio summary.
     */
    async getPortfolioSummary() {
        const portfolioEndpoint = `${this.apiUrl}/portfolio/summary`;
        return this._makeRequest('GET', portfolioEndpoint);
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const USERNAME = 'your_username';
const PASSWORD = 'your_password';

(async () => {
    // Create API client
    const client = new FridayAPIClient(API_URL, {
        username: USERNAME,
        password: PASSWORD,
        maxRetries: 3,
        retryDelay: 1000
    });
    
    try {
        // Authenticate
        await client.authenticate();
        
        // Check system status
        const status = await client.getSystemStatus();
        console.log('System Status:');
        console.log(`  API Server: ${status.api_server}`);
        console.log(`  MongoDB: ${status.mongodb}`);
        console.log(`  Redis: ${status.redis}`);
        
        // Get portfolio summary
        const portfolio = await client.getPortfolioSummary();
        console.log('\nPortfolio Summary:');
        console.log(`  Total Value: $${portfolio.total_value.toFixed(2)}`);
        console.log(`  Cash Balance: $${portfolio.cash_balance.toFixed(2)}`);
        console.log(`  Invested Value: $${portfolio.invested_value.toFixed(2)}`);
        
        // Place an order
        const order = await client.placeOrder('AAPL', 'market', 'buy', 10);
        console.log('\nOrder placed successfully:');
        console.log(`  Order ID: ${order.order_id}`);
        console.log(`  Symbol: ${order.symbol}`);
        console.log(`  Type: ${order.order_type}`);
        console.log(`  Side: ${order.side}`);
        console.log(`  Quantity: ${order.quantity}`);
        console.log(`  Status: ${order.status}`);
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
})();
```

This document provides practical examples to help you get started with the Friday AI Trading System API. For more detailed information about the API endpoints and parameters, please refer to the [API Documentation](API_DOCUMENTATION.md).

# Place a limit sell order
limit_order = place_order(
    API_URL,
    HEADERS,
    "AAPL",
    "limit",
    "sell",
    5,
    price=150.00
)

if limit_order:
    print("\nLimit order placed successfully:")
    print(f"  Order ID: {limit_order['order_id']}")
    print(f"  Symbol: {limit_order['symbol']}")
    print(f"  Type: {limit_order['order_type']}")
    print(f"  Side: {limit_order['side']}")
    print(f"  Quantity: {limit_order['quantity']}")
    print(f"  Price: ${limit_order['price']}")
    print(f"  Status: {limit_order['status']}")
else:
    print("Failed to place limit order.")
```

### JavaScript Example: Getting Order Status

```javascript
const axios = require('axios');

async function getOrderStatus(apiUrl, headers, orderId) {
    const orderEndpoint = `${apiUrl}/trading/orders/${orderId}`;
    
    try {
        const response = await axios.get(orderEndpoint, { headers });
        return response.data;
    } catch (error) {
        console.error('Error getting order status:', error.message);
        return null;
    }
}

async function getAllOrders(apiUrl, headers, status = null, limit = 10) {
    const ordersEndpoint = `${apiUrl}/trading/orders`;
    
    const params = { limit };
    if (status) {
        params.status = status;
    }
    
    try {
        const response = await axios.get(ordersEndpoint, { 
            headers,
            params
        });
        return response.data;
    } catch (error) {
        console.error('Error getting orders:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const HEADERS = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

(async () => {
    // Get a specific order
    const orderId = 'your_order_id';
    const order = await getOrderStatus(API_URL, HEADERS, orderId);
    
    if (order) {
        console.log('Order details:');
        console.log(`  Order ID: ${order.order_id}`);
        console.log(`  Symbol: ${order.symbol}`);
        console.log(`  Type: ${order.order_type}`);
        console.log(`  Side: ${order.side}`);
        console.log(`  Quantity: ${order.quantity}`);
        console.log(`  Status: ${order.status}`);
        if (order.price) {
            console.log(`  Price: $${order.price}`);
        }
        console.log(`  Created At: ${new Date(order.created_at).toLocaleString()}`);
        if (order.executed_at) {
            console.log(`  Executed At: ${new Date(order.executed_at).toLocaleString()}`);
        }
    } else {
        console.log(`Failed to retrieve order ${orderId}.`);
    }
    
    // Get all open orders
    const openOrders = await getAllOrders(API_URL, HEADERS, 'open');
    
    if (openOrders && openOrders.orders) {
        console.log('\nOpen Orders:');
        if (openOrders.orders.length === 0) {
            console.log('  No open orders.');
        } else {
            openOrders.orders.forEach((order, index) => {
                console.log(`  ${index + 1}. ${order.symbol} - ${order.side.toUpperCase()} ${order.quantity} @ ${order.order_type.toUpperCase()}${order.price ? ' $' + order.price : ''}`);
            });
        }
    } else {
        console.log('Failed to retrieve open orders.');
    }
})();
```

## Portfolio Management

### Python Example: Getting Portfolio Summary

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

def get_portfolio_summary(api_url, headers):
    """Get a summary of the user's portfolio."""
    portfolio_endpoint = f"{api_url}/portfolio/summary"
    
    try:
        response = requests.get(portfolio_endpoint, headers=headers)
        response.raise_for_status()
        
        portfolio_data = response.json()
        return portfolio_data
    except requests.exceptions.RequestException as e:
        print(f"Error getting portfolio summary: {e}")
        return None

def get_portfolio_positions(api_url, headers):
    """Get the current positions in the user's portfolio."""
    positions_endpoint = f"{api_url}/portfolio/positions"
    
    try:
        response = requests.get(positions_endpoint, headers=headers)
        response.raise_for_status()
        
        positions_data = response.json()
        return positions_data["positions"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting portfolio positions: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get portfolio summary
portfolio = get_portfolio_summary(API_URL, HEADERS)

if portfolio:
    print("Portfolio Summary:")
    print(f"  Total Value: ${portfolio['total_value']:.2f}")
    print(f"  Cash Balance: ${portfolio['cash_balance']:.2f}")
    print(f"  Invested Value: ${portfolio['invested_value']:.2f}")
    print(f"  Day Change: ${portfolio['day_change']:.2f} ({portfolio['day_change_percent']:.2f}%)")
    print(f"  Total Return: ${portfolio['total_return']:.2f} ({portfolio['total_return_percent']:.2f}%)")
else:
    print("Failed to retrieve portfolio summary.")

# Get portfolio positions
positions = get_portfolio_positions(API_URL, HEADERS)

if positions:
    # Convert to pandas DataFrame
    df = pd.DataFrame(positions)
    
    # Calculate position values
    df['position_value'] = df['quantity'] * df['current_price']
    
    # Sort by position value
    df = df.sort_values('position_value', ascending=False)
    
    print("\nPortfolio Positions:")
    for index, row in df.iterrows():
        print(f"  {row['symbol']}: {row['quantity']} shares @ ${row['current_price']:.2f} = ${row['position_value']:.2f} ({row['unrealized_pl_percent']:.2f}%)")
    
    # Create a pie chart of portfolio allocation
    plt.figure(figsize=(10, 8))
    plt.pie(df['position_value'], labels=df['symbol'], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Portfolio Allocation by Position Value')
    plt.tight_layout()
    plt.show()
else:
    print("Failed to retrieve portfolio positions.")
```

### JavaScript Example: Portfolio Performance

```javascript
const axios = require('axios');
const moment = require('moment');

async function getPortfolioPerformance(apiUrl, headers, timeframe = '1m') {
    const performanceEndpoint = `${apiUrl}/portfolio/performance`;
    
    try {
        const response = await axios.get(performanceEndpoint, { 
            headers,
            params: { timeframe }
        });
        return response.data;
    } catch (error) {
        console.error('Error getting portfolio performance:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const HEADERS = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

(async () => {
    // Get portfolio performance for the last month
    const performance = await getPortfolioPerformance(API_URL, HEADERS, '1m');
    
    if (performance && performance.data) {
        console.log('Portfolio Performance (Last Month):');
        
        // Format the data for display
        const formattedData = performance.data.map(point => ({
            date: moment(point.timestamp).format('YYYY-MM-DD'),
            value: point.value.toFixed(2),
            change: point.daily_change ? point.daily_change.toFixed(2) : 'N/A',
            changePercent: point.daily_change_percent ? `${point.daily_change_percent.toFixed(2)}%` : 'N/A'
        }));
        
        // Display the first and last few points
        console.log('\nInitial Performance:');
        formattedData.slice(0, 3).forEach(point => {
            console.log(`  ${point.date}: $${point.value} (Change: $${point.change}, ${point.changePercent})`);
        });
        
        console.log('\nRecent Performance:');
        formattedData.slice(-3).forEach(point => {
            console.log(`  ${point.date}: $${point.value} (Change: $${point.change}, ${point.changePercent})`);
        });
        
        // Calculate overall performance
        const firstValue = performance.data[0].value;
        const lastValue = performance.data[performance.data.length - 1].value;
        const totalChange = lastValue - firstValue;
        const totalChangePercent = (totalChange / firstValue) * 100;
        
        console.log('\nOverall Performance:');
        console.log(`  Start Value: $${firstValue.toFixed(2)}`);
        console.log(`  End Value: $${lastValue.toFixed(2)}`);
        console.log(`  Total Change: $${totalChange.toFixed(2)} (${totalChangePercent.toFixed(2)}%)`);
    } else {
        console.log('Failed to retrieve portfolio performance.');
    }
})();
```

## Models and Predictions

### Python Example: Getting Model Predictions

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

def get_model_predictions(api_url, headers, symbol, model_id=None):
    """Get predictions for a symbol from a specific model or all models."""
    predictions_endpoint = f"{api_url}/models/predictions"
    
    params = {"symbol": symbol}
    if model_id:
        params["model_id"] = model_id
    
    try:
        response = requests.get(predictions_endpoint, headers=headers, params=params)
        response.raise_for_status()
        
        predictions_data = response.json()
        return predictions_data["predictions"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting model predictions: {e}")
        return None

def get_available_models(api_url, headers):
    """Get a list of available models."""
    models_endpoint = f"{api_url}/models"
    
    try:
        response = requests.get(models_endpoint, headers=headers)
        response.raise_for_status()
        
        models_data = response.json()
        return models_data["models"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting available models: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get available models
models = get_available_models(API_URL, HEADERS)

if models:
    print("Available Models:")
    for model in models:
        print(f"  {model['model_id']}: {model['name']} (Type: {model['type']}, Accuracy: {model['accuracy']:.2f})")
    
    # Select the first model for predictions
    selected_model = models[0]['model_id']
else:
    print("Failed to retrieve available models.")
    selected_model = None

# Get predictions for AAPL
predictions = get_model_predictions(API_URL, HEADERS, "AAPL", selected_model)

if predictions:
    # Convert to pandas DataFrame
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['predicted_price'], 'b-', label='Predicted Price')
    if 'actual_price' in df.columns:
        plt.plot(df.index, df['actual_price'], 'g-', label='Actual Price')
    plt.title(f"AAPL Price Predictions ({selected_model})")
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"\nPredictions for AAPL using model {selected_model}:")
    for index, row in df.iterrows():
        print(f"  {index.date()}: Predicted ${row['predicted_price']:.2f}")
        if 'actual_price' in row and not pd.isna(row['actual_price']):
            print(f"    Actual: ${row['actual_price']:.2f} (Error: {row['error_percent']:.2f}%)")
else:
    print("Failed to retrieve predictions.")
```

### JavaScript Example: Model Performance

```javascript
const axios = require('axios');
const moment = require('moment');

async function getModelPerformance(apiUrl, headers, modelId, timeframe = '1m') {
    const performanceEndpoint = `${apiUrl}/models/${modelId}/performance`;
    
    try {
        const response = await axios.get(performanceEndpoint, { 
            headers,
            params: { timeframe }
        });
        return response.data;
    } catch (error) {
        console.error('Error getting model performance:', error.message);
        return null;
    }
}

async function getAvailableModels(apiUrl, headers) {
    const modelsEndpoint = `${apiUrl}/models`;
    
    try {
        const response = await axios.get(modelsEndpoint, { headers });
        return response.data.models;
    } catch (error) {
        console.error('Error getting available models:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const HEADERS = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

(async () => {
    // Get available models
    const models = await getAvailableModels(API_URL, HEADERS);
    
    if (models && models.length > 0) {
        console.log('Available Models:');
        models.forEach(model => {
            console.log(`  ${model.model_id}: ${model.name} (Type: ${model.type}, Accuracy: ${model.accuracy.toFixed(2)})`);
        });
        
        // Select the first model for performance analysis
        const selectedModel = models[0].model_id;
        
        // Get model performance for the last month
        const performance = await getModelPerformance(API_URL, HEADERS, selectedModel, '1m');
        
        if (performance && performance.data) {
            console.log(`\nModel Performance (${models[0].name}, Last Month):`);
            
            // Calculate average metrics
            const accuracyValues = performance.data.map(p => p.accuracy);
            const precisionValues = performance.data.map(p => p.precision);
            const recallValues = performance.data.map(p => p.recall);
            const f1Values = performance.data.map(p => p.f1_score);
            
            const avgAccuracy = accuracyValues.reduce((a, b) => a + b, 0) / accuracyValues.length;
            const avgPrecision = precisionValues.reduce((a, b) => a + b, 0) / precisionValues.length;
            const avgRecall = recallValues.reduce((a, b) => a + b, 0) / recallValues.length;
            const avgF1 = f1Values.reduce((a, b) => a + b, 0) / f1Values.length;
            
            console.log('Average Metrics:');
            console.log(`  Accuracy: ${avgAccuracy.toFixed(4)}`);
            console.log(`  Precision: ${avgPrecision.toFixed(4)}`);
            console.log(`  Recall: ${avgRecall.toFixed(4)}`);
            console.log(`  F1 Score: ${avgF1.toFixed(4)}`);
            
            // Display recent performance
            console.log('\nRecent Performance:');
            performance.data.slice(-5).forEach(point => {
                const date = moment(point.timestamp).format('YYYY-MM-DD');
                console.log(`  ${date}: Accuracy=${point.accuracy.toFixed(4)}, Precision=${point.precision.toFixed(4)}, Recall=${point.recall.toFixed(4)}, F1=${point.f1_score.toFixed(4)}`);
            });
        } else {
            console.log(`Failed to retrieve performance for model ${selectedModel}.`);
        }
    } else {
        console.log('Failed to retrieve available models.');
    }
})();
```

## Backtesting

### Python Example: Running a Backtest

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_backtest(api_url, headers, strategy_id, start_date, end_date, initial_capital=10000):
    """Run a backtest for a trading strategy."""
    backtest_endpoint = f"{api_url}/backtest/run"
    
    payload = {
        "strategy_id": strategy_id,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital
    }
    
    try:
        response = requests.post(backtest_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        backtest_data = response.json()
        return backtest_data
    except requests.exceptions.RequestException as e:
        print(f"Error running backtest: {e}")
        return None

def get_available_strategies(api_url, headers):
    """Get a list of available trading strategies."""
    strategies_endpoint = f"{api_url}/strategies"
    
    try:
        response = requests.get(strategies_endpoint, headers=headers)
        response.raise_for_status()
        
        strategies_data = response.json()
        return strategies_data["strategies"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting available strategies: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get available strategies
strategies = get_available_strategies(API_URL, HEADERS)

if strategies:
    print("Available Strategies:")
    for strategy in strategies:
        print(f"  {strategy['strategy_id']}: {strategy['name']} (Type: {strategy['type']})")
    
    # Select the first strategy for backtesting
    selected_strategy = strategies[0]['strategy_id']
else:
    print("Failed to retrieve available strategies.")
    selected_strategy = None

# Run a backtest for the last year
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

backtest = run_backtest(
    API_URL, 
    HEADERS, 
    selected_strategy, 
    start_date, 
    end_date, 
    initial_capital=10000
)

if backtest:
    print(f"\nBacktest Results for {backtest['strategy_name']} ({start_date} to {end_date}):")
    print(f"  Initial Capital: ${backtest['initial_capital']:.2f}")
    print(f"  Final Capital: ${backtest['final_capital']:.2f}")
    print(f"  Total Return: ${backtest['total_return']:.2f} ({backtest['total_return_percent']:.2f}%)")
    print(f"  Annualized Return: {backtest['annualized_return']:.2f}%")
    print(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest['max_drawdown']:.2f}%")
    print(f"  Win Rate: {backtest['win_rate']:.2f}%")
    print(f"  Total Trades: {backtest['total_trades']}")
    
    # Convert performance data to DataFrame
    performance_data = backtest['performance_data']
    df = pd.DataFrame(performance_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity'], 'b-', label='Strategy')
    plt.plot(df.index, df['benchmark'], 'r-', label='Benchmark')
    plt.title(f"Backtest Equity Curve: {backtest['strategy_name']}")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['drawdown'] * 100, 'r-')
    plt.title(f"Drawdown: {backtest['strategy_name']}")
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Failed to run backtest.")
```

### JavaScript Example: Comparing Backtest Results

```javascript
const axios = require('axios');
const moment = require('moment');

async function runBacktest(apiUrl, headers, strategyId, startDate, endDate, initialCapital = 10000) {
    const backtestEndpoint = `${apiUrl}/backtest/run`;
    
    const payload = {
        strategy_id: strategyId,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital
    };
    
    try {
        const response = await axios.post(backtestEndpoint, payload, { headers });
        return response.data;
    } catch (error) {
        console.error('Error running backtest:', error.message);
        return null;
    }
}

async function getAvailableStrategies(apiUrl, headers) {
    const strategiesEndpoint = `${apiUrl}/strategies`;
    
    try {
        const response = await axios.get(strategiesEndpoint, { headers });
        return response.data.strategies;
    } catch (error) {
        console.error('Error getting available strategies:', error.message);
        return null;
    }
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const HEADERS = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

(async () => {
    // Get available strategies
    const strategies = await getAvailableStrategies(API_URL, HEADERS);
    
    if (strategies && strategies.length >= 2) {
        console.log('Available Strategies:');
        strategies.forEach(strategy => {
            console.log(`  ${strategy.strategy_id}: ${strategy.name} (Type: ${strategy.type})`);
        });
        
        // Select the first two strategies for comparison
        const strategy1 = strategies[0].strategy_id;
        const strategy2 = strategies[1].strategy_id;
        
        // Set backtest period (last year)
        const endDate = moment().format('YYYY-MM-DD');
        const startDate = moment().subtract(1, 'year').format('YYYY-MM-DD');
        
        // Run backtests
        console.log(`\nRunning backtests from ${startDate} to ${endDate}...`);
        
        const backtest1 = await runBacktest(API_URL, HEADERS, strategy1, startDate, endDate);
        const backtest2 = await runBacktest(API_URL, HEADERS, strategy2, startDate, endDate);
        
        if (backtest1 && backtest2) {
            console.log('\nBacktest Comparison:');
            
            // Create comparison table
            const comparison = [
                ['Metric', backtest1.strategy_name, backtest2.strategy_name],
                ['Total Return', `${backtest1.total_return_percent.toFixed(2)}%`, `${backtest2.total_return_percent.toFixed(2)}%`],
                ['Annualized Return', `${backtest1.annualized_return.toFixed(2)}%`, `${backtest2.annualized_return.toFixed(2)}%`],
                ['Sharpe Ratio', backtest1.sharpe_ratio.toFixed(2), backtest2.sharpe_ratio.toFixed(2)],
                ['Max Drawdown', `${backtest1.max_drawdown.toFixed(2)}%`, `${backtest2.max_drawdown.toFixed(2)}%`],
                ['Win Rate', `${backtest1.win_rate.toFixed(2)}%`, `${backtest2.win_rate.toFixed(2)}%`],
                ['Total Trades', backtest1.total_trades, backtest2.total_trades]
            ];
            
            // Display comparison table
            comparison.forEach(row => {
                console.log(`  ${row[0].padEnd(20)} ${row[1].toString().padEnd(15)} ${row[2]}`);
            });
            
            // Determine winner
            let winner = '';
            if (backtest1.total_return > backtest2.total_return) {
                winner = backtest1.strategy_name;
            } else if (backtest2.total_return > backtest1.total_return) {
                winner = backtest2.strategy_name;
            } else {
                winner = 'Tie';
            }
            
            console.log(`\nWinner based on Total Return: ${winner}`);
        } else {
            console.log('Failed to run one or both backtests.');
        }
    } else {
        console.log('Failed to retrieve at least two strategies for comparison.');
    }
})();
```

## WebSocket Examples

### Python Example: Real-time Trading Signals

```python
import websocket
import json
import threading
import time

def on_message(ws, message):
    """Handle incoming WebSocket messages."""
    data = json.loads(message)
    
    if data.get("channel") == "trading_signals":
        signal = data.get("data", {})
        print(f"\nNew Trading Signal:")
        print(f"  Symbol: {signal.get('symbol')}")
        print(f"  Action: {signal.get('action')}")
        print(f"  Price: ${signal.get('price'):.2f}")
        print(f"  Confidence: {signal.get('confidence'):.2f}")
        print(f"  Model: {signal.get('model')}")
        print(f"  Timestamp: {signal.get('timestamp')}")
        
        # You could implement automatic trading here based on the signals

def on_error(ws, error):
    """Handle WebSocket errors."""
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket connection close."""
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    """Handle WebSocket connection open."""
    print("WebSocket connection established")
    
    # Subscribe to trading signals
    subscribe_message = {
        "action": "subscribe",
        "channel": "trading_signals",
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    }
    
    ws.send(json.dumps(subscribe_message))
    print(f"Subscribed to trading signals for {subscribe_message['symbols']}")

def get_websocket_url(api_url, headers):
    """Get the WebSocket URL from the API."""
    ws_info_endpoint = f"{api_url}/websocket-info"
    
    try:
        response = requests.get(ws_info_endpoint, headers=headers)
        response.raise_for_status()
        
        ws_info = response.json()
        return ws_info.get("websocket_url")
    except requests.exceptions.RequestException as e:
        print(f"Error getting WebSocket URL: {e}")
        return None

# Usage example
API_URL = "http://localhost:8000/api/v1"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get WebSocket URL
ws_url = get_websocket_url(API_URL, HEADERS)

if ws_url:
    # Add token to WebSocket URL
    if "?" in ws_url:
        ws_url += f"&token={token}"
    else:
        ws_url += f"?token={token}"
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Start WebSocket connection in a separate thread
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    print("Listening for trading signals. Press Ctrl+C to exit.")
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Close WebSocket connection on Ctrl+C
        ws.close()
        print("WebSocket connection closed")
else:
    print("Failed to get WebSocket URL.")
```

### JavaScript Example: Real-time Portfolio Updates

```javascript
const WebSocket = require('ws');
const axios = require('axios');

async function getWebSocketUrl(apiUrl, headers) {
    const wsInfoEndpoint = `${apiUrl}/websocket-info`;
    
    try {
        const response = await axios.get(wsInfoEndpoint, { headers });
        return response.data.websocket_url;
    } catch (error) {
        console.error('Error getting WebSocket URL:', error.message);
        return null;
    }
}

async function subscribeToPortfolioUpdates(apiUrl, token) {
    // Get WebSocket URL
    const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    };
    
    const wsUrl = await getWebSocketUrl(apiUrl, headers);
    
    if (!wsUrl) {
        console.error('Failed to get WebSocket URL.');
        return null;
    }
    
    // Add token to WebSocket URL
    const wsUrlWithToken = wsUrl.includes('?') 
        ? `${wsUrl}&token=${token}` 
        : `${wsUrl}?token=${token}`;
    
    // Create WebSocket connection
    const ws = new WebSocket(wsUrlWithToken);
    
    ws.on('open', () => {
        console.log('WebSocket connection established');
        
        // Subscribe to portfolio updates
        const subscribeMessage = {
            action: 'subscribe',
            channel: 'portfolio_updates'
        };
        
        ws.send(JSON.stringify(subscribeMessage));
        console.log('Subscribed to portfolio updates');
    });
    
    ws.on('message', (data) => {
        const message = JSON.parse(data);
        
        if (message.channel === 'portfolio_updates') {
            const update = message.data;
            
            console.log('\nPortfolio Update:');
            console.log(`  Total Value: $${update.total_value.toFixed(2)}`);
            console.log(`  Cash Balance: $${update.cash_balance.toFixed(2)}`);
            console.log(`  Invested Value: $${update.invested_value.toFixed(2)}`);
            
            if (update.recent_trades && update.recent_trades.length > 0) {
                console.log('  Recent Trades:');
                update.recent_trades.forEach(trade => {
                    console.log(`    ${trade.action.toUpperCase()} ${trade.quantity} ${trade.symbol} @ $${trade.price.toFixed(2)}`);
                });
            }
            
            if (update.position_changes && update.position_changes.length > 0) {
                console.log('  Position Changes:');
                update.position_changes.forEach(change => {
                    const direction = change.change_value >= 0 ? '↑' : '↓';
                    console.log(`    ${change.symbol}: ${direction} $${Math.abs(change.change_value).toFixed(2)} (${change.change_percent.toFixed(2)}%)`);
                });
            }
        }
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error.message);
    });
    
    ws.on('close', (code, reason) => {
        console.log(`WebSocket connection closed: ${code} - ${reason}`);
    });
    
    // Return the WebSocket instance for later use
    return ws;
}

// Usage example
const API_URL = 'http://localhost:8000/api/v1';
const TOKEN = 'your_jwt_token';

(async () => {
    const ws = await subscribeToPortfolioUpdates(API_URL, TOKEN);
    
    if (ws) {
        console.log('Listening for portfolio updates. Press Ctrl+C to exit.');
        
        // Handle process termination
        process.on('SIGINT', () => {
            console.log('Closing WebSocket connection...');
            
            if (ws.readyState === WebSocket.OPEN) {
                const unsubscribeMessage = {
                    action: 'unsubscribe',
                    channel: 'portfolio_updates'
                };
                
                ws.send(JSON.stringify(unsubscribeMessage));
                ws.close();
            }
            
            process.exit(0);
        });
    }
})();
```

## Error Handling

### Python Example: Robust Error Handling

```python
import requests
import json
import time
from requests.exceptions import RequestException, Timeout, ConnectionError

class FridayAPIClient:
    """A client for the Friday AI Trading System API with robust error handling."""
    
    def __init__(self, api_url, username=None, password=None, token=None, max_retries=3, retry_delay=1):
        self.api_url = api_url
        self.username = username
        self.password = password
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {}
        
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        elif username and password:
            self.authenticate()
        
        self.headers["Content-Type"] = "application/json"
    
    def authenticate(self):
        """Authenticate with the API and get a JWT token."""
        auth_endpoint = f"{self.api_url}/auth/token"
        
        payload = {
            "username": self.username,
            "password": self.password
        }
        
        try:
            response = self._make_request("POST", auth_endpoint, json=payload)
            self.token = response["access_token"]
            self.headers["Authorization"] = f"Bearer {self.token}"
            return True
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def _make_request(self, method, endpoint, params=None, json=None, retry_count=0):
        """Make a request to the API with retry logic and error handling."""
        try:
            if method == "GET":
                response = requests.get(endpoint, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(endpoint, headers=self.headers, json=json)
            elif method == "PUT":
                response = requests.put(endpoint, headers=self.headers, json=json)
            elif method == "DELETE":
                response = requests.delete(endpoint, headers=self.headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for API errors
            if response.status_code == 401:
                # Token expired, try to re-authenticate
                if self.username and self.password and retry_count < self.max_retries:
                    print("Token expired, re-authenticating...")
                    self.authenticate()
                    return self._make_request(method, endpoint, params, json, retry_count + 1)
                else:
                    raise Exception("Authentication failed: Invalid credentials or token expired")
            
            response.raise_for_status()
            
            return response.json()
        except Timeout:
            # Handle timeout errors
            if retry_count < self.max_retries:
                print(f"Request timed out, retrying ({retry_count + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
                return self._make_request(method, endpoint, params, json, retry_count + 1)
            else:
                raise Exception(f"Request timed out after {self.max_retries} retries")
        except ConnectionError:
            # Handle connection errors
            if retry_count < self.max_retries:
                print(f"Connection error, retrying ({retry_count + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
                return self._make_request(method, endpoint, params, json, retry_count + 1)
            else:
                raise Exception(f"Connection error after {self.max_retries} retries")
        except RequestException as e:
            # Handle other request errors
            if 500 <= e.response.status_code < 600 and retry_count < self.max_retries:
                print(f"Server error {e.response.status_code}, retrying ({retry_count + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
                return self._make_request(method, endpoint, params, json, retry_count + 1)
            else:
                error_message = str(e)
                try:
                    error_data = e.response.json()
                    if "detail" in error_data:
                        error_message = error_data["detail"]
                except:
                    pass
                raise Exception(f"API error ({e.response.status_code}): {error_message}")
    
    def get_system_status(self):
        """Get the system status."""
        status_endpoint = f"{self.api_url}/system/status"
        return self._make_request("GET", status_endpoint)
    
    def get_historical_data(self, symbol, interval, start_date, end_date):
        """Get historical market data."""
        historical_endpoint = f"{self.api_url}/market/historical"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date
        }
        
        response = self._make_request("GET", historical_endpoint, params=params)
        return response["data"]
    
    def place_order(self, symbol, order_type, side, quantity, price=None):
        """Place a trading order."""
        order_endpoint = f"{self.api_url}/trading/orders"
        
        payload = {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity
        }
        
        if order_type in ["limit", "stop_limit"] and price is not None:
            payload["price"] = price
        
        return self._make_request("POST", order_endpoint, json=payload)
    
    def get_portfolio_summary(self):
        """Get the portfolio summary."""
        portfolio_endpoint = f"{self.api_url}/portfolio/summary"
        return self._make_request("GET", portfolio_endpoint)

# Usage example
API_URL = "http://localhost:8000/api/v1"
USERNAME = "your_username"
PASSWORD = "your_password"

# Create API client
client = FridayAPIClient(API_URL, username=USERNAME, password=PASSWORD)

try:
    # Check system status
    status = client.get_system_status()
    print("System Status:")
    print(f"  API Server: {status['api_server']}")
    print(f"  MongoDB: {status['mongodb']}")
    print(f"  Redis: {status['redis']}")
    
    # Get portfolio summary
    portfolio = client.get_portfolio_summary()
    print("\nPortfolio Summary:")
    print(f"  Total Value: ${portfolio['total_value']:.2f}")
    print(f"  Cash Balance: ${portfolio['cash_balance']:.2f}")
    print(f"  Invested Value: ${portfolio['invested_value']:.2f}")
    
    # Place an order
    order = client.place_order("AAPL", "market", "buy", 10)
    print("\nOrder placed successfully:")
    print(f"  Order ID: {order['order_id']}")
    print(f"  Symbol: {order['symbol']}")
    print(f"  Type: {order['order_type']}")
    print(f"  Side: {order['side']}")
    print(f"  Quantity: {order['quantity']}")
    print(f"  Status: {order['status']}")
except Exception as e:
    print(f"Error: {e}")

# End of examples

This document provides practical examples to help you get started with the Friday AI Trading System API. For more detailed information about the API endpoints and parameters, please refer to the [API Documentation](API_DOCUMENTATION.md).