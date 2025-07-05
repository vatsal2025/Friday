# Friday AI Trading System - API Documentation

## Overview

The Friday AI Trading System provides a comprehensive RESTful API that allows users to interact with the system programmatically. This document provides detailed information about the available endpoints, request/response formats, authentication, and best practices.

## Table of Contents

1. [API Basics](#api-basics)
2. [Authentication](#authentication)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [Endpoints](#endpoints)
   - [System](#system-endpoints)
   - [Market Data](#market-data-endpoints)
   - [Trading](#trading-endpoints)
   - [Portfolio](#portfolio-endpoints)
   - [Models](#model-endpoints)
   - [Backtesting](#backtesting-endpoints)
   - [Analytics](#analytics-endpoints)
6. [WebSocket API](#websocket-api)
7. [Examples](#examples)
8. [Client Libraries](#client-libraries)
9. [API Versioning](#api-versioning)

## API Basics

### Base URL

The base URL for all API endpoints is:

```
http://<host>:<port>/api/v1
```

Where `<host>` and `<port>` are the hostname and port where the Friday API server is running. By default, this is `localhost:8000` for local development.

### Request Format

All requests should be sent with the appropriate HTTP method (GET, POST, PUT, DELETE) and include the required headers and parameters.

#### Headers

- `Content-Type`: `application/json` for all requests with a body
- `Authorization`: `Bearer <token>` for authenticated requests

#### Request Body

For POST and PUT requests, the request body should be a valid JSON object containing the required parameters.

### Response Format

All responses are returned as JSON objects with the following structure:

```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

Or in case of an error:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message"
  }
}
```

## Authentication

The Friday API uses JWT (JSON Web Tokens) for authentication. To access protected endpoints, you need to include a valid JWT token in the `Authorization` header of your requests.

### Obtaining a Token

To obtain a token, send a POST request to the `/api/v1/auth/login` endpoint with your credentials:

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

If the credentials are valid, the server will respond with a token:

```json
{
  "status": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_at": "2023-12-31T23:59:59Z"
  },
  "message": "Login successful"
}
```

### Using the Token

Include the token in the `Authorization` header of your requests:

```http
GET /api/v1/portfolio/positions
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Expiration

Tokens have a limited lifetime (typically 24 hours). When a token expires, you need to obtain a new one by logging in again.

### Token Refresh

To avoid frequent logins, you can refresh your token before it expires by sending a POST request to the `/api/v1/auth/refresh` endpoint with your current token:

```http
POST /api/v1/auth/refresh
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

If the token is valid, the server will respond with a new token:

```json
{
  "status": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_at": "2023-12-31T23:59:59Z"
  },
  "message": "Token refreshed successfully"
}
```

## Error Handling

The Friday API uses standard HTTP status codes to indicate the success or failure of a request. In addition, the response body contains detailed information about the error.

### HTTP Status Codes

- `200 OK`: The request was successful
- `201 Created`: The resource was created successfully
- `400 Bad Request`: The request was invalid or cannot be served
- `401 Unauthorized`: Authentication failed or user doesn't have permissions
- `403 Forbidden`: The request is valid but the user doesn't have permissions
- `404 Not Found`: The requested resource was not found
- `422 Unprocessable Entity`: The request was well-formed but was unable to be followed due to semantic errors
- `429 Too Many Requests`: The user has sent too many requests in a given amount of time
- `500 Internal Server Error`: An error occurred on the server

### Error Codes

In addition to HTTP status codes, the API returns specific error codes in the response body to provide more detailed information about the error.

Example error codes:

- `INVALID_CREDENTIALS`: The provided credentials are invalid
- `TOKEN_EXPIRED`: The provided token has expired
- `INSUFFICIENT_PERMISSIONS`: The user doesn't have sufficient permissions
- `RESOURCE_NOT_FOUND`: The requested resource was not found
- `VALIDATION_ERROR`: The request contains invalid data
- `RATE_LIMIT_EXCEEDED`: The user has exceeded the rate limit
- `INTERNAL_ERROR`: An internal error occurred on the server

## Rate Limiting

To ensure fair usage and protect the system from abuse, the Friday API implements rate limiting. Each user is allowed a certain number of requests per minute.

### Rate Limit Headers

The API includes the following headers in each response to provide information about the rate limit:

- `X-RateLimit-Limit`: The maximum number of requests allowed per minute
- `X-RateLimit-Remaining`: The number of requests remaining in the current minute
- `X-RateLimit-Reset`: The time at which the rate limit will reset (Unix timestamp)

### Rate Limit Exceeded

If you exceed the rate limit, the API will respond with a `429 Too Many Requests` status code and the following response body:

```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later."
  }
}
```

## Endpoints

### System Endpoints

#### Get System Status

```http
GET /api/v1/system/status
```

Returns the current status of the system, including the status of each component (MongoDB, Redis, MCP servers, API server).

**Response:**

```json
{
  "status": "success",
  "data": {
    "system_status": "operational",
    "components": {
      "mongodb": {
        "status": "operational",
        "version": "4.4.6"
      },
      "redis": {
        "status": "operational",
        "version": "6.2.5"
      },
      "mcp_servers": {
        "memory": {
          "status": "operational"
        },
        "sequential_thinking": {
          "status": "operational"
        }
      },
      "api_server": {
        "status": "operational",
        "version": "1.0.0"
      }
    },
    "uptime": 86400,
    "last_updated": "2023-01-01T00:00:00Z"
  },
  "message": "System status retrieved successfully"
}
```

#### Get System Configuration

```http
GET /api/v1/system/configuration
```

Returns the current system configuration (excluding sensitive information).

**Response:**

```json
{
  "status": "success",
  "data": {
    "configuration": {
      "database": {
        "host": "localhost",
        "port": 27017,
        "database": "friday"
      },
      "cache": {
        "host": "localhost",
        "port": 6379
      },
      "api": {
        "host": "0.0.0.0",
        "port": 8000
      },
      "trading": {
        "default_risk_per_trade": 0.02,
        "default_stop_loss_percent": 0.05
      }
    }
  },
  "message": "System configuration retrieved successfully"
}
```

### Market Data Endpoints

#### Get Market Data

```http
GET /api/v1/market/data
```

Returns market data for the specified symbol and timeframe.

**Parameters:**

- `symbol` (required): The trading symbol (e.g., AAPL, BTCUSD)
- `timeframe` (required): The timeframe (e.g., 1m, 5m, 15m, 1h, 1d)
- `start_time` (optional): The start time in ISO 8601 format (e.g., 2023-01-01T00:00:00Z)
- `end_time` (optional): The end time in ISO 8601 format (e.g., 2023-01-31T23:59:59Z)
- `limit` (optional): The maximum number of data points to return (default: 100)

**Response:**

```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "timeframe": "1d",
    "data": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 153.0,
        "volume": 1000000
      },
      {
        "timestamp": "2023-01-02T00:00:00Z",
        "open": 153.0,
        "high": 158.0,
        "low": 152.0,
        "close": 157.0,
        "volume": 1200000
      }
    ]
  },
  "message": "Market data retrieved successfully"
}
```

#### Get Available Symbols

```http
GET /api/v1/market/symbols
```

Returns a list of available trading symbols.

**Parameters:**

- `exchange` (optional): Filter symbols by exchange (e.g., NYSE, NASDAQ)
- `asset_class` (optional): Filter symbols by asset class (e.g., stock, crypto, forex)

**Response:**

```json
{
  "status": "success",
  "data": {
    "symbols": [
      {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "exchange": "NASDAQ",
        "asset_class": "stock"
      },
      {
        "symbol": "MSFT",
        "name": "Microsoft Corporation",
        "exchange": "NASDAQ",
        "asset_class": "stock"
      },
      {
        "symbol": "BTCUSD",
        "name": "Bitcoin/USD",
        "exchange": "Coinbase",
        "asset_class": "crypto"
      }
    ]
  },
  "message": "Symbols retrieved successfully"
}
```

### Trading Endpoints

#### Place Order

```http
POST /api/v1/trading/orders
Content-Type: application/json
Authorization: Bearer <token>

{
  "symbol": "AAPL",
  "side": "buy",
  "type": "market",
  "quantity": 10,
  "time_in_force": "day"
}
```

Places a new order.

**Parameters:**

- `symbol` (required): The trading symbol (e.g., AAPL, BTCUSD)
- `side` (required): The order side (buy, sell)
- `type` (required): The order type (market, limit, stop, stop_limit)
- `quantity` (required): The order quantity
- `price` (required for limit and stop_limit orders): The order price
- `stop_price` (required for stop and stop_limit orders): The stop price
- `time_in_force` (optional): The time in force (day, gtc, ioc, fok)

**Response:**

```json
{
  "status": "success",
  "data": {
    "order": {
      "id": "12345",
      "symbol": "AAPL",
      "side": "buy",
      "type": "market",
      "quantity": 10,
      "status": "filled",
      "filled_quantity": 10,
      "average_price": 153.5,
      "time_in_force": "day",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:01Z"
    }
  },
  "message": "Order placed successfully"
}
```

#### Get Orders

```http
GET /api/v1/trading/orders
Authorization: Bearer <token>
```

Returns a list of orders.

**Parameters:**

- `status` (optional): Filter orders by status (open, filled, canceled, all)
- `symbol` (optional): Filter orders by symbol
- `side` (optional): Filter orders by side (buy, sell)
- `start_time` (optional): Filter orders by creation time (start)
- `end_time` (optional): Filter orders by creation time (end)
- `limit` (optional): The maximum number of orders to return (default: 100)

**Response:**

```json
{
  "status": "success",
  "data": {
    "orders": [
      {
        "id": "12345",
        "symbol": "AAPL",
        "side": "buy",
        "type": "market",
        "quantity": 10,
        "status": "filled",
        "filled_quantity": 10,
        "average_price": 153.5,
        "time_in_force": "day",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:01Z"
      },
      {
        "id": "12346",
        "symbol": "MSFT",
        "side": "sell",
        "type": "limit",
        "quantity": 5,
        "price": 300.0,
        "status": "open",
        "filled_quantity": 0,
        "average_price": null,
        "time_in_force": "gtc",
        "created_at": "2023-01-01T12:30:00Z",
        "updated_at": "2023-01-01T12:30:00Z"
      }
    ]
  },
  "message": "Orders retrieved successfully"
}
```

#### Get Order

```http
GET /api/v1/trading/orders/{order_id}
Authorization: Bearer <token>
```

Returns details for a specific order.

**Parameters:**

- `order_id` (required): The order ID

**Response:**

```json
{
  "status": "success",
  "data": {
    "order": {
      "id": "12345",
      "symbol": "AAPL",
      "side": "buy",
      "type": "market",
      "quantity": 10,
      "status": "filled",
      "filled_quantity": 10,
      "average_price": 153.5,
      "time_in_force": "day",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:01Z"
    }
  },
  "message": "Order retrieved successfully"
}
```

#### Cancel Order

```http
DELETE /api/v1/trading/orders/{order_id}
Authorization: Bearer <token>
```

Cancels a specific order.

**Parameters:**

- `order_id` (required): The order ID

**Response:**

```json
{
  "status": "success",
  "data": {
    "order": {
      "id": "12346",
      "symbol": "MSFT",
      "side": "sell",
      "type": "limit",
      "quantity": 5,
      "price": 300.0,
      "status": "canceled",
      "filled_quantity": 0,
      "average_price": null,
      "time_in_force": "gtc",
      "created_at": "2023-01-01T12:30:00Z",
      "updated_at": "2023-01-01T12:35:00Z"
    }
  },
  "message": "Order canceled successfully"
}
```

### Portfolio Endpoints

#### Get Portfolio

```http
GET /api/v1/portfolio
Authorization: Bearer <token>
```

Returns the current portfolio.

**Response:**

```json
{
  "status": "success",
  "data": {
    "portfolio": {
      "total_value": 100000.0,
      "cash": 50000.0,
      "positions": [
        {
          "symbol": "AAPL",
          "quantity": 100,
          "average_price": 150.0,
          "current_price": 155.0,
          "market_value": 15500.0,
          "unrealized_pl": 500.0,
          "unrealized_pl_percent": 3.33
        },
        {
          "symbol": "MSFT",
          "quantity": 50,
          "average_price": 280.0,
          "current_price": 290.0,
          "market_value": 14500.0,
          "unrealized_pl": 500.0,
          "unrealized_pl_percent": 3.57
        }
      ],
      "allocation": {
        "cash": 50.0,
        "stocks": 30.0,
        "crypto": 20.0
      }
    }
  },
  "message": "Portfolio retrieved successfully"
}
```

#### Get Positions

```http
GET /api/v1/portfolio/positions
Authorization: Bearer <token>
```

Returns the current positions.

**Parameters:**

- `symbol` (optional): Filter positions by symbol
- `asset_class` (optional): Filter positions by asset class (e.g., stock, crypto, forex)

**Response:**

```json
{
  "status": "success",
  "data": {
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 100,
        "average_price": 150.0,
        "current_price": 155.0,
        "market_value": 15500.0,
        "unrealized_pl": 500.0,
        "unrealized_pl_percent": 3.33
      },
      {
        "symbol": "MSFT",
        "quantity": 50,
        "average_price": 280.0,
        "current_price": 290.0,
        "market_value": 14500.0,
        "unrealized_pl": 500.0,
        "unrealized_pl_percent": 3.57
      }
    ]
  },
  "message": "Positions retrieved successfully"
}
```

#### Get Transactions

```http
GET /api/v1/portfolio/transactions
Authorization: Bearer <token>
```

Returns the transaction history.

**Parameters:**

- `symbol` (optional): Filter transactions by symbol
- `type` (optional): Filter transactions by type (buy, sell, dividend, deposit, withdrawal)
- `start_time` (optional): Filter transactions by time (start)
- `end_time` (optional): Filter transactions by time (end)
- `limit` (optional): The maximum number of transactions to return (default: 100)

**Response:**

```json
{
  "status": "success",
  "data": {
    "transactions": [
      {
        "id": "12345",
        "type": "buy",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 153.5,
        "amount": 1535.0,
        "fees": 5.0,
        "timestamp": "2023-01-01T12:00:01Z"
      },
      {
        "id": "12346",
        "type": "sell",
        "symbol": "MSFT",
        "quantity": 5,
        "price": 290.0,
        "amount": 1450.0,
        "fees": 5.0,
        "timestamp": "2023-01-01T12:35:01Z"
      }
    ]
  },
  "message": "Transactions retrieved successfully"
}
```

### Model Endpoints

#### Get Models

```http
GET /api/v1/models
Authorization: Bearer <token>
```

Returns a list of available models.

**Parameters:**

- `type` (optional): Filter models by type (e.g., classification, regression, reinforcement_learning)
- `status` (optional): Filter models by status (e.g., active, inactive, training)

**Response:**

```json
{
  "status": "success",
  "data": {
    "models": [
      {
        "id": "model_1",
        "name": "Price Prediction Model",
        "type": "regression",
        "description": "Predicts future price movements based on historical data",
        "status": "active",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "performance": {
          "accuracy": 0.85,
          "precision": 0.87,
          "recall": 0.83,
          "f1_score": 0.85
        }
      },
      {
        "id": "model_2",
        "name": "Trend Classification Model",
        "type": "classification",
        "description": "Classifies market trends as bullish, bearish, or neutral",
        "status": "active",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "performance": {
          "accuracy": 0.78,
          "precision": 0.80,
          "recall": 0.76,
          "f1_score": 0.78
        }
      }
    ]
  },
  "message": "Models retrieved successfully"
}
```

#### Get Model

```http
GET /api/v1/models/{model_id}
Authorization: Bearer <token>
```

Returns details for a specific model.

**Parameters:**

- `model_id` (required): The model ID

**Response:**

```json
{
  "status": "success",
  "data": {
    "model": {
      "id": "model_1",
      "name": "Price Prediction Model",
      "type": "regression",
      "description": "Predicts future price movements based on historical data",
      "status": "active",
      "created_at": "2023-01-01T00:00:00Z",
      "updated_at": "2023-01-01T00:00:00Z",
      "performance": {
        "accuracy": 0.85,
        "precision": 0.87,
        "recall": 0.83,
        "f1_score": 0.85
      },
      "features": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_20",
        "ema_50",
        "rsi_14",
        "macd_12_26_9"
      ],
      "target": "close_1d",
      "hyperparameters": {
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 100
      }
    }
  },
  "message": "Model retrieved successfully"
}
```

#### Get Model Predictions

```http
GET /api/v1/models/{model_id}/predictions
Authorization: Bearer <token>
```

Returns predictions from a specific model.

**Parameters:**

- `model_id` (required): The model ID
- `symbol` (required): The trading symbol (e.g., AAPL, BTCUSD)
- `timeframe` (optional): The timeframe (e.g., 1m, 5m, 15m, 1h, 1d)

**Response:**

```json
{
  "status": "success",
  "data": {
    "predictions": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "symbol": "AAPL",
        "prediction": 155.0,
        "confidence": 0.85
      },
      {
        "timestamp": "2023-01-02T00:00:00Z",
        "symbol": "AAPL",
        "prediction": 157.0,
        "confidence": 0.82
      }
    ]
  },
  "message": "Predictions retrieved successfully"
}
```

### Backtesting Endpoints

#### Create Backtest

```http
POST /api/v1/backtesting/backtests
Content-Type: application/json
Authorization: Bearer <token>

{
  "strategy_id": "strategy_1",
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "initial_capital": 100000,
  "parameters": {
    "sma_short": 20,
    "sma_long": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  }
}
```

Creates a new backtest.

**Parameters:**

- `strategy_id` (required): The strategy ID
- `symbols` (required): An array of trading symbols
- `start_date` (required): The start date in ISO 8601 format
- `end_date` (required): The end date in ISO 8601 format
- `initial_capital` (required): The initial capital
- `parameters` (optional): Strategy-specific parameters

**Response:**

```json
{
  "status": "success",
  "data": {
    "backtest": {
      "id": "backtest_1",
      "strategy_id": "strategy_1",
      "symbols": ["AAPL", "MSFT"],
      "start_date": "2023-01-01T00:00:00Z",
      "end_date": "2023-01-31T23:59:59Z",
      "initial_capital": 100000,
      "parameters": {
        "sma_short": 20,
        "sma_long": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
      },
      "status": "running",
      "created_at": "2023-02-01T00:00:00Z"
    }
  },
  "message": "Backtest created successfully"
}
```

#### Get Backtests

```http
GET /api/v1/backtesting/backtests
Authorization: Bearer <token>
```

Returns a list of backtests.

**Parameters:**

- `strategy_id` (optional): Filter backtests by strategy ID
- `status` (optional): Filter backtests by status (e.g., running, completed, failed)
- `limit` (optional): The maximum number of backtests to return (default: 100)

**Response:**

```json
{
  "status": "success",
  "data": {
    "backtests": [
      {
        "id": "backtest_1",
        "strategy_id": "strategy_1",
        "symbols": ["AAPL", "MSFT"],
        "start_date": "2023-01-01T00:00:00Z",
        "end_date": "2023-01-31T23:59:59Z",
        "initial_capital": 100000,
        "status": "completed",
        "created_at": "2023-02-01T00:00:00Z",
        "completed_at": "2023-02-01T00:05:00Z",
        "performance": {
          "total_return": 0.05,
          "annualized_return": 0.6,
          "sharpe_ratio": 1.2,
          "max_drawdown": 0.02
        }
      },
      {
        "id": "backtest_2",
        "strategy_id": "strategy_2",
        "symbols": ["BTCUSD"],
        "start_date": "2023-01-01T00:00:00Z",
        "end_date": "2023-01-31T23:59:59Z",
        "initial_capital": 100000,
        "status": "running",
        "created_at": "2023-02-01T00:10:00Z"
      }
    ]
  },
  "message": "Backtests retrieved successfully"
}
```

#### Get Backtest

```http
GET /api/v1/backtesting/backtests/{backtest_id}
Authorization: Bearer <token>
```

Returns details for a specific backtest.

**Parameters:**

- `backtest_id` (required): The backtest ID

**Response:**

```json
{
  "status": "success",
  "data": {
    "backtest": {
      "id": "backtest_1",
      "strategy_id": "strategy_1",
      "symbols": ["AAPL", "MSFT"],
      "start_date": "2023-01-01T00:00:00Z",
      "end_date": "2023-01-31T23:59:59Z",
      "initial_capital": 100000,
      "parameters": {
        "sma_short": 20,
        "sma_long": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
      },
      "status": "completed",
      "created_at": "2023-02-01T00:00:00Z",
      "completed_at": "2023-02-01T00:05:00Z",
      "performance": {
        "total_return": 0.05,
        "annualized_return": 0.6,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.02,
        "win_rate": 0.65,
        "profit_factor": 1.8,
        "average_win": 0.02,
        "average_loss": 0.01,
        "largest_win": 0.05,
        "largest_loss": 0.03
      },
      "equity_curve": [
        {
          "timestamp": "2023-01-01T00:00:00Z",
          "equity": 100000
        },
        {
          "timestamp": "2023-01-02T00:00:00Z",
          "equity": 100500
        },
        {
          "timestamp": "2023-01-03T00:00:00Z",
          "equity": 101000
        }
      ],
      "trades": [
        {
          "id": "trade_1",
          "symbol": "AAPL",
          "side": "buy",
          "entry_time": "2023-01-02T10:00:00Z",
          "entry_price": 150.0,
          "exit_time": "2023-01-03T14:00:00Z",
          "exit_price": 155.0,
          "quantity": 10,
          "profit_loss": 50.0,
          "profit_loss_percent": 3.33
        },
        {
          "id": "trade_2",
          "symbol": "MSFT",
          "side": "buy",
          "entry_time": "2023-01-05T09:30:00Z",
          "entry_price": 280.0,
          "exit_time": "2023-01-07T15:45:00Z",
          "exit_price": 290.0,
          "quantity": 5,
          "profit_loss": 50.0,
          "profit_loss_percent": 3.57
        }
      ]
    }
  },
  "message": "Backtest retrieved successfully"
}
```

### Analytics Endpoints

#### Get Performance Metrics

```http
GET /api/v1/analytics/performance
Authorization: Bearer <token>
```

Returns performance metrics for the portfolio.

**Parameters:**

- `start_date` (optional): The start date in ISO 8601 format
- `end_date` (optional): The end date in ISO 8601 format

**Response:**

```json
{
  "status": "success",
  "data": {
    "performance": {
      "total_return": 0.05,
      "annualized_return": 0.6,
      "sharpe_ratio": 1.2,
      "sortino_ratio": 1.5,
      "max_drawdown": 0.02,
      "win_rate": 0.65,
      "profit_factor": 1.8,
      "average_win": 0.02,
      "average_loss": 0.01,
      "largest_win": 0.05,
      "largest_loss": 0.03,
      "equity_curve": [
        {
          "timestamp": "2023-01-01T00:00:00Z",
          "equity": 100000
        },
        {
          "timestamp": "2023-01-02T00:00:00Z",
          "equity": 100500
        },
        {
          "timestamp": "2023-01-03T00:00:00Z",
          "equity": 101000
        }
      ]
    }
  },
  "message": "Performance metrics retrieved successfully"
}
```

#### Get Trade Statistics

```http
GET /api/v1/analytics/trades
Authorization: Bearer <token>
```

Returns statistics for trades.

**Parameters:**

- `symbol` (optional): Filter trades by symbol
- `start_date` (optional): The start date in ISO 8601 format
- `end_date` (optional): The end date in ISO 8601 format

**Response:**

```json
{
  "status": "success",
  "data": {
    "trade_statistics": {
      "total_trades": 100,
      "winning_trades": 65,
      "losing_trades": 35,
      "win_rate": 0.65,
      "profit_factor": 1.8,
      "average_win": 0.02,
      "average_loss": 0.01,
      "largest_win": 0.05,
      "largest_loss": 0.03,
      "average_holding_period": 2.5,
      "trades_by_symbol": {
        "AAPL": {
          "total_trades": 50,
          "winning_trades": 35,
          "losing_trades": 15,
          "win_rate": 0.7,
          "profit_factor": 2.0
        },
        "MSFT": {
          "total_trades": 50,
          "winning_trades": 30,
          "losing_trades": 20,
          "win_rate": 0.6,
          "profit_factor": 1.6
        }
      }
    }
  },
  "message": "Trade statistics retrieved successfully"
}
```

## WebSocket API

The Friday AI Trading System also provides a WebSocket API for real-time updates.

### Connection

To connect to the WebSocket API, use the following URL:

```
ws://<host>:<port>/api/v1/ws
```

Where `<host>` and `<port>` are the hostname and port where the Friday API server is running. By default, this is `localhost:8000` for local development.

### Authentication

To authenticate with the WebSocket API, include your JWT token as a query parameter:

```
ws://<host>:<port>/api/v1/ws?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Messages

All messages sent and received through the WebSocket API are JSON objects.

#### Subscribe to Market Data

```json
{
  "action": "subscribe",
  "channel": "market_data",
  "symbols": ["AAPL", "MSFT"],
  "timeframe": "1m"
}
```

#### Market Data Update

```json
{
  "channel": "market_data",
  "data": {
    "symbol": "AAPL",
    "timeframe": "1m",
    "timestamp": "2023-01-01T12:00:00Z",
    "open": 150.0,
    "high": 155.0,
    "low": 149.0,
    "close": 153.0,
    "volume": 1000000
  }
}
```

#### Subscribe to Order Updates

```json
{
  "action": "subscribe",
  "channel": "orders"
}
```

#### Order Update

```json
{
  "channel": "orders",
  "data": {
    "id": "12345",
    "symbol": "AAPL",
    "side": "buy",
    "type": "market",
    "quantity": 10,
    "status": "filled",
    "filled_quantity": 10,
    "average_price": 153.5,
    "time_in_force": "day",
    "created_at": "2023-01-01T12:00:00Z",
    "updated_at": "2023-01-01T12:00:01Z"
  }
}
```

#### Subscribe to Portfolio Updates

```json
{
  "action": "subscribe",
  "channel": "portfolio"
}
```

#### Portfolio Update

```json
{
  "channel": "portfolio",
  "data": {
    "total_value": 100000.0,
    "cash": 50000.0,
    "positions": [
      {
        "symbol": "AAPL",
        "quantity": 100,
        "average_price": 150.0,
        "current_price": 155.0,
        "market_value": 15500.0,
        "unrealized_pl": 500.0,
        "unrealized_pl_percent": 3.33
      },
      {
        "symbol": "MSFT",
        "quantity": 50,
        "average_price": 280.0,
        "current_price": 290.0,
        "market_value": 14500.0,
        "unrealized_pl": 500.0,
        "unrealized_pl_percent": 3.57
      }
    ]
  }
}
```

#### Unsubscribe

```json
{
  "action": "unsubscribe",
  "channel": "market_data",
  "symbols": ["AAPL", "MSFT"]
}
```

## Examples

### Python Example

```python
import requests
import json
import websocket
import threading

# API Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Authentication
def login(username, password):
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": username, "password": password}
    )
    return response.json()["data"]["token"]

# Get Market Data
def get_market_data(token, symbol, timeframe, start_time=None, end_time=None, limit=100):
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit
    }
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    
    response = requests.get(
        f"{BASE_URL}/market/data",
        headers=headers,
        params=params
    )
    return response.json()

# Place Order
def place_order(token, symbol, side, order_type, quantity, price=None, stop_price=None, time_in_force="day"):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "time_in_force": time_in_force
    }
    if price:
        data["price"] = price
    if stop_price:
        data["stop_price"] = stop_price
    
    response = requests.post(
        f"{BASE_URL}/trading/orders",
        headers=headers,
        json=data
    )
    return response.json()

# WebSocket Example
def on_message(ws, message):
    print(f"Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    # Subscribe to market data
    ws.send(json.dumps({
        "action": "subscribe",
        "channel": "market_data",
        "symbols": ["AAPL", "MSFT"],
        "timeframe": "1m"
    }))

def start_websocket(token):
    ws_url = f"ws://localhost:8000/api/v1/ws?token={token}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    return ws

# Example Usage
def main():
    # Login
    token = login("your_username", "your_password")
    print(f"Token: {token}")
    
    # Get Market Data
    market_data = get_market_data(token, "AAPL", "1d", limit=10)
    print(f"Market Data: {json.dumps(market_data, indent=2)}")
    
    # Place Order
    order = place_order(token, "AAPL", "buy", "market", 10)
    print(f"Order: {json.dumps(order, indent=2)}")
    
    # Start WebSocket
    ws = start_websocket(token)
    
    # Keep the main thread running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        ws.close()

if __name__ == "__main__":
    main()
```

### JavaScript Example

```javascript
// API Base URL
const BASE_URL = "http://localhost:8000/api/v1";

// Authentication
async function login(username, password) {
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ username, password })
  });
  const data = await response.json();
  return data.data.token;
}

// Get Market Data
async function getMarketData(token, symbol, timeframe, startTime = null, endTime = null, limit = 100) {
  const params = new URLSearchParams({
    symbol,
    timeframe,
    limit
  });
  if (startTime) params.append("start_time", startTime);
  if (endTime) params.append("end_time", endTime);
  
  const response = await fetch(`${BASE_URL}/market/data?${params.toString()}`, {
    headers: {
      "Authorization": `Bearer ${token}`
    }
  });
  return response.json();
}

// Place Order
async function placeOrder(token, symbol, side, orderType, quantity, price = null, stopPrice = null, timeInForce = "day") {
  const data = {
    symbol,
    side,
    type: orderType,
    quantity,
    time_in_force: timeInForce
  };
  if (price) data.price = price;
  if (stopPrice) data.stop_price = stopPrice;
  
  const response = await fetch(`${BASE_URL}/trading/orders`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  });
  return response.json();
}

// WebSocket Example
function startWebSocket(token) {
  const ws = new WebSocket(`ws://localhost:8000/api/v1/ws?token=${token}`);
  
  ws.onopen = () => {
    console.log("Connection opened");
    // Subscribe to market data
    ws.send(JSON.stringify({
      action: "subscribe",
      channel: "market_data",
      symbols: ["AAPL", "MSFT"],
      timeframe: "1m"
    }));
  };
  
  ws.onmessage = (event) => {
    console.log(`Received: ${event.data}`);
  };
  
  ws.onerror = (error) => {
    console.error(`Error: ${error}`);
  };
  
  ws.onclose = () => {
    console.log("Connection closed");
  };
  
  return ws;
}

// Example Usage
async function main() {
  try {
    // Login
    const token = await login("your_username", "your_password");
    console.log(`Token: ${token}`);
    
    // Get Market Data
    const marketData = await getMarketData(token, "AAPL", "1d", null, null, 10);
    console.log("Market Data:", marketData);
    
    // Place Order
    const order = await placeOrder(token, "AAPL", "buy", "market", 10);
    console.log("Order:", order);
    
    // Start WebSocket
    const ws = startWebSocket(token);
    
    // Close WebSocket after 1 minute
    setTimeout(() => {
      ws.close();
    }, 60000);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
```

## Client Libraries

The Friday AI Trading System provides official client libraries for several programming languages:

- Python: [friday-client-python](https://github.com/yourusername/friday-client-python)
- JavaScript: [friday-client-js](https://github.com/yourusername/friday-client-js)
- Java: [friday-client-java](https://github.com/yourusername/friday-client-java)
- C#: [friday-client-csharp](https://github.com/yourusername/friday-client-csharp)

These client libraries provide a convenient way to interact with the Friday API without having to deal with the low-level details of HTTP requests and WebSocket connections.

## API Versioning

The Friday API uses versioning to ensure backward compatibility. The current version is `v1`.

When a new version is released, the previous version will continue to be supported for a period of time to allow clients to migrate to the new version.

The version is included in the URL path (e.g., `/api/v1/market/data`). When a new version is released, it will be available at a new URL path (e.g., `/api/v2/market/data`).

Changes that may warrant a new version include:

- Breaking changes to existing endpoints
- Significant changes to the request or response format
- Removal of deprecated endpoints

Minor changes that don't break backward compatibility (e.g., adding new endpoints, adding new optional parameters) will be made within the current version.