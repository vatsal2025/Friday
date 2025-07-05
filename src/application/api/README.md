# Friday AI Trading System - API Gateway

## Overview

The API Gateway serves as the central entry point for all external interactions with the Friday AI Trading System. It provides a unified interface for accessing various services including broker operations, notifications, and event streaming.

## Features

- **Authentication & Authorization**: Secure access using JWT tokens and API keys with scope-based permissions
- **Rate Limiting**: Protection against abuse with configurable rate limits
- **Real-time Event Streaming**: Server-Sent Events (SSE) for real-time updates
- **Broker Integration**: Endpoints for placing, modifying, and canceling orders
- **Notification System**: Send and manage notifications across multiple channels
- **Logging & Monitoring**: Comprehensive request logging and monitoring

## Architecture

The API Gateway is built using FastAPI and follows a modular design with the following components:

- **API Gateway Core** (`api_gateway.py`): Handles authentication, rate limiting, and CORS
- **Auth Router** (`auth_router.py`): Manages user authentication and API key operations
- **Broker Router** (`broker_router.py`): Handles broker-related operations
- **Notification Router** (`notification_router.py`): Manages notification operations
- **Event Router** (`event_router.py`): Provides real-time event streaming
- **Main Application** (`main.py`): Entry point that assembles all components

## API Endpoints

### Authentication

- `POST /auth/register`: Register a new user
- `POST /auth/login`: Login and get JWT token
- `GET /auth/me`: Get current user information
- `GET /auth/status`: Check authentication status
- `POST /auth/api-keys`: Create a new API key
- `DELETE /auth/api-keys/{key_id}`: Delete an API key

### Broker Operations

- `POST /broker/orders`: Place a new order
- `PUT /broker/orders/{order_id}`: Modify an existing order
- `DELETE /broker/orders/{order_id}`: Cancel an order
- `GET /broker/market-data/{symbol}`: Get market data for a symbol
- `POST /broker/authenticate`: Authenticate with a broker
- `GET /broker/status`: Get broker connection status

### Notifications

- `POST /notifications/send`: Send a notification
- `GET /notifications/history`: Get notification history
- `GET /notifications/preferences`: Get notification preferences
- `PUT /notifications/preferences`: Update notification preferences
- `GET /notifications/channels`: Get available notification channels

### Events

- `GET /events/stream`: Stream events in real-time (SSE)
- `POST /events/emit`: Emit a new event
- `GET /events/history`: Get event history

### System

- `GET /health`: Health check endpoint
- `GET /version`: Get API version

## Authentication

The API Gateway supports two authentication methods:

1. **JWT Tokens**: Used for user authentication in web applications
2. **API Keys**: Used for programmatic access with specific scopes

### API Key Scopes

- `trading`: Access to order placement and management
- `market_data`: Access to market data
- `notification`: Access to notification operations
- `events`: Access to event operations
- `broker_auth`: Access to broker authentication

## Usage Examples

### Authentication

```python
# Login and get JWT token
import requests

response = requests.post(
    "http://localhost:8000/auth/login",
    json={"username": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]

# Use JWT token for authenticated requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/auth/me", headers=headers)
```

### Using API Keys

```python
# Create an API key (requires JWT authentication)
response = requests.post(
    "http://localhost:8000/auth/api-keys",
    json={"name": "Trading Bot", "scopes": ["trading", "market_data"]},
    headers={"Authorization": f"Bearer {token}"}
)
api_key = response.json()["key"]

# Use API key for authenticated requests
headers = {"X-API-Key": api_key}
response = requests.post(
    "http://localhost:8000/broker/orders",
    json={
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": "Market",
        "side": "Buy",
        "time_in_force": "DAY"
    },
    headers=headers
)
```

### Streaming Events

```python
# Stream events using SSE (Server-Sent Events)
import sseclient
import requests

headers = {"X-API-Key": api_key}
response = requests.get(
    "http://localhost:8000/events/stream?event_types=order.filled,order.canceled",
    headers=headers,
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)
```

## Running the API Gateway

```bash
# Start the API server
python -m src.application.api.main
```

The API will be available at `http://localhost:8000` by default.

## Configuration

The API Gateway can be configured through environment variables or a configuration file. Key configuration options include:

- `API_HOST`: Host to bind the server (default: 0.0.0.0)
- `API_PORT`: Port to bind the server (default: 8000)
- `API_DEBUG`: Enable debug mode (default: False)
- `API_WORKERS`: Number of worker processes (default: 1)
- `JWT_SECRET_KEY`: Secret key for JWT token generation
- `JWT_ALGORITHM`: Algorithm for JWT token generation (default: HS256)
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiration time in minutes (default: 30)
- `RATE_LIMIT_ENABLED`: Enable rate limiting (default: True)
- `RATE_LIMIT_REQUESTS`: Maximum number of requests per window (default: 100)
- `RATE_LIMIT_WINDOW_SECONDS`: Rate limiting window in seconds (default: 60)

## Development

### Running Tests

```bash
# Run API Gateway tests
pytest tests/application/api/
```

### Adding New Endpoints

To add new endpoints, create a new router file or extend an existing one, then register it in `main.py`.

## Security Considerations

- API keys should be kept secure and not shared
- Use HTTPS in production environments
- Implement proper error handling to avoid information leakage
- Regularly rotate JWT secret keys and API keys