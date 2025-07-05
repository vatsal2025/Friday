# WebSocket Streaming Implementation

## Overview

This document provides an overview of the WebSocket streaming implementation for the Friday AI Trading System. The implementation enables real-time market data streaming via WebSocket connections, with support for high-frequency updates, throttling, and backpressure mechanisms.

## Architecture

The WebSocket streaming implementation consists of the following components:

1. **WebSocketAdapter**: Implements the `DataSourceAdapter` interface for WebSocket connections, handling connection management, authentication, and data processing.

2. **WebSocketDataStream**: Extends the `RealTimeDataStream` class to provide WebSocket-specific streaming functionality, including reconnection logic and event handling.

3. **StreamingMarketDataConnector**: Integrates the WebSocket data stream with the portfolio system, handling high-frequency updates, throttling, and backpressure.

4. **WebSocketStreamFactory**: Provides a factory for creating WebSocket adapters, streams, and connectors, simplifying the creation process.

## Component Details

### WebSocketAdapter

The `WebSocketAdapter` class implements the `DataSourceAdapter` interface for WebSocket connections. It handles:

- WebSocket connection establishment and management
- Authentication with the data source
- Subscription to symbols and timeframes
- Processing incoming messages (data, heartbeats, errors)
- Emitting system events for connection status and data updates

### WebSocketDataStream

The `WebSocketDataStream` class extends the `RealTimeDataStream` class to provide WebSocket-specific streaming functionality. It handles:

- Stream initialization and management
- Reconnection logic with exponential backoff
- Event handling for WebSocket connection, disconnection, error, and data events
- Data buffering and processing
- Emitting categorized data events (tick, trade, quote, bar, orderbook, custom)

### StreamingMarketDataConnector

The `StreamingMarketDataConnector` class integrates the WebSocket data stream with the portfolio system. It handles:

- High-frequency updates with throttling
- Backpressure mechanisms to prevent system overload
- Symbol prioritization for important market data
- Event handling for various stream events
- Processing and forwarding market data to the portfolio system

### WebSocketStreamFactory

The `WebSocketStreamFactory` class provides a factory for creating WebSocket adapters, streams, and connectors. It simplifies the creation process with methods for:

- Creating WebSocket adapters with specific configurations
- Creating WebSocket data streams with adapters
- Creating streaming market data connectors with streams
- Creating a complete WebSocket streaming stack in one step

## Configuration Options

The WebSocket streaming implementation supports various configuration options:

### Connection Configuration

- **WebSocket URL**: The endpoint URL for the WebSocket connection
- **Authentication Parameters**: API keys, tokens, or other authentication credentials
- **Reconnection Attempts**: Maximum number of reconnection attempts
- **Reconnection Delay**: Initial delay between reconnection attempts (with exponential backoff)

### Throttling and Backpressure

- **Throttle Interval**: Minimum time between updates for each symbol
- **Backpressure Threshold**: Maximum number of events per second before applying backpressure
- **Symbol Priorities**: Priority levels for different symbols (higher priority symbols bypass throttling)

### Data Buffering

- **Buffer Size**: Maximum number of data points to store for each symbol and timeframe
- **Heartbeat Interval**: Maximum time between heartbeats before considering the connection dead

## Usage Examples

### Basic Usage

```python
from src.data.acquisition.websocket_stream_factory import WebSocketStreamFactory
from src.data.acquisition.data_fetcher import DataTimeframe

# Create WebSocket stream factory
factory = WebSocketStreamFactory()

# Create complete WebSocket streaming stack
connector = factory.create_complete_stack(
    url="wss://example.com/marketdata/ws",
    auth_params={"api_key": "your_api_key_here"}
)

# Start the connector
connector.start()

# Subscribe to symbols and timeframes
connector.subscribe("AAPL", [DataTimeframe.ONE_MINUTE, DataTimeframe.FIVE_MINUTES])
connector.subscribe("MSFT", [DataTimeframe.ONE_MINUTE])

# Get latest data
latest_aapl_data = connector.get_latest_data("AAPL", DataTimeframe.ONE_MINUTE)

# Configure throttling and backpressure
connector.set_throttle_interval(0.1)  # 100ms throttle interval
connector.set_backpressure_threshold(100)  # Apply backpressure at 100 events/second
connector.set_symbol_priority("AAPL", 10)  # Higher priority for AAPL

# Stop the connector when done
connector.stop()
```

### Event Handling

```python
from src.infrastructure.event import EventSystem, Event

# Create event system
event_system = EventSystem()

# Register event handlers
event_system.subscribe("market_data.tick", handle_market_data_tick)
event_system.subscribe("market_data.bar", handle_market_data_bar)
event_system.subscribe("market_data.connected", handle_market_data_connected)
event_system.subscribe("market_data.disconnected", handle_market_data_disconnected)
event_system.subscribe("market_data.error", handle_market_data_error)

# Create factory with event system
factory = WebSocketStreamFactory(event_system=event_system)

# Create and start connector
connector = factory.create_complete_stack(...)
connector.start()
```

## Event Types

The WebSocket streaming implementation emits various event types:

### Stream Events

- `stream.tick`: Tick data event
- `stream.bar`: Bar data event
- `stream.trade`: Trade data event
- `stream.quote`: Quote data event
- `stream.orderbook`: Orderbook data event
- `stream.custom`: Custom data event
- `stream.event`: Stream status event (connected, disconnected, error)

### Market Data Events

- `market_data.tick`: Market tick data event
- `market_data.bar`: Market bar data event
- `market_data.trade`: Market trade data event
- `market_data.quote`: Market quote data event
- `market_data.orderbook`: Market orderbook data event
- `market_data.custom`: Market custom data event
- `market_data.connected`: Market data connected event
- `market_data.disconnected`: Market data disconnected event
- `market_data.error`: Market data error event

### WebSocket Events

- `websocket.connected`: WebSocket connected event
- `websocket.disconnected`: WebSocket disconnected event
- `websocket.error`: WebSocket error event
- `websocket.data`: WebSocket data event

## Testing

The WebSocket streaming implementation includes comprehensive unit tests for all components:

- `test_websocket_adapter.py`: Tests for the WebSocketAdapter class
- `test_websocket_data_stream.py`: Tests for the WebSocketDataStream class
- `test_streaming_market_data_connector.py`: Tests for the StreamingMarketDataConnector class
- `test_websocket_stream_factory.py`: Tests for the WebSocketStreamFactory class

## Implementation Phases

The WebSocket streaming implementation follows a phased approach:

1. **Phase 1: Core Streaming Infrastructure**
   - WebSocketAdapter implementation
   - WebSocketDataStream implementation
   - Basic event handling and data processing

2. **Phase 2: High-Frequency Update Handling**
   - StreamingMarketDataConnector implementation
   - Throttling mechanisms
   - Symbol prioritization

3. **Phase 3: Backpressure and Throttling**
   - Backpressure mechanisms
   - Advanced throttling options
   - Performance optimizations

4. **Phase 4: Advanced Features and Multiple Sources**
   - WebSocketStreamFactory implementation
   - Support for multiple data sources
   - Advanced configuration options

## Future Enhancements

Potential future enhancements for the WebSocket streaming implementation:

- Support for binary WebSocket protocols (protobuf, msgpack)
- Integration with additional market data providers
- Advanced data normalization and transformation
- Machine learning-based anomaly detection for market data
- Adaptive throttling based on system load and market conditions
- Enhanced monitoring and metrics for stream performance