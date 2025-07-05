# Event System API Documentation

## Overview

The Friday AI Trading System Event System provides a robust, production-ready event infrastructure for building event-driven applications. This document provides comprehensive API documentation for developers integrating with the event system.

## Core Components

### Event

```python
class Event:
    def __init__(self, event_type: str, data: Any, source: str = None, event_id: str = None, timestamp: float = None)
```

Represents a single event in the system.

**Parameters:**
- `event_type` (str): The type of the event (e.g., "market_data", "trade_signal")
- `data` (Any): The payload of the event
- `source` (str, optional): The source of the event
- `event_id` (str, optional): Unique identifier for the event (auto-generated if not provided)
- `timestamp` (float, optional): Event creation timestamp (auto-generated if not provided)

**Methods:**
- `to_dict() -> Dict`: Convert the event to a dictionary
- `to_json() -> str`: Convert the event to a JSON string
- `from_dict(cls, data: Dict) -> Event`: Create an event from a dictionary
- `from_json(cls, json_str: str) -> Event`: Create an event from a JSON string

### EventHandler

```python
class EventHandler:
    def __init__(self, callback: callable, event_types: List[str] = None, filter_func: callable = None)
```

Handles events of specified types.

**Parameters:**
- `callback` (callable): Function to call when an event is handled
- `event_types` (List[str], optional): List of event types this handler processes
- `filter_func` (callable, optional): Additional filtering function

**Methods:**
- `handles_event_type(event_type: str) -> bool`: Check if this handler processes a specific event type
- `should_handle(event: Event) -> bool`: Determine if this handler should process an event
- `handle(event: Event) -> bool`: Process an event

### EventQueue

```python
class EventQueue:
    def __init__(self, max_size: int = 1000)
```

Thread-safe queue for storing events.

**Parameters:**
- `max_size` (int, optional): Maximum number of events in the queue

**Methods:**
- `put(event: Event) -> bool`: Add an event to the queue
- `get() -> Optional[Event]`: Get the next event from the queue
- `size() -> int`: Get the current size of the queue
- `is_full() -> bool`: Check if the queue is full
- `is_empty() -> bool`: Check if the queue is empty
- `clear() -> None`: Clear all events from the queue

### EventBus

```python
class EventBus:
    def __init__(self, max_queue_size: int = 1000)
```

Central event bus for publishing and handling events.

**Parameters:**
- `max_queue_size` (int, optional): Maximum size of the event queue

**Methods:**
- `register_handler(callback: callable, event_types: List[str] = None, filter_func: callable = None) -> EventHandler`: Register a new event handler
- `unregister_handler(handler: EventHandler) -> bool`: Unregister an event handler
- `publish(event: Event) -> bool`: Publish an event to the bus
- `start() -> None`: Start the event bus
- `stop() -> None`: Stop the event bus
- `is_running() -> bool`: Check if the event bus is running

### EventStore

```python
class EventStore:
    def __init__(self, max_events: int = 10000, persistence_path: str = None, persistence_enabled: bool = False, persistence_interval: int = 60)
```

Stores events for persistence and replay.

**Parameters:**
- `max_events` (int, optional): Maximum number of events to store in memory
- `persistence_path` (str, optional): Path to store events on disk
- `persistence_enabled` (bool, optional): Whether to persist events to disk
- `persistence_interval` (int, optional): How often to persist events (in seconds)

**Methods:**
- `store(event: Event) -> bool`: Store an event
- `persist() -> bool`: Persist events to disk
- `get_events(event_type: str = None, start_time: float = None, end_time: float = None, max_count: int = None) -> List[Event]`: Get stored events
- `clear() -> None`: Clear all stored events
- `start() -> None`: Start the event store
- `stop() -> None`: Stop the event store

### EventSystem

```python
class EventSystem:
    def __init__(self, max_queue_size: int = 1000, max_events: int = 10000, persistence_path: str = None, persistence_enabled: bool = False, persistence_interval: int = 60)
```

Main entry point for the event system.

**Parameters:**
- `max_queue_size` (int, optional): Maximum size of the event queue
- `max_events` (int, optional): Maximum number of events to store
- `persistence_path` (str, optional): Path to store events on disk
- `persistence_enabled` (bool, optional): Whether to persist events to disk
- `persistence_interval` (int, optional): How often to persist events (in seconds)

**Methods:**
- `start() -> None`: Start the event system
- `stop() -> None`: Stop the event system
- `is_running() -> bool`: Check if the event system is running
- `emit(event: Event) -> bool`: Emit an event to the system
- `register_handler(callback: callable, event_types: List[str] = None, filter_func: callable = None) -> EventHandler`: Register a new event handler
- `unregister_handler(handler: EventHandler) -> bool`: Unregister an event handler
- `get_events(event_type: str = None, start_time: float = None, end_time: float = None, max_count: int = None) -> List[Event]`: Get stored events

## Monitoring Components

### EventMonitor

```python
class EventMonitor:
    def __init__(self, event_system: EventSystem)
```

Monitors event system metrics.

**Parameters:**
- `event_system` (EventSystem): The event system to monitor

**Methods:**
- `start() -> None`: Start monitoring
- `stop() -> None`: Stop monitoring
- `get_event_count(event_type: str = None) -> int`: Get the count of events
- `get_event_rate(event_type: str = None, window_seconds: int = 60) -> float`: Get the rate of events
- `get_event_size_stats(event_type: str = None) -> Dict[str, float]`: Get statistics about event sizes
- `get_summary() -> Dict[str, Any]`: Get a summary of all metrics

### EventHealthCheck

```python
class EventHealthCheck:
    def __init__(self, event_system: EventSystem, event_monitor: EventMonitor, check_interval: int = 60, alert_threshold: int = 80)
```

Checks the health of the event system.

**Parameters:**
- `event_system` (EventSystem): The event system to check
- `event_monitor` (EventMonitor): The event monitor to use
- `check_interval` (int, optional): How often to check health (in seconds)
- `alert_threshold` (int, optional): Threshold for alerts (percentage)

**Methods:**
- `start() -> None`: Start health checks
- `stop() -> None`: Stop health checks
- `get_health_status() -> Dict[str, Any]`: Get the current health status
- `check_queue_health() -> Dict[str, Any]`: Check the health of the event queue
- `check_event_rate_health() -> Dict[str, Any]`: Check the health of event rates

### EventDashboard

```python
class EventDashboard:
    def __init__(self, event_system: EventSystem, event_monitor: EventMonitor, health_check: EventHealthCheck)
```

Provides a dashboard for event system metrics.

**Parameters:**
- `event_system` (EventSystem): The event system to monitor
- `event_monitor` (EventMonitor): The event monitor to use
- `health_check` (EventHealthCheck): The health check to use

**Methods:**
- `generate_text_report() -> str`: Generate a text report
- `generate_json_report() -> Dict[str, Any]`: Generate a JSON report

### setup_event_monitoring

```python
def setup_event_monitoring(event_system: EventSystem, check_interval: int = 60, alert_threshold: int = 80) -> Tuple[EventMonitor, EventHealthCheck, EventDashboard]
```

Set up monitoring for an event system.

**Parameters:**
- `event_system` (EventSystem): The event system to monitor
- `check_interval` (int, optional): How often to check health (in seconds)
- `alert_threshold` (int, optional): Threshold for alerts (percentage)

**Returns:**
- `Tuple[EventMonitor, EventHealthCheck, EventDashboard]`: The monitoring components

## Configuration Components

### EventQueueConfig

```python
class EventQueueConfig:
    def __init__(self, max_size: int = 1000)
```

Configuration for the event queue.

**Parameters:**
- `max_size` (int, optional): Maximum size of the event queue

### EventStoreConfig

```python
class EventStoreConfig:
    def __init__(self, enabled: bool = False, max_events: int = 10000, persistence_path: str = None, persistence_interval: int = 60)
```

Configuration for the event store.

**Parameters:**
- `enabled` (bool, optional): Whether to enable the event store
- `max_events` (int, optional): Maximum number of events to store
- `persistence_path` (str, optional): Path to store events on disk
- `persistence_interval` (int, optional): How often to persist events (in seconds)

### EventMonitoringConfig

```python
class EventMonitoringConfig:
    def __init__(self, enabled: bool = True, check_interval: int = 60, alert_threshold: int = 80)
```

Configuration for event monitoring.

**Parameters:**
- `enabled` (bool, optional): Whether to enable monitoring
- `check_interval` (int, optional): How often to check health (in seconds)
- `alert_threshold` (int, optional): Threshold for alerts (percentage)

### EventHandlerConfig

```python
class EventHandlerConfig:
    def __init__(self, max_retry_count: int = 3, retry_delay: int = 5)
```

Configuration for event handlers.

**Parameters:**
- `max_retry_count` (int, optional): Maximum number of retries for failed handlers
- `retry_delay` (int, optional): Delay between retries (in seconds)

### EventSystemConfig

```python
class EventSystemConfig:
    def __init__(self, queue: EventQueueConfig = None, store: EventStoreConfig = None, monitoring: EventMonitoringConfig = None, handler: EventHandlerConfig = None)
```

Configuration for the event system.

**Parameters:**
- `queue` (EventQueueConfig, optional): Configuration for the event queue
- `store` (EventStoreConfig, optional): Configuration for the event store
- `monitoring` (EventMonitoringConfig, optional): Configuration for monitoring
- `handler` (EventHandlerConfig, optional): Configuration for event handlers

### Configuration Utility Functions

```python
def get_development_config() -> EventSystemConfig
def get_testing_config() -> EventSystemConfig
def get_production_config() -> EventSystemConfig
def load_config_from_file(file_path: str) -> EventSystemConfig
def save_config_to_file(config: EventSystemConfig, file_path: str) -> bool
def get_config_from_env() -> EventSystemConfig
```

Utility functions for working with configurations.

## Integration Components

### EventSystemIntegration

```python
class EventSystemIntegration:
    def __init__(self, config: Optional[EventSystemConfig] = None, auto_start: bool = False)
```

Integration class for the event system.

**Parameters:**
- `config` (EventSystemConfig, optional): Configuration for the event system
- `auto_start` (bool, optional): Whether to automatically start the event system

**Methods:**
- `start() -> None`: Start the event system and monitoring
- `stop() -> None`: Stop the event system and monitoring
- `register_component(component_id: str, component_type: str, event_types: List[str], description: str = "", metadata: Optional[Dict[str, Any]] = None) -> None`: Register a component
- `unregister_component(component_id: str) -> bool`: Unregister a component
- `get_component(component_id: str) -> Optional[Dict[str, Any]]`: Get information about a component
- `get_components_by_type(component_type: str) -> List[Dict[str, Any]]`: Get components by type
- `get_components_by_event_type(event_type: str) -> List[Dict[str, Any]]`: Get components by event type
- `get_system_status() -> Dict[str, Any]`: Get the current system status
- `save_system_status(file_path: str) -> None`: Save the system status to a file
- `integrate_with_external_system(system_name: str, event_types: List[str], forward_callback: callable, receive_callback: Optional[callable] = None) -> Tuple[callable, Optional[callable]]`: Integrate with an external system

## Common Event Types

The Friday AI Trading System uses the following common event types:

| Event Type | Description | Example Data |
|------------|-------------|---------------|
| `market_data` | Market data updates | `{"symbol": "AAPL", "price": 150.0, "volume": 1000}` |
| `market_status` | Market status changes | `{"market": "NYSE", "status": "open", "timestamp": 1625097600}` |
| `trade_signal` | Trading signals | `{"symbol": "AAPL", "action": "BUY", "confidence": 0.85}` |
| `order` | Order information | `{"order_id": "ord123", "symbol": "AAPL", "side": "BUY", "quantity": 10, "price": 150.0}` |
| `execution` | Order execution | `{"order_id": "ord123", "execution_id": "exec456", "symbol": "AAPL", "quantity": 10, "price": 150.0}` |
| `model_prediction` | Model predictions | `{"model_id": "model123", "symbol": "AAPL", "prediction": "BUY", "confidence": 0.85}` |
| `model_status` | Model status changes | `{"model_id": "model123", "status": "training", "progress": 0.75}` |
| `system_health` | System health status | `{"status": "healthy", "metrics": {...}}` |
| `error` | Error events | `{"error_type": "connection_error", "message": "Failed to connect to API", "source": "market_data_service"}` |
| `component_registered` | Component registration | `{"component_id": "market_data_service", "component_type": "data_source"}` |
| `component_unregistered` | Component unregistration | `{"component_id": "market_data_service", "component_type": "data_source"}` |

## Best Practices

1. **Event Design**:
   - Keep events small and focused
   - Include all necessary context in the event data
   - Use consistent naming conventions for event types

2. **Error Handling**:
   - Always handle exceptions in event handlers
   - Use the error event type for reporting errors
   - Consider retry logic for transient failures

3. **Performance**:
   - Monitor event queue size and processing rates
   - Use filtered handlers to reduce unnecessary processing
   - Consider batching for high-volume events

4. **Integration**:
   - Use the `EventSystemIntegration` class for integrating with other components
   - Register all components that emit or handle events
   - Use the monitoring tools to track system health

## Example Usage

### Basic Usage

```python
from src.infrastructure.event import Event, EventSystem

# Create an event system
event_system = EventSystem()
event_system.start()

# Define a handler
def handle_market_data(event):
    print(f"Received market data: {event.data}")
    return True

# Register the handler
event_system.register_handler(
    callback=handle_market_data,
    event_types=["market_data"]
)

# Emit an event
event_system.emit(Event(
    event_type="market_data",
    data={"symbol": "AAPL", "price": 150.0, "volume": 1000},
    source="example"
))

# Clean up
event_system.stop()
```

### Using the Integration Class

```python
from src.infrastructure.event import Event, EventSystemIntegration, get_production_config

# Create the integration with production configuration
config = get_production_config()
integration = EventSystemIntegration(config=config, auto_start=True)

# Register a component
integration.register_component(
    component_id="market_data_service",
    component_type="data_source",
    event_types=["market_data", "market_status"],
    description="Service providing real-time market data"
)

# Define a handler
def handle_market_data(event):
    print(f"Received market data: {event.data}")
    return True

# Register the handler
integration.event_system.register_handler(
    callback=handle_market_data,
    event_types=["market_data"]
)

# Emit an event
integration.event_system.emit(Event(
    event_type="market_data",
    data={"symbol": "AAPL", "price": 150.0, "volume": 1000},
    source="example"
))

# Get system status
status = integration.get_system_status()
print(f"System status: {status['health']['status']}")

# Clean up
integration.stop()
```

## Troubleshooting

### Common Issues

1. **Event handlers not being called**:
   - Check that the event type matches exactly
   - Verify that the handler is registered correctly
   - Check for exceptions in the handler

2. **High memory usage**:
   - Reduce the `max_events` parameter in the event store
   - Check for memory leaks in event handlers
   - Consider disabling persistence if not needed

3. **Slow event processing**:
   - Check for blocking operations in event handlers
   - Monitor event queue size and processing rates
   - Consider increasing the queue size or adding more workers

### Debugging

The event system integrates with the logging system to provide detailed logs:

```python
from src.infrastructure.logging import get_logger

logger = get_logger("event_system")
```

You can also use the monitoring tools to get detailed metrics:

```python
from src.infrastructure.event import setup_event_monitoring

monitor, health_check, dashboard = setup_event_monitoring(event_system)
print(dashboard.generate_text_report())
```

## Support

For additional support, please contact the infrastructure team or refer to the deployment guide for more information.