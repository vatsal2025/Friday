# Friday AI Trading System - Event System

## Overview

The Event System is a core infrastructure component of the Friday AI Trading System, providing a robust, scalable, and flexible mechanism for inter-component communication and event handling. It implements an event-driven architecture that allows different parts of the system to communicate asynchronously through events.

## Key Features

- **Asynchronous Communication**: Components can communicate without direct dependencies
- **Decoupled Architecture**: Publishers and subscribers are decoupled, enhancing modularity
- **Event Persistence**: Optional storage of events for auditing, replay, and analysis
- **Flexible Filtering**: Events can be filtered by type and custom criteria
- **Production-Ready Monitoring**: Built-in monitoring, health checks, and visualization
- **Thread-Safe**: Designed for concurrent access in a multi-threaded environment
- **Scalable**: Can handle high event throughput with configurable queue sizes

## Core Components

### Event

The fundamental data structure representing an occurrence in the system.

```python
from src.infrastructure.event import Event

# Create a new event
event = Event(
    event_type="trade_signal",
    data={"symbol": "AAPL", "action": "BUY", "price": 150.25},
    source="trading_strategy_a"
)

# Access event properties
print(event.event_type)  # "trade_signal"
print(event.data)        # {"symbol": "AAPL", "action": "BUY", "price": 150.25}
print(event.timestamp)   # ISO-format timestamp when the event was created
print(event.source)      # "trading_strategy_a"
print(event.event_id)    # Unique ID for the event

# Convert to dictionary or JSON
event_dict = event.to_dict()
event_json = event.to_json()

# Create from dictionary or JSON
event_from_dict = Event.from_dict(event_dict)
event_from_json = Event.from_json(event_json)
```

### EventHandler

Handles specific types of events with custom logic.

```python
from src.infrastructure.event import EventHandler

# Create a handler for specific event types
def handle_trade_signal(event):
    print(f"Received trade signal: {event.data}")

handler = EventHandler(
    callback=handle_trade_signal,
    event_types=["trade_signal", "market_data"],
    filter_func=lambda event: event.data.get("symbol") == "AAPL"
)

# Check if the handler should process an event
should_handle = handler.should_handle(event)

# Process an event
if should_handle:
    handler.handle(event)
```

### EventQueue

Threadsafe queue for storing and retrieving events.

```python
from src.infrastructure.event import EventQueue

# Create a queue with a maximum size
queue = EventQueue(max_size=1000)

# Add an event to the queue
queue.put(event)

# Get an event from the queue (blocks until an event is available)
event = queue.get()

# Get an event with a timeout (returns None if no event is available)
event = queue.get(timeout=1.0)

# Check the current size of the queue
size = queue.size()
```

### EventBus

Central hub for publishing events and dispatching them to handlers.

```python
from src.infrastructure.event import EventBus

# Create an event bus
bus = EventBus(max_queue_size=1000)

# Register a handler
bus.register_handler(handler)

# Unregister a handler
bus.unregister_handler(handler)

# Publish an event
bus.publish(event)

# Start and stop the event bus
bus.start()
bus.stop()
```

### EventStore

Stores events for later retrieval, analysis, or replay.

```python
from src.infrastructure.event import EventStore

# Create an event store with a maximum capacity
store = EventStore(max_events=10000)

# Store an event
store.store_event(event)

# Get all events
all_events = store.get_events()

# Get events of a specific type
trade_events = store.get_events(event_type="trade_signal")

# Get events matching a filter
apple_events = store.get_events(
    filter_func=lambda e: e.data.get("symbol") == "AAPL"
)

# Clear all events
store.clear_events()

# Start and stop the event store
store.start()
store.stop()
```

### EventSystem

Main entry point for using the event system, combining the EventBus and EventStore.

```python
from src.infrastructure.event import EventSystem

# Create an event system
event_system = EventSystem(
    max_queue_size=1000,
    max_events=10000,  # Set to None to disable event storage
    persist_events=False  # Set to True to enable event persistence
)

# Start and stop the event system
event_system.start()
event_system.stop()

# Emit an event
event_system.emit(event)

# Register a handler
def handle_event(event):
    print(f"Handling event: {event.event_type}")

event_system.register_handler(
    callback=handle_event,
    event_types=["trade_signal"],
    filter_func=lambda event: event.data.get("priority") == "high"
)

# Unregister a handler
event_system.unregister_handler(handler)
```

## Monitoring and Health Checks

The event system includes built-in monitoring and health check capabilities for production use.

### EventMonitor

Monitors event metrics such as counts, rates, and sizes.

```python
from src.infrastructure.event import EventMonitor

# Create an event monitor
monitor = EventMonitor(event_system, sampling_interval=1.0)

# Start and stop the monitor
monitor.start()
monitor.stop()

# Get metrics
event_counts = monitor.get_event_counts()
event_rates = monitor.get_event_rates(window_seconds=60)  # 1-minute window
event_sizes = monitor.get_event_sizes()

# Get a summary of all metrics
summary = monitor.get_summary()

# Reset metrics
monitor.reset_metrics()
```

### EventHealthCheck

Checks the health of the event system and detects issues.

```python
from src.infrastructure.event import EventHealthCheck

# Create a health check with custom thresholds
health_check = EventHealthCheck(
    event_system,
    event_monitor,
    check_interval=60.0,
    thresholds={
        "queue_size_warning": 100,
        "queue_size_critical": 500,
        "event_rate_warning": 1000,  # events/second
        "event_rate_critical": 5000,  # events/second
    }
)

# Start and stop the health check
health_check.start()
health_check.stop()

# Get health status
status = health_check.get_health_status()

# Check if the system is healthy
is_healthy = health_check.is_healthy()
```

### EventDashboard

Generates reports and visualizations of event system metrics.

```python
from src.infrastructure.event import EventDashboard

# Create a dashboard
dashboard = EventDashboard(event_monitor, event_health_check)

# Generate reports
text_report = dashboard.generate_text_report()
json_report = dashboard.generate_json_report()
```

### Setup Helper

Convenience function to set up monitoring components.

```python
from src.infrastructure.event import setup_event_monitoring

# Set up monitoring for an event system
monitor, health_check, dashboard = setup_event_monitoring(event_system)
```

## Common Event Types

The Friday AI Trading System uses the following common event types:

| Event Type | Description | Example Data |
|------------|-------------|-------------|
| `market_data` | Market data updates | `{"symbol": "AAPL", "price": 150.25, "timestamp": "2023-01-01T12:00:00"}` |
| `trade_signal` | Trading signals | `{"symbol": "AAPL", "action": "BUY", "price": 150.25, "confidence": 0.85}` |
| `order` | Order information | `{"order_id": "123", "symbol": "AAPL", "type": "MARKET", "side": "BUY", "quantity": 100}` |
| `execution` | Order execution | `{"order_id": "123", "execution_id": "456", "price": 150.25, "quantity": 100}` |
| `model_prediction` | ML model predictions | `{"model": "price_predictor", "symbol": "AAPL", "prediction": 155.50, "confidence": 0.75}` |
| `system_status` | System status updates | `{"component": "data_service", "status": "OK", "message": "Service started"}` |
| `error` | Error notifications | `{"component": "order_service", "error_code": "E1001", "message": "Failed to place order"}` |

## Best Practices

1. **Define Clear Event Types**: Use consistent naming conventions for event types
2. **Structured Event Data**: Keep event data well-structured and documented
3. **Targeted Handlers**: Create handlers that focus on specific event types or conditions
4. **Error Handling**: Always include error handling in event handlers
5. **Monitoring**: Use the built-in monitoring tools in production
6. **Performance Tuning**: Adjust queue sizes based on expected event volumes
7. **Testing**: Write unit tests for event handlers and event flows

## Example: Complete Event Flow

```python
from src.infrastructure.event import Event, EventSystem, setup_event_monitoring

# Create and start the event system
event_system = EventSystem(max_queue_size=1000, max_events=10000)
event_system.start()

# Set up monitoring
monitor, health_check, dashboard = setup_event_monitoring(event_system)

# Register handlers
def handle_trade_signal(event):
    print(f"Processing trade signal: {event.data}")
    # Process the trade signal...
    
    # Emit a new event as a result
    order_event = Event(
        event_type="order",
        data={
            "symbol": event.data["symbol"],
            "side": event.data["action"],
            "quantity": 100,
            "type": "MARKET"
        },
        source="order_manager"
    )
    event_system.emit(order_event)

def handle_order(event):
    print(f"Processing order: {event.data}")
    # Process the order...

# Register handlers with the event system
event_system.register_handler(
    callback=handle_trade_signal,
    event_types=["trade_signal"]
)

event_system.register_handler(
    callback=handle_order,
    event_types=["order"]
)

# Emit an event to start the flow
trade_signal = Event(
    event_type="trade_signal",
    data={"symbol": "AAPL", "action": "BUY", "price": 150.25},
    source="trading_strategy"
)

event_system.emit(trade_signal)

# Later, stop the event system
event_system.stop()
monitor.stop()
health_check.stop()
```

## Integration with Other Components

The event system integrates with other components of the Friday AI Trading System:

- **Logging**: Events can be logged using the logging infrastructure
- **Security**: Sensitive event data can be encrypted using the security infrastructure
- **Configuration**: Event system settings can be managed through the unified configuration
- **Database**: Events can be persisted to a database for long-term storage

## Troubleshooting

### Common Issues

1. **Events not being processed**: Check if the event system is started and handlers are registered
2. **High event queue backlog**: Increase queue size or add more processing capacity
3. **Memory usage growing**: Check for event leaks or reduce max_events in EventStore
4. **Slow event processing**: Optimize handler code or increase thread count

### Debugging

Use the monitoring tools to diagnose issues:

```python
# Get current metrics
metrics = monitor.get_summary()
print(f"Event counts: {metrics['counts']}")
print(f"Event rates: {metrics['rates']['1min']}")

# Check health status
health = health_check.get_health_status()
print(f"Health status: {health['status']}")
print(f"Issues: {health['issues']}")

# Generate a report
report = dashboard.generate_text_report()
print(report)
```

## Conclusion

The Event System provides a robust foundation for building event-driven applications within the Friday AI Trading System. By following the guidelines and examples in this documentation, developers can effectively leverage the event system to create decoupled, scalable, and maintainable components.