# Event System Deployment Guide

This guide provides instructions and best practices for deploying the Friday AI Trading System's event system in production environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Scaling Strategies](#scaling-strategies)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Performance Tuning](#performance-tuning)
7. [Disaster Recovery](#disaster-recovery)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Recommendations

- **CPU**: 4+ cores for moderate event throughput (up to 1,000 events/second)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for high-throughput systems
- **Disk**: SSD storage recommended, especially for event persistence
- **Network**: Low-latency network connections between system components

### Software Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux (recommended), Windows, or macOS
- **Dependencies**: All packages listed in requirements.txt

## Installation

1. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Configure the event system settings in your application configuration file.

3. Initialize the event system early in your application startup sequence.

## Configuration

### Basic Configuration

The event system can be configured through the `EventSystem` constructor:

```python
from src.infrastructure.event import EventSystem

event_system = EventSystem(
    max_queue_size=10000,  # Maximum events in queue before blocking
    max_events=100000,     # Maximum events to store in EventStore
    persistence_path="/path/to/event/storage",  # Optional: for event persistence
    persistence_enabled=True,  # Enable/disable event persistence
    persistence_interval=60,  # Seconds between persistence operations
)
```

### Production Configuration Recommendations

```python
from src.infrastructure.event import EventSystem, setup_event_monitoring
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger("event_system")

# Create event system with production settings
event_system = EventSystem(
    max_queue_size=50000,  # Higher queue size for production
    max_events=1000000,    # Store more events in production
    persistence_path="/var/friday/events",  # Use dedicated storage location
    persistence_enabled=True,
    persistence_interval=300,  # Less frequent persistence to reduce I/O
)

# Set up monitoring
monitor, health_check, dashboard = setup_event_monitoring(
    event_system,
    check_interval=60,  # Check health every minute
    alert_threshold=0.8,  # Alert when queue is 80% full
)

# Start the event system
event_system.start()
```

## Scaling Strategies

### Vertical Scaling

The event system can be scaled vertically by increasing resources on a single machine:

- Increase `max_queue_size` for higher throughput
- Allocate more memory to the application
- Use machines with more CPU cores for parallel event processing

### Horizontal Scaling

For very high throughput requirements, consider these approaches:

1. **Multiple Event Systems**: Partition events by type or source across multiple event system instances
2. **External Message Broker**: For extreme scale, integrate with external message brokers like RabbitMQ, Kafka, or Redis

```python
# Example: Integrating with external message broker (Redis)
from src.infrastructure.event import Event, EventSystem
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Custom handler to forward events to Redis
def redis_forwarder(event):
    redis_client.publish(f"events:{event.event_type}", event.to_json())

# Register the forwarder with the event system
event_system.register_handler(
    callback=redis_forwarder,
    event_types=None  # Handle all event types
)
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Event Queue Size**: Monitor for potential backpressure
2. **Event Processing Rate**: Events processed per second
3. **Event Latency**: Time from emission to processing
4. **Error Rates**: Failed event processing attempts
5. **Resource Usage**: CPU, memory, and disk I/O

### Setting Up Alerts

Configure alerts for these conditions:

- Queue size exceeds 80% of maximum capacity
- Event processing latency exceeds acceptable thresholds
- Error rate exceeds normal baseline
- Event system stops processing events

### Integration with Monitoring Systems

The `EventMonitor` and `EventHealthCheck` classes can be integrated with external monitoring systems:

```python
from src.infrastructure.event import EventSystem, setup_event_monitoring
import time
import requests

event_system = EventSystem()
monitor, health_check, dashboard = setup_event_monitoring(event_system)

# Example: Periodically send metrics to a monitoring service
def report_metrics():
    while True:
        metrics = monitor.get_summary()
        health = health_check.get_health_status()
        
        # Send to monitoring service
        requests.post(
            "https://monitoring-service.example.com/metrics",
            json={
                "event_system": {
                    "metrics": metrics,
                    "health": health
                }
            }
        )
        
        time.sleep(60)  # Report every minute

# Start reporting in a separate thread
import threading
reporting_thread = threading.Thread(target=report_metrics, daemon=True)
reporting_thread.start()
```

## Performance Tuning

### Optimizing Event Handlers

1. **Keep Handlers Lightweight**: Event handlers should complete quickly
2. **Offload Heavy Processing**: Use separate threads or processes for CPU-intensive operations
3. **Batch Processing**: Consider batching events for efficiency when appropriate

```python
# Example: Offloading heavy processing
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for heavy processing
processor_pool = ThreadPoolExecutor(max_workers=4)

def heavy_processing_handler(event):
    # Submit the heavy work to the thread pool
    processor_pool.submit(do_heavy_processing, event)
    
    # Return immediately, allowing the event bus to process more events
    return True

def do_heavy_processing(event):
    # Perform CPU-intensive work here
    # ...
    pass

# Register the handler
event_system.register_handler(
    callback=heavy_processing_handler,
    event_types=["data_processed", "model_prediction"]
)
```

### Tuning Event Persistence

1. **Adjust Persistence Interval**: Increase for higher throughput, decrease for better durability
2. **Selective Persistence**: Only persist critical events
3. **Storage Optimization**: Use fast storage for event persistence

### Running Performance Tests

Use the included performance testing tools to benchmark your configuration:

```python
from src.infrastructure.event import EventSystemPerformanceTest

# Create and run performance tests
test = EventSystemPerformanceTest(
    event_system=your_event_system,  # Optional: use your existing event system
    max_queue_size=10000,
    max_events=100000
)

# Run all tests
results = test.run_all_tests()

# Or run specific tests
throughput_results = test.run_throughput_test(
    num_events=100000,
    batch_size=100,
    event_size=1000,
    num_threads=4
)

# Save results
test.save_results("performance_test_results.json")
```

## Disaster Recovery

### Event Persistence

Enable event persistence for critical events to ensure they can be recovered after a system failure:

```python
event_system = EventSystem(
    persistence_enabled=True,
    persistence_path="/var/friday/events",
    persistence_interval=60  # Save every minute
)
```

### Backup Strategies

1. **Regular Backups**: Schedule regular backups of the event persistence directory
2. **Replication**: Consider replicating critical events to multiple storage locations
3. **Event Replay**: Implement mechanisms to replay events from persistence storage after recovery

### Recovery Procedures

1. **System Restart**: Procedures for clean restart of the event system
2. **Event Replay**: How to replay missed events after an outage
3. **Consistency Checks**: Verify system state after recovery

## Security Considerations

### Event Data Protection

1. **Sensitive Data**: Avoid storing sensitive information in events, or ensure it's encrypted
2. **Access Control**: Restrict access to event persistence storage
3. **Encryption**: Consider encrypting persisted events

```python
from src.infrastructure.security import encrypt_data, decrypt_data

# Example: Encrypting sensitive event data
def secure_event_handler(event):
    if "sensitive_data" in event.data:
        # Encrypt sensitive data before processing
        encryption_key = get_encryption_key()  # Your key management function
        event.data["sensitive_data"] = encrypt_data(
            event.data["sensitive_data"],
            encryption_key
        )
    
    # Process the event
    process_event(event)

# Register the handler
event_system.register_handler(
    callback=secure_event_handler,
    event_types=["user_data", "account_info"]
)
```

### Audit Logging

Implement audit logging for security-relevant events:

```python
from src.infrastructure.logging import get_logger

# Create a dedicated audit logger
audit_logger = get_logger("security.audit")

def audit_event_handler(event):
    if event.event_type in ["user_login", "permission_change", "api_access"]:
        audit_logger.info(
            f"Security event: {event.event_type} from {event.source} "
            f"at {event.timestamp} with ID {event.event_id}"
        )

# Register the audit handler
event_system.register_handler(
    callback=audit_event_handler,
    event_types=["user_login", "permission_change", "api_access"]
)
```

## Troubleshooting

### Common Issues

1. **Queue Overflow**: Events being rejected due to full queue
   - Solution: Increase `max_queue_size` or optimize event processing

2. **High Latency**: Slow event processing
   - Solution: Profile and optimize event handlers, consider offloading heavy processing

3. **Memory Usage**: Excessive memory consumption
   - Solution: Limit `max_events` in EventStore, implement event expiration

4. **Lost Events**: Events not being processed
   - Solution: Enable persistence, check handler registration, verify event types

### Diagnostic Tools

1. **Event Dashboard**: Use the `EventDashboard` for real-time system status

```python
from src.infrastructure.event import setup_event_monitoring

# Set up monitoring
monitor, health_check, dashboard = setup_event_monitoring(event_system)

# Get a text report
status_report = dashboard.generate_text_report()
print(status_report)

# Get a JSON report for programmatic use
status_json = dashboard.generate_json_report()
```

2. **Health Checks**: Use `EventHealthCheck` to diagnose issues

```python
health_status = health_check.get_health_status()
if health_status["status"] != "healthy":
    print(f"Event system issues: {health_status['issues']}")
```

3. **Performance Testing**: Use `EventSystemPerformanceTest` to identify bottlenecks

```python
from src.infrastructure.event import EventSystemPerformanceTest

test = EventSystemPerformanceTest(event_system=event_system)
latency_results = test.run_latency_test(num_events=1000)
print(f"Average latency: {latency_results['results']['latency']['avg']} ms")
```

### Logging

Enable detailed logging for troubleshooting:

```python
import logging
from src.infrastructure.logging import setup_logging

# Set up detailed logging for the event system
setup_logging(log_level=logging.DEBUG, log_file="/var/log/friday/event_system.log")
```

---

## Conclusion

A properly deployed and configured event system is critical to the reliability and performance of the Friday AI Trading System. Follow these guidelines to ensure your event system operates efficiently in production environments.

For additional support or questions, refer to the project documentation or contact the development team.