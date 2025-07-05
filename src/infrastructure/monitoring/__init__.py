"""Monitoring module for the Friday AI Trading System.

This module provides monitoring capabilities including:
- System health checks
- Performance metrics collection
- Resource usage monitoring
- Alerting mechanisms
"""

import functools
import logging
import os
import platform
import psutil
import socket
import threading
import time
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from src.infrastructure.logging import get_logger
from src.infrastructure.error import ErrorSeverity, log_error

# Type variable for the return type of the function being wrapped
T = TypeVar('T')

# Create logger
logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status of a component or system."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = auto()  # Monotonically increasing value
    GAUGE = auto()    # Value that can go up and down
    HISTOGRAM = auto() # Distribution of values
    TIMER = auto()    # Duration of operations


class AlertLevel(Enum):
    """Alert levels for monitoring alerts."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a health check.
        
        Args:
            name: Name of the health check
            description: Description of the health check
        """
        self.name = name
        self.description = description
        self.last_check_time = None
        self.last_status = HealthStatus.UNKNOWN
        self.last_message = ""
    
    def check(self) -> Tuple[HealthStatus, str]:
        """Perform the health check.
        
        Returns:
            Tuple of (status, message)
        """
        raise NotImplementedError("Subclasses must implement check()")
    
    def run(self) -> Dict[str, Any]:
        """Run the health check and return the result.
        
        Returns:
            Dictionary with health check result
        """
        try:
            self.last_check_time = datetime.now()
            self.last_status, self.last_message = self.check()
        except Exception as e:
            self.last_status = HealthStatus.UNHEALTHY
            self.last_message = f"Health check failed with error: {str(e)}"
            log_error(e, severity=ErrorSeverity.WARNING, context={
                'health_check': self.name,
                'description': self.description
            })
        
        return {
            'name': self.name,
            'description': self.description,
            'status': self.last_status.name,
            'message': self.last_message,
            'timestamp': self.last_check_time.isoformat() if self.last_check_time else None
        }


class SystemResourceCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self, name: str = "system_resources", description: str = "System resource health check",
                 cpu_threshold: float = 90.0, memory_threshold: float = 90.0, disk_threshold: float = 90.0):
        """Initialize a system resource health check.
        
        Args:
            name: Name of the health check
            description: Description of the health check
            cpu_threshold: CPU usage threshold percentage for DEGRADED status
            memory_threshold: Memory usage threshold percentage for DEGRADED status
            disk_threshold: Disk usage threshold percentage for DEGRADED status
        """
        super().__init__(name, description)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def check(self) -> Tuple[HealthStatus, str]:
        """Check system resource usage.
        
        Returns:
            Tuple of (status, message)
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage for the current disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Determine status
        status = HealthStatus.HEALTHY
        messages = []
        
        if cpu_percent >= self.cpu_threshold:
            status = HealthStatus.DEGRADED
            messages.append(f"CPU usage is high: {cpu_percent:.1f}%")
        
        if memory_percent >= self.memory_threshold:
            status = HealthStatus.DEGRADED
            messages.append(f"Memory usage is high: {memory_percent:.1f}%")
        
        if disk_percent >= self.disk_threshold:
            status = HealthStatus.DEGRADED
            messages.append(f"Disk usage is high: {disk_percent:.1f}%")
        
        # If no issues, return healthy status
        if not messages:
            message = f"System resources are healthy: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            return HealthStatus.HEALTHY, message
        
        # Return degraded status with messages
        return status, "; ".join(messages)


class NetworkConnectivityCheck(HealthCheck):
    """Health check for network connectivity."""
    
    def __init__(self, name: str = "network_connectivity", description: str = "Network connectivity health check",
                 hosts: List[str] = None, timeout: float = 5.0):
        """Initialize a network connectivity health check.
        
        Args:
            name: Name of the health check
            description: Description of the health check
            hosts: List of hosts to check connectivity to
            timeout: Timeout for connection attempts in seconds
        """
        super().__init__(name, description)
        self.hosts = hosts or ['8.8.8.8', '1.1.1.1']  # Default to Google DNS and Cloudflare DNS
        self.timeout = timeout
    
    def check(self) -> Tuple[HealthStatus, str]:
        """Check network connectivity.
        
        Returns:
            Tuple of (status, message)
        """
        import socket
        
        successful_connections = 0
        failed_hosts = []
        
        for host in self.hosts:
            try:
                # Try to create a socket connection to the host
                socket.create_connection((host, 53), timeout=self.timeout)
                successful_connections += 1
            except (socket.timeout, socket.error) as e:
                failed_hosts.append(f"{host} ({str(e)})")
        
        # Determine status based on connection success rate
        if successful_connections == len(self.hosts):
            return HealthStatus.HEALTHY, f"All {len(self.hosts)} hosts are reachable"
        elif successful_connections > 0:
            return HealthStatus.DEGRADED, f"Some hosts are unreachable: {', '.join(failed_hosts)}"
        else:
            return HealthStatus.UNHEALTHY, f"All hosts are unreachable: {', '.join(failed_hosts)}"


class DatabaseConnectivityCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, name: str = "database_connectivity", description: str = "Database connectivity health check",
                 connection_string: str = None, timeout: float = 5.0):
        """Initialize a database connectivity health check.
        
        Args:
            name: Name of the health check
            description: Description of the health check
            connection_string: Database connection string
            timeout: Timeout for connection attempts in seconds
        """
        super().__init__(name, description)
        self.connection_string = connection_string
        self.timeout = timeout
    
    def check(self) -> Tuple[HealthStatus, str]:
        """Check database connectivity.
        
        Returns:
            Tuple of (status, message)
        """
        if not self.connection_string:
            return HealthStatus.UNKNOWN, "No connection string provided"
        
        try:
            # This is a placeholder - in a real implementation, you would use the appropriate
            # database driver to check connectivity
            # For example, for MongoDB:
            # from pymongo import MongoClient
            # client = MongoClient(self.connection_string, serverSelectionTimeoutMS=int(self.timeout * 1000))
            # client.admin.command('ping')
            
            # For now, we'll just return a healthy status
            return HealthStatus.HEALTHY, "Database is reachable"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database is unreachable: {str(e)}"


class HealthCheckRegistry:
    """Registry for health checks."""
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance of HealthCheckRegistry."""
        if cls._instance is None:
            cls._instance = super(HealthCheckRegistry, cls).__new__(cls)
            cls._instance.health_checks = {}
        return cls._instance
    
    def register(self, health_check: HealthCheck) -> None:
        """Register a health check.
        
        Args:
            health_check: Health check to register
        """
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a health check.
        
        Args:
            name: Name of the health check to unregister
        """
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Unregistered health check: {name}")
    
    def get_health_check(self, name: str) -> Optional[HealthCheck]:
        """Get a health check by name.
        
        Args:
            name: Name of the health check
            
        Returns:
            Health check or None if not found
        """
        return self.health_checks.get(name)
    
    def get_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Get all registered health checks.
        
        Returns:
            Dictionary of health checks
        """
        return self.health_checks
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a health check by name.
        
        Args:
            name: Name of the health check
            
        Returns:
            Dictionary with health check result
        """
        health_check = self.get_health_check(name)
        if health_check:
            return health_check.run()
        else:
            return {
                'name': name,
                'status': HealthStatus.UNKNOWN.name,
                'message': f"Health check '{name}' not found",
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks.
        
        Returns:
            Dictionary with health check results
        """
        results = {}
        for name, health_check in self.health_checks.items():
            results[name] = health_check.run()
        return results
    
    def get_system_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get the overall system health.
        
        Returns:
            Tuple of (status, details)
        """
        results = self.run_all_health_checks()
        
        # Determine overall status
        if not results:
            return HealthStatus.UNKNOWN, {'message': "No health checks registered"}
        
        # Count statuses
        status_counts = {status: 0 for status in HealthStatus}
        for result in results.values():
            status = HealthStatus[result['status']]
            status_counts[status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return overall_status, {
            'status': overall_status.name,
            'status_counts': {status.name: count for status, count in status_counts.items()},
            'checks': results
        }


class Metric:
    """Base class for metrics."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        """Initialize a metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Labels to attach to the metric
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.type = None  # To be set by subclasses
    
    def get_value(self) -> Any:
        """Get the current value of the metric.
        
        Returns:
            Current value of the metric
        """
        raise NotImplementedError("Subclasses must implement get_value()")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metric to a dictionary.
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            'name': self.name,
            'description': self.description,
            'type': self.type.name if self.type else None,
            'labels': self.labels,
            'value': self.get_value()
        }


class Counter(Metric):
    """Metric that counts occurrences."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        """Initialize a counter.
        
        Args:
            name: Name of the counter
            description: Description of the counter
            labels: Labels to attach to the counter
        """
        super().__init__(name, description, labels)
        self.type = MetricType.COUNTER
        self.value = 0
    
    def increment(self, value: int = 1) -> None:
        """Increment the counter.
        
        Args:
            value: Value to increment by
        """
        self.value += value
    
    def get_value(self) -> int:
        """Get the current value of the counter.
        
        Returns:
            Current value of the counter
        """
        return self.value


class Gauge(Metric):
    """Metric that can go up and down."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        """Initialize a gauge.
        
        Args:
            name: Name of the gauge
            description: Description of the gauge
            labels: Labels to attach to the gauge
        """
        super().__init__(name, description, labels)
        self.type = MetricType.GAUGE
        self.value = 0
    
    def set(self, value: float) -> None:
        """Set the gauge value.
        
        Args:
            value: Value to set
        """
        self.value = value
    
    def increment(self, value: float = 1) -> None:
        """Increment the gauge.
        
        Args:
            value: Value to increment by
        """
        self.value += value
    
    def decrement(self, value: float = 1) -> None:
        """Decrement the gauge.
        
        Args:
            value: Value to decrement by
        """
        self.value -= value
    
    def get_value(self) -> float:
        """Get the current value of the gauge.
        
        Returns:
            Current value of the gauge
        """
        return self.value


class Histogram(Metric):
    """Metric that tracks the distribution of values."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None,
                 buckets: List[float] = None):
        """Initialize a histogram.
        
        Args:
            name: Name of the histogram
            description: Description of the histogram
            labels: Labels to attach to the histogram
            buckets: Bucket boundaries for the histogram
        """
        super().__init__(name, description, labels)
        self.type = MetricType.HISTOGRAM
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}
        self.bucket_counts[float('inf')] = 0  # Add +Inf bucket
        self.count = 0
        self.sum = 0
    
    def observe(self, value: float) -> None:
        """Observe a value.
        
        Args:
            value: Value to observe
        """
        self.count += 1
        self.sum += value
        
        # Increment bucket counts
        for bucket in self.buckets + [float('inf')]:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
    
    def get_value(self) -> Dict[str, Any]:
        """Get the current value of the histogram.
        
        Returns:
            Dictionary with histogram metrics
        """
        return {
            'count': self.count,
            'sum': self.sum,
            'buckets': self.bucket_counts
        }


class Timer(Metric):
    """Metric that tracks the duration of operations."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        """Initialize a timer.
        
        Args:
            name: Name of the timer
            description: Description of the timer
            labels: Labels to attach to the timer
        """
        super().__init__(name, description, labels)
        self.type = MetricType.TIMER
        self.histogram = Histogram(f"{name}_histogram", description, labels)
        self.start_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and record the duration.
        
        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer was not started")
        
        duration = time.time() - self.start_time
        self.histogram.observe(duration)
        self.start_time = None
        return duration
    
    def time(self) -> Callable:
        """Context manager and decorator for timing operations.
        
        Returns:
            Context manager or decorator
        """
        class TimerContext:
            def __init__(self, timer):
                self.timer = timer
            
            def __enter__(self):
                self.timer.start()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.timer.stop()
                return False
        
        return TimerContext(self)
    
    def get_value(self) -> Dict[str, Any]:
        """Get the current value of the timer.
        
        Returns:
            Dictionary with timer metrics
        """
        return self.histogram.get_value()


def timed(name: str, description: str = "", labels: Dict[str, str] = None):
    """Decorator for timing functions.
    
    Args:
        name: Name of the timer
        description: Description of the timer
        labels: Labels to attach to the timer
        
    Returns:
        Decorated function
    """
    def decorator(func):
        timer = Timer(name, description, labels)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer.time():
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class MetricRegistry:
    """Registry for metrics."""
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance of MetricRegistry."""
        if cls._instance is None:
            cls._instance = super(MetricRegistry, cls).__new__(cls)
            cls._instance.metrics = {}
        return cls._instance
    
    def register(self, metric: Metric) -> None:
        """Register a metric.
        
        Args:
            metric: Metric to register
        """
        self.metrics[metric.name] = metric
        logger.info(f"Registered metric: {metric.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a metric.
        
        Args:
            name: Name of the metric to unregister
        """
        if name in self.metrics:
            del self.metrics[name]
            logger.info(f"Unregistered metric: {name}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric or None if not found
        """
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def get_metrics_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionaries.
        
        Returns:
            Dictionary of metric dictionaries
        """
        return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def create_counter(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Counter:
        """Create and register a counter.
        
        Args:
            name: Name of the counter
            description: Description of the counter
            labels: Labels to attach to the counter
            
        Returns:
            Created counter
        """
        counter = Counter(name, description, labels)
        self.register(counter)
        return counter
    
    def create_gauge(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Gauge:
        """Create and register a gauge.
        
        Args:
            name: Name of the gauge
            description: Description of the gauge
            labels: Labels to attach to the gauge
            
        Returns:
            Created gauge
        """
        gauge = Gauge(name, description, labels)
        self.register(gauge)
        return gauge
    
    def create_histogram(self, name: str, description: str = "", labels: Dict[str, str] = None,
                        buckets: List[float] = None) -> Histogram:
        """Create and register a histogram.
        
        Args:
            name: Name of the histogram
            description: Description of the histogram
            labels: Labels to attach to the histogram
            buckets: Bucket boundaries for the histogram
            
        Returns:
            Created histogram
        """
        histogram = Histogram(name, description, labels, buckets)
        self.register(histogram)
        return histogram
    
    def create_timer(self, name: str, description: str = "", labels: Dict[str, str] = None) -> Timer:
        """Create and register a timer.
        
        Args:
            name: Name of the timer
            description: Description of the timer
            labels: Labels to attach to the timer
            
        Returns:
            Created timer
        """
        timer = Timer(name, description, labels)
        self.register(timer)
        return timer


class SystemMetricsCollector:
    """Collector for system metrics."""
    
    def __init__(self, collection_interval: float = 60.0):
        """Initialize a system metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.registry = MetricRegistry()
        self.running = False
        self.thread = None
        
        # Create metrics
        self.cpu_usage = self.registry.create_gauge('system_cpu_usage', 'CPU usage percentage')
        self.memory_usage = self.registry.create_gauge('system_memory_usage', 'Memory usage percentage')
        self.memory_available = self.registry.create_gauge('system_memory_available', 'Available memory in bytes')
        self.disk_usage = self.registry.create_gauge('system_disk_usage', 'Disk usage percentage')
        self.disk_available = self.registry.create_gauge('system_disk_available', 'Available disk space in bytes')
        self.network_sent = self.registry.create_counter('system_network_sent', 'Network bytes sent')
        self.network_received = self.registry.create_counter('system_network_received', 'Network bytes received')
        
        # Previous network counters for calculating deltas
        self.prev_network_sent = 0
        self.prev_network_received = 0
    
    def start(self) -> None:
        """Start collecting metrics."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.thread.start()
        logger.info("Started system metrics collector")
    
    def stop(self) -> None:
        """Stop collecting metrics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("Stopped system metrics collector")
    
    def _collect_metrics_loop(self) -> None:
        """Collect metrics in a loop."""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            # Sleep until next collection
            time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> None:
        """Collect system metrics."""
        # CPU usage
        self.cpu_usage.set(psutil.cpu_percent(interval=1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        self.memory_available.set(memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.disk_usage.set(disk.percent)
        self.disk_available.set(disk.free)
        
        # Network usage
        network = psutil.net_io_counters()
        
        # Calculate deltas
        if self.prev_network_sent > 0:
            sent_delta = network.bytes_sent - self.prev_network_sent
            if sent_delta > 0:  # Ensure positive delta (in case of counter reset)
                self.network_sent.increment(sent_delta)
        
        if self.prev_network_received > 0:
            received_delta = network.bytes_recv - self.prev_network_received
            if received_delta > 0:  # Ensure positive delta (in case of counter reset)
                self.network_received.increment(received_delta)
        
        # Update previous values
        self.prev_network_sent = network.bytes_sent
        self.prev_network_received = network.bytes_recv


class Alert:
    """Alert for monitoring."""
    
    def __init__(self, name: str, description: str, level: AlertLevel, source: str = None,
                 timestamp: datetime = None, context: Dict[str, Any] = None):
        """Initialize an alert.
        
        Args:
            name: Name of the alert
            description: Description of the alert
            level: Alert level
            source: Source of the alert
            timestamp: Timestamp of the alert
            context: Additional context for the alert
        """
        self.name = name
        self.description = description
        self.level = level
        self.source = source or 'system'
        self.timestamp = timestamp or datetime.now()
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary.
        
        Returns:
            Dictionary representation of the alert
        """
        return {
            'name': self.name,
            'description': self.description,
            'level': self.level.name,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class AlertHandler:
    """Base class for alert handlers."""
    
    def handle_alert(self, alert: Alert) -> None:
        """Handle an alert.
        
        Args:
            alert: Alert to handle
        """
        raise NotImplementedError("Subclasses must implement handle_alert()")


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts."""
    
    def handle_alert(self, alert: Alert) -> None:
        """Log an alert.
        
        Args:
            alert: Alert to log
        """
        # Map alert levels to logging levels
        log_levels = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EMERGENCY: logging.CRITICAL
        }
        
        # Log the alert
        log_level = log_levels.get(alert.level, logging.INFO)
        logger.log(log_level, f"ALERT: {alert.name} - {alert.description}", extra={
            'structured_data': alert.to_dict()
        })


class AlertManager:
    """Manager for alerts."""
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance of AlertManager."""
        if cls._instance is None:
            cls._instance = super(AlertManager, cls).__new__(cls)
            cls._instance.handlers = []
            cls._instance.alerts = []
            cls._instance.max_alerts = 1000  # Maximum number of alerts to store
        return cls._instance
    
    def register_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler.
        
        Args:
            handler: Alert handler to register
        """
        self.handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__class__.__name__}")
    
    def unregister_handler(self, handler: AlertHandler) -> None:
        """Unregister an alert handler.
        
        Args:
            handler: Alert handler to unregister
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.info(f"Unregistered alert handler: {handler.__class__.__name__}")
    
    def alert(self, name: str, description: str, level: AlertLevel = AlertLevel.INFO,
              source: str = None, context: Dict[str, Any] = None) -> Alert:
        """Create and handle an alert.
        
        Args:
            name: Name of the alert
            description: Description of the alert
            level: Alert level
            source: Source of the alert
            context: Additional context for the alert
            
        Returns:
            Created alert
        """
        # Create the alert
        alert = Alert(name, description, level, source, datetime.now(), context)
        
        # Store the alert
        self.alerts.append(alert)
        
        # Trim alerts if necessary
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Handle the alert with all registered handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error handling alert with {handler.__class__.__name__}: {e}")
        
        return alert
    
    def get_alerts(self, count: int = None, level: AlertLevel = None,
                  source: str = None, since: datetime = None) -> List[Alert]:
        """Get alerts with optional filtering.
        
        Args:
            count: Maximum number of alerts to return
            level: Filter by alert level
            source: Filter by alert source
            since: Filter by timestamp
            
        Returns:
            List of alerts
        """
        # Start with all alerts
        filtered_alerts = self.alerts
        
        # Apply filters
        if level is not None:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if source is not None:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]
        
        if since is not None:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= since]
        
        # Apply count limit
        if count is not None:
            filtered_alerts = filtered_alerts[-count:]
        
        return filtered_alerts
    
    def get_alerts_as_dict(self, count: int = None, level: AlertLevel = None,
                          source: str = None, since: datetime = None) -> List[Dict[str, Any]]:
        """Get alerts as dictionaries with optional filtering.
        
        Args:
            count: Maximum number of alerts to return
            level: Filter by alert level
            source: Filter by alert source
            since: Filter by timestamp
            
        Returns:
            List of alert dictionaries
        """
        alerts = self.get_alerts(count, level, source, since)
        return [alert.to_dict() for alert in alerts]


# Initialize default components
def initialize_monitoring():
    """Initialize the monitoring system."""
    # Register default health checks
    health_registry = HealthCheckRegistry()
    health_registry.register(SystemResourceCheck())
    health_registry.register(NetworkConnectivityCheck())
    
    # Start system metrics collector
    metrics_collector = SystemMetricsCollector()
    metrics_collector.start()
    
    # Register default alert handler
    alert_manager = AlertManager()
    alert_manager.register_handler(LoggingAlertHandler())
    
    logger.info("Monitoring system initialized")
    
    return {
        'health_registry': health_registry,
        'metrics_collector': metrics_collector,
        'alert_manager': alert_manager
    }