"""Health monitoring for external system integrations.

This module provides utilities for monitoring the health and performance of external systems,
including health checks, performance metrics, and alerting.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import time
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json
import os

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.event import EventSystem, Event

# Create logger
logger = get_logger(__name__)


class HealthCheckError(FridayError):
    """Exception raised for errors in health checks."""
    pass


class HealthStatus(Enum):
    """Health status of an external system."""
    UNKNOWN = 'unknown'
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'


class AlertLevel(Enum):
    """Alert levels for health check alerts."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class HealthMetrics:
    """Class for tracking health metrics of an external system."""
    
    def __init__(self, system_id: str, window_size: int = 100):
        """Initialize health metrics.
        
        Args:
            system_id: The ID of the external system.
            window_size: The number of data points to keep for metrics.
        """
        self.system_id = system_id
        self.window_size = window_size
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.request_counts: Dict[str, int] = {}
        self.last_check_time: Optional[datetime] = None
        self.last_status: HealthStatus = HealthStatus.UNKNOWN
        self.consecutive_failures = 0
        self.uptime_start: Optional[datetime] = None
        self.downtime_periods: List[Tuple[datetime, datetime]] = []
        self.lock = threading.RLock()
        
    def record_response_time(self, response_time: float):
        """Record a response time.
        
        Args:
            response_time: The response time in seconds.
        """
        with self.lock:
            self.response_times.append(response_time)
            if len(self.response_times) > self.window_size:
                self.response_times.pop(0)
                
    def record_request(self, endpoint: str):
        """Record a request.
        
        Args:
            endpoint: The endpoint that was requested.
        """
        with self.lock:
            self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
            
    def record_error(self, error_type: str):
        """Record an error.
        
        Args:
            error_type: The type of error that occurred.
        """
        with self.lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
    def record_check(self, status: HealthStatus):
        """Record a health check.
        
        Args:
            status: The status of the health check.
        """
        with self.lock:
            now = datetime.now()
            self.last_check_time = now
            
            # Update consecutive failures
            if status == HealthStatus.UNHEALTHY:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0
                
            # Update uptime tracking
            if status == HealthStatus.HEALTHY:
                if self.uptime_start is None:
                    self.uptime_start = now
            elif status == HealthStatus.UNHEALTHY and self.last_status != HealthStatus.UNHEALTHY:
                if self.uptime_start is not None:
                    self.downtime_periods.append((now, None))
                    self.uptime_start = None
            elif status != HealthStatus.UNHEALTHY and self.last_status == HealthStatus.UNHEALTHY:
                if self.downtime_periods and self.downtime_periods[-1][1] is None:
                    self.downtime_periods[-1] = (self.downtime_periods[-1][0], now)
                    self.uptime_start = now
                    
            self.last_status = status
            
    def get_average_response_time(self) -> Optional[float]:
        """Get the average response time.
        
        Returns:
            Optional[float]: The average response time in seconds, or None if no data.
        """
        with self.lock:
            if not self.response_times:
                return None
            return statistics.mean(self.response_times)
            
    def get_percentile_response_time(self, percentile: float) -> Optional[float]:
        """Get a percentile of response times.
        
        Args:
            percentile: The percentile to get (0-100).
            
        Returns:
            Optional[float]: The percentile response time in seconds, or None if no data.
        """
        with self.lock:
            if not self.response_times:
                return None
            return statistics.quantiles(sorted(self.response_times), n=100)[int(percentile) - 1]
            
    def get_error_rate(self) -> float:
        """Get the error rate.
        
        Returns:
            float: The error rate as a percentage.
        """
        with self.lock:
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            if total_requests == 0:
                return 0.0
            return (total_errors / total_requests) * 100.0
            
    def get_uptime_percentage(self) -> float:
        """Get the uptime percentage.
        
        Returns:
            float: The uptime percentage.
        """
        with self.lock:
            if self.uptime_start is None and not self.downtime_periods:
                return 0.0
                
            now = datetime.now()
            total_time = timedelta()
            uptime = timedelta()
            
            # Add time from the first uptime start or the first downtime start
            start_time = self.uptime_start
            if start_time is None and self.downtime_periods:
                start_time = self.downtime_periods[0][0]
                
            if start_time is not None:
                total_time = now - start_time
                
                # Add uptime from current period
                if self.uptime_start is not None:
                    uptime += now - self.uptime_start
                    
                # Add uptime from previous periods
                for down_start, down_end in self.downtime_periods:
                    if down_end is not None:
                        # If we have a complete downtime period, subtract it from total time
                        downtime = down_end - down_start
                        uptime += total_time - downtime
                        
            if total_time.total_seconds() == 0:
                return 100.0
                
            return (uptime.total_seconds() / total_time.total_seconds()) * 100.0
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics.
        
        Returns:
            Dict[str, Any]: A dictionary of all metrics.
        """
        with self.lock:
            avg_response_time = self.get_average_response_time()
            p95_response_time = self.get_percentile_response_time(95)
            error_rate = self.get_error_rate()
            uptime = self.get_uptime_percentage()
            
            return {
                'system_id': self.system_id,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'last_status': self.last_status.value,
                'consecutive_failures': self.consecutive_failures,
                'average_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'error_rate': error_rate,
                'uptime_percentage': uptime,
                'request_counts': self.request_counts,
                'error_counts': self.error_counts
            }
            
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.response_times = []
            self.error_counts = {}
            self.request_counts = {}
            self.last_check_time = None
            self.last_status = HealthStatus.UNKNOWN
            self.consecutive_failures = 0
            self.uptime_start = None
            self.downtime_periods = []


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, system_id: str, name: str, description: str, interval: int = 60):
        """Initialize a health check.
        
        Args:
            system_id: The ID of the external system.
            name: The name of the health check.
            description: The description of the health check.
            interval: The interval between checks in seconds.
        """
        self.system_id = system_id
        self.name = name
        self.description = description
        self.interval = interval
        self.last_run_time: Optional[datetime] = None
        self.last_status: HealthStatus = HealthStatus.UNKNOWN
        self.last_message: str = ""
        self.metrics = HealthMetrics(system_id)
        
    def check(self) -> Tuple[HealthStatus, str]:
        """Perform the health check.
        
        Returns:
            Tuple[HealthStatus, str]: The status and a message.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement check()")
        
    def run(self) -> Tuple[HealthStatus, str]:
        """Run the health check and update metrics.
        
        Returns:
            Tuple[HealthStatus, str]: The status and a message.
        """
        start_time = time.time()
        try:
            status, message = self.check()
        except Exception as e:
            logger.error(f"Health check '{self.name}' for system '{self.system_id}' failed: {str(e)}")
            status = HealthStatus.UNHEALTHY
            message = f"Health check failed: {str(e)}"
            
        end_time = time.time()
        response_time = end_time - start_time
        
        # Update metrics
        self.metrics.record_response_time(response_time)
        self.metrics.record_check(status)
        if status == HealthStatus.UNHEALTHY:
            self.metrics.record_error("health_check_failure")
            
        self.last_run_time = datetime.now()
        self.last_status = status
        self.last_message = message
        
        return status, message
        
    def should_run(self) -> bool:
        """Check if the health check should run.
        
        Returns:
            bool: True if the health check should run, False otherwise.
        """
        if self.last_run_time is None:
            return True
            
        now = datetime.now()
        elapsed = (now - self.last_run_time).total_seconds()
        return elapsed >= self.interval
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the health check.
        
        Returns:
            Dict[str, Any]: Information about the health check.
        """
        return {
            'system_id': self.system_id,
            'name': self.name,
            'description': self.description,
            'interval': self.interval,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'last_status': self.last_status.value,
            'last_message': self.last_message
        }


class HttpHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""
    
    def __init__(self, system_id: str, name: str, url: str, method: str = 'GET',
                 headers: Optional[Dict[str, str]] = None, body: Optional[str] = None,
                 timeout: int = 10, expected_status: int = 200,
                 expected_content: Optional[str] = None, description: str = "",
                 interval: int = 60):
        """Initialize an HTTP health check.
        
        Args:
            system_id: The ID of the external system.
            name: The name of the health check.
            url: The URL to check.
            method: The HTTP method to use.
            headers: The headers to include in the request.
            body: The body to include in the request.
            timeout: The timeout for the request in seconds.
            expected_status: The expected HTTP status code.
            expected_content: The expected content in the response.
            description: The description of the health check.
            interval: The interval between checks in seconds.
        """
        super().__init__(system_id, name, description or f"HTTP health check for {url}", interval)
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.body = body
        self.timeout = timeout
        self.expected_status = expected_status
        self.expected_content = expected_content
        
    def check(self) -> Tuple[HealthStatus, str]:
        """Perform the health check.
        
        Returns:
            Tuple[HealthStatus, str]: The status and a message.
        """
        import requests
        
        try:
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                data=self.body,
                timeout=self.timeout
            )
            
            # Check status code
            if response.status_code != self.expected_status:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Unexpected status code: {response.status_code} (expected {self.expected_status})"
                )
                
            # Check content
            if self.expected_content and self.expected_content not in response.text:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Expected content not found in response"
                )
                
            return HealthStatus.HEALTHY, "Health check passed"
        except requests.exceptions.Timeout:
            return HealthStatus.UNHEALTHY, "Request timed out"
        except requests.exceptions.ConnectionError:
            return HealthStatus.UNHEALTHY, "Connection error"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Health check failed: {str(e)}"


class WebSocketHealthCheck(HealthCheck):
    """Health check for WebSocket endpoints."""
    
    def __init__(self, system_id: str, name: str, url: str, message: Optional[str] = None,
                 timeout: int = 10, expected_response: Optional[str] = None,
                 description: str = "", interval: int = 60):
        """Initialize a WebSocket health check.
        
        Args:
            system_id: The ID of the external system.
            name: The name of the health check.
            url: The WebSocket URL to check.
            message: The message to send.
            timeout: The timeout for the connection in seconds.
            expected_response: The expected response.
            description: The description of the health check.
            interval: The interval between checks in seconds.
        """
        super().__init__(system_id, name, description or f"WebSocket health check for {url}", interval)
        self.url = url
        self.message = message
        self.timeout = timeout
        self.expected_response = expected_response
        
    def check(self) -> Tuple[HealthStatus, str]:
        """Perform the health check.
        
        Returns:
            Tuple[HealthStatus, str]: The status and a message.
        """
        import websocket
        
        try:
            # Connect to the WebSocket
            ws = websocket.create_connection(self.url, timeout=self.timeout)
            
            # Send a message if specified
            if self.message:
                ws.send(self.message)
                
                # Wait for a response if expected
                if self.expected_response:
                    response = ws.recv()
                    if self.expected_response not in response:
                        ws.close()
                        return (
                            HealthStatus.UNHEALTHY,
                            f"Expected response not found"
                        )
                        
            # Close the connection
            ws.close()
            
            return HealthStatus.HEALTHY, "Health check passed"
        except websocket.WebSocketTimeoutException:
            return HealthStatus.UNHEALTHY, "Connection timed out"
        except websocket.WebSocketConnectionClosedException:
            return HealthStatus.UNHEALTHY, "Connection closed unexpectedly"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Health check failed: {str(e)}"


class CustomHealthCheck(HealthCheck):
    """Custom health check using a provided function."""
    
    def __init__(self, system_id: str, name: str, check_func: Callable[[], Tuple[bool, str]],
                 description: str = "", interval: int = 60):
        """Initialize a custom health check.
        
        Args:
            system_id: The ID of the external system.
            name: The name of the health check.
            check_func: A function that performs the health check and returns a tuple of (success, message).
            description: The description of the health check.
            interval: The interval between checks in seconds.
        """
        super().__init__(system_id, name, description or "Custom health check", interval)
        self.check_func = check_func
        
    def check(self) -> Tuple[HealthStatus, str]:
        """Perform the health check.
        
        Returns:
            Tuple[HealthStatus, str]: The status and a message.
        """
        try:
            success, message = self.check_func()
            if success:
                return HealthStatus.HEALTHY, message
            else:
                return HealthStatus.UNHEALTHY, message
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Health check failed: {str(e)}"


class HealthAlert:
    """Class for health check alerts."""
    
    def __init__(self, system_id: str, check_name: str, status: HealthStatus,
                 message: str, level: AlertLevel, timestamp: datetime):
        """Initialize a health alert.
        
        Args:
            system_id: The ID of the external system.
            check_name: The name of the health check.
            status: The status of the health check.
            message: The message from the health check.
            level: The alert level.
            timestamp: The time the alert was generated.
        """
        self.system_id = system_id
        self.check_name = check_name
        self.status = status
        self.message = message
        self.level = level
        self.timestamp = timestamp
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary.
        
        Returns:
            Dict[str, Any]: The alert as a dictionary.
        """
        return {
            'system_id': self.system_id,
            'check_name': self.check_name,
            'status': self.status.value,
            'message': self.message,
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, alert_dict: Dict[str, Any]) -> 'HealthAlert':
        """Create an alert from a dictionary.
        
        Args:
            alert_dict: The alert dictionary.
            
        Returns:
            HealthAlert: The created alert.
        """
        return cls(
            system_id=alert_dict['system_id'],
            check_name=alert_dict['check_name'],
            status=HealthStatus(alert_dict['status']),
            message=alert_dict['message'],
            level=AlertLevel(alert_dict['level']),
            timestamp=datetime.fromisoformat(alert_dict['timestamp'])
        )


class HealthAlertEvent(Event):
    """Event for health alerts."""
    
    def __init__(self, alert: HealthAlert):
        """Initialize a health alert event.
        
        Args:
            alert: The health alert.
        """
        super().__init__(f"health_alert.{alert.system_id}.{alert.check_name}")
        self.alert = alert
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['alert'] = self.alert.to_dict()
        return event_dict


class HealthMonitor:
    """Class for monitoring the health of external systems."""
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize a health monitor.
        
        Args:
            event_system: The event system to use for alerts.
        """
        self.checks: Dict[str, List[HealthCheck]] = {}
        self.metrics: Dict[str, HealthMetrics] = {}
        self.alerts: List[HealthAlert] = []
        self.event_system = event_system
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
    def add_check(self, check: HealthCheck):
        """Add a health check.
        
        Args:
            check: The health check to add.
        """
        with self.lock:
            if check.system_id not in self.checks:
                self.checks[check.system_id] = []
            self.checks[check.system_id].append(check)
            
            # Add metrics if not already present
            if check.system_id not in self.metrics:
                self.metrics[check.system_id] = HealthMetrics(check.system_id)
                
    def remove_check(self, system_id: str, check_name: str) -> bool:
        """Remove a health check.
        
        Args:
            system_id: The ID of the external system.
            check_name: The name of the health check.
            
        Returns:
            bool: True if the check was removed, False otherwise.
        """
        with self.lock:
            if system_id not in self.checks:
                return False
                
            for i, check in enumerate(self.checks[system_id]):
                if check.name == check_name:
                    self.checks[system_id].pop(i)
                    return True
                    
            return False
            
    def get_check(self, system_id: str, check_name: str) -> Optional[HealthCheck]:
        """Get a health check.
        
        Args:
            system_id: The ID of the external system.
            check_name: The name of the health check.
            
        Returns:
            Optional[HealthCheck]: The health check, or None if not found.
        """
        with self.lock:
            if system_id not in self.checks:
                return None
                
            for check in self.checks[system_id]:
                if check.name == check_name:
                    return check
                    
            return None
            
    def get_checks(self, system_id: str) -> List[HealthCheck]:
        """Get all health checks for a system.
        
        Args:
            system_id: The ID of the external system.
            
        Returns:
            List[HealthCheck]: The health checks for the system.
        """
        with self.lock:
            return self.checks.get(system_id, [])
            
    def get_all_checks(self) -> Dict[str, List[HealthCheck]]:
        """Get all health checks.
        
        Returns:
            Dict[str, List[HealthCheck]]: All health checks.
        """
        with self.lock:
            return self.checks.copy()
            
    def get_metrics(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a system.
        
        Args:
            system_id: The ID of the external system.
            
        Returns:
            Optional[Dict[str, Any]]: The metrics for the system, or None if not found.
        """
        with self.lock:
            if system_id not in self.metrics:
                return None
            return self.metrics[system_id].get_metrics()
            
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all systems.
        
        Returns:
            Dict[str, Dict[str, Any]]: Metrics for all systems.
        """
        with self.lock:
            return {system_id: metrics.get_metrics() for system_id, metrics in self.metrics.items()}
            
    def get_alerts(self, system_id: Optional[str] = None, limit: int = 100) -> List[HealthAlert]:
        """Get alerts.
        
        Args:
            system_id: The ID of the external system, or None for all systems.
            limit: The maximum number of alerts to return.
            
        Returns:
            List[HealthAlert]: The alerts.
        """
        with self.lock:
            if system_id is None:
                return sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
            else:
                return sorted(
                    [a for a in self.alerts if a.system_id == system_id],
                    key=lambda a: a.timestamp,
                    reverse=True
                )[:limit]
                
    def add_alert(self, alert: HealthAlert):
        """Add an alert.
        
        Args:
            alert: The alert to add.
        """
        with self.lock:
            self.alerts.append(alert)
            
            # Limit the number of alerts
            if len(self.alerts) > 1000:
                self.alerts = sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:1000]
                
            # Send an event if an event system is available
            if self.event_system is not None:
                self.event_system.publish(HealthAlertEvent(alert))
                
    def run_check(self, system_id: str, check_name: str) -> Tuple[HealthStatus, str]:
        """Run a health check.
        
        Args:
            system_id: The ID of the external system.
            check_name: The name of the health check.
            
        Returns:
            Tuple[HealthStatus, str]: The status and message from the check.
            
        Raises:
            HealthCheckError: If the check is not found.
        """
        check = self.get_check(system_id, check_name)
        if check is None:
            raise HealthCheckError(f"Health check '{check_name}' for system '{system_id}' not found")
            
        status, message = check.run()
        
        # Update system metrics
        with self.lock:
            if system_id in self.metrics:
                self.metrics[system_id].record_check(status)
                
        # Generate an alert if the status is unhealthy
        if status == HealthStatus.UNHEALTHY:
            alert = HealthAlert(
                system_id=system_id,
                check_name=check_name,
                status=status,
                message=message,
                level=AlertLevel.ERROR,
                timestamp=datetime.now()
            )
            self.add_alert(alert)
            
        return status, message
        
    def run_all_checks(self, system_id: str) -> Dict[str, Tuple[HealthStatus, str]]:
        """Run all health checks for a system.
        
        Args:
            system_id: The ID of the external system.
            
        Returns:
            Dict[str, Tuple[HealthStatus, str]]: The results of the checks.
        """
        results = {}
        for check in self.get_checks(system_id):
            try:
                status, message = self.run_check(system_id, check.name)
                results[check.name] = (status, message)
            except Exception as e:
                logger.error(f"Failed to run health check '{check.name}' for system '{system_id}': {str(e)}")
                results[check.name] = (HealthStatus.UNHEALTHY, f"Failed to run health check: {str(e)}")
                
        return results
        
    def get_system_status(self, system_id: str) -> HealthStatus:
        """Get the overall status of a system.
        
        Args:
            system_id: The ID of the external system.
            
        Returns:
            HealthStatus: The overall status of the system.
        """
        checks = self.get_checks(system_id)
        if not checks:
            return HealthStatus.UNKNOWN
            
        # Count the number of checks in each status
        status_counts = {status: 0 for status in HealthStatus}
        for check in checks:
            status_counts[check.last_status] += 1
            
        # If any check is unhealthy, the system is unhealthy
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
            
        # If any check is degraded, the system is degraded
        if status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
            
        # If all checks are healthy, the system is healthy
        if status_counts[HealthStatus.HEALTHY] == len(checks):
            return HealthStatus.HEALTHY
            
        # Otherwise, the status is unknown
        return HealthStatus.UNKNOWN
        
    def get_all_system_statuses(self) -> Dict[str, HealthStatus]:
        """Get the overall status of all systems.
        
        Returns:
            Dict[str, HealthStatus]: The overall status of all systems.
        """
        return {system_id: self.get_system_status(system_id) for system_id in self.checks.keys()}
        
    def start_monitoring(self):
        """Start the monitoring thread."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_thread, daemon=True)
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        with self.lock:
            self.running = False
            if self.monitor_thread is not None:
                self.monitor_thread.join(timeout=5.0)
                self.monitor_thread = None
                
    def _monitor_thread(self):
        """Thread for running health checks."""
        while self.running:
            try:
                # Get all checks that should run
                checks_to_run = []
                with self.lock:
                    for system_checks in self.checks.values():
                        for check in system_checks:
                            if check.should_run():
                                checks_to_run.append((check.system_id, check.name))
                                
                # Run the checks
                for system_id, check_name in checks_to_run:
                    try:
                        self.run_check(system_id, check_name)
                    except Exception as e:
                        logger.error(f"Failed to run health check '{check_name}' for system '{system_id}': {str(e)}")
            except Exception as e:
                logger.error(f"Error in health monitor thread: {str(e)}")
                
            # Sleep for a short time
            time.sleep(1.0)
            
    def save_metrics(self, file_path: str):
        """Save metrics to a file.
        
        Args:
            file_path: The path to save the metrics to.
            
        Raises:
            HealthCheckError: If the metrics cannot be saved.
        """
        try:
            metrics = self.get_all_metrics()
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the metrics
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Saved health metrics to {file_path}")
        except Exception as e:
            raise HealthCheckError(f"Failed to save health metrics: {str(e)}") from e
            
    def load_metrics(self, file_path: str):
        """Load metrics from a file.
        
        Args:
            file_path: The path to load the metrics from.
            
        Raises:
            HealthCheckError: If the metrics cannot be loaded.
        """
        try:
            # Load the metrics
            with open(file_path, 'r') as f:
                metrics_data = json.load(f)
                
            # Create metrics objects
            with self.lock:
                for system_id, metrics_dict in metrics_data.items():
                    if system_id not in self.metrics:
                        self.metrics[system_id] = HealthMetrics(system_id)
                        
            logger.info(f"Loaded health metrics from {file_path}")
        except Exception as e:
            raise HealthCheckError(f"Failed to load health metrics: {str(e)}") from e
            
    def save_alerts(self, file_path: str):
        """Save alerts to a file.
        
        Args:
            file_path: The path to save the alerts to.
            
        Raises:
            HealthCheckError: If the alerts cannot be saved.
        """
        try:
            # Get alerts
            with self.lock:
                alerts_data = [alert.to_dict() for alert in self.alerts]
                
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the alerts
            with open(file_path, 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
            logger.info(f"Saved health alerts to {file_path}")
        except Exception as e:
            raise HealthCheckError(f"Failed to save health alerts: {str(e)}") from e
            
    def load_alerts(self, file_path: str):
        """Load alerts from a file.
        
        Args:
            file_path: The path to load the alerts from.
            
        Raises:
            HealthCheckError: If the alerts cannot be loaded.
        """
        try:
            # Load the alerts
            with open(file_path, 'r') as f:
                alerts_data = json.load(f)
                
            # Create alert objects
            with self.lock:
                self.alerts = [HealthAlert.from_dict(alert_dict) for alert_dict in alerts_data]
                
            logger.info(f"Loaded health alerts from {file_path}")
        except Exception as e:
            raise HealthCheckError(f"Failed to load health alerts: {str(e)}") from e


def create_http_health_check(system_id: str, name: str, url: str, **kwargs) -> HttpHealthCheck:
    """Create an HTTP health check.
    
    Args:
        system_id: The ID of the external system.
        name: The name of the health check.
        url: The URL to check.
        **kwargs: Additional arguments for the health check.
        
    Returns:
        HttpHealthCheck: The created health check.
    """
    return HttpHealthCheck(system_id, name, url, **kwargs)


def create_websocket_health_check(system_id: str, name: str, url: str, **kwargs) -> WebSocketHealthCheck:
    """Create a WebSocket health check.
    
    Args:
        system_id: The ID of the external system.
        name: The name of the health check.
        url: The WebSocket URL to check.
        **kwargs: Additional arguments for the health check.
        
    Returns:
        WebSocketHealthCheck: The created health check.
    """
    return WebSocketHealthCheck(system_id, name, url, **kwargs)


def create_custom_health_check(system_id: str, name: str, check_func: Callable[[], Tuple[bool, str]],
                             **kwargs) -> CustomHealthCheck:
    """Create a custom health check.
    
    Args:
        system_id: The ID of the external system.
        name: The name of the health check.
        check_func: A function that performs the health check and returns a tuple of (success, message).
        **kwargs: Additional arguments for the health check.
        
    Returns:
        CustomHealthCheck: The created health check.
    """
    return CustomHealthCheck(system_id, name, check_func, **kwargs)


def create_health_check_from_config(system_id: str, config: Dict[str, Any]) -> HealthCheck:
    """Create a health check from a configuration dictionary.
    
    Args:
        system_id: The ID of the external system.
        config: The configuration dictionary.
        
    Returns:
        HealthCheck: The created health check.
        
    Raises:
        HealthCheckError: If the configuration is invalid.
    """
    try:
        check_type = config.get('type')
        name = config.get('name')
        
        if not check_type or not name:
            raise HealthCheckError("Health check type and name are required")
            
        if check_type == 'http':
            url = config.get('url')
            if not url:
                raise HealthCheckError("URL is required for HTTP health check")
                
            return create_http_health_check(
                system_id=system_id,
                name=name,
                url=url,
                method=config.get('method', 'GET'),
                headers=config.get('headers'),
                body=config.get('body'),
                timeout=config.get('timeout', 10),
                expected_status=config.get('expected_status', 200),
                expected_content=config.get('expected_content'),
                description=config.get('description', ''),
                interval=config.get('interval', 60)
            )
        elif check_type == 'websocket':
            url = config.get('url')
            if not url:
                raise HealthCheckError("URL is required for WebSocket health check")
                
            return create_websocket_health_check(
                system_id=system_id,
                name=name,
                url=url,
                message=config.get('message'),
                timeout=config.get('timeout', 10),
                expected_response=config.get('expected_response'),
                description=config.get('description', ''),
                interval=config.get('interval', 60)
            )
        else:
            raise HealthCheckError(f"Unknown health check type: {check_type}")
    except Exception as e:
        raise HealthCheckError(f"Failed to create health check: {str(e)}") from e


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance.
    
    Returns:
        HealthMonitor: The global health monitor instance.
    """
    # Use a global variable to store the health monitor instance
    global _health_monitor
    
    # Create the health monitor if it doesn't exist
    if '_health_monitor' not in globals():
        from src.infrastructure.event import get_event_system
        _health_monitor = HealthMonitor(get_event_system())
        
    return _health_monitor