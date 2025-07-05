"""Event monitoring and visualization for the Friday AI Trading System.

This module provides tools for monitoring, analyzing, and visualizing
events in the event system.
"""

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.event.event_system import Event, EventSystem
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class EventMonitor:
    """Monitor for tracking and analyzing events in the system.
    
    This class provides functionality to track event rates, types,
    and other metrics for monitoring and debugging purposes.
    """
    
    def __init__(self, event_system: EventSystem, sampling_interval: float = 1.0):
        """Initialize the event monitor.
        
        Args:
            event_system: The event system to monitor.
            sampling_interval: The interval in seconds at which to sample metrics.
        """
        self.event_system = event_system
        self.sampling_interval = sampling_interval
        
        # Metrics storage
        self._event_counts: Dict[str, int] = Counter()
        self._event_rates: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self._event_latencies: Dict[str, List[float]] = defaultdict(list)
        self._event_sizes: Dict[str, List[int]] = defaultdict(list)
        
        # Time windows for rate calculation
        self._time_windows = [60, 300, 900]  # 1min, 5min, 15min in seconds
        
        # Thread for periodic sampling
        self._sampling_thread: Optional[Thread] = None
        self._running = False
        self._lock = Lock()
        
        # Register with event system
        self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register handlers with the event system."""
        self.event_system.register_handler(
            callback=self._handle_event,
            # Monitor all events
        )
    
    def _handle_event(self, event: Event) -> None:
        """Handle an event for monitoring.
        
        Args:
            event: The event to monitor.
        """
        with self._lock:
            # Update counts
            self._event_counts[event.event_type] += 1
            
            # Calculate event size (approximate)
            event_size = len(json.dumps(event.to_dict()))
            self._event_sizes[event.event_type].append(event_size)
            
            # Keep only the last 1000 size samples
            if len(self._event_sizes[event.event_type]) > 1000:
                self._event_sizes[event.event_type] = self._event_sizes[event.event_type][-1000:]
    
    def start(self) -> None:
        """Start the event monitor."""
        if self._running:
            return
        
        with self._lock:
            self._running = True
            self._sampling_thread = Thread(target=self._sampling_loop, daemon=True)
            self._sampling_thread.start()
            logger.info("Event monitor started")
    
    def stop(self) -> None:
        """Stop the event monitor."""
        if not self._running:
            return
        
        with self._lock:
            self._running = False
            if self._sampling_thread:
                self._sampling_thread.join(timeout=2.0)
                self._sampling_thread = None
            logger.info("Event monitor stopped")
    
    def _sampling_loop(self) -> None:
        """Sampling loop for periodic metric collection."""
        while self._running:
            try:
                self._sample_metrics()
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in event monitor sampling loop: {str(e)}")
    
    def _sample_metrics(self) -> None:
        """Sample metrics at the current point in time."""
        with self._lock:
            current_time = time.time()
            
            # Sample event rates
            for event_type, count in self._event_counts.items():
                self._event_rates[event_type].append((current_time, count))
            
            # Prune old rate samples
            max_window = max(self._time_windows)
            for event_type in self._event_rates:
                # Keep samples within the largest time window
                self._event_rates[event_type] = [
                    (t, c) for t, c in self._event_rates[event_type]
                    if current_time - t <= max_window
                ]
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get the current event counts.
        
        Returns:
            Dict[str, int]: A dictionary mapping event types to counts.
        """
        with self._lock:
            return dict(self._event_counts)
    
    def get_event_rates(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get the event rates over a time window.
        
        Args:
            window_seconds: The time window in seconds over which to calculate rates.
                Defaults to 60 seconds (1 minute).
        
        Returns:
            Dict[str, float]: A dictionary mapping event types to rates (events/second).
        """
        with self._lock:
            current_time = time.time()
            rates = {}
            
            for event_type, samples in self._event_rates.items():
                # Filter samples within the window
                window_samples = [
                    (t, c) for t, c in samples
                    if current_time - t <= window_seconds
                ]
                
                if len(window_samples) >= 2:
                    # Calculate rate from oldest to newest sample in window
                    oldest_time, oldest_count = window_samples[0]
                    newest_time, newest_count = window_samples[-1]
                    
                    time_diff = newest_time - oldest_time
                    count_diff = newest_count - oldest_count
                    
                    if time_diff > 0:
                        rates[event_type] = count_diff / time_diff
                    else:
                        rates[event_type] = 0.0
                else:
                    rates[event_type] = 0.0
            
            return rates
    
    def get_event_sizes(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about event sizes.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping event types to size statistics.
        """
        with self._lock:
            size_stats = {}
            
            for event_type, sizes in self._event_sizes.items():
                if sizes:
                    size_stats[event_type] = {
                        "min": min(sizes),
                        "max": max(sizes),
                        "avg": sum(sizes) / len(sizes),
                        "samples": len(sizes)
                    }
                else:
                    size_stats[event_type] = {
                        "min": 0,
                        "max": 0,
                        "avg": 0,
                        "samples": 0
                    }
            
            return size_stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all monitored metrics.
        
        Returns:
            Dict[str, Any]: A dictionary containing all monitored metrics.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "counts": self.get_event_counts(),
            "rates": {
                "1min": self.get_event_rates(60),
                "5min": self.get_event_rates(300),
                "15min": self.get_event_rates(900)
            },
            "sizes": self.get_event_sizes()
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._event_counts.clear()
            self._event_rates.clear()
            self._event_latencies.clear()
            self._event_sizes.clear()
            logger.info("Event monitor metrics reset")


class EventHealthCheck:
    """Health check for the event system.
    
    This class provides functionality to check the health of the event system
    and detect issues like queue backlog, handler errors, or system overload.
    """
    
    def __init__(self, 
                 event_system: EventSystem, 
                 event_monitor: EventMonitor,
                 check_interval: float = 60.0,
                 thresholds: Dict[str, Any] = None):
        """Initialize the event health check.
        
        Args:
            event_system: The event system to check.
            event_monitor: The event monitor to use for metrics.
            check_interval: The interval in seconds at which to perform health checks.
            thresholds: Dictionary of threshold values for health checks.
        """
        self.event_system = event_system
        self.event_monitor = event_monitor
        self.check_interval = check_interval
        
        # Default thresholds
        self.thresholds = thresholds or {
            "queue_size_warning": 100,
            "queue_size_critical": 500,
            "event_rate_warning": 1000,  # events/second
            "event_rate_critical": 5000,  # events/second
            "handler_error_rate_warning": 0.01,  # 1% error rate
            "handler_error_rate_critical": 0.05,  # 5% error rate
        }
        
        # Health check thread
        self._check_thread: Optional[Thread] = None
        self._running = False
        self._lock = Lock()
        
        # Health status
        self._health_status = {
            "status": "OK",
            "issues": [],
            "last_check": None
        }
    
    def start(self) -> None:
        """Start the health check."""
        if self._running:
            return
        
        with self._lock:
            self._running = True
            self._check_thread = Thread(target=self._check_loop, daemon=True)
            self._check_thread.start()
            logger.info("Event health check started")
    
    def stop(self) -> None:
        """Stop the health check."""
        if not self._running:
            return
        
        with self._lock:
            self._running = False
            if self._check_thread:
                self._check_thread.join(timeout=2.0)
                self._check_thread = None
            logger.info("Event health check stopped")
    
    def _check_loop(self) -> None:
        """Health check loop for periodic checks."""
        while self._running:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in event health check loop: {str(e)}")
    
    def _perform_health_check(self) -> None:
        """Perform a health check of the event system."""
        issues = []
        status = "OK"
        
        try:
            # Check if event system is running
            if not self.event_system.is_running():
                issues.append("Event system is not running")
                status = "CRITICAL"
            
            # Check queue size (if accessible)
            if hasattr(self.event_system, "_event_queue") and hasattr(self.event_system._event_queue, "size"):
                queue_size = self.event_system._event_queue.size()
                
                if queue_size >= self.thresholds["queue_size_critical"]:
                    issues.append(f"Event queue size critical: {queue_size} events")
                    status = "CRITICAL"
                elif queue_size >= self.thresholds["queue_size_warning"]:
                    issues.append(f"Event queue size warning: {queue_size} events")
                    if status != "CRITICAL":
                        status = "WARNING"
            
            # Check event rates
            event_rates = self.event_monitor.get_event_rates(60)  # 1-minute rates
            total_rate = sum(event_rates.values())
            
            if total_rate >= self.thresholds["event_rate_critical"]:
                issues.append(f"Event rate critical: {total_rate:.2f} events/second")
                status = "CRITICAL"
            elif total_rate >= self.thresholds["event_rate_warning"]:
                issues.append(f"Event rate warning: {total_rate:.2f} events/second")
                if status != "CRITICAL":
                    status = "WARNING"
            
            # Update health status
            with self._lock:
                self._health_status = {
                    "status": status,
                    "issues": issues,
                    "last_check": datetime.now().isoformat(),
                    "metrics": {
                        "queue_size": queue_size if "queue_size" in locals() else "unknown",
                        "event_rate": total_rate
                    }
                }
            
            # Log health status
            if status != "OK":
                log_level = "error" if status == "CRITICAL" else "warning"
                getattr(logger, log_level)(f"Event system health check: {status}")
                for issue in issues:
                    getattr(logger, log_level)(f"- {issue}")
            else:
                logger.debug("Event system health check: OK")
        
        except Exception as e:
            logger.error(f"Error performing event system health check: {str(e)}")
            
            with self._lock:
                self._health_status = {
                    "status": "UNKNOWN",
                    "issues": [f"Error performing health check: {str(e)}"],
                    "last_check": datetime.now().isoformat()
                }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status.
        
        Returns:
            Dict[str, Any]: The current health status.
        """
        with self._lock:
            return dict(self._health_status)
    
    def is_healthy(self) -> bool:
        """Check if the event system is healthy.
        
        Returns:
            bool: True if the event system is healthy, False otherwise.
        """
        with self._lock:
            return self._health_status["status"] == "OK"


class EventDashboard:
    """Dashboard for visualizing event system metrics.
    
    This class provides functionality to generate reports and visualizations
    of event system metrics for monitoring and analysis.
    """
    
    def __init__(self, event_monitor: EventMonitor, event_health_check: EventHealthCheck):
        """Initialize the event dashboard.
        
        Args:
            event_monitor: The event monitor to use for metrics.
            event_health_check: The health check to use for health status.
        """
        self.event_monitor = event_monitor
        self.event_health_check = event_health_check
    
    def generate_text_report(self) -> str:
        """Generate a text report of event system metrics.
        
        Returns:
            str: A text report of event system metrics.
        """
        # Get metrics
        summary = self.event_monitor.get_summary()
        health = self.event_health_check.get_health_status()
        
        # Format report
        report = []
        report.append("=== Event System Dashboard ===")
        report.append(f"Generated: {summary['timestamp']}")
        report.append(f"Health Status: {health['status']}")
        
        if health['issues']:
            report.append("Health Issues:")
            for issue in health['issues']:
                report.append(f"  - {issue}")
        
        report.append("\nEvent Counts:")
        for event_type, count in sorted(summary['counts'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {event_type}: {count}")
        
        report.append("\nEvent Rates (events/second):")
        report.append("  Event Type      | 1min    | 5min    | 15min   ")
        report.append("  ---------------- | ------- | ------- | -------")
        
        # Get all event types across all time windows
        all_event_types = set()
        for window in summary['rates'].values():
            all_event_types.update(window.keys())
        
        # Sort event types by 1-minute rate
        sorted_types = sorted(
            all_event_types,
            key=lambda et: summary['rates']['1min'].get(et, 0),
            reverse=True
        )
        
        for event_type in sorted_types:
            rate_1min = summary['rates']['1min'].get(event_type, 0)
            rate_5min = summary['rates']['5min'].get(event_type, 0)
            rate_15min = summary['rates']['15min'].get(event_type, 0)
            
            report.append(f"  {event_type:16} | {rate_1min:7.2f} | {rate_5min:7.2f} | {rate_15min:7.2f}")
        
        report.append("\nEvent Sizes (bytes):")
        report.append("  Event Type      | Min     | Max     | Avg     | Samples")
        report.append("  ---------------- | ------- | ------- | ------- | -------")
        
        for event_type, stats in sorted(summary['sizes'].items(), key=lambda x: x[0]):
            if stats['samples'] > 0:
                report.append(
                    f"  {event_type:16} | {stats['min']:7d} | {stats['max']:7d} | "
                    f"{stats['avg']:7.1f} | {stats['samples']:7d}"
                )
        
        return "\n".join(report)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a JSON report of event system metrics.
        
        Returns:
            Dict[str, Any]: A JSON report of event system metrics.
        """
        # Get metrics
        summary = self.event_monitor.get_summary()
        health = self.event_health_check.get_health_status()
        
        # Format report
        return {
            "timestamp": summary["timestamp"],
            "health": health,
            "metrics": {
                "counts": summary["counts"],
                "rates": summary["rates"],
                "sizes": summary["sizes"]
            }
        }


def setup_event_monitoring(event_system: EventSystem) -> Tuple[EventMonitor, EventHealthCheck, EventDashboard]:
    """Set up event monitoring for an event system.
    
    This is a convenience function to set up monitoring, health checks,
    and a dashboard for an event system.
    
    Args:
        event_system: The event system to monitor.
        
    Returns:
        Tuple[EventMonitor, EventHealthCheck, EventDashboard]: The monitoring components.
    """
    # Create monitoring components
    monitor = EventMonitor(event_system)
    health_check = EventHealthCheck(event_system, monitor)
    dashboard = EventDashboard(monitor, health_check)
    
    # Start monitoring
    monitor.start()
    health_check.start()
    
    return monitor, health_check, dashboard