"""Performance Monitoring for Trading Engine.

This module provides tools for monitoring and analyzing the performance of the trading engine,
including execution quality, latency measurements, system health, and trading metrics.
"""

import time
import datetime
import statistics
import threading
import logging
import json
import os
import csv
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from enum import Enum
from collections import deque, defaultdict
import uuid
import psutil

# Configure logger
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be monitored."""
    LATENCY = "latency"  # Time measurements
    COUNTER = "counter"  # Incremental counts
    GAUGE = "gauge"  # Point-in-time values
    HISTOGRAM = "histogram"  # Distribution of values
    EXECUTION_QUALITY = "execution_quality"  # Order execution metrics
    SYSTEM = "system"  # System resource metrics
    CUSTOM = "custom"  # User-defined metrics


class MetricUnit(Enum):
    """Units for metrics."""
    MILLISECONDS = "ms"
    MICROSECONDS = "μs"
    SECONDS = "s"
    COUNT = "count"
    PERCENTAGE = "percent"
    BYTES = "bytes"
    MEGABYTES = "MB"
    CURRENCY = "currency"  # Base currency
    BPS = "bps"  # Basis points
    TICKS = "ticks"
    RATIO = "ratio"
    CUSTOM = "custom"  # User-defined unit


class PerformanceMetric:
    """Represents a single performance metric."""
    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        unit: MetricUnit,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        max_samples: int = 1000,
        aggregation_window: Optional[int] = None,  # in seconds
    ):
        """Initialize a performance metric.
        
        Args:
            name: Name of the metric
            metric_type: Type of metric (from MetricType enum)
            unit: Unit of measurement (from MetricUnit enum)
            description: Description of what the metric measures
            tags: Dictionary of tags for categorizing the metric
            max_samples: Maximum number of samples to keep in memory
            aggregation_window: Time window in seconds for aggregation (None for all samples)
        """
        self.name = name
        self.metric_type = metric_type
        self.unit = unit
        self.description = description
        self.tags = tags or {}
        self.max_samples = max_samples
        self.aggregation_window = aggregation_window
        
        # Storage for metric values with timestamps
        self.values = deque(maxlen=max_samples)
        
        # For COUNTER type, maintain a running total
        self.total = 0 if metric_type == MetricType.COUNTER else None
        
        # For GAUGE type, store the latest value
        self.latest_value = None if metric_type == MetricType.GAUGE else None
        
        # For HISTOGRAM type, maintain min/max
        self.min_value = None if metric_type == MetricType.HISTOGRAM else None
        self.max_value = None if metric_type == MetricType.HISTOGRAM else None
    
    def add_value(self, value: Union[int, float], timestamp: Optional[float] = None) -> None:
        """Add a new value to the metric.
        
        Args:
            value: The metric value to add
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store the value with its timestamp
        self.values.append((timestamp, value))
        
        # Update type-specific tracking
        if self.metric_type == MetricType.COUNTER:
            self.total += value
        
        elif self.metric_type == MetricType.GAUGE:
            self.latest_value = value
        
        elif self.metric_type == MetricType.HISTOGRAM:
            if self.min_value is None or value < self.min_value:
                self.min_value = value
            if self.max_value is None or value > self.max_value:
                self.max_value = value
    
    def get_values(self, window_seconds: Optional[int] = None) -> List[Tuple[float, Union[int, float]]]:
        """Get values within a time window.
        
        Args:
            window_seconds: Time window in seconds (None for all values)
            
        Returns:
            List of (timestamp, value) tuples
        """
        if not self.values:
            return []
        
        if window_seconds is None:
            return list(self.values)
        
        # Filter values within the time window
        now = time.time()
        cutoff = now - window_seconds
        return [(ts, val) for ts, val in self.values if ts >= cutoff]
    
    def get_statistics(self, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Calculate statistics for the metric.
        
        Args:
            window_seconds: Time window in seconds (None for all values)
            
        Returns:
            Dictionary of statistics
        """
        values_in_window = self.get_values(window_seconds)
        if not values_in_window:
            return {"count": 0}
        
        # Extract just the values (not timestamps)
        values = [val for _, val in values_in_window]
        
        stats = {"count": len(values)}
        
        if self.metric_type == MetricType.COUNTER:
            stats["total"] = self.total
            stats["rate_per_second"] = self._calculate_rate(values_in_window)
        
        elif self.metric_type == MetricType.GAUGE:
            stats["latest"] = self.latest_value
            stats["mean"] = statistics.mean(values) if values else None
        
        elif self.metric_type in [MetricType.LATENCY, MetricType.HISTOGRAM, MetricType.EXECUTION_QUALITY]:
            if values:
                stats.update({
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0
                })
        
        return stats
    
    def _calculate_rate(self, values_with_timestamps: List[Tuple[float, Union[int, float]]]) -> float:
        """Calculate the rate of events per second.
        
        Args:
            values_with_timestamps: List of (timestamp, value) tuples
            
        Returns:
            Rate per second
        """
        if len(values_with_timestamps) < 2:
            return 0.0
        
        # Get first and last timestamp
        first_ts = values_with_timestamps[0][0]
        last_ts = values_with_timestamps[-1][0]
        
        # Calculate time span
        time_span = last_ts - first_ts
        if time_span <= 0:
            return 0.0
        
        # Sum all values
        total = sum(val for _, val in values_with_timestamps)
        
        return total / time_span
    
    def _percentile(self, values: List[Union[int, float]], percentile: float) -> float:
        """Calculate a percentile value.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        
        # Calculate index
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = int(k)
        c = int(k) + 1 if k > f else f
        
        if f >= len(sorted_values):
            return float(sorted_values[-1])
        elif c >= len(sorted_values):
            return float(sorted_values[-1])
        else:
            # Interpolate
            d0 = sorted_values[f] * (c - k)
            d1 = sorted_values[c] * (k - f)
            return float(d0 + d1)
    
    def reset(self) -> None:
        """Reset the metric values."""
        self.values.clear()
        
        if self.metric_type == MetricType.COUNTER:
            self.total = 0
        elif self.metric_type == MetricType.GAUGE:
            self.latest_value = None
        elif self.metric_type == MetricType.HISTOGRAM:
            self.min_value = None
            self.max_value = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metric to a dictionary.
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "unit": self.unit.value,
            "description": self.description,
            "tags": self.tags,
            "statistics": self.get_statistics(self.aggregation_window),
            "sample_count": len(self.values)
        }


class LatencyTracker:
    """Utility for tracking operation latency."""
    def __init__(self, performance_monitor: 'PerformanceMonitor'):
        """Initialize the latency tracker.
        
        Args:
            performance_monitor: The performance monitor to record metrics
        """
        self.performance_monitor = performance_monitor
        self.start_times = {}
    
    def start(self, operation_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation being timed
            context: Optional context information
            
        Returns:
            Tracking ID for the operation
        """
        tracking_id = str(uuid.uuid4())
        self.start_times[tracking_id] = {
            "start_time": time.time(),
            "operation": operation_name,
            "context": context or {}
        }
        return tracking_id
    
    def stop(self, tracking_id: str, status: str = "success", additional_tags: Optional[Dict[str, str]] = None) -> float:
        """Stop timing an operation and record the latency.
        
        Args:
            tracking_id: Tracking ID returned by start()
            status: Outcome status (success, error, etc.)
            additional_tags: Additional tags to add to the metric
            
        Returns:
            Latency in milliseconds
        """
        if tracking_id not in self.start_times:
            logger.warning(f"Tracking ID {tracking_id} not found")
            return 0.0
        
        end_time = time.time()
        start_data = self.start_times.pop(tracking_id)
        start_time = start_data["start_time"]
        operation = start_data["operation"]
        context = start_data["context"]
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000.0
        
        # Create tags
        tags = {
            "operation": operation,
            "status": status
        }
        
        # Add context as tags
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                tags[key] = str(value)
        
        # Add additional tags
        if additional_tags:
            tags.update(additional_tags)
        
        # Record the latency
        metric_name = f"latency.{operation}"
        self.performance_monitor.record_metric(
            name=metric_name,
            value=latency_ms,
            metric_type=MetricType.LATENCY,
            unit=MetricUnit.MILLISECONDS,
            tags=tags
        )
        
        return latency_ms
    
    def measure(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for measuring operation latency.
        
        Args:
            operation_name: Name of the operation being timed
            context: Optional context information
            
        Returns:
            Context manager that measures latency
        """
        return LatencyMeasurement(self, operation_name, context)


class LatencyMeasurement:
    """Context manager for measuring operation latency."""
    def __init__(self, tracker: LatencyTracker, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Initialize the latency measurement.
        
        Args:
            tracker: The latency tracker
            operation_name: Name of the operation being timed
            context: Optional context information
        """
        self.tracker = tracker
        self.operation_name = operation_name
        self.context = context
        self.tracking_id = None
        self.status = "success"
        self.additional_tags = {}
    
    def __enter__(self):
        """Start the latency measurement."""
        self.tracking_id = self.tracker.start(self.operation_name, self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the latency measurement."""
        if exc_type is not None:
            self.status = "error"
            self.additional_tags["error_type"] = exc_type.__name__
        
        self.tracker.stop(self.tracking_id, self.status, self.additional_tags)
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the measurement.
        
        Args:
            key: Tag key
            value: Tag value
        """
        self.additional_tags[key] = value
    
    def set_status(self, status: str) -> None:
        """Set the status of the measurement.
        
        Args:
            status: Status value
        """
        self.status = status


class ExecutionQualityMetrics:
    """Calculates and tracks execution quality metrics."""
    def __init__(self, performance_monitor: 'PerformanceMonitor'):
        """Initialize execution quality metrics.
        
        Args:
            performance_monitor: The performance monitor to record metrics
        """
        self.performance_monitor = performance_monitor
    
    def record_fill(self, 
                    order_id: str, 
                    symbol: str, 
                    side: str, 
                    expected_price: float, 
                    fill_price: float, 
                    quantity: float, 
                    order_type: str,
                    strategy: Optional[str] = None,
                    venue: Optional[str] = None) -> Dict[str, float]:
        """Record a fill and calculate execution quality metrics.
        
        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (buy/sell)
            expected_price: Expected execution price
            fill_price: Actual fill price
            quantity: Fill quantity
            order_type: Type of order (market, limit, etc.)
            strategy: Optional execution strategy
            venue: Optional execution venue
            
        Returns:
            Dictionary of calculated metrics
        """
        # Calculate slippage in ticks and basis points
        price_diff = fill_price - expected_price if side.lower() == "buy" else expected_price - fill_price
        slippage_bps = (price_diff / expected_price) * 10000  # Basis points (1% = 100 bps)
        
        # Calculate implementation shortfall (slippage * quantity)
        implementation_shortfall = price_diff * quantity
        
        # Create tags
        tags = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type
        }
        
        if strategy:
            tags["strategy"] = strategy
        if venue:
            tags["venue"] = venue
        
        # Record metrics
        self.performance_monitor.record_metric(
            name="execution.slippage_bps",
            value=slippage_bps,
            metric_type=MetricType.EXECUTION_QUALITY,
            unit=MetricUnit.BPS,
            tags=tags
        )
        
        self.performance_monitor.record_metric(
            name="execution.implementation_shortfall",
            value=implementation_shortfall,
            metric_type=MetricType.EXECUTION_QUALITY,
            unit=MetricUnit.CURRENCY,
            tags=tags
        )
        
        # Record price improvement (negative slippage is good)
        if slippage_bps < 0:
            self.performance_monitor.record_metric(
                name="execution.price_improvement",
                value=abs(slippage_bps),
                metric_type=MetricType.EXECUTION_QUALITY,
                unit=MetricUnit.BPS,
                tags=tags
            )
        
        # Return the calculated metrics
        return {
            "slippage_bps": slippage_bps,
            "implementation_shortfall": implementation_shortfall,
            "price_diff": price_diff
        }
    
    def record_order_lifecycle(self, 
                              order_id: str, 
                              symbol: str, 
                              creation_time: float, 
                              submission_time: float, 
                              first_fill_time: Optional[float] = None,
                              completion_time: Optional[float] = None,
                              cancellation_time: Optional[float] = None,
                              tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Record order lifecycle timing metrics.
        
        Args:
            order_id: Order identifier
            symbol: Trading symbol
            creation_time: Time when order was created
            submission_time: Time when order was submitted to broker
            first_fill_time: Time of first fill (None if no fills)
            completion_time: Time when order was fully filled (None if not complete)
            cancellation_time: Time when order was cancelled (None if not cancelled)
            tags: Additional tags
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Create base tags
        base_tags = {
            "order_id": order_id,
            "symbol": symbol
        }
        
        if tags:
            base_tags.update(tags)
        
        # Calculate submission latency (internal processing time)
        submission_latency = (submission_time - creation_time) * 1000  # ms
        metrics["submission_latency_ms"] = submission_latency
        
        self.performance_monitor.record_metric(
            name="order.submission_latency",
            value=submission_latency,
            metric_type=MetricType.LATENCY,
            unit=MetricUnit.MILLISECONDS,
            tags=base_tags
        )
        
        # Calculate time to first fill if available
        if first_fill_time is not None:
            time_to_first_fill = (first_fill_time - submission_time) * 1000  # ms
            metrics["time_to_first_fill_ms"] = time_to_first_fill
            
            self.performance_monitor.record_metric(
                name="order.time_to_first_fill",
                value=time_to_first_fill,
                metric_type=MetricType.LATENCY,
                unit=MetricUnit.MILLISECONDS,
                tags=base_tags
            )
        
        # Calculate total execution time if order completed
        if completion_time is not None:
            total_execution_time = (completion_time - submission_time) * 1000  # ms
            metrics["total_execution_time_ms"] = total_execution_time
            
            self.performance_monitor.record_metric(
                name="order.total_execution_time",
                value=total_execution_time,
                metric_type=MetricType.LATENCY,
                unit=MetricUnit.MILLISECONDS,
                tags=base_tags
            )
        
        # Calculate time to cancellation if order was cancelled
        if cancellation_time is not None:
            time_to_cancellation = (cancellation_time - submission_time) * 1000  # ms
            metrics["time_to_cancellation_ms"] = time_to_cancellation
            
            self.performance_monitor.record_metric(
                name="order.time_to_cancellation",
                value=time_to_cancellation,
                metric_type=MetricType.LATENCY,
                unit=MetricUnit.MILLISECONDS,
                tags=base_tags
            )
        
        return metrics


class SystemMetricsCollector:
    """Collects system-level metrics like CPU, memory, and disk usage."""
    def __init__(self, performance_monitor: 'PerformanceMonitor', collection_interval: int = 60):
        """Initialize the system metrics collector.
        
        Args:
            performance_monitor: The performance monitor to record metrics
            collection_interval: Interval in seconds between collections
        """
        self.performance_monitor = performance_monitor
        self.collection_interval = collection_interval
        self.running = False
        self.collection_thread = None
        self.process = psutil.Process()
    
    def start(self) -> None:
        """Start collecting system metrics."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("System metrics collection started")
    
    def stop(self) -> None:
        """Stop collecting system metrics."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
            self.collection_thread = None
        logger.info("System metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop that runs in a separate thread."""
        while self.running:
            try:
                self.collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            # Sleep until next collection
            time.sleep(self.collection_interval)
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect and record system metrics.
        
        Returns:
            Dictionary of collected metrics
        """
        metrics = {}
        
        # CPU usage (system-wide)
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["cpu_percent"] = cpu_percent
        
        self.performance_monitor.record_metric(
            name="system.cpu_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.PERCENTAGE
        )
        
        # Process-specific CPU usage
        process_cpu_percent = self.process.cpu_percent(interval=1) / psutil.cpu_count()
        metrics["process_cpu_percent"] = process_cpu_percent
        
        self.performance_monitor.record_metric(
            name="system.process_cpu_percent",
            value=process_cpu_percent,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.PERCENTAGE
        )
        
        # Memory usage (system-wide)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics["memory_percent"] = memory_percent
        
        self.performance_monitor.record_metric(
            name="system.memory_percent",
            value=memory_percent,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.PERCENTAGE
        )
        
        # Process-specific memory usage
        process_memory_mb = self.process.memory_info().rss / (1024 * 1024)  # MB
        metrics["process_memory_mb"] = process_memory_mb
        
        self.performance_monitor.record_metric(
            name="system.process_memory_mb",
            value=process_memory_mb,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.MEGABYTES
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        metrics["disk_percent"] = disk_percent
        
        self.performance_monitor.record_metric(
            name="system.disk_percent",
            value=disk_percent,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.PERCENTAGE
        )
        
        # Network I/O counters
        try:
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            
            # Store as gauges (point-in-time values)
            self.performance_monitor.record_metric(
                name="system.network_bytes_sent",
                value=bytes_sent,
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.BYTES
            )
            
            self.performance_monitor.record_metric(
                name="system.network_bytes_recv",
                value=bytes_recv,
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.BYTES
            )
            
            metrics["network_bytes_sent"] = bytes_sent
            metrics["network_bytes_recv"] = bytes_recv
        except Exception as e:
            logger.warning(f"Failed to collect network metrics: {e}")
        
        # Thread count
        thread_count = threading.active_count()
        metrics["thread_count"] = thread_count
        
        self.performance_monitor.record_metric(
            name="system.thread_count",
            value=thread_count,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT
        )
        
        return metrics


class PerformanceMonitor:
    """Central manager for performance monitoring."""
    def __init__(self, 
                 metrics_dir: Optional[str] = None,
                 enable_system_metrics: bool = True,
                 system_metrics_interval: int = 60,
                 auto_flush_interval: Optional[int] = 300):
        """Initialize the performance monitor.
        
        Args:
            metrics_dir: Directory to store metrics data files
            enable_system_metrics: Whether to collect system metrics
            system_metrics_interval: Interval in seconds for system metrics collection
            auto_flush_interval: Interval in seconds for auto-flushing metrics to disk (None to disable)
        """
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metrics_dir = metrics_dir
        
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        
        # Create helper components
        self.latency_tracker = LatencyTracker(self)
        self.execution_quality = ExecutionQualityMetrics(self)
        
        # System metrics collector
        self.system_metrics_collector = None
        if enable_system_metrics:
            self.system_metrics_collector = SystemMetricsCollector(
                self, collection_interval=system_metrics_interval)
        
        # Auto-flush mechanism
        self.auto_flush_interval = auto_flush_interval
        self.auto_flush_thread = None
        self.running = False
        
        # Callback registry for metric thresholds
        self.threshold_callbacks: Dict[str, List[Tuple[float, Callable]]] = defaultdict(list)
        
        # Start system metrics collection and auto-flush if enabled
        if self.system_metrics_collector:
            self.system_metrics_collector.start()
        
        if auto_flush_interval:
            self.start_auto_flush()
    
    def start_auto_flush(self) -> None:
        """Start the auto-flush thread."""
        if self.auto_flush_thread is not None:
            return
        
        self.running = True
        self.auto_flush_thread = threading.Thread(target=self._auto_flush_loop, daemon=True)
        self.auto_flush_thread.start()
    
    def stop_auto_flush(self) -> None:
        """Stop the auto-flush thread."""
        self.running = False
        if self.auto_flush_thread:
            self.auto_flush_thread.join(timeout=2.0)
            self.auto_flush_thread = None
    
    def _auto_flush_loop(self) -> None:
        """Auto-flush loop that runs in a separate thread."""
        while self.running:
            try:
                time.sleep(self.auto_flush_interval)
                if self.running:  # Check again after sleep
                    self.flush_metrics_to_disk()
            except Exception as e:
                logger.error(f"Error in auto-flush: {e}")
    
    def record_metric(self, 
                      name: str, 
                      value: Union[int, float], 
                      metric_type: Optional[MetricType] = None,
                      unit: Optional[MetricUnit] = None,
                      description: str = "",
                      tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric (required for new metrics)
            unit: Unit of measurement (required for new metrics)
            description: Description of the metric
            tags: Tags for the metric
        """
        # Create the metric if it doesn't exist
        if name not in self.metrics:
            if metric_type is None or unit is None:
                raise ValueError(f"Metric type and unit are required for new metric '{name}'")
            
            self.metrics[name] = PerformanceMetric(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description,
                tags=tags
            )
        
        # Add the value
        self.metrics[name].add_value(value)
        
        # Check thresholds
        self._check_thresholds(name, value, tags)
    
    def _check_thresholds(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Check if a metric value exceeds any registered thresholds.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Tags for the metric
        """
        if name not in self.threshold_callbacks:
            return
        
        for threshold, callback in self.threshold_callbacks[name]:
            if value >= threshold:
                try:
                    callback(name, value, threshold, tags)
                except Exception as e:
                    logger.error(f"Error in threshold callback for {name}: {e}")
    
    def register_threshold_callback(self, 
                                    metric_name: str, 
                                    threshold: float, 
                                    callback: Callable[[str, float, float, Optional[Dict[str, str]]], None]) -> None:
        """Register a callback for when a metric exceeds a threshold.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold: Threshold value
            callback: Function to call when threshold is exceeded
        """
        self.threshold_callbacks[metric_name].append((threshold, callback))
    
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            PerformanceMetric or None if not found
        """
        return self.metrics.get(name)
    
    def get_metrics_by_prefix(self, prefix: str) -> Dict[str, PerformanceMetric]:
        """Get metrics that start with a prefix.
        
        Args:
            prefix: Metric name prefix
            
        Returns:
            Dictionary of matching metrics
        """
        return {name: metric for name, metric in self.metrics.items() if name.startswith(prefix)}
    
    def get_metrics_by_type(self, metric_type: MetricType) -> Dict[str, PerformanceMetric]:
        """Get metrics of a specific type.
        
        Args:
            metric_type: Metric type to filter by
            
        Returns:
            Dictionary of matching metrics
        """
        return {name: metric for name, metric in self.metrics.items() if metric.metric_type == metric_type}
    
    def get_metrics_by_tag(self, tag_key: str, tag_value: Optional[str] = None) -> Dict[str, PerformanceMetric]:
        """Get metrics with a specific tag.
        
        Args:
            tag_key: Tag key to filter by
            tag_value: Optional tag value to filter by
            
        Returns:
            Dictionary of matching metrics
        """
        result = {}
        for name, metric in self.metrics.items():
            if tag_key in metric.tags:
                if tag_value is None or metric.tags[tag_key] == tag_value:
                    result[name] = metric
        return result
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionaries.
        
        Returns:
            Dictionary of metric dictionaries
        """
        return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def reset_metrics(self, prefix: Optional[str] = None) -> None:
        """Reset metrics, optionally filtering by prefix.
        
        Args:
            prefix: Optional prefix to filter metrics to reset
        """
        if prefix is None:
            # Reset all metrics
            for metric in self.metrics.values():
                metric.reset()
        else:
            # Reset metrics with matching prefix
            for name, metric in self.metrics.items():
                if name.startswith(prefix):
                    metric.reset()
    
    def flush_metrics_to_disk(self) -> Optional[str]:
        """Flush metrics to disk.
        
        Returns:
            Path to the metrics file or None if metrics_dir is not set
        """
        if not self.metrics_dir:
            return None
        
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
        
        # Get all metrics
        metrics_data = self.get_all_metrics()
        
        # Add metadata
        metrics_data["_metadata"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metric_count": len(metrics_data) - 1  # Subtract 1 for _metadata
        }
        
        # Write to file
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics flushed to {metrics_file}")
        return metrics_file
    
    def export_csv(self, metric_names: Optional[List[str]] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export metrics to CSV files.
        
        Args:
            metric_names: List of metric names to export (all if None)
            output_dir: Directory to write CSV files (metrics_dir if None)
            
        Returns:
            Dictionary mapping metric names to CSV file paths
        """
        if output_dir is None:
            if self.metrics_dir is None:
                raise ValueError("No output directory specified and metrics_dir is not set")
            output_dir = self.metrics_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which metrics to export
        if metric_names is None:
            metrics_to_export = list(self.metrics.keys())
        else:
            metrics_to_export = [name for name in metric_names if name in self.metrics]
        
        result = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name in metrics_to_export:
            metric = self.metrics[name]
            values = metric.get_values()
            
            if not values:
                continue
            
            # Create a safe filename
            safe_name = name.replace('.', '_').replace('/', '_').replace('\\', '_')
            csv_file = os.path.join(output_dir, f"{safe_name}_{timestamp}.csv")
            
            # Write to CSV
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ["timestamp", "value"]
                writer.writerow(header)
                
                # Write data
                for ts, val in values:
                    # Convert timestamp to ISO format
                    iso_time = datetime.datetime.fromtimestamp(ts).isoformat()
                    writer.writerow([iso_time, val])
            
            result[name] = csv_file
        
        return result
    
    def generate_report(self, 
                        output_file: Optional[str] = None, 
                        include_metrics: Optional[List[str]] = None,
                        exclude_metrics: Optional[List[str]] = None,
                        window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Generate a performance report.
        
        Args:
            output_file: File to write the report to (optional)
            include_metrics: List of metrics to include (all if None)
            exclude_metrics: List of metrics to exclude
            window_seconds: Time window in seconds for statistics
            
        Returns:
            Report data as a dictionary
        """
        # Filter metrics
        metrics_to_include = set(self.metrics.keys())
        
        if include_metrics:
            metrics_to_include = metrics_to_include.intersection(include_metrics)
        
        if exclude_metrics:
            metrics_to_include = metrics_to_include.difference(exclude_metrics)
        
        # Generate report data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "window_seconds": window_seconds,
            "metrics": {}
        }
        
        for name in sorted(metrics_to_include):
            metric = self.metrics[name]
            stats = metric.get_statistics(window_seconds)
            
            report["metrics"][name] = {
                "type": metric.metric_type.value,
                "unit": metric.unit.value,
                "statistics": stats,
                "tags": metric.tags
            }
        
        # Group metrics by type
        report["by_type"] = {}
        for metric_type in MetricType:
            type_metrics = self.get_metrics_by_type(metric_type)
            type_names = [name for name in type_metrics.keys() if name in metrics_to_include]
            
            if type_names:
                report["by_type"][metric_type.value] = type_names
        
        # Add system information
        try:
            report["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
                "platform": os.name,
                "hostname": os.environ.get("COMPUTERNAME", "unknown")
            }
        except Exception as e:
            logger.warning(f"Failed to collect system information: {e}")
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def shutdown(self) -> None:
        """Shutdown the performance monitor and its components."""
        # Stop system metrics collection
        if self.system_metrics_collector:
            self.system_metrics_collector.stop()
        
        # Stop auto-flush
        self.stop_auto_flush()
        
        # Final flush to disk
        if self.metrics_dir:
            self.flush_metrics_to_disk()


# Factory function
def create_performance_monitor(metrics_dir: Optional[str] = None,
                               enable_system_metrics: bool = True,
                               system_metrics_interval: int = 60,
                               auto_flush_interval: Optional[int] = 300) -> PerformanceMonitor:
    """Create a performance monitor.
    
    Args:
        metrics_dir: Directory to store metrics data files
        enable_system_metrics: Whether to collect system metrics
        system_metrics_interval: Interval in seconds for system metrics collection
        auto_flush_interval: Interval in seconds for auto-flushing metrics to disk
        
    Returns:
        Configured PerformanceMonitor
    """
    return PerformanceMonitor(
        metrics_dir=metrics_dir,
        enable_system_metrics=enable_system_metrics,
        system_metrics_interval=system_metrics_interval,
        auto_flush_interval=auto_flush_interval
    )


# Utility functions

def calculate_execution_statistics(performance_monitor: PerformanceMonitor, 
                                  window_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Calculate execution quality statistics.
    
    Args:
        performance_monitor: Performance monitor instance
        window_seconds: Time window in seconds
        
    Returns:
        Dictionary of execution statistics
    """
    # Get execution quality metrics
    slippage_metric = performance_monitor.get_metric("execution.slippage_bps")
    shortfall_metric = performance_monitor.get_metric("execution.implementation_shortfall")
    price_improvement_metric = performance_monitor.get_metric("execution.price_improvement")
    
    result = {}
    
    # Calculate slippage statistics
    if slippage_metric:
        result["slippage_bps"] = slippage_metric.get_statistics(window_seconds)
    
    # Calculate implementation shortfall statistics
    if shortfall_metric:
        result["implementation_shortfall"] = shortfall_metric.get_statistics(window_seconds)
    
    # Calculate price improvement statistics
    if price_improvement_metric:
        result["price_improvement"] = price_improvement_metric.get_statistics(window_seconds)
    
    # Calculate overall execution quality
    if slippage_metric and slippage_metric.values:
        slippage_values = [val for _, val in slippage_metric.get_values(window_seconds)]
        
        if slippage_values:
            # Count positive and negative slippage instances
            positive_slippage = sum(1 for val in slippage_values if val > 0)
            negative_slippage = sum(1 for val in slippage_values if val < 0)
            zero_slippage = sum(1 for val in slippage_values if val == 0)
            
            total_count = len(slippage_values)
            
            result["execution_quality"] = {
                "total_executions": total_count,
                "positive_slippage_count": positive_slippage,
                "negative_slippage_count": negative_slippage,
                "zero_slippage_count": zero_slippage,
                "positive_slippage_percent": (positive_slippage / total_count) * 100 if total_count > 0 else 0,
                "negative_slippage_percent": (negative_slippage / total_count) * 100 if total_count > 0 else 0,
                "zero_slippage_percent": (zero_slippage / total_count) * 100 if total_count > 0 else 0
            }
    
    return result


def calculate_latency_statistics(performance_monitor: PerformanceMonitor,
                               operation_prefix: str = "latency.",
                               window_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Calculate latency statistics for operations.
    
    Args:
        performance_monitor: Performance monitor instance
        operation_prefix: Prefix for latency metrics
        window_seconds: Time window in seconds
        
    Returns:
        Dictionary of latency statistics by operation
    """
    # Get all latency metrics
    latency_metrics = performance_monitor.get_metrics_by_prefix(operation_prefix)
    
    result = {}
    
    for name, metric in latency_metrics.items():
        # Extract operation name from metric name
        operation = name[len(operation_prefix):] if name.startswith(operation_prefix) else name
        
        # Get statistics
        stats = metric.get_statistics(window_seconds)
        
        if stats["count"] > 0:
            result[operation] = stats
    
    # Calculate overall latency statistics
    all_latencies = []
    for metric in latency_metrics.values():
        all_latencies.extend([val for _, val in metric.get_values(window_seconds)])
    
    if all_latencies:
        result["overall"] = {
            "count": len(all_latencies),
            "min": min(all_latencies),
            "max": max(all_latencies),
            "mean": statistics.mean(all_latencies),
            "median": statistics.median(all_latencies),
            "p95": metric._percentile(all_latencies, 95),
            "p99": metric._percentile(all_latencies, 99),
            "stddev": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0
        }
    
    return result


def calculate_system_health(performance_monitor: PerformanceMonitor,
                          window_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Calculate system health metrics.
    
    Args:
        performance_monitor: Performance monitor instance
        window_seconds: Time window in seconds
        
    Returns:
        Dictionary of system health metrics
    """
    # Get system metrics
    cpu_metric = performance_monitor.get_metric("system.cpu_percent")
    process_cpu_metric = performance_monitor.get_metric("system.process_cpu_percent")
    memory_metric = performance_monitor.get_metric("system.memory_percent")
    process_memory_metric = performance_monitor.get_metric("system.process_memory_mb")
    
    result = {}
    
    # CPU usage
    if cpu_metric:
        cpu_values = [val for _, val in cpu_metric.get_values(window_seconds)]
        if cpu_values:
            result["cpu"] = {
                "current": cpu_values[-1],
                "min": min(cpu_values),
                "max": max(cpu_values),
                "mean": statistics.mean(cpu_values)
            }
    
    # Process CPU usage
    if process_cpu_metric:
        process_cpu_values = [val for _, val in process_cpu_metric.get_values(window_seconds)]
        if process_cpu_values:
            result["process_cpu"] = {
                "current": process_cpu_values[-1],
                "min": min(process_cpu_values),
                "max": max(process_cpu_values),
                "mean": statistics.mean(process_cpu_values)
            }
    
    # Memory usage
    if memory_metric:
        memory_values = [val for _, val in memory_metric.get_values(window_seconds)]
        if memory_values:
            result["memory"] = {
                "current": memory_values[-1],
                "min": min(memory_values),
                "max": max(memory_values),
                "mean": statistics.mean(memory_values)
            }
    
    # Process memory usage
    if process_memory_metric:
        process_memory_values = [val for _, val in process_memory_metric.get_values(window_seconds)]
        if process_memory_values:
            result["process_memory"] = {
                "current": process_memory_values[-1],
                "min": min(process_memory_values),
                "max": max(process_memory_values),
                "mean": statistics.mean(process_memory_values)
            }
    
    # Calculate overall health score (0-100)
    health_score = 100
    
    # Reduce score based on CPU usage
    if "cpu" in result:
        cpu_current = result["cpu"]["current"]
        if cpu_current > 90:
            health_score -= 30
        elif cpu_current > 75:
            health_score -= 15
        elif cpu_current > 50:
            health_score -= 5
    
    # Reduce score based on memory usage
    if "memory" in result:
        memory_current = result["memory"]["current"]
        if memory_current > 90:
            health_score -= 30
        elif memory_current > 75:
            health_score -= 15
        elif memory_current > 50:
            health_score -= 5
    
    result["health_score"] = max(0, health_score)
    
    # Add health status
    if health_score >= 90:
        result["health_status"] = "excellent"
    elif health_score >= 75:
        result["health_status"] = "good"
    elif health_score >= 50:
        result["health_status"] = "fair"
    elif health_score >= 25:
        result["health_status"] = "poor"
    else:
        result["health_status"] = "critical"
    
    return result