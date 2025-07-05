"""Tests for the monitoring module.

This module contains tests for the monitoring functionality, including:
- Health checks
- Metrics collection and registration
- System metrics collection
- Alerts and alert handlers
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from src.infrastructure.monitoring import (
    HealthStatus, MetricType, AlertLevel,
    HealthCheck, SystemResourceCheck, NetworkConnectivityCheck, DatabaseConnectivityCheck,
    HealthCheckRegistry, Metric, Counter, Gauge, Histogram, Timer,
    timed, MetricRegistry, SystemMetricsCollector,
    Alert, AlertHandler, LoggingAlertHandler, AlertManager,
    initialize_monitoring
)


class TestHealthChecks(unittest.TestCase):
    """Tests for health check functionality."""

    def test_base_health_check(self):
        """Test the base HealthCheck class."""
        # Create a simple health check implementation
        class TestHealthCheck(HealthCheck):
            def __init__(self):
                super().__init__("test_check", "Test Check")

            def check_health(self):
                return HealthStatus.HEALTHY, "Test check passed", {"test_data": "value"}

        # Run the health check
        check = TestHealthCheck()
        status, message, data = check.run()

        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertEqual(message, "Test check passed")
        self.assertEqual(data, {"test_data": "value"})

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_resource_check(self, mock_disk_usage, mock_virtual_memory, mock_cpu_percent):
        """Test the SystemResourceCheck class."""
        # Mock the system resource functions
        mock_cpu_percent.return_value = 50.0
        mock_virtual_memory.return_value = MagicMock(percent=60.0)
        mock_disk_usage.return_value = MagicMock(percent=70.0)

        # Create and run the check
        check = SystemResourceCheck(
            cpu_threshold=80.0,
            memory_threshold=80.0,
            disk_threshold=80.0
        )
        status, message, data = check.run()

        # All resources are below thresholds, so should be healthy
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertIn("System resources are within acceptable limits", message)
        self.assertEqual(data["cpu_usage"], 50.0)
        self.assertEqual(data["memory_usage"], 60.0)
        self.assertEqual(data["disk_usage"], 70.0)

        # Now test with CPU above threshold
        mock_cpu_percent.return_value = 90.0
        status, message, data = check.run()

        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertIn("CPU usage is above threshold", message)

    @patch("socket.create_connection")
    def test_network_connectivity_check(self, mock_create_connection):
        """Test the NetworkConnectivityCheck class."""
        # Mock successful connection
        mock_create_connection.return_value = MagicMock()

        # Create and run the check
        check = NetworkConnectivityCheck("example.com", 80)
        status, message, data = check.run()

        # Connection successful, should be healthy
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertIn("Successfully connected", message)

        # Mock connection failure
        mock_create_connection.side_effect = Exception("Connection failed")
        status, message, data = check.run()

        # Connection failed, should be unhealthy
        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertIn("Failed to connect", message)

    @patch("pymongo.MongoClient")
    def test_database_connectivity_check(self, mock_mongo_client):
        """Test the DatabaseConnectivityCheck class."""
        # Mock successful database connection
        mock_client = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_mongo_client.return_value = mock_client

        # Create and run the check
        check = DatabaseConnectivityCheck("mongodb://localhost:27017")
        status, message, data = check.run()

        # Connection successful, should be healthy
        self.assertEqual(status, HealthStatus.HEALTHY)
        self.assertIn("Successfully connected", message)

        # Mock connection failure
        mock_client.admin.command.side_effect = Exception("Connection failed")
        status, message, data = check.run()

        # Connection failed, should be unhealthy
        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertIn("Failed to connect", message)


class TestHealthCheckRegistry(unittest.TestCase):
    """Tests for the HealthCheckRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        HealthCheckRegistry._instance = None
        self.registry = HealthCheckRegistry()

    def test_register_and_get_health_check(self):
        """Test registering and retrieving health checks."""
        # Create a mock health check
        mock_check = MagicMock()
        mock_check.name = "test_check"

        # Register the check
        self.registry.register(mock_check)

        # Retrieve the check
        retrieved_check = self.registry.get("test_check")
        self.assertEqual(retrieved_check, mock_check)

    def test_run_all_health_checks(self):
        """Test running all registered health checks."""
        # Create mock health checks
        mock_check1 = MagicMock()
        mock_check1.name = "check1"
        mock_check1.run.return_value = (HealthStatus.HEALTHY, "Check 1 passed", {})

        mock_check2 = MagicMock()
        mock_check2.name = "check2"
        mock_check2.run.return_value = (HealthStatus.UNHEALTHY, "Check 2 failed", {})

        # Register the checks
        self.registry.register(mock_check1)
        self.registry.register(mock_check2)

        # Run all checks
        results = self.registry.run_all()

        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results["check1"]["status"], HealthStatus.HEALTHY)
        self.assertEqual(results["check2"]["status"], HealthStatus.UNHEALTHY)


class TestMetrics(unittest.TestCase):
    """Tests for metric classes."""

    def test_counter(self):
        """Test the Counter metric class."""
        counter = Counter("test_counter", "Test counter")

        # Initial value should be 0
        self.assertEqual(counter.value, 0)

        # Increment by 1
        counter.increment()
        self.assertEqual(counter.value, 1)

        # Increment by custom value
        counter.increment(5)
        self.assertEqual(counter.value, 6)

        # Reset counter
        counter.reset()
        self.assertEqual(counter.value, 0)

    def test_gauge(self):
        """Test the Gauge metric class."""
        gauge = Gauge("test_gauge", "Test gauge")

        # Initial value should be 0
        self.assertEqual(gauge.value, 0)

        # Set value
        gauge.set(42)
        self.assertEqual(gauge.value, 42)

        # Increment value
        gauge.increment(8)
        self.assertEqual(gauge.value, 50)

        # Decrement value
        gauge.decrement(10)
        self.assertEqual(gauge.value, 40)

    def test_histogram(self):
        """Test the Histogram metric class."""
        histogram = Histogram("test_histogram", "Test histogram")

        # Add values
        histogram.update(5)
        histogram.update(10)
        histogram.update(15)

        # Check statistics
        self.assertEqual(histogram.count, 3)
        self.assertEqual(histogram.sum, 30)
        self.assertEqual(histogram.min, 5)
        self.assertEqual(histogram.max, 15)
        self.assertEqual(histogram.mean, 10)

        # Reset histogram
        histogram.reset()
        self.assertEqual(histogram.count, 0)
        self.assertEqual(histogram.sum, 0)

    def test_timer(self):
        """Test the Timer metric class."""
        timer = Timer("test_timer", "Test timer")

        # Start and stop timer
        timer.start()
        time.sleep(0.01)  # Sleep for a short time
        timer.stop()

        # Check that elapsed time is positive
        self.assertGreater(timer.value, 0)

        # Test with context manager
        with timer:
            time.sleep(0.01)  # Sleep for a short time

        # Check that elapsed time is updated
        self.assertGreater(timer.value, 0)

    def test_timed_decorator(self):
        """Test the timed decorator."""
        # Create a mock registry
        mock_registry = MagicMock()
        mock_timer = MagicMock()
        mock_registry.get_or_create_metric.return_value = mock_timer

        # Apply the decorator to a function
        @timed("test_function_timer", registry=mock_registry)
        def test_function():
            time.sleep(0.01)  # Sleep for a short time
            return "result"

        # Call the function
        result = test_function()

        # Check that the function returned the correct result
        self.assertEqual(result, "result")

        # Check that the timer was used
        mock_registry.get_or_create_metric.assert_called_with(
            "test_function_timer", MetricType.TIMER, "Execution time for test_function"
        )
        mock_timer.__enter__.assert_called_once()
        mock_timer.__exit__.assert_called_once()


class TestMetricRegistry(unittest.TestCase):
    """Tests for the MetricRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        MetricRegistry._instance = None
        self.registry = MetricRegistry()

    def test_register_and_get_metric(self):
        """Test registering and retrieving metrics."""
        # Create a metric
        counter = Counter("test_counter", "Test counter")

        # Register the metric
        self.registry.register(counter)

        # Retrieve the metric
        retrieved_metric = self.registry.get("test_counter")
        self.assertEqual(retrieved_metric, counter)

    def test_get_or_create_metric(self):
        """Test getting or creating metrics."""
        # Get or create a counter
        counter = self.registry.get_or_create_metric(
            "test_counter", MetricType.COUNTER, "Test counter"
        )
        self.assertIsInstance(counter, Counter)

        # Get the same counter again
        same_counter = self.registry.get_or_create_metric(
            "test_counter", MetricType.COUNTER, "Test counter"
        )
        self.assertEqual(counter, same_counter)

        # Get or create a gauge
        gauge = self.registry.get_or_create_metric(
            "test_gauge", MetricType.GAUGE, "Test gauge"
        )
        self.assertIsInstance(gauge, Gauge)

        # Get or create a histogram
        histogram = self.registry.get_or_create_metric(
            "test_histogram", MetricType.HISTOGRAM, "Test histogram"
        )
        self.assertIsInstance(histogram, Histogram)

        # Get or create a timer
        timer = self.registry.get_or_create_metric(
            "test_timer", MetricType.TIMER, "Test timer"
        )
        self.assertIsInstance(timer, Timer)

    def test_get_all_metrics(self):
        """Test getting all registered metrics."""
        # Register some metrics
        counter = Counter("test_counter", "Test counter")
        gauge = Gauge("test_gauge", "Test gauge")

        self.registry.register(counter)
        self.registry.register(gauge)

        # Get all metrics
        metrics = self.registry.get_all()

        # Check that all registered metrics are returned
        self.assertEqual(len(metrics), 2)
        self.assertIn("test_counter", metrics)
        self.assertIn("test_gauge", metrics)


class TestSystemMetricsCollector(unittest.TestCase):
    """Tests for the SystemMetricsCollector class."""

    @patch("threading.Thread")
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_metrics_collector(self, mock_disk_usage, mock_virtual_memory, 
                                     mock_cpu_percent, mock_thread):
        """Test the SystemMetricsCollector class."""
        # Mock the system resource functions
        mock_cpu_percent.return_value = 50.0
        mock_virtual_memory.return_value = MagicMock(percent=60.0)
        mock_disk_usage.return_value = MagicMock(percent=70.0)

        # Create a mock registry
        mock_registry = MagicMock()
        mock_cpu_gauge = MagicMock()
        mock_memory_gauge = MagicMock()
        mock_disk_gauge = MagicMock()

        mock_registry.get_or_create_metric.side_effect = [
            mock_cpu_gauge, mock_memory_gauge, mock_disk_gauge
        ]

        # Create the collector
        collector = SystemMetricsCollector(registry=mock_registry, interval=1)

        # Call the collect method directly
        collector.collect()

        # Check that metrics were updated
        mock_cpu_gauge.set.assert_called_with(50.0)
        mock_memory_gauge.set.assert_called_with(60.0)
        mock_disk_gauge.set.assert_called_with(70.0)

        # Check that the thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.daemon.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


class TestAlerts(unittest.TestCase):
    """Tests for alert functionality."""

    def test_alert_creation(self):
        """Test creating an Alert."""
        alert = Alert(
            name="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert message",
            source="test_source",
            data={"key": "value"}
        )

        self.assertEqual(alert.name, "test_alert")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.source, "test_source")
        self.assertEqual(alert.data, {"key": "value"})
        self.assertIsNotNone(alert.timestamp)

    def test_alert_handler(self):
        """Test the AlertHandler base class."""
        # Create a simple alert handler implementation
        class TestAlertHandler(AlertHandler):
            def __init__(self):
                self.alerts = []

            def handle(self, alert):
                self.alerts.append(alert)

        # Create an alert and handler
        alert = Alert("test_alert", AlertLevel.WARNING, "Test message")
        handler = TestAlertHandler()

        # Handle the alert
        handler.handle(alert)

        # Check that the alert was handled
        self.assertEqual(len(handler.alerts), 1)
        self.assertEqual(handler.alerts[0], alert)

    @patch("logging.Logger.warning")
    @patch("logging.Logger.error")
    @patch("logging.Logger.critical")
    def test_logging_alert_handler(self, mock_critical, mock_error, mock_warning):
        """Test the LoggingAlertHandler class."""
        # Create a logger and handler
        logger = logging.getLogger("test_logger")
        handler = LoggingAlertHandler(logger)

        # Create and handle alerts at different levels
        warning_alert = Alert("warning_alert", AlertLevel.WARNING, "Warning message")
        handler.handle(warning_alert)
        mock_warning.assert_called_once()

        error_alert = Alert("error_alert", AlertLevel.ERROR, "Error message")
        handler.handle(error_alert)
        mock_error.assert_called_once()

        critical_alert = Alert("critical_alert", AlertLevel.CRITICAL, "Critical message")
        handler.handle(critical_alert)
        mock_critical.assert_called_once()


class TestAlertManager(unittest.TestCase):
    """Tests for the AlertManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        AlertManager._instance = None
        self.manager = AlertManager()

    def test_register_and_get_handler(self):
        """Test registering and retrieving alert handlers."""
        # Create a mock handler
        mock_handler = MagicMock()
        mock_handler.name = "test_handler"

        # Register the handler
        self.manager.register_handler(mock_handler)

        # Retrieve the handler
        retrieved_handler = self.manager.get_handler("test_handler")
        self.assertEqual(retrieved_handler, mock_handler)

    def test_trigger_alert(self):
        """Test triggering alerts."""
        # Create mock handlers
        mock_handler1 = MagicMock()
        mock_handler1.name = "handler1"

        mock_handler2 = MagicMock()
        mock_handler2.name = "handler2"

        # Register the handlers
        self.manager.register_handler(mock_handler1)
        self.manager.register_handler(mock_handler2)

        # Trigger an alert
        self.manager.trigger(
            "test_alert",
            AlertLevel.WARNING,
            "Test alert message",
            "test_source",
            {"key": "value"}
        )

        # Check that both handlers were called
        mock_handler1.handle.assert_called_once()
        mock_handler2.handle.assert_called_once()

        # Check that the alert was created correctly
        alert = mock_handler1.handle.call_args[0][0]
        self.assertEqual(alert.name, "test_alert")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.source, "test_source")
        self.assertEqual(alert.data, {"key": "value"})


class TestInitializeMonitoring(unittest.TestCase):
    """Tests for the initialize_monitoring function."""

    @patch("src.infrastructure.monitoring.HealthCheckRegistry")
    @patch("src.infrastructure.monitoring.MetricRegistry")
    @patch("src.infrastructure.monitoring.AlertManager")
    @patch("src.infrastructure.monitoring.SystemMetricsCollector")
    @patch("src.infrastructure.monitoring.LoggingAlertHandler")
    def test_initialize_monitoring(self, mock_logging_handler, mock_collector,
                                  mock_alert_manager, mock_metric_registry,
                                  mock_health_registry):
        """Test that initialize_monitoring sets up monitoring correctly."""
        # Create mock instances
        mock_health_registry_instance = MagicMock()
        mock_metric_registry_instance = MagicMock()
        mock_alert_manager_instance = MagicMock()
        mock_collector_instance = MagicMock()
        mock_logging_handler_instance = MagicMock()

        mock_health_registry.return_value = mock_health_registry_instance
        mock_metric_registry.return_value = mock_metric_registry_instance
        mock_alert_manager.return_value = mock_alert_manager_instance
        mock_collector.return_value = mock_collector_instance
        mock_logging_handler.return_value = mock_logging_handler_instance

        # Call initialize_monitoring
        initialize_monitoring()

        # Check that health checks were registered
        self.assertEqual(mock_health_registry_instance.register.call_count, 2)

        # Check that the metrics collector was started
        mock_collector.assert_called_once()
        mock_collector_instance.start.assert_called_once()

        # Check that the logging alert handler was registered
        mock_alert_manager_instance.register_handler.assert_called_once_with(
            mock_logging_handler_instance
        )


if __name__ == "__main__":
    unittest.main()