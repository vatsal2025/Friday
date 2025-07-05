"""Tests for the diagnostics module.

This module contains tests for the diagnostics functionality, including:
- System information collection
- Memory diagnostics
- Performance diagnostics
- Deadlock detection
- Request tracing
- Diagnostic reports
"""

import unittest
from unittest.mock import MagicMock, patch
import threading
import time

from src.infrastructure.diagnostics import (
    DiagnosticLevel, SystemInfo, MemoryDiagnostics, PerformanceDiagnostics,
    DeadlockDetector, RequestTracer, DiagnosticReport, DiagnosticServer,
    initialize_diagnostics
)


class TestSystemInfo(unittest.TestCase):
    """Tests for the SystemInfo class."""

    def test_get_basic_system_info(self):
        """Test getting basic system information."""
        system_info = SystemInfo()
        info = system_info.get_basic_info()

        # Check that basic info contains expected keys
        self.assertIn("os", info)
        self.assertIn("python_version", info)
        self.assertIn("cpu_count", info)
        self.assertIn("hostname", info)

    @patch("platform.uname")
    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_detailed_system_info(self, mock_virtual_memory, mock_cpu_count, mock_uname):
        """Test getting detailed system information."""
        # Mock system information functions
        mock_uname.return_value = MagicMock(
            system="Linux",
            node="test-host",
            release="5.10.0",
            version="#1 SMP",
            machine="x86_64"
        )
        mock_cpu_count.return_value = 8
        mock_virtual_memory.return_value = MagicMock(
            total=16000000000,  # 16 GB
            available=8000000000  # 8 GB
        )

        system_info = SystemInfo()
        info = system_info.get_detailed_info()

        # Check that detailed info contains expected keys and values
        self.assertEqual(info["system"]["os"], "Linux")
        self.assertEqual(info["system"]["hostname"], "test-host")
        self.assertEqual(info["system"]["cpu_count"], 8)
        self.assertEqual(info["system"]["total_memory_gb"], 14.9)  # Approx 16 GB
        self.assertEqual(info["system"]["available_memory_gb"], 7.45)  # Approx 8 GB

    def test_get_python_info(self):
        """Test getting Python information."""
        system_info = SystemInfo()
        info = system_info.get_python_info()

        # Check that Python info contains expected keys
        self.assertIn("version", info)
        self.assertIn("implementation", info)
        self.assertIn("executable", info)

    def test_get_thread_info(self):
        """Test getting thread information."""
        system_info = SystemInfo()
        info = system_info.get_thread_info()

        # Check that thread info contains expected keys
        self.assertIn("current_thread", info)
        self.assertIn("active_threads", info)
        self.assertIn("thread_count", info)
        self.assertGreaterEqual(info["thread_count"], 1)  # At least the main thread

    @patch.dict("os.environ", {"TEST_VAR": "test_value", "SECRET_KEY": "secret"})
    def test_get_environment_variables(self):
        """Test getting environment variables with sensitive data filtering."""
        system_info = SystemInfo()
        env_vars = system_info.get_environment_variables()

        # Check that non-sensitive variables are included
        self.assertIn("TEST_VAR", env_vars)
        self.assertEqual(env_vars["TEST_VAR"], "test_value")

        # Check that sensitive variables are masked
        self.assertIn("SECRET_KEY", env_vars)
        self.assertEqual(env_vars["SECRET_KEY"], "*****")


class TestMemoryDiagnostics(unittest.TestCase):
    """Tests for the MemoryDiagnostics class."""

    @patch("psutil.Process")
    def test_get_process_memory_usage(self, mock_process):
        """Test getting process memory usage."""
        # Mock process memory information
        mock_process.return_value.memory_info.return_value = MagicMock(
            rss=104857600,  # 100 MB
            vms=209715200   # 200 MB
        )

        memory_diagnostics = MemoryDiagnostics()
        usage = memory_diagnostics.get_process_memory_usage()

        # Check that memory usage contains expected keys and values
        self.assertIn("rss_mb", usage)
        self.assertIn("vms_mb", usage)
        self.assertEqual(usage["rss_mb"], 100.0)
        self.assertEqual(usage["vms_mb"], 200.0)

    @patch("psutil.virtual_memory")
    def test_get_system_memory_usage(self, mock_virtual_memory):
        """Test getting system memory usage."""
        # Mock system memory information
        mock_virtual_memory.return_value = MagicMock(
            total=16000000000,  # 16 GB
            available=8000000000,  # 8 GB
            used=8000000000,  # 8 GB
            percent=50.0
        )

        memory_diagnostics = MemoryDiagnostics()
        usage = memory_diagnostics.get_system_memory_usage()

        # Check that memory usage contains expected keys and values
        self.assertIn("total_gb", usage)
        self.assertIn("available_gb", usage)
        self.assertIn("used_gb", usage)
        self.assertIn("percent", usage)
        self.assertEqual(usage["total_gb"], 14.9)  # Approx 16 GB
        self.assertEqual(usage["available_gb"], 7.45)  # Approx 8 GB
        self.assertEqual(usage["used_gb"], 7.45)  # Approx 8 GB
        self.assertEqual(usage["percent"], 50.0)

    @patch("psutil.swap_memory")
    def test_get_swap_memory_usage(self, mock_swap_memory):
        """Test getting swap memory usage."""
        # Mock swap memory information
        mock_swap_memory.return_value = MagicMock(
            total=8000000000,  # 8 GB
            used=1000000000,  # 1 GB
            percent=12.5
        )

        memory_diagnostics = MemoryDiagnostics()
        usage = memory_diagnostics.get_swap_memory_usage()

        # Check that swap usage contains expected keys and values
        self.assertIn("total_gb", usage)
        self.assertIn("used_gb", usage)
        self.assertIn("percent", usage)
        self.assertEqual(usage["total_gb"], 7.45)  # Approx 8 GB
        self.assertEqual(usage["used_gb"], 0.93)  # Approx 1 GB
        self.assertEqual(usage["percent"], 12.5)

    def test_get_object_counts(self):
        """Test getting object counts."""
        memory_diagnostics = MemoryDiagnostics()
        counts = memory_diagnostics.get_object_counts()

        # Check that object counts contains expected keys
        self.assertIn("total_objects", counts)
        self.assertIn("type_counts", counts)
        self.assertGreater(counts["total_objects"], 0)
        self.assertGreater(len(counts["type_counts"]), 0)

    @patch("gc.get_objects")
    def test_find_memory_leaks(self, mock_get_objects):
        """Test finding potential memory leaks."""
        # Create some test objects
        test_objects = [
            "string1", "string2", "string3",  # 3 strings
            [1, 2, 3], [4, 5, 6],  # 2 lists
            {"key": "value"}  # 1 dict
        ]
        mock_get_objects.return_value = test_objects

        memory_diagnostics = MemoryDiagnostics()
        leaks = memory_diagnostics.find_potential_leaks(threshold=2)

        # Check that potential leaks contains expected types
        self.assertIn("str", leaks)
        self.assertEqual(leaks["str"], 3)  # 3 strings
        self.assertIn("list", leaks)
        self.assertEqual(leaks["list"], 2)  # 2 lists
        self.assertNotIn("dict", leaks)  # Only 1 dict, below threshold

    def test_track_memory_usage_decorator(self):
        """Test the track_memory_usage decorator."""
        # Create a mock logger
        mock_logger = MagicMock()

        # Apply the decorator to a function
        @MemoryDiagnostics.track_memory_usage(logger=mock_logger)
        def test_function():
            # Allocate some memory
            data = [0] * 1000000  # Allocate a large list
            return "result"

        # Call the function
        result = test_function()

        # Check that the function returned the correct result
        self.assertEqual(result, "result")

        # Check that the logger was called
        mock_logger.info.assert_called()
        # The log message should contain memory usage information
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("Memory usage", log_message)


class TestPerformanceDiagnostics(unittest.TestCase):
    """Tests for the PerformanceDiagnostics class."""

    def test_profile_decorator(self):
        """Test the profile decorator."""
        # Create a mock logger
        mock_logger = MagicMock()

        # Apply the decorator to a function
        @PerformanceDiagnostics.profile(logger=mock_logger)
        def test_function(n):
            result = 0
            for i in range(n):
                result += i
            return result

        # Call the function
        result = test_function(1000)

        # Check that the function returned the correct result
        self.assertEqual(result, 499500)  # Sum of 0 to 999

        # Check that the logger was called
        mock_logger.info.assert_called()
        # The log message should contain performance information
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("executed in", log_message)

    def test_detailed_profile_decorator(self):
        """Test the detailed_profile decorator."""
        # Create a mock logger
        mock_logger = MagicMock()

        # Apply the decorator to a function
        @PerformanceDiagnostics.detailed_profile(logger=mock_logger)
        def test_function(n):
            result = 0
            for i in range(n):
                result += i
            return result

        # Call the function
        result = test_function(1000)

        # Check that the function returned the correct result
        self.assertEqual(result, 499500)  # Sum of 0 to 999

        # Check that the logger was called
        mock_logger.info.assert_called()
        # The log message should contain detailed performance information
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("Profile results", log_message)

    def test_measure_time_context_manager(self):
        """Test the measure_time context manager."""
        # Create a mock logger
        mock_logger = MagicMock()

        # Use the context manager
        with PerformanceDiagnostics.measure_time("test_operation", logger=mock_logger):
            # Perform some operation
            time.sleep(0.01)

        # Check that the logger was called
        mock_logger.info.assert_called()
        # The log message should contain timing information
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("test_operation", log_message)
        self.assertIn("executed in", log_message)

    def test_measure_detailed_context_manager(self):
        """Test the measure_detailed context manager."""
        # Create a mock logger
        mock_logger = MagicMock()

        # Use the context manager
        with PerformanceDiagnostics.measure_detailed("test_operation", logger=mock_logger):
            # Perform some operation
            time.sleep(0.01)

        # Check that the logger was called
        mock_logger.info.assert_called()
        # The log message should contain detailed timing information
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("test_operation", log_message)
        self.assertIn("Profile results", log_message)


class TestDeadlockDetector(unittest.TestCase):
    """Tests for the DeadlockDetector class."""

    def test_detect_deadlocks_no_deadlock(self):
        """Test detecting deadlocks when there are none."""
        detector = DeadlockDetector()
        deadlocks = detector.detect_deadlocks()

        # There should be no deadlocks
        self.assertEqual(len(deadlocks), 0)

    @patch("threading.enumerate")
    @patch("threading._current_frames")
    def test_detect_deadlocks_with_deadlock(self, mock_current_frames, mock_enumerate):
        """Test detecting deadlocks when there is a deadlock."""
        # Create mock threads
        thread1 = MagicMock()
        thread1.ident = 1
        thread1.name = "Thread-1"

        thread2 = MagicMock()
        thread2.ident = 2
        thread2.name = "Thread-2"

        mock_enumerate.return_value = [thread1, thread2]

        # Create mock frames that indicate a deadlock
        # Thread 1 is waiting for a lock held by Thread 2
        # Thread 2 is waiting for a lock held by Thread 1
        frame1 = MagicMock()
        frame1.f_code.co_name = "acquire"
        frame1.f_code.co_filename = "threading.py"

        frame2 = MagicMock()
        frame2.f_code.co_name = "acquire"
        frame2.f_code.co_filename = "threading.py"

        mock_current_frames.return_value = {1: frame1, 2: frame2}

        detector = DeadlockDetector()
        deadlocks = detector.detect_deadlocks()

        # There should be a potential deadlock
        self.assertEqual(len(deadlocks), 2)
        self.assertIn(1, deadlocks)
        self.assertIn(2, deadlocks)

    @patch("threading.Thread")
    def test_monitor_deadlocks(self, mock_thread):
        """Test the monitor_deadlocks function."""
        from src.infrastructure.diagnostics import monitor_deadlocks

        # Call monitor_deadlocks
        monitor_deadlocks(interval=1, logger=MagicMock())

        # Check that a thread was created and started
        mock_thread.assert_called_once()
        mock_thread.return_value.daemon.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


class TestRequestTracer(unittest.TestCase):
    """Tests for the RequestTracer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        RequestTracer._instance = None
        self.tracer = RequestTracer()

    def test_start_trace(self):
        """Test starting a trace."""
        trace_id = self.tracer.start_trace("test_request")

        # Check that a trace was created
        self.assertIsNotNone(trace_id)
        trace = self.tracer.get_trace(trace_id)
        self.assertEqual(trace["request_type"], "test_request")
        self.assertIn("start_time", trace)
        self.assertIn("events", trace)
        self.assertEqual(len(trace["events"]), 0)

    def test_add_event(self):
        """Test adding events to a trace."""
        trace_id = self.tracer.start_trace("test_request")

        # Add events to the trace
        self.tracer.add_event(trace_id, "event1", {"key1": "value1"})
        self.tracer.add_event(trace_id, "event2", {"key2": "value2"})

        # Check that events were added
        trace = self.tracer.get_trace(trace_id)
        self.assertEqual(len(trace["events"]), 2)
        self.assertEqual(trace["events"][0]["name"], "event1")
        self.assertEqual(trace["events"][0]["data"], {"key1": "value1"})
        self.assertEqual(trace["events"][1]["name"], "event2")
        self.assertEqual(trace["events"][1]["data"], {"key2": "value2"})

    def test_end_trace(self):
        """Test ending a trace."""
        trace_id = self.tracer.start_trace("test_request")

        # End the trace
        self.tracer.end_trace(trace_id, {"result": "success"})

        # Check that the trace was ended
        trace = self.tracer.get_trace(trace_id)
        self.assertIn("end_time", trace)
        self.assertIn("duration_ms", trace)
        self.assertEqual(trace["result"], {"result": "success"})

    def test_trace_function_decorator(self):
        """Test the trace_function decorator."""
        # Apply the decorator to a function
        @self.tracer.trace_function("test_function")
        def test_function(arg1, arg2=None):
            # Add an event within the function
            trace_id = self.tracer.get_current_trace_id()
            self.tracer.add_event(trace_id, "inside_function", {"arg1": arg1, "arg2": arg2})
            return arg1 + (arg2 or 0)

        # Call the function
        result = test_function(1, arg2=2)

        # Check that the function returned the correct result
        self.assertEqual(result, 3)

        # Get the trace
        traces = self.tracer.get_all_traces()
        self.assertEqual(len(traces), 1)

        trace = list(traces.values())[0]
        self.assertEqual(trace["request_type"], "test_function")
        self.assertEqual(len(trace["events"]), 1)
        self.assertEqual(trace["events"][0]["name"], "inside_function")
        self.assertEqual(trace["events"][0]["data"], {"arg1": 1, "arg2": 2})
        self.assertEqual(trace["result"], {"return_value": 3})


class TestDiagnosticReport(unittest.TestCase):
    """Tests for the DiagnosticReport class."""

    def test_generate_basic_report(self):
        """Test generating a basic diagnostic report."""
        report = DiagnosticReport()
        basic_report = report.generate_report(DiagnosticLevel.BASIC)

        # Check that the report contains expected sections
        self.assertIn("timestamp", basic_report)
        self.assertIn("level", basic_report)
        self.assertEqual(basic_report["level"], "BASIC")
        self.assertIn("system_info", basic_report)
        self.assertIn("memory_usage", basic_report)

    def test_generate_detailed_report(self):
        """Test generating a detailed diagnostic report."""
        report = DiagnosticReport()
        detailed_report = report.generate_report(DiagnosticLevel.DETAILED)

        # Check that the report contains expected sections
        self.assertIn("timestamp", detailed_report)
        self.assertIn("level", detailed_report)
        self.assertEqual(detailed_report["level"], "DETAILED")
        self.assertIn("system_info", detailed_report)
        self.assertIn("memory_usage", detailed_report)
        self.assertIn("python_info", detailed_report)
        self.assertIn("thread_info", detailed_report)
        self.assertIn("environment_variables", detailed_report)

    def test_generate_intensive_report(self):
        """Test generating an intensive diagnostic report."""
        report = DiagnosticReport()
        intensive_report = report.generate_report(DiagnosticLevel.INTENSIVE)

        # Check that the report contains expected sections
        self.assertIn("timestamp", intensive_report)
        self.assertIn("level", intensive_report)
        self.assertEqual(intensive_report["level"], "INTENSIVE")
        self.assertIn("system_info", intensive_report)
        self.assertIn("memory_usage", intensive_report)
        self.assertIn("python_info", intensive_report)
        self.assertIn("thread_info", intensive_report)
        self.assertIn("environment_variables", intensive_report)
        self.assertIn("object_counts", intensive_report)
        self.assertIn("potential_memory_leaks", intensive_report)
        self.assertIn("active_traces", intensive_report)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_save_report_to_file(self, mock_open):
        """Test saving a report to a file."""
        report = DiagnosticReport()
        basic_report = report.generate_report(DiagnosticLevel.BASIC)

        # Save the report to a file
        report.save_report_to_file(basic_report, "test_report.json")

        # Check that the file was opened and written to
        mock_open.assert_called_once_with("test_report.json", "w")
        mock_open().write.assert_called_once()

    @patch("builtins.print")
    def test_print_report(self, mock_print):
        """Test printing a report."""
        report = DiagnosticReport()
        basic_report = report.generate_report(DiagnosticLevel.BASIC)

        # Print the report
        report.print_report(basic_report)

        # Check that print was called
        mock_print.assert_called()


class TestDiagnosticServer(unittest.TestCase):
    """Tests for the DiagnosticServer class."""

    @patch("threading.Thread")
    @patch("http.server.HTTPServer")
    def test_start_server(self, mock_http_server, mock_thread):
        """Test starting the diagnostic server."""
        server = DiagnosticServer(port=8080)
        server.start()

        # Check that the server was created and started
        mock_http_server.assert_called_once()
        mock_thread.assert_called_once()
        mock_thread.return_value.daemon.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch("http.server.HTTPServer")
    def test_stop_server(self, mock_http_server):
        """Test stopping the diagnostic server."""
        server = DiagnosticServer(port=8080)
        server.start()

        # Stop the server
        server.stop()

        # Check that the server was shut down
        mock_http_server.return_value.shutdown.assert_called_once()


class TestInitializeDiagnostics(unittest.TestCase):
    """Tests for the initialize_diagnostics function."""

    @patch("src.infrastructure.diagnostics.RequestTracer")
    @patch("src.infrastructure.diagnostics.monitor_deadlocks")
    def test_initialize_diagnostics(self, mock_monitor_deadlocks, mock_request_tracer):
        """Test that initialize_diagnostics sets up diagnostics correctly."""
        # Create mock instances
        mock_request_tracer_instance = MagicMock()
        mock_request_tracer.return_value = mock_request_tracer_instance

        # Call initialize_diagnostics
        initialize_diagnostics()

        # Check that the request tracer was initialized
        mock_request_tracer.assert_called_once()

        # Check that deadlock monitoring was started
        mock_monitor_deadlocks.assert_called_once()


if __name__ == "__main__":
    unittest.main()