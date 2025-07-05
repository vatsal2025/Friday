"""Diagnostics module for the Friday AI Trading System.

This module provides diagnostic capabilities including:
- System diagnostics
- Performance profiling
- Memory leak detection
- Deadlock detection
- Request tracing
"""

import functools
import gc
import inspect
import io
import logging
import os
import platform
import psutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Tuple, TypeVar, Union

from src.infrastructure.logging import get_logger
from src.infrastructure.error import ErrorSeverity, log_error

# Type variable for the return type of the function being wrapped
T = TypeVar('T')

# Create logger
logger = get_logger(__name__)


class DiagnosticLevel(Enum):
    """Diagnostic level for diagnostic operations."""
    BASIC = auto()      # Basic diagnostics, safe to run in production
    DETAILED = auto()   # Detailed diagnostics, may impact performance
    INTENSIVE = auto()  # Intensive diagnostics, significant performance impact


class SystemInfo:
    """Collects and provides system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get basic system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
            'cpu_count': os.cpu_count(),
            'hostname': platform.node(),
            'process_id': os.getpid(),
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add psutil information if available
        try:
            process = psutil.Process()
            info.update({
                'memory_usage': process.memory_info().rss,
                'cpu_usage': process.cpu_percent(interval=0.1),
                'open_files': len(process.open_files()),
                'threads': process.num_threads(),
                'connections': len(process.connections()),
                'uptime': time.time() - process.create_time()
            })
        except Exception as e:
            logger.warning(f"Error getting psutil information: {e}")
        
        return info
    
    @staticmethod
    def get_detailed_system_info() -> Dict[str, Any]:
        """Get detailed system information.
        
        Returns:
            Dictionary with detailed system information
        """
        info = SystemInfo.get_system_info()
        
        # Add more detailed information
        try:
            # Memory information
            memory = psutil.virtual_memory()
            info['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
            
            # CPU information
            info['cpu'] = {
                'percent': psutil.cpu_percent(interval=1, percpu=True),
                'times': psutil.cpu_times()._asdict()
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            info['disk'] = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
            
            # Network information
            network = psutil.net_io_counters()
            info['network'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'errin': network.errin,
                'errout': network.errout,
                'dropin': network.dropin,
                'dropout': network.dropout
            }
            
            # Process information
            process = psutil.Process()
            info['process'] = {
                'name': process.name(),
                'exe': process.exe(),
                'cwd': process.cwd(),
                'cmdline': process.cmdline(),
                'status': process.status(),
                'username': process.username(),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
                'terminal': process.terminal(),
                'memory_maps': [m._asdict() for m in process.memory_maps()],
                'io_counters': process.io_counters()._asdict() if process.io_counters() else None,
                'nice': process.nice(),
                'ionice': process.ionice()._asdict(),
                'cpu_affinity': process.cpu_affinity(),
                'memory_percent': process.memory_percent(),
                'memory_full_info': process.memory_full_info()._asdict(),
                'num_ctx_switches': process.num_ctx_switches()._asdict(),
                'num_handles': process.num_handles() if hasattr(process, 'num_handles') else None,
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
            }
        except Exception as e:
            logger.warning(f"Error getting detailed system information: {e}")
        
        return info
    
    @staticmethod
    def get_python_info() -> Dict[str, Any]:
        """Get Python-specific information.
        
        Returns:
            Dictionary with Python information
        """
        info = {
            'version': sys.version,
            'implementation': platform.python_implementation(),
            'build': platform.python_build(),
            'compiler': platform.python_compiler(),
            'branch': platform.python_branch(),
            'revision': platform.python_revision(),
            'path': sys.path,
            'executable': sys.executable,
            'modules': list(sys.modules.keys()),
            'recursion_limit': sys.getrecursionlimit(),
            'thread_switch_interval': sys.getswitchinterval(),
            'platform': sys.platform,
            'flags': {flag: getattr(sys.flags, flag) for flag in dir(sys.flags) if not flag.startswith('_')}
        }
        
        # Add garbage collector information
        info['gc'] = {
            'enabled': gc.isenabled(),
            'thresholds': gc.get_threshold(),
            'counts': gc.get_count(),
            'stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
        }
        
        return info
    
    @staticmethod
    def get_thread_info() -> Dict[str, Any]:
        """Get information about threads.
        
        Returns:
            Dictionary with thread information
        """
        current_thread = threading.current_thread()
        all_threads = threading.enumerate()
        
        thread_info = {
            'current_thread': {
                'name': current_thread.name,
                'id': current_thread.ident,
                'daemon': current_thread.daemon,
                'alive': current_thread.is_alive()
            },
            'thread_count': len(all_threads),
            'threads': [{
                'name': thread.name,
                'id': thread.ident,
                'daemon': thread.daemon,
                'alive': thread.is_alive()
            } for thread in all_threads]
        }
        
        return thread_info
    
    @staticmethod
    def get_environment_variables() -> Dict[str, str]:
        """Get environment variables.
        
        Returns:
            Dictionary with environment variables
        """
        # Filter out sensitive environment variables
        sensitive_prefixes = ['API_', 'KEY_', 'SECRET_', 'PASSWORD_', 'TOKEN_']
        
        env_vars = {}
        for key, value in os.environ.items():
            # Check if the key has a sensitive prefix
            is_sensitive = any(key.startswith(prefix) for prefix in sensitive_prefixes)
            
            # Add the environment variable, masking sensitive values
            env_vars[key] = '********' if is_sensitive else value
        
        return env_vars


class MemoryDiagnostics:
    """Provides memory diagnostics."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory usage information
        """
        usage = {}
        
        # Get process memory usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            usage['process'] = {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'shared': getattr(memory_info, 'shared', None),  # Shared memory
                'text': getattr(memory_info, 'text', None),  # Text (code)
                'data': getattr(memory_info, 'data', None),  # Data + stack
                'lib': getattr(memory_info, 'lib', None),  # Library
                'dirty': getattr(memory_info, 'dirty', None),  # Dirty pages
                'percent': process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Error getting process memory usage: {e}")
        
        # Get system memory usage
        try:
            memory = psutil.virtual_memory()
            usage['system'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent,
                'cached': getattr(memory, 'cached', None),
                'buffers': getattr(memory, 'buffers', None),
                'shared': getattr(memory, 'shared', None)
            }
        except Exception as e:
            logger.warning(f"Error getting system memory usage: {e}")
        
        # Get swap memory usage
        try:
            swap = psutil.swap_memory()
            usage['swap'] = {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent,
                'sin': getattr(swap, 'sin', None),
                'sout': getattr(swap, 'sout', None)
            }
        except Exception as e:
            logger.warning(f"Error getting swap memory usage: {e}")
        
        return usage
    
    @staticmethod
    def get_object_counts() -> Dict[str, int]:
        """Get counts of Python objects in memory.
        
        Returns:
            Dictionary with object counts
        """
        # Collect all objects
        objects = gc.get_objects()
        
        # Count objects by type
        counts = {}
        for obj in objects:
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        
        return counts
    
    @staticmethod
    def find_memory_leaks(threshold: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """Find potential memory leaks.
        
        Args:
            threshold: Threshold for number of objects to consider a potential leak
            
        Returns:
            Dictionary with potential memory leaks
        """
        # Collect all objects
        objects = gc.get_objects()
        
        # Count objects by type
        counts = {}
        for obj in objects:
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        
        # Find potential leaks
        leaks = {}
        for obj_type, count in counts.items():
            if count > threshold:
                # Get sample objects of this type
                samples = [obj for obj in objects if type(obj).__name__ == obj_type][:10]
                
                # Get information about sample objects
                sample_info = []
                for sample in samples:
                    try:
                        sample_info.append({
                            'type': type(sample).__name__,
                            'id': id(sample),
                            'size': sys.getsizeof(sample),
                            'repr': repr(sample)[:100]  # Limit representation to 100 characters
                        })
                    except Exception:
                        # Some objects may not be representable
                        pass
                
                leaks[obj_type] = {
                    'count': count,
                    'samples': sample_info
                }
        
        return leaks
    
    @staticmethod
    def track_memory_usage(func):
        """Decorator to track memory usage of a function.
        
        Args:
            func: Function to track
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory usage before function call
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Get memory usage after function call
            memory_after = process.memory_info().rss
            memory_diff = memory_after - memory_before
            
            # Log memory usage
            logger.info(f"Memory usage for {func.__name__}: {memory_diff} bytes")
            
            return result
        
        return wrapper


class PerformanceDiagnostics:
    """Provides performance diagnostics."""
    
    @staticmethod
    def profile(func):
        """Decorator to profile a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get start time
            start_time = time.time()
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Get end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Log performance
            logger.info(f"Performance for {func.__name__}: {elapsed_time:.6f} seconds")
            
            return result
        
        return wrapper
    
    @staticmethod
    def detailed_profile(func):
        """Decorator to profile a function in detail.
        
        Args:
            func: Function to profile
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get start time and resources
            start_time = time.time()
            start_process = psutil.Process()
            start_cpu_times = start_process.cpu_times()
            start_memory = start_process.memory_info()
            start_io = start_process.io_counters() if hasattr(start_process, 'io_counters') else None
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Get end time and resources
            end_time = time.time()
            end_process = psutil.Process()
            end_cpu_times = end_process.cpu_times()
            end_memory = end_process.memory_info()
            end_io = end_process.io_counters() if hasattr(end_process, 'io_counters') else None
            
            # Calculate differences
            elapsed_time = end_time - start_time
            cpu_user_diff = end_cpu_times.user - start_cpu_times.user
            cpu_system_diff = end_cpu_times.system - start_cpu_times.system
            memory_rss_diff = end_memory.rss - start_memory.rss
            
            # Calculate IO differences if available
            io_read_diff = None
            io_write_diff = None
            if start_io and end_io:
                io_read_diff = end_io.read_bytes - start_io.read_bytes
                io_write_diff = end_io.write_bytes - start_io.write_bytes
            
            # Log performance
            logger.info(f"Detailed performance for {func.__name__}:")
            logger.info(f"  Elapsed time: {elapsed_time:.6f} seconds")
            logger.info(f"  CPU user time: {cpu_user_diff:.6f} seconds")
            logger.info(f"  CPU system time: {cpu_system_diff:.6f} seconds")
            logger.info(f"  Memory usage: {memory_rss_diff} bytes")
            
            if io_read_diff is not None and io_write_diff is not None:
                logger.info(f"  IO read: {io_read_diff} bytes")
                logger.info(f"  IO write: {io_write_diff} bytes")
            
            return result
        
        return wrapper
    
    @staticmethod
    @contextmanager
    def measure_time(name: str):
        """Context manager to measure execution time.
        
        Args:
            name: Name of the operation being measured
        """
        start_time = time.time()
        yield
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Time for {name}: {elapsed_time:.6f} seconds")
    
    @staticmethod
    @contextmanager
    def measure_detailed(name: str):
        """Context manager to measure detailed performance metrics.
        
        Args:
            name: Name of the operation being measured
        """
        # Get start time and resources
        start_time = time.time()
        start_process = psutil.Process()
        start_cpu_times = start_process.cpu_times()
        start_memory = start_process.memory_info()
        start_io = start_process.io_counters() if hasattr(start_process, 'io_counters') else None
        
        yield
        
        # Get end time and resources
        end_time = time.time()
        end_process = psutil.Process()
        end_cpu_times = end_process.cpu_times()
        end_memory = end_process.memory_info()
        end_io = end_process.io_counters() if hasattr(end_process, 'io_counters') else None
        
        # Calculate differences
        elapsed_time = end_time - start_time
        cpu_user_diff = end_cpu_times.user - start_cpu_times.user
        cpu_system_diff = end_cpu_times.system - start_cpu_times.system
        memory_rss_diff = end_memory.rss - start_memory.rss
        
        # Calculate IO differences if available
        io_read_diff = None
        io_write_diff = None
        if start_io and end_io:
            io_read_diff = end_io.read_bytes - start_io.read_bytes
            io_write_diff = end_io.write_bytes - start_io.write_bytes
        
        # Log performance
        logger.info(f"Detailed performance for {name}:")
        logger.info(f"  Elapsed time: {elapsed_time:.6f} seconds")
        logger.info(f"  CPU user time: {cpu_user_diff:.6f} seconds")
        logger.info(f"  CPU system time: {cpu_system_diff:.6f} seconds")
        logger.info(f"  Memory usage: {memory_rss_diff} bytes")
        
        if io_read_diff is not None and io_write_diff is not None:
            logger.info(f"  IO read: {io_read_diff} bytes")
            logger.info(f"  IO write: {io_write_diff} bytes")


class DeadlockDetector:
    """Detects potential deadlocks in the application."""
    
    @staticmethod
    def detect_deadlocks() -> List[Dict[str, Any]]:
        """Detect potential deadlocks.
        
        Returns:
            List of potential deadlocks
        """
        # This is a simplified implementation that looks for threads waiting on locks
        # A more sophisticated implementation would build a wait-for graph and detect cycles
        
        deadlocks = []
        
        # Get all threads
        all_threads = threading.enumerate()
        
        for thread in all_threads:
            # Skip the current thread
            if thread == threading.current_thread():
                continue
            
            # Check if the thread is alive but not running
            if thread.is_alive() and not thread.daemon:
                # Get the thread's stack frames
                stack_frames = sys._current_frames().get(thread.ident)
                
                if stack_frames:
                    # Check if the thread is waiting on a lock
                    stack_trace = traceback.format_stack(stack_frames)
                    is_waiting_on_lock = any('_acquire_lock' in frame or 'wait' in frame for frame in stack_trace)
                    
                    if is_waiting_on_lock:
                        deadlocks.append({
                            'thread_id': thread.ident,
                            'thread_name': thread.name,
                            'stack_trace': stack_trace
                        })
        
        return deadlocks
    
    @staticmethod
    def monitor_deadlocks(interval: float = 60.0, callback: Callable[[List[Dict[str, Any]]], None] = None):
        """Start a thread to monitor for deadlocks.
        
        Args:
            interval: Interval between checks in seconds
            callback: Callback function to call with deadlocks
        """
        def monitor():
            while True:
                try:
                    deadlocks = DeadlockDetector.detect_deadlocks()
                    if deadlocks:
                        logger.warning(f"Detected {len(deadlocks)} potential deadlocks")
                        if callback:
                            callback(deadlocks)
                except Exception as e:
                    logger.error(f"Error in deadlock monitor: {e}")
                
                time.sleep(interval)
        
        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started deadlock monitor with interval {interval} seconds")
        
        return monitor_thread


class RequestTracer:
    """Traces requests through the system."""
    
    def __init__(self):
        """Initialize a request tracer."""
        self.traces = {}
        self.lock = threading.Lock()
    
    def start_trace(self, request_id: str, context: Dict[str, Any] = None) -> None:
        """Start tracing a request.
        
        Args:
            request_id: ID of the request
            context: Additional context for the trace
        """
        with self.lock:
            self.traces[request_id] = {
                'start_time': time.time(),
                'context': context or {},
                'events': []
            }
        
        logger.debug(f"Started trace for request {request_id}")
    
    def add_event(self, request_id: str, event: str, context: Dict[str, Any] = None) -> None:
        """Add an event to a trace.
        
        Args:
            request_id: ID of the request
            event: Event name
            context: Additional context for the event
        """
        if request_id not in self.traces:
            logger.warning(f"Trace for request {request_id} not found")
            return
        
        with self.lock:
            self.traces[request_id]['events'].append({
                'time': time.time(),
                'event': event,
                'context': context or {}
            })
        
        logger.debug(f"Added event {event} to trace for request {request_id}")
    
    def end_trace(self, request_id: str, result: Any = None) -> Dict[str, Any]:
        """End tracing a request.
        
        Args:
            request_id: ID of the request
            result: Result of the request
            
        Returns:
            Trace information
        """
        if request_id not in self.traces:
            logger.warning(f"Trace for request {request_id} not found")
            return {}
        
        with self.lock:
            trace = self.traces[request_id]
            trace['end_time'] = time.time()
            trace['duration'] = trace['end_time'] - trace['start_time']
            trace['result'] = result
            
            # Remove the trace from the active traces
            del self.traces[request_id]
        
        logger.debug(f"Ended trace for request {request_id}, duration: {trace['duration']:.6f} seconds")
        
        return trace
    
    def get_trace(self, request_id: str) -> Dict[str, Any]:
        """Get a trace by request ID.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Trace information or empty dict if not found
        """
        with self.lock:
            return self.traces.get(request_id, {}).copy()
    
    def get_active_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all active traces.
        
        Returns:
            Dictionary of active traces
        """
        with self.lock:
            return {request_id: trace.copy() for request_id, trace in self.traces.items()}
    
    def clear_traces(self) -> None:
        """Clear all traces."""
        with self.lock:
            self.traces.clear()
        
        logger.debug("Cleared all traces")
    
    def trace_function(self, request_id: str, event_prefix: str = ""):
        """Decorator to trace a function.
        
        Args:
            request_id: ID of the request
            event_prefix: Prefix for event names
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start event
                event_name = f"{event_prefix}{func.__name__}" if event_prefix else func.__name__
                self.add_event(request_id, f"{event_name}_start", {
                    'args': [repr(arg) for arg in args],
                    'kwargs': {key: repr(value) for key, value in kwargs.items()}
                })
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # End event with success
                    self.add_event(request_id, f"{event_name}_end", {
                        'status': 'success',
                        'result': repr(result)
                    })
                    
                    return result
                except Exception as e:
                    # End event with error
                    self.add_event(request_id, f"{event_name}_end", {
                        'status': 'error',
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator


class DiagnosticReport:
    """Generates diagnostic reports."""
    
    @staticmethod
    def generate_basic_report() -> Dict[str, Any]:
        """Generate a basic diagnostic report.
        
        Returns:
            Dictionary with diagnostic information
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': SystemInfo.get_system_info(),
            'memory_usage': MemoryDiagnostics.get_memory_usage(),
            'thread_info': SystemInfo.get_thread_info()
        }
        
        return report
    
    @staticmethod
    def generate_detailed_report() -> Dict[str, Any]:
        """Generate a detailed diagnostic report.
        
        Returns:
            Dictionary with detailed diagnostic information
        """
        report = DiagnosticReport.generate_basic_report()
        
        # Add more detailed information
        report.update({
            'detailed_system_info': SystemInfo.get_detailed_system_info(),
            'python_info': SystemInfo.get_python_info(),
            'object_counts': MemoryDiagnostics.get_object_counts(),
            'potential_deadlocks': DeadlockDetector.detect_deadlocks()
        })
        
        return report
    
    @staticmethod
    def generate_intensive_report() -> Dict[str, Any]:
        """Generate an intensive diagnostic report.
        
        Returns:
            Dictionary with intensive diagnostic information
        """
        report = DiagnosticReport.generate_detailed_report()
        
        # Add more intensive information
        report.update({
            'memory_leaks': MemoryDiagnostics.find_memory_leaks(),
            'environment_variables': SystemInfo.get_environment_variables()
        })
        
        return report
    
    @staticmethod
    def generate_report(level: DiagnosticLevel = DiagnosticLevel.BASIC) -> Dict[str, Any]:
        """Generate a diagnostic report at the specified level.
        
        Args:
            level: Diagnostic level
            
        Returns:
            Dictionary with diagnostic information
        """
        if level == DiagnosticLevel.BASIC:
            return DiagnosticReport.generate_basic_report()
        elif level == DiagnosticLevel.DETAILED:
            return DiagnosticReport.generate_detailed_report()
        elif level == DiagnosticLevel.INTENSIVE:
            return DiagnosticReport.generate_intensive_report()
        else:
            raise ValueError(f"Invalid diagnostic level: {level}")
    
    @staticmethod
    def save_report(report: Dict[str, Any], file_path: str = None) -> str:
        """Save a diagnostic report to a file.
        
        Args:
            report: Diagnostic report
            file_path: Path to save the report to, or None to generate a default path
            
        Returns:
            Path to the saved report
        """
        import json
        
        # Generate a default file path if not provided
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"diagnostic_report_{timestamp}.json"
        
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the report
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved diagnostic report to {file_path}")
        
        return file_path
    
    @staticmethod
    def print_report(report: Dict[str, Any], output: TextIO = None, format_json: bool = True) -> None:
        """Print a diagnostic report.
        
        Args:
            report: Diagnostic report
            output: Output stream, or None to use sys.stdout
            format_json: Whether to format the JSON output
        """
        import json
        
        # Use sys.stdout if no output stream is provided
        if output is None:
            output = sys.stdout
        
        # Print the report
        if format_json:
            json.dump(report, output, indent=2, default=str)
        else:
            json.dump(report, output, default=str)
        
        # Add a newline
        output.write('\n')
        output.flush()


class DiagnosticServer:
    """Server for providing diagnostic information."""
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        """Initialize a diagnostic server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
    
    def start(self) -> None:
        """Start the diagnostic server."""
        if self.running:
            return
        
        # Import here to avoid dependencies if not used
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class DiagnosticHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    # Parse the path
                    path = self.path.strip('/')
                    
                    # Generate the appropriate report
                    if path == 'basic':
                        report = DiagnosticReport.generate_basic_report()
                    elif path == 'detailed':
                        report = DiagnosticReport.generate_detailed_report()
                    elif path == 'intensive':
                        report = DiagnosticReport.generate_intensive_report()
                    else:
                        report = {'error': f"Invalid path: {path}"}
                    
                    # Send the response
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(report, default=str).encode('utf-8'))
                except Exception as e:
                    # Send an error response
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        
        # Create the server
        self.server = HTTPServer((self.host, self.port), DiagnosticHandler)
        
        # Start the server in a separate thread
        def run_server():
            logger.info(f"Starting diagnostic server on {self.host}:{self.port}")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        self.running = True
    
    def stop(self) -> None:
        """Stop the diagnostic server."""
        if not self.running:
            return
        
        # Stop the server
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        
        # Wait for the thread to stop
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        self.running = False
        logger.info("Stopped diagnostic server")


# Initialize the diagnostic system
def initialize_diagnostics():
    """Initialize the diagnostic system."""
    logger.info("Initializing diagnostic system")
    
    # Create a request tracer
    request_tracer = RequestTracer()
    
    # Start deadlock detection
    deadlock_thread = DeadlockDetector.monitor_deadlocks()
    
    return {
        'request_tracer': request_tracer,
        'deadlock_thread': deadlock_thread
    }