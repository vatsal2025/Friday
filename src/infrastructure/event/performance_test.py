"""Performance testing for the event system.

This module provides tools for testing the performance of the event system
under various load conditions.
"""

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event as ThreadEvent
from typing import Dict, List, Optional, Tuple

from src.infrastructure.event import Event, EventSystem, setup_event_monitoring
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class EventSystemPerformanceTest:
    """Performance test for the event system.
    
    This class provides functionality to test the performance of the event system
    under various load conditions, measuring throughput, latency, and resource usage.
    """
    
    def __init__(
        self,
        event_system: Optional[EventSystem] = None,
        max_queue_size: int = 10000,
        max_events: int = 100000,
        event_types: List[str] = None,
        event_sizes: List[int] = None,
        num_handlers: int = 5,
        handler_processing_time: float = 0.001,  # 1ms
    ):
        """Initialize the performance test.
        
        Args:
            event_system: An existing event system to test, or None to create a new one.
            max_queue_size: Maximum size of the event queue if creating a new event system.
            max_events: Maximum number of events to store if creating a new event system.
            event_types: List of event types to use in the test.
            event_sizes: List of event data sizes (in bytes) to use in the test.
            num_handlers: Number of event handlers to register.
            handler_processing_time: Simulated processing time for each handler in seconds.
        """
        # Use provided event system or create a new one
        self.event_system = event_system or EventSystem(
            max_queue_size=max_queue_size,
            max_events=max_events
        )
        
        # Test parameters
        self.event_types = event_types or [
            "market_data", "trade_signal", "order", "execution",
            "model_prediction", "system_status", "error"
        ]
        self.event_sizes = event_sizes or [100, 500, 1000, 5000, 10000]  # bytes
        self.num_handlers = num_handlers
        self.handler_processing_time = handler_processing_time
        
        # Test results
        self.results: Dict[str, Dict] = {}
        
        # Monitoring
        self.monitor = None
        self.health_check = None
        self.dashboard = None
        
        # Control flags
        self._stop_event = ThreadEvent()
    
    def setup(self):
        """Set up the test environment."""
        logger.info("Setting up performance test environment")
        
        # Start the event system
        if not self.event_system.is_running():
            self.event_system.start()
        
        # Set up monitoring
        self.monitor, self.health_check, self.dashboard = setup_event_monitoring(self.event_system)
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"Registered {self.num_handlers} handlers")
    
    def _register_handlers(self):
        """Register test handlers with the event system."""
        for i in range(self.num_handlers):
            # Determine which event types this handler will handle
            # Distribute handlers across event types
            handler_event_types = [self.event_types[i % len(self.event_types)]]
            
            # Create a handler with simulated processing time
            def create_handler(handler_id):
                def handle_event(event):
                    # Simulate processing time
                    time.sleep(self.handler_processing_time)
                    # logger.debug(f"Handler {handler_id} processed event {event.event_id}")
                return handle_event
            
            # Register the handler
            self.event_system.register_handler(
                callback=create_handler(i),
                event_types=handler_event_types
            )
    
    def teardown(self):
        """Clean up the test environment."""
        logger.info("Tearing down performance test environment")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
        if self.health_check:
            self.health_check.stop()
        
        # Stop the event system if we created it
        if self.event_system and self.event_system.is_running():
            self.event_system.stop()
    
    def _generate_test_event(self, size_bytes: int) -> Event:
        """Generate a test event with the specified data size.
        
        Args:
            size_bytes: Approximate size of the event data in bytes.
            
        Returns:
            Event: A test event with the specified data size.
        """
        # Choose a random event type
        event_type = random.choice(self.event_types)
        
        # Generate random data to achieve the target size
        # The actual size will be approximate due to JSON overhead
        data_size = max(1, size_bytes - 100)  # Account for event metadata
        
        # Create a base structure with common fields
        data = {
            "timestamp": datetime.now().isoformat(),
            "test_id": f"perf_test_{int(time.time())}",
        }
        
        # Add event type specific fields
        if event_type == "market_data":
            data.update({
                "symbol": random.choice(["AAPL", "MSFT", "GOOG", "AMZN", "FB"]),
                "price": round(random.uniform(100, 1000), 2),
                "volume": random.randint(100, 10000),
                "bid": round(random.uniform(100, 1000), 2),
                "ask": round(random.uniform(100, 1000), 2),
            })
        elif event_type == "trade_signal":
            data.update({
                "symbol": random.choice(["AAPL", "MSFT", "GOOG", "AMZN", "FB"]),
                "action": random.choice(["BUY", "SELL"]),
                "price": round(random.uniform(100, 1000), 2),
                "confidence": round(random.uniform(0, 1), 2),
                "strategy": f"strategy_{random.randint(1, 10)}",
            })
        elif event_type == "order":
            data.update({
                "order_id": f"order_{random.randint(1000, 9999)}",
                "symbol": random.choice(["AAPL", "MSFT", "GOOG", "AMZN", "FB"]),
                "type": random.choice(["MARKET", "LIMIT", "STOP"]),
                "side": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(1, 1000),
                "price": round(random.uniform(100, 1000), 2),
            })
        
        # Fill the rest with random data to reach the target size
        current_size = len(json.dumps(data))
        if current_size < data_size:
            # Add random key-value pairs to reach the target size
            for i in range(data_size - current_size):
                key = f"field_{i}"
                value = f"value_{i}"
                data[key] = value
                
                # Check if we've reached the target size
                if len(json.dumps(data)) >= data_size:
                    break
        
        return Event(event_type=event_type, data=data, source="performance_test")
    
    def run_throughput_test(
        self,
        num_events: int = 10000,
        batch_size: int = 100,
        event_size: int = 1000,
        num_threads: int = 4,
        test_duration: float = 60.0,
    ) -> Dict:
        """Run a throughput test.
        
        Args:
            num_events: Total number of events to emit.
            batch_size: Number of events to emit in each batch.
            event_size: Size of each event in bytes.
            num_threads: Number of threads to use for emitting events.
            test_duration: Maximum duration of the test in seconds.
            
        Returns:
            Dict: Test results.
        """
        logger.info(f"Starting throughput test with {num_events} events, "
                   f"{batch_size} batch size, {event_size} bytes per event, "
                   f"{num_threads} threads")
        
        # Reset monitoring metrics
        if self.monitor:
            self.monitor.reset_metrics()
        
        # Prepare test data
        events_per_thread = num_events // num_threads
        remaining_events = num_events % num_threads
        
        # Function for each thread to emit events
        def emit_events(thread_id: int, num_events_to_emit: int):
            events_emitted = 0
            start_time = time.time()
            
            while (events_emitted < num_events_to_emit and 
                   time.time() - start_time < test_duration and
                   not self._stop_event.is_set()):
                
                # Determine batch size for this iteration
                current_batch_size = min(batch_size, num_events_to_emit - events_emitted)
                
                # Generate and emit a batch of events
                batch_start_time = time.time()
                for _ in range(current_batch_size):
                    event = self._generate_test_event(event_size)
                    self.event_system.emit(event)
                    events_emitted += 1
                batch_end_time = time.time()
                
                # Calculate batch throughput
                batch_duration = batch_end_time - batch_start_time
                batch_throughput = current_batch_size / batch_duration if batch_duration > 0 else 0
                
                logger.debug(f"Thread {thread_id}: Emitted {current_batch_size} events "
                           f"in {batch_duration:.3f}s ({batch_throughput:.1f} events/s)")
            
            return events_emitted
        
        # Start the test
        total_start_time = time.time()
        total_events_emitted = 0
        
        # Use a thread pool to emit events
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks for each thread
            futures = []
            for i in range(num_threads):
                # Distribute remaining events
                thread_events = events_per_thread + (1 if i < remaining_events else 0)
                futures.append(executor.submit(emit_events, i, thread_events))
            
            # Wait for all threads to complete
            for future in futures:
                total_events_emitted += future.result()
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Calculate results
        throughput = total_events_emitted / total_duration if total_duration > 0 else 0
        
        # Get monitoring metrics
        monitoring_metrics = {}
        if self.monitor:
            monitoring_metrics = self.monitor.get_summary()
        
        # Store and return results
        results = {
            "test_type": "throughput",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "num_events": num_events,
                "batch_size": batch_size,
                "event_size": event_size,
                "num_threads": num_threads,
                "test_duration": test_duration,
            },
            "results": {
                "events_emitted": total_events_emitted,
                "duration": total_duration,
                "throughput": throughput,  # events/second
                "monitoring": monitoring_metrics,
            }
        }
        
        self.results[f"throughput_{int(time.time())}"] = results
        
        logger.info(f"Throughput test completed: {total_events_emitted} events "
                   f"in {total_duration:.3f}s ({throughput:.1f} events/s)")
        
        return results
    
    def run_latency_test(
        self,
        num_events: int = 1000,
        event_size: int = 1000,
        test_duration: float = 30.0,
    ) -> Dict:
        """Run a latency test.
        
        Args:
            num_events: Number of events to emit.
            event_size: Size of each event in bytes.
            test_duration: Maximum duration of the test in seconds.
            
        Returns:
            Dict: Test results.
        """
        logger.info(f"Starting latency test with {num_events} events, "
                   f"{event_size} bytes per event")
        
        # Reset monitoring metrics
        if self.monitor:
            self.monitor.reset_metrics()
        
        # Create a special handler to measure latency
        latencies = []
        latency_event = ThreadEvent()
        
        def latency_handler(event):
            # Calculate latency from event creation to handler execution
            if hasattr(event, "_test_emit_time"):
                latency = time.time() - event._test_emit_time
                latencies.append(latency)
                # logger.debug(f"Latency: {latency * 1000:.2f}ms")
            
            # Signal completion of the last event
            if len(latencies) >= num_events:
                latency_event.set()
        
        # Register the latency handler for all event types
        latency_handler_obj = self.event_system.register_handler(
            callback=latency_handler,
            event_types=self.event_types
        )
        
        try:
            # Start the test
            start_time = time.time()
            events_emitted = 0
            
            while (events_emitted < num_events and 
                   time.time() - start_time < test_duration and
                   not self._stop_event.is_set()):
                
                # Generate and emit an event with timestamp
                event = self._generate_test_event(event_size)
                event._test_emit_time = time.time()  # Add emit time for latency calculation
                self.event_system.emit(event)
                events_emitted += 1
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.001)
            
            # Wait for all events to be processed or timeout
            timeout = max(test_duration, 10.0)  # At least 10 seconds
            latency_event.wait(timeout=timeout)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate latency statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                p50_latency = statistics.median(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            else:
                avg_latency = min_latency = max_latency = p50_latency = p95_latency = p99_latency = 0
            
            # Get monitoring metrics
            monitoring_metrics = {}
            if self.monitor:
                monitoring_metrics = self.monitor.get_summary()
            
            # Store and return results
            results = {
                "test_type": "latency",
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "num_events": num_events,
                    "event_size": event_size,
                    "test_duration": test_duration,
                },
                "results": {
                    "events_emitted": events_emitted,
                    "events_processed": len(latencies),
                    "duration": total_duration,
                    "latency": {
                        "avg": avg_latency * 1000,  # ms
                        "min": min_latency * 1000,  # ms
                        "max": max_latency * 1000,  # ms
                        "p50": p50_latency * 1000,  # ms
                        "p95": p95_latency * 1000,  # ms
                        "p99": p99_latency * 1000,  # ms
                    },
                    "monitoring": monitoring_metrics,
                }
            }
            
            self.results[f"latency_{int(time.time())}"] = results
            
            logger.info(f"Latency test completed: {len(latencies)}/{events_emitted} events processed")
            logger.info(f"Latency (ms): avg={avg_latency*1000:.2f}, min={min_latency*1000:.2f}, "
                       f"max={max_latency*1000:.2f}, p50={p50_latency*1000:.2f}, "
                       f"p95={p95_latency*1000:.2f}, p99={p99_latency*1000:.2f}")
            
            return results
        
        finally:
            # Unregister the latency handler
            self.event_system.unregister_handler(latency_handler_obj)
    
    def run_stress_test(
        self,
        duration: float = 300.0,  # 5 minutes
        ramp_up_time: float = 60.0,  # 1 minute
        max_events_per_second: int = 1000,
        event_size: int = 1000,
        num_threads: int = 4,
    ) -> Dict:
        """Run a stress test with gradually increasing load.
        
        Args:
            duration: Total duration of the test in seconds.
            ramp_up_time: Time to ramp up to maximum load in seconds.
            max_events_per_second: Maximum events per second at peak load.
            event_size: Size of each event in bytes.
            num_threads: Number of threads to use for emitting events.
            
        Returns:
            Dict: Test results.
        """
        logger.info(f"Starting stress test with {duration}s duration, "
                   f"{ramp_up_time}s ramp-up, {max_events_per_second} max events/s, "
                   f"{event_size} bytes per event, {num_threads} threads")
        
        # Reset monitoring metrics
        if self.monitor:
            self.monitor.reset_metrics()
        
        # Function for each thread to emit events
        def emit_events(thread_id: int, stop_time: float):
            events_emitted = 0
            start_time = time.time()
            
            while (time.time() < stop_time and not self._stop_event.is_set()):
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Calculate current target rate based on ramp-up
                if elapsed < ramp_up_time:
                    # Linear ramp-up
                    target_rate = (elapsed / ramp_up_time) * max_events_per_second
                else:
                    target_rate = max_events_per_second
                
                # Calculate target rate per thread
                thread_target_rate = target_rate / num_threads
                
                # Calculate sleep time to achieve target rate
                # Each thread emits 1 event per iteration
                sleep_time = 1.0 / thread_target_rate if thread_target_rate > 0 else 0.1
                
                # Emit an event
                event = self._generate_test_event(event_size)
                self.event_system.emit(event)
                events_emitted += 1
                
                # Sleep to maintain target rate
                time.sleep(sleep_time)
            
            return events_emitted
        
        # Start the test
        total_start_time = time.time()
        stop_time = total_start_time + duration
        total_events_emitted = 0
        
        # Use a thread pool to emit events
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks for each thread
            futures = []
            for i in range(num_threads):
                futures.append(executor.submit(emit_events, i, stop_time))
            
            # Wait for all threads to complete
            for future in futures:
                total_events_emitted += future.result()
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Calculate results
        throughput = total_events_emitted / total_duration if total_duration > 0 else 0
        
        # Get monitoring metrics
        monitoring_metrics = {}
        if self.monitor:
            monitoring_metrics = self.monitor.get_summary()
        
        # Get health status
        health_status = {}
        if self.health_check:
            health_status = self.health_check.get_health_status()
        
        # Store and return results
        results = {
            "test_type": "stress",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "duration": duration,
                "ramp_up_time": ramp_up_time,
                "max_events_per_second": max_events_per_second,
                "event_size": event_size,
                "num_threads": num_threads,
            },
            "results": {
                "events_emitted": total_events_emitted,
                "duration": total_duration,
                "throughput": throughput,  # events/second
                "monitoring": monitoring_metrics,
                "health": health_status,
            }
        }
        
        self.results[f"stress_{int(time.time())}"] = results
        
        logger.info(f"Stress test completed: {total_events_emitted} events "
                   f"in {total_duration:.3f}s ({throughput:.1f} events/s)")
        logger.info(f"Health status: {health_status['status']}")
        if health_status.get('issues'):
            logger.info(f"Health issues: {health_status['issues']}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all performance tests.
        
        Returns:
            Dict[str, Dict]: All test results.
        """
        logger.info("Running all performance tests")
        
        try:
            # Set up the test environment
            self.setup()
            
            # Run throughput tests with different configurations
            self.run_throughput_test(num_events=1000, batch_size=10, event_size=100, num_threads=1)
            self.run_throughput_test(num_events=10000, batch_size=100, event_size=1000, num_threads=4)
            
            # Run latency tests with different configurations
            self.run_latency_test(num_events=100, event_size=100)
            self.run_latency_test(num_events=1000, event_size=1000)
            
            # Run a stress test
            self.run_stress_test(duration=60.0, ramp_up_time=10.0, max_events_per_second=500)
            
            return self.results
        
        finally:
            # Clean up
            self.teardown()
    
    def save_results(self, filename: str = None) -> str:
        """Save test results to a file.
        
        Args:
            filename: Name of the file to save results to.
                If None, a default name will be generated.
                
        Returns:
            str: The filename where results were saved.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_system_performance_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved performance test results to {filename}")
        return filename


def main():
    """Run performance tests from the command line."""
    parser = argparse.ArgumentParser(description="Event System Performance Tests")
    parser.add_argument("--test", choices=["throughput", "latency", "stress", "all"],
                        default="all", help="Test to run")
    parser.add_argument("--events", type=int, default=10000,
                        help="Number of events for throughput/latency tests")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for throughput test")
    parser.add_argument("--event-size", type=int, default=1000,
                        help="Event size in bytes")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads for throughput/stress tests")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=float, default=10.0,
                        help="Ramp-up time for stress test in seconds")
    parser.add_argument("--max-rate", type=int, default=1000,
                        help="Maximum events per second for stress test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for test results")
    
    args = parser.parse_args()
    
    # Create and set up the test
    test = EventSystemPerformanceTest()
    test.setup()
    
    try:
        # Run the specified test
        if args.test == "throughput" or args.test == "all":
            test.run_throughput_test(
                num_events=args.events,
                batch_size=args.batch_size,
                event_size=args.event_size,
                num_threads=args.threads,
                test_duration=args.duration
            )
        
        if args.test == "latency" or args.test == "all":
            test.run_latency_test(
                num_events=args.events,
                event_size=args.event_size,
                test_duration=args.duration
            )
        
        if args.test == "stress" or args.test == "all":
            test.run_stress_test(
                duration=args.duration,
                ramp_up_time=args.ramp_up,
                max_events_per_second=args.max_rate,
                event_size=args.event_size,
                num_threads=args.threads
            )
        
        # Save results
        test.save_results(args.output)
    
    finally:
        # Clean up
        test.teardown()


if __name__ == "__main__":
    main()