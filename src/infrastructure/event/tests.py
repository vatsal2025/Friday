"""Unit tests for the event system.

This module contains tests for the event system components to ensure
they work correctly in production.
"""

import json
import time
import unittest
from queue import Queue
from threading import Event as ThreadEvent
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.infrastructure.event.event_system import (
    Event, EventBus, EventHandler, EventQueue, EventStore, EventSystem
)
from src.infrastructure.event.monitoring import (
    EventMonitor, EventHealthCheck, EventDashboard, setup_event_monitoring
)


class TestEvent(unittest.TestCase):
    """Tests for the Event class."""
    
    def test_event_creation(self):
        """Test creating an event."""
        event = Event(event_type="test", data={"key": "value"})
        
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, {"key": "value"})
        self.assertIsNotNone(event.timestamp)
        self.assertIsNotNone(event.event_id)
    
    def test_event_to_dict(self):
        """Test converting an event to a dictionary."""
        event = Event(event_type="test", data={"key": "value"}, source="test_source")
        event_dict = event.to_dict()
        
        self.assertEqual(event_dict["event_type"], "test")
        self.assertEqual(event_dict["data"], {"key": "value"})
        self.assertEqual(event_dict["source"], "test_source")
        self.assertIn("timestamp", event_dict)
        self.assertIn("event_id", event_dict)
    
    def test_event_from_dict(self):
        """Test creating an event from a dictionary."""
        event_dict = {
            "event_type": "test",
            "data": {"key": "value"},
            "timestamp": "2023-01-01T00:00:00",
            "source": "test_source",
            "event_id": "test_id"
        }
        
        event = Event.from_dict(event_dict)
        
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, {"key": "value"})
        self.assertEqual(event.timestamp, "2023-01-01T00:00:00")
        self.assertEqual(event.source, "test_source")
        self.assertEqual(event.event_id, "test_id")
    
    def test_event_to_json(self):
        """Test converting an event to JSON."""
        event = Event(event_type="test", data={"key": "value"})
        event_json = event.to_json()
        
        # Parse the JSON string back to a dictionary
        event_dict = json.loads(event_json)
        
        self.assertEqual(event_dict["event_type"], "test")
        self.assertEqual(event_dict["data"], {"key": "value"})
    
    def test_event_from_json(self):
        """Test creating an event from JSON."""
        event_json = json.dumps({
            "event_type": "test",
            "data": {"key": "value"},
            "timestamp": "2023-01-01T00:00:00",
            "source": "test_source",
            "event_id": "test_id"
        })
        
        event = Event.from_json(event_json)
        
        self.assertEqual(event.event_type, "test")
        self.assertEqual(event.data, {"key": "value"})
        self.assertEqual(event.timestamp, "2023-01-01T00:00:00")
        self.assertEqual(event.source, "test_source")
        self.assertEqual(event.event_id, "test_id")


class TestEventQueue(unittest.TestCase):
    """Tests for the EventQueue class."""
    
    def test_queue_put_get(self):
        """Test putting and getting events from the queue."""
        queue = EventQueue(max_size=10)
        event = Event(event_type="test", data={"key": "value"})
        
        # Put an event in the queue
        queue.put(event)
        
        # Get the event from the queue
        retrieved_event = queue.get()
        
        self.assertEqual(retrieved_event.event_type, event.event_type)
        self.assertEqual(retrieved_event.data, event.data)
    
    def test_queue_size(self):
        """Test getting the size of the queue."""
        queue = EventQueue(max_size=10)
        
        # Queue should be empty initially
        self.assertEqual(queue.size(), 0)
        
        # Add some events
        for i in range(5):
            queue.put(Event(event_type=f"test_{i}", data={}))
        
        # Queue should have 5 events
        self.assertEqual(queue.size(), 5)
    
    def test_queue_max_size(self):
        """Test that the queue respects its maximum size."""
        max_size = 5
        queue = EventQueue(max_size=max_size)
        
        # Add more events than the max size
        for i in range(max_size + 5):
            queue.put(Event(event_type=f"test_{i}", data={}))
        
        # Queue size should be limited to max_size
        self.assertEqual(queue.size(), max_size)
    
    def test_queue_get_timeout(self):
        """Test that get() times out correctly."""
        queue = EventQueue(max_size=10)
        
        # Try to get from an empty queue with a short timeout
        start_time = time.time()
        event = queue.get(timeout=0.1)
        elapsed_time = time.time() - start_time
        
        # Event should be None
        self.assertIsNone(event)
        
        # Should have waited at least the timeout period
        self.assertGreaterEqual(elapsed_time, 0.1)


class TestEventHandler(unittest.TestCase):
    """Tests for the EventHandler class."""
    
    def test_handler_creation(self):
        """Test creating an event handler."""
        callback = lambda event: None
        handler = EventHandler(callback=callback, event_types=["test"])
        
        self.assertEqual(handler.callback, callback)
        self.assertEqual(handler.event_types, ["test"])
        self.assertIsNone(handler.filter_func)
    
    def test_handles_event_type(self):
        """Test checking if a handler handles an event type."""
        handler = EventHandler(callback=lambda event: None, event_types=["type1", "type2"])
        
        self.assertTrue(handler.handles_event_type("type1"))
        self.assertTrue(handler.handles_event_type("type2"))
        self.assertFalse(handler.handles_event_type("type3"))
        
        # Test with no event types (handles all)
        handler = EventHandler(callback=lambda event: None)
        self.assertTrue(handler.handles_event_type("any_type"))
    
    def test_should_handle(self):
        """Test checking if a handler should handle an event."""
        handler = EventHandler(callback=lambda event: None, event_types=["type1"])
        
        # Should handle events of the right type
        event1 = Event(event_type="type1", data={})
        self.assertTrue(handler.should_handle(event1))
        
        # Should not handle events of the wrong type
        event2 = Event(event_type="type2", data={})
        self.assertFalse(handler.should_handle(event2))
        
        # Test with a filter function
        handler = EventHandler(
            callback=lambda event: None,
            event_types=["type1"],
            filter_func=lambda event: event.data.get("key") == "value"
        )
        
        # Should handle events that pass the filter
        event3 = Event(event_type="type1", data={"key": "value"})
        self.assertTrue(handler.should_handle(event3))
        
        # Should not handle events that don't pass the filter
        event4 = Event(event_type="type1", data={"key": "wrong_value"})
        self.assertFalse(handler.should_handle(event4))
    
    def test_handle(self):
        """Test handling an event."""
        # Create a mock callback
        mock_callback = MagicMock()
        handler = EventHandler(callback=mock_callback, event_types=["test"])
        
        # Create an event
        event = Event(event_type="test", data={"key": "value"})
        
        # Handle the event
        handler.handle(event)
        
        # Check that the callback was called with the event
        mock_callback.assert_called_once_with(event)


class TestEventBus(unittest.TestCase):
    """Tests for the EventBus class."""
    
    def setUp(self):
        """Set up for each test."""
        self.bus = EventBus(max_queue_size=10)
    
    def tearDown(self):
        """Clean up after each test."""
        if self.bus._running:
            self.bus.stop()
    
    def test_register_handler(self):
        """Test registering a handler."""
        handler = EventHandler(callback=lambda event: None, event_types=["test"])
        
        # Register the handler
        self.bus.register_handler(handler)
        
        # Check that the handler was registered
        self.assertIn(handler, self.bus._handlers)
    
    def test_unregister_handler(self):
        """Test unregistering a handler."""
        handler = EventHandler(callback=lambda event: None, event_types=["test"])
        
        # Register the handler
        self.bus.register_handler(handler)
        
        # Unregister the handler
        self.bus.unregister_handler(handler)
        
        # Check that the handler was unregistered
        self.assertNotIn(handler, self.bus._handlers)
    
    def test_publish(self):
        """Test publishing an event."""
        # Create a mock event queue
        mock_queue = MagicMock()
        self.bus._event_queue = mock_queue
        
        # Create an event
        event = Event(event_type="test", data={"key": "value"})
        
        # Publish the event
        self.bus.publish(event)
        
        # Check that the event was put in the queue
        mock_queue.put.assert_called_once_with(event)
    
    def test_start_stop(self):
        """Test starting and stopping the event bus."""
        # Start the bus
        self.bus.start()
        
        # Check that the bus is running
        self.assertTrue(self.bus._running)
        self.assertIsNotNone(self.bus._worker_thread)
        
        # Stop the bus
        self.bus.stop()
        
        # Check that the bus is not running
        self.assertFalse(self.bus._running)
    
    def test_process_events(self):
        """Test processing events."""
        # Create a mock event
        event = Event(event_type="test", data={"key": "value"})
        
        # Create a mock handler
        mock_handler = MagicMock()
        mock_handler.should_handle.return_value = True
        
        # Register the handler
        self.bus._handlers.append(mock_handler)
        
        # Create a mock queue that returns our event once, then None
        mock_queue = MagicMock()
        mock_queue.get.side_effect = [event, None]
        self.bus._event_queue = mock_queue
        
        # Set up a flag to stop processing after one event
        stop_flag = ThreadEvent()
        
        def side_effect(event):
            stop_flag.set()
            return None
        
        mock_handler.handle.side_effect = side_effect
        
        # Start processing events
        self.bus._running = True
        self.bus._process_events()
        
        # Check that the handler was called with the event
        mock_handler.should_handle.assert_called_with(event)
        mock_handler.handle.assert_called_with(event)


class TestEventSystem(unittest.TestCase):
    """Tests for the EventSystem class."""
    
    def setUp(self):
        """Set up for each test."""
        self.event_bus = MagicMock()
        self.event_store = MagicMock()
        self.event_system = EventSystem(event_bus=self.event_bus, event_store=self.event_store)
    
    def test_start(self):
        """Test starting the event system."""
        self.event_system.start()
        
        # Check that the bus and store were started
        self.event_bus.start.assert_called_once()
        self.event_store.start.assert_called_once()
    
    def test_stop(self):
        """Test stopping the event system."""
        self.event_system.stop()
        
        # Check that the bus and store were stopped
        self.event_bus.stop.assert_called_once()
        self.event_store.stop.assert_called_once()
    
    def test_emit(self):
        """Test emitting an event."""
        # Create an event
        event = Event(event_type="test", data={"key": "value"})
        
        # Emit the event
        self.event_system.emit(event)
        
        # Check that the event was published to the bus
        self.event_bus.publish.assert_called_once_with(event)
        
        # Check that the event was stored
        self.event_store.store_event.assert_called_once_with(event)
    
    def test_register_handler(self):
        """Test registering a handler."""
        # Create a handler
        callback = lambda event: None
        event_types = ["test"]
        filter_func = lambda event: True
        
        # Register the handler
        handler = self.event_system.register_handler(
            callback=callback,
            event_types=event_types,
            filter_func=filter_func
        )
        
        # Check that the handler was created correctly
        self.assertEqual(handler.callback, callback)
        self.assertEqual(handler.event_types, event_types)
        self.assertEqual(handler.filter_func, filter_func)
        
        # Check that the handler was registered with the bus
        self.event_bus.register_handler.assert_called_once_with(handler)
    
    def test_unregister_handler(self):
        """Test unregistering a handler."""
        # Create a handler
        handler = EventHandler(callback=lambda event: None)
        
        # Unregister the handler
        self.event_system.unregister_handler(handler)
        
        # Check that the handler was unregistered from the bus
        self.event_bus.unregister_handler.assert_called_once_with(handler)


class TestEventStore(unittest.TestCase):
    """Tests for the EventStore class."""
    
    def setUp(self):
        """Set up for each test."""
        self.store = EventStore(max_events=5)
    
    def tearDown(self):
        """Clean up after each test."""
        if self.store._running:
            self.store.stop()
    
    def test_store_event(self):
        """Test storing an event."""
        event = Event(event_type="test", data={"key": "value"})
        
        # Store the event
        self.store.store_event(event)
        
        # Check that the event was stored
        self.assertIn(event, self.store._events)
    
    def test_get_events(self):
        """Test getting events."""
        # Store some events
        events = [
            Event(event_type="type1", data={"key": "value1"}),
            Event(event_type="type2", data={"key": "value2"}),
            Event(event_type="type1", data={"key": "value3"})
        ]
        
        for event in events:
            self.store.store_event(event)
        
        # Get all events
        all_events = self.store.get_events()
        self.assertEqual(len(all_events), 3)
        
        # Get events of a specific type
        type1_events = self.store.get_events(event_type="type1")
        self.assertEqual(len(type1_events), 2)
        self.assertEqual(type1_events[0].event_type, "type1")
        self.assertEqual(type1_events[1].event_type, "type1")
        
        # Get events with a filter
        value1_events = self.store.get_events(filter_func=lambda e: e.data["key"] == "value1")
        self.assertEqual(len(value1_events), 1)
        self.assertEqual(value1_events[0].data["key"], "value1")
    
    def test_clear_events(self):
        """Test clearing events."""
        # Store some events
        events = [
            Event(event_type="type1", data={"key": "value1"}),
            Event(event_type="type2", data={"key": "value2"})
        ]
        
        for event in events:
            self.store.store_event(event)
        
        # Clear the events
        self.store.clear_events()
        
        # Check that the events were cleared
        self.assertEqual(len(self.store.get_events()), 0)
    
    def test_max_events(self):
        """Test that the store respects its maximum size."""
        max_events = 3
        store = EventStore(max_events=max_events)
        
        # Store more events than the max
        for i in range(max_events + 2):
            store.store_event(Event(event_type=f"test_{i}", data={}))
        
        # Check that only the most recent events were kept
        events = store.get_events()
        self.assertEqual(len(events), max_events)
        self.assertEqual(events[0].event_type, f"test_{2}")
        self.assertEqual(events[1].event_type, f"test_{3}")
        self.assertEqual(events[2].event_type, f"test_{4}")


class TestEventMonitor(unittest.TestCase):
    """Tests for the EventMonitor class."""
    
    def setUp(self):
        """Set up for each test."""
        self.event_system = MagicMock()
        self.monitor = EventMonitor(event_system=self.event_system, sampling_interval=0.1)
    
    def tearDown(self):
        """Clean up after each test."""
        if self.monitor._running:
            self.monitor.stop()
    
    def test_handle_event(self):
        """Test handling an event for monitoring."""
        # Create an event
        event = Event(event_type="test", data={"key": "value"})
        
        # Handle the event
        self.monitor._handle_event(event)
        
        # Check that the event was counted
        self.assertEqual(self.monitor._event_counts["test"], 1)
        
        # Handle another event of the same type
        self.monitor._handle_event(event)
        
        # Check that the count was incremented
        self.assertEqual(self.monitor._event_counts["test"], 2)
    
    def test_get_event_counts(self):
        """Test getting event counts."""
        # Set up some counts
        self.monitor._event_counts["type1"] = 5
        self.monitor._event_counts["type2"] = 3
        
        # Get the counts
        counts = self.monitor.get_event_counts()
        
        # Check the counts
        self.assertEqual(counts["type1"], 5)
        self.assertEqual(counts["type2"], 3)
    
    def test_get_event_sizes(self):
        """Test getting event size statistics."""
        # Set up some sizes
        self.monitor._event_sizes["type1"] = [100, 200, 300]
        
        # Get the size stats
        size_stats = self.monitor.get_event_sizes()
        
        # Check the stats
        self.assertEqual(size_stats["type1"]["min"], 100)
        self.assertEqual(size_stats["type1"]["max"], 300)
        self.assertEqual(size_stats["type1"]["avg"], 200)
        self.assertEqual(size_stats["type1"]["samples"], 3)
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        # Set up some metrics
        self.monitor._event_counts["type1"] = 5
        self.monitor._event_sizes["type1"] = [100, 200, 300]
        
        # Reset the metrics
        self.monitor.reset_metrics()
        
        # Check that the metrics were reset
        self.assertEqual(len(self.monitor._event_counts), 0)
        self.assertEqual(len(self.monitor._event_sizes), 0)


class TestEventHealthCheck(unittest.TestCase):
    """Tests for the EventHealthCheck class."""
    
    def setUp(self):
        """Set up for each test."""
        self.event_system = MagicMock()
        self.event_monitor = MagicMock()
        self.health_check = EventHealthCheck(
            event_system=self.event_system,
            event_monitor=self.event_monitor,
            check_interval=0.1
        )
    
    def tearDown(self):
        """Clean up after each test."""
        if self.health_check._running:
            self.health_check.stop()
    
    def test_get_health_status(self):
        """Test getting health status."""
        # Set up a health status
        self.health_check._health_status = {
            "status": "WARNING",
            "issues": ["Test issue"],
            "last_check": "2023-01-01T00:00:00"
        }
        
        # Get the health status
        status = self.health_check.get_health_status()
        
        # Check the status
        self.assertEqual(status["status"], "WARNING")
        self.assertEqual(status["issues"], ["Test issue"])
        self.assertEqual(status["last_check"], "2023-01-01T00:00:00")
    
    def test_is_healthy(self):
        """Test checking if the system is healthy."""
        # Set up a healthy status
        self.health_check._health_status = {"status": "OK"}
        
        # Check if healthy
        self.assertTrue(self.health_check.is_healthy())
        
        # Set up an unhealthy status
        self.health_check._health_status = {"status": "WARNING"}
        
        # Check if healthy
        self.assertFalse(self.health_check.is_healthy())


class TestEventDashboard(unittest.TestCase):
    """Tests for the EventDashboard class."""
    
    def setUp(self):
        """Set up for each test."""
        self.event_monitor = MagicMock()
        self.event_health_check = MagicMock()
        self.dashboard = EventDashboard(
            event_monitor=self.event_monitor,
            event_health_check=self.event_health_check
        )
    
    def test_generate_json_report(self):
        """Test generating a JSON report."""
        # Set up mock return values
        self.event_monitor.get_summary.return_value = {
            "timestamp": "2023-01-01T00:00:00",
            "counts": {"type1": 5, "type2": 3},
            "rates": {"1min": {"type1": 1.0}, "5min": {}, "15min": {}},
            "sizes": {"type1": {"min": 100, "max": 300, "avg": 200, "samples": 3}}
        }
        
        self.event_health_check.get_health_status.return_value = {
            "status": "OK",
            "issues": [],
            "last_check": "2023-01-01T00:00:00"
        }
        
        # Generate a report
        report = self.dashboard.generate_json_report()
        
        # Check the report
        self.assertEqual(report["timestamp"], "2023-01-01T00:00:00")
        self.assertEqual(report["health"]["status"], "OK")
        self.assertEqual(report["metrics"]["counts"]["type1"], 5)
        self.assertEqual(report["metrics"]["rates"]["1min"]["type1"], 1.0)
        self.assertEqual(report["metrics"]["sizes"]["type1"]["min"], 100)


class TestSetupEventMonitoring(unittest.TestCase):
    """Tests for the setup_event_monitoring function."""
    
    def test_setup_event_monitoring(self):
        """Test setting up event monitoring."""
        # Create a mock event system
        event_system = MagicMock()
        
        # Set up monitoring
        with patch("src.infrastructure.event.monitoring.EventMonitor.start") as mock_monitor_start, \
             patch("src.infrastructure.event.monitoring.EventHealthCheck.start") as mock_health_check_start:
            
            monitor, health_check, dashboard = setup_event_monitoring(event_system)
            
            # Check that the components were created
            self.assertIsInstance(monitor, EventMonitor)
            self.assertIsInstance(health_check, EventHealthCheck)
            self.assertIsInstance(dashboard, EventDashboard)
            
            # Check that monitoring was started
            mock_monitor_start.assert_called_once()
            mock_health_check_start.assert_called_once()


if __name__ == "__main__":
    unittest.main()