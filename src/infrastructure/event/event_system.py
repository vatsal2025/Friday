"""Event system for the Friday AI Trading System.

This module provides an event-driven architecture with message queueing
for handling events across the system.
"""

import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..logging import get_logger

# Create logger
logger = get_logger(__name__)


class Event:
    """Event class for the event system.

    Attributes:
        event_type: The type of the event.
        data: The data associated with the event.
        timestamp: The timestamp when the event was created.
        source: The source of the event.
        id: The unique identifier of the event.
    """

    def __init__(
        self,
        event_type: str,
        data: Any,
        source: Optional[str] = None,
        event_id: Optional[str] = None,
    ):
        """Initialize an event.

        Args:
            event_type: The type of the event.
            data: The data associated with the event.
            source: The source of the event. Defaults to None.
            event_id: The unique identifier of the event. Defaults to None.
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now().isoformat()
        self.source = source
        self.id = event_id or f"{int(time.time() * 1000)}-{threading.get_ident()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.

        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def to_json(self) -> str:
        """Convert the event to a JSON string.

        Returns:
            str: The event as a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> "Event":
        """Create an event from a dictionary.

        Args:
            event_dict: The dictionary to create the event from.

        Returns:
            Event: The created event.
        """
        event = cls(
            event_type=event_dict["event_type"],
            data=event_dict["data"],
            source=event_dict.get("source"),
            event_id=event_dict.get("id"),
        )
        event.timestamp = event_dict.get("timestamp", event.timestamp)
        return event

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create an event from a JSON string.

        Args:
            json_str: The JSON string to create the event from.

        Returns:
            Event: The created event.
        """
        event_dict = json.loads(json_str)
        return cls.from_dict(event_dict)


class EventQueue:
    """Queue for storing events.

    Attributes:
        queue: The underlying queue.
        max_size: The maximum size of the queue.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize an event queue.

        Args:
            max_size: The maximum size of the queue. Defaults to 1000.
        """
        self.queue: Queue = Queue(maxsize=max_size)
        self.max_size = max_size

    def put(self, event: Event) -> None:
        """Put an event in the queue.

        Args:
            event: The event to put in the queue.

        Raises:
            Full: If the queue is full.
        """
        self.queue.put(event, block=False)

    def get(self) -> Event:
        """Get an event from the queue.

        Returns:
            Event: The event from the queue.

        Raises:
            Empty: If the queue is empty.
        """
        return self.queue.get(block=True)

    def get_nowait(self) -> Optional[Event]:
        """Get an event from the queue without blocking.

        Returns:
            Optional[Event]: The event from the queue, or None if the queue is empty.
        """
        try:
            return self.queue.get(block=False)
        except Exception:
            return None

    def task_done(self) -> None:
        """Mark a task as done."""
        self.queue.task_done()

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self.queue.empty()

    def full(self) -> bool:
        """Check if the queue is full.

        Returns:
            bool: True if the queue is full, False otherwise.
        """
        return self.queue.full()

    def qsize(self) -> int:
        """Get the size of the queue.

        Returns:
            int: The size of the queue.
        """
        return self.queue.qsize()


class EventHandler:
    """Handler for processing events.

    Attributes:
        callback: The callback function to call when an event is received.
        event_types: The types of events to handle.
        filter_func: A function to filter events.
    """

    def __init__(
        self,
        callback: Callable[[Event], None],
        event_types: Optional[List[str]] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ):
        """Initialize an event handler.

        Args:
            callback: The callback function to call when an event is received.
            event_types: The types of events to handle. Defaults to None.
            filter_func: A function to filter events. Defaults to None.
        """
        self.callback = callback
        self.event_types = set(event_types) if event_types else set()
        self.filter_func = filter_func

    def handles_event_type(self, event_type: str) -> bool:
        """Check if the handler handles a specific event type.

        Args:
            event_type: The event type to check.

        Returns:
            bool: True if the handler handles the event type, False otherwise.
        """
        return not self.event_types or event_type in self.event_types

    def should_handle(self, event: Event) -> bool:
        """Check if the handler should handle an event.

        Args:
            event: The event to check.

        Returns:
            bool: True if the handler should handle the event, False otherwise.
        """
        if not self.handles_event_type(event.event_type):
            return False

        if self.filter_func and not self.filter_func(event):
            return False

        return True

    def handle(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: The event to handle.
        """
        if self.should_handle(event):
            try:
                self.callback(event)
            except Exception as e:
                logger.error(f"Error handling event {event.id}: {str(e)}")


class EventBus:
    """Event bus for distributing events to handlers.

    Attributes:
        handlers: The registered event handlers.
        event_queue: The event queue.
        worker_thread: The worker thread for processing events.
        running: Whether the event bus is running.
    """

    def __init__(self, max_queue_size: int = 1000):
        """Initialize an event bus.

        Args:
            max_queue_size: The maximum size of the event queue. Defaults to 1000.
        """
        self.handlers: List[EventHandler] = []
        self.event_queue = EventQueue(max_size=max_queue_size)
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()

    def register_handler(
        self,
        callback: Callable[[Event], None],
        event_types: Optional[List[str]] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ) -> None:
        """Register an event handler.

        Args:
            callback: The callback function to call when an event is received.
            event_types: The types of events to handle. Defaults to None.
            filter_func: A function to filter events. Defaults to None.
        """
        with self._lock:
            handler = EventHandler(callback, event_types, filter_func)
            self.handlers.append(handler)

    def unregister_handler(self, callback: Callable[[Event], None]) -> None:
        """Unregister an event handler.

        Args:
            callback: The callback function to unregister.
        """
        with self._lock:
            self.handlers = [h for h in self.handlers if h.callback != callback]

    def publish(self, event: Event) -> None:
        """Publish an event to the event bus.

        Args:
            event: The event to publish.

        Raises:
            RuntimeError: If the event bus is not running.
        """
        if not self.running:
            raise RuntimeError("Event bus is not running")

        try:
            self.event_queue.put(event)
            logger.debug(f"Published event {event.id} of type {event.event_type}")
        except Exception as e:
            logger.error(f"Error publishing event {event.id}: {str(e)}")

    def _process_events(self) -> None:
        """Process events from the event queue."""
        while self.running:
            try:
                event = self.event_queue.get()
                if event is None:
                    continue

                logger.debug(f"Processing event {event.id} of type {event.event_type}")

                with self._lock:
                    handlers_copy = self.handlers.copy()

                for handler in handlers_copy:
                    try:
                        handler.handle(event)
                    except Exception as e:
                        logger.error(
                            f"Error in handler for event {event.id}: {str(e)}"
                        )

                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")

    def start(self) -> None:
        """Start the event bus."""
        if self.running:
            return

        with self._lock:
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._process_events, daemon=True
            )
            self.worker_thread.start()
            logger.info("Event bus started")

    def stop(self) -> None:
        """Stop the event bus."""
        if not self.running:
            return

        with self._lock:
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=1.0)
                self.worker_thread = None
            logger.info("Event bus stopped")


class EventSystem:
    """Event system for the Friday AI Trading System.

    This class provides an event-driven architecture with message queueing
    for handling events across the system.

    Attributes:
        event_bus: The event bus for distributing events.
        event_store: The store for persisting events.
    """

    def __init__(self, max_queue_size: int = 1000, enable_persistence: bool = False):
        """Initialize the event system.

        Args:
            max_queue_size: The maximum size of the event queue. Defaults to 1000.
            enable_persistence: Whether to enable event persistence. Defaults to False.
        """
        self.event_bus = EventBus(max_queue_size=max_queue_size)
        self.event_store = EventStore() if enable_persistence else None
        self._started = False

    def start(self) -> None:
        """Start the event system."""
        if self._started:
            return

        self.event_bus.start()
        if self.event_store:
            self.event_store.start()
        self._started = True
        logger.info("Event system started")

    def stop(self) -> None:
        """Stop the event system."""
        if not self._started:
            return

        self.event_bus.stop()
        if self.event_store:
            self.event_store.stop()
        self._started = False
        logger.info("Event system stopped")

    def emit(self, event_type: str, data: Any, source: Optional[str] = None) -> Event:
        """Emit an event.

        Args:
            event_type: The type of the event.
            data: The data associated with the event.
            source: The source of the event. Defaults to None.

        Returns:
            Event: The emitted event.

        Raises:
            RuntimeError: If the event system is not started.
        """
        if not self._started:
            raise RuntimeError("Event system is not started")

        event = Event(event_type=event_type, data=data, source=source)
        self.event_bus.publish(event)

        if self.event_store:
            self.event_store.store_event(event)

        return event

    def register_handler(
        self,
        callback: Callable[[Event], None],
        event_types: Optional[List[str]] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ) -> None:
        """Register an event handler.

        Args:
            callback: The callback function to call when an event is received.
            event_types: The types of events to handle. Defaults to None.
            filter_func: A function to filter events. Defaults to None.
        """
        self.event_bus.register_handler(callback, event_types, filter_func)

    def unregister_handler(self, callback: Callable[[Event], None]) -> None:
        """Unregister an event handler.

        Args:
            callback: The callback function to unregister.
        """
        self.event_bus.unregister_handler(callback)


class EventStore:
    """Store for persisting events.

    Attributes:
        events: The stored events.
        max_events: The maximum number of events to store.
        persistence_thread: The thread for persisting events.
        running: Whether the event store is running.
    """

    def __init__(self, max_events: int = 10000):
        """Initialize an event store.

        Args:
            max_events: The maximum number of events to store. Defaults to 10000.
        """
        self.events: Dict[str, List[Event]] = defaultdict(list)
        self.max_events = max_events
        self.event_queue = EventQueue()
        self.persistence_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()

    def store_event(self, event: Event) -> None:
        """Store an event.

        Args:
            event: The event to store.
        """
        if not self.running:
            return

        try:
            self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Error queueing event for storage: {str(e)}")

    def _persist_events(self) -> None:
        """Persist events from the queue."""
        while self.running:
            try:
                event = self.event_queue.get()
                if event is None:
                    continue

                with self._lock:
                    # Add event to the appropriate list
                    event_list = self.events[event.event_type]
                    event_list.append(event)

                    # Trim if necessary
                    if len(event_list) > self.max_events:
                        self.events[event.event_type] = event_list[-self.max_events:]

                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"Error persisting event: {str(e)}")

    def get_events(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> List[Event]:
        """Get stored events.

        Args:
            event_type: The type of events to get. If None, get all events.
            limit: The maximum number of events to get. Defaults to 100.

        Returns:
            List[Event]: The stored events.
        """
        with self._lock:
            if event_type:
                events = self.events.get(event_type, [])
                return events[-limit:] if events else []
            else:
                # Flatten all events and sort by timestamp
                all_events = []
                for event_list in self.events.values():
                    all_events.extend(event_list)

                # Sort by timestamp (newest first) and limit
                all_events.sort(key=lambda e: e.timestamp, reverse=True)
                return all_events[:limit]

    def clear_events(self, event_type: Optional[str] = None) -> None:
        """Clear stored events.

        Args:
            event_type: The type of events to clear. If None, clear all events.
        """
        with self._lock:
            if event_type:
                self.events[event_type] = []
            else:
                self.events.clear()

    def start(self) -> None:
        """Start the event store."""
        if self.running:
            return

        with self._lock:
            self.running = True
            self.persistence_thread = threading.Thread(
                target=self._persist_events, daemon=True
            )
            self.persistence_thread.start()
            logger.info("Event store started")

    def stop(self) -> None:
        """Stop the event store."""
        if not self.running:
            return

        with self._lock:
            self.running = False
            if self.persistence_thread:
                self.persistence_thread.join(timeout=1.0)
                self.persistence_thread = None
            logger.info("Event store stopped")