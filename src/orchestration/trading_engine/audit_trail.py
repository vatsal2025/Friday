"""Audit Trail for Trading Engine.

This module provides comprehensive audit trail functionality for tracking all trading activities,
including order submissions, executions, cancellations, modifications, and system events.
The audit trail is essential for compliance, debugging, and performance analysis.
"""

import time
import datetime
import json
import os
import csv
import logging
import uuid
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import deque

# Configure logger
logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of events that can be recorded in the audit trail."""
    # Order lifecycle events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"
    ORDER_MODIFIED = "order_modified"
    ORDER_REPLACED = "order_replaced"
    
    # Trade lifecycle events
    TRADE_CREATED = "trade_created"
    TRADE_UPDATED = "trade_updated"
    TRADE_COMPLETED = "trade_completed"
    TRADE_CANCELLED = "trade_cancelled"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_PROCESSED = "signal_processed"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Execution strategy events
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_UPDATED = "strategy_updated"
    STRATEGY_COMPLETED = "strategy_completed"
    STRATEGY_FAILED = "strategy_failed"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_INFO = "system_info"
    
    # Market events
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    MARKET_HALT = "market_halt"
    MARKET_RESUME = "market_resume"
    
    # Risk management events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    
    # Validation events
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    
    # Emergency events
    EMERGENCY_DECLARED = "emergency_declared"
    EMERGENCY_RESOLVED = "emergency_resolved"
    TRADING_PAUSED = "trading_paused"
    TRADING_RESUMED = "trading_resumed"
    TRADING_HALTED = "trading_halted"
    
    # Configuration events
    CONFIG_LOADED = "config_loaded"
    CONFIG_UPDATED = "config_updated"
    
    # Custom event
    CUSTOM = "custom"


class AuditEvent:
    """Represents a single event in the audit trail."""
    def __init__(
        self,
        event_type: AuditEventType,
        event_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        source: str = "trading_engine",
        user: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        related_ids: Optional[Dict[str, str]] = None,
        severity: str = "info"
    ):
        """Initialize an audit event.
        
        Args:
            event_type: Type of event from AuditEventType enum
            event_id: Unique identifier for the event (generated if None)
            timestamp: Event timestamp (current time if None)
            source: Source component that generated the event
            user: User associated with the event (if applicable)
            details: Additional event details
            related_ids: Dictionary of related identifiers (order_id, trade_id, etc.)
            severity: Event severity (info, warning, error, critical)
        """
        self.event_type = event_type
        self.event_id = event_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.source = source
        self.user = user
        self.details = details or {}
        self.related_ids = related_ids or {}
        self.severity = severity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.datetime.fromtimestamp(self.timestamp).isoformat(),
            "source": self.source,
            "user": self.user,
            "details": self.details,
            "related_ids": self.related_ids,
            "severity": self.severity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create an event from a dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            AuditEvent instance
        """
        # Convert string event_type to enum
        event_type_str = data.get("event_type")
        event_type = AuditEventType(event_type_str) if event_type_str else AuditEventType.CUSTOM
        
        return cls(
            event_type=event_type,
            event_id=data.get("event_id"),
            timestamp=data.get("timestamp"),
            source=data.get("source", "unknown"),
            user=data.get("user"),
            details=data.get("details", {}),
            related_ids=data.get("related_ids", {}),
            severity=data.get("severity", "info")
        )


class AuditTrail:
    """Manages the audit trail for the trading engine."""
    def __init__(
        self,
        max_in_memory_events: int = 10000,
        audit_dir: Optional[str] = None,
        auto_flush_interval: Optional[int] = 300,  # 5 minutes
        auto_flush_count: Optional[int] = 1000,
        include_details_in_log: bool = False
    ):
        """Initialize the audit trail.
        
        Args:
            max_in_memory_events: Maximum number of events to keep in memory
            audit_dir: Directory to store audit files
            auto_flush_interval: Interval in seconds for auto-flushing events to disk
            auto_flush_count: Number of events that triggers auto-flush
            include_details_in_log: Whether to include event details in log messages
        """
        self.events = deque(maxlen=max_in_memory_events)
        self.audit_dir = audit_dir
        self.auto_flush_interval = auto_flush_interval
        self.auto_flush_count = auto_flush_count
        self.include_details_in_log = include_details_in_log
        
        # Create audit directory if specified
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        
        # Event counts by type
        self.event_counts: Dict[AuditEventType, int] = {event_type: 0 for event_type in AuditEventType}
        
        # Event filters for callbacks
        self.event_callbacks: Dict[AuditEventType, List[callable]] = {}
        
        # Auto-flush mechanism
        self.last_flush_time = time.time()
        self.events_since_flush = 0
        self.flush_lock = threading.Lock()
        self.auto_flush_thread = None
        self.running = False
        
        # Start auto-flush thread if interval is specified
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
                    self.flush_to_disk()
            except Exception as e:
                logger.error(f"Error in audit trail auto-flush: {e}")
    
    def record_event(self, event: AuditEvent) -> str:
        """Record an event in the audit trail.
        
        Args:
            event: The event to record
            
        Returns:
            Event ID
        """
        # Add event to the in-memory queue
        self.events.append(event)
        
        # Update event count
        self.event_counts[event.event_type] = self.event_counts.get(event.event_type, 0) + 1
        self.events_since_flush += 1
        
        # Log the event
        self._log_event(event)
        
        # Trigger callbacks for this event type
        self._trigger_callbacks(event)
        
        # Check if auto-flush is needed
        if self.auto_flush_count and self.events_since_flush >= self.auto_flush_count:
            self.flush_to_disk()
        
        return event.event_id
    
    def create_event(self, 
                     event_type: AuditEventType, 
                     details: Optional[Dict[str, Any]] = None, 
                     related_ids: Optional[Dict[str, str]] = None,
                     source: str = "trading_engine",
                     user: Optional[str] = None,
                     severity: str = "info") -> str:
        """Create and record an event.
        
        Args:
            event_type: Type of event
            details: Event details
            related_ids: Related identifiers
            source: Event source
            user: Associated user
            severity: Event severity
            
        Returns:
            Event ID
        """
        event = AuditEvent(
            event_type=event_type,
            details=details,
            related_ids=related_ids,
            source=source,
            user=user,
            severity=severity
        )
        
        return self.record_event(event)
    
    def _log_event(self, event: AuditEvent) -> None:
        """Log an event based on its severity.
        
        Args:
            event: The event to log
        """
        # Determine log level based on severity
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(event.severity.lower(), logging.INFO)
        
        # Create log message
        log_message = f"[{event.event_type.value}] {event.source}"
        
        # Add related IDs to log message
        if event.related_ids:
            id_str = ", ".join(f"{k}={v}" for k, v in event.related_ids.items())
            log_message += f" ({id_str})"
        
        # Add details if configured
        if self.include_details_in_log and event.details:
            # Limit details to prevent huge log messages
            details_str = str(event.details)
            if len(details_str) > 200:
                details_str = details_str[:197] + "..."
            log_message += f": {details_str}"
        
        # Log the message
        logger.log(log_level, log_message)
    
    def _trigger_callbacks(self, event: AuditEvent) -> None:
        """Trigger callbacks registered for this event type.
        
        Args:
            event: The event that triggered callbacks
        """
        # Get callbacks for this event type
        callbacks = self.event_callbacks.get(event.event_type, [])
        
        # Also get callbacks for ALL event types
        all_callbacks = self.event_callbacks.get(None, [])
        
        # Combine callbacks
        all_callbacks = callbacks + all_callbacks
        
        # Trigger callbacks
        for callback in all_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in audit event callback: {e}")
    
    def register_callback(self, callback: callable, event_type: Optional[AuditEventType] = None) -> None:
        """Register a callback for a specific event type.
        
        Args:
            callback: Function to call when event occurs
            event_type: Event type to trigger callback (None for all events)
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        
        self.event_callbacks[event_type].append(callback)
    
    def unregister_callback(self, callback: callable, event_type: Optional[AuditEventType] = None) -> bool:
        """Unregister a callback.
        
        Args:
            callback: Callback to unregister
            event_type: Event type the callback was registered for
            
        Returns:
            True if callback was found and removed, False otherwise
        """
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
            return True
        return False
    
    def get_events(self, 
                  event_types: Optional[List[AuditEventType]] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  sources: Optional[List[str]] = None,
                  related_id_filters: Optional[Dict[str, str]] = None,
                  max_events: Optional[int] = None) -> List[AuditEvent]:
        """Get events matching specified filters.
        
        Args:
            event_types: List of event types to include
            start_time: Start time for filtering events
            end_time: End time for filtering events
            sources: List of sources to include
            related_id_filters: Dictionary of related ID filters
            max_events: Maximum number of events to return
            
        Returns:
            List of matching events
        """
        # Start with all events
        filtered_events = list(self.events)
        
        # Filter by event type
        if event_types:
            filtered_events = [e for e in filtered_events if e.event_type in event_types]
        
        # Filter by time range
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Filter by source
        if sources:
            filtered_events = [e for e in filtered_events if e.source in sources]
        
        # Filter by related IDs
        if related_id_filters:
            for key, value in related_id_filters.items():
                filtered_events = [e for e in filtered_events if e.related_ids.get(key) == value]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Limit number of events
        if max_events and len(filtered_events) > max_events:
            filtered_events = filtered_events[:max_events]
        
        return filtered_events
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get counts of events by type.
        
        Returns:
            Dictionary mapping event type names to counts
        """
        return {event_type.value: count for event_type, count in self.event_counts.items()}
    
    def get_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get an event by its ID.
        
        Args:
            event_id: Event ID to find
            
        Returns:
            AuditEvent if found, None otherwise
        """
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
    
    def get_related_events(self, id_type: str, id_value: str) -> List[AuditEvent]:
        """Get events related to a specific ID.
        
        Args:
            id_type: Type of ID (order_id, trade_id, etc.)
            id_value: Value of the ID
            
        Returns:
            List of related events
        """
        return [e for e in self.events if e.related_ids.get(id_type) == id_value]
    
    def flush_to_disk(self) -> Optional[str]:
        """Flush events to disk.
        
        Returns:
            Path to the audit file or None if audit_dir is not set
        """
        if not self.audit_dir or not self.events:
            return None
        
        with self.flush_lock:
            # Create a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audit_file = os.path.join(self.audit_dir, f"audit_{timestamp}.json")
            
            # Convert events to dictionaries
            events_data = [event.to_dict() for event in self.events]
            
            # Add metadata
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event_count": len(events_data),
                "event_counts_by_type": self.get_event_counts()
            }
            
            # Create the full data structure
            data = {
                "metadata": metadata,
                "events": events_data
            }
            
            # Write to file
            with open(audit_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Reset counters
            self.last_flush_time = time.time()
            self.events_since_flush = 0
            
            logger.info(f"Audit trail flushed to {audit_file} ({len(events_data)} events)")
            return audit_file
    
    def export_csv(self, output_file: str, events: Optional[List[AuditEvent]] = None) -> str:
        """Export events to a CSV file.
        
        Args:
            output_file: Path to the output CSV file
            events: Events to export (all in-memory events if None)
            
        Returns:
            Path to the CSV file
        """
        if events is None:
            events = list(self.events)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write to CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Event ID", "Event Type", "Timestamp", "ISO Timestamp", 
                "Source", "User", "Severity", "Related IDs", "Details"
            ])
            
            # Write data
            for event in sorted(events, key=lambda e: e.timestamp):
                writer.writerow([
                    event.event_id,
                    event.event_type.value,
                    event.timestamp,
                    datetime.datetime.fromtimestamp(event.timestamp).isoformat(),
                    event.source,
                    event.user or "",
                    event.severity,
                    json.dumps(event.related_ids),
                    json.dumps(event.details)
                ])
        
        return output_file
    
    def clear(self) -> None:
        """Clear all in-memory events."""
        self.events.clear()
        self.event_counts = {event_type: 0 for event_type in AuditEventType}
        self.events_since_flush = 0
    
    def shutdown(self) -> None:
        """Shutdown the audit trail and flush events to disk."""
        # Stop auto-flush thread
        self.stop_auto_flush()
        
        # Final flush to disk
        if self.audit_dir:
            self.flush_to_disk()


class OrderAuditTrail:
    """Helper class for recording order-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the order audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_order_created(self, 
                            order_id: str, 
                            order_details: Dict[str, Any],
                            source: str = "order_manager",
                            user: Optional[str] = None) -> str:
        """Record an order created event.
        
        Args:
            order_id: Order identifier
            order_details: Order details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_CREATED,
            details=order_details,
            related_ids={"order_id": order_id},
            source=source,
            user=user
        )
    
    def record_order_submitted(self, 
                              order_id: str, 
                              broker_details: Dict[str, Any],
                              source: str = "order_manager",
                              user: Optional[str] = None) -> str:
        """Record an order submitted event.
        
        Args:
            order_id: Order identifier
            broker_details: Broker submission details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            details=broker_details,
            related_ids={"order_id": order_id},
            source=source,
            user=user
        )
    
    def record_order_accepted(self, 
                             order_id: str, 
                             broker_order_id: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "broker_service",
                             user: Optional[str] = None) -> str:
        """Record an order accepted event.
        
        Args:
            order_id: Order identifier
            broker_order_id: Broker's order identifier
            details: Additional details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"order_id": order_id}
        if broker_order_id:
            related_ids["broker_order_id"] = broker_order_id
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_ACCEPTED,
            details=details or {},
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_order_rejected(self, 
                             order_id: str, 
                             reason: str,
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "broker_service",
                             user: Optional[str] = None) -> str:
        """Record an order rejected event.
        
        Args:
            order_id: Order identifier
            reason: Rejection reason
            details: Additional details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["rejection_reason"] = reason
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_REJECTED,
            details=event_details,
            related_ids={"order_id": order_id},
            source=source,
            user=user,
            severity="warning"
        )
    
    def record_order_filled(self, 
                           order_id: str, 
                           fill_details: Dict[str, Any],
                           source: str = "broker_service",
                           user: Optional[str] = None) -> str:
        """Record an order filled event.
        
        Args:
            order_id: Order identifier
            fill_details: Fill details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"order_id": order_id}
        
        # Add trade_id if available
        if "trade_id" in fill_details:
            related_ids["trade_id"] = fill_details["trade_id"]
        
        # Add execution_id if available
        if "execution_id" in fill_details:
            related_ids["execution_id"] = fill_details["execution_id"]
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_FILLED,
            details=fill_details,
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_order_partially_filled(self, 
                                     order_id: str, 
                                     fill_details: Dict[str, Any],
                                     source: str = "broker_service",
                                     user: Optional[str] = None) -> str:
        """Record an order partially filled event.
        
        Args:
            order_id: Order identifier
            fill_details: Fill details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"order_id": order_id}
        
        # Add trade_id if available
        if "trade_id" in fill_details:
            related_ids["trade_id"] = fill_details["trade_id"]
        
        # Add execution_id if available
        if "execution_id" in fill_details:
            related_ids["execution_id"] = fill_details["execution_id"]
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_PARTIALLY_FILLED,
            details=fill_details,
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_order_cancelled(self, 
                              order_id: str, 
                              details: Optional[Dict[str, Any]] = None,
                              source: str = "order_manager",
                              user: Optional[str] = None) -> str:
        """Record an order cancelled event.
        
        Args:
            order_id: Order identifier
            details: Cancellation details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_CANCELLED,
            details=details or {},
            related_ids={"order_id": order_id},
            source=source,
            user=user
        )
    
    def record_order_expired(self, 
                            order_id: str, 
                            details: Optional[Dict[str, Any]] = None,
                            source: str = "broker_service",
                            user: Optional[str] = None) -> str:
        """Record an order expired event.
        
        Args:
            order_id: Order identifier
            details: Expiration details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_EXPIRED,
            details=details or {},
            related_ids={"order_id": order_id},
            source=source,
            user=user
        )
    
    def record_order_modified(self, 
                             order_id: str, 
                             modification_details: Dict[str, Any],
                             source: str = "order_manager",
                             user: Optional[str] = None) -> str:
        """Record an order modified event.
        
        Args:
            order_id: Order identifier
            modification_details: Modification details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.ORDER_MODIFIED,
            details=modification_details,
            related_ids={"order_id": order_id},
            source=source,
            user=user
        )


class TradeAuditTrail:
    """Helper class for recording trade-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the trade audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_trade_created(self, 
                            trade_id: str, 
                            trade_details: Dict[str, Any],
                            source: str = "trade_lifecycle_manager",
                            user: Optional[str] = None) -> str:
        """Record a trade created event.
        
        Args:
            trade_id: Trade identifier
            trade_details: Trade details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"trade_id": trade_id}
        
        # Add order_id if available
        if "order_id" in trade_details:
            related_ids["order_id"] = trade_details["order_id"]
        
        # Add signal_id if available
        if "signal_id" in trade_details:
            related_ids["signal_id"] = trade_details["signal_id"]
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADE_CREATED,
            details=trade_details,
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_trade_updated(self, 
                            trade_id: str, 
                            update_details: Dict[str, Any],
                            source: str = "trade_lifecycle_manager",
                            user: Optional[str] = None) -> str:
        """Record a trade updated event.
        
        Args:
            trade_id: Trade identifier
            update_details: Update details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADE_UPDATED,
            details=update_details,
            related_ids={"trade_id": trade_id},
            source=source,
            user=user
        )
    
    def record_trade_completed(self, 
                              trade_id: str, 
                              completion_details: Dict[str, Any],
                              source: str = "trade_lifecycle_manager",
                              user: Optional[str] = None) -> str:
        """Record a trade completed event.
        
        Args:
            trade_id: Trade identifier
            completion_details: Completion details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADE_COMPLETED,
            details=completion_details,
            related_ids={"trade_id": trade_id},
            source=source,
            user=user
        )
    
    def record_trade_cancelled(self, 
                              trade_id: str, 
                              cancellation_details: Dict[str, Any],
                              source: str = "trade_lifecycle_manager",
                              user: Optional[str] = None) -> str:
        """Record a trade cancelled event.
        
        Args:
            trade_id: Trade identifier
            cancellation_details: Cancellation details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADE_CANCELLED,
            details=cancellation_details,
            related_ids={"trade_id": trade_id},
            source=source,
            user=user,
            severity="warning"
        )


class SignalAuditTrail:
    """Helper class for recording signal-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the signal audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_signal_generated(self, 
                               signal_id: str, 
                               signal_details: Dict[str, Any],
                               source: str = "signal_generator",
                               user: Optional[str] = None) -> str:
        """Record a signal generated event.
        
        Args:
            signal_id: Signal identifier
            signal_details: Signal details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"signal_id": signal_id}
        
        # Add model_id if available
        if "model_id" in signal_details:
            related_ids["model_id"] = signal_details["model_id"]
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            details=signal_details,
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_signal_processed(self, 
                               signal_id: str, 
                               processing_details: Dict[str, Any],
                               source: str = "trading_engine",
                               user: Optional[str] = None) -> str:
        """Record a signal processed event.
        
        Args:
            signal_id: Signal identifier
            processing_details: Processing details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        related_ids = {"signal_id": signal_id}
        
        # Add order_id if available
        if "order_id" in processing_details:
            related_ids["order_id"] = processing_details["order_id"]
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SIGNAL_PROCESSED,
            details=processing_details,
            related_ids=related_ids,
            source=source,
            user=user
        )
    
    def record_signal_rejected(self, 
                              signal_id: str, 
                              reason: str,
                              details: Optional[Dict[str, Any]] = None,
                              source: str = "trading_engine",
                              user: Optional[str] = None) -> str:
        """Record a signal rejected event.
        
        Args:
            signal_id: Signal identifier
            reason: Rejection reason
            details: Additional details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["rejection_reason"] = reason
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SIGNAL_REJECTED,
            details=event_details,
            related_ids={"signal_id": signal_id},
            source=source,
            user=user,
            severity="warning"
        )


class SystemAuditTrail:
    """Helper class for recording system-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the system audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_system_startup(self, 
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "trading_engine",
                             user: Optional[str] = None) -> str:
        """Record a system startup event.
        
        Args:
            details: Startup details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            details=details or {},
            source=source,
            user=user
        )
    
    def record_system_shutdown(self, 
                              details: Optional[Dict[str, Any]] = None,
                              source: str = "trading_engine",
                              user: Optional[str] = None) -> str:
        """Record a system shutdown event.
        
        Args:
            details: Shutdown details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            details=details or {},
            source=source,
            user=user
        )
    
    def record_system_error(self, 
                           error_message: str,
                           details: Optional[Dict[str, Any]] = None,
                           source: str = "trading_engine",
                           user: Optional[str] = None) -> str:
        """Record a system error event.
        
        Args:
            error_message: Error message
            details: Error details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["error_message"] = error_message
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            details=event_details,
            source=source,
            user=user,
            severity="error"
        )
    
    def record_system_warning(self, 
                             warning_message: str,
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "trading_engine",
                             user: Optional[str] = None) -> str:
        """Record a system warning event.
        
        Args:
            warning_message: Warning message
            details: Warning details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["warning_message"] = warning_message
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SYSTEM_WARNING,
            details=event_details,
            source=source,
            user=user,
            severity="warning"
        )
    
    def record_system_info(self, 
                          info_message: str,
                          details: Optional[Dict[str, Any]] = None,
                          source: str = "trading_engine",
                          user: Optional[str] = None) -> str:
        """Record a system info event.
        
        Args:
            info_message: Info message
            details: Info details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["info_message"] = info_message
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.SYSTEM_INFO,
            details=event_details,
            source=source,
            user=user
        )
    
    def record_config_loaded(self, 
                            config_details: Dict[str, Any],
                            source: str = "trading_engine",
                            user: Optional[str] = None) -> str:
        """Record a configuration loaded event.
        
        Args:
            config_details: Configuration details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.CONFIG_LOADED,
            details=config_details,
            source=source,
            user=user
        )
    
    def record_config_updated(self, 
                             config_details: Dict[str, Any],
                             source: str = "trading_engine",
                             user: Optional[str] = None) -> str:
        """Record a configuration updated event.
        
        Args:
            config_details: Configuration details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.CONFIG_UPDATED,
            details=config_details,
            source=source,
            user=user
        )


class RiskAuditTrail:
    """Helper class for recording risk-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the risk audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_risk_limit_breach(self, 
                                limit_type: str,
                                limit_value: float,
                                actual_value: float,
                                details: Optional[Dict[str, Any]] = None,
                                source: str = "risk_manager",
                                user: Optional[str] = None) -> str:
        """Record a risk limit breach event.
        
        Args:
            limit_type: Type of limit breached
            limit_value: Value of the limit
            actual_value: Actual value that breached the limit
            details: Additional details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "limit_type": limit_type,
            "limit_value": limit_value,
            "actual_value": actual_value,
            "breach_percentage": (actual_value / limit_value) * 100 if limit_value != 0 else float('inf')
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.RISK_LIMIT_BREACH,
            details=event_details,
            source=source,
            user=user,
            severity="error"
        )
    
    def record_risk_check_passed(self, 
                                check_type: str,
                                details: Optional[Dict[str, Any]] = None,
                                related_ids: Optional[Dict[str, str]] = None,
                                source: str = "risk_manager",
                                user: Optional[str] = None) -> str:
        """Record a risk check passed event.
        
        Args:
            check_type: Type of risk check
            details: Check details
            related_ids: Related identifiers
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["check_type"] = check_type
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.RISK_CHECK_PASSED,
            details=event_details,
            related_ids=related_ids or {},
            source=source,
            user=user
        )
    
    def record_risk_check_failed(self, 
                                check_type: str,
                                reason: str,
                                details: Optional[Dict[str, Any]] = None,
                                related_ids: Optional[Dict[str, str]] = None,
                                source: str = "risk_manager",
                                user: Optional[str] = None) -> str:
        """Record a risk check failed event.
        
        Args:
            check_type: Type of risk check
            reason: Failure reason
            details: Check details
            related_ids: Related identifiers
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "check_type": check_type,
            "failure_reason": reason
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.RISK_CHECK_FAILED,
            details=event_details,
            related_ids=related_ids or {},
            source=source,
            user=user,
            severity="warning"
        )


class ValidationAuditTrail:
    """Helper class for recording validation-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the validation audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_validation_passed(self, 
                                validation_type: str,
                                entity_type: str,
                                details: Optional[Dict[str, Any]] = None,
                                related_ids: Optional[Dict[str, str]] = None,
                                source: str = "validator",
                                user: Optional[str] = None) -> str:
        """Record a validation passed event.
        
        Args:
            validation_type: Type of validation
            entity_type: Type of entity validated
            details: Validation details
            related_ids: Related identifiers
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "validation_type": validation_type,
            "entity_type": entity_type
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.VALIDATION_PASSED,
            details=event_details,
            related_ids=related_ids or {},
            source=source,
            user=user
        )
    
    def record_validation_failed(self, 
                                validation_type: str,
                                entity_type: str,
                                reason: str,
                                details: Optional[Dict[str, Any]] = None,
                                related_ids: Optional[Dict[str, str]] = None,
                                source: str = "validator",
                                user: Optional[str] = None) -> str:
        """Record a validation failed event.
        
        Args:
            validation_type: Type of validation
            entity_type: Type of entity validated
            reason: Failure reason
            details: Validation details
            related_ids: Related identifiers
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "validation_type": validation_type,
            "entity_type": entity_type,
            "failure_reason": reason
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.VALIDATION_FAILED,
            details=event_details,
            related_ids=related_ids or {},
            source=source,
            user=user,
            severity="warning"
        )


class EmergencyAuditTrail:
    """Helper class for recording emergency-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the emergency audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_emergency_declared(self, 
                                 emergency_level: str,
                                 reason: str,
                                 details: Optional[Dict[str, Any]] = None,
                                 source: str = "emergency_handler",
                                 user: Optional[str] = None) -> str:
        """Record an emergency declared event.
        
        Args:
            emergency_level: Emergency level
            reason: Emergency reason
            details: Emergency details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "emergency_level": emergency_level,
            "reason": reason
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.EMERGENCY_DECLARED,
            details=event_details,
            source=source,
            user=user,
            severity="critical" if emergency_level.lower() in ["critical", "severe"] else "error"
        )
    
    def record_emergency_resolved(self, 
                                emergency_level: str,
                                details: Optional[Dict[str, Any]] = None,
                                source: str = "emergency_handler",
                                user: Optional[str] = None) -> str:
        """Record an emergency resolved event.
        
        Args:
            emergency_level: Emergency level that was resolved
            details: Resolution details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["emergency_level"] = emergency_level
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.EMERGENCY_RESOLVED,
            details=event_details,
            source=source,
            user=user
        )
    
    def record_trading_paused(self, 
                             reason: str,
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "emergency_handler",
                             user: Optional[str] = None) -> str:
        """Record a trading paused event.
        
        Args:
            reason: Pause reason
            details: Pause details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["reason"] = reason
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADING_PAUSED,
            details=event_details,
            source=source,
            user=user,
            severity="warning"
        )
    
    def record_trading_resumed(self, 
                              details: Optional[Dict[str, Any]] = None,
                              source: str = "emergency_handler",
                              user: Optional[str] = None) -> str:
        """Record a trading resumed event.
        
        Args:
            details: Resume details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADING_RESUMED,
            details=details or {},
            source=source,
            user=user
        )
    
    def record_trading_halted(self, 
                             reason: str,
                             details: Optional[Dict[str, Any]] = None,
                             source: str = "emergency_handler",
                             user: Optional[str] = None) -> str:
        """Record a trading halted event.
        
        Args:
            reason: Halt reason
            details: Halt details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["reason"] = reason
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.TRADING_HALTED,
            details=event_details,
            source=source,
            user=user,
            severity="error"
        )


class MarketAuditTrail:
    """Helper class for recording market-related audit events."""
    def __init__(self, audit_trail: AuditTrail):
        """Initialize the market audit trail.
        
        Args:
            audit_trail: The main audit trail
        """
        self.audit_trail = audit_trail
    
    def record_market_open(self, 
                          market: str,
                          details: Optional[Dict[str, Any]] = None,
                          source: str = "market_monitor",
                          user: Optional[str] = None) -> str:
        """Record a market open event.
        
        Args:
            market: Market identifier
            details: Open details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["market"] = market
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.MARKET_OPEN,
            details=event_details,
            source=source,
            user=user
        )
    
    def record_market_close(self, 
                           market: str,
                           details: Optional[Dict[str, Any]] = None,
                           source: str = "market_monitor",
                           user: Optional[str] = None) -> str:
        """Record a market close event.
        
        Args:
            market: Market identifier
            details: Close details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["market"] = market
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.MARKET_CLOSE,
            details=event_details,
            source=source,
            user=user
        )
    
    def record_market_halt(self, 
                          market: str,
                          reason: str,
                          details: Optional[Dict[str, Any]] = None,
                          source: str = "market_monitor",
                          user: Optional[str] = None) -> str:
        """Record a market halt event.
        
        Args:
            market: Market identifier
            reason: Halt reason
            details: Halt details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details.update({
            "market": market,
            "reason": reason
        })
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.MARKET_HALT,
            details=event_details,
            source=source,
            user=user,
            severity="warning"
        )
    
    def record_market_resume(self, 
                            market: str,
                            details: Optional[Dict[str, Any]] = None,
                            source: str = "market_monitor",
                            user: Optional[str] = None) -> str:
        """Record a market resume event.
        
        Args:
            market: Market identifier
            details: Resume details
            source: Event source
            user: Associated user
            
        Returns:
            Event ID
        """
        event_details = details or {}
        event_details["market"] = market
        
        return self.audit_trail.create_event(
            event_type=AuditEventType.MARKET_RESUME,
            details=event_details,
            source=source,
            user=user
        )


# Factory function
def create_audit_trail(audit_dir: Optional[str] = None,
                      max_in_memory_events: int = 10000,
                      auto_flush_interval: Optional[int] = 300,
                      auto_flush_count: Optional[int] = 1000,
                      include_details_in_log: bool = False) -> Tuple[AuditTrail, Dict[str, Any]]:
    """Create an audit trail and its helper components.
    
    Args:
        audit_dir: Directory to store audit files
        max_in_memory_events: Maximum number of events to keep in memory
        auto_flush_interval: Interval in seconds for auto-flushing events to disk
        auto_flush_count: Number of events that triggers auto-flush
        include_details_in_log: Whether to include event details in log messages
        
    Returns:
        Tuple containing the main AuditTrail instance and a dictionary of helper components
    """
    # Create the main audit trail
    audit_trail = AuditTrail(
        max_in_memory_events=max_in_memory_events,
        audit_dir=audit_dir,
        auto_flush_interval=auto_flush_interval,
        auto_flush_count=auto_flush_count,
        include_details_in_log=include_details_in_log
    )
    
    # Create helper components
    order_audit = OrderAuditTrail(audit_trail)
    trade_audit = TradeAuditTrail(audit_trail)
    signal_audit = SignalAuditTrail(audit_trail)
    system_audit = SystemAuditTrail(audit_trail)
    risk_audit = RiskAuditTrail(audit_trail)
    validation_audit = ValidationAuditTrail(audit_trail)
    emergency_audit = EmergencyAuditTrail(audit_trail)
    market_audit = MarketAuditTrail(audit_trail)
    
    # Create a dictionary of helper components
    helpers = {
        "order": order_audit,
        "trade": trade_audit,
        "signal": signal_audit,
        "system": system_audit,
        "risk": risk_audit,
        "validation": validation_audit,
        "emergency": emergency_audit,
        "market": market_audit
    }
    
    return audit_trail, helpers