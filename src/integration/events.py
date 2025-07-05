"""Event handling for external system integration.

This module provides event handlers and utilities for external system integration events.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
import logging
import time
from datetime import datetime

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.event import EventSystem, EventHandler, Event

# Create logger
logger = get_logger(__name__)


class IntegrationEventError(FridayError):
    """Exception raised for errors in integration events."""
    pass


# Define event types
CONNECTION_ESTABLISHED = "integration.connection.established"
CONNECTION_LOST = "integration.connection.lost"
CONNECTION_FAILED = "integration.connection.failed"
AUTHENTICATION_SUCCESS = "integration.authentication.success"
AUTHENTICATION_FAILED = "integration.authentication.failed"
DATA_RECEIVED = "integration.data.received"
DATA_SENT = "integration.data.sent"
DATA_ERROR = "integration.data.error"
SYSTEM_REGISTERED = "integration.system.registered"
SYSTEM_UNREGISTERED = "integration.system.unregistered"
SYSTEM_STATUS_CHANGED = "integration.system.status_changed"
SYSTEM_ERROR = "integration.system.error"
MOCK_SERVICE_STARTED = "integration.mock.started"
MOCK_SERVICE_STOPPED = "integration.mock.stopped"
MOCK_REQUEST_RECEIVED = "integration.mock.request_received"
MOCK_RESPONSE_SENT = "integration.mock.response_sent"


class IntegrationEvent(Event):
    """Base class for integration events."""
    
    def __init__(self, event_type: str, system_id: str, timestamp: Optional[float] = None, **kwargs):
        """Initialize an integration event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the external system.
            timestamp: The timestamp of the event (defaults to current time).
            **kwargs: Additional event data.
        """
        super().__init__(event_type)
        self.system_id = system_id
        self.timestamp = timestamp or time.time()
        self.data = kwargs
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        return {
            'event_type': self.event_type,
            'system_id': self.system_id,
            'timestamp': self.timestamp,
            'data': self.data
        }
        
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'IntegrationEvent':
        """Create an event from a dictionary.
        
        Args:
            event_dict: The event dictionary.
            
        Returns:
            IntegrationEvent: The created event.
        """
        event_type = event_dict.get('event_type')
        system_id = event_dict.get('system_id')
        timestamp = event_dict.get('timestamp')
        data = event_dict.get('data', {})
        
        return cls(event_type, system_id, timestamp, **data)


class ConnectionEvent(IntegrationEvent):
    """Event for connection status changes."""
    
    def __init__(self, event_type: str, system_id: str, status: str, 
                 timestamp: Optional[float] = None, **kwargs):
        """Initialize a connection event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the external system.
            status: The connection status.
            timestamp: The timestamp of the event.
            **kwargs: Additional event data.
        """
        super().__init__(event_type, system_id, timestamp, status=status, **kwargs)


class AuthenticationEvent(IntegrationEvent):
    """Event for authentication status changes."""
    
    def __init__(self, event_type: str, system_id: str, success: bool, 
                 timestamp: Optional[float] = None, **kwargs):
        """Initialize an authentication event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the external system.
            success: Whether authentication was successful.
            timestamp: The timestamp of the event.
            **kwargs: Additional event data.
        """
        super().__init__(event_type, system_id, timestamp, success=success, **kwargs)


class DataEvent(IntegrationEvent):
    """Event for data transfer."""
    
    def __init__(self, event_type: str, system_id: str, data_type: str, 
                 data_id: Optional[str] = None, timestamp: Optional[float] = None, **kwargs):
        """Initialize a data event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the external system.
            data_type: The type of data.
            data_id: The ID of the data.
            timestamp: The timestamp of the event.
            **kwargs: Additional event data.
        """
        super().__init__(event_type, system_id, timestamp, data_type=data_type, 
                        data_id=data_id, **kwargs)


class SystemEvent(IntegrationEvent):
    """Event for system status changes."""
    
    def __init__(self, event_type: str, system_id: str, status: Optional[str] = None, 
                 timestamp: Optional[float] = None, **kwargs):
        """Initialize a system event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the external system.
            status: The system status.
            timestamp: The timestamp of the event.
            **kwargs: Additional event data.
        """
        super().__init__(event_type, system_id, timestamp, status=status, **kwargs)


class MockServiceEvent(IntegrationEvent):
    """Event for mock service status changes."""
    
    def __init__(self, event_type: str, system_id: str, service_type: str, 
                 timestamp: Optional[float] = None, **kwargs):
        """Initialize a mock service event.
        
        Args:
            event_type: The type of event.
            system_id: The ID of the mock service.
            service_type: The type of mock service.
            timestamp: The timestamp of the event.
            **kwargs: Additional event data.
        """
        super().__init__(event_type, system_id, timestamp, service_type=service_type, **kwargs)


class IntegrationEventHandler(EventHandler):
    """Base class for integration event handlers."""
    
    def __init__(self, event_system: EventSystem):
        """Initialize an integration event handler.
        
        Args:
            event_system: The event system to register with.
        """
        super().__init__(event_system)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def register(self):
        """Register the handler with the event system."""
        raise NotImplementedError("Subclasses must implement register")


class ConnectionEventHandler(IntegrationEventHandler):
    """Handler for connection events."""
    
    def register(self):
        """Register the handler with the event system."""
        self.event_system.register(CONNECTION_ESTABLISHED, self.handle_connection_established)
        self.event_system.register(CONNECTION_LOST, self.handle_connection_lost)
        self.event_system.register(CONNECTION_FAILED, self.handle_connection_failed)
        
    def handle_connection_established(self, event: IntegrationEvent):
        """Handle a connection established event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        self.logger.info(f"Connection established with system {system_id}")
        
    def handle_connection_lost(self, event: IntegrationEvent):
        """Handle a connection lost event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        self.logger.warning(f"Connection lost with system {system_id}")
        
    def handle_connection_failed(self, event: IntegrationEvent):
        """Handle a connection failed event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        reason = event.data.get('reason', 'Unknown reason')
        self.logger.error(f"Connection failed with system {system_id}: {reason}")


class AuthenticationEventHandler(IntegrationEventHandler):
    """Handler for authentication events."""
    
    def register(self):
        """Register the handler with the event system."""
        self.event_system.register(AUTHENTICATION_SUCCESS, self.handle_authentication_success)
        self.event_system.register(AUTHENTICATION_FAILED, self.handle_authentication_failed)
        
    def handle_authentication_success(self, event: IntegrationEvent):
        """Handle an authentication success event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        self.logger.info(f"Authentication successful with system {system_id}")
        
    def handle_authentication_failed(self, event: IntegrationEvent):
        """Handle an authentication failed event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        reason = event.data.get('reason', 'Unknown reason')
        self.logger.error(f"Authentication failed with system {system_id}: {reason}")


class DataEventHandler(IntegrationEventHandler):
    """Handler for data events."""
    
    def register(self):
        """Register the handler with the event system."""
        self.event_system.register(DATA_RECEIVED, self.handle_data_received)
        self.event_system.register(DATA_SENT, self.handle_data_sent)
        self.event_system.register(DATA_ERROR, self.handle_data_error)
        
    def handle_data_received(self, event: IntegrationEvent):
        """Handle a data received event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        data_type = event.data.get('data_type', 'Unknown')
        data_id = event.data.get('data_id', 'Unknown')
        self.logger.debug(f"Received {data_type} data from system {system_id} (ID: {data_id})")
        
    def handle_data_sent(self, event: IntegrationEvent):
        """Handle a data sent event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        data_type = event.data.get('data_type', 'Unknown')
        data_id = event.data.get('data_id', 'Unknown')
        self.logger.debug(f"Sent {data_type} data to system {system_id} (ID: {data_id})")
        
    def handle_data_error(self, event: IntegrationEvent):
        """Handle a data error event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        data_type = event.data.get('data_type', 'Unknown')
        error = event.data.get('error', 'Unknown error')
        self.logger.error(f"Data error with system {system_id} for {data_type}: {error}")


class SystemEventHandler(IntegrationEventHandler):
    """Handler for system events."""
    
    def register(self):
        """Register the handler with the event system."""
        self.event_system.register(SYSTEM_REGISTERED, self.handle_system_registered)
        self.event_system.register(SYSTEM_UNREGISTERED, self.handle_system_unregistered)
        self.event_system.register(SYSTEM_STATUS_CHANGED, self.handle_system_status_changed)
        self.event_system.register(SYSTEM_ERROR, self.handle_system_error)
        
    def handle_system_registered(self, event: IntegrationEvent):
        """Handle a system registered event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        system_type = event.data.get('system_type', 'Unknown')
        self.logger.info(f"System {system_id} of type {system_type} registered")
        
    def handle_system_unregistered(self, event: IntegrationEvent):
        """Handle a system unregistered event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        self.logger.info(f"System {system_id} unregistered")
        
    def handle_system_status_changed(self, event: IntegrationEvent):
        """Handle a system status changed event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        status = event.data.get('status', 'Unknown')
        self.logger.info(f"System {system_id} status changed to {status}")
        
    def handle_system_error(self, event: IntegrationEvent):
        """Handle a system error event.
        
        Args:
            event: The event to handle.
        """
        system_id = event.system_id
        error = event.data.get('error', 'Unknown error')
        self.logger.error(f"System {system_id} error: {error}")


class MockServiceEventHandler(IntegrationEventHandler):
    """Handler for mock service events."""
    
    def register(self):
        """Register the handler with the event system."""
        self.event_system.register(MOCK_SERVICE_STARTED, self.handle_mock_service_started)
        self.event_system.register(MOCK_SERVICE_STOPPED, self.handle_mock_service_stopped)
        self.event_system.register(MOCK_REQUEST_RECEIVED, self.handle_mock_request_received)
        self.event_system.register(MOCK_RESPONSE_SENT, self.handle_mock_response_sent)
        
    def handle_mock_service_started(self, event: IntegrationEvent):
        """Handle a mock service started event.
        
        Args:
            event: The event to handle.
        """
        service_id = event.system_id
        service_type = event.data.get('service_type', 'Unknown')
        self.logger.info(f"Mock service {service_id} of type {service_type} started")
        
    def handle_mock_service_stopped(self, event: IntegrationEvent):
        """Handle a mock service stopped event.
        
        Args:
            event: The event to handle.
        """
        service_id = event.system_id
        self.logger.info(f"Mock service {service_id} stopped")
        
    def handle_mock_request_received(self, event: IntegrationEvent):
        """Handle a mock request received event.
        
        Args:
            event: The event to handle.
        """
        service_id = event.system_id
        endpoint = event.data.get('endpoint', 'Unknown')
        self.logger.debug(f"Mock service {service_id} received request to {endpoint}")
        
    def handle_mock_response_sent(self, event: IntegrationEvent):
        """Handle a mock response sent event.
        
        Args:
            event: The event to handle.
        """
        service_id = event.system_id
        endpoint = event.data.get('endpoint', 'Unknown')
        status = event.data.get('status', 'Unknown')
        self.logger.debug(f"Mock service {service_id} sent response from {endpoint} with status {status}")


def register_integration_event_handlers(event_system: EventSystem):
    """Register all integration event handlers with the event system.
    
    Args:
        event_system: The event system to register with.
    """
    handlers = [
        ConnectionEventHandler(event_system),
        AuthenticationEventHandler(event_system),
        DataEventHandler(event_system),
        SystemEventHandler(event_system),
        MockServiceEventHandler(event_system)
    ]
    
    for handler in handlers:
        handler.register()
        
    logger.info("Registered integration event handlers")


def create_connection_established_event(system_id: str, **kwargs) -> IntegrationEvent:
    """Create a connection established event.
    
    Args:
        system_id: The ID of the external system.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return ConnectionEvent(CONNECTION_ESTABLISHED, system_id, 'established', **kwargs)


def create_connection_lost_event(system_id: str, **kwargs) -> IntegrationEvent:
    """Create a connection lost event.
    
    Args:
        system_id: The ID of the external system.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return ConnectionEvent(CONNECTION_LOST, system_id, 'lost', **kwargs)


def create_connection_failed_event(system_id: str, reason: str, **kwargs) -> IntegrationEvent:
    """Create a connection failed event.
    
    Args:
        system_id: The ID of the external system.
        reason: The reason for the failure.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return ConnectionEvent(CONNECTION_FAILED, system_id, 'failed', reason=reason, **kwargs)


def create_authentication_success_event(system_id: str, **kwargs) -> IntegrationEvent:
    """Create an authentication success event.
    
    Args:
        system_id: The ID of the external system.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return AuthenticationEvent(AUTHENTICATION_SUCCESS, system_id, True, **kwargs)


def create_authentication_failed_event(system_id: str, reason: str, **kwargs) -> IntegrationEvent:
    """Create an authentication failed event.
    
    Args:
        system_id: The ID of the external system.
        reason: The reason for the failure.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return AuthenticationEvent(AUTHENTICATION_FAILED, system_id, False, reason=reason, **kwargs)


def create_data_received_event(system_id: str, data_type: str, data_id: Optional[str] = None, 
                             **kwargs) -> IntegrationEvent:
    """Create a data received event.
    
    Args:
        system_id: The ID of the external system.
        data_type: The type of data.
        data_id: The ID of the data.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return DataEvent(DATA_RECEIVED, system_id, data_type, data_id, **kwargs)


def create_data_sent_event(system_id: str, data_type: str, data_id: Optional[str] = None, 
                         **kwargs) -> IntegrationEvent:
    """Create a data sent event.
    
    Args:
        system_id: The ID of the external system.
        data_type: The type of data.
        data_id: The ID of the data.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return DataEvent(DATA_SENT, system_id, data_type, data_id, **kwargs)


def create_data_error_event(system_id: str, data_type: str, error: str, 
                          data_id: Optional[str] = None, **kwargs) -> IntegrationEvent:
    """Create a data error event.
    
    Args:
        system_id: The ID of the external system.
        data_type: The type of data.
        error: The error message.
        data_id: The ID of the data.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return DataEvent(DATA_ERROR, system_id, data_type, data_id, error=error, **kwargs)


def create_system_registered_event(system_id: str, system_type: str, **kwargs) -> IntegrationEvent:
    """Create a system registered event.
    
    Args:
        system_id: The ID of the external system.
        system_type: The type of system.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return SystemEvent(SYSTEM_REGISTERED, system_id, system_type=system_type, **kwargs)


def create_system_unregistered_event(system_id: str, **kwargs) -> IntegrationEvent:
    """Create a system unregistered event.
    
    Args:
        system_id: The ID of the external system.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return SystemEvent(SYSTEM_UNREGISTERED, system_id, **kwargs)


def create_system_status_changed_event(system_id: str, status: str, **kwargs) -> IntegrationEvent:
    """Create a system status changed event.
    
    Args:
        system_id: The ID of the external system.
        status: The new status.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return SystemEvent(SYSTEM_STATUS_CHANGED, system_id, status, **kwargs)


def create_system_error_event(system_id: str, error: str, **kwargs) -> IntegrationEvent:
    """Create a system error event.
    
    Args:
        system_id: The ID of the external system.
        error: The error message.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return SystemEvent(SYSTEM_ERROR, system_id, error=error, **kwargs)


def create_mock_service_started_event(service_id: str, service_type: str, **kwargs) -> IntegrationEvent:
    """Create a mock service started event.
    
    Args:
        service_id: The ID of the mock service.
        service_type: The type of mock service.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return MockServiceEvent(MOCK_SERVICE_STARTED, service_id, service_type, **kwargs)


def create_mock_service_stopped_event(service_id: str, service_type: str, **kwargs) -> IntegrationEvent:
    """Create a mock service stopped event.
    
    Args:
        service_id: The ID of the mock service.
        service_type: The type of mock service.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return MockServiceEvent(MOCK_SERVICE_STOPPED, service_id, service_type, **kwargs)


def create_mock_request_received_event(service_id: str, service_type: str, endpoint: str, 
                                     **kwargs) -> IntegrationEvent:
    """Create a mock request received event.
    
    Args:
        service_id: The ID of the mock service.
        service_type: The type of mock service.
        endpoint: The endpoint that received the request.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return MockServiceEvent(MOCK_REQUEST_RECEIVED, service_id, service_type, endpoint=endpoint, **kwargs)


def create_mock_response_sent_event(service_id: str, service_type: str, endpoint: str, 
                                  status: str, **kwargs) -> IntegrationEvent:
    """Create a mock response sent event.
    
    Args:
        service_id: The ID of the mock service.
        service_type: The type of mock service.
        endpoint: The endpoint that sent the response.
        status: The status of the response.
        **kwargs: Additional event data.
        
    Returns:
        IntegrationEvent: The created event.
    """
    return MockServiceEvent(MOCK_RESPONSE_SENT, service_id, service_type, endpoint=endpoint, 
                          status=status, **kwargs)