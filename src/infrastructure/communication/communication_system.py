"""
Communication System for Friday AI Trading System
Handles inter-component communication, API endpoints, and message passing.
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import signal
import sys
from collections import defaultdict, deque

from ..logging import get_logger
from ..event.event_system import EventSystem, Event

# Create logger
logger = get_logger(__name__)


class MessageType(Enum):
    """Types of messages in the communication system."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


class ComponentType(Enum):
    """Types of components in the system."""
    DATA_SOURCE = "data_source"
    DATA_PIPELINE = "data_pipeline"
    KNOWLEDGE_ENGINE = "knowledge_engine"
    TRADING_ENGINE = "trading_engine"
    RISK_ENGINE = "risk_engine"
    PORTFOLIO_MANAGER = "portfolio_manager"
    NOTIFICATION_SERVICE = "notification_service"
    API_GATEWAY = "api_gateway"


@dataclass
class Message:
    """Message structure for communication between components."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    source: str = ""
    destination: str = ""
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "destination": self.destination,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "request")),
            source=data.get("source", ""),
            destination=data.get("destination", ""),
            topic=data.get("topic", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to")
        )


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return response."""
        pass
    
    @abstractmethod
    def get_supported_topics(self) -> List[str]:
        """Get list of topics this handler supports."""
        pass


class CommunicationBusMetrics:
    """Metrics for monitoring the Communication Bus."""
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.latencies = []
    
    def record_success(self, latency):
        self.success_count += 1
        self.latencies.append(latency)

    def record_failure(self):
        self.failure_count += 1

    def get_metrics(self):
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_latency": sum(self.latencies) / len(self.latencies) if self.latencies else 0
        }

class CommunicationBus:
    """Central communication bus for message routing."""
    
    def __init__(self, max_queue_size=100, rate_limit=10):
        self.handlers: Dict[str, List[MessageHandler]] = {}
        self.components: Dict[str, ComponentType] = {}
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        self.event_system: Optional[EventSystem] = None
        self.rate_limit = rate_limit
        self.metrics = CommunicationBusMetrics()

        # Record the last time a message was published
        self.last_publish_time = time.time()
        
    def register_component(self, component_id: str, component_type: ComponentType):
        """Register a component with the communication bus."""
        self.components[component_id] = component_type
        logger.info(f"Registered component {component_id} of type {component_type.value}")
    
    def register_handler(self, topic: str, handler: MessageHandler):
        """Register a message handler for a specific topic."""
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)
        logger.info(f"Registered handler for topic: {topic}")
    
    def unregister_handler(self, topic: str, handler: MessageHandler):
        """Unregister a message handler."""
        if topic in self.handlers and handler in self.handlers[topic]:
            self.handlers[topic].remove(handler)
            logger.info(f"Unregistered handler for topic: {topic}")
    
    async def publish_message(self, message: Message):
        """Publish a message to the bus."""
        # Rate limiting
        current_time = time.time()
        if (current_time - self.last_publish_time) < 1.0 / self.rate_limit:
            logger.warning("Rate limit exceeded, dropping message")
            return
        self.last_publish_time = current_time
        await self.message_queue.put(message)
        
        # Also emit as event if event system is available
        if self.event_system:
            try:
                self.event_system.emit(
                    f"message.{message.topic}",
                    message.to_dict(),
                    source=message.source
                )
            except Exception as e:
                logger.warning(f"Failed to emit message as event: {e}")
    
    async def send_request(self, destination: str, topic: str, payload: Dict[str, Any], 
                          source: str = "system") -> Optional[Message]:
        """Send a request and wait for response."""
        message = Message(
            type=MessageType.REQUEST,
            source=source,
            destination=destination,
            topic=topic,
            payload=payload,
            reply_to=source
        )
        
        # Store correlation ID for response tracking
        correlation_id = message.id
        response_future = asyncio.Future()
        
        # Register temporary handler for response
        async def response_handler(msg: Message) -> None:
            if msg.correlation_id == correlation_id:
                response_future.set_result(msg)
        
        # Publish request
        await self.publish_message(message)
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for message {correlation_id}")
            return None
    
    async def broadcast_message(self, topic: str, payload: Dict[str, Any], source: str = "system"):
        """Broadcast a message to all interested components."""
        message = Message(
            type=MessageType.BROADCAST,
            source=source,
            destination="*",
            topic=topic,
            payload=payload
        )
        await self.publish_message(message)
    
    async def _process_messages(self, reconnect_attempts=5, backoff_factor=2):
        """Process messages from the queue."""
        reconnect_count = 0
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.record_failure()
                logger.error(f"Error processing message: {e}")
                # Attempt reconnection and exponential back-off
                if reconnect_count < reconnect_attempts:
                    wait_time = backoff_factor ** reconnect_count
                    logger.info(f"Attempting reconnect in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    reconnect_count += 1
                    continue
                else:
                    logger.error("Max reconnect attempts reached, stopping bus.")
                    self.running = False
                    break
    
    async def _route_message(self, message: Message):
        """Route message to appropriate handlers."""
        start_time = time.time()
        handlers = self.handlers.get(message.topic, [])
        
        if not handlers:
            logger.warning(f"No handlers found for topic: {message.topic}")
            return
        
        # Route to specific destination or all handlers for broadcast
        for handler in handlers:
            try:
                if message.destination == "*" or message.destination == "":
                    # Broadcast or general message
                    response = await handler.handle_message(message)
                else:
                    # Specific destination - check if handler should receive
                    response = await handler.handle_message(message)
                
                # Send response if provided
                if response and message.reply_to:
                    response.correlation_id = message.id
                    response.destination = message.reply_to
                    await self.publish_message(response)
                
                # Record success metrics
                latency = time.time() - start_time
                self.metrics.record_success(latency)
                    
            except Exception as e:
                self.metrics.record_failure()
                logger.error(f"Error in message handler: {e}")
    
    async def start(self):
        """Start the communication bus."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        if self.running:
            return
        
        self.running = True
        
        # Start message processing task
        self._message_task = asyncio.create_task(self._process_messages())
        logger.info("Communication bus started")
    
    async def stop(self):
        """Stop the communication bus."""
        self.running = False
        logger.info("Communication bus stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit-breaker metrics."""
        return self.metrics.get_metrics()
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping bus...")
        self.running = False
        sys.exit(0)


class NotificationService:
    """Service for handling notifications and alerts."""
    
    def __init__(self, comm_bus: CommunicationBus):
        self.comm_bus = comm_bus
        self.subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to notification events."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type} notifications")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from notification events."""
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.info(f"Unsubscribed from {event_type} notifications")
    
    async def send_notification(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Send a notification."""
        notification_data = {
            "event_type": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to communication bus
        await self.comm_bus.broadcast_message(
            topic=f"notification.{event_type}",
            payload=notification_data,
            source="notification_service"
        )
        
        # Call local subscribers
        for callback in self.subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    async def send_alert(self, severity: str, message: str, component: str = "system"):
        """Send an alert notification."""
        await self.send_notification(
            "alert",
            message,
            {
                "severity": severity,
                "component": component,
                "requires_action": severity in ["high", "critical"]
            }
        )


class APIGateway:
    """API Gateway for external communication."""
    
    def __init__(self, comm_bus: CommunicationBus):
        self.comm_bus = comm_bus
        self.endpoints: Dict[str, Callable] = {}
        
    def register_endpoint(self, path: str, handler: Callable):
        """Register an API endpoint."""
        self.endpoints[path] = handler
        logger.info(f"Registered API endpoint: {path}")
    
    async def handle_request(self, path: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming API request."""
        if path not in self.endpoints:
            return {"error": "Endpoint not found", "status": 404}
        
        try:
            handler = self.endpoints[path]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(method, data)
            else:
                result = handler(method, data)
            
            return {"data": result, "status": 200}
        except Exception as e:
            logger.error(f"Error handling API request {path}: {e}")
            return {"error": str(e), "status": 500}


class PhaseIntegrationBridge:
    """Bridge for integration between different phases."""
    
    def __init__(self, comm_bus: CommunicationBus):
        self.comm_bus = comm_bus
        self.phase_handlers: Dict[str, MessageHandler] = {}
    
    def register_phase_handler(self, phase: str, handler: MessageHandler):
        """Register a handler for a specific phase."""
        self.phase_handlers[phase] = handler
        
        # Register with communication bus for phase-specific topics
        for topic in handler.get_supported_topics():
            self.comm_bus.register_handler(f"phase.{phase}.{topic}", handler)
        
        logger.info(f"Registered phase {phase} handler")
    
    async def send_to_phase(self, phase: str, topic: str, data: Dict[str, Any]) -> Optional[Message]:
        """Send message to specific phase."""
        full_topic = f"phase.{phase}.{topic}"
        return await self.comm_bus.send_request(
            destination=phase,
            topic=full_topic,
            payload=data,
            source="phase_bridge"
        )


class CommunicationSystem:
    """Main communication system that coordinates all components."""
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        self.comm_bus = CommunicationBus()
        self.notification_service = NotificationService(self.comm_bus)
        self.api_gateway = APIGateway(self.comm_bus)
        self.phase_bridge = PhaseIntegrationBridge(self.comm_bus)
        
        # Connect event system if provided
        if event_system:
            self.comm_bus.event_system = event_system
    
    async def initialize(self):
        """Initialize the communication system."""
        await self.comm_bus.start()
        
        # Register system components
        self.comm_bus.register_component("notification_service", ComponentType.NOTIFICATION_SERVICE)
        self.comm_bus.register_component("api_gateway", ComponentType.API_GATEWAY)
        
        logger.info("Communication system initialized")
    
    async def shutdown(self):
        """Shutdown the communication system."""
        await self.comm_bus.stop()
        logger.info("Communication system shutdown")
    
    def get_bus(self) -> CommunicationBus:
        """Get the communication bus."""
        return self.comm_bus
    
    def get_notification_service(self) -> NotificationService:
        """Get the notification service."""
        return self.notification_service
    
    def get_api_gateway(self) -> APIGateway:
        """Get the API gateway."""
        return self.api_gateway
    
    def get_phase_bridge(self) -> PhaseIntegrationBridge:
        """Get the phase integration bridge."""
        return self.phase_bridge


# Example message handlers
class SystemHealthHandler(MessageHandler):
    """Handler for system health messages."""
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle system health requests."""
        if message.topic == "system.health":
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "communication": "active",
                    "database": "connected",
                    "event_system": "running"
                }
            }
            
            return Message(
                type=MessageType.RESPONSE,
                source="system_health",
                topic="system.health.response",
                payload=health_data
            )
        return None
    
    def get_supported_topics(self) -> List[str]:
        """Get supported topics."""
        return ["system.health"]


class DataRequestHandler(MessageHandler):
    """Handler for data requests."""
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle data requests."""
        if message.topic.startswith("data."):
            # Process data request
            requested_data = message.payload.get("data_type", "unknown")
            
            response_data = {
                "data_type": requested_data,
                "status": "processed",
                "timestamp": datetime.now().isoformat()
            }
            
            return Message(
                type=MessageType.RESPONSE,
                source="data_handler",
                topic=f"{message.topic}.response",
                payload=response_data
            )
        return None
    
    def get_supported_topics(self) -> List[str]:
        """Get supported topics."""
        return ["data.request", "data.historical", "data.realtime"]
