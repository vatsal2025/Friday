"""
Communication Infrastructure for Friday AI Trading System.

This module provides communication, messaging, and integration capabilities.
"""

from .communication_system import (
    CommunicationSystem,
    CommunicationBus,
    Message,
    MessageType,
    MessageHandler,
    NotificationService,
    APIGateway,
    PhaseIntegrationBridge,
    ComponentType,
    SystemHealthHandler,
    DataRequestHandler
)

__all__ = [
    "CommunicationSystem",
    "CommunicationBus", 
    "Message",
    "MessageType",
    "MessageHandler",
    "NotificationService",
    "APIGateway",
    "PhaseIntegrationBridge",
    "ComponentType",
    "SystemHealthHandler",
    "DataRequestHandler"
]
