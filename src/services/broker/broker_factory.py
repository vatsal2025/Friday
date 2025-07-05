"""Broker factory for the Friday AI Trading System.

This module provides a factory for creating broker instances
based on the configuration.
"""

from typing import Dict, Optional, Type

from src.infrastructure.config import get_config
from src.infrastructure.event import EventSystem
from src.services.broker.broker_interface import BrokerInterface
from src.services.broker.zerodha_adapter import ZerodhaAdapter


class BrokerFactory:
    """Factory for creating broker instances.

    This class provides methods for creating and retrieving broker instances
    based on the configuration.

    Attributes:
        _brokers: A dictionary of broker instances.
        _broker_classes: A dictionary mapping broker names to broker classes.
        event_system: The event system for publishing events.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the broker factory.

        Args:
            event_system: The event system for publishing events. Defaults to None.
        """
        self._brokers: Dict[str, BrokerInterface] = {}
        self._broker_classes: Dict[str, Type[BrokerInterface]] = {
            "zerodha": ZerodhaAdapter,
            # Add more broker adapters here as they are implemented
        }
        self.event_system = event_system

    def get_broker(self, broker_name: Optional[str] = None) -> BrokerInterface:
        """Get a broker instance.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker
                from the configuration.

        Returns:
            BrokerInterface: The broker instance.

        Raises:
            ValueError: If the broker is not supported.
        """
        if broker_name is None:
            # Get the default broker from the configuration
            config = get_config("BROKER_CONFIG")
            broker_name = config.get("default_broker", "zerodha")

        # Check if the broker is already instantiated
        if broker_name in self._brokers:
            return self._brokers[broker_name]

        # Check if the broker is supported
        if broker_name not in self._broker_classes:
            raise ValueError(f"Broker '{broker_name}' is not supported")

        # Create a new broker instance
        broker_class = self._broker_classes[broker_name]
        broker = broker_class(self.event_system)
        self._brokers[broker_name] = broker

        return broker

    def register_broker(self, broker_name: str, broker_class: Type[BrokerInterface]) -> None:
        """Register a new broker class.

        Args:
            broker_name: The name of the broker.
            broker_class: The broker class.
        """
        self._broker_classes[broker_name] = broker_class