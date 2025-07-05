"""Broker interface for the Friday AI Trading System.

This module provides an abstract interface for broker integrations,
allowing the system to interact with different brokers through a common API.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from src.infrastructure.event import EventSystem, Event


class BrokerInterface(ABC):
    """Abstract interface for broker integrations.

    This class defines the methods that all broker implementations must provide.
    It serves as a contract for broker integrations, ensuring that the system
    can interact with different brokers through a common API.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the broker interface.

        Args:
            event_system: The event system for publishing events. Defaults to None.
        """
        self.event_system = event_system
        self.authenticated = False

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event through the event system.

        Args:
            event_type: The type of the event.
            data: The data associated with the event.
        """
        if self.event_system:
            self.event_system.emit(
                event_type=event_type,
                data=data,
                source=self.__class__.__name__
            )

    @abstractmethod
    def authenticate(self, *args, **kwargs) -> bool:
        """Authenticate with the broker.

        Returns:
            bool: Whether authentication was successful.
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.

        Returns:
            Dict[str, Any]: The account information.
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions.

        Returns:
            List[Dict[str, Any]]: The positions.
        """
        pass

    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get orders.

        Returns:
            List[Dict[str, Any]]: The orders.
        """
        pass

    @abstractmethod
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order.

        Args:
            order_params: The order parameters.
                This can include standard order parameters like symbol, quantity, price,
                as well as additional metadata like asset_class and sector.

        Returns:
            Dict[str, Any]: The order response.
        """
        pass

    @abstractmethod
    def modify_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an order.

        Args:
            order_params: The order parameters.

        Returns:
            Dict[str, Any]: The order response.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, *args, **kwargs) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: The order ID.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The order response.
        """
        pass

    @abstractmethod
    def get_historical_data(
        self,
        instrument_id: Union[str, int],
        from_date: datetime,
        to_date: datetime,
        interval: str,
        *args,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get historical data for an instrument.

        Args:
            instrument_id: The instrument ID.
            from_date: The start date.
            to_date: The end date.
            interval: The candle interval (minute, day, etc.).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: The historical data.
        """
        pass

    @abstractmethod
    def get_quote(self, instrument_ids: List[Union[str, int]]) -> Dict[str, Any]:
        """Get quotes for instruments.

        Args:
            instrument_ids: The instrument IDs.

        Returns:
            Dict[str, Any]: The quotes.
        """
        pass

    @abstractmethod
    def get_instruments(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get instruments list.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: The instruments list.
        """
        pass

    @abstractmethod
    def subscribe_symbols(self, instrument_ids: List[Union[str, int]]) -> None:
        """Subscribe to symbols for real-time data.

        Args:
            instrument_ids: The instrument IDs to subscribe to.
        """
        pass

    @abstractmethod
    def unsubscribe_symbols(self, instrument_ids: List[Union[str, int]]) -> None:
        """Unsubscribe from symbols.

        Args:
            instrument_ids: The instrument IDs to unsubscribe from.
        """
        pass