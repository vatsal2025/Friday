"""Broker service for the Friday AI Trading System.

This module provides a service for interacting with brokers,
managing authentication, and handling broker-related events.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from src.infrastructure.config import get_config
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger
from src.services.broker.broker_factory import BrokerFactory
from src.services.broker.broker_interface import BrokerInterface

# Create logger
logger = get_logger(__name__)


class BrokerService:
    """Service for interacting with brokers.

    This class provides methods for authenticating with brokers,
    fetching market data, and executing trades.

    Attributes:
        event_system: The event system for publishing events.
        broker_factory: The factory for creating broker instances.
        default_broker: The default broker instance.
        config: The broker service configuration.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the broker service.

        Args:
            event_system: The event system for publishing events. Defaults to None.
        """
        self.event_system = event_system
        self.broker_factory = BrokerFactory(event_system)
        self.config = get_config("BROKER_CONFIG")
        self.default_broker_name = self.config.get("default_broker", "zerodha")
        self.default_broker = self.broker_factory.get_broker(self.default_broker_name)

        # Register event handlers if event system is provided
        if self.event_system:
            self.register_event_handlers()

    def register_event_handlers(self) -> None:
        """Register event handlers with the event system."""
        if not self.event_system:
            return

        self.event_system.register_handler(
            callback=self._handle_broker_event,
            event_types=[
                "order_place", "order_modify", "order_cancel",
                "market_data_request", "broker_authenticate"
            ]
        )

    def _handle_broker_event(self, event: Event) -> None:
        """Handle broker-related events.

        Args:
            event: The event to handle.
        """
        try:
            if event.event_type == "broker_authenticate":
                broker_name = event.data.get("broker", self.default_broker_name)
                request_token = event.data.get("request_token")
                if not request_token:
                    self._emit_event("broker_error", {
                        "error": "No request token provided",
                        "broker": broker_name,
                        "original_event": event.to_dict()
                    })
                    return
                self.authenticate(request_token, broker_name)
            elif event.event_type == "order_place":
                broker_name = event.data.get("broker", self.default_broker_name)
                broker = self.broker_factory.get_broker(broker_name)
                order_params = event.data.get("params", {})
                self.place_order(order_params, broker_name)
            elif event.event_type == "order_modify":
                broker_name = event.data.get("broker", self.default_broker_name)
                broker = self.broker_factory.get_broker(broker_name)
                order_params = event.data.get("params", {})
                self.modify_order(order_params, broker_name)
            elif event.event_type == "order_cancel":
                broker_name = event.data.get("broker", self.default_broker_name)
                broker = self.broker_factory.get_broker(broker_name)
                order_id = event.data.get("order_id")
                variety = event.data.get("variety", "regular")
                self.cancel_order(order_id, variety, broker_name)
            elif event.event_type == "market_data_request":
                broker_name = event.data.get("broker", self.default_broker_name)
                broker = self.broker_factory.get_broker(broker_name)
                request_type = event.data.get("request_type")
                if request_type == "historical":
                    self.get_historical_data(
                        instrument_id=event.data.get("instrument_id"),
                        from_date=event.data.get("from_date"),
                        to_date=event.data.get("to_date"),
                        interval=event.data.get("interval"),
                        broker_name=broker_name
                    )
                elif request_type == "quote":
                    self.get_quote(
                        instrument_ids=event.data.get("instrument_ids"),
                        broker_name=broker_name
                    )
                elif request_type == "subscribe":
                    self.subscribe_symbols(
                        instrument_ids=event.data.get("instrument_ids"),
                        broker_name=broker_name
                    )
                elif request_type == "unsubscribe":
                    self.unsubscribe_symbols(
                        instrument_ids=event.data.get("instrument_ids"),
                        broker_name=broker_name
                    )
        except Exception as e:
            logger.error(f"Error handling broker event: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "original_event": event.to_dict()
            })

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
                source="broker_service"
            )

    def get_login_url(self, broker_name: Optional[str] = None) -> str:
        """Get the login URL for broker authentication.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            str: The login URL.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        # Check if the broker has a get_login_url method
        if hasattr(broker, "get_login_url") and callable(getattr(broker, "get_login_url")):
            return broker.get_login_url()
        else:
            raise NotImplementedError(f"Broker '{broker_name or self.default_broker_name}' does not support get_login_url")

    def authenticate(self, request_token: str, broker_name: Optional[str] = None) -> bool:
        """Authenticate with a broker.

        Args:
            request_token: The request token from the callback URL.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            bool: Whether authentication was successful.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            result = broker.authenticate(request_token)
            if result:
                self._emit_event("broker_authenticated", {
                    "broker": broker_name or self.default_broker_name,
                    "timestamp": datetime.now().isoformat()
                })
            return result
        except Exception as e:
            logger.error(f"Error authenticating with broker: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "authenticate",
                "broker": broker_name or self.default_broker_name
            })
            return False

    def get_account_info(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get account information.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            Dict[str, Any]: The account information.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            return broker.get_account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_account_info",
                "broker": broker_name or self.default_broker_name
            })
            raise

    def get_positions(self, broker_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            List[Dict[str, Any]]: The positions.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            positions = broker.get_positions()
            self._emit_event("positions_fetched", {
                "broker": broker_name or self.default_broker_name,
                "count": len(positions),
                "timestamp": datetime.now().isoformat()
            })
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_positions",
                "broker": broker_name or self.default_broker_name
            })
            raise

    def get_orders(self, broker_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            List[Dict[str, Any]]: The orders.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            orders = broker.get_orders()
            self._emit_event("orders_fetched", {
                "broker": broker_name or self.default_broker_name,
                "count": len(orders),
                "timestamp": datetime.now().isoformat()
            })
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_orders",
                "broker": broker_name or self.default_broker_name
            })
            raise

    def place_order(self, order_params: Dict[str, Any], broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Place an order.

        Args:
            order_params: The order parameters.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            Dict[str, Any]: The order response.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            response = broker.place_order(order_params)
            event_data = {
                "broker": broker_name or self.default_broker_name,
                "order_id": response.get("order_id"),
                "params": order_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Include asset_class and sector in the event if available
            if "asset_class" in order_params:
                event_data["asset_class"] = order_params["asset_class"]
            if "sector" in order_params:
                event_data["sector"] = order_params["sector"]
                
            self._emit_event("order_placed", event_data)
            return response
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "place_order",
                "params": order_params,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def modify_order(self, order_params: Dict[str, Any], broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Modify an order.

        Args:
            order_params: The order parameters.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            Dict[str, Any]: The order response.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            response = broker.modify_order(order_params)
            self._emit_event("order_modified", {
                "broker": broker_name or self.default_broker_name,
                "order_id": response.get("order_id"),
                "params": order_params,
                "timestamp": datetime.now().isoformat()
            })
            return response
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "modify_order",
                "params": order_params,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def cancel_order(self, order_id: str, variety: str = "regular", broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: The order ID.
            variety: The order variety. Defaults to "regular".
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            Dict[str, Any]: The order response.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            response = broker.cancel_order(order_id, variety=variety)
            self._emit_event("order_cancelled", {
                "broker": broker_name or self.default_broker_name,
                "order_id": order_id,
                "timestamp": datetime.now().isoformat()
            })
            return response
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "cancel_order",
                "order_id": order_id,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def get_historical_data(
        self,
        instrument_id: Union[str, int],
        from_date: datetime,
        to_date: datetime,
        interval: str,
        broker_name: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get historical data for an instrument.

        Args:
            instrument_id: The instrument ID.
            from_date: The start date.
            to_date: The end date.
            interval: The candle interval (minute, day, etc.).
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
            **kwargs: Additional keyword arguments to pass to the broker.

        Returns:
            List[Dict[str, Any]]: The historical data.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            data = broker.get_historical_data(
                instrument_id=instrument_id,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                **kwargs
            )
            self._emit_event("historical_data_fetched", {
                "broker": broker_name or self.default_broker_name,
                "instrument_id": instrument_id,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "interval": interval,
                "count": len(data),
                "timestamp": datetime.now().isoformat()
            })
            return data
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_historical_data",
                "instrument_id": instrument_id,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def get_quote(self, instrument_ids: List[Union[str, int]], broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get quotes for instruments.

        Args:
            instrument_ids: The instrument IDs.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.

        Returns:
            Dict[str, Any]: The quotes.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            quotes = broker.get_quote(instrument_ids)
            self._emit_event("quotes_fetched", {
                "broker": broker_name or self.default_broker_name,
                "instrument_ids": instrument_ids,
                "timestamp": datetime.now().isoformat()
            })
            return quotes
        except Exception as e:
            logger.error(f"Error getting quotes: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_quote",
                "instrument_ids": instrument_ids,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def get_instruments(self, broker_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get instruments list.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
            **kwargs: Additional keyword arguments to pass to the broker.

        Returns:
            List[Dict[str, Any]]: The instruments list.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            instruments = broker.get_instruments(**kwargs)
            self._emit_event("instruments_fetched", {
                "broker": broker_name or self.default_broker_name,
                "count": len(instruments),
                "timestamp": datetime.now().isoformat()
            })
            return instruments
        except Exception as e:
            logger.error(f"Error getting instruments: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_instruments",
                "broker": broker_name or self.default_broker_name
            })
            raise

    def subscribe_symbols(self, instrument_ids: List[Union[str, int]], broker_name: Optional[str] = None) -> None:
        """Subscribe to symbols for real-time data.

        Args:
            instrument_ids: The instrument IDs to subscribe to.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            broker.subscribe_symbols(instrument_ids)
            self._emit_event("symbols_subscribed", {
                "broker": broker_name or self.default_broker_name,
                "instrument_ids": instrument_ids,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "subscribe_symbols",
                "instrument_ids": instrument_ids,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def unsubscribe_symbols(self, instrument_ids: List[Union[str, int]], broker_name: Optional[str] = None) -> None:
        """Unsubscribe from symbols.

        Args:
            instrument_ids: The instrument IDs to unsubscribe from.
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        try:
            broker.unsubscribe_symbols(instrument_ids)
            self._emit_event("symbols_unsubscribed", {
                "broker": broker_name or self.default_broker_name,
                "instrument_ids": instrument_ids,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "unsubscribe_symbols",
                "instrument_ids": instrument_ids,
                "broker": broker_name or self.default_broker_name
            })
            raise

    def start_ticker(self, broker_name: Optional[str] = None) -> None:
        """Start the ticker for real-time data.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        # Check if the broker has a start_ticker method
        if hasattr(broker, "start_ticker") and callable(getattr(broker, "start_ticker")):
            try:
                broker.start_ticker()
                self._emit_event("ticker_started", {
                    "broker": broker_name or self.default_broker_name,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error starting ticker: {str(e)}")
                self._emit_event("broker_error", {
                    "error": str(e),
                    "action": "start_ticker",
                    "broker": broker_name or self.default_broker_name
                })
                raise
        else:
            logger.warning(f"Broker '{broker_name or self.default_broker_name}' does not support start_ticker")

    def stop_ticker(self, broker_name: Optional[str] = None) -> None:
        """Stop the ticker.

        Args:
            broker_name: The name of the broker. Defaults to None, which will use the default broker.
        """
        broker = self.broker_factory.get_broker(broker_name) if broker_name else self.default_broker
        
        # Check if the broker has a stop_ticker method
        if hasattr(broker, "stop_ticker") and callable(getattr(broker, "stop_ticker")):
            try:
                broker.stop_ticker()
                self._emit_event("ticker_stopped", {
                    "broker": broker_name or self.default_broker_name,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error stopping ticker: {str(e)}")
                self._emit_event("broker_error", {
                    "error": str(e),
                    "action": "stop_ticker",
                    "broker": broker_name or self.default_broker_name
                })
                raise
        else:
            logger.warning(f"Broker '{broker_name or self.default_broker_name}' does not support stop_ticker")