"""Zerodha broker adapter for the Friday AI Trading System.

This module provides an adapter that implements the BrokerInterface
for Zerodha's Kite Connect API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from src.infrastructure.event import EventSystem
from src.services.broker.broker_interface import BrokerInterface
from src.services.broker.zerodha_broker import ZerodhaBroker


class ZerodhaAdapter(BrokerInterface):
    """Adapter for Zerodha broker that implements the BrokerInterface.

    This class adapts the ZerodhaBroker to the BrokerInterface,
    allowing the system to interact with Zerodha through the common broker API.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the Zerodha adapter.

        Args:
            event_system: The event system for publishing events. Defaults to None.
        """
        super().__init__(event_system)
        self.broker = ZerodhaBroker(event_system)

    def authenticate(self, request_token: str) -> bool:
        """Authenticate with Zerodha using the request token.

        Args:
            request_token: The request token from the callback URL.

        Returns:
            bool: Whether authentication was successful.
        """
        try:
            self.broker.generate_session(request_token)
            self.authenticated = self.broker.authenticated
            return self.authenticated
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "authenticate"
            })
            return False

    def get_login_url(self) -> str:
        """Get the login URL for Zerodha authentication.

        Returns:
            str: The login URL.
        """
        return self.broker.get_login_url()

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.

        Returns:
            Dict[str, Any]: The account information.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get account info: Not authenticated with Zerodha")

        # Zerodha doesn't have a direct method for account info, so we combine margins and profile
        try:
            margins = self.broker.get_margins()
            # We would normally get profile info here, but Zerodha API doesn't expose it directly
            # We'll use the margins as a substitute
            return {
                "margins": margins,
                "broker": "zerodha"
            }
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_account_info"
            })
            raise

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions.

        Returns:
            List[Dict[str, Any]]: The positions.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get positions: Not authenticated with Zerodha")

        try:
            positions_data = self.broker.get_positions()
            # Zerodha returns positions in a dict with 'day' and 'net' keys
            # We'll use 'net' positions as they represent the current holdings
            positions = positions_data.get("net", [])
            return positions
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_positions"
            })
            raise

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get orders.

        Returns:
            List[Dict[str, Any]]: The orders.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get orders: Not authenticated with Zerodha")

        try:
            return self.broker.get_orders()
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_orders"
            })
            raise

    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order.

        Args:
            order_params: The order parameters.
                This can include standard order parameters like exchange, tradingsymbol, 
                transaction_type, quantity, price, etc., as well as additional metadata 
                like asset_class and sector for portfolio categorization.

        Returns:
            Dict[str, Any]: The order response.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot place order: Not authenticated with Zerodha")

        try:
            return self.broker._place_order(order_params)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "place_order",
                "params": order_params
            })
            raise

    def modify_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an order.

        Args:
            order_params: The order parameters.

        Returns:
            Dict[str, Any]: The order response.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot modify order: Not authenticated with Zerodha")

        try:
            return self.broker._modify_order(order_params)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "modify_order",
                "params": order_params
            })
            raise

    def cancel_order(self, order_id: str, variety: str = "regular") -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: The order ID.
            variety: The order variety. Defaults to "regular".

        Returns:
            Dict[str, Any]: The order response.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot cancel order: Not authenticated with Zerodha")

        try:
            return self.broker._cancel_order({
                "order_id": order_id,
                "variety": variety
            })
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "cancel_order",
                "order_id": order_id
            })
            raise

    def get_historical_data(
        self,
        instrument_id: Union[str, int],
        from_date: datetime,
        to_date: datetime,
        interval: str,
        continuous: bool = False
    ) -> List[Dict[str, Any]]:
        """Get historical data for an instrument.

        Args:
            instrument_id: The instrument ID (token).
            from_date: The start date.
            to_date: The end date.
            interval: The candle interval (minute, day, etc.).
            continuous: Whether to get continuous data. Defaults to False.

        Returns:
            List[Dict[str, Any]]: The historical data.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get historical data: Not authenticated with Zerodha")

        try:
            # Ensure instrument_id is an integer for Zerodha
            instrument_token = int(instrument_id)
            return self.broker.get_historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous
            )
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_historical_data",
                "instrument_id": instrument_id
            })
            raise

    def get_quote(self, instrument_ids: List[Union[str, int]]) -> Dict[str, Any]:
        """Get quotes for instruments.

        Args:
            instrument_ids: The instrument IDs (tokens).

        Returns:
            Dict[str, Any]: The quotes.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get quote: Not authenticated with Zerodha")

        try:
            # Ensure all instrument_ids are integers for Zerodha
            instrument_tokens = [int(id) for id in instrument_ids]
            return self.broker.get_quote(instrument_tokens)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_quote",
                "instrument_ids": instrument_ids
            })
            raise

    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get instruments list.

        Args:
            exchange: The exchange. Defaults to None.

        Returns:
            List[Dict[str, Any]]: The instruments list.
        """
        try:
            return self.broker.get_instruments(exchange=exchange)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_instruments",
                "exchange": exchange
            })
            raise

    def subscribe_symbols(self, instrument_ids: List[Union[str, int]]) -> None:
        """Subscribe to symbols for real-time data.

        Args:
            instrument_ids: The instrument IDs (tokens) to subscribe to.

        Raises:
            RuntimeError: If not authenticated or ticker not running.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot subscribe: Not authenticated with Zerodha")

        try:
            # Ensure all instrument_ids are integers for Zerodha
            instrument_tokens = [int(id) for id in instrument_ids]
            
            # Start ticker if not already running
            if not self.broker.ticker_running:
                self.broker.start_ticker()
                
            self.broker.subscribe_symbols(instrument_tokens)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "subscribe_symbols",
                "instrument_ids": instrument_ids
            })
            raise

    def unsubscribe_symbols(self, instrument_ids: List[Union[str, int]]) -> None:
        """Unsubscribe from symbols.

        Args:
            instrument_ids: The instrument IDs (tokens) to unsubscribe from.

        Raises:
            RuntimeError: If not authenticated or ticker not running.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot unsubscribe: Not authenticated with Zerodha")

        try:
            # Ensure all instrument_ids are integers for Zerodha
            instrument_tokens = [int(id) for id in instrument_ids]
            self.broker.unsubscribe_symbols(instrument_tokens)
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "unsubscribe_symbols",
                "instrument_ids": instrument_ids
            })
            raise

    def start_ticker(self) -> None:
        """Start the ticker for real-time data.

        Raises:
            RuntimeError: If not authenticated or ticker already running.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot start ticker: Not authenticated with Zerodha")

        try:
            self.broker.start_ticker()
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "start_ticker"
            })
            raise

    def stop_ticker(self) -> None:
        """Stop the ticker."""
        try:
            self.broker.stop_ticker()
        except Exception as e:
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "stop_ticker"
            })
            raise