"""Zerodha Kite Connect API integration for the Friday AI Trading System.

This module provides a broker interface implementation for Zerodha's Kite Connect API,
allowing the system to authenticate, fetch market data, and execute trades.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Import the Kite Connect API library
try:
    from kiteconnect import KiteConnect, KiteTicker
except ImportError:
    raise ImportError(
        "kiteconnect package is required for Zerodha integration. "
        "Install it using: pip install kiteconnect"
    )

from src.infrastructure.config import get_config
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class ZerodhaBroker:
    """Zerodha Kite Connect API integration.

    This class provides methods for authenticating with Zerodha,
    fetching market data, and executing trades.

    Attributes:
        kite: The KiteConnect instance.
        ticker: The KiteTicker instance for real-time data.
        event_system: The event system for publishing events.
        config: The Zerodha configuration.
        authenticated: Whether the broker is authenticated.
        access_token: The access token for API access.
        ticker_running: Whether the ticker is running.
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the Zerodha broker.

        Args:
            event_system: The event system for publishing events. Defaults to None.
        """
        self.config = get_config("ZERODHA_CONFIG")
        self.kite = KiteConnect(api_key=self.config["api_key"])
        self.ticker: Optional[KiteTicker] = None
        self.event_system = event_system
        self.authenticated = False
        self.access_token: Optional[str] = None
        self.ticker_running = False
        self.ticker_reconnect_count = 0
        self.ticker_max_reconnects = self.config.get("ticker_max_reconnects", 5)
        self.ticker_reconnect_delay = self.config.get("ticker_reconnect_delay", 5)
        self.ticker_reconnect_backoff = self.config.get("ticker_reconnect_backoff", 2)
        self.ticker_subscriptions: List[int] = []

        # Register event handlers if event system is provided
        if self.event_system:
            self.register_event_handlers()

    def register_event_handlers(self) -> None:
        """Register event handlers with the event system."""
        if not self.event_system:
            return

        self.event_system.register_handler(
            callback=self._handle_order_event,
            event_types=["order_place", "order_modify", "order_cancel"]
        )

    def _handle_order_event(self, event: Event) -> None:
        """Handle order-related events.

        Args:
            event: The event to handle.
        """
        if not self.authenticated:
            logger.error("Cannot handle order event: Not authenticated with Zerodha")
            self._emit_event("broker_error", {
                "error": "Not authenticated",
                "original_event": event.to_dict()
            })
            return

        try:
            if event.event_type == "order_place":
                self._place_order(event.data)
            elif event.event_type == "order_modify":
                self._modify_order(event.data)
            elif event.event_type == "order_cancel":
                self._cancel_order(event.data)
        except Exception as e:
            logger.error(f"Error handling order event: {str(e)}")
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
                source="zerodha_broker"
            )

    def get_login_url(self) -> str:
        """Get the login URL for Zerodha authentication.

        Returns:
            str: The login URL.
        """
        return self.kite.login_url()

    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Generate a session with the request token.

        Args:
            request_token: The request token from the callback URL.

        Returns:
            Dict[str, Any]: The user profile and session details.

        Raises:
            Exception: If there is an error generating the session.
        """
        try:
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.config["api_secret"]
            )
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.authenticated = True
            
            # Initialize ticker with the access token
            self.ticker = KiteTicker(
                api_key=self.config["api_key"],
                access_token=self.access_token
            )
            self._setup_ticker_callbacks()
            
            # Emit authentication success event
            self._emit_event("broker_authenticated", {
                "broker": "zerodha",
                "user_id": data["user_id"],
                "user_name": data.get("user_name", ""),
                "timestamp": datetime.now().isoformat()
            })
            
            return data
        except Exception as e:
            logger.error(f"Error generating session: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "generate_session"
            })
            raise

    def _setup_ticker_callbacks(self) -> None:
        """Set up callbacks for the ticker."""
        if not self.ticker:
            return

        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close
        self.ticker.on_error = self._on_error
        self.ticker.on_reconnect = self._on_reconnect
        self.ticker.on_noreconnect = self._on_noreconnect
        self.ticker.on_order_update = self._on_order_update

    def _on_ticks(self, ws, ticks) -> None:
        """Callback when ticks are received.

        Args:
            ws: The WebSocket instance.
            ticks: The ticks data.
        """
        for tick in ticks:
            self._emit_event("market_tick", {
                "instrument_token": tick["instrument_token"],
                "timestamp": tick.get("timestamp", datetime.now().isoformat()),
                "last_price": tick.get("last_price"),
                "volume": tick.get("volume"),
                "buy_quantity": tick.get("buy_quantity"),
                "sell_quantity": tick.get("sell_quantity"),
                "ohlc": tick.get("ohlc", {}),
                "change": tick.get("change")
            })

    def _on_connect(self, ws, response) -> None:
        """Callback when connection is established.

        Args:
            ws: The WebSocket instance.
            response: The connection response.
        """
        logger.info("Ticker connected")
        self._emit_event("ticker_connected", {
            "timestamp": datetime.now().isoformat()
        })
        
        # Reset reconnect count on successful connection
        self.ticker_reconnect_count = 0
        
        # Resubscribe to instruments if any
        if self.ticker_subscriptions:
            self.ticker.subscribe(self.ticker_subscriptions)

    def _on_close(self, ws, code, reason) -> None:
        """Callback when connection is closed.

        Args:
            ws: The WebSocket instance.
            code: The close code.
            reason: The close reason.
        """
        logger.warning(f"Ticker connection closed: {code} - {reason}")
        self._emit_event("ticker_disconnected", {
            "code": code,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        self.ticker_running = False

    def _on_error(self, ws, code, reason) -> None:
        """Callback when an error occurs.

        Args:
            ws: The WebSocket instance.
            code: The error code.
            reason: The error reason.
        """
        logger.error(f"Ticker error: {code} - {reason}")
        self._emit_event("ticker_error", {
            "code": code,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

    def _on_reconnect(self, ws, attempts_count) -> None:
        """Callback when reconnection is attempted.

        Args:
            ws: The WebSocket instance.
            attempts_count: The number of reconnection attempts.
        """
        logger.info(f"Ticker reconnecting: attempt {attempts_count}")
        self._emit_event("ticker_reconnecting", {
            "attempts": attempts_count,
            "timestamp": datetime.now().isoformat()
        })
        self.ticker_reconnect_count = attempts_count

    def _on_noreconnect(self, ws) -> None:
        """Callback when reconnection fails.

        Args:
            ws: The WebSocket instance.
        """
        logger.error("Ticker failed to reconnect")
        self._emit_event("ticker_reconnect_failed", {
            "timestamp": datetime.now().isoformat()
        })
        
        # Try to manually reconnect if within limits
        if self.ticker_reconnect_count < self.ticker_max_reconnects:
            delay = self.ticker_reconnect_delay * (self.ticker_reconnect_backoff ** self.ticker_reconnect_count)
            logger.info(f"Manual reconnect attempt in {delay} seconds")
            time.sleep(delay)
            self.start_ticker()

    def _on_order_update(self, ws, data) -> None:
        """Callback when an order update is received.

        Args:
            ws: The WebSocket instance.
            data: The order update data.
        """
        self._emit_event("order_update", data)

    def start_ticker(self) -> None:
        """Start the ticker for real-time data.

        Raises:
            RuntimeError: If not authenticated or ticker already running.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot start ticker: Not authenticated with Zerodha")

        if self.ticker_running:
            logger.warning("Ticker is already running")
            return

        if not self.ticker:
            self.ticker = KiteTicker(
                api_key=self.config["api_key"],
                access_token=self.access_token
            )
            self._setup_ticker_callbacks()

        try:
            self.ticker.connect(threaded=True)
            self.ticker_running = True
            logger.info("Ticker started")
        except Exception as e:
            logger.error(f"Error starting ticker: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "start_ticker"
            })
            raise

    def stop_ticker(self) -> None:
        """Stop the ticker."""
        if not self.ticker_running or not self.ticker:
            return

        try:
            self.ticker.close()
            self.ticker_running = False
            logger.info("Ticker stopped")
        except Exception as e:
            logger.error(f"Error stopping ticker: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "stop_ticker"
            })

    def subscribe_symbols(self, instrument_tokens: List[int]) -> None:
        """Subscribe to symbols for real-time data.

        Args:
            instrument_tokens: The instrument tokens to subscribe to.

        Raises:
            RuntimeError: If ticker is not running.
        """
        if not self.ticker_running or not self.ticker:
            raise RuntimeError("Cannot subscribe: Ticker is not running")

        try:
            self.ticker.subscribe(instrument_tokens)
            self.ticker_subscriptions.extend(instrument_tokens)
            # Remove duplicates
            self.ticker_subscriptions = list(set(self.ticker_subscriptions))
            logger.info(f"Subscribed to {len(instrument_tokens)} instruments")
        except Exception as e:
            logger.error(f"Error subscribing to instruments: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "subscribe_symbols"
            })
            raise

    def unsubscribe_symbols(self, instrument_tokens: List[int]) -> None:
        """Unsubscribe from symbols.

        Args:
            instrument_tokens: The instrument tokens to unsubscribe from.

        Raises:
            RuntimeError: If ticker is not running.
        """
        if not self.ticker_running or not self.ticker:
            raise RuntimeError("Cannot unsubscribe: Ticker is not running")

        try:
            self.ticker.unsubscribe(instrument_tokens)
            # Remove from subscriptions list
            self.ticker_subscriptions = [t for t in self.ticker_subscriptions if t not in instrument_tokens]
            logger.info(f"Unsubscribed from {len(instrument_tokens)} instruments")
        except Exception as e:
            logger.error(f"Error unsubscribing from instruments: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "unsubscribe_symbols"
            })
            raise

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
        continuous: bool = False
    ) -> List[Dict[str, Any]]:
        """Get historical data for an instrument.

        Args:
            instrument_token: The instrument token.
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
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous
            )
            logger.info(f"Retrieved {len(data)} historical candles for instrument {instrument_token}")
            return data
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_historical_data",
                "instrument_token": instrument_token
            })
            raise

    def get_quote(self, instrument_tokens: List[int]) -> Dict[str, Any]:
        """Get quotes for instruments.

        Args:
            instrument_tokens: The instrument tokens.

        Returns:
            Dict[str, Any]: The quotes.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get quote: Not authenticated with Zerodha")

        try:
            quotes = self.kite.quote(instrument_tokens)
            return quotes
        except Exception as e:
            logger.error(f"Error getting quotes: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_quote",
                "instrument_tokens": instrument_tokens
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
            instruments = self.kite.instruments(exchange=exchange)
            logger.info(f"Retrieved {len(instruments)} instruments")
            return instruments
        except Exception as e:
            logger.error(f"Error getting instruments: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_instruments",
                "exchange": exchange
            })
            raise

    def get_margins(self) -> Dict[str, Any]:
        """Get user margins.

        Returns:
            Dict[str, Any]: The margins.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get margins: Not authenticated with Zerodha")

        try:
            margins = self.kite.margins()
            return margins
        except Exception as e:
            logger.error(f"Error getting margins: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_margins"
            })
            raise

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get user positions.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The positions.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get positions: Not authenticated with Zerodha")

        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_positions"
            })
            raise

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get user orders.

        Returns:
            List[Dict[str, Any]]: The orders.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot get orders: Not authenticated with Zerodha")

        try:
            orders = self.kite.orders()
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "get_orders"
            })
            raise

    def _place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order.

        Args:
            order_params: The order parameters.

        Returns:
            Dict[str, Any]: The order response.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot place order: Not authenticated with Zerodha")

        try:
            order_id = self.kite.place_order(
                variety=order_params.get("variety", "regular"),
                exchange=order_params["exchange"],
                tradingsymbol=order_params["tradingsymbol"],
                transaction_type=order_params["transaction_type"],
                quantity=order_params["quantity"],
                price=order_params.get("price"),
                product=order_params.get("product", "CNC"),
                order_type=order_params.get("order_type", "MARKET"),
                validity=order_params.get("validity", "DAY"),
                disclosed_quantity=order_params.get("disclosed_quantity"),
                trigger_price=order_params.get("trigger_price"),
                squareoff=order_params.get("squareoff"),
                stoploss=order_params.get("stoploss"),
                trailing_stoploss=order_params.get("trailing_stoploss")
            )
            
            logger.info(f"Order placed: {order_id}")
            
            # Prepare event data
            event_data = {
                "order_id": order_id,
                "params": order_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Include asset_class and sector in the event if available
            if "asset_class" in order_params:
                event_data["asset_class"] = order_params["asset_class"]
                logger.info(f"Order asset class: {order_params['asset_class']}")
            if "sector" in order_params:
                event_data["sector"] = order_params["sector"]
                logger.info(f"Order sector: {order_params['sector']}")
            
            # Emit order placed event
            self._emit_event("order_placed", event_data)
            
            return {"order_id": order_id}
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "place_order",
                "params": order_params
            })
            raise

    def _modify_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
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
            order_id = self.kite.modify_order(
                variety=order_params.get("variety", "regular"),
                order_id=order_params["order_id"],
                quantity=order_params.get("quantity"),
                price=order_params.get("price"),
                order_type=order_params.get("order_type"),
                validity=order_params.get("validity"),
                disclosed_quantity=order_params.get("disclosed_quantity"),
                trigger_price=order_params.get("trigger_price")
            )
            
            logger.info(f"Order modified: {order_id}")
            
            # Emit order modified event
            self._emit_event("order_modified", {
                "order_id": order_id,
                "params": order_params,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"order_id": order_id}
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "modify_order",
                "params": order_params
            })
            raise

    def _cancel_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            order_params: The order parameters.

        Returns:
            Dict[str, Any]: The order response.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.authenticated:
            raise RuntimeError("Cannot cancel order: Not authenticated with Zerodha")

        try:
            order_id = self.kite.cancel_order(
                variety=order_params.get("variety", "regular"),
                order_id=order_params["order_id"]
            )
            
            logger.info(f"Order cancelled: {order_id}")
            
            # Emit order cancelled event
            self._emit_event("order_cancelled", {
                "order_id": order_id,
                "params": order_params,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"order_id": order_id}
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            self._emit_event("broker_error", {
                "error": str(e),
                "action": "cancel_order",
                "params": order_params
            })
            raise