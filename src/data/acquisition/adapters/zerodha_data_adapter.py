"""Zerodha data adapter for the Friday AI Trading System.

This module provides an adapter that implements the DataSourceAdapter interface
for Zerodha's Kite Connect API, allowing the system to fetch real-time and historical
market data through the common data API.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker

from src.data.acquisition.data_fetcher import (
    DataSourceAdapter,
    DataTimeframe,
    DataConnectionError,
    DataValidationError
)
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.event import Event, EventSystem

# Create logger
logger = get_logger(__name__)


class ZerodhaDataAdapter(DataSourceAdapter):
    """Adapter for Zerodha that implements the DataSourceAdapter interface.

    This class adapts the Zerodha Kite Connect API to the DataSourceAdapter interface,
    allowing the system to fetch data from Zerodha through the common data API.
    
    Attributes:
        api_key: The Zerodha API key.
        api_secret: The Zerodha API secret.
        kite: The KiteConnect instance.
        ticker: The KiteTicker instance for real-time data.
        connected: Whether the adapter is connected to Zerodha.
        authenticated: Whether the adapter is authenticated with Zerodha.
        access_token: The access token for API access.
        ticker_running: Whether the ticker is running.
        available_timeframes: List of available timeframes.
        event_system: The event system for publishing events.
    """

    # Mapping from DataTimeframe to Zerodha intervals
    TIMEFRAME_MAP = {
        DataTimeframe.ONE_MINUTE: "minute",
        DataTimeframe.FIVE_MINUTES: "5minute",
        DataTimeframe.FIFTEEN_MINUTES: "15minute",
        DataTimeframe.THIRTY_MINUTES: "30minute",
        DataTimeframe.ONE_HOUR: "60minute",
        DataTimeframe.ONE_DAY: "day",
        DataTimeframe.ONE_WEEK: "week",
        DataTimeframe.ONE_MONTH: "month"
    }

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, event_system: Optional[EventSystem] = None):
        """Initialize the Zerodha adapter.

        Args:
            api_key: The Zerodha API key. If None, it will be loaded from config.
            api_secret: The Zerodha API secret. If None, it will be loaded from config.
            event_system: The event system for publishing events. If None, a new one will be created.
        """
        self.config = ConfigManager()
        self.api_key = api_key or self.config.get("zerodha.api_key")
        self.api_secret = api_secret or self.config.get("zerodha.api_secret")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Zerodha API key and secret must be provided or configured")
        
        self.kite = KiteConnect(api_key=self.api_key)
        self.ticker = None
        self.connected = False
        self.authenticated = False
        self.access_token = None
        self.ticker_running = False
        
        self.available_timeframes = [
            DataTimeframe.ONE_MINUTE,
            DataTimeframe.FIVE_MINUTES,
            DataTimeframe.FIFTEEN_MINUTES,
            DataTimeframe.THIRTY_MINUTES,
            DataTimeframe.ONE_HOUR,
            DataTimeframe.ONE_DAY,
            DataTimeframe.ONE_WEEK,
            DataTimeframe.ONE_MONTH
        ]
        
        self.event_system = event_system or EventSystem()
        
        # Subscription tracking
        self.subscribed_symbols = set()
        self.instrument_token_map = {}
        self.symbol_token_map = {}
        
        # Data buffer for real-time data
        self.data_buffer = {}

    def connect(self) -> bool:
        """Connect to Zerodha.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        if self.connected:
            logger.info("Already connected to Zerodha")
            return True

        try:
            # Test connection by fetching a simple API endpoint
            self.kite.margins()
            self.connected = True
            logger.info("Connected to Zerodha")
            return True
        except Exception as e:
            error_msg = f"Failed to connect to Zerodha: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg)

    def disconnect(self) -> bool:
        """Disconnect from Zerodha.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        if not self.connected:
            return True

        try:
            if self.ticker_running and self.ticker:
                self.ticker.close()
                self.ticker_running = False

            self.connected = False
            self.authenticated = False
            self.access_token = None
            logger.info("Disconnected from Zerodha")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Zerodha: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to Zerodha.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connected

    def authenticate(self, request_token: str) -> bool:
        """Authenticate with Zerodha using the request token.

        Args:
            request_token: The request token from the callback URL.

        Returns:
            bool: Whether authentication was successful.

        Raises:
            DataConnectionError: If authentication fails.
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.authenticated = True
            logger.info("Authenticated with Zerodha")
            
            # Initialize the ticker with the access token
            self.ticker = KiteTicker(api_key=self.api_key, access_token=self.access_token)
            self._setup_ticker_callbacks()
            
            # Load instrument token mappings
            self._load_instrument_tokens()
            
            return True
        except Exception as e:
            error_msg = f"Failed to authenticate with Zerodha: {str(e)}"
            logger.error(error_msg)
            self._emit_event("data_error", {
                "error": str(e),
                "action": "authenticate"
            })
            raise DataConnectionError(error_msg)

    def _load_instrument_tokens(self) -> None:
        """Load instrument tokens for all available instruments."""
        try:
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                symbol = instrument["tradingsymbol"]
                token = instrument["instrument_token"]
                self.instrument_token_map[symbol] = token
                self.symbol_token_map[token] = symbol
            
            logger.info(f"Loaded {len(instruments)} instrument tokens")
        except Exception as e:
            logger.error(f"Error loading instrument tokens: {str(e)}")

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
            token = tick["instrument_token"]
            symbol = self.symbol_token_map.get(token)
            
            if not symbol:
                continue
                
            # Process and store the tick data
            timestamp = tick.get("timestamp") or datetime.now()
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
                
            tick_data = {
                "timestamp": timestamp,
                "open": tick.get("ohlc", {}).get("open", tick.get("last_price")),
                "high": tick.get("ohlc", {}).get("high", tick.get("last_price")),
                "low": tick.get("ohlc", {}).get("low", tick.get("last_price")),
                "close": tick.get("last_price"),
                "volume": tick.get("volume"),
            }
            
            # Store in buffer
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = []
            
            self.data_buffer[symbol].append(tick_data)
            
            # Limit buffer size
            if len(self.data_buffer[symbol]) > 1000:  # Arbitrary limit
                self.data_buffer[symbol] = self.data_buffer[symbol][-1000:]
            
            # Emit event for real-time data subscribers
            self._emit_event("market_tick", {
                "symbol": symbol,
                "data": tick_data,
                "timeframe": DataTimeframe.TICK.value
            })

    def _on_connect(self, ws) -> None:
        """Callback when WebSocket connects.

        Args:
            ws: The WebSocket instance.
        """
        logger.info("Ticker connected")
        self._emit_event("ticker_connected", {})
        
        # Resubscribe to symbols
        if self.subscribed_symbols:
            self._subscribe_symbols()

    def _on_close(self, ws) -> None:
        """Callback when WebSocket closes.

        Args:
            ws: The WebSocket instance.
        """
        logger.info("Ticker disconnected")
        self._emit_event("ticker_disconnected", {})

    def _on_error(self, ws, error) -> None:
        """Callback when WebSocket error occurs.

        Args:
            ws: The WebSocket instance.
            error: The error.
        """
        logger.error(f"Ticker error: {str(error)}")
        self._emit_event("ticker_error", {"error": str(error)})

    def _on_reconnect(self, ws, attempts_count) -> None:
        """Callback when WebSocket reconnects.

        Args:
            ws: The WebSocket instance.
            attempts_count: The number of reconnection attempts.
        """
        logger.info(f"Ticker reconnecting: attempt {attempts_count}")
        self._emit_event("ticker_reconnecting", {"attempts": attempts_count})

    def _on_noreconnect(self, ws) -> None:
        """Callback when reconnection fails.

        Args:
            ws: The WebSocket instance.
        """
        logger.error("Ticker failed to reconnect")
        self._emit_event("ticker_reconnect_failed", {})

    def _on_order_update(self, ws, data) -> None:
        """Callback when order update is received.

        Args:
            ws: The WebSocket instance.
            data: The order update data.
        """
        self._emit_event("order_update", data)

    def start_ticker(self) -> None:
        """Start the ticker for real-time data.

        Raises:
            DataConnectionError: If not authenticated or ticker already running.
        """
        if not self.authenticated:
            raise DataConnectionError("Cannot start ticker: Not authenticated with Zerodha")

        if self.ticker_running:
            logger.warning("Ticker is already running")
            return

        if not self.ticker:
            self.ticker = KiteTicker(
                api_key=self.api_key,
                access_token=self.access_token
            )
            self._setup_ticker_callbacks()

        try:
            self.ticker.connect(threaded=True)
            self.ticker_running = True
            logger.info("Ticker started")
        except Exception as e:
            error_msg = f"Error starting ticker: {str(e)}"
            logger.error(error_msg)
            self._emit_event("data_error", {
                "error": str(e),
                "action": "start_ticker"
            })
            raise DataConnectionError(error_msg)

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
            self._emit_event("data_error", {
                "error": str(e),
                "action": "stop_ticker"
            })

    def subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to symbols for real-time data.

        Args:
            symbols: The symbols to subscribe to.

        Raises:
            DataConnectionError: If not authenticated or ticker not running.
        """
        if not self.authenticated:
            raise DataConnectionError("Cannot subscribe to symbols: Not authenticated with Zerodha")

        # Add symbols to subscription set
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)

        # Subscribe if ticker is running
        if self.ticker_running:
            self._subscribe_symbols()

    def _subscribe_symbols(self) -> None:
        """Subscribe to the current set of symbols."""
        if not self.ticker or not self.ticker_running:
            return

        try:
            # Convert symbols to instrument tokens
            tokens = []
            for symbol in self.subscribed_symbols:
                token = self.instrument_token_map.get(symbol)
                if token:
                    tokens.append(token)

            if tokens:
                self.ticker.subscribe(tokens)
                logger.info(f"Subscribed to {len(tokens)} symbols")
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {str(e)}")
            self._emit_event("data_error", {
                "error": str(e),
                "action": "subscribe_symbols"
            })

    def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols for real-time data.

        Args:
            symbols: The symbols to unsubscribe from.
        """
        if not self.ticker or not self.ticker_running:
            return

        try:
            # Convert symbols to instrument tokens and remove from subscription set
            tokens = []
            for symbol in symbols:
                self.subscribed_symbols.discard(symbol)
                token = self.instrument_token_map.get(symbol)
                if token:
                    tokens.append(token)

            if tokens:
                self.ticker.unsubscribe(tokens)
                logger.info(f"Unsubscribed from {len(tokens)} symbols")
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {str(e)}")
            self._emit_event("data_error", {
                "error": str(e),
                "action": "unsubscribe_symbols"
            })

    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from Zerodha.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to Zerodha fails.
            DataValidationError: If the fetched data is invalid.
        """
        if not self.is_connected():
            self.connect()

        if not self.authenticated:
            raise DataConnectionError("Cannot fetch data: Not authenticated with Zerodha")

        # For tick data, return from buffer if available
        if timeframe == DataTimeframe.TICK:
            return self._get_tick_data(symbol, start_date, end_date, limit)

        # For other timeframes, fetch historical data
        try:
            # Get instrument token for the symbol
            instrument_token = self.instrument_token_map.get(symbol)
            if not instrument_token:
                raise DataValidationError(f"Unknown symbol: {symbol}")

            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                # Default to 1 year of data
                start_date = end_date - timedelta(days=365)

            # Get Zerodha interval from timeframe
            interval = self.TIMEFRAME_MAP.get(timeframe)
            if not interval:
                raise DataValidationError(f"Unsupported timeframe: {timeframe}")

            # Fetch data from Zerodha
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
                continuous=False
            )

            if not data:
                logger.warning(f"No data returned for {symbol} with timeframe {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            df.rename(columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            }, inplace=True)

            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise DataValidationError("Data missing timestamp column")

            # Apply limit if specified
            if limit is not None and limit > 0:
                df = df.tail(limit)

            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)

            logger.info(f"Fetched {len(df)} candles for {symbol} with timeframe {timeframe}")
            return df

        except Exception as e:
            error_msg = f"Error fetching data from Zerodha: {str(e)}"
            logger.error(error_msg)
            self._emit_event("data_error", {
                "error": str(e),
                "action": "fetch_data",
                "symbol": symbol,
                "timeframe": timeframe.value
            })
            raise DataConnectionError(error_msg)

    def _get_tick_data(self, symbol: str, start_date: Optional[datetime], end_date: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Get tick data from the buffer.

        Args:
            symbol: The symbol to get data for.
            start_date: The start date for the data.
            end_date: The end date for the data.
            limit: The maximum number of data points to get.

        Returns:
            pd.DataFrame: The tick data as a pandas DataFrame.
        """
        if symbol not in self.data_buffer or not self.data_buffer[symbol]:
            logger.warning(f"No tick data available for {symbol}")
            return pd.DataFrame()

        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer[symbol])

        # Apply date filters if specified
        if start_date is not None:
            df = df[df["timestamp"] >= start_date]
        if end_date is not None:
            df = df[df["timestamp"] <= end_date]

        # Apply limit if specified
        if limit is not None and limit > 0:
            df = df.tail(limit)

        # Sort by timestamp
        df.sort_values("timestamp", inplace=True)

        return df

    def get_symbols(self) -> List[str]:
        """Get available symbols from Zerodha.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to Zerodha fails.
        """
        if not self.is_connected():
            self.connect()

        if not self.authenticated:
            raise DataConnectionError("Cannot get symbols: Not authenticated with Zerodha")

        try:
            # Return cached symbols if available
            if self.instrument_token_map:
                return list(self.instrument_token_map.keys())

            # Otherwise, load instruments and return symbols
            self._load_instrument_tokens()
            return list(self.instrument_token_map.keys())
        except Exception as e:
            error_msg = f"Error getting symbols from Zerodha: {str(e)}"
            logger.error(error_msg)
            self._emit_event("data_error", {
                "error": str(e),
                "action": "get_symbols"
            })
            raise DataConnectionError(error_msg)

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from Zerodha.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event through the event system.

        Args:
            event_type: The type of the event.
            data: The event data.
        """
        if self.event_system:
            self.event_system.emit(Event(f"data.zerodha.{event_type}", data))