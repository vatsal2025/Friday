"""WebSocket adapter for the Friday AI Trading System.

This module provides an adapter that implements the DataSourceAdapter interface
for WebSocket-based market data sources.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
from typing import Dict, List, Optional, Union, Any, Callable

import pandas as pd
import websocket

from src.data.acquisition.data_fetcher import (
    DataSourceAdapter,
    DataTimeframe,
    DataConnectionError,
    DataValidationError
)
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.event import Event as SystemEvent, EventSystem

# Create logger
logger = get_logger(__name__)


class WebSocketAdapter(DataSourceAdapter):
    """Adapter for WebSocket-based market data that implements the DataSourceAdapter interface.

    This class adapts WebSocket connections to the DataSourceAdapter interface,
    allowing the system to fetch real-time streaming data through the common data API.
    
    Attributes:
        url: The WebSocket endpoint URL.
        connected: Whether the adapter is connected to the WebSocket server.
        ws: The WebSocket client instance.
        ws_thread: The thread running the WebSocket connection.
        data_queue: Queue for storing received data.
        available_symbols: List of available symbols.
        available_timeframes: List of available timeframes.
        auth_params: Authentication parameters for the WebSocket connection.
        event_system: The event system for publishing events.
        config: Configuration manager.
    """

    def __init__(
        self,
        url: str,
        auth_params: Optional[Dict[str, Any]] = None,
        available_symbols: Optional[List[str]] = None,
        available_timeframes: Optional[List[DataTimeframe]] = None,
        event_system: Optional[EventSystem] = None,
        config: Optional[ConfigManager] = None,
    ):
        """Initialize the WebSocket adapter.

        Args:
            url: The WebSocket endpoint URL.
            auth_params: Authentication parameters for the WebSocket connection.
            available_symbols: List of available symbols. If None, will be populated on connect.
            available_timeframes: List of available timeframes. If None, will use defaults.
            event_system: The event system for publishing events. If None, a new one will be created.
            config: Configuration manager. If None, a new one will be created.
        """
        self.url = url
        self.auth_params = auth_params or {}
        self.connected = False
        self.ws = None
        self.ws_thread = None
        self.data_queue = Queue()
        self.available_symbols = available_symbols or []
        self.available_timeframes = available_timeframes or [
            DataTimeframe.TICK,
            DataTimeframe.ONE_MINUTE,
            DataTimeframe.FIVE_MINUTES,
            DataTimeframe.FIFTEEN_MINUTES,
            DataTimeframe.THIRTY_MINUTES,
            DataTimeframe.ONE_HOUR,
            DataTimeframe.ONE_DAY,
        ]
        self.event_system = event_system or EventSystem()
        self.config = config or ConfigManager()
        
        # Load configuration
        self.reconnect_attempts = self.config.get("data.websocket.reconnect_attempts", 5)
        self.reconnect_delay = self.config.get("data.websocket.reconnect_delay", 5)  # seconds
        self.heartbeat_interval = self.config.get("data.websocket.heartbeat_interval", 30)  # seconds
        
        # Subscription tracking
        self.subscribed_symbols = set()
        self.subscribed_timeframes = set()
        
        # Last received data timestamp for monitoring connection health
        self.last_data_timestamp = datetime.now()
        
        # Reconnection tracking
        self.reconnect_count = 0
        self.reconnect_backoff_factor = self.config.get("data.websocket.reconnect_backoff_factor", 1.5)

    def connect(self) -> bool:
        """Connect to the WebSocket server.

        Returns:
            bool: True if connection is successful, False otherwise.

        Raises:
            DataConnectionError: If connection fails.
        """
        if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
            logger.debug("Already connected to WebSocket server")
            return True
            
        try:
            # Initialize WebSocket client with callbacks
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                header=self._get_headers()
            )
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.ws_thread.start()
            
            # Wait for connection to establish or fail
            timeout = time.time() + 10  # 10 seconds timeout
            while time.time() < timeout:
                if self.connected:
                    return True
                time.sleep(0.1)
                
            # If we get here, connection timed out
            logger.warning("Timeout while connecting to WebSocket server")
            return False
            
        except Exception as e:
            error_msg = f"Failed to connect to WebSocket server: {str(e)}"
            logger.error(error_msg)
            raise DataConnectionError(error_msg) from e

    def disconnect(self) -> bool:
        """Disconnect from the WebSocket server.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        if not self.connected or not self.ws:
            logger.debug("Not connected to WebSocket server")
            return True
            
        try:
            # Close WebSocket connection
            self.ws.close()
            
            # Wait for thread to finish
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5.0)
                
            self.connected = False
            logger.info("Disconnected from WebSocket server")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from WebSocket server: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to the WebSocket server.

        Returns:
            bool: True if connected, False otherwise.
        """
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.connected = True
            return True
        else:
            self.connected = False
            return False

    def fetch_data(
        self,
        symbol: str,
        timeframe: DataTimeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from the WebSocket server.

        For WebSocket connections, this method returns the latest data received
        from the WebSocket stream for the specified symbol and timeframe.

        Args:
            symbol: The symbol to fetch data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Ignored for WebSocket.
            end_date: The end date for the data. Ignored for WebSocket.
            limit: The maximum number of data points to fetch. Defaults to 100.

        Returns:
            pd.DataFrame: The fetched data as a pandas DataFrame.

        Raises:
            DataConnectionError: If connection to the WebSocket server fails.
            DataValidationError: If the fetched data is invalid.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e
                
        # Subscribe to the symbol if not already subscribed
        if symbol not in self.subscribed_symbols:
            self._subscribe_symbol(symbol, [timeframe])
            
        # Get data from queue
        data = []
        limit = limit or 100
        
        # Non-blocking get from queue with timeout
        timeout = time.time() + 1  # 1 second timeout
        while len(data) < limit and time.time() < timeout:
            try:
                item = self.data_queue.get(block=False)
                if item["symbol"] == symbol and item["timeframe"] == timeframe:
                    data.append(item["data"])
                self.data_queue.task_done()
            except Exception:
                # Queue is empty
                break
                
        if not data:
            logger.warning(f"No data available for {symbol} with timeframe {timeframe.value}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate data
        try:
            self._validate_data(df)
            return df
        except DataValidationError as e:
            logger.error(f"Data validation error: {str(e)}")
            return pd.DataFrame()

    def get_symbols(self) -> List[str]:
        """Get available symbols from the WebSocket server.

        Returns:
            List[str]: List of available symbols.

        Raises:
            DataConnectionError: If connection to the WebSocket server fails.
        """
        if not self.is_connected():
            try:
                self.connect()
            except DataConnectionError as e:
                raise e
                
        return self.available_symbols

    def get_timeframes(self) -> List[DataTimeframe]:
        """Get available timeframes from the WebSocket server.

        Returns:
            List[DataTimeframe]: List of available timeframes.
        """
        return self.available_timeframes

    def _run_websocket(self) -> None:
        """Run the WebSocket connection in a loop with reconnection logic."""
        while True:
            try:
                # Run WebSocket connection (this blocks until connection is closed)
                self.ws.run_forever()
                
                # If we get here, connection was closed
                logger.info("WebSocket connection closed")
                
                # If we're intentionally disconnected, break the loop
                if not self.connected:
                    break
                    
                # Otherwise, try to reconnect
                if self.reconnect_count < self.reconnect_attempts:
                    self.reconnect_count += 1
                    delay = self.reconnect_delay * (self.reconnect_backoff_factor ** (self.reconnect_count - 1))
                    logger.info(f"Reconnecting to WebSocket server in {delay:.1f} seconds (attempt {self.reconnect_count}/{self.reconnect_attempts})")
                    time.sleep(delay)
                    
                    # Reinitialize WebSocket client
                    self.ws = websocket.WebSocketApp(
                        self.url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                        header=self._get_headers()
                    )
                else:
                    logger.error(f"Failed to reconnect after {self.reconnect_attempts} attempts")
                    break
            except Exception as e:
                logger.error(f"Error in WebSocket thread: {str(e)}")
                break

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for WebSocket connection.

        Returns:
            Dict[str, str]: Headers dictionary.
        """
        headers = {}
        
        # Add authentication headers if provided
        if "api_key" in self.auth_params:
            headers["X-API-Key"] = self.auth_params["api_key"]
            
        if "token" in self.auth_params:
            headers["Authorization"] = f"Bearer {self.auth_params['token']}"
            
        return headers

    def _on_open(self, ws) -> None:
        """Callback when WebSocket connection is opened.

        Args:
            ws: The WebSocket instance.
        """
        logger.info("Connected to WebSocket server")
        self.connected = True
        self.reconnect_count = 0
        self.last_data_timestamp = datetime.now()
        
        # Authenticate if needed
        self._authenticate()
        
        # Resubscribe to symbols
        for symbol in self.subscribed_symbols:
            self._subscribe_symbol(symbol, list(self.subscribed_timeframes))
            
        # Emit connection event
        self._emit_event("websocket.connected", {
            "url": self.url,
            "timestamp": datetime.now().isoformat()
        })

    def _on_message(self, ws, message) -> None:
        """Callback when a message is received.

        Args:
            ws: The WebSocket instance.
            message: The received message.
        """
        try:
            # Update last data timestamp
            self.last_data_timestamp = datetime.now()
            
            # Parse message
            data = json.loads(message)
            
            # Process data based on message type
            if "type" in data:
                if data["type"] == "heartbeat":
                    logger.debug("Received heartbeat")
                    return
                    
                if data["type"] == "error":
                    logger.error(f"Received error from WebSocket server: {data.get('message', 'Unknown error')}")
                    return
                    
            # Extract symbol and data
            symbol = data.get("symbol")
            if not symbol:
                logger.warning(f"Received data without symbol: {data}")
                return
                
            # Determine timeframe
            timeframe = self._determine_timeframe(data)
            if not timeframe:
                logger.warning(f"Could not determine timeframe for data: {data}")
                return
                
            # Normalize data format
            normalized_data = self._normalize_data(data)
            
            # Add to queue
            self.data_queue.put({
                "symbol": symbol,
                "timeframe": timeframe,
                "data": normalized_data
            })
            
            # Emit data event
            self._emit_event("websocket.data", {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "data": normalized_data,
                "timestamp": datetime.now().isoformat()
            })
            
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")

    def _on_error(self, ws, error) -> None:
        """Callback when an error occurs.

        Args:
            ws: The WebSocket instance.
            error: The error.
        """
        logger.error(f"WebSocket error: {str(error)}")
        
        # Emit error event
        self._emit_event("websocket.error", {
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })

    def _on_close(self, ws, close_status_code, close_reason) -> None:
        """Callback when WebSocket connection is closed.

        Args:
            ws: The WebSocket instance.
            close_status_code: The close status code.
            close_reason: The close reason.
        """
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_reason}")
        self.connected = False
        
        # Emit disconnection event
        self._emit_event("websocket.disconnected", {
            "code": close_status_code,
            "reason": close_reason,
            "timestamp": datetime.now().isoformat()
        })

    def _authenticate(self) -> None:
        """Authenticate with the WebSocket server if required."""
        if not self.auth_params:
            return
            
        try:
            auth_message = {
                "type": "auth",
                **self.auth_params
            }
            
            self.ws.send(json.dumps(auth_message))
            logger.debug("Sent authentication message")
            
        except Exception as e:
            logger.error(f"Error authenticating with WebSocket server: {str(e)}")

    def _subscribe_symbol(self, symbol: str, timeframes: List[DataTimeframe]) -> None:
        """Subscribe to a symbol.

        Args:
            symbol: The symbol to subscribe to.
            timeframes: The timeframes to subscribe to.
        """
        if not self.is_connected():
            logger.warning("Cannot subscribe: Not connected to WebSocket server")
            return
            
        try:
            # Add to subscribed symbols and timeframes
            self.subscribed_symbols.add(symbol)
            for tf in timeframes:
                self.subscribed_timeframes.add(tf)
                
            # Create subscription message
            subscribe_message = {
                "type": "subscribe",
                "symbol": symbol,
                "timeframes": [tf.value for tf in timeframes]
            }
            
            # Send subscription message
            self.ws.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {symbol} with timeframes {[tf.value for tf in timeframes]}")
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {str(e)}")

    def _unsubscribe_symbol(self, symbol: str, timeframes: Optional[List[DataTimeframe]] = None) -> None:
        """Unsubscribe from a symbol.

        Args:
            symbol: The symbol to unsubscribe from.
            timeframes: The timeframes to unsubscribe from. If None, unsubscribe from all timeframes.
        """
        if not self.is_connected():
            logger.warning("Cannot unsubscribe: Not connected to WebSocket server")
            return
            
        try:
            # Create unsubscription message
            unsubscribe_message = {
                "type": "unsubscribe",
                "symbol": symbol
            }
            
            if timeframes:
                unsubscribe_message["timeframes"] = [tf.value for tf in timeframes]
                
            # Send unsubscription message
            self.ws.send(json.dumps(unsubscribe_message))
            
            # Update subscribed symbols and timeframes
            if not timeframes:
                self.subscribed_symbols.discard(symbol)
            else:
                # Only remove specific timeframes
                for tf in timeframes:
                    self.subscribed_timeframes.discard(tf)
                    
                # If no timeframes left for this symbol, remove it completely
                if not any(tf in self.subscribed_timeframes for tf in timeframes):
                    self.subscribed_symbols.discard(symbol)
                    
            logger.info(f"Unsubscribed from {symbol}" + 
                      (f" with timeframes {[tf.value for tf in timeframes]}" if timeframes else ""))
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {str(e)}")

    def _determine_timeframe(self, data: Dict[str, Any]) -> Optional[DataTimeframe]:
        """Determine the timeframe of the data.

        Args:
            data: The data to determine timeframe for.

        Returns:
            Optional[DataTimeframe]: The determined timeframe, or None if it cannot be determined.
        """
        # If timeframe is explicitly provided
        if "timeframe" in data:
            timeframe_str = data["timeframe"]
            for tf in DataTimeframe:
                if tf.value == timeframe_str:
                    return tf
                    
        # If this is tick data
        if "price" in data and "timestamp" in data and not any(k in data for k in ["open", "high", "low", "close"]):
            return DataTimeframe.TICK
            
        # If this is OHLCV data
        if all(k in data for k in ["open", "high", "low", "close"]):
            # Try to determine timeframe from interval if provided
            if "interval" in data:
                interval = data["interval"]
                if interval == "1m":
                    return DataTimeframe.ONE_MINUTE
                elif interval == "5m":
                    return DataTimeframe.FIVE_MINUTES
                elif interval == "15m":
                    return DataTimeframe.FIFTEEN_MINUTES
                elif interval == "30m":
                    return DataTimeframe.THIRTY_MINUTES
                elif interval == "1h":
                    return DataTimeframe.ONE_HOUR
                elif interval == "1d":
                    return DataTimeframe.ONE_DAY
                    
            # Default to 1-minute for OHLCV data
            return DataTimeframe.ONE_MINUTE
            
        # Could not determine timeframe
        return None

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data to a standard format.

        Args:
            data: The data to normalize.

        Returns:
            Dict[str, Any]: The normalized data.
        """
        normalized = {}
        
        # Copy timestamp if available
        if "timestamp" in data:
            normalized["timestamp"] = data["timestamp"]
        else:
            normalized["timestamp"] = datetime.now().isoformat()
            
        # Copy symbol if available
        if "symbol" in data:
            normalized["symbol"] = data["symbol"]
            
        # Handle tick data
        if "price" in data:
            normalized["price"] = data["price"]
            if "volume" in data:
                normalized["volume"] = data["volume"]
                
        # Handle OHLCV data
        for field in ["open", "high", "low", "close", "volume"]:
            if field in data:
                normalized[field] = data[field]
                
        # Handle quote data
        for field in ["bid", "ask", "bid_size", "ask_size"]:
            if field in data:
                normalized[field] = data[field]
                
        return normalized

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data.

        Args:
            data: The data to validate.

        Returns:
            bool: True if data is valid.

        Raises:
            DataValidationError: If the data is invalid.
        """
        if data.empty:
            raise DataValidationError("Data is empty")
            
        # Check for required columns based on data type
        if "price" in data.columns:
            # Tick data
            required_columns = ["price", "timestamp"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns for tick data: {missing_columns}")
                
        elif all(col in data.columns for col in ["open", "high", "low", "close"]):
            # OHLCV data
            required_columns = ["open", "high", "low", "close", "timestamp"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns for OHLCV data: {missing_columns}")
                
        elif all(col in data.columns for col in ["bid", "ask"]):
            # Quote data
            required_columns = ["bid", "ask", "timestamp"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns for quote data: {missing_columns}")
                
        else:
            raise DataValidationError("Unknown data format")
            
        return True

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event.

        Args:
            event_type: The event type.
            data: The event data.
        """
        system_event = SystemEvent(
            event_type=event_type,
            data=data,
            source="WebSocketAdapter"
        )
        
        self.event_system.emit(system_event)