"""WebSocket-based real-time data stream implementation.

This module provides a WebSocket-based implementation of the RealTimeDataStream
for streaming market data in real-time.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set

import pandas as pd

from src.data.acquisition.real_time_data_stream import (
    RealTimeDataStream,
    StreamStatus,
    StreamEvent
)
from src.data.acquisition.adapters.websocket_adapter import WebSocketAdapter
from src.data.acquisition.data_fetcher import (
    DataSourceType,
    DataTimeframe,
    DataConnectionError
)
from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.event import Event as SystemEvent, EventSystem

# Create logger
logger = get_logger(__name__)


class WebSocketDataStream(RealTimeDataStream):
    """WebSocket-based implementation of RealTimeDataStream.

    This class extends the RealTimeDataStream to provide real-time market data
    streaming through WebSocket connections.
    
    Attributes:
        adapter: The WebSocket adapter instance.
        event_system: The event system for publishing events.
        config: Configuration manager.
        subscribed_symbols: Set of subscribed symbols.
        subscribed_timeframes: Set of subscribed timeframes.
        data_buffers: Dictionary of data buffers for each symbol and timeframe.
        buffer_size: Maximum size of data buffers.
        throttle_interval: Interval for throttling data updates.
        last_throttled_update: Dictionary tracking last update time for throttling.
    """

    def __init__(
        self,
        adapter: WebSocketAdapter,
        event_system: Optional[EventSystem] = None,
        config: Optional[ConfigManager] = None
    ):
        """Initialize the WebSocket data stream.

        Args:
            adapter: The WebSocket adapter instance.
            event_system: The event system for publishing events. If None, a new one will be created.
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(
            source_type=DataSourceType.WEBSOCKET,
            adapter=adapter,
            event_system=event_system,
            config=config
        )
        
        # Load WebSocket-specific configuration
        self.throttle_interval = self.config.get("data.websocket.throttle_interval", 0.0)  # seconds
        self.last_throttled_update = {}
        
        # Set up event handlers for WebSocket events
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for WebSocket events."""
        self.event_system.subscribe("websocket.connected", self._handle_websocket_connected)
        self.event_system.subscribe("websocket.disconnected", self._handle_websocket_disconnected)
        self.event_system.subscribe("websocket.error", self._handle_websocket_error)
        self.event_system.subscribe("websocket.data", self._handle_websocket_data)

    def _handle_websocket_connected(self, event: SystemEvent) -> None:
        """Handle WebSocket connected event.

        Args:
            event: The event object.
        """
        logger.info("WebSocket connected")
        self._set_status(StreamStatus.CONNECTED)
        
        # Resubscribe to symbols
        for symbol in self.subscribed_symbols:
            self.subscribe(symbol, list(self.subscribed_timeframes))

    def _handle_websocket_disconnected(self, event: SystemEvent) -> None:
        """Handle WebSocket disconnected event.

        Args:
            event: The event object.
        """
        logger.info("WebSocket disconnected")
        self._set_status(StreamStatus.DISCONNECTED)

    def _handle_websocket_error(self, event: SystemEvent) -> None:
        """Handle WebSocket error event.

        Args:
            event: The event object.
        """
        error = event.data.get("error", "Unknown error")
        logger.error(f"WebSocket error: {error}")
        
        # Don't change status here, let the reconnection logic handle it

    def _handle_websocket_data(self, event: SystemEvent) -> None:
        """Handle WebSocket data event.

        Args:
            event: The event object.
        """
        data = event.data
        symbol = data.get("symbol")
        timeframe_str = data.get("timeframe")
        market_data = data.get("data")
        
        if not symbol or not timeframe_str or not market_data:
            logger.warning(f"Received incomplete data event: {data}")
            return
            
        # Convert timeframe string to enum
        timeframe = None
        for tf in DataTimeframe:
            if tf.value == timeframe_str:
                timeframe = tf
                break
                
        if not timeframe:
            logger.warning(f"Received data with unknown timeframe: {timeframe_str}")
            return
            
        # Update last data timestamp
        self.last_data_timestamp = datetime.now()
        
        # Apply throttling if configured
        if self.throttle_interval > 0:
            key = f"{symbol}_{timeframe.value}"
            now = time.time()
            last_update = self.last_throttled_update.get(key, 0)
            
            if now - last_update < self.throttle_interval:
                # Skip this update due to throttling
                return
                
            # Update last throttled update time
            self.last_throttled_update[key] = now
            
        # Process the data
        self._process_stream_data(symbol, timeframe, market_data)

    def _process_stream_data(self, symbol: str, timeframe: DataTimeframe, data: Dict[str, Any]) -> None:
        """Process streaming data.

        Args:
            symbol: The symbol of the data.
            timeframe: The timeframe of the data.
            data: The market data.
        """
        # Update data buffer
        self._update_data_buffer(symbol, timeframe, data)
        
        # Emit data event based on data type
        if timeframe == DataTimeframe.TICK:
            # Tick data
            self._emit_data_event(StreamEvent.TICK, symbol, timeframe, data)
        elif "bid" in data and "ask" in data:
            # Quote data
            self._emit_data_event(StreamEvent.QUOTE, symbol, timeframe, data)
        elif all(field in data for field in ["open", "high", "low", "close"]):
            # Bar data
            self._emit_data_event(StreamEvent.BAR, symbol, timeframe, data)
        elif "price" in data and "volume" in data:
            # Trade data
            self._emit_data_event(StreamEvent.TRADE, symbol, timeframe, data)
        else:
            # Custom data
            self._emit_data_event(StreamEvent.CUSTOM, symbol, timeframe, data)

    def connect(self) -> bool:
        """Connect to the WebSocket stream.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        if self.is_connected():
            logger.debug("Already connected to WebSocket stream")
            return True
            
        try:
            # Connect using the adapter
            success = self.adapter.connect()
            
            if success:
                self._set_status(StreamStatus.CONNECTED)
                logger.info("Connected to WebSocket stream")
                return True
            else:
                self._set_status(StreamStatus.ERROR)
                logger.error("Failed to connect to WebSocket stream")
                return False
                
        except DataConnectionError as e:
            self._set_status(StreamStatus.ERROR)
            logger.error(f"Error connecting to WebSocket stream: {str(e)}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the WebSocket stream.

        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        if not self.is_connected():
            logger.debug("Not connected to WebSocket stream")
            return True
            
        try:
            # Disconnect using the adapter
            success = self.adapter.disconnect()
            
            if success:
                self._set_status(StreamStatus.DISCONNECTED)
                logger.info("Disconnected from WebSocket stream")
                return True
            else:
                logger.error("Failed to disconnect from WebSocket stream")
                return False
                
        except Exception as e:
            logger.error(f"Error disconnecting from WebSocket stream: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to the WebSocket stream.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.adapter.is_connected()

    def subscribe(self, symbol: str, timeframes: List[DataTimeframe]) -> bool:
        """Subscribe to a symbol with specified timeframes.

        Args:
            symbol: The symbol to subscribe to.
            timeframes: The timeframes to subscribe to.

        Returns:
            bool: True if subscription is successful, False otherwise.
        """
        if not self.is_connected():
            logger.warning("Cannot subscribe: Not connected to WebSocket stream")
            return False
            
        try:
            # Use WebSocketAdapter's _subscribe_symbol method
            if hasattr(self.adapter, "_subscribe_symbol"):
                self.adapter._subscribe_symbol(symbol, timeframes)
            else:
                # Fallback if adapter doesn't have the method
                logger.warning("Adapter does not support direct subscription")
                
            # Update subscribed symbols and timeframes
            self.subscribed_symbols.add(symbol)
            for tf in timeframes:
                self.subscribed_timeframes.add(tf)
                
            # Initialize data buffers for this symbol and timeframes
            for tf in timeframes:
                buffer_key = f"{symbol}_{tf.value}"
                if buffer_key not in self.data_buffers:
                    self.data_buffers[buffer_key] = []
                    
            logger.info(f"Subscribed to {symbol} with timeframes {[tf.value for tf in timeframes]}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {str(e)}")
            return False

    def unsubscribe(self, symbol: str, timeframes: Optional[List[DataTimeframe]] = None) -> bool:
        """Unsubscribe from a symbol with specified timeframes.

        Args:
            symbol: The symbol to unsubscribe from.
            timeframes: The timeframes to unsubscribe from. If None, unsubscribe from all timeframes.

        Returns:
            bool: True if unsubscription is successful, False otherwise.
        """
        if not self.is_connected():
            logger.warning("Cannot unsubscribe: Not connected to WebSocket stream")
            return False
            
        try:
            # Use WebSocketAdapter's _unsubscribe_symbol method
            if hasattr(self.adapter, "_unsubscribe_symbol"):
                self.adapter._unsubscribe_symbol(symbol, timeframes)
            else:
                # Fallback if adapter doesn't have the method
                logger.warning("Adapter does not support direct unsubscription")
                
            # Update subscribed symbols and timeframes
            if not timeframes:
                # Unsubscribe from all timeframes
                self.subscribed_symbols.discard(symbol)
                
                # Remove all data buffers for this symbol
                for key in list(self.data_buffers.keys()):
                    if key.startswith(f"{symbol}_"):
                        del self.data_buffers[key]
            else:
                # Unsubscribe from specific timeframes
                for tf in timeframes:
                    # Remove data buffer for this symbol and timeframe
                    buffer_key = f"{symbol}_{tf.value}"
                    if buffer_key in self.data_buffers:
                        del self.data_buffers[buffer_key]
                        
                # Check if any timeframes left for this symbol
                symbol_timeframes = [key.split("_")[1] for key in self.data_buffers.keys() 
                                    if key.startswith(f"{symbol}_")]
                                    
                if not symbol_timeframes:
                    # No timeframes left, remove symbol from subscribed symbols
                    self.subscribed_symbols.discard(symbol)
                    
            logger.info(f"Unsubscribed from {symbol}" + 
                      (f" with timeframes {[tf.value for tf in timeframes]}" if timeframes else ""))
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {str(e)}")
            return False

    def _stream_worker(self) -> None:
        """Worker thread for streaming data.

        This method is called by the parent class's start() method.
        For WebSocket streams, this mainly handles reconnection and heartbeat checks.
        """
        logger.info("WebSocket stream worker started")
        
        while not self.stop_event.is_set():
            try:
                # Check if connected
                if not self.is_connected():
                    # Try to reconnect
                    if self.reconnect_attempts > 0:
                        logger.info(f"Attempting to reconnect to WebSocket stream (attempt {self.reconnect_count + 1}/{self.reconnect_attempts})")
                        
                        # Increment reconnect count
                        self.reconnect_count += 1
                        
                        # Calculate backoff delay
                        backoff_factor = 1.5  # Exponential backoff factor
                        delay = self.reconnect_delay * (backoff_factor ** (self.reconnect_count - 1))
                        
                        # Try to reconnect
                        success = self.connect()
                        
                        if success:
                            # Reset reconnect count on successful connection
                            self.reconnect_count = 0
                            logger.info("Successfully reconnected to WebSocket stream")
                        else:
                            # Failed to reconnect
                            if self.reconnect_count >= self.reconnect_attempts:
                                logger.error(f"Failed to reconnect after {self.reconnect_attempts} attempts")
                                self._set_status(StreamStatus.ERROR)
                                break
                                
                            # Wait before next reconnection attempt
                            logger.info(f"Waiting {delay:.1f} seconds before next reconnection attempt")
                            self.stop_event.wait(delay)
                    else:
                        # No reconnection attempts left
                        logger.error("WebSocket stream disconnected and reconnection is disabled")
                        self._set_status(StreamStatus.ERROR)
                        break
                else:
                    # Connected, check heartbeat
                    self._check_heartbeat()
                    
                    # Wait a bit before next iteration
                    self.stop_event.wait(1.0)
                    
            except Exception as e:
                logger.error(f"Error in WebSocket stream worker: {str(e)}")
                self._set_status(StreamStatus.ERROR)
                
                # Wait before retrying
                self.stop_event.wait(self.reconnect_delay)
                
        logger.info("WebSocket stream worker stopped")