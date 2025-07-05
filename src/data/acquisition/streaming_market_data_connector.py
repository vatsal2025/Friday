"""Streaming Market Data Connector for the Friday AI Trading System.

This module provides a connector for streaming market data from various sources,
including WebSocket connections, to the portfolio system.
"""

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
from src.data.acquisition.websocket_data_stream import WebSocketDataStream
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


class StreamingMarketDataConnector:
    """Connector for streaming market data to the portfolio system.

    This class connects streaming market data sources to the portfolio system,
    handling high-frequency updates, backpressure, and throttling.
    
    Attributes:
        stream: The real-time data stream instance.
        event_system: The event system for publishing events.
        config: Configuration manager.
        throttle_enabled: Whether throttling is enabled.
        throttle_interval: Interval for throttling data updates.
        backpressure_enabled: Whether backpressure handling is enabled.
        backpressure_queue_size: Maximum size of the backpressure queue.
        backpressure_queue: Queue for handling backpressure.
        symbol_priorities: Dictionary of symbol priorities for throttling.
        subscribed_symbols: Set of subscribed symbols.
        subscribed_timeframes: Set of subscribed timeframes.
    """

    def __init__(
        self,
        stream: RealTimeDataStream,
        event_system: Optional[EventSystem] = None,
        config: Optional[ConfigManager] = None
    ):
        """Initialize the streaming market data connector.

        Args:
            stream: The real-time data stream instance.
            event_system: The event system for publishing events. If None, a new one will be created.
            config: Configuration manager. If None, a new one will be created.
        """
        self.stream = stream
        self.event_system = event_system or EventSystem()
        self.config = config or ConfigManager()
        
        # Load configuration
        self.throttle_enabled = self.config.get("data.streaming.throttle.enabled", False)
        self.throttle_interval = self.config.get("data.streaming.throttle.interval", 0.1)  # seconds
        self.backpressure_enabled = self.config.get("data.streaming.backpressure.enabled", False)
        self.backpressure_queue_size = self.config.get("data.streaming.backpressure.queue_size", 1000)
        
        # Symbol priorities for throttling (higher priority = less throttling)
        self.symbol_priorities = self.config.get("data.streaming.symbol_priorities", {})
        
        # Backpressure queue
        self.backpressure_queue = []
        
        # Throttling state
        self.last_update_time = {}
        
        # Subscription tracking
        self.subscribed_symbols = set()
        self.subscribed_timeframes = set()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Worker thread for backpressure handling
        self.backpressure_thread = None
        self.stop_event = threading.Event()

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for stream events."""
        # Subscribe to stream events
        self.event_system.subscribe(StreamEvent.TICK.value, self._handle_tick_event)
        self.event_system.subscribe(StreamEvent.TRADE.value, self._handle_trade_event)
        self.event_system.subscribe(StreamEvent.QUOTE.value, self._handle_quote_event)
        self.event_system.subscribe(StreamEvent.BAR.value, self._handle_bar_event)
        self.event_system.subscribe(StreamEvent.ORDERBOOK.value, self._handle_orderbook_event)
        self.event_system.subscribe(StreamEvent.CUSTOM.value, self._handle_custom_event)
        
        # Subscribe to stream status events
        self.event_system.subscribe("stream.status.connected", self._handle_stream_connected)
        self.event_system.subscribe("stream.status.disconnected", self._handle_stream_disconnected)
        self.event_system.subscribe("stream.status.error", self._handle_stream_error)

    def start(self) -> bool:
        """Start the streaming market data connector.

        Returns:
            bool: True if start is successful, False otherwise.
        """
        try:
            # Start the stream
            if not self.stream.start():
                logger.error("Failed to start stream")
                return False
                
            # Start backpressure handling if enabled
            if self.backpressure_enabled:
                self.stop_event.clear()
                self.backpressure_thread = threading.Thread(
                    target=self._backpressure_worker,
                    daemon=True
                )
                self.backpressure_thread.start()
                
            logger.info("Streaming market data connector started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming market data connector: {str(e)}")
            return False

    def stop(self) -> bool:
        """Stop the streaming market data connector.

        Returns:
            bool: True if stop is successful, False otherwise.
        """
        try:
            # Stop the stream
            if not self.stream.stop():
                logger.error("Failed to stop stream")
                return False
                
            # Stop backpressure handling if enabled
            if self.backpressure_enabled and self.backpressure_thread:
                self.stop_event.set()
                self.backpressure_thread.join(timeout=5.0)
                
            logger.info("Streaming market data connector stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping streaming market data connector: {str(e)}")
            return False

    def subscribe(self, symbol: str, timeframes: List[DataTimeframe]) -> bool:
        """Subscribe to a symbol with specified timeframes.

        Args:
            symbol: The symbol to subscribe to.
            timeframes: The timeframes to subscribe to.

        Returns:
            bool: True if subscription is successful, False otherwise.
        """
        try:
            # Subscribe to the stream
            if not self.stream.subscribe(symbol, timeframes):
                logger.error(f"Failed to subscribe to {symbol}")
                return False
                
            # Update subscribed symbols and timeframes
            self.subscribed_symbols.add(symbol)
            for tf in timeframes:
                self.subscribed_timeframes.add(tf)
                
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
        try:
            # Unsubscribe from the stream
            if not self.stream.unsubscribe(symbol, timeframes):
                logger.error(f"Failed to unsubscribe from {symbol}")
                return False
                
            # Update subscribed symbols and timeframes
            if not timeframes:
                # Unsubscribe from all timeframes
                self.subscribed_symbols.discard(symbol)
            else:
                # Check if any timeframes left for this symbol
                remaining_timeframes = self.stream.get_subscribed_timeframes(symbol)
                if not remaining_timeframes:
                    self.subscribed_symbols.discard(symbol)
                    
            logger.info(f"Unsubscribed from {symbol}" + 
                      (f" with timeframes {[tf.value for tf in timeframes]}" if timeframes else ""))
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {str(e)}")
            return False

    def get_latest_data(self, symbol: str, timeframe: DataTimeframe) -> Optional[pd.DataFrame]:
        """Get the latest data for a symbol and timeframe.

        Args:
            symbol: The symbol to get data for.
            timeframe: The timeframe of the data.

        Returns:
            Optional[pd.DataFrame]: The latest data as a pandas DataFrame, or None if not available.
        """
        return self.stream.get_latest_data(symbol, timeframe)

    def get_status(self) -> StreamStatus:
        """Get the current status of the stream.

        Returns:
            StreamStatus: The current status.
        """
        return self.stream.get_status()

    def get_subscribed_symbols(self) -> Set[str]:
        """Get the set of subscribed symbols.

        Returns:
            Set[str]: The set of subscribed symbols.
        """
        return self.subscribed_symbols.copy()

    def get_subscribed_timeframes(self, symbol: Optional[str] = None) -> Set[DataTimeframe]:
        """Get the set of subscribed timeframes.

        Args:
            symbol: The symbol to get timeframes for. If None, get all timeframes.

        Returns:
            Set[DataTimeframe]: The set of subscribed timeframes.
        """
        if symbol:
            return self.stream.get_subscribed_timeframes(symbol)
        else:
            return self.subscribed_timeframes.copy()

    def _handle_tick_event(self, event: SystemEvent) -> None:
        """Handle tick event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.TICK)

    def _handle_trade_event(self, event: SystemEvent) -> None:
        """Handle trade event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.TRADE)

    def _handle_quote_event(self, event: SystemEvent) -> None:
        """Handle quote event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.QUOTE)

    def _handle_bar_event(self, event: SystemEvent) -> None:
        """Handle bar event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.BAR)

    def _handle_orderbook_event(self, event: SystemEvent) -> None:
        """Handle orderbook event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.ORDERBOOK)

    def _handle_custom_event(self, event: SystemEvent) -> None:
        """Handle custom event.

        Args:
            event: The event object.
        """
        self._process_market_data_event(event, StreamEvent.CUSTOM)

    def _handle_stream_connected(self, event: SystemEvent) -> None:
        """Handle stream connected event.

        Args:
            event: The event object.
        """
        logger.info("Stream connected")
        
        # Emit portfolio system event
        self._emit_portfolio_event("market_data.connected", {
            "timestamp": datetime.now().isoformat(),
            "source": self.stream.source_type.value
        })

    def _handle_stream_disconnected(self, event: SystemEvent) -> None:
        """Handle stream disconnected event.

        Args:
            event: The event object.
        """
        logger.info("Stream disconnected")
        
        # Emit portfolio system event
        self._emit_portfolio_event("market_data.disconnected", {
            "timestamp": datetime.now().isoformat(),
            "source": self.stream.source_type.value
        })

    def _handle_stream_error(self, event: SystemEvent) -> None:
        """Handle stream error event.

        Args:
            event: The event object.
        """
        error = event.data.get("error", "Unknown error")
        logger.error(f"Stream error: {error}")
        
        # Emit portfolio system event
        self._emit_portfolio_event("market_data.error", {
            "timestamp": datetime.now().isoformat(),
            "source": self.stream.source_type.value,
            "error": error
        })

    def _process_market_data_event(self, event: SystemEvent, event_type: StreamEvent) -> None:
        """Process market data event.

        Args:
            event: The event object.
            event_type: The type of event.
        """
        data = event.data
        symbol = data.get("symbol")
        timeframe_str = data.get("timeframe")
        market_data = data.get("data")
        
        if not symbol or not timeframe_str or not market_data:
            logger.warning(f"Received incomplete market data event: {data}")
            return
            
        # Convert timeframe string to enum
        timeframe = None
        for tf in DataTimeframe:
            if tf.value == timeframe_str:
                timeframe = tf
                break
                
        if not timeframe:
            logger.warning(f"Received market data with unknown timeframe: {timeframe_str}")
            return
            
        # Apply throttling if enabled
        if self.throttle_enabled and not self._should_process_update(symbol, timeframe, event_type):
            # Skip this update due to throttling
            return
            
        # Apply backpressure if enabled
        if self.backpressure_enabled:
            # Add to backpressure queue
            if len(self.backpressure_queue) < self.backpressure_queue_size:
                self.backpressure_queue.append((event, event_type))
            else:
                logger.warning("Backpressure queue full, dropping market data update")
        else:
            # Process immediately
            self._emit_market_data_event(event, event_type)

    def _should_process_update(self, symbol: str, timeframe: DataTimeframe, event_type: StreamEvent) -> bool:
        """Check if an update should be processed based on throttling rules.

        Args:
            symbol: The symbol of the update.
            timeframe: The timeframe of the update.
            event_type: The type of event.

        Returns:
            bool: True if the update should be processed, False otherwise.
        """
        # Get symbol priority (higher priority = less throttling)
        priority = self.symbol_priorities.get(symbol, 1.0)
        
        # Adjust throttle interval based on priority
        adjusted_interval = self.throttle_interval / priority
        
        # Further adjust based on event type (ticks are more frequent than bars)
        if event_type == StreamEvent.TICK:
            adjusted_interval *= 0.5  # Process ticks more frequently
        elif event_type == StreamEvent.BAR:
            adjusted_interval *= 2.0  # Process bars less frequently
            
        # Check if enough time has passed since last update
        key = f"{symbol}_{timeframe.value}_{event_type.value}"
        now = time.time()
        last_update = self.last_update_time.get(key, 0)
        
        if now - last_update < adjusted_interval:
            # Not enough time has passed
            return False
            
        # Update last update time
        self.last_update_time[key] = now
        return True

    def _backpressure_worker(self) -> None:
        """Worker thread for handling backpressure."""
        logger.info("Backpressure worker started")
        
        while not self.stop_event.is_set():
            try:
                # Process items in the backpressure queue
                if self.backpressure_queue:
                    # Get the next item
                    event, event_type = self.backpressure_queue.pop(0)
                    
                    # Process the item
                    self._emit_market_data_event(event, event_type)
                    
                    # Small delay to prevent CPU hogging
                    time.sleep(0.001)
                else:
                    # No items in queue, wait a bit
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in backpressure worker: {str(e)}")
                time.sleep(0.1)
                
        logger.info("Backpressure worker stopped")

    def _emit_market_data_event(self, event: SystemEvent, event_type: StreamEvent) -> None:
        """Emit market data event to the portfolio system.

        Args:
            event: The event object.
            event_type: The type of event.
        """
        data = event.data
        symbol = data.get("symbol")
        timeframe_str = data.get("timeframe")
        market_data = data.get("data")
        
        # Determine portfolio event type based on stream event type
        if event_type == StreamEvent.TICK:
            portfolio_event_type = "market_data.tick"
        elif event_type == StreamEvent.TRADE:
            portfolio_event_type = "market_data.trade"
        elif event_type == StreamEvent.QUOTE:
            portfolio_event_type = "market_data.quote"
        elif event_type == StreamEvent.BAR:
            portfolio_event_type = "market_data.bar"
        elif event_type == StreamEvent.ORDERBOOK:
            portfolio_event_type = "market_data.orderbook"
        else:
            portfolio_event_type = "market_data.custom"
            
        # Emit portfolio system event
        self._emit_portfolio_event(portfolio_event_type, {
            "symbol": symbol,
            "timeframe": timeframe_str,
            "data": market_data,
            "timestamp": datetime.now().isoformat(),
            "source": self.stream.source_type.value
        })

    def _emit_portfolio_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to the portfolio system.

        Args:
            event_type: The event type.
            data: The event data.
        """
        system_event = SystemEvent(
            event_type=event_type,
            data=data,
            source="StreamingMarketDataConnector"
        )
        
        self.event_system.emit(system_event)