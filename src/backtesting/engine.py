"""Event-driven backtesting engine implementation.

This module provides the core event-driven backtesting engine for simulating
trading strategies with historical data.
"""

import datetime
import logging
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.infrastructure.event.event_system import Event as InfraEvent
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the backtesting system."""
    MARKET_DATA = auto()  # New market data (tick, bar)
    SIGNAL = auto()       # Strategy signal
    ORDER = auto()        # Order to be executed
    FILL = auto()         # Order fill confirmation
    CUSTOM = auto()       # Custom event type


@dataclass
class Event:
    """Base event class for the backtesting engine.
    
    All events in the backtesting system inherit from this class.
    """
    type: EventType
    timestamp: datetime.datetime
    data: Dict[str, Any]
    event_id: str = None
    
    def __post_init__(self):
        """Initialize event_id if not provided."""
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())


class BacktestEngine:
    """Event-driven backtesting engine.
    
    This class implements an event-driven backtesting engine that simulates
    trading strategies with historical data.
    """
    
    def __init__(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        initial_capital: float = 100000.0,
        data_frequency: str = "1d",
        commission_model: Optional[Any] = None,
        slippage_model: Optional[Any] = None,
        market_impact_model: Optional[Any] = None,
    ):
        """Initialize the backtesting engine.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            data_frequency: Frequency of the data (e.g., "1d", "1h", "1m")
            commission_model: Commission model to use
            slippage_model: Slippage model to use
            market_impact_model: Market impact model to use
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_frequency = data_frequency
        
        # Set transaction cost models
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        
        # Initialize event queue and handlers
        self.events: List[Event] = []
        self.event_handlers: Dict[EventType, List[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }
        
        # Initialize data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_datetime: Optional[datetime.datetime] = None
        
        # Initialize portfolio state
        self.portfolio = {
            "cash": initial_capital,
            "positions": {},
            "equity": initial_capital,
            "history": []
        }
        
        # Performance tracking
        self.performance_metrics = {
            "returns": [],
            "equity_curve": [],
            "drawdowns": [],
            "trades": []
        }
        
        # Execution tracking
        self.orders = []
        self.fills = []
        
        # Strategy components
        self.strategies = []
        
        # Simulation state
        self.is_running = False
        self.current_step = 0
        self.total_steps = 0
        
        logger.info(f"Initialized backtesting engine from {start_date} to {end_date}")
    
    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add market data for a symbol.
        
        Args:
            symbol: The market symbol (e.g., "AAPL")
            data: DataFrame with market data (must have datetime index)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        # Filter data to backtest date range
        filtered_data = data[
            (data.index >= self.start_date) & 
            (data.index <= self.end_date)
        ].copy()
        
        if filtered_data.empty:
            logger.warning(f"No data for {symbol} in the specified date range")
            return
        
        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in filtered_data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            # Add missing columns with NaN values
            for col in missing_columns:
                filtered_data[col] = np.nan
        
        self.data[symbol] = filtered_data
        self.total_steps = max(self.total_steps, len(filtered_data))
        
        logger.info(f"Added data for {symbol} with {len(filtered_data)} bars")
    
    def register_strategy(self, strategy: Any) -> None:
        """Register a trading strategy.
        
        Args:
            strategy: Strategy object with on_data method
        """
        if not hasattr(strategy, "on_data"):
            raise ValueError("Strategy must have an on_data method")
        
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.__class__.__name__}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.name} events")
    
    def dispatch_event(self, event: Event) -> None:
        """Dispatch an event to all registered handlers.
        
        Args:
            event: Event to dispatch
        """
        for handler in self.event_handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {str(e)}")
    
    def add_event(self, event: Event) -> None:
        """Add an event to the event queue.
        
        Args:
            event: Event to add
        """
        self.events.append(event)
    
    def process_events(self) -> None:
        """Process all events in the event queue."""
        # Sort events by timestamp
        self.events.sort(key=lambda e: e.timestamp)
        
        # Process events until queue is empty
        while self.events:
            event = self.events.pop(0)
            self.dispatch_event(event)
    
    def update_portfolio(self, timestamp: datetime.datetime) -> None:
        """Update portfolio state with current market prices.
        
        Args:
            timestamp: Current timestamp
        """
        equity = self.portfolio["cash"]
        
        # Update position values
        for symbol, position in self.portfolio["positions"].items():
            if symbol in self.data and not self.data[symbol].empty:
                # Get latest price
                latest_price = self._get_price_at_time(symbol, timestamp)
                if latest_price is not None:
                    position_value = position["quantity"] * latest_price
                    position["current_price"] = latest_price
                    position["market_value"] = position_value
                    equity += position_value
        
        # Update portfolio equity
        self.portfolio["equity"] = equity
        
        # Record portfolio state
        self.portfolio["history"].append({
            "timestamp": timestamp,
            "cash": self.portfolio["cash"],
            "equity": equity
        })
        
        # Update performance metrics
        self.performance_metrics["equity_curve"].append((timestamp, equity))
        
        # Calculate returns if we have at least two equity points
        if len(self.performance_metrics["equity_curve"]) >= 2:
            prev_equity = self.performance_metrics["equity_curve"][-2][1]
            if prev_equity > 0:
                returns = (equity / prev_equity) - 1
                self.performance_metrics["returns"].append((timestamp, returns))
    
    def _get_price_at_time(self, symbol: str, timestamp: datetime.datetime) -> Optional[float]:
        """Get the price of a symbol at a specific time.
        
        Args:
            symbol: Market symbol
            timestamp: Timestamp to get price at
            
        Returns:
            Price at the specified time or None if not available
        """
        if symbol not in self.data:
            return None
        
        df = self.data[symbol]
        
        # Find the closest timestamp that's not after the requested timestamp
        idx = df.index.asof(timestamp)
        if idx is not None:
            return df.loc[idx, "close"]
        
        return None
    
    def execute_order(self, order_event: Event) -> None:
        """Execute a trade order.
        
        Args:
            order_event: Order event to execute
        """
        order_data = order_event.data
        symbol = order_data.get("symbol")
        order_type = order_data.get("order_type")
        quantity = order_data.get("quantity", 0)
        price = order_data.get("price")
        
        if not symbol or quantity == 0:
            logger.warning("Invalid order: missing symbol or quantity")
            return
        
        # Get current price if not specified
        if price is None:
            price = self._get_price_at_time(symbol, order_event.timestamp)
            if price is None:
                logger.warning(f"Cannot execute order: no price data for {symbol}")
                return
        
        # Apply slippage if model exists
        if self.slippage_model:
            price = self.slippage_model.apply_slippage(price, quantity, order_type)
        
        # Apply market impact if model exists
        if self.market_impact_model:
            price = self.market_impact_model.apply_impact(price, quantity, symbol)
        
        # Calculate commission if model exists
        commission = 0.0
        if self.commission_model:
            commission = self.commission_model.calculate_commission(price, quantity, symbol)
        
        # Calculate total cost
        cost = price * abs(quantity) + commission
        
        # Check if we have enough cash for a buy order
        if quantity > 0 and cost > self.portfolio["cash"]:
            logger.warning(f"Insufficient funds to execute buy order for {symbol}")
            return
        
        # Update portfolio
        if quantity > 0:  # Buy
            self.portfolio["cash"] -= cost
        else:  # Sell
            self.portfolio["cash"] += (price * abs(quantity)) - commission
        
        # Update position
        if symbol not in self.portfolio["positions"]:
            self.portfolio["positions"][symbol] = {
                "quantity": 0,
                "cost_basis": 0,
                "current_price": price,
                "market_value": 0
            }
        
        position = self.portfolio["positions"][symbol]
        old_quantity = position["quantity"]
        new_quantity = old_quantity + quantity
        
        # Update cost basis for buys
        if quantity > 0:
            old_cost = position["cost_basis"] * old_quantity
            new_cost = price * quantity
            position["cost_basis"] = (old_cost + new_cost) / new_quantity if new_quantity > 0 else 0
        
        position["quantity"] = new_quantity
        position["current_price"] = price
        position["market_value"] = price * new_quantity
        
        # Remove position if quantity is zero
        if position["quantity"] == 0:
            del self.portfolio["positions"][symbol]
        
        # Record the fill
        fill_event = Event(
            type=EventType.FILL,
            timestamp=order_event.timestamp,
            data={
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "order_id": order_data.get("order_id"),
            }
        )
        
        # Add to fills history
        self.fills.append(fill_event.data)
        
        # Dispatch fill event
        self.dispatch_event(fill_event)
        
        # Record trade for performance analysis
        self.performance_metrics["trades"].append({
            "timestamp": order_event.timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "cost": cost,
            "type": "buy" if quantity > 0 else "sell"
        })
        
        logger.info(
            f"Executed order: {symbol}, {quantity} shares at {price:.2f}, "
            f"commission: {commission:.2f}"
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest.
        
        Returns:
            Dict with backtest results
        """
        if not self.data:
            raise ValueError("No data added to backtest engine")
        
        if not self.strategies:
            raise ValueError("No strategies registered")
        
        logger.info("Starting backtest")
        self.is_running = True
        
        # Register default handlers
        self.register_event_handler(EventType.ORDER, self.execute_order)
        
        # Create a unified timeline of all data points
        all_timestamps = set()
        for symbol, df in self.data.items():
            all_timestamps.update(df.index.to_pydatetime())
        
        timeline = sorted(all_timestamps)
        self.total_steps = len(timeline)
        
        # Main event loop
        for timestamp in timeline:
            if not self.is_running:
                break
            
            self.current_datetime = timestamp
            self.current_step += 1
            
            # Create market data events for this timestamp
            for symbol, df in self.data.items():
                if timestamp in df.index:
                    market_data = df.loc[timestamp].to_dict()
                    market_data["symbol"] = symbol
                    
                    market_event = Event(
                        type=EventType.MARKET_DATA,
                        timestamp=timestamp,
                        data=market_data
                    )
                    
                    # Add to event queue
                    self.add_event(market_event)
                    
                    # Notify strategies
                    for strategy in self.strategies:
                        try:
                            strategy.on_data(market_event, self)
                        except Exception as e:
                            logger.error(f"Error in strategy: {str(e)}")
            
            # Process all events
            self.process_events()
            
            # Update portfolio state
            self.update_portfolio(timestamp)
            
            # Log progress periodically
            if self.current_step % 100 == 0 or self.current_step == self.total_steps:
                progress = (self.current_step / self.total_steps) * 100
                logger.info(f"Backtest progress: {progress:.1f}% ({self.current_step}/{self.total_steps})")
        
        self.is_running = False
        logger.info("Backtest completed")
        
        # Prepare results
        results = self.get_results()
        return results
    
    def stop(self) -> None:
        """Stop the backtest."""
        self.is_running = False
        logger.info("Backtest stopped")
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results.
        
        Returns:
            Dict with backtest results
        """
        # Convert equity curve to DataFrame
        equity_curve = pd.DataFrame(
            self.performance_metrics["equity_curve"],
            columns=["timestamp", "equity"]
        ).set_index("timestamp")
        
        # Convert returns to DataFrame
        returns = pd.DataFrame(
            self.performance_metrics["returns"],
            columns=["timestamp", "returns"]
        ).set_index("timestamp")
        
        # Convert trades to DataFrame
        trades = pd.DataFrame(self.performance_metrics["trades"])
        
        # Convert portfolio history to DataFrame
        portfolio_history = pd.DataFrame(self.portfolio["history"])
        if not portfolio_history.empty:
            portfolio_history = portfolio_history.set_index("timestamp")
        
        # Calculate basic performance metrics
        total_return = ((equity_curve["equity"].iloc[-1] / self.initial_capital) - 1) * 100 if not equity_curve.empty else 0
        
        # Calculate drawdowns
        if not equity_curve.empty:
            equity_curve["peak"] = equity_curve["equity"].cummax()
            equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["peak"]) - 1
            max_drawdown = equity_curve["drawdown"].min() * 100
        else:
            max_drawdown = 0
        
        # Calculate Sharpe ratio (annualized)
        if not returns.empty and len(returns) > 1:
            # Assuming daily returns
            sharpe_ratio = returns["returns"].mean() / returns["returns"].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Prepare results dictionary
        results = {
            "initial_capital": self.initial_capital,
            "final_equity": equity_curve["equity"].iloc[-1] if not equity_curve.empty else self.initial_capital,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "equity_curve": equity_curve,
            "returns": returns,
            "trades": trades,
            "portfolio_history": portfolio_history,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "duration": (self.end_date - self.start_date).days,
        }
        
        return results