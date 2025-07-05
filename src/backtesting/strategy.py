"""Strategy module for backtesting framework.

This module provides base classes and interfaces for implementing trading
strategies in the backtesting framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from src.backtesting.engine import Event, EventType
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types for trading strategies."""
    MARKET = "market"  # Market order, executed at current market price
    LIMIT = "limit"    # Limit order, executed at specified price or better
    STOP = "stop"      # Stop order, becomes market order when price reaches trigger
    STOP_LIMIT = "stop_limit"  # Stop-limit order, becomes limit order when price reaches trigger


class OrderSide(Enum):
    """Order sides for trading strategies."""
    BUY = "buy"        # Buy order
    SELL = "sell"      # Sell order


class OrderStatus(Enum):
    """Order statuses for tracking order lifecycle."""
    CREATED = "created"        # Order created but not submitted
    SUBMITTED = "submitted"    # Order submitted to broker
    PARTIAL = "partial"        # Order partially filled
    FILLED = "filled"          # Order completely filled
    CANCELED = "canceled"      # Order canceled
    REJECTED = "rejected"      # Order rejected
    EXPIRED = "expired"        # Order expired


class Order:
    """Order class for trading strategies.
    
    This class represents an order in the backtesting framework.
    """
    
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",  # GTC = Good Till Canceled
        order_id: Optional[str] = None,
    ):
        """Initialize an order.
        
        Args:
            symbol: Symbol to trade
            order_type: Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
            side: Side of order (BUY, SELL)
            quantity: Quantity to trade
            price: Price for limit orders (default: None)
            stop_price: Stop price for stop orders (default: None)
            time_in_force: Time in force (default: "GTC")
            order_id: Order ID (default: None, auto-generated)
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.order_id = order_id or self._generate_order_id()
        
        # Order status tracking
        self.status = OrderStatus.CREATED
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.filled_time = None
        self.transaction_costs = 0.0
        
        # Validate order
        self._validate()
    
    def _validate(self) -> None:
        """Validate the order parameters.
        
        Raises:
            ValueError: If order parameters are invalid
        """
        # Check required price fields based on order type
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop price")
        
        if self.order_type == OrderType.STOP_LIMIT and (self.price is None or self.stop_price is None):
            raise ValueError("Stop-limit orders require both a price and a stop price")
        
        # Check quantity
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID.
        
        Returns:
            Unique order ID
        """
        import uuid
        return str(uuid.uuid4())
    
    def update_status(self, status: OrderStatus) -> None:
        """Update the order status.
        
        Args:
            status: New order status
        """
        self.status = status
    
    def fill(self, quantity: float, price: float, time: pd.Timestamp, transaction_costs: float = 0.0) -> None:
        """Fill the order (partially or completely).
        
        Args:
            quantity: Quantity filled
            price: Fill price
            time: Fill time
            transaction_costs: Transaction costs (default: 0.0)
        """
        # Update filled quantity
        self.filled_quantity += quantity
        
        # Update average filled price
        if self.filled_price == 0.0:
            self.filled_price = price
        else:
            # Calculate weighted average price
            total_quantity = self.filled_quantity
            previous_quantity = total_quantity - quantity
            self.filled_price = (previous_quantity * self.filled_price + quantity * price) / total_quantity
        
        # Update filled time
        self.filled_time = time
        
        # Update transaction costs
        self.transaction_costs += transaction_costs
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.status = OrderStatus.CANCELED
    
    def reject(self) -> None:
        """Reject the order."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.status = OrderStatus.REJECTED
    
    def expire(self) -> None:
        """Expire the order."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.status = OrderStatus.EXPIRED
    
    def is_active(self) -> bool:
        """Check if the order is active.
        
        Returns:
            True if the order is active, False otherwise
        """
        return self.status in [OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    def is_filled(self) -> bool:
        """Check if the order is filled.
        
        Returns:
            True if the order is filled, False otherwise
        """
        return self.status == OrderStatus.FILLED
    
    def is_canceled(self) -> bool:
        """Check if the order is canceled.
        
        Returns:
            True if the order is canceled, False otherwise
        """
        return self.status == OrderStatus.CANCELED
    
    def is_rejected(self) -> bool:
        """Check if the order is rejected.
        
        Returns:
            True if the order is rejected, False otherwise
        """
        return self.status == OrderStatus.REJECTED
    
    def is_expired(self) -> bool:
        """Check if the order is expired.
        
        Returns:
            True if the order is expired, False otherwise
        """
        return self.status == OrderStatus.EXPIRED
    
    def __str__(self) -> str:
        """String representation of the order.
        
        Returns:
            String representation
        """
        return f"Order(id={self.order_id}, symbol={self.symbol}, type={self.order_type.value}, side={self.side.value}, quantity={self.quantity}, price={self.price}, status={self.status.value})"


class Position:
    """Position class for tracking holdings.
    
    This class represents a position in the backtesting framework.
    """
    
    def __init__(self, symbol: str, quantity: float = 0.0, average_price: float = 0.0):
        """Initialize a position.
        
        Args:
            symbol: Symbol of the position
            quantity: Quantity held (default: 0.0)
            average_price: Average entry price (default: 0.0)
        """
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.cost_basis = quantity * average_price if quantity != 0 else 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_price = 0.0
        self.last_update_time = None
    
    def update(self, price: float, time: pd.Timestamp) -> None:
        """Update the position with current market price.
        
        Args:
            price: Current market price
            time: Current time
        """
        self.current_price = price
        self.last_update_time = time
        
        # Calculate unrealized P&L
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def add(self, quantity: float, price: float, time: pd.Timestamp) -> float:
        """Add to the position.
        
        Args:
            quantity: Quantity to add (positive for buy, negative for sell)
            price: Price of the transaction
            time: Time of the transaction
            
        Returns:
            Realized P&L from the transaction
        """
        realized_pnl = 0.0
        
        # Handle selling (reducing position)
        if quantity < 0 and self.quantity > 0:
            # Selling long position
            sell_quantity = min(abs(quantity), self.quantity)
            realized_pnl = (price - self.average_price) * sell_quantity
            self.realized_pnl += realized_pnl
            
            # Update position
            new_quantity = self.quantity - sell_quantity
            if new_quantity > 0:
                # Still have some position left
                self.quantity = new_quantity
            else:
                # Position closed
                self.quantity = 0
                self.average_price = 0.0
                self.cost_basis = 0.0
                
                # Handle remaining quantity (short position)
                remaining_quantity = abs(quantity) - sell_quantity
                if remaining_quantity > 0:
                    self.quantity = -remaining_quantity
                    self.average_price = price
                    self.cost_basis = self.quantity * self.average_price
        
        # Handle buying (reducing short position)
        elif quantity > 0 and self.quantity < 0:
            # Buying to cover short position
            buy_quantity = min(quantity, abs(self.quantity))
            realized_pnl = (self.average_price - price) * buy_quantity
            self.realized_pnl += realized_pnl
            
            # Update position
            new_quantity = self.quantity + buy_quantity
            if new_quantity < 0:
                # Still have some short position left
                self.quantity = new_quantity
            else:
                # Position closed
                self.quantity = 0
                self.average_price = 0.0
                self.cost_basis = 0.0
                
                # Handle remaining quantity (long position)
                remaining_quantity = quantity - buy_quantity
                if remaining_quantity > 0:
                    self.quantity = remaining_quantity
                    self.average_price = price
                    self.cost_basis = self.quantity * self.average_price
        
        # Handle adding to existing position
        else:
            # Calculate new average price and cost basis
            old_cost = self.cost_basis
            additional_cost = quantity * price
            new_quantity = self.quantity + quantity
            
            if new_quantity != 0:
                self.average_price = (old_cost + additional_cost) / new_quantity
            else:
                self.average_price = 0.0
            
            self.quantity = new_quantity
            self.cost_basis = self.quantity * self.average_price
        
        # Update with current price
        self.update(price, time)
        
        return realized_pnl
    
    def close(self, price: float, time: pd.Timestamp) -> float:
        """Close the position.
        
        Args:
            price: Price to close at
            time: Time of closing
            
        Returns:
            Realized P&L from closing the position
        """
        if self.quantity == 0:
            return 0.0
        
        # Calculate realized P&L
        if self.quantity > 0:
            realized_pnl = (price - self.average_price) * self.quantity
        else:
            realized_pnl = (self.average_price - price) * abs(self.quantity)
        
        # Update position
        self.realized_pnl += realized_pnl
        self.quantity = 0.0
        self.average_price = 0.0
        self.cost_basis = 0.0
        self.unrealized_pnl = 0.0
        
        # Update with current price
        self.update(price, time)
        
        return realized_pnl
    
    def market_value(self) -> float:
        """Calculate the market value of the position.
        
        Returns:
            Market value
        """
        return self.quantity * self.current_price
    
    def total_pnl(self) -> float:
        """Calculate the total P&L (realized + unrealized).
        
        Returns:
            Total P&L
        """
        return self.realized_pnl + self.unrealized_pnl
    
    def is_long(self) -> bool:
        """Check if the position is long.
        
        Returns:
            True if the position is long, False otherwise
        """
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if the position is short.
        
        Returns:
            True if the position is short, False otherwise
        """
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if the position is flat (no position).
        
        Returns:
            True if the position is flat, False otherwise
        """
        return self.quantity == 0
    
    def __str__(self) -> str:
        """String representation of the position.
        
        Returns:
            String representation
        """
        return f"Position(symbol={self.symbol}, quantity={self.quantity}, avg_price={self.average_price:.2f}, unrealized_pnl={self.unrealized_pnl:.2f}, realized_pnl={self.realized_pnl:.2f})"


class Portfolio:
    """Portfolio class for tracking positions and performance.
    
    This class represents a portfolio in the backtesting framework.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize a portfolio.
        
        Args:
            initial_capital: Initial capital (default: 100000.0)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> Order
        
        # Performance tracking
        self.equity = initial_capital
        self.equity_curve = pd.Series()
        self.returns = pd.Series()
        self.transactions = []
        self.current_time = None
    
    def update(self, prices: Dict[str, float], time: pd.Timestamp) -> None:
        """Update the portfolio with current market prices.
        
        Args:
            prices: Dictionary of current prices (symbol -> price)
            time: Current time
        """
        self.current_time = time
        
        # Update positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update(prices[symbol], time)
        
        # Calculate equity
        self.equity = self.cash + sum(position.market_value() for position in self.positions.values())
        
        # Update equity curve and returns
        if len(self.equity_curve) == 0:
            self.equity_curve = pd.Series([self.equity], index=[time])
            self.returns = pd.Series([0.0], index=[time])
        else:
            self.equity_curve = pd.concat([self.equity_curve, pd.Series([self.equity], index=[time])])
            self.returns = pd.concat([self.returns, pd.Series([self.equity / self.equity_curve.iloc[-2] - 1], index=[time])])
    
    def get_position(self, symbol: str) -> Position:
        """Get a position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position for the symbol
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        return self.positions[symbol]
    
    def create_order(self, order: Order) -> str:
        """Create an order.
        
        Args:
            order: Order to create
            
        Returns:
            Order ID
        """
        self.orders[order.order_id] = order
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if the order was canceled, False otherwise
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active():
                order.cancel()
                return True
        
        return False
    
    def execute_order(self, order_id: str, price: float, quantity: float, time: pd.Timestamp, transaction_costs: float = 0.0) -> bool:
        """Execute an order (fill it partially or completely).
        
        Args:
            order_id: ID of the order to execute
            price: Execution price
            quantity: Quantity to execute
            time: Execution time
            transaction_costs: Transaction costs (default: 0.0)
            
        Returns:
            True if the order was executed, False otherwise
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if not order.is_active():
            return False
        
        # Calculate actual quantity to execute (limited by order quantity)
        quantity = min(quantity, order.quantity - order.filled_quantity)
        if quantity <= 0:
            return False
        
        # Calculate cash impact
        cash_impact = price * quantity
        if order.side == OrderSide.BUY:
            cash_impact = -cash_impact
        
        # Check if we have enough cash for a buy order
        if order.side == OrderSide.BUY and self.cash < cash_impact + transaction_costs:
            # Not enough cash, reject the order
            order.reject()
            return False
        
        # Update cash and position
        self.cash += cash_impact - transaction_costs
        
        # Get position
        position = self.get_position(order.symbol)
        
        # Update position
        position_quantity = quantity
        if order.side == OrderSide.SELL:
            position_quantity = -quantity
        
        realized_pnl = position.add(position_quantity, price, time)
        
        # Fill the order
        order.fill(quantity, price, time, transaction_costs)
        
        # Record transaction
        self.transactions.append({
            "time": time,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "transaction_costs": transaction_costs,
            "realized_pnl": realized_pnl,
        })
        
        return True
    
    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve.
        
        Returns:
            Equity curve as a pandas Series
        """
        return self.equity_curve
    
    def get_returns(self) -> pd.Series:
        """Get the returns.
        
        Returns:
            Returns as a pandas Series
        """
        return self.returns
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """Get the transactions.
        
        Returns:
            List of transactions
        """
        return self.transactions
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions.
        
        Returns:
            Dictionary of positions (symbol -> Position)
        """
        return self.positions
    
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders.
        
        Returns:
            Dictionary of active orders (order_id -> Order)
        """
        return {order_id: order for order_id, order in self.orders.items() if order.is_active()}
    
    def get_total_value(self) -> float:
        """Get the total portfolio value.
        
        Returns:
            Total portfolio value
        """
        return self.equity
    
    def get_cash(self) -> float:
        """Get the cash balance.
        
        Returns:
            Cash balance
        """
        return self.cash
    
    def get_total_realized_pnl(self) -> float:
        """Get the total realized P&L.
        
        Returns:
            Total realized P&L
        """
        return sum(position.realized_pnl for position in self.positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """Get the total unrealized P&L.
        
        Returns:
            Total unrealized P&L
        """
        return sum(position.unrealized_pnl for position in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Get the total P&L (realized + unrealized).
        
        Returns:
            Total P&L
        """
        return self.get_total_realized_pnl() + self.get_total_unrealized_pnl()
    
    def __str__(self) -> str:
        """String representation of the portfolio.
        
        Returns:
            String representation
        """
        return f"Portfolio(cash={self.cash:.2f}, equity={self.equity:.2f}, positions={len(self.positions)}, active_orders={len(self.get_active_orders())})"


class Strategy(ABC):
    """Abstract base class for trading strategies.
    
    This class defines the interface for trading strategies in the backtesting
    framework. Concrete strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = None):
        """Initialize the strategy.
        
        Args:
            name: Strategy name (default: None, uses class name)
        """
        self.name = name or self.__class__.__name__
        self.portfolio = None
        self.data = {}
        self.current_time = None
        self.is_initialized = False
    
    def initialize(self, portfolio: Portfolio) -> None:
        """Initialize the strategy with a portfolio.
        
        Args:
            portfolio: Portfolio to use
        """
        self.portfolio = portfolio
        self.is_initialized = True
        self.on_start()
    
    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add data for a symbol.
        
        Args:
            symbol: Symbol to add data for
            data: Data for the symbol
        """
        self.data[symbol] = data
    
    def on_start(self) -> None:
        """Called when the strategy starts.
        
        This method can be overridden by concrete strategies to perform
        initialization tasks.
        """
        pass
    
    def on_end(self) -> None:
        """Called when the strategy ends.
        
        This method can be overridden by concrete strategies to perform
        cleanup tasks.
        """
        pass
    
    @abstractmethod
    def on_bar(self, event: Event) -> None:
        """Called when a new bar is received.
        
        This method must be implemented by concrete strategies to define
        the trading logic.
        
        Args:
            event: Bar event
        """
        pass
    
    def on_order_filled(self, event: Event) -> None:
        """Called when an order is filled.
        
        This method can be overridden by concrete strategies to handle
        order fills.
        
        Args:
            event: Order filled event
        """
        pass
    
    def on_order_canceled(self, event: Event) -> None:
        """Called when an order is canceled.
        
        This method can be overridden by concrete strategies to handle
        order cancellations.
        
        Args:
            event: Order canceled event
        """
        pass
    
    def on_order_rejected(self, event: Event) -> None:
        """Called when an order is rejected.
        
        This method can be overridden by concrete strategies to handle
        order rejections.
        
        Args:
            event: Order rejected event
        """
        pass
    
    def handle_event(self, event: Event) -> None:
        """Handle an event.
        
        This method dispatches events to the appropriate handler methods.
        
        Args:
            event: Event to handle
        """
        self.current_time = event.time
        
        if event.type == EventType.BAR:
            self.on_bar(event)
        elif event.type == EventType.ORDER_FILLED:
            self.on_order_filled(event)
        elif event.type == EventType.ORDER_CANCELED:
            self.on_order_canceled(event)
        elif event.type == EventType.ORDER_REJECTED:
            self.on_order_rejected(event)
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float) -> str:
        """Create a market order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY, SELL)
            quantity: Quantity to trade
            
        Returns:
            Order ID
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
        )
        
        return self.portfolio.create_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float) -> str:
        """Create a limit order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY, SELL)
            quantity: Quantity to trade
            price: Limit price
            
        Returns:
            Order ID
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            price=price,
        )
        
        return self.portfolio.create_order(order)
    
    def create_stop_order(self, symbol: str, side: OrderSide, quantity: float, stop_price: float) -> str:
        """Create a stop order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY, SELL)
            quantity: Quantity to trade
            stop_price: Stop price
            
        Returns:
            Order ID
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.STOP,
            side=side,
            quantity=quantity,
            stop_price=stop_price,
        )
        
        return self.portfolio.create_order(order)
    
    def create_stop_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float, stop_price: float) -> str:
        """Create a stop-limit order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY, SELL)
            quantity: Quantity to trade
            price: Limit price
            stop_price: Stop price
            
        Returns:
            Order ID
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.STOP_LIMIT,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
        
        return self.portfolio.create_order(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if the order was canceled, False otherwise
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        return self.portfolio.cancel_order(order_id)
    
    def get_position(self, symbol: str) -> Position:
        """Get a position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position for the symbol
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        return self.portfolio.get_position(symbol)
    
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders.
        
        Returns:
            Dictionary of active orders (order_id -> Order)
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        return self.portfolio.get_active_orders()
    
    def get_cash(self) -> float:
        """Get the cash balance.
        
        Returns:
            Cash balance
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        return self.portfolio.get_cash()
    
    def get_equity(self) -> float:
        """Get the total portfolio value.
        
        Returns:
            Total portfolio value
        """
        if not self.is_initialized:
            raise ValueError("Strategy not initialized")
        
        return self.portfolio.get_total_value()
    
    def get_current_time(self) -> pd.Timestamp:
        """Get the current time.
        
        Returns:
            Current time
        """
        return self.current_time


class MovingAverageStrategy(Strategy):
    """Simple moving average crossover strategy.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    def __init__(self, symbol: str, fast_period: int = 10, slow_period: int = 30, position_size: float = 1.0):
        """Initialize the strategy.
        
        Args:
            symbol: Symbol to trade
            fast_period: Fast moving average period (default: 10)
            slow_period: Slow moving average period (default: 30)
            position_size: Position size as a fraction of portfolio value (default: 1.0)
        """
        super().__init__()
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        
        # Initialize indicators
        self.fast_ma = None
        self.slow_ma = None
        
        # Initialize state
        self.current_position = 0
        self.last_signal = None
    
    def on_start(self) -> None:
        """Called when the strategy starts."""
        # Calculate moving averages
        if self.symbol in self.data:
            data = self.data[self.symbol]
            if "close" in data.columns:
                self.fast_ma = data["close"].rolling(window=self.fast_period).mean()
                self.slow_ma = data["close"].rolling(window=self.slow_period).mean()
    
    def on_bar(self, event: Event) -> None:
        """Called when a new bar is received.
        
        Args:
            event: Bar event
        """
        # Check if we have data for the symbol
        if self.symbol not in self.data or event.symbol != self.symbol:
            return
        
        # Get current data
        data = self.data[self.symbol]
        current_index = data.index.get_loc(event.time)
        
        # Check if we have enough data
        if current_index < self.slow_period:
            return
        
        # Get current price and moving averages
        current_price = data.loc[event.time, "close"]
        current_fast_ma = self.fast_ma.iloc[current_index]
        current_slow_ma = self.slow_ma.iloc[current_index]
        
        # Get previous moving averages
        if current_index > 0:
            previous_fast_ma = self.fast_ma.iloc[current_index - 1]
            previous_slow_ma = self.slow_ma.iloc[current_index - 1]
        else:
            previous_fast_ma = current_fast_ma
            previous_slow_ma = current_slow_ma
        
        # Check for crossover
        if previous_fast_ma <= previous_slow_ma and current_fast_ma > current_slow_ma:
            # Buy signal
            self._enter_long(current_price)
        elif previous_fast_ma >= previous_slow_ma and current_fast_ma < current_slow_ma:
            # Sell signal
            self._enter_short(current_price)
    
    def _enter_long(self, price: float) -> None:
        """Enter a long position.
        
        Args:
            price: Current price
        """
        # Close any existing short position
        position = self.get_position(self.symbol)
        if position.is_short():
            # Close short position
            self.create_market_order(self.symbol, OrderSide.BUY, abs(position.quantity))
        
        # Calculate position size
        equity = self.get_equity()
        position_value = equity * self.position_size
        quantity = position_value / price
        
        # Enter long position
        if quantity > 0 and not position.is_long():
            self.create_market_order(self.symbol, OrderSide.BUY, quantity)
            self.current_position = 1
            self.last_signal = "buy"
    
    def _enter_short(self, price: float) -> None:
        """Enter a short position.
        
        Args:
            price: Current price
        """
        # Close any existing long position
        position = self.get_position(self.symbol)
        if position.is_long():
            # Close long position
            self.create_market_order(self.symbol, OrderSide.SELL, position.quantity)
        
        # Calculate position size
        equity = self.get_equity()
        position_value = equity * self.position_size
        quantity = position_value / price
        
        # Enter short position
        if quantity > 0 and not position.is_short():
            self.create_market_order(self.symbol, OrderSide.SELL, quantity)
            self.current_position = -1
            self.last_signal = "sell"


class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) strategy.
    
    This strategy generates buy signals when the RSI falls below the oversold level
    and then rises above it, and sell signals when the RSI rises above the overbought
    level and then falls below it.
    """
    
    def __init__(
        self,
        symbol: str,
        rsi_period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        position_size: float = 1.0,
    ):
        """Initialize the strategy.
        
        Args:
            symbol: Symbol to trade
            rsi_period: RSI period (default: 14)
            overbought: Overbought level (default: 70.0)
            oversold: Oversold level (default: 30.0)
            position_size: Position size as a fraction of portfolio value (default: 1.0)
        """
        super().__init__()
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.position_size = position_size
        
        # Initialize indicators
        self.rsi = None
        
        # Initialize state
        self.current_position = 0
        self.last_signal = None
        self.was_overbought = False
        self.was_oversold = False
    
    def on_start(self) -> None:
        """Called when the strategy starts."""
        # Calculate RSI
        if self.symbol in self.data:
            data = self.data[self.symbol]
            if "close" in data.columns:
                # Calculate price changes
                delta = data["close"].diff()
                
                # Calculate gains and losses
                gains = delta.copy()
                losses = delta.copy()
                gains[gains < 0] = 0
                losses[losses > 0] = 0
                losses = abs(losses)
                
                # Calculate average gains and losses
                avg_gain = gains.rolling(window=self.rsi_period).mean()
                avg_loss = losses.rolling(window=self.rsi_period).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                self.rsi = 100 - (100 / (1 + rs))
    
    def on_bar(self, event: Event) -> None:
        """Called when a new bar is received.
        
        Args:
            event: Bar event
        """
        # Check if we have data for the symbol
        if self.symbol not in self.data or event.symbol != self.symbol:
            return
        
        # Get current data
        data = self.data[self.symbol]
        current_index = data.index.get_loc(event.time)
        
        # Check if we have enough data
        if current_index < self.rsi_period:
            return
        
        # Get current price and RSI
        current_price = data.loc[event.time, "close"]
        current_rsi = self.rsi.iloc[current_index]
        
        # Check for overbought/oversold conditions
        if current_rsi > self.overbought:
            self.was_overbought = True
        elif current_rsi < self.oversold:
            self.was_oversold = True
        
        # Check for signals
        if self.was_oversold and current_rsi > self.oversold:
            # Buy signal
            self._enter_long(current_price)
            self.was_oversold = False
        elif self.was_overbought and current_rsi < self.overbought:
            # Sell signal
            self._enter_short(current_price)
            self.was_overbought = False
    
    def _enter_long(self, price: float) -> None:
        """Enter a long position.
        
        Args:
            price: Current price
        """
        # Close any existing short position
        position = self.get_position(self.symbol)
        if position.is_short():
            # Close short position
            self.create_market_order(self.symbol, OrderSide.BUY, abs(position.quantity))
        
        # Calculate position size
        equity = self.get_equity()
        position_value = equity * self.position_size
        quantity = position_value / price
        
        # Enter long position
        if quantity > 0 and not position.is_long():
            self.create_market_order(self.symbol, OrderSide.BUY, quantity)
            self.current_position = 1
            self.last_signal = "buy"
    
    def _enter_short(self, price: float) -> None:
        """Enter a short position.
        
        Args:
            price: Current price
        """
        # Close any existing long position
        position = self.get_position(self.symbol)
        if position.is_long():
            # Close long position
            self.create_market_order(self.symbol, OrderSide.SELL, position.quantity)
        
        # Calculate position size
        equity = self.get_equity()
        position_value = equity * self.position_size
        quantity = position_value / price
        
        # Enter short position
        if quantity > 0 and not position.is_short():
            self.create_market_order(self.symbol, OrderSide.SELL, quantity)
            self.current_position = -1
            self.last_signal = "sell"