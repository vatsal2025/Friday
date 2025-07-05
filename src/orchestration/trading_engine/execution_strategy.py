"""Execution Strategy Module for Trading Engine.

This module provides various execution strategies for optimizing order execution,
including TWAP, VWAP, and other advanced algorithms.
"""

import logging
import math
import time
import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger(__name__)


class ExecutionStrategyType(Enum):
    """Types of execution strategies."""
    MARKET = "market"  # Immediate execution at market price
    LIMIT = "limit"  # Limit order execution
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ARRIVAL_PRICE = "arrival_price"  # Execution relative to arrival price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # Implementation shortfall strategy
    PERCENTAGE_OF_VOLUME = "percentage_of_volume"  # Percentage of volume strategy (POV)
    ADAPTIVE = "adaptive"  # Adaptive strategy based on market conditions
    ICEBERG = "iceberg"  # Iceberg/reserve strategy
    CUSTOM = "custom"  # Custom execution strategy


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = "day"  # Valid for the day
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date


@dataclass
class MarketData:
    """Market data for execution strategies."""
    symbol: str  # Symbol/ticker
    current_price: float  # Current market price
    bid: float = 0.0  # Best bid price
    ask: float = 0.0  # Best ask price
    last_volume: float = 0.0  # Last trade volume
    daily_volume: float = 0.0  # Daily volume
    average_daily_volume: float = 0.0  # Average daily volume (historical)
    historical_volume_profile: Dict[str, float] = field(default_factory=dict)  # Historical volume profile by time
    volatility: float = 0.0  # Price volatility
    spread: float = 0.0  # Bid-ask spread
    liquidity: float = 1.0  # Liquidity factor
    is_stressed: bool = False  # Whether the market is in a stressed condition


@dataclass
class ExecutionParameters:
    """Parameters for execution strategies."""
    strategy_type: ExecutionStrategyType = ExecutionStrategyType.MARKET
    
    # General parameters
    start_time: Optional[datetime.datetime] = None  # Start time for execution
    end_time: Optional[datetime.datetime] = None  # End time for execution
    max_participation_rate: float = 0.3  # Maximum participation rate (0.0-1.0)
    urgency: float = 0.5  # Execution urgency (0.0-1.0, higher = more urgent)
    
    # TWAP parameters
    twap_num_slices: int = 10  # Number of slices for TWAP
    twap_slice_interval_seconds: int = 60  # Interval between slices in seconds
    
    # VWAP parameters
    vwap_use_historical_profile: bool = True  # Whether to use historical volume profile
    vwap_custom_profile: Dict[str, float] = field(default_factory=dict)  # Custom volume profile
    
    # POV parameters
    pov_target_rate: float = 0.1  # Target participation rate (0.0-1.0)
    pov_min_rate: float = 0.05  # Minimum participation rate
    pov_max_rate: float = 0.2  # Maximum participation rate
    
    # Iceberg parameters
    iceberg_display_size: float = 0.0  # Display size for iceberg orders
    iceberg_min_size: float = 0.0  # Minimum size for iceberg slices
    
    # Adaptive parameters
    adaptive_min_size: float = 0.0  # Minimum order size for adaptive strategy
    adaptive_max_size: float = 0.0  # Maximum order size for adaptive strategy
    adaptive_size_scale_factor: float = 1.0  # Scaling factor for order size
    
    # Limit order parameters
    limit_price_offset: float = 0.0  # Offset from reference price for limit orders
    limit_price_aggression: float = 0.5  # Aggression level for limit price (0.0-1.0)
    
    # Implementation shortfall parameters
    is_risk_aversion: float = 0.5  # Risk aversion parameter (0.0-1.0)
    is_alpha_decay: float = 0.1  # Alpha decay parameter
    
    # Custom parameters
    custom_strategy: Optional[Callable] = None  # Custom strategy function
    custom_params: Dict[str, Any] = field(default_factory=dict)  # Custom parameters


@dataclass
class ExecutionState:
    """State of an execution strategy."""
    symbol: str  # Symbol/ticker
    side: OrderSide  # Order side
    total_quantity: float  # Total quantity to execute
    executed_quantity: float = 0.0  # Quantity executed so far
    remaining_quantity: float = 0.0  # Quantity remaining to execute
    start_price: float = 0.0  # Price at start of execution
    last_execution_price: float = 0.0  # Price of last execution
    vwap_price: float = 0.0  # Volume-weighted average execution price
    start_time: Optional[datetime.datetime] = None  # Start time of execution
    end_time: Optional[datetime.datetime] = None  # End time of execution
    is_complete: bool = False  # Whether execution is complete
    is_cancelled: bool = False  # Whether execution was cancelled
    child_orders: List[Dict[str, Any]] = field(default_factory=list)  # Child orders created
    execution_logs: List[Dict[str, Any]] = field(default_factory=list)  # Execution logs


class ExecutionStrategy:
    """Base class for execution strategies."""
    def __init__(self, params: ExecutionParameters):
        """Initialize the execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        self.params = params
        self.state = None  # Will be initialized in start_execution
    
    def start_execution(self, symbol: str, side: OrderSide, quantity: float, 
                       current_price: float) -> ExecutionState:
        """Start execution of an order.
        
        Args:
            symbol: Symbol/ticker
            side: Order side (buy or sell)
            quantity: Total quantity to execute
            current_price: Current market price
            
        Returns:
            Initial execution state
        """
        # Initialize execution state
        self.state = ExecutionState(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            remaining_quantity=quantity,
            start_price=current_price,
            start_time=datetime.datetime.now(),
            end_time=None
        )
        
        # Log start of execution
        logger.info(f"Starting execution for {quantity} {symbol} {side.value} using {self.params.strategy_type.value} strategy")
        
        return self.state
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        raise NotImplementedError("Subclasses must implement process_market_update")
    
    def process_execution(self, order_id: str, executed_quantity: float, 
                         execution_price: float) -> None:
        """Process an execution report.
        
        Args:
            order_id: ID of the executed order
            executed_quantity: Quantity executed
            execution_price: Execution price
        """
        if self.state is None:
            logger.error("Execution state not initialized")
            return
        
        # Update execution state
        self.state.executed_quantity += executed_quantity
        self.state.remaining_quantity = self.state.total_quantity - self.state.executed_quantity
        self.state.last_execution_price = execution_price
        
        # Update VWAP price
        total_value = 0.0
        total_qty = 0.0
        
        # Add current execution
        self.state.execution_logs.append({
            "order_id": order_id,
            "time": datetime.datetime.now(),
            "quantity": executed_quantity,
            "price": execution_price
        })
        
        # Calculate VWAP from all executions
        for log in self.state.execution_logs:
            total_value += log["quantity"] * log["price"]
            total_qty += log["quantity"]
        
        if total_qty > 0:
            self.state.vwap_price = total_value / total_qty
        
        # Check if execution is complete
        if math.isclose(self.state.executed_quantity, self.state.total_quantity, rel_tol=1e-5) or \
           self.state.executed_quantity >= self.state.total_quantity:
            self.state.is_complete = True
            self.state.end_time = datetime.datetime.now()
            logger.info(f"Execution complete for {self.state.symbol} {self.state.side.value}. "
                       f"VWAP: {self.state.vwap_price:.4f}")
    
    def cancel_execution(self) -> None:
        """Cancel the execution strategy."""
        if self.state is None:
            logger.error("Execution state not initialized")
            return
        
        self.state.is_cancelled = True
        self.state.end_time = datetime.datetime.now()
        logger.info(f"Execution cancelled for {self.state.symbol} {self.state.side.value}. "
                   f"Executed: {self.state.executed_quantity}/{self.state.total_quantity}")
    
    def get_execution_state(self) -> Optional[ExecutionState]:
        """Get the current execution state.
        
        Returns:
            Current execution state or None if not initialized
        """
        return self.state
    
    def get_execution_progress(self) -> float:
        """Get the execution progress as a percentage.
        
        Returns:
            Execution progress (0.0-1.0)
        """
        if self.state is None or self.state.total_quantity == 0:
            return 0.0
        
        return self.state.executed_quantity / self.state.total_quantity
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics.
        
        Returns:
            Dictionary of execution metrics
        """
        if self.state is None:
            return {}
        
        # Calculate execution time
        start_time = self.state.start_time or datetime.datetime.now()
        end_time = self.state.end_time or datetime.datetime.now()
        execution_time_seconds = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        metrics = {
            "symbol": self.state.symbol,
            "side": self.state.side.value,
            "strategy": self.params.strategy_type.value,
            "total_quantity": self.state.total_quantity,
            "executed_quantity": self.state.executed_quantity,
            "execution_progress": self.get_execution_progress(),
            "start_price": self.state.start_price,
            "vwap_price": self.state.vwap_price,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat() if self.state.end_time else None,
            "execution_time_seconds": execution_time_seconds,
            "is_complete": self.state.is_complete,
            "is_cancelled": self.state.is_cancelled,
            "num_child_orders": len(self.state.child_orders),
            "num_executions": len(self.state.execution_logs)
        }
        
        # Calculate slippage if execution is complete or has executions
        if self.state.vwap_price > 0:
            if self.state.side == OrderSide.BUY:
                # For buys, positive slippage means execution price is lower than start price
                metrics["slippage_bps"] = (self.state.start_price - self.state.vwap_price) / self.state.start_price * 10000
            else:
                # For sells, positive slippage means execution price is higher than start price
                metrics["slippage_bps"] = (self.state.vwap_price - self.state.start_price) / self.state.start_price * 10000
        
        return metrics


class MarketExecutionStrategy(ExecutionStrategy):
    """Market execution strategy.
    
    Executes the entire order immediately at market price.
    """
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        # For market execution, we submit a single market order for the entire quantity
        if self.state.remaining_quantity > 0 and len(self.state.child_orders) == 0:
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": self.state.remaining_quantity,
                "order_type": OrderType.MARKET.value,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value
            }
            
            self.state.child_orders.append(order)
            return [order]
        
        return []


class LimitExecutionStrategy(ExecutionStrategy):
    """Limit execution strategy.
    
    Executes the order using limit orders with specified price offset.
    """
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        # Calculate limit price based on current market data and parameters
        limit_price = self._calculate_limit_price(market_data)
        
        # Check if we need to submit a new order or update existing ones
        if self.state.remaining_quantity > 0 and len(self.state.child_orders) == 0:
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": self.state.remaining_quantity,
                "order_type": OrderType.LIMIT.value,
                "limit_price": limit_price,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value
            }
            
            self.state.child_orders.append(order)
            return [order]
        
        return []
    
    def _calculate_limit_price(self, market_data: MarketData) -> float:
        """Calculate limit price based on market data and parameters.
        
        Args:
            market_data: Current market data
            
        Returns:
            Calculated limit price
        """
        # Get reference price (use mid-price if bid/ask available, otherwise use current price)
        reference_price = market_data.current_price
        if market_data.bid > 0 and market_data.ask > 0:
            reference_price = (market_data.bid + market_data.ask) / 2
        
        # Calculate offset based on aggression level and spread
        spread = market_data.spread if market_data.spread > 0 else (market_data.ask - market_data.bid)
        if spread <= 0:
            # If spread is not available, use a default percentage of price
            spread = reference_price * 0.001  # 0.1% default spread
        
        # Scale offset by aggression (higher aggression = smaller offset)
        offset = spread * (1.0 - self.params.limit_price_aggression)
        
        # Add fixed offset if specified
        offset += self.params.limit_price_offset
        
        # Calculate limit price based on side
        if self.state.side == OrderSide.BUY:
            # For buys, limit price is below reference price
            limit_price = reference_price - offset
        else:
            # For sells, limit price is above reference price
            limit_price = reference_price + offset
        
        return limit_price


class TWAPExecutionStrategy(ExecutionStrategy):
    """Time-Weighted Average Price (TWAP) execution strategy.
    
    Executes the order in equal-sized slices over a specified time period.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the TWAP execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.next_slice_time = None
        self.slice_count = 0
    
    def start_execution(self, symbol: str, side: OrderSide, quantity: float, 
                       current_price: float) -> ExecutionState:
        """Start execution of an order.
        
        Args:
            symbol: Symbol/ticker
            side: Order side (buy or sell)
            quantity: Total quantity to execute
            current_price: Current market price
            
        Returns:
            Initial execution state
        """
        state = super().start_execution(symbol, side, quantity, current_price)
        
        # Initialize TWAP-specific state
        self.next_slice_time = datetime.datetime.now()
        self.slice_count = 0
        
        return state
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        # Check if it's time for the next slice
        current_time = datetime.datetime.now()
        if self.next_slice_time is None or current_time >= self.next_slice_time:
            # Calculate slice size
            remaining_slices = self.params.twap_num_slices - self.slice_count
            if remaining_slices <= 0:
                return []
            
            slice_size = self.state.remaining_quantity / remaining_slices
            if slice_size <= 0:
                return []
            
            # Create order for this slice
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": slice_size,
                "order_type": OrderType.MARKET.value,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value,
                "slice_number": self.slice_count + 1
            }
            
            # Update state
            self.state.child_orders.append(order)
            self.slice_count += 1
            self.next_slice_time = current_time + datetime.timedelta(seconds=self.params.twap_slice_interval_seconds)
            
            logger.info(f"TWAP slice {self.slice_count}/{self.params.twap_num_slices} for {self.state.symbol} "
                       f"size: {slice_size:.2f}, next slice at {self.next_slice_time}")
            
            return [order]
        
        return []


class VWAPExecutionStrategy(ExecutionStrategy):
    """Volume-Weighted Average Price (VWAP) execution strategy.
    
    Executes the order according to expected volume profile over a specified time period.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the VWAP execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.volume_profile = {}
        self.next_check_time = None
        self.executed_profile_pct = 0.0
    
    def start_execution(self, symbol: str, side: OrderSide, quantity: float, 
                       current_price: float) -> ExecutionState:
        """Start execution of an order.
        
        Args:
            symbol: Symbol/ticker
            side: Order side (buy or sell)
            quantity: Total quantity to execute
            current_price: Current market price
            
        Returns:
            Initial execution state
        """
        state = super().start_execution(symbol, side, quantity, current_price)
        
        # Initialize VWAP-specific state
        self._initialize_volume_profile()
        self.next_check_time = datetime.datetime.now()
        self.executed_profile_pct = 0.0
        
        return state
    
    def _initialize_volume_profile(self) -> None:
        """Initialize the volume profile for VWAP execution."""
        # Use custom profile if provided
        if self.params.vwap_custom_profile:
            self.volume_profile = self.params.vwap_custom_profile.copy()
            return
        
        # Use a default volume profile if historical data not available or not requested
        if not self.params.vwap_use_historical_profile:
            # Default U-shaped volume profile (higher volume at open and close)
            self.volume_profile = {
                "09:30": 0.08, "10:00": 0.06, "10:30": 0.05, "11:00": 0.04, "11:30": 0.04,
                "12:00": 0.03, "12:30": 0.03, "13:00": 0.03, "13:30": 0.04, "14:00": 0.04,
                "14:30": 0.05, "15:00": 0.06, "15:30": 0.08, "16:00": 0.10
            }
    
    def _get_target_executed_percentage(self, current_time: datetime.datetime) -> float:
        """Get the target executed percentage based on the current time and volume profile.
        
        Args:
            current_time: Current time
            
        Returns:
            Target executed percentage (0.0-1.0)
        """
        # Convert current time to a string key in HH:MM format
        time_key = current_time.strftime("%H:%M")
        
        # Find the closest time key in the volume profile
        closest_key = None
        min_diff = float('inf')
        
        for key in self.volume_profile.keys():
            key_time = datetime.datetime.strptime(key, "%H:%M").time()
            current_time_only = current_time.time()
            
            # Calculate time difference in minutes
            key_minutes = key_time.hour * 60 + key_time.minute
            current_minutes = current_time_only.hour * 60 + current_time_only.minute
            diff = abs(key_minutes - current_minutes)
            
            if diff < min_diff:
                min_diff = diff
                closest_key = key
        
        if closest_key is None:
            return 0.0
        
        # Calculate cumulative percentage up to this time
        cumulative_pct = 0.0
        for key, pct in self.volume_profile.items():
            if key <= closest_key:
                cumulative_pct += pct
        
        return min(cumulative_pct, 1.0)
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        # Check if it's time to evaluate VWAP progress
        current_time = datetime.datetime.now()
        if self.next_check_time is None or current_time >= self.next_check_time:
            # Get target executed percentage based on volume profile
            target_pct = self._get_target_executed_percentage(current_time)
            
            # Calculate how much we should have executed by now
            target_executed = self.state.total_quantity * target_pct
            
            # Calculate how much more we need to execute in this interval
            additional_qty = target_executed - self.state.executed_quantity
            
            # Only create an order if we need to execute more
            if additional_qty > 0:
                order = {
                    "symbol": self.state.symbol,
                    "side": self.state.side.value,
                    "quantity": additional_qty,
                    "order_type": OrderType.MARKET.value,
                    "time_in_force": TimeInForce.DAY.value,
                    "parent_strategy": self.params.strategy_type.value,
                    "target_pct": target_pct
                }
                
                self.state.child_orders.append(order)
                self.executed_profile_pct = target_pct
                
                logger.info(f"VWAP order for {self.state.symbol} at {current_time.strftime('%H:%M:%S')}, "
                           f"target: {target_pct:.2%}, qty: {additional_qty:.2f}")
                
                # Set next check time (every 5 minutes by default)
                self.next_check_time = current_time + datetime.timedelta(minutes=5)
                
                return [order]
        
        return []


class PercentageOfVolumeStrategy(ExecutionStrategy):
    """Percentage of Volume (POV) execution strategy.
    
    Executes the order as a percentage of market volume.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the POV execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.last_volume_check_time = None
        self.accumulated_market_volume = 0.0
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        current_time = datetime.datetime.now()
        
        # Initialize volume tracking if this is the first update
        if self.last_volume_check_time is None:
            self.last_volume_check_time = current_time
            self.accumulated_market_volume = 0.0
            return []
        
        # Calculate time since last check
        time_diff_seconds = (current_time - self.last_volume_check_time).total_seconds()
        
        # Only check periodically (e.g., every 30 seconds)
        if time_diff_seconds < 30:
            return []
        
        # Get market volume since last check
        new_volume = market_data.last_volume
        if new_volume <= 0:
            # If no volume data, estimate based on average daily volume
            if market_data.average_daily_volume > 0:
                # Estimate volume for the time period (assuming 6.5 hour trading day)
                trading_seconds_per_day = 6.5 * 3600
                estimated_volume = market_data.average_daily_volume * (time_diff_seconds / trading_seconds_per_day)
                new_volume = estimated_volume
            else:
                # No volume data available
                self.last_volume_check_time = current_time
                return []
        
        # Add to accumulated volume
        self.accumulated_market_volume += new_volume
        
        # Calculate target participation
        target_participation = self.params.pov_target_rate
        
        # Adjust participation rate based on market conditions
        if market_data.is_stressed:
            # Reduce participation in stressed markets
            target_participation = max(self.params.pov_min_rate, target_participation * 0.5)
        elif market_data.volatility > 0 and market_data.volatility > 0.02:  # High volatility
            # Reduce participation in volatile markets
            target_participation = max(self.params.pov_min_rate, target_participation * 0.7)
        
        # Cap at maximum participation rate
        target_participation = min(target_participation, self.params.pov_max_rate)
        
        # Calculate quantity to execute
        target_qty = self.accumulated_market_volume * target_participation
        
        # Cap at remaining quantity
        target_qty = min(target_qty, self.state.remaining_quantity)
        
        # Only create an order if quantity is significant
        if target_qty > 0.01:  # Minimum order size threshold
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": target_qty,
                "order_type": OrderType.MARKET.value,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value,
                "participation_rate": target_participation
            }
            
            self.state.child_orders.append(order)
            
            logger.info(f"POV order for {self.state.symbol} at {current_time.strftime('%H:%M:%S')}, "
                       f"participation: {target_participation:.2%}, qty: {target_qty:.2f}")
            
            # Reset accumulated volume
            self.accumulated_market_volume = 0.0
            self.last_volume_check_time = current_time
            
            return [order]
        
        # Update last check time
        self.last_volume_check_time = current_time
        
        return []


class IcebergExecutionStrategy(ExecutionStrategy):
    """Iceberg execution strategy.
    
    Executes the order by showing only a portion of the total quantity at a time.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the iceberg execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.current_iceberg_order = None
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        # If we don't have an active iceberg order and have remaining quantity, create one
        if self.current_iceberg_order is None and self.state.remaining_quantity > 0:
            # Calculate display size
            display_size = self.params.iceberg_display_size
            if display_size <= 0:
                # Default to 5% of total quantity if not specified
                display_size = self.state.total_quantity * 0.05
            
            # Ensure minimum size
            if self.params.iceberg_min_size > 0:
                display_size = max(display_size, self.params.iceberg_min_size)
            
            # Cap at remaining quantity
            display_size = min(display_size, self.state.remaining_quantity)
            
            # Create iceberg order
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": self.state.remaining_quantity,  # Total quantity
                "display_quantity": display_size,  # Displayed quantity
                "order_type": OrderType.LIMIT.value,
                "limit_price": self._calculate_limit_price(market_data),
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value,
                "is_iceberg": True
            }
            
            self.current_iceberg_order = order
            self.state.child_orders.append(order)
            
            logger.info(f"Iceberg order for {self.state.symbol}, total: {self.state.remaining_quantity:.2f}, "
                       f"display: {display_size:.2f}")
            
            return [order]
        
        return []
    
    def _calculate_limit_price(self, market_data: MarketData) -> float:
        """Calculate limit price for the iceberg order.
        
        Args:
            market_data: Current market data
            
        Returns:
            Calculated limit price
        """
        # Use mid-price if bid/ask available, otherwise use current price
        price = market_data.current_price
        if market_data.bid > 0 and market_data.ask > 0:
            price = (market_data.bid + market_data.ask) / 2
        
        # Apply offset based on side
        offset = price * 0.001  # Default 0.1% offset
        
        if self.state.side == OrderSide.BUY:
            return price - offset
        else:
            return price + offset
    
    def process_execution(self, order_id: str, executed_quantity: float, 
                         execution_price: float) -> None:
        """Process an execution report.
        
        Args:
            order_id: ID of the executed order
            executed_quantity: Quantity executed
            execution_price: Execution price
        """
        super().process_execution(order_id, executed_quantity, execution_price)
        
        # Reset current iceberg order if fully executed
        if self.state.is_complete:
            self.current_iceberg_order = None


class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adaptive execution strategy.
    
    Adjusts execution parameters based on market conditions.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the adaptive execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.last_order_time = None
        self.market_condition_score = 0.5  # 0.0 = very passive, 1.0 = very aggressive
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        current_time = datetime.datetime.now()
        
        # Update market condition score
        self._update_market_condition_score(market_data)
        
        # Determine if we should place an order now
        should_order = False
        
        # If this is the first order or enough time has passed since last order
        if self.last_order_time is None:
            should_order = True
        else:
            # Calculate time since last order
            time_diff_seconds = (current_time - self.last_order_time).total_seconds()
            
            # More aggressive in favorable conditions (shorter wait time)
            min_wait_time = 30 * (1.0 - self.market_condition_score) + 5
            
            if time_diff_seconds >= min_wait_time:
                should_order = True
        
        if should_order and self.state.remaining_quantity > 0:
            # Calculate order size based on market conditions
            order_size = self._calculate_adaptive_size(market_data)
            
            # Cap at remaining quantity
            order_size = min(order_size, self.state.remaining_quantity)
            
            # Determine order type based on market conditions
            order_type = OrderType.MARKET.value
            limit_price = None
            
            # Use limit orders in less favorable conditions
            if self.market_condition_score < 0.7:
                order_type = OrderType.LIMIT.value
                limit_price = self._calculate_adaptive_limit_price(market_data)
            
            # Create order
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": order_size,
                "order_type": order_type,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value,
                "market_condition_score": self.market_condition_score
            }
            
            if limit_price is not None:
                order["limit_price"] = limit_price
            
            self.state.child_orders.append(order)
            self.last_order_time = current_time
            
            logger.info(f"Adaptive order for {self.state.symbol}, size: {order_size:.2f}, "
                       f"type: {order_type}, condition: {self.market_condition_score:.2f}")
            
            return [order]
        
        return []
    
    def _update_market_condition_score(self, market_data: MarketData) -> None:
        """Update the market condition score based on current market data.
        
        Args:
            market_data: Current market data
        """
        # Start with neutral score
        score = 0.5
        
        # Adjust based on volatility (lower score for higher volatility)
        if market_data.volatility > 0:
            vol_factor = max(0.0, min(1.0, 1.0 - (market_data.volatility / 0.03)))
            score = score * 0.7 + vol_factor * 0.3
        
        # Adjust based on spread (lower score for wider spread)
        if market_data.spread > 0 and market_data.current_price > 0:
            spread_bps = market_data.spread / market_data.current_price * 10000
            spread_factor = max(0.0, min(1.0, 1.0 - (spread_bps / 20)))
            score = score * 0.7 + spread_factor * 0.3
        
        # Adjust based on liquidity (lower score for lower liquidity)
        if market_data.liquidity > 0:
            liq_factor = min(1.0, market_data.liquidity)
            score = score * 0.8 + liq_factor * 0.2
        
        # Significant reduction for stressed markets
        if market_data.is_stressed:
            score *= 0.5
        
        # Ensure score is within bounds
        self.market_condition_score = max(0.1, min(0.9, score))
    
    def _calculate_adaptive_size(self, market_data: MarketData) -> float:
        """Calculate adaptive order size based on market conditions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Calculated order size
        """
        # Base size as a percentage of remaining quantity
        base_pct = 0.1 + (self.market_condition_score * 0.3)  # 10% to 40% based on conditions
        base_size = self.state.remaining_quantity * base_pct
        
        # Apply min/max constraints if specified
        if self.params.adaptive_min_size > 0:
            base_size = max(base_size, self.params.adaptive_min_size)
        
        if self.params.adaptive_max_size > 0:
            base_size = min(base_size, self.params.adaptive_max_size)
        
        # Scale by factor
        base_size *= self.params.adaptive_size_scale_factor
        
        return base_size
    
    def _calculate_adaptive_limit_price(self, market_data: MarketData) -> float:
        """Calculate adaptive limit price based on market conditions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Calculated limit price
        """
        # Use mid-price if bid/ask available, otherwise use current price
        price = market_data.current_price
        if market_data.bid > 0 and market_data.ask > 0:
            price = (market_data.bid + market_data.ask) / 2
        
        # Calculate offset based on market condition score
        # More aggressive (smaller offset) in favorable conditions
        base_offset_bps = 10 * (1.0 - self.market_condition_score)  # 0 to 10 bps
        offset = price * (base_offset_bps / 10000)
        
        # Apply offset based on side
        if self.state.side == OrderSide.BUY:
            return price - offset
        else:
            return price + offset


class ImplementationShortfallStrategy(ExecutionStrategy):
    """Implementation Shortfall execution strategy.
    
    Balances market impact and timing risk based on price prediction and risk model.
    """
    def __init__(self, params: ExecutionParameters):
        """Initialize the Implementation Shortfall execution strategy.
        
        Args:
            params: Parameters for the execution strategy
        """
        super().__init__(params)
        self.last_evaluation_time = None
        self.remaining_trading_time_seconds = 0
        self.initial_trading_time_seconds = 0
    
    def start_execution(self, symbol: str, side: OrderSide, quantity: float, 
                       current_price: float) -> ExecutionState:
        """Start execution of an order.
        
        Args:
            symbol: Symbol/ticker
            side: Order side (buy or sell)
            quantity: Total quantity to execute
            current_price: Current market price
            
        Returns:
            Initial execution state
        """
        state = super().start_execution(symbol, side, quantity, current_price)
        
        # Initialize strategy-specific state
        self.last_evaluation_time = datetime.datetime.now()
        
        # Set trading time window
        if self.params.end_time is not None:
            self.remaining_trading_time_seconds = (self.params.end_time - self.last_evaluation_time).total_seconds()
            self.initial_trading_time_seconds = self.remaining_trading_time_seconds
        else:
            # Default to 1 hour if not specified
            self.remaining_trading_time_seconds = 3600
            self.initial_trading_time_seconds = 3600
        
        return state
    
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        current_time = datetime.datetime.now()
        
        # Update remaining trading time
        if self.params.end_time is not None:
            self.remaining_trading_time_seconds = max(0, (self.params.end_time - current_time).total_seconds())
        else:
            time_elapsed = (current_time - self.last_evaluation_time).total_seconds()
            self.remaining_trading_time_seconds = max(0, self.remaining_trading_time_seconds - time_elapsed)
        
        # Only evaluate periodically (e.g., every minute)
        time_since_last = (current_time - self.last_evaluation_time).total_seconds()
        if time_since_last < 60 and self.remaining_trading_time_seconds > 60:
            return []
        
        # Calculate optimal trading rate based on IS model
        trading_rate = self._calculate_optimal_trading_rate(market_data)
        
        # Calculate target quantity for this period
        target_qty = trading_rate * time_since_last * self.state.total_quantity / self.initial_trading_time_seconds
        
        # Cap at remaining quantity
        target_qty = min(target_qty, self.state.remaining_quantity)
        
        # Only create an order if quantity is significant
        if target_qty > 0.01:  # Minimum order size threshold
            # Determine order type based on urgency
            order_type = OrderType.MARKET.value
            limit_price = None
            
            # Use limit orders for less urgent executions
            if trading_rate < 0.8 and self.remaining_trading_time_seconds > 300:
                order_type = OrderType.LIMIT.value
                limit_price = self._calculate_limit_price(market_data)
            
            # Create order
            order = {
                "symbol": self.state.symbol,
                "side": self.state.side.value,
                "quantity": target_qty,
                "order_type": order_type,
                "time_in_force": TimeInForce.DAY.value,
                "parent_strategy": self.params.strategy_type.value,
                "trading_rate": trading_rate
            }
            
            if limit_price is not None:
                order["limit_price"] = limit_price
            
            self.state.child_orders.append(order)
            self.last_evaluation_time = current_time
            
            logger.info(f"IS order for {self.state.symbol}, qty: {target_qty:.2f}, "
                       f"rate: {trading_rate:.2f}, remaining time: {self.remaining_trading_time_seconds:.0f}s")
            
            return [order]
        
        # Update last evaluation time
        self.last_evaluation_time = current_time
        
        return []
    
    def _calculate_optimal_trading_rate(self, market_data: MarketData) -> float:
        """Calculate optimal trading rate based on Implementation Shortfall model.
        
        Args:
            market_data: Current market data
            
        Returns:
            Optimal trading rate (0.0-1.0)
        """
        # Implementation of Almgren-Chriss model (simplified)
        # Trading rate increases with:
        # - Less remaining time
        # - Higher risk aversion
        # - Lower market impact
        # - Higher volatility
        
        # Default parameters if not available
        volatility = market_data.volatility if market_data.volatility > 0 else 0.01
        
        # Time-based component (accelerate as we approach the deadline)
        time_remaining_ratio = self.remaining_trading_time_seconds / self.initial_trading_time_seconds
        time_component = 1.0 - time_remaining_ratio
        
        # Risk aversion component
        risk_component = self.params.is_risk_aversion
        
        # Market impact component (lower impact = higher rate)
        impact_factor = 0.5
        if market_data.liquidity > 0:
            impact_factor = min(1.0, 1.0 / market_data.liquidity)
        
        # Volatility component (higher volatility = higher rate with high risk aversion)
        vol_component = volatility * self.params.is_risk_aversion
        
        # Combine components
        trading_rate = 0.3 * time_component + 0.3 * risk_component + 0.2 * (1.0 - impact_factor) + 0.2 * vol_component
        
        # Adjust for market stress
        if market_data.is_stressed:
            if self.params.is_risk_aversion > 0.5:
                # High risk aversion: accelerate in stressed markets
                trading_rate = min(1.0, trading_rate * 1.5)
            else:
                # Low risk aversion: slow down in stressed markets
                trading_rate = trading_rate * 0.7
        
        # Ensure rate is within bounds
        trading_rate = max(0.1, min(1.0, trading_rate))
        
        return trading_rate
    
    def _calculate_limit_price(self, market_data: MarketData) -> float:
        """Calculate limit price for IS strategy.
        
        Args:
            market_data: Current market data
            
        Returns:
            Calculated limit price
        """
        # Use mid-price if bid/ask available, otherwise use current price
        price = market_data.current_price
        if market_data.bid > 0 and market_data.ask > 0:
            price = (market_data.bid + market_data.ask) / 2
        
        # Calculate offset based on remaining time and risk aversion
        # More aggressive (smaller offset) with less time or higher risk aversion
        time_factor = 1.0 - (self.remaining_trading_time_seconds / self.initial_trading_time_seconds)
        base_offset_bps = 5 * (1.0 - (time_factor + self.params.is_risk_aversion) / 2)  # 0 to 5 bps
        offset = price * (base_offset_bps / 10000)
        
        # Apply offset based on side
        if self.state.side == OrderSide.BUY:
            return price - offset
        else:
            return price + offset


class CustomExecutionStrategy(ExecutionStrategy):
    """Custom execution strategy.
    
    Uses a custom function to determine execution behavior.
    """
    def process_market_update(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Process a market data update and generate orders if needed.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to submit
        """
        if self.state is None or self.state.is_complete or self.state.is_cancelled:
            return []
        
        if self.params.custom_strategy is None:
            logger.warning("No custom strategy function provided")
            return []
        
        try:
            # Call custom strategy function
            orders = self.params.custom_strategy(
                self.state, market_data, self.params.custom_params)
            
            # Add orders to child orders list
            if orders:
                for order in orders:
                    self.state.child_orders.append(order)
            
            return orders
        except Exception as e:
            logger.error(f"Error in custom execution strategy: {e}")
            return []


class ExecutionStrategyFactory:
    """Factory for creating execution strategies."""
    @staticmethod
    def create_strategy(params: ExecutionParameters) -> ExecutionStrategy:
        """Create an execution strategy based on parameters.
        
        Args:
            params: Parameters for the execution strategy
            
        Returns:
            ExecutionStrategy instance
        """
        if params.strategy_type == ExecutionStrategyType.MARKET:
            return MarketExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.LIMIT:
            return LimitExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.TWAP:
            return TWAPExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.VWAP:
            return VWAPExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.PERCENTAGE_OF_VOLUME:
            return PercentageOfVolumeStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.ICEBERG:
            return IcebergExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.ADAPTIVE:
            return AdaptiveExecutionStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.IMPLEMENTATION_SHORTFALL:
            return ImplementationShortfallStrategy(params)
        elif params.strategy_type == ExecutionStrategyType.CUSTOM:
            return CustomExecutionStrategy(params)
        else:
            logger.warning(f"Unknown strategy type: {params.strategy_type}, using market strategy")
            return MarketExecutionStrategy(params)
    
    @staticmethod
    def create_default_strategy() -> ExecutionStrategy:
        """Create a default execution strategy.
        
        Returns:
            Default ExecutionStrategy instance
        """
        params = ExecutionParameters()
        return MarketExecutionStrategy(params)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> ExecutionStrategy:
        """Create an execution strategy from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ExecutionStrategy instance
        """
        # Parse strategy type
        strategy_type_str = config.get("strategy_type", "market")
        try:
            strategy_type = ExecutionStrategyType(strategy_type_str)
        except ValueError:
            logger.warning(f"Invalid strategy type: {strategy_type_str}, using market strategy")
            strategy_type = ExecutionStrategyType.MARKET
        
        # Create parameters
        params = ExecutionParameters(
            strategy_type=strategy_type,
            max_participation_rate=config.get("max_participation_rate", 0.3),
            urgency=config.get("urgency", 0.5)
        )
        
        # Parse start/end times if provided
        if "start_time" in config and isinstance(config["start_time"], str):
            try:
                params.start_time = datetime.datetime.fromisoformat(config["start_time"])
            except ValueError:
                logger.warning(f"Invalid start_time format: {config['start_time']}")
        
        if "end_time" in config and isinstance(config["end_time"], str):
            try:
                params.end_time = datetime.datetime.fromisoformat(config["end_time"])
            except ValueError:
                logger.warning(f"Invalid end_time format: {config['end_time']}")
        
        # Add strategy-specific parameters
        if strategy_type == ExecutionStrategyType.TWAP:
            params.twap_num_slices = config.get("twap_num_slices", 10)
            params.twap_slice_interval_seconds = config.get("twap_slice_interval_seconds", 60)
        
        elif strategy_type == ExecutionStrategyType.VWAP:
            params.vwap_use_historical_profile = config.get("vwap_use_historical_profile", True)
            if "vwap_custom_profile" in config and isinstance(config["vwap_custom_profile"], dict):
                params.vwap_custom_profile = config["vwap_custom_profile"]
        
        elif strategy_type == ExecutionStrategyType.PERCENTAGE_OF_VOLUME:
            params.pov_target_rate = config.get("pov_target_rate", 0.1)
            params.pov_min_rate = config.get("pov_min_rate", 0.05)
            params.pov_max_rate = config.get("pov_max_rate", 0.2)
        
        elif strategy_type == ExecutionStrategyType.ICEBERG:
            params.iceberg_display_size = config.get("iceberg_display_size", 0.0)
            params.iceberg_min_size = config.get("iceberg_min_size", 0.0)
        
        elif strategy_type == ExecutionStrategyType.ADAPTIVE:
            params.adaptive_min_size = config.get("adaptive_min_size", 0.0)
            params.adaptive_max_size = config.get("adaptive_max_size", 0.0)
            params.adaptive_size_scale_factor = config.get("adaptive_size_scale_factor", 1.0)
        
        elif strategy_type == ExecutionStrategyType.IMPLEMENTATION_SHORTFALL:
            params.is_risk_aversion = config.get("is_risk_aversion", 0.5)
            params.is_alpha_decay = config.get("is_alpha_decay", 0.1)
        
        # Add custom parameters if present
        if "custom_params" in config and isinstance(config["custom_params"], dict):
            params.custom_params = config["custom_params"]
        
        return ExecutionStrategyFactory.create_strategy(params)

    @staticmethod
    def create_twap_strategy(num_slices: int = 10, slice_interval_seconds: int = 60) -> ExecutionStrategy:
        """Create a TWAP execution strategy.
        
        Args:
            num_slices: Number of slices for TWAP
            slice_interval_seconds: Interval between slices in seconds
            
        Returns:
            TWAP ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.TWAP,
            twap_num_slices=num_slices,
            twap_slice_interval_seconds=slice_interval_seconds
        )
        return TWAPExecutionStrategy(params)
    
    @staticmethod
    def create_vwap_strategy(use_historical_profile: bool = True,
                           custom_profile: Optional[Dict[str, float]] = None) -> ExecutionStrategy:
        """Create a VWAP execution strategy.
        
        Args:
            use_historical_profile: Whether to use historical volume profile
            custom_profile: Custom volume profile
            
        Returns:
            VWAP ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.VWAP,
            vwap_use_historical_profile=use_historical_profile
        )
        
        if custom_profile is not None:
            params.vwap_custom_profile = custom_profile
        
        return VWAPExecutionStrategy(params)
    
    @staticmethod
    def create_pov_strategy(target_rate: float = 0.1,
                          min_rate: float = 0.05,
                          max_rate: float = 0.2) -> ExecutionStrategy:
        """Create a Percentage of Volume execution strategy.
        
        Args:
            target_rate: Target participation rate (0.0-1.0)
            min_rate: Minimum participation rate
            max_rate: Maximum participation rate
            
        Returns:
            POV ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.PERCENTAGE_OF_VOLUME,
            pov_target_rate=target_rate,
            pov_min_rate=min_rate,
            pov_max_rate=max_rate
        )
        return PercentageOfVolumeStrategy(params)
    
    @staticmethod
    def create_iceberg_strategy(display_size: float = 0.0,
                              min_size: float = 0.0) -> ExecutionStrategy:
        """Create an Iceberg execution strategy.
        
        Args:
            display_size: Display size for iceberg orders
            min_size: Minimum size for iceberg slices
            
        Returns:
            Iceberg ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.ICEBERG,
            iceberg_display_size=display_size,
            iceberg_min_size=min_size
        )
        return IcebergExecutionStrategy(params)
    
    @staticmethod
    def create_adaptive_strategy(min_size: float = 0.0,
                               max_size: float = 0.0,
                               size_scale_factor: float = 1.0) -> ExecutionStrategy:
        """Create an Adaptive execution strategy.
        
        Args:
            min_size: Minimum order size
            max_size: Maximum order size
            size_scale_factor: Scaling factor for order size
            
        Returns:
            Adaptive ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.ADAPTIVE,
            adaptive_min_size=min_size,
            adaptive_max_size=max_size,
            adaptive_size_scale_factor=size_scale_factor
        )
        return AdaptiveExecutionStrategy(params)
    
    @staticmethod
    def create_implementation_shortfall_strategy(risk_aversion: float = 0.5,
                                               alpha_decay: float = 0.1) -> ExecutionStrategy:
        """Create an Implementation Shortfall execution strategy.
        
        Args:
            risk_aversion: Risk aversion parameter (0.0-1.0)
            alpha_decay: Alpha decay parameter
            
        Returns:
            Implementation Shortfall ExecutionStrategy instance
        """
        params = ExecutionParameters(
            strategy_type=ExecutionStrategyType.IMPLEMENTATION_SHORTFALL,
            is_risk_aversion=risk_aversion,
            is_alpha_decay=alpha_decay
        )
        return ImplementationShortfallStrategy(params)


# Production-ready execution strategy configurations
def create_production_twap_strategy(symbol_type: str = "equity") -> ExecutionStrategy:
    """Create a production-ready TWAP strategy with appropriate parameters based on symbol type.
    
    Args:
        symbol_type: Type of symbol (equity, futures, options, forex, crypto)
        
    Returns:
        Production-configured TWAP strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.TWAP)
    
    # Configure based on symbol type
    if symbol_type == "equity":
        params.twap_num_slices = 12
        params.twap_slice_interval_seconds = 300  # 5 minutes
    elif symbol_type == "futures":
        params.twap_num_slices = 20
        params.twap_slice_interval_seconds = 180  # 3 minutes
    elif symbol_type == "options":
        params.twap_num_slices = 8
        params.twap_slice_interval_seconds = 450  # 7.5 minutes
    elif symbol_type == "forex":
        params.twap_num_slices = 24
        params.twap_slice_interval_seconds = 150  # 2.5 minutes
    elif symbol_type == "crypto":
        params.twap_num_slices = 30
        params.twap_slice_interval_seconds = 120  # 2 minutes
    else:
        # Default configuration
        params.twap_num_slices = 12
        params.twap_slice_interval_seconds = 300  # 5 minutes
    
    return TWAPExecutionStrategy(params)


def create_production_vwap_strategy(market: str = "US") -> ExecutionStrategy:
    """Create a production-ready VWAP strategy with appropriate parameters based on market.
    
    Args:
        market: Market (US, Europe, Asia)
        
    Returns:
        Production-configured VWAP strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.VWAP)
    params.vwap_use_historical_profile = True
    
    # Configure volume profiles based on market
    if market == "US":
        # U.S. market typical volume profile (higher at open and close)
        params.vwap_custom_profile = {
            "09:30": 0.08, "10:00": 0.07, "10:30": 0.06, "11:00": 0.05, "11:30": 0.04,
            "12:00": 0.03, "12:30": 0.03, "13:00": 0.03, "13:30": 0.04, "14:00": 0.05,
            "14:30": 0.06, "15:00": 0.07, "15:30": 0.09, "16:00": 0.10
        }
    elif market == "Europe":
        # European market volume profile
        params.vwap_custom_profile = {
            "08:00": 0.07, "08:30": 0.06, "09:00": 0.06, "09:30": 0.05, "10:00": 0.05,
            "10:30": 0.04, "11:00": 0.04, "11:30": 0.04, "12:00": 0.03, "12:30": 0.03,
            "13:00": 0.03, "13:30": 0.04, "14:00": 0.04, "14:30": 0.05, "15:00": 0.05,
            "15:30": 0.06, "16:00": 0.07, "16:30": 0.09, "17:00": 0.10
        }
    elif market == "Asia":
        # Asian market volume profile
        params.vwap_custom_profile = {
            "09:00": 0.09, "09:30": 0.07, "10:00": 0.06, "10:30": 0.05, "11:00": 0.04,
            "11:30": 0.03, "12:00": 0.02, "12:30": 0.02, "13:00": 0.02, "13:30": 0.03,
            "14:00": 0.05, "14:30": 0.06, "15:00": 0.08, "15:30": 0.10, "16:00": 0.08,
            "16:30": 0.06, "17:00": 0.04, "17:30": 0.03, "18:00": 0.02, "18:30": 0.02,
            "19:00": 0.03
        }
    else:
        # Default profile
        params.vwap_custom_profile = {
            "09:30": 0.08, "10:00": 0.06, "10:30": 0.05, "11:00": 0.04, "11:30": 0.04,
            "12:00": 0.03, "12:30": 0.03, "13:00": 0.03, "13:30": 0.04, "14:00": 0.04,
            "14:30": 0.05, "15:00": 0.06, "15:30": 0.08, "16:00": 0.10
        }
    
    return VWAPExecutionStrategy(params)


def create_production_pov_strategy(liquidity_profile: str = "normal") -> ExecutionStrategy:
    """Create a production-ready Percentage of Volume strategy with appropriate parameters.
    
    Args:
        liquidity_profile: Liquidity profile (low, normal, high)
        
    Returns:
        Production-configured POV strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.PERCENTAGE_OF_VOLUME)
    
    # Configure participation rate based on liquidity profile
    if liquidity_profile == "low":
        # Conservative settings for low liquidity
        params.pov_target_percentage = 0.05  # 5% participation
        params.pov_min_volume_threshold = 1000  # Minimum volume to participate
        params.pov_max_percentage = 0.08  # Cap at 8%
    elif liquidity_profile == "high":
        # Aggressive settings for high liquidity
        params.pov_target_percentage = 0.15  # 15% participation
        params.pov_min_volume_threshold = 500  # Lower threshold for high liquidity
        params.pov_max_percentage = 0.20  # Cap at 20%
    else:  # normal
        # Standard settings for normal liquidity
        params.pov_target_percentage = 0.10  # 10% participation
        params.pov_min_volume_threshold = 750  # Standard threshold
        params.pov_max_percentage = 0.15  # Cap at 15%
    
    # Common settings
    params.pov_check_interval_seconds = 30  # Check volume every 30 seconds
    
    return PercentageOfVolumeStrategy(params)


def create_production_iceberg_strategy(market_impact_profile: str = "medium") -> ExecutionStrategy:
    """Create a production-ready Iceberg strategy with appropriate parameters.
    
    Args:
        market_impact_profile: Market impact profile (low, medium, high)
        
    Returns:
        Production-configured Iceberg strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.ICEBERG)
    
    # Configure display size based on market impact profile
    if market_impact_profile == "low":
        # Larger display size for low market impact
        params.iceberg_display_size_percentage = 0.15  # 15% of total order
        params.iceberg_min_display_size = 200
        params.iceberg_max_display_size = 2000
    elif market_impact_profile == "high":
        # Smaller display size for high market impact
        params.iceberg_display_size_percentage = 0.05  # 5% of total order
        params.iceberg_min_display_size = 50
        params.iceberg_max_display_size = 500
    else:  # medium
        # Standard settings for medium market impact
        params.iceberg_display_size_percentage = 0.10  # 10% of total order
        params.iceberg_min_display_size = 100
        params.iceberg_max_display_size = 1000
    
    # Common settings
    params.iceberg_variance_percentage = 0.20  # 20% random variance in display size
    params.iceberg_time_between_slices_ms = 2000  # 2 seconds between slices
    
    return IcebergStrategy(params)


def create_production_adaptive_strategy(volatility_profile: str = "normal") -> ExecutionStrategy:
    """Create a production-ready Adaptive strategy with appropriate parameters.
    
    Args:
        volatility_profile: Volatility profile (low, normal, high)
        
    Returns:
        Production-configured Adaptive strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.ADAPTIVE)
    
    # Configure adaptive parameters based on volatility profile
    if volatility_profile == "low":
        # More aggressive in low volatility
        params.adaptive_initial_rate = 0.15  # Start at 15% participation
        params.adaptive_min_rate = 0.10  # Minimum 10% participation
        params.adaptive_max_rate = 0.25  # Maximum 25% participation
        params.adaptive_volatility_sensitivity = 0.5  # Less sensitive to volatility
    elif volatility_profile == "high":
        # More conservative in high volatility
        params.adaptive_initial_rate = 0.05  # Start at 5% participation
        params.adaptive_min_rate = 0.02  # Minimum 2% participation
        params.adaptive_max_rate = 0.10  # Maximum 10% participation
        params.adaptive_volatility_sensitivity = 2.0  # More sensitive to volatility
    else:  # normal
        # Balanced settings for normal volatility
        params.adaptive_initial_rate = 0.10  # Start at 10% participation
        params.adaptive_min_rate = 0.05  # Minimum 5% participation
        params.adaptive_max_rate = 0.15  # Maximum 15% participation
        params.adaptive_volatility_sensitivity = 1.0  # Standard sensitivity
    
    # Common settings
    params.adaptive_volume_lookback_periods = 5  # Look back 5 periods for volume analysis
    params.adaptive_price_lookback_periods = 10  # Look back 10 periods for price analysis
    
    return AdaptiveStrategy(params)


def create_production_implementation_shortfall_strategy(risk_profile: str = "balanced") -> ExecutionStrategy:
    """Create a production-ready Implementation Shortfall strategy with appropriate parameters.
    
    Args:
        risk_profile: Risk profile (risk_averse, balanced, alpha_seeking)
        
    Returns:
        Production-configured Implementation Shortfall strategy
    """
    params = ExecutionParameters(strategy_type=ExecutionStrategyType.IMPLEMENTATION_SHORTFALL)
    
    # Configure risk parameters based on risk profile
    if risk_profile == "risk_averse":
        # More risk-averse settings
        params.is_risk_aversion = 0.8  # High risk aversion
        params.is_alpha_decay = 0.05  # Slower alpha decay
    elif risk_profile == "alpha_seeking":
        # More alpha-seeking settings
        params.is_risk_aversion = 0.3  # Low risk aversion
        params.is_alpha_decay = 0.15  # Faster alpha decay
    else:  # balanced
        # Balanced settings
        params.is_risk_aversion = 0.5  # Medium risk aversion
        params.is_alpha_decay = 0.10  # Standard alpha decay
    
    return ImplementationShortfallStrategy(params)


# Emergency market condition handling
class EmergencyMarketConditionHandler:
    """Handler for adapting execution strategies during emergency market conditions.
    
    This class provides methods to adjust execution strategy parameters based on
    different market emergency conditions such as high volatility, low liquidity,
    wide spreads, and circuit breakers.
    """
    
    @staticmethod
    def adjust_for_high_volatility(strategy: ExecutionStrategy) -> ExecutionStrategy:
        """Adjust strategy parameters for high volatility conditions.
        
        Args:
            strategy: The original execution strategy
            
        Returns:
            Adjusted execution strategy for high volatility
        """
        params = strategy.parameters
        strategy_type = params.strategy_type
        
        if strategy_type == ExecutionStrategyType.TWAP:
            # Increase number of slices and reduce slice size
            params.twap_num_slices = int(params.twap_num_slices * 1.5)
            params.twap_slice_interval_seconds = int(params.twap_slice_interval_seconds * 0.8)
        
        elif strategy_type == ExecutionStrategyType.VWAP:
            # Adjust to be more conservative with volume profile
            if hasattr(params, 'vwap_custom_profile') and params.vwap_custom_profile:
                # Flatten the profile to be more evenly distributed
                total = sum(params.vwap_custom_profile.values())
                keys = list(params.vwap_custom_profile.keys())
                avg_value = total / len(keys)
                
                # Move values 30% toward the average
                for key in keys:
                    current = params.vwap_custom_profile[key]
                    params.vwap_custom_profile[key] = current * 0.7 + avg_value * 0.3
        
        elif strategy_type == ExecutionStrategyType.PERCENTAGE_OF_VOLUME:
            # Reduce participation rate
            params.pov_target_percentage = max(0.02, params.pov_target_percentage * 0.5)
            params.pov_max_percentage = max(0.05, params.pov_max_percentage * 0.5)
            # Increase minimum volume threshold
            params.pov_min_volume_threshold = int(params.pov_min_volume_threshold * 2)
        
        elif strategy_type == ExecutionStrategyType.ICEBERG:
            # Reduce display size
            params.iceberg_display_size_percentage = max(0.02, params.iceberg_display_size_percentage * 0.5)
            params.iceberg_max_display_size = int(params.iceberg_max_display_size * 0.5)
            # Increase time between slices
            params.iceberg_time_between_slices_ms = int(params.iceberg_time_between_slices_ms * 2)
        
        elif strategy_type == ExecutionStrategyType.ADAPTIVE:
            # Make more conservative
            params.adaptive_initial_rate = max(0.02, params.adaptive_initial_rate * 0.5)
            params.adaptive_min_rate = max(0.01, params.adaptive_min_rate * 0.5)
            params.adaptive_max_rate = max(0.05, params.adaptive_max_rate * 0.5)
            # Increase volatility sensitivity
            params.adaptive_volatility_sensitivity = params.adaptive_volatility_sensitivity * 2
        
        elif strategy_type == ExecutionStrategyType.IMPLEMENTATION_SHORTFALL:
            # Increase risk aversion
            params.is_risk_aversion = min(0.95, params.is_risk_aversion * 1.5)
            # Reduce alpha decay
            params.is_alpha_decay = max(0.01, params.is_alpha_decay * 0.5)
        
        return strategy
    
    @staticmethod
    def adjust_for_low_liquidity(strategy: ExecutionStrategy) -> ExecutionStrategy:
        """Adjust strategy parameters for low liquidity conditions.
        
        Args:
            strategy: The original execution strategy
            
        Returns:
            Adjusted execution strategy for low liquidity
        """
        params = strategy.parameters
        strategy_type = params.strategy_type
        
        if strategy_type == ExecutionStrategyType.TWAP:
            # Increase number of slices and extend time horizon
            params.twap_num_slices = int(params.twap_num_slices * 1.5)
            params.twap_slice_interval_seconds = int(params.twap_slice_interval_seconds * 1.5)
        
        elif strategy_type == ExecutionStrategyType.PERCENTAGE_OF_VOLUME:
            # Reduce participation rate significantly
            params.pov_target_percentage = max(0.01, params.pov_target_percentage * 0.3)
            params.pov_max_percentage = max(0.03, params.pov_max_percentage * 0.3)
            # Increase minimum volume threshold
            params.pov_min_volume_threshold = int(params.pov_min_volume_threshold * 3)
        
        elif strategy_type == ExecutionStrategyType.ICEBERG:
            # Reduce display size significantly
            params.iceberg_display_size_percentage = max(0.01, params.iceberg_display_size_percentage * 0.3)
            params.iceberg_max_display_size = int(params.iceberg_max_display_size * 0.3)
            # Increase time between slices
            params.iceberg_time_between_slices_ms = int(params.iceberg_time_between_slices_ms * 3)
        
        elif strategy_type == ExecutionStrategyType.ADAPTIVE:
            # Make extremely conservative
            params.adaptive_initial_rate = max(0.01, params.adaptive_initial_rate * 0.3)
            params.adaptive_min_rate = max(0.005, params.adaptive_min_rate * 0.3)
            params.adaptive_max_rate = max(0.03, params.adaptive_max_rate * 0.3)
        
        return strategy
    
    @staticmethod
    def adjust_for_wide_spreads(strategy: ExecutionStrategy) -> ExecutionStrategy:
        """Adjust strategy parameters for wide spread conditions.
        
        Args:
            strategy: The original execution strategy
            
        Returns:
            Adjusted execution strategy for wide spreads
        """
        params = strategy.parameters
        strategy_type = params.strategy_type
        
        if strategy_type == ExecutionStrategyType.ICEBERG:
            # Use smaller display sizes
            params.iceberg_display_size_percentage = max(0.03, params.iceberg_display_size_percentage * 0.6)
            # Increase variance to avoid pattern detection
            params.iceberg_variance_percentage = min(0.5, params.iceberg_variance_percentage * 1.5)
        
        elif strategy_type == ExecutionStrategyType.ADAPTIVE:
            # Adjust to be more passive
            params.adaptive_initial_rate = max(0.03, params.adaptive_initial_rate * 0.6)
            params.adaptive_max_rate = max(0.05, params.adaptive_max_rate * 0.6)
        
        return strategy
    
    @staticmethod
    def adjust_for_circuit_breaker(strategy: ExecutionStrategy) -> ExecutionStrategy:
        """Adjust strategy parameters after a circuit breaker event.
        
        Args:
            strategy: The original execution strategy
            
        Returns:
            Adjusted execution strategy for post-circuit breaker conditions
        """
        params = strategy.parameters
        strategy_type = params.strategy_type
        
        # For all strategies, make them extremely conservative initially
        # after a circuit breaker event
        
        if strategy_type == ExecutionStrategyType.TWAP:
            # Delay execution by increasing interval
            params.twap_slice_interval_seconds = int(params.twap_slice_interval_seconds * 3)
            # First slice should be smaller
            if hasattr(params, 'twap_custom_weights') and params.twap_custom_weights:
                # Adjust first slice to be smaller
                keys = list(params.twap_custom_weights.keys())
                if keys:
                    first_key = keys[0]
                    params.twap_custom_weights[first_key] = params.twap_custom_weights[first_key] * 0.3
        
        elif strategy_type == ExecutionStrategyType.PERCENTAGE_OF_VOLUME:
            # Minimal participation initially
            params.pov_target_percentage = max(0.01, params.pov_target_percentage * 0.2)
            params.pov_max_percentage = max(0.02, params.pov_max_percentage * 0.2)
            # Very high volume threshold
            params.pov_min_volume_threshold = int(params.pov_min_volume_threshold * 5)
        
        elif strategy_type == ExecutionStrategyType.ADAPTIVE:
            # Start extremely passive
            params.adaptive_initial_rate = max(0.01, params.adaptive_initial_rate * 0.2)
            params.adaptive_min_rate = max(0.005, params.adaptive_min_rate * 0.2)
            params.adaptive_max_rate = max(0.02, params.adaptive_max_rate * 0.2)
            # Maximum sensitivity to volatility
            params.adaptive_volatility_sensitivity = params.adaptive_volatility_sensitivity * 3
        
        return strategy


# Production configuration for emergency handling
def get_emergency_adjusted_strategy(strategy: ExecutionStrategy, 
                                   emergency_condition: str) -> ExecutionStrategy:
    """Get an execution strategy adjusted for specific emergency market conditions.
    
    Args:
        strategy: The original execution strategy
        emergency_condition: Type of emergency condition (high_volatility, low_liquidity, 
                            wide_spreads, circuit_breaker)
        
    Returns:
        Adjusted execution strategy for the emergency condition
    """
    handler = EmergencyMarketConditionHandler()
    
    if emergency_condition == "high_volatility":
        return handler.adjust_for_high_volatility(strategy)
    elif emergency_condition == "low_liquidity":
        return handler.adjust_for_low_liquidity(strategy)
    elif emergency_condition == "wide_spreads":
        return handler.adjust_for_wide_spreads(strategy)
    elif emergency_condition == "circuit_breaker":
        return handler.adjust_for_circuit_breaker(strategy)
    else:
        # Return original strategy if condition not recognized
        return strategy