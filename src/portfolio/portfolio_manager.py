import logging
import pandas as pd
import numpy as np
import time
import functools
from typing import Dict, List, Optional, Union, Any, Tuple, TypedDict, NamedTuple, Callable
from datetime import datetime, timedelta
from functools import lru_cache

# Import performance optimization components
from .cache_manager import CacheManager
from .batch_processor import BatchProcessor
from .performance_optimizer import PerformanceOptimizer

# Try to import risk components using relative imports (when part of the main package)
try:
    from ..risk.risk_metrics_calculator import RiskMetricsCalculator
    from ..risk.portfolio_risk_manager import PortfolioRiskManager
    RISK_MODULE_AVAILABLE = True
except ImportError:
    try:
        from risk.risk_metrics_calculator import RiskMetricsCalculator
        from risk.portfolio_risk_manager import PortfolioRiskManager
        RISK_MODULE_AVAILABLE = True
    except ImportError:
        RISK_MODULE_AVAILABLE = False
        # Define placeholder classes if risk module is not available
        class RiskMetricsCalculator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
        class PortfolioRiskManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

# Configure logger with appropriate levels
logger = logging.getLogger(__name__)

# Helper function for performance logging
def log_execution_time(func: Callable) -> Callable:
    """Decorator to log execution time of functions.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Type aliases for improved readability
Position = Dict[str, Any]
Positions = Dict[str, Position]

class HistoricalSnapshot(TypedDict):
    """Type definition for historical portfolio snapshots."""
    timestamp: datetime
    positions: Dict[str, Dict[str, Any]]
    portfolio_value: float

class MissingSymbolsInfo(TypedDict):
    """Type definition for missing symbols information."""
    count: int
    percentage: float
    symbols: List[str]

class PortfolioManager:
    """
    Portfolio Manager for tracking positions, performance, and portfolio state.

    This class provides functionality for:
    - Real-time portfolio state tracking
    - Position management
    - Performance tracking
    - Integration with risk management
    - Historical portfolio state storage
    """

    def __init__(self,
                 portfolio_id: str,
                 initial_capital: float = 100000.0,
                 risk_manager: Optional[PortfolioRiskManager] = None,
                 enable_optimization: bool = True):
        """
        Initialize the Portfolio Manager.

        Args:
            portfolio_id: Unique identifier for the portfolio
            initial_capital: Initial capital amount (must be positive)
            risk_manager: Optional risk manager for risk calculations
            enable_optimization: Whether to enable performance optimization
            
        Raises:
            ValueError: If portfolio_id is empty or initial_capital is not positive
        """
        # Input validation
        if not portfolio_id or not isinstance(portfolio_id, str):
            logger.error("Invalid portfolio_id provided")
            raise ValueError("portfolio_id must be a non-empty string")
            
        if initial_capital <= 0:
            logger.error(f"Invalid initial_capital: {initial_capital}")
            raise ValueError("initial_capital must be a positive number")
        
        self.portfolio_id = portfolio_id
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> position details
        self.portfolio_value = initial_capital
        self.risk_manager = risk_manager

        # Historical tracking
        self.portfolio_history = []  # List of (timestamp, portfolio_value, cash, positions)
        self.transaction_history = []  # List of transaction records

        # Performance tracking
        self.returns = []  # List of (timestamp, return_value)
        self.start_date = datetime.now()
        
        # Initialize NumPy arrays for efficient calculations
        self._returns_array = np.array([])
        self._portfolio_values_array = np.array([initial_capital])
        
        # Initialize performance optimization components
        self.enable_optimization = enable_optimization
        if enable_optimization:
            logger.info(f"Initializing performance optimization for portfolio {portfolio_id}")
            self.performance_optimizer = PerformanceOptimizer(
                enable_caching=True,
                enable_batching=True,
                enable_lazy_calc=True,
                enable_memory_opt=True,
                cache_ttl=60.0,  # 1 minute default cache TTL
                max_batch_size=100,
                max_batch_wait=0.1
            )
            # Initialize cache manager for direct access
            self.cache_manager = self.performance_optimizer.cache_manager
        else:
            self.performance_optimizer = None
            self.cache_manager = None

        logger.info(f"Portfolio Manager initialized: ID={portfolio_id}, Capital=${initial_capital:.2f}, Optimization={enable_optimization}")
        if risk_manager:
            logger.info(f"Risk manager attached to portfolio {portfolio_id}")
        else:
            logger.debug(f"No risk manager attached to portfolio {portfolio_id}")

    @log_execution_time
    def update_portfolio(self,
                        positions: Dict[str, Dict[str, Any]],
                        cash: float,
                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update the portfolio state with current positions and cash.

        Args:
            positions: Dictionary of positions {symbol: position_details}
            cash: Current cash balance (must be non-negative)
            timestamp: Timestamp for this update (default: now)

        Returns:
            Dict with updated portfolio state
            
        Raises:
            ValueError: If cash is negative or positions is invalid
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("update_portfolio"):
                return self._update_portfolio(positions, cash, timestamp)
        else:
            return self._update_portfolio(positions, cash, timestamp)
    
    def _update_portfolio(self,
                         positions: Dict[str, Dict[str, Any]],
                         cash: float,
                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Internal implementation of portfolio update."""
        start_time = time.time()
        
        # Input validation
        if cash < 0:
            logger.error(f"Invalid cash value: {cash}")
            raise ValueError("Cash balance cannot be negative")
            
        if not isinstance(positions, dict):
            logger.error(f"Invalid positions type: {type(positions)}")
            raise ValueError("Positions must be a dictionary")
            
        if timestamp is None:
            timestamp = datetime.now()

        logger.debug(f"Updating portfolio {self.portfolio_id} at {timestamp}")
        
        self.positions = positions
        self.cash = cash

        # Calculate portfolio value using NumPy for better performance
        if positions:
            # Extract market values as a NumPy array for faster summation
            market_values = np.array([position.get("market_value", 0.0) for position in positions.values()])
            positions_value = np.sum(market_values)
            logger.debug(f"Positions value: ${positions_value:.2f}")
        else:
            positions_value = 0.0
            logger.debug("No positions in portfolio")
            
        previous_value = self.portfolio_value
        self.portfolio_value = positions_value + cash

        # Calculate return since last update
        period_return = None
        if self.portfolio_history:
            last_value = self.portfolio_history[-1][1]  # Last portfolio value
            if last_value > 0:
                period_return = (self.portfolio_value / last_value) - 1
                self.returns.append((timestamp, period_return))
                
                # Update returns array for efficient calculations
                self._returns_array = np.append(self._returns_array, period_return)
                
                # Update portfolio values array
                self._portfolio_values_array = np.append(self._portfolio_values_array, self.portfolio_value)
                
                logger.info(f"Portfolio {self.portfolio_id} period return: {period_return:.4%}")

        # Store historical state
        self.portfolio_history.append((timestamp, self.portfolio_value, cash, positions.copy()))

        # Update risk manager if available
        if self.risk_manager:
            logger.debug("Updating risk manager")
            self.risk_manager.update_portfolio(positions, self.portfolio_value, timestamp)
            
        # Set last update time for cache invalidation
        self._last_update_time = time.time()
        
        # Log performance metrics
        execution_time = time.time() - start_time
        logger.debug(f"Portfolio update completed in {execution_time:.4f} seconds")
        logger.info(f"Portfolio {self.portfolio_id} value: ${self.portfolio_value:.2f} (Cash: ${cash:.2f})")

        return self.get_portfolio_state()

    @log_execution_time
    def execute_trade(self,
                      symbol: str,
                      quantity: float,
                      price: float,
                      timestamp: Optional[datetime] = None,
                      transaction_costs: float = 0.0,
                      sector: Optional[str] = None,
                      asset_class: Optional[str] = None) -> 'PortfolioManager':
        """
        Execute a trade and update the portfolio.

        Args:
            symbol: Symbol of the asset (must be non-empty string)
            quantity: Quantity to trade (positive for buy, negative for sell)
            price: Execution price (must be positive)
            timestamp: Timestamp for this trade (default: now)
            transaction_costs: Transaction costs for this trade (must be non-negative)
            sector: Sector of the asset (for new positions)
            asset_class: Asset class (e.g., 'equities', 'options', 'futures')

        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input parameters are invalid
            InsufficientFundsError: If there is not enough cash for a buy order
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("execute_trade"):
                return self._execute_trade(symbol, quantity, price, timestamp, transaction_costs, sector, asset_class)
        else:
            return self._execute_trade(symbol, quantity, price, timestamp, transaction_costs, sector, asset_class)
    
    def _execute_trade(self,
                      symbol: str,
                      quantity: float,
                      price: float,
                      timestamp: Optional[datetime] = None,
                      transaction_costs: float = 0.0,
                      sector: Optional[str] = None,
                      asset_class: Optional[str] = None) -> 'PortfolioManager':
        """Internal implementation of trade execution."""
        
        # Input validation
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            raise ValueError("Symbol must be a non-empty string")
            
        if price <= 0:
            logger.error(f"Invalid price: {price}")
            raise ValueError("Price must be positive")
            
        if transaction_costs < 0:
            logger.error(f"Invalid transaction costs: {transaction_costs}")
            raise ValueError("Transaction costs cannot be negative")
            
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"Executing trade: {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)} @ ${price:.2f}")
        
        # Calculate trade value
        trade_value = quantity * price
        total_cost = trade_value + transaction_costs

        # Check if we have enough cash for a buy
        if quantity > 0 and total_cost > self.cash:
            logger.error(f"Insufficient cash for trade: {symbol} {quantity} @ ${price:.2f}. Required: ${total_cost:.2f}, Available: ${self.cash:.2f}")
            raise ValueError(f"Insufficient funds for trade: required ${total_cost:.2f}, available ${self.cash:.2f}")
            
        logger.debug(f"Trade value: ${trade_value:.2f}, Transaction costs: ${transaction_costs:.2f}")
        logger.debug(f"Total cost: ${total_cost:.2f}")
        logger.debug(f"Cash before trade: ${self.cash:.2f}")
        
        start_time = time.time()

        # Update cash
        previous_cash = self.cash
        self.cash -= total_cost
        logger.debug(f"Cash after trade: ${self.cash:.2f} (change: ${self.cash - previous_cash:.2f})")

        # Update position
        position_action = ""  # For logging
        if symbol not in self.positions:
            if quantity > 0:  # Only create new position for buys
                self.positions[symbol] = {
                    "quantity": quantity,
                    "price": price,
                    "market_value": trade_value,
                    "cost_basis": price,
                    "sector": sector or "Unknown",
                    "asset_class": asset_class or "Unknown"
                }
                position_action = "New position created"
                logger.info(f"New position created: {symbol}, {quantity} shares @ ${price:.2f}")
        else:
            current_position = self.positions[symbol]
            current_quantity = current_position.get("quantity", 0)
            current_value = current_position.get("market_value", 0)
            previous_quantity = current_quantity

            # Calculate new position details
            new_quantity = current_quantity + quantity
            logger.debug(f"Position {symbol}: Quantity changing from {current_quantity} to {new_quantity}")

            if new_quantity <= 0:  # Sold entire position or short
                if new_quantity == 0:  # Sold entire position
                    del self.positions[symbol]
                    position_action = "Position closed"
                    logger.info(f"Position closed: {symbol}, sold {abs(quantity)} shares @ ${price:.2f}")
                else:  # Short position
                    self.positions[symbol] = {
                        "quantity": new_quantity,
                        "price": price,
                        "market_value": new_quantity * price,
                        "cost_basis": price,
                        "sector": current_position.get("sector", "Unknown"),
                        "asset_class": current_position.get("asset_class", asset_class or "Unknown")
                    }
                    position_action = "Short position created"
                    logger.info(f"Short position created: {symbol}, {abs(new_quantity)} shares @ ${price:.2f}")
            else:  # Still have a position
                # Calculate new cost basis (weighted average)
                old_cost_basis = current_position.get("cost_basis", 0)
                if current_quantity > 0 and quantity > 0:  # Adding to long position
                    new_cost_basis = ((current_quantity * old_cost_basis) +
                                     (quantity * price)) / new_quantity
                    position_action = "Position increased"
                    logger.info(f"Position increased: {symbol}, added {quantity} shares @ ${price:.2f}")
                else:  # Reducing position or other scenarios
                    new_cost_basis = old_cost_basis
                    position_action = "Position reduced"
                    logger.info(f"Position reduced: {symbol}, sold {abs(quantity)} shares @ ${price:.2f}")
                
                logger.debug(f"Cost basis changed from ${old_cost_basis:.2f} to ${new_cost_basis:.2f}")

                self.positions[symbol] = {
                    "quantity": new_quantity,
                    "price": price,  # Current market price
                    "market_value": new_quantity * price,
                    "cost_basis": new_cost_basis,
                    "sector": current_position.get("sector", "Unknown"),
                    "asset_class": current_position.get("asset_class", asset_class or "Unknown")
                }

        # Record transaction
        transaction = {
            "timestamp": timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "transaction_costs": transaction_costs,
            "total_cost": total_cost,
            "action": "BUY" if quantity > 0 else "SELL",
            "sector": self.positions.get(symbol, {}).get("sector", sector or "Unknown"),
            "asset_class": self.positions.get(symbol, {}).get("asset_class", asset_class or "Unknown")
        }
        self.transaction_history.append(transaction)
        logger.debug(f"Transaction recorded: {transaction['action']} {symbol} {abs(quantity)} @ ${price:.2f}")
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.debug(f"Trade execution completed in {execution_time:.4f} seconds")
        
        # Update portfolio state
        self.update_portfolio(self.positions, self.cash, timestamp)
        
        # Return self for method chaining
        return self

    @log_execution_time
    def update_prices(self,
                     price_data: Dict[str, float],
                     timestamp: Optional[datetime] = None) -> 'PortfolioManager':
        """
        Update market prices for positions.

        Args:
            price_data: Dictionary of {symbol: price} (prices must be positive)
            timestamp: Timestamp for this update (default: now)

        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If price_data is invalid or contains negative prices
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("update_prices"):
                return self._update_prices(price_data, timestamp)
        else:
            return self._update_prices(price_data, timestamp)
    
    def _update_prices(self,
                      price_data: Dict[str, float],
                      timestamp: Optional[datetime] = None) -> 'PortfolioManager':
        """Internal implementation of price updates."""
        
        # Input validation
        if not isinstance(price_data, dict):
            logger.error(f"Invalid price_data type: {type(price_data)}")
            raise ValueError("price_data must be a dictionary")
            
        # Check for negative prices
        negative_prices = {symbol: price for symbol, price in price_data.items() if price <= 0}
        if negative_prices:
            logger.error(f"Negative or zero prices found: {negative_prices}")
            raise ValueError(f"Prices must be positive. Found negative or zero prices: {negative_prices}")
        
        if timestamp is None:
            timestamp = datetime.now()
            
        start_time = time.time()
        logger.info(f"Updating prices for {len(price_data)} symbols")
        
        # Track which positions were updated
        updated_positions = []
        missing_positions = []
        
        # Update position prices and market values
        for symbol, price in price_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                old_price = position.get("price", 0)
                old_market_value = position.get("market_value", 0)
                quantity = position.get("quantity", 0)
                
                position["price"] = price
                position["market_value"] = quantity * price
                
                price_change_pct = ((price / old_price) - 1) * 100 if old_price > 0 else 0
                updated_positions.append(symbol)
                
                logger.debug(f"Updated {symbol}: Price ${old_price:.2f} -> ${price:.2f} ({price_change_pct:.2f}%)")
            else:
                missing_positions.append(symbol)
                
        if missing_positions:
            logger.warning(f"Prices provided for {len(missing_positions)} symbols not in portfolio: {', '.join(missing_positions[:5])}{' and more' if len(missing_positions) > 5 else ''}")
            
        if len(updated_positions) < len(self.positions):
            missing_updates = set(self.positions.keys()) - set(updated_positions)
            logger.warning(f"No price updates for {len(missing_updates)} positions: {', '.join(list(missing_updates)[:5])}{' and more' if len(missing_updates) > 5 else ''}")

        # Log execution time
        execution_time = time.time() - start_time
        logger.debug(f"Price update completed in {execution_time:.4f} seconds")
        
        # Update portfolio state
        self.update_portfolio(self.positions, self.cash, timestamp)
        
        # Return self for method chaining
        return self

    @log_execution_time
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current portfolio state with detailed portfolio metrics.

        Returns:
            Dict with portfolio state details including:
                - timestamp: Current time
                - portfolio_value: Total portfolio value
                - cash: Available cash
                - positions: Dictionary of all positions
                - position_count: Number of positions
                - equity: Portfolio value minus cash
                - cash_percentage: Percentage of portfolio in cash
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_portfolio_state"):
                return self._get_portfolio_state_with_cache()
        else:
            return self._get_portfolio_state_with_cache()
    
    def _get_portfolio_state_with_cache(self) -> Dict[str, Any]:
        """
        Internal implementation of portfolio state retrieval with caching.
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"portfolio_state_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached portfolio state for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._get_portfolio_state()
            self.cache_manager.set(cache_key, result, ttl=5.0)  # Cache for 5 seconds
            return result
        else:
            return self._get_portfolio_state()
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Internal implementation of portfolio state calculation."""
        logger.debug(f"Retrieving portfolio state for {self.portfolio_id}")
        
        # Calculate additional metrics
        equity = self.portfolio_value - self.cash
        cash_percentage = (self.cash / self.portfolio_value) * 100 if self.portfolio_value > 0 else 0
        
        # Log portfolio summary
        logger.info(f"Portfolio summary - Value: ${self.portfolio_value:.2f}, Cash: ${self.cash:.2f} ({cash_percentage:.2f}%), Positions: {len(self.positions)}")
        
        return {
            "timestamp": datetime.now(),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions": self.positions,
            "position_count": len(self.positions),
            "equity": equity,
            "cash_percentage": cash_percentage / 100  # Return as decimal for backward compatibility
        }

    def get_portfolio_value(self) -> float:
        """
        Get the current total portfolio value.

        Returns:
            Current portfolio value as a float
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_portfolio_value"):
                return self._get_portfolio_value()
        else:
            return self._get_portfolio_value()
    
    def _get_portfolio_value(self) -> float:
        """
        Internal implementation of portfolio value retrieval.
        
        Returns:
            Current portfolio value as a float
        """
        logger.debug(f"Retrieving portfolio value for {self.portfolio_id}")
        return self.portfolio_value
        
    def get_positions_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all positions in the portfolio.
        
        Returns:
            Dictionary of position summaries with symbol as key and position details as value
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_positions_summary"):
                return self._get_positions_summary_with_cache()
        else:
            return self._get_positions_summary_with_cache()
    
    def _get_positions_summary_with_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Internal implementation of positions summary retrieval with caching.
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"positions_summary_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached positions summary for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._get_positions_summary()
            self.cache_manager.set(cache_key, result, ttl=5.0)  # Cache for 5 seconds
            return result
        else:
            return self._get_positions_summary()
    
    def _get_positions_summary(self) -> Dict[str, Dict[str, Any]]:
        """Internal implementation of positions summary calculation."""
        logger.debug(f"Retrieving positions summary for {self.portfolio_id}")
        
        positions_summary = {}
        for symbol, position in self.positions.items():
            # Create a copy of the position to avoid modifying the original
            position_summary = position.copy()
            
            # Add any additional calculated fields if needed
            if 'market_value' in position and 'cost_basis' in position and position['cost_basis'] > 0:
                position_summary['unrealized_pl'] = position['market_value'] - position['cost_basis']
                position_summary['unrealized_pl_percent'] = (position_summary['unrealized_pl'] / position['cost_basis']) * 100
            
            # Add allocation percentage
            if 'market_value' in position and self.portfolio_value > 0:
                position_summary['allocation'] = (position['market_value'] / self.portfolio_value) * 100
            
            positions_summary[symbol] = position_summary
        
        logger.info(f"Retrieved summary for {len(positions_summary)} positions")
        return positions_summary

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific position.

        Args:
            symbol: Symbol of the position (must be a non-empty string)

        Returns:
            Position details or None if not found
            
        Raises:
            ValueError: If symbol is invalid
        """
        # Input validation
        if not isinstance(symbol, str) or not symbol.strip():
            logger.error(f"Invalid symbol: {symbol}")
            raise ValueError("Symbol must be a non-empty string")
        
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_position"):
                return self._get_position_with_cache(symbol)
        else:
            return self._get_position_with_cache(symbol)
    
    def _get_position_with_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Internal implementation of position retrieval with caching.
        
        Args:
            symbol: Symbol of the position
            
        Returns:
            Position details or None if not found
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"position_{self.portfolio_id}_{symbol}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached position data for {symbol}")
                return cached_result
            
            # If no valid cache, get position and store in cache
            position = self._get_position(symbol)
            if position is not None:
                # Make a copy to avoid modifying the original through the cache
                position_copy = position.copy()
                self.cache_manager.set(cache_key, position_copy, ttl=5.0)  # Cache for 5 seconds
                return position_copy
            return None
        else:
            return self._get_position(symbol)
    
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Internal implementation of position retrieval.
        
        Args:
            symbol: Symbol of the position
            
        Returns:
            Position details or None if not found
        """
        position = self.positions.get(symbol)
        
        if position:
            logger.debug(f"Retrieved position for {symbol}: Quantity: {position.get('quantity')}, Value: ${position.get('market_value', 0):.2f}")
        else:
            logger.debug(f"Position not found for symbol: {symbol}")
            
        return position

    @log_execution_time
    def get_historical_values(self) -> List[Tuple[datetime, float]]:
        """
        Get historical portfolio values over time.

        Returns:
            List of (timestamp, portfolio_value) tuples in chronological order
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_historical_values"):
                return self._get_historical_values_with_cache()
        else:
            return self._get_historical_values_with_cache()
    
    def _get_historical_values_with_cache(self) -> List[Tuple[datetime, float]]:
        """
        Internal implementation of historical values retrieval with caching.
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"historical_values_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached historical values for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._get_historical_values()
            # Use longer TTL for historical data since it changes less frequently
            self.cache_manager.set(cache_key, result, ttl=30.0)  # Cache for 30 seconds
            return result
        else:
            return self._get_historical_values()
    
    def _get_historical_values(self) -> List[Tuple[datetime, float]]:
        """Internal implementation of historical values retrieval."""
        logger.debug(f"Retrieving historical values for portfolio {self.portfolio_id}")
        
        historical_values = [(ts, val) for ts, val, _, _ in self.portfolio_history]
        
        if historical_values:
            start_date = historical_values[0][0]
            end_date = historical_values[-1][0]
            date_range = (end_date - start_date).days
            logger.info(f"Retrieved {len(historical_values)} historical values spanning {date_range} days")
        else:
            logger.warning(f"No historical values found for portfolio {self.portfolio_id}")
            
        return historical_values

    @log_execution_time
    def get_historical_returns(self) -> List[Tuple[datetime, float]]:
        """
        Get historical portfolio returns over time.

        Returns:
            List of (timestamp, return_value) tuples in chronological order,
            where return_value is the percentage change from the previous period
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_historical_returns"):
                return self._get_historical_returns_with_cache()
        else:
            return self._get_historical_returns_with_cache()
    
    def _get_historical_returns_with_cache(self) -> List[Tuple[datetime, float]]:
        """
        Internal implementation of historical returns retrieval with caching.
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"historical_returns_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached historical returns for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._get_historical_returns()
            # Use longer TTL for historical data since it changes less frequently
            self.cache_manager.set(cache_key, result, ttl=30.0)  # Cache for 30 seconds
            return result
        else:
            return self._get_historical_returns()
    
    def _get_historical_returns(self) -> List[Tuple[datetime, float]]:
        """Internal implementation of historical returns retrieval."""
        logger.debug(f"Retrieving historical returns for portfolio {self.portfolio_id}")
        
        if self.returns:
            start_date = self.returns[0][0]
            end_date = self.returns[-1][0]
            date_range = (end_date - start_date).days
            
            # Calculate some basic statistics
            return_values = [ret for _, ret in self.returns]
            avg_return = sum(return_values) / len(return_values) if return_values else 0
            max_return = max(return_values) if return_values else 0
            min_return = min(return_values) if return_values else 0
            
            logger.info(f"Retrieved {len(self.returns)} historical returns spanning {date_range} days")
            logger.debug(f"Return statistics - Avg: {avg_return:.4f}%, Max: {max_return:.4f}%, Min: {min_return:.4f}%")
        else:
            logger.warning(f"No historical returns found for portfolio {self.portfolio_id}")
            
        return self.returns

    @log_execution_time
    def get_transactions(self,
                        symbol: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get transaction history with optional filtering by symbol and date range.

        Args:
            symbol: Filter by symbol (optional, must be a non-empty string if provided)
            start_date: Filter by start date (optional, inclusive)
            end_date: Filter by end date (optional, inclusive)

        Returns:
            List of transaction records sorted by timestamp (newest first)
            
        Raises:
            ValueError: If symbol is provided but invalid, or if date range is invalid
        """
        # Input validation
        if symbol is not None and (not isinstance(symbol, str) or not symbol.strip()):
            logger.error(f"Invalid symbol for transaction filtering: {symbol}")
            raise ValueError("Symbol must be a non-empty string")
            
        if start_date is not None and end_date is not None and start_date > end_date:
            logger.error(f"Invalid date range: start_date {start_date} is after end_date {end_date}")
            raise ValueError("start_date must be before or equal to end_date")
        
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_transactions"):
                return self._get_transactions_with_cache(symbol, start_date, end_date)
        else:
            return self._get_transactions_with_cache(symbol, start_date, end_date)
    
    def _get_transactions_with_cache(self,
                                    symbol: Optional[str] = None,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Internal implementation of transaction retrieval with caching.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional, inclusive)
            end_date: Filter by end date (optional, inclusive)
            
        Returns:
            List of transaction records sorted by timestamp (newest first)
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Create a cache key based on the filter parameters
            cache_key = f"transactions_{self.portfolio_id}_{symbol}_{start_date}_{end_date}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached transaction data for filters: symbol={symbol}, start_date={start_date}, end_date={end_date}")
                return cached_result
            
            # If no valid cache, process transactions and store in cache
            result = self._get_transactions(symbol, start_date, end_date)
            # Use a shorter TTL for transaction data since it might change more frequently
            self.cache_manager.set(cache_key, result, ttl=15.0)  # Cache for 15 seconds
            return result
        else:
            return self._get_transactions(symbol, start_date, end_date)
    
    def _get_transactions(self,
                         symbol: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Internal implementation of transaction retrieval with filtering."""
        logger.debug(f"Retrieving transactions for portfolio {self.portfolio_id} with filters: "
                   f"symbol={symbol}, start_date={start_date}, end_date={end_date}")
        
        filtered_transactions = self.transaction_history
        original_count = len(filtered_transactions)

        # Apply filters - use batch processing if optimization is enabled and we have many transactions
        if self.enable_optimization and self.performance_optimizer and len(filtered_transactions) > 100:
            # Use batch processor for filtering large transaction sets
            batch_processor = self.performance_optimizer.get_batch_processor()
            
            # Define filter functions
            def symbol_filter(transaction):
                return transaction["symbol"] == symbol if symbol else True
                
            def start_date_filter(transaction):
                return transaction["timestamp"] >= start_date if start_date else True
                
            def end_date_filter(transaction):
                return transaction["timestamp"] <= end_date if end_date else True
            
            # Apply filters in batches
            for transaction in filtered_transactions:
                batch_processor.add_item(transaction)
            
            # Process with all filters
            filtered_transactions = batch_processor.process_items(
                lambda item: symbol_filter(item) and start_date_filter(item) and end_date_filter(item)
            )
            
            logger.debug(f"Batch processed {original_count} transactions -> {len(filtered_transactions)} after filtering")
            batch_processor.flush()
        else:
            # Traditional filtering for smaller sets
            if symbol:
                filtered_transactions = [t for t in filtered_transactions if t["symbol"] == symbol]
                logger.debug(f"Symbol filter applied: {original_count} -> {len(filtered_transactions)} transactions")

            if start_date:
                filtered_transactions = [t for t in filtered_transactions if t["timestamp"] >= start_date]
                logger.debug(f"Start date filter applied: {original_count} -> {len(filtered_transactions)} transactions")

            if end_date:
                filtered_transactions = [t for t in filtered_transactions if t["timestamp"] <= end_date]
                logger.debug(f"End date filter applied: {original_count} -> {len(filtered_transactions)} transactions")

        # Sort by timestamp (newest first)
        filtered_transactions = sorted(filtered_transactions, key=lambda t: t["timestamp"], reverse=True)
        
        if filtered_transactions:
            earliest = min(t["timestamp"] for t in filtered_transactions)
            latest = max(t["timestamp"] for t in filtered_transactions)
            logger.info(f"Retrieved {len(filtered_transactions)} transactions from {earliest} to {latest}")
        else:
            logger.info(f"No transactions found matching the specified criteria")
            
        return filtered_transactions

    @functools.lru_cache(maxsize=32)
    def _calculate_performance_metrics_cached(self, history_length: int, last_update_time: float) -> Dict[str, Any]:
        """
        Internal cached version of performance metrics calculation. This method uses LRU caching
        to avoid recalculating metrics when the portfolio hasn't changed.
        
        Args:
            history_length: Length of portfolio history (used for cache invalidation)
            last_update_time: Timestamp of last portfolio update (used for cache invalidation)
            
        Returns:
            Dict with performance metrics including:
                - total_return: Total return since inception
                - annualized_return: Annualized return
                - volatility: Annualized volatility
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - current_drawdown: Current drawdown
                
        Note:
            The cache is invalidated when either history_length or last_update_time changes.
        """
        start_time = time.time()
        logger.debug(f"Calculating performance metrics for portfolio {self.portfolio_id}")
        
        if not self.portfolio_history:
            logger.warning(f"No portfolio history available for {self.portfolio_id}")
            return {"error": "No portfolio history available"}

        # Calculate returns
        start_value = self.initial_capital
        current_value = self.portfolio_value
        total_return = (current_value / start_value) - 1
        
        logger.debug(f"Basic metrics - Start value: ${start_value:.2f}, Current value: ${current_value:.2f}, Total return: {total_return:.4f} ({total_return*100:.2f}%)")
        

        # Calculate time-weighted returns if we have history
        if len(self.returns) > 0:
            logger.debug(f"Calculating time-weighted returns from {len(self.returns)} data points")
            
            # Use NumPy arrays for efficient calculations
            # Convert returns to numpy array if not already
            if not hasattr(self, '_returns_array') or len(self._returns_array) != len(self.returns):
                self._returns_array = np.array([r[1] for r in self.returns], dtype=np.float64)
                logger.debug(f"Created new returns array with {len(self._returns_array)} elements")
            
            returns_array = self._returns_array

            # Calculate metrics using vectorized operations
            # Assuming daily returns (252 trading days per year)
            trading_days_per_year = 252
            annualized_return = np.mean(returns_array) * trading_days_per_year
            volatility = np.std(returns_array, ddof=1) * np.sqrt(trading_days_per_year)  # Use ddof=1 for sample std dev
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            logger.debug(f"Performance metrics - Annualized return: {annualized_return:.4f}, "
                       f"Volatility: {volatility:.4f}, Sharpe ratio: {sharpe_ratio:.4f}")

            # Calculate drawdown more efficiently using vectorized operations
            logger.debug("Calculating drawdown metrics")
            # Pre-allocate arrays for better performance
            cumulative_returns = np.empty_like(returns_array)
            np.add(returns_array, 1, out=cumulative_returns)
            np.cumprod(cumulative_returns, out=cumulative_returns)
            
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = np.divide(cumulative_returns, running_max) - 1
            max_drawdown = np.min(drawdown)
            current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
            
            logger.debug(f"Drawdown metrics - Max drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%), "
                       f"Current drawdown: {current_drawdown:.4f} ({current_drawdown*100:.2f}%)")
        else:
            logger.warning("No return data available for calculating performance metrics")
            annualized_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            current_drawdown = 0

        # Calculate days since inception
        days = (datetime.now() - self.start_date).days
        if days <= 0:
            days = 1  # Avoid division by zero

        # Calculate CAGR (Compound Annual Growth Rate)
        cagr = (current_value / start_value) ** (365 / days) - 1 if days > 0 else 0
        logger.debug(f"Time-based metrics - Days since inception: {days}, CAGR: {cagr:.4f} ({cagr*100:.2f}%)")
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.debug(f"Performance metrics calculation completed in {execution_time:.4f} seconds")

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "days": days,
            "cagr": cagr
        }
        
    @log_execution_time
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic performance metrics with caching for better performance.
        This method uses an LRU-cached internal method to avoid recalculating metrics
        when the portfolio hasn't changed.

        Returns:
            Dict with performance metrics including:
                - total_return: Total return since inception
                - annualized_return: Annualized return
                - volatility: Annualized volatility
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - current_drawdown: Current drawdown
                - days: Days since inception
                - cagr: Compound Annual Growth Rate
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("calculate_performance_metrics"):
                return self._calculate_performance_metrics_with_cache()
        else:
            return self._calculate_performance_metrics_with_cache()
    
    def _calculate_performance_metrics_with_cache(self) -> Dict[str, Any]:
        """
        Internal implementation of performance metrics calculation with caching.
        """
        # Use our caching system if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"performance_metrics_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached performance metrics for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._calculate_performance_metrics()
                
            # Cache the result - performance metrics can be cached longer since they don't change as frequently
            self.cache_manager.set(cache_key, result, ttl=60.0)  # Cache for 60 seconds
            return result
        else:
            return self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Internal implementation of performance metrics calculation.
        """
        logger.debug(f"Calculating performance metrics for portfolio {self.portfolio_id}")
        
        # Use parameters that will invalidate the cache when portfolio changes
        history_length = len(self.portfolio_history)
        last_update_time = getattr(self, '_last_update_time', time.time())
        
        # Check if we have enough data
        if history_length == 0:
            logger.warning(f"No portfolio history available for {self.portfolio_id}, metrics will be limited")
            
        # Still use the functools.lru_cache as a second layer of caching
        metrics = self._calculate_performance_metrics_cached(history_length, last_update_time)
        
        # Log key metrics
        logger.info(f"Performance metrics - Total return: {metrics.get('total_return', 0)*100:.2f}%, "
                  f"CAGR: {metrics.get('cagr', 0)*100:.2f}%, "
                  f"Max drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        
        return metrics

    @log_execution_time
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get risk metrics from the risk manager if available.
        This method delegates to the attached risk manager to calculate various risk metrics.

        Returns:
            Dict with risk metrics (e.g., VaR, CVaR, beta, etc.) or empty dict if no risk manager is attached
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("get_risk_metrics"):
                return self._get_risk_metrics_with_cache()
        else:
            return self._get_risk_metrics_with_cache()
    
    def _get_risk_metrics_with_cache(self) -> Dict[str, Any]:
        """
        Internal implementation of risk metrics retrieval with caching.
        """
        # Use caching if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Check if we have a cached result that's still valid
            cache_key = f"risk_metrics_{self.portfolio_id}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached risk metrics for {self.portfolio_id}")
                return cached_result
            
            # If no valid cache, calculate and store in cache
            result = self._get_risk_metrics()
                
            # Cache the result - risk metrics can be cached for a moderate time
            self.cache_manager.set(cache_key, result, ttl=30.0)  # Cache for 30 seconds
            return result
        else:
            return self._get_risk_metrics()
    
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """
        Internal implementation of risk metrics retrieval.
        """
        logger.debug(f"Getting risk metrics for portfolio {self.portfolio_id}")
        
        if self.risk_manager:
            logger.debug(f"Delegating risk calculation to risk manager: {self.risk_manager.__class__.__name__}")
            metrics = self.risk_manager.calculate_risk_metrics()
            
            # Log key risk metrics if available
            if metrics:
                log_items = []
                for key in ['var_95', 'cvar_95', 'beta', 'volatility']:
                    if key in metrics:
                        log_items.append(f"{key}: {metrics[key]:.4f}")
                        
                if log_items:
                    logger.info(f"Risk metrics - {', '.join(log_items)}")
            return metrics
            
        logger.warning(f"No risk manager attached to portfolio {self.portfolio_id}, returning empty risk metrics")
        return {}
        
    @log_execution_time
    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_price: float = None,
                              volatility: float = None,
                              risk_multiplier: float = 1.0,
                              sector: str = None,
                              asset_class: str = None,
                              method: str = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size for a new trade using the risk manager.
        This method delegates to the attached risk manager to determine appropriate position size
        based on various risk parameters and portfolio constraints.

        Args:
            symbol: Asset symbol (must be non-empty string)
            entry_price: Entry price for the position (must be positive)
            stop_price: Stop loss price (required for risk-based sizing)
            volatility: Asset volatility (required for volatility-based sizing)
            risk_multiplier: Multiplier to adjust risk (1.0 = 100% of normal risk)
            sector: Asset sector (for sector exposure limits)
            asset_class: Asset class (e.g., 'equities', 'options', 'futures')
            method: Position sizing method (overrides default)

        Returns:
            Tuple of (position_size, details dictionary with calculation information)
            
        Raises:
            ValueError: If symbol is empty or entry_price is not positive
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("calculate_position_size"):
                return self._calculate_position_size(symbol, entry_price, stop_price, volatility,
                                                  risk_multiplier, sector, asset_class, method)
        else:
            return self._calculate_position_size(symbol, entry_price, stop_price, volatility,
                                              risk_multiplier, sector, asset_class, method)
    
    def _calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_price: float = None,
                               volatility: float = None,
                               risk_multiplier: float = 1.0,
                               sector: str = None,
                               asset_class: str = None,
                               method: str = None) -> Tuple[float, Dict[str, Any]]:
        """
        Internal implementation of position size calculation.
        """
        # Input validation
        if not symbol or not isinstance(symbol, str):
            error_msg = f"Symbol must be a non-empty string, got {type(symbol).__name__}: {symbol}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not entry_price or entry_price <= 0:
            error_msg = f"Entry price must be positive, got {entry_price}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if risk_multiplier <= 0:
            error_msg = f"Risk multiplier must be positive, got {risk_multiplier}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Calculating position size for {symbol} at ${entry_price:.2f}")
        logger.debug(f"Parameters - Stop: ${stop_price:.2f if stop_price else None}, "
                   f"Volatility: {volatility:.4f if volatility else None}, "
                   f"Risk multiplier: {risk_multiplier:.2f}")
        
        if not self.risk_manager:
            error_msg = "No risk manager available for position sizing"
            logger.warning(error_msg)
            return 0, {"error": error_msg, "portfolio_id": self.portfolio_id}
            
        # Call the risk manager's calculate_position_size method with asset_class
        logger.debug(f"Delegating position size calculation to risk manager: {self.risk_manager.__class__.__name__}")
        position_size, details = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            volatility=volatility,
            risk_multiplier=risk_multiplier,
            sector=sector,
            asset_class=asset_class,
            method=method
        )
        
        # Log the results
        if position_size > 0:
            logger.info(f"Calculated position size for {symbol}: {position_size:.2f} units (${position_size * entry_price:.2f})")
            if 'risk_amount' in details:
                logger.debug(f"Risk amount: ${details['risk_amount']:.2f}, {details.get('risk_percent', 0)*100:.2f}% of portfolio")
        else:
            logger.warning(f"Zero position size calculated for {symbol} - Reason: {details.get('reason', 'Unknown')}")
            
        return position_size, details
        
    def set_risk_manager(self, risk_manager: PortfolioRiskManager) -> 'PortfolioManager':
        """
        Set or update the risk manager for this portfolio.
        
        Args:
            risk_manager: The risk manager instance to use
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If risk_manager is None or not a PortfolioRiskManager instance
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("set_risk_manager"):
                return self._set_risk_manager(risk_manager)
        else:
            return self._set_risk_manager(risk_manager)
    
    def _set_risk_manager(self, risk_manager: PortfolioRiskManager) -> 'PortfolioManager':
        """
        Internal implementation of setting the risk manager.
        """
        if risk_manager is None:
            error_msg = "Risk manager cannot be None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not isinstance(risk_manager, PortfolioRiskManager):
            error_msg = f"Risk manager must be a PortfolioRiskManager instance, got {type(risk_manager).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Clear any cached risk metrics when changing risk manager
        if self.enable_optimization and self.cache_manager and hasattr(self, 'risk_manager') and self.risk_manager:
            # Clear any cached risk metrics
            cache_keys = ["get_risk_metrics"]
            for key in cache_keys:
                self.cache_manager.invalidate_cache(key, portfolio_id=self.portfolio_id)
            logger.debug(f"Cleared cached risk metrics for portfolio {self.portfolio_id}")
            
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for portfolio {self.portfolio_id}: {risk_manager.__class__.__name__}")
        
        return self

    @log_execution_time
    def reset(self, initial_capital: Optional[float] = None) -> 'PortfolioManager':
        """
        Reset the portfolio to initial state, clearing all positions and transaction history.
        
        Args:
            initial_capital: New initial capital (optional, defaults to original value)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If initial_capital is provided but not positive
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("reset"):
                return self._reset(initial_capital)
        else:
            return self._reset(initial_capital)
    
    def _reset(self, initial_capital: Optional[float] = None) -> 'PortfolioManager':
        """
        Internal implementation of portfolio reset.
        """
        logger.info(f"Resetting portfolio {self.portfolio_id}")
        
        if initial_capital is not None:
            if initial_capital <= 0:
                error_msg = f"Initial capital must be positive, got {initial_capital}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"Changing initial capital from {self.initial_capital} to {initial_capital}")
            self.initial_capital = initial_capital
            
        # Reset portfolio state
        self.cash = self.initial_capital
        old_positions_count = len(self.positions)
        self.positions = {}
        self.portfolio_value = self.initial_capital
        
        # Reset history
        self.portfolio_history = []
        self.transactions = []
        
        # Reset arrays
        self._returns_array = np.array([])
        self._portfolio_values_array = np.array([])
        
        # Reset other tracking variables
        self.returns = []
        self.start_date = datetime.now()
        
        # Reset cached data
        if hasattr(self, '_historical_snapshots'):
            self._historical_snapshots = []
            
        # Set last update time for cache invalidation
        self._last_update_time = time.time()
        
        # Clear cache if optimization is enabled
        if self.enable_optimization and self.cache_manager:
            # Clear all cache entries related to this portfolio
            cache_prefix = f"{self.portfolio_id}_"
            self.cache_manager.clear_pattern(cache_prefix)
            logger.debug(f"Cleared cache entries for portfolio {self.portfolio_id}")
        
        # Reset risk manager if available
        if self.risk_manager:
            logger.debug(f"Resetting attached risk manager: {self.risk_manager.__class__.__name__}")
            self.risk_manager.reset()
            
        logger.info(f"Portfolio reset complete - Cleared {old_positions_count} positions, starting with ${self.initial_capital:.2f}")
        return self
        
    @log_execution_time
    def add_historical_prices(self, price_data: Dict[str, float], timestamp: datetime, max_history_size: Optional[int] = None, missing_threshold: float = 0.2, handle_missing: str = 'warn') -> Optional[HistoricalSnapshot]:
        """
        Add historical price data for assets and update the risk manager.
        
        This method is used to provide historical price data to the risk manager
        for calculating risk metrics based on historical returns. It creates a
        snapshot of the portfolio using the provided historical prices and
        updates the risk manager with this data.
        
        The risk manager can then use this historical data to calculate metrics
        such as Value at Risk (VaR), volatility, and other risk measures that
        require historical price information.
        
        Args:
            price_data: Dictionary mapping asset symbols to their historical prices.
                       Only assets that exist in the current portfolio and have
                       prices in this dictionary will be included in the update.
            timestamp: The historical timestamp associated with the provided prices.
                       Used for chronological ordering of data points.
            max_history_size: Maximum number of historical data points to retain.
                               If provided, older data points will be removed to prevent
                               memory issues with large datasets. Defaults to None (no limit).
            missing_threshold: Maximum acceptable percentage of missing symbols (0.0-1.0).
                               If the percentage of missing symbols exceeds this threshold,
                               a warning will be logged. Defaults to 0.2 (20%).
            handle_missing: How to handle missing price data. Options:
                           'warn' - Log a warning and continue (default)
                           'error' - Raise an exception
                           'skip' - Skip missing symbols silently
                           'interpolate' - Use last known price for missing symbols
        
        Returns:
            A dictionary containing the historical snapshot with positions and portfolio value.
            Returns None if no risk manager is available.
             
        Examples:
            >>> # Add historical prices for a specific date
            >>> historical_date = datetime(2023, 1, 1)
            >>> portfolio.add_historical_prices({"AAPL": 145.0, "MSFT": 240.0}, historical_date)
            
            >>> # Add historical prices for a week ago
            >>> last_week = datetime.now() - timedelta(days=7)
            >>> portfolio.add_historical_prices({"AAPL": 140.0, "MSFT": 240.0}, last_week)
            >>> # Add historical prices with a limit on history size
            >>> portfolio.add_historical_prices({"AAPL": 140.0, "MSFT": 240.0}, last_week, max_history_size=100)
        
        Note:
            This method requires a risk manager to be set. If no risk manager is available,
            the method will log a debug message and return without making any changes.
            
        Raises:
            ValueError: If timestamp is in the future or if max_history_size is negative.
            TypeError: If price_data contains non-numeric values.
            KeyError: If handle_missing='error' and symbols are missing price data.
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            with self.performance_optimizer.monitor_performance("add_historical_prices"):
                return self._add_historical_prices(price_data, timestamp, max_history_size, missing_threshold, handle_missing)
        else:
            return self._add_historical_prices(price_data, timestamp, max_history_size, missing_threshold, handle_missing)
    
    def _add_historical_prices(self, price_data: Dict[str, float], timestamp: datetime, max_history_size: Optional[int] = None, missing_threshold: float = 0.2, handle_missing: str = 'warn') -> Optional[HistoricalSnapshot]:
        """
        Internal implementation of adding historical prices.
        """
        # Validate input parameters
        if not isinstance(price_data, dict):
            raise TypeError("price_data must be a dictionary mapping symbols to prices")
            
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime object")
            
        # Check if timestamp is in the future
        if timestamp > datetime.now():
            raise ValueError("timestamp cannot be in the future")
            
        # Validate max_history_size
        if max_history_size is not None:
            if not isinstance(max_history_size, int):
                raise TypeError("max_history_size must be an integer")
            if max_history_size <= 0:
                raise ValueError("max_history_size must be positive")
                
        # Validate price data values
        for symbol, price in price_data.items():
            if not isinstance(price, (int, float)):
                raise TypeError(f"Price for {symbol} must be a number, got {type(price).__name__}")
            if price <= 0:
                raise ValueError(f"Price for {symbol} must be positive, got {price}")
        
        logger.debug(f"Adding historical prices for portfolio {self.portfolio_id} at {timestamp}")
        logger.debug(f"Received price data for {len(price_data)} symbols")
        
        # Use batch processing for large datasets if optimization is enabled
        if self.enable_optimization and self.performance_optimizer and len(price_data) > 50:
            logger.debug(f"Using batch processing for {len(price_data)} price data points")
            batch_processor = self.performance_optimizer.get_batch_processor()
            
            # Process position snapshots in batches
            positions_snapshot, portfolio_value, missing_symbols, included_symbols = self._process_historical_prices_batch(
                price_data, batch_processor)
        else:
            # Standard processing for smaller datasets
            positions_snapshot, portfolio_value, missing_symbols, included_symbols = self._process_historical_prices_standard(
                price_data)
        
        logger.debug(f"Created snapshot with {len(included_symbols)} positions, portfolio value: ${portfolio_value:.2f}")
        if included_symbols:
            logger.debug(f"Included symbols: {', '.join(included_symbols[:5])}{' and more' if len(included_symbols) > 5 else ''}")
        
        # Check for missing symbols threshold
        if missing_symbols and len(self.positions) > 0:
            missing_count = len(missing_symbols)
            total_positions = len(self.positions)
            missing_percentage = missing_count / total_positions
            
            missing_info: MissingSymbolsInfo = {
                "count": missing_count,
                "percentage": missing_percentage,
                "symbols": missing_symbols
            }
            
            # Handle missing symbols based on the handle_missing parameter
            if handle_missing == 'error' and missing_count > 0:
                raise KeyError(f"Missing historical prices for {missing_count} symbols: {', '.join(missing_symbols)}")
            elif handle_missing == 'warn' and missing_percentage > missing_threshold:
                logger.warning(
                    f"Missing historical prices for {missing_count}/{total_positions} symbols "
                    f"({missing_percentage:.1%}) at {timestamp}, exceeding threshold of {missing_threshold:.1%}"
                )
                logger.debug(f"Missing symbols: {', '.join(missing_symbols)}")
            elif handle_missing == 'interpolate' and missing_count > 0:
                # Use last known prices for missing symbols
                for symbol in missing_symbols:
                    if symbol in self.positions and hasattr(self, '_historical_snapshots') and self._historical_snapshots:
                        # Try to find the most recent price for this symbol
                        for snapshot in reversed(self._historical_snapshots):
                            if symbol in snapshot['positions'] and 'price' in snapshot['positions'][symbol]:
                                last_price = snapshot['positions'][symbol]['price']
                                quantity = self.positions[symbol].get("quantity", 0)
                                market_value = quantity * last_price
                                
                                positions_snapshot[symbol] = {
                                    **self.positions[symbol],
                                    "price": last_price,
                                    "market_value": market_value
                                }
                                
                                portfolio_value += market_value
                                logger.debug(f"Interpolated price for {symbol} using last known price: {last_price}")
                                break
            # For 'skip' option, we do nothing and continue with available data
        
        # Update risk manager if available
        if self.risk_manager:
            logger.debug(f"Updating risk manager with historical data at {timestamp}")
            self.risk_manager.update_portfolio(positions_snapshot, portfolio_value, timestamp)
        else:
            logger.warning(f"No risk manager available for historical price updates for portfolio {self.portfolio_id}")
            return None
            
        # Implement caching for frequently calculated metrics
        if not hasattr(self, '_historical_snapshots'):
            self._historical_snapshots: List[HistoricalSnapshot] = []
        
        # Store the snapshot for potential future use
        snapshot: HistoricalSnapshot = {
            "timestamp": timestamp,
            "positions": positions_snapshot,
            "portfolio_value": portfolio_value
        }
        
        self._historical_snapshots.append(snapshot)
        
        # Limit history size if specified
        if max_history_size and len(self._historical_snapshots) > max_history_size:
            # Remove oldest entries to maintain the size limit
            self._historical_snapshots = self._historical_snapshots[-max_history_size:]
            
        # Store portfolio values in NumPy array for efficient calculations
        if not hasattr(self, '_portfolio_values_array'):
            self._portfolio_values_array = np.array([s["portfolio_value"] for s in self._historical_snapshots])
        else:
            # Pre-allocate array if needed for better performance
            if len(self._portfolio_values_array) >= max_history_size and max_history_size:
                # Shift array left and add new value at the end
                self._portfolio_values_array = np.roll(self._portfolio_values_array, -1)
                self._portfolio_values_array[-1] = portfolio_value
            else:
                # Append to array
                self._portfolio_values_array = np.append(self._portfolio_values_array, portfolio_value)
                if max_history_size and len(self._portfolio_values_array) > max_history_size:
                    self._portfolio_values_array = self._portfolio_values_array[-max_history_size:]
        
        # Log summary of the operation
        logger.info(f"Added historical prices for {len(price_data)} assets at {timestamp}")
        
        if max_history_size:
            logger.debug(f"Historical snapshots: {len(self._historical_snapshots)}/{max_history_size} (max size)")
        else:
            logger.debug(f"Historical snapshots: {len(self._historical_snapshots)} (no size limit)")
        
        # Return both the snapshot and self for flexible usage
        return snapshot
        
    def _process_historical_prices_standard(self, price_data: Dict[str, float]) -> Tuple[Positions, float, List[str], List[str]]:
        """
        Process historical prices using standard approach.
        """
        positions_snapshot: Positions = {}
        portfolio_value: float = self.cash
        missing_symbols: List[str] = []
        included_symbols: List[str] = []
        
        # Create a snapshot of positions with historical prices
        for symbol, position in self.positions.items():
            if symbol in price_data:
                price = price_data[symbol]
                quantity = position.get("quantity", 0)
                market_value = quantity * price
                
                positions_snapshot[symbol] = {
                    **position,
                    "price": price,
                    "market_value": market_value
                }
                
                portfolio_value += market_value
                included_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)
                logger.debug(f"No historical price data available for {symbol}")
                
        return positions_snapshot, portfolio_value, missing_symbols, included_symbols
    
    def _process_historical_prices_batch(self, price_data: Dict[str, float], batch_processor) -> Tuple[Positions, float, List[str], List[str]]:
        """
        Process historical prices using batch processing for better performance.
        """
        positions_snapshot: Positions = {}
        portfolio_value: float = self.cash
        missing_symbols: List[str] = []
        included_symbols: List[str] = []
        
        # Convert positions to list for batch processing
        position_items = list(self.positions.items())
        
        # Process in batches
        batch_size = 50  # Adjust based on performance testing
        for i in range(0, len(position_items), batch_size):
            batch = position_items[i:i+batch_size]
            
            # Process this batch
            batch_results = batch_processor.process_batch(
                batch,
                lambda item: self._process_position_price(item, price_data)
            )
            
            # Combine results
            for result in batch_results:
                if result['included']:
                    symbol = result['symbol']
                    positions_snapshot[symbol] = result['position_data']
                    portfolio_value += result['market_value']
                    included_symbols.append(symbol)
                else:
                    missing_symbols.append(result['symbol'])
        
        return positions_snapshot, portfolio_value, missing_symbols, included_symbols
    
    def _process_position_price(self, position_item, price_data):
        """
        Process a single position for historical price update.
        Used by the batch processor.
        """
        symbol, position = position_item
        
        if symbol in price_data:
            price = price_data[symbol]
            quantity = position.get("quantity", 0)
            market_value = quantity * price
            
            position_data = {
                **position,
                "price": price,
                "market_value": market_value
            }
            
            return {
                'symbol': symbol,
                'included': True,
                'position_data': position_data,
                'market_value': market_value
            }
        else:
            return {
                'symbol': symbol,
                'included': False
            }
