import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from .position_sizer import PositionSizer
from .stop_loss_manager import StopLossManager, StopLossType
from .portfolio_risk_manager import PortfolioRiskManager
from .risk_metrics_calculator import RiskMetricsCalculator
from .circuit_breaker import CircuitBreakerManager, AccountCircuitBreaker, MarketWideCircuitBreaker
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class AdvancedRiskManager(RiskManager):
    """
    Advanced Risk Manager that integrates all risk management components.

    This class provides a unified interface for:
    - Position sizing
    - Stop-loss management
    - Portfolio risk controls
    - Circuit breakers

    It serves as the main entry point for the risk management system.
    """

    def __init__(self,
                 position_sizer: PositionSizer = None,
                 stop_loss_manager: StopLossManager = None,
                 portfolio_risk_manager: PortfolioRiskManager = None,
                 risk_metrics_calculator: RiskMetricsCalculator = None,
                 circuit_breaker_manager: CircuitBreakerManager = None,
                 initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 max_portfolio_var_percent: float = 0.02,
                 max_drawdown_percent: float = 0.15,
                 max_sector_exposure: float = 0.25,
                 max_asset_exposure: float = 0.10,
                 max_correlation_exposure: float = 0.40,
                 default_stop_loss_percent: float = 0.02,
                 default_trailing_percent: float = 0.03,
                 default_atr_multiplier: float = 3.0,
                 default_time_stop_days: int = 10,
                 default_profit_target_ratio: float = 2.0,
                 account_soft_loss_limit_percent: float = 0.05,
                 account_hard_loss_limit_percent: float = 0.10,
                 market_circuit_breaker_percent: float = 0.07,
                 position_sizing_method: str = "risk_based",
                 max_signal_strength: float = 1.0,
                 min_signal_strength: float = 0.1,
                 risk_adjustment_factor: float = 1.0,
                 enable_signal_filtering: bool = True):
        """
        Initialize the advanced risk manager.

        Args:
            initial_capital: Initial capital for the trading account
            risk_per_trade: Risk per trade as a percentage of capital
            max_portfolio_var_percent: Maximum portfolio Value at Risk as percentage
            max_drawdown_percent: Maximum allowable drawdown percentage
            max_sector_exposure: Maximum exposure to any single sector
            max_asset_exposure: Maximum exposure to any single asset
            max_correlation_exposure: Maximum exposure to correlated assets
            default_stop_loss_percent: Default fixed stop loss percentage
            default_trailing_percent: Default trailing stop percentage
            default_atr_multiplier: Default ATR multiplier for volatility-based stops
            default_time_stop_days: Default number of days for time-based exits
            default_profit_target_ratio: Default profit target as a ratio of risk
            account_soft_loss_limit_percent: Soft loss limit for account circuit breaker
            account_hard_loss_limit_percent: Hard loss limit for account circuit breaker
            market_circuit_breaker_percent: Market circuit breaker trigger percentage
            position_sizing_method: Default position sizing method
            max_signal_strength: Maximum allowed signal strength
            min_signal_strength: Minimum signal strength to consider valid
            risk_adjustment_factor: Factor to adjust signal strength
            enable_signal_filtering: Whether to enable signal filtering
        """
        # Initialize the base RiskManager
        super().__init__(
            max_signal_strength=max_signal_strength,
            min_signal_strength=min_signal_strength,
            risk_adjustment_factor=risk_adjustment_factor,
            enable_signal_filtering=enable_signal_filtering
        )
        
        # Initialize components
        if position_sizer:
            self.position_sizer = position_sizer
        else:
            self.position_sizer = PositionSizer(
                risk_per_trade=risk_per_trade,
                max_position_percent=max_asset_exposure,
                sizing_method=position_sizing_method
            )

        if stop_loss_manager:
            self.stop_loss_manager = stop_loss_manager
        else:
            self.stop_loss_manager = StopLossManager(
                default_stop_loss_percent=default_stop_loss_percent,
                default_trailing_percent=default_trailing_percent,
                default_atr_multiplier=default_atr_multiplier,
                default_time_stop_days=default_time_stop_days,
                default_profit_target_ratio=default_profit_target_ratio
            )

        if portfolio_risk_manager:
            self.portfolio_risk_manager = portfolio_risk_manager
        else:
            self.portfolio_risk_manager = PortfolioRiskManager(
                max_portfolio_var_percent=max_portfolio_var_percent,
                max_drawdown_percent=max_drawdown_percent,
                max_sector_allocation=max_sector_exposure,
                max_position_size=max_asset_exposure,
                max_correlation_exposure=max_correlation_exposure,
                max_history_size=252
            )

        if risk_metrics_calculator:
            self.risk_metrics_calculator = risk_metrics_calculator
        else:
            self.risk_metrics_calculator = RiskMetricsCalculator(
                confidence_level=0.95  # Default 95% confidence level for VaR
            )

        if circuit_breaker_manager:
            self.circuit_breaker_manager = circuit_breaker_manager
        else:
            self.circuit_breaker_manager = CircuitBreakerManager(circuit_breakers=[], emergency_handlers=[])

        # Add default circuit breakers
        account_circuit_breaker = AccountCircuitBreaker(
            account_id="main",
            daily_loss_limit_percent=account_soft_loss_limit_percent,
            weekly_loss_limit_percent=account_hard_loss_limit_percent,
            max_drawdown_percent=account_hard_loss_limit_percent * 1.5,
            initial_balance=initial_capital
        )

        market_circuit_breaker = MarketWideCircuitBreaker(
            market="global",
            level_1_percent=market_circuit_breaker_percent,
            level_2_percent=market_circuit_breaker_percent * 1.5,
            level_3_percent=market_circuit_breaker_percent * 2.0,
            level_1_duration_minutes=60,
            level_2_duration_minutes=120,
            level_3_duration_minutes=180
        )

        self.circuit_breaker_manager.add_circuit_breaker(account_circuit_breaker)
        self.circuit_breaker_manager.add_circuit_breaker(market_circuit_breaker)

        # Start circuit breaker check thread
        self.circuit_breaker_manager.start()

        # State tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = {}
        self.positions = {}
        self.risk_per_trade = risk_per_trade
        self.position_sizing_method = position_sizing_method

        logger.info("Initialized AdvancedRiskManager")

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
        Calculate position size for a new trade, integrating all risk constraints.

        Args:
            symbol: Asset symbol
            entry_price: Entry price for the position
            stop_price: Stop loss price (required for risk-based sizing)
            volatility: Asset volatility (required for volatility-based sizing)
            risk_multiplier: Multiplier to adjust risk (1.0 = 100% of normal risk)
            sector: Asset sector (for sector exposure limits)
            asset_class: Asset class (e.g., 'equities', 'options', 'futures')
            method: Position sizing method (overrides default)

        Returns:
            Tuple of (position_size, details)
        """
        if method is None:
            method = self.position_sizing_method

        # Calculate base position size using position sizer
        if method == "risk_based" and stop_price is not None:
            base_size, details = self.position_sizer.calculate_risk_based_size(
                capital=self.current_capital,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_per_trade=self.risk_per_trade * risk_multiplier
            )
        elif method == "volatility_based" and volatility is not None:
            base_size, details = self.position_sizer.calculate_volatility_based_size(
                capital=self.current_capital,
                entry_price=entry_price,
                volatility=volatility,
                risk_per_trade=self.risk_per_trade * risk_multiplier
            )
        else:  # Default to fixed percentage
            base_size, details = self.position_sizer.calculate_fixed_percent_size(
                capital=self.current_capital,
                entry_price=entry_price,
                position_percent=self.risk_per_trade * risk_multiplier
            )

        # Apply portfolio-level constraints
        max_size_portfolio = self.portfolio_risk_manager.calculate_max_position_size(
            symbol=symbol,
            price=entry_price,
            sector=sector
        )

        # If we have volatility data, apply VaR-adjusted constraints
        if volatility is not None:
            correlation = {}  # In a real system, this would be populated
            max_size_var = self.portfolio_risk_manager.calculate_var_adjusted_position_size(
                symbol=symbol,
                price=entry_price,
                volatility=volatility,
                correlation=correlation
            )
            max_size_portfolio = min(max_size_portfolio, max_size_var)

        # Apply drawdown-based adjustment
        max_size_drawdown = self.portfolio_risk_manager.calculate_drawdown_adjusted_position_size(
            base_position_size=base_size
        )
        
        # Apply asset class-specific position limits if available
        max_size_asset_class = float('inf')
        if asset_class and hasattr(self, 'position_limits_by_asset') and asset_class in self.position_limits_by_asset:
            limits = self.position_limits_by_asset[asset_class]
            max_percentage = limits.get('max_position_percentage', self.position_sizer.max_position_size_percentage)
            max_value = limits.get('max_position_value', self.position_sizer.max_position_value)
            
            # Calculate max size based on percentage of portfolio
            max_size_by_pct = (self.current_capital * max_percentage) / entry_price
            
            # Calculate max size based on absolute value
            max_size_by_value = max_value / entry_price
            
            # Take the minimum
            max_size_asset_class = min(max_size_by_pct, max_size_by_value)
            
            logger.debug(f"Applied {asset_class} position limits: {max_size_asset_class} units")

        # Take the minimum of all constraints
        final_size = min(base_size, max_size_portfolio, max_size_drawdown, max_size_asset_class)

        # Round to appropriate precision (e.g., whole shares for stocks)
        final_size = max(0, round(final_size, 8))

        # Update details
        details.update({
            "original_size": base_size,
            "portfolio_constrained_size": max_size_portfolio,
            "drawdown_adjusted_size": max_size_drawdown,
            "asset_class_constrained_size": max_size_asset_class if max_size_asset_class < float('inf') else None,
            "final_size": final_size,
            "method": method,
            "risk_multiplier": risk_multiplier,
            "asset_class": asset_class
        })

        logger.info(f"Calculated position size for {symbol}: {final_size} units")
        return final_size, details

    def set_stop_loss(self,
                     trade_id: str,
                     symbol: str,
                     entry_price: float,
                     entry_time: datetime,
                     direction: str,
                     stop_type: str = StopLossType.FIXED,
                     stop_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set a stop loss for a trade.

        Args:
            trade_id: Unique identifier for the trade
            symbol: Asset symbol
            entry_price: Entry price of the position
            entry_time: Entry time of the position
            direction: Trade direction ("long" or "short")
            stop_type: Type of stop loss (fixed, trailing, volatility, time, profit)
            stop_params: Additional parameters for the stop loss

        Returns:
            Stop loss details
        """
        if stop_params is None:
            stop_params = {}

        # Add symbol to stop params for tracking
        stop_params["symbol"] = symbol

        # Set stop loss using stop loss manager
        stop_details = self.stop_loss_manager.set_stop_loss(
            trade_id=trade_id,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_type=stop_type,
            stop_params=stop_params
        )

        # Store trade information
        self.trades[trade_id] = {
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "direction": direction,
            "stop_details": stop_details,
            "status": "open"
        }

        logger.info(f"Set {stop_type} stop loss for trade {trade_id} on {symbol}")
        return stop_details

    def update_stop_loss(self,
                        trade_id: str,
                        current_price: float,
                        current_time: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Update a stop loss based on current market conditions.

        Args:
            trade_id: Unique identifier for the trade
            current_price: Current market price
            current_time: Current time (default: now)

        Returns:
            Tuple of (is_triggered, updated_stop_details)
        """
        # Update stop loss using stop loss manager
        is_triggered, stop_details = self.stop_loss_manager.update_stop_loss(
            trade_id=trade_id,
            current_price=current_price,
            current_time=current_time
        )

        # Update trade status if stop is triggered
        if is_triggered and trade_id in self.trades:
            self.trades[trade_id]["status"] = "closed"
            self.trades[trade_id]["exit_price"] = current_price
            self.trades[trade_id]["exit_time"] = current_time or datetime.now()
            self.trades[trade_id]["exit_reason"] = "stop_loss"

            # Remove from positions if present
            symbol = self.trades[trade_id]["symbol"]
            if symbol in self.positions:
                del self.positions[symbol]

            logger.info(f"Stop loss triggered for trade {trade_id} at price {current_price}")

        return is_triggered, stop_details

    def check_all_stops(self,
                       current_prices: Dict[str, float],
                       current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Check all active stops against current market prices.

        Args:
            current_prices: Dictionary of current prices {symbol: price}
            current_time: Current time (default: now)

        Returns:
            List of triggered stop details
        """
        # Check all stops using stop loss manager
        triggered_stops = self.stop_loss_manager.check_all_stops(
            current_prices=current_prices,
            current_time=current_time
        )

        # Update trade statuses for triggered stops
        for stop in triggered_stops:
            trade_id = stop.get("trade_id")
            if trade_id in self.trades:
                symbol = self.trades[trade_id]["symbol"]
                current_price = current_prices.get(symbol)

                self.trades[trade_id]["status"] = "closed"
                self.trades[trade_id]["exit_price"] = current_price
                self.trades[trade_id]["exit_time"] = current_time or datetime.now()
                self.trades[trade_id]["exit_reason"] = "stop_loss"

                # Remove from positions if present
                if symbol in self.positions:
                    del self.positions[symbol]

        return triggered_stops

    def update_portfolio(self,
                        positions: Dict[str, Dict[str, Any]],
                        portfolio_value: float,
                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update the portfolio state with current positions and value.

        Args:
            positions: Dictionary of positions {symbol: position_details}
            portfolio_value: Current total portfolio value
            timestamp: Timestamp for this update (default: now)

        Returns:
            Portfolio summary
        """
        # Update portfolio risk manager
        self.portfolio_risk_manager.update_portfolio(
            positions=positions,
            portfolio_value=portfolio_value,
            timestamp=timestamp
        )

        # Update circuit breaker manager with account balance
        self.circuit_breaker_manager.update_account_balance(
            account_id="main",
            balance=portfolio_value
        )

        # Update state
        self.positions = positions
        self.current_capital = portfolio_value

        # Get portfolio summary
        summary = self.portfolio_risk_manager.get_portfolio_summary()

        # Check for risk alerts
        risk_alerts = self.portfolio_risk_manager.get_risk_alerts()
        if risk_alerts:
            logger.warning(f"Risk alerts detected: {len(risk_alerts)}")
            summary["risk_alerts"] = risk_alerts

        return summary

    def update_market_data(self,
                         market_id: str,
                         current_value: float,
                         previous_value: float) -> bool:
        """
        Update market data for circuit breaker monitoring.

        Args:
            market_id: Market identifier
            current_value: Current market value
            previous_value: Previous market value

        Returns:
            True if circuit breaker triggered, False otherwise
        """
        # Update circuit breaker manager with market data
        is_triggered = self.circuit_breaker_manager.update_market_data(
            market_id=market_id,
            current_value=current_value,
            previous_value=previous_value
        )

        if is_triggered:
            logger.warning(f"Market circuit breaker triggered for {market_id}")

        return is_triggered

    def get_active_circuit_breakers(self) -> List[Dict[str, Any]]:
        """
        Get active circuit breakers.

        Returns:
            List of active circuit breakers
        """
        return self.circuit_breaker_manager.get_active_circuit_breakers()

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        # Use the RiskMetricsCalculator to get comprehensive risk metrics
        return self.risk_metrics_calculator.calculate_all_metrics()
        
    def set_position_limits_for_asset_class(self,
                                          asset_class: str,
                                          max_position_percentage: float,
                                          max_position_value: float) -> None:
        """
        Set position limits for a specific asset class.
        
        Args:
            asset_class: The asset class (e.g., 'equities', 'options', 'futures')
            max_position_percentage: Maximum position size as percentage of portfolio
            max_position_value: Maximum position value in account currency
        """
        # Update position sizer with asset class specific limits
        if not hasattr(self, 'position_limits_by_asset'):
            self.position_limits_by_asset = {}
            
        self.position_limits_by_asset[asset_class] = {
            "max_position_percentage": max_position_percentage,
            "max_position_value": max_position_value
        }
        
        logger.info(f"Set position limits for {asset_class}: {max_position_percentage:.1%} max, "
                   f"${max_position_value:,.2f} max value")

    def get_trade_status(self, trade_id: str) -> Dict[str, Any]:
        """
        Get status of a specific trade.

        Args:
            trade_id: Unique identifier for the trade

        Returns:
            Trade status details
        """
        if trade_id not in self.trades:
            return {}

        trade_info = self.trades[trade_id].copy()

        # Add stop loss details if available
        stop_details = self.stop_loss_manager.get_stop_loss(trade_id)
        if stop_details:
            trade_info["stop_details"] = stop_details

        return trade_info

    def get_all_trades(self, status: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all trades, optionally filtered by status.

        Args:
            status: Filter by trade status ("open", "closed", or None for all)

        Returns:
            Dictionary of trades {trade_id: trade_details}
        """
        if status is None:
            return self.trades

        return {tid: trade for tid, trade in self.trades.items()
                if trade.get("status") == status}

    def reset_circuit_breakers(self) -> None:
        """
        Reset all circuit breakers.
        """
        self.circuit_breaker_manager.reset_circuit_breakers()
        logger.info("Reset all circuit breakers")

    def shutdown(self) -> None:
        """
        Shutdown the risk manager and its components.
        """
        # Stop circuit breaker check thread
        self.circuit_breaker_manager.stop_check_thread()
        logger.info("Shut down AdvancedRiskManager")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the risk manager state to a dictionary.

        Returns:
            Dictionary representation of the risk manager
        """
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "risk_per_trade": self.risk_per_trade,
            "position_sizing_method": self.position_sizing_method,
            "num_trades": len(self.trades),
            "num_positions": len(self.positions),
            "portfolio_risk": self.portfolio_risk_manager.to_dict(),
            "active_circuit_breakers": len(self.get_active_circuit_breakers()),
            "timestamp": datetime.now()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedRiskManager':
        """
        Create a risk manager from a dictionary.

        Args:
            data: Dictionary representation of a risk manager

        Returns:
            AdvancedRiskManager instance
        """
        manager = cls(
            initial_capital=data.get("initial_capital", 100000.0),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            position_sizing_method=data.get("position_sizing_method", "risk_based")
        )

        # Restore state if provided
        if "current_capital" in data:
            manager.current_capital = data["current_capital"]

        return manager
