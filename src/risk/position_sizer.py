import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pandas as pd

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Position sizing algorithm that calculates optimal position sizes based on risk parameters.

    This class implements various position sizing strategies including:
    - Fixed fractional risk (Kelly criterion)
    - Volatility-based sizing
    - Portfolio-level constraints
    """

    def __init__(self,
                 default_risk_per_trade: float = 0.02,  # 2% risk per trade by default
                 max_position_size_percentage: float = 0.05,  # 5% max position size
                 max_position_value: float = 50000.0,  # $50K max position
                 volatility_lookback_days: int = 20,
                 position_sizing_method: str = "risk_based"):
        """
        Initialize the position sizer.

        Args:
            default_risk_per_trade: Default risk per trade as a percentage of portfolio (0.02 = 2%)
            max_position_size_percentage: Maximum position size as a percentage of portfolio
            max_position_value: Maximum position value in account currency
            volatility_lookback_days: Number of days to look back for volatility calculation
            position_sizing_method: Method to use for position sizing (risk_based, volatility_based, fixed)
        """
        self.default_risk_per_trade = default_risk_per_trade
        self.max_position_size_percentage = max_position_size_percentage
        self.max_position_value = max_position_value
        self.volatility_lookback_days = volatility_lookback_days
        self.position_sizing_method = position_sizing_method

        logger.info(f"Initialized PositionSizer with {position_sizing_method} method")

    def calculate_position_size(self,
                               capital: float,
                               risk_per_trade: Optional[float] = None,
                               stop_loss_percent: Optional[float] = None,
                               volatility: Optional[float] = None,
                               current_price: Optional[float] = None,
                               asset_class: Optional[str] = None) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as a percentage of capital (0.02 = 2%)
            stop_loss_percent: Stop loss percentage (0.02 = 2%)
            volatility: Asset volatility (standard deviation of returns)
            current_price: Current price of the asset
            asset_class: Asset class (equities, options, futures, forex, crypto)

        Returns:
            Position size in units of the asset
        """
        if not capital or capital <= 0:
            logger.warning("Invalid capital value provided")
            return 0.0

        # Use default risk per trade if not provided
        if risk_per_trade is None:
            risk_per_trade = self.default_risk_per_trade

        # Calculate risk amount in currency
        risk_amount = capital * risk_per_trade

        # Choose position sizing method
        if self.position_sizing_method == "risk_based":
            return self._calculate_risk_based_size(capital, risk_amount, stop_loss_percent, current_price)
        elif self.position_sizing_method == "volatility_based":
            return self._calculate_volatility_based_size(capital, risk_amount, volatility, current_price)
        elif self.position_sizing_method == "fixed":
            return self._calculate_fixed_size(capital, current_price)
        else:
            logger.warning(f"Unknown position sizing method: {self.position_sizing_method}")
            return self._calculate_risk_based_size(capital, risk_amount, stop_loss_percent, current_price)

    def _calculate_risk_based_size(self,
                                 capital: float,
                                 risk_amount: float,
                                 stop_loss_percent: Optional[float],
                                 current_price: Optional[float]) -> float:
        """
        Calculate position size based on risk per trade and stop loss percentage.

        Args:
            capital: Available capital
            risk_amount: Amount of capital to risk per trade
            stop_loss_percent: Stop loss percentage (0.02 = 2%)
            current_price: Current price of the asset

        Returns:
            Position size in units of the asset
        """
        if not stop_loss_percent or stop_loss_percent <= 0:
            logger.warning("Invalid stop loss percentage provided")
            return 0.0

        if not current_price or current_price <= 0:
            logger.warning("Invalid current price provided")
            return 0.0

        # Calculate position size in currency
        position_value = risk_amount / stop_loss_percent

        # Convert to units
        position_size = position_value / current_price

        # Apply position limits
        position_size = self._apply_position_limits(position_size, capital, current_price)

        return position_size

    def _calculate_volatility_based_size(self,
                                       capital: float,
                                       risk_amount: float,
                                       volatility: Optional[float],
                                       current_price: Optional[float]) -> float:
        """
        Calculate position size based on volatility.

        Args:
            capital: Available capital
            risk_amount: Amount of capital to risk per trade
            volatility: Asset volatility (standard deviation of returns)
            current_price: Current price of the asset

        Returns:
            Position size in units of the asset
        """
        if not volatility or volatility <= 0:
            logger.warning("Invalid volatility provided")
            return 0.0

        if not current_price or current_price <= 0:
            logger.warning("Invalid current price provided")
            return 0.0

        # Use volatility as a proxy for risk
        # Higher volatility = smaller position size
        position_value = risk_amount / volatility

        # Convert to units
        position_size = position_value / current_price

        # Apply position limits
        position_size = self._apply_position_limits(position_size, capital, current_price)

        return position_size

    def _calculate_fixed_size(self,
                            capital: float,
                            current_price: Optional[float]) -> float:
        """
        Calculate position size based on a fixed percentage of capital.

        Args:
            capital: Available capital
            current_price: Current price of the asset

        Returns:
            Position size in units of the asset
        """
        if not current_price or current_price <= 0:
            logger.warning("Invalid current price provided")
            return 0.0

        # Use max position size percentage
        position_value = capital * self.max_position_size_percentage

        # Apply absolute maximum
        position_value = min(position_value, self.max_position_value)

        # Convert to units
        position_size = position_value / current_price

        return position_size

    def _apply_position_limits(self,
                             position_size: float,
                             capital: float,
                             current_price: float) -> float:
        """
        Apply position limits to the calculated position size.

        Args:
            position_size: Calculated position size in units
            capital: Available capital
            current_price: Current price of the asset

        Returns:
            Adjusted position size in units of the asset
        """
        # Calculate position value
        position_value = position_size * current_price

        # Apply percentage limit
        max_value_by_percentage = capital * self.max_position_size_percentage

        # Apply absolute limit
        max_value = min(max_value_by_percentage, self.max_position_value)

        # Adjust position size if necessary
        if position_value > max_value:
            position_size = max_value / current_price

        return position_size

    def calculate_position_sizes(self,
                               signals: Dict[str, Dict[str, Any]],
                               portfolio: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes for multiple signals based on portfolio constraints.

        Args:
            signals: Dictionary of trading signals with metadata
                     {symbol: {"direction": "long"/"short", "confidence": 0.0-1.0, "stop_loss": 0.02, ...}}
            portfolio: Portfolio information including capital, positions, etc.

        Returns:
            Dictionary of position sizes for each symbol {symbol: position_size}
        """
        position_sizes = {}
        available_capital = portfolio.get("capital", 0.0)

        if not signals or not available_capital:
            return position_sizes

        # Get current positions
        current_positions = portfolio.get("positions", {})

        # Calculate total portfolio value
        portfolio_value = available_capital
        for symbol, position in current_positions.items():
            portfolio_value += position.get("market_value", 0.0)

        # Adjust available capital based on existing positions
        adjusted_capital = portfolio_value

        # Sort signals by confidence (highest first)
        sorted_signals = sorted(
            [(symbol, data) for symbol, data in signals.items()],
            key=lambda x: x[1].get("confidence", 0.0),
            reverse=True
        )

        # Calculate position sizes for each signal
        for symbol, signal_data in sorted_signals:
            # Extract signal parameters
            stop_loss_percent = signal_data.get("stop_loss", 0.02)
            confidence = signal_data.get("confidence", 0.5)
            current_price = signal_data.get("current_price")
            volatility = signal_data.get("volatility")

            # Adjust risk per trade based on confidence
            risk_per_trade = self.default_risk_per_trade * confidence

            # Calculate position size
            position_size = self.calculate_position_size(
                capital=adjusted_capital,
                risk_per_trade=risk_per_trade,
                stop_loss_percent=stop_loss_percent,
                volatility=volatility,
                current_price=current_price
            )

            position_sizes[symbol] = position_size

        # Apply portfolio-level constraints
        position_sizes = self._apply_portfolio_constraints(position_sizes, signals, portfolio_value)

        return position_sizes

    def _apply_portfolio_constraints(self,
                                   position_sizes: Dict[str, float],
                                   signals: Dict[str, Dict[str, Any]],
                                   portfolio_value: float) -> Dict[str, float]:
        """
        Apply portfolio-level constraints to position sizes.

        Args:
            position_sizes: Dictionary of calculated position sizes
            signals: Dictionary of trading signals with metadata
            portfolio_value: Total portfolio value

        Returns:
            Adjusted position sizes
        """
        # Calculate total position value
        total_value = 0.0
        for symbol, size in position_sizes.items():
            price = signals[symbol].get("current_price", 0.0)
            total_value += size * price

        # If total exceeds max exposure, scale down proportionally
        max_exposure = portfolio_value * 0.8  # 80% max exposure

        if total_value > max_exposure and total_value > 0:
            scale_factor = max_exposure / total_value
            for symbol in position_sizes:
                position_sizes[symbol] *= scale_factor

        return position_sizes

    def adjust_for_volatility(self,
                            position_size: float,
                            historical_prices: pd.Series,
                            current_price: float) -> float:
        """
        Adjust position size based on recent market volatility.

        Args:
            position_size: Original position size
            historical_prices: Series of historical prices
            current_price: Current price of the asset

        Returns:
            Adjusted position size
        """
        if historical_prices is None or len(historical_prices) < self.volatility_lookback_days:
            return position_size

        # Calculate daily returns
        returns = historical_prices.pct_change().dropna()

        # Calculate volatility (standard deviation of returns)
        volatility = returns.std()

        # Calculate average volatility over the lookback period
        avg_volatility = 0.01  # 1% daily volatility as baseline

        # Adjust position size based on relative volatility
        # Higher volatility = smaller position size
        if volatility > 0:
            volatility_ratio = avg_volatility / volatility
            adjusted_size = position_size * min(max(volatility_ratio, 0.5), 2.0)
            return adjusted_size

        return position_size
