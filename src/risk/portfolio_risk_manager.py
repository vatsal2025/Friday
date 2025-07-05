import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from .risk_metrics_calculator import RiskMetricsCalculator

logger = logging.getLogger(__name__)

class PortfolioRiskManager:
    """
    Portfolio Risk Manager for monitoring and controlling portfolio-level risk.

    This class provides functionality for:
    - Value at Risk (VaR) calculation and monitoring
    - Drawdown monitoring and controls
    - Sector/asset exposure limits
    - Correlation risk management
    - Portfolio-level position sizing
    - Risk-adjusted performance metrics
    """

    def __init__(self, 
                 max_portfolio_var_percent: float = 0.05,
                 max_drawdown_percent: float = 0.25,
                 max_sector_allocation: float = 0.30,
                 max_position_size: float = 0.10,
                 max_correlation_exposure: float = 0.50,
                 confidence_level: float = 0.95,
                 var_lookback_days: int = 252,
                 drawdown_window_days: int = 252,
                 max_history_size: int = 252,
                 risk_metrics_calculator: Optional[RiskMetricsCalculator] = None):
        """
        Initialize the portfolio risk manager.

        Args:
            max_portfolio_var_percent: Maximum portfolio Value at Risk as percentage
            max_drawdown_percent: Maximum allowable drawdown percentage
            max_sector_allocation: Maximum allocation to any single sector
            max_position_size: Maximum size of any single position
            max_correlation_exposure: Maximum exposure to correlated assets
            confidence_level: Confidence level for VaR calculation
            var_lookback_days: Lookback period for VaR calculation
            drawdown_window_days: Window for drawdown calculation
            max_history_size: Maximum number of historical data points to retain
                             (default: 252, approximately 1 year of trading days)
        """
        self.max_portfolio_var_percent = max_portfolio_var_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.max_sector_exposure = max_sector_allocation  # Renamed parameter
        self.max_asset_exposure = max_position_size  # Renamed parameter
        self.max_correlation_exposure = max_correlation_exposure if 'max_correlation_exposure' in locals() else 0.50
        self.confidence_level = confidence_level
        self.var_lookback_days = var_lookback_days
        self.max_history_size = max_history_size

        # Store the risk metrics calculator if provided, otherwise it will be created later
        self.risk_metrics_calculator = risk_metrics_calculator
        self.drawdown_window_days = drawdown_window_days

        # Initialize the risk metrics calculator if not provided
        if self.risk_metrics_calculator is None:
            from .risk_metrics_calculator import RiskMetricsCalculator
            self.risk_metrics_calculator = RiskMetricsCalculator(
                confidence_level=self.confidence_level,
                max_history_size=self.max_history_size
            )

        # Portfolio state
        self.portfolio_value = 0.0
        self.positions = {}
        self.sector_allocations = {}
        self.historical_values = []
        self.historical_returns = []
        self.correlation_matrix = None
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        self.peak_value = 0.0
        
        # Initialize NumPy arrays for efficient calculations
        self._values_array = np.array([])
        self._returns_array = np.array([])

        # Risk metrics
        self.current_var = 0.0
        self.current_var_percent = 0.0
        self.current_sector_exposure = {}
        self.current_asset_exposure = {}

        # Risk alerts
        self.risk_alerts = []

        logger.info("Initialized PortfolioRiskManager")

    def update_portfolio(self,
                        positions: Dict[str, Dict[str, Any]],
                        portfolio_value: float,
                        timestamp: Optional[datetime] = None) -> None:
        """
        Update the portfolio state with current positions and value.

        Args:
            positions: Dictionary of positions {symbol: position_details}
            portfolio_value: Current total portfolio value
            timestamp: Timestamp for this update (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.positions = positions
        self.portfolio_value = portfolio_value

        # Update historical values
        self.historical_values.append((timestamp, portfolio_value))

        # Update the risk metrics calculator with the new portfolio value
        if self.risk_metrics_calculator is not None:
            self.risk_metrics_calculator.add_portfolio_value_observation(portfolio_value, timestamp)

        # Keep only the most recent values within the max_history_size
        if len(self.historical_values) > self.max_history_size:
            self.historical_values = self.historical_values[-self.max_history_size:]
            
        # Initialize NumPy array for historical values if not already done
        if not hasattr(self, '_values_array'):
            self._values_array = np.array([val for _, val in self.historical_values])
        else:
            self._values_array = np.append(self._values_array, portfolio_value)
            if len(self._values_array) > self.max_history_size:
                self._values_array = self._values_array[-self.max_history_size:]

        # Calculate returns if we have at least two data points
        if len(self.historical_values) >= 2:
            prev_value = self.historical_values[-2][1]
            if prev_value > 0:
                daily_return = (portfolio_value / prev_value) - 1
                self.historical_returns.append((timestamp, daily_return))
                
                # Keep only the most recent returns within the max_history_size
                if len(self.historical_returns) > self.max_history_size:
                    self.historical_returns = self.historical_returns[-self.max_history_size:]
                
                # Initialize NumPy array for returns if not already done
                if not hasattr(self, '_returns_array'):
                    self._returns_array = np.array([ret for _, ret in self.historical_returns])
                else:
                    self._returns_array = np.append(self._returns_array, daily_return)
                    if len(self._returns_array) > self.max_history_size:
                        self._returns_array = self._returns_array[-self.max_history_size:]

                # Update the risk metrics calculator with the new return
                if self.risk_metrics_calculator is not None:
                    self.risk_metrics_calculator.add_return_observation(daily_return, timestamp)

        # Update peak value for drawdown calculation
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = 1 - (portfolio_value / self.peak_value)
            self.max_historical_drawdown = max(self.max_historical_drawdown, self.current_drawdown)

        # Update sector allocations
        self._update_sector_allocations()

        # Update risk metrics
        self._update_risk_metrics()

        logger.info(f"Updated portfolio: value={portfolio_value:.2f}, positions={len(positions)}")

    def _update_sector_allocations(self) -> None:
        """
        Update sector allocations based on current positions.
        """
        self.sector_allocations = {}

        for symbol, position in self.positions.items():
            sector = position.get("sector", "Unknown")
            position_value = position.get("market_value", 0.0)

            if sector not in self.sector_allocations:
                self.sector_allocations[sector] = 0.0

            self.sector_allocations[sector] += position_value

    def _update_risk_metrics(self) -> None:
        """
        Update all risk metrics based on current portfolio state.
        """
        # Calculate VaR
        self._calculate_var()

        # Calculate sector and asset exposures
        self._calculate_exposures()

        # Update correlation matrix if we have positions
        if len(self.positions) > 1:
            self._update_correlation_matrix()

        # Check for risk limit breaches
        self._check_risk_limits()

        # Update the risk metrics calculator if available
        if self.risk_metrics_calculator is not None:
            # Add the latest return if available
            if self.historical_returns:
                timestamp, return_value = self.historical_returns[-1]
                self.risk_metrics_calculator.add_return_observation(return_value, timestamp)

            # Add the latest portfolio value
            if hasattr(self, 'portfolio_value') and self.portfolio_value is not None:
                self.risk_metrics_calculator.add_portfolio_value_observation(self.portfolio_value, datetime.now())

    def _calculate_var(self) -> None:
        """
        Calculate Value at Risk (VaR) using historical simulation method.
        """
        # Check if we have sufficient data for VaR calculation
        if len(self._returns_array) < 30:
            logger.warning("Insufficient data for VaR calculation")
            self.current_var = 0.0
            self.current_var_percent = 0.0
            return

        # Calculate VaR at the specified confidence level using NumPy array
        var_percentile = 1.0 - self.confidence_level
        var_percent = np.percentile(self._returns_array, var_percentile * 100)

        # Convert to absolute value (VaR is typically positive)
        var_percent = abs(var_percent)
        var_amount = self.portfolio_value * var_percent

        self.current_var = var_amount
        self.current_var_percent = var_percent

        logger.info(f"Updated VaR: {var_percent:.2%} (${var_amount:.2f})")

    def _calculate_exposures(self) -> None:
        """
        Calculate sector and asset exposures as percentages of portfolio value.
        """
        # Reset exposures
        self.current_sector_exposure = {}
        self.current_asset_exposure = {}

        if self.portfolio_value <= 0:
            return

        # Calculate sector exposures
        for sector, allocation in self.sector_allocations.items():
            self.current_sector_exposure[sector] = allocation / self.portfolio_value

        # Calculate asset exposures
        for symbol, position in self.positions.items():
            position_value = position.get("market_value", 0.0)
            self.current_asset_exposure[symbol] = position_value / self.portfolio_value

    def _update_correlation_matrix(self) -> None:
        """
        Update correlation matrix between assets in the portfolio.

        Note: This requires historical price data for each asset, which would
        typically be provided by a data service. This implementation is simplified.
        """
        # In a real implementation, this would fetch historical price data for each asset
        # and calculate the correlation matrix. For now, we'll just log a placeholder.
        logger.info("Correlation matrix update would happen here")

        # Placeholder for correlation matrix
        self.correlation_matrix = {}

    def _check_risk_limits(self) -> None:
        """
        Check if any risk limits are breached and generate alerts.
        """
        # Clear previous alerts
        self.risk_alerts = []

        # Check VaR limit
        if self.current_var_percent > self.max_portfolio_var_percent:
            alert = {
                "type": "VAR_BREACH",
                "timestamp": datetime.now(),
                "details": {
                    "current_var": self.current_var_percent,
                    "limit": self.max_portfolio_var_percent,
                    "excess": self.current_var_percent - self.max_portfolio_var_percent
                }
            }
            self.risk_alerts.append(alert)
            logger.warning(f"VaR limit breach: {self.current_var_percent:.2%} > {self.max_portfolio_var_percent:.2%}")

        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_percent:
            alert = {
                "type": "DRAWDOWN_BREACH",
                "timestamp": datetime.now(),
                "details": {
                    "current_drawdown": self.current_drawdown,
                    "limit": self.max_drawdown_percent,
                    "excess": self.current_drawdown - self.max_drawdown_percent
                }
            }
            self.risk_alerts.append(alert)
            logger.warning(f"Drawdown limit breach: {self.current_drawdown:.2%} > {self.max_drawdown_percent:.2%}")

        # Check sector exposure limits
        for sector, exposure in self.current_sector_exposure.items():
            if exposure > self.max_sector_allocation:
                alert = {
                    "type": "SECTOR_EXPOSURE_BREACH",
                    "timestamp": datetime.now(),
                    "details": {
                        "sector": sector,
                        "current_exposure": exposure,
                        "limit": self.max_sector_allocation,
                        "excess": exposure - self.max_sector_allocation
                    }
                }
                self.risk_alerts.append(alert)
                logger.warning(f"Sector exposure limit breach: {sector} at {exposure:.2%} > {self.max_sector_allocation:.2%}")

        # Check asset exposure limits
        for symbol, exposure in self.current_asset_exposure.items():
            if exposure > self.max_position_size:
                alert = {
                    "type": "ASSET_EXPOSURE_BREACH",
                    "timestamp": datetime.now(),
                    "details": {
                        "symbol": symbol,
                        "current_exposure": exposure,
                        "limit": self.max_position_size,
                        "excess": exposure - self.max_position_size
                    }
                }
                self.risk_alerts.append(alert)
                logger.warning(f"Asset exposure limit breach: {symbol} at {exposure:.2%} > {self.max_position_size:.2%}")

    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current risk alerts.

        Returns:
            List of risk alerts
        """
        return self.risk_alerts

    def calculate_max_position_size(self,
                                  symbol: str,
                                  price: float,
                                  sector: Optional[str] = None) -> float:
        """
        Calculate the maximum allowed position size for a new trade.

        Args:
            symbol: Asset symbol
            price: Current price of the asset
            sector: Sector of the asset (if known)

        Returns:
            Maximum position size in units
        """
        if self.portfolio_value <= 0 or price <= 0:
            return 0.0

        # Start with maximum position size
        max_value = self.portfolio_value * self.max_position_size

        # Adjust for existing position if any
        if symbol in self.positions:
            existing_value = self.positions[symbol].get("market_value", 0.0)
            max_value -= existing_value

        # Adjust for sector exposure if sector is provided
        if sector is not None:
            current_sector_value = self.sector_allocations.get(sector, 0.0)
            max_sector_value = self.portfolio_value * self.max_sector_allocation
            remaining_sector_capacity = max_sector_value - current_sector_value
            max_value = min(max_value, remaining_sector_capacity)

        # Adjust for correlation risk (simplified)
        # In a real implementation, this would use the correlation matrix

        # Convert to units
        max_units = max_value / price

        return max(0.0, max_units)

    def calculate_var_adjusted_position_size(self,
                                           symbol: str,
                                           price: float,
                                           volatility: float,
                                           correlation: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate position size adjusted for contribution to portfolio VaR.

        Args:
            symbol: Asset symbol
            price: Current price of the asset
            volatility: Asset volatility (standard deviation of returns)
            correlation: Correlation of this asset with other portfolio assets

        Returns:
            VaR-adjusted maximum position size in units
        """
        if self.portfolio_value <= 0 or price <= 0 or volatility <= 0:
            return 0.0

        # Calculate maximum VaR contribution allowed for this position
        # This is a simplified approach; a full implementation would use
        # marginal VaR contribution calculations
        max_var_contribution = self.portfolio_value * self.max_portfolio_var_percent * 0.2  # 20% of total VaR budget

        # Calculate position size based on volatility and confidence level
        # Using scipy's norm.ppf for more efficient z-score calculation
        from scipy import stats
        z_score = abs(stats.norm.ppf(1 - self.confidence_level))
        max_value = max_var_contribution / (volatility * z_score)

        # Adjust for correlation (simplified)
        if correlation is not None and len(correlation) > 0:
            avg_correlation = sum(correlation.values()) / len(correlation)
            # Reduce position size as correlation increases
            correlation_factor = 1.0 - (avg_correlation * 0.5)  # 50% reduction at perfect correlation
            max_value *= correlation_factor

        # Convert to units
        max_units = max_value / price

        return max(0.0, max_units)

    def calculate_drawdown_adjusted_position_size(self,
                                               base_position_size: float) -> float:
        """
        Adjust position size based on current drawdown.

        Args:
            base_position_size: Base position size to adjust

        Returns:
            Drawdown-adjusted position size
        """
        if base_position_size <= 0:
            return 0.0

        # Calculate drawdown factor (reduce position as drawdown increases)
        # At max drawdown, position size is reduced by 75%
        drawdown_ratio = self.current_drawdown / self.max_drawdown_percent
        drawdown_factor = 1.0 - (min(1.0, drawdown_ratio) * 0.75)

        return base_position_size * drawdown_factor

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk-adjusted performance metrics.

        Returns:
            Dictionary of risk metrics
        """
        # If we have a risk metrics calculator, use it for comprehensive metrics
        if self.risk_metrics_calculator is not None:
            # Make sure the calculator has the latest data
            for timestamp, return_value in self.historical_returns:
                self.risk_metrics_calculator.add_return_observation(return_value, timestamp)

            # Add the latest portfolio value
            if hasattr(self, 'portfolio_value') and self.portfolio_value is not None:
                self.risk_metrics_calculator.add_portfolio_value_observation(self.portfolio_value, datetime.now())

            # Get comprehensive metrics from the calculator
            metrics = self.risk_metrics_calculator.calculate_all_metrics()

            # Add portfolio-specific metrics that the calculator doesn't have
            metrics.update({
                "sector_exposure": self.current_sector_exposure,
                "asset_exposure": self.current_asset_exposure,
            })

            return metrics

        # Fall back to original implementation if no calculator is available
        metrics = {
            "var": self.current_var,
            "var_percent": self.current_var_percent,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_historical_drawdown,
            "sector_exposure": self.current_sector_exposure,
            "asset_exposure": self.current_asset_exposure,
            "portfolio_value": self.portfolio_value,
            "timestamp": datetime.now()
        }

        # Calculate additional metrics if we have sufficient data
        if len(self._returns_array) >= 30:
            # Use NumPy array for more efficient calculations
            # Annualized return (assuming daily returns)
            avg_daily_return = np.mean(self._returns_array)
            annualized_return = ((1 + avg_daily_return) ** 252) - 1
            metrics["annualized_return"] = annualized_return

            # Volatility (annualized) - using NumPy array for efficiency
            daily_volatility = np.std(self._returns_array)
            annualized_volatility = daily_volatility * np.sqrt(252)
            metrics["annualized_volatility"] = annualized_volatility

            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility
                metrics["sharpe_ratio"] = sharpe_ratio

            # Sortino ratio (downside deviation) - using NumPy array for efficiency
            negative_returns = self._returns_array[self._returns_array < 0]
            if len(negative_returns) > 0 and annualized_return > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
                metrics["sortino_ratio"] = sortino_ratio

            # Calmar ratio
            if self.max_historical_drawdown > 0:
                calmar_ratio = annualized_return / self.max_historical_drawdown
                metrics["calmar_ratio"] = calmar_ratio

        return metrics

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio state and risk metrics.

        Returns:
            Dictionary with portfolio summary
        """
        summary = {
            "portfolio_value": self.portfolio_value,
            "num_positions": len(self.positions),
            "risk_metrics": self.calculate_risk_metrics(),
            "risk_alerts": self.get_risk_alerts(),
            "timestamp": datetime.now()
        }

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the portfolio risk manager state to a dictionary.

        Returns:
            Dictionary representation of the portfolio risk manager
        """
        return {
            "max_portfolio_var_percent": self.max_portfolio_var_percent,
            "max_drawdown_percent": self.max_drawdown_percent,
            "max_sector_allocation": self.max_sector_allocation,
            "max_position_size": self.max_position_size,
            "max_correlation_exposure": self.max_correlation_exposure,
            "confidence_level": self.confidence_level,
            "var_lookback_days": self.var_lookback_days,
            "drawdown_window_days": self.drawdown_window_days,
            "portfolio_value": self.portfolio_value,
            "current_drawdown": self.current_drawdown,
            "max_historical_drawdown": self.max_historical_drawdown,
            "current_var_percent": self.current_var_percent,
            "num_positions": len(self.positions),
            "num_sectors": len(self.sector_allocations),
            "timestamp": datetime.now()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioRiskManager':
        """
        Create a portfolio risk manager from a dictionary.

        Args:
            data: Dictionary representation of a portfolio risk manager

        Returns:
            PortfolioRiskManager instance
        """
        manager = cls(
            max_portfolio_var_percent=data.get("max_portfolio_var_percent", 0.02),
            max_drawdown_percent=data.get("max_drawdown_percent", 0.15),
            max_sector_allocation=data.get("max_sector_allocation", data.get("max_sector_exposure", 0.25)),
            max_position_size=data.get("max_position_size", data.get("max_asset_exposure", 0.10)),
            max_correlation_exposure=data.get("max_correlation_exposure", 0.30),
            confidence_level=data.get("confidence_level", 0.95),
            var_lookback_days=data.get("var_lookback_days", 252),
            drawdown_window_days=data.get("drawdown_window_days", 60),
            max_history_size=data.get("max_history_size", 252)
        )

        # Restore state if provided
        if "portfolio_value" in data:
            manager.portfolio_value = data["portfolio_value"]
        if "current_drawdown" in data:
            manager.current_drawdown = data["current_drawdown"]
        if "max_historical_drawdown" in data:
            manager.max_historical_drawdown = data["max_historical_drawdown"]

        return manager
