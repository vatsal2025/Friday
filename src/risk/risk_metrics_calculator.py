from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime

class RiskMetricsCalculator:
    """
    Risk Metrics Calculator for computing various risk and performance metrics.

    This class provides functionality for:
    - Value at Risk (VaR) calculation using historical simulation
    - Drawdown calculation and tracking
    - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar ratios)
    - Volatility and correlation analysis
    """

    def __init__(self, confidence_level: float = 0.95, max_history_size: int = 252):
        """
        Initialize the risk metrics calculator.

        Args:
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            max_history_size: Maximum number of historical data points to retain (default: 252,
                             approximately 1 year of trading days)
        """
        self.confidence_level = confidence_level
        self.max_history_size = max_history_size
        self.historical_returns = []
        self.historical_values = []
        self.peak_value = 0.0
        
        # Use NumPy arrays for efficient calculations
        self._returns_array = np.array([])
        self._values_array = np.array([])

    def add_return_observation(self, return_value: float, timestamp: datetime) -> None:
        """
        Add a return observation to the historical returns.

        Args:
            return_value: The return value
            timestamp: The timestamp of the return observation
        """
        self.historical_returns.append((timestamp, return_value))

        # Keep only the specified maximum history size
        if len(self.historical_returns) > self.max_history_size:
            self.historical_returns = self.historical_returns[-self.max_history_size:]
            
        # Update NumPy array for efficient calculations
        self._returns_array = np.append(self._returns_array, return_value)
        if len(self._returns_array) > self.max_history_size:
            self._returns_array = self._returns_array[-self.max_history_size:]

    def add_portfolio_value_observation(self, value: float, timestamp: datetime) -> None:
        """
        Add a portfolio value observation to track drawdown.

        Args:
            value: The portfolio value
            timestamp: The timestamp of the value observation
        """
        self.historical_values.append((timestamp, value))
        
        # Keep only the specified maximum history size
        if len(self.historical_values) > self.max_history_size:
            self.historical_values = self.historical_values[-self.max_history_size:]
            
        # Update NumPy array for efficient calculations
        self._values_array = np.append(self._values_array, value)
        if len(self._values_array) > self.max_history_size:
            self._values_array = self._values_array[-self.max_history_size:]

        # Update peak value if this is a new peak
        if value > self.peak_value:
            self.peak_value = value

        # Keep only the last 252 trading days (approximately 1 year)
        if len(self.historical_values) > 252:
            # Remove oldest value
            self.historical_values.pop(0)

            # Recalculate peak value if we removed the peak
            if len(self.historical_values) > 0:
                self.peak_value = max(value for _, value in self.historical_values)

    def calculate_var(self, portfolio_value: float) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) using historical simulation method.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (VaR amount, VaR as percentage of portfolio)
        """
        if len(self.historical_returns) < 30:
            # Not enough data for reliable VaR calculation
            return 0.0, 0.0

        # Extract return values
        returns = [r for _, r in self.historical_returns]

        # Sort returns in ascending order (worst to best)
        sorted_returns = sorted(returns)

        # Find the return at the specified confidence level
        index = int((1 - self.confidence_level) * len(sorted_returns))
        var_return = abs(sorted_returns[index])

        # Calculate VaR amount and percentage
        var_amount = portfolio_value * var_return
        var_percent = var_return

        return var_amount, var_percent

    def calculate_drawdown(self) -> Tuple[float, float]:
        """
        Calculate current drawdown and maximum historical drawdown.

        Returns:
            Tuple of (current drawdown percentage, maximum historical drawdown percentage)
        """
        if not self.historical_values or self.peak_value <= 0:
            return 0.0, 0.0

        # Current portfolio value is the most recent value
        current_value = self.historical_values[-1][1]

        # Calculate current drawdown
        current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0

        # Calculate maximum historical drawdown
        drawdowns = []
        peak = self.historical_values[0][1]

        for _, value in self.historical_values:
            if value > peak:
                peak = value
            elif peak > 0:
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)

        max_drawdown = max(drawdowns) if drawdowns else 0.0

        return current_drawdown, max_drawdown

    def calculate_volatility(self, annualize: bool = True) -> float:
        """
        Calculate return volatility (standard deviation).

        Args:
            annualize: Whether to annualize the volatility (default: True)

        Returns:
            Volatility value
        """
        if len(self.historical_returns) < 2:
            return 0.0

        # Extract return values
        returns = [r for _, r in self.historical_returns]

        # Calculate standard deviation
        volatility = np.std(returns)

        # Annualize if requested (assuming daily returns, multiply by sqrt(252))
        if annualize:
            volatility *= np.sqrt(252)

        return volatility

    def calculate_sharpe_ratio(self, annualized_return: float, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (return per unit of risk).

        Args:
            annualized_return: Annualized portfolio return
            risk_free_rate: Risk-free rate (default: 0.0)

        Returns:
            Sharpe ratio value
        """
        volatility = self.calculate_volatility()

        if volatility <= 0:
            return 0.0

        return (annualized_return - risk_free_rate) / volatility

    def calculate_sortino_ratio(self, annualized_return: float, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (return per unit of downside risk).

        Args:
            annualized_return: Annualized portfolio return
            risk_free_rate: Risk-free rate (default: 0.0)

        Returns:
            Sortino ratio value
        """
        if len(self.historical_returns) < 2:
            return 0.0

        # Extract return values
        returns = [r for _, r in self.historical_returns]

        # Calculate downside deviation (standard deviation of negative returns)
        negative_returns = [r for r in returns if r < 0]

        if not negative_returns:
            return 0.0 if annualized_return <= risk_free_rate else float('inf')

        downside_deviation = np.std(negative_returns)

        # Annualize downside deviation (assuming daily returns)
        downside_deviation *= np.sqrt(252)

        if downside_deviation <= 0:
            return 0.0

        return (annualized_return - risk_free_rate) / downside_deviation

    def calculate_calmar_ratio(self, annualized_return: float) -> float:
        """
        Calculate Calmar ratio (return per unit of maximum drawdown).

        Args:
            annualized_return: Annualized portfolio return

        Returns:
            Calmar ratio value
        """
        _, max_drawdown = self.calculate_drawdown()

        if max_drawdown <= 0:
            return 0.0 if annualized_return <= 0 else float('inf')

        return annualized_return / max_drawdown

    def calculate_all_metrics(self, portfolio_value: float, annualized_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate all risk metrics.

        Args:
            portfolio_value: Current portfolio value
            annualized_return: Annualized portfolio return (if None, calculated from historical returns)

        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            "timestamp": datetime.now()
        }

        # Calculate VaR
        var_amount, var_percent = self.calculate_var(portfolio_value)
        metrics["var"] = var_amount
        metrics["var_percent"] = var_percent

        # Calculate drawdown
        current_drawdown, max_drawdown = self.calculate_drawdown()
        metrics["current_drawdown"] = current_drawdown
        metrics["max_drawdown"] = max_drawdown

        # Calculate volatility
        volatility = self.calculate_volatility()
        metrics["annualized_volatility"] = volatility

        # Calculate annualized return if not provided
        if annualized_return is None and len(self.historical_returns) >= 30:
            returns = [r for _, r in self.historical_returns]
            avg_daily_return = np.mean(returns)
            annualized_return = ((1 + avg_daily_return) ** 252) - 1

        # Calculate risk-adjusted metrics if we have annualized return
        if annualized_return is not None:
            metrics["annualized_return"] = annualized_return

            # Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(annualized_return)
            metrics["sharpe_ratio"] = sharpe_ratio

            # Sortino ratio
            sortino_ratio = self.calculate_sortino_ratio(annualized_return)
            metrics["sortino_ratio"] = sortino_ratio

            # Calmar ratio
            calmar_ratio = self.calculate_calmar_ratio(annualized_return)
            metrics["calmar_ratio"] = calmar_ratio

        return metrics
