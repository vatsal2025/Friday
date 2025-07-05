"""Performance analytics for backtesting framework.

This module provides tools for calculating and analyzing trading strategy
performance metrics.
"""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PerformanceMetrics(Enum):
    """Standard performance metrics for trading strategies."""
    TOTAL_RETURN = "total_return"                # Total return percentage
    ANNUALIZED_RETURN = "annualized_return"      # Annualized return percentage
    VOLATILITY = "volatility"                    # Return volatility
    SHARPE_RATIO = "sharpe_ratio"                # Sharpe ratio
    SORTINO_RATIO = "sortino_ratio"              # Sortino ratio
    MAX_DRAWDOWN = "max_drawdown"                # Maximum drawdown
    CALMAR_RATIO = "calmar_ratio"                # Calmar ratio
    ALPHA = "alpha"                              # Jensen's alpha
    BETA = "beta"                                # Beta to benchmark
    INFORMATION_RATIO = "information_ratio"      # Information ratio
    WINNING_PERCENTAGE = "winning_percentage"    # Percentage of winning trades
    PROFIT_FACTOR = "profit_factor"              # Profit factor
    EXPECTANCY = "expectancy"                    # Average trade expectancy
    AVERAGE_TRADE = "average_trade"              # Average trade return
    AVERAGE_WIN = "average_win"                  # Average winning trade
    AVERAGE_LOSS = "average_loss"                # Average losing trade
    WIN_LOSS_RATIO = "win_loss_ratio"            # Win/loss ratio
    MAX_CONSECUTIVE_WINS = "max_consecutive_wins"  # Maximum consecutive wins
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"  # Maximum consecutive losses
    RECOVERY_FACTOR = "recovery_factor"          # Recovery factor
    ULCER_INDEX = "ulcer_index"                  # Ulcer index
    PAIN_INDEX = "pain_index"                    # Pain index
    CAGR = "cagr"                                # Compound annual growth rate
    VALUE_AT_RISK = "value_at_risk"              # Value at risk
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"  # Conditional value at risk


class PerformanceAnalytics:
    """Performance analytics for trading strategies.
    
    This class calculates and analyzes trading strategy performance metrics.
    """
    
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        frequency: str = "daily",
    ):
        """Initialize the performance analytics.
        
        Args:
            equity_curve: DataFrame with equity curve (index=timestamp, columns=[equity])
            returns: DataFrame with returns (index=timestamp, columns=[returns])
            trades: DataFrame with trade details
            benchmark_returns: Optional DataFrame with benchmark returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            frequency: Data frequency ("daily", "weekly", "monthly")
        """
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades = trades
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        # Determine annualization factor based on frequency
        self.annualization_factor = self._get_annualization_factor(frequency)
        
        # Calculate daily risk-free rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / self.annualization_factor) - 1
        
        # Initialize metrics dictionary
        self.metrics = {}
        
        # Calculate all metrics
        self._calculate_all_metrics()
        
        logger.info("Initialized performance analytics")
    
    def _get_annualization_factor(self, frequency: str) -> int:
        """Get annualization factor based on data frequency.
        
        Args:
            frequency: Data frequency ("daily", "weekly", "monthly")
            
        Returns:
            Annualization factor
        """
        frequency_map = {
            "daily": 252,    # Trading days in a year
            "weekly": 52,   # Weeks in a year
            "monthly": 12,  # Months in a year
            "hourly": 252 * 6.5,  # Trading hours in a year (approx)
            "minute": 252 * 6.5 * 60,  # Trading minutes in a year (approx)
        }
        
        return frequency_map.get(frequency.lower(), 252)
    
    def _calculate_all_metrics(self) -> None:
        """Calculate all performance metrics."""
        # Return metrics
        self.metrics[PerformanceMetrics.TOTAL_RETURN.value] = self._calculate_total_return()
        self.metrics[PerformanceMetrics.ANNUALIZED_RETURN.value] = self._calculate_annualized_return()
        self.metrics[PerformanceMetrics.VOLATILITY.value] = self._calculate_volatility()
        self.metrics[PerformanceMetrics.CAGR.value] = self._calculate_cagr()
        
        # Risk-adjusted metrics
        self.metrics[PerformanceMetrics.SHARPE_RATIO.value] = self._calculate_sharpe_ratio()
        self.metrics[PerformanceMetrics.SORTINO_RATIO.value] = self._calculate_sortino_ratio()
        self.metrics[PerformanceMetrics.MAX_DRAWDOWN.value] = self._calculate_max_drawdown()
        self.metrics[PerformanceMetrics.CALMAR_RATIO.value] = self._calculate_calmar_ratio()
        self.metrics[PerformanceMetrics.ULCER_INDEX.value] = self._calculate_ulcer_index()
        self.metrics[PerformanceMetrics.PAIN_INDEX.value] = self._calculate_pain_index()
        self.metrics[PerformanceMetrics.VALUE_AT_RISK.value] = self._calculate_value_at_risk()
        self.metrics[PerformanceMetrics.CONDITIONAL_VALUE_AT_RISK.value] = self._calculate_conditional_value_at_risk()
        
        # Benchmark comparison metrics
        if self.benchmark_returns is not None:
            self.metrics[PerformanceMetrics.ALPHA.value] = self._calculate_alpha()
            self.metrics[PerformanceMetrics.BETA.value] = self._calculate_beta()
            self.metrics[PerformanceMetrics.INFORMATION_RATIO.value] = self._calculate_information_ratio()
        
        # Trade metrics
        if not self.trades.empty:
            self.metrics[PerformanceMetrics.WINNING_PERCENTAGE.value] = self._calculate_winning_percentage()
            self.metrics[PerformanceMetrics.PROFIT_FACTOR.value] = self._calculate_profit_factor()
            self.metrics[PerformanceMetrics.EXPECTANCY.value] = self._calculate_expectancy()
            self.metrics[PerformanceMetrics.AVERAGE_TRADE.value] = self._calculate_average_trade()
            self.metrics[PerformanceMetrics.AVERAGE_WIN.value] = self._calculate_average_win()
            self.metrics[PerformanceMetrics.AVERAGE_LOSS.value] = self._calculate_average_loss()
            self.metrics[PerformanceMetrics.WIN_LOSS_RATIO.value] = self._calculate_win_loss_ratio()
            self.metrics[PerformanceMetrics.MAX_CONSECUTIVE_WINS.value] = self._calculate_max_consecutive_wins()
            self.metrics[PerformanceMetrics.MAX_CONSECUTIVE_LOSSES.value] = self._calculate_max_consecutive_losses()
            self.metrics[PerformanceMetrics.RECOVERY_FACTOR.value] = self._calculate_recovery_factor()
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage.
        
        Returns:
            Total return percentage
        """
        if self.equity_curve.empty:
            return 0.0
        
        initial_equity = self.equity_curve["equity"].iloc[0]
        final_equity = self.equity_curve["equity"].iloc[-1]
        
        return ((final_equity / initial_equity) - 1) * 100
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return percentage.
        
        Returns:
            Annualized return percentage
        """
        if self.equity_curve.empty:
            return 0.0
        
        total_return = self._calculate_total_return() / 100  # Convert to decimal
        years = self._calculate_years()
        
        if years <= 0:
            return 0.0
        
        return (((1 + total_return) ** (1 / years)) - 1) * 100
    
    def _calculate_years(self) -> float:
        """Calculate number of years in the backtest.
        
        Returns:
            Number of years
        """
        if self.equity_curve.empty:
            return 0.0
        
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        days = (end_date - start_date).days
        
        return days / 365.25
    
    def _calculate_volatility(self) -> float:
        """Calculate return volatility (annualized).
        
        Returns:
            Annualized volatility percentage
        """
        if self.returns.empty:
            return 0.0
        
        return self.returns["returns"].std() * np.sqrt(self.annualization_factor) * 100
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio.
        
        Returns:
            Sharpe ratio
        """
        if self.returns.empty:
            return 0.0
        
        excess_returns = self.returns["returns"] - self.daily_risk_free_rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.annualization_factor)
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio.
        
        Returns:
            Sortino ratio
        """
        if self.returns.empty:
            return 0.0
        
        excess_returns = self.returns["returns"] - self.daily_risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(self.annualization_factor)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage.
        
        Returns:
            Maximum drawdown percentage
        """
        if self.equity_curve.empty:
            return 0.0
        
        # Calculate running maximum
        running_max = self.equity_curve["equity"].cummax()
        
        # Calculate drawdown percentage
        drawdown = (self.equity_curve["equity"] / running_max - 1) * 100
        
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio.
        
        Returns:
            Calmar ratio
        """
        annualized_return = self._calculate_annualized_return() / 100  # Convert to decimal
        max_drawdown = self._calculate_max_drawdown() / 100  # Convert to decimal
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_alpha(self) -> float:
        """Calculate Jensen's alpha.
        
        Returns:
            Alpha value
        """
        if self.returns.empty or self.benchmark_returns is None:
            return 0.0
        
        # Align returns with benchmark
        aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_data.empty:
            return 0.0
        
        # Calculate beta
        beta = self._calculate_beta()
        
        # Calculate alpha
        strategy_return = aligned_data["returns"].mean() * self.annualization_factor
        benchmark_return = aligned_data[aligned_data.columns[1]].mean() * self.annualization_factor
        
        return strategy_return - self.risk_free_rate - beta * (benchmark_return - self.risk_free_rate)
    
    def _calculate_beta(self) -> float:
        """Calculate beta to benchmark.
        
        Returns:
            Beta value
        """
        if self.returns.empty or self.benchmark_returns is None:
            return 0.0
        
        # Align returns with benchmark
        aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_data.empty:
            return 0.0
        
        # Calculate covariance and variance
        cov = aligned_data["returns"].cov(aligned_data[aligned_data.columns[1]])
        var = aligned_data[aligned_data.columns[1]].var()
        
        if var == 0:
            return 0.0
        
        return cov / var
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio.
        
        Returns:
            Information ratio
        """
        if self.returns.empty or self.benchmark_returns is None:
            return 0.0
        
        # Align returns with benchmark
        aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_data.empty:
            return 0.0
        
        # Calculate tracking error
        tracking_error = (aligned_data["returns"] - aligned_data[aligned_data.columns[1]]).std()
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate information ratio
        excess_return = aligned_data["returns"].mean() - aligned_data[aligned_data.columns[1]].mean()
        
        return excess_return / tracking_error * np.sqrt(self.annualization_factor)
    
    def _calculate_winning_percentage(self) -> float:
        """Calculate percentage of winning trades.
        
        Returns:
            Winning percentage
        """
        if self.trades.empty:
            return 0.0
        
        # Determine if each trade is a win or loss
        self.trades["profit"] = self.trades.apply(
            lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
            axis=1
        )
        
        # Count winning trades
        winning_trades = len(self.trades[self.trades["profit"] > 0])
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return 0.0
        
        return (winning_trades / total_trades) * 100
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor.
        
        Returns:
            Profit factor
        """
        if self.trades.empty:
            return 0.0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        # Sum profits and losses
        gross_profit = self.trades[self.trades["profit"] > 0]["profit"].sum()
        gross_loss = abs(self.trades[self.trades["profit"] < 0]["profit"].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_expectancy(self) -> float:
        """Calculate average trade expectancy.
        
        Returns:
            Expectancy value
        """
        if self.trades.empty:
            return 0.0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        # Calculate win rate and average win/loss
        win_rate = self._calculate_winning_percentage() / 100
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        
        if avg_loss == 0:
            return 0.0
        
        # Calculate R-multiple (average win / average loss)
        r_multiple = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calculate expectancy
        return (win_rate * r_multiple) - (1 - win_rate)
    
    def _calculate_average_trade(self) -> float:
        """Calculate average trade return.
        
        Returns:
            Average trade return
        """
        if self.trades.empty:
            return 0.0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        return self.trades["profit"].mean()
    
    def _calculate_average_win(self) -> float:
        """Calculate average winning trade.
        
        Returns:
            Average winning trade
        """
        if self.trades.empty:
            return 0.0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        winning_trades = self.trades[self.trades["profit"] > 0]
        
        if winning_trades.empty:
            return 0.0
        
        return winning_trades["profit"].mean()
    
    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade.
        
        Returns:
            Average losing trade (as a negative number)
        """
        if self.trades.empty:
            return 0.0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        losing_trades = self.trades[self.trades["profit"] < 0]
        
        if losing_trades.empty:
            return 0.0
        
        return losing_trades["profit"].mean()
    
    def _calculate_win_loss_ratio(self) -> float:
        """Calculate win/loss ratio.
        
        Returns:
            Win/loss ratio
        """
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return abs(avg_win / avg_loss)
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins.
        
        Returns:
            Maximum consecutive wins
        """
        if self.trades.empty:
            return 0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        # Create a list of wins (1) and losses (0)
        results = [1 if profit > 0 else 0 for profit in self.trades["profit"]]
        
        # Calculate maximum consecutive wins
        max_consecutive = 0
        current_consecutive = 0
        
        for result in results:
            if result == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses.
        
        Returns:
            Maximum consecutive losses
        """
        if self.trades.empty:
            return 0
        
        # Calculate profit for each trade
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1
            )
        
        # Create a list of wins (0) and losses (1)
        results = [1 if profit < 0 else 0 for profit in self.trades["profit"]]
        
        # Calculate maximum consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        
        for result in results:
            if result == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_recovery_factor(self) -> float:
        """Calculate recovery factor.
        
        Returns:
            Recovery factor
        """
        total_return = self._calculate_total_return() / 100  # Convert to decimal
        max_drawdown = self._calculate_max_drawdown() / 100  # Convert to decimal
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_drawdown
    
    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer index.
        
        Returns:
            Ulcer index
        """
        if self.equity_curve.empty:
            return 0.0
        
        # Calculate running maximum
        running_max = self.equity_curve["equity"].cummax()
        
        # Calculate percentage drawdown
        drawdown = (self.equity_curve["equity"] / running_max - 1) * 100
        
        # Calculate squared drawdowns
        squared_drawdowns = drawdown ** 2
        
        # Calculate Ulcer index
        return np.sqrt(squared_drawdowns.mean())
    
    def _calculate_pain_index(self) -> float:
        """Calculate Pain index.
        
        Returns:
            Pain index
        """
        if self.equity_curve.empty:
            return 0.0
        
        # Calculate running maximum
        running_max = self.equity_curve["equity"].cummax()
        
        # Calculate percentage drawdown
        drawdown = (self.equity_curve["equity"] / running_max - 1) * 100
        
        # Calculate Pain index (average drawdown)
        return abs(drawdown.mean())
    
    def _calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate (CAGR).
        
        Returns:
            CAGR percentage
        """
        if self.equity_curve.empty:
            return 0.0
        
        initial_equity = self.equity_curve["equity"].iloc[0]
        final_equity = self.equity_curve["equity"].iloc[-1]
        years = self._calculate_years()
        
        if years <= 0 or initial_equity <= 0:
            return 0.0
        
        return (((final_equity / initial_equity) ** (1 / years)) - 1) * 100
    
    def _calculate_value_at_risk(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Value at Risk percentage
        """
        if self.returns.empty:
            return 0.0
        
        # Calculate VaR
        return abs(np.percentile(self.returns["returns"] * 100, 100 * (1 - confidence)))
    
    def _calculate_conditional_value_at_risk(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Conditional Value at Risk percentage
        """
        if self.returns.empty:
            return 0.0
        
        # Calculate VaR
        var = np.percentile(self.returns["returns"], 100 * (1 - confidence))
        
        # Calculate CVaR (average of returns below VaR)
        cvar_returns = self.returns[self.returns["returns"] <= var]["returns"]
        
        if cvar_returns.empty:
            return 0.0
        
        return abs(cvar_returns.mean() * 100)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all calculated performance metrics.
        
        Returns:
            Dict with performance metrics
        """
        return self.metrics
    
    def get_metric(self, metric: Union[str, PerformanceMetrics]) -> float:
        """Get a specific performance metric.
        
        Args:
            metric: Metric name or PerformanceMetrics enum
            
        Returns:
            Metric value
        """
        if isinstance(metric, PerformanceMetrics):
            metric = metric.value
        
        return self.metrics.get(metric, 0.0)
    
    def get_summary(self) -> Dict[str, float]:
        """Get a summary of key performance metrics.
        
        Returns:
            Dict with key performance metrics
        """
        return {
            "total_return": self.get_metric(PerformanceMetrics.TOTAL_RETURN),
            "annualized_return": self.get_metric(PerformanceMetrics.ANNUALIZED_RETURN),
            "volatility": self.get_metric(PerformanceMetrics.VOLATILITY),
            "sharpe_ratio": self.get_metric(PerformanceMetrics.SHARPE_RATIO),
            "max_drawdown": self.get_metric(PerformanceMetrics.MAX_DRAWDOWN),
            "calmar_ratio": self.get_metric(PerformanceMetrics.CALMAR_RATIO),
            "winning_percentage": self.get_metric(PerformanceMetrics.WINNING_PERCENTAGE),
            "profit_factor": self.get_metric(PerformanceMetrics.PROFIT_FACTOR),
        }


class BenchmarkComparison:
    """Benchmark comparison for trading strategies.
    
    This class compares trading strategy performance with benchmark performance.
    """
    
    def __init__(
        self,
        strategy_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ):
        """Initialize the benchmark comparison.
        
        Args:
            strategy_returns: DataFrame with strategy returns (index=timestamp, columns=[returns])
            benchmark_returns: DataFrame with benchmark returns (index=timestamp, columns=[returns])
            risk_free_rate: Annual risk-free rate (default: 0.0)
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Align returns
        self.aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        if not self.aligned_data.empty:
            self.aligned_data.columns = ["strategy", "benchmark"]
        
        # Calculate daily risk-free rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1
        
        # Initialize metrics dictionary
        self.metrics = {}
        
        # Calculate all metrics
        self._calculate_all_metrics()
        
        logger.info("Initialized benchmark comparison")
    
    def _calculate_all_metrics(self) -> None:
        """Calculate all benchmark comparison metrics."""
        if self.aligned_data.empty:
            return
        
        # Calculate alpha and beta
        self.metrics["alpha"] = self._calculate_alpha()
        self.metrics["beta"] = self._calculate_beta()
        
        # Calculate correlation
        self.metrics["correlation"] = self._calculate_correlation()
        
        # Calculate tracking error
        self.metrics["tracking_error"] = self._calculate_tracking_error()
        
        # Calculate information ratio
        self.metrics["information_ratio"] = self._calculate_information_ratio()
        
        # Calculate up/down capture ratios
        self.metrics["up_capture"] = self._calculate_up_capture()
        self.metrics["down_capture"] = self._calculate_down_capture()
        
        # Calculate outperformance percentage
        self.metrics["outperformance_percentage"] = self._calculate_outperformance_percentage()
    
    def _calculate_alpha(self) -> float:
        """Calculate Jensen's alpha.
        
        Returns:
            Alpha value
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Calculate beta
        beta = self._calculate_beta()
        
        # Calculate alpha
        strategy_return = self.aligned_data["strategy"].mean() * 252
        benchmark_return = self.aligned_data["benchmark"].mean() * 252
        
        return strategy_return - self.risk_free_rate - beta * (benchmark_return - self.risk_free_rate)
    
    def _calculate_beta(self) -> float:
        """Calculate beta to benchmark.
        
        Returns:
            Beta value
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Calculate covariance and variance
        cov = self.aligned_data["strategy"].cov(self.aligned_data["benchmark"])
        var = self.aligned_data["benchmark"].var()
        
        if var == 0:
            return 0.0
        
        return cov / var
    
    def _calculate_correlation(self) -> float:
        """Calculate correlation with benchmark.
        
        Returns:
            Correlation coefficient
        """
        if self.aligned_data.empty:
            return 0.0
        
        return self.aligned_data["strategy"].corr(self.aligned_data["benchmark"])
    
    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error.
        
        Returns:
            Tracking error (annualized)
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Calculate tracking difference
        tracking_diff = self.aligned_data["strategy"] - self.aligned_data["benchmark"]
        
        # Calculate tracking error
        return tracking_diff.std() * np.sqrt(252) * 100
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio.
        
        Returns:
            Information ratio
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Calculate tracking error
        tracking_error = self._calculate_tracking_error() / 100  # Convert to decimal
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate excess return
        excess_return = (self.aligned_data["strategy"].mean() - self.aligned_data["benchmark"].mean()) * 252
        
        return excess_return / tracking_error
    
    def _calculate_up_capture(self) -> float:
        """Calculate up-market capture ratio.
        
        Returns:
            Up-market capture ratio
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Filter for up-market periods
        up_market = self.aligned_data[self.aligned_data["benchmark"] > 0]
        
        if up_market.empty:
            return 0.0
        
        # Calculate up-market capture ratio
        strategy_up_return = (up_market["strategy"] + 1).prod() - 1
        benchmark_up_return = (up_market["benchmark"] + 1).prod() - 1
        
        if benchmark_up_return == 0:
            return float('inf') if strategy_up_return > 0 else 0.0
        
        return (strategy_up_return / benchmark_up_return) * 100
    
    def _calculate_down_capture(self) -> float:
        """Calculate down-market capture ratio.
        
        Returns:
            Down-market capture ratio
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Filter for down-market periods
        down_market = self.aligned_data[self.aligned_data["benchmark"] < 0]
        
        if down_market.empty:
            return 0.0
        
        # Calculate down-market capture ratio
        strategy_down_return = (down_market["strategy"] + 1).prod() - 1
        benchmark_down_return = (down_market["benchmark"] + 1).prod() - 1
        
        if benchmark_down_return == 0:
            return 0.0
        
        return (strategy_down_return / benchmark_down_return) * 100
    
    def _calculate_outperformance_percentage(self) -> float:
        """Calculate percentage of periods outperforming benchmark.
        
        Returns:
            Outperformance percentage
        """
        if self.aligned_data.empty:
            return 0.0
        
        # Calculate outperformance periods
        outperformance = (self.aligned_data["strategy"] > self.aligned_data["benchmark"]).sum()
        total_periods = len(self.aligned_data)
        
        if total_periods == 0:
            return 0.0
        
        return (outperformance / total_periods) * 100
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all calculated benchmark comparison metrics.
        
        Returns:
            Dict with benchmark comparison metrics
        """
        return self.metrics
    
    def get_metric(self, metric: str) -> float:
        """Get a specific benchmark comparison metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Metric value
        """
        return self.metrics.get(metric, 0.0)
    
    def get_summary(self) -> Dict[str, float]:
        """Get a summary of key benchmark comparison metrics.
        
        Returns:
            Dict with key benchmark comparison metrics
        """
        return {
            "alpha": self.get_metric("alpha"),
            "beta": self.get_metric("beta"),
            "correlation": self.get_metric("correlation"),
            "information_ratio": self.get_metric("information_ratio"),
            "up_capture": self.get_metric("up_capture"),
            "down_capture": self.get_metric("down_capture"),
            "outperformance_percentage": self.get_metric("outperformance_percentage"),
        }