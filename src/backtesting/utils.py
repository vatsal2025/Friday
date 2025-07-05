"""Utilities module for backtesting framework.

This module provides utility functions and helper classes for the backtesting framework.
"""

import os
import json
import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class TimeFrame(Enum):
    """Time frames for resampling data."""
    TICK = "tick"
    SECOND = "1S"
    MINUTE = "1T"
    FIVE_MINUTE = "5T"
    FIFTEEN_MINUTE = "15T"
    THIRTY_MINUTE = "30T"
    HOUR = "1H"
    FOUR_HOUR = "4H"
    DAY = "1D"
    WEEK = "1W"
    MONTH = "1M"
    QUARTER = "3M"
    YEAR = "1Y"


class TradeDirection(Enum):
    """Trade directions."""
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade statuses."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"


class Trade:
    """Trade class for tracking individual trades."""
    
    def __init__(
        self,
        symbol: str,
        direction: Union[str, TradeDirection],
        entry_time: pd.Timestamp,
        entry_price: float,
        quantity: float,
        entry_order_id: Optional[str] = None,
        exit_time: Optional[pd.Timestamp] = None,
        exit_price: Optional[float] = None,
        exit_order_id: Optional[str] = None,
        status: Union[str, TradeStatus] = TradeStatus.OPEN,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a trade.
        
        Args:
            symbol: Symbol of the trade
            direction: Direction of the trade (long or short)
            entry_time: Entry time
            entry_price: Entry price
            quantity: Quantity of the trade
            entry_order_id: Entry order ID (default: None)
            exit_time: Exit time (default: None)
            exit_price: Exit price (default: None)
            exit_order_id: Exit order ID (default: None)
            status: Status of the trade (default: OPEN)
            stop_loss: Stop loss price (default: None)
            take_profit: Take profit price (default: None)
            commission: Commission paid (default: 0.0)
            slippage: Slippage incurred (default: 0.0)
            tags: List of tags for the trade (default: None)
            metadata: Additional metadata for the trade (default: None)
        """
        # Convert direction to TradeDirection if it's a string
        if isinstance(direction, str):
            try:
                direction = TradeDirection(direction.lower())
            except ValueError:
                raise ValueError(f"Invalid trade direction: {direction}")
        
        # Convert status to TradeStatus if it's a string
        if isinstance(status, str):
            try:
                status = TradeStatus(status.lower())
            except ValueError:
                raise ValueError(f"Invalid trade status: {status}")
        
        self.id = f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}_{direction.value}"
        self.symbol = symbol
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_order_id = entry_order_id
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_order_id = exit_order_id
        self.status = status
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        self.slippage = slippage
        self.tags = tags or []
        self.metadata = metadata or {}
        
        # Calculate trade metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate trade metrics."""
        # Initialize metrics
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.duration = None
        self.annualized_return = None
        self.risk_reward_ratio = None
        self.total_cost = self.commission + self.slippage
        
        # Calculate metrics for closed trades
        if self.status == TradeStatus.CLOSED and self.exit_price is not None:
            # Calculate P&L
            if self.direction == TradeDirection.LONG:
                self.pnl = (self.exit_price - self.entry_price) * self.quantity - self.total_cost
                self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100
            else:  # SHORT
                self.pnl = (self.entry_price - self.exit_price) * self.quantity - self.total_cost
                self.pnl_pct = (self.entry_price / self.exit_price - 1) * 100
            
            # Calculate duration
            if self.exit_time is not None:
                self.duration = self.exit_time - self.entry_time
                
                # Calculate annualized return
                days = self.duration.total_seconds() / (24 * 60 * 60)
                if days > 0:
                    self.annualized_return = ((1 + self.pnl_pct / 100) ** (365 / days) - 1) * 100
            
            # Calculate risk-reward ratio
            if self.stop_loss is not None and self.direction == TradeDirection.LONG:
                risk = self.entry_price - self.stop_loss
                reward = self.exit_price - self.entry_price
                if risk > 0 and reward != 0:
                    self.risk_reward_ratio = abs(reward / risk)
            elif self.stop_loss is not None and self.direction == TradeDirection.SHORT:
                risk = self.stop_loss - self.entry_price
                reward = self.entry_price - self.exit_price
                if risk > 0 and reward != 0:
                    self.risk_reward_ratio = abs(reward / risk)
    
    def close(self, exit_time: pd.Timestamp, exit_price: float, exit_order_id: Optional[str] = None) -> None:
        """Close the trade.
        
        Args:
            exit_time: Exit time
            exit_price: Exit price
            exit_order_id: Exit order ID (default: None)
        """
        if self.status != TradeStatus.OPEN:
            raise ValueError(f"Cannot close trade with status {self.status}")
        
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_order_id = exit_order_id
        self.status = TradeStatus.CLOSED
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def cancel(self) -> None:
        """Cancel the trade."""
        if self.status != TradeStatus.OPEN:
            raise ValueError(f"Cannot cancel trade with status {self.status}")
        
        self.status = TradeStatus.CANCELED
    
    def update_stop_loss(self, price: float) -> None:
        """Update the stop loss price.
        
        Args:
            price: New stop loss price
        """
        self.stop_loss = price
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def update_take_profit(self, price: float) -> None:
        """Update the take profit price.
        
        Args:
            price: New take profit price
        """
        self.take_profit = price
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the trade.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the trade.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trade.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trade to a dictionary.
        
        Returns:
            Dictionary representation of the trade
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_order_id": self.entry_order_id,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_order_id": self.exit_order_id,
            "status": self.status.value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "commission": self.commission,
            "slippage": self.slippage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "duration": str(self.duration) if self.duration else None,
            "annualized_return": self.annualized_return,
            "risk_reward_ratio": self.risk_reward_ratio,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create a trade from a dictionary.
        
        Args:
            data: Dictionary representation of the trade
            
        Returns:
            Trade object
        """
        # Convert ISO format strings to timestamps
        entry_time = pd.Timestamp(data["entry_time"])
        exit_time = pd.Timestamp(data["exit_time"]) if data.get("exit_time") else None
        
        # Create trade object
        trade = cls(
            symbol=data["symbol"],
            direction=data["direction"],
            entry_time=entry_time,
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            entry_order_id=data.get("entry_order_id"),
            exit_time=exit_time,
            exit_price=data.get("exit_price"),
            exit_order_id=data.get("exit_order_id"),
            status=data["status"],
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            commission=data.get("commission", 0.0),
            slippage=data.get("slippage", 0.0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
        
        return trade


class TradeAnalyzer:
    """Trade analyzer for analyzing trade performance."""
    
    def __init__(self, trades: List[Trade] = None):
        """Initialize the trade analyzer.
        
        Args:
            trades: List of trades to analyze (default: None)
        """
        self.trades = trades or []
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the analyzer.
        
        Args:
            trade: Trade to add
        """
        self.trades.append(trade)
    
    def add_trades(self, trades: List[Trade]) -> None:
        """Add multiple trades to the analyzer.
        
        Args:
            trades: Trades to add
        """
        self.trades.extend(trades)
    
    def get_trades(self, status: Union[str, TradeStatus] = None) -> List[Trade]:
        """Get trades with a specific status.
        
        Args:
            status: Trade status to filter by (default: None, returns all trades)
            
        Returns:
            List of trades
        """
        if status is None:
            return self.trades
        
        # Convert status to TradeStatus if it's a string
        if isinstance(status, str):
            try:
                status = TradeStatus(status.lower())
            except ValueError:
                raise ValueError(f"Invalid trade status: {status}")
        
        return [trade for trade in self.trades if trade.status == status]
    
    def get_closed_trades(self) -> List[Trade]:
        """Get closed trades.
        
        Returns:
            List of closed trades
        """
        return self.get_trades(TradeStatus.CLOSED)
    
    def get_open_trades(self) -> List[Trade]:
        """Get open trades.
        
        Returns:
            List of open trades
        """
        return self.get_trades(TradeStatus.OPEN)
    
    def get_canceled_trades(self) -> List[Trade]:
        """Get canceled trades.
        
        Returns:
            List of canceled trades
        """
        return self.get_trades(TradeStatus.CANCELED)
    
    def get_winning_trades(self) -> List[Trade]:
        """Get winning trades.
        
        Returns:
            List of winning trades
        """
        return [trade for trade in self.get_closed_trades() if trade.pnl > 0]
    
    def get_losing_trades(self) -> List[Trade]:
        """Get losing trades.
        
        Returns:
            List of losing trades
        """
        return [trade for trade in self.get_closed_trades() if trade.pnl <= 0]
    
    def get_long_trades(self) -> List[Trade]:
        """Get long trades.
        
        Returns:
            List of long trades
        """
        return [trade for trade in self.trades if trade.direction == TradeDirection.LONG]
    
    def get_short_trades(self) -> List[Trade]:
        """Get short trades.
        
        Returns:
            List of short trades
        """
        return [trade for trade in self.trades if trade.direction == TradeDirection.SHORT]
    
    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get trades for a specific symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of trades for the symbol
        """
        return [trade for trade in self.trades if trade.symbol == symbol]
    
    def get_trades_by_tag(self, tag: str) -> List[Trade]:
        """Get trades with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of trades with the tag
        """
        return [trade for trade in self.trades if tag in trade.tags]
    
    def get_trades_by_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Trade]:
        """Get trades within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trades within the date range
        """
        return [trade for trade in self.trades if start_date <= trade.entry_time <= end_date]
    
    def get_total_pnl(self) -> float:
        """Get the total P&L of all closed trades.
        
        Returns:
            Total P&L
        """
        return sum(trade.pnl for trade in self.get_closed_trades())
    
    def get_win_rate(self) -> float:
        """Get the win rate of closed trades.
        
        Returns:
            Win rate (percentage)
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        winning_trades = self.get_winning_trades()
        return len(winning_trades) / len(closed_trades) * 100
    
    def get_average_pnl(self) -> float:
        """Get the average P&L of closed trades.
        
        Returns:
            Average P&L
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        return self.get_total_pnl() / len(closed_trades)
    
    def get_average_winning_trade(self) -> float:
        """Get the average P&L of winning trades.
        
        Returns:
            Average P&L of winning trades
        """
        winning_trades = self.get_winning_trades()
        if not winning_trades:
            return 0.0
        
        return sum(trade.pnl for trade in winning_trades) / len(winning_trades)
    
    def get_average_losing_trade(self) -> float:
        """Get the average P&L of losing trades.
        
        Returns:
            Average P&L of losing trades
        """
        losing_trades = self.get_losing_trades()
        if not losing_trades:
            return 0.0
        
        return sum(trade.pnl for trade in losing_trades) / len(losing_trades)
    
    def get_profit_factor(self) -> float:
        """Get the profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor
        """
        winning_trades = self.get_winning_trades()
        losing_trades = self.get_losing_trades()
        
        gross_profit = sum(trade.pnl for trade in winning_trades)
        gross_loss = abs(sum(trade.pnl for trade in losing_trades))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_expectancy(self) -> float:
        """Get the expectancy (average P&L per trade).
        
        Returns:
            Expectancy
        """
        return self.get_average_pnl()
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Get the Sharpe ratio of closed trades.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sharpe ratio
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        # Calculate returns
        returns = [trade.pnl_pct / 100 for trade in closed_trades]
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        # Annualize Sharpe ratio (assuming 252 trading days per year)
        sharpe_ratio *= np.sqrt(252)
        
        return sharpe_ratio
    
    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Get the Sortino ratio of closed trades.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sortino ratio
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        # Calculate returns
        returns = [trade.pnl_pct / 100 for trade in closed_trades]
        
        # Calculate mean and downside deviation
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if mean_return > risk_free_rate else 0.0
        
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
        
        if downside_deviation == 0:
            return 0.0
        
        # Calculate Sortino ratio
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        
        # Annualize Sortino ratio (assuming 252 trading days per year)
        sortino_ratio *= np.sqrt(252)
        
        return sortino_ratio
    
    def get_max_drawdown(self) -> float:
        """Get the maximum drawdown of closed trades.
        
        Returns:
            Maximum drawdown (percentage)
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        # Sort trades by entry time
        sorted_trades = sorted(closed_trades, key=lambda x: x.entry_time)
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum([trade.pnl for trade in sorted_trades])
        
        # Calculate drawdown
        peak = 0
        drawdown = 0
        max_drawdown = 0
        
        for i, pnl in enumerate(cumulative_pnl):
            if pnl > peak:
                peak = pnl
            
            drawdown = peak - pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Convert to percentage
        if peak > 0:
            max_drawdown_pct = max_drawdown / peak * 100
        else:
            max_drawdown_pct = 0.0
        
        return max_drawdown_pct
    
    def get_average_duration(self) -> pd.Timedelta:
        """Get the average duration of closed trades.
        
        Returns:
            Average duration
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return pd.Timedelta(0)
        
        durations = [trade.duration for trade in closed_trades if trade.duration is not None]
        if not durations:
            return pd.Timedelta(0)
        
        return sum(durations, pd.Timedelta(0)) / len(durations)
    
    def get_average_risk_reward_ratio(self) -> float:
        """Get the average risk-reward ratio of closed trades.
        
        Returns:
            Average risk-reward ratio
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            return 0.0
        
        ratios = [trade.risk_reward_ratio for trade in closed_trades if trade.risk_reward_ratio is not None]
        if not ratios:
            return 0.0
        
        return sum(ratios) / len(ratios)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of trade performance.
        
        Returns:
            Dictionary with trade performance metrics
        """
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(self.get_closed_trades()),
            "open_trades": len(self.get_open_trades()),
            "canceled_trades": len(self.get_canceled_trades()),
            "winning_trades": len(self.get_winning_trades()),
            "losing_trades": len(self.get_losing_trades()),
            "long_trades": len(self.get_long_trades()),
            "short_trades": len(self.get_short_trades()),
            "total_pnl": self.get_total_pnl(),
            "win_rate": self.get_win_rate(),
            "average_pnl": self.get_average_pnl(),
            "average_winning_trade": self.get_average_winning_trade(),
            "average_losing_trade": self.get_average_losing_trade(),
            "profit_factor": self.get_profit_factor(),
            "expectancy": self.get_expectancy(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "max_drawdown": self.get_max_drawdown(),
            "average_duration": str(self.get_average_duration()),
            "average_risk_reward_ratio": self.get_average_risk_reward_ratio(),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to a DataFrame.
        
        Returns:
            DataFrame with trade data
        """
        if not self.trades:
            return pd.DataFrame()
        
        # Convert trades to dictionaries
        trade_dicts = [trade.to_dict() for trade in self.trades]
        
        # Create DataFrame
        df = pd.DataFrame(trade_dicts)
        
        # Convert string timestamps to datetime
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
        
        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(df["exit_time"])
        
        return df
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (10, 6), title: str = "Equity Curve") -> plt.Figure:
        """Plot the equity curve.
        
        Args:
            figsize: Figure size (default: (10, 6))
            title: Plot title (default: "Equity Curve")
            
        Returns:
            Matplotlib figure
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")
            ax.text(0.5, 0.5, "No closed trades", ha="center", va="center", transform=ax.transAxes)
            return fig
        
        # Sort trades by entry time
        sorted_trades = sorted(closed_trades, key=lambda x: x.entry_time)
        
        # Calculate cumulative P&L
        dates = [trade.exit_time for trade in sorted_trades]
        cumulative_pnl = np.cumsum([trade.pnl for trade in sorted_trades])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(dates, cumulative_pnl, label="Equity Curve")
        
        # Add drawdown shading
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        ax.fill_between(dates, cumulative_pnl, peak, alpha=0.3, color="red", label="Drawdown")
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        
        # Add summary statistics
        summary = self.get_summary()
        stats_text = (
            f"Total P&L: ${summary['total_pnl']:.2f}\n"
            f"Win Rate: {summary['win_rate']:.2f}%\n"
            f"Profit Factor: {summary['profit_factor']:.2f}\n"
            f"Max Drawdown: {summary['max_drawdown']:.2f}%"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot_drawdown(self, figsize: Tuple[int, int] = (10, 6), title: str = "Drawdown") -> plt.Figure:
        """Plot the drawdown.
        
        Args:
            figsize: Figure size (default: (10, 6))
            title: Plot title (default: "Drawdown")
            
        Returns:
            Matplotlib figure
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown (%)")
            ax.text(0.5, 0.5, "No closed trades", ha="center", va="center", transform=ax.transAxes)
            return fig
        
        # Sort trades by entry time
        sorted_trades = sorted(closed_trades, key=lambda x: x.entry_time)
        
        # Calculate cumulative P&L
        dates = [trade.exit_time for trade in sorted_trades]
        cumulative_pnl = np.cumsum([trade.pnl for trade in sorted_trades])
        
        # Calculate drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / np.maximum(peak, 1e-10) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdown
        ax.fill_between(dates, 0, drawdown, color="red", alpha=0.5)
        ax.plot(dates, drawdown, color="red", label="Drawdown")
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}%"))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        
        # Add max drawdown text
        max_dd = np.max(drawdown)
        max_dd_date = dates[np.argmax(drawdown)]
        ax.text(
            0.02, 0.98,
            f"Max Drawdown: {max_dd:.2f}% on {max_dd_date.strftime('%Y-%m-%d')}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot_monthly_returns(self, figsize: Tuple[int, int] = (10, 6), title: str = "Monthly Returns") -> plt.Figure:
        """Plot monthly returns.
        
        Args:
            figsize: Figure size (default: (10, 6))
            title: Plot title (default: "Monthly Returns")
            
        Returns:
            Matplotlib figure
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.set_xlabel("Month")
            ax.set_ylabel("Return (%)")
            ax.text(0.5, 0.5, "No closed trades", ha="center", va="center", transform=ax.transAxes)
            return fig
        
        # Create DataFrame with trade data
        df = self.to_dataframe()
        
        # Group by month and calculate returns
        df["month"] = df["exit_time"].dt.to_period("M")
        monthly_returns = df.groupby("month")["pnl"].sum().reset_index()
        monthly_returns["month_str"] = monthly_returns["month"].dt.strftime("%Y-%m")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot monthly returns
        bars = ax.bar(
            monthly_returns["month_str"],
            monthly_returns["pnl"],
            color=["green" if x > 0 else "red" for x in monthly_returns["pnl"]],
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.1 if height > 0 else -0.1),
                f"${height:.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                rotation=0,
                fontsize=8,
            )
        
        # Format x-axis
        plt.xticks(rotation=45, ha="right")
        
        # Format y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
        
        # Add grid
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Return ($)")
        
        # Add summary statistics
        positive_months = sum(monthly_returns["pnl"] > 0)
        negative_months = sum(monthly_returns["pnl"] <= 0)
        total_months = len(monthly_returns)
        
        stats_text = (
            f"Positive Months: {positive_months} ({positive_months / total_months * 100:.2f}%)\n"
            f"Negative Months: {negative_months} ({negative_months / total_months * 100:.2f}%)\n"
            f"Best Month: ${monthly_returns['pnl'].max():.2f}\n"
            f"Worst Month: ${monthly_returns['pnl'].min():.2f}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot_trade_distribution(self, figsize: Tuple[int, int] = (10, 6), title: str = "Trade P&L Distribution") -> plt.Figure:
        """Plot the distribution of trade P&Ls.
        
        Args:
            figsize: Figure size (default: (10, 6))
            title: Plot title (default: "Trade P&L Distribution")
            
        Returns:
            Matplotlib figure
        """
        closed_trades = self.get_closed_trades()
        if not closed_trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title)
            ax.set_xlabel("P&L ($)")
            ax.set_ylabel("Frequency")
            ax.text(0.5, 0.5, "No closed trades", ha="center", va="center", transform=ax.transAxes)
            return fig
        
        # Get P&Ls
        pnls = [trade.pnl for trade in closed_trades]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        n, bins, patches = ax.hist(pnls, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        
        # Color positive and negative bars
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor("green")
            else:
                patch.set_facecolor("red")
        
        # Add normal distribution curve
        mu = np.mean(pnls)
        sigma = np.std(pnls)
        x = np.linspace(min(pnls), max(pnls), 100)
        y = len(pnls) * (bins[1] - bins[0]) * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        ax.plot(x, y, "k--", linewidth=1.5, label="Normal Distribution")
        
        # Format x-axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        
        # Add summary statistics
        stats_text = (
            f"Mean: ${mu:.2f}\n"
            f"Std Dev: ${sigma:.2f}\n"
            f"Min: ${min(pnls):.2f}\n"
            f"Max: ${max(pnls):.2f}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        
        return fig


def calculate_returns(prices: Union[pd.Series, np.ndarray], method: str = "simple") -> np.ndarray:
    """Calculate returns from prices.
    
    Args:
        prices: Price series
        method: Return calculation method (default: "simple", options: "simple", "log")
        
    Returns:
        Returns array
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if method.lower() == "simple":
        returns = np.diff(prices) / prices[:-1]
    elif method.lower() == "log":
        returns = np.diff(np.log(prices))
    else:
        raise ValueError(f"Invalid return calculation method: {method}")
    
    return returns


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily returns)
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate mean and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    
    # Annualize Sharpe ratio
    sharpe_ratio *= np.sqrt(periods_per_year)
    
    return sharpe_ratio


def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily returns)
        
    Returns:
        Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate mean and downside deviation
    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    
    if downside_deviation == 0:
        return 0.0
    
    # Calculate Sortino ratio
    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
    
    # Annualize Sortino ratio
    sortino_ratio *= np.sqrt(periods_per_year)
    
    return sortino_ratio


def calculate_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> Tuple[float, int, int]:
    """Calculate maximum drawdown.
    
    Args:
        equity_curve: Equity curve
        
    Returns:
        Tuple of (maximum drawdown, peak index, valley index)
    """
    if isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdown)
    valley_idx = np.argmax(drawdown)
    peak_idx = np.argmax(equity_curve[:valley_idx])
    
    return max_drawdown, peak_idx, valley_idx


def calculate_cagr(equity_curve: Union[pd.Series, np.ndarray], days: int) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        equity_curve: Equity curve
        days: Number of days in the equity curve
        
    Returns:
        CAGR
    """
    if isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate CAGR
    start_value = equity_curve[0]
    end_value = equity_curve[-1]
    
    if start_value <= 0:
        return 0.0
    
    years = days / 365.0
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    return cagr


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio.
    
    Args:
        cagr: Compound Annual Growth Rate
        max_drawdown: Maximum drawdown
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float('inf') if cagr > 0 else 0.0
    
    return cagr / max_drawdown


def calculate_omega_ratio(returns: Union[pd.Series, np.ndarray], threshold: float = 0.0) -> float:
    """Calculate Omega ratio.
    
    Args:
        returns: Return series
        threshold: Return threshold (default: 0.0)
        
    Returns:
        Omega ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Separate returns above and below threshold
    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns < threshold]
    
    if len(returns_below) == 0 or np.sum(returns_below) == 0:
        return float('inf') if len(returns_above) > 0 else 0.0
    
    # Calculate Omega ratio
    omega_ratio = np.sum(returns_above) / np.sum(returns_below)
    
    return omega_ratio


def calculate_var(returns: Union[pd.Series, np.ndarray], alpha: float = 0.05, method: str = "historical") -> float:
    """Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        alpha: Significance level (default: 0.05 for 95% VaR)
        method: Calculation method (default: "historical", options: "historical", "gaussian")
        
    Returns:
        VaR
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if method.lower() == "historical":
        # Historical VaR
        var = np.percentile(returns, alpha * 100)
    elif method.lower() == "gaussian":
        # Gaussian VaR
        mean = np.mean(returns)
        std = np.std(returns)
        var = mean + std * np.percentile(np.random.standard_normal(10000), alpha * 100)
    else:
        raise ValueError(f"Invalid VaR calculation method: {method}")
    
    return var


def calculate_cvar(returns: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR).
    
    Args:
        returns: Return series
        alpha: Significance level (default: 0.05 for 95% CVaR)
        
    Returns:
        CVaR
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate VaR
    var = calculate_var(returns, alpha, method="historical")
    
    # Calculate CVaR
    cvar = np.mean(returns[returns <= var])
    
    return cvar


def calculate_beta(returns: Union[pd.Series, np.ndarray], benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate beta.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        
    Returns:
        Beta
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    # Ensure returns have the same length
    min_length = min(len(returns), len(benchmark_returns))
    returns = returns[-min_length:]
    benchmark_returns = benchmark_returns[-min_length:]
    
    # Calculate covariance and variance
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    
    if variance == 0:
        return 0.0
    
    # Calculate beta
    beta = covariance / variance
    
    return beta


def calculate_alpha(returns: Union[pd.Series, np.ndarray], benchmark_returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0) -> float:
    """Calculate alpha.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Alpha
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    # Ensure returns have the same length
    min_length = min(len(returns), len(benchmark_returns))
    returns = returns[-min_length:]
    benchmark_returns = benchmark_returns[-min_length:]
    
    # Calculate beta
    beta = calculate_beta(returns, benchmark_returns)
    
    # Calculate alpha
    mean_return = np.mean(returns)
    mean_benchmark_return = np.mean(benchmark_returns)
    alpha = mean_return - risk_free_rate - beta * (mean_benchmark_return - risk_free_rate)
    
    return alpha


def calculate_information_ratio(returns: Union[pd.Series, np.ndarray], benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate information ratio.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        
    Returns:
        Information ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    # Ensure returns have the same length
    min_length = min(len(returns), len(benchmark_returns))
    returns = returns[-min_length:]
    benchmark_returns = benchmark_returns[-min_length:]
    
    # Calculate tracking error
    tracking_error = np.std(returns - benchmark_returns)
    
    if tracking_error == 0:
        return 0.0
    
    # Calculate information ratio
    information_ratio = np.mean(returns - benchmark_returns) / tracking_error
    
    return information_ratio


def calculate_treynor_ratio(returns: Union[pd.Series, np.ndarray], benchmark_returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0) -> float:
    """Calculate Treynor ratio.
    
    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Treynor ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    # Calculate beta
    beta = calculate_beta(returns, benchmark_returns)
    
    if beta == 0:
        return 0.0
    
    # Calculate Treynor ratio
    mean_return = np.mean(returns)
    treynor_ratio = (mean_return - risk_free_rate) / beta
    
    return treynor_ratio


def resample_ohlcv(df: pd.DataFrame, timeframe: Union[str, TimeFrame], price_col: str = "close") -> pd.DataFrame:
    """Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe
        price_col: Price column name (default: "close")
        
    Returns:
        Resampled DataFrame
    """
    # Convert timeframe to string if it's a TimeFrame enum
    if isinstance(timeframe, TimeFrame):
        timeframe = timeframe.value
    
    # Check if DataFrame has an index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Check for required columns
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # If only price column is available, create OHLCV data
        if price_col in df.columns:
            df = df.copy()
            df["open"] = df[price_col]
            df["high"] = df[price_col]
            df["low"] = df[price_col]
            df["close"] = df[price_col]
            df["volume"] = 0
        else:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Resample data
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    
    return resampled


def calculate_drawdowns(equity_curve: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
    """Calculate drawdowns from an equity curve.
    
    Args:
        equity_curve: Equity curve
        
    Returns:
        DataFrame with drawdown information
    """
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    
    # Calculate drawdown
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    drawdown_pct = drawdown * 100
    
    # Find drawdown periods
    is_drawdown = drawdown > 0
    starts = is_drawdown.ne(is_drawdown.shift()).cumsum()
    
    # Group by drawdown periods
    grouped = drawdown_pct.groupby(starts)
    
    # Calculate drawdown statistics
    result = []
    
    for _, group in grouped.groups.items():
        if drawdown_pct.iloc[group].max() > 0:  # Only include actual drawdowns
            start_idx = group[0]
            end_idx = group[-1]
            max_idx = drawdown_pct.iloc[group].idxmax()
            
            # Find recovery date (if any)
            recovery_idx = None
            if end_idx < len(equity_curve) - 1 and drawdown_pct.iloc[end_idx + 1] == 0:
                # Find the first index after the drawdown where the equity curve reaches a new peak
                for i in range(end_idx + 1, len(equity_curve)):
                    if equity_curve.iloc[i] >= peak.iloc[max_idx]:
                        recovery_idx = equity_curve.index[i]
                        break
            
            result.append({
                "start_date": equity_curve.index[start_idx],
                "valley_date": max_idx,
                "end_date": equity_curve.index[end_idx],
                "recovery_date": recovery_idx,
                "drawdown_pct": drawdown_pct.iloc[group].max(),
                "drawdown_value": (peak.iloc[max_idx] - equity_curve.iloc[max_idx]),
                "duration": equity_curve.index[end_idx] - equity_curve.index[start_idx],
                "recovery_duration": (recovery_idx - equity_curve.index[end_idx]) if recovery_idx else None,
            })
    
    # Create DataFrame and sort by drawdown percentage
    df = pd.DataFrame(result)
    if not df.empty:
        df = df.sort_values("drawdown_pct", ascending=False).reset_index(drop=True)
    
    return df


def calculate_rolling_metrics(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """Calculate rolling performance metrics.
    
    Args:
        returns: Return series
        window: Rolling window size (default: 20)
        
    Returns:
        DataFrame with rolling metrics
    """
    # Calculate rolling metrics
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
    rolling_drawdown = returns.rolling(window=window).apply(lambda x: calculate_max_drawdown(1 + x.cumsum())[0])
    rolling_win_rate = returns.rolling(window=window).apply(lambda x: np.sum(x > 0) / len(x) * 100)
    
    # Create DataFrame with rolling metrics
    df = pd.DataFrame({
        "mean": rolling_mean,
        "std": rolling_std,
        "sharpe": rolling_sharpe,
        "drawdown": rolling_drawdown,
        "win_rate": rolling_win_rate,
    })
    
    return df


def calculate_trade_statistics(trades: List[Trade]) -> Dict[str, Any]:
    """Calculate trade statistics.
    
    Args:
        trades: List of trades
        
    Returns:
        Dictionary with trade statistics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "average_winning_trade": 0.0,
            "average_losing_trade": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_duration": None,
            "max_consecutive_winners": 0,
            "max_consecutive_losers": 0,
        }
    
    # Filter closed trades
    closed_trades = [trade for trade in trades if trade.status == TradeStatus.CLOSED]
    
    if not closed_trades:
        return {
            "total_trades": len(trades),
            "closed_trades": 0,
            "open_trades": len([t for t in trades if t.status == TradeStatus.OPEN]),
            "canceled_trades": len([t for t in trades if t.status == TradeStatus.CANCELED]),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "average_winning_trade": 0.0,
            "average_losing_trade": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_duration": None,
            "max_consecutive_winners": 0,
            "max_consecutive_losers": 0,
        }
    
    # Calculate basic statistics
    winning_trades = [trade for trade in closed_trades if trade.pnl > 0]
    losing_trades = [trade for trade in closed_trades if trade.pnl <= 0]
    
    total_pnl = sum(trade.pnl for trade in closed_trades)
    average_pnl = total_pnl / len(closed_trades) if closed_trades else 0.0
    
    average_winning_trade = sum(trade.pnl for trade in winning_trades) / len(winning_trades) if winning_trades else 0.0
    average_losing_trade = sum(trade.pnl for trade in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    gross_profit = sum(trade.pnl for trade in winning_trades)
    gross_loss = abs(sum(trade.pnl for trade in losing_trades))
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    # Calculate win rate
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0.0
    
    # Calculate average duration
    durations = [trade.duration for trade in closed_trades if trade.duration is not None]
    average_duration = sum(durations, datetime.timedelta()) / len(durations) if durations else None
    
    # Calculate consecutive winners and losers
    results = [1 if trade.pnl > 0 else 0 for trade in closed_trades]
    
    max_consecutive_winners = 0
    max_consecutive_losers = 0
    current_winners = 0
    current_losers = 0
    
    for result in results:
        if result == 1:
            current_winners += 1
            current_losers = 0
            max_consecutive_winners = max(max_consecutive_winners, current_winners)
        else:
            current_losers += 1
            current_winners = 0
            max_consecutive_losers = max(max_consecutive_losers, current_losers)
    
    # Calculate additional statistics
    pnl_values = [trade.pnl for trade in closed_trades]
    pnl_std = np.std(pnl_values) if len(pnl_values) > 1 else 0.0
    
    # Calculate risk-adjusted metrics
    sharpe_ratio = average_pnl / pnl_std if pnl_std > 0 else 0.0
    
    # Calculate expectancy
    expectancy = (win_rate / 100 * average_winning_trade) + ((1 - win_rate / 100) * average_losing_trade)
    
    # Return statistics
    return {
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "open_trades": len([t for t in trades if t.status == TradeStatus.OPEN]),
        "canceled_trades": len([t for t in trades if t.status == TradeStatus.CANCELED]),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_pnl": average_pnl,
        "average_winning_trade": average_winning_trade,
        "average_losing_trade": average_losing_trade,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_duration": average_duration,
        "max_consecutive_winners": max_consecutive_winners,
        "max_consecutive_losers": max_consecutive_losers,
        "pnl_std": pnl_std,
        "sharpe_ratio": sharpe_ratio,
    }


def save_backtest_results(results: Dict[str, Any], file_path: str, file_format: str = "json") -> None:
    """Save backtest results to a file.
    
    Args:
        results: Backtest results
        file_path: File path
        file_format: File format (default: "json")
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save results based on file format
    if file_format.lower() == "json":
        # Convert non-serializable objects
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (datetime.datetime, pd.Timestamp)):
                serializable_results[key] = value.isoformat()
            elif isinstance(value, datetime.timedelta):
                serializable_results[key] = str(value)
            else:
                serializable_results[key] = value
        
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
    
    elif file_format.lower() == "csv":
        # Check if results contain a DataFrame
        if "equity_curve" in results and isinstance(results["equity_curve"], pd.DataFrame):
            results["equity_curve"].to_csv(file_path)
        elif "trades" in results and isinstance(results["trades"], pd.DataFrame):
            results["trades"].to_csv(file_path)
        else:
            # Convert dictionary to DataFrame
            df = pd.DataFrame([results])
            df.to_csv(file_path, index=False)
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_backtest_results(file_path: str, file_format: str = "json") -> Dict[str, Any]:
    """Load backtest results from a file.
    
    Args:
        file_path: File path
        file_format: File format (default: "json")
        
    Returns:
        Backtest results
    """
    # Load results based on file format
    if file_format.lower() == "json":
        with open(file_path, "r") as f:
            results = json.load(f)
        
        # Convert dictionaries to DataFrames
        for key, value in results.items():
            if isinstance(value, dict) and "index" in value and "data" in value:
                results[key] = pd.DataFrame(value["data"], index=value["index"])
        
        return results
    
    elif file_format.lower() == "csv":
        return {"data": pd.read_csv(file_path)}
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def run_monte_carlo_simulation(returns: pd.Series, initial_capital: float = 10000.0, num_simulations: int = 1000, num_periods: int = 252) -> Dict[str, Any]:
    """Run Monte Carlo simulation on returns.
    
    Args:
        returns: Return series
        initial_capital: Initial capital (default: 10000.0)
        num_simulations: Number of simulations (default: 1000)
        num_periods: Number of periods to simulate (default: 252)
        
    Returns:
        Dictionary with simulation results
    """
    # Convert returns to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate mean and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Generate random returns
    random_returns = np.random.normal(mean_return, std_return, size=(num_simulations, num_periods))
    
    # Calculate equity curves
    equity_curves = np.zeros((num_simulations, num_periods + 1))
    equity_curves[:, 0] = initial_capital
    
    for i in range(num_periods):
        equity_curves[:, i + 1] = equity_curves[:, i] * (1 + random_returns[:, i])
    
    # Calculate statistics
    final_values = equity_curves[:, -1]
    pct_changes = (final_values - initial_capital) / initial_capital * 100
    
    # Calculate percentiles
    percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
    
    # Calculate max drawdowns
    max_drawdowns = np.zeros(num_simulations)
    for i in range(num_simulations):
        max_drawdowns[i] = calculate_max_drawdown(equity_curves[i, :])[0] * 100
    
    # Calculate CAGR
    cagrs = (final_values / initial_capital) ** (252 / num_periods) - 1
    
    # Create results dictionary
    results = {
        "equity_curves": equity_curves,
        "final_values": final_values,
        "pct_changes": pct_changes,
        "percentiles": {
            "5%": percentiles[0],
            "25%": percentiles[1],
            "50%": percentiles[2],
            "75%": percentiles[3],
            "95%": percentiles[4],
        },
        "max_drawdowns": {
            "min": np.min(max_drawdowns),
            "max": np.max(max_drawdowns),
            "mean": np.mean(max_drawdowns),
            "median": np.median(max_drawdowns),
            "std": np.std(max_drawdowns),
            "percentiles": np.percentile(max_drawdowns, [5, 25, 50, 75, 95]),
        },
        "cagrs": {
            "min": np.min(cagrs),
            "max": np.max(cagrs),
            "mean": np.mean(cagrs),
            "median": np.median(cagrs),
            "std": np.std(cagrs),
            "percentiles": np.percentile(cagrs, [5, 25, 50, 75, 95]),
        },
        "probability_profit": np.mean(pct_changes > 0) * 100,
        "probability_loss": np.mean(pct_changes <= 0) * 100,
        "probability_target": {
            "10%": np.mean(pct_changes > 10) * 100,
            "20%": np.mean(pct_changes > 20) * 100,
            "30%": np.mean(pct_changes > 30) * 100,
            "50%": np.mean(pct_changes > 50) * 100,
            "100%": np.mean(pct_changes > 100) * 100,
        },
    }
    
    return results


def plot_monte_carlo_simulation(simulation_results: Dict[str, Any], figsize: Tuple[int, int] = (10, 6), title: str = "Monte Carlo Simulation") -> plt.Figure:
    """Plot Monte Carlo simulation results.
    
    Args:
        simulation_results: Simulation results from run_monte_carlo_simulation
        figsize: Figure size (default: (10, 6))
        title: Plot title (default: "Monte Carlo Simulation")
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    equity_curves = simulation_results["equity_curves"]
    percentiles = simulation_results["percentiles"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curves (sample of 100 curves)
    num_curves = min(100, equity_curves.shape[0])
    indices = np.random.choice(equity_curves.shape[0], num_curves, replace=False)
    
    for i in indices:
        ax.plot(equity_curves[i, :], color="skyblue", alpha=0.1)
    
    # Plot median
    median_curve = np.median(equity_curves, axis=0)
    ax.plot(median_curve, color="blue", linewidth=2, label="Median")
    
    # Plot percentiles
    percentile_5 = np.percentile(equity_curves, 5, axis=0)
    percentile_95 = np.percentile(equity_curves, 95, axis=0)
    
    ax.plot(percentile_5, color="red", linewidth=1.5, linestyle="--", label="5th Percentile")
    ax.plot(percentile_95, color="green", linewidth=1.5, linestyle="--", label="95th Percentile")
    
    # Fill between percentiles
    ax.fill_between(range(len(percentile_5)), percentile_5, percentile_95, color="gray", alpha=0.2)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Periods")
    ax.set_ylabel("Equity")
    
    # Add summary statistics
    stats_text = (
        f"Final Value (Median): ${percentiles['50%']:.2f}\n"
        f"5th Percentile: ${percentiles['5%']:.2f}\n"
        f"95th Percentile: ${percentiles['95%']:.2f}\n"
        f"Probability of Profit: {simulation_results['probability_profit']:.2f}%"
    )
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    plt.tight_layout()
    
    return fig


def run_walk_forward_analysis(data: pd.DataFrame, strategy_class: type, train_size: int, test_size: int, step_size: int, strategy_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Run walk-forward analysis.
    
    Args:
        data: DataFrame with OHLCV data
        strategy_class: Strategy class
        train_size: Training window size
        test_size: Testing window size
        step_size: Step size for moving window
        strategy_params: Strategy parameters
        **kwargs: Additional keyword arguments for strategy
        
    Returns:
        Dictionary with walk-forward analysis results
    """
    # Check if data has enough rows
    if len(data) < train_size + test_size:
        raise ValueError(f"Data has {len(data)} rows, but train_size + test_size = {train_size + test_size}")
    
    # Initialize results
    results = {
        "windows": [],
        "train_results": [],
        "test_results": [],
        "optimized_params": [],
        "equity_curves": [],
        "trades": [],
    }
    
    # Calculate number of windows
    num_windows = (len(data) - train_size - test_size) // step_size + 1
    
    # Run walk-forward analysis
    for i in range(num_windows):
        # Calculate window indices
        train_start = i * step_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        # Get train and test data
        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()
        
        # Optimize strategy on train data
        # This is a placeholder for optimization
        # In a real implementation, you would optimize strategy parameters here
        optimized_params = strategy_params.copy()
        
        # Create and run strategy on test data
        strategy = strategy_class(**optimized_params, **kwargs)
        strategy.run(test_data)
        
        # Get test results
        test_results = strategy.get_results()
        
        # Store results
        results["windows"].append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_start_date": data.index[train_start],
            "train_end_date": data.index[train_end - 1],
            "test_start_date": data.index[test_start],
            "test_end_date": data.index[test_end - 1],
        })
        
        results["train_results"].append(None)  # Placeholder for train results
        results["test_results"].append(test_results)
        results["optimized_params"].append(optimized_params)
        results["equity_curves"].append(test_results.get("equity_curve"))
        results["trades"].append(test_results.get("trades"))
    
    # Combine equity curves
    combined_equity_curve = pd.concat(results["equity_curves"])
    
    # Combine trades
    combined_trades = pd.concat(results["trades"])
    
    # Calculate overall performance metrics
    overall_metrics = {
        "total_return": combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0] - 1,
        "annualized_return": calculate_cagr(combined_equity_curve, (combined_equity_curve.index[-1] - combined_equity_curve.index[0]).days),
        "sharpe_ratio": calculate_sharpe_ratio(combined_equity_curve.pct_change().dropna()),
        "max_drawdown": calculate_max_drawdown(combined_equity_curve)[0],
        "win_rate": len(combined_trades[combined_trades["pnl"] > 0]) / len(combined_trades) * 100,
        "profit_factor": combined_trades[combined_trades["pnl"] > 0]["pnl"].sum() / abs(combined_trades[combined_trades["pnl"] <= 0]["pnl"].sum()),
    }
    
    # Add overall metrics to results
    results["overall_metrics"] = overall_metrics
    results["combined_equity_curve"] = combined_equity_curve
    results["combined_trades"] = combined_trades
    
    return results


def plot_walk_forward_analysis(wfa_results: Dict[str, Any], figsize: Tuple[int, int] = (10, 6), title: str = "Walk-Forward Analysis") -> plt.Figure:
    """Plot walk-forward analysis results.
    
    Args:
        wfa_results: Walk-forward analysis results from run_walk_forward_analysis
        figsize: Figure size (default: (10, 6))
        title: Plot title (default: "Walk-Forward Analysis")
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    windows = wfa_results["windows"]
    equity_curves = wfa_results["equity_curves"]
    combined_equity_curve = wfa_results["combined_equity_curve"]
    overall_metrics = wfa_results["overall_metrics"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curves for each window
    for i, (window, equity_curve) in enumerate(zip(windows, equity_curves)):
        # Normalize equity curve to start at 1.0
        normalized_curve = equity_curve / equity_curve.iloc[0]
        
        # Plot equity curve
        ax.plot(
            equity_curve.index,
            normalized_curve,
            label=f"Window {i+1}",
            alpha=0.7,
        )
        
        # Add vertical lines for window boundaries
        ax.axvline(window["test_start_date"], color="gray", linestyle="--", alpha=0.5)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Equity")
    
    # Add summary statistics
    stats_text = (
        f"Total Return: {overall_metrics['total_return'] * 100:.2f}%\n"
        f"Annualized Return: {overall_metrics['annualized_return'] * 100:.2f}%\n"
        f"Sharpe Ratio: {overall_metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {overall_metrics['max_drawdown'] * 100:.2f}%\n"
        f"Win Rate: {overall_metrics['win_rate']:.2f}%\n"
        f"Profit Factor: {overall_metrics['profit_factor']:.2f}"
    )
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    plt.tight_layout()
    
    return fig