"""Reporting module for backtesting framework.

This module provides tools for generating visual reports and analysis of backtest results.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.backtesting.performance import PerformanceAnalytics, PerformanceMetrics
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class ChartType(Enum):
    """Chart types for backtest reporting."""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    MONTHLY_RETURNS = "monthly_returns"
    ROLLING_RETURNS = "rolling_returns"
    ROLLING_VOLATILITY = "rolling_volatility"
    ROLLING_SHARPE = "rolling_sharpe"
    UNDERWATER = "underwater"
    TRADE_ANALYSIS = "trade_analysis"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    POSITION_SIZE = "position_size"
    TRADE_DURATION = "trade_duration"
    CUSTOM = "custom"


class BacktestReport:
    """Backtest report generator.
    
    This class generates visual reports and analysis of backtest results.
    """
    
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.DataFrame,
        trades: pd.DataFrame,
        positions: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        period: str = "daily",
        output_dir: str = "reports",
    ):
        """Initialize the backtest report generator.
        
        Args:
            equity_curve: DataFrame with equity curve (index=timestamp, columns=[equity])
            returns: DataFrame with returns (index=timestamp, columns=[returns])
            trades: DataFrame with trade details
            positions: Optional DataFrame with position details
            benchmark_returns: Optional DataFrame with benchmark returns
            strategy_name: Strategy name (default: "Strategy")
            initial_capital: Initial capital (default: 100000.0)
            risk_free_rate: Annual risk-free rate (default: 0.0)
            period: Data frequency ("daily", "weekly", "monthly")
            output_dir: Output directory for reports (default: "reports")
        """
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades = trades
        self.positions = positions
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.period = period
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize performance analytics
        self.performance = PerformanceAnalytics(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            frequency=period,
        )
        
        # Initialize figures dictionary
        self.figures = {}
        
        logger.info(f"Initialized backtest report generator for {strategy_name}")
    
    def generate_report(
        self,
        report_format: Union[str, ReportFormat] = ReportFormat.HTML,
        include_charts: Optional[List[Union[str, ChartType]]] = None,
        filename: Optional[str] = None,
        show_plots: bool = False,
    ) -> str:
        """Generate a backtest report.
        
        Args:
            report_format: Report format (default: HTML)
            include_charts: List of charts to include (default: all)
            filename: Output filename (default: auto-generated)
            show_plots: Whether to show plots (default: False)
            
        Returns:
            Path to the generated report
        """
        if isinstance(report_format, str):
            report_format = ReportFormat(report_format.lower())
        
        # Generate all charts
        self._generate_all_charts(include_charts)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name.replace(' ', '_')}_{timestamp}"
        
        # Generate report based on format
        if report_format == ReportFormat.HTML:
            report_path = self._generate_html_report(filename)
        elif report_format == ReportFormat.PDF:
            report_path = self._generate_pdf_report(filename)
        elif report_format in [ReportFormat.PNG, ReportFormat.JPG, ReportFormat.SVG]:
            report_path = self._generate_image_report(filename, report_format.value)
        elif report_format == ReportFormat.JSON:
            report_path = self._generate_json_report(filename)
        elif report_format == ReportFormat.CSV:
            report_path = self._generate_csv_report(filename)
        elif report_format == ReportFormat.EXCEL:
            report_path = self._generate_excel_report(filename)
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        # Show plots if requested
        if show_plots:
            plt.show()
        else:
            plt.close("all")
        
        logger.info(f"Generated {report_format.value} report: {report_path}")
        
        return report_path
    
    def _generate_all_charts(
        self,
        include_charts: Optional[List[Union[str, ChartType]]] = None,
    ) -> None:
        """Generate all charts for the report.
        
        Args:
            include_charts: List of charts to include (default: all)
        """
        # Define all available charts
        all_charts = [
            ChartType.EQUITY_CURVE,
            ChartType.DRAWDOWN,
            ChartType.RETURNS_DISTRIBUTION,
            ChartType.MONTHLY_RETURNS,
            ChartType.ROLLING_RETURNS,
            ChartType.ROLLING_VOLATILITY,
            ChartType.ROLLING_SHARPE,
            ChartType.UNDERWATER,
            ChartType.TRADE_ANALYSIS,
        ]
        
        # Add benchmark comparison if benchmark returns are available
        if self.benchmark_returns is not None:
            all_charts.append(ChartType.BENCHMARK_COMPARISON)
        
        # Add position-based charts if positions are available
        if self.positions is not None:
            all_charts.extend([
                ChartType.EXPOSURE,
                ChartType.POSITION_SIZE,
            ])
        
        # Filter charts if include_charts is provided
        if include_charts is not None:
            # Convert string chart types to ChartType enum
            include_charts = [
                ChartType(chart) if isinstance(chart, str) else chart
                for chart in include_charts
            ]
            charts_to_generate = [chart for chart in all_charts if chart in include_charts]
        else:
            charts_to_generate = all_charts
        
        # Generate each chart
        for chart_type in charts_to_generate:
            self._generate_chart(chart_type)
    
    def _generate_chart(self, chart_type: ChartType) -> Figure:
        """Generate a specific chart.
        
        Args:
            chart_type: Chart type
            
        Returns:
            Matplotlib figure
        """
        if chart_type == ChartType.EQUITY_CURVE:
            fig = self._generate_equity_curve()
        elif chart_type == ChartType.DRAWDOWN:
            fig = self._generate_drawdown()
        elif chart_type == ChartType.RETURNS_DISTRIBUTION:
            fig = self._generate_returns_distribution()
        elif chart_type == ChartType.MONTHLY_RETURNS:
            fig = self._generate_monthly_returns()
        elif chart_type == ChartType.ROLLING_RETURNS:
            fig = self._generate_rolling_returns()
        elif chart_type == ChartType.ROLLING_VOLATILITY:
            fig = self._generate_rolling_volatility()
        elif chart_type == ChartType.ROLLING_SHARPE:
            fig = self._generate_rolling_sharpe()
        elif chart_type == ChartType.UNDERWATER:
            fig = self._generate_underwater()
        elif chart_type == ChartType.TRADE_ANALYSIS:
            fig = self._generate_trade_analysis()
        elif chart_type == ChartType.BENCHMARK_COMPARISON:
            fig = self._generate_benchmark_comparison()
        elif chart_type == ChartType.EXPOSURE:
            fig = self._generate_exposure()
        elif chart_type == ChartType.POSITION_SIZE:
            fig = self._generate_position_size()
        elif chart_type == ChartType.TRADE_DURATION:
            fig = self._generate_trade_duration()
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Store figure in figures dictionary
        self.figures[chart_type.value] = fig
        
        return fig
    
    def _generate_equity_curve(self) -> Figure:
        """Generate equity curve chart.
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot equity curve
        ax.plot(self.equity_curve.index, self.equity_curve["equity"], label="Equity Curve")
        
        # Plot benchmark if available
        if self.benchmark_returns is not None:
            # Calculate benchmark equity curve
            benchmark_equity = self.initial_capital * (1 + self.benchmark_returns.cumsum())
            ax.plot(benchmark_equity.index, benchmark_equity.iloc[:, 0], label="Benchmark", alpha=0.7)
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        
        # Add key metrics as text
        metrics_text = (
            f"Total Return: {self.performance.get_metric(PerformanceMetrics.TOTAL_RETURN):.2f}%\n"
            f"CAGR: {self.performance.get_metric(PerformanceMetrics.CAGR):.2f}%\n"
            f"Sharpe Ratio: {self.performance.get_metric(PerformanceMetrics.SHARPE_RATIO):.2f}\n"
            f"Max Drawdown: {self.performance.get_metric(PerformanceMetrics.MAX_DRAWDOWN):.2f}%"
        )
        
        # Position text in the upper left corner with a slight offset
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_drawdown(self) -> Figure:
        """Generate drawdown chart.
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate running maximum
        running_max = self.equity_curve["equity"].cummax()
        
        # Calculate drawdown percentage
        drawdown = (self.equity_curve["equity"] / running_max - 1) * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3, label="Drawdown")
        ax.plot(drawdown.index, drawdown, color="red", alpha=0.5)
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis (drawdowns are negative)
        ax.invert_yaxis()
        
        # Add max drawdown as text
        max_drawdown = self.performance.get_metric(PerformanceMetrics.MAX_DRAWDOWN)
        ax.text(
            0.02, 0.98,
            f"Max Drawdown: {max_drawdown:.2f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_returns_distribution(self) -> Figure:
        """Generate returns distribution chart.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert returns to percentage
        returns_pct = self.returns["returns"] * 100
        
        # Plot histogram
        n, bins, patches = ax.hist(
            returns_pct,
            bins=50,
            alpha=0.7,
            color="blue",
            density=True,
            label="Returns",
        )
        
        # Plot normal distribution
        mu = returns_pct.mean()
        sigma = returns_pct.std()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        ax.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                linewidth=2, color="red", label="Normal Distribution")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Returns Distribution")
        ax.set_xlabel("Returns (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Mean: {mu:.2f}%\n"
            f"Std Dev: {sigma:.2f}%\n"
            f"Skewness: {returns_pct.skew():.2f}\n"
            f"Kurtosis: {returns_pct.kurtosis():.2f}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_monthly_returns(self) -> Figure:
        """Generate monthly returns heatmap.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty:
            return plt.figure()
        
        # Resample returns to monthly
        monthly_returns = self.returns["returns"].resample("M").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period("M")
        returns_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        im = ax.imshow(returns_table, cmap=cmap, aspect="auto")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Returns (%)")
        
        # Set labels
        ax.set_title(f"{self.strategy_name} - Monthly Returns (%)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        
        # Set x-axis labels (months)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels(months)
        
        # Set y-axis labels (years)
        years = returns_table.index.astype(str).tolist()
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        
        # Add text annotations to the heatmap cells
        for i in range(len(years)):
            for j in range(len(months)):
                if j < returns_table.shape[1] and not np.isnan(returns_table.iloc[i, j]):
                    text_color = "black" if abs(returns_table.iloc[i, j]) < 10 else "white"
                    ax.text(j, i, f"{returns_table.iloc[i, j]:.1f}",
                            ha="center", va="center", color=text_color)
        
        fig.tight_layout()
        
        return fig
    
    def _generate_rolling_returns(self) -> Figure:
        """Generate rolling returns chart.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate rolling returns for different windows
        windows = [21, 63, 126, 252]  # 1 month, 3 months, 6 months, 1 year
        labels = ["1 Month", "3 Months", "6 Months", "1 Year"]
        colors = ["blue", "green", "orange", "red"]
        
        for window, label, color in zip(windows, labels, colors):
            if len(self.returns) >= window:
                rolling_returns = self.returns["returns"].rolling(window=window).apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100
                ax.plot(rolling_returns.index, rolling_returns, label=label, color=color, alpha=0.7)
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Rolling Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Returns (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        
        fig.tight_layout()
        
        return fig
    
    def _generate_rolling_volatility(self) -> Figure:
        """Generate rolling volatility chart.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate rolling volatility (annualized)
        window = 63  # 3 months
        annualization_factor = 252 if self.period == "daily" else 52 if self.period == "weekly" else 12
        rolling_vol = self.returns["returns"].rolling(window=window).std() * np.sqrt(annualization_factor) * 100
        
        # Plot rolling volatility
        ax.plot(rolling_vol.index, rolling_vol, color="red", label=f"{window}-day Rolling Volatility")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Rolling Volatility (Annualized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add overall volatility as text
        overall_vol = self.performance.get_metric(PerformanceMetrics.VOLATILITY)
        ax.text(
            0.02, 0.98,
            f"Overall Volatility: {overall_vol:.2f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_rolling_sharpe(self) -> Figure:
        """Generate rolling Sharpe ratio chart.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate rolling Sharpe ratio
        window = 126  # 6 months
        annualization_factor = 252 if self.period == "daily" else 52 if self.period == "weekly" else 12
        excess_returns = self.returns["returns"] - self.risk_free_rate / annualization_factor
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(annualization_factor)
        
        # Plot rolling Sharpe ratio
        ax.plot(rolling_sharpe.index, rolling_sharpe, color="purple", label=f"{window}-day Rolling Sharpe Ratio")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Rolling Sharpe Ratio")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        
        # Add overall Sharpe ratio as text
        overall_sharpe = self.performance.get_metric(PerformanceMetrics.SHARPE_RATIO)
        ax.text(
            0.02, 0.98,
            f"Overall Sharpe Ratio: {overall_sharpe:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_underwater(self) -> Figure:
        """Generate underwater (drawdown) chart.
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate running maximum
        running_max = self.equity_curve["equity"].cummax()
        
        # Calculate drawdown percentage
        drawdown = (self.equity_curve["equity"] / running_max - 1) * 100
        
        # Plot underwater chart
        ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Underwater Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis (drawdowns are negative)
        ax.invert_yaxis()
        
        # Add drawdown statistics as text
        max_drawdown = self.performance.get_metric(PerformanceMetrics.MAX_DRAWDOWN)
        calmar_ratio = self.performance.get_metric(PerformanceMetrics.CALMAR_RATIO)
        recovery_factor = self.performance.get_metric(PerformanceMetrics.RECOVERY_FACTOR)
        
        stats_text = (
            f"Max Drawdown: {max_drawdown:.2f}%\n"
            f"Calmar Ratio: {calmar_ratio:.2f}\n"
            f"Recovery Factor: {recovery_factor:.2f}"
        )
        
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_trade_analysis(self) -> Figure:
        """Generate trade analysis chart.
        
        Returns:
            Matplotlib figure
        """
        if self.trades.empty:
            return plt.figure()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Calculate profit for each trade if not already present
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1,
            )
        
        # Subplot 1: Profit distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.trades["profit"], bins=30, alpha=0.7, color="blue")
        ax1.set_title("Profit Distribution")
        ax1.set_xlabel("Profit ($)")
        ax1.set_ylabel("Frequency")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Cumulative profit
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_profit = self.trades["profit"].cumsum()
        ax2.plot(range(len(cumulative_profit)), cumulative_profit, color="green")
        ax2.set_title("Cumulative Profit")
        ax2.set_xlabel("Trade Number")
        ax2.set_ylabel("Cumulative Profit ($)")
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Win/loss ratio
        ax3 = fig.add_subplot(gs[1, 0])
        winning_trades = len(self.trades[self.trades["profit"] > 0])
        losing_trades = len(self.trades[self.trades["profit"] < 0])
        ax3.pie(
            [winning_trades, losing_trades],
            labels=["Winning", "Losing"],
            colors=["green", "red"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.1, 0),
        )
        ax3.set_title("Win/Loss Ratio")
        
        # Subplot 4: Trade metrics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        
        # Calculate trade metrics
        total_trades = len(self.trades)
        winning_percentage = self.performance.get_metric(PerformanceMetrics.WINNING_PERCENTAGE)
        profit_factor = self.performance.get_metric(PerformanceMetrics.PROFIT_FACTOR)
        expectancy = self.performance.get_metric(PerformanceMetrics.EXPECTANCY)
        avg_win = self.performance.get_metric(PerformanceMetrics.AVERAGE_WIN)
        avg_loss = self.performance.get_metric(PerformanceMetrics.AVERAGE_LOSS)
        max_win = self.trades["profit"].max() if not self.trades.empty else 0
        max_loss = self.trades["profit"].min() if not self.trades.empty else 0
        
        # Create metrics text
        metrics_text = (
            f"Total Trades: {total_trades}\n\n"
            f"Winning Percentage: {winning_percentage:.2f}%\n\n"
            f"Profit Factor: {profit_factor:.2f}\n\n"
            f"Expectancy: {expectancy:.2f}\n\n"
            f"Average Win: ${avg_win:.2f}\n\n"
            f"Average Loss: ${avg_loss:.2f}\n\n"
            f"Max Win: ${max_win:.2f}\n\n"
            f"Max Loss: ${max_loss:.2f}"
        )
        
        ax4.text(
            0.5, 0.5, metrics_text,
            transform=ax4.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        # Add overall title
        fig.suptitle(f"{self.strategy_name} - Trade Analysis", fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig
    
    def _generate_benchmark_comparison(self) -> Figure:
        """Generate benchmark comparison chart.
        
        Returns:
            Matplotlib figure
        """
        if self.returns.empty or self.benchmark_returns is None:
            return plt.figure()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Subplot 1: Cumulative returns comparison
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        # Calculate cumulative returns
        strategy_cum_returns = (1 + self.returns["returns"]).cumprod() - 1
        benchmark_cum_returns = (1 + self.benchmark_returns.iloc[:, 0]).cumprod() - 1
        
        # Plot cumulative returns
        ax1.plot(strategy_cum_returns.index, strategy_cum_returns * 100, label="Strategy", color="blue")
        ax1.plot(benchmark_cum_returns.index, benchmark_cum_returns * 100, label="Benchmark", color="red", alpha=0.7)
        
        # Add labels and title
        ax1.set_title("Cumulative Returns Comparison")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Rolling beta
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate rolling beta
        window = 126  # 6 months
        if len(self.returns) >= window:
            # Align returns with benchmark
            aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
            aligned_data.columns = ["strategy", "benchmark"]
            
            # Calculate rolling beta
            rolling_cov = aligned_data["strategy"].rolling(window=window).cov(aligned_data["benchmark"])
            rolling_var = aligned_data["benchmark"].rolling(window=window).var()
            rolling_beta = rolling_cov / rolling_var
            
            # Plot rolling beta
            ax2.plot(rolling_beta.index, rolling_beta, color="purple")
            
            # Add horizontal line at 1
            ax2.axhline(y=1, color="black", linestyle="--", alpha=0.3)
        
        # Add labels and title
        ax2.set_title("Rolling Beta")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Beta")
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Comparison metrics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        
        # Calculate comparison metrics
        alpha = self.performance.get_metric(PerformanceMetrics.ALPHA)
        beta = self.performance.get_metric(PerformanceMetrics.BETA)
        info_ratio = self.performance.get_metric(PerformanceMetrics.INFORMATION_RATIO)
        
        # Calculate correlation
        aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        
        # Calculate outperformance
        strategy_return = self.performance.get_metric(PerformanceMetrics.TOTAL_RETURN)
        benchmark_return = (benchmark_cum_returns.iloc[-1] * 100) if not benchmark_cum_returns.empty else 0
        outperformance = strategy_return - benchmark_return
        
        # Create metrics text
        metrics_text = (
            f"Alpha: {alpha:.4f}\n\n"
            f"Beta: {beta:.2f}\n\n"
            f"Information Ratio: {info_ratio:.2f}\n\n"
            f"Correlation: {correlation:.2f}\n\n"
            f"Strategy Return: {strategy_return:.2f}%\n\n"
            f"Benchmark Return: {benchmark_return:.2f}%\n\n"
            f"Outperformance: {outperformance:.2f}%"
        )
        
        ax3.text(
            0.5, 0.5, metrics_text,
            transform=ax3.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        # Add overall title
        fig.suptitle(f"{self.strategy_name} - Benchmark Comparison", fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig
    
    def _generate_exposure(self) -> Figure:
        """Generate exposure chart.
        
        Returns:
            Matplotlib figure
        """
        if self.positions is None or self.positions.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate exposure (sum of absolute position values / equity)
        if "exposure" in self.positions.columns:
            exposure = self.positions["exposure"]
        else:
            # Calculate exposure from position values and equity
            position_values = self.positions["value"] if "value" in self.positions.columns else self.positions["position_value"]
            equity = self.equity_curve["equity"]
            
            # Align indexes
            position_values = position_values.reindex(equity.index, method="ffill")
            
            # Calculate exposure
            exposure = position_values.abs() / equity
        
        # Plot exposure
        ax.plot(exposure.index, exposure * 100, color="blue")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Market Exposure")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exposure (%)")
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 100%
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.7)
        
        # Add average exposure as text
        avg_exposure = exposure.mean() * 100
        max_exposure = exposure.max() * 100
        
        ax.text(
            0.02, 0.98,
            f"Average Exposure: {avg_exposure:.2f}%\nMax Exposure: {max_exposure:.2f}%",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_position_size(self) -> Figure:
        """Generate position size chart.
        
        Returns:
            Matplotlib figure
        """
        if self.positions is None or self.positions.empty:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get position sizes
        if "size" in self.positions.columns:
            position_sizes = self.positions["size"]
        elif "quantity" in self.positions.columns:
            position_sizes = self.positions["quantity"]
        else:
            position_sizes = self.positions["position_size"]
        
        # Plot position sizes
        ax.plot(position_sizes.index, position_sizes, color="green")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Position Size")
        ax.set_xlabel("Date")
        ax.set_ylabel("Position Size (units)")
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        
        # Add average position size as text
        avg_position_size = position_sizes.abs().mean()
        max_position_size = position_sizes.abs().max()
        
        ax.text(
            0.02, 0.98,
            f"Average Position Size: {avg_position_size:.2f}\nMax Position Size: {max_position_size:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_trade_duration(self) -> Figure:
        """Generate trade duration chart.
        
        Returns:
            Matplotlib figure
        """
        if self.trades.empty or "duration" not in self.trades.columns:
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot trade duration histogram
        ax.hist(self.trades["duration"], bins=30, alpha=0.7, color="blue")
        
        # Add labels and title
        ax.set_title(f"{self.strategy_name} - Trade Duration")
        ax.set_xlabel("Duration (days)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add average duration as text
        avg_duration = self.trades["duration"].mean()
        max_duration = self.trades["duration"].max()
        
        ax.text(
            0.98, 0.98,
            f"Average Duration: {avg_duration:.2f} days\nMax Duration: {max_duration:.2f} days",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        
        fig.tight_layout()
        
        return fig
    
    def _generate_html_report(self, filename: str) -> str:
        """Generate HTML report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import base64
        from io import BytesIO
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.strategy_name} - Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric h3 {{ margin-top: 0; color: #333; }}
                .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>{self.strategy_name} - Backtest Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics">
        """
        
        # Add key metrics
        metrics = [
            ("Total Return", f"{self.performance.get_metric(PerformanceMetrics.TOTAL_RETURN):.2f}%"),
            ("CAGR", f"{self.performance.get_metric(PerformanceMetrics.CAGR):.2f}%"),
            ("Sharpe Ratio", f"{self.performance.get_metric(PerformanceMetrics.SHARPE_RATIO):.2f}"),
            ("Sortino Ratio", f"{self.performance.get_metric(PerformanceMetrics.SORTINO_RATIO):.2f}"),
            ("Max Drawdown", f"{self.performance.get_metric(PerformanceMetrics.MAX_DRAWDOWN):.2f}%"),
            ("Volatility", f"{self.performance.get_metric(PerformanceMetrics.VOLATILITY):.2f}%"),
            ("Calmar Ratio", f"{self.performance.get_metric(PerformanceMetrics.CALMAR_RATIO):.2f}"),
            ("Winning %", f"{self.performance.get_metric(PerformanceMetrics.WINNING_PERCENTAGE):.2f}%"),
        ]
        
        for name, value in metrics:
            html_content += f"""
                <div class="metric">
                    <h3>{name}</h3>
                    <p>{value}</p>
                </div>
            """
        
        html_content += "</div>\n"
        
        # Add charts
        for chart_type, fig in self.figures.items():
            # Convert figure to base64 image
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            
            # Add chart to HTML
            html_content += f"""
            <h2>{chart_type.replace("_", " ").title()}</h2>
            <div class="chart">
                <img src="data:image/png;base64,{img_str}" alt="{chart_type}" width="800">
            </div>
            """
        
        # Add detailed metrics table
        html_content += """
            <h2>Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add all metrics to table
        all_metrics = self.performance.get_metrics()
        for metric, value in all_metrics.items():
            metric_name = metric.replace("_", " ").title()
            formatted_value = f"{value:.4f}"
            
            # Add percentage sign for return metrics
            if "return" in metric.lower() or "drawdown" in metric.lower() or "volatility" in metric.lower():
                formatted_value += "%"
            
            html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{formatted_value}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Save HTML to file
        if not filename.endswith(".html"):
            filename += ".html"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_pdf_report(self, filename: str) -> str:
        """Generate PDF report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        except ImportError:
            logger.error("ReportLab is required for PDF generation. Install with 'pip install reportlab'")
            return ""
        
        # Create PDF filename
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        elements.append(Paragraph(f"{self.strategy_name} - Backtest Report", styles["Title"]))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        
        # Add key metrics table
        elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
        
        metrics_data = [
            ["Metric", "Value"],
            ["Total Return", f"{self.performance.get_metric(PerformanceMetrics.TOTAL_RETURN):.2f}%"],
            ["CAGR", f"{self.performance.get_metric(PerformanceMetrics.CAGR):.2f}%"],
            ["Sharpe Ratio", f"{self.performance.get_metric(PerformanceMetrics.SHARPE_RATIO):.2f}"],
            ["Sortino Ratio", f"{self.performance.get_metric(PerformanceMetrics.SORTINO_RATIO):.2f}"],
            ["Max Drawdown", f"{self.performance.get_metric(PerformanceMetrics.MAX_DRAWDOWN):.2f}%"],
            ["Volatility", f"{self.performance.get_metric(PerformanceMetrics.VOLATILITY):.2f}%"],
            ["Calmar Ratio", f"{self.performance.get_metric(PerformanceMetrics.CALMAR_RATIO):.2f}"],
            ["Winning %", f"{self.performance.get_metric(PerformanceMetrics.WINNING_PERCENTAGE):.2f}%"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[200, 100])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (1, 0), "CENTER"),
            ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (1, 0), 12),
            ("BACKGROUND", (0, 1), (1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 12))
        
        # Add charts
        for chart_type, fig in self.figures.items():
            elements.append(Paragraph(chart_type.replace("_", " ").title(), styles["Heading2"]))
            
            # Save figure to temporary file
            chart_filename = os.path.join(self.output_dir, f"{chart_type}_temp.png")
            fig.savefig(chart_filename, format="png", dpi=100)
            
            # Add image to PDF
            img = Image(chart_filename, width=500, height=300)
            elements.append(img)
            elements.append(Spacer(1, 12))
            
            # Remove temporary file
            os.remove(chart_filename)
        
        # Build PDF
        doc.build(elements)
        
        return output_path
    
    def _generate_image_report(self, filename: str, format: str) -> str:
        """Generate image report.
        
        Args:
            filename: Output filename
            format: Image format (png, jpg, svg)
            
        Returns:
            Path to the generated report
        """
        # Create figure with all charts
        num_charts = len(self.figures)
        rows = int(np.ceil(num_charts / 2))
        
        fig = plt.figure(figsize=(20, 10 * rows))
        gs = GridSpec(rows, 2, figure=fig)
        
        # Add charts to figure
        for i, (chart_type, chart_fig) in enumerate(self.figures.items()):
            row = i // 2
            col = i % 2
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Copy chart to subplot
            for item in chart_fig.get_axes()[0].get_children():
                if hasattr(item, "get_data"):
                    x, y = item.get_data()
                    ax.plot(x, y, color=item.get_color(), alpha=item.get_alpha(), label=item.get_label())
                elif hasattr(item, "get_paths") and hasattr(item, "get_facecolor"):
                    # For filled areas
                    ax.fill_between(item.get_paths()[0].vertices[:, 0], item.get_paths()[0].vertices[:, 1], color=item.get_facecolor()[0], alpha=item.get_alpha())
            
            # Copy title and labels
            ax.set_title(chart_fig.get_axes()[0].get_title())
            ax.set_xlabel(chart_fig.get_axes()[0].get_xlabel())
            ax.set_ylabel(chart_fig.get_axes()[0].get_ylabel())
            
            # Copy legend if exists
            if chart_fig.get_axes()[0].get_legend() is not None:
                ax.legend()
            
            # Copy grid
            ax.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f"{self.strategy_name} - Backtest Report", fontsize=20)
        
        # Save figure to file
        if not filename.endswith(f".{format}"):
            filename += f".{format}"
        
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, format=format, dpi=100, bbox_inches="tight")
        
        return output_path
    
    def _generate_json_report(self, filename: str) -> str:
        """Generate JSON report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import json
        
        # Create report data
        report_data = {
            "strategy_name": self.strategy_name,
            "generated_at": datetime.now().isoformat(),
            "metrics": self.performance.get_metrics(),
            "summary": self.performance.get_summary(),
        }
        
        # Add trade statistics if available
        if not self.trades.empty:
            report_data["trade_statistics"] = {
                "total_trades": len(self.trades),
                "winning_trades": len(self.trades[self.trades["profit"] > 0]) if "profit" in self.trades.columns else 0,
                "losing_trades": len(self.trades[self.trades["profit"] < 0]) if "profit" in self.trades.columns else 0,
                "total_profit": float(self.trades["profit"].sum()) if "profit" in self.trades.columns else 0,
            }
        
        # Save to file
        if not filename.endswith(".json"):
            filename += ".json"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=4)
        
        return output_path
    
    def _generate_csv_report(self, filename: str) -> str:
        """Generate CSV report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            "Metric": list(self.performance.get_metrics().keys()),
            "Value": list(self.performance.get_metrics().values()),
        })
        
        # Save to file
        if not filename.endswith(".csv"):
            filename += ".csv"
        
        output_path = os.path.join(self.output_dir, filename)
        metrics_df.to_csv(output_path, index=False)
        
        return output_path
    
    def _generate_excel_report(self, filename: str) -> str:
        """Generate Excel report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        try:
            import xlsxwriter
        except ImportError:
            logger.error("XlsxWriter is required for Excel generation. Install with 'pip install xlsxwriter'")
            return ""
        
        # Create Excel filename
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
        
        # Create workbook and add worksheets
        workbook = writer.book
        
        # Add metrics worksheet
        metrics_df = pd.DataFrame({
            "Metric": list(self.performance.get_metrics().keys()),
            "Value": list(self.performance.get_metrics().values()),
        })
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        
        # Format metrics worksheet
        metrics_sheet = writer.sheets["Metrics"]
        metrics_sheet.set_column("A:A", 30)
        metrics_sheet.set_column("B:B", 15)
        
        # Add header format
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#D7E4BC",
            "border": 1,
        })
        
        # Write headers with format
        for col_num, value in enumerate(metrics_df.columns.values):
            metrics_sheet.write(0, col_num, value, header_format)
        
        # Add equity curve worksheet
        if not self.equity_curve.empty:
            self.equity_curve.to_excel(writer, sheet_name="Equity Curve")
        
        # Add returns worksheet
        if not self.returns.empty:
            self.returns.to_excel(writer, sheet_name="Returns")
        
        # Add trades worksheet
        if not self.trades.empty:
            self.trades.to_excel(writer, sheet_name="Trades")
        
        # Add positions worksheet
        if self.positions is not None and not self.positions.empty:
            self.positions.to_excel(writer, sheet_name="Positions")
        
        # Save Excel file
        writer.close()
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the backtest results.
        
        Returns:
            Dictionary with backtest summary
        """
        # Get key metrics
        metrics = self.performance.get_metrics()
        
        # Create summary dictionary
        summary = {
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "period": self.period,
            "start_date": self.equity_curve.index[0].strftime("%Y-%m-%d") if not self.equity_curve.empty else None,
            "end_date": self.equity_curve.index[-1].strftime("%Y-%m-%d") if not self.equity_curve.empty else None,
            "duration_days": (self.equity_curve.index[-1] - self.equity_curve.index[0]).days if not self.equity_curve.empty else 0,
            "final_equity": float(self.equity_curve["equity"].iloc[-1]) if not self.equity_curve.empty else 0,
            "total_return_pct": metrics.get("total_return", 0),
            "cagr": metrics.get("cagr", 0),
            "volatility": metrics.get("volatility", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "calmar_ratio": metrics.get("calmar_ratio", 0),
            "winning_percentage": metrics.get("winning_percentage", 0),
            "profit_factor": metrics.get("profit_factor", 0),
        }
        
        # Add trade statistics if available
        if not self.trades.empty:
            summary["total_trades"] = len(self.trades)
            summary["winning_trades"] = len(self.trades[self.trades["profit"] > 0]) if "profit" in self.trades.columns else 0
            summary["losing_trades"] = len(self.trades[self.trades["profit"] < 0]) if "profit" in self.trades.columns else 0
            summary["total_profit"] = float(self.trades["profit"].sum()) if "profit" in self.trades.columns else 0
        
        # Add benchmark comparison if available
        if self.benchmark_returns is not None:
            benchmark_return = ((1 + self.benchmark_returns.iloc[:, 0]).cumprod().iloc[-1] - 1) * 100
            summary["benchmark_return"] = float(benchmark_return)
            summary["outperformance"] = summary["total_return_pct"] - float(benchmark_return)
            summary["alpha"] = metrics.get("alpha", 0)
            summary["beta"] = metrics.get("beta", 0)
            summary["information_ratio"] = metrics.get("information_ratio", 0)
        
        return summary