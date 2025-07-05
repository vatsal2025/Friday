"""Comparative and scenario analysis module for portfolio analysis.

This module provides tools for comparing different strategies, portfolios,
and analyzing the impact of various scenarios on portfolio performance.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ComparativeAnalyzer:
    """Base class for comparative analysis."""
    
    def __init__(self):
        """Initialize the comparative analyzer."""
        pass
    
    def compare(self, *args, **kwargs) -> pd.DataFrame:
        """Compare different strategies or portfolios.
        
        This is a base method that should be implemented by subclasses.
        
        Returns:
            DataFrame with comparison results
        """
        raise NotImplementedError("Subclasses must implement compare()")
    
    def plot_comparison(self, *args, **kwargs) -> Any:
        """Plot comparison results.
        
        This is a base method that should be implemented by subclasses.
        
        Returns:
            Figure object
        """
        raise NotImplementedError("Subclasses must implement plot_comparison()")


class StrategyComparator(ComparativeAnalyzer):
    """Comparator for analyzing and comparing different investment strategies."""
    
    def __init__(self):
        """Initialize the strategy comparator."""
        super().__init__()
    
    def compare(
        self,
        returns_dict: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        frequency: str = 'D',
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare different investment strategies based on performance metrics.
        
        Args:
            returns_dict: Dictionary mapping strategy names to return series
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            frequency: Frequency of returns ('D' for daily, 'M' for monthly, etc.)
            metrics: List of metrics to calculate (if None, calculates all available metrics)
            
        Returns:
            DataFrame with performance metrics for each strategy
        """
        # Default metrics to calculate
        if metrics is None:
            metrics = [
                'total_return', 'annualized_return', 'annualized_volatility',
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio',
                'best_month', 'worst_month', 'skewness', 'kurtosis',
                'positive_months', 'win_rate'
            ]
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=metrics)
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            returns_dict['Benchmark'] = benchmark_returns
        
        # Calculate annualization factor based on frequency
        if frequency == 'D':
            annualization_factor = 252
        elif frequency == 'W':
            annualization_factor = 52
        elif frequency == 'M':
            annualization_factor = 12
        elif frequency == 'Q':
            annualization_factor = 4
        else:
            annualization_factor = 1  # Annual
        
        # Calculate metrics for each strategy
        for strategy_name, returns in returns_dict.items():
            strategy_metrics = {}
            
            # Ensure returns are a pandas Series
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)
            
            # Total return
            if 'total_return' in metrics:
                strategy_metrics['total_return'] = (1 + returns).prod() - 1
            
            # Annualized return
            if 'annualized_return' in metrics:
                strategy_metrics['annualized_return'] = (
                    (1 + returns).prod() ** (annualization_factor / len(returns)) - 1
                )
            
            # Annualized volatility
            if 'annualized_volatility' in metrics:
                strategy_metrics['annualized_volatility'] = (
                    returns.std() * np.sqrt(annualization_factor)
                )
            
            # Sharpe ratio
            if 'sharpe_ratio' in metrics:
                excess_returns = returns - risk_free_rate / annualization_factor
                strategy_metrics['sharpe_ratio'] = (
                    excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
                    if returns.std() > 0 else 0
                )
            
            # Sortino ratio
            if 'sortino_ratio' in metrics:
                excess_returns = returns - risk_free_rate / annualization_factor
                downside_returns = returns[returns < 0]
                downside_deviation = (
                    downside_returns.std() * np.sqrt(annualization_factor)
                    if len(downside_returns) > 0 else 0
                )
                strategy_metrics['sortino_ratio'] = (
                    excess_returns.mean() / downside_deviation * np.sqrt(annualization_factor)
                    if downside_deviation > 0 else 0
                )
            
            # Maximum drawdown
            if 'max_drawdown' in metrics:
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns - running_max) / running_max
                strategy_metrics['max_drawdown'] = drawdown.min()
            
            # Calmar ratio
            if 'calmar_ratio' in metrics and 'annualized_return' in strategy_metrics and 'max_drawdown' in strategy_metrics:
                strategy_metrics['calmar_ratio'] = (
                    -strategy_metrics['annualized_return'] / strategy_metrics['max_drawdown']
                    if strategy_metrics['max_drawdown'] < 0 else 0
                )
            
            # Best and worst months
            if ('best_month' in metrics or 'worst_month' in metrics) and len(returns) > 0:
                # Resample to monthly if not already monthly
                if frequency != 'M':
                    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                else:
                    monthly_returns = returns
                
                if 'best_month' in metrics:
                    strategy_metrics['best_month'] = monthly_returns.max()
                
                if 'worst_month' in metrics:
                    strategy_metrics['worst_month'] = monthly_returns.min()
            
            # Skewness and kurtosis
            if 'skewness' in metrics and len(returns) > 2:
                strategy_metrics['skewness'] = returns.skew()
            
            if 'kurtosis' in metrics and len(returns) > 3:
                strategy_metrics['kurtosis'] = returns.kurtosis()
            
            # Positive months and win rate
            if 'positive_months' in metrics:
                positive_months = (returns > 0).sum()
                strategy_metrics['positive_months'] = positive_months / len(returns)
            
            if 'win_rate' in metrics and benchmark_returns is not None:
                # Align returns with benchmark
                aligned_returns = returns.reindex(benchmark_returns.index).dropna()
                aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
                
                # Calculate win rate (outperformance rate)
                wins = (aligned_returns > aligned_benchmark).sum()
                strategy_metrics['win_rate'] = wins / len(aligned_returns) if len(aligned_returns) > 0 else 0
            
            # Add metrics to results DataFrame
            results[strategy_name] = pd.Series(strategy_metrics)
        
        return results
    
    def plot_comparison(self,
                       comparison_data: pd.DataFrame,
                       title: str = "Strategy Comparison",
                       interactive: bool = False,
                       plot_type: str = "bar") -> Any:
        """Plot strategy comparison results.
        
        Args:
            comparison_data: DataFrame with comparison results from compare()
            title: Chart title
            interactive: Whether to create an interactive plot
            plot_type: Type of plot ('bar', 'radar', or 'heatmap')
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if plot_type == "bar":
            return self._plot_bar_comparison(comparison_data, title, interactive)
        elif plot_type == "radar":
            return self._plot_radar_comparison(comparison_data, title, interactive)
        elif plot_type == "heatmap":
            return self._plot_heatmap_comparison(comparison_data, title, interactive)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'bar', 'radar', or 'heatmap'")
    
    def _plot_bar_comparison(self,
                           comparison_data: pd.DataFrame,
                           title: str,
                           interactive: bool) -> Any:
        """Create a bar chart comparison of strategies."""
        # Select metrics to display
        display_metrics = [
            'annualized_return', 'annualized_volatility', 'sharpe_ratio',
            'max_drawdown', 'sortino_ratio', 'calmar_ratio'
        ]
        
        # Filter metrics that exist in the data
        display_metrics = [m for m in display_metrics if m in comparison_data.index]
        
        # Extract data for selected metrics
        plot_data = comparison_data.loc[display_metrics]
        
        # Format metric names for display
        plot_data.index = [m.replace('_', ' ').title() for m in plot_data.index]
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive grouped bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each strategy
            for strategy in plot_data.columns:
                fig.add_trace(go.Bar(
                    name=strategy,
                    x=plot_data.index,
                    y=plot_data[strategy],
                    text=plot_data[strategy].apply(lambda x: f"{x:.4f}" if abs(x) < 0.01 else f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"),
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Metric",
                yaxis_title="Value",
                barmode='group',
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            return fig
        
        else:
            # Create static grouped bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            plot_data.T.plot(kind='bar', ax=ax)
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Strategy", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            
            # Format y-axis for percentage metrics
            percentage_metrics = ['Annualized Return', 'Max Drawdown', 'Positive Months', 'Win Rate']
            if any(metric in plot_data.index for metric in percentage_metrics):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title="Metric")
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_radar_comparison(self,
                             comparison_data: pd.DataFrame,
                             title: str,
                             interactive: bool) -> Any:
        """Create a radar chart comparison of strategies."""
        # Select metrics to display
        display_metrics = [
            'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'positive_months', 'win_rate'
        ]
        
        # Filter metrics that exist in the data
        display_metrics = [m for m in display_metrics if m in comparison_data.index]
        
        # Extract data for selected metrics
        plot_data = comparison_data.loc[display_metrics]
        
        # Format metric names for display
        plot_data.index = [m.replace('_', ' ').title() for m in plot_data.index]
        
        # Normalize data for radar chart (0-1 scale)
        normalized_data = plot_data.copy()
        for metric in normalized_data.index:
            if metric == 'Max Drawdown':  # Invert max drawdown (lower is better)
                normalized_data.loc[metric] = 1 - (plot_data.loc[metric] - plot_data.loc[metric].min()) / (
                    plot_data.loc[metric].max() - plot_data.loc[metric].min()) if plot_data.loc[metric].max() != plot_data.loc[metric].min() else 0.5
            else:  # Higher is better
                normalized_data.loc[metric] = (plot_data.loc[metric] - plot_data.loc[metric].min()) / (
                    plot_data.loc[metric].max() - plot_data.loc[metric].min()) if plot_data.loc[metric].max() != plot_data.loc[metric].min() else 0.5
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive radar chart with plotly
            fig = go.Figure()
            
            # Add traces for each strategy
            for strategy in normalized_data.columns:
                # Add first point at the end to close the polygon
                metrics = normalized_data.index.tolist()
                values = normalized_data[strategy].tolist()
                metrics.append(metrics[0])
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=strategy
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static radar chart with matplotlib
            # Number of metrics
            num_metrics = len(normalized_data.index)
            
            # Create figure and polar axis
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Set the angles for each metric (evenly spaced around the circle)
            angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
            
            # Close the polygon by repeating the first angle
            angles.append(angles[0])
            
            # Set the labels for each angle
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(normalized_data.index, fontsize=10)
            
            # Plot each strategy
            for i, strategy in enumerate(normalized_data.columns):
                values = normalized_data[strategy].tolist()
                values.append(values[0])  # Close the polygon
                
                ax.plot(angles, values, linewidth=2, label=strategy)
                ax.fill(angles, values, alpha=0.1)
            
            # Set y-limits
            ax.set_ylim(0, 1)
            
            # Add legend and title
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title(title, fontsize=14, y=1.08)
            
            return fig
    
    def _plot_heatmap_comparison(self,
                               comparison_data: pd.DataFrame,
                               title: str,
                               interactive: bool) -> Any:
        """Create a heatmap comparison of strategies."""
        # Format metric names for display
        plot_data = comparison_data.copy()
        plot_data.index = [m.replace('_', ' ').title() for m in plot_data.index]
        
        # Normalize data for heatmap (z-score normalization)
        normalized_data = plot_data.copy()
        for metric in normalized_data.index:
            if metric == 'Max Drawdown':  # Invert max drawdown (lower is better)
                normalized_data.loc[metric] = -stats.zscore(plot_data.loc[metric])
            else:  # Higher is better
                normalized_data.loc[metric] = stats.zscore(plot_data.loc[metric])
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive heatmap with plotly
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                z=normalized_data.values,
                x=normalized_data.columns,
                y=normalized_data.index,
                colorscale=[
                    [0, 'red'],
                    [0.5, 'white'],
                    [1, 'green']
                ],
                zmid=0,
                text=[[f"{plot_data.iloc[i, j]:.2%}" if abs(plot_data.iloc[i, j]) < 10 else f"{plot_data.iloc[i, j]:.2f}" 
                       for j in range(plot_data.shape[1])] for i in range(plot_data.shape[0])],
                texttemplate="%{text}",
                colorbar=dict(title="Z-Score")
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Strategy",
                yaxis_title="Metric",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static heatmap with matplotlib and seaborn
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(
                normalized_data,
                annot=plot_data.applymap(lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}"),
                fmt="",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                ax=ax
            )
            
            # Set title
            ax.set_title(title, fontsize=14)
            
            # Rotate y-axis labels for better readability
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class ScenarioAnalyzer(ComparativeAnalyzer):
    """Analyzer for scenario analysis and stress testing."""
    
    def __init__(self):
        """Initialize the scenario analyzer."""
        super().__init__()
    
    def define_scenario(
        self,
        name: str,
        asset_returns: Dict[str, float],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Define a scenario with expected asset returns.
        
        Args:
            name: Scenario name
            asset_returns: Dictionary mapping asset names to expected returns
            description: Optional description of the scenario
            
        Returns:
            Dictionary with scenario definition
        """
        return {
            'name': name,
            'asset_returns': asset_returns,
            'description': description
        }
    
    def analyze_scenario(
        self,
        portfolio_weights: Dict[str, float],
        scenario: Dict[str, Any]
    ) -> float:
        """Analyze the impact of a scenario on a portfolio.
        
        Args:
            portfolio_weights: Dictionary mapping asset names to portfolio weights
            scenario: Scenario definition from define_scenario()
            
        Returns:
            Expected portfolio return under the scenario
        """
        # Extract asset returns from scenario
        asset_returns = scenario['asset_returns']
        
        # Calculate portfolio return
        portfolio_return = 0.0
        for asset, weight in portfolio_weights.items():
            if asset in asset_returns:
                portfolio_return += weight * asset_returns[asset]
        
        return portfolio_return
    
    def compare(
        self,
        portfolios: Dict[str, Dict[str, float]],
        scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Compare multiple portfolios under different scenarios.
        
        Args:
            portfolios: Dictionary mapping portfolio names to weight dictionaries
            scenarios: List of scenario definitions from define_scenario()
            
        Returns:
            DataFrame with portfolio returns under each scenario
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=[s['name'] for s in scenarios], columns=list(portfolios.keys()))
        
        # Calculate returns for each portfolio under each scenario
        for scenario in scenarios:
            scenario_name = scenario['name']
            for portfolio_name, weights in portfolios.items():
                results.loc[scenario_name, portfolio_name] = self.analyze_scenario(weights, scenario)
        
        return results
    
    def plot_comparison(self,
                       comparison_data: pd.DataFrame,
                       title: str = "Scenario Analysis",
                       interactive: bool = False,
                       plot_type: str = "heatmap") -> Any:
        """Plot scenario analysis results.
        
        Args:
            comparison_data: DataFrame with comparison results from compare()
            title: Chart title
            interactive: Whether to create an interactive plot
            plot_type: Type of plot ('heatmap' or 'bar')
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if plot_type == "heatmap":
            return self._plot_heatmap_comparison(comparison_data, title, interactive)
        elif plot_type == "bar":
            return self._plot_bar_comparison(comparison_data, title, interactive)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'heatmap' or 'bar'")
    
    def _plot_heatmap_comparison(self,
                               comparison_data: pd.DataFrame,
                               title: str,
                               interactive: bool) -> Any:
        """Create a heatmap of scenario analysis results."""
        # Convert to percentage for display
        plot_data = comparison_data * 100
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive heatmap with plotly
            fig = go.Figure()
            
            fig.add_trace(go.Heatmap(
                z=plot_data.values,
                x=plot_data.columns,
                y=plot_data.index,
                colorscale=[
                    [0, 'red'],
                    [0.5, 'white'],
                    [1, 'green']
                ],
                zmid=0,
                text=[[f"{val:.2f}%" for val in row] for row in plot_data.values],
                texttemplate="%{text}",
                colorbar=dict(title="Return (%)")
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Portfolio",
                yaxis_title="Scenario",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static heatmap with matplotlib and seaborn
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(
                plot_data,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                ax=ax
            )
            
            # Set title
            ax.set_title(title, fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_bar_comparison(self,
                           comparison_data: pd.DataFrame,
                           title: str,
                           interactive: bool) -> Any:
        """Create a bar chart of scenario analysis results."""
        # Convert to percentage for display
        plot_data = comparison_data * 100
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive grouped bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each portfolio
            for portfolio in plot_data.columns:
                fig.add_trace(go.Bar(
                    name=portfolio,
                    x=plot_data.index,
                    y=plot_data[portfolio],
                    text=plot_data[portfolio].apply(lambda x: f"{x:.2f}%"),
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Scenario",
                yaxis_title="Return (%)",
                barmode='group',
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            return fig
        
        else:
            # Create static grouped bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            plot_data.plot(kind='bar', ax=ax)
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Scenario", fontsize=12)
            ax.set_ylabel("Return (%)", fontsize=12)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title="Portfolio")
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class MonteCarloSimulator(ComparativeAnalyzer):
    """Monte Carlo simulator for portfolio analysis."""
    
    def __init__(self):
        """Initialize the Monte Carlo simulator."""
        super().__init__()
    
    def simulate(
        self,
        portfolio_weights: Dict[str, float],
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        num_simulations: int = 1000,
        time_horizon: int = 252,  # Default to 1 year of daily returns
        initial_investment: float = 10000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a portfolio.
        
        Args:
            portfolio_weights: Dictionary mapping asset names to portfolio weights
            expected_returns: Dictionary mapping asset names to expected returns
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            num_simulations: Number of simulations to run
            time_horizon: Time horizon in days
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary with simulation results
        """
        # Convert inputs to numpy arrays
        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[asset] for asset in assets])
        returns = np.array([expected_returns[asset] for asset in assets])
        
        # Extract covariance matrix for the assets in the portfolio
        cov_matrix = covariance_matrix.loc[assets, assets].values
        
        # Calculate portfolio expected return and volatility
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.multivariate_normal(
            returns / 252,  # Convert annual returns to daily
            cov_matrix / 252,  # Convert annual covariance to daily
            (num_simulations, time_horizon)
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.sum(daily_returns * weights, axis=2)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        
        # Calculate final values
        final_values = initial_investment * cumulative_returns[:, -1]
        
        # Calculate statistics
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        min_final_value = np.min(final_values)
        max_final_value = np.max(final_values)
        
        # Calculate percentiles
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
        
        # Store simulation paths and results
        simulation_paths = initial_investment * cumulative_returns
        
        return {
            'simulation_paths': simulation_paths,
            'final_values': final_values,
            'mean_final_value': mean_final_value,
            'median_final_value': median_final_value,
            'min_final_value': min_final_value,
            'max_final_value': max_final_value,
            'percentiles': {
                '5%': percentiles[0],
                '25%': percentiles[1],
                '50%': percentiles[2],
                '75%': percentiles[3],
                '95%': percentiles[4]
            },
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0,
            'initial_investment': initial_investment,
            'time_horizon': time_horizon
        }
    
    def compare(
        self,
        portfolios: Dict[str, Dict[str, float]],
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        num_simulations: int = 1000,
        time_horizon: int = 252,
        initial_investment: float = 10000
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple portfolios using Monte Carlo simulation.
        
        Args:
            portfolios: Dictionary mapping portfolio names to weight dictionaries
            expected_returns: Dictionary mapping asset names to expected returns
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            num_simulations: Number of simulations to run
            time_horizon: Time horizon in days
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary mapping portfolio names to simulation results
        """
        results = {}
        
        for portfolio_name, weights in portfolios.items():
            results[portfolio_name] = self.simulate(
                weights,
                expected_returns,
                covariance_matrix,
                risk_free_rate,
                num_simulations,
                time_horizon,
                initial_investment
            )
        
        return results
    
    def plot_comparison(self,
                       comparison_data: Dict[str, Dict[str, Any]],
                       title: str = "Monte Carlo Simulation Comparison",
                       interactive: bool = False,
                       plot_type: str = "paths") -> Any:
        """Plot Monte Carlo simulation comparison results.
        
        Args:
            comparison_data: Dictionary with comparison results from compare()
            title: Chart title
            interactive: Whether to create an interactive plot
            plot_type: Type of plot ('paths', 'histogram', or 'boxplot')
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if plot_type == "paths":
            return self._plot_paths_comparison(comparison_data, title, interactive)
        elif plot_type == "histogram":
            return self._plot_histogram_comparison(comparison_data, title, interactive)
        elif plot_type == "boxplot":
            return self._plot_boxplot_comparison(comparison_data, title, interactive)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'paths', 'histogram', or 'boxplot'")
    
    def _plot_paths_comparison(self,
                             comparison_data: Dict[str, Dict[str, Any]],
                             title: str,
                             interactive: bool) -> Any:
        """Create a comparison of Monte Carlo simulation paths."""
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive line chart with plotly
            fig = go.Figure()
            
            # Add a subset of paths for each portfolio
            for portfolio_name, results in comparison_data.items():
                paths = results['simulation_paths']
                time_horizon = results['time_horizon']
                
                # Only plot a subset of paths to avoid overcrowding
                num_paths_to_plot = min(50, paths.shape[0])
                indices = np.linspace(0, paths.shape[0] - 1, num_paths_to_plot, dtype=int)
                
                # Create x-axis values (days)
                x = np.arange(time_horizon + 1)  # Include day 0
                
                # Add initial investment at day 0
                initial_investment = results['initial_investment']
                
                for i in indices:
                    # Add initial investment at day 0
                    path_with_initial = np.insert(paths[i], 0, initial_investment)
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=path_with_initial,
                        mode='lines',
                        opacity=0.3,
                        line=dict(width=1),
                        showlegend=False,
                        name=portfolio_name
                    ))
                
                # Add mean path
                mean_path = np.mean(paths, axis=0)
                mean_path_with_initial = np.insert(mean_path, 0, initial_investment)
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=mean_path_with_initial,
                    mode='lines',
                    line=dict(width=3, color=px.colors.qualitative.Plotly[list(comparison_data.keys()).index(portfolio_name)]),
                    name=f"{portfolio_name} (Mean)"
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Time (Days)",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            return fig
        
        else:
            # Create static line chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot paths for each portfolio
            for i, (portfolio_name, results) in enumerate(comparison_data.items()):
                paths = results['simulation_paths']
                time_horizon = results['time_horizon']
                
                # Only plot a subset of paths to avoid overcrowding
                num_paths_to_plot = min(50, paths.shape[0])
                indices = np.linspace(0, paths.shape[0] - 1, num_paths_to_plot, dtype=int)
                
                # Create x-axis values (days)
                x = np.arange(time_horizon + 1)  # Include day 0
                
                # Add initial investment at day 0
                initial_investment = results['initial_investment']
                
                # Get color for this portfolio
                color = plt.cm.tab10(i % 10)
                
                for j in indices:
                    # Add initial investment at day 0
                    path_with_initial = np.insert(paths[j], 0, initial_investment)
                    
                    ax.plot(x, path_with_initial, color=color, alpha=0.1, linewidth=0.5)
                
                # Add mean path
                mean_path = np.mean(paths, axis=0)
                mean_path_with_initial = np.insert(mean_path, 0, initial_investment)
                
                ax.plot(x, mean_path_with_initial, color=color, linewidth=2, label=f"{portfolio_name} (Mean)")
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Time (Days)", fontsize=12)
            ax.set_ylabel("Portfolio Value ($)", fontsize=12)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_histogram_comparison(self,
                                 comparison_data: Dict[str, Dict[str, Any]],
                                 title: str,
                                 interactive: bool) -> Any:
        """Create a histogram comparison of final values."""
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive histogram with plotly
            fig = go.Figure()
            
            # Add histogram for each portfolio
            for portfolio_name, results in comparison_data.items():
                final_values = results['final_values']
                
                fig.add_trace(go.Histogram(
                    x=final_values,
                    name=portfolio_name,
                    opacity=0.7,
                    nbinsx=30
                ))
                
                # Add vertical line for mean
                fig.add_trace(go.Scatter(
                    x=[results['mean_final_value'], results['mean_final_value']],
                    y=[0, 100],  # Will be scaled automatically
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    name=f"{portfolio_name} Mean"
                ))
            
            # Update layout
            fig.update_layout(
                title=title + " (Final Values)",
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Frequency",
                barmode='overlay',
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            return fig
        
        else:
            # Create static histogram with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot histogram for each portfolio
            for i, (portfolio_name, results) in enumerate(comparison_data.items()):
                final_values = results['final_values']
                
                # Get color for this portfolio
                color = plt.cm.tab10(i % 10)
                
                ax.hist(final_values, bins=30, alpha=0.7, color=color, label=portfolio_name)
                
                # Add vertical line for mean
                ax.axvline(results['mean_final_value'], color=color, linestyle='--', linewidth=2)
                
                # Add text for mean
                ax.text(
                    results['mean_final_value'],
                    ax.get_ylim()[1] * 0.9 - i * ax.get_ylim()[1] * 0.1,
                    f"{portfolio_name} Mean: ${results['mean_final_value']:.2f}",
                    color=color,
                    fontsize=10,
                    ha='center'
                )
            
            # Set labels and title
            ax.set_title(title + " (Final Values)", fontsize=14)
            ax.set_xlabel("Final Portfolio Value ($)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_boxplot_comparison(self,
                               comparison_data: Dict[str, Dict[str, Any]],
                               title: str,
                               interactive: bool) -> Any:
        """Create a boxplot comparison of final values."""
        # Extract final values for each portfolio
        final_values = {}
        for portfolio_name, results in comparison_data.items():
            final_values[portfolio_name] = results['final_values']
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive box plot with plotly
            fig = go.Figure()
            
            # Add box plot for each portfolio
            for portfolio_name, values in final_values.items():
                fig.add_trace(go.Box(
                    y=values,
                    name=portfolio_name,
                    boxmean=True  # Show mean as a dashed line
                ))
            
            # Update layout
            fig.update_layout(
                title=title + " (Final Values)",
                xaxis_title="Portfolio",
                yaxis_title="Final Portfolio Value ($)",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static box plot with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create box plot
            ax.boxplot(
                list(final_values.values()),
                labels=list(final_values.keys()),
                showmeans=True,
                meanline=True
            )
            
            # Set labels and title
            ax.set_title(title + " (Final Values)", fontsize=14)
            ax.set_xlabel("Portfolio", fontsize=12)
            ax.set_ylabel("Final Portfolio Value ($)", fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


# Export all analyzer classes
__all__ = [
    'ComparativeAnalyzer',
    'StrategyComparator',
    'ScenarioAnalyzer',
    'MonteCarloSimulator'
]