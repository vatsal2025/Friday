"""Attribution analysis module for portfolio performance attribution.

This module provides tools for analyzing and attributing portfolio performance
to various factors, including sector allocation, security selection, and factor exposures.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class AttributionAnalyzer:
    """Base class for attribution analysis."""
    
    def __init__(self):
        """Initialize the attribution analyzer."""
        pass
    
    def calculate_attribution(self, *args, **kwargs) -> pd.DataFrame:
        """Calculate attribution effects.
        
        This is a base method that should be implemented by subclasses.
        
        Returns:
            DataFrame with attribution results
        """
        raise NotImplementedError("Subclasses must implement calculate_attribution()")
    
    def plot_attribution(self, *args, **kwargs) -> Any:
        """Plot attribution effects.
        
        This is a base method that should be implemented by subclasses.
        
        Returns:
            Figure object
        """
        raise NotImplementedError("Subclasses must implement plot_attribution()")


class BrinsionAttributionAnalyzer(AttributionAnalyzer):
    """Brinson attribution model for sector-based performance attribution.
    
    The Brinson model decomposes portfolio performance into allocation effect,
    selection effect, and interaction effect.
    """
    
    def __init__(self):
        """Initialize the Brinson attribution analyzer."""
        super().__init__()
    
    def calculate_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        total_portfolio_return: float,
        total_benchmark_return: float
    ) -> pd.DataFrame:
        """Calculate Brinson attribution effects.
        
        Args:
            portfolio_weights: DataFrame with portfolio weights by sector
            benchmark_weights: DataFrame with benchmark weights by sector
            portfolio_returns: DataFrame with portfolio returns by sector
            benchmark_returns: DataFrame with benchmark returns by sector
            total_portfolio_return: Total portfolio return
            total_benchmark_return: Total benchmark return
            
        Returns:
            DataFrame with allocation, selection, and interaction effects by sector
        """
        # Ensure all inputs are aligned
        sectors = portfolio_weights.index
        result = pd.DataFrame(index=sectors)
        
        # Store input data
        result['portfolio_weight'] = portfolio_weights
        result['benchmark_weight'] = benchmark_weights
        result['portfolio_return'] = portfolio_returns
        result['benchmark_return'] = benchmark_returns
        
        # Calculate active weights and returns
        result['active_weight'] = result['portfolio_weight'] - result['benchmark_weight']
        result['active_return'] = result['portfolio_return'] - result['benchmark_return']
        
        # Calculate attribution effects
        # Allocation effect: (portfolio_weight - benchmark_weight) * benchmark_return
        result['allocation_effect'] = result['active_weight'] * result['benchmark_return']
        
        # Selection effect: benchmark_weight * (portfolio_return - benchmark_return)
        result['selection_effect'] = result['benchmark_weight'] * result['active_return']
        
        # Interaction effect: (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
        result['interaction_effect'] = result['active_weight'] * result['active_return']
        
        # Calculate total contribution
        result['total_effect'] = result['allocation_effect'] + result['selection_effect'] + result['interaction_effect']
        
        # Calculate contribution to active return
        active_return = total_portfolio_return - total_benchmark_return
        result['contribution_to_active'] = result['total_effect'] / active_return if active_return != 0 else 0
        
        return result
    
    def plot_attribution(self,
                        attribution_data: pd.DataFrame,
                        title: str = "Brinson Attribution Analysis",
                        interactive: bool = False) -> Any:
        """Plot attribution effects.
        
        Args:
            attribution_data: DataFrame with attribution results from calculate_attribution
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        # Extract relevant columns
        plot_data = attribution_data[['allocation_effect', 'selection_effect', 'interaction_effect']].copy()
        
        # Sort by total effect
        plot_data['total_effect'] = plot_data.sum(axis=1)
        plot_data = plot_data.sort_values('total_effect', ascending=False)
        
        # Drop the total effect column as we don't want to plot it
        plot_data = plot_data.drop('total_effect', axis=1)
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive stacked bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each effect
            for column in plot_data.columns:
                fig.add_trace(go.Bar(
                    name=column.replace('_', ' ').title(),
                    x=plot_data.index,
                    y=plot_data[column],
                    text=plot_data[column].apply(lambda x: f"{x:.2%}"),
                    textposition='auto'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Sector",
                yaxis_title="Attribution Effect",
                barmode='relative',
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=".2%")
            
            return fig
        
        else:
            # Create static stacked bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create stacked bar chart
            plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Sector", fontsize=12)
            ax.set_ylabel("Attribution Effect", fontsize=12)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title="Effect Type")
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class FactorAttributionAnalyzer(AttributionAnalyzer):
    """Factor-based attribution analysis.
    
    Decomposes portfolio performance into contributions from various factors
    such as market, size, value, momentum, etc.
    """
    
    def __init__(self):
        """Initialize the factor attribution analyzer."""
        super().__init__()
    
    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        specific_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculate factor attribution effects.
        
        Args:
            portfolio_returns: Series of portfolio returns
            factor_returns: DataFrame of factor returns
            factor_exposures: DataFrame of portfolio exposures to factors
            specific_returns: Series of specific (idiosyncratic) returns
            
        Returns:
            DataFrame with factor contributions and summary statistics
        """
        # Align all inputs to the same dates
        common_dates = portfolio_returns.index.intersection(
            factor_returns.index.intersection(factor_exposures.index)
        )
        
        portfolio_returns = portfolio_returns.loc[common_dates]
        factor_returns = factor_returns.loc[common_dates]
        factor_exposures = factor_exposures.loc[common_dates]
        
        if specific_returns is not None:
            specific_returns = specific_returns.loc[common_dates]
        
        # Calculate factor contributions for each period
        factor_contributions = pd.DataFrame(index=common_dates, columns=factor_returns.columns)
        
        for date in common_dates:
            for factor in factor_returns.columns:
                factor_contributions.loc[date, factor] = (
                    factor_exposures.loc[date, factor] * factor_returns.loc[date, factor]
                )
        
        # Calculate cumulative contributions
        cumulative_contributions = factor_contributions.cumsum()
        
        # Calculate total contribution by factor
        total_contribution_by_factor = factor_contributions.sum()
        
        # Calculate percentage contribution to total return
        total_portfolio_return = portfolio_returns.sum()
        percentage_contribution = total_contribution_by_factor / total_portfolio_return if total_portfolio_return != 0 else 0
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'total_contribution': total_contribution_by_factor,
            'percentage_contribution': percentage_contribution
        })
        
        # Calculate specific contribution if specific returns are provided
        if specific_returns is not None:
            specific_contribution = specific_returns.sum()
            specific_percentage = specific_contribution / total_portfolio_return if total_portfolio_return != 0 else 0
            
            # Add specific contribution to summary
            summary.loc['Specific'] = [specific_contribution, specific_percentage]
        
        # Calculate average factor exposures
        average_exposures = factor_exposures.mean()
        summary['average_exposure'] = average_exposures
        
        # For specific returns, set average exposure to 1
        if specific_returns is not None:
            summary.loc['Specific', 'average_exposure'] = 1.0
        
        # Store period-by-period contributions for later analysis
        self.factor_contributions = factor_contributions
        self.cumulative_contributions = cumulative_contributions
        
        return summary
    
    def plot_attribution(self,
                        attribution_data: pd.DataFrame,
                        title: str = "Factor Attribution Analysis",
                        interactive: bool = False,
                        plot_type: str = "bar") -> Any:
        """Plot factor attribution effects.
        
        Args:
            attribution_data: DataFrame with attribution results from calculate_attribution
            title: Chart title
            interactive: Whether to create an interactive plot
            plot_type: Type of plot ('bar', 'waterfall', or 'time_series')
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if plot_type == "bar":
            return self._plot_bar_attribution(attribution_data, title, interactive)
        elif plot_type == "waterfall":
            return self._plot_waterfall_attribution(attribution_data, title, interactive)
        elif plot_type == "time_series":
            if not hasattr(self, 'cumulative_contributions'):
                raise ValueError("Time series plot requires calculate_attribution to be called first")
            return self._plot_time_series_attribution(title, interactive)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'bar', 'waterfall', or 'time_series'")
    
    def _plot_bar_attribution(self,
                            attribution_data: pd.DataFrame,
                            title: str,
                            interactive: bool) -> Any:
        """Create a bar chart of factor attribution."""
        # Extract relevant columns and sort
        plot_data = attribution_data['percentage_contribution'].sort_values(ascending=False)
        
        # Convert to percentage for display
        plot_data = plot_data * 100
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each factor
            colors = [px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i in range(len(plot_data))]            
            
            fig.add_trace(go.Bar(
                x=plot_data.index,
                y=plot_data.values,
                marker_color=colors,
                text=plot_data.apply(lambda x: f"{x:.2f}%"),
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Factor",
                yaxis_title="Contribution to Return (%)",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bar chart
            bars = ax.bar(plot_data.index, plot_data.values, color=sns.color_palette("viridis", len(plot_data)))
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}%",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Factor", fontsize=12)
            ax.set_ylabel("Contribution to Return (%)", fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_waterfall_attribution(self,
                                  attribution_data: pd.DataFrame,
                                  title: str,
                                  interactive: bool) -> Any:
        """Create a waterfall chart of factor attribution."""
        # Extract relevant columns and sort by absolute contribution
        plot_data = attribution_data['total_contribution'].sort_values(key=abs, ascending=False)
        
        # Add total as the last item
        total = plot_data.sum()
        plot_data = pd.concat([plot_data, pd.Series({'Total': total})])
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive waterfall chart with plotly
            fig = go.Figure()
            
            # Add bars for each factor
            measure = ['relative'] * (len(plot_data) - 1) + ['total']
            
            # Determine colors based on positive/negative values
            colors = []
            for value in plot_data.values:
                if value > 0:
                    colors.append('green')
                elif value < 0:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            fig.add_trace(go.Waterfall(
                name="Factor Attribution",
                orientation="v",
                measure=measure,
                x=plot_data.index,
                y=plot_data.values,
                connector={"line":{"color":"rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"red"}},
                increasing={"marker":{"color":"green"}},
                totals={"marker":{"color":"blue"}},
                text=plot_data.apply(lambda x: f"{x:.2%}"),
                textposition='outside'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Factor",
                yaxis_title="Contribution to Return",
                template="plotly_white",
                showlegend=False
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=".2%")
            
            return fig
        
        else:
            # Create static waterfall chart with matplotlib
            # This is more complex with matplotlib, so we'll use a bar chart with invisible bars
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create data for waterfall chart
            indices = range(len(plot_data))
            cumulative = plot_data.cumsum()
            cumulative_prev = cumulative.shift(1).fillna(0)
            
            # Create invisible bars for the base of each segment
            ax.bar(indices, cumulative_prev, width=0.5, color='none', edgecolor='none')
            
            # Create visible bars for each segment
            for i, (idx, value) in enumerate(plot_data.items()):
                color = 'green' if value >= 0 else 'red'
                if idx == 'Total':
                    color = 'blue'
                
                ax.bar(i, value, bottom=cumulative_prev[i], width=0.5, color=color, alpha=0.7)
                
                # Add value labels
                if idx == 'Total':
                    label = f"Total: {value:.2%}"
                else:
                    label = f"{value:.2%}"
                
                ax.text(i, cumulative[i] + (0.01 if value >= 0 else -0.01),
                       label, ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
            
            # Add connecting lines
            for i in range(len(plot_data) - 1):
                ax.plot([i + 0.25, i + 0.75], [cumulative[i], cumulative[i]], 'k--', alpha=0.3)
            
            # Set labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Factor", fontsize=12)
            ax.set_ylabel("Contribution to Return", fontsize=12)
            
            # Set x-tick labels
            ax.set_xticks(indices)
            ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_time_series_attribution(self,
                                    title: str,
                                    interactive: bool) -> Any:
        """Create a time series chart of cumulative factor contributions."""
        # Get cumulative contributions data
        plot_data = self.cumulative_contributions
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive line chart with plotly
            fig = go.Figure()
            
            # Add lines for each factor
            for factor in plot_data.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data[factor],
                    mode='lines',
                    name=factor
                ))
            
            # Update layout
            fig.update_layout(
                title=title + " (Cumulative)",
                xaxis_title="Date",
                yaxis_title="Cumulative Contribution",
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=".2%")
            
            return fig
        
        else:
            # Create static line chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot cumulative contributions for each factor
            plot_data.plot(ax=ax, linewidth=2)
            
            # Set labels and title
            ax.set_title(title + " (Cumulative)", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Contribution", fontsize=12)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(title="Factor")
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class RiskAttributionAnalyzer(AttributionAnalyzer):
    """Risk attribution analysis.
    
    Analyzes the contribution of different assets or factors to portfolio risk.
    """
    
    def __init__(self):
        """Initialize the risk attribution analyzer."""
        super().__init__()
    
    def calculate_attribution(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        portfolio_volatility: Optional[float] = None
    ) -> pd.DataFrame:
        """Calculate risk attribution effects.
        
        Args:
            weights: Series of portfolio weights
            covariance_matrix: Covariance matrix of asset returns
            portfolio_volatility: Portfolio volatility (if None, will be calculated)
            
        Returns:
            DataFrame with risk contributions and percentage contributions
        """
        # Ensure weights and covariance matrix are aligned
        weights = weights.reindex(covariance_matrix.index)
        
        # Calculate portfolio volatility if not provided
        if portfolio_volatility is None:
            portfolio_variance = weights.dot(covariance_matrix).dot(weights)
            portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate marginal contributions to risk (MCTR)
        mctr = covariance_matrix.dot(weights) / portfolio_volatility
        
        # Calculate component contributions to risk (CCTR)
        cctr = weights * mctr
        
        # Calculate percentage contributions to risk
        pctr = cctr / portfolio_volatility
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'weight': weights,
            'marginal_contribution': mctr,
            'risk_contribution': cctr,
            'percentage_contribution': pctr
        })
        
        # Add portfolio volatility as an attribute
        self.portfolio_volatility = portfolio_volatility
        
        return summary
    
    def plot_attribution(self,
                        attribution_data: pd.DataFrame,
                        title: str = "Risk Attribution Analysis",
                        interactive: bool = False,
                        plot_type: str = "pie") -> Any:
        """Plot risk attribution effects.
        
        Args:
            attribution_data: DataFrame with attribution results from calculate_attribution
            title: Chart title
            interactive: Whether to create an interactive plot
            plot_type: Type of plot ('pie' or 'bar')
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if plot_type == "pie":
            return self._plot_pie_attribution(attribution_data, title, interactive)
        elif plot_type == "bar":
            return self._plot_bar_attribution(attribution_data, title, interactive)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'pie' or 'bar'")
    
    def _plot_pie_attribution(self,
                            attribution_data: pd.DataFrame,
                            title: str,
                            interactive: bool) -> Any:
        """Create a pie chart of risk attribution."""
        # Extract percentage contributions and sort
        plot_data = attribution_data['percentage_contribution'].sort_values(ascending=False)
        
        # Convert to percentage for display
        plot_data = plot_data * 100
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive pie chart with plotly
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=plot_data.index,
                values=plot_data.values,
                textinfo='label+percent',
                insidetextorientation='radial',
                hole=0.3
            ))
            
            # Update layout
            fig.update_layout(
                title=title + f" (Portfolio Volatility: {self.portfolio_volatility:.2%})",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static pie chart with matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                plot_data.values,
                labels=plot_data.index,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                textprops={'fontsize': 9}
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Set title
            ax.set_title(title + f" (Portfolio Volatility: {self.portfolio_volatility:.2%})", fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def _plot_bar_attribution(self,
                            attribution_data: pd.DataFrame,
                            title: str,
                            interactive: bool) -> Any:
        """Create a bar chart of risk attribution."""
        # Extract percentage contributions and sort
        plot_data = attribution_data['percentage_contribution'].sort_values(ascending=False)
        
        # Convert to percentage for display
        plot_data = plot_data * 100
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each asset
            fig.add_trace(go.Bar(
                x=plot_data.index,
                y=plot_data.values,
                marker_color=px.colors.qualitative.Plotly,
                text=plot_data.apply(lambda x: f"{x:.2f}%"),
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title=title + f" (Portfolio Volatility: {self.portfolio_volatility:.2%})",
                xaxis_title="Asset",
                yaxis_title="Contribution to Risk (%)",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bar chart
            bars = ax.bar(plot_data.index, plot_data.values, color=sns.color_palette("viridis", len(plot_data)))
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}%",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
            
            # Set labels and title
            ax.set_title(title + f" (Portfolio Volatility: {self.portfolio_volatility:.2%})", fontsize=14)
            ax.set_xlabel("Asset", fontsize=12)
            ax.set_ylabel("Contribution to Risk (%)", fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


# Export all analyzer classes
__all__ = [
    'AttributionAnalyzer',
    'BrinsionAttributionAnalyzer',
    'FactorAttributionAnalyzer',
    'RiskAttributionAnalyzer'
]