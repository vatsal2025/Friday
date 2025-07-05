# Enhanced Reporting and Analytics - Visualization Module

"""
This module provides comprehensive visualization capabilities for portfolio analysis,
including performance, allocation, risk, and tax visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from enum import Enum

# Import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import bokeh for web-based visualizations
try:
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot, column, row
    from bokeh.models import ColumnDataSource, HoverTool, Legend
    from bokeh.palettes import Category10, Viridis256
    from bokeh.io import output_file, save
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from .reporting import VisualizationType, ReportFormat


class BaseVisualizer:
    """Base class for all visualizers."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """Initialize the base visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set default color schemes
        self.color_scheme = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "positive": "#2ca02c",
            "negative": "#d62728",
            "neutral": "#7f7f7f",
            "highlight": "#9467bd",
            "benchmark": "#8c564b"
        }
        
        # Set default figure size
        self.figure_size = (10, 6)
        
        # Set default DPI for raster formats
        self.dpi = 100
        
        # Set default font sizes
        self.font_sizes = {
            "title": 16,
            "subtitle": 14,
            "axis": 12,
            "tick": 10,
            "legend": 10,
            "annotation": 9
        }
    
    def set_style(self, style_dict: Dict[str, Any]) -> None:
        """Update the visualizer style.
        
        Args:
            style_dict: Dictionary with style settings
        """
        if "color_scheme" in style_dict:
            self.color_scheme.update(style_dict["color_scheme"])
        
        if "figure_size" in style_dict:
            self.figure_size = style_dict["figure_size"]
        
        if "dpi" in style_dict:
            self.dpi = style_dict["dpi"]
        
        if "font_sizes" in style_dict:
            self.font_sizes.update(style_dict["font_sizes"])
    
    def save_figure(self, fig, filename: str, formats: List[ReportFormat] = None) -> Dict[ReportFormat, str]:
        """Save a figure in multiple formats.
        
        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            formats: List of formats to save (if None, save as PNG)
            
        Returns:
            Dict: Dictionary mapping formats to file paths
        """
        if formats is None:
            formats = [ReportFormat.PNG]
        
        saved_files = {}
        
        for fmt in formats:
            if fmt == ReportFormat.PNG:
                filepath = os.path.join(self.output_dir, f"{filename}.png")
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.JPG or fmt == ReportFormat.JPEG:
                filepath = os.path.join(self.output_dir, f"{filename}.jpg")
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.SVG:
                filepath = os.path.join(self.output_dir, f"{filename}.svg")
                fig.savefig(filepath, bbox_inches='tight')
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.PDF:
                filepath = os.path.join(self.output_dir, f"{filename}.pdf")
                fig.savefig(filepath, bbox_inches='tight')
                saved_files[fmt] = filepath
        
        return saved_files
    
    def save_plotly_figure(self, fig, filename: str, formats: List[ReportFormat] = None) -> Dict[ReportFormat, str]:
        """Save a Plotly figure in multiple formats.
        
        Args:
            fig: Plotly figure to save
            filename: Base filename (without extension)
            formats: List of formats to save (if None, save as HTML)
            
        Returns:
            Dict: Dictionary mapping formats to file paths
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not available. Install it with 'pip install plotly'.")
        
        if formats is None:
            formats = [ReportFormat.HTML]
        
        saved_files = {}
        
        for fmt in formats:
            if fmt == ReportFormat.HTML or fmt == ReportFormat.INTERACTIVE:
                filepath = os.path.join(self.output_dir, f"{filename}.html")
                fig.write_html(filepath, include_plotlyjs=True, full_html=True)
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.PNG:
                filepath = os.path.join(self.output_dir, f"{filename}.png")
                fig.write_image(filepath, width=self.figure_size[0]*100, height=self.figure_size[1]*100, scale=1)
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.JPG or fmt == ReportFormat.JPEG:
                filepath = os.path.join(self.output_dir, f"{filename}.jpg")
                fig.write_image(filepath, width=self.figure_size[0]*100, height=self.figure_size[1]*100, scale=1)
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.SVG:
                filepath = os.path.join(self.output_dir, f"{filename}.svg")
                fig.write_image(filepath)
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.PDF:
                filepath = os.path.join(self.output_dir, f"{filename}.pdf")
                fig.write_image(filepath)
                saved_files[fmt] = filepath
            
            elif fmt == ReportFormat.JSON:
                filepath = os.path.join(self.output_dir, f"{filename}.json")
                fig.write_json(filepath)
                saved_files[fmt] = filepath
        
        return saved_files
    
    def save_bokeh_figure(self, fig, filename: str, formats: List[ReportFormat] = None) -> Dict[ReportFormat, str]:
        """Save a Bokeh figure in multiple formats.
        
        Args:
            fig: Bokeh figure to save
            filename: Base filename (without extension)
            formats: List of formats to save (if None, save as HTML)
            
        Returns:
            Dict: Dictionary mapping formats to file paths
        """
        if not BOKEH_AVAILABLE:
            raise ImportError("Bokeh is not available. Install it with 'pip install bokeh'.")
        
        if formats is None:
            formats = [ReportFormat.HTML]
        
        saved_files = {}
        
        for fmt in formats:
            if fmt == ReportFormat.HTML or fmt == ReportFormat.INTERACTIVE:
                filepath = os.path.join(self.output_dir, f"{filename}.html")
                output_file(filepath, title=filename)
                save(fig)
                saved_files[fmt] = filepath
            
            # Bokeh doesn't directly support other formats, but can be exported via selenium
            # This would require additional dependencies, so we'll skip it for now
        
        return saved_files


class PerformanceVisualizer(BaseVisualizer):
    """Visualizer for performance-related charts and graphs."""
    
    def __init__(self, output_dir: str = "./visualizations/performance"):
        """Initialize the performance visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_equity_curve(self, 
                         equity_curve: pd.Series, 
                         benchmark_equity: Optional[pd.Series] = None,
                         metrics: Optional[Dict[str, float]] = None,
                         title: str = "Equity Curve",
                         log_scale: bool = False,
                         interactive: bool = False) -> Any:
        """Plot the equity curve.
        
        Args:
            equity_curve: Series of portfolio equity values over time
            benchmark_equity: Series of benchmark equity values over time
            metrics: Dictionary of performance metrics to display
            title: Chart title
            log_scale: Whether to use logarithmic scale
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive plot with plotly
            fig = go.Figure()
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Strategy',
                line=dict(color=self.color_scheme["primary"], width=2)
            ))
            
            # Add benchmark if available
            if benchmark_equity is not None:
                fig.add_trace(go.Scatter(
                    x=benchmark_equity.index,
                    y=benchmark_equity.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_scheme["benchmark"], width=2, dash='dash')
                ))
            
            # Add metrics as annotations if available
            if metrics is not None:
                annotations = []
                y_pos = 1.05
                for i, (key, value) in enumerate(metrics.items()):
                    if key in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown"]:
                        if key in ["total_return", "annualized_return", "max_drawdown"]:
                            value_str = f"{value:.2%}"
                        else:
                            value_str = f"{value:.2f}"
                        
                        annotations.append(dict(
                            x=0.02 + (i * 0.24),
                            y=y_pos,
                            xref="paper",
                            yref="paper",
                            text=f"<b>{key.replace('_', ' ').title()}:</b> {value_str}",
                            showarrow=False,
                            font=dict(size=12)
                        ))
                
                fig.update_layout(annotations=annotations)
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white"
            )
            
            if log_scale:
                fig.update_layout(yaxis_type="log")
            
            return fig
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot equity curve
            equity_curve.plot(ax=ax, color=self.color_scheme["primary"], linewidth=2, label='Strategy')
            
            # Plot benchmark if available
            if benchmark_equity is not None:
                benchmark_equity.plot(ax=ax, color=self.color_scheme["benchmark"], 
                                     linewidth=2, linestyle='--', label='Benchmark')
            
            # Add metrics as text if available
            if metrics is not None:
                text_str = ""
                for key, value in metrics.items():
                    if key in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown"]:
                        if key in ["total_return", "annualized_return", "max_drawdown"]:
                            value_str = f"{value:.2%}"
                        else:
                            value_str = f"{value:.2f}"
                        
                        text_str += f"{key.replace('_', ' ').title()}: {value_str}\n"
                
                # Add text box with metrics
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Equity ($)", fontsize=self.font_sizes["axis"])
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Set log scale if requested
            if log_scale:
                ax.set_yscale('log')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_drawdown(self, 
                     equity_curve: pd.Series,
                     title: str = "Drawdown",
                     interactive: bool = False) -> Any:
        """Plot the drawdown chart.
        
        Args:
            equity_curve: Series of portfolio equity values over time
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Calculate drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive plot with plotly
            fig = go.Figure()
            
            # Add drawdown curve
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,  # Convert to percentage
                mode='lines',
                name='Drawdown',
                line=dict(color=self.color_scheme["negative"], width=2),
                fill='tozeroy'
            ))
            
            # Add max drawdown marker
            fig.add_trace(go.Scatter(
                x=[max_drawdown_date],
                y=[max_drawdown * 100],
                mode='markers+text',
                name='Max Drawdown',
                marker=dict(color=self.color_scheme["highlight"], size=10),
                text=[f"{max_drawdown:.2%}"],
                textposition="bottom center"
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white",
                yaxis=dict(tickformat=".1%", autorange="reversed")
            )
            
            return fig
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot drawdown
            ax.fill_between(drawdown.index, 0, drawdown.values * 100, color=self.color_scheme["negative"], alpha=0.5)
            ax.plot(drawdown.index, drawdown.values * 100, color=self.color_scheme["negative"], linewidth=1.5)
            
            # Mark max drawdown
            ax.scatter(max_drawdown_date, max_drawdown * 100, color=self.color_scheme["highlight"], 
                      s=50, zorder=5)
            ax.annotate(f"{max_drawdown:.2%}", 
                       (max_drawdown_date, max_drawdown * 100),
                       xytext=(0, -20),
                       textcoords="offset points",
                       ha='center',
                       fontsize=self.font_sizes["annotation"],
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Drawdown (%)", fontsize=self.font_sizes["axis"])
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Invert y-axis
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_returns_distribution(self,
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                title: str = "Returns Distribution",
                                bins: int = 50,
                                interactive: bool = False) -> Any:
        """Plot the returns distribution.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            title: Chart title
            bins: Number of bins for histogram
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Calculate statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive plot with plotly
            fig = go.Figure()
            
            # Add returns histogram
            fig.add_trace(go.Histogram(
                x=returns.values * 100,  # Convert to percentage
                name='Returns',
                marker=dict(color=self.color_scheme["primary"], opacity=0.7),
                histnorm='probability density',
                nbinsx=bins
            ))
            
            # Add normal distribution curve
            x = np.linspace(returns.min() * 1.5, returns.max() * 1.5, 1000) * 100
            y = (1 / (std_return * np.sqrt(2 * np.pi))) * np.exp(-(x/100 - mean_return)**2 / (2 * std_return**2))
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.color_scheme["secondary"], width=2, dash='dash')
            ))
            
            # Add benchmark returns histogram if available
            if benchmark_returns is not None:
                fig.add_trace(go.Histogram(
                    x=benchmark_returns.values * 100,  # Convert to percentage
                    name='Benchmark Returns',
                    marker=dict(color=self.color_scheme["benchmark"], opacity=0.5),
                    histnorm='probability density',
                    nbinsx=bins
                ))
            
            # Add statistics as annotations
            annotations = [
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Mean:</b> {mean_return:.2%}",
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    x=0.02,
                    y=0.93,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Std Dev:</b> {std_return:.2%}",
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    x=0.02,
                    y=0.88,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Skewness:</b> {skew:.2f}",
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    x=0.02,
                    y=0.83,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Kurtosis:</b> {kurtosis:.2f}",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
            
            fig.update_layout(annotations=annotations)
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Returns (%)",
                yaxis_title="Probability Density",
                legend=dict(x=0.99, y=0.99, xanchor='right', bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="closest",
                template="plotly_white",
                barmode='overlay'
            )
            
            return fig
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot returns histogram
            returns_hist = ax.hist(returns * 100, bins=bins, alpha=0.7, density=True, 
                                 color=self.color_scheme["primary"], label='Returns')
            
            # Plot benchmark returns histogram if available
            if benchmark_returns is not None:
                ax.hist(benchmark_returns * 100, bins=bins, alpha=0.5, density=True,
                       color=self.color_scheme["benchmark"], label='Benchmark Returns')
            
            # Plot normal distribution curve
            x = np.linspace(returns.min() * 1.5, returns.max() * 1.5, 1000) * 100
            y = (1 / (std_return * np.sqrt(2 * np.pi))) * np.exp(-(x/100 - mean_return)**2 / (2 * std_return**2))
            ax.plot(x, y, color=self.color_scheme["secondary"], linestyle='--', 
                   linewidth=2, label='Normal Distribution')
            
            # Add statistics as text
            stats_text = f"Mean: {mean_return:.2%}\nStd Dev: {std_return:.2%}\nSkewness: {skew:.2f}\nKurtosis: {kurtosis:.2f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Returns (%)", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Probability Density", fontsize=self.font_sizes["axis"])
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_monthly_returns_heatmap(self,
                                   returns: pd.Series,
                                   title: str = "Monthly Returns Heatmap",
                                   interactive: bool = False) -> Any:
        """Plot the monthly returns heatmap.
        
        Args:
            returns: Series of portfolio returns
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Resample returns to monthly and convert to percentage
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        returns_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Returns': monthly_returns.values
        })
        
        pivot_table = returns_table.pivot_table(
            index='Year', columns='Month', values='Returns')
        
        # Add row with monthly average
        monthly_avg = pivot_table.mean(axis=0)
        pivot_table.loc['Avg'] = monthly_avg
        
        # Add column with yearly average
        yearly_avg = pivot_table.mean(axis=1)
        pivot_table['Avg'] = yearly_avg
        
        # Replace month numbers with month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Avg']
        pivot_table.columns = [month_names[i-1] if i <= 12 else 'Avg' for i in pivot_table.columns]
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive heatmap with plotly
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale=[
                    [0, self.color_scheme["negative"]],
                    [0.5, '#ffffff'],
                    [1, self.color_scheme["positive"]]
                ],
                zmid=0,
                text=[[f"{val:.2f}%" if not np.isnan(val) else "" for val in row] for row in pivot_table.values],
                texttemplate="%{text}",
                textfont={"size":10},
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Month",
                yaxis_title="Year",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static heatmap with matplotlib and seaborn
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create heatmap
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            heatmap = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, center=0,
                               linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Month", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Year", fontsize=self.font_sizes["axis"])
            
            # Rotate x-axis labels
            plt.xticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class AllocationVisualizer(BaseVisualizer):
    """Visualizer for allocation-related charts and graphs."""
    
    def __init__(self, output_dir: str = "./visualizations/allocation"):
        """Initialize the allocation visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_asset_allocation(self,
                            allocations: Dict[str, float],
                            title: str = "Asset Allocation",
                            chart_type: str = "pie",
                            interactive: bool = False) -> Any:
        """Plot the asset allocation.
        
        Args:
            allocations: Dictionary mapping asset names to allocation percentages
            title: Chart title
            chart_type: Type of chart ('pie', 'bar', or 'treemap')
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Convert allocations to DataFrame
        df = pd.DataFrame(list(allocations.items()), columns=['Asset', 'Allocation'])
        df = df.sort_values('Allocation', ascending=False)
        
        # Calculate percentage
        total = df['Allocation'].sum()
        df['Percentage'] = df['Allocation'] / total * 100
        
        if interactive and PLOTLY_AVAILABLE:
            if chart_type == 'pie':
                # Create interactive pie chart with plotly
                fig = go.Figure(data=go.Pie(
                    labels=df['Asset'],
                    values=df['Allocation'],
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    template="plotly_white"
                )
                
                return fig
            
            elif chart_type == 'bar':
                # Create interactive bar chart with plotly
                fig = go.Figure(data=go.Bar(
                    x=df['Asset'],
                    y=df['Percentage'],
                    text=df['Percentage'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto',
                    marker_color=px.colors.qualitative.Plotly
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Asset",
                    yaxis_title="Allocation (%)",
                    template="plotly_white",
                    yaxis=dict(range=[0, max(df['Percentage']) * 1.1])
                )
                
                return fig
            
            elif chart_type == 'treemap':
                # Create interactive treemap with plotly
                fig = px.treemap(
                    df,
                    path=['Asset'],
                    values='Allocation',
                    color='Percentage',
                    color_continuous_scale='RdBu',
                    title=title
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_white"
                )
                
                return fig
        
        else:
            if chart_type == 'pie':
                # Create static pie chart with matplotlib
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    df['Allocation'],
                    labels=df['Asset'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.5),  # For donut chart
                    textprops=dict(color="k", fontsize=self.font_sizes["annotation"])
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                
                # Set title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                
                # Add legend
                ax.legend(wedges, df['Asset'], title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
            
            elif chart_type == 'bar':
                # Create static bar chart with matplotlib
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Create bar chart
                bars = ax.bar(df['Asset'], df['Percentage'], color=sns.color_palette("husl", len(df)))
                
                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.1f}%",
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=self.font_sizes["annotation"])
                
                # Set labels and title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                ax.set_xlabel("Asset", fontsize=self.font_sizes["axis"])
                ax.set_ylabel("Allocation (%)", fontsize=self.font_sizes["axis"])
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
            
            elif chart_type == 'treemap':
                # Create static treemap with matplotlib (using squarify)
                try:
                    import squarify
                except ImportError:
                    raise ImportError("Squarify is not available. Install it with 'pip install squarify'.")
                
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Create treemap
                cmap = plt.cm.viridis
                colors = [cmap(i/len(df)) for i in range(len(df))]
                squarify.plot(sizes=df['Allocation'], label=df['Asset'], alpha=0.8, color=colors, ax=ax)
                
                # Set title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                
                # Remove axes
                ax.axis('off')
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
    
    def plot_allocation_drift(self,
                            allocations_over_time: pd.DataFrame,
                            target_allocations: Optional[Dict[str, float]] = None,
                            title: str = "Allocation Drift Over Time",
                            interactive: bool = False) -> Any:
        """Plot the allocation drift over time.
        
        Args:
            allocations_over_time: DataFrame with dates as index and assets as columns
            target_allocations: Dictionary mapping asset names to target allocation percentages
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive area chart with plotly
            fig = go.Figure()
            
            # Add area traces for each asset
            for column in allocations_over_time.columns:
                fig.add_trace(go.Scatter(
                    x=allocations_over_time.index,
                    y=allocations_over_time[column] * 100,  # Convert to percentage
                    mode='lines',
                    name=column,
                    stackgroup='one'
                ))
            
            # Add target allocation lines if available
            if target_allocations is not None:
                # Calculate cumulative target allocations for stacked view
                cumulative = 0
                for i, (asset, target) in enumerate(target_allocations.items()):
                    if asset in allocations_over_time.columns:
                        cumulative += target * 100
                        fig.add_trace(go.Scatter(
                            x=[allocations_over_time.index[0], allocations_over_time.index[-1]],
                            y=[cumulative, cumulative],
                            mode='lines',
                            name=f"{asset} Target",
                            line=dict(color='black', width=1, dash='dash')
                        ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Allocation (%)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white",
                yaxis=dict(range=[0, 100])
            )
            
            return fig
        
        else:
            # Create static area chart with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create stacked area chart
            allocations_over_time_pct = allocations_over_time * 100  # Convert to percentage
            ax.stackplot(allocations_over_time.index, 
                        [allocations_over_time_pct[col] for col in allocations_over_time.columns],
                        labels=allocations_over_time.columns,
                        alpha=0.8)
            
            # Add target allocation lines if available
            if target_allocations is not None:
                # Calculate cumulative target allocations for stacked view
                cumulative = 0
                for asset, target in target_allocations.items():
                    if asset in allocations_over_time.columns:
                        cumulative += target * 100
                        ax.axhline(y=cumulative, color='black', linestyle='--', linewidth=1, 
                                  alpha=0.7, label=f"{asset} Target" if asset == list(target_allocations.keys())[0] else "")
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Allocation (%)", fontsize=self.font_sizes["axis"])
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class RiskVisualizer(BaseVisualizer):
    """Visualizer for risk-related charts and graphs."""
    
    def __init__(self, output_dir: str = "./visualizations/risk"):
        """Initialize the risk visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_correlation_matrix(self,
                              returns_data: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              interactive: bool = False) -> Any:
        """Plot the correlation matrix.
        
        Args:
            returns_data: DataFrame with asset returns as columns
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive heatmap with plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[
                    [0, self.color_scheme["negative"]],
                    [0.5, '#ffffff'],
                    [1, self.color_scheme["positive"]]
                ],
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                textfont={"size":10},
                hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>"
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static heatmap with matplotlib and seaborn
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create heatmap
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Optional: mask upper triangle
            heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
                               mask=mask, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            # Set title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_risk_contribution(self,
                             risk_contributions: Dict[str, float],
                             title: str = "Risk Contribution",
                             chart_type: str = "pie",
                             interactive: bool = False) -> Any:
        """Plot the risk contribution.
        
        Args:
            risk_contributions: Dictionary mapping asset names to risk contribution percentages
            title: Chart title
            chart_type: Type of chart ('pie', 'bar', or 'waterfall')
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Convert risk contributions to DataFrame
        df = pd.DataFrame(list(risk_contributions.items()), columns=['Asset', 'Contribution'])
        df = df.sort_values('Contribution', ascending=False)
        
        # Calculate percentage
        total = df['Contribution'].sum()
        df['Percentage'] = df['Contribution'] / total * 100
        
        if interactive and PLOTLY_AVAILABLE:
            if chart_type == 'pie':
                # Create interactive pie chart with plotly
                fig = go.Figure(data=go.Pie(
                    labels=df['Asset'],
                    values=df['Contribution'],
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    template="plotly_white"
                )
                
                return fig
            
            elif chart_type == 'bar':
                # Create interactive bar chart with plotly
                fig = go.Figure(data=go.Bar(
                    x=df['Asset'],
                    y=df['Percentage'],
                    text=df['Percentage'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto',
                    marker_color=px.colors.qualitative.Plotly
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Asset",
                    yaxis_title="Risk Contribution (%)",
                    template="plotly_white",
                    yaxis=dict(range=[0, max(df['Percentage']) * 1.1])
                )
                
                return fig
            
            elif chart_type == 'waterfall':
                # Create interactive waterfall chart with plotly
                fig = go.Figure(go.Waterfall(
                    name="Risk Contribution",
                    orientation="v",
                    measure=["relative"] * len(df) + ["total"],
                    x=list(df['Asset']) + ["Total"],
                    textposition="outside",
                    text=list(df['Percentage'].apply(lambda x: f"{x:.1f}%")) + ["100.0%"],
                    y=list(df['Contribution']) + [total],
                    connector={"line":{"color":"rgb(63, 63, 63)"}},
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Asset",
                    yaxis_title="Risk Contribution",
                    template="plotly_white",
                    showlegend=False
                )
                
                return fig
        
        else:
            if chart_type == 'pie':
                # Create static pie chart with matplotlib
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    df['Contribution'],
                    labels=df['Asset'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.5),  # For donut chart
                    textprops=dict(color="k", fontsize=self.font_sizes["annotation"])
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                
                # Set title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                
                # Add legend
                ax.legend(wedges, df['Asset'], title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
            
            elif chart_type == 'bar':
                # Create static bar chart with matplotlib
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Create bar chart
                bars = ax.bar(df['Asset'], df['Percentage'], color=sns.color_palette("husl", len(df)))
                
                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.1f}%",
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=self.font_sizes["annotation"])
                
                # Set labels and title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                ax.set_xlabel("Asset", fontsize=self.font_sizes["axis"])
                ax.set_ylabel("Risk Contribution (%)", fontsize=self.font_sizes["axis"])
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
            
            elif chart_type == 'waterfall':
                # Create static waterfall chart with matplotlib
                fig, ax = plt.subplots(figsize=self.figure_size)
                
                # Prepare data for waterfall chart
                indices = range(len(df) + 1)
                values = list(df['Contribution']) + [0]  # Add a zero for the total bar
                
                # Create the cumulative sum
                cumulative = np.zeros(len(indices))
                cumulative[1:] = np.cumsum(values[:-1])
                
                # Create the waterfall chart
                for i, (x, y, value) in enumerate(zip(indices, cumulative, values)):
                    if i < len(indices) - 1:  # Regular bars
                        color = self.color_scheme["positive"] if value >= 0 else self.color_scheme["negative"]
                        ax.bar(x, value, bottom=y, color=color, alpha=0.7, width=0.6)
                    else:  # Total bar
                        ax.bar(x, total, color=self.color_scheme["neutral"], alpha=0.7, width=0.6)
                    
                    # Add labels
                    if i < len(indices) - 1:
                        label = f"{df['Percentage'].iloc[i]:.1f}%"
                    else:
                        label = "100.0%"
                    
                    ax.annotate(label,
                               xy=(x, y + value/2 if i < len(indices) - 1 else total/2),
                               xytext=(0, 0),
                               textcoords="offset points",
                               ha='center', va='center',
                               fontsize=self.font_sizes["annotation"],
                               color='white')
                
                # Add connecting lines
                for i in range(1, len(indices)):
                    ax.plot([i-1, i], [cumulative[i], cumulative[i]], 'k-', alpha=0.3)
                
                # Set x-axis labels
                labels = list(df['Asset']) + ["Total"]
                ax.set_xticks(indices)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Set labels and title
                ax.set_title(title, fontsize=self.font_sizes["title"])
                ax.set_ylabel("Risk Contribution", fontsize=self.font_sizes["axis"])
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Adjust layout
                plt.tight_layout()
                
                return fig


class TaxVisualizer(BaseVisualizer):
    """Visualizer for tax-related charts and graphs."""
    
    def __init__(self, output_dir: str = "./visualizations/tax"):
        """Initialize the tax visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_realized_gains(self,
                          realized_gains: pd.DataFrame,
                          title: str = "Realized Gains/Losses",
                          interactive: bool = False) -> Any:
        """Plot the realized gains/losses.
        
        Args:
            realized_gains: DataFrame with dates as index and columns for short-term and long-term gains
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Ensure the DataFrame has the required columns
        required_columns = ['short_term_gains', 'long_term_gains']
        for col in required_columns:
            if col not in realized_gains.columns:
                realized_gains[col] = 0
        
        # Calculate total gains
        realized_gains['total_gains'] = realized_gains['short_term_gains'] + realized_gains['long_term_gains']
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive stacked bar chart with plotly
            fig = go.Figure()
            
            # Add short-term gains/losses
            fig.add_trace(go.Bar(
                x=realized_gains.index,
                y=realized_gains['short_term_gains'],
                name='Short-term Gains/Losses',
                marker_color=self.color_scheme["primary"]
            ))
            
            # Add long-term gains/losses
            fig.add_trace(go.Bar(
                x=realized_gains.index,
                y=realized_gains['long_term_gains'],
                name='Long-term Gains/Losses',
                marker_color=self.color_scheme["secondary"]
            ))
            
            # Add total gains/losses line
            fig.add_trace(go.Scatter(
                x=realized_gains.index,
                y=realized_gains['total_gains'],
                mode='lines+markers',
                name='Total Gains/Losses',
                line=dict(color=self.color_scheme["highlight"], width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white",
                barmode='stack'
            )
            
            return fig
        
        else:
            # Create static stacked bar chart with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create stacked bar chart
            width = 0.8
            ax.bar(realized_gains.index, realized_gains['short_term_gains'], width, 
                  label='Short-term Gains/Losses', color=self.color_scheme["primary"], alpha=0.7)
            ax.bar(realized_gains.index, realized_gains['long_term_gains'], width, 
                  bottom=realized_gains['short_term_gains'], label='Long-term Gains/Losses', 
                  color=self.color_scheme["secondary"], alpha=0.7)
            
            # Add total gains/losses line
            ax2 = ax.twinx()
            ax2.plot(realized_gains.index, realized_gains['total_gains'], 'o-', 
                    color=self.color_scheme["highlight"], linewidth=2, label='Total Gains/Losses')
            ax2.set_ylabel('Total Gains/Losses ($)', fontsize=self.font_sizes["axis"])
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Amount ($)", fontsize=self.font_sizes["axis"])
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_tax_impact(self,
                      tax_impact: pd.DataFrame,
                      title: str = "Tax Impact Analysis",
                      interactive: bool = False) -> Any:
        """Plot the tax impact analysis.
        
        Args:
            tax_impact: DataFrame with tax impact data including columns for 'pre_tax_return', 'post_tax_return', and 'tax_drag'
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Ensure the DataFrame has the required columns
        required_columns = ['pre_tax_return', 'post_tax_return', 'tax_drag']
        for col in required_columns:
            if col not in tax_impact.columns:
                raise ValueError(f"Tax impact DataFrame must contain '{col}' column")
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive bar chart with plotly
            fig = go.Figure()
            
            # Add pre-tax returns
            fig.add_trace(go.Bar(
                x=tax_impact.index,
                y=tax_impact['pre_tax_return'] * 100,  # Convert to percentage
                name='Pre-Tax Return',
                marker_color=self.color_scheme["primary"]
            ))
            
            # Add post-tax returns
            fig.add_trace(go.Bar(
                x=tax_impact.index,
                y=tax_impact['post_tax_return'] * 100,  # Convert to percentage
                name='Post-Tax Return',
                marker_color=self.color_scheme["secondary"]
            ))
            
            # Add tax drag line
            fig.add_trace(go.Scatter(
                x=tax_impact.index,
                y=tax_impact['tax_drag'] * 100,  # Convert to percentage
                mode='lines+markers',
                name='Tax Drag',
                line=dict(color=self.color_scheme["highlight"], width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Period",
                yaxis_title="Return (%)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static bar chart with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Set width of bars
            width = 0.35
            x = np.arange(len(tax_impact.index))
            
            # Create bars
            ax.bar(x - width/2, tax_impact['pre_tax_return'] * 100, width, 
                  label='Pre-Tax Return', color=self.color_scheme["primary"], alpha=0.7)
            ax.bar(x + width/2, tax_impact['post_tax_return'] * 100, width, 
                  label='Post-Tax Return', color=self.color_scheme["secondary"], alpha=0.7)
            
            # Add tax drag line on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(x, tax_impact['tax_drag'] * 100, 'o-', 
                    color=self.color_scheme["highlight"], linewidth=2, label='Tax Drag')
            ax2.set_ylabel('Tax Drag (%)', fontsize=self.font_sizes["axis"])
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Period", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Return (%)", fontsize=self.font_sizes["axis"])
            
            # Set x-tick labels
            ax.set_xticks(x)
            ax.set_xticklabels(tax_impact.index, rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_tax_efficiency_metrics(self,
                                  tax_metrics: Dict[str, float],
                                  title: str = "Tax Efficiency Metrics",
                                  interactive: bool = False) -> Any:
        """Plot tax efficiency metrics as a radar chart or bar chart.
        
        Args:
            tax_metrics: Dictionary of tax efficiency metrics
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
    
    def plot_tax_optimization_metrics(self,
                                    optimization_history: List[Dict[str, Any]],
                                    title: str = "Tax Optimization Metrics Over Time",
                                    interactive: bool = False) -> Any:
        """Plot tax optimization metrics over time to track optimization progress.
        
        Args:
            optimization_history: List of tax optimization history entries from TaxOptimizer
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Filter history for tax efficiency metrics entries
        metrics_history = [entry['data'] for entry in optimization_history 
                          if entry['type'] == 'tax_efficiency_metrics']
        
        if not metrics_history:
            raise ValueError("No tax efficiency metrics found in optimization history")
        
        # Extract timestamps and metrics
        timestamps = [entry['timestamp'] for entry in metrics_history]
        tax_efficiency_ratios = [entry['tax_efficiency']['tax_efficiency_ratio'] for entry in metrics_history]
        long_term_gain_ratios = [entry['tax_efficiency']['long_term_gain_ratio'] for entry in metrics_history]
        loss_harvesting_efficiencies = [entry['tax_efficiency']['loss_harvesting_efficiency'] for entry in metrics_history]
        tax_benefits = [entry['tax_efficiency']['tax_benefit_from_harvesting'] for entry in metrics_history]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Tax Efficiency Ratio': tax_efficiency_ratios,
            'Long-Term Gain Ratio': long_term_gain_ratios,
            'Loss Harvesting Efficiency': loss_harvesting_efficiencies,
            'Tax Benefit from Harvesting': tax_benefits
        }, index=timestamps)
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive line chart with plotly
            fig = go.Figure()
            
            # Add metrics lines
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Tax Efficiency Ratio'],
                mode='lines+markers',
                name='Tax Efficiency Ratio',
                line=dict(color=self.color_scheme["primary"], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Long-Term Gain Ratio'],
                mode='lines+markers',
                name='Long-Term Gain Ratio',
                line=dict(color=self.color_scheme["secondary"], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Loss Harvesting Efficiency'],
                mode='lines+markers',
                name='Loss Harvesting Efficiency',
                line=dict(color=self.color_scheme["tertiary"], width=2)
            ))
            
            # Add tax benefit as bar chart on secondary y-axis
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Tax Benefit from Harvesting'],
                name='Tax Benefit from Harvesting',
                marker_color=self.color_scheme["highlight"],
                opacity=0.7,
                yaxis="y2"
            ))
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Ratio",
                yaxis2=dict(
                    title="Tax Benefit ($)",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static line chart with matplotlib
            fig, ax1 = plt.subplots(figsize=self.figure_size)
            
            # Plot ratios on primary y-axis
            ax1.plot(df.index, df['Tax Efficiency Ratio'], 'o-', 
                    color=self.color_scheme["primary"], linewidth=2, label='Tax Efficiency Ratio')
            ax1.plot(df.index, df['Long-Term Gain Ratio'], 's-', 
                    color=self.color_scheme["secondary"], linewidth=2, label='Long-Term Gain Ratio')
            ax1.plot(df.index, df['Loss Harvesting Efficiency'], '^-', 
                    color=self.color_scheme["tertiary"], linewidth=2, label='Loss Harvesting Efficiency')
            
            # Set labels for primary y-axis
            ax1.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax1.set_ylabel("Ratio", fontsize=self.font_sizes["axis"])
            
            # Create secondary y-axis for tax benefit
            ax2 = ax1.twinx()
            ax2.bar(df.index, df['Tax Benefit from Harvesting'], alpha=0.3, 
                   color=self.color_scheme["highlight"], label='Tax Benefit from Harvesting')
            ax2.set_ylabel("Tax Benefit ($)", fontsize=self.font_sizes["axis"])
            
            # Set title
            ax1.set_title(title, fontsize=self.font_sizes["title"])
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        # Extract metrics and values
        metrics = list(tax_metrics.keys())
        values = list(tax_metrics.values())
        
        # Normalize values for radar chart (between 0 and 1)
        normalized_values = []
        for value in values:
            # Handle different types of metrics
            if isinstance(value, (int, float)):
                # For percentage metrics, divide by 100
                if 'ratio' in metrics[values.index(value)] or 'efficiency' in metrics[values.index(value)]:
                    normalized_values.append(min(max(value, 0), 1))  # Clamp between 0 and 1
                else:
                    # For other metrics, normalize to 0-1 range (assuming positive values)
                    normalized_values.append(min(max(value / 100, 0), 1) if value > 1 else value)
            else:
                # Skip non-numeric values
                normalized_values.append(0)
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive radar chart with plotly
            fig = go.Figure()
            
            # Add radar chart
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metrics,
                fill='toself',
                name='Tax Efficiency',
                line_color=self.color_scheme["primary"]
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
                showlegend=False
            )
            
            return fig
        
        else:
            # Create static radar chart with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(polar=True))
            
            # Number of metrics
            N = len(metrics)
            
            # Compute angle for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values and close the loop
            values_plot = normalized_values + [normalized_values[0]]
            
            # Add metrics labels and close the loop
            metrics_plot = metrics + [metrics[0]]
            
            # Draw the chart
            ax.plot(angles, values_plot, 'o-', linewidth=2, color=self.color_scheme["primary"])
            ax.fill(angles, values_plot, alpha=0.25, color=self.color_scheme["primary"])
            
            # Set labels
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
            
            # Set title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            
            # Add original values as annotations
            for i, (angle, value, metric) in enumerate(zip(angles[:-1], values[:-1], metrics[:-1])):
                # Format value based on metric type
                if 'ratio' in metric or 'efficiency' in metric:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.1f}%" if value > 1 else f"{value:.2f}"
                
                # Calculate annotation position
                x = 1.25 * np.cos(angle)
                y = 1.25 * np.sin(angle)
                
                # Add annotation
                ax.annotate(
                    formatted_value,
                    xy=(angle, normalized_values[i]),
                    xytext=(x, y),
                    textcoords='data',
                    fontsize=self.font_sizes["annotation"],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


class InteractiveVisualizer(BaseVisualizer):
    """Visualizer for interactive visualizations and dashboards."""
    
    def __init__(self, output_dir: str = "./visualizations/interactive"):
        """Initialize the interactive visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
        
        # Check if required libraries are available
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations. Install it with 'pip install plotly'.")
    
    def create_dashboard(self,
                       performance_data: Dict[str, Any],
                       allocation_data: Dict[str, Any],
                       risk_data: Dict[str, Any],
                       title: str = "Portfolio Dashboard") -> Any:
        """Create an interactive dashboard with multiple visualizations.
        
        Args:
            performance_data: Dictionary with performance-related data
            allocation_data: Dictionary with allocation-related data
            risk_data: Dictionary with risk-related data
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Equity Curve", "Asset Allocation",
                "Drawdown", "Risk Contribution",
                "Monthly Returns", "Correlation Matrix"
            ),
            specs=[
                [{"type": "xy"}, {"type": "domain"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Equity Curve (top left)
        equity_curve = performance_data.get('equity_curve')
        benchmark_equity = performance_data.get('benchmark_equity')
        
        if equity_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode='lines',
                    name='Strategy',
                    line=dict(color=self.color_scheme["primary"], width=2)
                ),
                row=1, col=1
            )
            
            if benchmark_equity is not None:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_equity.index,
                        y=benchmark_equity.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.color_scheme["benchmark"], width=2, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # 2. Asset Allocation (top right)
        allocations = allocation_data.get('allocations')
        
        if allocations is not None:
            fig.add_trace(
                go.Pie(
                    labels=list(allocations.keys()),
                    values=list(allocations.values()),
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3
                ),
                row=1, col=2
            )
        
        # 3. Drawdown (middle left)
        if equity_curve is not None:
            # Calculate drawdown
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100  # Convert to percentage
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.color_scheme["negative"], width=2),
                    fill='tozeroy'
                ),
                row=2, col=1
            )
        
        # 4. Risk Contribution (middle right)
        risk_contributions = risk_data.get('risk_contributions')
        
        if risk_contributions is not None:
            # Convert to DataFrame for easier manipulation
            risk_df = pd.DataFrame(list(risk_contributions.items()), columns=['Asset', 'Contribution'])
            risk_df = risk_df.sort_values('Contribution', ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=risk_df['Asset'],
                    y=risk_df['Contribution'],
                    name='Risk Contribution',
                    marker_color=px.colors.qualitative.Plotly
                ),
                row=2, col=2
            )
        
        # 5. Monthly Returns Heatmap (bottom left)
        returns = performance_data.get('returns')
        
        if returns is not None:
            # Resample returns to monthly and convert to percentage
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            
            # Create a pivot table with years as rows and months as columns
            monthly_returns.index = monthly_returns.index.to_period('M')
            returns_table = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Returns': monthly_returns.values
            })
            
            pivot_table = returns_table.pivot_table(
                index='Year', columns='Month', values='Returns')
            
            # Replace month numbers with month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale=[
                        [0, self.color_scheme["negative"]],
                        [0.5, '#ffffff'],
                        [1, self.color_scheme["positive"]]
                    ],
                    zmid=0,
                    text=[[f"{val:.2f}%" if not np.isnan(val) else "" for val in row] for row in pivot_table.values],
                    texttemplate="%{text}",
                    textfont={"size":8}
                ),
                row=3, col=1
            )
        
        # 6. Correlation Matrix (bottom right)
        returns_data = risk_data.get('returns_data')
        
        if returns_data is not None:
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=[
                        [0, self.color_scheme["negative"]],
                        [0.5, '#ffffff'],
                        [1, self.color_scheme["positive"]]
                    ],
                    zmid=0,
                    text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    textfont={"size":8}
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=900,
            width=1200,
            template="plotly_white",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Asset", row=2, col=2)
        fig.update_yaxes(title_text="Risk Contribution", row=2, col=2)
        
        fig.update_xaxes(title_text="Month", row=3, col=1)
        fig.update_yaxes(title_text="Year", row=3, col=1)
        
        # Invert y-axis for drawdown
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        
        return fig


class ComparativeVisualizer(BaseVisualizer):
    """Visualizer for comparative analysis and scenario testing."""
    
    def __init__(self, output_dir: str = "./visualizations/comparative"):
        """Initialize the comparative visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        super().__init__(output_dir)
    
    def plot_strategy_comparison(self,
                               equity_curves: Dict[str, pd.Series],
                               title: str = "Strategy Comparison",
                               metrics: Optional[Dict[str, Dict[str, float]]] = None,
                               log_scale: bool = False,
                               interactive: bool = False) -> Any:
        """Plot a comparison of multiple strategies.
        
        Args:
            equity_curves: Dictionary mapping strategy names to equity curves
            title: Chart title
            metrics: Dictionary mapping strategy names to performance metrics
            log_scale: Whether to use logarithmic scale
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive plot with plotly
            fig = go.Figure()
            
            # Add equity curves for each strategy
            for strategy_name, equity_curve in equity_curves.items():
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode='lines',
                    name=strategy_name
                ))
            
            # Add metrics as a table if available
            if metrics is not None and len(metrics) > 0:
                # Create a metrics table
                metric_names = list(next(iter(metrics.values())).keys())
                strategy_names = list(metrics.keys())
                
                # Create header row
                header = ['Metric'] + strategy_names
                cells = []
                
                # Add metric names column
                cells.append(metric_names)
                
                # Add values for each strategy
                for strategy in strategy_names:
                    strategy_metrics = metrics[strategy]
                    cells.append([strategy_metrics.get(metric, "") for metric in metric_names])
                
                # Format the values
                formatted_cells = []
                formatted_cells.append(cells[0])  # Metric names
                
                for i in range(1, len(cells)):
                    formatted_values = []
                    for j in range(len(cells[i])):
                        value = cells[i][j]
                        if isinstance(value, float):
                            if metric_names[j] in ["total_return", "annualized_return", "max_drawdown"]:
                                formatted_values.append(f"{value:.2%}")
                            else:
                                formatted_values.append(f"{value:.2f}")
                        else:
                            formatted_values.append(str(value))
                    formatted_cells.append(formatted_values)
                
                # Add table as an annotation
                table_text = "<br>".join([
                    "&nbsp;&nbsp;".join([header[j] if j == 0 else f"<b>{header[j]}</b>" for j in range(len(header))]),
                    *["&nbsp;&nbsp;".join([f"<b>{formatted_cells[0][i]}</b>"] + [formatted_cells[j+1][i] for j in range(len(strategy_names))]) for i in range(len(metric_names))]
                ])
                
                fig.add_annotation(
                    x=0.5,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=table_text,
                    showarrow=False,
                    font=dict(size=12),
                    align="left"
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                hovermode="x unified",
                template="plotly_white",
                margin=dict(t=150 if metrics else 100)  # Add margin for the table
            )
            
            if log_scale:
                fig.update_layout(yaxis_type="log")
            
            return fig
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot equity curves for each strategy
            for strategy_name, equity_curve in equity_curves.items():
                equity_curve.plot(ax=ax, linewidth=2, label=strategy_name)
            
            # Add metrics as a table if available
            if metrics is not None and len(metrics) > 0:
                # Create a metrics table
                metric_names = list(next(iter(metrics.values())).keys())
                strategy_names = list(metrics.keys())
                
                # Format the metrics for display
                cell_text = []
                for metric in metric_names:
                    row = [metric.replace('_', ' ').title()]
                    for strategy in strategy_names:
                        value = metrics[strategy].get(metric, "")
                        if isinstance(value, float):
                            if metric in ["total_return", "annualized_return", "max_drawdown"]:
                                row.append(f"{value:.2%}")
                            else:
                                row.append(f"{value:.2f}")
                        else:
                            row.append(str(value))
                    cell_text.append(row)
                
                # Create the table
                table = ax.table(
                    cellText=cell_text,
                    colLabels=["Metric"] + strategy_names,
                    loc='top',
                    cellLoc='center',
                    colLoc='center'
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                # Adjust figure to make room for the table
                plt.subplots_adjust(top=0.85)
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Date", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Equity ($)", fontsize=self.font_sizes["axis"])
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            # Set log scale if requested
            if log_scale:
                ax.set_yscale('log')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
    
    def plot_scenario_analysis(self,
                             scenario_results: Dict[str, float],
                             baseline: Optional[float] = None,
                             title: str = "Scenario Analysis",
                             interactive: bool = False) -> Any:
        """Plot the results of scenario analysis.
        
        Args:
            scenario_results: Dictionary mapping scenario names to portfolio returns
            baseline: Baseline return for comparison
            title: Chart title
            interactive: Whether to create an interactive plot
            
        Returns:
            Figure object (matplotlib, plotly, or bokeh)
        """
        # Convert scenario results to DataFrame
        df = pd.DataFrame(list(scenario_results.items()), columns=['Scenario', 'Return'])
        df = df.sort_values('Return')
        
        # Convert returns to percentage
        df['Return'] = df['Return'] * 100
        
        # Add color based on positive/negative returns
        df['Color'] = df['Return'].apply(
            lambda x: self.color_scheme["positive"] if x >= 0 else self.color_scheme["negative"])
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive bar chart with plotly
            fig = go.Figure()
            
            # Add bars for each scenario
            fig.add_trace(go.Bar(
                x=df['Scenario'],
                y=df['Return'],
                marker_color=df['Color'],
                text=df['Return'].apply(lambda x: f"{x:.2f}%"),
                textposition='auto'
            ))
            
            # Add baseline if available
            if baseline is not None:
                baseline_pct = baseline * 100
                fig.add_trace(go.Scatter(
                    x=df['Scenario'],
                    y=[baseline_pct] * len(df),
                    mode='lines',
                    name='Baseline',
                    line=dict(color='black', width=2, dash='dash')
                ))
                
                # Add baseline label
                fig.add_annotation(
                    x=df['Scenario'].iloc[-1],
                    y=baseline_pct,
                    text=f"Baseline: {baseline_pct:.2f}%",
                    showarrow=False,
                    xshift=10,
                    yshift=10,
                    font=dict(size=10)
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Scenario",
                yaxis_title="Return (%)",
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Create static bar chart with matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create bar chart
            bars = ax.bar(df['Scenario'], df['Return'], color=df['Color'])
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}%",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=self.font_sizes["annotation"])
            
            # Add baseline if available
            if baseline is not None:
                baseline_pct = baseline * 100
                ax.axhline(y=baseline_pct, color='black', linestyle='--', linewidth=1)
                ax.annotate(f"Baseline: {baseline_pct:.2f}%",
                           xy=(df['Scenario'].iloc[-1], baseline_pct),
                           xytext=(5, 0),  # 5 points horizontal offset
                           textcoords="offset points",
                           va='center',
                           fontsize=self.font_sizes["annotation"])
            
            # Set labels and title
            ax.set_title(title, fontsize=self.font_sizes["title"])
            ax.set_xlabel("Scenario", fontsize=self.font_sizes["axis"])
            ax.set_ylabel("Return (%)", fontsize=self.font_sizes["axis"])
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig


# Export all visualizer classes
__all__ = [
    'BaseVisualizer',
    'PerformanceVisualizer',
    'AllocationVisualizer',
    'RiskVisualizer',
    'TaxVisualizer',
    'InteractiveVisualizer',
    'ComparativeVisualizer'
]