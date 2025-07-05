"""Utility functions for the analytics module.

This module provides helper functions and utilities for the analytics components,
including data processing, formatting, and common calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import bokeh
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot
    from bokeh.models import ColumnDataSource, HoverTool
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


# Data processing functions
def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate common performance metrics from returns series.
    
    Args:
        returns: Series of returns (daily, monthly, etc.)
        
    Returns:
        Dictionary of performance metrics
    """
    if returns.empty:
        return {}
    
    # Annualization factor based on return frequency
    freq = pd.infer_freq(returns.index)
    if freq in ['D', 'B']:
        ann_factor = 252
    elif freq in ['W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']:
        ann_factor = 52
    elif freq in ['M', 'MS', 'BM', 'BMS']:
        ann_factor = 12
    elif freq in ['Q', 'QS', 'BQ', 'BQS']:
        ann_factor = 4
    else:
        # Default to daily
        ann_factor = 252
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Calculate drawdowns
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    max_drawdown = drawdowns.min()
    
    # Calculate Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
    sortino = ann_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Calmar ratio
    calmar = -ann_return / max_drawdown if max_drawdown < 0 else 0
    
    # Calculate win rate
    win_rate = len(returns[returns > 0]) / len(returns)
    
    # Calculate average win/loss
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    
    # Calculate profit factor
    total_wins = returns[returns > 0].sum()
    total_losses = abs(returns[returns < 0].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> Dict[str, pd.Series]:
    """Calculate rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Dictionary of rolling metrics series
    """
    if returns.empty or len(returns) < window:
        return {}
    
    # Annualization factor based on return frequency
    freq = pd.infer_freq(returns.index)
    if freq in ['D', 'B']:
        ann_factor = 252
    elif freq in ['W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']:
        ann_factor = 52
    elif freq in ['M', 'MS', 'BM', 'BMS']:
        ann_factor = 12
    elif freq in ['Q', 'QS', 'BQ', 'BQS']:
        ann_factor = 4
    else:
        # Default to daily
        ann_factor = 252
    
    # Calculate rolling returns
    rolling_return = returns.rolling(window=window).apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(ann_factor)
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = rolling_return / rolling_vol
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling drawdown
    rolling_dd = pd.Series(index=returns.index)
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        cum_returns = (1 + window_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        rolling_dd.iloc[i-1] = drawdowns.min()
    
    # Calculate rolling Sortino ratio
    def downside_deviation(x):
        downside_returns = x[x < 0]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.std() * np.sqrt(ann_factor)
    
    rolling_downside_dev = returns.rolling(window=window).apply(downside_deviation)
    rolling_sortino = rolling_return / rolling_downside_dev
    rolling_sortino = rolling_sortino.replace([np.inf, -np.inf], np.nan)
    
    return {
        'rolling_return': rolling_return,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_dd,
        'rolling_sortino': rolling_sortino
    }


def calculate_drawdowns(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Calculate drawdown periods and statistics.
    
    Args:
        returns: Series of returns
        top_n: Number of largest drawdowns to return
        
    Returns:
        DataFrame with drawdown statistics
    """
    if returns.empty:
        return pd.DataFrame()
    
    # Calculate cumulative returns and drawdowns
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    
    # Find drawdown periods
    is_drawdown = drawdowns < 0
    
    # Initialize variables for tracking drawdown periods
    drawdown_periods = []
    current_drawdown_start = None
    
    # Iterate through the drawdowns to identify periods
    for date, in_drawdown in is_drawdown.items():
        if in_drawdown and current_drawdown_start is None:
            # Start of a new drawdown period
            current_drawdown_start = date
        elif not in_drawdown and current_drawdown_start is not None:
            # End of a drawdown period
            end_date = date
            drawdown_period = drawdowns[current_drawdown_start:end_date]
            max_drawdown = drawdown_period.min()
            max_drawdown_date = drawdown_period.idxmin()
            
            # Calculate recovery time
            recovery_time = (end_date - max_drawdown_date).days
            
            # Calculate drawdown duration
            duration = (end_date - current_drawdown_start).days
            
            drawdown_periods.append({
                'start_date': current_drawdown_start,
                'end_date': end_date,
                'max_drawdown_date': max_drawdown_date,
                'max_drawdown': max_drawdown,
                'duration_days': duration,
                'recovery_days': recovery_time
            })
            
            current_drawdown_start = None
    
    # Check if we're still in a drawdown at the end of the series
    if current_drawdown_start is not None:
        end_date = drawdowns.index[-1]
        drawdown_period = drawdowns[current_drawdown_start:end_date]
        max_drawdown = drawdown_period.min()
        max_drawdown_date = drawdown_period.idxmin()
        
        # Calculate drawdown duration (ongoing)
        duration = (end_date - current_drawdown_start).days
        
        drawdown_periods.append({
            'start_date': current_drawdown_start,
            'end_date': end_date,
            'max_drawdown_date': max_drawdown_date,
            'max_drawdown': max_drawdown,
            'duration_days': duration,
            'recovery_days': np.nan  # Recovery not complete
        })
    
    # Convert to DataFrame and sort by max_drawdown
    if drawdown_periods:
        df_drawdowns = pd.DataFrame(drawdown_periods)
        df_drawdowns = df_drawdowns.sort_values('max_drawdown').head(top_n)
        return df_drawdowns
    else:
        return pd.DataFrame()


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Calculate monthly returns table.
    
    Args:
        returns: Series of returns (preferably daily)
        
    Returns:
        DataFrame with monthly returns (rows=years, columns=months)
    """
    if returns.empty:
        return pd.DataFrame()
    
    # Resample to monthly if not already
    if pd.infer_freq(returns.index) not in ['M', 'MS', 'BM', 'BMS']:
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    else:
        monthly_returns = returns
    
    # Create a pivot table with years as rows and months as columns
    monthly_returns.index = pd.MultiIndex.from_arrays([
        monthly_returns.index.year,
        monthly_returns.index.month
    ])
    monthly_returns = monthly_returns.reset_index()
    monthly_returns.columns = ['Year', 'Month', 'Return']
    
    pivot_table = monthly_returns.pivot(index='Year', columns='Month', values='Return')
    
    # Rename columns to month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    pivot_table = pivot_table.rename(columns=month_names)
    
    # Add annual returns
    annual_returns = []
    for year in pivot_table.index:
        year_returns = pivot_table.loc[year].dropna()
        annual_return = (1 + year_returns).prod() - 1
        annual_returns.append(annual_return)
    
    pivot_table['Annual'] = annual_returns
    
    return pivot_table


def calculate_correlation_matrix(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate correlation matrix from multiple return series.
    
    Args:
        returns_dict: Dictionary of return series
        
    Returns:
        Correlation matrix DataFrame
    """
    if not returns_dict:
        return pd.DataFrame()
    
    # Combine returns into a single DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    return corr_matrix


def calculate_risk_contribution(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate risk contribution of each asset.
    
    Args:
        weights: Series of asset weights
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Series of risk contributions
    """
    if weights.empty or cov_matrix.empty:
        return pd.Series()
    
    # Ensure weights and covariance matrix have matching indices
    weights = weights.reindex(cov_matrix.index)
    
    # Calculate portfolio volatility
    port_vol = np.sqrt(weights.dot(cov_matrix).dot(weights))
    
    # Calculate marginal contribution to risk
    mcr = cov_matrix.dot(weights) / port_vol
    
    # Calculate risk contribution
    rc = weights * mcr
    
    # Normalize to sum to 1
    rc = rc / rc.sum()
    
    return rc


def calculate_factor_contribution(returns: pd.Series, factor_returns: pd.DataFrame, 
                                factor_exposures: pd.DataFrame) -> pd.DataFrame:
    """Calculate factor contribution to returns.
    
    Args:
        returns: Series of portfolio returns
        factor_returns: DataFrame of factor returns
        factor_exposures: DataFrame of factor exposures
        
    Returns:
        DataFrame of factor contributions
    """
    if returns.empty or factor_returns.empty or factor_exposures.empty:
        return pd.DataFrame()
    
    # Align dates
    common_dates = returns.index.intersection(factor_returns.index)
    returns = returns.loc[common_dates]
    factor_returns = factor_returns.loc[common_dates]
    
    # Calculate factor contribution for each period
    contributions = pd.DataFrame(index=common_dates, columns=factor_returns.columns)
    
    for date in common_dates:
        # Get factor exposures for the date or the most recent date before it
        exposure_dates = factor_exposures.index[factor_exposures.index <= date]
        if len(exposure_dates) > 0:
            latest_exposure_date = exposure_dates[-1]
            exposures = factor_exposures.loc[latest_exposure_date]
            
            # Calculate contribution
            contributions.loc[date] = exposures * factor_returns.loc[date]
    
    # Add specific return (unexplained by factors)
    contributions['Specific'] = returns - contributions.sum(axis=1)
    
    return contributions


# Formatting functions
def format_percentage(value: float, precision: int = 2) -> str:
    """Format a value as a percentage string.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if np.isnan(value):
        return 'N/A'
    
    return f"{value * 100:.{precision}f}%"


def format_currency(value: float, currency: str = '$', precision: int = 2) -> str:
    """Format a value as a currency string.
    
    Args:
        value: Value to format
        currency: Currency symbol
        precision: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if np.isnan(value):
        return 'N/A'
    
    if abs(value) >= 1e9:
        return f"{currency}{value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{currency}{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value / 1e3:.{precision}f}K"
    else:
        return f"{currency}{value:.{precision}f}"


def format_number(value: float, precision: int = 2) -> str:
    """Format a value as a number string with thousands separator.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if np.isnan(value):
        return 'N/A'
    
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


# Visualization helper functions
def get_color_palette(n_colors: int = 10, palette_name: str = 'viridis') -> List[str]:
    """Get a color palette for visualizations.
    
    Args:
        n_colors: Number of colors needed
        palette_name: Name of the seaborn color palette
        
    Returns:
        List of color hex codes
    """
    return sns.color_palette(palette_name, n_colors=n_colors).as_hex()


def create_custom_colormap(start_color: str = '#0571b0', end_color: str = '#ca0020', 
                          mid_color: str = '#f7f7f7') -> LinearSegmentedColormap:
    """Create a custom colormap for heatmaps.
    
    Args:
        start_color: Color for negative values
        end_color: Color for positive values
        mid_color: Color for zero
        
    Returns:
        Matplotlib colormap
    """
    return LinearSegmentedColormap.from_list(
        'custom_diverging',
        [start_color, mid_color, end_color],
        N=256
    )


def add_value_labels(ax: plt.Axes, spacing: int = 5, precision: int = 2, 
                    percentage: bool = True) -> None:
    """Add value labels to a bar chart.
    
    Args:
        ax: Matplotlib axes object
        spacing: Spacing between bar and label
        precision: Number of decimal places
        percentage: Whether to format as percentage
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        
        if percentage:
            label = format_percentage(y_value, precision)
        else:
            label = format_number(y_value, precision)
        
        va = 'bottom' if y_value >= 0 else 'top'
        
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords="offset points",
            ha='center',
            va=va,
            fontsize=8
        )


def format_axis_ticks(ax: plt.Axes, x_format: Optional[str] = None, 
                     y_format: Optional[str] = None) -> None:
    """Format axis ticks with custom formatters.
    
    Args:
        ax: Matplotlib axes object
        x_format: Format for x-axis ('percentage', 'currency', or None)
        y_format: Format for y-axis ('percentage', 'currency', or None)
    """
    if x_format == 'percentage':
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    elif x_format == 'currency':
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, pos: f"${x:,.2f}" if abs(x) < 1e6 else f"${x/1e6:,.1f}M"
        ))
    
    if y_format == 'percentage':
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    elif y_format == 'currency':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, pos: f"${y:,.2f}" if abs(y) < 1e6 else f"${y/1e6:,.1f}M"
        ))


# File and path utilities
def ensure_directory_exists(directory_path: str) -> None:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_default_output_path(report_name: str, format_extension: str) -> str:
    """Get default output path for reports.
    
    Args:
        report_name: Name of the report
        format_extension: File extension for the format
        
    Returns:
        Default output path
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.getcwd(), 'reports')
    ensure_directory_exists(reports_dir)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{report_name}_{timestamp}.{format_extension}"
    
    return os.path.join(reports_dir, filename)


def save_figure(fig: Any, output_path: str, dpi: int = 300) -> None:
    """Save a figure to file, handling different figure types.
    
    Args:
        fig: Figure object (matplotlib, plotly, or bokeh)
        output_path: Path to save the figure
        dpi: DPI for raster formats
    """
    # Ensure directory exists
    directory = os.path.dirname(output_path)
    ensure_directory_exists(directory)
    
    # Get file extension
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()
    
    # Handle different figure types
    if isinstance(fig, plt.Figure):
        # Matplotlib figure
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    elif PLOTLY_AVAILABLE and isinstance(fig, (go.Figure, go.FigureWidget)):
        # Plotly figure
        if ext == '.html':
            fig.write_html(output_path)
        elif ext == '.json':
            fig.write_json(output_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
            fig.write_image(output_path)
    elif BOKEH_AVAILABLE and 'bokeh' in str(type(fig)):
        # Bokeh figure
        from bokeh.io import save
        save(fig, filename=output_path)
    else:
        raise ValueError(f"Unsupported figure type: {type(fig)}")


# Export functions
__all__ = [
    # Data processing
    'calculate_returns_metrics',
    'calculate_rolling_metrics',
    'calculate_drawdowns',
    'calculate_monthly_returns',
    'calculate_correlation_matrix',
    'calculate_risk_contribution',
    'calculate_factor_contribution',
    
    # Formatting
    'format_percentage',
    'format_currency',
    'format_number',
    
    # Visualization helpers
    'get_color_palette',
    'create_custom_colormap',
    'add_value_labels',
    'format_axis_ticks',
    
    # File and path utilities
    'ensure_directory_exists',
    'get_default_output_path',
    'save_figure'
]