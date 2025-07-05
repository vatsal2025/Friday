"""Interactive dashboard module for portfolio analysis.

This module provides tools for creating interactive dashboards for portfolio analysis,
including performance, allocation, risk, and comparative analysis dashboards.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    PLOTLY_AVAILABLE = True
    DASH_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    DASH_AVAILABLE = False

# Import local modules
from ..analytics.visualization import (
    PerformanceVisualizer, AllocationVisualizer, 
    RiskVisualizer, InteractiveVisualizer
)
from ..analytics.comparative import StrategyComparator, ScenarioAnalyzer


class DashboardBase:
    """Base class for interactive dashboards."""
    
    def __init__(self, title: str = "Portfolio Analysis Dashboard"):
        """Initialize the dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.app = None
        
        # Check if required dependencies are available
        if not PLOTLY_AVAILABLE or not DASH_AVAILABLE:
            raise ImportError(
                "Plotly and Dash are required for interactive dashboards. "
                "Please install them with 'pip install plotly dash'."
            )
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the dashboard.
        
        Returns:
            Dash app instance
        """
        raise NotImplementedError("Subclasses must implement create_app()")
    
    def run_server(self, debug: bool = False, port: int = 8050, host: str = "127.0.0.1") -> None:
        """Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
            host: Host to run the server on
        """
        if self.app is None:
            self.app = self.create_app()
        
        self.app.run_server(debug=debug, port=port, host=host)


class PerformanceDashboard(DashboardBase):
    """Interactive dashboard for portfolio performance analysis."""
    
    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Portfolio Performance Dashboard"
    ):
        """Initialize the performance dashboard.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns series
            title: Dashboard title
        """
        super().__init__(title)
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.performance_visualizer = PerformanceVisualizer()
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the performance dashboard.
        
        Returns:
            Dash app instance
        """
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Time Period"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=self.returns.index.min(),
                        max_date_allowed=self.returns.index.max(),
                        start_date=self.returns.index.min(),
                        end_date=self.returns.index.max()
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Chart Type"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {'label': 'Equity Curve', 'value': 'equity'},
                            {'label': 'Drawdown', 'value': 'drawdown'},
                            {'label': 'Returns Distribution', 'value': 'distribution'},
                            {'label': 'Monthly Returns Heatmap', 'value': 'heatmap'},
                            {'label': 'Rolling Statistics', 'value': 'rolling'}
                        ],
                        value='equity'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Rolling Window"),
                    dcc.Slider(
                        id='rolling-window',
                        min=5,
                        max=252,
                        step=5,
                        value=21,
                        marks={i: str(i) for i in range(0, 253, 21)}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='performance-chart')
            ]),
            
            html.Div([
                html.H3("Performance Metrics", style={'textAlign': 'center'}),
                html.Div(id='performance-metrics')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('performance-chart', 'figure'),
             Output('performance-metrics', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('chart-type', 'value'),
             Input('rolling-window', 'value')]
        )
        def update_chart(start_date, end_date, chart_type, rolling_window):
            # Filter returns by date range
            filtered_returns = self.returns.loc[start_date:end_date]
            
            # Filter benchmark returns if available
            filtered_benchmark = None
            if self.benchmark_returns is not None:
                filtered_benchmark = self.benchmark_returns.loc[start_date:end_date]
            
            # Create figure based on chart type
            if chart_type == 'equity':
                fig = self.performance_visualizer.plot_equity_curve(
                    filtered_returns,
                    benchmark_returns=filtered_benchmark,
                    title="Equity Curve",
                    interactive=True
                )
            elif chart_type == 'drawdown':
                fig = self.performance_visualizer.plot_drawdown(
                    filtered_returns,
                    title="Drawdown",
                    interactive=True
                )
            elif chart_type == 'distribution':
                fig = self.performance_visualizer.plot_returns_distribution(
                    filtered_returns,
                    benchmark_returns=filtered_benchmark,
                    title="Returns Distribution",
                    interactive=True
                )
            elif chart_type == 'heatmap':
                fig = self.performance_visualizer.plot_monthly_returns_heatmap(
                    filtered_returns,
                    title="Monthly Returns Heatmap",
                    interactive=True
                )
            elif chart_type == 'rolling':
                fig = self.performance_visualizer.plot_rolling_statistics(
                    filtered_returns,
                    window=rolling_window,
                    benchmark_returns=filtered_benchmark,
                    title=f"Rolling Statistics ({rolling_window}-day window)",
                    interactive=True
                )
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(filtered_returns, filtered_benchmark)
            
            # Create metrics table
            metrics_table = html.Table(
                # Header
                [html.Tr([html.Th(col) for col in metrics.columns])] +
                # Rows
                [html.Tr([html.Td(metrics.iloc[i][col]) for col in metrics.columns])
                 for i in range(len(metrics))],
                style={'margin': 'auto'}
            )
            
            return fig, metrics_table
        
        return app
    
    def _calculate_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate performance metrics.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns series
            
        Returns:
            DataFrame with performance metrics
        """
        # Calculate annualization factor based on frequency
        if returns.index.freq == 'D' or pd.infer_freq(returns.index) in ['D', 'B']:
            annualization_factor = 252
        elif returns.index.freq == 'W' or pd.infer_freq(returns.index) == 'W':
            annualization_factor = 52
        elif returns.index.freq == 'M' or pd.infer_freq(returns.index) == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252  # Default to daily
        
        # Initialize metrics dictionary
        metrics_dict = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Calmar Ratio',
                'Best Month',
                'Worst Month',
                'Positive Months %'
            ],
            'Portfolio': []
        }
        
        # Add benchmark column if available
        if benchmark_returns is not None:
            metrics_dict['Benchmark'] = []
        
        # Calculate portfolio metrics
        # Total return
        total_return = (1 + returns).prod() - 1
        metrics_dict['Portfolio'].append(f"{total_return:.2%}")
        
        # Annualized return
        ann_return = (1 + returns).prod() ** (annualization_factor / len(returns)) - 1
        metrics_dict['Portfolio'].append(f"{ann_return:.2%}")
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(annualization_factor)
        metrics_dict['Portfolio'].append(f"{ann_vol:.2%}")
        
        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        metrics_dict['Portfolio'].append(f"{sharpe:.2f}")
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_deviation if downside_deviation > 0 else 0
        metrics_dict['Portfolio'].append(f"{sortino:.2f}")
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics_dict['Portfolio'].append(f"{max_drawdown:.2%}")
        
        # Calmar ratio
        calmar = -ann_return / max_drawdown if max_drawdown < 0 else 0
        metrics_dict['Portfolio'].append(f"{calmar:.2f}")
        
        # Best and worst months
        if returns.index.freq == 'M' or pd.infer_freq(returns.index) == 'M':
            monthly_returns = returns
        else:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        metrics_dict['Portfolio'].append(f"{best_month:.2%}")
        metrics_dict['Portfolio'].append(f"{worst_month:.2%}")
        
        # Positive months percentage
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns)
        metrics_dict['Portfolio'].append(f"{positive_months:.2%}")
        
        # Calculate benchmark metrics if available
        if benchmark_returns is not None:
            # Total return
            b_total_return = (1 + benchmark_returns).prod() - 1
            metrics_dict['Benchmark'].append(f"{b_total_return:.2%}")
            
            # Annualized return
            b_ann_return = (1 + benchmark_returns).prod() ** (annualization_factor / len(benchmark_returns)) - 1
            metrics_dict['Benchmark'].append(f"{b_ann_return:.2%}")
            
            # Annualized volatility
            b_ann_vol = benchmark_returns.std() * np.sqrt(annualization_factor)
            metrics_dict['Benchmark'].append(f"{b_ann_vol:.2%}")
            
            # Sharpe ratio
            b_sharpe = b_ann_return / b_ann_vol if b_ann_vol > 0 else 0
            metrics_dict['Benchmark'].append(f"{b_sharpe:.2f}")
            
            # Sortino ratio
            b_downside_returns = benchmark_returns[benchmark_returns < 0]
            b_downside_deviation = b_downside_returns.std() * np.sqrt(annualization_factor) if len(b_downside_returns) > 0 else 0
            b_sortino = b_ann_return / b_downside_deviation if b_downside_deviation > 0 else 0
            metrics_dict['Benchmark'].append(f"{b_sortino:.2f}")
            
            # Maximum drawdown
            b_cumulative_returns = (1 + benchmark_returns).cumprod()
            b_running_max = b_cumulative_returns.cummax()
            b_drawdown = (b_cumulative_returns - b_running_max) / b_running_max
            b_max_drawdown = b_drawdown.min()
            metrics_dict['Benchmark'].append(f"{b_max_drawdown:.2%}")
            
            # Calmar ratio
            b_calmar = -b_ann_return / b_max_drawdown if b_max_drawdown < 0 else 0
            metrics_dict['Benchmark'].append(f"{b_calmar:.2f}")
            
            # Best and worst months
            if benchmark_returns.index.freq == 'M' or pd.infer_freq(benchmark_returns.index) == 'M':
                b_monthly_returns = benchmark_returns
            else:
                b_monthly_returns = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            b_best_month = b_monthly_returns.max()
            b_worst_month = b_monthly_returns.min()
            metrics_dict['Benchmark'].append(f"{b_best_month:.2%}")
            metrics_dict['Benchmark'].append(f"{b_worst_month:.2%}")
            
            # Positive months percentage
            b_positive_months = (b_monthly_returns > 0).sum() / len(b_monthly_returns)
            metrics_dict['Benchmark'].append(f"{b_positive_months:.2%}")
        
        return pd.DataFrame(metrics_dict)


class AllocationDashboard(DashboardBase):
    """Interactive dashboard for portfolio allocation analysis."""
    
    def __init__(
        self,
        portfolio_weights: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
        historical_weights: Optional[pd.DataFrame] = None,
        title: str = "Portfolio Allocation Dashboard"
    ):
        """Initialize the allocation dashboard.
        
        Args:
            portfolio_weights: Dictionary mapping asset names to current weights
            target_weights: Optional dictionary mapping asset names to target weights
            historical_weights: Optional DataFrame with historical weights
            title: Dashboard title
        """
        super().__init__(title)
        self.portfolio_weights = portfolio_weights
        self.target_weights = target_weights
        self.historical_weights = historical_weights
        self.allocation_visualizer = AllocationVisualizer()
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the allocation dashboard.
        
        Returns:
            Dash app instance
        """
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Chart Type"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {'label': 'Asset Allocation', 'value': 'allocation'},
                            {'label': 'Allocation Drift', 'value': 'drift'},
                            {'label': 'Allocation History', 'value': 'history'},
                            {'label': 'Sector Allocation', 'value': 'sector'}
                        ],
                        value='allocation'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Chart Style"),
                    dcc.Dropdown(
                        id='chart-style',
                        options=[
                            {'label': 'Pie Chart', 'value': 'pie'},
                            {'label': 'Bar Chart', 'value': 'bar'},
                            {'label': 'Treemap', 'value': 'treemap'}
                        ],
                        value='pie'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Group By"),
                    dcc.Dropdown(
                        id='group-by',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Asset Class', 'value': 'asset_class'},
                            {'label': 'Sector', 'value': 'sector'},
                            {'label': 'Country', 'value': 'country'}
                        ],
                        value='none'
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='allocation-chart')
            ]),
            
            html.Div([
                html.H3("Allocation Details", style={'textAlign': 'center'}),
                html.Div(id='allocation-details')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('allocation-chart', 'figure'),
             Output('allocation-details', 'children')],
            [Input('chart-type', 'value'),
             Input('chart-style', 'value'),
             Input('group-by', 'value')]
        )
        def update_chart(chart_type, chart_style, group_by):
            # Create figure based on chart type
            if chart_type == 'allocation':
                fig = self.allocation_visualizer.plot_asset_allocation(
                    self.portfolio_weights,
                    title="Current Asset Allocation",
                    plot_type=chart_style,
                    interactive=True
                )
            elif chart_type == 'drift' and self.target_weights is not None:
                fig = self.allocation_visualizer.plot_allocation_drift(
                    self.portfolio_weights,
                    self.target_weights,
                    title="Allocation Drift from Target",
                    interactive=True
                )
            elif chart_type == 'history' and self.historical_weights is not None:
                fig = self.allocation_visualizer.plot_allocation_history(
                    self.historical_weights,
                    title="Historical Allocation",
                    interactive=True
                )
            elif chart_type == 'sector':
                # This would require sector data, which we don't have
                # For now, just use the regular allocation chart
                fig = self.allocation_visualizer.plot_asset_allocation(
                    self.portfolio_weights,
                    title="Asset Allocation by Sector",
                    plot_type=chart_style,
                    interactive=True
                )
            else:
                # Default to asset allocation
                fig = self.allocation_visualizer.plot_asset_allocation(
                    self.portfolio_weights,
                    title="Current Asset Allocation",
                    plot_type=chart_style,
                    interactive=True
                )
            
            # Create allocation details table
            details = self._create_allocation_details()
            
            return fig, details
        
        return app
    
    def _create_allocation_details(self) -> html.Table:
        """Create allocation details table.
        
        Returns:
            HTML table with allocation details
        """
        # Create DataFrame with current and target weights
        data = {'Asset': [], 'Current Weight': [], 'Target Weight': [], 'Difference': []}
        
        for asset, weight in self.portfolio_weights.items():
            data['Asset'].append(asset)
            data['Current Weight'].append(f"{weight:.2%}")
            
            if self.target_weights is not None and asset in self.target_weights:
                target = self.target_weights[asset]
                data['Target Weight'].append(f"{target:.2%}")
                data['Difference'].append(f"{weight - target:.2%}")
            else:
                data['Target Weight'].append("N/A")
                data['Difference'].append("N/A")
        
        # Convert to DataFrame and sort by current weight
        df = pd.DataFrame(data)
        df = df.sort_values('Current Weight', ascending=False)
        
        # Create HTML table
        table = html.Table(
            # Header
            [html.Tr([html.Th(col) for col in df.columns])] +
            # Rows
            [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
             for i in range(len(df))],
            style={'margin': 'auto'}
        )
        
        return table


class RiskDashboard(DashboardBase):
    """Interactive dashboard for portfolio risk analysis."""
    
    def __init__(
        self,
        returns: pd.Series,
        asset_returns: Optional[pd.DataFrame] = None,
        factor_exposures: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_contributions: Optional[Dict[str, float]] = None,
        title: str = "Portfolio Risk Dashboard"
    ):
        """Initialize the risk dashboard.
        
        Args:
            returns: Portfolio returns series
            asset_returns: Optional DataFrame with asset returns
            factor_exposures: Optional DataFrame with factor exposures
            factor_returns: Optional DataFrame with factor returns
            risk_contributions: Optional dictionary with risk contributions
            title: Dashboard title
        """
        super().__init__(title)
        self.returns = returns
        self.asset_returns = asset_returns
        self.factor_exposures = factor_exposures
        self.factor_returns = factor_returns
        self.risk_contributions = risk_contributions
        self.risk_visualizer = RiskVisualizer()
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the risk dashboard.
        
        Returns:
            Dash app instance
        """
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Chart Type"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {'label': 'Risk Contribution', 'value': 'risk_contribution'},
                            {'label': 'Correlation Matrix', 'value': 'correlation'},
                            {'label': 'Factor Exposures', 'value': 'factor_exposures'},
                            {'label': 'Rolling Volatility', 'value': 'rolling_vol'},
                            {'label': 'Drawdown Analysis', 'value': 'drawdown'},
                            {'label': 'Value at Risk', 'value': 'var'}
                        ],
                        value='risk_contribution'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Time Period"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=self.returns.index.min(),
                        max_date_allowed=self.returns.index.max(),
                        start_date=self.returns.index.min(),
                        end_date=self.returns.index.max()
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Rolling Window"),
                    dcc.Slider(
                        id='rolling-window',
                        min=5,
                        max=252,
                        step=5,
                        value=21,
                        marks={i: str(i) for i in range(0, 253, 21)}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='risk-chart')
            ]),
            
            html.Div([
                html.H3("Risk Metrics", style={'textAlign': 'center'}),
                html.Div(id='risk-metrics')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('risk-chart', 'figure'),
             Output('risk-metrics', 'children')],
            [Input('chart-type', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('rolling-window', 'value')]
        )
        def update_chart(chart_type, start_date, end_date, rolling_window):
            # Filter returns by date range
            filtered_returns = self.returns.loc[start_date:end_date]
            
            # Filter asset returns if available
            filtered_asset_returns = None
            if self.asset_returns is not None:
                filtered_asset_returns = self.asset_returns.loc[start_date:end_date]
            
            # Create figure based on chart type
            if chart_type == 'risk_contribution' and self.risk_contributions is not None:
                fig = self.risk_visualizer.plot_risk_contribution(
                    self.risk_contributions,
                    title="Risk Contribution",
                    interactive=True
                )
            elif chart_type == 'correlation' and filtered_asset_returns is not None:
                fig = self.risk_visualizer.plot_correlation_matrix(
                    filtered_asset_returns,
                    title="Correlation Matrix",
                    interactive=True
                )
            elif chart_type == 'factor_exposures' and self.factor_exposures is not None:
                fig = self.risk_visualizer.plot_factor_exposures(
                    self.factor_exposures,
                    title="Factor Exposures",
                    interactive=True
                )
            elif chart_type == 'rolling_vol':
                fig = self.risk_visualizer.plot_rolling_volatility(
                    filtered_returns,
                    window=rolling_window,
                    title=f"Rolling Volatility ({rolling_window}-day window)",
                    interactive=True
                )
            elif chart_type == 'drawdown':
                fig = self.risk_visualizer.plot_drawdown_analysis(
                    filtered_returns,
                    title="Drawdown Analysis",
                    interactive=True
                )
            elif chart_type == 'var':
                fig = self.risk_visualizer.plot_value_at_risk(
                    filtered_returns,
                    title="Value at Risk",
                    interactive=True
                )
            else:
                # Default to rolling volatility
                fig = self.risk_visualizer.plot_rolling_volatility(
                    filtered_returns,
                    window=rolling_window,
                    title=f"Rolling Volatility ({rolling_window}-day window)",
                    interactive=True
                )
            
            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(filtered_returns)
            
            # Create metrics table
            metrics_table = html.Table(
                # Header
                [html.Tr([html.Th(col) for col in metrics.columns])] +
                # Rows
                [html.Tr([html.Td(metrics.iloc[i][col]) for col in metrics.columns])
                 for i in range(len(metrics))],
                style={'margin': 'auto'}
            )
            
            return fig, metrics_table
        
        return app
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate risk metrics.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            DataFrame with risk metrics
        """
        # Calculate annualization factor based on frequency
        if returns.index.freq == 'D' or pd.infer_freq(returns.index) in ['D', 'B']:
            annualization_factor = 252
        elif returns.index.freq == 'W' or pd.infer_freq(returns.index) == 'W':
            annualization_factor = 52
        elif returns.index.freq == 'M' or pd.infer_freq(returns.index) == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252  # Default to daily
        
        # Initialize metrics dictionary
        metrics_dict = {
            'Metric': [
                'Annualized Volatility',
                'Max Drawdown',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Skewness',
                'Kurtosis',
                'Downside Deviation',
                'Sortino Ratio',
                'Calmar Ratio',
                'Ulcer Index'
            ],
            'Value': []
        }
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(annualization_factor)
        metrics_dict['Value'].append(f"{ann_vol:.2%}")
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics_dict['Value'].append(f"{max_drawdown:.2%}")
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        metrics_dict['Value'].append(f"{var_95:.2%}")
        
        # Conditional VaR (95%)
        cvar_95 = returns[returns <= var_95].mean()
        metrics_dict['Value'].append(f"{cvar_95:.2%}")
        
        # Skewness
        skewness = returns.skew()
        metrics_dict['Value'].append(f"{skewness:.2f}")
        
        # Kurtosis
        kurtosis = returns.kurtosis()
        metrics_dict['Value'].append(f"{kurtosis:.2f}")
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 0
        metrics_dict['Value'].append(f"{downside_deviation:.2%}")
        
        # Sortino ratio
        ann_return = (1 + returns).prod() ** (annualization_factor / len(returns)) - 1
        sortino = ann_return / downside_deviation if downside_deviation > 0 else 0
        metrics_dict['Value'].append(f"{sortino:.2f}")
        
        # Calmar ratio
        calmar = -ann_return / max_drawdown if max_drawdown < 0 else 0
        metrics_dict['Value'].append(f"{calmar:.2f}")
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.sum(drawdown ** 2) / len(drawdown))
        metrics_dict['Value'].append(f"{ulcer_index:.2%}")
        
        return pd.DataFrame(metrics_dict)


class ComparativeDashboard(DashboardBase):
    """Interactive dashboard for comparative analysis."""
    
    def __init__(
        self,
        returns_dict: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Comparative Analysis Dashboard"
    ):
        """Initialize the comparative dashboard.
        
        Args:
            returns_dict: Dictionary mapping strategy names to return series
            benchmark_returns: Optional benchmark return series
            title: Dashboard title
        """
        super().__init__(title)
        self.returns_dict = returns_dict
        self.benchmark_returns = benchmark_returns
        self.strategy_comparator = StrategyComparator()
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the comparative dashboard.
        
        Returns:
            Dash app instance
        """
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Time Period"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=min([returns.index.min() for returns in self.returns_dict.values()]),
                        max_date_allowed=max([returns.index.max() for returns in self.returns_dict.values()]),
                        start_date=min([returns.index.min() for returns in self.returns_dict.values()]),
                        end_date=max([returns.index.max() for returns in self.returns_dict.values()])
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Chart Type"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {'label': 'Equity Curves', 'value': 'equity'},
                            {'label': 'Drawdowns', 'value': 'drawdown'},
                            {'label': 'Rolling Returns', 'value': 'rolling_returns'},
                            {'label': 'Rolling Sharpe', 'value': 'rolling_sharpe'},
                            {'label': 'Performance Metrics', 'value': 'metrics'}
                        ],
                        value='equity'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Rolling Window"),
                    dcc.Slider(
                        id='rolling-window',
                        min=5,
                        max=252,
                        step=5,
                        value=21,
                        marks={i: str(i) for i in range(0, 253, 21)}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='comparison-chart')
            ]),
            
            html.Div([
                html.H3("Performance Comparison", style={'textAlign': 'center'}),
                html.Div(id='comparison-metrics')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('comparison-chart', 'figure'),
             Output('comparison-metrics', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('chart-type', 'value'),
             Input('rolling-window', 'value')]
        )
        def update_chart(start_date, end_date, chart_type, rolling_window):
            # Filter returns by date range
            filtered_returns_dict = {}
            for strategy, returns in self.returns_dict.items():
                filtered_returns_dict[strategy] = returns.loc[start_date:end_date]
            
            # Filter benchmark returns if available
            filtered_benchmark = None
            if self.benchmark_returns is not None:
                filtered_benchmark = self.benchmark_returns.loc[start_date:end_date]
            
            # Create figure based on chart type
            if chart_type == 'equity':
                # Create equity curves
                fig = go.Figure()
                
                for strategy, returns in filtered_returns_dict.items():
                    equity_curve = (1 + returns).cumprod()
                    fig.add_trace(go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve.values,
                        mode='lines',
                        name=strategy
                    ))
                
                if filtered_benchmark is not None:
                    benchmark_equity = (1 + filtered_benchmark).cumprod()
                    fig.add_trace(go.Scatter(
                        x=benchmark_equity.index,
                        y=benchmark_equity.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title="Equity Curves",
                    xaxis_title="Date",
                    yaxis_title="Growth of $1",
                    template="plotly_white"
                )
            
            elif chart_type == 'drawdown':
                # Create drawdown charts
                fig = go.Figure()
                
                for strategy, returns in filtered_returns_dict.items():
                    equity_curve = (1 + returns).cumprod()
                    running_max = equity_curve.cummax()
                    drawdown = (equity_curve - running_max) / running_max
                    
                    fig.add_trace(go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        mode='lines',
                        name=strategy
                    ))
                
                if filtered_benchmark is not None:
                    benchmark_equity = (1 + filtered_benchmark).cumprod()
                    benchmark_max = benchmark_equity.cummax()
                    benchmark_drawdown = (benchmark_equity - benchmark_max) / benchmark_max
                    
                    fig.add_trace(go.Scatter(
                        x=benchmark_drawdown.index,
                        y=benchmark_drawdown.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title="Drawdowns",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    template="plotly_white",
                    yaxis=dict(tickformat='%')
                )
            
            elif chart_type == 'rolling_returns':
                # Create rolling returns charts
                fig = go.Figure()
                
                for strategy, returns in filtered_returns_dict.items():
                    rolling_returns = returns.rolling(window=rolling_window).apply(
                        lambda x: (1 + x).prod() - 1
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_returns.index,
                        y=rolling_returns.values,
                        mode='lines',
                        name=strategy
                    ))
                
                if filtered_benchmark is not None:
                    benchmark_rolling = filtered_benchmark.rolling(window=rolling_window).apply(
                        lambda x: (1 + x).prod() - 1
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=benchmark_rolling.index,
                        y=benchmark_rolling.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Rolling {rolling_window}-Day Returns",
                    xaxis_title="Date",
                    yaxis_title="Return",
                    template="plotly_white",
                    yaxis=dict(tickformat='%')
                )
            
            elif chart_type == 'rolling_sharpe':
                # Create rolling Sharpe ratio charts
                fig = go.Figure()
                
                for strategy, returns in filtered_returns_dict.items():
                    rolling_returns = returns.rolling(window=rolling_window).mean() * 252
                    rolling_vol = returns.rolling(window=rolling_window).std() * np.sqrt(252)
                    rolling_sharpe = rolling_returns / rolling_vol
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name=strategy
                    ))
                
                if filtered_benchmark is not None:
                    benchmark_returns = filtered_benchmark.rolling(window=rolling_window).mean() * 252
                    benchmark_vol = filtered_benchmark.rolling(window=rolling_window).std() * np.sqrt(252)
                    benchmark_sharpe = benchmark_returns / benchmark_vol
                    
                    fig.add_trace(go.Scatter(
                        x=benchmark_sharpe.index,
                        y=benchmark_sharpe.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Rolling {rolling_window}-Day Sharpe Ratio",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    template="plotly_white"
                )
            
            elif chart_type == 'metrics':
                # Compare performance metrics
                comparison_data = self.strategy_comparator.compare(
                    filtered_returns_dict,
                    benchmark_returns=filtered_benchmark
                )
                
                fig = self.strategy_comparator.plot_comparison(
                    comparison_data,
                    title="Performance Metrics Comparison",
                    interactive=True,
                    plot_type="bar"
                )
            
            else:
                # Default to equity curves
                fig = go.Figure()
                
                for strategy, returns in filtered_returns_dict.items():
                    equity_curve = (1 + returns).cumprod()
                    fig.add_trace(go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve.values,
                        mode='lines',
                        name=strategy
                    ))
                
                if filtered_benchmark is not None:
                    benchmark_equity = (1 + filtered_benchmark).cumprod()
                    fig.add_trace(go.Scatter(
                        x=benchmark_equity.index,
                        y=benchmark_equity.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(dash='dash')
                    ))
                
                fig.update_layout(
                    title="Equity Curves",
                    xaxis_title="Date",
                    yaxis_title="Growth of $1",
                    template="plotly_white"
                )
            
            # Calculate comparison metrics
            comparison_data = self.strategy_comparator.compare(
                filtered_returns_dict,
                benchmark_returns=filtered_benchmark
            )
            
            # Format metrics for display
            metrics_df = comparison_data.copy()
            metrics_df.index = [idx.replace('_', ' ').title() for idx in metrics_df.index]
            
            # Select key metrics to display
            display_metrics = [
                'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
                'Max Drawdown', 'Sortino Ratio', 'Calmar Ratio'
            ]
            
            # Filter metrics that exist in the data
            display_metrics = [m for m in display_metrics if m in metrics_df.index]
            
            # Create metrics table
            metrics_table = html.Table(
                # Header
                [html.Tr([html.Th('Metric')] + [html.Th(col) for col in metrics_df.columns])] +
                # Rows
                [html.Tr([html.Td(metric)] + 
                         [html.Td(f"{metrics_df.loc[metric, col]:.2%}" if abs(metrics_df.loc[metric, col]) < 10 else f"{metrics_df.loc[metric, col]:.2f}")
                          for col in metrics_df.columns])
                 for metric in display_metrics],
                style={'margin': 'auto'}
            )
            
            return fig, metrics_table
        
        return app


class ScenarioDashboard(DashboardBase):
    """Interactive dashboard for scenario analysis."""
    
    def __init__(
        self,
        portfolios: Dict[str, Dict[str, float]],
        scenarios: List[Dict[str, Any]],
        title: str = "Scenario Analysis Dashboard"
    ):
        """Initialize the scenario dashboard.
        
        Args:
            portfolios: Dictionary mapping portfolio names to weight dictionaries
            scenarios: List of scenario definitions
            title: Dashboard title
        """
        super().__init__(title)
        self.portfolios = portfolios
        self.scenarios = scenarios
        self.scenario_analyzer = ScenarioAnalyzer()
    
    def create_app(self) -> dash.Dash:
        """Create a Dash app for the scenario dashboard.
        
        Returns:
            Dash app instance
        """
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H3("Chart Type"),
                    dcc.Dropdown(
                        id='chart-type',
                        options=[
                            {'label': 'Heatmap', 'value': 'heatmap'},
                            {'label': 'Bar Chart', 'value': 'bar'}
                        ],
                        value='heatmap'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Portfolio Selection"),
                    dcc.Dropdown(
                        id='portfolio-selection',
                        options=[{'label': name, 'value': name} for name in self.portfolios.keys()],
                        value=list(self.portfolios.keys()),
                        multi=True
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Scenario Selection"),
                    dcc.Dropdown(
                        id='scenario-selection',
                        options=[{'label': s['name'], 'value': s['name']} for s in self.scenarios],
                        value=[s['name'] for s in self.scenarios],
                        multi=True
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                dcc.Graph(id='scenario-chart')
            ]),
            
            html.Div([
                html.H3("Scenario Details", style={'textAlign': 'center'}),
                html.Div(id='scenario-details')
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('scenario-chart', 'figure'),
             Output('scenario-details', 'children')],
            [Input('chart-type', 'value'),
             Input('portfolio-selection', 'value'),
             Input('scenario-selection', 'value')]
        )
        def update_chart(chart_type, portfolio_selection, scenario_selection):
            # Filter portfolios and scenarios based on selection
            selected_portfolios = {name: self.portfolios[name] for name in portfolio_selection}
            selected_scenarios = [s for s in self.scenarios if s['name'] in scenario_selection]
            
            # Run scenario analysis
            comparison_data = self.scenario_analyzer.compare(selected_portfolios, selected_scenarios)
            
            # Create figure based on chart type
            fig = self.scenario_analyzer.plot_comparison(
                comparison_data,
                title="Scenario Analysis",
                interactive=True,
                plot_type=chart_type
            )
            
            # Create scenario details
            details = self._create_scenario_details(selected_scenarios)
            
            return fig, details
        
        return app
    
    def _create_scenario_details(self, scenarios: List[Dict[str, Any]]) -> html.Div:
        """Create scenario details.
        
        Args:
            scenarios: List of scenario definitions
            
        Returns:
            HTML div with scenario details
        """
        details = []
        
        for scenario in scenarios:
            # Create scenario card
            card = html.Div([
                html.H4(scenario['name']),
                html.P(scenario['description'] if scenario['description'] else "No description provided."),
                html.H5("Asset Returns:"),
                html.Table(
                    # Header
                    [html.Tr([html.Th("Asset"), html.Th("Expected Return")])],
                    # Rows
                    [html.Tr([html.Td(asset), html.Td(f"{return_value:.2%}")])
                     for asset, return_value in scenario['asset_returns'].items()]
                )
            ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
            
            details.append(card)
        
        return html.Div(details, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})


# Export all dashboard classes
__all__ = [
    'DashboardBase',
    'PerformanceDashboard',
    'AllocationDashboard',
    'RiskDashboard',
    'ComparativeDashboard',
    'ScenarioDashboard'
]