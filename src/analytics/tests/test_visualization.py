"""Tests for the analytics visualization module."""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import visualization module
from analytics.visualization import (
    BaseVisualizer,
    PerformanceVisualizer,
    AllocationVisualizer,
    RiskVisualizer,
    TaxVisualizer,
    InteractiveVisualizer,
    ComparativeVisualizer
)


class TestBaseVisualizer(unittest.TestCase):
    """Test cases for BaseVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = BaseVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_init(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.visualizer.figsize, (10, 6))
        self.assertEqual(self.visualizer.style, 'seaborn')
        self.assertEqual(self.visualizer.dpi, 100)

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        visualizer = BaseVisualizer(figsize=(12, 8), style='ggplot', dpi=200)
        self.assertEqual(visualizer.figsize, (12, 8))
        self.assertEqual(visualizer.style, 'ggplot')
        self.assertEqual(visualizer.dpi, 200)

    def test_create_figure(self):
        """Test creating a figure."""
        fig, ax = self.visualizer.create_figure()
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

    def test_create_figure_with_params(self):
        """Test creating a figure with custom parameters."""
        fig, ax = self.visualizer.create_figure(figsize=(8, 4), dpi=150)
        self.assertEqual(fig.get_figwidth(), 8)
        self.assertEqual(fig.get_figheight(), 4)
        self.assertEqual(fig.dpi, 150)

    def test_save_figure(self):
        """Test saving a figure."""
        fig, _ = self.visualizer.create_figure()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_name = temp.name

        try:
            self.visualizer.save_figure(fig, temp_name)
            self.assertTrue(os.path.exists(temp_name))
            self.assertTrue(os.path.getsize(temp_name) > 0)
        finally:
            os.unlink(temp_name)

    def test_save_figure_with_params(self):
        """Test saving a figure with custom parameters."""
        fig, _ = self.visualizer.create_figure()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_name = temp.name

        try:
            self.visualizer.save_figure(fig, temp_name, dpi=200, bbox_inches='tight')
            self.assertTrue(os.path.exists(temp_name))
            self.assertTrue(os.path.getsize(temp_name) > 0)
        finally:
            os.unlink(temp_name)

    @patch('matplotlib.pyplot.show')
    def test_show_figure(self, mock_show):
        """Test showing a figure."""
        fig, _ = self.visualizer.create_figure()
        self.visualizer.show_figure(fig)
        mock_show.assert_called_once()


class TestPerformanceVisualizer(unittest.TestCase):
    """Test cases for PerformanceVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        cumulative_returns = (1 + returns).cumprod() - 1
        drawdowns = np.random.uniform(-0.3, 0, 100)
        drawdowns.sort()  # Sort to simulate realistic drawdowns

        self.returns = pd.Series(returns, index=dates)
        self.cumulative_returns = pd.Series(cumulative_returns, index=dates)
        self.drawdowns = pd.Series(drawdowns, index=dates)

        # Create benchmark data
        benchmark_returns = np.random.normal(0.0008, 0.018, 100)
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates)
        self.benchmark_cumulative_returns = pd.Series(benchmark_cumulative_returns, index=dates)

        # Create monthly returns
        months = 12
        years = 3
        monthly_returns = np.random.normal(0.01, 0.03, months * years).reshape(years, months)
        self.monthly_returns = pd.DataFrame(
            monthly_returns,
            index=pd.date_range(start='2020-01-01', periods=years, freq='YS').year,
            columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )

        self.visualizer = PerformanceVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_plot_equity_curve(self):
        """Test plotting equity curve."""
        fig = self.visualizer.plot_equity_curve(self.cumulative_returns)
        self.assertIsInstance(fig, plt.Figure)

        # Test with benchmark
        fig = self.visualizer.plot_equity_curve(
            self.cumulative_returns,
            benchmark=self.benchmark_cumulative_returns
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_drawdown(self):
        """Test plotting drawdown."""
        fig = self.visualizer.plot_drawdown(self.drawdowns)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_returns_distribution(self):
        """Test plotting returns distribution."""
        fig = self.visualizer.plot_returns_distribution(self.returns)
        self.assertIsInstance(fig, plt.Figure)

        # Test with benchmark
        fig = self.visualizer.plot_returns_distribution(
            self.returns,
            benchmark=self.benchmark_returns
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_monthly_returns_heatmap(self):
        """Test plotting monthly returns heatmap."""
        fig = self.visualizer.plot_monthly_returns_heatmap(self.monthly_returns)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_rolling_returns(self):
        """Test plotting rolling returns."""
        fig = self.visualizer.plot_rolling_returns(self.returns, window=30)
        self.assertIsInstance(fig, plt.Figure)

        # Test with benchmark
        fig = self.visualizer.plot_rolling_returns(
            self.returns,
            benchmark=self.benchmark_returns,
            window=30
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_rolling_volatility(self):
        """Test plotting rolling volatility."""
        fig = self.visualizer.plot_rolling_volatility(self.returns, window=30)
        self.assertIsInstance(fig, plt.Figure)

        # Test with benchmark
        fig = self.visualizer.plot_rolling_volatility(
            self.returns,
            benchmark=self.benchmark_returns,
            window=30
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_rolling_sharpe(self):
        """Test plotting rolling Sharpe ratio."""
        fig = self.visualizer.plot_rolling_sharpe(self.returns, window=30)
        self.assertIsInstance(fig, plt.Figure)

        # Test with benchmark
        fig = self.visualizer.plot_rolling_sharpe(
            self.returns,
            benchmark=self.benchmark_returns,
            window=30
        )
        self.assertIsInstance(fig, plt.Figure)


class TestAllocationVisualizer(unittest.TestCase):
    """Test cases for AllocationVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.asset_classes = ['Equities', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
        self.allocations = [0.4, 0.3, 0.1, 0.1, 0.1]
        self.target_allocations = [0.45, 0.25, 0.15, 0.1, 0.05]

        # Create allocation history
        dates = pd.date_range(start='2020-01-01', periods=10, freq='M')
        allocation_history = np.random.uniform(0, 1, (10, 5))
        # Normalize to sum to 1
        allocation_history = allocation_history / allocation_history.sum(axis=1)[:, np.newaxis]
        self.allocation_history = pd.DataFrame(
            allocation_history,
            index=dates,
            columns=self.asset_classes
        )

        # Create sector allocations
        self.sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Other']
        self.sector_allocations = [0.25, 0.2, 0.15, 0.15, 0.1, 0.15]

        self.visualizer = AllocationVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_plot_asset_allocation(self):
        """Test plotting asset allocation."""
        fig = self.visualizer.plot_asset_allocation(
            self.asset_classes,
            self.allocations
        )
        self.assertIsInstance(fig, plt.Figure)

        # Test with targets
        fig = self.visualizer.plot_asset_allocation(
            self.asset_classes,
            self.allocations,
            targets=self.target_allocations
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_allocation_drift(self):
        """Test plotting allocation drift."""
        fig = self.visualizer.plot_allocation_drift(
            self.asset_classes,
            self.allocations,
            self.target_allocations
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_allocation_history(self):
        """Test plotting allocation history."""
        fig = self.visualizer.plot_allocation_history(self.allocation_history)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_sector_allocation(self):
        """Test plotting sector allocation."""
        fig = self.visualizer.plot_sector_allocation(
            self.sectors,
            self.sector_allocations
        )
        self.assertIsInstance(fig, plt.Figure)


class TestRiskVisualizer(unittest.TestCase):
    """Test cases for RiskVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.asset_classes = ['Equities', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
        self.risk_contributions = [0.6, 0.2, 0.1, 0.08, 0.02]

        # Create correlation matrix
        np.random.seed(42)  # For reproducibility
        corr = np.random.uniform(-1, 1, (5, 5))
        # Make it symmetric
        corr = (corr + corr.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(corr, 1)
        self.correlation_matrix = pd.DataFrame(
            corr,
            index=self.asset_classes,
            columns=self.asset_classes
        )

        # Create factor exposures
        self.factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        self.factor_exposures = [0.8, 0.3, 0.1, -0.2, 0.4]

        self.visualizer = RiskVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_plot_risk_contribution(self):
        """Test plotting risk contribution."""
        fig = self.visualizer.plot_risk_contribution(
            self.asset_classes,
            self.risk_contributions
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_correlation_matrix(self):
        """Test plotting correlation matrix."""
        fig = self.visualizer.plot_correlation_matrix(self.correlation_matrix)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_factor_exposures(self):
        """Test plotting factor exposures."""
        fig = self.visualizer.plot_factor_exposures(
            self.factors,
            self.factor_exposures
        )
        self.assertIsInstance(fig, plt.Figure)


class TestTaxVisualizer(unittest.TestCase):
    """Test cases for TaxVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=4, freq='Q')
        self.realized_gains = pd.Series([10000, 5000, -3000, 8000], index=dates)
        self.tax_impact = pd.Series([2500, 1250, -750, 2000], index=dates)

        self.visualizer = TaxVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_plot_realized_gains(self):
        """Test plotting realized gains."""
        fig = self.visualizer.plot_realized_gains(self.realized_gains)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_tax_impact(self):
        """Test plotting tax impact."""
        fig = self.visualizer.plot_tax_impact(self.tax_impact)
        self.assertIsInstance(fig, plt.Figure)
        
    def test_plot_tax_optimization_metrics(self):
        """Test plotting tax optimization metrics."""
        # Create sample optimization history
        optimization_history = [
            {
                'type': 'tax_efficiency_metrics',
                'data': {
                    'timestamp': pd.Timestamp('2020-01-01'),
                    'tax_efficiency': {
                        'tax_efficiency_ratio': 0.25,
                        'long_term_gain_ratio': 0.6,
                        'loss_harvesting_efficiency': 0.15,
                        'tax_benefit_from_harvesting': 1200
                    }
                }
            },
            {
                'type': 'tax_efficiency_metrics',
                'data': {
                    'timestamp': pd.Timestamp('2020-04-01'),
                    'tax_efficiency': {
                        'tax_efficiency_ratio': 0.22,
                        'long_term_gain_ratio': 0.65,
                        'loss_harvesting_efficiency': 0.18,
                        'tax_benefit_from_harvesting': 1500
                    }
                }
            },
            {
                'type': 'tax_efficiency_metrics',
                'data': {
                    'timestamp': pd.Timestamp('2020-07-01'),
                    'tax_efficiency': {
                        'tax_efficiency_ratio': 0.20,
                        'long_term_gain_ratio': 0.70,
                        'loss_harvesting_efficiency': 0.22,
                        'tax_benefit_from_harvesting': 1800
                    }
                }
            }
        ]
        
        fig = self.visualizer.plot_tax_optimization_metrics(optimization_history)
        self.assertIsInstance(fig, plt.Figure)


# Mock Plotly for testing InteractiveVisualizer
class MockFigure:
    """Mock Plotly Figure class."""
    def __init__(self, *args, **kwargs):
        self.data = []
        self.layout = {}

    def add_trace(self, *args, **kwargs):
        """Mock add_trace method."""
        return self

    def update_layout(self, *args, **kwargs):
        """Mock update_layout method."""
        return self

    def update_xaxes(self, *args, **kwargs):
        """Mock update_xaxes method."""
        return self

    def update_yaxes(self, *args, **kwargs):
        """Mock update_yaxes method."""
        return self


@patch('analytics.visualization.go', MagicMock())
@patch('analytics.visualization.make_subplots', MagicMock(return_value=MockFigure()))
@patch('analytics.visualization.px', MagicMock())
@patch('analytics.visualization.PLOTLY_AVAILABLE', True)
class TestInteractiveVisualizer(unittest.TestCase):
    """Test cases for InteractiveVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data (same as in TestPerformanceVisualizer)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, 100)
        cumulative_returns = (1 + returns).cumprod() - 1
        drawdowns = np.random.uniform(-0.3, 0, 100)
        drawdowns.sort()  # Sort to simulate realistic drawdowns

        self.returns = pd.Series(returns, index=dates)
        self.cumulative_returns = pd.Series(cumulative_returns, index=dates)
        self.drawdowns = pd.Series(drawdowns, index=dates)

        # Create asset allocation data
        self.asset_classes = ['Equities', 'Bonds', 'Real Estate', 'Commodities', 'Cash']
        self.allocations = [0.4, 0.3, 0.1, 0.1, 0.1]
        self.risk_contributions = [0.6, 0.2, 0.1, 0.08, 0.02]

        # Create correlation matrix
        np.random.seed(42)  # For reproducibility
        corr = np.random.uniform(-1, 1, (5, 5))
        # Make it symmetric
        corr = (corr + corr.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(corr, 1)
        self.correlation_matrix = pd.DataFrame(
            corr,
            index=self.asset_classes,
            columns=self.asset_classes
        )

        # Create monthly returns
        months = 12
        years = 3
        monthly_returns = np.random.normal(0.01, 0.03, months * years).reshape(years, months)
        self.monthly_returns = pd.DataFrame(
            monthly_returns,
            index=pd.date_range(start='2020-01-01', periods=years, freq='YS').year,
            columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )

        self.visualizer = InteractiveVisualizer()

    def test_create_dashboard(self, *args):
        """Test creating an interactive dashboard."""
        fig = self.visualizer.create_dashboard(
            self.cumulative_returns,
            self.drawdowns,
            self.returns,
            self.asset_classes,
            self.allocations,
            self.risk_contributions,
            self.correlation_matrix,
            self.monthly_returns
        )
        self.assertIsInstance(fig, MockFigure)


@patch('analytics.visualization.go', MagicMock())
@patch('analytics.visualization.make_subplots', MagicMock(return_value=MockFigure()))
@patch('analytics.visualization.px', MagicMock())
@patch('analytics.visualization.PLOTLY_AVAILABLE', True)
class TestComparativeVisualizer(unittest.TestCase):
    """Test cases for ComparativeVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for strategy comparison
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create returns for multiple strategies
        strategy_returns = {}
        strategy_cumulative_returns = {}
        for i, strategy in enumerate(['Strategy A', 'Strategy B', 'Strategy C']):
            returns = np.random.normal(0.001 * (i + 1), 0.02, 100)
            cumulative_returns = (1 + returns).cumprod() - 1
            strategy_returns[strategy] = pd.Series(returns, index=dates)
            strategy_cumulative_returns[strategy] = pd.Series(cumulative_returns, index=dates)
        
        self.strategy_returns = pd.DataFrame(strategy_returns)
        self.strategy_cumulative_returns = pd.DataFrame(strategy_cumulative_returns)
        
        # Create metrics for strategies
        self.metrics = pd.DataFrame({
            'Total Return': [0.15, 0.12, 0.18],
            'Sharpe Ratio': [1.2, 0.9, 1.5],
            'Max Drawdown': [-0.12, -0.10, -0.15],
            'Volatility': [0.10, 0.08, 0.12]
        }, index=['Strategy A', 'Strategy B', 'Strategy C'])
        
        # Create scenario analysis data
        self.scenarios = ['Baseline', 'Market Crash', 'Rising Rates', 'Inflation Shock']
        self.scenario_results = pd.DataFrame({
            'Strategy A': [0.10, -0.20, -0.05, 0.03],
            'Strategy B': [0.08, -0.15, -0.02, 0.01],
            'Strategy C': [0.12, -0.25, -0.08, 0.05]
        }, index=self.scenarios)
        
        self.visualizer = ComparativeVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_plot_strategy_comparison_static(self, *args):
        """Test plotting static strategy comparison."""
        fig = self.visualizer.plot_strategy_comparison(
            self.strategy_cumulative_returns,
            metrics=self.metrics,
            interactive=False
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_strategy_comparison_interactive(self, *args):
        """Test plotting interactive strategy comparison."""
        fig = self.visualizer.plot_strategy_comparison(
            self.strategy_cumulative_returns,
            metrics=self.metrics,
            interactive=True
        )
        self.assertIsInstance(fig, MockFigure)

    def test_plot_scenario_analysis_static(self, *args):
        """Test plotting static scenario analysis."""
        fig = self.visualizer.plot_scenario_analysis(
            self.scenario_results,
            baseline='Baseline',
            interactive=False
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_scenario_analysis_interactive(self, *args):
        """Test plotting interactive scenario analysis."""
        fig = self.visualizer.plot_scenario_analysis(
            self.scenario_results,
            baseline='Baseline',
            interactive=True
        )
        self.assertIsInstance(fig, MockFigure)


if __name__ == '__main__':
    unittest.main()