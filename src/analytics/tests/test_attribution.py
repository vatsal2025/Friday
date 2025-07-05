"""Tests for the analytics attribution module."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import attribution module
from analytics.attribution import (
    AttributionAnalyzer,
    BrinsionAttributionAnalyzer,
    FactorAttributionAnalyzer,
    RiskAttributionAnalyzer
)


class TestAttributionAnalyzer(unittest.TestCase):
    """Test cases for AttributionAnalyzer base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AttributionAnalyzer()

    def test_init(self):
        """Test initialization."""
        self.assertIsNotNone(self.analyzer)

    def test_format_result(self):
        """Test formatting attribution results."""
        # Create sample data
        data = {
            'Component A': 0.0123,
            'Component B': -0.0045,
            'Component C': 0.0567
        }
        result = self.analyzer._format_result(data)
        
        # Check that values are formatted as percentages
        self.assertEqual(result['Component A'], '1.23%')
        self.assertEqual(result['Component B'], '-0.45%')
        self.assertEqual(result['Component C'], '5.67%')

    def test_validate_data(self):
        """Test data validation."""
        # Valid data
        valid_data = pd.Series([1, 2, 3])
        self.assertTrue(self.analyzer._validate_data(valid_data))
        
        # Invalid data (None)
        with self.assertRaises(ValueError):
            self.analyzer._validate_data(None)
        
        # Invalid data (empty)
        with self.assertRaises(ValueError):
            self.analyzer._validate_data(pd.Series([]))


class TestBrinsionAttributionAnalyzer(unittest.TestCase):
    """Test cases for BrinsionAttributionAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BrinsionAttributionAnalyzer()
        
        # Create sample data
        # Portfolio weights and returns by sector
        self.portfolio_weights = pd.Series({
            'Technology': 0.30,
            'Healthcare': 0.20,
            'Financials': 0.15,
            'Consumer': 0.15,
            'Energy': 0.10,
            'Other': 0.10
        })
        
        self.portfolio_returns = pd.Series({
            'Technology': 0.08,
            'Healthcare': 0.05,
            'Financials': 0.03,
            'Consumer': 0.04,
            'Energy': -0.02,
            'Other': 0.01
        })
        
        # Benchmark weights and returns by sector
        self.benchmark_weights = pd.Series({
            'Technology': 0.25,
            'Healthcare': 0.15,
            'Financials': 0.20,
            'Consumer': 0.15,
            'Energy': 0.15,
            'Other': 0.10
        })
        
        self.benchmark_returns = pd.Series({
            'Technology': 0.07,
            'Healthcare': 0.04,
            'Financials': 0.02,
            'Consumer': 0.03,
            'Energy': -0.01,
            'Other': 0.01
        })

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_analyze(self):
        """Test Brinson attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_weights,
            self.portfolio_returns,
            self.benchmark_weights,
            self.benchmark_returns
        )
        
        # Check that result contains expected components
        self.assertIn('allocation_effect', result)
        self.assertIn('selection_effect', result)
        self.assertIn('interaction_effect', result)
        self.assertIn('total_effect', result)
        
        # Check that effects sum to total effect
        total_effect = result['allocation_effect'] + result['selection_effect'] + result['interaction_effect']
        self.assertAlmostEqual(total_effect, result['total_effect'], places=10)
        
        # Check sector-level results
        self.assertIn('sector_effects', result)
        sector_effects = result['sector_effects']
        self.assertEqual(len(sector_effects), len(self.portfolio_weights))
        
        # Check that sector effects contain expected columns
        for sector in sector_effects:
            self.assertIn('allocation', sector_effects[sector])
            self.assertIn('selection', sector_effects[sector])
            self.assertIn('interaction', sector_effects[sector])
            self.assertIn('total', sector_effects[sector])

    def test_plot_attribution_static(self):
        """Test plotting static attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_weights,
            self.portfolio_returns,
            self.benchmark_weights,
            self.benchmark_returns
        )
        
        fig = self.analyzer.plot_attribution(result, interactive=False)
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.make_subplots', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_attribution_interactive(self):
        """Test plotting interactive attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_weights,
            self.portfolio_returns,
            self.benchmark_weights,
            self.benchmark_returns
        )
        
        fig = self.analyzer.plot_attribution(result, interactive=True)
        self.assertIsNotNone(fig)

    def test_plot_sector_attribution_static(self):
        """Test plotting static sector attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_weights,
            self.portfolio_returns,
            self.benchmark_weights,
            self.benchmark_returns
        )
        
        fig = self.analyzer.plot_sector_attribution(result, interactive=False)
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.make_subplots', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_sector_attribution_interactive(self):
        """Test plotting interactive sector attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_weights,
            self.portfolio_returns,
            self.benchmark_weights,
            self.benchmark_returns
        )
        
        fig = self.analyzer.plot_sector_attribution(result, interactive=True)
        self.assertIsNotNone(fig)


class TestFactorAttributionAnalyzer(unittest.TestCase):
    """Test cases for FactorAttributionAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FactorAttributionAnalyzer()
        
        # Create sample data
        # Portfolio returns
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        
        # Factor returns
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        factor_returns_data = np.random.normal(0.0005, 0.01, (100, 5))
        self.factor_returns = pd.DataFrame(factor_returns_data, index=dates, columns=factors)
        
        # Factor exposures
        self.factor_exposures = pd.Series({
            'Market': 1.0,
            'Size': 0.2,
            'Value': 0.3,
            'Momentum': -0.1,
            'Quality': 0.5
        })

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_analyze(self):
        """Test factor attribution analysis."""
        result = self.analyzer.analyze(
            self.portfolio_returns,
            self.factor_returns,
            self.factor_exposures
        )
        
        # Check that result contains expected components
        self.assertIn('factor_contributions', result)
        self.assertIn('specific_return', result)
        self.assertIn('total_return', result)
        
        # Check that factor contributions and specific return sum to total return
        factor_contribution_sum = sum(result['factor_contributions'].values())
        total_contribution = factor_contribution_sum + result['specific_return']
        self.assertAlmostEqual(total_contribution, result['total_return'], places=10)
        
        # Check factor contributions
        factor_contributions = result['factor_contributions']
        self.assertEqual(len(factor_contributions), len(self.factor_exposures))
        for factor in self.factor_exposures.index:
            self.assertIn(factor, factor_contributions)

    def test_plot_factor_contribution_static(self):
        """Test plotting static factor contribution."""
        result = self.analyzer.analyze(
            self.portfolio_returns,
            self.factor_returns,
            self.factor_exposures
        )
        
        fig = self.analyzer.plot_factor_contribution(result, interactive=False)
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_factor_contribution_interactive(self):
        """Test plotting interactive factor contribution."""
        result = self.analyzer.analyze(
            self.portfolio_returns,
            self.factor_returns,
            self.factor_exposures
        )
        
        fig = self.analyzer.plot_factor_contribution(result, interactive=True)
        self.assertIsNotNone(fig)

    def test_plot_factor_contribution_over_time_static(self):
        """Test plotting static factor contribution over time."""
        result = self.analyzer.analyze_over_time(
            self.portfolio_returns,
            self.factor_returns,
            self.factor_exposures,
            window=30
        )
        
        fig = self.analyzer.plot_factor_contribution_over_time(result, interactive=False)
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_factor_contribution_over_time_interactive(self):
        """Test plotting interactive factor contribution over time."""
        result = self.analyzer.analyze_over_time(
            self.portfolio_returns,
            self.factor_returns,
            self.factor_exposures,
            window=30
        )
        
        fig = self.analyzer.plot_factor_contribution_over_time(result, interactive=True)
        self.assertIsNotNone(fig)


class TestRiskAttributionAnalyzer(unittest.TestCase):
    """Test cases for RiskAttributionAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RiskAttributionAnalyzer()
        
        # Create sample data
        # Asset weights
        self.asset_weights = pd.Series({
            'Asset A': 0.25,
            'Asset B': 0.20,
            'Asset C': 0.15,
            'Asset D': 0.15,
            'Asset E': 0.10,
            'Asset F': 0.15
        })
        
        # Asset returns
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        assets = self.asset_weights.index
        returns_data = np.random.normal(0.001, 0.02, (100, len(assets)))
        self.asset_returns = pd.DataFrame(returns_data, index=dates, columns=assets)
        
        # Factor exposures
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        exposures_data = np.random.normal(0, 1, (len(assets), len(factors)))
        self.factor_exposures = pd.DataFrame(exposures_data, index=assets, columns=factors)
        
        # Factor covariance matrix
        factor_cov_data = np.random.uniform(-0.001, 0.001, (len(factors), len(factors)))
        factor_cov_data = factor_cov_data.dot(factor_cov_data.T)  # Make positive semi-definite
        np.fill_diagonal(factor_cov_data, np.random.uniform(0.001, 0.01, len(factors)))  # Set diagonal to positive values
        self.factor_covariance = pd.DataFrame(factor_cov_data, index=factors, columns=factors)

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_analyze_asset_risk(self):
        """Test analyzing asset risk contribution."""
        result = self.analyzer.analyze_asset_risk(
            self.asset_weights,
            self.asset_returns
        )
        
        # Check that result contains expected components
        self.assertIn('asset_risk_contribution', result)
        self.assertIn('portfolio_volatility', result)
        self.assertIn('marginal_contribution', result)
        
        # Check that asset risk contributions sum to portfolio volatility
        risk_contribution_sum = sum(result['asset_risk_contribution'].values())
        self.assertAlmostEqual(risk_contribution_sum, result['portfolio_volatility'], places=10)
        
        # Check asset risk contributions
        asset_risk_contribution = result['asset_risk_contribution']
        self.assertEqual(len(asset_risk_contribution), len(self.asset_weights))
        for asset in self.asset_weights.index:
            self.assertIn(asset, asset_risk_contribution)

    def test_analyze_factor_risk(self):
        """Test analyzing factor risk contribution."""
        result = self.analyzer.analyze_factor_risk(
            self.asset_weights,
            self.factor_exposures,
            self.factor_covariance
        )
        
        # Check that result contains expected components
        self.assertIn('factor_risk_contribution', result)
        self.assertIn('portfolio_volatility', result)
        self.assertIn('marginal_contribution', result)
        
        # Check that factor risk contributions sum approximately to portfolio volatility
        # (allowing for specific risk)
        factor_risk_contribution_sum = sum(result['factor_risk_contribution'].values())
        self.assertLessEqual(factor_risk_contribution_sum, result['portfolio_volatility'] * 1.1)  # Allow 10% margin for specific risk
        
        # Check factor risk contributions
        factor_risk_contribution = result['factor_risk_contribution']
        self.assertEqual(len(factor_risk_contribution), len(self.factor_covariance))
        for factor in self.factor_covariance.index:
            self.assertIn(factor, factor_risk_contribution)

    def test_plot_risk_contribution_static(self):
        """Test plotting static risk contribution."""
        result = self.analyzer.analyze_asset_risk(
            self.asset_weights,
            self.asset_returns
        )
        
        fig = self.analyzer.plot_risk_contribution(result, interactive=False)
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_risk_contribution_interactive(self):
        """Test plotting interactive risk contribution."""
        result = self.analyzer.analyze_asset_risk(
            self.asset_weights,
            self.asset_returns
        )
        
        fig = self.analyzer.plot_risk_contribution(result, interactive=True)
        self.assertIsNotNone(fig)

    def test_plot_risk_contribution_comparison_static(self):
        """Test plotting static risk contribution comparison."""
        asset_result = self.analyzer.analyze_asset_risk(
            self.asset_weights,
            self.asset_returns
        )
        
        factor_result = self.analyzer.analyze_factor_risk(
            self.asset_weights,
            self.factor_exposures,
            self.factor_covariance
        )
        
        fig = self.analyzer.plot_risk_contribution_comparison(
            asset_result['asset_risk_contribution'],
            factor_result['factor_risk_contribution'],
            interactive=False
        )
        self.assertIsInstance(fig, plt.Figure)

    @patch('analytics.attribution.go', MagicMock())
    @patch('analytics.attribution.make_subplots', MagicMock())
    @patch('analytics.attribution.PLOTLY_AVAILABLE', True)
    def test_plot_risk_contribution_comparison_interactive(self):
        """Test plotting interactive risk contribution comparison."""
        asset_result = self.analyzer.analyze_asset_risk(
            self.asset_weights,
            self.asset_returns
        )
        
        factor_result = self.analyzer.analyze_factor_risk(
            self.asset_weights,
            self.factor_exposures,
            self.factor_covariance
        )
        
        fig = self.analyzer.plot_risk_contribution_comparison(
            asset_result['asset_risk_contribution'],
            factor_result['factor_risk_contribution'],
            interactive=True
        )
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()