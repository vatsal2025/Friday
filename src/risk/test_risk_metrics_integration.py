import unittest
from datetime import datetime, timedelta
import numpy as np
from risk_management_factory import RiskManagementFactory
from production_config import RiskManagementProductionConfig
from risk_metrics_calculator import RiskMetricsCalculator
from portfolio_risk_manager import PortfolioRiskManager
from advanced_risk_manager import AdvancedRiskManager

class TestRiskMetricsIntegration(unittest.TestCase):
    """Test the integration of RiskMetricsCalculator with the risk management components."""

    def setUp(self):
        """Set up the test environment."""
        # Create a production configuration
        self.config = RiskManagementProductionConfig(
            var_confidence_level=0.95,
            max_portfolio_var_percent=0.02,
            max_drawdown_percent=0.15,
            max_sector_allocation=0.25,
            max_position_size=0.10,
            max_history_size=252
        )

        # Create a factory with the configuration
        self.factory = RiskManagementFactory(self.config)

        # Create test data
        self.test_positions = {
            "AAPL": {"market_value": 10000, "sector": "Technology"},
            "MSFT": {"market_value": 8000, "sector": "Technology"},
            "JPM": {"market_value": 7000, "sector": "Financials"}
        }
        self.portfolio_value = 25000

    def test_risk_metrics_calculator_creation(self):
        """Test that the factory creates a RiskMetricsCalculator with the correct configuration."""
        calculator = self.factory.create_risk_metrics_calculator()
        self.assertIsInstance(calculator, RiskMetricsCalculator)
        self.assertEqual(calculator.confidence_level, self.config.var_confidence_level)

    def test_portfolio_risk_manager_integration(self):
        """Test that PortfolioRiskManager correctly integrates with RiskMetricsCalculator."""
        # Create a risk metrics calculator
        calculator = self.factory.create_risk_metrics_calculator()

        # Create a portfolio risk manager with the calculator
        portfolio_manager = PortfolioRiskManager(
            max_portfolio_var_percent=self.config.max_portfolio_var_percent,
            max_drawdown_percent=self.config.max_drawdown_percent,
            max_sector_allocation=self.config.max_sector_allocation,
            max_position_size=self.config.max_position_size,
            max_history_size=self.config.max_history_size,
            risk_metrics_calculator=calculator
        )

        # Update the portfolio with test data
        now = datetime.now()
        portfolio_manager.update_portfolio(self.test_positions, self.portfolio_value, now)

        # Add some historical data
        for i in range(1, 31):
            timestamp = now - timedelta(days=i)
            value = self.portfolio_value * (1 + np.random.normal(0.0005, 0.01))
            portfolio_manager.update_portfolio(self.test_positions, value, timestamp)

        # Get risk metrics
        metrics = portfolio_manager.calculate_risk_metrics()

        # Verify that the metrics include both portfolio-specific and calculator metrics
        self.assertIn('var', metrics)
        self.assertIn('current_drawdown', metrics)
        self.assertIn('sector_allocations', metrics)
        self.assertIn('position_sizes', metrics)
        self.assertIn('timestamp', metrics)

        # If we have enough data, we should also have these metrics
        self.assertIn('annualized_return', metrics)
        self.assertIn('annualized_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)

    def test_advanced_risk_manager_integration(self):
        """Test that AdvancedRiskManager correctly integrates with RiskMetricsCalculator."""
        # Create an advanced risk manager using the factory
        advanced_manager = self.factory.create_advanced_risk_manager()

        # Verify that it has a risk metrics calculator
        self.assertIsInstance(advanced_manager.risk_metrics_calculator, RiskMetricsCalculator)

        # Update the portfolio with test data
        now = datetime.now()
        advanced_manager.portfolio_risk_manager.update_portfolio(self.test_positions, self.portfolio_value, now)

        # Add some historical data
        for i in range(1, 31):
            timestamp = now - timedelta(days=i)
            value = self.portfolio_value * (1 + np.random.normal(0.0005, 0.01))
            advanced_manager.portfolio_risk_manager.update_portfolio(self.test_positions, value, timestamp)

        # Get risk metrics
        metrics = advanced_manager.get_risk_metrics()

        # Verify that the metrics include the expected fields
        self.assertIn('var', metrics)
        self.assertIn('timestamp', metrics)

        # If we have enough data, we should also have these metrics
        self.assertIn('annualized_return', metrics)
        self.assertIn('annualized_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)

if __name__ == '__main__':
    unittest.main()
