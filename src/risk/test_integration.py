#!/usr/bin/env python
"""
Comprehensive integration test for the risk management system.

This test verifies that all components of the risk management system work together
correctly, including the factory pattern, RiskMetricsCalculator, and all risk
management components.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal

from risk_management_factory import RiskManagementFactory
from production_config import RiskManagementProductionConfig, create_default_production_config
from risk_metrics_calculator import RiskMetricsCalculator
from portfolio_risk_manager import PortfolioRiskManager
from advanced_risk_manager import AdvancedRiskManager
from position_sizer import PositionSizer
from circuit_breaker import MarketWideCircuitBreaker, AccountCircuitBreaker
from stop_loss_manager import StopLossManager
from circuit_breaker import CircuitBreakerManager


class TestRiskManagementIntegration(unittest.TestCase):
    """Test the integration of all risk management components."""

    def setUp(self):
        """Set up the test environment."""
        # Create a production configuration
        self.config = create_default_production_config()
        self.config.var_confidence_level = 0.95
        self.config.max_portfolio_var_percent = 0.02
        self.config.max_drawdown_percent = 0.15
        self.config.max_sector_exposure = 0.25  # Used as max_sector_allocation in PortfolioRiskManager
        self.config.max_asset_exposure = 0.10  # Used as max_position_size in PortfolioRiskManager
        self.config.risk_per_trade = 0.01
        self.config.initial_capital = 100000.0

        # Create a factory with the configuration
        self.factory = RiskManagementFactory(self.config)

        # Create test data
        self.test_positions = {
            "AAPL": {"market_value": 10000, "sector": "Technology"},
            "MSFT": {"market_value": 8000, "sector": "Technology"},
            "JPM": {"market_value": 7000, "sector": "Financials"}
        }
        self.portfolio_value = 25000

    def test_factory_creates_all_components(self):
        """Test that the factory creates all components correctly."""
        # Create components using the factory
        risk_metrics_calculator = self.factory.create_risk_metrics_calculator()
        position_sizer = self.factory.create_position_sizer()
        stop_loss_manager = self.factory.create_stop_loss_manager()
        portfolio_risk_manager = self.factory.create_portfolio_risk_manager()
        circuit_breaker_manager = self.factory.create_circuit_breaker_manager()
        advanced_risk_manager = self.factory.create_advanced_risk_manager()

        # Verify that all components are created with the correct type
        self.assertIsInstance(risk_metrics_calculator, RiskMetricsCalculator)
        self.assertIsInstance(position_sizer, PositionSizer)
        self.assertIsInstance(stop_loss_manager, StopLossManager)
        self.assertIsInstance(portfolio_risk_manager, PortfolioRiskManager)
        self.assertIsInstance(circuit_breaker_manager, CircuitBreakerManager)
        self.assertIsInstance(advanced_risk_manager, AdvancedRiskManager)

        # Verify that the components are configured correctly
        self.assertEqual(risk_metrics_calculator.confidence_level, self.config.var_confidence_level)
        self.assertEqual(position_sizer.risk_per_trade, self.config.risk_per_trade)
        self.assertEqual(portfolio_risk_manager.max_portfolio_var_percent, self.config.max_portfolio_var_percent)

    def test_advanced_risk_manager_integration(self):
        """Test that AdvancedRiskManager integrates all components correctly."""
        # Create an advanced risk manager using the factory
        advanced_manager = self.factory.create_advanced_risk_manager()

        # Verify that it has all the required components
        self.assertIsNotNone(advanced_manager.position_sizer)
        self.assertIsNotNone(advanced_manager.stop_loss_manager)
        self.assertIsNotNone(advanced_manager.portfolio_risk_manager)
        self.assertIsNotNone(advanced_manager.circuit_breaker_manager)
        self.assertIsNotNone(advanced_manager.risk_metrics_calculator)

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
        self.assertIn('current_drawdown', metrics)
        self.assertIn('sector_exposure', metrics)
        self.assertIn('asset_exposure', metrics)
        self.assertIn('timestamp', metrics)

    def test_position_sizing(self):
        """Test that position sizing works correctly with the integrated system."""
        # Create an advanced risk manager using the factory
        advanced_manager = self.factory.create_advanced_risk_manager()

        # Calculate position size
        symbol = "AAPL"
        entry_price = 150.0
        stop_loss_price = 145.0

        # Calculate position size using the position sizer
        position_size = advanced_manager.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=Decimal(str(entry_price)),
            stop_loss_price=Decimal(str(stop_loss_price))
        )

        # Verify that the position size is calculated correctly
        expected_risk_amount = self.config.initial_capital * self.config.risk_per_trade
        expected_risk_per_share = entry_price - stop_loss_price
        expected_position_size = expected_risk_amount / expected_risk_per_share

        self.assertAlmostEqual(float(position_size), float(expected_position_size), delta=0.01)

    def test_stop_loss_management(self):
        """Test that stop loss management works correctly with the integrated system."""
        # Create an advanced risk manager using the factory
        advanced_manager = self.factory.create_advanced_risk_manager()

        # Set up a trade
        symbol = "AAPL"
        entry_price = Decimal("150.0")
        position_size = Decimal("10")
        stop_type = "fixed"
        stop_percent = Decimal("0.03")  # 3% stop loss

        # Set the stop loss
        advanced_manager.stop_loss_manager.set_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            stop_type=stop_type,
            stop_percent=stop_percent
        )

        # Get the stop loss price
        stop_loss_price = advanced_manager.stop_loss_manager.get_stop_loss_price(symbol)

        # Verify that the stop loss price is calculated correctly
        expected_stop_loss_price = entry_price * (1 - stop_percent)
        self.assertEqual(stop_loss_price, expected_stop_loss_price)

    def test_risk_metrics_calculation(self):
        """Test that risk metrics calculation works correctly with the integrated system."""
        # Create a risk metrics calculator using the factory
        calculator = self.factory.create_risk_metrics_calculator()

        # Add some historical returns
        now = datetime.now()
        for i in range(30):
            timestamp = now - timedelta(days=i)
            daily_return = np.random.normal(0.0005, 0.01)
            calculator.add_return_observation(daily_return, timestamp)

        # Calculate all metrics
        metrics = calculator.calculate_all_metrics()

        # Verify that the metrics include the expected fields
        self.assertIn('var', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('annualized_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

    def test_circuit_breaker_integration(self):
        """Test that circuit breakers work correctly with the integrated system."""
        # Create an advanced risk manager using the factory
        advanced_manager = self.factory.create_advanced_risk_manager()

        # Add a circuit breaker
        breaker_id = "test_breaker"
        metric_name = "var"
        threshold = 0.05  # 5% VaR threshold
        action = "alert"

        # Create a market-wide circuit breaker
        market_cb = MarketWideCircuitBreaker(
            market=breaker_id,
            level_1_percent=threshold,
            level_2_percent=threshold * 1.5,
            level_3_percent=threshold * 2.0,
            level_1_duration_minutes=15,
            level_2_duration_minutes=30,
            level_3_duration_minutes=60
        )
        advanced_manager.circuit_breaker_manager.add_circuit_breaker(market_cb)

        # Update the portfolio with test data to trigger the circuit breaker
        now = datetime.now()
        advanced_manager.portfolio_risk_manager.update_portfolio(self.test_positions, self.portfolio_value, now)

        # Add some volatile historical data to increase VaR
        for i in range(1, 31):
            timestamp = now - timedelta(days=i)
            # Add some large negative returns to increase VaR
            if i % 5 == 0:
                value = self.portfolio_value * 0.95  # 5% drop
            else:
                value = self.portfolio_value * (1 + np.random.normal(0.0005, 0.01))
            advanced_manager.portfolio_risk_manager.update_portfolio(self.test_positions, value, timestamp)

        # Check if the circuit breaker is triggered
        metrics = advanced_manager.get_risk_metrics()
        triggered_breakers = advanced_manager.circuit_breaker_manager.check_circuit_breakers(metrics)

        # If VaR is above the threshold, the circuit breaker should be triggered
        if metrics['var'] > threshold:
            self.assertIn(breaker_id, triggered_breakers)
        else:
            self.assertNotIn(breaker_id, triggered_breakers)


if __name__ == '__main__':
    unittest.main()
