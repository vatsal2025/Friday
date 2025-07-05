import unittest
import numpy as np
from datetime import datetime, timedelta
from .risk_metrics_calculator import RiskMetricsCalculator

class TestRiskMetricsCalculator(unittest.TestCase):
    """Test cases for the RiskMetricsCalculator class."""

    def setUp(self):
        self.calculator = RiskMetricsCalculator(confidence_level=0.95)

        # Add some historical returns for testing
        base_date = datetime(2023, 1, 1)
        for i in range(100):
            date = base_date + timedelta(days=i)
            # Generate some random returns between -0.05 and 0.05
            return_value = (np.random.random() - 0.5) * 0.1
            self.calculator.add_return_observation(return_value, date)

        # Add some historical portfolio values for testing
        initial_value = 1000000.0
        current_value = initial_value
        for i in range(100):
            date = base_date + timedelta(days=i)
            # Apply the return to the portfolio value
            return_value = self.calculator.historical_returns[i][1]
            current_value *= (1 + return_value)
            self.calculator.add_portfolio_value_observation(current_value, date)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        portfolio_value = 1000000.0
        var_amount, var_percent = self.calculator.calculate_var(portfolio_value)

        # VaR should be positive
        self.assertGreater(var_amount, 0)
        self.assertGreater(var_percent, 0)

        # VaR percent should be less than 1 (100%)
        self.assertLess(var_percent, 1)

        # VaR amount should be approximately equal to portfolio_value * var_percent
        self.assertAlmostEqual(var_amount, portfolio_value * var_percent)

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        current_drawdown, max_drawdown = self.calculator.calculate_drawdown()

        # Drawdowns should be between 0 and 1
        self.assertGreaterEqual(current_drawdown, 0)
        self.assertLessEqual(current_drawdown, 1)
        self.assertGreaterEqual(max_drawdown, 0)
        self.assertLessEqual(max_drawdown, 1)

        # Max drawdown should be at least as large as current drawdown
        self.assertGreaterEqual(max_drawdown, current_drawdown)

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        volatility = self.calculator.calculate_volatility()

        # Volatility should be positive
        self.assertGreater(volatility, 0)

        # Test non-annualized volatility
        daily_volatility = self.calculator.calculate_volatility(annualize=False)

        # Annualized volatility should be approximately daily_volatility * sqrt(252)
        self.assertAlmostEqual(volatility, daily_volatility * np.sqrt(252))

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Test with positive return
        sharpe_ratio = self.calculator.calculate_sharpe_ratio(0.10)  # 10% annualized return

        # Sharpe ratio should be a float
        self.assertIsInstance(sharpe_ratio, float)

        # Test with zero volatility (should return 0)
        original_volatility = self.calculator.calculate_volatility
        self.calculator.calculate_volatility = lambda annualize=True: 0.0
        zero_vol_sharpe = self.calculator.calculate_sharpe_ratio(0.10)
        self.assertEqual(zero_vol_sharpe, 0.0)

        # Restore original method
        self.calculator.calculate_volatility = original_volatility

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        # Test with positive return
        sortino_ratio = self.calculator.calculate_sortino_ratio(0.10)  # 10% annualized return

        # Sortino ratio should be a float
        self.assertIsInstance(sortino_ratio, float)

        # Test with no negative returns
        original_returns = self.calculator.historical_returns
        self.calculator.historical_returns = [(datetime.now(), 0.01) for _ in range(10)]  # All positive returns
        inf_sortino = self.calculator.calculate_sortino_ratio(0.10)
        self.assertEqual(inf_sortino, float('inf'))

        # Restore original returns
        self.calculator.historical_returns = original_returns

    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation."""
        # Test with positive return
        calmar_ratio = self.calculator.calculate_calmar_ratio(0.10)  # 10% annualized return

        # Calmar ratio should be a float
        self.assertIsInstance(calmar_ratio, float)

        # Test with zero max drawdown
        original_drawdown = self.calculator.calculate_drawdown
        self.calculator.calculate_drawdown = lambda: (0.0, 0.0)
        inf_calmar = self.calculator.calculate_calmar_ratio(0.10)
        self.assertEqual(inf_calmar, float('inf'))

        # Test with zero max drawdown and negative return
        zero_calmar = self.calculator.calculate_calmar_ratio(-0.10)
        self.assertEqual(zero_calmar, 0.0)

        # Restore original method
        self.calculator.calculate_drawdown = original_drawdown

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        portfolio_value = 1000000.0
        metrics = self.calculator.calculate_all_metrics(portfolio_value)

        # Check that all expected metrics are present
        expected_metrics = [
            "timestamp", "var", "var_percent", "current_drawdown", "max_drawdown",
            "annualized_volatility", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio"
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)

        # Test with provided annualized return
        metrics_with_return = self.calculator.calculate_all_metrics(
            portfolio_value=portfolio_value,
            annualized_return=0.15  # 15% annualized return
        )

        self.assertEqual(metrics_with_return["annualized_return"], 0.15)

if __name__ == "__main__":
    unittest.main()
