import unittest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import portfolio components
from portfolio.portfolio_manager import PortfolioManager
from portfolio.performance_calculator import PerformanceCalculator
from portfolio.tax_manager import TaxManager, TaxLotMethod
from portfolio.allocation_manager import AllocationManager, RebalanceMethod
from portfolio.portfolio_factory import PortfolioFactory

# Try to import risk components
try:
    from risk.risk_management_factory import RiskManagementFactory
    from risk.advanced_risk_manager import AdvancedRiskManager
    RISK_MODULE_AVAILABLE = True
except ImportError:
    RISK_MODULE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPortfolioIntegration(unittest.TestCase):
    """Integration tests for the portfolio management system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a portfolio factory with test configuration
        self.config = {
            "portfolio_manager": {
                "portfolio_id": "test-portfolio",
                "initial_cash": 100000.0
            },
            "performance_calculator": {
                "benchmark_symbol": "SPY"
            },
            "tax_manager": {
                "default_method": "FIFO"
            },
            "allocation_manager": {
                "rebalance_method": "THRESHOLD",
                "default_threshold": 5.0,
                "rebalance_frequency_days": 90,
                "allocation_targets": [
                    {"name": "AAPL", "target_percentage": 25.0, "category": "stocks"},
                    {"name": "MSFT", "target_percentage": 25.0, "category": "stocks"},
                    {"name": "GOOGL", "target_percentage": 20.0, "category": "stocks"},
                    {"name": "BND", "target_percentage": 30.0, "category": "bonds"}
                ]
            }
        }

        if RISK_MODULE_AVAILABLE:
            self.config["risk_manager"] = {
                "max_portfolio_var_percent": 2.0,
                "max_drawdown_percent": 15.0,
                "risk_free_rate": 0.02,
                "confidence_level": 0.95
            }

        self.factory = PortfolioFactory(self.config)

        # Create individual components for testing
        self.portfolio_manager = self.factory.create_portfolio_manager()
        self.performance_calculator = self.factory.create_performance_calculator()
        self.tax_manager = self.factory.create_tax_manager()
        self.allocation_manager = self.factory.create_allocation_manager()

        if RISK_MODULE_AVAILABLE:
            self.risk_manager = self.factory.create_risk_manager()
        else:
            self.risk_manager = None

    def test_factory_creation(self):
        """Test that the factory correctly creates all components."""
        # Test individual component creation
        self.assertIsInstance(self.portfolio_manager, PortfolioManager)
        self.assertEqual(self.portfolio_manager.portfolio_id, "test-portfolio")
        self.assertEqual(self.portfolio_manager.cash, 100000.0)

        self.assertIsInstance(self.performance_calculator, PerformanceCalculator)
        self.assertEqual(self.performance_calculator.benchmark_symbol, "SPY")

        self.assertIsInstance(self.tax_manager, TaxManager)
        self.assertEqual(self.tax_manager.default_method, TaxLotMethod.FIFO)

        self.assertIsInstance(self.allocation_manager, AllocationManager)
        self.assertEqual(self.allocation_manager.rebalance_method, RebalanceMethod.THRESHOLD)
        self.assertEqual(self.allocation_manager.default_threshold, 5.0)

        # Test complete system creation
        system = self.factory.create_complete_portfolio_system()
        self.assertIsInstance(system["portfolio_manager"], PortfolioManager)
        self.assertIsInstance(system["performance_calculator"], PerformanceCalculator)
        self.assertIsInstance(system["tax_manager"], TaxManager)
        self.assertIsInstance(system["allocation_manager"], AllocationManager)

        if RISK_MODULE_AVAILABLE:
            self.assertIsInstance(system["risk_manager"], AdvancedRiskManager)

    def test_portfolio_manager_basic_operations(self):
        """Test basic portfolio manager operations."""
        # Initial state
        self.assertEqual(self.portfolio_manager.cash, 100000.0)
        self.assertEqual(len(self.portfolio_manager.positions), 0)

        # Execute trades
        self.portfolio_manager.execute_trade("AAPL", 50, 150.0, datetime.now(), asset_class="equities", sector="Technology")
        self.portfolio_manager.execute_trade("MSFT", 40, 250.0, datetime.now(), asset_class="equities", sector="Technology")

        # Check positions
        self.assertEqual(len(self.portfolio_manager.positions), 2)
        self.assertEqual(self.portfolio_manager.positions["AAPL"]["quantity"], 50)
        self.assertEqual(self.portfolio_manager.positions["MSFT"]["quantity"], 40)
        self.assertEqual(self.portfolio_manager.positions["AAPL"]["asset_class"], "equities")
        self.assertEqual(self.portfolio_manager.positions["MSFT"]["asset_class"], "equities")
        self.assertEqual(self.portfolio_manager.positions["AAPL"]["sector"], "Technology")
        self.assertEqual(self.portfolio_manager.positions["MSFT"]["sector"], "Technology")

        # Check cash balance (100000 - 50*150 - 40*250 = 100000 - 7500 - 10000 = 82500)
        self.assertEqual(self.portfolio_manager.cash, 82500.0)

        # Update prices
        prices = {"AAPL": 160.0, "MSFT": 260.0}
        self.portfolio_manager.update_prices(prices)

        # Check portfolio value (82500 + 50*160 + 40*260 = 82500 + 8000 + 10400 = 100900)
        self.assertEqual(self.portfolio_manager.get_portfolio_value(), 100900.0)

        # Test transaction history
        transactions = self.portfolio_manager.get_transaction_history()
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["symbol"], "AAPL")
        self.assertEqual(transactions[1]["symbol"], "MSFT")

    def test_performance_calculator(self):
        """Test performance calculator functionality."""
        # Add portfolio returns
        dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
        returns = [0.01, -0.005, 0.02, 0.01, -0.01, 0.005, 0.015, -0.008, 0.01, 0.005]
        benchmark_returns = [0.008, -0.003, 0.015, 0.012, -0.005, 0.004, 0.01, -0.006, 0.008, 0.006]

        for i, date in enumerate(dates):
            self.performance_calculator.add_portfolio_return_observation(returns[i], date)
            self.performance_calculator.add_benchmark_return_observation(benchmark_returns[i], date)

        # Add portfolio values
        values = [100000 * (1 + sum(returns[:i+1])) for i in range(len(returns))]
        for i, date in enumerate(dates):
            self.performance_calculator.add_portfolio_value_observation(values[i], date)

        # Test performance metrics
        metrics = self.performance_calculator.calculate_performance_metrics()

        # Basic checks
        self.assertIn("total_return", metrics)
        self.assertIn("annualized_return", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("benchmark_comparison", metrics)

        # Check benchmark comparison
        benchmark_comp = metrics["benchmark_comparison"]
        self.assertIn("alpha", benchmark_comp)
        self.assertIn("beta", benchmark_comp)
        self.assertIn("tracking_error", benchmark_comp)
        self.assertIn("information_ratio", benchmark_comp)

    def test_tax_manager(self):
        """Test tax manager functionality."""
        # Add tax lots
        lot1 = self.tax_manager.add_tax_lot("AAPL", 20, 150.0, datetime.now() - timedelta(days=400))
        lot2 = self.tax_manager.add_tax_lot("AAPL", 30, 160.0, datetime.now() - timedelta(days=200))
        lot3 = self.tax_manager.add_tax_lot("MSFT", 15, 240.0, datetime.now() - timedelta(days=100))

        # Check tax lots
        lots = self.tax_manager.get_tax_lots()
        self.assertEqual(len(lots), 2)  # Two symbols
        self.assertEqual(len(lots["AAPL"]), 2)  # Two AAPL lots
        self.assertEqual(len(lots["MSFT"]), 1)  # One MSFT lot

        # Sell some AAPL using FIFO
        sale = self.tax_manager.sell_tax_lots("AAPL", 25, 170.0)

        # Check sale results
        self.assertEqual(sale["quantity_sold"], 25)
        self.assertEqual(sale["method"], "First-In-First-Out")
        self.assertTrue(sale["realized_gain"] > 0)  # Should have a gain

        # Check remaining lots
        lots = self.tax_manager.get_tax_lots()
        self.assertEqual(len(lots["AAPL"]), 2)  # Still two lots but first one reduced
        self.assertEqual(lots["AAPL"][0].quantity, 20 - (25 - 20))  # 20 - 5 = 15

        # Check realized gains
        gains = self.tax_manager.get_realized_gains()
        self.assertEqual(len(gains), 1)  # One sale
        self.assertEqual(gains[0]["symbol"], "AAPL")

    def test_allocation_manager(self):
        """Test allocation manager functionality."""
        # Check allocation targets
        targets = self.allocation_manager.get_allocation_targets()
        self.assertEqual(len(targets), 2)  # Two categories: stocks and bonds
        self.assertEqual(len(targets["stocks"]), 3)  # Three stock targets
        self.assertEqual(len(targets["bonds"]), 1)  # One bond target

        # Update current allocations
        portfolio_values = {
            "AAPL": 30000.0,  # 30%
            "MSFT": 20000.0,  # 20%
            "GOOGL": 15000.0,  # 15%
            "BND": 25000.0,   # 25%
            "CASH": 10000.0   # 10%
        }

        categories = {
            "AAPL": "stocks",
            "MSFT": "stocks",
            "GOOGL": "stocks",
            "BND": "bonds",
            "CASH": "cash"
        }

        self.allocation_manager.update_allocation_from_portfolio(portfolio_values, categories)

        # Check current allocations
        allocations = self.allocation_manager.get_current_allocations()
        self.assertIn("stocks", allocations)
        self.assertIn("bonds", allocations)
        self.assertIn("cash", allocations)

        # Check if rebalance is needed
        rebalance_check = self.allocation_manager.check_rebalance_needed()

        # Generate rebalance plan
        plan = self.allocation_manager.generate_rebalance_plan(portfolio_values, categories)

        # Check plan
        if plan["rebalance_needed"]:
            self.assertIn("trades", plan)
            self.assertTrue(len(plan["trades"]) > 0)

    def test_portfolio_with_risk_integration(self):
        """Test integration between portfolio and risk management."""
        if not RISK_MODULE_AVAILABLE:
            self.skipTest("Risk management module not available")

        # Create a portfolio with risk manager
        portfolio = self.factory.create_portfolio_manager(risk_manager=self.risk_manager)

        # Execute some trades
        portfolio.execute_trade("AAPL", 50, 150.0, datetime.now())
        portfolio.execute_trade("MSFT", 40, 250.0, datetime.now())

        # Update prices
        prices = {"AAPL": 160.0, "MSFT": 260.0}
        portfolio.update_prices(prices)

        # Add historical data for risk calculations
        dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        aapl_prices = [150.0 * (1 + 0.0003 * (i - np.sin(i/10))) for i in range(100)]
        msft_prices = [250.0 * (1 + 0.0004 * (i - np.sin(i/12))) for i in range(100)]

        for i, date in enumerate(dates):
            historical_prices = {"AAPL": aapl_prices[i], "MSFT": msft_prices[i]}
            portfolio.add_historical_prices(historical_prices, date)

        # Get risk metrics
        risk_metrics = portfolio.get_risk_metrics()

        # Check that risk metrics are available
        self.assertIsNotNone(risk_metrics)
        self.assertIn("portfolio_var", risk_metrics)
        self.assertIn("portfolio_volatility", risk_metrics)
        
    def test_historical_prices_workflow(self):
        """Test the entire workflow from adding historical prices to calculating risk metrics."""
        # Create a portfolio with risk manager
        portfolio = self.factory.create_portfolio_manager(risk_manager=True)
        
        # Execute trades
        portfolio.execute_trade("AAPL", 10, 150.0)  # $1,500 invested
        portfolio.execute_trade("MSFT", 5, 200.0)   # $1,000 invested
        portfolio.execute_trade("GOOGL", 2, 1500.0) # $3,000 invested
        
        # Initial portfolio value: $10,000 - $5,500 + current market value
        initial_cash = 10000.0
        invested_amount = 5500.0
        current_cash = initial_cash - invested_amount
        
        # Add historical data for multiple dates
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ]
        
        # Historical price data showing some volatility
        historical_prices = [
            {"AAPL": 145.0, "MSFT": 195.0, "GOOGL": 1450.0},  # Day 1
            {"AAPL": 147.0, "MSFT": 198.0, "GOOGL": 1470.0},  # Day 2
            {"AAPL": 144.0, "MSFT": 192.0, "GOOGL": 1430.0},  # Day 3
            {"AAPL": 146.0, "MSFT": 197.0, "GOOGL": 1460.0},  # Day 4
            {"AAPL": 149.0, "MSFT": 201.0, "GOOGL": 1480.0}   # Day 5
        ]
        
        # Add historical prices with max_history_size parameter
        snapshots = []
        for i, date in enumerate(dates):
            snapshot = portfolio.add_historical_prices(historical_prices[i], date, max_history_size=10)
            snapshots.append(snapshot)
            
        # Verify snapshots were created correctly
        self.assertEqual(len(snapshots), 5)
        
        # Verify the content of the first snapshot
        self.assertEqual(snapshots[0]["timestamp"], dates[0])
        self.assertEqual(len(snapshots[0]["positions"]), 3)
        
        # Calculate expected portfolio values for each day
        expected_values = []
        for prices in historical_prices:
            value = current_cash
            value += 10 * prices["AAPL"]    # 10 shares of AAPL
            value += 5 * prices["MSFT"]     # 5 shares of MSFT
            value += 2 * prices["GOOGL"]    # 2 shares of GOOGL
            expected_values.append(value)
        
        # Verify portfolio values in snapshots
        for i, snapshot in enumerate(snapshots):
            self.assertAlmostEqual(snapshot["portfolio_value"], expected_values[i], delta=0.01)
        
        # Get risk metrics after adding historical data
        risk_metrics = portfolio.get_risk_metrics()
        
        # Verify risk metrics are available and contain expected fields
        self.assertIsNotNone(risk_metrics)
        self.assertIn("portfolio_var", risk_metrics)
        self.assertIn("position_var", risk_metrics)
        
        # Test with missing symbols
        incomplete_prices = {"AAPL": 150.0, "MSFT": 200.0}  # GOOGL missing
        incomplete_date = datetime(2023, 1, 6)
        
        incomplete_snapshot = portfolio.add_historical_prices(incomplete_prices, incomplete_date)
        
        # Verify only two positions in the incomplete snapshot
        self.assertEqual(len(incomplete_snapshot["positions"]), 2)
        self.assertIn("AAPL", incomplete_snapshot["positions"])
        self.assertIn("MSFT", incomplete_snapshot["positions"])
        self.assertNotIn("GOOGL", incomplete_snapshot["positions"])
        
        # Verify history size limitation works
        # Add more historical data points to exceed the limit
        for i in range(10):
            date = datetime(2023, 1, 10 + i)
            portfolio.add_historical_prices(historical_prices[0], date, max_history_size=5)
        
        # Access the internal historical snapshots (for testing purposes only)
        self.assertEqual(len(portfolio._historical_snapshots), 5)
        
        # Verify the oldest snapshots were removed
        self.assertGreaterEqual(portfolio._historical_snapshots[0]["timestamp"], datetime(2023, 1, 15))
        
        # Test with empty price data
        empty_date = datetime(2023, 1, 20)
        empty_snapshot = portfolio.add_historical_prices({}, empty_date)
        
        # Verify empty positions in the snapshot
        self.assertEqual(len(empty_snapshot["positions"]), 0)
        self.assertEqual(empty_snapshot["portfolio_value"], current_cash)

    def test_complete_portfolio_system(self):
        """Test the complete portfolio system with all components."""
        # Create complete system
        system = self.factory.create_complete_portfolio_system()
        portfolio = system["portfolio_manager"]
        performance = system["performance_calculator"]
        tax = system["tax_manager"]
        allocation = system["allocation_manager"]

        # Execute trades
        portfolio.execute_trade("AAPL", 50, 150.0, datetime.now())
        portfolio.execute_trade("MSFT", 40, 250.0, datetime.now())
        portfolio.execute_trade("GOOGL", 10, 1500.0, datetime.now())
        portfolio.execute_trade("BND", 100, 85.0, datetime.now())

        # Update prices
        prices = {"AAPL": 160.0, "MSFT": 260.0, "GOOGL": 1550.0, "BND": 86.0}
        portfolio.update_prices(prices)

        # Add tax lots
        for symbol, position in portfolio.positions.items():
            tax.add_tax_lot(symbol, position["quantity"], position["average_price"], position["acquisition_date"])

        # Update allocations
        portfolio_values = {
            symbol: position["quantity"] * prices.get(symbol, position["last_price"])
            for symbol, position in portfolio.positions.items()
        }
        portfolio_values["CASH"] = portfolio.cash

        categories = {
            "AAPL": "stocks",
            "MSFT": "stocks",
            "GOOGL": "stocks",
            "BND": "bonds",
            "CASH": "cash"
        }

        allocation.update_allocation_from_portfolio(portfolio_values, categories)

        # Add performance data
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        performance.add_portfolio_value_observation(portfolio.get_portfolio_value(), today)
        performance.add_portfolio_value_observation(portfolio.get_portfolio_value() * 0.99, yesterday)

        # Check integration
        self.assertEqual(portfolio.get_portfolio_value(),
                        sum(portfolio_values.values()))

        # Check allocation
        rebalance_plan = allocation.generate_rebalance_plan(portfolio_values, categories)

        # Check performance
        metrics = performance.calculate_performance_metrics()
        self.assertIn("total_return", metrics)

        # Check tax lots
        lots = tax.get_tax_lots_dataframe()
        self.assertEqual(len(lots), 4)  # Four symbols

if __name__ == "__main__":
    unittest.main()
