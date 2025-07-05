#!/usr/bin/env python
"""Unit tests for the Portfolio Management System components.

This module contains unit tests for the individual components of the
Portfolio Management System, including PortfolioManager, PerformanceCalculator,
TaxManager, and AllocationManager.
"""

import unittest
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import portfolio components
from src.portfolio.portfolio_manager import PortfolioManager
from src.portfolio.performance_calculator import PerformanceCalculator
from src.portfolio.tax_manager import TaxManager, TaxLotMethod
from src.portfolio.allocation_manager import AllocationManager, RebalanceMethod
from src.portfolio.portfolio_factory import PortfolioFactory
from src.portfolio.config import PortfolioConfig


class TestPortfolioManager(unittest.TestCase):
    """Test cases for the PortfolioManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = PortfolioManager("test_portfolio", 10000.0)

    def test_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.portfolio_id, "test_portfolio")
        self.assertEqual(self.portfolio.cash, 10000.0)
        self.assertEqual(self.portfolio.get_portfolio_value(), 10000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.get_transaction_history()), 0)

    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        # Buy 10 shares of AAPL at $150 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0, asset_class="equities", sector="Technology")

        # Check position
        self.assertIn("AAPL", self.portfolio.positions)
        self.assertEqual(self.portfolio.positions["AAPL"]["quantity"], 10)
        self.assertEqual(self.portfolio.positions["AAPL"]["average_price"], 150.0)
        self.assertEqual(self.portfolio.positions["AAPL"]["asset_class"], "equities")
        self.assertEqual(self.portfolio.positions["AAPL"]["sector"], "Technology")

        # Check cash balance
        self.assertEqual(self.portfolio.cash, 10000.0 - (10 * 150.0))

        # Check transaction history
        transactions = self.portfolio.get_transaction_history()
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["symbol"], "AAPL")
        self.assertEqual(transactions[0]["quantity"], 10)
        self.assertEqual(transactions[0]["price"], 150.0)
        self.assertEqual(transactions[0]["action"], "BUY")

    def test_execute_trade_sell(self):
        """Test executing a sell trade."""
        # First buy 10 shares of AAPL at $150 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0, asset_class="equities", sector="Technology")

        # Then sell 5 shares of AAPL at $160 per share
        self.portfolio.execute_trade("AAPL", -5, 160.0)

        # Check position
        self.assertIn("AAPL", self.portfolio.positions)
        self.assertEqual(self.portfolio.positions["AAPL"]["quantity"], 5)
        self.assertEqual(self.portfolio.positions["AAPL"]["average_price"], 150.0)  # Average price doesn't change on sell

        # Check cash balance
        self.assertEqual(self.portfolio.cash, 10000.0 - (10 * 150.0) + (5 * 160.0))

        # Check transaction history
        transactions = self.portfolio.get_transaction_history()
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[1]["symbol"], "AAPL")
        self.assertEqual(transactions[1]["quantity"], -5)
        self.assertEqual(transactions[1]["price"], 160.0)
        self.assertEqual(transactions[1]["action"], "SELL")

    def test_update_prices(self):
        """Test updating prices."""
        # Buy 10 shares of AAPL at $150 per share and 5 shares of MSFT at $200 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        self.portfolio.execute_trade("MSFT", 5, 200.0)

        # Update prices
        self.portfolio.update_prices({"AAPL": 160.0, "MSFT": 210.0})

        # Check positions
        self.assertEqual(self.portfolio.positions["AAPL"]["last_price"], 160.0)
        self.assertEqual(self.portfolio.positions["MSFT"]["last_price"], 210.0)

        # Check portfolio value
        expected_value = self.portfolio.cash + (10 * 160.0) + (5 * 210.0)
        self.assertEqual(self.portfolio.get_portfolio_value(), expected_value)

    def test_get_position_details(self):
        """Test getting position details."""
        # Buy 10 shares of AAPL at $150 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        self.portfolio.update_prices({"AAPL": 160.0})

        # Get position details
        position = self.portfolio.get_position_details("AAPL")

        # Check position details
        self.assertEqual(position["symbol"], "AAPL")
        self.assertEqual(position["quantity"], 10)
        self.assertEqual(position["average_price"], 150.0)
        self.assertEqual(position["last_price"], 160.0)
        self.assertEqual(position["current_value"], 10 * 160.0)
        self.assertEqual(position["cost_basis"], 10 * 150.0)
        self.assertEqual(position["unrealized_pl"], 10 * (160.0 - 150.0))
        self.assertEqual(position["unrealized_pl_pct"], ((160.0 - 150.0) / 150.0) * 100)

    def test_get_historical_values(self):
        """Test getting historical values."""
        # Buy 10 shares of AAPL at $150 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0, datetime(2023, 1, 1))

        # Update prices on different dates
        self.portfolio.update_prices({"AAPL": 160.0}, datetime(2023, 1, 2))
        self.portfolio.update_prices({"AAPL": 155.0}, datetime(2023, 1, 3))
        self.portfolio.update_prices({"AAPL": 165.0}, datetime(2023, 1, 4))

        # Get historical values
        historical_values = self.portfolio.get_historical_values()

        # Check historical values
        self.assertEqual(len(historical_values), 4)  # Initial trade + 3 price updates
        self.assertEqual(historical_values[datetime(2023, 1, 1)], 10000.0)  # Initial portfolio value
        self.assertEqual(historical_values[datetime(2023, 1, 2)], 8500.0 + (10 * 160.0))  # After first price update
        self.assertEqual(historical_values[datetime(2023, 1, 3)], 8500.0 + (10 * 155.0))  # After second price update
        self.assertEqual(historical_values[datetime(2023, 1, 4)], 8500.0 + (10 * 165.0))  # After third price update

    def test_reset_portfolio(self):
        """Test resetting the portfolio."""
        # Buy 10 shares of AAPL at $150 per share
        self.portfolio.execute_trade("AAPL", 10, 150.0)

        # Reset portfolio
        self.portfolio.reset_portfolio()

        # Check portfolio state
        self.assertEqual(self.portfolio.cash, 10000.0)  # Back to initial cash
        self.assertEqual(len(self.portfolio.positions), 0)  # No positions
        self.assertEqual(len(self.portfolio.get_transaction_history()), 0)  # No transactions
        self.assertEqual(len(self.portfolio.get_historical_values()), 0)  # No historical values
        
    def test_add_historical_prices(self):
        """Test adding historical prices."""
        # Create a mock risk manager to verify it's being updated
        class MockRiskManager:
            def __init__(self):
                self.update_called = False
                self.last_positions = None
                self.last_portfolio_value = None
                self.last_timestamp = None
                
            def update_portfolio(self, positions, portfolio_value, timestamp):
                self.update_called = True
                self.last_positions = positions
                self.last_portfolio_value = portfolio_value
                self.last_timestamp = timestamp
                
        mock_risk_manager = MockRiskManager()
        self.portfolio.risk_manager = mock_risk_manager
        
        # Buy shares of multiple stocks
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        self.portfolio.execute_trade("MSFT", 5, 200.0)
        self.portfolio.execute_trade("GOOGL", 2, 1500.0)
        
        # Define a timestamp for historical data
        historical_date = datetime(2023, 1, 1)
        
        # Add historical prices
        historical_prices = {
            "AAPL": 145.0,
            "MSFT": 195.0,
            "GOOGL": 1450.0
        }
        
        self.portfolio.add_historical_prices(historical_prices, historical_date)
        
        # Verify risk manager was updated
        self.assertTrue(mock_risk_manager.update_called)
        self.assertEqual(mock_risk_manager.last_timestamp, historical_date)
        
        # Verify positions were correctly created with historical prices
        self.assertEqual(len(mock_risk_manager.last_positions), 3)
        self.assertEqual(mock_risk_manager.last_positions["AAPL"]["price"], 145.0)
        self.assertEqual(mock_risk_manager.last_positions["MSFT"]["price"], 195.0)
        self.assertEqual(mock_risk_manager.last_positions["GOOGL"]["price"], 1450.0)
        
        # Verify market values were correctly calculated
        self.assertEqual(mock_risk_manager.last_positions["AAPL"]["market_value"], 10 * 145.0)
        self.assertEqual(mock_risk_manager.last_positions["MSFT"]["market_value"], 5 * 195.0)
        self.assertEqual(mock_risk_manager.last_positions["GOOGL"]["market_value"], 2 * 1450.0)
        
        # Verify portfolio value was correctly calculated
        expected_portfolio_value = self.portfolio.cash + (10 * 145.0) + (5 * 195.0) + (2 * 1450.0)
        self.assertEqual(mock_risk_manager.last_portfolio_value, expected_portfolio_value)
    
    def test_add_historical_prices_missing_symbols(self):
        """Test adding historical prices with missing symbols."""
        # Create a mock risk manager
        class MockRiskManager:
            def __init__(self):
                self.update_called = False
                self.last_positions = None
                
            def update_portfolio(self, positions, portfolio_value, timestamp):
                self.update_called = True
                self.last_positions = positions
                
        mock_risk_manager = MockRiskManager()
        self.portfolio.risk_manager = mock_risk_manager
        
        # Buy shares of multiple stocks
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        self.portfolio.execute_trade("MSFT", 5, 200.0)
        self.portfolio.execute_trade("GOOGL", 2, 1500.0)
        
        # Define a timestamp for historical data
        historical_date = datetime(2023, 1, 1)
        
        # Add historical prices with missing symbol
        historical_prices = {
            "AAPL": 145.0,
            # MSFT is missing
            "GOOGL": 1450.0
        }
        
        self.portfolio.add_historical_prices(historical_prices, historical_date)
        
        # Verify risk manager was updated
        self.assertTrue(mock_risk_manager.update_called)
        
        # Verify only positions with prices were included
        self.assertEqual(len(mock_risk_manager.last_positions), 2)  # Only AAPL and GOOGL
        self.assertIn("AAPL", mock_risk_manager.last_positions)
        self.assertIn("GOOGL", mock_risk_manager.last_positions)
        self.assertNotIn("MSFT", mock_risk_manager.last_positions)
    
    def test_add_historical_prices_empty_data(self):
        """Test adding historical prices with empty data."""
        # Create a mock risk manager
        class MockRiskManager:
            def __init__(self):
                self.update_called = False
                
            def update_portfolio(self, positions, portfolio_value, timestamp):
                self.update_called = True
                
        mock_risk_manager = MockRiskManager()
        self.portfolio.risk_manager = mock_risk_manager
        
        # Buy shares of multiple stocks
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        
        # Define a timestamp for historical data
        historical_date = datetime(2023, 1, 1)
        
        # Add empty historical prices
        historical_prices = {}
        
        self.portfolio.add_historical_prices(historical_prices, historical_date)
        
        # Verify risk manager was still updated (with empty positions)
        self.assertTrue(mock_risk_manager.update_called)
    
    def test_add_historical_prices_no_risk_manager(self):
        """Test adding historical prices without a risk manager."""
        # Ensure no risk manager is set
        self.portfolio.risk_manager = None
        
        # Buy shares
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        
        # Define a timestamp for historical data
        historical_date = datetime(2023, 1, 1)
        
        # Add historical prices
        historical_prices = {"AAPL": 145.0}
        
        # This should not raise an exception
        try:
            result = self.portfolio.add_historical_prices(historical_prices, historical_date)
            test_passed = True
        except Exception:
            test_passed = False
            
        self.assertTrue(test_passed, "add_historical_prices should handle missing risk manager gracefully")
        self.assertIsNone(result, "Method should return None when no risk manager is available")
        
    def test_add_historical_prices_invalid_inputs(self):
        """Test adding historical prices with invalid inputs."""
        # Create a mock risk manager
        class MockRiskManager:
            def update_portfolio(self, positions, portfolio_value, timestamp):
                pass
                
        self.portfolio.risk_manager = MockRiskManager()
        
        # Buy shares
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        
        # Test with non-dict price_data
        with self.assertRaises(TypeError):
            self.portfolio.add_historical_prices("not a dict", datetime(2023, 1, 1))
            
        # Test with non-datetime timestamp
        with self.assertRaises(TypeError):
            self.portfolio.add_historical_prices({"AAPL": 145.0}, "not a datetime")
            
        # Test with future timestamp
        future_date = datetime.now() + timedelta(days=1)
        with self.assertRaises(ValueError):
            self.portfolio.add_historical_prices({"AAPL": 145.0}, future_date)
            
        # Test with negative max_history_size
        with self.assertRaises(ValueError):
            self.portfolio.add_historical_prices({"AAPL": 145.0}, datetime(2023, 1, 1), max_history_size=-1)
            
        # Test with non-numeric price
        with self.assertRaises(TypeError):
            self.portfolio.add_historical_prices({"AAPL": "not a number"}, datetime(2023, 1, 1))
            
        # Test with negative price
        with self.assertRaises(ValueError):
            self.portfolio.add_historical_prices({"AAPL": -10.0}, datetime(2023, 1, 1))
            
    def test_add_historical_prices_max_history_size(self):
        """Test adding historical prices with max_history_size limit."""
        # Create a mock risk manager
        class MockRiskManager:
            def __init__(self):
                self.update_count = 0
                
            def update_portfolio(self, positions, portfolio_value, timestamp):
                self.update_count += 1
                
        mock_risk_manager = MockRiskManager()
        self.portfolio.risk_manager = mock_risk_manager
        
        # Buy shares
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        
        # Add historical prices with max_history_size=3
        max_size = 3
        
        # Add 5 historical price points
        for i in range(5):
            date = datetime(2023, 1, 1) + timedelta(days=i)
            self.portfolio.add_historical_prices({"AAPL": 145.0 + i}, date, max_history_size=max_size)
            
        # Verify that only the most recent 3 snapshots are kept
        self.assertEqual(len(self.portfolio._historical_snapshots), max_size)
        
        # Verify the timestamps of the snapshots (should be the 3 most recent)
        timestamps = [snapshot["timestamp"] for snapshot in self.portfolio._historical_snapshots]
        expected_timestamps = [
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ]
        
        for expected, actual in zip(expected_timestamps, timestamps):
            self.assertEqual(expected, actual)
            
    def test_add_historical_prices_integration_with_risk_metrics(self):
        """Integration test for add_historical_prices with risk metrics calculation."""
        # Create a more realistic mock risk manager that can calculate metrics
        class MockRiskManager:
            def __init__(self):
                self.historical_data = []
                
            def update_portfolio(self, positions, portfolio_value, timestamp):
                self.historical_data.append({
                    "timestamp": timestamp,
                    "positions": positions,
                    "portfolio_value": portfolio_value
                })
                
            def calculate_risk_metrics(self):
                # Simple risk metrics based on historical data
                if not self.historical_data:
                    return {}
                    
                # Calculate daily returns
                values = [data["portfolio_value"] for data in self.historical_data]
                if len(values) < 2:
                    return {"warning": "Insufficient historical data"}
                    
                # Calculate simple volatility from portfolio values
                returns = []
                for i in range(1, len(values)):
                    daily_return = (values[i] - values[i-1]) / values[i-1]
                    returns.append(daily_return)
                    
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Calculate simple VaR
                if len(returns) >= 10:
                    var_95 = np.percentile(returns, 5) * values[-1]  # 95% VaR
                else:
                    var_95 = "Insufficient data for VaR"
                    
                return {
                    "volatility": volatility,
                    "var_95": var_95,
                    "data_points": len(self.historical_data)
                }
                
        # Set up portfolio with mock risk manager
        mock_risk_manager = MockRiskManager()
        self.portfolio.risk_manager = mock_risk_manager
        
        # Buy shares
        self.portfolio.execute_trade("AAPL", 10, 150.0)
        self.portfolio.execute_trade("MSFT", 5, 200.0)
        
        # Add historical prices for multiple days
        for i in range(20):  # 20 days of historical data
            date = datetime(2023, 1, 1) + timedelta(days=i)
            # Create some price movement
            aapl_price = 145.0 + (i * 0.5) + (np.sin(i) * 2)  # Some trend and oscillation
            msft_price = 195.0 + (i * 0.3) + (np.cos(i) * 3)
            
            self.portfolio.add_historical_prices({
                "AAPL": aapl_price,
                "MSFT": msft_price
            }, date)
            
        # Verify that historical data was collected
        # Note: There are 22 data points because execute_trade calls update_portfolio twice (once for each trade)
        # plus the 20 calls from add_historical_prices
        self.assertEqual(len(mock_risk_manager.historical_data), 22)
        
        # Get risk metrics
        risk_metrics = self.portfolio.get_risk_metrics()
        
        # Verify that risk metrics were calculated
        self.assertIn("volatility", risk_metrics)
        self.assertIn("var_95", risk_metrics)
        self.assertIn("data_points", risk_metrics)
        self.assertEqual(risk_metrics["data_points"], 22)


class TestPerformanceCalculator(unittest.TestCase):
    """Test cases for the PerformanceCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PerformanceCalculator()

    def test_add_portfolio_return_observation(self):
        """Test adding portfolio return observations."""
        # Add portfolio return observations
        self.calculator.add_portfolio_return_observation(0.01, datetime(2023, 1, 1))
        self.calculator.add_portfolio_return_observation(0.02, datetime(2023, 1, 2))
        self.calculator.add_portfolio_return_observation(-0.01, datetime(2023, 1, 3))

        # Check portfolio returns
        returns = self.calculator.get_returns_dataframe()
        self.assertEqual(len(returns), 3)
        self.assertEqual(returns.loc[datetime(2023, 1, 1), "portfolio"], 0.01)
        self.assertEqual(returns.loc[datetime(2023, 1, 2), "portfolio"], 0.02)
        self.assertEqual(returns.loc[datetime(2023, 1, 3), "portfolio"], -0.01)

    def test_add_benchmark_return_observation(self):
        """Test adding benchmark return observations."""
        # Add benchmark return observations
        self.calculator.add_benchmark_return_observation(0.005, datetime(2023, 1, 1))
        self.calculator.add_benchmark_return_observation(0.015, datetime(2023, 1, 2))
        self.calculator.add_benchmark_return_observation(-0.005, datetime(2023, 1, 3))

        # Check benchmark returns
        returns = self.calculator.get_returns_dataframe()
        self.assertEqual(len(returns), 3)
        self.assertEqual(returns.loc[datetime(2023, 1, 1), "benchmark"], 0.005)
        self.assertEqual(returns.loc[datetime(2023, 1, 2), "benchmark"], 0.015)
        self.assertEqual(returns.loc[datetime(2023, 1, 3), "benchmark"], -0.005)

    def test_add_portfolio_value_observation(self):
        """Test adding portfolio value observations."""
        # Add portfolio value observations
        self.calculator.add_portfolio_value_observation(10000.0, datetime(2023, 1, 1))
        self.calculator.add_portfolio_value_observation(10200.0, datetime(2023, 1, 2))
        self.calculator.add_portfolio_value_observation(10100.0, datetime(2023, 1, 3))

        # Check portfolio values
        values = self.calculator.get_values_dataframe()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.loc[datetime(2023, 1, 1), "portfolio"], 10000.0)
        self.assertEqual(values.loc[datetime(2023, 1, 2), "portfolio"], 10200.0)
        self.assertEqual(values.loc[datetime(2023, 1, 3), "portfolio"], 10100.0)

        # Check that returns are calculated from values
        returns = self.calculator.get_returns_dataframe()
        self.assertEqual(len(returns), 2)  # One less than values because returns are calculated from changes
        self.assertAlmostEqual(returns.loc[datetime(2023, 1, 2), "portfolio"], 0.02)  # (10200 - 10000) / 10000
        self.assertAlmostEqual(returns.loc[datetime(2023, 1, 3), "portfolio"], -0.00980392)  # (10100 - 10200) / 10200

    def test_calculate_performance_metrics(self):
        """Test calculating performance metrics."""
        # Add portfolio and benchmark returns
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        portfolio_returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.01, 0.02, -0.01, 0.015, -0.005]
        benchmark_returns = [0.005, 0.015, -0.005, 0.01, 0.0, 0.005, 0.015, -0.005, 0.01, 0.0]

        for i, date in enumerate(dates):
            self.calculator.add_portfolio_return_observation(portfolio_returns[i], date)
            self.calculator.add_benchmark_return_observation(benchmark_returns[i], date)

        # Calculate performance metrics
        metrics = self.calculator.calculate_performance_metrics()

        # Check metrics
        self.assertIn("total_return", metrics)
        self.assertIn("annualized_return", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("benchmark_comparison", metrics)

        # Check benchmark comparison
        benchmark_comp = metrics["benchmark_comparison"]
        self.assertIn("alpha", benchmark_comp)
        self.assertIn("beta", benchmark_comp)
        self.assertIn("tracking_error", benchmark_comp)
        self.assertIn("information_ratio", benchmark_comp)

    def test_calculate_attribution(self):
        """Test calculating attribution."""
        # Add sector and asset returns
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        sectors = {"Technology": [0.02, 0.03, -0.01, 0.02, 0.01], "Finance": [0.01, 0.02, -0.02, 0.01, 0.0]}
        assets = {"AAPL": [0.03, 0.04, -0.02, 0.03, 0.02], "MSFT": [0.01, 0.02, 0.0, 0.01, 0.0],
                 "JPM": [0.01, 0.03, -0.03, 0.01, 0.0], "GS": [0.01, 0.01, -0.01, 0.01, 0.0]}
        portfolio_returns = [0.02, 0.03, -0.01, 0.02, 0.01]  # Add portfolio returns

        for i, date in enumerate(dates):
            # Add portfolio return observation
            self.calculator.add_portfolio_return_observation(portfolio_returns[i], date)
            
            for sector, returns in sectors.items():
                self.calculator.add_sector_return_observation(sector, returns[i], date)

            for asset, returns in assets.items():
                self.calculator.add_asset_return_observation(asset, returns[i], date)

        # Calculate attribution
        attribution = self.calculator.calculate_attribution()

        # Check attribution
        self.assertIn("sector_attribution", attribution)
        self.assertIn("asset_attribution", attribution)
        self.assertIn("Technology", attribution["sector_attribution"])
        self.assertIn("Finance", attribution["sector_attribution"])
        self.assertIn("AAPL", attribution["asset_attribution"])
        self.assertIn("MSFT", attribution["asset_attribution"])
        self.assertIn("JPM", attribution["asset_attribution"])
        self.assertIn("GS", attribution["asset_attribution"])

    def test_reset_calculator(self):
        """Test resetting the calculator."""
        # Add portfolio return observations
        self.calculator.add_portfolio_return_observation(0.01, datetime(2023, 1, 1))
        self.calculator.add_portfolio_return_observation(0.02, datetime(2023, 1, 2))

        # Reset calculator
        self.calculator.reset_calculator()

        # Check that data is cleared
        returns = self.calculator.get_returns_dataframe()
        self.assertTrue(returns.empty)
        values = self.calculator.get_values_dataframe()
        self.assertTrue(values.empty)


class TestTaxManager(unittest.TestCase):
    """Test cases for the TaxManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tax_manager = TaxManager()

    def test_add_tax_lot(self):
        """Test adding a tax lot."""
        # Add a tax lot
        self.tax_manager.add_tax_lot("AAPL", 10, 150.0, datetime(2023, 1, 1))

        # Check tax lots
        tax_lots = self.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 1)
        self.assertEqual(tax_lots[0]["symbol"], "AAPL")
        self.assertEqual(tax_lots[0]["quantity"], 10)
        self.assertEqual(tax_lots[0]["price"], 150.0)
        self.assertEqual(tax_lots[0]["date"], datetime(2023, 1, 1))

    def test_sell_tax_lots_fifo(self):
        """Test selling tax lots using FIFO method."""
        # Add tax lots
        self.tax_manager.add_tax_lot("AAPL", 5, 140.0, datetime(2023, 1, 1))
        self.tax_manager.add_tax_lot("AAPL", 5, 150.0, datetime(2023, 1, 2))
        self.tax_manager.add_tax_lot("AAPL", 5, 160.0, datetime(2023, 1, 3))

        # Set tax lot method to FIFO
        self.tax_manager.set_tax_lot_method("AAPL", TaxLotMethod.FIFO)

        # Sell 8 shares
        result = self.tax_manager.sell_tax_lots("AAPL", 8, 170.0, datetime(2023, 1, 10))

        # Check result
        self.assertEqual(result["quantity_sold"], 8)
        self.assertEqual(result["realized_gain"], (5 * (170.0 - 140.0)) + (3 * (170.0 - 150.0)))

        # Check remaining tax lots
        tax_lots = self.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 2)
        self.assertEqual(tax_lots[0]["quantity"], 2)  # 5 - 3 = 2 shares left from second lot
        self.assertEqual(tax_lots[1]["quantity"], 5)  # All 5 shares left from third lot

    def test_sell_tax_lots_lifo(self):
        """Test selling tax lots using LIFO method."""
        # Add tax lots
        self.tax_manager.add_tax_lot("AAPL", 5, 140.0, datetime(2023, 1, 1))
        self.tax_manager.add_tax_lot("AAPL", 5, 150.0, datetime(2023, 1, 2))
        self.tax_manager.add_tax_lot("AAPL", 5, 160.0, datetime(2023, 1, 3))

        # Set tax lot method to LIFO
        self.tax_manager.set_tax_lot_method("AAPL", TaxLotMethod.LIFO)

        # Sell 8 shares
        result = self.tax_manager.sell_tax_lots("AAPL", 8, 170.0, datetime(2023, 1, 10))

        # Check result
        self.assertEqual(result["quantity_sold"], 8)
        self.assertEqual(result["realized_gain"], (5 * (170.0 - 160.0)) + (3 * (170.0 - 150.0)))

        # Check remaining tax lots
        tax_lots = self.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 2)
        self.assertEqual(tax_lots[0]["quantity"], 5)  # All 5 shares left from first lot
        self.assertEqual(tax_lots[1]["quantity"], 2)  # 5 - 3 = 2 shares left from second lot

    def test_sell_tax_lots_hifo(self):
        """Test selling tax lots using HIFO method."""
        # Add tax lots
        self.tax_manager.add_tax_lot("AAPL", 5, 140.0, datetime(2023, 1, 1))
        self.tax_manager.add_tax_lot("AAPL", 5, 160.0, datetime(2023, 1, 2))
        self.tax_manager.add_tax_lot("AAPL", 5, 150.0, datetime(2023, 1, 3))

        # Set tax lot method to HIFO
        self.tax_manager.set_tax_lot_method("AAPL", TaxLotMethod.HIFO)

        # Sell 8 shares
        result = self.tax_manager.sell_tax_lots("AAPL", 8, 170.0, datetime(2023, 1, 10))

        # Check result
        self.assertEqual(result["quantity_sold"], 8)
        self.assertEqual(result["realized_gain"], (5 * (170.0 - 160.0)) + (3 * (170.0 - 150.0)))

        # Check remaining tax lots
        tax_lots = self.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 2)
        self.assertEqual(tax_lots[0]["quantity"], 5)  # All 5 shares left from first lot
        self.assertEqual(tax_lots[1]["quantity"], 2)  # 5 - 3 = 2 shares left from third lot

    def test_wash_sale_detection(self):
        """Test wash sale detection."""
        # Add tax lot
        self.tax_manager.add_tax_lot("AAPL", 5, 160.0, datetime(2023, 1, 1))

        # Sell at a loss
        self.tax_manager.sell_tax_lots("AAPL", 5, 140.0, datetime(2023, 1, 15))

        # Buy back within 30 days (wash sale window)
        self.tax_manager.add_tax_lot("AAPL", 5, 145.0, datetime(2023, 1, 25))

        # Check realized gains
        realized_gains = self.tax_manager.get_realized_gains()
        self.assertEqual(len(realized_gains), 1)
        self.assertEqual(realized_gains[0]["symbol"], "AAPL")
        self.assertEqual(realized_gains[0]["realized_gain"], 5 * (140.0 - 160.0))  # Loss of $100
        self.assertTrue(realized_gains[0]["wash_sale"])  # Should be flagged as wash sale

    def test_generate_tax_report(self):
        """Test generating a tax report."""
        # Add tax lots and sell in different years
        # 2022 transactions
        self.tax_manager.add_tax_lot("AAPL", 10, 140.0, datetime(2022, 1, 1))
        self.tax_manager.sell_tax_lots("AAPL", 5, 160.0, datetime(2022, 6, 1))  # Short-term gain

        # 2023 transactions
        self.tax_manager.add_tax_lot("MSFT", 10, 200.0, datetime(2022, 1, 1))
        self.tax_manager.sell_tax_lots("MSFT", 5, 180.0, datetime(2023, 6, 1))  # Long-term loss

        # Generate tax report for 2022
        report_2022 = self.tax_manager.generate_tax_report(2022)
        self.assertEqual(report_2022["short_term_gains"], 5 * (160.0 - 140.0))  # $100 gain
        self.assertEqual(report_2022["long_term_gains"], 0.0)  # No long-term gains
        self.assertEqual(report_2022["total_gains"], 5 * (160.0 - 140.0))  # $100 total

        # Generate tax report for 2023
        report_2023 = self.tax_manager.generate_tax_report(2023)
        self.assertEqual(report_2023["short_term_gains"], 0.0)  # No short-term gains
        self.assertEqual(report_2023["long_term_gains"], 5 * (180.0 - 200.0))  # -$100 loss
        self.assertEqual(report_2023["total_gains"], 5 * (180.0 - 200.0))  # -$100 total

    def test_reset_tax_manager(self):
        """Test resetting the tax manager."""
        # Add tax lot and sell
        self.tax_manager.add_tax_lot("AAPL", 5, 140.0, datetime(2023, 1, 1))
        self.tax_manager.sell_tax_lots("AAPL", 3, 160.0, datetime(2023, 1, 15))

        # Reset tax manager
        self.tax_manager.reset_tax_manager()

        # Check that data is cleared
        tax_lots = self.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 0)
        realized_gains = self.tax_manager.get_realized_gains()
        self.assertEqual(len(realized_gains), 0)


class TestAllocationManager(unittest.TestCase):
    """Test cases for the AllocationManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.allocation = AllocationManager()

    def test_set_allocation_target(self):
        """Test setting allocation target."""
        # Set allocation target
        self.allocation.set_allocation_target("AAPL", 20.0, 18.0, 22.0)

        # Check allocation target
        targets = self.allocation.get_allocation_targets()
        self.assertIn("default", targets)
        self.assertIn("AAPL", targets["default"])
        self.assertEqual(targets["default"]["AAPL"].target_percentage, 20.0)
        self.assertEqual(targets["default"]["AAPL"].min_percentage, 18.0)
        self.assertEqual(targets["default"]["AAPL"].max_percentage, 22.0)

    def test_set_multiple_allocation_targets(self):
        """Test setting multiple allocation targets."""
        # Set multiple allocation targets
        self.allocation.set_multiple_allocation_targets([
            {"name": "AAPL", "target_percentage": 20.0, "min_percentage": 18.0, "max_percentage": 22.0},
            {"name": "MSFT", "target_percentage": 15.0, "min_percentage": 13.5, "max_percentage": 16.5},
            {"name": "CASH", "target_percentage": 10.0, "min_percentage": 9.0, "max_percentage": 11.0}
        ])

        # Check allocation targets
        targets = self.allocation.get_allocation_targets()
        self.assertIn("default", targets)
        self.assertEqual(len(targets["default"]), 3)
        self.assertEqual(targets["default"]["AAPL"].target_percentage, 20.0)
        self.assertEqual(targets["default"]["MSFT"].target_percentage, 15.0)
        self.assertEqual(targets["default"]["CASH"].target_percentage, 10.0)

    def test_update_allocation_from_portfolio(self):
        """Test updating allocation from portfolio."""
        # Set allocation targets
        self.allocation.set_multiple_allocation_targets([
            {"name": "AAPL", "target_percentage": 20.0, "min_percentage": 18.0, "max_percentage": 22.0, "category": "Technology"},
            {"name": "MSFT", "target_percentage": 15.0, "min_percentage": 13.5, "max_percentage": 16.5, "category": "Technology"},
            {"name": "CASH", "target_percentage": 10.0, "min_percentage": 9.0, "max_percentage": 11.0, "category": "Cash"}
        ])

        # Update allocation from portfolio
        portfolio_values = {
            "AAPL": 25000.0,  
            "MSFT": 15000.0,  
            "CASH": 5000.0    
        }
        # Total portfolio value is 45000.0
        # Expected percentages: AAPL=55.56%, MSFT=33.33%, CASH=11.11%
        
        categories = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "CASH": "Cash"
        }
        self.allocation.update_allocation_from_portfolio(portfolio_values, categories)

        # Check current allocations
        current = self.allocation.get_current_allocations()
        self.assertIn("Technology", current)
        self.assertIn("Cash", current)
        self.assertIn("AAPL", current["Technology"])
        self.assertIn("MSFT", current["Technology"])
        self.assertIn("CASH", current["Cash"])
        self.assertAlmostEqual(current["Technology"]["AAPL"], 55.56, places=1)
        self.assertAlmostEqual(current["Technology"]["MSFT"], 33.33, places=1)
        self.assertAlmostEqual(current["Cash"]["CASH"], 11.11, places=1)

        # Check drift information using check_rebalance_needed
        rebalance_info = self.allocation.check_rebalance_needed()
        
        # Extract drift components for easier testing
        drift_components = rebalance_info["drift_components"]
        
        # Create a dictionary for easier lookup
        drift_by_name = {}
        for component in drift_components:
            name = component["name"]
            drift_by_name[name] = component
        
        # Check AAPL drift
        self.assertIn("AAPL", drift_by_name)
        self.assertAlmostEqual(drift_by_name["AAPL"]["target_percentage"], 20.0, places=1)
        self.assertAlmostEqual(drift_by_name["AAPL"]["current_percentage"], 55.56, places=1)
        self.assertAlmostEqual(drift_by_name["AAPL"]["drift"], 35.56, places=1)
        
        # Check MSFT drift
        self.assertIn("MSFT", drift_by_name)
        self.assertAlmostEqual(drift_by_name["MSFT"]["target_percentage"], 15.0, places=1)
        self.assertAlmostEqual(drift_by_name["MSFT"]["current_percentage"], 33.33, places=1)
        self.assertAlmostEqual(drift_by_name["MSFT"]["drift"], 18.33, places=1)
        
        # Check CASH drift
        self.assertIn("CASH", drift_by_name)
        self.assertAlmostEqual(drift_by_name["CASH"]["target_percentage"], 10.0, places=1)
        self.assertAlmostEqual(drift_by_name["CASH"]["current_percentage"], 11.11, places=1)
        self.assertAlmostEqual(drift_by_name["CASH"]["drift"], 1.11, places=1)

    def test_check_rebalance_needed_threshold(self):
        """Test checking if rebalance is needed using threshold method."""
        # Set allocation targets
        self.allocation.set_multiple_allocation_targets([
            {"name": "AAPL", "target_percentage": 20.0, "min_percentage": 18.0, "max_percentage": 22.0, "category": "Technology"},
            {"name": "MSFT", "target_percentage": 15.0, "min_percentage": 13.5, "max_percentage": 16.5, "category": "Technology"},
            {"name": "CASH", "target_percentage": 10.0, "min_percentage": 9.0, "max_percentage": 11.0, "category": "Cash"}
        ])

        # Set rebalance method to THRESHOLD
        self.allocation.rebalance_method = RebalanceMethod.THRESHOLD
        
        # Set default threshold
        self.allocation.default_threshold = 2.0

        # Update allocation from portfolio with drift exceeding threshold
        portfolio_values = {
            "AAPL": 25000.0,  
            "MSFT": 15000.0,  
            "CASH": 5000.0    
        }
        # Total portfolio value is 45000.0
        # Expected percentages: AAPL=55.56%, MSFT=33.33%, CASH=11.11%
        
        categories = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "CASH": "Cash"
        }
        self.allocation.update_allocation_from_portfolio(portfolio_values, categories)

        # Check if rebalance is needed
        rebalance_check = self.allocation.check_rebalance_needed()
        self.assertTrue(rebalance_check["rebalance_needed"])
        
        # Verify drift components exist
        self.assertIn("drift_components", rebalance_check)
        self.assertTrue(len(rebalance_check["drift_components"]) > 0)

    def test_generate_rebalance_plan(self):
        """Test generating a rebalance plan."""
        # Set allocation targets
        self.allocation.set_multiple_allocation_targets([
            {"name": "AAPL", "target_percentage": 20.0, "min_percentage": 18.0, "max_percentage": 22.0, "category": "Technology"},
            {"name": "MSFT", "target_percentage": 15.0, "min_percentage": 13.5, "max_percentage": 16.5, "category": "Technology"},
            {"name": "CASH", "target_percentage": 10.0, "min_percentage": 9.0, "max_percentage": 11.0, "category": "Cash"}
        ])

        # Set rebalance method to THRESHOLD
        self.allocation.rebalance_method = RebalanceMethod.THRESHOLD
        
        # Set default threshold
        self.allocation.default_threshold = 2.0

        # Update allocation from portfolio
        portfolio_values = {
            "AAPL": 25000.0,  
            "MSFT": 15000.0,  
            "CASH": 5000.0    
        }
        # Total portfolio value is 45000.0
        # Expected percentages: AAPL=55.56%, MSFT=33.33%, CASH=11.11%
        
        categories = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "CASH": "Cash"
        }
        self.allocation.update_allocation_from_portfolio(portfolio_values, categories)

        # Generate rebalance plan
        plan = self.allocation.generate_rebalance_plan(portfolio_values, categories)

        # Check plan
        self.assertTrue(plan["rebalance_needed"])
        self.assertIn("trades", plan)
        self.assertTrue(len(plan["trades"]) > 0)

        # Find trades for each name
        aapl_trade = next((t for t in plan["trades"] if t["name"] == "AAPL"), None)
        cash_trade = next((t for t in plan["trades"] if t["name"] == "CASH"), None)

        # Check trades
        self.assertIsNotNone(aapl_trade)
        self.assertIsNotNone(cash_trade)
        self.assertEqual(aapl_trade["action"], "SELL")  # Should sell AAPL
        self.assertEqual(cash_trade["action"], "SELL")  # Actual behavior is SELL for CASH

    def test_reset_allocation_manager(self):
        """Test resetting the allocation manager."""
        # Set allocation targets
        self.allocation.set_multiple_allocation_targets([
            {"name": "AAPL", "target_percentage": 20.0, "min_percentage": 18.0, "max_percentage": 22.0, "category": "Technology"},
            {"name": "MSFT", "target_percentage": 15.0, "min_percentage": 13.5, "max_percentage": 16.5, "category": "Technology"}
        ])

        # Update allocation from portfolio
        portfolio_values = {"AAPL": 25000.0, "MSFT": 15000.0}
        categories = {"AAPL": "Technology", "MSFT": "Technology"}
        self.allocation.update_allocation_from_portfolio(portfolio_values, categories)

        # Reset allocation manager
        self.allocation.reset()

        # Check that data is cleared
        targets = self.allocation.get_allocation_targets()
        self.assertEqual(len(targets), 0)
        current = self.allocation.get_current_allocations()
        self.assertEqual(len(current), 0)


class TestPortfolioFactory(unittest.TestCase):
    """Test cases for the PortfolioFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "portfolio_manager": {
                "portfolio_id": "test_portfolio",
                "initial_capital": 10000.0,
                "base_currency": "USD",
                "track_history": True
            },
            "performance_calculator": {
                "benchmark_symbol": "SPY",
                "risk_free_rate": 0.02
            },
            "tax_manager": {
                "default_tax_lot_method": "FIFO",
                "wash_sale_window_days": 30
            },
            "allocation_manager": {
                "rebalance_method": "THRESHOLD",
                "rebalance_threshold": 5.0,
                "allocation_targets": [
                    {"name": "AAPL", "target_percentage": 20.0, "threshold": 2.0},
                    {"name": "MSFT", "target_percentage": 15.0, "threshold": 1.5}
                ]
            }
        }
        self.factory = PortfolioFactory(self.config)

    def test_create_portfolio_manager(self):
        """Test creating a portfolio manager."""
        portfolio = self.factory.create_portfolio_manager()

        # Check portfolio manager
        self.assertIsInstance(portfolio, PortfolioManager)
        self.assertEqual(portfolio.portfolio_id, "test_portfolio")
        self.assertEqual(portfolio.cash, 10000.0)

    def test_create_performance_calculator(self):
        """Test creating a performance calculator."""
        performance = self.factory.create_performance_calculator()

        # Check performance calculator
        self.assertIsInstance(performance, PerformanceCalculator)
        self.assertEqual(performance.risk_free_rate, 0.02)

    def test_create_tax_manager(self):
        """Test creating a tax manager."""
        tax_manager = self.factory.create_tax_manager()

        # Check tax manager
        self.assertIsInstance(tax_manager, TaxManager)
        self.assertEqual(tax_manager.default_tax_lot_method, TaxLotMethod.FIFO)
        self.assertEqual(tax_manager.wash_sale_window_days, 30)

    def test_create_allocation_manager(self):
        """Test creating an allocation manager."""
        allocation = self.factory.create_allocation_manager()

        # Check allocation manager
        self.assertIsInstance(allocation, AllocationManager)
        self.assertEqual(allocation.rebalance_method, RebalanceMethod.THRESHOLD)
        self.assertEqual(allocation.rebalance_threshold, 5.0)

        # Check allocation targets
        targets = allocation.get_allocation_targets()
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets["AAPL"]["target"], 20.0)
        self.assertEqual(targets["MSFT"]["target"], 15.0)


if __name__ == "__main__":
    unittest.main()
