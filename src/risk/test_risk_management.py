import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .position_sizer import PositionSizer
from .stop_loss_manager import StopLossManager, StopLossType
from .portfolio_risk_manager import PortfolioRiskManager
from .advanced_risk_manager import AdvancedRiskManager

class TestPositionSizer(unittest.TestCase):
    """Test cases for the PositionSizer class."""

    def setUp(self):
        self.position_sizer = PositionSizer(risk_per_trade=0.01)  # 1% risk per trade

    def test_risk_based_sizing(self):
        # Test with 1% risk on $100,000 capital with a 2% stop loss
        capital = 100000.0
        entry_price = 50.0
        stop_price = 49.0  # 2% stop loss
        risk_per_trade = 0.01  # 1% of capital

        # Expected position size: ($100,000 * 0.01) / ($50 - $49) = $1,000 / $1 = 1,000 shares
        expected_size = 1000.0

        size, details = self.position_sizer.calculate_risk_based_size(
            capital=capital,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_per_trade=risk_per_trade
        )

        self.assertEqual(size, expected_size)
        self.assertEqual(details['risk_amount'], capital * risk_per_trade)

    def test_fixed_percent_sizing(self):
        # Test with 5% position size on $100,000 capital
        capital = 100000.0
        entry_price = 50.0
        position_percent = 0.05  # 5% of capital

        # Expected position size: ($100,000 * 0.05) / $50 = $5,000 / $50 = 100 shares
        expected_size = 100.0

        size, details = self.position_sizer.calculate_fixed_percent_size(
            capital=capital,
            entry_price=entry_price,
            position_percent=position_percent
        )

        self.assertEqual(size, expected_size)
        self.assertEqual(details['position_value'], capital * position_percent)

    def test_volatility_based_sizing(self):
        # Test with 1% risk on $100,000 capital with 2% volatility
        capital = 100000.0
        entry_price = 50.0
        volatility = 0.02  # 2% daily volatility
        risk_per_trade = 0.01  # 1% of capital

        # Expected position size will depend on the volatility calculation
        size, details = self.position_sizer.calculate_volatility_based_size(
            capital=capital,
            entry_price=entry_price,
            volatility=volatility,
            risk_per_trade=risk_per_trade
        )

        # We can't predict the exact size due to the volatility calculation,
        # but we can check that it's reasonable
        self.assertGreater(size, 0)
        self.assertLess(size, capital / entry_price)  # Can't be more than 100% of capital

class TestStopLossManager(unittest.TestCase):
    """Test cases for the StopLossManager class."""

    def setUp(self):
        self.stop_loss_manager = StopLossManager()

    def test_fixed_stop_loss(self):
        # Test fixed stop loss for long position
        entry_price = 100.0
        stop_percent = 0.05  # 5% stop loss

        # Expected stop price for long: $100 * (1 - 0.05) = $95
        expected_stop_price = 95.0

        stop_price = self.stop_loss_manager.calculate_fixed_stop(
            entry_price=entry_price,
            direction="long",
            stop_percent=stop_percent
        )

        self.assertEqual(stop_price, expected_stop_price)

        # Test fixed stop loss for short position
        # Expected stop price for short: $100 * (1 + 0.05) = $105
        expected_stop_price = 105.0

        stop_price = self.stop_loss_manager.calculate_fixed_stop(
            entry_price=entry_price,
            direction="short",
            stop_percent=stop_percent
        )

        self.assertEqual(stop_price, expected_stop_price)

    def test_trailing_stop_loss(self):
        # Test trailing stop loss for long position
        entry_price = 100.0
        current_price = 110.0  # Price moved up 10%
        trailing_percent = 0.03  # 3% trailing stop

        # Expected stop price for long: $110 * (1 - 0.03) = $106.7
        expected_stop_price = 106.7

        stop_price = self.stop_loss_manager.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            direction="long",
            trailing_percent=trailing_percent
        )

        self.assertAlmostEqual(stop_price, expected_stop_price, places=1)

        # Test trailing stop loss for short position
        current_price = 90.0  # Price moved down 10%

        # Expected stop price for short: $90 * (1 + 0.03) = $92.7
        expected_stop_price = 92.7

        stop_price = self.stop_loss_manager.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            direction="short",
            trailing_percent=trailing_percent
        )

        self.assertAlmostEqual(stop_price, expected_stop_price, places=1)

    def test_profit_target(self):
        # Test profit target for long position
        entry_price = 100.0
        stop_price = 95.0  # $5 risk
        risk_reward_ratio = 3.0  # 3:1 reward-to-risk

        # Expected target for long: $100 + ($5 * 3) = $115
        expected_target = 115.0

        target_price = self.stop_loss_manager.calculate_profit_target(
            entry_price=entry_price,
            stop_price=stop_price,
            direction="long",
            risk_reward_ratio=risk_reward_ratio
        )

        self.assertEqual(target_price, expected_target)

        # Test profit target for short position
        # Expected target for short: $100 - ($5 * 3) = $85
        expected_target = 85.0

        target_price = self.stop_loss_manager.calculate_profit_target(
            entry_price=entry_price,
            stop_price=105.0,  # $5 risk
            direction="short",
            risk_reward_ratio=risk_reward_ratio
        )

        self.assertEqual(target_price, expected_target)

    def test_stop_loss_triggering(self):
        # Set up a stop loss
        trade_id = "test_trade_1"
        entry_price = 100.0
        entry_time = datetime.now()
        direction = "long"

        stop_details = self.stop_loss_manager.set_stop_loss(
            trade_id=trade_id,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_type=StopLossType.FIXED,
            stop_params={"stop_percent": 0.05}  # 5% stop loss
        )

        # Verify stop price
        self.assertEqual(stop_details["stop_price"], 95.0)

        # Test stop not triggered when price is above stop
        is_triggered, _ = self.stop_loss_manager.update_stop_loss(
            trade_id=trade_id,
            current_price=98.0
        )

        self.assertFalse(is_triggered)

        # Test stop triggered when price is below stop
        is_triggered, updated_stop = self.stop_loss_manager.update_stop_loss(
            trade_id=trade_id,
            current_price=94.0
        )

        self.assertTrue(is_triggered)
        self.assertFalse(updated_stop["is_active"])

class TestPortfolioRiskManager(unittest.TestCase):
    """Test cases for the PortfolioRiskManager class."""

    def setUp(self):
        self.portfolio_risk_manager = PortfolioRiskManager(
            max_portfolio_var_percent=0.02,
            max_drawdown_percent=0.15,
            max_sector_allocation=0.25,
            max_position_size=0.10,
            max_history_size=252
        )

    def test_sector_exposure_calculation(self):
        # Set up a portfolio with positions in different sectors
        positions = {
            "AAPL": {"market_value": 10000.0, "sector": "Technology"},
            "MSFT": {"market_value": 15000.0, "sector": "Technology"},
            "JPM": {"market_value": 8000.0, "sector": "Financials"},
            "XOM": {"market_value": 7000.0, "sector": "Energy"}
        }

        portfolio_value = 100000.0

        # Update portfolio
        self.portfolio_risk_manager.update_portfolio(
            positions=positions,
            portfolio_value=portfolio_value
        )

        # Check sector allocations
        expected_technology_exposure = 0.25  # (10000 + 15000) / 100000
        expected_financials_exposure = 0.08  # 8000 / 100000
        expected_energy_exposure = 0.07  # 7000 / 100000

        self.assertAlmostEqual(
            self.portfolio_risk_manager.sector_allocations.get("Technology", 0),
            expected_technology_exposure
        )

        self.assertAlmostEqual(
            self.portfolio_risk_manager.sector_allocations.get("Financials", 0),
            expected_financials_exposure
        )

        self.assertAlmostEqual(
            self.portfolio_risk_manager.sector_allocations.get("Energy", 0),
            expected_energy_exposure
        )

    def test_max_position_size_calculation(self):
        # Set up a portfolio with existing positions
        positions = {
            "AAPL": {"market_value": 8000.0, "sector": "Technology"},
            "MSFT": {"market_value": 7000.0, "sector": "Technology"}
        }

        portfolio_value = 100000.0

        # Update portfolio
        self.portfolio_risk_manager.update_portfolio(
            positions=positions,
            portfolio_value=portfolio_value
        )

        # Calculate max position size for a new technology stock
        symbol = "GOOGL"
        price = 2000.0
        sector = "Technology"

        # Expected max size based on sector allocation:
        # Max technology allocation: 0.25 * 100000 = 25000
        # Current technology allocation: 8000 + 7000 = 15000
        # Remaining capacity: 25000 - 15000 = 10000
        # Max units: 10000 / 2000 = 5 shares
        expected_max_size = 5.0

        max_size = self.portfolio_risk_manager.calculate_max_position_size(
            symbol=symbol,
            price=price,
            sector=sector
        )

        self.assertAlmostEqual(max_size, expected_max_size)

        # Calculate max position size for a new energy stock (no existing exposure)
        symbol = "XOM"
        price = 50.0
        sector = "Energy"

        # Expected max size based on position size limit (no sector constraint):
        # Max position size: 0.10 * 100000 = 10000
        # Max units: 10000 / 50 = 200 shares
        expected_max_size = 200.0

        max_size = self.portfolio_risk_manager.calculate_max_position_size(
            symbol=symbol,
            price=price,
            sector=sector
        )

        self.assertAlmostEqual(max_size, expected_max_size)

class TestAdvancedRiskManager(unittest.TestCase):
    """Test cases for the AdvancedRiskManager class."""

    def setUp(self):
        self.risk_manager = AdvancedRiskManager(
            initial_capital=100000.0,
            risk_per_trade=0.01,  # 1% risk per trade
            max_portfolio_var_percent=0.02,
            max_drawdown_percent=0.15,
            max_sector_allocation=0.25,
            max_position_size=0.10,
            default_stop_loss_percent=0.02,  # 2% default stop loss
            max_history_size=252
        )

    def test_position_size_calculation(self):
        # Test risk-based position sizing
        symbol = "AAPL"
        entry_price = 150.0
        stop_price = 147.0  # 2% stop loss

        # Expected position size: ($100,000 * 0.01) / ($150 - $147) = $1,000 / $3 = 333.33 shares
        expected_size = 333.33

        size, details = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            method="risk_based"
        )

        # Allow for some rounding differences
        self.assertAlmostEqual(size, round(expected_size, 8), places=2)

    def test_stop_loss_integration(self):
        # Set a stop loss
        trade_id = "test_trade_1"
        symbol = "AAPL"
        entry_price = 150.0
        entry_time = datetime.now()
        direction = "long"

        stop_details = self.risk_manager.set_stop_loss(
            trade_id=trade_id,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_type=StopLossType.FIXED,
            stop_params={"stop_percent": 0.02}  # 2% stop loss
        )

        # Verify stop price
        self.assertEqual(stop_details["stop_price"], 147.0)

        # Verify trade is tracked
        self.assertIn(trade_id, self.risk_manager.trades)
        self.assertEqual(self.risk_manager.trades[trade_id]["symbol"], symbol)

        # Test stop not triggered
        is_triggered, _ = self.risk_manager.update_stop_loss(
            trade_id=trade_id,
            current_price=148.0
        )

        self.assertFalse(is_triggered)
        self.assertEqual(self.risk_manager.trades[trade_id]["status"], "open")

        # Test stop triggered
        is_triggered, _ = self.risk_manager.update_stop_loss(
            trade_id=trade_id,
            current_price=146.0
        )

        self.assertTrue(is_triggered)
        self.assertEqual(self.risk_manager.trades[trade_id]["status"], "closed")
        self.assertEqual(self.risk_manager.trades[trade_id]["exit_reason"], "stop_loss")

    def test_portfolio_update(self):
        # Set up a portfolio with positions
        positions = {
            "AAPL": {"market_value": 10000.0, "sector": "Technology", "quantity": 66.67, "price": 150.0},
            "MSFT": {"market_value": 8000.0, "sector": "Technology", "quantity": 32.0, "price": 250.0},
            "JPM": {"market_value": 7000.0, "sector": "Financials", "quantity": 50.0, "price": 140.0}
        }

        portfolio_value = 100000.0

        # Update portfolio
        summary = self.risk_manager.update_portfolio(
            positions=positions,
            portfolio_value=portfolio_value
        )

        # Verify portfolio was updated
        self.assertEqual(self.risk_manager.current_capital, portfolio_value)
        self.assertEqual(len(self.risk_manager.positions), 3)

        # Verify summary contains risk metrics
        self.assertIn("risk_metrics", summary)
        self.assertIn("portfolio_value", summary["risk_metrics"])
        self.assertEqual(summary["risk_metrics"]["portfolio_value"], portfolio_value)

if __name__ == "__main__":
    unittest.main()
