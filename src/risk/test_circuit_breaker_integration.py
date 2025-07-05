import unittest
from datetime import datetime, timedelta
import time

from .circuit_breaker import (
    CircuitBreakerType, CircuitBreakerLevel, CircuitBreakerStatus,
    CircuitBreakerEvent, MarketWideCircuitBreaker, AccountCircuitBreaker,
    CircuitBreakerManager, create_circuit_breaker_manager
)
from .advanced_risk_manager import AdvancedRiskManager
from .position_sizer import PositionSizer
from .stop_loss_manager import StopLossManager, StopLossType
from .portfolio_risk_manager import PortfolioRiskManager


class MockEmergencyHandler:
    """Mock emergency handler for testing."""

    def __init__(self):
        self.events = []

    def _handle_circuit_breaker(self, event_data):
        self.events.append(event_data)


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Test the integration of circuit breakers with the risk management system."""

    def setUp(self):
        # Create mock emergency handler
        self.emergency_handler = MockEmergencyHandler()

        # Create circuit breaker manager
        self.cb_manager = create_circuit_breaker_manager(self.emergency_handler)

        # Create advanced risk manager
        self.risk_manager = AdvancedRiskManager(
            initial_capital=100000.0,
            risk_per_trade=0.01,
            max_portfolio_var_percent=0.02,
            max_drawdown_percent=0.15,
            max_sector_exposure=0.25,  # Used as max_sector_allocation in PortfolioRiskManager
            max_asset_exposure=0.10,  # Used as max_position_size in PortfolioRiskManager
            default_stop_loss_percent=0.02,
            circuit_breaker_manager=self.cb_manager
        )

        # Add market circuit breaker
        market_cb = MarketWideCircuitBreaker(
            market="SPY",
            level_1_percent=5.0,  # 5% drop for Level 1
            level_2_percent=10.0,  # 10% drop for Level 2
            level_3_percent=15.0,  # 15% drop for Level 3
            reference_index="SPY",
            level_1_duration_minutes=15,
            level_2_duration_minutes=30,
            level_3_duration_minutes=0  # Rest of day
        )
        market_cb.update_reference_value(400.0)  # Set reference value
        self.cb_manager.add_market_circuit_breaker(market_cb)

        # Add account circuit breaker
        account_cb = AccountCircuitBreaker(
            account_id="test_account",
            daily_loss_percent_warning=3.0,  # 3% daily loss warning
            daily_loss_percent_soft=5.0,     # 5% daily loss soft limit
            daily_loss_percent_hard=7.0      # 7% daily loss hard limit
        )
        account_cb.update_starting_balance(100000.0)  # Set starting balance
        self.cb_manager.add_account_circuit_breaker(account_cb)

    def tearDown(self):
        # Stop circuit breaker manager
        self.cb_manager.stop()

    def test_market_circuit_breaker_integration(self):
        """Test market circuit breaker integration with risk manager."""
        # Update market data with a 6% drop (should trigger Level 1)
        market_data = {
            "SPY": {
                "price": 376.0,  # 6% drop from 400
                "volume": 1000000,
                "volatility": 0.02
            }
        }

        # Update market data in risk manager
        self.risk_manager.update_market_data(market_data)

        # Give time for async processing
        time.sleep(0.1)

        # Check that circuit breaker was triggered
        active_cbs = self.cb_manager.get_active_circuit_breakers()
        self.assertEqual(len(active_cbs), 1)

        # Check that emergency handler was notified
        self.assertEqual(len(self.emergency_handler.events), 1)
        self.assertEqual(self.emergency_handler.events[0]["circuit_breaker_type"],
                         CircuitBreakerType.MARKET_WIDE.value)
        self.assertEqual(self.emergency_handler.events[0]["level"],
                         CircuitBreakerLevel.LEVEL_1.value)

        # Check that risk manager has the circuit breaker info
        cb_status = self.risk_manager.get_circuit_breaker_status()
        self.assertTrue(cb_status["market_circuit_breakers_triggered"])
        self.assertEqual(len(cb_status["active_circuit_breakers"]), 1)

    def test_account_circuit_breaker_integration(self):
        """Test account circuit breaker integration with risk manager."""
        # Set up portfolio with a loss
        positions = {
            "AAPL": {"market_value": 20000.0, "sector": "Technology", "quantity": 100, "price": 200.0},
            "MSFT": {"market_value": 30000.0, "sector": "Technology", "quantity": 100, "price": 300.0},
            "GOOGL": {"market_value": 40000.0, "sector": "Technology", "quantity": 20, "price": 2000.0}
        }

        # Update portfolio with a 6% loss (should trigger soft limit)
        portfolio_value = 94000.0  # 6% loss from 100000

        # Update portfolio in risk manager
        self.risk_manager.update_portfolio(positions, portfolio_value)

        # Update account PnL in circuit breaker manager
        self.cb_manager.update_account_pnl("test_account", -6000.0)

        # Give time for async processing
        time.sleep(0.1)

        # Check that circuit breaker was triggered
        active_cbs = self.cb_manager.get_active_circuit_breakers()
        self.assertEqual(len(active_cbs), 1)

        # Check that emergency handler was notified
        self.assertEqual(len(self.emergency_handler.events), 1)
        self.assertEqual(self.emergency_handler.events[0]["circuit_breaker_type"],
                         CircuitBreakerType.ACCOUNT_LEVEL.value)
        self.assertEqual(self.emergency_handler.events[0]["level"],
                         CircuitBreakerLevel.SOFT_LIMIT.value)

        # Check that risk manager has the circuit breaker info
        cb_status = self.risk_manager.get_circuit_breaker_status()
        self.assertTrue(cb_status["account_circuit_breakers_triggered"])
        self.assertEqual(len(cb_status["active_circuit_breakers"]), 1)

    def test_position_sizing_with_circuit_breaker(self):
        """Test position sizing adjustment when circuit breaker is active."""
        # Calculate normal position size
        symbol = "AAPL"
        entry_price = 150.0
        stop_price = 147.0  # 2% stop loss

        normal_size, _ = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            method="risk_based"
        )

        # Trigger a market circuit breaker
        market_data = {
            "SPY": {
                "price": 360.0,  # 10% drop from 400 (Level 2)
                "volume": 1000000,
                "volatility": 0.03
            }
        }

        # Update market data in risk manager
        self.risk_manager.update_market_data(market_data)

        # Give time for async processing
        time.sleep(0.1)

        # Calculate position size with circuit breaker active
        cb_size, details = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            method="risk_based"
        )

        # Position size should be reduced due to circuit breaker
        self.assertLess(cb_size, normal_size)
        self.assertIn("circuit_breaker_adjustment", details)
        self.assertTrue(details["circuit_breaker_adjustment"])

    def test_stop_loss_adjustment_with_circuit_breaker(self):
        """Test stop loss adjustment when circuit breaker is active."""
        # Set a normal stop loss
        trade_id = "test_trade_1"
        symbol = "AAPL"
        entry_price = 150.0
        entry_time = datetime.now()
        direction = "long"

        normal_stop = self.risk_manager.set_stop_loss(
            trade_id=trade_id,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_type=StopLossType.FIXED,
            stop_params={"stop_percent": 0.02}  # 2% stop loss
        )

        # Trigger a market circuit breaker
        market_data = {
            "SPY": {
                "price": 360.0,  # 10% drop from 400 (Level 2)
                "volume": 1000000,
                "volatility": 0.03
            }
        }

        # Update market data in risk manager
        self.risk_manager.update_market_data(market_data)

        # Give time for async processing
        time.sleep(0.1)

        # Set another stop loss with circuit breaker active
        trade_id_2 = "test_trade_2"
        cb_stop = self.risk_manager.set_stop_loss(
            trade_id=trade_id_2,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_type=StopLossType.FIXED,
            stop_params={"stop_percent": 0.02}  # 2% stop loss
        )

        # Stop loss should be tighter due to circuit breaker
        self.assertGreater(cb_stop["stop_price"], normal_stop["stop_price"])
        self.assertIn("circuit_breaker_adjustment", cb_stop)
        self.assertTrue(cb_stop["circuit_breaker_adjustment"])

    def test_risk_alerts_with_circuit_breaker(self):
        """Test risk alerts when circuit breaker is active."""
        # Set up portfolio with high technology exposure
        positions = {
            "AAPL": {"market_value": 20000.0, "sector": "Technology", "quantity": 100, "price": 200.0},
            "MSFT": {"market_value": 30000.0, "sector": "Technology", "quantity": 100, "price": 300.0},
            "GOOGL": {"market_value": 40000.0, "sector": "Technology", "quantity": 20, "price": 2000.0}
        }

        # Update portfolio
        portfolio_value = 100000.0
        self.risk_manager.update_portfolio(positions, portfolio_value)

        # Check risk alerts (should have sector exposure alert)
        alerts = self.risk_manager.check_risk_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "sector_exposure_breach")

        # Trigger a market circuit breaker
        market_data = {
            "SPY": {
                "price": 360.0,  # 10% drop from 400 (Level 2)
                "volume": 1000000,
                "volatility": 0.03
            }
        }

        # Update market data in risk manager
        self.risk_manager.update_market_data(market_data)

        # Give time for async processing
        time.sleep(0.1)

        # Check risk alerts with circuit breaker active
        alerts = self.risk_manager.check_risk_alerts()

        # Should have more alerts due to circuit breaker
        self.assertGreater(len(alerts), 1)

        # Should have a market volatility alert
        volatility_alerts = [a for a in alerts if a["type"] == "market_volatility"]
        self.assertEqual(len(volatility_alerts), 1)


if __name__ == "__main__":
    unittest.main()
