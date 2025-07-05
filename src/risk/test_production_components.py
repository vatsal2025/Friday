import unittest
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.risk.production_config import RiskManagementProductionConfig, load_production_config, save_production_config
from src.risk.risk_management_factory import RiskManagementFactory
from src.risk.risk_management_service import RiskManagementService

class TestProductionConfig(unittest.TestCase):
    """Test cases for the production configuration module."""

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_default_config(self):
        """Test that the default configuration has expected values."""
        config = RiskManagementProductionConfig()

        # Check some key values
        self.assertEqual(config.risk_per_trade, 0.005)
        self.assertEqual(config.max_portfolio_var_percent, 0.015)
        self.assertEqual(config.max_drawdown_percent, 0.10)
        self.assertEqual(config.max_sector_allocation, 0.20)
        self.assertEqual(config.max_position_size, 0.05)
        self.assertEqual(config.max_history_size, 252)

        # Check position limits for a specific asset class
        self.assertIn("equities", config.position_limits_by_asset)
        self.assertEqual(config.position_limits_by_asset["equities"]["max_position_percentage"], 0.05)

    def test_save_and_load_config(self):
        """Test saving and loading configuration to/from a file."""
        # Create a custom configuration
        config = RiskManagementProductionConfig()
        config.risk_per_trade = 0.01
        config.max_portfolio_var_percent = 0.02

        # Save the configuration
        success = save_production_config(config, self.config_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.config_path))

        # Load the configuration
        loaded_config = load_production_config(self.config_path)

        # Check that the loaded configuration has the expected values
        self.assertEqual(loaded_config.risk_per_trade, 0.01)
        self.assertEqual(loaded_config.max_portfolio_var_percent, 0.02)

class TestRiskManagementFactory(unittest.TestCase):
    """Test cases for the risk management factory."""

    def test_create_position_sizer(self):
        """Test creating a position sizer with the factory."""
        factory = RiskManagementFactory()
        position_sizer = factory.create_position_sizer()

        # Check that the position sizer has the expected configuration
        self.assertEqual(position_sizer.default_risk_per_trade, factory.config.risk_per_trade)
        self.assertEqual(position_sizer.max_position_size_percentage, factory.config.max_position_size_percentage)
        self.assertEqual(position_sizer.max_position_value, factory.config.max_position_value)

    def test_create_stop_loss_manager(self):
        """Test creating a stop loss manager with the factory."""
        factory = RiskManagementFactory()
        stop_loss_manager = factory.create_stop_loss_manager()

        # Check that the stop loss manager has the expected configuration
        self.assertEqual(stop_loss_manager.default_stop_loss_percent, factory.config.default_stop_loss_percent)
        self.assertEqual(stop_loss_manager.default_trailing_percent, factory.config.default_trailing_percent)
        self.assertEqual(stop_loss_manager.default_atr_multiplier, factory.config.default_atr_multiplier)

    def test_create_portfolio_risk_manager(self):
        """Test creating a portfolio risk manager with the factory."""
        factory = RiskManagementFactory()
        portfolio_risk_manager = factory.create_portfolio_risk_manager()

        # Check that the portfolio risk manager has the expected configuration
        self.assertEqual(portfolio_risk_manager.max_var_percent, factory.config.max_portfolio_var_percent)
        self.assertEqual(portfolio_risk_manager.max_drawdown_percent, factory.config.max_drawdown_percent)
        self.assertEqual(portfolio_risk_manager.max_sector_allocation, factory.config.max_sector_allocation)
        self.assertEqual(portfolio_risk_manager.max_position_size, factory.config.max_position_size)
        self.assertEqual(portfolio_risk_manager.max_history_size, factory.config.max_history_size)

    def test_create_circuit_breakers(self):
        """Test creating circuit breakers with the factory."""
        factory = RiskManagementFactory()
        circuit_breakers = factory.create_circuit_breakers()

        # Check that the circuit breakers were created
        self.assertGreater(len(circuit_breakers), 0)

        # Check that at least one market circuit breaker was created
        market_cbs = [cb for cb in circuit_breakers if cb.cb_type.name == "MARKET"]
        self.assertGreater(len(market_cbs), 0)

        # Check that at least one account circuit breaker was created
        account_cbs = [cb for cb in circuit_breakers if cb.cb_type.name == "ACCOUNT"]
        self.assertGreater(len(account_cbs), 0)

    def test_create_advanced_risk_manager(self):
        """Test creating an advanced risk manager with the factory."""
        factory = RiskManagementFactory()
        emergency_handler = MagicMock()
        advanced_risk_manager = factory.create_advanced_risk_manager([emergency_handler])

        # Check that the advanced risk manager has all the expected components
        self.assertIsNotNone(advanced_risk_manager.position_sizer)
        self.assertIsNotNone(advanced_risk_manager.stop_loss_manager)
        self.assertIsNotNone(advanced_risk_manager.portfolio_risk_manager)
        self.assertIsNotNone(advanced_risk_manager.circuit_breaker_manager)

class TestRiskManagementService(unittest.TestCase):
    """Test cases for the risk management service."""

    def setUp(self):
        # Create a temporary directory for persistence
        self.test_dir = tempfile.mkdtemp()

        # Create a custom configuration for testing
        self.config = RiskManagementProductionConfig()
        self.config.persistence_path = self.test_dir
        self.config.persistence_enabled = True
        self.config.risk_metrics_calculation_interval = 0.1  # Fast interval for testing
        self.config.persistence_interval = 0.1  # Fast interval for testing

        # Create the service with the test configuration
        self.service = RiskManagementService(self.config)

    def tearDown(self):
        # Stop the service if it's running
        if self.service.is_running:
            self.service.stop()

        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_service_start_stop(self):
        """Test starting and stopping the service."""
        # Start the service
        self.service.start()
        self.assertTrue(self.service.is_running)
        self.assertIsNotNone(self.service.risk_manager)

        # Check that the monitoring threads are running
        self.assertIsNotNone(self.service.metrics_thread)
        self.assertTrue(self.service.metrics_thread.is_alive())

        self.assertIsNotNone(self.service.persistence_thread)
        self.assertTrue(self.service.persistence_thread.is_alive())

        # Stop the service
        self.service.stop()
        self.assertFalse(self.service.is_running)

    def test_alert_callbacks(self):
        """Test registering and triggering alert callbacks."""
        # Create a mock callback
        callback = MagicMock()

        # Register the callback
        self.service.register_alert_callback(callback)

        # Start the service
        self.service.start()

        # Manually trigger an alert
        test_alert = {"type": "test_alert", "severity": "medium", "message": "Test alert"}
        self.service._handle_risk_alert(test_alert)

        # Check that the callback was called with the alert
        callback.assert_called_once_with(test_alert)

        # Stop the service
        self.service.stop()

    @patch('src.risk.risk_management_service.pickle.dump')
    def test_persistence(self, mock_pickle_dump):
        """Test that the service persists its state."""
        # Start the service
        self.service.start()

        # Manually trigger persistence
        self.service._persist_state()

        # Check that pickle.dump was called to persist the state
        mock_pickle_dump.assert_called_once()

        # Stop the service
        self.service.stop()

    def test_health_status(self):
        """Test getting the health status of the service."""
        # Start the service
        self.service.start()

        # Get the health status
        health = self.service.get_health_status()

        # Check that the health status has the expected fields
        self.assertEqual(health["status"], "running")
        self.assertIn("start_time", health)

        # Stop the service
        self.service.stop()

        # Get the health status again
        health = self.service.get_health_status()

        # Check that the status is now stopped
        self.assertEqual(health["status"], "stopped")

if __name__ == "__main__":
    unittest.main()
