"""Tests for the MultiPortfolioIntegration class.

This module contains unit tests for the MultiPortfolioIntegration class,
which extends the system to support multiple portfolios simultaneously.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.portfolio.multi_portfolio_integration import MultiPortfolioIntegration
from src.portfolio.portfolio_registry import PortfolioRegistry
from src.portfolio.portfolio_integration import PortfolioIntegration


class TestMultiPortfolioIntegration(unittest.TestCase):
    """Test cases for the MultiPortfolioIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.event_system = MagicMock()
        self.trading_engine = MagicMock()
        self.market_data_service = MagicMock()
        
        # Create a mock for the PortfolioRegistry class
        self.registry_patcher = patch('src.portfolio.multi_portfolio_integration.PortfolioRegistry')
        self.mock_registry_class = self.registry_patcher.start()
        self.mock_registry = self.mock_registry_class.return_value
        
        # Create a mock for the PortfolioIntegration class
        self.integration_patcher = patch('src.portfolio.multi_portfolio_integration.PortfolioIntegration')
        self.mock_integration_class = self.integration_patcher.start()
        self.mock_integration = self.mock_integration_class.return_value
        
        # Create the MultiPortfolioIntegration instance
        self.multi_integration = MultiPortfolioIntegration(
            event_system=self.event_system,
            trading_engine=self.trading_engine,
            market_data_service=self.market_data_service
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.registry_patcher.stop()
        self.integration_patcher.stop()
    
    def test_init(self):
        """Test initialization of MultiPortfolioIntegration."""
        # Check that the registry was created
        self.mock_registry_class.assert_called_once()
        
        # Check that the event subscriptions were set up
        self.event_system.subscribe.assert_called()
        
        # Check that the instance variables were set correctly
        self.assertEqual(self.multi_integration.event_system, self.event_system)
        self.assertEqual(self.multi_integration.trading_engine, self.trading_engine)
        self.assertEqual(self.multi_integration.market_data_service, self.market_data_service)
        self.assertEqual(self.multi_integration.registry, self.mock_registry)
        self.assertEqual(self.multi_integration.portfolio_integrations, {})
    
    def test_create_portfolio(self):
        """Test creating a portfolio."""
        # Set up the mock registry to return a portfolio ID and system components
        portfolio_id = "test_portfolio"
        self.mock_registry.create_portfolio.return_value = portfolio_id
        
        portfolio_manager = MagicMock()
        performance_calculator = MagicMock()
        tax_manager = MagicMock()
        allocation_manager = MagicMock()
        risk_manager = MagicMock()
        
        self.mock_registry.get_portfolio_system.return_value = {
            "portfolio_manager": portfolio_manager,
            "performance_calculator": performance_calculator,
            "tax_manager": tax_manager,
            "allocation_manager": allocation_manager,
            "risk_manager": risk_manager
        }
        
        # Call the method
        result = self.multi_integration.create_portfolio(
            portfolio_id=portfolio_id,
            initial_capital=100000.0,
            name="Test Portfolio",
            description="A test portfolio",
            tags=["test", "demo"]
        )
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.create_portfolio.assert_called_once_with(
            portfolio_id=portfolio_id,
            config=None,
            initial_capital=100000.0,
            name="Test Portfolio",
            description="A test portfolio",
            tags=["test", "demo"],
            auto_activate=True
        )
        
        # Check that the portfolio system was retrieved
        self.mock_registry.get_portfolio_system.assert_called_once_with(portfolio_id)
        
        # Check that a PortfolioIntegration was created
        self.mock_integration_class.assert_called_once_with(
            portfolio_manager=portfolio_manager,
            performance_calculator=performance_calculator,
            tax_manager=tax_manager,
            allocation_manager=allocation_manager,
            risk_manager=risk_manager,
            event_system=self.event_system,
            trading_engine=self.trading_engine,
            market_data_service=self.market_data_service
        )
        
        # Check that the integration was stored
        self.assertEqual(self.multi_integration.portfolio_integrations[portfolio_id], self.mock_integration)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the portfolio ID
        self.assertEqual(result, portfolio_id)
    
    def test_delete_portfolio(self):
        """Test deleting a portfolio."""
        # Set up the mock registry
        portfolio_id = "test_portfolio"
        self.mock_registry.portfolios = {portfolio_id: MagicMock()}
        self.mock_registry.delete_portfolio.return_value = True
        
        # Add a mock integration
        mock_integration = MagicMock()
        self.multi_integration.portfolio_integrations[portfolio_id] = mock_integration
        
        # Call the method
        result = self.multi_integration.delete_portfolio(portfolio_id)
        
        # Check that the integration's subscriptions were cleaned up
        mock_integration.cleanup_subscriptions.assert_called_once()
        
        # Check that the registry method was called
        self.mock_registry.delete_portfolio.assert_called_once_with(portfolio_id)
        
        # Check that the integration was removed
        self.assertNotIn(portfolio_id, self.multi_integration.portfolio_integrations)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_delete_nonexistent_portfolio(self):
        """Test deleting a portfolio that doesn't exist."""
        # Set up the mock registry
        portfolio_id = "nonexistent_portfolio"
        self.mock_registry.portfolios = {}
        
        # Call the method
        result = self.multi_integration.delete_portfolio(portfolio_id)
        
        # Check that the registry method was not called
        self.mock_registry.delete_portfolio.assert_not_called()
        
        # Check that no event was published
        self.event_system.publish.assert_not_called()
        
        # Check that the result is False
        self.assertFalse(result)
    
    def test_activate_portfolio(self):
        """Test activating a portfolio."""
        # Set up the mock registry
        portfolio_id = "test_portfolio"
        self.mock_registry.portfolios = {portfolio_id: MagicMock()}
        self.mock_registry.activate_portfolio.return_value = True
        
        # Call the method
        result = self.multi_integration.activate_portfolio(portfolio_id)
        
        # Check that the registry method was called
        self.mock_registry.activate_portfolio.assert_called_once_with(portfolio_id)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_deactivate_portfolio(self):
        """Test deactivating a portfolio."""
        # Set up the mock registry
        portfolio_id = "test_portfolio"
        self.mock_registry.portfolios = {portfolio_id: MagicMock()}
        self.mock_registry.deactivate_portfolio.return_value = True
        
        # Call the method
        result = self.multi_integration.deactivate_portfolio(portfolio_id)
        
        # Check that the registry method was called
        self.mock_registry.deactivate_portfolio.assert_called_once_with(portfolio_id)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_set_active_portfolio(self):
        """Test setting the active portfolio."""
        # Set up the mock registry
        portfolio_id = "test_portfolio"
        self.mock_registry.set_active_portfolio.return_value = True
        
        # Call the method
        result = self.multi_integration.set_active_portfolio(portfolio_id)
        
        # Check that the registry method was called
        self.mock_registry.set_active_portfolio.assert_called_once_with(portfolio_id)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_get_active_portfolio_id(self):
        """Test getting the active portfolio ID."""
        # Set up the mock registry
        portfolio_id = "test_portfolio"
        self.mock_registry.get_active_portfolio_id.return_value = portfolio_id
        
        # Call the method
        result = self.multi_integration.get_active_portfolio_id()
        
        # Check that the registry method was called
        self.mock_registry.get_active_portfolio_id.assert_called_once()
        
        # Check that the result is the portfolio ID
        self.assertEqual(result, portfolio_id)
    
    def test_get_portfolio_integration(self):
        """Test getting a portfolio integration."""
        # Set up the mock registry and integrations
        portfolio_id = "test_portfolio"
        self.mock_registry.get_active_portfolio_id.return_value = portfolio_id
        
        mock_integration = MagicMock()
        self.multi_integration.portfolio_integrations[portfolio_id] = mock_integration
        
        # Call the method with no portfolio ID (should use active)
        result = self.multi_integration.get_portfolio_integration()
        
        # Check that the registry method was called
        self.mock_registry.get_active_portfolio_id.assert_called_once()
        
        # Check that the result is the integration
        self.assertEqual(result, mock_integration)
        
        # Call the method with a specific portfolio ID
        other_portfolio_id = "other_portfolio"
        other_mock_integration = MagicMock()
        self.multi_integration.portfolio_integrations[other_portfolio_id] = other_mock_integration
        
        result = self.multi_integration.get_portfolio_integration(other_portfolio_id)
        
        # Check that the result is the correct integration
        self.assertEqual(result, other_mock_integration)
    
    def test_create_portfolio_group(self):
        """Test creating a portfolio group."""
        # Set up the mock registry
        group_id = "test_group"
        self.mock_registry.create_portfolio_group.return_value = group_id
        
        # Call the method
        result = self.multi_integration.create_portfolio_group(
            name="Test Group",
            portfolio_ids=["portfolio1", "portfolio2"],
            description="A test group",
            allocation={"portfolio1": 0.6, "portfolio2": 0.4}
        )
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.create_portfolio_group.assert_called_once_with(
            name="Test Group",
            portfolio_ids=["portfolio1", "portfolio2"],
            group_id=None,
            description="A test group",
            allocation={"portfolio1": 0.6, "portfolio2": 0.4}
        )
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the group ID
        self.assertEqual(result, group_id)
    
    def test_delete_portfolio_group(self):
        """Test deleting a portfolio group."""
        # Set up the mock registry
        group_id = "test_group"
        self.mock_registry.delete_portfolio_group.return_value = True
        
        # Call the method
        result = self.multi_integration.delete_portfolio_group(group_id)
        
        # Check that the registry method was called
        self.mock_registry.delete_portfolio_group.assert_called_once_with(group_id)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_add_to_group(self):
        """Test adding a portfolio to a group."""
        # Set up the mock registry
        group_id = "test_group"
        portfolio_id = "test_portfolio"
        allocation = 0.5
        self.mock_registry.add_to_group.return_value = True
        
        # Call the method
        result = self.multi_integration.add_to_group(group_id, portfolio_id, allocation)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.add_to_group.assert_called_once_with(group_id, portfolio_id, allocation)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_remove_from_group(self):
        """Test removing a portfolio from a group."""
        # Set up the mock registry
        group_id = "test_group"
        portfolio_id = "test_portfolio"
        self.mock_registry.remove_from_group.return_value = True
        
        # Call the method
        result = self.multi_integration.remove_from_group(group_id, portfolio_id)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.remove_from_group.assert_called_once_with(group_id, portfolio_id)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is True
        self.assertTrue(result)
    
    def test_compare_portfolios(self):
        """Test comparing portfolios."""
        # Set up the mock registry
        portfolio_ids = ["portfolio1", "portfolio2"]
        metrics = ["returns", "volatility", "sharpe_ratio"]
        comparison_result = {"returns": {"portfolio1": 0.1, "portfolio2": 0.15}}
        self.mock_registry.compare_portfolios.return_value = comparison_result
        
        # Call the method
        result = self.multi_integration.compare_portfolios(portfolio_ids, metrics)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.compare_portfolios.assert_called_once_with(portfolio_ids, metrics)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the comparison result
        self.assertEqual(result, comparison_result)
    
    def test_calculate_correlation(self):
        """Test calculating correlation between portfolios."""
        # Set up the mock registry
        portfolio_ids = ["portfolio1", "portfolio2", "portfolio3"]
        correlation_result = {"matrix": [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]]}
        self.mock_registry.calculate_correlation.return_value = correlation_result
        
        # Call the method
        result = self.multi_integration.calculate_correlation(portfolio_ids)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.calculate_correlation.assert_called_once_with(portfolio_ids)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the correlation result
        self.assertEqual(result, correlation_result)
    
    def test_analyze_diversification(self):
        """Test analyzing diversification across portfolios."""
        # Set up the mock registry
        portfolio_ids = ["portfolio1", "portfolio2"]
        diversification_result = {"overlap": 0.3, "unique_assets": 15}
        self.mock_registry.analyze_diversification.return_value = diversification_result
        
        # Call the method
        result = self.multi_integration.analyze_diversification(portfolio_ids)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.analyze_diversification.assert_called_once_with(portfolio_ids)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the diversification result
        self.assertEqual(result, diversification_result)
    
    def test_generate_consolidated_report(self):
        """Test generating a consolidated report."""
        # Set up the mock registry
        portfolio_ids = ["portfolio1", "portfolio2"]
        report_result = {"portfolio_count": 2, "total_value": 250000.0}
        self.mock_registry.generate_consolidated_report.return_value = report_result
        self.mock_registry.active_portfolios = {"portfolio1": True, "portfolio2": True}
        
        # Call the method
        result = self.multi_integration.generate_consolidated_report(portfolio_ids)
        
        # Check that the registry method was called with the correct arguments
        self.mock_registry.generate_consolidated_report.assert_called_once_with(portfolio_ids)
        
        # Check that an event was published
        self.event_system.publish.assert_called_once()
        
        # Check that the result is the report result
        self.assertEqual(result, report_result)
    
    def test_cleanup_subscriptions(self):
        """Test cleaning up event subscriptions."""
        # Set up subscriptions
        subscription1 = MagicMock()
        subscription2 = MagicMock()
        self.multi_integration.subscriptions = {
            "type1": [subscription1],
            "type2": [subscription2]
        }
        
        # Add a mock integration
        portfolio_id = "test_portfolio"
        mock_integration = MagicMock()
        self.multi_integration.portfolio_integrations[portfolio_id] = mock_integration
        
        # Call the method
        self.multi_integration.cleanup_subscriptions()
        
        # Check that unsubscribe was called for each subscription
        self.event_system.unsubscribe.assert_any_call(subscription1)
        self.event_system.unsubscribe.assert_any_call(subscription2)
        
        # Check that cleanup_subscriptions was called for the integration
        mock_integration.cleanup_subscriptions.assert_called_once()
        
        # Check that the subscriptions were cleared
        self.assertEqual(self.multi_integration.subscriptions, {})
    
    def test_event_handlers(self):
        """Test event handlers."""
        # Create mock events
        portfolio_create_event = MagicMock()
        portfolio_create_event.data = {
            "portfolio_id": "test_portfolio",
            "initial_capital": 100000.0,
            "name": "Test Portfolio",
            "description": "A test portfolio",
            "tags": ["test", "demo"],
            "auto_activate": True
        }
        
        portfolio_delete_event = MagicMock()
        portfolio_delete_event.data = {"portfolio_id": "test_portfolio"}
        
        portfolio_activate_event = MagicMock()
        portfolio_activate_event.data = {"portfolio_id": "test_portfolio"}
        
        portfolio_deactivate_event = MagicMock()
        portfolio_deactivate_event.data = {"portfolio_id": "test_portfolio"}
        
        group_create_event = MagicMock()
        group_create_event.data = {
            "name": "Test Group",
            "portfolio_ids": ["portfolio1", "portfolio2"],
            "description": "A test group",
            "allocation": {"portfolio1": 0.6, "portfolio2": 0.4}
        }
        
        group_delete_event = MagicMock()
        group_delete_event.data = {"group_id": "test_group"}
        
        cross_portfolio_analysis_event = MagicMock()
        cross_portfolio_analysis_event.data = {
            "analysis_type": "comparison",
            "portfolio_ids": ["portfolio1", "portfolio2"],
            "metrics": ["returns", "volatility"]
        }
        
        consolidated_report_request_event = MagicMock()
        consolidated_report_request_event.data = {
            "portfolio_ids": ["portfolio1", "portfolio2"]
        }
        
        # Mock the methods that the handlers call
        self.multi_integration.create_portfolio = MagicMock()
        self.multi_integration.delete_portfolio = MagicMock()
        self.multi_integration.activate_portfolio = MagicMock()
        self.multi_integration.deactivate_portfolio = MagicMock()
        self.multi_integration.create_portfolio_group = MagicMock()
        self.multi_integration.delete_portfolio_group = MagicMock()
        self.multi_integration.compare_portfolios = MagicMock()
        self.multi_integration.calculate_correlation = MagicMock()
        self.multi_integration.analyze_diversification = MagicMock()
        self.multi_integration.generate_consolidated_report = MagicMock()
        
        # Call the handlers
        self.multi_integration._handle_portfolio_create_event(portfolio_create_event)
        self.multi_integration._handle_portfolio_delete_event(portfolio_delete_event)
        self.multi_integration._handle_portfolio_activate_event(portfolio_activate_event)
        self.multi_integration._handle_portfolio_deactivate_event(portfolio_deactivate_event)
        self.multi_integration._handle_group_create_event(group_create_event)
        self.multi_integration._handle_group_delete_event(group_delete_event)
        self.multi_integration._handle_cross_portfolio_analysis_event(cross_portfolio_analysis_event)
        self.multi_integration._handle_consolidated_report_request_event(consolidated_report_request_event)
        
        # Check that the methods were called with the correct arguments
        self.multi_integration.create_portfolio.assert_called_once_with(
            portfolio_id="test_portfolio",
            config=None,
            initial_capital=100000.0,
            name="Test Portfolio",
            description="A test portfolio",
            tags=["test", "demo"],
            auto_activate=True
        )
        
        self.multi_integration.delete_portfolio.assert_called_once_with("test_portfolio")
        self.multi_integration.activate_portfolio.assert_called_once_with("test_portfolio")
        self.multi_integration.deactivate_portfolio.assert_called_once_with("test_portfolio")
        
        self.multi_integration.create_portfolio_group.assert_called_once_with(
            name="Test Group",
            portfolio_ids=["portfolio1", "portfolio2"],
            group_id=None,
            description="A test group",
            allocation={"portfolio1": 0.6, "portfolio2": 0.4}
        )
        
        self.multi_integration.delete_portfolio_group.assert_called_once_with("test_group")
        
        self.multi_integration.compare_portfolios.assert_called_once_with(
            ["portfolio1", "portfolio2"],
            ["returns", "volatility"]
        )
        
        self.multi_integration.generate_consolidated_report.assert_called_once_with(
            ["portfolio1", "portfolio2"]
        )


if __name__ == "__main__":
    unittest.main()