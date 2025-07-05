"""Unit tests for the PortfolioRegistry class.

This module contains tests for the PortfolioRegistry class, which is responsible
for managing multiple portfolio instances, including creation, activation,
deactivation, deletion, grouping, and cross-portfolio operations.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from src.portfolio.portfolio_registry import PortfolioRegistry, PortfolioGroup
from src.portfolio.portfolio_manager import PortfolioManager


class TestPortfolioRegistry(unittest.TestCase):
    """Test cases for the PortfolioRegistry class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_system_mock = Mock()
        self.portfolio_factory_mock = Mock()
        
        # Configure the portfolio factory mock to return a mock portfolio manager
        self.portfolio_manager_mock = Mock(spec=PortfolioManager)
        self.portfolio_factory_mock.create_portfolio_manager.return_value = self.portfolio_manager_mock
        self.portfolio_factory_mock.create_performance_calculator.return_value = Mock()
        self.portfolio_factory_mock.create_tax_manager.return_value = Mock()
        self.portfolio_factory_mock.create_allocation_manager.return_value = Mock()
        self.portfolio_factory_mock.create_risk_manager.return_value = Mock()
        
        # Create the registry
        self.registry = PortfolioRegistry(
            event_system=self.event_system_mock,
            portfolio_factory=self.portfolio_factory_mock,
            default_config={}
        )

    def test_create_portfolio(self):
        """Test creating a new portfolio."""
        # Act
        portfolio_id = self.registry.create_portfolio(
            name="Test Portfolio",
            initial_capital=100000.0,
            description="Test portfolio description",
            tags=["test", "example"],
            config={}
        )
        
        # Assert
        self.assertIsNotNone(portfolio_id)
        self.assertIn(portfolio_id, self.registry.get_all_portfolio_ids())
        self.portfolio_factory_mock.create_portfolio_manager.assert_called_once()
        self.assertEqual(self.registry.get_portfolio_name(portfolio_id), "Test Portfolio")
        self.assertEqual(self.registry.get_portfolio_description(portfolio_id), "Test portfolio description")
        self.assertEqual(self.registry.get_portfolio_tags(portfolio_id), ["test", "example"])

    def test_delete_portfolio(self):
        """Test deleting a portfolio."""
        # Arrange
        portfolio_id = self.registry.create_portfolio(
            name="Test Portfolio",
            initial_capital=100000.0
        )
        
        # Act
        self.registry.delete_portfolio(portfolio_id)
        
        # Assert
        self.assertNotIn(portfolio_id, self.registry.get_all_portfolio_ids())

    def test_activate_deactivate_portfolio(self):
        """Test activating and deactivating a portfolio."""
        # Arrange
        portfolio_id = self.registry.create_portfolio(
            name="Test Portfolio",
            initial_capital=100000.0
        )
        
        # Act - Deactivate
        self.registry.deactivate_portfolio(portfolio_id)
        
        # Assert
        self.assertNotIn(portfolio_id, self.registry.get_active_portfolio_ids())
        self.assertIn(portfolio_id, self.registry.get_all_portfolio_ids())
        
        # Act - Activate
        self.registry.activate_portfolio(portfolio_id)
        
        # Assert
        self.assertIn(portfolio_id, self.registry.get_active_portfolio_ids())

    def test_set_active_portfolio(self):
        """Test setting the active portfolio."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Act
        self.registry.set_active_portfolio(portfolio_id2)
        
        # Assert
        self.assertEqual(self.registry.get_active_portfolio_id(), portfolio_id2)

    def test_create_portfolio_group(self):
        """Test creating a portfolio group."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Act
        group_id = self.registry.create_portfolio_group(
            name="Test Group",
            portfolio_ids=[portfolio_id1, portfolio_id2],
            description="Test group description",
            allocation={
                portfolio_id1: 0.4,
                portfolio_id2: 0.6
            }
        )
        
        # Assert
        self.assertIsNotNone(group_id)
        self.assertIn(group_id, self.registry.get_all_group_ids())
        group = self.registry.get_portfolio_group(group_id)
        self.assertIsInstance(group, PortfolioGroup)
        self.assertEqual(group.name, "Test Group")
        self.assertEqual(group.description, "Test group description")
        self.assertEqual(len(group.portfolio_ids), 2)
        self.assertIn(portfolio_id1, group.portfolio_ids)
        self.assertIn(portfolio_id2, group.portfolio_ids)
        self.assertEqual(group.allocation[portfolio_id1], 0.4)
        self.assertEqual(group.allocation[portfolio_id2], 0.6)

    def test_delete_portfolio_group(self):
        """Test deleting a portfolio group."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        group_id = self.registry.create_portfolio_group(
            name="Test Group",
            portfolio_ids=[portfolio_id1, portfolio_id2]
        )
        
        # Act
        self.registry.delete_portfolio_group(group_id)
        
        # Assert
        self.assertNotIn(group_id, self.registry.get_all_group_ids())
        # Portfolios should still exist
        self.assertIn(portfolio_id1, self.registry.get_all_portfolio_ids())
        self.assertIn(portfolio_id2, self.registry.get_all_portfolio_ids())

    def test_add_to_group(self):
        """Test adding a portfolio to a group."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        group_id = self.registry.create_portfolio_group(
            name="Test Group",
            portfolio_ids=[portfolio_id1]
        )
        
        # Act
        self.registry.add_to_group(group_id, portfolio_id2, allocation=0.5)
        
        # Assert
        group = self.registry.get_portfolio_group(group_id)
        self.assertIn(portfolio_id2, group.portfolio_ids)
        self.assertEqual(group.allocation[portfolio_id2], 0.5)

    def test_remove_from_group(self):
        """Test removing a portfolio from a group."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        group_id = self.registry.create_portfolio_group(
            name="Test Group",
            portfolio_ids=[portfolio_id1, portfolio_id2]
        )
        
        # Act
        self.registry.remove_from_group(group_id, portfolio_id2)
        
        # Assert
        group = self.registry.get_portfolio_group(group_id)
        self.assertNotIn(portfolio_id2, group.portfolio_ids)
        self.assertNotIn(portfolio_id2, group.allocation)

    @patch('src.portfolio.portfolio_registry.np.corrcoef')
    def test_calculate_correlation(self, mock_corrcoef):
        """Test calculating correlation between portfolios."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Configure mocks
        mock_returns1 = np.array([0.01, 0.02, -0.01, 0.03])
        mock_returns2 = np.array([0.02, 0.01, 0.01, 0.02])
        
        # Mock the portfolio managers to return historical returns
        portfolio1 = self.registry.get_portfolio_components(portfolio_id1)
        portfolio1['performance_calculator'].get_historical_returns.return_value = mock_returns1
        
        portfolio2 = self.registry.get_portfolio_components(portfolio_id2)
        portfolio2['performance_calculator'].get_historical_returns.return_value = mock_returns2
        
        # Configure the corrcoef mock
        mock_corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        mock_corrcoef.return_value = mock_corr_matrix
        
        # Act
        correlation = self.registry.calculate_correlation([portfolio_id1, portfolio_id2])
        
        # Assert
        mock_corrcoef.assert_called_once()
        self.assertEqual(correlation[portfolio_id1][portfolio_id2], 0.5)

    def test_compare_portfolios(self):
        """Test comparing portfolios across different metrics."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Configure mocks
        portfolio1 = self.registry.get_portfolio_components(portfolio_id1)
        portfolio1['performance_calculator'].get_total_return.return_value = 0.15
        portfolio1['performance_calculator'].get_volatility.return_value = 0.10
        portfolio1['performance_calculator'].get_sharpe_ratio.return_value = 1.2
        portfolio1['performance_calculator'].get_max_drawdown.return_value = 0.05
        
        portfolio2 = self.registry.get_portfolio_components(portfolio_id2)
        portfolio2['performance_calculator'].get_total_return.return_value = 0.12
        portfolio2['performance_calculator'].get_volatility.return_value = 0.08
        portfolio2['performance_calculator'].get_sharpe_ratio.return_value = 1.1
        portfolio2['performance_calculator'].get_max_drawdown.return_value = 0.04
        
        # Act
        comparison = self.registry.compare_portfolios(
            [portfolio_id1, portfolio_id2],
            metrics=["returns", "volatility", "sharpe_ratio", "max_drawdown"]
        )
        
        # Assert
        self.assertEqual(comparison[portfolio_id1]["returns"], 0.15)
        self.assertEqual(comparison[portfolio_id1]["volatility"], 0.10)
        self.assertEqual(comparison[portfolio_id1]["sharpe_ratio"], 1.2)
        self.assertEqual(comparison[portfolio_id1]["max_drawdown"], 0.05)
        
        self.assertEqual(comparison[portfolio_id2]["returns"], 0.12)
        self.assertEqual(comparison[portfolio_id2]["volatility"], 0.08)
        self.assertEqual(comparison[portfolio_id2]["sharpe_ratio"], 1.1)
        self.assertEqual(comparison[portfolio_id2]["max_drawdown"], 0.04)

    def test_analyze_diversification(self):
        """Test analyzing diversification across portfolios."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Configure mocks
        portfolio1 = self.registry.get_portfolio_components(portfolio_id1)
        portfolio1['portfolio_manager'].get_positions.return_value = {
            "AAPL": {"quantity": 100, "value": 15000},
            "MSFT": {"quantity": 50, "value": 10000},
            "GOOGL": {"quantity": 20, "value": 20000},
        }
        
        portfolio2 = self.registry.get_portfolio_components(portfolio_id2)
        portfolio2['portfolio_manager'].get_positions.return_value = {
            "AMZN": {"quantity": 30, "value": 30000},
            "MSFT": {"quantity": 100, "value": 20000},
            "JNJ": {"quantity": 200, "value": 25000},
        }
        
        # Act
        diversification = self.registry.analyze_diversification([portfolio_id1, portfolio_id2])
        
        # Assert
        self.assertIn("unique_assets", diversification)
        self.assertIn("common_assets", diversification)
        self.assertIn("diversification_score", diversification)
        
        # There should be 5 unique assets across both portfolios
        self.assertEqual(len(diversification["unique_assets"]), 5)
        
        # MSFT is common to both portfolios
        self.assertEqual(len(diversification["common_assets"]), 1)
        self.assertIn("MSFT", diversification["common_assets"])

    def test_generate_consolidated_report(self):
        """Test generating a consolidated report for all portfolios."""
        # Arrange
        portfolio_id1 = self.registry.create_portfolio(
            name="Portfolio 1",
            initial_capital=100000.0
        )
        portfolio_id2 = self.registry.create_portfolio(
            name="Portfolio 2",
            initial_capital=200000.0
        )
        
        # Configure mocks
        portfolio1 = self.registry.get_portfolio_components(portfolio_id1)
        portfolio1['portfolio_manager'].get_portfolio_value.return_value = 110000.0
        portfolio1['performance_calculator'].get_total_return.return_value = 0.10
        
        portfolio2 = self.registry.get_portfolio_components(portfolio_id2)
        portfolio2['portfolio_manager'].get_portfolio_value.return_value = 220000.0
        portfolio2['performance_calculator'].get_total_return.return_value = 0.10
        
        # Act
        report = self.registry.generate_consolidated_report()
        
        # Assert
        self.assertIn("total_value", report)
        self.assertIn("portfolios", report)
        self.assertEqual(report["total_value"], 330000.0)  # 110000 + 220000
        self.assertEqual(len(report["portfolios"]), 2)
        
        # Check that each portfolio is included in the report
        portfolio_names = [p["name"] for p in report["portfolios"]]
        self.assertIn("Portfolio 1", portfolio_names)
        self.assertIn("Portfolio 2", portfolio_names)


class TestPortfolioGroup(unittest.TestCase):
    """Test cases for the PortfolioGroup class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.portfolio_id1 = "portfolio1"
        self.portfolio_id2 = "portfolio2"
        self.portfolio_id3 = "portfolio3"
        
        self.group = PortfolioGroup(
            group_id="group1",
            name="Test Group",
            portfolio_ids=[self.portfolio_id1, self.portfolio_id2],
            description="Test group description",
            allocation={
                self.portfolio_id1: 0.6,
                self.portfolio_id2: 0.4
            }
        )

    def test_add_portfolio(self):
        """Test adding a portfolio to the group."""
        # Act
        self.group.add_portfolio(self.portfolio_id3, allocation=0.2)
        
        # Assert
        self.assertIn(self.portfolio_id3, self.group.portfolio_ids)
        self.assertEqual(self.group.allocation[self.portfolio_id3], 0.2)

    def test_remove_portfolio(self):
        """Test removing a portfolio from the group."""
        # Act
        self.group.remove_portfolio(self.portfolio_id2)
        
        # Assert
        self.assertNotIn(self.portfolio_id2, self.group.portfolio_ids)
        self.assertNotIn(self.portfolio_id2, self.group.allocation)

    def test_update_allocation(self):
        """Test updating the allocation for a portfolio in the group."""
        # Act
        self.group.update_allocation(self.portfolio_id1, 0.7)
        self.group.update_allocation(self.portfolio_id2, 0.3)
        
        # Assert
        self.assertEqual(self.group.allocation[self.portfolio_id1], 0.7)
        self.assertEqual(self.group.allocation[self.portfolio_id2], 0.3)

    def test_normalize_allocation(self):
        """Test normalizing allocations to ensure they sum to 1.0."""
        # Arrange
        self.group.allocation = {
            self.portfolio_id1: 2.0,
            self.portfolio_id2: 3.0
        }
        
        # Act
        self.group.normalize_allocation()
        
        # Assert
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id1], 0.4, places=6)
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id2], 0.6, places=6)
        self.assertAlmostEqual(sum(self.group.allocation.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()