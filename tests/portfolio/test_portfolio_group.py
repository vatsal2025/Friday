"""Unit tests for the PortfolioGroup class.

This module contains tests for the PortfolioGroup class, which is responsible
for managing collections of portfolios, including allocation management and
consolidated reporting.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from src.portfolio.portfolio_registry import PortfolioGroup


class TestPortfolioGroup(unittest.TestCase):
    """Test cases for the PortfolioGroup class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.portfolio_id1 = "portfolio1"
        self.portfolio_id2 = "portfolio2"
        self.portfolio_id3 = "portfolio3"
        
        # Create a portfolio group with two portfolios
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

    def test_initialization(self):
        """Test that the group is initialized correctly."""
        self.assertEqual(self.group.group_id, "group1")
        self.assertEqual(self.group.name, "Test Group")
        self.assertEqual(self.group.description, "Test group description")
        self.assertEqual(len(self.group.portfolio_ids), 2)
        self.assertIn(self.portfolio_id1, self.group.portfolio_ids)
        self.assertIn(self.portfolio_id2, self.group.portfolio_ids)
        self.assertEqual(self.group.allocation[self.portfolio_id1], 0.6)
        self.assertEqual(self.group.allocation[self.portfolio_id2], 0.4)

    def test_add_portfolio(self):
        """Test adding a portfolio to the group."""
        # Act
        self.group.add_portfolio(self.portfolio_id3, allocation=0.2)
        
        # Assert
        self.assertIn(self.portfolio_id3, self.group.portfolio_ids)
        self.assertEqual(self.group.allocation[self.portfolio_id3], 0.2)
        self.assertEqual(len(self.group.portfolio_ids), 3)

    def test_add_portfolio_without_allocation(self):
        """Test adding a portfolio without specifying an allocation."""
        # Act
        self.group.add_portfolio(self.portfolio_id3)
        
        # Assert
        self.assertIn(self.portfolio_id3, self.group.portfolio_ids)
        # Default allocation should be equal weight
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id3], 1/3, places=6)
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id1], 1/3, places=6)
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id2], 1/3, places=6)

    def test_remove_portfolio(self):
        """Test removing a portfolio from the group."""
        # Act
        self.group.remove_portfolio(self.portfolio_id2)
        
        # Assert
        self.assertNotIn(self.portfolio_id2, self.group.portfolio_ids)
        self.assertNotIn(self.portfolio_id2, self.group.allocation)
        self.assertEqual(len(self.group.portfolio_ids), 1)
        # Remaining allocation should be normalized to 1.0
        self.assertEqual(self.group.allocation[self.portfolio_id1], 1.0)

    def test_update_allocation(self):
        """Test updating the allocation for a portfolio in the group."""
        # Act
        self.group.update_allocation(self.portfolio_id1, 0.7)
        self.group.update_allocation(self.portfolio_id2, 0.3)
        
        # Assert
        self.assertEqual(self.group.allocation[self.portfolio_id1], 0.7)
        self.assertEqual(self.group.allocation[self.portfolio_id2], 0.3)

    def test_update_allocation_invalid_portfolio(self):
        """Test updating the allocation for a portfolio not in the group."""
        # Act/Assert
        with self.assertRaises(ValueError):
            self.group.update_allocation("non_existent_portfolio", 0.5)

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

    def test_normalize_allocation_zero_sum(self):
        """Test normalizing allocations when the sum is zero."""
        # Arrange
        self.group.allocation = {
            self.portfolio_id1: 0.0,
            self.portfolio_id2: 0.0
        }
        
        # Act
        self.group.normalize_allocation()
        
        # Assert - should default to equal weights
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id1], 0.5, places=6)
        self.assertAlmostEqual(self.group.allocation[self.portfolio_id2], 0.5, places=6)
        self.assertAlmostEqual(sum(self.group.allocation.values()), 1.0, places=6)

    def test_get_portfolio_ids(self):
        """Test getting the portfolio IDs in the group."""
        # Act
        portfolio_ids = self.group.get_portfolio_ids()
        
        # Assert
        self.assertEqual(len(portfolio_ids), 2)
        self.assertIn(self.portfolio_id1, portfolio_ids)
        self.assertIn(self.portfolio_id2, portfolio_ids)

    def test_get_allocation(self):
        """Test getting the allocation for a specific portfolio."""
        # Act
        allocation1 = self.group.get_allocation(self.portfolio_id1)
        allocation2 = self.group.get_allocation(self.portfolio_id2)
        
        # Assert
        self.assertEqual(allocation1, 0.6)
        self.assertEqual(allocation2, 0.4)

    def test_get_allocation_invalid_portfolio(self):
        """Test getting the allocation for a portfolio not in the group."""
        # Act/Assert
        with self.assertRaises(ValueError):
            self.group.get_allocation("non_existent_portfolio")

    def test_to_dict(self):
        """Test converting the group to a dictionary representation."""
        # Act
        group_dict = self.group.to_dict()
        
        # Assert
        self.assertEqual(group_dict["group_id"], "group1")
        self.assertEqual(group_dict["name"], "Test Group")
        self.assertEqual(group_dict["description"], "Test group description")
        self.assertEqual(len(group_dict["portfolio_ids"]), 2)
        self.assertIn(self.portfolio_id1, group_dict["portfolio_ids"])
        self.assertIn(self.portfolio_id2, group_dict["portfolio_ids"])
        self.assertEqual(group_dict["allocation"][self.portfolio_id1], 0.6)
        self.assertEqual(group_dict["allocation"][self.portfolio_id2], 0.4)

    def test_from_dict(self):
        """Test creating a group from a dictionary representation."""
        # Arrange
        group_dict = {
            "group_id": "group2",
            "name": "Another Group",
            "description": "Another group description",
            "portfolio_ids": [self.portfolio_id1, self.portfolio_id3],
            "allocation": {
                self.portfolio_id1: 0.3,
                self.portfolio_id3: 0.7
            }
        }
        
        # Act
        group = PortfolioGroup.from_dict(group_dict)
        
        # Assert
        self.assertEqual(group.group_id, "group2")
        self.assertEqual(group.name, "Another Group")
        self.assertEqual(group.description, "Another group description")
        self.assertEqual(len(group.portfolio_ids), 2)
        self.assertIn(self.portfolio_id1, group.portfolio_ids)
        self.assertIn(self.portfolio_id3, group.portfolio_ids)
        self.assertEqual(group.allocation[self.portfolio_id1], 0.3)
        self.assertEqual(group.allocation[self.portfolio_id3], 0.7)

    def test_equal_weight_allocation(self):
        """Test creating a group with equal weight allocation."""
        # Arrange
        portfolio_ids = [self.portfolio_id1, self.portfolio_id2, self.portfolio_id3]
        
        # Act
        group = PortfolioGroup(
            group_id="equal_weight_group",
            name="Equal Weight Group",
            portfolio_ids=portfolio_ids,
            description="Equal weight allocation"
            # No allocation specified, should default to equal weights
        )
        
        # Assert
        self.assertEqual(len(group.portfolio_ids), 3)
        self.assertAlmostEqual(group.allocation[self.portfolio_id1], 1/3, places=6)
        self.assertAlmostEqual(group.allocation[self.portfolio_id2], 1/3, places=6)
        self.assertAlmostEqual(group.allocation[self.portfolio_id3], 1/3, places=6)
        self.assertAlmostEqual(sum(group.allocation.values()), 1.0, places=6)

    def test_validate_allocation(self):
        """Test validating the allocation."""
        # Arrange
        invalid_allocation = {
            self.portfolio_id1: 0.6,
            self.portfolio_id2: 0.5  # Sum > 1.0
        }
        
        # Act/Assert
        with self.assertRaises(ValueError):
            self.group.validate_allocation(invalid_allocation)
        
        # Test with negative allocation
        invalid_allocation = {
            self.portfolio_id1: -0.1,
            self.portfolio_id2: 1.1
        }
        
        # Act/Assert
        with self.assertRaises(ValueError):
            self.group.validate_allocation(invalid_allocation)

    def test_validate_portfolio_ids(self):
        """Test validating the portfolio IDs."""
        # Arrange
        invalid_portfolio_ids = [self.portfolio_id1, self.portfolio_id1]  # Duplicate ID
        
        # Act/Assert
        with self.assertRaises(ValueError):
            self.group.validate_portfolio_ids(invalid_portfolio_ids)


if __name__ == "__main__":
    unittest.main()