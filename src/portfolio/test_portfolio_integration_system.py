"""Integration tests for the Portfolio Management System integration.

This module contains tests that validate the integration of the Portfolio
Management System with other components of the Friday AI Trading System.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, Any, List

# Import portfolio components
from .portfolio_integration import PortfolioIntegration, create_portfolio_integration
from .portfolio_factory import PortfolioFactory
from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager
from .allocation_manager import AllocationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEventSystem:
    """Mock implementation of the Event System for testing."""

    def __init__(self):
        self.subscriptions = {}
        self.published_events = []

    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(callback)
        return True

    def publish(self, event_type, data):
        """Publish an event."""
        self.published_events.append({"event_type": event_type, "data": data})
        if event_type in self.subscriptions:
            for callback in self.subscriptions[event_type]:
                callback(data)
        return True

    def unsubscribe_all(self, subscriber):
        """Unsubscribe from all events."""
        return True

    def get_published_events(self, event_type=None):
        """Get all published events, optionally filtered by type."""
        if event_type is None:
            return self.published_events
        return [e for e in self.published_events if e["event_type"] == event_type]


class MockTradingEngine:
    """Mock implementation of the Trading Engine for testing."""

    def __init__(self):
        self.portfolio_manager = None
        self.risk_manager = None
        self.executed_trades = []

    def register_portfolio_manager(self, portfolio_manager):
        """Register a portfolio manager."""
        self.portfolio_manager = portfolio_manager
        return True

    def register_risk_manager(self, risk_manager):
        """Register a risk manager."""
        self.risk_manager = risk_manager
        return True

    def execute_trade(self, symbol, quantity, price, **kwargs):
        """Execute a trade."""
        trade = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": kwargs.get("timestamp", datetime.now()),
            "trade_id": kwargs.get("trade_id", f"trade-{len(self.executed_trades)}"),
            "commission": kwargs.get("commission", 0.0)
        }
        self.executed_trades.append(trade)
        return trade


class MockMarketDataService:
    """Mock implementation of the Market Data Service for testing."""

    def __init__(self):
        self.price_update_callback = None
        self.latest_prices = {}

    def register_price_update_callback(self, callback):
        """Register a callback for price updates."""
        self.price_update_callback = callback
        return True

    def get_latest_prices(self, symbols=None):
        """Get the latest prices for the given symbols."""
        if symbols is None:
            return self.latest_prices
        return {s: p for s, p in self.latest_prices.items() if s in symbols}

    def update_prices(self, prices):
        """Update prices and trigger callback if registered."""
        self.latest_prices.update(prices)
        if self.price_update_callback:
            self.price_update_callback({"prices": prices})
        return True


class TestPortfolioIntegration(unittest.TestCase):
    """Test cases for the Portfolio Integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.event_system = MockEventSystem()
        self.trading_engine = MockTradingEngine()
        self.market_data_service = MockMarketDataService()

        # Create test configuration
        self.config = {
            "portfolio_manager": {
                "portfolio_id": "test-portfolio",
                "initial_cash": 100000.0
            },
            "performance_calculator": {
                "benchmark_symbol": "SPY",
                "risk_free_rate": 0.02
            },
            "tax_manager": {
                "default_method": "FIFO",
                "wash_sale_window_days": 30
            },
            "allocation_manager": {
                "rebalance_method": "THRESHOLD",
                "default_threshold": 5.0,
                "rebalance_frequency_days": 90,
                "allocation_targets": [
                    {"symbol": "AAPL", "target": 0.20},
                    {"symbol": "MSFT", "target": 0.20},
                    {"symbol": "GOOGL", "target": 0.15},
                    {"symbol": "BND", "target": 0.25},
                    {"symbol": "VTI", "target": 0.20}
                ]
            }
        }

        # Create portfolio integration
        self.integration = create_portfolio_integration(
            config=self.config,
            event_system=self.event_system,
            trading_engine=self.trading_engine,
            market_data_service=self.market_data_service,
            auto_start=True
        )

    def test_initialization(self):
        """Test that the integration initializes correctly."""
        # Check that components are properly initialized
        self.assertIsNotNone(self.integration.portfolio_manager)
        self.assertIsNotNone(self.integration.performance_calculator)
        self.assertIsNotNone(self.integration.tax_manager)
        self.assertIsNotNone(self.integration.allocation_manager)

        # Check that the integration is started
        self.assertTrue(self.integration._started)

        # Check that event subscriptions are set up
        self.assertIn("market_data_update", self.event_system.subscriptions)
        self.assertIn("trade_executed", self.event_system.subscriptions)
        self.assertIn("portfolio_update_request", self.event_system.subscriptions)
        self.assertIn("portfolio_rebalance_request", self.event_system.subscriptions)

    def test_market_data_integration(self):
        """Test integration with market data service."""
        # Update prices through market data service
        prices = {
            "AAPL": 150.0,
            "MSFT": 250.0,
            "GOOGL": 1500.0,
            "BND": 85.0,
            "VTI": 200.0
        }
        self.market_data_service.update_prices(prices)

        # Check that portfolio prices are updated
        portfolio_prices = self.integration.portfolio_manager.prices
        for symbol, price in prices.items():
            self.assertEqual(portfolio_prices.get(symbol), price)

        # Check that a portfolio value update event was published
        value_updates = self.event_system.get_published_events("portfolio_value_update")
        self.assertEqual(len(value_updates), 1)
        self.assertEqual(value_updates[0]["data"]["portfolio_id"], "test-portfolio")

    def test_trade_execution_integration(self):
        """Test integration with trade execution."""
        # Execute a trade through the event system
        trade_data = {
            "symbol": "AAPL",
            "quantity": 50,
            "price": 150.0,
            "timestamp": datetime.now(),
            "trade_id": "test-trade-1",
            "commission": 5.0
        }
        self.event_system.publish("trade_executed", trade_data)

        # Check that the trade was executed in the portfolio
        positions = self.integration.portfolio_manager.get_positions()
        self.assertIn("AAPL", positions)
        self.assertEqual(positions["AAPL"].quantity, 50)

        # Check that a tax lot was added
        tax_lots = self.integration.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 1)
        self.assertEqual(tax_lots[0].quantity, 50)
        self.assertEqual(tax_lots[0].price, 150.0)

        # Check that allocations were updated
        allocations = self.integration.allocation_manager.get_current_allocations()
        self.assertIn("AAPL", allocations)

        # Check that a portfolio updated event was published
        portfolio_updates = self.event_system.get_published_events("portfolio_updated")
        self.assertEqual(len(portfolio_updates), 1)
        self.assertEqual(portfolio_updates[0]["data"]["portfolio_id"], "test-portfolio")

    def test_portfolio_update_request(self):
        """Test handling of portfolio update requests."""
        # Execute some trades to populate the portfolio
        self.integration.portfolio_manager.execute_trade("AAPL", 50, 150.0)
        self.integration.portfolio_manager.execute_trade("MSFT", 40, 250.0)

        # Update prices
        prices = {
            "AAPL": 160.0,
            "MSFT": 260.0
        }
        self.integration.portfolio_manager.update_prices(prices)

        # Send a portfolio update request
        self.event_system.publish("portfolio_update_request", {})

        # Check that a portfolio state event was published
        state_events = self.event_system.get_published_events("portfolio_state")
        self.assertEqual(len(state_events), 1)
        state_data = state_events[0]["data"]

        # Verify the state data
        self.assertEqual(state_data["portfolio_id"], "test-portfolio")
        self.assertIn("value", state_data)
        self.assertIn("cash", state_data)
        self.assertIn("positions", state_data)
        self.assertIn("transaction_history", state_data)

    def test_rebalance_request(self):
        """Test handling of rebalance requests."""
        # Execute trades to create an unbalanced portfolio
        self.integration.portfolio_manager.execute_trade("AAPL", 100, 150.0)  # 15,000 (30%)
        self.integration.portfolio_manager.execute_trade("MSFT", 20, 250.0)   # 5,000 (10%)

        # Update prices
        prices = {
            "AAPL": 160.0,  # Now 16,000 (32%)
            "MSFT": 260.0   # Now 5,200 (10.4%)
        }
        self.integration.portfolio_manager.update_prices(prices)

        # Update allocations
        self.integration.allocation_manager.update_allocation_from_portfolio(
            self.integration.portfolio_manager.get_positions_value(),
            self.integration.portfolio_manager.get_portfolio_value()
        )

        # Send a rebalance request
        self.event_system.publish("portfolio_rebalance_request", {})

        # Check that a rebalance plan event was published
        rebalance_events = self.event_system.get_published_events("portfolio_rebalance_plan")
        self.assertEqual(len(rebalance_events), 1)
        rebalance_data = rebalance_events[0]["data"]

        # Verify the rebalance plan
        self.assertEqual(rebalance_data["portfolio_id"], "test-portfolio")
        self.assertIn("rebalance_plan", rebalance_data)
        self.assertGreater(len(rebalance_data["rebalance_plan"]), 0)

    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        # Execute some trades
        self.integration.portfolio_manager.execute_trade("AAPL", 50, 150.0)
        self.integration.portfolio_manager.execute_trade("MSFT", 40, 250.0)

        # Update prices
        prices = {
            "AAPL": 160.0,
            "MSFT": 260.0
        }
        self.integration.portfolio_manager.update_prices(prices)

        # Get portfolio summary
        summary = self.integration.get_portfolio_summary()

        # Verify summary contents
        self.assertEqual(summary["portfolio_id"], "test-portfolio")
        self.assertIn("value", summary)
        self.assertIn("cash", summary)
        self.assertIn("positions", summary)
        self.assertIn("AAPL", summary["positions"])
        self.assertIn("MSFT", summary["positions"])

    def test_stop_integration(self):
        """Test stopping the integration."""
        # Stop the integration
        self.integration.stop()

        # Check that the integration is stopped
        self.assertFalse(self.integration._started)


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow tests for the Portfolio Integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.event_system = MockEventSystem()
        self.trading_engine = MockTradingEngine()
        self.market_data_service = MockMarketDataService()

        # Create test configuration
        self.config = {
            "portfolio_manager": {
                "portfolio_id": "e2e-portfolio",
                "initial_cash": 100000.0
            },
            "performance_calculator": {
                "benchmark_symbol": "SPY",
                "risk_free_rate": 0.02,
                "max_history_size": 100
            },
            "tax_manager": {
                "default_method": "FIFO",
                "wash_sale_window_days": 30
            },
            "allocation_manager": {
                "rebalance_method": "THRESHOLD",
                "default_threshold": 5.0,
                "rebalance_frequency_days": 90,
                "allocation_targets": [
                    {"symbol": "AAPL", "target": 0.20, "sector": "Technology"},
                    {"symbol": "MSFT", "target": 0.20, "sector": "Technology"},
                    {"symbol": "GOOGL", "target": 0.15, "sector": "Technology"},
                    {"symbol": "BND", "target": 0.25, "sector": "Bonds"},
                    {"symbol": "VTI", "target": 0.20, "sector": "Equities"}
                ]
            }
        }

        # Create portfolio integration
        self.integration = create_portfolio_integration(
            config=self.config,
            event_system=self.event_system,
            trading_engine=self.trading_engine,
            market_data_service=self.market_data_service,
            auto_start=True
        )

    def test_complete_workflow(self):
        """Test a complete portfolio management workflow."""
        # Step 1: Execute initial trades
        initial_trades = [
            {"symbol": "AAPL", "quantity": 50, "price": 150.0, "sector": "Technology"},
            {"symbol": "MSFT", "quantity": 40, "price": 250.0, "sector": "Technology"},
            {"symbol": "GOOGL", "quantity": 10, "price": 1500.0, "sector": "Technology"},
            {"symbol": "BND", "quantity": 200, "price": 85.0, "sector": "Bonds"},
            {"symbol": "VTI", "quantity": 50, "price": 200.0, "sector": "Equities"}
        ]

        for trade in initial_trades:
            self.event_system.publish("trade_executed", trade)

        # Step 2: Update prices over time
        price_updates = [
            # Day 1
            {
                "AAPL": 155.0,
                "MSFT": 255.0,
                "GOOGL": 1520.0,
                "BND": 85.5,
                "VTI": 202.0,
                "SPY": 400.0  # Benchmark
            },
            # Day 2
            {
                "AAPL": 160.0,
                "MSFT": 260.0,
                "GOOGL": 1550.0,
                "BND": 86.0,
                "VTI": 205.0,
                "SPY": 405.0  # Benchmark
            },
            # Day 3
            {
                "AAPL": 165.0,
                "MSFT": 265.0,
                "GOOGL": 1580.0,
                "BND": 86.5,
                "VTI": 208.0,
                "SPY": 410.0  # Benchmark
            }
        ]

        # Apply price updates and add performance data
        for i, prices in enumerate(price_updates):
            # Set timestamp for this update
            timestamp = datetime.now() - timedelta(days=len(price_updates) - i)

            # Update prices through market data service
            self.market_data_service.update_prices(prices)

            # Add performance data point
            portfolio_value = self.integration.portfolio_manager.get_portfolio_value()
            benchmark_value = prices["SPY"]
            self.integration.performance_calculator.add_observation(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                benchmark_value=benchmark_value
            )

        # Step 3: Execute additional trades
        additional_trades = [
            {"symbol": "AAPL", "quantity": 10, "price": 165.0, "sector": "Technology"},
            {"symbol": "MSFT", "quantity": -5, "price": 265.0, "sector": "Technology"}  # Sell some MSFT
        ]

        for trade in additional_trades:
            self.event_system.publish("trade_executed", trade)

        # Step 4: Request portfolio update
        self.event_system.publish("portfolio_update_request", {})

        # Step 5: Request rebalance
        self.event_system.publish("portfolio_rebalance_request", {})

        # Step 6: Get final portfolio summary
        summary = self.integration.get_portfolio_summary()

        # Verify the final state
        self.assertEqual(summary["portfolio_id"], "e2e-portfolio")

        # Check positions
        positions = summary["positions"]
        self.assertEqual(positions["AAPL"]["quantity"], 60)  # 50 + 10
        self.assertEqual(positions["MSFT"]["quantity"], 35)  # 40 - 5

        # Check performance metrics
        if "performance_metrics" in summary:
            metrics = summary["performance_metrics"]
            self.assertIn("total_return", metrics)
            self.assertIn("benchmark_return", metrics)

        # Check tax lots
        tax_lots = self.integration.tax_manager.get_tax_lots("AAPL")
        self.assertEqual(len(tax_lots), 2)  # Initial purchase + additional purchase

        # Check allocation summary
        if "allocation_summary" in summary:
            allocations = summary["allocation_summary"]
            self.assertIn("current_allocations", allocations)
            self.assertIn("target_allocations", allocations)
            self.assertIn("drift", allocations)

        # Check published events
        portfolio_updates = self.event_system.get_published_events("portfolio_updated")
        self.assertGreaterEqual(len(portfolio_updates), len(initial_trades) + len(additional_trades))

        rebalance_plans = self.event_system.get_published_events("portfolio_rebalance_plan")
        self.assertGreaterEqual(len(rebalance_plans), 1)

        portfolio_states = self.event_system.get_published_events("portfolio_state")
        self.assertGreaterEqual(len(portfolio_states), 1)


if __name__ == "__main__":
    unittest.main()
