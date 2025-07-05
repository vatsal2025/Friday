"""Example usage of the Portfolio Management System.

This module provides examples of how to use the Portfolio Management System,
including creating a portfolio, executing trades, and analyzing performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import portfolio components
from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager, TaxLotMethod
from .allocation_manager import AllocationManager, RebalanceMethod
from .portfolio_factory import PortfolioFactory
from .config import PortfolioConfig

# Try to import risk components
try:
    from ..risk.risk_management_factory import RiskManagementFactory
    from ..risk.advanced_risk_manager import AdvancedRiskManager
    RISK_MODULE_AVAILABLE = True
except ImportError:
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


def basic_portfolio_example() -> None:
    """Basic example of creating and managing a portfolio."""
    logger.info("Running basic portfolio example")

    # Create a portfolio manager
    portfolio = PortfolioManager(portfolio_id="example-portfolio", initial_cash=100000.0)

    # Execute some trades
    trade_date = datetime.now() - timedelta(days=30)
    portfolio.execute_trade("AAPL", 50, 150.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("MSFT", 40, 250.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("GOOGL", 10, 1500.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("BND", 100, 85.0, trade_date, sector="Fixed Income", asset_class="bonds")

    # Update prices (simulate price changes over time)
    dates = [trade_date + timedelta(days=i) for i in range(1, 31)]
    for i, date in enumerate(dates):
        # Simulate some price movements
        day_change = 0.002 * np.sin(i/5) + 0.001  # Small oscillation with upward trend

        prices = {
            "AAPL": 150.0 * (1 + 0.1 * np.sin(i/10) + i * day_change),
            "MSFT": 250.0 * (1 + 0.08 * np.sin(i/12) + i * day_change),
            "GOOGL": 1500.0 * (1 + 0.12 * np.sin(i/8) + i * day_change),
            "BND": 85.0 * (1 + 0.02 * np.sin(i/15) + i * day_change * 0.3)  # Bonds move less
        }

        portfolio.update_prices(prices, date)

    # Print portfolio summary
    logger.info(f"Portfolio ID: {portfolio.portfolio_id}")
    logger.info(f"Cash balance: ${portfolio.cash:.2f}")
    logger.info(f"Portfolio value: ${portfolio.get_portfolio_value():.2f}")
    logger.info(f"Number of positions: {len(portfolio.positions)}")

    # Print positions
    logger.info("\nPositions:")
    for symbol, position in portfolio.positions.items():
        current_value = position["quantity"] * position["last_price"]
        profit_loss = current_value - (position["quantity"] * position["average_price"])
        profit_loss_pct = (profit_loss / (position["quantity"] * position["average_price"])) * 100

        logger.info(f"{symbol}: {position['quantity']} shares @ ${position['average_price']:.2f} avg cost, "
                   f"current price: ${position['last_price']:.2f}, "
                   f"value: ${current_value:.2f}, "
                   f"P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")

    # Print transaction history
    logger.info("\nTransaction History:")
    for tx in portfolio.get_transaction_history():
        logger.info(f"{tx['date'].strftime('%Y-%m-%d')}: {tx['action']} {tx['quantity']} {tx['symbol']} @ ${tx['price']:.2f}")

    # Print historical values
    logger.info("\nHistorical Portfolio Values:")
    historical_values = portfolio.get_historical_values()
    for date, value in list(historical_values.items())[-5:]:
        logger.info(f"{date.strftime('%Y-%m-%d')}: ${value:.2f}")

    return portfolio


def performance_calculation_example(portfolio: PortfolioManager):
    """Example of calculating portfolio performance."""
    logger.info("\nRunning performance calculation example")

    # Create a performance calculator
    performance = PerformanceCalculator(benchmark_symbol="SPY")

    # Add portfolio values
    historical_values = portfolio.get_historical_values()
    for date, value in historical_values.items():
        performance.add_portfolio_value_observation(value, date)

    # Add benchmark returns (simulated)
    dates = sorted(historical_values.keys())
    for i in range(1, len(dates)):
        # Simulate benchmark returns (correlated with portfolio but different)
        benchmark_return = 0.001 * np.sin(i/7) + 0.0005  # Small oscillation with upward trend
        performance.add_benchmark_return_observation(benchmark_return, dates[i])

    # Calculate performance metrics
    metrics = performance.calculate_performance_metrics()

    # Print performance metrics
    logger.info("Performance Metrics:")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"Volatility: {metrics['volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Print benchmark comparison
    benchmark_comp = metrics["benchmark_comparison"]
    logger.info("\nBenchmark Comparison:")
    logger.info(f"Alpha: {benchmark_comp['alpha']:.4f}")
    logger.info(f"Beta: {benchmark_comp['beta']:.2f}")
    logger.info(f"Tracking Error: {benchmark_comp['tracking_error']:.2%}")
    logger.info(f"Information Ratio: {benchmark_comp['information_ratio']:.2f}")

    return performance


def tax_management_example(portfolio: PortfolioManager):
    """Example of tax-aware trading and reporting."""
    logger.info("\nRunning tax management example")

    # Create a tax manager
    tax_manager = TaxManager(default_method=TaxLotMethod.FIFO)

    # Add tax lots from portfolio positions
    for symbol, position in portfolio.positions.items():
        tax_manager.add_tax_lot(symbol, position["quantity"], position["average_price"], position["acquisition_date"])

    # Print tax lots
    logger.info("Tax Lots:")
    lots = tax_manager.get_tax_lots()
    for symbol, symbol_lots in lots.items():
        for lot in symbol_lots:
            logger.info(f"{symbol}: {lot.quantity} shares @ ${lot.price:.2f}, acquired {lot.acquisition_date.strftime('%Y-%m-%d')}")

    # Sell some shares
    logger.info("\nSelling shares:")
    sale_date = datetime.now()

    # Sell AAPL using FIFO
    aapl_sale = tax_manager.sell_tax_lots("AAPL", 20, 170.0, sale_date)
    logger.info(f"AAPL sale using {aapl_sale['method']}: {aapl_sale['quantity_sold']} shares @ $170.00")
    logger.info(f"Realized gain: ${aapl_sale['realized_gain']:.2f}")

    # Sell MSFT using LIFO
    tax_manager.set_symbol_method("MSFT", TaxLotMethod.LIFO)
    msft_sale = tax_manager.sell_tax_lots("MSFT", 15, 280.0, sale_date)
    logger.info(f"MSFT sale using {msft_sale['method']}: {msft_sale['quantity_sold']} shares @ $280.00")
    logger.info(f"Realized gain: ${msft_sale['realized_gain']:.2f}")

    # Generate tax report
    current_year = datetime.now().year
    tax_report = tax_manager.generate_tax_report(current_year)

    logger.info(f"\nTax Report for {current_year}:")
    logger.info(f"Short-term gains: ${tax_report['short_term_gains']:.2f}")
    logger.info(f"Long-term gains: ${tax_report['long_term_gains']:.2f}")
    logger.info(f"Total realized gains: ${tax_report['total_gains']:.2f}")
    logger.info(f"Wash sales: ${tax_report['wash_sale_amount']:.2f}")

    return tax_manager


def allocation_management_example(portfolio: PortfolioManager):
    """Example of asset allocation management."""
    logger.info("\nRunning allocation management example")

    # Create an allocation manager
    allocation = AllocationManager(rebalance_method=RebalanceMethod.THRESHOLD, default_threshold=5.0)

    # Set allocation targets
    allocation.set_allocation_targets([
        {"name": "AAPL", "target_percentage": 25.0, "category": "stocks"},
        {"name": "MSFT", "target_percentage": 20.0, "category": "stocks"},
        {"name": "GOOGL", "target_percentage": 15.0, "category": "stocks"},
        {"name": "BND", "target_percentage": 30.0, "category": "bonds"},
        {"name": "CASH", "target_percentage": 10.0, "category": "cash"}
    ])

    # Get current portfolio values
    portfolio_values = {
        symbol: position["quantity"] * position["last_price"]
        for symbol, position in portfolio.positions.items()
    }
    portfolio_values["CASH"] = portfolio.cash

    # Define asset categories
    categories = {
        "AAPL": "stocks",
        "MSFT": "stocks",
        "GOOGL": "stocks",
        "BND": "bonds",
        "CASH": "cash"
    }

    # Update current allocations
    allocation.update_allocation_from_portfolio(portfolio_values, categories)

    # Print current allocations
    current_allocations = allocation.get_current_allocations()
    logger.info("Current Allocations:")
    for category, assets in current_allocations.items():
        logger.info(f"{category.capitalize()}:")
        for asset, percentage in assets.items():
            logger.info(f"  {asset}: {percentage:.2f}%")

    # Check if rebalance is needed
    rebalance_check = allocation.check_rebalance_needed()
    logger.info(f"\nRebalance needed: {rebalance_check['rebalance_needed']}")
    if rebalance_check["rebalance_needed"]:
        logger.info(f"Reason: {rebalance_check['reason']}")

    # Generate rebalance plan
    plan = allocation.generate_rebalance_plan(portfolio_values, categories)

    if plan["rebalance_needed"]:
        logger.info("\nRebalance Plan:")
        for trade in plan["trades"]:
            action = "Buy" if trade["quantity"] > 0 else "Sell"
            logger.info(f"{action} {abs(trade['quantity']):.2f} units of {trade['symbol']} "
                       f"(${abs(trade['estimated_value']):.2f})")

    # Print drift information
    drift_info = allocation.get_drift_information()
    logger.info("\nAllocation Drift:")
    for symbol, drift in drift_info.items():
        logger.info(f"{symbol}: Target {drift['target']:.2f}%, Current {drift['current']:.2f}%, "
                   f"Drift {drift['drift']:.2f}%, Threshold {drift['threshold']:.2f}%")

    return allocation


def factory_example() -> None:
    """Example of using the PortfolioFactory to create components."""
    logger.info("Running factory example")

    # Create configuration
    config = {
        "portfolio_manager": {
            "portfolio_id": "factory-portfolio",
            "initial_cash": 100000.0
        },
        "performance_calculator": {
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02
        },
        "tax_manager": {
            "default_method": "FIFO",
            "symbol_methods": {
                "MSFT": "LIFO",
                "GOOGL": "HIFO"
            }
        },
        "allocation_manager": {
            "rebalance_method": "THRESHOLD",
            "default_threshold": 5.0,
            "allocation_targets": [
                {"name": "AAPL", "target_percentage": 25.0, "category": "stocks"},
                {"name": "MSFT", "target_percentage": 20.0, "category": "stocks"},
                {"name": "GOOGL", "target_percentage": 15.0, "category": "stocks"},
                {"name": "BND", "target_percentage": 30.0, "category": "bonds"},
                {"name": "CASH", "target_percentage": 10.0, "category": "cash"}
            ]
        }
    }

    # Add risk manager config if available
    if RISK_MODULE_AVAILABLE:
        config["risk_manager"] = {
            "max_portfolio_var_percent": 2.0,
            "max_drawdown_percent": 15.0,
            "risk_free_rate": 0.02,
            "confidence_level": 0.95
        }

    # Create factory
    factory = PortfolioFactory(config)

    # Create complete system
    system = factory.create_complete_portfolio_system()

    # Print system components
    logger.info("Portfolio System Components:")
    for component_name, component in system.items():
        logger.info(f"{component_name}: {type(component).__name__}")

    # Execute some trades
    portfolio = system["portfolio_manager"]
    trade_date = datetime.now() - timedelta(days=30)
    portfolio.execute_trade("AAPL", 50, 150.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("MSFT", 40, 250.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("GOOGL", 10, 1500.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("BND", 100, 85.0, trade_date, sector="Fixed Income", asset_class="bonds")

    # Update prices
    prices = {"AAPL": 160.0, "MSFT": 260.0, "GOOGL": 1550.0, "BND": 86.0}
    portfolio.update_prices(prices)

    # Print portfolio summary
    logger.info(f"\nPortfolio ID: {portfolio.portfolio_id}")
    logger.info(f"Cash balance: ${portfolio.cash:.2f}")
    logger.info(f"Portfolio value: ${portfolio.get_portfolio_value():.2f}")

    return system


def risk_integration_example() -> None:
    """Example of integrating risk management with portfolio."""
    logger.info("Running risk integration example")

    # Create configuration with risk management
    config = {
        "portfolio_manager": {
            "portfolio_id": "risk-portfolio",
            "initial_cash": 100000.0
        },
        "risk_manager": {
            "max_portfolio_var_percent": 2.0,
            "max_drawdown_percent": 15.0,
            "risk_free_rate": 0.02,
            "confidence_level": 0.95
        }
    }

    # Create factory
    factory = PortfolioFactory(config)

    # Create portfolio with risk manager
    portfolio = factory.create_portfolio_manager()
    risk_manager = factory.create_risk_manager()
    portfolio.set_risk_manager(risk_manager)

    # Execute some trades
    trade_date = datetime.now() - timedelta(days=100)
    portfolio.execute_trade("AAPL", 50, 150.0, trade_date, sector="Technology", asset_class="equities")
    portfolio.execute_trade("MSFT", 40, 250.0, trade_date, sector="Technology", asset_class="equities")

    # Add historical data for risk calculations
    dates = [trade_date + timedelta(days=i) for i in range(100)]
    for i, date in enumerate(dates):
        # Simulate price movements
        aapl_price = 150.0 * (1 + 0.0003 * (i - np.sin(i/10)))
        msft_price = 250.0 * (1 + 0.0004 * (i - np.sin(i/12)))

        historical_prices = {"AAPL": aapl_price, "MSFT": msft_price}
        portfolio.add_historical_prices(historical_prices, date)

    # Update current prices
    current_prices = {"AAPL": 160.0, "MSFT": 260.0}
    portfolio.update_prices(current_prices)

    # Get risk metrics
    risk_metrics = portfolio.get_risk_metrics()

    if risk_metrics:
        logger.info("Risk Metrics:")
        logger.info(f"Value at Risk (VaR): ${risk_metrics['portfolio_var']:.2f}")
        logger.info(f"VaR %: {risk_metrics['portfolio_var_percent']:.2%}")
        logger.info(f"Expected Shortfall: ${risk_metrics['expected_shortfall']:.2f}")
        logger.info(f"Portfolio Volatility: {risk_metrics['portfolio_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
        logger.info(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")

    return portfolio


def portfolio_integration_example() -> None:
    """Example of using the Portfolio Integration module."""
    logger.info("\nRunning portfolio integration example")
    
    # Try to import integration components
    try:
        from .portfolio_integration import create_portfolio_integration, PortfolioIntegration
        INTEGRATION_MODULE_AVAILABLE = True
    except ImportError:
        logger.error("Portfolio integration module not available")
        return
    
    # Create mock event system for demonstration
    class MockEventSystem:
        def __init__(self):
            self.subscriptions = {}
            self.published_events = []
        
        def subscribe(self, event_type, callback):
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []
            self.subscriptions[event_type].append(callback)
            logger.info(f"Subscribed to event: {event_type}")
            return True
        
        def publish(self, event_type, data):
            self.published_events.append({"event_type": event_type, "data": data})
            logger.info(f"Published event: {event_type}")
            if event_type in self.subscriptions:
                for callback in self.subscriptions[event_type]:
                    callback(data)
            return True
        
        def unsubscribe_all(self, subscriber):
            logger.info("Unsubscribed from all events")
            return True
    
    # Create configuration
    config = {
        "portfolio_manager": {
            "portfolio_id": "integrated-portfolio",
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
                {"symbol": "AAPL", "target": 0.15},
                {"symbol": "MSFT", "target": 0.15},
                {"symbol": "GOOGL", "target": 0.10},
                {"symbol": "BND", "target": 0.30},
                {"symbol": "VTI", "target": 0.30}
            ]
        }
    }
    
    # Create event system
    event_system = MockEventSystem()
    
    # Create portfolio integration
    integration = create_portfolio_integration(
        config=config,
        event_system=event_system,
        auto_start=True
    )
    
    logger.info("Portfolio integration created and started")
    
    # Execute trades through event system
    trades = [
        {
            "symbol": "AAPL",
            "quantity": 50,
            "price": 150.0,
            "timestamp": datetime.now(),
            "trade_id": "trade-1",
            "commission": 5.0,
            "sector": "Technology"
        },
        {
            "symbol": "MSFT",
            "quantity": 40,
            "price": 250.0,
            "timestamp": datetime.now(),
            "trade_id": "trade-2",
            "commission": 5.0,
            "sector": "Technology"
        },
        {
            "symbol": "GOOGL",
            "quantity": 10,
            "price": 1500.0,
            "timestamp": datetime.now(),
            "trade_id": "trade-3",
            "commission": 5.0,
            "sector": "Technology"
        }
    ]
    
    for trade in trades:
        event_system.publish("trade_executed", trade)
    
    # Update prices through market data update event
    prices = {
        "AAPL": 160.0,
        "MSFT": 260.0,
        "GOOGL": 1550.0,
        "BND": 85.0,
        "VTI": 200.0
    }
    
    event_system.publish("market_data_update", {"prices": prices})
    
    # Request portfolio update
    event_system.publish("portfolio_update_request", {})
    
    # Request rebalance
    event_system.publish("portfolio_rebalance_request", {})
    
    # Get portfolio summary
    summary = integration.get_portfolio_summary()
    
    # Print summary
    logger.info(f"\nPortfolio Summary:")
    logger.info(f"Portfolio ID: {summary['portfolio_id']}")
    logger.info(f"Portfolio Value: ${summary['value']:.2f}")
    logger.info(f"Cash: ${summary['cash']:.2f}")
    logger.info("Positions:")
    for symbol, position in summary['positions'].items():
        logger.info(f"  {symbol}: {position['quantity']} shares, ${position['value']:.2f}")
    
    # Stop integration
    integration.stop()
    logger.info("Portfolio integration stopped")
    
    return integration


def run_all_examples() -> None:
    """Run all example functions."""
    basic_portfolio_example()
    factory_example()
    if RISK_MODULE_AVAILABLE:
        risk_integration_example()
    else:
        logger.warning("Risk module not available, skipping risk integration example")
    
    # Run portfolio integration example
    portfolio_integration_example()


if __name__ == "__main__":
    run_all_examples()
