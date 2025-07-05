#!/usr/bin/env python
"""Command-line interface for the Portfolio Management System.

This module provides a command-line interface for interacting with the
Portfolio Management System, allowing users to create and manage portfolios,
execute trades, analyze performance, and generate reports.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import portfolio components
from portfolio.portfolio_manager import PortfolioManager
from portfolio.performance_calculator import PerformanceCalculator
from portfolio.tax_manager import TaxManager, TaxLotMethod
from portfolio.allocation_manager import AllocationManager, RebalanceMethod
from portfolio.portfolio_factory import PortfolioFactory
from portfolio.config import PortfolioConfig

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use default.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        try:
            return PortfolioConfig.load_from_file(config_path)
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration instead")

    return PortfolioConfig.get_default_config()


def create_portfolio(args: argparse.Namespace) -> None:
    """Create a new portfolio.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)

    # Override portfolio ID and initial cash if provided
    if args.portfolio_id:
        config["portfolio_manager"]["portfolio_id"] = args.portfolio_id
    if args.initial_cash is not None:
        config["portfolio_manager"]["initial_cash"] = args.initial_cash

    # Create portfolio
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()

    logger.info(f"Created portfolio '{portfolio.portfolio_id}' with initial cash ${portfolio.cash:.2f}")

    # Save configuration if output path provided
    if args.output:
        try:
            PortfolioConfig.save_to_file(config, args.output)
            logger.info(f"Saved configuration to {args.output}")
        except Exception as e:
            logger.error(f"Error saving configuration to {args.output}: {e}")


def execute_trade(args: argparse.Namespace) -> None:
    """Execute a trade in the portfolio.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()

    # Parse trade date
    if args.date:
        try:
            trade_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Expected format: YYYY-MM-DD")
            return
    else:
        trade_date = datetime.now()

    # Execute trade
    try:
        portfolio.execute_trade(
            args.symbol, 
            args.quantity, 
            args.price, 
            trade_date,
            sector=args.sector,
            asset_class=args.asset_class
        )
        logger.info(f"Executed trade: {args.quantity} {args.symbol} @ ${args.price:.2f}")
        if args.sector or args.asset_class:
            details = []
            if args.sector:
                details.append(f"sector: {args.sector}")
            if args.asset_class:
                details.append(f"asset class: {args.asset_class}")
            logger.info(f"Trade details: {', '.join(details)}")

        # Update tax manager if available
        if args.update_tax and "tax_manager" in config:
            tax_manager = factory.create_tax_manager()
            if args.quantity > 0:  # Buy
                tax_manager.add_tax_lot(args.symbol, args.quantity, args.price, trade_date)
                logger.info(f"Added tax lot: {args.quantity} {args.symbol} @ ${args.price:.2f}")
            else:  # Sell
                sale = tax_manager.sell_tax_lots(args.symbol, abs(args.quantity), args.price, trade_date)
                logger.info(f"Sold tax lots: {sale['quantity_sold']} {args.symbol} @ ${args.price:.2f}")
                logger.info(f"Realized gain: ${sale['realized_gain']:.2f}")
    except Exception as e:
        logger.error(f"Error executing trade: {e}")


def update_prices(args: argparse.Namespace) -> None:
    """Update prices in the portfolio.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()

    # Parse prices
    prices = {}
    for price_str in args.prices:
        try:
            symbol, price = price_str.split(":")
            prices[symbol] = float(price)
        except ValueError:
            logger.error(f"Invalid price format: {price_str}. Expected format: SYMBOL:PRICE")
            continue

    # Parse date
    if args.date:
        try:
            price_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Expected format: YYYY-MM-DD")
            return
    else:
        price_date = datetime.now()

    # Update prices
    try:
        portfolio.update_prices(prices, price_date)
        logger.info(f"Updated prices for {len(prices)} symbols")
        for symbol, price in prices.items():
            logger.info(f"  {symbol}: ${price:.2f}")
    except Exception as e:
        logger.error(f"Error updating prices: {e}")


def show_portfolio(args: argparse.Namespace) -> None:
    """Show portfolio information.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()

    # Print portfolio summary
    logger.info(f"Portfolio ID: {portfolio.portfolio_id}")
    logger.info(f"Cash balance: ${portfolio.cash:.2f}")
    logger.info(f"Portfolio value: ${portfolio.get_portfolio_value():.2f}")
    logger.info(f"Number of positions: {len(portfolio.positions)}")

    # Print positions
    if portfolio.positions:
        logger.info("\nPositions:")
        for symbol, position in portfolio.positions.items():
            current_value = position["quantity"] * position["last_price"]
            profit_loss = current_value - (position["quantity"] * position["average_price"])
            profit_loss_pct = (profit_loss / (position["quantity"] * position["average_price"])) * 100

            logger.info(f"{symbol}: {position['quantity']} shares @ ${position['average_price']:.2f} avg cost, "
                       f"current price: ${position['last_price']:.2f}, "
                       f"value: ${current_value:.2f}, "
                       f"P/L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")

    # Print transaction history if requested
    if args.transactions:
        transactions = portfolio.get_transaction_history()
        if transactions:
            logger.info("\nTransaction History:")
            for tx in transactions:
                logger.info(f"{tx['date'].strftime('%Y-%m-%d')}: {tx['action']} {tx['quantity']} {tx['symbol']} @ ${tx['price']:.2f}")
        else:
            logger.info("No transactions found")

    # Print historical values if requested
    if args.history:
        historical_values = portfolio.get_historical_values()
        if historical_values:
            logger.info("\nHistorical Portfolio Values:")
            for date, value in historical_values.items():
                logger.info(f"{date.strftime('%Y-%m-%d')}: ${value:.2f}")
        else:
            logger.info("No historical values found")


def analyze_performance(args: argparse.Namespace) -> None:
    """Analyze portfolio performance.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()
    performance = factory.create_performance_calculator()

    # Add portfolio values to performance calculator if not already added
    historical_values = portfolio.get_historical_values()
    if historical_values and not performance.portfolio_values:
        for date, value in historical_values.items():
            performance.add_portfolio_value_observation(value, date)

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

    # Print benchmark comparison if available
    if "benchmark_comparison" in metrics and metrics["benchmark_comparison"]:
        benchmark_comp = metrics["benchmark_comparison"]
        logger.info("\nBenchmark Comparison:")
        logger.info(f"Alpha: {benchmark_comp['alpha']:.4f}")
        logger.info(f"Beta: {benchmark_comp['beta']:.2f}")
        logger.info(f"Tracking Error: {benchmark_comp['tracking_error']:.2%}")
        logger.info(f"Information Ratio: {benchmark_comp['information_ratio']:.2f}")

    # Print attribution if requested
    if args.attribution and "attribution" in metrics:
        attribution = metrics["attribution"]
        logger.info("\nPerformance Attribution:")
        for category, contribution in attribution["by_category"].items():
            logger.info(f"{category}: {contribution:.2%}")

        logger.info("\nAsset Contribution:")
        for asset, contribution in attribution["by_asset"].items():
            logger.info(f"{asset}: {contribution:.2%}")


def check_allocation(args: argparse.Namespace) -> None:
    """Check portfolio allocation and generate rebalance plan.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    portfolio = factory.create_portfolio_manager()
    allocation = factory.create_allocation_manager()

    # Get current portfolio values
    portfolio_values = {
        symbol: position["quantity"] * position["last_price"]
        for symbol, position in portfolio.positions.items()
    }
    portfolio_values["CASH"] = portfolio.cash

    # Define asset categories (use from config or default to "default")
    categories = {}
    for symbol in portfolio_values.keys():
        if symbol == "CASH":
            categories[symbol] = "cash"
            continue

        # Try to find category in allocation targets
        category = "default"
        for target in config["allocation_manager"].get("allocation_targets", []):
            if target["name"] == symbol:
                category = target.get("category", "default")
                break

        categories[symbol] = category

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

    # Generate rebalance plan if requested
    if args.rebalance:
        plan = allocation.generate_rebalance_plan(portfolio_values, categories)

        if plan["rebalance_needed"]:
            logger.info("\nRebalance Plan:")
            for trade in plan["trades"]:
                action = "Buy" if trade["quantity"] > 0 else "Sell"
                logger.info(f"{action} {abs(trade['quantity']):.2f} units of {trade['symbol']} "
                           f"(${abs(trade['estimated_value']):.2f})")
        else:
            logger.info("No rebalancing needed")

    # Print drift information
    drift_info = allocation.get_drift_information()
    logger.info("\nAllocation Drift:")
    for symbol, drift in drift_info.items():
        logger.info(f"{symbol}: Target {drift['target']:.2f}%, Current {drift['current']:.2f}%, "
                   f"Drift {drift['drift']:.2f}%, Threshold {drift['threshold']:.2f}%")


def generate_tax_report(args: argparse.Namespace) -> None:
    """Generate tax report.

    Args:
        args: Command-line arguments
    """
    config = load_config(args.config)
    factory = PortfolioFactory(config)
    tax_manager = factory.create_tax_manager()

    # Parse year
    year = args.year if args.year else datetime.now().year

    # Generate tax report
    tax_report = tax_manager.generate_tax_report(year)

    # Print tax report
    logger.info(f"Tax Report for {year}:")
    logger.info(f"Short-term gains: ${tax_report['short_term_gains']:.2f}")
    logger.info(f"Long-term gains: ${tax_report['long_term_gains']:.2f}")
    logger.info(f"Total realized gains: ${tax_report['total_gains']:.2f}")
    logger.info(f"Wash sales: ${tax_report['wash_sale_amount']:.2f}")

    # Print detailed realized gains if requested
    if args.detailed:
        realized_gains = tax_manager.get_realized_gains(year=year)
        if realized_gains:
            logger.info("\nDetailed Realized Gains:")
            for gain in realized_gains:
                gain_type = "Long-term" if gain["long_term"] else "Short-term"
                wash_sale = " (Wash Sale)" if gain["wash_sale"] else ""
                logger.info(f"{gain['date'].strftime('%Y-%m-%d')}: {gain_type}{wash_sale} "
                           f"{gain['symbol']} {gain['quantity']} shares, "
                           f"gain/loss: ${gain['realized_gain']:.2f}")
        else:
            logger.info("No realized gains found for the specified year")

    # Export to CSV if requested
    if args.export:
        try:
            df = tax_manager.get_realized_gains_dataframe(year=year)
            df.to_csv(args.export, index=False)
            logger.info(f"Exported realized gains to {args.export}")
        except Exception as e:
            logger.error(f"Error exporting realized gains: {e}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Portfolio Management System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", "-c", help="Path to configuration file")

    # Create portfolio command
    create_parser = subparsers.add_parser("create", help="Create a new portfolio", parents=[common_parser])
    create_parser.add_argument("--portfolio-id", "-p", help="Portfolio ID")
    create_parser.add_argument("--initial-cash", "-i", type=float, help="Initial cash amount")
    create_parser.add_argument("--output", "-o", help="Path to save configuration")
    create_parser.set_defaults(func=create_portfolio)

    # Execute trade command
    trade_parser = subparsers.add_parser("trade", help="Execute a trade", parents=[common_parser])
    trade_parser.add_argument("symbol", help="Symbol to trade")
    trade_parser.add_argument("quantity", type=float, help="Quantity to trade (positive for buy, negative for sell)")
    trade_parser.add_argument("price", type=float, help="Price per unit")
    trade_parser.add_argument("--date", "-d", help="Trade date (YYYY-MM-DD)")
    trade_parser.add_argument("--update-tax", "-t", action="store_true", help="Update tax lots")
    trade_parser.add_argument("--sector", "-s", help="Sector of the asset (e.g., 'Technology', 'Healthcare')")
    trade_parser.add_argument("--asset-class", "-a", help="Asset class (e.g., 'equities', 'options', 'futures', 'forex', 'crypto')")
    trade_parser.set_defaults(func=execute_trade)

    # Update prices command
    prices_parser = subparsers.add_parser("prices", help="Update prices", parents=[common_parser])
    prices_parser.add_argument("prices", nargs="+", help="Prices in format SYMBOL:PRICE")
    prices_parser.add_argument("--date", "-d", help="Price date (YYYY-MM-DD)")
    prices_parser.set_defaults(func=update_prices)

    # Show portfolio command
    show_parser = subparsers.add_parser("show", help="Show portfolio information", parents=[common_parser])
    show_parser.add_argument("--transactions", "-t", action="store_true", help="Show transaction history")
    show_parser.add_argument("--history", "-h", action="store_true", help="Show historical values")
    show_parser.set_defaults(func=show_portfolio)

    # Analyze performance command
    performance_parser = subparsers.add_parser("performance", help="Analyze portfolio performance", parents=[common_parser])
    performance_parser.add_argument("--attribution", "-a", action="store_true", help="Show performance attribution")
    performance_parser.set_defaults(func=analyze_performance)

    # Check allocation command
    allocation_parser = subparsers.add_parser("allocation", help="Check portfolio allocation", parents=[common_parser])
    allocation_parser.add_argument("--rebalance", "-r", action="store_true", help="Generate rebalance plan")
    allocation_parser.set_defaults(func=check_allocation)

    # Generate tax report command
    tax_parser = subparsers.add_parser("tax", help="Generate tax report", parents=[common_parser])
    tax_parser.add_argument("--year", "-y", type=int, help="Tax year")
    tax_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed realized gains")
    tax_parser.add_argument("--export", "-e", help="Export realized gains to CSV")
    tax_parser.set_defaults(func=generate_tax_report)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
