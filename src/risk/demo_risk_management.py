#!/usr/bin/env python
"""
Demo script to demonstrate the usage of the risk management system with the integrated RiskMetricsCalculator.

This script creates a simulated portfolio and shows how the risk management system
can be used to monitor and manage risk using the factory pattern and the RiskMetricsCalculator.
"""

import random
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from risk_management_factory import RiskManagementFactory
from production_config import RiskManagementProductionConfig, create_default_production_config


def generate_random_portfolio(num_assets=10, min_value=5000, max_value=20000):
    """
    Generate a random portfolio of assets.

    Args:
        num_assets: Number of assets in the portfolio
        min_value: Minimum market value per asset
        max_value: Maximum market value per asset

    Returns:
        Dictionary of positions with market values and sectors
    """
    sectors = ["Technology", "Financials", "Healthcare", "Consumer", "Energy", "Utilities"]
    tickers = [f"STOCK{i}" for i in range(1, num_assets + 1)]

    positions = {}
    for ticker in tickers:
        positions[ticker] = {
            "market_value": random.uniform(min_value, max_value),
            "sector": random.choice(sectors)
        }

    return positions


def simulate_portfolio_performance(initial_value, days=60, daily_return_mean=0.0005, daily_return_std=0.01):
    """
    Simulate portfolio performance over time.

    Args:
        initial_value: Initial portfolio value
        days: Number of days to simulate
        daily_return_mean: Mean daily return
        daily_return_std: Standard deviation of daily returns

    Returns:
        List of (timestamp, portfolio_value) tuples
    """
    now = datetime.now()
    performance = []
    current_value = initial_value

    # Generate past data first (oldest to newest)
    for i in range(days, 0, -1):
        timestamp = now - timedelta(days=i)
        daily_return = np.random.normal(daily_return_mean, daily_return_std)
        current_value *= (1 + daily_return)
        performance.append((timestamp, current_value))

    # Add current day
    performance.append((now, current_value))

    return performance


def plot_risk_metrics(timestamps, values, var_values, drawdown_values):
    """
    Plot portfolio performance and risk metrics.

    Args:
        timestamps: List of timestamps
        values: List of portfolio values
        var_values: List of VaR values
        drawdown_values: List of drawdown values
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot portfolio value
    ax1.plot(timestamps, values, label="Portfolio Value")
    ax1.set_title("Portfolio Performance")
    ax1.set_ylabel("Value ($)")
    ax1.legend()
    ax1.grid(True)

    # Plot risk metrics
    ax2.plot(timestamps, var_values, label="Value at Risk (VaR)")
    ax2.plot(timestamps, drawdown_values, label="Drawdown")
    ax2.set_title("Risk Metrics")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Risk Measure (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("risk_metrics_demo.png")
    plt.show()


def main():
    # Create a production configuration
    config = create_default_production_config()
    config.var_confidence_level = 0.95

    # Create a factory with the configuration
    factory = RiskManagementFactory(config)

    # Create an advanced risk manager using the factory
    risk_manager = factory.create_advanced_risk_manager()

    # Generate a random portfolio
    initial_portfolio_value = 100000
    positions = generate_random_portfolio(num_assets=8)

    # Simulate portfolio performance
    performance_data = simulate_portfolio_performance(initial_portfolio_value, days=60)

    # Track risk metrics over time
    timestamps = []
    portfolio_values = []
    var_values = []
    drawdown_values = []

    # Update the risk manager with historical data
    print("Updating risk manager with historical data...")
    for timestamp, portfolio_value in performance_data:
        # Update positions slightly each day to simulate trading
        updated_positions = {}
        for ticker, position in positions.items():
            updated_positions[ticker] = {
                "market_value": position["market_value"] * (1 + np.random.normal(0, 0.005)),
                "sector": position["sector"]
            }
        positions = updated_positions

        # Update the risk manager
        risk_manager.portfolio_risk_manager.update_portfolio(positions, portfolio_value, timestamp)

        # Get risk metrics
        metrics = risk_manager.get_risk_metrics()

        # Store data for plotting
        timestamps.append(timestamp)
        portfolio_values.append(portfolio_value)
        var_values.append(metrics.get('var', 0) * 100)  # Convert to percentage
        drawdown_values.append(metrics.get('current_drawdown', 0) * 100)  # Convert to percentage

    # Print final risk metrics
    print("\nFinal Risk Metrics:")
    final_metrics = risk_manager.get_risk_metrics()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot the results
    plot_risk_metrics(timestamps, portfolio_values, var_values, drawdown_values)


if __name__ == "__main__":
    main()
