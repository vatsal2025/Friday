# Portfolio Management System Documentation

## Overview

The Portfolio Management System is a comprehensive solution for managing investment portfolios, tracking performance, handling tax implications, and maintaining target allocations. It is designed to be modular, extensible, and integrate with other components of the Friday AI Trading System.

## Components

### PortfolioManager

The `PortfolioManager` is responsible for tracking the real-time state of a portfolio, including positions, cash balance, and transaction history.

#### Key Features

- Real-time portfolio state tracking
- Position management (buys, sells, updates)
- Performance tracking
- Integration with risk management
- Historical data storage

#### Usage

```python
from portfolio.portfolio_manager import PortfolioManager

# Initialize a portfolio with ID and initial cash
portfolio = PortfolioManager("my_portfolio", 100000.0)

# Execute trades
portfolio.execute_trade("AAPL", 10, 150.0)  # Buy 10 shares at $150 each
portfolio.execute_trade("MSFT", 5, 200.0)   # Buy 5 shares at $200 each
portfolio.execute_trade("AAPL", -3, 160.0)  # Sell 3 shares at $160 each

# Update prices
portfolio.update_prices({"AAPL": 155.0, "MSFT": 210.0})

# Get portfolio value
value = portfolio.get_portfolio_value()
print(f"Portfolio value: ${value:.2f}")

# Get position details
aapl_position = portfolio.get_position_details("AAPL")
print(f"AAPL position: {aapl_position}")

# Get transaction history
transactions = portfolio.get_transaction_history()
for tx in transactions:
    print(f"{tx['date']}: {tx['action']} {tx['quantity']} {tx['symbol']} @ ${tx['price']:.2f}")
```

### PerformanceCalculator

The `PerformanceCalculator` is responsible for measuring and analyzing portfolio performance, including returns, risk metrics, and attribution.

#### Key Features

- Performance measurement (returns, volatility, Sharpe ratio, etc.)
- Performance attribution (sector, asset contributions)
- Benchmark comparison
- Risk-adjusted metrics

#### Usage

```python
from portfolio.performance_calculator import PerformanceCalculator
from datetime import datetime

# Initialize a performance calculator
performance = PerformanceCalculator()

# Add portfolio returns
performance.add_portfolio_return_observation(0.01, datetime(2023, 1, 1))
performance.add_portfolio_return_observation(0.02, datetime(2023, 1, 2))
performance.add_portfolio_return_observation(-0.01, datetime(2023, 1, 3))

# Add benchmark returns
performance.add_benchmark_return_observation(0.005, datetime(2023, 1, 1))
performance.add_benchmark_return_observation(0.015, datetime(2023, 1, 2))
performance.add_benchmark_return_observation(-0.005, datetime(2023, 1, 3))

# Calculate performance metrics
metrics = performance.calculate_performance_metrics()
print(f"Total return: {metrics['total_return']:.2%}")
print(f"Annualized return: {metrics['annualized_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

# Calculate attribution
attribution = performance.calculate_attribution()
for sector, contribution in attribution["by_category"].items():
    print(f"{sector}: {contribution:.2%}")
```

### TaxManager

The `TaxManager` is responsible for tracking tax lots, calculating realized gains/losses, and handling tax-related considerations.

#### Key Features

- Tax lot tracking (FIFO, LIFO, HIFO, LOFO, Specific ID)
- Capital gains/losses calculation
- Wash sale detection
- Tax reporting

#### Usage

```python
from portfolio.tax_manager import TaxManager, TaxLotMethod
from datetime import datetime

# Initialize a tax manager
tax_manager = TaxManager()

# Set tax lot method for a symbol
tax_manager.set_tax_lot_method("AAPL", TaxLotMethod.HIFO)

# Add tax lots
tax_manager.add_tax_lot("AAPL", 10, 150.0, datetime(2023, 1, 1))
tax_manager.add_tax_lot("AAPL", 5, 160.0, datetime(2023, 1, 15))

# Sell tax lots
result = tax_manager.sell_tax_lots("AAPL", 8, 170.0, datetime(2023, 2, 1))
print(f"Sold {result['quantity_sold']} shares with realized gain: ${result['realized_gain']:.2f}")

# Generate tax report
report = tax_manager.generate_tax_report(2023)
print(f"Short-term gains: ${report['short_term_gains']:.2f}")
print(f"Long-term gains: ${report['long_term_gains']:.2f}")
print(f"Total gains: ${report['total_gains']:.2f}")
```

### AllocationManager

The `AllocationManager` is responsible for setting and tracking target allocations, monitoring drift, and generating rebalancing recommendations.

#### Key Features

- Setting and tracking target allocations
- Monitoring drift
- Generating rebalancing recommendations
- Supporting different rebalancing methods (Threshold, Calendar, Constant Mix, Buy and Hold)

#### Usage

```python
from portfolio.allocation_manager import AllocationManager, RebalanceMethod

# Initialize an allocation manager
allocation = AllocationManager()

# Set rebalance method
allocation.set_rebalance_method(RebalanceMethod.THRESHOLD)
allocation.set_rebalance_threshold(5.0)

# Set allocation targets
allocation.set_allocation_target("AAPL", 20.0, 2.0)
allocation.set_allocation_target("MSFT", 15.0, 1.5)
allocation.set_allocation_target("CASH", 10.0, 1.0)

# Update current allocations from portfolio
portfolio_values = {"AAPL": 25000.0, "MSFT": 15000.0, "CASH": 5000.0}
categories = {"AAPL": "Technology", "MSFT": "Technology", "CASH": "Cash"}
allocation.update_allocation_from_portfolio(portfolio_values, categories)

# Check if rebalance is needed
rebalance_check = allocation.check_rebalance_needed()
print(f"Rebalance needed: {rebalance_check['rebalance_needed']}")
if rebalance_check["rebalance_needed"]:
    print(f"Reason: {rebalance_check['reason']}")

# Generate rebalance plan
plan = allocation.generate_rebalance_plan(portfolio_values, categories)
if plan["rebalance_needed"]:
    for trade in plan["trades"]:
        action = "Buy" if trade["quantity"] > 0 else "Sell"
        print(f"{action} {abs(trade['quantity']):.2f} units of {trade['symbol']} (${abs(trade['estimated_value']):.2f})")
```

### PortfolioFactory

The `PortfolioFactory` is responsible for creating and configuring portfolio management components based on a configuration.

#### Key Features

- Centralized creation of portfolio components
- Configuration-based initialization
- Integration with risk management

#### Usage

```python
from portfolio.portfolio_factory import PortfolioFactory

# Define configuration
config = {
    "portfolio_manager": {
        "portfolio_id": "my_portfolio",
        "initial_cash": 100000.0,
        "base_currency": "USD"
    },
    "performance_calculator": {
        "benchmark_symbol": "SPY",
        "risk_free_rate": 0.02
    },
    "tax_manager": {
        "default_tax_lot_method": "FIFO",
        "wash_sale_window_days": 30
    },
    "allocation_manager": {
        "rebalance_method": "THRESHOLD",
        "rebalance_threshold": 5.0,
        "allocation_targets": [
            {"name": "AAPL", "target_percentage": 20.0, "threshold": 2.0},
            {"name": "MSFT", "target_percentage": 15.0, "threshold": 1.5}
        ]
    }
}

# Create factory
factory = PortfolioFactory(config)

# Create components
portfolio = factory.create_portfolio_manager()
performance = factory.create_performance_calculator()
tax_manager = factory.create_tax_manager()
allocation = factory.create_allocation_manager()

# Use components
print(f"Portfolio ID: {portfolio.portfolio_id}")
print(f"Initial cash: ${portfolio.cash:.2f}")
```

## Command-Line Interface

The Portfolio Management System includes a command-line interface (CLI) for interacting with the system.

### Usage

```bash
# Create a new portfolio
python -m portfolio.cli create --portfolio-id my_portfolio --initial-cash 100000.0

# Execute a trade
python -m portfolio.cli trade AAPL 10 150.0 --date 2023-01-01 --update-tax

# Update prices
python -m portfolio.cli prices AAPL:155.0 MSFT:210.0 --date 2023-01-02

# Show portfolio information
python -m portfolio.cli show --transactions --history

# Analyze performance
python -m portfolio.cli performance --attribution

# Check allocation and generate rebalance plan
python -m portfolio.cli allocation --rebalance

# Generate tax report
python -m portfolio.cli tax --year 2023 --detailed --export tax_report_2023.csv
```

## Configuration

The Portfolio Management System can be configured using a JSON configuration file. A sample configuration file is provided in `sample_config.json`.

### Example Configuration

```json
{
    "portfolio_manager": {
        "portfolio_id": "sample_portfolio",
        "initial_cash": 100000.0,
        "base_currency": "USD",
        "track_history": true,
        "history_granularity": "daily"
    },
    "performance_calculator": {
        "benchmark_symbol": "SPY",
        "risk_free_rate": 0.02,
        "calculation_frequency": "daily",
        "rolling_window_days": 252
    },
    "tax_manager": {
        "default_tax_lot_method": "FIFO",
        "wash_sale_window_days": 30,
        "long_term_threshold_days": 365,
        "symbol_specific_methods": {
            "AAPL": "HIFO",
            "MSFT": "LIFO"
        }
    },
    "allocation_manager": {
        "rebalance_method": "THRESHOLD",
        "rebalance_threshold": 5.0,
        "calendar_rebalance_frequency": "QUARTERLY",
        "allocation_targets": [
            {
                "name": "US_STOCKS",
                "category": "equity",
                "target_percentage": 60.0,
                "threshold": 5.0,
                "assets": [
                    {"symbol": "AAPL", "target_percentage": 15.0},
                    {"symbol": "MSFT", "target_percentage": 15.0},
                    {"symbol": "AMZN", "target_percentage": 10.0},
                    {"symbol": "GOOGL", "target_percentage": 10.0},
                    {"symbol": "META", "target_percentage": 10.0}
                ]
            },
            {
                "name": "BONDS",
                "category": "fixed_income",
                "target_percentage": 30.0,
                "threshold": 3.0,
                "assets": [
                    {"symbol": "AGG", "target_percentage": 50.0},
                    {"symbol": "BND", "target_percentage": 50.0}
                ]
            },
            {
                "name": "CASH",
                "category": "cash",
                "target_percentage": 10.0,
                "threshold": 2.0,
                "assets": [
                    {"symbol": "CASH", "target_percentage": 100.0}
                ]
            }
        ]
    },
    "risk_management": {
        "use_risk_manager": true,
        "max_position_size_percentage": 20.0,
        "max_sector_exposure_percentage": 40.0,
        "stop_loss_percentage": 15.0,
        "var_confidence_level": 0.95,
        "var_time_horizon_days": 1,
        "stress_test_scenarios": [
            {
                "name": "Market Crash",
                "asset_shocks": {
                    "equity": -0.30,
                    "fixed_income": -0.10,
                    "cash": 0.0
                }
            },
            {
                "name": "Rising Rates",
                "asset_shocks": {
                    "equity": -0.05,
                    "fixed_income": -0.15,
                    "cash": 0.01
                }
            }
        ]
    }
}
```

## Integration with Risk Management System

The Portfolio Management System can integrate with the Risk Management System to provide risk metrics and constraints.

### Example Integration

```python
from portfolio.portfolio_factory import PortfolioFactory
from risk.risk_management_factory import RiskManagementFactory

# Create portfolio factory
portfolio_factory = PortfolioFactory(portfolio_config)

# Create risk management factory
risk_factory = RiskManagementFactory(risk_config)

# Create portfolio manager
portfolio = portfolio_factory.create_portfolio_manager()

# Create risk manager
risk_manager = risk_factory.create_risk_manager()

# Set risk manager in portfolio manager
portfolio.set_risk_manager(risk_manager)

# Get risk metrics
risk_metrics = portfolio.get_risk_metrics()
print(f"Value at Risk (95%): ${risk_metrics['var_95']:.2f}")
print(f"Expected Shortfall: ${risk_metrics['expected_shortfall']:.2f}")
```

## Testing

The Portfolio Management System includes comprehensive unit tests and integration tests.

### Running Tests

```bash
# Run unit tests
python -m unittest discover -s src/portfolio

# Run integration tests
python -m portfolio.test_portfolio_integration
```

## Dependencies

The Portfolio Management System has the following dependencies:

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- PyFolio (optional)
- Empyrical (optional)
- scikit-learn (optional)
- statsmodels (optional)

See `requirements.txt` for specific version requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
