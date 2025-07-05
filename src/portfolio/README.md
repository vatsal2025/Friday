# Portfolio Management System

The Portfolio Management System is a comprehensive solution for tracking, analyzing, and managing investment portfolios. It provides tools for portfolio state tracking, performance measurement, tax-aware trading, and asset allocation.

## Components

### PortfolioManager

The `PortfolioManager` is responsible for real-time portfolio state tracking, including:

- Position management (quantities, average prices, current values)
- Cash balance tracking
- Trade execution and recording
- Historical portfolio value tracking
- Integration with risk management (optional)

```python
from portfolio.portfolio_manager import PortfolioManager

# Create a portfolio manager
portfolio = PortfolioManager(portfolio_id="my-portfolio", initial_cash=100000.0)

# Execute trades
portfolio.execute_trade("AAPL", 50, 150.0, datetime.now())

# Update prices
portfolio.update_prices({"AAPL": 160.0})

# Get portfolio value
value = portfolio.get_portfolio_value()
```

### PerformanceCalculator

The `PerformanceCalculator` provides comprehensive performance measurement, including:

- Return calculations (total, annualized, period-specific)
- Risk-adjusted metrics (Sharpe ratio, Sortino ratio)
- Performance attribution (by sector, asset)
- Benchmark comparison (alpha, beta, tracking error)

```python
from portfolio.performance_calculator import PerformanceCalculator

# Create a performance calculator
performance = PerformanceCalculator(benchmark_symbol="SPY")

# Add portfolio returns and values
performance.add_portfolio_return_observation(0.01, datetime.now())
performance.add_portfolio_value_observation(101000.0, datetime.now())

# Calculate performance metrics
metrics = performance.calculate_performance_metrics()
```

### TaxManager

The `TaxManager` handles tax-aware trading and reporting, including:

- Tax lot tracking with multiple methods (FIFO, LIFO, HIFO, etc.)
- Capital gains/losses calculation
- Wash sale detection
- Tax reporting

```python
from portfolio.tax_manager import TaxManager, TaxLotMethod

# Create a tax manager
tax_manager = TaxManager(default_method=TaxLotMethod.FIFO)

# Add tax lots
tax_manager.add_tax_lot("AAPL", 50, 150.0, datetime.now() - timedelta(days=100))

# Sell tax lots
sale = tax_manager.sell_tax_lots("AAPL", 25, 170.0)

# Get realized gains
gains = tax_manager.get_realized_gains()
```

### AllocationManager

The `AllocationManager` manages asset allocation, including:

- Setting and tracking target allocations
- Monitoring allocation drift
- Generating rebalancing recommendations
- Supporting multiple rebalancing methods (threshold, calendar, etc.)

```python
from portfolio.allocation_manager import AllocationManager, RebalanceMethod

# Create an allocation manager
allocation = AllocationManager(rebalance_method=RebalanceMethod.THRESHOLD, default_threshold=5.0)

# Set allocation targets
allocation.set_allocation_target("AAPL", 25.0, "stocks")
allocation.set_allocation_target("BND", 30.0, "bonds")

# Update current allocations
portfolio_values = {"AAPL": 30000.0, "BND": 25000.0, "CASH": 10000.0}
categories = {"AAPL": "stocks", "BND": "bonds", "CASH": "cash"}
allocation.update_allocation_from_portfolio(portfolio_values, categories)

# Check if rebalance is needed
rebalance_needed = allocation.check_rebalance_needed()

# Generate rebalance plan
plan = allocation.generate_rebalance_plan(portfolio_values, categories)
```

### PortfolioFactory

The `PortfolioFactory` simplifies the creation and configuration of portfolio components:

```python
from portfolio.portfolio_factory import PortfolioFactory

# Configuration
config = {
    "portfolio_manager": {
        "portfolio_id": "my-portfolio",
        "initial_cash": 100000.0
    },
    "performance_calculator": {
        "benchmark_symbol": "SPY"
    },
    "tax_manager": {
        "default_method": "FIFO"
    },
    "allocation_manager": {
        "rebalance_method": "THRESHOLD",
        "default_threshold": 5.0,
        "allocation_targets": [
            {"name": "AAPL", "target_percentage": 25.0, "category": "stocks"},
            {"name": "BND", "target_percentage": 30.0, "category": "bonds"}
        ]
    }
}

# Create factory
factory = PortfolioFactory(config)

# Create complete system
system = factory.create_complete_portfolio_system()
portfolio = system["portfolio_manager"]
performance = system["performance_calculator"]
tax = system["tax_manager"]
allocation = system["allocation_manager"]
```

## Integration with Risk Management

```python
# Create a portfolio with a risk manager
from risk.risk_manager import RiskManager
from datetime import datetime

risk_manager = RiskManager()
portfolio = PortfolioFactory.create_portfolio(risk_manager=risk_manager)

# Execute trades
portfolio.execute_trade("AAPL", 10, 150.0)
portfolio.update_prices({"AAPL": 155.0})

# Add historical price data for risk analysis
historical_date = datetime(2023, 1, 1)
historical_prices = {"AAPL": 145.0}
portfolio.add_historical_prices(historical_prices, historical_date)

# Get risk metrics
risk_metrics = portfolio.get_risk_metrics()
print(f"Current VaR: {risk_metrics['var_95']}")
print(f"Portfolio volatility: {risk_metrics['volatility']}")
```

## Historical Price Management

```python
from datetime import datetime, timedelta
import numpy as np

# Create portfolio
portfolio = PortfolioFactory.create_portfolio(risk_manager=risk_manager)

# Execute trades
portfolio.execute_trade("AAPL", 10, 150.0)
portfolio.execute_trade("MSFT", 5, 200.0)

# Add historical prices for multiple days
for i in range(30):  # 30 days of historical data
    date = datetime(2023, 1, 1) + timedelta(days=i)
    # Create some price movement
    aapl_price = 145.0 + (i * 0.5) + (np.sin(i) * 2)  # Some trend and oscillation
    msft_price = 195.0 + (i * 0.3) + (np.cos(i) * 3)
    
    # Add historical prices with a maximum history size of 90 days
    portfolio.add_historical_prices({
        "AAPL": aapl_price,
        "MSFT": msft_price
    }, date, max_history_size=90)

# Get historical values
historical_values = portfolio.get_historical_values()
print(f"Number of historical snapshots: {len(historical_values)}")

# Calculate performance metrics using historical data
performance_metrics = portfolio.calculate_performance_metrics()
print(f"Total return: {performance_metrics['total_return']}")
```

## Running Tests

To run the integration tests for the Portfolio Management System:

```bash
python -m unittest portfolio.test_portfolio_integration
```
