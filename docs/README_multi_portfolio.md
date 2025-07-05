# Multi-Portfolio Support

## Overview

The multi-portfolio support feature extends the Friday trading system to manage multiple portfolios simultaneously, with portfolio isolation, portfolio-specific event routing, portfolio grouping, and cross-portfolio analysis capabilities. This document provides an overview of the new functionality and how to use it.

## Key Features

### Multiple Portfolio Management

- **Simultaneous Management**: Manage multiple portfolios with different strategies, risk profiles, and objectives.
- **Portfolio Isolation**: Each portfolio has its own isolated state, including positions, cash, performance metrics, and risk parameters.
- **Lifecycle Management**: Create, activate, deactivate, and delete portfolios as needed.

### Portfolio-Specific Event Routing

- **Targeted Events**: Route market data updates, trade executions, and other events to specific portfolios.
- **Isolated Processing**: Each portfolio processes events independently, ensuring changes in one portfolio don't affect others.

### Portfolio Grouping and Aggregation

- **Portfolio Groups**: Organize portfolios into logical groups based on strategy, asset class, or other criteria.
- **Allocation Management**: Assign and track target allocations within portfolio groups.
- **Consolidated Reporting**: Generate reports that aggregate data across multiple portfolios or groups.

### Cross-Portfolio Analysis

- **Performance Comparison**: Compare performance metrics across portfolios.
- **Correlation Analysis**: Calculate correlations between portfolio returns.
- **Diversification Analysis**: Analyze diversification benefits across multiple portfolios.

## Architecture

The multi-portfolio support is implemented through the following key components:

### PortfolioRegistry

The `PortfolioRegistry` class is the central component that manages multiple portfolio instances. It provides methods for:

- Creating, activating, deactivating, and deleting portfolios
- Managing portfolio groups
- Performing cross-portfolio analysis

### MultiPortfolioIntegration

The `MultiPortfolioIntegration` class extends the existing `PortfolioIntegration` class to work with the `PortfolioRegistry`. It provides a high-level interface for:

- Managing multiple portfolios and groups
- Integrating with the event system, trading engine, and market data service
- Routing events to the appropriate portfolios

### PortfolioGroup

The `PortfolioGroup` class represents a collection of portfolios with associated metadata and allocation information. It provides methods for:

- Adding and removing portfolios from the group
- Managing allocations within the group
- Generating consolidated reports for the group

## Usage Examples

### Creating and Managing Portfolios

```python
# Initialize the multi-portfolio integration
from src.portfolio.multi_portfolio_integration import MultiPortfolioIntegration

multi_integration = MultiPortfolioIntegration(
    event_system=event_system,
    trading_engine=trading_engine,
    market_data_service=market_data_service,
    default_config=config
)

# Create portfolios with different strategies
growth_portfolio_id = multi_integration.create_portfolio(
    name="Growth Portfolio",
    initial_capital=100000.0,
    description="Aggressive growth strategy",
    tags=["growth", "high-risk"],
    config={
        "risk_management": {
            "max_position_size": 0.08  # 8% of portfolio
        }
    }
)

income_portfolio_id = multi_integration.create_portfolio(
    name="Income Portfolio",
    initial_capital=200000.0,
    description="Income-focused strategy",
    tags=["income", "low-risk"],
    config={
        "risk_management": {
            "max_position_size": 0.04  # 4% of portfolio
        }
    }
)

# Set the active portfolio
multi_integration.set_active_portfolio(growth_portfolio_id)

# Deactivate a portfolio when not in use
multi_integration.deactivate_portfolio(income_portfolio_id)

# Delete a portfolio when no longer needed
multi_integration.delete_portfolio(income_portfolio_id)
```

### Creating and Managing Portfolio Groups

```python
# Create a portfolio group
equity_group_id = multi_integration.create_portfolio_group(
    name="Equity Strategies",
    portfolio_ids=[growth_portfolio_id, value_portfolio_id],
    description="Portfolios focused on equity investments",
    allocation={
        growth_portfolio_id: 0.6,  # 60% allocation
        value_portfolio_id: 0.4     # 40% allocation
    }
)

# Add a portfolio to a group
multi_integration.add_to_group(equity_group_id, international_portfolio_id, allocation=0.3)

# Remove a portfolio from a group
multi_integration.remove_from_group(equity_group_id, value_portfolio_id)

# Delete a portfolio group
multi_integration.delete_portfolio_group(equity_group_id)
```

### Cross-Portfolio Analysis

```python
# Compare performance metrics across portfolios
comparison = multi_integration.compare_portfolios(
    portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id],
    metrics=["returns", "volatility", "sharpe_ratio", "max_drawdown"]
)

# Calculate correlation between portfolios
correlation = multi_integration.calculate_correlation(
    portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id]
)

# Analyze diversification across portfolios
diversification = multi_integration.analyze_diversification(
    portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id]
)

# Generate a consolidated report for all portfolios
report = multi_integration.generate_consolidated_report()
```

### Event Routing

```python
# Create an event with a specific portfolio_id
event = {
    "type": "MARKET_DATA_UPDATE",
    "data": {
        "portfolio_id": growth_portfolio_id,  # Route to specific portfolio
        "symbol": "AAPL",
        "price": 150.0
    }
}

# Publish the event
event_system.publish(event)

# For backward compatibility, if no portfolio_id is specified,
# the event is routed to the active portfolio
event = {
    "type": "MARKET_DATA_UPDATE",
    "data": {
        "symbol": "MSFT",
        "price": 250.0
    }
}

event_system.publish(event)  # Routed to active portfolio
```

## Configuration

### Default Configuration

The default configuration is applied to all portfolios unless overridden by portfolio-specific configurations:

```python
default_config = {
    "risk_management": {
        "max_position_size": 0.05,  # 5% of portfolio
        "max_sector_exposure": 0.25,  # 25% of portfolio
        "stop_loss_percentage": 0.15  # 15% stop loss
    },
    "tax_settings": {
        "tax_rate_short_term": 0.35,  # 35% short-term tax rate
        "tax_rate_long_term": 0.15,  # 15% long-term tax rate
        "tax_loss_harvesting": True
    },
    "performance_settings": {
        "benchmark": "SPY",  # S&P 500 ETF as benchmark
        "risk_free_rate": 0.02  # 2% risk-free rate
    }
}
```

### Portfolio-Specific Configuration

Each portfolio can have its own configuration that overrides the default settings:

```python
portfolio_config = {
    "risk_management": {
        "max_position_size": 0.08,  # Override just this setting
        # Other settings inherit from default_config
    }
}
```

## Migration from Single-Portfolio System

For guidance on migrating from the single-portfolio system to the new multi-portfolio system, please refer to the [Migration Guide](migration_to_multi_portfolio.md).

## Examples

For complete examples of using the multi-portfolio support, see the [Multi-Portfolio Configuration Example](../examples/multi_portfolio_config.py).

## API Reference

### MultiPortfolioIntegration

#### Portfolio Management

- `create_portfolio(name, initial_capital, description=None, tags=None, config=None)`: Create a new portfolio
- `delete_portfolio(portfolio_id)`: Delete a portfolio
- `activate_portfolio(portfolio_id)`: Activate a portfolio
- `deactivate_portfolio(portfolio_id)`: Deactivate a portfolio
- `set_active_portfolio(portfolio_id)`: Set the active portfolio
- `get_active_portfolio_id()`: Get the ID of the active portfolio
- `get_all_portfolio_ids()`: Get IDs of all portfolios
- `get_active_portfolio_ids()`: Get IDs of all active portfolios

#### Group Management

- `create_portfolio_group(name, portfolio_ids, description=None, allocation=None)`: Create a portfolio group
- `delete_portfolio_group(group_id)`: Delete a portfolio group
- `add_to_group(group_id, portfolio_id, allocation=None)`: Add a portfolio to a group
- `remove_from_group(group_id, portfolio_id)`: Remove a portfolio from a group
- `get_all_group_ids()`: Get IDs of all portfolio groups
- `get_portfolio_group(group_id)`: Get a portfolio group by ID

#### Cross-Portfolio Analysis

- `compare_portfolios(portfolio_ids, metrics)`: Compare portfolios across metrics
- `calculate_correlation(portfolio_ids)`: Calculate correlation between portfolios
- `analyze_diversification(portfolio_ids)`: Analyze diversification across portfolios
- `generate_consolidated_report()`: Generate a consolidated report for all portfolios

#### Event Handling

- `handle_market_data_update(event)`: Handle market data update events
- `handle_trade_execution(event)`: Handle trade execution events
- `handle_portfolio_update_request(event)`: Handle portfolio update request events
- `handle_rebalance_request(event)`: Handle rebalance request events

### PortfolioRegistry

- `create_portfolio(name, initial_capital, description=None, tags=None, config=None)`: Create a new portfolio
- `delete_portfolio(portfolio_id)`: Delete a portfolio
- `activate_portfolio(portfolio_id)`: Activate a portfolio
- `deactivate_portfolio(portfolio_id)`: Deactivate a portfolio
- `get_portfolio_components(portfolio_id)`: Get all components of a portfolio
- `get_portfolio_manager(portfolio_id)`: Get the portfolio manager for a portfolio
- `get_performance_calculator(portfolio_id)`: Get the performance calculator for a portfolio
- `get_tax_manager(portfolio_id)`: Get the tax manager for a portfolio
- `get_allocation_manager(portfolio_id)`: Get the allocation manager for a portfolio
- `get_risk_manager(portfolio_id)`: Get the risk manager for a portfolio

### PortfolioGroup

- `add_portfolio(portfolio_id, allocation=None)`: Add a portfolio to the group
- `remove_portfolio(portfolio_id)`: Remove a portfolio from the group
- `update_allocation(portfolio_id, allocation)`: Update the allocation for a portfolio
- `normalize_allocation()`: Normalize allocations to ensure they sum to 1.0

## Conclusion

The multi-portfolio support feature significantly enhances the capabilities of the Friday trading system, allowing for more sophisticated portfolio management strategies and better organization of investment activities. By leveraging this feature, users can manage multiple investment strategies simultaneously, analyze relationships between portfolios, and generate comprehensive reports across their entire investment universe.