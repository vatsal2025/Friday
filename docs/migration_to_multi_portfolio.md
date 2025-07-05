# Migration Guide: Single Portfolio to Multi-Portfolio System

This document provides a comprehensive guide for migrating from the single-portfolio architecture to the new multi-portfolio system. It outlines the steps required to transition existing code and configurations, ensuring a smooth upgrade process with minimal disruption.

## Table of Contents

1. [Overview](#overview)
2. [Backward Compatibility](#backward-compatibility)
3. [Migration Steps](#migration-steps)
4. [Configuration Changes](#configuration-changes)
5. [API Changes](#api-changes)
6. [Event System Changes](#event-system-changes)
7. [Testing Your Migration](#testing-your-migration)
8. [Troubleshooting](#troubleshooting)

## Overview

The multi-portfolio system extends the existing portfolio management capabilities to support multiple portfolios simultaneously, with portfolio isolation, portfolio-specific event routing, portfolio grouping, and cross-portfolio analysis. This migration guide will help you transition your existing code to take advantage of these new features.

## Backward Compatibility

The multi-portfolio system has been designed with backward compatibility in mind. The existing `PortfolioIntegration` class continues to work as before, and the new `MultiPortfolioIntegration` class extends its functionality. This means you can migrate at your own pace, starting with the core components and gradually adopting the new features.

## Migration Steps

### Step 1: Update Dependencies

Ensure you have the latest version of the system with multi-portfolio support installed.

### Step 2: Replace PortfolioIntegration with MultiPortfolioIntegration

Update your imports and instantiation code:

```python
# Old code
from src.portfolio.portfolio_integration import PortfolioIntegration

portfolio_integration = PortfolioIntegration(
    event_system=event_system,
    trading_engine=trading_engine,
    market_data_service=market_data_service,
    config=config
)

# New code
from src.portfolio.multi_portfolio_integration import MultiPortfolioIntegration

multi_portfolio_integration = MultiPortfolioIntegration(
    event_system=event_system,
    trading_engine=trading_engine,
    market_data_service=market_data_service,
    default_config=config  # Note: renamed from 'config' to 'default_config'
)

# Create your default portfolio (equivalent to the single portfolio in the old system)
default_portfolio_id = multi_portfolio_integration.create_portfolio(
    name="Default Portfolio",
    initial_capital=config.get("initial_capital", 100000.0),
    description="Default portfolio migrated from single-portfolio system"
)

# Set it as the active portfolio
multi_portfolio_integration.set_active_portfolio(default_portfolio_id)
```

### Step 3: Update Event Handlers

If you have custom event handlers that interact with the portfolio system, update them to work with the multi-portfolio system:

```python
# Old code
def handle_market_data_update(event):
    portfolio_integration.handle_market_data_update(event)

# New code
def handle_market_data_update(event):
    # The event now needs a portfolio_id field to route to the correct portfolio
    if 'portfolio_id' not in event.data:
        # For backward compatibility, use the active portfolio if no portfolio_id is specified
        event.data['portfolio_id'] = multi_portfolio_integration.get_active_portfolio_id()
    
    multi_portfolio_integration.handle_market_data_update(event)
```

### Step 4: Update API Calls

Update any direct API calls to the portfolio system:

```python
# Old code
portfolio_value = portfolio_integration.get_portfolio_manager().get_portfolio_value()

# New code
portfolio_id = multi_portfolio_integration.get_active_portfolio_id()
portfolio_value = multi_portfolio_integration.get_portfolio_value(portfolio_id)
```

## Configuration Changes

### Portfolio Configuration

The configuration structure has been updated to support multiple portfolios. Each portfolio can have its own configuration, which overrides the default configuration:

```python
# Old configuration
config = {
    "risk_management": {
        "max_position_size": 0.05,
        "max_sector_exposure": 0.25
    },
    "tax_settings": {
        "tax_rate_short_term": 0.35,
        "tax_rate_long_term": 0.15
    }
}

# New configuration
default_config = {
    "risk_management": {
        "max_position_size": 0.05,
        "max_sector_exposure": 0.25
    },
    "tax_settings": {
        "tax_rate_short_term": 0.35,
        "tax_rate_long_term": 0.15
    }
}

# Portfolio-specific configuration (overrides default settings)
portfolio_config = {
    "risk_management": {
        "max_position_size": 0.08  # Override just this setting
    }
}

# Create portfolio with specific configuration
portfolio_id = multi_portfolio_integration.create_portfolio(
    name="Growth Portfolio",
    initial_capital=100000.0,
    config=portfolio_config
)
```

### Group Configuration

Portfolio groups are a new concept in the multi-portfolio system. They allow you to organize portfolios and perform operations on them as a group:

```python
# Create a portfolio group
group_id = multi_portfolio_integration.create_portfolio_group(
    name="Equity Strategies",
    portfolio_ids=[portfolio_id1, portfolio_id2],
    description="Portfolios focused on equity investments",
    allocation={
        portfolio_id1: 0.6,  # 60% allocation
        portfolio_id2: 0.4   # 40% allocation
    }
)
```

## API Changes

### New APIs

The multi-portfolio system introduces several new APIs:

- Portfolio Management:
  - `create_portfolio(name, initial_capital, description=None, tags=None, config=None)`
  - `delete_portfolio(portfolio_id)`
  - `activate_portfolio(portfolio_id)`
  - `deactivate_portfolio(portfolio_id)`
  - `set_active_portfolio(portfolio_id)`
  - `get_active_portfolio_id()`
  - `get_all_portfolio_ids()`
  - `get_active_portfolio_ids()`

- Group Management:
  - `create_portfolio_group(name, portfolio_ids, description=None, allocation=None)`
  - `delete_portfolio_group(group_id)`
  - `add_to_group(group_id, portfolio_id, allocation=None)`
  - `remove_from_group(group_id, portfolio_id)`
  - `get_all_group_ids()`
  - `get_portfolio_group(group_id)`

- Cross-Portfolio Analysis:
  - `compare_portfolios(portfolio_ids, metrics)`
  - `calculate_correlation(portfolio_ids)`
  - `analyze_diversification(portfolio_ids)`
  - `generate_consolidated_report()`

### Modified APIs

Existing APIs have been modified to include a portfolio_id parameter:

```python
# Old API
portfolio_integration.get_portfolio_manager().get_portfolio_value()

# New API
multi_portfolio_integration.get_portfolio_value(portfolio_id)
```

## Event System Changes

The event system has been extended to support portfolio-specific events. Events now include a `portfolio_id` field to route them to the correct portfolio:

```python
# Old event
event = {
    "type": "MARKET_DATA_UPDATE",
    "data": {
        "symbol": "AAPL",
        "price": 150.0
    }
}

# New event
event = {
    "type": "MARKET_DATA_UPDATE",
    "data": {
        "portfolio_id": "portfolio-123",  # New field
        "symbol": "AAPL",
        "price": 150.0
    }
}
```

For backward compatibility, if no `portfolio_id` is specified, the event is routed to the active portfolio.

## Testing Your Migration

After migrating to the multi-portfolio system, it's important to test your implementation thoroughly:

1. **Unit Tests**: Update your unit tests to work with the new APIs and ensure they pass.

2. **Integration Tests**: Test the integration between the multi-portfolio system and other components (event system, trading engine, market data service).

3. **Backward Compatibility Tests**: Ensure that existing code that expects the single-portfolio behavior continues to work with the multi-portfolio system.

4. **Performance Tests**: Verify that the performance of the system remains acceptable with multiple portfolios.

## Troubleshooting

### Common Issues

1. **Event Routing Issues**: If events are not being routed to the correct portfolio, check that the `portfolio_id` field is correctly set in the event data.

2. **Configuration Issues**: If portfolio-specific configurations are not being applied, check that the configuration is correctly structured and passed to the `create_portfolio` method.

3. **Performance Issues**: If you experience performance degradation with multiple portfolios, consider deactivating portfolios that are not currently in use.

### Getting Help

If you encounter issues during migration, please refer to the following resources:

- Documentation: Check the updated documentation for detailed information on the multi-portfolio system.
- Examples: Review the example code in the `examples` directory for guidance on using the new features.
- Support: Contact the development team for assistance with complex migration issues.

---

By following this migration guide, you should be able to transition smoothly from the single-portfolio architecture to the new multi-portfolio system, taking advantage of the enhanced capabilities while maintaining compatibility with existing code.