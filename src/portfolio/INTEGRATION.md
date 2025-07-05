# Portfolio Management System Integration

## Overview

The Portfolio Management System is designed to integrate seamlessly with other components of the Friday AI Trading System. This document outlines the integration capabilities, configuration options, and usage examples.

## Integration Components

The Portfolio Management System can integrate with the following components:

- **Event System**: For publishing and subscribing to portfolio events
- **Trading Engine**: For executing trades and receiving trade events
- **Market Data Service**: For receiving market data updates
- **Risk Management System**: For risk assessment and constraints

## Integration Architecture

The integration is facilitated through the `PortfolioIntegration` class, which serves as a bridge between the Portfolio Management System and other components. This class:

1. Creates and configures all portfolio components using `PortfolioFactory`
2. Sets up event subscriptions with the Event System
3. Registers portfolio and risk managers with the Trading Engine
4. Configures callbacks for market data updates
5. Handles events and routes them to the appropriate portfolio components

## Configuration

The integration can be configured through a dictionary or a configuration file. Here's an example configuration:

```python
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
```

## Event Types

The Portfolio Integration publishes and subscribes to the following event types:

### Subscribed Events

- `market_data_update`: Updates portfolio prices based on market data
- `trade_executed`: Processes executed trades in the portfolio
- `portfolio_update_request`: Generates and publishes portfolio state information
- `portfolio_rebalance_request`: Checks if rebalancing is needed and generates a plan

### Published Events

- `portfolio_value_update`: Published when portfolio values are updated
- `portfolio_updated`: Published when portfolio positions change
- `portfolio_state`: Published in response to update requests
- `portfolio_rebalance_plan`: Published when a rebalance plan is generated
- `portfolio_rebalance_not_needed`: Published when rebalancing is not needed

## Usage Examples

### Basic Integration Setup

```python
from portfolio.portfolio_integration import create_portfolio_integration
from infrastructure.event.event_system import EventSystem
from orchestration.trading_engine.integration import TradingEngineIntegrator
from data.market_data_service import MarketDataService

# Create components
event_system = EventSystem()
event_system.start()

trading_engine = TradingEngineIntegrator()
trading_engine.start()

market_data_service = MarketDataService()
market_data_service.start()

# Create portfolio integration
integration = create_portfolio_integration(
    config=config,
    event_system=event_system,
    trading_engine=trading_engine,
    market_data_service=market_data_service,
    auto_start=True
)

# Access portfolio components
portfolio_manager = integration.portfolio_manager
performance_calculator = integration.performance_calculator
tax_manager = integration.tax_manager
allocation_manager = integration.allocation_manager

# Get portfolio summary
summary = integration.get_portfolio_summary()
print(f"Portfolio Value: ${summary['value']:.2f}")

# Stop integration when done
integration.stop()
```

### Handling Portfolio Events

You can subscribe to portfolio events through the event system:

```python
def handle_portfolio_update(data):
    print(f"Portfolio updated: {data['portfolio_id']}")
    print(f"New value: ${data['value']:.2f}")
    print(f"Positions: {len(data['positions'])} symbols")

event_system.subscribe(
    event_type="portfolio_updated",
    callback=handle_portfolio_update
)
```

### Triggering Portfolio Actions

You can trigger portfolio actions by publishing events:

```python
# Request portfolio update
event_system.publish(
    event_type="portfolio_update_request",
    data={}
)

# Request portfolio rebalance
event_system.publish(
    event_type="portfolio_rebalance_request",
    data={}
)

# Simulate trade execution
event_system.publish(
    event_type="trade_executed",
    data={
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "timestamp": datetime.now(),
        "trade_id": "trade-123",
        "commission": 5.0
    }
)
```

## Integration with Risk Management

The Portfolio Integration automatically integrates with the Risk Management System if available:

```python
# Create portfolio integration with risk management
from risk.risk_management_factory import RiskManagementFactory

# Create risk factory
risk_factory = RiskManagementFactory()

# Portfolio factory will automatically detect and use risk management
integration = create_portfolio_integration(
    config=config,
    event_system=event_system,
    trading_engine=trading_engine,
    market_data_service=market_data_service,
    auto_start=True
)

# Access risk metrics
if integration.risk_manager:
    risk_metrics = integration.portfolio_manager.get_risk_metrics()
    print(f"Value at Risk (95%): ${risk_metrics['var_95']:.2f}")
    print(f"Expected Shortfall: ${risk_metrics['expected_shortfall']:.2f}")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
```

## Testing

The Portfolio Integration includes comprehensive integration tests that validate the integration with other components. These tests use mock implementations of the Event System, Trading Engine, and Market Data Service to simulate the behavior of these components.

To run the integration tests:

```bash
python -m unittest portfolio.test_portfolio_integration_system
```

## Error Handling

The Portfolio Integration includes robust error handling to ensure that failures in one component do not affect the entire system. It uses try-except blocks to catch and log errors, and it gracefully degrades functionality when components are not available.

## Conclusion

The Portfolio Management System's integration capabilities enable it to work seamlessly with other components of the Friday AI Trading System. By using the `PortfolioIntegration` class, you can easily connect the portfolio management functionality with the event system, trading engine, market data service, and risk management system.