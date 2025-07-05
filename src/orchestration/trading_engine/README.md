# Trading Engine

The Trading Engine is a core component of the Friday AI Trading System, responsible for processing model predictions, generating trading signals, managing orders, executing trades, and tracking trade lifecycle.

## Overview

The Trading Engine bridges the gap between model predictions and actual trades. It takes model predictions as input, applies trading rules and risk management, generates trading signals, creates and manages orders, and tracks the lifecycle of trades.

## Components

### Core Components

- **TradingEngine**: The main engine that coordinates signal generation, order management, and trade execution.
- **SignalGenerator**: Generates trading signals from model predictions.
- **OrderManager**: Creates and manages orders based on trading signals.

### Execution Strategies

- **ExecutionStrategy**: Base class for execution strategies.
- **ImmediateExecution**: Executes orders immediately (market or limit).
- **TWAPExecution**: Time-Weighted Average Price execution strategy.
- **ExecutionFactory**: Factory for creating execution strategies.
- **MarketImpactEstimator**: Estimates market impact of orders and recommends execution strategies.

### Trade Lifecycle Management

- **TradeLifecycleManager**: Tracks trades through different states.
- **TradeReporter**: Generates trade summaries and reports.

### Integration

- **TradingEngineIntegrator**: Integrates the trading engine with other components.
- **ModelTradingBridgeIntegration**: Integrates the Model Trading Bridge with the Trading Engine.

### Configuration

- **TradingEngineConfig**: Configuration for the trading engine.
- **OrderConfig**: Configuration for order execution.
- **SignalConfig**: Configuration for signal generation.
- **TradingHoursConfig**: Configuration for trading hours.
- **RiskLimitsConfig**: Configuration for risk limits.

## Usage

### Basic Usage

```python
from src.infrastructure.event import EventSystem
from src.orchestration.trading_engine import (
    create_trading_engine,
    create_model_trading_bridge
)

# Create event system
event_system = EventSystem()

# Create trading engine
trading_engine = create_trading_engine(event_system)

# Create model trading bridge
bridge = create_model_trading_bridge(trading_engine, event_system)

# Process a model prediction
prediction = {
    "symbol": "AAPL",
    "prediction_type": "price_movement",
    "value": 0.75,  # Positive value indicates upward movement
    "horizon": "1d",
    "timestamp": "2023-01-01T12:00:00",
    "model_id": "example_model",
    "confidence": 0.85
}

signal = bridge.process_prediction(prediction, prediction["confidence"])
```

### Executing Orders

```python
# Execute an order
order_params = {
    "symbol": "AAPL",
    "quantity": 10,
    "price": 150.0,
    "side": "buy",
    "order_type": "limit",
    "time_in_force": "day"
}

result = trading_engine.execute_order(order_params, "immediate")
```

### Continuous Trading

```python
# Start continuous trading
trading_engine.start_continuous_trading(interval_seconds=60.0)

# Stop continuous trading
trading_engine.stop_continuous_trading()
```

### Trade Lifecycle Management

```python
# Get a trade
trade = trading_engine.get_trade(trade_id)

# Generate a trade summary
summary = trading_engine.generate_trade_summary(trade_id)

# Generate a daily report
report = trading_engine.generate_daily_report()
```

### Event Handling

```python
# Register an event handler
def handle_order_filled(event):
    print(f"Order filled: {event.data}")

event_system.register_handler(
    callback=handle_order_filled,
    event_types=["order_filled"]
)
```

## Configuration

```python
from src.orchestration.trading_engine import (
    TradingEngineConfig,
    get_default_config,
    load_config
)

# Get default configuration
config = get_default_config()

# Customize configuration
config.signal_config.min_signal_strength = 0.3
config.order_config.default_execution_strategy = "twap"

# Save configuration
config.save_to_file("trading_engine_config.json")

# Load configuration
config = load_config("trading_engine_config.json")
```

## Examples

See the `examples.py` file for more examples of how to use the trading engine.

## Integration with Other Components

The Trading Engine integrates with the following components:

- **Event System**: For publishing and subscribing to events.
- **Broker Service**: For executing trades with brokers.
- **Risk Manager**: For controlling risk.
- **Model Trading Bridge**: For processing model predictions.

## Development

### Adding a New Execution Strategy

1. Create a new class that inherits from `ExecutionStrategy`.
2. Implement the `execute` method.
3. Register the strategy with the `ExecutionFactory`.

### Adding a New Trading Rule

```python
def my_trading_rule(signal):
    # Return True if the signal passes the rule, False otherwise
    return signal["strength"] > 0.5

trading_engine.add_trading_rule(my_trading_rule, "my_rule")
```

## Future Enhancements

- Support for more execution strategies (VWAP, TWAP, etc.)
- Integration with more brokers
- Advanced risk management
- Performance optimization
- Support for more asset classes