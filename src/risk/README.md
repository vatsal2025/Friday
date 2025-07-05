# Risk Management Module

This module provides a comprehensive risk management system for the Friday AI Trading System. It includes components for position sizing, stop-loss management, portfolio risk controls, circuit breakers, and risk metrics calculation. The system is designed to be production-ready with features for persistence, monitoring, and alerting, and uses a factory pattern for easy configuration and instantiation.

## Components

### Position Sizer

The `PositionSizer` class calculates appropriate position sizes based on various methods:

- **Risk-based sizing**: Calculates position size based on account risk per trade and stop loss distance
- **Fixed percentage sizing**: Calculates position size based on a fixed percentage of account capital
- **Volatility-based sizing**: Adjusts position size based on market volatility

### Stop Loss Manager

The `StopLossManager` class manages various types of stop losses for trades:

- **Fixed stops**: Simple percentage-based stop losses
- **Trailing stops**: Stops that move with favorable price movement
- **Volatility-based stops**: Stops based on market volatility (ATR)
- **Time-based stops**: Exits trades after a specified time period
- **Profit targets**: Sets take-profit levels based on risk-reward ratios

### Portfolio Risk Manager

The `PortfolioRiskManager` class manages portfolio-level risk controls:

- **Value at Risk (VaR)**: Calculates and monitors portfolio VaR
- **Drawdown monitoring**: Tracks portfolio drawdown and enforces limits
- **Sector/asset exposure**: Monitors and limits exposure to specific sectors or assets
- **Correlation risk**: Monitors portfolio correlation and concentration
- **Risk-adjusted metrics**: Calculates Sharpe ratio, Sortino ratio, etc.

### Advanced Risk Manager

The `AdvancedRiskManager` class integrates all risk management components into a unified system:

- Combines position sizing, stop-loss management, and portfolio risk controls
- Provides a simple interface for the trading system
- Manages circuit breakers for market, account, and system-level risk events
- Tracks trades and maintains risk metrics
- Integrates with the RiskMetricsCalculator for comprehensive risk analysis

### Risk Metrics Calculator

The `RiskMetricsCalculator` class provides comprehensive risk metrics calculation:

- Value at Risk (VaR) calculation with configurable confidence levels
- Historical and parametric VaR methods
- Performance metrics including Sharpe ratio, Sortino ratio, and Calmar ratio
- Volatility and drawdown calculations
- Time-series analysis of portfolio performance

### Factory Pattern Implementation

The module uses a factory pattern for creating and configuring risk management components:

- `RiskManagementFactory`: Creates pre-configured risk management components
- `RiskManagementProductionConfig`: Provides configuration parameters for production environments
- Simplifies the creation and integration of risk management components
- Ensures consistent configuration across the system

## Usage

### Basic Setup with Factory Pattern

```python
# Create a configuration
from risk.production_config import create_default_production_config
config = create_default_production_config()

# Create a factory with the configuration
from risk.risk_management_factory import RiskManagementFactory
factory = RiskManagementFactory(config)

# Create risk management components
risk_metrics_calculator = factory.create_risk_metrics_calculator()
advanced_risk_manager = factory.create_advanced_risk_manager()

# Update the portfolio
positions = {"AAPL": {"market_value": 10000, "sector": "Technology"}}  # Example positions
portfolio_value = 100000
timestamp = datetime.now()
advanced_risk_manager.portfolio_risk_manager.update_portfolio(positions, portfolio_value, timestamp)

# Get risk metrics
metrics = advanced_risk_manager.get_risk_metrics()
print(f"Portfolio VaR: {metrics['var']:.2%}")
print(f"Current Drawdown: {metrics['current_drawdown']:.2%}")
```

### Manual Setup (Alternative)

```python
from risk.advanced_risk_manager import AdvancedRiskManager
from risk.portfolio_risk_manager import PortfolioRiskManager
from risk.position_sizer import PositionSizer
from risk.stop_loss_manager import StopLossManager
from risk.risk_metrics_calculator import RiskMetricsCalculator

# Create components manually
risk_metrics_calculator = RiskMetricsCalculator(confidence_level=0.95)
portfolio_risk_manager = PortfolioRiskManager(
    max_portfolio_var_percent=0.02,
    max_drawdown_percent=0.15,
    risk_metrics_calculator=risk_metrics_calculator
)

# Create the advanced risk manager with the components
risk_manager = AdvancedRiskManager(
    initial_capital=100000,
    risk_per_trade=0.01,
    portfolio_risk_manager=portfolio_risk_manager,
    risk_metrics_calculator=risk_metrics_calculator
)
```

### Demo Script

A demo script is provided to demonstrate the usage of the risk management system:

```bash
python -m risk.demo_risk_management
```


)
```

### Position Sizing

```python
# Calculate position size based on risk
symbol = "AAPL"
entry_price = 150.0
stop_price = 147.0  # 2% stop loss

size, details = risk_manager.calculate_position_size(
    symbol=symbol,
    entry_price=entry_price,
    stop_price=stop_price,
    method="risk_based"  # Options: "risk_based", "fixed_percent", "volatility_based"
)

print(f"Position size: {size} shares")
print(f"Details: {details}")
```

### Stop Loss Management

```python
from risk.stop_loss_manager import StopLossType

# Set a stop loss for a trade
trade_id = "T12345"
symbol = "AAPL"
entry_price = 150.0
direction = "long"

stop_details = risk_manager.set_stop_loss(
    trade_id=trade_id,
    symbol=symbol,
    entry_price=entry_price,
    entry_time=datetime.now(),
    direction=direction,
    stop_type=StopLossType.TRAILING,
    stop_params={"trailing_percent": 0.02}  # 2% trailing stop
)

print(f"Stop loss set: {stop_details}")

# Update stop loss with current price
is_triggered, updated_stop = risk_manager.update_stop_loss(
    trade_id=trade_id,
    current_price=155.0  # Price moved up
)

print(f"Stop loss triggered: {is_triggered}")
print(f"Updated stop price: {updated_stop['stop_price']}")
```

### Portfolio Risk Management

```python
# Update portfolio with current positions
positions = {
    "AAPL": {"market_value": 15000.0, "sector": "Technology", "quantity": 100, "price": 150.0},
    "MSFT": {"market_value": 20000.0, "sector": "Technology", "quantity": 80, "price": 250.0},
    "JPM": {"market_value": 10000.0, "sector": "Financials", "quantity": 100, "price": 100.0}
}

portfolio_value = 100000.0

summary = risk_manager.update_portfolio(
    positions=positions,
    portfolio_value=portfolio_value
)

print("Portfolio risk metrics:")
for metric, value in summary['risk_metrics'].items():
    print(f"  {metric}: {value}")

# Check for risk alerts
alerts = risk_manager.check_risk_alerts()
for alert in alerts:
    print(f"Risk alert: {alert['type']} - {alert['message']}")
```

### Circuit Breaker Integration

```python
# Update market data for circuit breaker monitoring
market_data = {
    "AAPL": {"price": 145.0, "volume": 1000000, "volatility": 0.02},
    "MSFT": {"price": 240.0, "volume": 800000, "volatility": 0.018},
    "SPY": {"price": 400.0, "volume": 5000000, "volatility": 0.015}
}

risk_manager.update_market_data(market_data)

# Check for circuit breaker triggers
circuit_breakers = risk_manager.check_circuit_breakers()
for breaker in circuit_breakers:
    print(f"Circuit breaker triggered: {breaker['type']} - {breaker['reason']}")
```

## Integration with Trading System

See `integration_example.py` for a complete example of how to integrate the risk management system with the trading engine and event system.

## Testing

Run the test suite to verify the functionality of the risk management system:

```bash
python -m unittest risk.test_risk_management
```

## Production-Ready Components

The risk management system includes several production-ready components:

### Production Configuration

The `production_config.py` module provides a production configuration for the risk management system:

- Defines conservative risk parameters suitable for production use
- Supports loading configuration from files in standard locations
- Includes asset class-specific position limits
- Configures logging, monitoring, and persistence settings

### Risk Management Factory

The `risk_management_factory.py` module provides a factory for creating production-ready risk management components:

- Creates and configures all risk management components based on production configuration
- Ensures consistent setup across the application
- Provides methods for creating individual components or the complete system

### Risk Management Service

The `risk_management_service.py` module provides a production-ready service with additional features:

- Periodic risk metrics calculation and monitoring
- State persistence to disk
- Automatic recovery from persistence
- Alert notifications
- Health checks

### Production Example

See `production_example.py` for a complete example of how to use the production-ready risk management system in a live trading environment.
