from datetime import datetime
import pandas as pd
import numpy as np

# Import risk management components
from .position_sizer import PositionSizer
from .stop_loss_manager import StopLossManager, StopLossType
from .portfolio_risk_manager import PortfolioRiskManager
from .advanced_risk_manager import AdvancedRiskManager

# Import other system components (these imports would be adjusted based on actual system structure)
from ..infrastructure.event.integration import EventSystem
from ..orchestration.trading_engine.engine import TradingEngine
from ..infrastructure.monitoring import performance_monitoring


def setup_risk_management_system(config):
    """
    Set up and configure the risk management system based on the provided configuration.

    Args:
        config: Configuration object containing risk parameters

    Returns:
        AdvancedRiskManager: Configured risk management system
    """
    # Extract risk parameters from config
    initial_capital = config.get('initial_capital', 100000.0)
    risk_per_trade = config.get('risk_per_trade', 0.01)  # Default 1% risk per trade
    max_portfolio_var_percent = config.get('max_portfolio_var_percent', 0.02)  # Default 2% VaR
    max_drawdown_percent = config.get('max_drawdown_percent', 0.15)  # Default 15% max drawdown
    max_sector_exposure = config.get('max_sector_exposure', 0.25)  # Default 25% sector exposure
    max_asset_exposure = config.get('max_asset_exposure', 0.10)  # Default 10% asset exposure
    default_stop_loss_percent = config.get('default_stop_loss_percent', 0.02)  # Default 2% stop loss

    # Create the advanced risk manager
    risk_manager = AdvancedRiskManager(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_portfolio_var_percent=max_portfolio_var_percent,
        max_drawdown_percent=max_drawdown_percent,
        max_sector_exposure=max_sector_exposure,
        max_asset_exposure=max_asset_exposure,
        default_stop_loss_percent=default_stop_loss_percent
    )

    return risk_manager


def integrate_with_trading_engine(trading_engine, risk_manager, event_system):
    """
    Integrate the risk management system with the trading engine and event system.

    Args:
        trading_engine: The trading engine instance
        risk_manager: The risk management system instance
        event_system: The event system instance
    """
    # Register risk manager with trading engine
    trading_engine.register_risk_manager(risk_manager)

    # Subscribe to relevant events
    event_system.subscribe('market_data_update', risk_manager.update_market_data)
    event_system.subscribe('portfolio_update', risk_manager.update_portfolio)
    event_system.subscribe('trade_executed', handle_trade_executed)
    event_system.subscribe('trade_exit', handle_trade_exit)

    # Register circuit breaker handlers
    event_system.subscribe('circuit_breaker_triggered', handle_circuit_breaker)

    # Register risk alert handlers
    event_system.subscribe('risk_alert', handle_risk_alert)


def handle_trade_executed(trade_data, risk_manager):
    """
    Handle trade executed event by setting appropriate stop losses.

    Args:
        trade_data: Data about the executed trade
        risk_manager: The risk management system instance
    """
    trade_id = trade_data['trade_id']
    symbol = trade_data['symbol']
    entry_price = trade_data['price']
    entry_time = trade_data['timestamp']
    direction = trade_data['direction']
    quantity = trade_data['quantity']

    # Set stop loss based on strategy or default settings
    stop_type = trade_data.get('stop_type', StopLossType.FIXED)
    stop_params = trade_data.get('stop_params', {'stop_percent': risk_manager.default_stop_loss_percent})

    # Set the stop loss
    stop_details = risk_manager.set_stop_loss(
        trade_id=trade_id,
        symbol=symbol,
        entry_price=entry_price,
        entry_time=entry_time,
        direction=direction,
        stop_type=stop_type,
        stop_params=stop_params,
        quantity=quantity
    )

    # Log the stop loss details
    print(f"Stop loss set for trade {trade_id}: {stop_details}")

    # Update trade in risk manager's tracking
    risk_manager.trades[trade_id].update({
        'strategy': trade_data.get('strategy', 'unknown'),
        'tags': trade_data.get('tags', []),
        'metadata': trade_data.get('metadata', {})
    })


def handle_trade_exit(trade_data, risk_manager):
    """
    Handle trade exit event by removing stop loss tracking.

    Args:
        trade_data: Data about the exited trade
        risk_manager: The risk management system instance
    """
    trade_id = trade_data['trade_id']
    exit_price = trade_data['price']
    exit_time = trade_data['timestamp']
    exit_reason = trade_data['reason']

    # Remove stop loss tracking
    risk_manager.remove_stop_loss(trade_id)

    # Update trade status in risk manager's tracking
    if trade_id in risk_manager.trades:
        risk_manager.trades[trade_id].update({
            'status': 'closed',
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason
        })

        # Calculate and log P&L
        entry_price = risk_manager.trades[trade_id]['entry_price']
        direction = risk_manager.trades[trade_id]['direction']
        quantity = risk_manager.trades[trade_id]['quantity']

        if direction == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity

        risk_manager.trades[trade_id]['pnl'] = pnl
        print(f"Trade {trade_id} exited: {exit_reason}, P&L: ${pnl:.2f}")


def handle_circuit_breaker(event_data, trading_engine):
    """
    Handle circuit breaker triggered event.

    Args:
        event_data: Data about the circuit breaker event
        trading_engine: The trading engine instance
    """
    breaker_type = event_data['type']
    reason = event_data['reason']
    severity = event_data['severity']

    print(f"Circuit breaker triggered: {breaker_type}, Reason: {reason}, Severity: {severity}")

    # Take action based on severity
    if severity == 'critical':
        # Halt all trading
        trading_engine.halt_trading(reason=f"Circuit breaker: {reason}")
    elif severity == 'high':
        # Reduce position sizes
        trading_engine.set_risk_multiplier(0.5)  # Reduce risk by 50%
    elif severity == 'medium':
        # Increase stop loss tightness
        trading_engine.set_stop_loss_multiplier(1.5)  # Tighten stops by 50%
    elif severity == 'low':
        # Just log the event
        pass


def handle_risk_alert(alert_data, trading_engine, risk_manager):
    """
    Handle risk alert event.

    Args:
        alert_data: Data about the risk alert
        trading_engine: The trading engine instance
        risk_manager: The risk management system instance
    """
    alert_type = alert_data['type']
    message = alert_data['message']
    metrics = alert_data.get('metrics', {})

    print(f"Risk alert: {alert_type}, Message: {message}")

    # Take action based on alert type
    if alert_type == 'portfolio_var_breach':
        # Reduce position sizes for new trades
        trading_engine.set_risk_multiplier(0.7)  # Reduce risk by 30%
    elif alert_type == 'drawdown_breach':
        # Reduce position sizes more aggressively
        trading_engine.set_risk_multiplier(0.5)  # Reduce risk by 50%
    elif alert_type == 'sector_exposure_breach':
        # Block new trades in the affected sector
        sector = metrics.get('sector')
        if sector:
            trading_engine.block_sector(sector)
    elif alert_type == 'asset_exposure_breach':
        # Block new trades in the affected asset
        symbol = metrics.get('symbol')
        if symbol:
            trading_engine.block_symbol(symbol)


def example_usage():
    """
    Example of how to use the risk management system in a trading application.
    """
    # Create configuration
    config = {
        'initial_capital': 1000000.0,  # $1M starting capital
        'risk_per_trade': 0.005,  # 0.5% risk per trade
        'max_portfolio_var_percent': 0.015,  # 1.5% VaR
        'max_drawdown_percent': 0.10,  # 10% max drawdown
        'max_sector_exposure': 0.20,  # 20% sector exposure
        'max_asset_exposure': 0.05,  # 5% asset exposure
        'default_stop_loss_percent': 0.015  # 1.5% default stop loss
    }

    # Set up risk management system
    risk_manager = setup_risk_management_system(config)

    # Create mock trading engine and event system
    trading_engine = TradingEngine()
    event_system = EventSystem()

    # Integrate systems
    integrate_with_trading_engine(trading_engine, risk_manager, event_system)

    # Example: Calculate position size for a trade
    symbol = "AAPL"
    entry_price = 175.50
    stop_price = 172.00

    size, details = risk_manager.calculate_position_size(
        symbol=symbol,
        entry_price=entry_price,
        stop_price=stop_price,
        method="risk_based"
    )

    print(f"Calculated position size for {symbol}: {size} shares")
    print(f"Position details: {details}")

    # Example: Simulate a trade execution
    trade_data = {
        'trade_id': 'T12345',
        'symbol': symbol,
        'price': entry_price,
        'timestamp': datetime.now(),
        'direction': 'long',
        'quantity': size,
        'strategy': 'momentum_breakout',
        'stop_type': StopLossType.TRAILING,
        'stop_params': {'trailing_percent': 0.02}  # 2% trailing stop
    }

    # Simulate trade execution event
    handle_trade_executed(trade_data, risk_manager)

    # Example: Update market data
    market_data = {
        symbol: {
            'price': 178.25,  # Price moved up
            'timestamp': datetime.now(),
            'volume': 1000000,
            'volatility': 0.018  # 1.8% volatility
        }
    }

    # Update market data and check stop loss
    risk_manager.update_market_data(market_data)
    is_triggered, _ = risk_manager.update_stop_loss('T12345', 178.25)

    print(f"Stop loss triggered: {is_triggered}")
    print(f"Updated stop price: {risk_manager.get_stop_loss('T12345')['stop_price']}")

    # Example: Update portfolio
    positions = {
        "AAPL": {"market_value": 178.25 * size, "sector": "Technology", "quantity": size, "price": 178.25},
        "MSFT": {"market_value": 350000.0, "sector": "Technology", "quantity": 1000.0, "price": 350.0},
        "JPM": {"market_value": 200000.0, "sector": "Financials", "quantity": 1250.0, "price": 160.0}
    }

    portfolio_value = 1050000.0  # Portfolio value increased

    # Update portfolio and get risk metrics
    summary = risk_manager.update_portfolio(positions, portfolio_value)

    print("Portfolio risk metrics:")
    for metric, value in summary['risk_metrics'].items():
        print(f"  {metric}: {value}")

    # Example: Simulate trade exit
    exit_data = {
        'trade_id': 'T12345',
        'price': 180.0,  # Exit price
        'timestamp': datetime.now(),
        'reason': 'target_reached'
    }

    # Simulate trade exit event
    handle_trade_exit(exit_data, risk_manager)


if __name__ == "__main__":
    example_usage()
