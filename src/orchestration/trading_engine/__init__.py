"""Trading Engine for the Friday AI Trading System.

This package provides the trading engine components for the Friday AI Trading System,
including signal generation, order management, execution strategies, and trade lifecycle management.
"""

from src.orchestration.trading_engine.engine import (
    TradingEngine,
    SignalGenerator,
    OrderManager
)

from src.orchestration.trading_engine.execution import (
    ExecutionStrategy,
    ImmediateExecution,
    TWAPExecution,
    ExecutionFactory,
    MarketImpactEstimator
)

from src.orchestration.trading_engine.lifecycle import (
    TradeState,
    TradeLifecycleManager,
    TradeReporter
)

from src.orchestration.trading_engine.integration import (
    TradingEngineIntegrator,
    ModelTradingBridgeIntegration,
    create_trading_engine,
    create_model_trading_bridge
)

from src.orchestration.trading_engine.config import (
    TradingEngineConfig,
    OrderConfig,
    SignalConfig,
    TradingHoursConfig,
    RiskLimitsConfig,
    get_default_config,
    load_config
)

from src.orchestration.trading_engine.utils import (
    generate_trade_id,
    generate_order_id,
    generate_signal_id,
    generate_execution_id,
    calculate_order_value,
    calculate_slippage,
    is_market_open,
    get_market_session,
    calculate_time_to_market_open,
    calculate_time_to_market_close,
    format_price,
    format_quantity,
    calculate_profit_loss,
    calculate_return_percentage
)

__all__ = [
    # Engine components
    'TradingEngine',
    'SignalGenerator',
    'OrderManager',
    
    # Execution strategies
    'ExecutionStrategy',
    'ImmediateExecution',
    'TWAPExecution',
    'ExecutionFactory',
    'MarketImpactEstimator',
    
    # Trade lifecycle
    'TradeState',
    'TradeLifecycleManager',
    'TradeReporter',
    
    # Integration
    'TradingEngineIntegrator',
    'ModelTradingBridgeIntegration',
    'create_trading_engine',
    'create_model_trading_bridge',
    
    # Configuration
    'TradingEngineConfig',
    'OrderConfig',
    'SignalConfig',
    'TradingHoursConfig',
    'RiskLimitsConfig',
    'get_default_config',
    'load_config',
    
    # Utilities
    'generate_trade_id',
    'generate_order_id',
    'generate_signal_id',
    'generate_execution_id',
    'calculate_order_value',
    'calculate_slippage',
    'is_market_open',
    'get_market_session',
    'calculate_time_to_market_open',
    'calculate_time_to_market_close',
    'format_price',
    'format_quantity',
    'calculate_profit_loss',
    'calculate_return_percentage'
]