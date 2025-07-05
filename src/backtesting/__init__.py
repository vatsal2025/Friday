"""Backtesting framework for strategy evaluation.

This package provides a comprehensive backtesting framework for evaluating trading strategies,
including an event-driven backtesting engine, performance analytics, transaction cost modeling,
and visual reporting capabilities.
"""

# Core backtesting engine
from src.backtesting.engine import BacktestEngine, Event, EventType

# Performance analytics
from src.backtesting.performance import (
    PerformanceAnalytics,
    PerformanceMetrics,
    BenchmarkComparison
)

# Transaction cost modeling
from src.backtesting.costs import (
    CostType,
    TransactionCostModel,
    FixedCostModel,
    PercentageCostModel,
    PerShareCostModel,
    SlippageCostModel as SlippageModel,  # Aliasing SlippageCostModel to SlippageModel for backward compatibility
    SpreadCostModel as SpreadModel,  # Aliasing SpreadCostModel to SpreadModel for backward compatibility
    MarketImpactCostModel as MarketImpactModel,  # Aliasing MarketImpactCostModel to MarketImpactModel for backward compatibility
    ExchangeFeeCostModel as ExchangeFeeModel,  # Aliasing ExchangeFeeCostModel to ExchangeFeeModel for backward compatibility
    TaxCostModel as TaxModel,  # Aliasing TaxCostModel to TaxModel for backward compatibility
    CompositeCostModel,
    ZeroCostModel,
    RealisticStockCostModel,
    RealisticCryptoCostModel,
    RealisticFuturesCostModel,
    TransactionCostCalculator,
    TransactionCostAnalyzer
)

# Reporting
from src.backtesting.reporting import (
    BacktestReport,
    ReportFormat,
    ChartType
)

# Strategy
from src.backtesting.strategy import (
    Strategy,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
    Portfolio,
    MovingAverageStrategy,
    RSIStrategy
)

# Data handling
from src.backtesting.data import (
    DataSource,
    DataHandler,
    CSVDataSource,
    SQLDataSource,
    APIDataSource,
    OHLCVDataHandler
)

# Integration
from src.backtesting.integration import (
    BacktestRunner,
    WalkForwardAnalyzer,
    MonteCarloSimulator
)

# Utilities
from src.backtesting.utils import (
    TimeFrame,
    TradeDirection,
    TradeStatus,
    Trade,
    TradeAnalyzer,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_var,
    calculate_cvar,
    calculate_beta,
    calculate_alpha,
    calculate_information_ratio,
    calculate_treynor_ratio,
    resample_ohlcv,
    calculate_drawdowns,
    calculate_rolling_metrics,
    calculate_trade_statistics,
    save_backtest_results,
    load_backtest_results,
    run_monte_carlo_simulation,
    plot_monte_carlo_simulation,
    run_walk_forward_analysis,
    plot_walk_forward_analysis
)

__all__ = [
    # Core backtesting engine
    'BacktestEngine', 'Event', 'EventType',
    
    # Performance analytics
    'PerformanceAnalytics', 'PerformanceMetrics', 'BenchmarkComparison',
    
    # Transaction cost modeling
    'CostType', 'TransactionCostModel', 'FixedCostModel', 'PercentageCostModel',
    'PerShareCostModel', 'SlippageModel', 'SpreadModel', 'MarketImpactModel',
    'ExchangeFeeModel', 'TaxModel', 'CompositeCostModel', 'ZeroCostModel',
    'RealisticStockCostModel', 'RealisticCryptoCostModel', 'RealisticFuturesCostModel',
    'TransactionCostCalculator', 'TransactionCostAnalyzer',
    
    # Reporting
    'BacktestReport', 'ReportFormat', 'ChartType',
    
    # Strategy
    'Strategy', 'Order', 'OrderType', 'OrderSide', 'OrderStatus', 'Position',
    'Portfolio', 'MovingAverageStrategy', 'RSIStrategy',
    
    # Data handling
    'DataSource', 'DataHandler', 'CSVDataSource', 'SQLDataSource', 'APIDataSource',
    'OHLCVDataHandler',
    
    # Integration
    'BacktestRunner', 'WalkForwardAnalyzer', 'MonteCarloSimulator',
    
    # Utilities
    'TimeFrame', 'TradeDirection', 'TradeStatus', 'Trade', 'TradeAnalyzer',
    'calculate_returns', 'calculate_sharpe_ratio', 'calculate_sortino_ratio',
    'calculate_max_drawdown', 'calculate_cagr', 'calculate_calmar_ratio',
    'calculate_omega_ratio', 'calculate_var', 'calculate_cvar', 'calculate_beta',
    'calculate_alpha', 'calculate_information_ratio', 'calculate_treynor_ratio',
    'resample_ohlcv', 'calculate_drawdowns', 'calculate_rolling_metrics',
    'calculate_trade_statistics', 'save_backtest_results', 'load_backtest_results',
    'run_monte_carlo_simulation', 'plot_monte_carlo_simulation',
    'run_walk_forward_analysis', 'plot_walk_forward_analysis'
]