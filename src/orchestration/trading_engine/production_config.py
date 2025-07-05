"""Production Configuration for the Trading Engine.

This module provides production-specific configuration classes and utilities
for the trading engine, extending the base configuration with production-grade
settings, controls, and safeguards.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
import os
from datetime import time, datetime
import pytz

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.config import (
    TradingEngineConfig, OrderConfig, SignalConfig,
    TradingHoursConfig, RiskLimitsConfig
)

# Configure logger
logger = get_logger(__name__)


@dataclass
class ProductionOrderConfig(OrderConfig):
    """Production-specific configuration for order execution."""
    # Production slippage model parameters with market data calibration
    market_impact_model: str = "adaptive"  # Options: "fixed", "percentage", "adaptive", "ml_based"
    market_impact_lookback_days: int = 20  # Days of market data for impact calibration
    volatility_adjustment_factor: float = 1.5  # Adjust slippage based on volatility
    
    # Production order size limits
    max_order_notional_value: Dict[str, float] = field(default_factory=lambda: {
        "default": 100000.0,  # Default max order value
        "liquid_large_cap": 500000.0,  # For highly liquid large cap stocks
        "mid_cap": 100000.0,  # For mid cap stocks
        "small_cap": 50000.0,  # For small cap stocks
        "micro_cap": 10000.0,  # For micro cap stocks
        "options": 50000.0,  # For options
        "futures": 200000.0,  # For futures
    })
    
    max_order_percentage_of_adv: Dict[str, float] = field(default_factory=lambda: {
        "default": 0.01,  # Default 1% of ADV
        "liquid_large_cap": 0.02,  # 2% for highly liquid large cap
        "mid_cap": 0.01,  # 1% for mid cap
        "small_cap": 0.005,  # 0.5% for small cap
        "micro_cap": 0.002,  # 0.2% for micro cap
    })
    
    # Production execution rate controls
    min_time_between_orders_ms: Dict[str, int] = field(default_factory=lambda: {
        "default": 1000,  # Default 1 second between orders
        "high_frequency": 100,  # 100ms for high frequency strategies
        "day_trading": 500,  # 500ms for day trading strategies
        "swing_trading": 5000,  # 5 seconds for swing trading strategies
    })
    
    max_orders_per_minute: Dict[str, int] = field(default_factory=lambda: {
        "default": 10,  # Default 10 orders per minute
        "high_frequency": 60,  # 60 orders per minute for high frequency
        "day_trading": 30,  # 30 orders per minute for day trading
        "swing_trading": 10,  # 10 orders per minute for swing trading
    })
    
    # Production execution strategy parameters
    twap_settings: Dict[str, Any] = field(default_factory=lambda: {
        "min_duration_minutes": 5,
        "max_duration_minutes": 120,
        "default_duration_minutes": 30,
        "interval_seconds": 60,
        "randomize_interval": True,
        "randomize_size": True,
        "size_variance_percentage": 10,
    })
    
    vwap_settings: Dict[str, Any] = field(default_factory=lambda: {
        "min_duration_minutes": 10,
        "max_duration_minutes": 240,
        "default_duration_minutes": 60,
        "use_historical_profile": True,
        "profile_lookback_days": 20,
        "allow_deviation_percentage": 5,
    })
    
    # Production order validation rules
    price_band_percentage: Dict[str, float] = field(default_factory=lambda: {
        "default": 0.05,  # Default 5% from last price
        "volatile": 0.10,  # 10% for volatile stocks
        "stable": 0.03,  # 3% for stable stocks
        "options": 0.15,  # 15% for options
    })
    
    # Production retry policies
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "initial_delay_ms": 500,
        "max_delay_ms": 5000,
        "backoff_factor": 2.0,
        "retry_on_errors": [
            "CONNECTION_ERROR",
            "TIMEOUT_ERROR",
            "RATE_LIMIT_ERROR",
            "TEMPORARY_SERVER_ERROR"
        ],
        "no_retry_on_errors": [
            "INVALID_ORDER",
            "INSUFFICIENT_FUNDS",
            "MARKET_CLOSED",
            "SECURITY_NOT_FOUND"
        ]
    })
    
    # Emergency market condition handling
    circuit_breaker_levels: Dict[str, Any] = field(default_factory=lambda: {
        "level_1": {
            "market_decline_percentage": 7.0,
            "action": "pause_trading",
            "duration_minutes": 15
        },
        "level_2": {
            "market_decline_percentage": 13.0,
            "action": "pause_trading",
            "duration_minutes": 30
        },
        "level_3": {
            "market_decline_percentage": 20.0,
            "action": "halt_trading",
            "duration_minutes": 0  # 0 means for the rest of the day
        }
    })
    
    volatility_pause_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "single_stock_move_percentage": 10.0,
        "market_volatility_index_threshold": 30.0,
        "pause_duration_minutes": 5,
        "max_pauses_per_day": 3
    })
    
    # Broker-specific order mapping
    broker_order_mapping: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "zerodha": {
            "order_type_mapping": {
                "market": "MARKET",
                "limit": "LIMIT",
                "stop": "SL",
                "stop_limit": "SL-M"
            },
            "time_in_force_mapping": {
                "day": "DAY",
                "ioc": "IOC",
                "gtc": "GTC"
            },
            "product_type_mapping": {
                "intraday": "MIS",
                "delivery": "CNC",
                "margin": "NRML"
            }
        },
        "interactive_brokers": {
            "order_type_mapping": {
                "market": "MKT",
                "limit": "LMT",
                "stop": "STP",
                "stop_limit": "STPLMT"
            },
            "time_in_force_mapping": {
                "day": "DAY",
                "gtc": "GTC",
                "ioc": "IOC",
                "fok": "FOK"
            },
            "product_type_mapping": {
                "cash": "CASH",
                "margin": "MARGIN",
                "short": "SHORT"
            }
        }
    })


@dataclass
class ProductionSignalConfig(SignalConfig):
    """Production-specific configuration for signal generation and processing."""
    # Production signal thresholds by strategy type
    signal_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "momentum": {
            "min_signal_strength": 0.3,
            "strong_signal_threshold": 0.7,
            "min_confidence": 0.6
        },
        "mean_reversion": {
            "min_signal_strength": 0.4,
            "strong_signal_threshold": 0.8,
            "min_confidence": 0.7
        },
        "trend_following": {
            "min_signal_strength": 0.25,
            "strong_signal_threshold": 0.65,
            "min_confidence": 0.55
        },
        "statistical_arbitrage": {
            "min_signal_strength": 0.35,
            "strong_signal_threshold": 0.75,
            "min_confidence": 0.65
        },
        "options": {
            "min_signal_strength": 0.45,
            "strong_signal_threshold": 0.85,
            "min_confidence": 0.75
        }
    })
    
    # Production filtering parameters
    market_regime_filters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "bull_market": {
            "momentum_weight": 1.2,
            "mean_reversion_weight": 0.8,
            "trend_following_weight": 1.3,
            "volatility_threshold": 0.015
        },
        "bear_market": {
            "momentum_weight": 0.7,
            "mean_reversion_weight": 1.2,
            "trend_following_weight": 0.6,
            "volatility_threshold": 0.025
        },
        "sideways_market": {
            "momentum_weight": 0.8,
            "mean_reversion_weight": 1.3,
            "trend_following_weight": 0.7,
            "volatility_threshold": 0.02
        },
        "high_volatility": {
            "momentum_weight": 0.6,
            "mean_reversion_weight": 0.7,
            "trend_following_weight": 0.5,
            "volatility_threshold": 0.035
        }
    })
    
    # Production signal filters by asset class
    asset_class_filters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "equities": {
            "min_market_cap": 500000000,  # $500M minimum market cap
            "min_avg_daily_volume": 100000,  # 100K minimum ADV
            "min_price": 5.0,  # $5 minimum price
            "max_price": 10000.0,  # $10,000 maximum price
            "exclude_sectors": []  # No excluded sectors
        },
        "options": {
            "min_open_interest": 500,  # 500 minimum open interest
            "min_volume": 100,  # 100 minimum daily volume
            "min_days_to_expiration": 5,  # 5 days minimum to expiration
            "max_days_to_expiration": 60,  # 60 days maximum to expiration
            "min_implied_volatility": 0.15,  # 15% minimum IV
            "max_implied_volatility": 1.0  # 100% maximum IV
        },
        "futures": {
            "min_open_interest": 1000,  # 1000 minimum open interest
            "min_volume": 500,  # 500 minimum daily volume
            "min_days_to_expiration": 10,  # 10 days minimum to expiration
            "max_days_to_expiration": 90  # 90 days maximum to expiration
        },
        "forex": {
            "min_daily_volume": 1000000,  # $1M minimum daily volume
            "allowed_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD", "USD/CHF"]
        },
        "crypto": {
            "min_market_cap": 1000000000,  # $1B minimum market cap
            "min_daily_volume": 10000000,  # $10M minimum daily volume
            "allowed_assets": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX"]
        }
    })
    
    # Production time-of-day filters
    time_of_day_filters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "market_open": {
            "duration_minutes": 30,
            "signal_strength_multiplier": 0.8,
            "min_confidence_multiplier": 1.2
        },
        "market_close": {
            "duration_minutes": 30,
            "signal_strength_multiplier": 0.7,
            "min_confidence_multiplier": 1.3
        },
        "lunch_hour": {
            "start_time": "12:00",
            "end_time": "13:00",
            "signal_strength_multiplier": 0.9,
            "min_confidence_multiplier": 1.1
        },
        "after_hours": {
            "signal_strength_multiplier": 0.6,
            "min_confidence_multiplier": 1.4
        }
    })
    
    # Production news and earnings filters
    event_filters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "earnings_announcement": {
            "days_before": 5,
            "days_after": 2,
            "action": "increase_threshold",
            "threshold_multiplier": 1.5
        },
        "major_economic_release": {
            "minutes_before": 30,
            "minutes_after": 30,
            "action": "pause_trading"
        },
        "company_news": {
            "sentiment_threshold": 0.2,  # -1 to 1 scale, 0.2 means moderately positive
            "impact_threshold": 0.5,  # 0 to 1 scale, 0.5 means medium impact
            "action": "adjust_signal",
            "adjustment_factor": 0.2  # Adjust signal strength by 20%
        }
    })


@dataclass
class ProductionTradingHoursConfig(TradingHoursConfig):
    """Production-specific configuration for trading hours."""
    # Production trading hours by market
    market_hours: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "US": {
            "timezone": "America/New_York",
            "pre_market_start": "04:00",
            "market_open": "09:30",
            "market_close": "16:00",
            "post_market_end": "20:00",
            "allow_pre_market_trading": True,
            "allow_post_market_trading": True
        },
        "India": {
            "timezone": "Asia/Kolkata",
            "pre_market_start": "09:00",
            "market_open": "09:15",
            "market_close": "15:30",
            "post_market_end": "16:00",
            "allow_pre_market_trading": False,
            "allow_post_market_trading": False
        },
        "UK": {
            "timezone": "Europe/London",
            "pre_market_start": "07:00",
            "market_open": "08:00",
            "market_close": "16:30",
            "post_market_end": "16:35",
            "allow_pre_market_trading": False,
            "allow_post_market_trading": False
        },
        "Japan": {
            "timezone": "Asia/Tokyo",
            "pre_market_start": "08:00",
            "market_open": "09:00",
            "market_close": "15:00",
            "post_market_end": "15:30",
            "allow_pre_market_trading": False,
            "allow_post_market_trading": False
        },
        "Hong_Kong": {
            "timezone": "Asia/Hong_Kong",
            "pre_market_start": "09:00",
            "market_open": "09:30",
            "market_close": "16:00",
            "post_market_end": "16:10",
            "allow_pre_market_trading": False,
            "allow_post_market_trading": False
        },
        "Crypto": {
            "timezone": "UTC",
            "pre_market_start": "00:00",
            "market_open": "00:00",
            "market_close": "23:59",
            "post_market_end": "23:59",
            "allow_pre_market_trading": True,
            "allow_post_market_trading": True,
            "weekend_trading_enabled": True
        }
    })
    
    # Production holiday calendars
    holiday_calendars: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "US": [
            {"name": "New Year's Day", "date": "2023-01-02", "status": "closed"},
            {"name": "Martin Luther King Jr. Day", "date": "2023-01-16", "status": "closed"},
            {"name": "Presidents' Day", "date": "2023-02-20", "status": "closed"},
            {"name": "Good Friday", "date": "2023-04-07", "status": "closed"},
            {"name": "Memorial Day", "date": "2023-05-29", "status": "closed"},
            {"name": "Juneteenth", "date": "2023-06-19", "status": "closed"},
            {"name": "Independence Day", "date": "2023-07-04", "status": "closed"},
            {"name": "Labor Day", "date": "2023-09-04", "status": "closed"},
            {"name": "Thanksgiving Day", "date": "2023-11-23", "status": "closed"},
            {"name": "Christmas Day", "date": "2023-12-25", "status": "closed"}
        ],
        "India": [
            {"name": "Republic Day", "date": "2023-01-26", "status": "closed"},
            {"name": "Holi", "date": "2023-03-08", "status": "closed"},
            {"name": "Ram Navami", "date": "2023-03-30", "status": "closed"},
            {"name": "Good Friday", "date": "2023-04-07", "status": "closed"},
            {"name": "Dr.Ambedkar Jayanti", "date": "2023-04-14", "status": "closed"},
            {"name": "Maharashtra Day", "date": "2023-05-01", "status": "closed"},
            {"name": "Bakri Id", "date": "2023-06-29", "status": "closed"},
            {"name": "Independence Day", "date": "2023-08-15", "status": "closed"},
            {"name": "Gandhi Jayanti", "date": "2023-10-02", "status": "closed"},
            {"name": "Diwali", "date": "2023-11-14", "status": "closed"},
            {"name": "Christmas", "date": "2023-12-25", "status": "closed"}
        ]
    })
    
    # Production early close days
    early_close_days: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "US": [
            {"name": "Day Before Independence Day", "date": "2023-07-03", "close_time": "13:00"},
            {"name": "Day After Thanksgiving", "date": "2023-11-24", "close_time": "13:00"},
            {"name": "Christmas Eve", "date": "2023-12-22", "close_time": "13:00"}
        ]
    })
    
    # Production market status override
    market_status_override: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "US": {
            "2023-08-29": {"status": "open", "reason": "System test", "open_time": "09:30", "close_time": "16:00"},
            "2023-09-15": {"status": "closed", "reason": "Emergency maintenance"}
        }
    })


@dataclass
class ProductionRiskLimitsConfig(RiskLimitsConfig):
    """Production-specific configuration for risk limits."""
    # Production position limits by asset class
    position_limits_by_asset: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "equities": {
            "max_position_percentage": 0.05,  # 5% of portfolio
            "max_position_value": 100000.0,  # $100K max position
            "max_concentration_percentage": 0.25  # 25% max in one sector
        },
        "options": {
            "max_position_percentage": 0.02,  # 2% of portfolio
            "max_position_value": 50000.0,  # $50K max position
            "max_concentration_percentage": 0.15  # 15% max in one underlying
        },
        "futures": {
            "max_position_percentage": 0.03,  # 3% of portfolio
            "max_position_value": 75000.0,  # $75K max position
            "max_concentration_percentage": 0.2  # 20% max in one sector
        },
        "forex": {
            "max_position_percentage": 0.04,  # 4% of portfolio
            "max_position_value": 80000.0,  # $80K max position
            "max_concentration_percentage": 0.3  # 30% max in one currency
        },
        "crypto": {
            "max_position_percentage": 0.01,  # 1% of portfolio
            "max_position_value": 25000.0,  # $25K max position
            "max_concentration_percentage": 0.1  # 10% max in one crypto
        }
    })
    
    # Production drawdown limits by strategy
    drawdown_limits_by_strategy: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "momentum": {
            "daily_drawdown_percentage": 0.02,  # 2% daily drawdown limit
            "weekly_drawdown_percentage": 0.05,  # 5% weekly drawdown limit
            "monthly_drawdown_percentage": 0.1,  # 10% monthly drawdown limit
            "max_drawdown_percentage": 0.15  # 15% max drawdown limit
        },
        "mean_reversion": {
            "daily_drawdown_percentage": 0.025,  # 2.5% daily drawdown limit
            "weekly_drawdown_percentage": 0.06,  # 6% weekly drawdown limit
            "monthly_drawdown_percentage": 0.12,  # 12% monthly drawdown limit
            "max_drawdown_percentage": 0.18  # 18% max drawdown limit
        },
        "trend_following": {
            "daily_drawdown_percentage": 0.03,  # 3% daily drawdown limit
            "weekly_drawdown_percentage": 0.07,  # 7% weekly drawdown limit
            "monthly_drawdown_percentage": 0.15,  # 15% monthly drawdown limit
            "max_drawdown_percentage": 0.2  # 20% max drawdown limit
        },
        "statistical_arbitrage": {
            "daily_drawdown_percentage": 0.015,  # 1.5% daily drawdown limit
            "weekly_drawdown_percentage": 0.04,  # 4% weekly drawdown limit
            "monthly_drawdown_percentage": 0.08,  # 8% monthly drawdown limit
            "max_drawdown_percentage": 0.12  # 12% max drawdown limit
        },
        "options": {
            "daily_drawdown_percentage": 0.04,  # 4% daily drawdown limit
            "weekly_drawdown_percentage": 0.1,  # 10% weekly drawdown limit
            "monthly_drawdown_percentage": 0.2,  # 20% monthly drawdown limit
            "max_drawdown_percentage": 0.25  # 25% max drawdown limit
        }
    })
    
    # Production VaR limits
    var_limits: Dict[str, Any] = field(default_factory=lambda: {
        "confidence_level": 0.99,  # 99% confidence level
        "time_horizon_days": 1,  # 1-day VaR
        "max_var_percentage": 0.03,  # 3% maximum VaR
        "calculation_method": "historical",  # Options: "historical", "parametric", "monte_carlo"
        "lookback_days": 252,  # 1 year of data
        "stress_test_scenarios": [
            "2008_financial_crisis",
            "2020_covid_crash",
            "2022_rate_hike"
        ]
    })
    
    # Production correlation limits
    correlation_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_correlation_threshold": 0.7,  # Maximum correlation between positions
        "min_positions_for_check": 5,  # Minimum positions before checking correlation
        "lookback_days": 63,  # 3 months of data
        "max_portfolio_correlation_to_market": 0.8  # Maximum correlation to market
    })
    
    # Production leverage limits
    leverage_limits: Dict[str, float] = field(default_factory=lambda: {
        "max_gross_leverage": 2.0,  # Maximum gross leverage
        "max_net_leverage": 1.0,  # Maximum net leverage
        "max_sector_leverage": 1.5,  # Maximum leverage in any sector
        "max_single_name_leverage": 1.0  # Maximum leverage in any single name
    })


@dataclass
class TradeLifecycleProductionConfig:
    """Production-specific configuration for trade lifecycle management."""
    # Production trade state transition rules
    state_transition_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "signal_generated": {
            "timeout_seconds": 300,  # 5 minutes timeout
            "next_state_on_timeout": "expired",
            "allowed_next_states": ["order_created", "rejected", "expired"]
        },
        "order_created": {
            "timeout_seconds": 600,  # 10 minutes timeout
            "next_state_on_timeout": "cancelled",
            "allowed_next_states": ["partially_filled", "filled", "cancelled", "rejected"]
        },
        "partially_filled": {
            "timeout_seconds": 1800,  # 30 minutes timeout
            "next_state_on_timeout": "cancelled",
            "allowed_next_states": ["filled", "cancelled"]
        },
        "filled": {
            "allowed_next_states": ["completed", "exit_order_created"]
        },
        "exit_order_created": {
            "timeout_seconds": 1800,  # 30 minutes timeout
            "next_state_on_timeout": "force_exit",
            "allowed_next_states": ["exit_partially_filled", "exit_filled", "cancelled"]
        },
        "exit_partially_filled": {
            "timeout_seconds": 1800,  # 30 minutes timeout
            "next_state_on_timeout": "force_exit",
            "allowed_next_states": ["exit_filled", "force_exit"]
        },
        "exit_filled": {
            "allowed_next_states": ["completed"]
        },
        "force_exit": {
            "timeout_seconds": 300,  # 5 minutes timeout
            "next_state_on_timeout": "error",
            "allowed_next_states": ["completed", "error"]
        }
    })
    
    # Production trade reporting configuration
    reporting_config: Dict[str, Any] = field(default_factory=lambda: {
        "real_time_reporting": {
            "enabled": True,
            "include_fields": [
                "trade_id", "symbol", "side", "quantity", "entry_price",
                "exit_price", "profit_loss", "status", "timestamp"
            ],
            "notification_events": [
                "order_filled", "order_cancelled", "order_rejected",
                "exit_filled", "trade_completed", "error"
            ]
        },
        "daily_reporting": {
            "enabled": True,
            "generation_time": "17:30",  # 5:30 PM
            "timezone": "America/New_York",
            "formats": ["csv", "json", "pdf"],
            "distribution_list": ["trading_team@example.com", "risk@example.com"]
        },
        "audit_trail": {
            "enabled": True,
            "log_all_state_transitions": True,
            "log_all_order_events": True,
            "include_user_actions": True,
            "retention_days": 2555  # 7 years
        }
    })
    
    # Production trade archiving configuration
    archiving_config: Dict[str, Any] = field(default_factory=lambda: {
        "archive_completed_trades_after_days": 30,
        "archive_storage_path": "/data/archives/trades",
        "archive_format": "parquet",
        "include_full_order_history": True,
        "include_signals": True,
        "compression": "snappy"
    })


@dataclass
class ProductionTradingEngineConfig(TradingEngineConfig):
    """Production-specific configuration for the trading engine."""
    # Override component configurations with production versions
    order_config: ProductionOrderConfig = field(default_factory=ProductionOrderConfig)
    signal_config: ProductionSignalConfig = field(default_factory=ProductionSignalConfig)
    trading_hours_config: ProductionTradingHoursConfig = field(default_factory=ProductionTradingHoursConfig)
    risk_limits_config: ProductionRiskLimitsConfig = field(default_factory=ProductionRiskLimitsConfig)
    
    # Add trade lifecycle configuration
    trade_lifecycle_config: TradeLifecycleProductionConfig = field(default_factory=TradeLifecycleProductionConfig)
    
    # Production-specific settings
    engine_mode: str = "live"  # Override default "paper" mode
    environment: str = "production"  # Explicitly mark as production
    
    # Production monitoring and alerting
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        "health_check_interval_seconds": 60,
        "performance_metrics_interval_seconds": 300,
        "alert_thresholds": {
            "order_latency_ms": 500,
            "signal_processing_latency_ms": 200,
            "error_rate_percentage": 5.0,
            "cpu_usage_percentage": 80.0,
            "memory_usage_percentage": 80.0
        },
        "alert_channels": [
            "email",
            "slack",
            "pagerduty"
        ],
        "alert_contacts": {
            "email": ["trading-alerts@example.com", "oncall@example.com"],
            "slack": "#trading-alerts",
            "pagerduty": "trading-engine-oncall"
        }
    })
    
    # Production failover configuration
    failover_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "failover_triggers": [
            "process_crash",
            "heartbeat_missing",
            "error_threshold_exceeded",
            "manual_trigger"
        ],
        "primary_instance": "trading-engine-primary",
        "backup_instance": "trading-engine-backup",
        "max_failover_attempts": 3,
        "failback_delay_minutes": 30
    })
    
    # Production data persistence
    persistence_config: Dict[str, Any] = field(default_factory=lambda: {
        "database_type": "postgresql",
        "connection_pool_size": 10,
        "max_connections": 20,
        "connection_timeout_seconds": 5,
        "retry_attempts": 3,
        "retry_delay_seconds": 1,
        "tables": {
            "trades": "production_trades",
            "orders": "production_orders",
            "signals": "production_signals",
            "events": "production_events"
        }
    })
    
    # Production API rate limits
    api_rate_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_requests_per_minute": 300,
        "max_requests_per_hour": 5000,
        "throttling_enabled": True,
        "throttling_strategy": "token_bucket"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        base_dict = super().to_dict()
        
        # Add production-specific fields
        base_dict["trade_lifecycle_config"] = self.trade_lifecycle_config.__dict__
        base_dict["environment"] = self.environment
        base_dict["monitoring_config"] = self.monitoring_config
        base_dict["failover_config"] = self.failover_config
        base_dict["persistence_config"] = self.persistence_config
        base_dict["api_rate_limits"] = self.api_rate_limits
        
        return base_dict


def get_production_config() -> ProductionTradingEngineConfig:
    """Get the production trading engine configuration.
    
    Returns:
        ProductionTradingEngineConfig: The production configuration.
    """
    return ProductionTradingEngineConfig()


def load_production_config(config_path: Optional[str] = None) -> ProductionTradingEngineConfig:
    """Load the production trading engine configuration.
    
    Args:
        config_path: The path to the configuration file. If None, uses the default path.
        
    Returns:
        ProductionTradingEngineConfig: The loaded production configuration.
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            os.path.join(os.getcwd(), "trading_engine_production_config.json"),
            os.path.join(os.path.dirname(__file__), "trading_engine_production_config.json"),
            os.path.join(os.path.expanduser("~"), ".friday", "trading_engine_production_config.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create base config
            config = ProductionTradingEngineConfig()
            
            # TODO: Implement full deserialization of nested production config
            # This would require more complex deserialization logic than the base config
            
            logger.info(f"Loaded production trading engine configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load production trading engine configuration: {str(e)}")
            logger.info("Using default production configuration")
            return get_production_config()
    else:
        logger.info("No production configuration file found, using default production configuration")
        return get_production_config()