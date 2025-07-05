"""Configuration module for the Trading Engine.

This module provides configuration classes and utilities for the trading engine.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
import os
from datetime import time

from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


@dataclass
class OrderConfig:
    """Configuration for order execution."""
    # Default slippage model parameters
    default_slippage_model: str = "fixed"  # Options: "fixed", "percentage", "market_impact"
    fixed_slippage_points: int = 1  # For fixed slippage model
    percentage_slippage: float = 0.0005  # 0.05% for percentage slippage model
    
    # Order execution parameters
    default_execution_strategy: str = "immediate"  # Options: "immediate", "twap", "vwap"
    twap_interval_seconds: int = 300  # 5 minutes for TWAP strategy
    twap_max_duration_minutes: int = 60  # 1 hour max duration for TWAP
    
    # Order validation parameters
    max_order_value: float = 100000.0  # Maximum order value in account currency
    min_order_value: float = 10.0  # Minimum order value in account currency
    
    # Order retry parameters
    max_retry_attempts: int = 3  # Maximum number of retry attempts for failed orders
    retry_delay_seconds: int = 5  # Delay between retry attempts


@dataclass
class SignalConfig:
    """Configuration for signal generation and processing."""
    # Signal thresholds
    min_signal_strength: float = 0.2  # Minimum signal strength to generate an order
    strong_signal_threshold: float = 0.7  # Threshold for strong signals
    
    # Signal expiration
    signal_expiry_seconds: int = 300  # Signals expire after 5 minutes by default
    
    # Signal aggregation parameters
    aggregation_method: str = "weighted_average"  # Options: "weighted_average", "highest_confidence", "consensus"
    min_signals_for_consensus: int = 3  # Minimum number of signals needed for consensus
    
    # Signal filtering
    apply_trend_filter: bool = True  # Apply market trend filter to signals
    apply_volatility_filter: bool = True  # Apply volatility filter to signals
    min_volatility_threshold: float = 0.01  # Minimum volatility for valid signals
    max_volatility_threshold: float = 0.05  # Maximum volatility for valid signals


@dataclass
class TradingHoursConfig:
    """Configuration for trading hours."""
    # Trading session times (24-hour format)
    pre_market_start: time = field(default_factory=lambda: time(8, 0))  # 8:00 AM
    market_open: time = field(default_factory=lambda: time(9, 30))  # 9:30 AM
    market_close: time = field(default_factory=lambda: time(16, 0))  # 4:00 PM
    post_market_end: time = field(default_factory=lambda: time(20, 0))  # 8:00 PM
    
    # Trading session flags
    allow_pre_market_trading: bool = False
    allow_post_market_trading: bool = False
    
    # Weekend trading
    weekend_trading_enabled: bool = False
    
    # Holiday calendar
    holiday_calendar_name: str = "NYSE"  # Options: "NYSE", "NASDAQ", "LSE", etc.


@dataclass
class RiskLimitsConfig:
    """Configuration for risk limits in the trading engine."""
    # Position limits
    max_position_size_percentage: float = 0.05  # Maximum position size as percentage of portfolio
    max_position_value: float = 50000.0  # Maximum position value in account currency
    
    # Exposure limits
    max_total_exposure_percentage: float = 0.8  # Maximum total exposure as percentage of portfolio
    max_sector_exposure_percentage: float = 0.3  # Maximum sector exposure as percentage of portfolio
    
    # Drawdown limits
    max_daily_drawdown_percentage: float = 0.03  # Maximum daily drawdown (3%)
    max_total_drawdown_percentage: float = 0.15  # Maximum total drawdown (15%)
    
    # Trade frequency limits
    max_daily_trades: int = 50  # Maximum number of trades per day
    min_time_between_trades_seconds: int = 60  # Minimum time between trades
    
    # Signal adjustment parameters
    risk_adjustment_factor: float = 1.0  # Factor to adjust signal strength based on risk assessment
    max_signal_strength: float = 1.0  # Maximum allowed signal strength
    min_signal_strength: float = 0.1  # Minimum signal strength to consider valid
    enable_signal_filtering: bool = True  # Whether to enable signal filtering


@dataclass
class TradingEngineConfig:
    """Main configuration for the trading engine."""
    # Component configurations
    order_config: OrderConfig = field(default_factory=OrderConfig)
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    trading_hours_config: TradingHoursConfig = field(default_factory=TradingHoursConfig)
    risk_limits_config: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    
    # General settings
    engine_mode: str = "paper"  # Options: "paper", "live", "backtest"
    default_market: str = "US"  # Default market for trading
    log_level: str = "INFO"  # Logging level
    
    # Performance monitoring
    performance_tracking_enabled: bool = True
    metrics_reporting_interval_seconds: int = 300  # Report metrics every 5 minutes
    
    # Event system settings
    event_system_enabled: bool = True
    event_types_to_publish: List[str] = field(default_factory=lambda: [
        "order_created", "order_filled", "order_cancelled", "trade_completed", 
        "signal_generated", "trading_engine_heartbeat"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return {
            "order_config": self.order_config.__dict__,
            "signal_config": self.signal_config.__dict__,
            "trading_hours_config": {
                **{k: str(v) if isinstance(v, time) else v 
                   for k, v in self.trading_hours_config.__dict__.items()}
            },
            "risk_limits_config": self.risk_limits_config.__dict__,
            "engine_mode": self.engine_mode,
            "default_market": self.default_market,
            "log_level": self.log_level,
            "performance_tracking_enabled": self.performance_tracking_enabled,
            "metrics_reporting_interval_seconds": self.metrics_reporting_interval_seconds,
            "event_system_enabled": self.event_system_enabled,
            "event_types_to_publish": self.event_types_to_publish
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save the configuration to a JSON file.
        
        Args:
            file_path: The path to save the configuration to.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Saved trading engine configuration to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save trading engine configuration: {str(e)}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'TradingEngineConfig':
        """Load the configuration from a JSON file.
        
        Args:
            file_path: The path to load the configuration from.
            
        Returns:
            TradingEngineConfig: The loaded configuration.
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create base config
            config = cls()
            
            # Update order config
            if "order_config" in config_dict:
                for k, v in config_dict["order_config"].items():
                    if hasattr(config.order_config, k):
                        setattr(config.order_config, k, v)
            
            # Update signal config
            if "signal_config" in config_dict:
                for k, v in config_dict["signal_config"].items():
                    if hasattr(config.signal_config, k):
                        setattr(config.signal_config, k, v)
            
            # Update trading hours config
            if "trading_hours_config" in config_dict:
                for k, v in config_dict["trading_hours_config"].items():
                    if hasattr(config.trading_hours_config, k):
                        # Convert time strings to time objects
                        if k in ["pre_market_start", "market_open", "market_close", "post_market_end"]:
                            if isinstance(v, str):
                                hour, minute = map(int, v.split(':'))
                                v = time(hour, minute)
                        setattr(config.trading_hours_config, k, v)
            
            # Update risk limits config
            if "risk_limits_config" in config_dict:
                for k, v in config_dict["risk_limits_config"].items():
                    if hasattr(config.risk_limits_config, k):
                        setattr(config.risk_limits_config, k, v)
            
            # Update general settings
            for k in ["engine_mode", "default_market", "log_level", 
                     "performance_tracking_enabled", "metrics_reporting_interval_seconds",
                     "event_system_enabled", "event_types_to_publish"]:
                if k in config_dict:
                    setattr(config, k, config_dict[k])
            
            logger.info(f"Loaded trading engine configuration from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load trading engine configuration: {str(e)}")
            logger.info("Using default configuration")
            return cls()


def get_default_config() -> TradingEngineConfig:
    """Get the default trading engine configuration.
    
    Returns:
        TradingEngineConfig: The default configuration.
    """
    return TradingEngineConfig()


def load_config(config_path: Optional[str] = None) -> TradingEngineConfig:
    """Load the trading engine configuration.
    
    Args:
        config_path: The path to the configuration file. If None, uses the default path.
        
    Returns:
        TradingEngineConfig: The loaded configuration.
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            os.path.join(os.getcwd(), "trading_engine_config.json"),
            os.path.join(os.path.dirname(__file__), "trading_engine_config.json"),
            os.path.join(os.path.expanduser("~"), ".friday", "trading_engine_config.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        return TradingEngineConfig.load_from_file(config_path)
    else:
        logger.info("No configuration file found, using default configuration")
        return get_default_config()