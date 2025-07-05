import logging
import os
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RiskManagementProductionConfig:
    """
    Production configuration for the risk management system.

    This class provides production-ready configuration settings for the risk management
    system, including position sizing, stop-loss management, portfolio risk controls,
    and circuit breakers.
    """
    # Capital and risk settings
    initial_capital: float = 1000000.0  # $1M starting capital
    risk_per_trade: float = 0.005  # 0.5% risk per trade (more conservative for production)

    # Portfolio risk limits
    max_portfolio_var_percent: float = 0.015  # 1.5% VaR (more conservative for production)
    max_drawdown_percent: float = 0.10  # 10% max drawdown
    max_sector_allocation: float = 0.20  # 20% max sector allocation (formerly max_sector_exposure)
    max_position_size: float = 0.05  # 5% max position size (formerly max_asset_exposure)
    max_history_size: int = 252  # 252 days (1 year) of price history to maintain
    var_confidence_level: float = 0.95  # 95% confidence level for VaR calculations

    # Stop loss settings
    default_stop_loss_percent: float = 0.015  # 1.5% default stop loss
    default_trailing_percent: float = 0.01  # 1% trailing stop
    default_atr_multiplier: float = 2.0  # 2x ATR for volatility stops
    default_time_stop_days: int = 10  # 10-day time stop
    default_profit_target_ratio: float = 2.0  # 2:1 reward-to-risk ratio

    # Position sizing settings
    position_sizing_method: str = "risk_based"  # risk_based, volatility_based, fixed
    max_position_size_percentage: float = 0.05  # 5% max position size
    max_position_value: float = 100000.0  # $100K max position
    volatility_lookback_days: int = 20  # 20-day lookback for volatility calculation

    # Circuit breaker settings
    market_circuit_breaker_enabled: bool = True
    account_circuit_breaker_enabled: bool = True
    market_volatility_threshold: float = 0.03  # 3% market volatility threshold
    daily_loss_limit_percent: float = 0.03  # 3% daily loss limit
    weekly_loss_limit_percent: float = 0.07  # 7% weekly loss limit

    # Position limits by asset class
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
            "max_position_percentage": 0.02,  # 2% of portfolio
            "max_position_value": 40000.0,  # $40K max position
            "max_concentration_percentage": 0.15  # 15% max in one coin
        }
    })

    # Logging and monitoring settings
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    risk_metrics_calculation_interval: int = 300  # Calculate risk metrics every 5 minutes
    alert_notification_enabled: bool = True

    # Persistence settings
    persistence_enabled: bool = True
    persistence_path: str = "/var/friday/risk"
    persistence_interval: int = 300  # Persist state every 5 minutes


def load_production_config(config_path: Optional[str] = None) -> RiskManagementProductionConfig:
    """
    Load the production risk management configuration.

    Args:
        config_path: The path to the configuration file. If None, uses the default path.

    Returns:
        RiskManagementProductionConfig: The loaded production configuration.
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            os.path.join(os.getcwd(), "config/risk_management_production.json"),
            "/etc/friday/risk_management_production.json",
            os.path.expanduser("~/.friday/risk_management_production.json")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    config = RiskManagementProductionConfig()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            logger.info(f"Loaded production risk management configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading production risk management configuration: {e}")
    else:
        logger.warning("No production risk management configuration file found, using defaults")

    return config


def save_production_config(config: RiskManagementProductionConfig, config_path: str) -> bool:
    """
    Save the production risk management configuration to a file.

    Args:
        config: The configuration to save.
        config_path: The path to save the configuration to.

    Returns:
        bool: True if the configuration was saved successfully, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Convert config to dictionary
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}

        # Save to file
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        logger.info(f"Saved production risk management configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving production risk management configuration: {e}")
        return False


def create_default_production_config(config_path: str) -> bool:
    """
    Create a default production risk management configuration file.

    Args:
        config_path: The path to save the configuration to.

    Returns:
        bool: True if the configuration was created successfully, False otherwise.
    """
    config = RiskManagementProductionConfig()
    return save_production_config(config, config_path)
