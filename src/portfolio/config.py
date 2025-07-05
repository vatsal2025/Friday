"""Configuration module for the Portfolio Management System.

This module provides configuration classes and utilities for the portfolio management system,
including default configurations and configuration validation.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
import logging
import json
import os
from datetime import datetime

from src.portfolio.tax_manager import TaxLotMethod
from src.portfolio.allocation_manager import RebalanceMethod

logger = logging.getLogger(__name__)

class PortfolioConfigValidator:
    """Validates portfolio management system configuration."""

    @staticmethod
    def validate_portfolio_manager_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio manager configuration.

        Args:
            config: Portfolio manager configuration dictionary

        Returns:
            Validated configuration with defaults applied

        Raises:
            ValueError: If configuration is invalid
        """
        validated = {}

        # Required fields
        if "portfolio_id" not in config:
            raise ValueError("portfolio_id is required in portfolio_manager configuration")
        validated["portfolio_id"] = config["portfolio_id"]

        # Optional fields with defaults
        validated["initial_cash"] = config.get("initial_cash", 0.0)
        validated["include_historical_data"] = config.get("include_historical_data", True)
        validated["max_history_length"] = config.get("max_history_length", 1000)

        return validated

    @staticmethod
    def validate_performance_calculator_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance calculator configuration.

        Args:
            config: Performance calculator configuration dictionary

        Returns:
            Validated configuration with defaults applied
        """
        validated = {}

        # Optional fields with defaults
        validated["benchmark_symbol"] = config.get("benchmark_symbol", "SPY")
        validated["risk_free_rate"] = config.get("risk_free_rate", 0.02)
        validated["max_history_length"] = config.get("max_history_length", 1000)
        validated["annualization_factor"] = config.get("annualization_factor", 252)  # Trading days in a year

        return validated

    @staticmethod
    def validate_tax_manager_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tax manager configuration.

        Args:
            config: Tax manager configuration dictionary

        Returns:
            Validated configuration with defaults applied

        Raises:
            ValueError: If configuration is invalid
        """
        validated = {}

        # Optional fields with defaults
        default_method = config.get("default_method", "FIFO")
        try:
            if isinstance(default_method, str):
                validated["default_method"] = TaxLotMethod[default_method]
            else:
                validated["default_method"] = default_method
        except KeyError:
            raise ValueError(f"Invalid tax lot method: {default_method}. Must be one of {[m.name for m in TaxLotMethod]}")

        validated["wash_sale_window_days"] = config.get("wash_sale_window_days", 30)
        validated["long_term_threshold_days"] = config.get("long_term_threshold_days", 365)

        # Symbol-specific methods
        symbol_methods = {}
        for symbol, method in config.get("symbol_methods", {}).items():
            try:
                if isinstance(method, str):
                    symbol_methods[symbol] = TaxLotMethod[method]
                else:
                    symbol_methods[symbol] = method
            except KeyError:
                raise ValueError(f"Invalid tax lot method for {symbol}: {method}. Must be one of {[m.name for m in TaxLotMethod]}")

        validated["symbol_methods"] = symbol_methods

        return validated

    @staticmethod
    def validate_allocation_manager_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate allocation manager configuration.

        Args:
            config: Allocation manager configuration dictionary

        Returns:
            Validated configuration with defaults applied

        Raises:
            ValueError: If configuration is invalid
        """
        validated = {}

        # Optional fields with defaults
        rebalance_method = config.get("rebalance_method", "THRESHOLD")
        try:
            if isinstance(rebalance_method, str):
                validated["rebalance_method"] = RebalanceMethod[rebalance_method]
            else:
                validated["rebalance_method"] = rebalance_method
        except KeyError:
            raise ValueError(f"Invalid rebalance method: {rebalance_method}. Must be one of {[m.name for m in RebalanceMethod]}")

        validated["default_threshold"] = config.get("default_threshold", 5.0)
        validated["rebalance_frequency_days"] = config.get("rebalance_frequency_days", 90)
        validated["last_rebalance_date"] = config.get("last_rebalance_date", None)

        # Allocation targets
        allocation_targets = []
        for target in config.get("allocation_targets", []):
            if "name" not in target or "target_percentage" not in target:
                raise ValueError("Allocation targets must include 'name' and 'target_percentage'")

            validated_target = {
                "name": target["name"],
                "target_percentage": target["target_percentage"],
                "category": target.get("category", "default"),
                "threshold": target.get("threshold", validated["default_threshold"])
            }
            allocation_targets.append(validated_target)

        validated["allocation_targets"] = allocation_targets

        # Validate total allocation percentage
        total_percentage = sum(target["target_percentage"] for target in allocation_targets)
        if total_percentage > 0 and abs(total_percentage - 100.0) > 0.01:
            logger.warning(f"Total allocation percentage is {total_percentage}%, not 100%")

        return validated

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete portfolio management system configuration.

        Args:
            config: Complete configuration dictionary

        Returns:
            Validated configuration with defaults applied

        Raises:
            ValueError: If configuration is invalid
        """
        validated = {}

        # Validate portfolio manager config
        if "portfolio_manager" in config:
            validated["portfolio_manager"] = PortfolioConfigValidator.validate_portfolio_manager_config(
                config["portfolio_manager"]
            )
        else:
            validated["portfolio_manager"] = PortfolioConfigValidator.validate_portfolio_manager_config({})

        # Validate performance calculator config
        if "performance_calculator" in config:
            validated["performance_calculator"] = PortfolioConfigValidator.validate_performance_calculator_config(
                config["performance_calculator"]
            )
        else:
            validated["performance_calculator"] = PortfolioConfigValidator.validate_performance_calculator_config({})

        # Validate tax manager config
        if "tax_manager" in config:
            validated["tax_manager"] = PortfolioConfigValidator.validate_tax_manager_config(
                config["tax_manager"]
            )
        else:
            validated["tax_manager"] = PortfolioConfigValidator.validate_tax_manager_config({})

        # Validate allocation manager config
        if "allocation_manager" in config:
            validated["allocation_manager"] = PortfolioConfigValidator.validate_allocation_manager_config(
                config["allocation_manager"]
            )
        else:
            validated["allocation_manager"] = PortfolioConfigValidator.validate_allocation_manager_config({})

        # Pass through risk manager config if present
        if "risk_manager" in config:
            validated["risk_manager"] = config["risk_manager"]

        return validated


class PortfolioConfig:
    """Portfolio management system configuration."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for the portfolio management system.

        Returns:
            Default configuration dictionary
        """
        return {
            "portfolio_manager": {
                "portfolio_id": "default-portfolio",
                "initial_cash": 100000.0,
                "include_historical_data": True,
                "max_history_length": 1000
            },
            "performance_calculator": {
                "benchmark_symbol": "SPY",
                "risk_free_rate": 0.02,
                "max_history_length": 1000,
                "annualization_factor": 252
            },
            "tax_manager": {
                "default_method": "FIFO",
                "wash_sale_window_days": 30,
                "long_term_threshold_days": 365,
                "symbol_methods": {}
            },
            "allocation_manager": {
                "rebalance_method": "THRESHOLD",
                "default_threshold": 5.0,
                "rebalance_frequency_days": 90,
                "allocation_targets": []
            }
        }

    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            Loaded and validated configuration dictionary

        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file is not valid JSON
            ValueError: If the configuration is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            config = json.load(f)

        return PortfolioConfigValidator.validate_config(config)

    @staticmethod
    def save_to_file(config: Dict[str, Any], file_path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            config: Configuration dictionary
            file_path: Path to save the configuration file

        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate configuration before saving
        validated_config = PortfolioConfigValidator.validate_config(config)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert Enum values to strings for JSON serialization
        serializable_config = PortfolioConfig._prepare_for_serialization(validated_config)

        with open(file_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)

    @staticmethod
    def _prepare_for_serialization(config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for JSON serialization by converting Enum values to strings.

        Args:
            config: Configuration dictionary

        Returns:
            Serializable configuration dictionary
        """
        serializable = {}

        for key, value in config.items():
            if isinstance(value, dict):
                serializable[key] = PortfolioConfig._prepare_for_serialization(value)
            elif isinstance(value, Enum):
                serializable[key] = value.name
            elif isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, list):
                serializable[key] = [
                    PortfolioConfig._prepare_for_serialization(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serializable[key] = value

        return serializable
