"""Configuration-aware market data validator factory.

This module provides a factory for creating market data validators based on 
configuration settings from the ConfigManager. It integrates the comprehensive
validation rules with the system configuration.
"""

from datetime import time
from typing import Optional, Set, List
import logging

from src.infrastructure.config.config_manager import ConfigurationManager
from src.data.processing.market_validation_rules import (
    build_comprehensive_market_validator,
    MarketDataValidationRules
)
from src.data.processing.data_validator import DataValidator

logger = logging.getLogger(__name__)


class ConfigAwareValidatorFactory:
    """Factory for creating configuration-aware market data validators."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the factory with a configuration manager.
        
        Args:
            config_manager: ConfigurationManager instance. If None, gets singleton instance.
        """
        self.config = config_manager or ConfigurationManager.get_instance()
        self._register_validation_schemas()
    
    def _register_validation_schemas(self) -> None:
        """Register validation schemas for market data configuration."""
        market_data_schema = {
            'max_timestamp_gap_minutes': {
                'type': int,
                'min': 1,
                'max': 1440,  # Max 24 hours
                'required': False
            },
            'min_price_bound': {
                'type': float,
                'min': 0.0001,
                'max': 1.0,
                'required': False
            },
            'max_price_bound': {
                'type': float,
                'min': 1.0,
                'max': 1000000.0,
                'required': False
            },
            'strict_type_validation': {
                'type': bool,
                'required': False
            },
            'allow_negative_prices': {
                'type': bool,
                'required': False
            },
            'allow_zero_volume': {
                'type': bool,
                'required': False
            },
            'enforce_trading_hours': {
                'type': bool,
                'required': False
            },
            'enable_symbol_whitelist': {
                'type': bool,
                'required': False
            },
            'warn_only_mode': {
                'type': bool,
                'required': False
            }
        }
        
        self.config.register_validator('validation.market_data', market_data_schema)
        logger.debug("Registered market data validation schemas")
    
    def create_validator_from_config(self, 
                                   override_warn_only: Optional[bool] = None,
                                   include_additional_rules: bool = True) -> DataValidator:
        """Create a market data validator based on current configuration.
        
        Args:
            override_warn_only: Override warn_only mode setting from config
            include_additional_rules: Include additional validation rules (price bounds, zero volume)
            
        Returns:
            DataValidator: Configured market data validator
        """
        # Load configuration parameters
        market_config = self.config.get('validation.market_data', {})
        
        # Extract parameters with defaults
        max_gap_minutes = market_config.get('max_timestamp_gap_minutes', 5)
        strict_types = market_config.get('strict_type_validation', False)
        
        # Trading hours configuration
        enforce_trading_hours = market_config.get('enforce_trading_hours', False)
        trading_start = None
        trading_end = None
        
        if enforce_trading_hours:
            start_str = market_config.get('trading_hours_start', '09:30')
            end_str = market_config.get('trading_hours_end', '16:00')
            
            try:
                start_hour, start_min = map(int, start_str.split(':'))
                end_hour, end_min = map(int, end_str.split(':'))
                trading_start = time(start_hour, start_min)
                trading_end = time(end_hour, end_min)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid trading hours format: {e}. Disabling trading hours validation.")
                trading_start = None
                trading_end = None
        
        # Symbol whitelist configuration
        enable_whitelist = market_config.get('enable_symbol_whitelist', False)
        symbol_whitelist = None
        
        if enable_whitelist:
            whitelist_symbols = market_config.get('symbol_whitelist', [])
            if whitelist_symbols:
                symbol_whitelist = set(whitelist_symbols)
        
        # Create the comprehensive validator
        validator = build_comprehensive_market_validator(
            symbol_whitelist=symbol_whitelist,
            trading_hours_start=trading_start,
            trading_hours_end=trading_end,
            max_timestamp_gap_minutes=max_gap_minutes,
            strict_type_validation=strict_types
        )
        
        # Add additional rules if requested
        if include_additional_rules:
            self._add_additional_rules(validator, market_config)
        
        logger.info(f"Created validator from config with {len(validator.validation_rules)} rules")
        
        return validator
    
    def _add_additional_rules(self, validator: DataValidator, market_config: dict) -> None:
        """Add additional validation rules based on configuration.
        
        Args:
            validator: DataValidator to add rules to
            market_config: Market data configuration dictionary
        """
        rules = MarketDataValidationRules()
        
        # Add price bounds rule if configured
        min_price = market_config.get('min_price_bound')
        max_price = market_config.get('max_price_bound')
        
        if min_price is not None and max_price is not None:
            validator.add_validation_rule(
                rules.create_price_bounds_rule(min_price=min_price, max_price=max_price)
            )
            logger.debug(f"Added price bounds rule: [{min_price}, {max_price}]")
        
        # Add zero volume rule if configured
        allow_zero_volume = market_config.get('allow_zero_volume', True)
        if not allow_zero_volume:
            validator.add_validation_rule(rules.create_no_zero_volume_rule())
            logger.debug("Added no zero volume rule")
    
    def create_realtime_validator(self) -> DataValidator:
        """Create a validator optimized for real-time data streams.
        
        Returns:
            DataValidator: Validator configured for real-time use
        """
        validator = self.create_validator_from_config(include_additional_rules=False)
        
        # For real-time, we typically want more lenient validation
        market_config = self.config.get('validation.market_data', {})
        warn_only_for_realtime = market_config.get('warn_only_for_realtime', True)
        
        if warn_only_for_realtime:
            logger.info("Real-time validator created with warn-only mode enabled")
        
        return validator
    
    def create_batch_validator(self) -> DataValidator:
        """Create a validator optimized for batch processing.
        
        Returns:
            DataValidator: Validator configured for batch processing
        """
        validator = self.create_validator_from_config(
            override_warn_only=False,
            include_additional_rules=True
        )
        
        logger.info("Batch validator created with comprehensive validation rules")
        return validator
    
    def validate_configuration(self) -> List[str]:
        """Validate the current market data validation configuration.
        
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = self.config.validate('validation.market_data')
        
        if errors:
            logger.warning(f"Configuration validation failed: {errors}")
        else:
            logger.debug("Configuration validation passed")
        
        return errors
    
    def get_validation_config_summary(self) -> dict:
        """Get a summary of the current validation configuration.
        
        Returns:
            dict: Summary of validation configuration
        """
        market_config = self.config.get('validation.market_data', {})
        
        summary = {
            'timestamp_validation': {
                'max_gap_minutes': market_config.get('max_timestamp_gap_minutes', 5),
                'require_monotonic': market_config.get('require_monotonic_timestamps', True),
                'allow_duplicates': market_config.get('allow_duplicate_timestamps', False)
            },
            'price_validation': {
                'min_bound': market_config.get('min_price_bound', 0.01),
                'max_bound': market_config.get('max_price_bound', 100000.0),
                'allow_negative': market_config.get('allow_negative_prices', False),
                'allow_zero': market_config.get('allow_zero_prices', False)
            },
            'volume_validation': {
                'allow_negative': market_config.get('allow_negative_volume', False),
                'allow_zero': market_config.get('allow_zero_volume', True),
                'min_bound': market_config.get('min_volume_bound', 0.0),
                'max_bound': market_config.get('max_volume_bound', 1e12)
            },
            'type_validation': {
                'strict_types': market_config.get('strict_type_validation', False),
                'required_columns': market_config.get('required_columns', ['open', 'high', 'low', 'close', 'volume'])
            },
            'trading_hours': {
                'enforce': market_config.get('enforce_trading_hours', False),
                'start': market_config.get('trading_hours_start', '09:30'),
                'end': market_config.get('trading_hours_end', '16:00'),
                'timezone': market_config.get('timezone', 'America/New_York')
            },
            'symbol_whitelist': {
                'enabled': market_config.get('enable_symbol_whitelist', False),
                'symbols': market_config.get('symbol_whitelist', [])
            },
            'warn_only': {
                'enabled': market_config.get('warn_only_mode', False),
                'realtime_auto': market_config.get('warn_only_for_realtime', True)
            },
            'performance': {
                'detailed_logging': market_config.get('enable_detailed_logging', True),
                'max_time_seconds': market_config.get('max_validation_time_seconds', 30),
                'batch_size': market_config.get('batch_validation_size', 10000)
            }
        }
        
        return summary
    
    def update_config_from_dict(self, config_updates: dict) -> None:
        """Update validation configuration from a dictionary.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        for key, value in config_updates.items():
            config_path = f'validation.market_data.{key}'
            self.config.set(config_path, value)
            logger.debug(f"Updated config: {config_path} = {value}")
        
        # Validate the updated configuration
        errors = self.validate_configuration()
        if errors:
            logger.error(f"Configuration validation failed after updates: {errors}")
            raise ValueError(f"Invalid configuration updates: {errors}")
    
    def reset_to_defaults(self) -> None:
        """Reset validation configuration to default values."""
        default_config = {
            'max_timestamp_gap_minutes': 5,
            'require_monotonic_timestamps': True,
            'allow_duplicate_timestamps': False,
            'min_price_bound': 0.01,
            'max_price_bound': 100000.0,
            'allow_negative_prices': False,
            'allow_zero_prices': False,
            'allow_negative_volume': False,
            'allow_zero_volume': True,
            'strict_type_validation': False,
            'enforce_trading_hours': False,
            'enable_symbol_whitelist': False,
            'warn_only_mode': False,
            'warn_only_for_realtime': True
        }
        
        self.update_config_from_dict(default_config)
        logger.info("Reset validation configuration to defaults")


# Convenience functions for creating validators
def create_validator_from_config(config_manager: Optional[ConfigurationManager] = None) -> DataValidator:
    """Create a market data validator from configuration.
    
    Args:
        config_manager: Optional ConfigurationManager instance
        
    Returns:
        DataValidator: Configured market data validator
    """
    factory = ConfigAwareValidatorFactory(config_manager)
    return factory.create_validator_from_config()


def create_realtime_validator(config_manager: Optional[ConfigurationManager] = None) -> DataValidator:
    """Create a validator optimized for real-time data.
    
    Args:
        config_manager: Optional ConfigurationManager instance
        
    Returns:
        DataValidator: Real-time optimized validator
    """
    factory = ConfigAwareValidatorFactory(config_manager)
    return factory.create_realtime_validator()


def create_batch_validator(config_manager: Optional[ConfigurationManager] = None) -> DataValidator:
    """Create a validator optimized for batch processing.
    
    Args:
        config_manager: Optional ConfigurationManager instance
        
    Returns:
        DataValidator: Batch processing optimized validator
    """
    factory = ConfigAwareValidatorFactory(config_manager)
    return factory.create_batch_validator()
