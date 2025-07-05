# -*- coding: utf-8 -*-
"""
Production Signal Thresholds and Filtering Parameters

This module defines production-ready signal thresholds, filtering parameters,
and validation rules for the trading engine. These configurations help ensure
that only high-quality signals are processed and executed in production.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import json
import os
import logging


class SignalType(Enum):
    """Types of trading signals supported by the system."""
    PRICE_BREAKOUT = "price_breakout"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_REGIME = "volatility_regime"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"


class AssetClass(Enum):
    """Asset classes supported by the system."""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"


class MarketType(Enum):
    """Market types for different regions."""
    US = "us"
    EUROPE = "europe"
    ASIA = "asia"
    GLOBAL = "global"


@dataclass
class SignalThreshold:
    """Configuration for signal thresholds and filtering parameters.
    
    This class defines the minimum requirements for a signal to be considered
    valid for trading in production environments.
    """
    # Signal identification
    signal_type: SignalType
    asset_class: AssetClass
    market: MarketType
    
    # Strength thresholds
    min_strength: float = 0.6  # Minimum signal strength (0.0-1.0)
    min_confidence: float = 0.7  # Minimum confidence level (0.0-1.0)
    
    # Time validity
    min_duration_seconds: int = 60  # Minimum signal duration in seconds
    max_duration_seconds: int = 86400  # Maximum signal duration (24 hours)
    
    # Performance requirements
    min_expected_return: float = 0.001  # Minimum expected return (0.1%)
    min_sharpe_ratio: float = 0.5  # Minimum Sharpe ratio
    max_drawdown: float = 0.02  # Maximum acceptable drawdown (2%)
    
    # Filtering parameters
    min_volume: Dict[str, int] = field(default_factory=lambda: {
        "equity": 100000,
        "futures": 5000,
        "options": 1000,
        "forex": 1000000,
        "crypto": 50000,
        "fixed_income": 1000000,
        "commodity": 5000
    })
    
    max_spread_bps: Dict[str, int] = field(default_factory=lambda: {
        "equity": 20,  # 20 basis points
        "futures": 5,
        "options": 50,
        "forex": 3,
        "crypto": 30,
        "fixed_income": 10,
        "commodity": 15
    })
    
    # Volatility constraints
    max_volatility: Dict[str, float] = field(default_factory=lambda: {
        "equity": 0.03,  # 3% daily volatility
        "futures": 0.025,
        "options": 0.05,
        "forex": 0.015,
        "crypto": 0.06,
        "fixed_income": 0.01,
        "commodity": 0.03
    })
    
    # Custom parameters for specific signal types
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the threshold configuration to a dictionary.
        
        Returns:
            Dictionary representation of the threshold configuration
        """
        return {
            "signal_type": self.signal_type.value,
            "asset_class": self.asset_class.value,
            "market": self.market.value,
            "min_strength": self.min_strength,
            "min_confidence": self.min_confidence,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "min_expected_return": self.min_expected_return,
            "min_sharpe_ratio": self.min_sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "min_volume": self.min_volume,
            "max_spread_bps": self.max_spread_bps,
            "max_volatility": self.max_volatility,
            "custom_parameters": self.custom_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalThreshold':
        """Create a threshold configuration from a dictionary.
        
        Args:
            data: Dictionary containing threshold configuration
            
        Returns:
            SignalThreshold instance
        """
        return cls(
            signal_type=SignalType(data["signal_type"]),
            asset_class=AssetClass(data["asset_class"]),
            market=MarketType(data["market"]),
            min_strength=data.get("min_strength", 0.6),
            min_confidence=data.get("min_confidence", 0.7),
            min_duration_seconds=data.get("min_duration_seconds", 60),
            max_duration_seconds=data.get("max_duration_seconds", 86400),
            min_expected_return=data.get("min_expected_return", 0.001),
            min_sharpe_ratio=data.get("min_sharpe_ratio", 0.5),
            max_drawdown=data.get("max_drawdown", 0.02),
            min_volume=data.get("min_volume", cls.__annotations__["min_volume"].default_factory()),
            max_spread_bps=data.get("max_spread_bps", cls.__annotations__["max_spread_bps"].default_factory()),
            max_volatility=data.get("max_volatility", cls.__annotations__["max_volatility"].default_factory()),
            custom_parameters=data.get("custom_parameters", {})
        )


@dataclass
class OrderSizeLimits:
    """Configuration for production order size limits.
    
    This class defines the maximum order sizes allowed for different
    asset classes and markets in production environments.
    """
    asset_class: AssetClass
    market: MarketType
    
    # Maximum order size as percentage of average daily volume
    max_pct_of_adv: Dict[str, float] = field(default_factory=lambda: {
        "equity": 0.01,  # 1% of ADV
        "futures": 0.02,
        "options": 0.05,
        "forex": 0.005,
        "crypto": 0.03,
        "fixed_income": 0.01,
        "commodity": 0.02
    })
    
    # Maximum order size in absolute units
    max_order_size: Dict[str, int] = field(default_factory=lambda: {
        "equity": 100000,  # shares
        "futures": 500,  # contracts
        "options": 1000,  # contracts
        "forex": 10000000,  # currency units
        "crypto": 100,  # coins/tokens
        "fixed_income": 10000000,  # face value
        "commodity": 500  # contracts
    })
    
    # Maximum notional value in USD
    max_notional_usd: Dict[str, int] = field(default_factory=lambda: {
        "equity": 5000000,  # $5M
        "futures": 10000000,  # $10M
        "options": 2000000,  # $2M
        "forex": 20000000,  # $20M
        "crypto": 1000000,  # $1M
        "fixed_income": 20000000,  # $20M
        "commodity": 5000000  # $5M
    })
    
    # Maximum percentage of position size for liquidation
    max_liquidation_pct: float = 0.25  # 25% of position per order
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the order size limits to a dictionary.
        
        Returns:
            Dictionary representation of the order size limits
        """
        return {
            "asset_class": self.asset_class.value,
            "market": self.market.value,
            "max_pct_of_adv": self.max_pct_of_adv,
            "max_order_size": self.max_order_size,
            "max_notional_usd": self.max_notional_usd,
            "max_liquidation_pct": self.max_liquidation_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderSizeLimits':
        """Create order size limits from a dictionary.
        
        Args:
            data: Dictionary containing order size limits
            
        Returns:
            OrderSizeLimits instance
        """
        return cls(
            asset_class=AssetClass(data["asset_class"]),
            market=MarketType(data["market"]),
            max_pct_of_adv=data.get("max_pct_of_adv", cls.__annotations__["max_pct_of_adv"].default_factory()),
            max_order_size=data.get("max_order_size", cls.__annotations__["max_order_size"].default_factory()),
            max_notional_usd=data.get("max_notional_usd", cls.__annotations__["max_notional_usd"].default_factory()),
            max_liquidation_pct=data.get("max_liquidation_pct", 0.25)
        )


@dataclass
class ExecutionRateControls:
    """Configuration for production execution rate controls.
    
    This class defines the maximum execution rates allowed for different
    asset classes and markets in production environments.
    """
    asset_class: AssetClass
    market: MarketType
    
    # Maximum orders per second
    max_orders_per_second: Dict[str, float] = field(default_factory=lambda: {
        "equity": 5.0,
        "futures": 10.0,
        "options": 3.0,
        "forex": 5.0,
        "crypto": 3.0,
        "fixed_income": 2.0,
        "commodity": 5.0
    })
    
    # Maximum orders per minute
    max_orders_per_minute: Dict[str, int] = field(default_factory=lambda: {
        "equity": 100,
        "futures": 200,
        "options": 50,
        "forex": 100,
        "crypto": 50,
        "fixed_income": 30,
        "commodity": 100
    })
    
    # Maximum notional value per minute in USD
    max_notional_per_minute_usd: Dict[str, int] = field(default_factory=lambda: {
        "equity": 2000000,  # $2M
        "futures": 5000000,  # $5M
        "options": 1000000,  # $1M
        "forex": 10000000,  # $10M
        "crypto": 500000,  # $500K
        "fixed_income": 5000000,  # $5M
        "commodity": 2000000  # $2M
    })
    
    # Cooldown period after hitting rate limits (seconds)
    rate_limit_cooldown_seconds: int = 300  # 5 minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution rate controls to a dictionary.
        
        Returns:
            Dictionary representation of the execution rate controls
        """
        return {
            "asset_class": self.asset_class.value,
            "market": self.market.value,
            "max_orders_per_second": self.max_orders_per_second,
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_notional_per_minute_usd": self.max_notional_per_minute_usd,
            "rate_limit_cooldown_seconds": self.rate_limit_cooldown_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionRateControls':
        """Create execution rate controls from a dictionary.
        
        Args:
            data: Dictionary containing execution rate controls
            
        Returns:
            ExecutionRateControls instance
        """
        return cls(
            asset_class=AssetClass(data["asset_class"]),
            market=MarketType(data["market"]),
            max_orders_per_second=data.get("max_orders_per_second", cls.__annotations__["max_orders_per_second"].default_factory()),
            max_orders_per_minute=data.get("max_orders_per_minute", cls.__annotations__["max_orders_per_minute"].default_factory()),
            max_notional_per_minute_usd=data.get("max_notional_per_minute_usd", cls.__annotations__["max_notional_per_minute_usd"].default_factory()),
            rate_limit_cooldown_seconds=data.get("rate_limit_cooldown_seconds", 300)
        )


class SignalThresholdManager:
    """Manager for signal thresholds and filtering parameters.
    
    This class provides methods to load, save, and retrieve signal thresholds,
    order size limits, and execution rate controls for different asset classes,
    markets, and signal types.
    """
    
    def __init__(self, config_dir: str = None):
        """Initialize the signal threshold manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "config")
        self.signal_thresholds: List[SignalThreshold] = []
        self.order_size_limits: List[OrderSizeLimits] = []
        self.execution_rate_controls: List[ExecutionRateControls] = []
        self.logger = logging.getLogger(__name__)
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load all configurations from files."""
        self._load_signal_thresholds()
        self._load_order_size_limits()
        self._load_execution_rate_controls()
    
    def _load_signal_thresholds(self) -> None:
        """Load signal thresholds from file."""
        file_path = os.path.join(self.config_dir, "signal_thresholds.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.signal_thresholds = [SignalThreshold.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.signal_thresholds)} signal thresholds")
            except Exception as e:
                self.logger.error(f"Error loading signal thresholds: {e}")
                # Initialize with default thresholds
                self._initialize_default_signal_thresholds()
        else:
            self.logger.info("Signal thresholds file not found, initializing defaults")
            self._initialize_default_signal_thresholds()
    
    def _load_order_size_limits(self) -> None:
        """Load order size limits from file."""
        file_path = os.path.join(self.config_dir, "order_size_limits.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.order_size_limits = [OrderSizeLimits.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.order_size_limits)} order size limits")
            except Exception as e:
                self.logger.error(f"Error loading order size limits: {e}")
                # Initialize with default limits
                self._initialize_default_order_size_limits()
        else:
            self.logger.info("Order size limits file not found, initializing defaults")
            self._initialize_default_order_size_limits()
    
    def _load_execution_rate_controls(self) -> None:
        """Load execution rate controls from file."""
        file_path = os.path.join(self.config_dir, "execution_rate_controls.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.execution_rate_controls = [ExecutionRateControls.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.execution_rate_controls)} execution rate controls")
            except Exception as e:
                self.logger.error(f"Error loading execution rate controls: {e}")
                # Initialize with default controls
                self._initialize_default_execution_rate_controls()
        else:
            self.logger.info("Execution rate controls file not found, initializing defaults")
            self._initialize_default_execution_rate_controls()
    
    def _initialize_default_signal_thresholds(self) -> None:
        """Initialize default signal thresholds for all combinations."""
        self.signal_thresholds = []
        
        # Create default thresholds for common combinations
        for signal_type in SignalType:
            for asset_class in AssetClass:
                for market in MarketType:
                    # Skip some uncommon combinations
                    if (signal_type == SignalType.FUNDAMENTAL and 
                        asset_class in [AssetClass.FOREX, AssetClass.CRYPTO]):
                        continue
                    
                    # Create default threshold
                    threshold = SignalThreshold(
                        signal_type=signal_type,
                        asset_class=asset_class,
                        market=market
                    )
                    
                    # Adjust parameters for specific combinations
                    if signal_type == SignalType.MOMENTUM:
                        threshold.min_strength = 0.7
                        threshold.min_duration_seconds = 300  # 5 minutes
                    elif signal_type == SignalType.MEAN_REVERSION:
                        threshold.min_confidence = 0.8
                        threshold.max_drawdown = 0.015  # 1.5%
                    elif signal_type == SignalType.MACHINE_LEARNING:
                        threshold.min_confidence = 0.85
                        threshold.min_sharpe_ratio = 0.8
                    
                    # Add custom parameters for specific signal types
                    if signal_type == SignalType.PRICE_BREAKOUT:
                        threshold.custom_parameters = {
                            "min_breakout_strength": 2.0,  # 2 standard deviations
                            "confirmation_period_seconds": 120  # 2 minutes
                        }
                    elif signal_type == SignalType.VOLUME_SPIKE:
                        threshold.custom_parameters = {
                            "min_volume_multiple": 3.0,  # 3x average volume
                            "lookback_periods": 20
                        }
                    
                    self.signal_thresholds.append(threshold)
        
        self.logger.info(f"Initialized {len(self.signal_thresholds)} default signal thresholds")
        self._save_signal_thresholds()
    
    def _initialize_default_order_size_limits(self) -> None:
        """Initialize default order size limits for all combinations."""
        self.order_size_limits = []
        
        # Create default limits for all asset classes and markets
        for asset_class in AssetClass:
            for market in MarketType:
                limits = OrderSizeLimits(
                    asset_class=asset_class,
                    market=market
                )
                
                # Adjust limits for specific markets
                if market == MarketType.ASIA:
                    # More conservative limits for Asian markets
                    for key in limits.max_pct_of_adv:
                        limits.max_pct_of_adv[key] *= 0.8
                    for key in limits.max_notional_usd:
                        limits.max_notional_usd[key] = int(limits.max_notional_usd[key] * 0.8)
                
                self.order_size_limits.append(limits)
        
        self.logger.info(f"Initialized {len(self.order_size_limits)} default order size limits")
        self._save_order_size_limits()
    
    def _initialize_default_execution_rate_controls(self) -> None:
        """Initialize default execution rate controls for all combinations."""
        self.execution_rate_controls = []
        
        # Create default controls for all asset classes and markets
        for asset_class in AssetClass:
            for market in MarketType:
                controls = ExecutionRateControls(
                    asset_class=asset_class,
                    market=market
                )
                
                # Adjust controls for specific markets
                if market == MarketType.US:
                    # Higher capacity for US markets
                    for key in controls.max_orders_per_second:
                        controls.max_orders_per_second[key] *= 1.2
                    for key in controls.max_orders_per_minute:
                        controls.max_orders_per_minute[key] = int(controls.max_orders_per_minute[key] * 1.2)
                
                self.execution_rate_controls.append(controls)
        
        self.logger.info(f"Initialized {len(self.execution_rate_controls)} default execution rate controls")
        self._save_execution_rate_controls()
    
    def _save_signal_thresholds(self) -> None:
        """Save signal thresholds to file."""
        file_path = os.path.join(self.config_dir, "signal_thresholds.json")
        try:
            with open(file_path, "w") as f:
                json.dump([threshold.to_dict() for threshold in self.signal_thresholds], f, indent=2)
            self.logger.info(f"Saved {len(self.signal_thresholds)} signal thresholds to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving signal thresholds: {e}")
    
    def _save_order_size_limits(self) -> None:
        """Save order size limits to file."""
        file_path = os.path.join(self.config_dir, "order_size_limits.json")
        try:
            with open(file_path, "w") as f:
                json.dump([limits.to_dict() for limits in self.order_size_limits], f, indent=2)
            self.logger.info(f"Saved {len(self.order_size_limits)} order size limits to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving order size limits: {e}")
    
    def _save_execution_rate_controls(self) -> None:
        """Save execution rate controls to file."""
        file_path = os.path.join(self.config_dir, "execution_rate_controls.json")
        try:
            with open(file_path, "w") as f:
                json.dump([controls.to_dict() for controls in self.execution_rate_controls], f, indent=2)
            self.logger.info(f"Saved {len(self.execution_rate_controls)} execution rate controls to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving execution rate controls: {e}")
    
    def get_signal_threshold(self, signal_type: Union[SignalType, str], 
                            asset_class: Union[AssetClass, str],
                            market: Union[MarketType, str]) -> Optional[SignalThreshold]:
        """Get signal threshold for the specified parameters.
        
        Args:
            signal_type: Signal type
            asset_class: Asset class
            market: Market type
            
        Returns:
            SignalThreshold if found, None otherwise
        """
        # Convert string parameters to enum values if needed
        if isinstance(signal_type, str):
            signal_type = SignalType(signal_type)
        if isinstance(asset_class, str):
            asset_class = AssetClass(asset_class)
        if isinstance(market, str):
            market = MarketType(market)
        
        # Find matching threshold
        for threshold in self.signal_thresholds:
            if (threshold.signal_type == signal_type and
                threshold.asset_class == asset_class and
                threshold.market == market):
                return threshold
        
        # Try to find a global market threshold if specific market not found
        for threshold in self.signal_thresholds:
            if (threshold.signal_type == signal_type and
                threshold.asset_class == asset_class and
                threshold.market == MarketType.GLOBAL):
                return threshold
        
        return None
    
    def get_order_size_limits(self, asset_class: Union[AssetClass, str],
                             market: Union[MarketType, str]) -> Optional[OrderSizeLimits]:
        """Get order size limits for the specified parameters.
        
        Args:
            asset_class: Asset class
            market: Market type
            
        Returns:
            OrderSizeLimits if found, None otherwise
        """
        # Convert string parameters to enum values if needed
        if isinstance(asset_class, str):
            asset_class = AssetClass(asset_class)
        if isinstance(market, str):
            market = MarketType(market)
        
        # Find matching limits
        for limits in self.order_size_limits:
            if (limits.asset_class == asset_class and
                limits.market == market):
                return limits
        
        # Try to find global market limits if specific market not found
        for limits in self.order_size_limits:
            if (limits.asset_class == asset_class and
                limits.market == MarketType.GLOBAL):
                return limits
        
        return None
    
    def get_execution_rate_controls(self, asset_class: Union[AssetClass, str],
                                  market: Union[MarketType, str]) -> Optional[ExecutionRateControls]:
        """Get execution rate controls for the specified parameters.
        
        Args:
            asset_class: Asset class
            market: Market type
            
        Returns:
            ExecutionRateControls if found, None otherwise
        """
        # Convert string parameters to enum values if needed
        if isinstance(asset_class, str):
            asset_class = AssetClass(asset_class)
        if isinstance(market, str):
            market = MarketType(market)
        
        # Find matching controls
        for controls in self.execution_rate_controls:
            if (controls.asset_class == asset_class and
                controls.market == market):
                return controls
        
        # Try to find global market controls if specific market not found
        for controls in self.execution_rate_controls:
            if (controls.asset_class == asset_class and
                controls.market == MarketType.GLOBAL):
                return controls
        
        return None


# Singleton instance
_signal_threshold_manager = None


def get_signal_threshold_manager(config_dir: str = None) -> SignalThresholdManager:
    """Get the singleton instance of the signal threshold manager.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        SignalThresholdManager instance
    """
    global _signal_threshold_manager
    if _signal_threshold_manager is None:
        _signal_threshold_manager = SignalThresholdManager(config_dir)
    return _signal_threshold_manager


# Production-ready signal validation function
def validate_signal(signal_type: Union[SignalType, str],
                   asset_class: Union[AssetClass, str],
                   market: Union[MarketType, str],
                   strength: float,
                   confidence: float,
                   expected_return: float,
                   volume: float,
                   spread_bps: float,
                   volatility: float,
                   duration_seconds: int,
                   additional_params: Dict[str, Any] = None) -> bool:
    """Validate a signal against production thresholds.
    
    Args:
        signal_type: Type of signal
        asset_class: Asset class
        market: Market type
        strength: Signal strength (0.0-1.0)
        confidence: Confidence level (0.0-1.0)
        expected_return: Expected return
        volume: Current volume
        spread_bps: Current spread in basis points
        volatility: Current volatility
        duration_seconds: Signal duration in seconds
        additional_params: Additional parameters for custom validation
        
    Returns:
        True if signal passes validation, False otherwise
    """
    # Get threshold configuration
    manager = get_signal_threshold_manager()
    threshold = manager.get_signal_threshold(signal_type, asset_class, market)
    
    if threshold is None:
        logging.warning(f"No threshold found for {signal_type}/{asset_class}/{market}")
        return False
    
    # Basic validation
    if strength < threshold.min_strength:
        logging.info(f"Signal rejected: strength {strength} < min {threshold.min_strength}")
        return False
    
    if confidence < threshold.min_confidence:
        logging.info(f"Signal rejected: confidence {confidence} < min {threshold.min_confidence}")
        return False
    
    if expected_return < threshold.min_expected_return:
        logging.info(f"Signal rejected: expected return {expected_return} < min {threshold.min_expected_return}")
        return False
    
    if duration_seconds < threshold.min_duration_seconds:
        logging.info(f"Signal rejected: duration {duration_seconds}s < min {threshold.min_duration_seconds}s")
        return False
    
    if duration_seconds > threshold.max_duration_seconds:
        logging.info(f"Signal rejected: duration {duration_seconds}s > max {threshold.max_duration_seconds}s")
        return False
    
    # Asset-specific validation
    asset_class_str = asset_class.value if isinstance(asset_class, AssetClass) else asset_class
    
    if volume < threshold.min_volume.get(asset_class_str, 0):
        logging.info(f"Signal rejected: volume {volume} < min {threshold.min_volume.get(asset_class_str, 0)}")
        return False
    
    if spread_bps > threshold.max_spread_bps.get(asset_class_str, float('inf')):
        logging.info(f"Signal rejected: spread {spread_bps} bps > max {threshold.max_spread_bps.get(asset_class_str, float('inf'))} bps")
        return False
    
    if volatility > threshold.max_volatility.get(asset_class_str, float('inf')):
        logging.info(f"Signal rejected: volatility {volatility} > max {threshold.max_volatility.get(asset_class_str, float('inf'))}")
        return False
    
    # Custom parameter validation
    if additional_params and threshold.custom_parameters:
        for key, min_value in threshold.custom_parameters.items():
            if key in additional_params and additional_params[key] < min_value:
                logging.info(f"Signal rejected: {key} {additional_params[key]} < min {min_value}")
                return False
    
    return True