# -*- coding: utf-8 -*-
"""
Production Order Validation Rules

This module defines production-grade order validation rules and constraints
for the trading engine. These rules ensure that orders meet all requirements
before being submitted to brokers in production environments.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
import json
import os
import logging
import re
from datetime import datetime, time, timedelta

# Import from other modules in the trading engine
try:
    from .signal_thresholds import AssetClass, MarketType
    from .holiday_calendar import get_calendar_manager
except ImportError:
    # For standalone usage
    class AssetClass(Enum):
        EQUITY = "equity"
        FUTURES = "futures"
        OPTIONS = "options"
        FOREX = "forex"
        CRYPTO = "crypto"
        FIXED_INCOME = "fixed_income"
        COMMODITY = "commodity"
    
    class MarketType(Enum):
        US = "us"
        EUROPE = "europe"
        ASIA = "asia"
        GLOBAL = "global"


class OrderType(Enum):
    """Types of orders supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order sides supported by the system."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


class TimeInForce(Enum):
    """Time-in-force options supported by the system."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class ValidationSeverity(Enum):
    """Severity levels for validation rules."""
    ERROR = auto()    # Order will be rejected
    WARNING = auto()   # Order will be accepted but with warning
    INFO = auto()     # Informational only


class ValidationCategory(Enum):
    """Categories of validation rules."""
    SYMBOL = "symbol"              # Symbol format and existence
    QUANTITY = "quantity"          # Order quantity constraints
    PRICE = "price"               # Price constraints
    NOTIONAL = "notional"          # Notional value constraints
    TIME = "time"                 # Time-related constraints
    MARKET_HOURS = "market_hours"  # Market hours constraints
    RISK = "risk"                 # Risk-related constraints
    REGULATORY = "regulatory"      # Regulatory constraints
    BROKER = "broker"             # Broker-specific constraints
    CUSTOM = "custom"             # Custom constraints


@dataclass
class ValidationResult:
    """Result of a validation rule check."""
    is_valid: bool
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    rule_id: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """A rule for validating orders."""
    rule_id: str
    description: str
    category: ValidationCategory
    severity: ValidationSeverity
    asset_classes: List[AssetClass] = field(default_factory=list)  # Empty means all
    markets: List[MarketType] = field(default_factory=list)      # Empty means all
    order_types: List[OrderType] = field(default_factory=list)   # Empty means all
    order_sides: List[OrderSide] = field(default_factory=list)   # Empty means all
    validator_func: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation rule to a dictionary.
        
        Returns:
            Dictionary representation of the validation rule
        """
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.name,
            "asset_classes": [ac.value for ac in self.asset_classes],
            "markets": [m.value for m in self.markets],
            "order_types": [ot.value for ot in self.order_types],
            "order_sides": [os.value for os in self.order_sides],
            "parameters": self.parameters,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationRule':
        """Create a validation rule from a dictionary.
        
        Args:
            data: Dictionary containing validation rule data
            
        Returns:
            ValidationRule instance
        """
        return cls(
            rule_id=data["rule_id"],
            description=data["description"],
            category=ValidationCategory(data["category"]),
            severity=ValidationSeverity[data["severity"]],
            asset_classes=[AssetClass(ac) for ac in data.get("asset_classes", [])],
            markets=[MarketType(m) for m in data.get("markets", [])],
            order_types=[OrderType(ot) for ot in data.get("order_types", [])],
            order_sides=[OrderSide(os) for os in data.get("order_sides", [])],
            parameters=data.get("parameters", {}),
            enabled=data.get("enabled", True)
        )


@dataclass
class Order:
    """Representation of an order for validation purposes."""
    symbol: str
    quantity: float
    order_type: OrderType
    order_side: OrderSide
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_date: Optional[datetime] = None
    asset_class: AssetClass = AssetClass.EQUITY
    market: MarketType = MarketType.US
    notional_value: Optional[float] = None
    account_id: Optional[str] = None
    broker_id: Optional[str] = None
    client_order_id: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    tags: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


class OrderValidator:
    """Validator for orders based on production rules.
    
    This class provides methods to validate orders against a set of rules
    before they are submitted to brokers in production environments.
    """
    
    def __init__(self, config_dir: str = None):
        """Initialize the order validator.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "config")
        self.rules: Dict[str, ValidationRule] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load rules
        self._load_rules()
        
        # Register built-in validators
        self._register_built_in_validators()
    
    def _load_rules(self) -> None:
        """Load validation rules from file."""
        file_path = os.path.join(self.config_dir, "validation_rules.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for rule_data in data:
                        rule = ValidationRule.from_dict(rule_data)
                        self.rules[rule.rule_id] = rule
                self.logger.info(f"Loaded {len(self.rules)} validation rules")
            except Exception as e:
                self.logger.error(f"Error loading validation rules: {e}")
                # Initialize with default rules
                self._initialize_default_rules()
        else:
            self.logger.info("Validation rules file not found, initializing defaults")
            self._initialize_default_rules()
    
    def _save_rules(self) -> None:
        """Save validation rules to file."""
        file_path = os.path.join(self.config_dir, "validation_rules.json")
        try:
            with open(file_path, "w") as f:
                json.dump([rule.to_dict() for rule in self.rules.values()], f, indent=2)
            self.logger.info(f"Saved {len(self.rules)} validation rules to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving validation rules: {e}")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        self.rules = {}
        
        # Symbol validation rules
        self._add_rule(
            rule_id="SYMBOL_FORMAT",
            description="Symbol must match the expected format for the asset class",
            category=ValidationCategory.SYMBOL,
            severity=ValidationSeverity.ERROR,
            parameters={
                "equity_pattern": r"^[A-Z]{1,5}$",  # US equities
                "futures_pattern": r"^[A-Z]{2}[FGHJKMNQUVXZ]\d{1,2}$",  # Futures
                "options_pattern": r"^[A-Z]{1,5}\d{6}[CP]\d{8}$",  # Options
                "forex_pattern": r"^[A-Z]{3}/[A-Z]{3}$",  # Forex
                "crypto_pattern": r"^[A-Z]{3,4}/[A-Z]{3,4}$"  # Crypto
            }
        )
        
        # Quantity validation rules
        self._add_rule(
            rule_id="MIN_QUANTITY",
            description="Order quantity must be greater than the minimum allowed",
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.ERROR,
            parameters={
                "min_quantity": {
                    "equity": 1,
                    "futures": 1,
                    "options": 1,
                    "forex": 1000,
                    "crypto": 0.001,
                    "fixed_income": 1000,
                    "commodity": 1
                }
            }
        )
        
        self._add_rule(
            rule_id="MAX_QUANTITY",
            description="Order quantity must be less than the maximum allowed",
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.ERROR,
            parameters={
                "max_quantity": {
                    "equity": 1000000,
                    "futures": 10000,
                    "options": 10000,
                    "forex": 10000000,
                    "crypto": 1000,
                    "fixed_income": 10000000,
                    "commodity": 10000
                }
            }
        )
        
        self._add_rule(
            rule_id="ROUND_LOT",
            description="Order quantity must be in round lots",
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.WARNING,
            asset_classes=[AssetClass.EQUITY],
            parameters={
                "lot_size": 100
            }
        )
        
        # Price validation rules
        self._add_rule(
            rule_id="PRICE_REQUIRED",
            description="Price is required for limit and stop-limit orders",
            category=ValidationCategory.PRICE,
            severity=ValidationSeverity.ERROR,
            order_types=[OrderType.LIMIT, OrderType.STOP_LIMIT]
        )
        
        self._add_rule(
            rule_id="STOP_PRICE_REQUIRED",
            description="Stop price is required for stop and stop-limit orders",
            category=ValidationCategory.PRICE,
            severity=ValidationSeverity.ERROR,
            order_types=[OrderType.STOP, OrderType.STOP_LIMIT]
        )
        
        self._add_rule(
            rule_id="PRICE_PRECISION",
            description="Price must have valid precision for the asset class",
            category=ValidationCategory.PRICE,
            severity=ValidationSeverity.ERROR,
            parameters={
                "max_decimals": {
                    "equity": 2,
                    "futures": 4,
                    "options": 2,
                    "forex": 5,
                    "crypto": 8,
                    "fixed_income": 6,
                    "commodity": 4
                }
            }
        )
        
        self._add_rule(
            rule_id="PRICE_RANGE",
            description="Price must be within a reasonable range",
            category=ValidationCategory.PRICE,
            severity=ValidationSeverity.WARNING,
            parameters={
                "max_price": {
                    "equity": 10000.0,
                    "futures": 1000000.0,
                    "options": 1000.0,
                    "forex": 1000.0,
                    "crypto": 1000000.0,
                    "fixed_income": 10000.0,
                    "commodity": 1000000.0
                }
            }
        )
        
        # Notional value validation rules
        self._add_rule(
            rule_id="MIN_NOTIONAL",
            description="Order notional value must be greater than the minimum allowed",
            category=ValidationCategory.NOTIONAL,
            severity=ValidationSeverity.ERROR,
            parameters={
                "min_notional": {
                    "equity": 100.0,
                    "futures": 1000.0,
                    "options": 10.0,
                    "forex": 1000.0,
                    "crypto": 10.0,
                    "fixed_income": 1000.0,
                    "commodity": 1000.0
                }
            }
        )
        
        self._add_rule(
            rule_id="MAX_NOTIONAL",
            description="Order notional value must be less than the maximum allowed",
            category=ValidationCategory.NOTIONAL,
            severity=ValidationSeverity.ERROR,
            parameters={
                "max_notional": {
                    "equity": 10000000.0,
                    "futures": 50000000.0,
                    "options": 5000000.0,
                    "forex": 50000000.0,
                    "crypto": 5000000.0,
                    "fixed_income": 50000000.0,
                    "commodity": 20000000.0
                }
            }
        )
        
        # Time-related validation rules
        self._add_rule(
            rule_id="EXPIRE_DATE_REQUIRED",
            description="Expire date is required for GTD orders",
            category=ValidationCategory.TIME,
            severity=ValidationSeverity.ERROR,
            parameters={}
        )
        
        self._add_rule(
            rule_id="EXPIRE_DATE_FUTURE",
            description="Expire date must be in the future",
            category=ValidationCategory.TIME,
            severity=ValidationSeverity.ERROR,
            parameters={}
        )
        
        # Market hours validation rules
        self._add_rule(
            rule_id="MARKET_OPEN",
            description="Order can only be submitted when the market is open",
            category=ValidationCategory.MARKET_HOURS,
            severity=ValidationSeverity.WARNING,
            order_types=[OrderType.MARKET],
            parameters={}
        )
        
        # Risk validation rules
        self._add_rule(
            rule_id="MAX_SHARES_OUTSTANDING",
            description="Order quantity must be less than the maximum percentage of shares outstanding",
            category=ValidationCategory.RISK,
            severity=ValidationSeverity.ERROR,
            asset_classes=[AssetClass.EQUITY],
            parameters={
                "max_pct_shares_outstanding": 0.01  # 1%
            }
        )
        
        self._add_rule(
            rule_id="MAX_ADV",
            description="Order quantity must be less than the maximum percentage of average daily volume",
            category=ValidationCategory.RISK,
            severity=ValidationSeverity.ERROR,
            parameters={
                "max_pct_adv": {
                    "equity": 0.1,  # 10% of ADV
                    "futures": 0.2,
                    "options": 0.3,
                    "forex": 0.05,
                    "crypto": 0.2,
                    "fixed_income": 0.1,
                    "commodity": 0.2
                }
            }
        )
        
        # Regulatory validation rules
        self._add_rule(
            rule_id="SHORT_SELL_RESTRICTION",
            description="Short selling is restricted for this symbol",
            category=ValidationCategory.REGULATORY,
            severity=ValidationSeverity.ERROR,
            order_sides=[OrderSide.SELL_SHORT],
            parameters={}
        )
        
        self._add_rule(
            rule_id="MARKET_REGULATION_T",
            description="Order must comply with Regulation T requirements",
            category=ValidationCategory.REGULATORY,
            severity=ValidationSeverity.ERROR,
            markets=[MarketType.US],
            parameters={}
        )
        
        # Broker-specific validation rules
        self._add_rule(
            rule_id="BROKER_ACCOUNT_REQUIRED",
            description="Broker account ID is required",
            category=ValidationCategory.BROKER,
            severity=ValidationSeverity.ERROR,
            parameters={}
        )
        
        self._add_rule(
            rule_id="BROKER_SUPPORTED_ORDER_TYPE",
            description="Order type must be supported by the broker",
            category=ValidationCategory.BROKER,
            severity=ValidationSeverity.ERROR,
            parameters={
                "supported_order_types": {
                    "broker1": ["market", "limit", "stop", "stop_limit"],
                    "broker2": ["market", "limit", "stop", "stop_limit", "trailing_stop"],
                    "broker3": ["market", "limit"]
                }
            }
        )
        
        self.logger.info(f"Initialized {len(self.rules)} default validation rules")
        self._save_rules()
    
    def _add_rule(self, rule_id: str, description: str, category: ValidationCategory,
                 severity: ValidationSeverity, asset_classes: List[AssetClass] = None,
                 markets: List[MarketType] = None, order_types: List[OrderType] = None,
                 order_sides: List[OrderSide] = None, parameters: Dict[str, Any] = None) -> None:
        """Add a validation rule.
        
        Args:
            rule_id: Unique identifier for the rule
            description: Description of the rule
            category: Category of the rule
            severity: Severity level of the rule
            asset_classes: Asset classes to which the rule applies
            markets: Markets to which the rule applies
            order_types: Order types to which the rule applies
            order_sides: Order sides to which the rule applies
            parameters: Parameters for the rule
        """
        rule = ValidationRule(
            rule_id=rule_id,
            description=description,
            category=category,
            severity=severity,
            asset_classes=asset_classes or [],
            markets=markets or [],
            order_types=order_types or [],
            order_sides=order_sides or [],
            parameters=parameters or {}
        )
        self.rules[rule_id] = rule
    
    def _register_built_in_validators(self) -> None:
        """Register built-in validator functions for rules."""
        # Symbol format validation
        self.rules["SYMBOL_FORMAT"].validator_func = self._validate_symbol_format
        
        # Quantity validation
        self.rules["MIN_QUANTITY"].validator_func = self._validate_min_quantity
        self.rules["MAX_QUANTITY"].validator_func = self._validate_max_quantity
        self.rules["ROUND_LOT"].validator_func = self._validate_round_lot
        
        # Price validation
        self.rules["PRICE_REQUIRED"].validator_func = self._validate_price_required
        self.rules["STOP_PRICE_REQUIRED"].validator_func = self._validate_stop_price_required
        self.rules["PRICE_PRECISION"].validator_func = self._validate_price_precision
        self.rules["PRICE_RANGE"].validator_func = self._validate_price_range
        
        # Notional value validation
        self.rules["MIN_NOTIONAL"].validator_func = self._validate_min_notional
        self.rules["MAX_NOTIONAL"].validator_func = self._validate_max_notional
        
        # Time-related validation
        self.rules["EXPIRE_DATE_REQUIRED"].validator_func = self._validate_expire_date_required
        self.rules["EXPIRE_DATE_FUTURE"].validator_func = self._validate_expire_date_future
        
        # Market hours validation
        self.rules["MARKET_OPEN"].validator_func = self._validate_market_open
        
        # Risk validation
        self.rules["MAX_SHARES_OUTSTANDING"].validator_func = self._validate_max_shares_outstanding
        self.rules["MAX_ADV"].validator_func = self._validate_max_adv
        
        # Regulatory validation
        self.rules["SHORT_SELL_RESTRICTION"].validator_func = self._validate_short_sell_restriction
        self.rules["MARKET_REGULATION_T"].validator_func = self._validate_market_regulation_t
        
        # Broker-specific validation
        self.rules["BROKER_ACCOUNT_REQUIRED"].validator_func = self._validate_broker_account_required
        self.rules["BROKER_SUPPORTED_ORDER_TYPE"].validator_func = self._validate_broker_supported_order_type
    
    def validate_order(self, order: Order, context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate an order against all applicable rules.
        
        Args:
            order: Order to validate
            context: Additional context for validation
            
        Returns:
            List of validation results
        """
        results = []
        context = context or {}
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this order
            if not self._rule_applies(rule, order):
                continue
            
            # Apply the rule
            if rule.validator_func:
                result = rule.validator_func(order, rule, context)
                if result:
                    results.append(result)
        
        return results
    
    def _rule_applies(self, rule: ValidationRule, order: Order) -> bool:
        """Check if a rule applies to an order.
        
        Args:
            rule: Validation rule
            order: Order to check
            
        Returns:
            True if the rule applies, False otherwise
        """
        # Check asset class
        if rule.asset_classes and order.asset_class not in rule.asset_classes:
            return False
        
        # Check market
        if rule.markets and order.market not in rule.markets:
            return False
        
        # Check order type
        if rule.order_types and order.order_type not in rule.order_types:
            return False
        
        # Check order side
        if rule.order_sides and order.order_side not in rule.order_sides:
            return False
        
        return True
    
    def has_errors(self, results: List[ValidationResult]) -> bool:
        """Check if validation results contain any errors.
        
        Args:
            results: List of validation results
            
        Returns:
            True if there are any errors, False otherwise
        """
        return any(result.severity == ValidationSeverity.ERROR for result in results)
    
    def get_error_messages(self, results: List[ValidationResult]) -> List[str]:
        """Get error messages from validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            List of error messages
        """
        return [result.message for result in results if result.severity == ValidationSeverity.ERROR]
    
    def get_warning_messages(self, results: List[ValidationResult]) -> List[str]:
        """Get warning messages from validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            List of warning messages
        """
        return [result.message for result in results if result.severity == ValidationSeverity.WARNING]
    
    # Built-in validator functions
    
    def _validate_symbol_format(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate symbol format.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        asset_class_str = order.asset_class.value
        pattern_key = f"{asset_class_str}_pattern"
        
        if pattern_key in rule.parameters:
            pattern = rule.parameters[pattern_key]
            if not re.match(pattern, order.symbol):
                return ValidationResult(
                    is_valid=False,
                    severity=rule.severity,
                    category=rule.category,
                    message=f"Symbol '{order.symbol}' does not match the expected format for {asset_class_str}",
                    rule_id=rule.rule_id,
                    details={
                        "symbol": order.symbol,
                        "expected_pattern": pattern
                    }
                )
        
        return None
    
    def _validate_min_quantity(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate minimum quantity.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        asset_class_str = order.asset_class.value
        min_quantity = rule.parameters.get("min_quantity", {}).get(asset_class_str, 0)
        
        if order.quantity < min_quantity:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order quantity {order.quantity} is less than the minimum allowed {min_quantity} for {asset_class_str}",
                rule_id=rule.rule_id,
                details={
                    "quantity": order.quantity,
                    "min_quantity": min_quantity
                }
            )
        
        return None
    
    def _validate_max_quantity(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate maximum quantity.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        asset_class_str = order.asset_class.value
        max_quantity = rule.parameters.get("max_quantity", {}).get(asset_class_str, float('inf'))
        
        if order.quantity > max_quantity:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order quantity {order.quantity} is greater than the maximum allowed {max_quantity} for {asset_class_str}",
                rule_id=rule.rule_id,
                details={
                    "quantity": order.quantity,
                    "max_quantity": max_quantity
                }
            )
        
        return None
    
    def _validate_round_lot(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate round lot.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        lot_size = rule.parameters.get("lot_size", 100)
        
        if order.quantity % lot_size != 0:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order quantity {order.quantity} is not a multiple of the lot size {lot_size}",
                rule_id=rule.rule_id,
                details={
                    "quantity": order.quantity,
                    "lot_size": lot_size,
                    "suggested_quantity": int(order.quantity / lot_size) * lot_size
                }
            )
        
        return None
    
    def _validate_price_required(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate price required.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.price is None:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Price is required for {order.order_type.value} orders",
                rule_id=rule.rule_id,
                details={}
            )
        
        return None
    
    def _validate_stop_price_required(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate stop price required.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.stop_price is None:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Stop price is required for {order.order_type.value} orders",
                rule_id=rule.rule_id,
                details={}
            )
        
        return None
    
    def _validate_price_precision(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate price precision.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.price is None:
            return None
        
        asset_class_str = order.asset_class.value
        max_decimals = rule.parameters.get("max_decimals", {}).get(asset_class_str, 2)
        
        # Check price precision
        price_str = str(order.price)
        if '.' in price_str:
            decimals = len(price_str.split('.')[1])
            if decimals > max_decimals:
                return ValidationResult(
                    is_valid=False,
                    severity=rule.severity,
                    category=rule.category,
                    message=f"Price {order.price} has {decimals} decimal places, maximum allowed is {max_decimals}",
                    rule_id=rule.rule_id,
                    details={
                        "price": order.price,
                        "decimals": decimals,
                        "max_decimals": max_decimals
                    }
                )
        
        # Check stop price precision if present
        if order.stop_price is not None:
            stop_price_str = str(order.stop_price)
            if '.' in stop_price_str:
                decimals = len(stop_price_str.split('.')[1])
                if decimals > max_decimals:
                    return ValidationResult(
                        is_valid=False,
                        severity=rule.severity,
                        category=rule.category,
                        message=f"Stop price {order.stop_price} has {decimals} decimal places, maximum allowed is {max_decimals}",
                        rule_id=rule.rule_id,
                        details={
                            "stop_price": order.stop_price,
                            "decimals": decimals,
                            "max_decimals": max_decimals
                        }
                    )
        
        return None
    
    def _validate_price_range(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate price range.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.price is None:
            return None
        
        asset_class_str = order.asset_class.value
        max_price = rule.parameters.get("max_price", {}).get(asset_class_str, float('inf'))
        
        if order.price > max_price:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Price {order.price} is greater than the maximum allowed {max_price} for {asset_class_str}",
                rule_id=rule.rule_id,
                details={
                    "price": order.price,
                    "max_price": max_price
                }
            )
        
        return None
    
    def _validate_min_notional(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate minimum notional value.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # Calculate notional value if not provided
        notional = order.notional_value
        if notional is None and order.price is not None:
            notional = order.quantity * order.price
        
        if notional is None:
            # Can't validate without notional value
            return None
        
        asset_class_str = order.asset_class.value
        min_notional = rule.parameters.get("min_notional", {}).get(asset_class_str, 0)
        
        if notional < min_notional:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Notional value {notional} is less than the minimum allowed {min_notional} for {asset_class_str}",
                rule_id=rule.rule_id,
                details={
                    "notional": notional,
                    "min_notional": min_notional
                }
            )
        
        return None
    
    def _validate_max_notional(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate maximum notional value.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # Calculate notional value if not provided
        notional = order.notional_value
        if notional is None and order.price is not None:
            notional = order.quantity * order.price
        
        if notional is None:
            # Can't validate without notional value
            return None
        
        asset_class_str = order.asset_class.value
        max_notional = rule.parameters.get("max_notional", {}).get(asset_class_str, float('inf'))
        
        if notional > max_notional:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Notional value {notional} is greater than the maximum allowed {max_notional} for {asset_class_str}",
                rule_id=rule.rule_id,
                details={
                    "notional": notional,
                    "max_notional": max_notional
                }
            )
        
        return None
    
    def _validate_expire_date_required(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate expire date required.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.time_in_force == TimeInForce.GTD and order.expire_date is None:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message="Expire date is required for GTD orders",
                rule_id=rule.rule_id,
                details={}
            )
        
        return None
    
    def _validate_expire_date_future(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate expire date is in the future.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if order.expire_date is not None and order.expire_date <= datetime.now():
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message="Expire date must be in the future",
                rule_id=rule.rule_id,
                details={
                    "expire_date": order.expire_date,
                    "current_time": datetime.now()
                }
            )
        
        return None
    
    def _validate_market_open(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate market is open.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        try:
            calendar_manager = get_calendar_manager()
            market_open = calendar_manager.is_market_open(order.market.value, order.asset_class.value)
            
            if not market_open:
                return ValidationResult(
                    is_valid=False,
                    severity=rule.severity,
                    category=rule.category,
                    message=f"Market is closed for {order.market.value} {order.asset_class.value}",
                    rule_id=rule.rule_id,
                    details={}
                )
        except Exception as e:
            # If calendar manager is not available, log warning and continue
            logging.warning(f"Could not check market hours: {e}")
        
        return None
    
    def _validate_max_shares_outstanding(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate maximum percentage of shares outstanding.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # This requires market data context
        shares_outstanding = context.get("shares_outstanding", None)
        if shares_outstanding is None:
            return None
        
        max_pct = rule.parameters.get("max_pct_shares_outstanding", 0.01)  # 1%
        max_shares = shares_outstanding * max_pct
        
        if order.quantity > max_shares:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order quantity {order.quantity} is greater than {max_pct*100:.2f}% of shares outstanding ({max_shares:.0f})",
                rule_id=rule.rule_id,
                details={
                    "quantity": order.quantity,
                    "shares_outstanding": shares_outstanding,
                    "max_pct": max_pct,
                    "max_shares": max_shares
                }
            )
        
        return None
    
    def _validate_max_adv(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate maximum percentage of average daily volume.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # This requires market data context
        adv = context.get("average_daily_volume", None)
        if adv is None:
            return None
        
        asset_class_str = order.asset_class.value
        max_pct = rule.parameters.get("max_pct_adv", {}).get(asset_class_str, 0.1)  # 10%
        max_volume = adv * max_pct
        
        if order.quantity > max_volume:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order quantity {order.quantity} is greater than {max_pct*100:.2f}% of ADV ({max_volume:.0f})",
                rule_id=rule.rule_id,
                details={
                    "quantity": order.quantity,
                    "adv": adv,
                    "max_pct": max_pct,
                    "max_volume": max_volume
                }
            )
        
        return None
    
    def _validate_short_sell_restriction(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate short sell restriction.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # This requires market data context
        short_restricted = context.get("short_restricted", False)
        if short_restricted:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Short selling is restricted for {order.symbol}",
                rule_id=rule.rule_id,
                details={}
            )
        
        return None
    
    def _validate_market_regulation_t(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate Regulation T requirements.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        # This is a placeholder for Regulation T validation
        # In a real implementation, this would check margin requirements
        return None
    
    def _validate_broker_account_required(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate broker account is provided.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if not order.account_id:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message="Broker account ID is required",
                rule_id=rule.rule_id,
                details={}
            )
        
        return None
    
    def _validate_broker_supported_order_type(self, order: Order, rule: ValidationRule, context: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate order type is supported by the broker.
        
        Args:
            order: Order to validate
            rule: Validation rule
            context: Additional context
            
        Returns:
            ValidationResult if invalid, None if valid
        """
        if not order.broker_id:
            return None
        
        supported_types = rule.parameters.get("supported_order_types", {}).get(order.broker_id, [])
        if supported_types and order.order_type.value not in supported_types:
            return ValidationResult(
                is_valid=False,
                severity=rule.severity,
                category=rule.category,
                message=f"Order type {order.order_type.value} is not supported by broker {order.broker_id}",
                rule_id=rule.rule_id,
                details={
                    "order_type": order.order_type.value,
                    "broker_id": order.broker_id,
                    "supported_types": supported_types
                }
            )
        
        return None


@dataclass
class BrokerOrderMapping:
    """Mapping between internal order fields and broker-specific fields.
    
    This class defines how internal order fields are mapped to broker-specific
    fields for different brokers in production environments.
    """
    broker_id: str
    asset_class: AssetClass
    market: MarketType
    
    # Field mappings
    field_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Value mappings for specific fields
    value_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Default values for fields not provided
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    # Required fields for this broker
    required_fields: Set[str] = field(default_factory=set)
    
    # Fields to exclude when sending to this broker
    excluded_fields: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the broker order mapping to a dictionary.
        
        Returns:
            Dictionary representation of the broker order mapping
        """
        return {
            "broker_id": self.broker_id,
            "asset_class": self.asset_class.value,
            "market": self.market.value,
            "field_mappings": self.field_mappings,
            "value_mappings": self.value_mappings,
            "default_values": self.default_values,
            "required_fields": list(self.required_fields),
            "excluded_fields": list(self.excluded_fields)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerOrderMapping':
        """Create a broker order mapping from a dictionary.
        
        Args:
            data: Dictionary containing broker order mapping data
            
        Returns:
            BrokerOrderMapping instance
        """
        return cls(
            broker_id=data["broker_id"],
            asset_class=AssetClass(data["asset_class"]),
            market=MarketType(data["market"]),
            field_mappings=data.get("field_mappings", {}),
            value_mappings=data.get("value_mappings", {}),
            default_values=data.get("default_values", {}),
            required_fields=set(data.get("required_fields", [])),
            excluded_fields=set(data.get("excluded_fields", []))
        )


class BrokerMappingManager:
    """Manager for broker-specific order mappings.
    
    This class provides methods to load, save, and retrieve broker-specific
    order mappings for different asset classes and markets.
    """
    
    def __init__(self, config_dir: str = None):
        """Initialize the broker mapping manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "config")
        self.mappings: List[BrokerOrderMapping] = []
        self.logger = logging.getLogger(__name__)
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load mappings
        self._load_mappings()
    
    def _load_mappings(self) -> None:
        """Load broker order mappings from file."""
        file_path = os.path.join(self.config_dir, "broker_mappings.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.mappings = [BrokerOrderMapping.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.mappings)} broker order mappings")
            except Exception as e:
                self.logger.error(f"Error loading broker order mappings: {e}")
                # Initialize with default mappings
                self._initialize_default_mappings()
        else:
            self.logger.info("Broker order mappings file not found, initializing defaults")
            self._initialize_default_mappings()
    
    def _save_mappings(self) -> None:
        """Save broker order mappings to file."""
        file_path = os.path.join(self.config_dir, "broker_mappings.json")
        try:
            with open(file_path, "w") as f:
                json.dump([mapping.to_dict() for mapping in self.mappings], f, indent=2)
            self.logger.info(f"Saved {len(self.mappings)} broker order mappings to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving broker order mappings: {e}")
    
    def _initialize_default_mappings(self) -> None:
        """Initialize default broker order mappings."""
        self.mappings = []
        
        # Add mappings for common brokers
        self._add_interactive_brokers_mappings()
        self._add_alpaca_mappings()
        self._add_td_ameritrade_mappings()
        
        self.logger.info(f"Initialized {len(self.mappings)} default broker order mappings")
        self._save_mappings()
    
    def _add_interactive_brokers_mappings(self) -> None:
        """Add Interactive Brokers mappings."""
        # US Equities
        self.mappings.append(BrokerOrderMapping(
            broker_id="interactive_brokers",
            asset_class=AssetClass.EQUITY,
            market=MarketType.US,
            field_mappings={
                "symbol": "symbol",
                "quantity": "quantity",
                "order_type": "orderType",
                "order_side": "action",
                "price": "lmtPrice",
                "stop_price": "auxPrice",
                "time_in_force": "tif"
            },
            value_mappings={
                "order_type": {
                    "market": "MKT",
                    "limit": "LMT",
                    "stop": "STP",
                    "stop_limit": "STOP_LIMIT",
                    "trailing_stop": "TRAIL"
                },
                "order_side": {
                    "buy": "BUY",
                    "sell": "SELL",
                    "buy_to_cover": "BUY",
                    "sell_short": "SELL"
                },
                "time_in_force": {
                    "day": "DAY",
                    "gtc": "GTC",
                    "ioc": "IOC",
                    "fok": "FOK"
                }
            },
            default_values={
                "account": "",  # Set in production
                "outsideRth": False,
                "transmit": True
            },
            required_fields={"symbol", "quantity", "orderType", "action"},
            excluded_fields={"client_order_id", "custom_params"}
        ))
        
        # US Futures
        self.mappings.append(BrokerOrderMapping(
            broker_id="interactive_brokers",
            asset_class=AssetClass.FUTURES,
            market=MarketType.US,
            field_mappings={
                "symbol": "symbol",
                "quantity": "quantity",
                "order_type": "orderType",
                "order_side": "action",
                "price": "lmtPrice",
                "stop_price": "auxPrice",
                "time_in_force": "tif"
            },
            value_mappings={
                "order_type": {
                    "market": "MKT",
                    "limit": "LMT",
                    "stop": "STP",
                    "stop_limit": "STOP_LIMIT",
                    "trailing_stop": "TRAIL"
                },
                "order_side": {
                    "buy": "BUY",
                    "sell": "SELL",
                    "buy_to_cover": "BUY",
                    "sell_short": "SELL"
                },
                "time_in_force": {
                    "day": "DAY",
                    "gtc": "GTC",
                    "ioc": "IOC",
                    "fok": "FOK"
                }
            },
            default_values={
                "account": "",  # Set in production
                "outsideRth": False,
                "transmit": True,
                "secType": "FUT"
            },
            required_fields={"symbol", "quantity", "orderType", "action", "secType"},
            excluded_fields={"client_order_id", "custom_params"}
        ))
    
    def _add_alpaca_mappings(self) -> None:
        """Add Alpaca mappings."""
        # US Equities
        self.mappings.append(BrokerOrderMapping(
            broker_id="alpaca",
            asset_class=AssetClass.EQUITY,
            market=MarketType.US,
            field_mappings={
                "symbol": "symbol",
                "quantity": "qty",
                "order_type": "type",
                "order_side": "side",
                "price": "limit_price",
                "stop_price": "stop_price",
                "time_in_force": "time_in_force",
                "client_order_id": "client_order_id"
            },
            value_mappings={
                "order_type": {
                    "market": "market",
                    "limit": "limit",
                    "stop": "stop",
                    "stop_limit": "stop_limit"
                },
                "order_side": {
                    "buy": "buy",
                    "sell": "sell",
                    "buy_to_cover": "buy",
                    "sell_short": "sell"
                },
                "time_in_force": {
                    "day": "day",
                    "gtc": "gtc",
                    "ioc": "ioc",
                    "fok": "fok"
                }
            },
            default_values={
                "extended_hours": False
            },
            required_fields={"symbol", "qty", "type", "side", "time_in_force"},
            excluded_fields={"custom_params"}
        ))
    
    def _add_td_ameritrade_mappings(self) -> None:
        """Add TD Ameritrade mappings."""
        # US Equities
        self.mappings.append(BrokerOrderMapping(
            broker_id="td_ameritrade",
            asset_class=AssetClass.EQUITY,
            market=MarketType.US,
            field_mappings={
                "symbol": "symbol",
                "quantity": "quantity",
                "order_type": "orderType",
                "order_side": "instruction",
                "price": "price",
                "stop_price": "stopPrice",
                "time_in_force": "duration"
            },
            value_mappings={
                "order_type": {
                    "market": "MARKET",
                    "limit": "LIMIT",
                    "stop": "STOP",
                    "stop_limit": "STOP_LIMIT"
                },
                "order_side": {
                    "buy": "BUY",
                    "sell": "SELL",
                    "buy_to_cover": "BUY_TO_COVER",
                    "sell_short": "SELL_SHORT"
                },
                "time_in_force": {
                    "day": "DAY",
                    "gtc": "GOOD_TILL_CANCEL",
                    "fok": "FILL_OR_KILL"
                }
            },
            default_values={
                "session": "NORMAL",
                "orderStrategyType": "SINGLE"
            },
            required_fields={"symbol", "quantity", "orderType", "instruction", "duration"},
            excluded_fields={"custom_params"}
        ))
    
    def get_mapping(self, broker_id: str, asset_class: AssetClass, market: MarketType) -> Optional[BrokerOrderMapping]:
        """Get a broker order mapping.
        
        Args:
            broker_id: Broker ID
            asset_class: Asset class
            market: Market
            
        Returns:
            BrokerOrderMapping if found, None otherwise
        """
        for mapping in self.mappings:
            if (mapping.broker_id == broker_id and 
                mapping.asset_class == asset_class and 
                mapping.market == market):
                return mapping
        return None
    
    def add_mapping(self, mapping: BrokerOrderMapping) -> None:
        """Add a broker order mapping.
        
        Args:
            mapping: Broker order mapping to add
        """
        # Remove existing mapping if any
        self.remove_mapping(mapping.broker_id, mapping.asset_class, mapping.market)
        
        # Add new mapping
        self.mappings.append(mapping)
        self._save_mappings()
    
    def remove_mapping(self, broker_id: str, asset_class: AssetClass, market: MarketType) -> bool:
        """Remove a broker order mapping.
        
        Args:
            broker_id: Broker ID
            asset_class: Asset class
            market: Market
            
        Returns:
            True if removed, False if not found
        """
        for i, mapping in enumerate(self.mappings):
            if (mapping.broker_id == broker_id and 
                mapping.asset_class == asset_class and 
                mapping.market == market):
                del self.mappings[i]
                self._save_mappings()
                return True
        return False
    
    def map_order_to_broker_format(self, order: Order, broker_id: str) -> Dict[str, Any]:
        """Map an order to broker-specific format.
        
        Args:
            order: Order to map
            broker_id: Broker ID
            
        Returns:
            Dictionary containing broker-specific order fields
        """
        mapping = self.get_mapping(broker_id, order.asset_class, order.market)
        if not mapping:
            raise ValueError(f"No mapping found for broker {broker_id}, asset class {order.asset_class.value}, market {order.market.value}")
        
        # Create broker order
        broker_order = {}
        
        # Apply field mappings
        for internal_field, broker_field in mapping.field_mappings.items():
            if hasattr(order, internal_field) and getattr(order, internal_field) is not None:
                value = getattr(order, internal_field)
                
                # Apply value mappings if available
                if internal_field in mapping.value_mappings and isinstance(value, Enum):
                    value_map = mapping.value_mappings[internal_field]
                    if value.value in value_map:
                        value = value_map[value.value]
                    else:
                        # Use the enum value directly if no mapping
                        value = value.value
                elif isinstance(value, Enum):
                    # Convert enum to string
                    value = value.value
                
                broker_order[broker_field] = value
        
        # Apply default values
        for field, value in mapping.default_values.items():
            if field not in broker_order:
                broker_order[field] = value
        
        # Add custom parameters
        if hasattr(order, "custom_params") and order.custom_params:
            for field, value in order.custom_params.items():
                if field not in mapping.excluded_fields:
                    broker_order[field] = value
        
        # Check required fields
        missing_fields = mapping.required_fields - set(broker_order.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields for broker {broker_id}: {missing_fields}")
        
        return broker_order


def create_order_validator(config_dir: str = None) -> OrderValidator:
    """Create an order validator.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        OrderValidator instance
    """
    return OrderValidator(config_dir)


def create_broker_mapping_manager(config_dir: str = None) -> BrokerMappingManager:
    """Create a broker mapping manager.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        BrokerMappingManager instance
    """
    return BrokerMappingManager(config_dir)


def validate_order_for_production(order: Order, context: Dict[str, Any] = None) -> List[ValidationResult]:
    """Validate an order for production trading.
    
    This is a convenience function that creates an order validator and validates
    the order against all applicable rules.
    
    Args:
        order: Order to validate
        context: Additional context for validation
        
    Returns:
        List of validation results
    """
    validator = create_order_validator()
    return validator.validate_order(order, context)


def map_order_to_broker(order: Order, broker_id: str, config_dir: str = None) -> Dict[str, Any]:
    """Map an order to broker-specific format.
    
    This is a convenience function that creates a broker mapping manager and maps
    the order to broker-specific format.
    
    Args:
        order: Order to map
        broker_id: Broker ID
        config_dir: Directory containing configuration files
        
    Returns:
        Dictionary containing broker-specific order fields
    """
    manager = create_broker_mapping_manager(config_dir)
    return manager.map_order_to_broker_format(order, broker_id)