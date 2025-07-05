"""Order and Signal Validation for Production Trading Engine.

This module provides comprehensive validation rules and utilities for orders
and signals in a production trading environment, ensuring that all trades
meet regulatory, risk, and business requirements before execution.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
import re
from datetime import datetime, time
import pytz

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.utils import (
    calculate_order_value,
    is_market_open,
    get_time_to_market_close
)

# Configure logger
logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation rules."""
    INFO = "INFO"  # Informational only, doesn't block execution
    WARNING = "WARNING"  # Warning, but allows execution with acknowledgment
    ERROR = "ERROR"  # Error, blocks execution unless explicitly overridden
    CRITICAL = "CRITICAL"  # Critical error, always blocks execution


class ValidationResult:
    """Result of a validation check."""
    def __init__(
        self,
        is_valid: bool,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        validation_id: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a validation result.
        
        Args:
            is_valid: Whether the validation passed
            message: Validation message
            severity: Severity level of the validation
            validation_id: Unique identifier for the validation rule
            metadata: Additional metadata about the validation
        """
        self.is_valid = is_valid
        self.message = message
        self.severity = severity
        self.validation_id = validation_id
        self.metadata = metadata or {}
        self.timestamp = datetime.now(pytz.UTC)
    
    def __bool__(self) -> bool:
        """Boolean representation of validation result."""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "severity": self.severity.value,
            "validation_id": self.validation_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ValidationRule:
    """A validation rule for orders or signals."""
    def __init__(
        self,
        rule_id: str,
        description: str,
        validation_fn: Callable[[Dict[str, Any], Dict[str, Any]], ValidationResult],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        applies_to: List[str] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a validation rule.
        
        Args:
            rule_id: Unique identifier for the rule
            description: Description of what the rule validates
            validation_fn: Function that performs the validation
            severity: Severity level if validation fails
            applies_to: List of order/signal types this rule applies to
            enabled: Whether this rule is enabled
            metadata: Additional metadata about the rule
        """
        self.rule_id = rule_id
        self.description = description
        self.validation_fn = validation_fn
        self.severity = severity
        self.applies_to = applies_to or ["all"]
        self.enabled = enabled
        self.metadata = metadata or {}
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Apply the validation rule to the data.
        
        Args:
            data: The order or signal data to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult: The result of the validation
        """
        if not self.enabled:
            return ValidationResult(
                True,
                f"Rule {self.rule_id} is disabled",
                ValidationSeverity.INFO,
                self.rule_id
            )
        
        # Check if rule applies to this type
        data_type = data.get("type", "unknown")
        if "all" not in self.applies_to and data_type not in self.applies_to:
            return ValidationResult(
                True,
                f"Rule {self.rule_id} does not apply to {data_type}",
                ValidationSeverity.INFO,
                self.rule_id
            )
        
        # Apply the validation function
        try:
            result = self.validation_fn(data, context)
            result.validation_id = self.rule_id
            return result
        except Exception as e:
            logger.error(f"Error applying validation rule {self.rule_id}: {str(e)}")
            return ValidationResult(
                False,
                f"Validation error: {str(e)}",
                self.severity,
                self.rule_id,
                {"exception": str(e)}
            )


class Validator:
    """Validator for orders and signals."""
    def __init__(self, rules: List[ValidationRule] = None):
        """Initialize a validator with rules.
        
        Args:
            rules: List of validation rules
        """
        self.rules = rules or []
        self.rule_map = {rule.rule_id: rule for rule in self.rules}
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.
        
        Args:
            rule: The validation rule to add
        """
        self.rules.append(rule)
        self.rule_map[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule.
        
        Args:
            rule_id: ID of the rule to remove
            
        Returns:
            bool: True if rule was removed, False if not found
        """
        if rule_id in self.rule_map:
            rule = self.rule_map[rule_id]
            self.rules.remove(rule)
            del self.rule_map[rule_id]
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule.
        
        Args:
            rule_id: ID of the rule to enable
            
        Returns:
            bool: True if rule was enabled, False if not found
        """
        if rule_id in self.rule_map:
            self.rule_map[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule.
        
        Args:
            rule_id: ID of the rule to disable
            
        Returns:
            bool: True if rule was disabled, False if not found
        """
        if rule_id in self.rule_map:
            self.rule_map[rule_id].enabled = False
            return True
        return False
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate data against all rules.
        
        Args:
            data: The order or signal data to validate
            context: Additional context for validation
            
        Returns:
            List[ValidationResult]: Results of all validations
        """
        context = context or {}
        results = []
        
        for rule in self.rules:
            result = rule.validate(data, context)
            results.append(result)
        
        return results
    
    def is_valid(self, data: Dict[str, Any], context: Dict[str, Any] = None, 
                 ignore_warnings: bool = False) -> Tuple[bool, List[ValidationResult]]:
        """Check if data is valid according to all rules.
        
        Args:
            data: The order or signal data to validate
            context: Additional context for validation
            ignore_warnings: Whether to ignore warning severity validations
            
        Returns:
            Tuple[bool, List[ValidationResult]]: (is_valid, validation_results)
        """
        results = self.validate(data, context)
        
        # Check if any validations failed
        for result in results:
            if not result.is_valid:
                if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    return False, results
                if result.severity == ValidationSeverity.WARNING and not ignore_warnings:
                    return False, results
        
        return True, results


# ===== Order Validation Rules =====

def create_order_validators() -> Validator:
    """Create a validator with standard order validation rules.
    
    Returns:
        Validator: Validator with order validation rules
    """
    rules = [
        # Symbol validation
        ValidationRule(
            "ORD-SYM-001",
            "Symbol must be provided and in valid format",
            lambda order, ctx: ValidationResult(
                bool(order.get("symbol")) and re.match(r'^[A-Z0-9.]{1,20}$', order.get("symbol", "")),
                "Symbol must be provided and in valid format (letters, numbers, dots only, max 20 chars)"
            )
        ),
        
        # Order type validation
        ValidationRule(
            "ORD-TYPE-001",
            "Order type must be valid",
            lambda order, ctx: ValidationResult(
                order.get("order_type") in ["market", "limit", "stop", "stop_limit"],
                f"Invalid order type: {order.get('order_type')}. Must be one of: market, limit, stop, stop_limit"
            )
        ),
        
        # Price validation for limit orders
        ValidationRule(
            "ORD-PRICE-001",
            "Limit orders must have a valid price",
            lambda order, ctx: ValidationResult(
                not (order.get("order_type") in ["limit", "stop_limit"]) or 
                (isinstance(order.get("price"), (int, float)) and order.get("price") > 0),
                "Limit orders must have a valid positive price"
            ),
            applies_to=["limit", "stop_limit"]
        ),
        
        # Stop price validation for stop orders
        ValidationRule(
            "ORD-STOP-001",
            "Stop orders must have a valid stop price",
            lambda order, ctx: ValidationResult(
                not (order.get("order_type") in ["stop", "stop_limit"]) or 
                (isinstance(order.get("stop_price"), (int, float)) and order.get("stop_price") > 0),
                "Stop orders must have a valid positive stop price"
            ),
            applies_to=["stop", "stop_limit"]
        ),
        
        # Quantity validation
        ValidationRule(
            "ORD-QTY-001",
            "Order quantity must be positive",
            lambda order, ctx: ValidationResult(
                isinstance(order.get("quantity"), (int, float)) and order.get("quantity") > 0,
                "Order quantity must be a positive number"
            )
        ),
        
        # Side validation
        ValidationRule(
            "ORD-SIDE-001",
            "Order side must be valid",
            lambda order, ctx: ValidationResult(
                order.get("side") in ["buy", "sell"],
                f"Invalid order side: {order.get('side')}. Must be one of: buy, sell"
            )
        ),
        
        # Time in force validation
        ValidationRule(
            "ORD-TIF-001",
            "Time in force must be valid",
            lambda order, ctx: ValidationResult(
                not order.get("time_in_force") or order.get("time_in_force") in ["day", "gtc", "ioc", "fok"],
                f"Invalid time in force: {order.get('time_in_force')}. Must be one of: day, gtc, ioc, fok"
            )
        ),
        
        # Market hours validation
        ValidationRule(
            "ORD-MKT-001",
            "Market must be open for non-extended hours orders",
            lambda order, ctx: ValidationResult(
                order.get("allow_outside_market_hours", False) or 
                is_market_open(order.get("symbol"), ctx.get("market", "US")),
                "Market is closed. Set allow_outside_market_hours=True to override"
            ),
            severity=ValidationSeverity.ERROR
        ),
        
        # Market close proximity validation
        ValidationRule(
            "ORD-MKT-002",
            "Order may not execute before market close",
            lambda order, ctx: ValidationResult(
                order.get("allow_outside_market_hours", False) or 
                get_time_to_market_close(ctx.get("market", "US")) > 300,  # 5 minutes
                "Market closing in less than 5 minutes. Order may not execute in time"
            ),
            severity=ValidationSeverity.WARNING
        ),
        
        # Order value validation
        ValidationRule(
            "ORD-VAL-001",
            "Order value must not exceed maximum allowed",
            lambda order, ctx: ValidationResult(
                calculate_order_value(order) <= ctx.get("max_order_value", float('inf')),
                f"Order value exceeds maximum allowed: {ctx.get('max_order_value', 'unlimited')}"
            )
        ),
        
        # Duplicate order validation
        ValidationRule(
            "ORD-DUP-001",
            "Duplicate order check",
            lambda order, ctx: ValidationResult(
                not ctx.get("recent_orders") or 
                not any(is_duplicate_order(order, recent_order) 
                        for recent_order in ctx.get("recent_orders", [])),
                "Potential duplicate order detected"
            ),
            severity=ValidationSeverity.WARNING
        ),
        
        # Wash trade prevention
        ValidationRule(
            "ORD-WASH-001",
            "Wash trade prevention",
            lambda order, ctx: ValidationResult(
                not ctx.get("positions") or 
                not is_potential_wash_trade(order, ctx.get("positions", {})),
                "Potential wash trade detected"
            ),
            severity=ValidationSeverity.ERROR
        ),
        
        # Restricted symbol validation
        ValidationRule(
            "ORD-REST-001",
            "Symbol not on restricted list",
            lambda order, ctx: ValidationResult(
                order.get("symbol") not in ctx.get("restricted_symbols", []),
                f"Symbol {order.get('symbol')} is on the restricted list"
            ),
            severity=ValidationSeverity.CRITICAL
        ),
        
        # Order rate limit validation
        ValidationRule(
            "ORD-RATE-001",
            "Order submission rate limit",
            lambda order, ctx: ValidationResult(
                ctx.get("order_count_last_minute", 0) < ctx.get("max_orders_per_minute", 100),
                f"Order rate limit exceeded: {ctx.get('max_orders_per_minute', 100)} per minute"
            )
        ),
        
        # Notional value limit by asset class
        ValidationRule(
            "ORD-NOTIONAL-001",
            "Notional value limit by asset class",
            lambda order, ctx: ValidationResult(
                calculate_order_value(order) <= get_max_notional_for_asset_class(
                    order.get("symbol"), 
                    ctx.get("asset_class_limits", {})
                ),
                f"Order value exceeds limit for this asset class"
            )
        ),
        
        # ADV percentage limit
        ValidationRule(
            "ORD-ADV-001",
            "Average Daily Volume percentage limit",
            lambda order, ctx: ValidationResult(
                not ctx.get("adv") or 
                order.get("quantity", 0) <= ctx.get("adv", 0) * ctx.get("max_adv_percentage", 0.1),
                f"Order quantity exceeds {ctx.get('max_adv_percentage', 0.1)*100}% of ADV"
            ),
            severity=ValidationSeverity.WARNING
        ),
        
        # Broker-specific validation
        ValidationRule(
            "ORD-BROKER-001",
            "Broker-specific order validation",
            lambda order, ctx: ValidationResult(
                not ctx.get("broker_validator") or 
                ctx.get("broker_validator")(order),
                "Order does not meet broker-specific requirements"
            )
        ),
    ]
    
    return Validator(rules)


# ===== Signal Validation Rules =====

def create_signal_validators() -> Validator:
    """Create a validator with standard signal validation rules.
    
    Returns:
        Validator: Validator with signal validation rules
    """
    rules = [
        # Symbol validation
        ValidationRule(
            "SIG-SYM-001",
            "Symbol must be provided and in valid format",
            lambda signal, ctx: ValidationResult(
                bool(signal.get("symbol")) and re.match(r'^[A-Z0-9.]{1,20}$', signal.get("symbol", "")),
                "Symbol must be provided and in valid format (letters, numbers, dots only, max 20 chars)"
            )
        ),
        
        # Direction validation
        ValidationRule(
            "SIG-DIR-001",
            "Signal direction must be valid",
            lambda signal, ctx: ValidationResult(
                signal.get("direction") in ["buy", "sell", "neutral"],
                f"Invalid signal direction: {signal.get('direction')}. Must be one of: buy, sell, neutral"
            )
        ),
        
        # Signal strength validation
        ValidationRule(
            "SIG-STR-001",
            "Signal strength must be between -1 and 1",
            lambda signal, ctx: ValidationResult(
                isinstance(signal.get("strength"), (int, float)) and -1 <= signal.get("strength", 0) <= 1,
                "Signal strength must be a number between -1 and 1"
            )
        ),
        
        # Confidence validation
        ValidationRule(
            "SIG-CONF-001",
            "Signal confidence must be between 0 and 1",
            lambda signal, ctx: ValidationResult(
                not signal.get("confidence") or 
                (isinstance(signal.get("confidence"), (int, float)) and 0 <= signal.get("confidence", 0) <= 1),
                "Signal confidence must be a number between 0 and 1"
            )
        ),
        
        # Strategy validation
        ValidationRule(
            "SIG-STRAT-001",
            "Signal must have a valid strategy",
            lambda signal, ctx: ValidationResult(
                bool(signal.get("strategy")),
                "Signal must have a strategy specified"
            )
        ),
        
        # Minimum strength threshold
        ValidationRule(
            "SIG-STR-002",
            "Signal strength must meet minimum threshold",
            lambda signal, ctx: ValidationResult(
                abs(signal.get("strength", 0)) >= ctx.get("min_signal_strength", 0.2),
                f"Signal strength below minimum threshold of {ctx.get('min_signal_strength', 0.2)}"
            ),
            severity=ValidationSeverity.WARNING
        ),
        
        # Minimum confidence threshold
        ValidationRule(
            "SIG-CONF-002",
            "Signal confidence must meet minimum threshold",
            lambda signal, ctx: ValidationResult(
                not signal.get("confidence") or 
                signal.get("confidence", 0) >= ctx.get("min_confidence", 0.5),
                f"Signal confidence below minimum threshold of {ctx.get('min_confidence', 0.5)}"
            ),
            severity=ValidationSeverity.WARNING
        ),
        
        # Restricted symbol validation
        ValidationRule(
            "SIG-REST-001",
            "Symbol not on restricted list",
            lambda signal, ctx: ValidationResult(
                signal.get("symbol") not in ctx.get("restricted_symbols", []),
                f"Symbol {signal.get('symbol')} is on the restricted list"
            ),
            severity=ValidationSeverity.ERROR
        ),
        
        # Market hours validation for signals
        ValidationRule(
            "SIG-MKT-001",
            "Market status check for signal",
            lambda signal, ctx: ValidationResult(
                is_market_open(signal.get("symbol"), ctx.get("market", "US")),
                "Market is closed. Signal may not be actionable until market opens"
            ),
            severity=ValidationSeverity.INFO
        ),
        
        # Signal expiration validation
        ValidationRule(
            "SIG-EXP-001",
            "Signal must not be expired",
            lambda signal, ctx: ValidationResult(
                not signal.get("expiration") or 
                datetime.fromisoformat(signal.get("expiration")) > datetime.now(pytz.UTC),
                "Signal has expired"
            )
        ),
        
        # Signal source validation
        ValidationRule(
            "SIG-SRC-001",
            "Signal must have a valid source",
            lambda signal, ctx: ValidationResult(
                bool(signal.get("source")),
                "Signal must have a source specified"
            )
        ),
        
        # Strategy-specific threshold validation
        ValidationRule(
            "SIG-STRAT-002",
            "Signal must meet strategy-specific thresholds",
            lambda signal, ctx: ValidationResult(
                not ctx.get("strategy_thresholds") or 
                not signal.get("strategy") or 
                signal.get("strategy") not in ctx.get("strategy_thresholds", {}) or 
                abs(signal.get("strength", 0)) >= ctx.get("strategy_thresholds", {}).get(
                    signal.get("strategy"), {}).get("min_signal_strength", 0),
                f"Signal strength below threshold for {signal.get('strategy')} strategy"
            ),
            severity=ValidationSeverity.WARNING
        ),
    ]
    
    return Validator(rules)


# ===== Helper Functions =====

def is_duplicate_order(order: Dict[str, Any], other_order: Dict[str, Any]) -> bool:
    """Check if an order is a duplicate of another recent order.
    
    Args:
        order: The order to check
        other_order: Another recent order to compare against
        
    Returns:
        bool: True if orders appear to be duplicates
    """
    # Check if orders have the same key properties
    return (
        order.get("symbol") == other_order.get("symbol") and
        order.get("side") == other_order.get("side") and
        order.get("order_type") == other_order.get("order_type") and
        abs(order.get("quantity", 0) - other_order.get("quantity", 0)) < 0.001 and
        (order.get("order_type") != "limit" or 
         abs(order.get("price", 0) - other_order.get("price", 0)) < 0.0001)
    )


def is_potential_wash_trade(order: Dict[str, Any], positions: Dict[str, Dict[str, Any]]) -> bool:
    """Check if an order might result in a wash trade.
    
    Args:
        order: The order to check
        positions: Current positions keyed by symbol
        
    Returns:
        bool: True if order might result in a wash trade
    """
    symbol = order.get("symbol")
    if symbol not in positions:
        return False
    
    position = positions[symbol]
    position_side = "buy" if position.get("quantity", 0) > 0 else "sell"
    order_side = order.get("side")
    
    # Wash trade if selling a recently bought position or buying a recently sold position
    return position_side != order_side and position.get("days_held", 30) <= 30


def get_max_notional_for_asset_class(symbol: str, asset_class_limits: Dict[str, float]) -> float:
    """Get the maximum notional value for an asset class.
    
    Args:
        symbol: The symbol to check
        asset_class_limits: Limits by asset class
        
    Returns:
        float: Maximum notional value allowed
    """
    # This is a simplified implementation - in production, you would determine
    # the asset class from the symbol using a more sophisticated method
    if symbol.endswith(".OPT"):
        return asset_class_limits.get("options", float('inf'))
    elif symbol.endswith(".FUT"):
        return asset_class_limits.get("futures", float('inf'))
    elif "/" in symbol:
        return asset_class_limits.get("forex", float('inf'))
    elif symbol.startswith("BTC") or symbol.startswith("ETH"):
        return asset_class_limits.get("crypto", float('inf'))
    else:
        return asset_class_limits.get("equities", float('inf'))


# ===== Factory Functions =====

def create_production_order_validator(config: Dict[str, Any] = None) -> Validator:
    """Create a production-ready order validator with configuration.
    
    Args:
        config: Configuration for the validator
        
    Returns:
        Validator: Configured order validator
    """
    validator = create_order_validators()
    
    if config:
        # Apply configuration to existing rules
        for rule_id, rule_config in config.get("rule_configs", {}).items():
            if rule_id in validator.rule_map:
                if "enabled" in rule_config:
                    validator.rule_map[rule_id].enabled = rule_config["enabled"]
                if "severity" in rule_config:
                    validator.rule_map[rule_id].severity = ValidationSeverity(rule_config["severity"])
        
        # Add custom rules if provided
        for rule_config in config.get("custom_rules", []):
            if all(k in rule_config for k in ["rule_id", "description", "validation_fn"]):
                validator.add_rule(ValidationRule(
                    rule_config["rule_id"],
                    rule_config["description"],
                    rule_config["validation_fn"],
                    ValidationSeverity(rule_config.get("severity", "ERROR")),
                    rule_config.get("applies_to"),
                    rule_config.get("enabled", True),
                    rule_config.get("metadata")
                ))
    
    return validator


def create_production_signal_validator(config: Dict[str, Any] = None) -> Validator:
    """Create a production-ready signal validator with configuration.
    
    Args:
        config: Configuration for the validator
        
    Returns:
        Validator: Configured signal validator
    """
    validator = create_signal_validators()
    
    if config:
        # Apply configuration to existing rules
        for rule_id, rule_config in config.get("rule_configs", {}).items():
            if rule_id in validator.rule_map:
                if "enabled" in rule_config:
                    validator.rule_map[rule_id].enabled = rule_config["enabled"]
                if "severity" in rule_config:
                    validator.rule_map[rule_id].severity = ValidationSeverity(rule_config["severity"])
        
        # Add custom rules if provided
        for rule_config in config.get("custom_rules", []):
            if all(k in rule_config for k in ["rule_id", "description", "validation_fn"]):
                validator.add_rule(ValidationRule(
                    rule_config["rule_id"],
                    rule_config["description"],
                    rule_config["validation_fn"],
                    ValidationSeverity(rule_config.get("severity", "ERROR")),
                    rule_config.get("applies_to"),
                    rule_config.get("enabled", True),
                    rule_config.get("metadata")
                ))
    
    return validator