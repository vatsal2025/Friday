"""Circuit Breakers for Trading Engine.

This module provides market-wide and account-level circuit breakers
to protect the trading system from extreme market conditions and
excessive losses.
"""

import datetime
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from enum import Enum
import json
import os

from src.orchestration.trading_engine.emergency import EmergencyHandler, EmergencyLevel, EmergencyTrigger, EmergencyAction
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    MARKET_WIDE = "market_wide"  # Affects all trading for specific markets
    ACCOUNT_LEVEL = "account_level"  # Affects trading for specific accounts
    STRATEGY_LEVEL = "strategy_level"  # Affects trading for specific strategies


class CircuitBreakerStatus(Enum):
    """Status of a circuit breaker."""
    NORMAL = "normal"  # Normal trading
    TRIGGERED = "triggered"  # Circuit breaker triggered
    MONITORING = "monitoring"  # Monitoring after trigger
    RESET = "reset"  # Reset after cooling period


class MarketDeclineLevel(Enum):
    """Market decline levels for market-wide circuit breakers."""
    LEVEL_1 = 1  # Level 1 decline (e.g., 7%)
    LEVEL_2 = 2  # Level 2 decline (e.g., 13%)
    LEVEL_3 = 3  # Level 3 decline (e.g., 20%)


class CircuitBreakerManager:
    """Manages circuit breakers for the trading engine."""
    
    def __init__(self, emergency_handlers=None, emergency_handler=None, config_path: Optional[str] = None):
        """Initialize the circuit breaker manager.
        
        Args:
            emergency_handlers: List of emergency handlers for declaring emergencies
            emergency_handler: Single emergency handler (for backward compatibility)
            config_path: Path to circuit breaker configuration file
        """
        # Handle both emergency_handlers list and single emergency_handler for backward compatibility
        if emergency_handlers:
            self.emergency_handler = emergency_handlers[0]
        else:
            self.emergency_handler = emergency_handler
            
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config", "circuit_breakers.json")
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize circuit breaker states
        self.market_circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.account_circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize from config
        self._initialize_circuit_breakers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load circuit breaker configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "market_wide": {
                "enabled": True,
                "decline_levels": {
                    "LEVEL_1": {
                        "threshold_percent": 7.0,
                        "halt_duration_minutes": 15,
                        "time_restrictions": {
                            "start_time": "09:30",
                            "end_time": "15:25"
                        }
                    },
                    "LEVEL_2": {
                        "threshold_percent": 13.0,
                        "halt_duration_minutes": 15,
                        "time_restrictions": {
                            "start_time": "09:30",
                            "end_time": "15:25"
                        }
                    },
                    "LEVEL_3": {
                        "threshold_percent": 20.0,
                        "halt_duration_minutes": 0,  # 0 means halt for the rest of the day
                        "time_restrictions": {
                            "start_time": "09:30",
                            "end_time": "16:00"
                        }
                    }
                },
                "reference_indices": [
                    {
                        "symbol": "SPY",
                        "name": "S&P 500 ETF",
                        "weight": 1.0
                    }
                ],
                "cooldown_period_minutes": 60,
                "max_triggers_per_day": 3
            },
            "account_level": {
                "enabled": True,
                "daily_loss_limit_percent": 5.0,  # 5% of account equity
                "intraday_loss_limit_percent": 3.0,  # 3% of account equity
                "max_drawdown_percent": 10.0,  # 10% of account equity
                "actions": {
                    "warning": {
                        "threshold_percent": 50.0,  # 50% of limit
                        "action": "NOTIFY"
                    },
                    "soft_limit": {
                        "threshold_percent": 80.0,  # 80% of limit
                        "action": "THROTTLE"
                    },
                    "hard_limit": {
                        "threshold_percent": 100.0,  # 100% of limit
                        "action": "HALT"
                    }
                },
                "reset_period": "daily",  # daily, weekly, monthly
                "exclude_accounts": []  # List of account IDs to exclude
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded circuit breaker configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading circuit breaker configuration: {str(e)}. Using default configuration.")
        else:
            logger.info(f"Circuit breaker configuration not found at {self.config_path}. Using default configuration.")
            # Save default configuration
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Saved default circuit breaker configuration to {self.config_path}")
            except Exception as e:
                logger.error(f"Error saving default circuit breaker configuration: {str(e)}")
        
        return default_config
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breaker states from configuration."""
        # Initialize market-wide circuit breakers
        if self.config.get("market_wide", {}).get("enabled", False):
            reference_indices = self.config.get("market_wide", {}).get("reference_indices", [])
            for index in reference_indices:
                symbol = index.get("symbol")
                if symbol:
                    self.market_circuit_breakers[symbol] = {
                        "status": CircuitBreakerStatus.NORMAL.value,
                        "reference_price": None,  # Will be set on first update
                        "current_price": None,
                        "decline_percent": 0.0,
                        "triggered_level": None,
                        "trigger_time": None,
                        "reset_time": None,
                        "triggers_today": 0,
                        "last_update_time": None
                    }
        
        # Initialize account-level circuit breakers
        if self.config.get("account_level", {}).get("enabled", False):
            # Account circuit breakers will be initialized when account data is first received
            pass
    
    def update_market_data(self, symbol: str, price: float, timestamp: Optional[datetime.datetime] = None) -> None:
        """Update market data and check for market-wide circuit breaker triggers.
        
        Args:
            symbol: Market symbol
            price: Current price
            timestamp: Timestamp of the price data (defaults to current time)
        """
        if not self.config.get("market_wide", {}).get("enabled", False):
            return
        
        # Check if this symbol is a reference index
        is_reference = False
        for index in self.config.get("market_wide", {}).get("reference_indices", []):
            if index.get("symbol") == symbol:
                is_reference = True
                break
        
        if not is_reference:
            return
        
        # Get or initialize circuit breaker state for this symbol
        if symbol not in self.market_circuit_breakers:
            self.market_circuit_breakers[symbol] = {
                "status": CircuitBreakerStatus.NORMAL.value,
                "reference_price": price,  # Initial reference price
                "current_price": price,
                "decline_percent": 0.0,
                "triggered_level": None,
                "trigger_time": None,
                "reset_time": None,
                "triggers_today": 0,
                "last_update_time": timestamp or datetime.datetime.now()
            }
            return
        
        # Get current state
        state = self.market_circuit_breakers[symbol]
        current_time = timestamp or datetime.datetime.now()
        
        # Set reference price if not set
        if state["reference_price"] is None:
            state["reference_price"] = price
            state["current_price"] = price
            state["last_update_time"] = current_time
            return
        
        # Update current price and calculate decline
        state["current_price"] = price
        state["last_update_time"] = current_time
        
        # Calculate decline percentage
        reference_price = state["reference_price"]
        decline_percent = ((reference_price - price) / reference_price) * 100.0
        state["decline_percent"] = decline_percent
        
        # Check if circuit breaker is already triggered
        if state["status"] == CircuitBreakerStatus.TRIGGERED.value:
            # Check if it's time to reset
            if state["reset_time"] and current_time >= state["reset_time"]:
                self._reset_market_circuit_breaker(symbol)
            return
        
        # Check for circuit breaker triggers
        self._check_market_circuit_breaker_triggers(symbol, decline_percent, current_time)
    
    def _check_market_circuit_breaker_triggers(self, symbol: str, decline_percent: float, timestamp: datetime.datetime) -> None:
        """Check if market-wide circuit breakers should be triggered.
        
        Args:
            symbol: Market symbol
            decline_percent: Current decline percentage
            timestamp: Current timestamp
        """
        # Get decline levels from config
        decline_levels = self.config.get("market_wide", {}).get("decline_levels", {})
        
        # Check if we've exceeded the maximum triggers per day
        state = self.market_circuit_breakers[symbol]
        max_triggers = self.config.get("market_wide", {}).get("max_triggers_per_day", 3)
        if state["triggers_today"] >= max_triggers:
            logger.info(f"Maximum circuit breaker triggers reached for {symbol} today")
            return
        
        # Check each level from highest to lowest
        for level_name in ["LEVEL_3", "LEVEL_2", "LEVEL_1"]:
            level_config = decline_levels.get(level_name)
            if not level_config:
                continue
            
            threshold = level_config.get("threshold_percent", 0.0)
            
            # Check if decline exceeds threshold
            if decline_percent >= threshold:
                # Check time restrictions
                time_restrictions = level_config.get("time_restrictions", {})
                start_time_str = time_restrictions.get("start_time", "09:30")
                end_time_str = time_restrictions.get("end_time", "16:00")
                
                # Parse time strings
                start_hour, start_minute = map(int, start_time_str.split(':'))
                end_hour, end_minute = map(int, end_time_str.split(':'))
                
                # Create datetime objects for comparison
                start_time = timestamp.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                end_time = timestamp.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
                
                # Check if current time is within restrictions
                if start_time <= timestamp <= end_time:
                    # Trigger circuit breaker
                    self._trigger_market_circuit_breaker(symbol, level_name, decline_percent, timestamp, level_config)
                    return
    
    def _trigger_market_circuit_breaker(self, symbol: str, level_name: str, decline_percent: float, 
                                       timestamp: datetime.datetime, level_config: Dict[str, Any]) -> None:
        """Trigger a market-wide circuit breaker.
        
        Args:
            symbol: Market symbol
            level_name: Level name (LEVEL_1, LEVEL_2, LEVEL_3)
            decline_percent: Current decline percentage
            timestamp: Current timestamp
            level_config: Configuration for this level
        """
        # Get halt duration
        halt_duration_minutes = level_config.get("halt_duration_minutes", 15)
        
        # Calculate reset time
        reset_time = None
        if halt_duration_minutes > 0:
            reset_time = timestamp + datetime.timedelta(minutes=halt_duration_minutes)
        
        # Update circuit breaker state
        state = self.market_circuit_breakers[symbol]
        state["status"] = CircuitBreakerStatus.TRIGGERED.value
        state["triggered_level"] = level_name
        state["trigger_time"] = timestamp
        state["reset_time"] = reset_time
        state["triggers_today"] += 1
        
        # Determine emergency level based on circuit breaker level
        emergency_level = EmergencyLevel.MEDIUM
        if level_name == "LEVEL_3":
            emergency_level = EmergencyLevel.CRITICAL
        elif level_name == "LEVEL_2":
            emergency_level = EmergencyLevel.HIGH
        
        # Determine emergency action
        emergency_action = EmergencyAction.PAUSE_TRADING
        if level_name == "LEVEL_3":
            emergency_action = EmergencyAction.HALT_TRADING
        
        # Declare emergency
        description = f"Market-wide circuit breaker triggered: {symbol} declined {decline_percent:.2f}% (Level {level_name[-1]})."
        if reset_time:
            description += f" Trading will resume at {reset_time.strftime('%H:%M:%S')}"
        else:
            description += " Trading halted for the remainder of the day."
        
        self.emergency_handler.declare_emergency(
            trigger=EmergencyTrigger.CIRCUIT_BREAKER,
            level=emergency_level,
            description=description,
            recommended_action=emergency_action,
            affected_markets=[symbol],
            metadata={
                "circuit_breaker_type": CircuitBreakerType.MARKET_WIDE.value,
                "symbol": symbol,
                "level": level_name,
                "decline_percent": decline_percent,
                "trigger_time": timestamp.isoformat(),
                "reset_time": reset_time.isoformat() if reset_time else None,
                "halt_duration_minutes": halt_duration_minutes
            }
        )
        
        logger.warning(description)
    
    def _reset_market_circuit_breaker(self, symbol: str) -> None:
        """Reset a market-wide circuit breaker after the halt period.
        
        Args:
            symbol: Market symbol
        """
        state = self.market_circuit_breakers[symbol]
        
        # Update state
        state["status"] = CircuitBreakerStatus.NORMAL.value
        state["triggered_level"] = None
        state["trigger_time"] = None
        state["reset_time"] = None
        
        # Update reference price to current price
        state["reference_price"] = state["current_price"]
        state["decline_percent"] = 0.0
        
        # Resolve emergency
        self.emergency_handler.resolve_emergency(
            trigger=EmergencyTrigger.CIRCUIT_BREAKER,
            metadata={
                "circuit_breaker_type": CircuitBreakerType.MARKET_WIDE.value,
                "symbol": symbol
            }
        )
        
        logger.info(f"Market-wide circuit breaker reset for {symbol}")
    
    def update_account_data(self, account_id: str, equity: float, starting_equity: Optional[float] = None,
                          daily_pnl: Optional[float] = None, timestamp: Optional[datetime.datetime] = None) -> None:
        """Update account data and check for account-level circuit breaker triggers.
        
        Args:
            account_id: Account ID
            equity: Current account equity
            starting_equity: Starting equity (e.g., beginning of day)
            daily_pnl: Daily profit and loss
            timestamp: Timestamp of the account data (defaults to current time)
        """
        if not self.config.get("account_level", {}).get("enabled", False):
            return
        
        # Check if this account is excluded
        exclude_accounts = self.config.get("account_level", {}).get("exclude_accounts", [])
        if account_id in exclude_accounts:
            return
        
        current_time = timestamp or datetime.datetime.now()
        
        # Get or initialize circuit breaker state for this account
        if account_id not in self.account_circuit_breakers:
            self.account_circuit_breakers[account_id] = {
                "status": CircuitBreakerStatus.NORMAL.value,
                "starting_equity": starting_equity or equity,
                "current_equity": equity,
                "highest_equity": equity,
                "daily_pnl": daily_pnl or 0.0,
                "daily_loss_percent": 0.0,
                "max_drawdown_percent": 0.0,
                "triggered_action": None,
                "trigger_time": None,
                "last_update_time": current_time
            }
            return
        
        # Get current state
        state = self.account_circuit_breakers[account_id]
        
        # Update equity and calculate metrics
        state["current_equity"] = equity
        state["last_update_time"] = current_time
        
        # Update highest equity if current equity is higher
        if equity > state["highest_equity"]:
            state["highest_equity"] = equity
        
        # Calculate daily loss percentage
        if starting_equity is not None:
            state["starting_equity"] = starting_equity
        
        daily_loss_percent = ((state["starting_equity"] - equity) / state["starting_equity"]) * 100.0
        state["daily_loss_percent"] = max(0.0, daily_loss_percent)  # Only consider losses
        
        # Calculate drawdown percentage
        drawdown_percent = ((state["highest_equity"] - equity) / state["highest_equity"]) * 100.0
        state["max_drawdown_percent"] = max(state["max_drawdown_percent"], drawdown_percent)
        
        # Update daily P&L if provided
        if daily_pnl is not None:
            state["daily_pnl"] = daily_pnl
        
        # Check for circuit breaker triggers
        self._check_account_circuit_breaker_triggers(account_id, current_time)
    
    def _check_account_circuit_breaker_triggers(self, account_id: str, timestamp: datetime.datetime) -> None:
        """Check if account-level circuit breakers should be triggered.
        
        Args:
            account_id: Account ID
            timestamp: Current timestamp
        """
        # Get account circuit breaker configuration
        account_config = self.config.get("account_level", {})
        daily_loss_limit = account_config.get("daily_loss_limit_percent", 5.0)
        max_drawdown_limit = account_config.get("max_drawdown_percent", 10.0)
        
        # Get current state
        state = self.account_circuit_breakers[account_id]
        
        # Skip if already triggered
        if state["status"] == CircuitBreakerStatus.TRIGGERED.value:
            return
        
        # Check daily loss limit
        daily_loss_percent = state["daily_loss_percent"]
        daily_loss_ratio = daily_loss_percent / daily_loss_limit
        
        # Check drawdown limit
        drawdown_percent = state["max_drawdown_percent"]
        drawdown_ratio = drawdown_percent / max_drawdown_limit
        
        # Use the higher of the two ratios
        loss_ratio = max(daily_loss_ratio, drawdown_ratio)
        
        # Check against thresholds
        actions = account_config.get("actions", {})
        
        # Check hard limit first (highest threshold)
        hard_limit = actions.get("hard_limit", {})
        hard_threshold = hard_limit.get("threshold_percent", 100.0) / 100.0
        
        if loss_ratio >= hard_threshold:
            self._trigger_account_circuit_breaker(
                account_id, "hard_limit", daily_loss_percent, drawdown_percent, timestamp)
            return
        
        # Check soft limit
        soft_limit = actions.get("soft_limit", {})
        soft_threshold = soft_limit.get("threshold_percent", 80.0) / 100.0
        
        if loss_ratio >= soft_threshold:
            self._trigger_account_circuit_breaker(
                account_id, "soft_limit", daily_loss_percent, drawdown_percent, timestamp)
            return
        
        # Check warning level
        warning = actions.get("warning", {})
        warning_threshold = warning.get("threshold_percent", 50.0) / 100.0
        
        if loss_ratio >= warning_threshold:
            self._trigger_account_circuit_breaker(
                account_id, "warning", daily_loss_percent, drawdown_percent, timestamp)
            return
    
    def _trigger_account_circuit_breaker(self, account_id: str, action_level: str, 
                                        daily_loss_percent: float, drawdown_percent: float,
                                        timestamp: datetime.datetime) -> None:
        """Trigger an account-level circuit breaker.
        
        Args:
            account_id: Account ID
            action_level: Action level (warning, soft_limit, hard_limit)
            daily_loss_percent: Daily loss percentage
            drawdown_percent: Drawdown percentage
            timestamp: Current timestamp
        """
        # Get account circuit breaker configuration
        account_config = self.config.get("account_level", {})
        actions = account_config.get("actions", {})
        action_config = actions.get(action_level, {})
        action_name = action_config.get("action", "NOTIFY")
        
        # Update circuit breaker state
        state = self.account_circuit_breakers[account_id]
        
        # Only update state for soft and hard limits
        if action_level in ["soft_limit", "hard_limit"]:
            state["status"] = CircuitBreakerStatus.TRIGGERED.value
        else:
            state["status"] = CircuitBreakerStatus.MONITORING.value
        
        state["triggered_action"] = action_level
        state["trigger_time"] = timestamp
        
        # Determine emergency level based on action level
        emergency_level = EmergencyLevel.LOW
        if action_level == "hard_limit":
            emergency_level = EmergencyLevel.HIGH
        elif action_level == "soft_limit":
            emergency_level = EmergencyLevel.MEDIUM
        
        # Determine emergency action
        emergency_action = EmergencyAction.MONITOR
        if action_name == "HALT":
            emergency_action = EmergencyAction.PAUSE_TRADING
        elif action_name == "THROTTLE":
            emergency_action = EmergencyAction.THROTTLE
        
        # Declare emergency for soft and hard limits only
        if action_level in ["soft_limit", "hard_limit"]:
            description = f"Account-level circuit breaker triggered for account {account_id}: "
            description += f"Daily loss: {daily_loss_percent:.2f}%, Max drawdown: {drawdown_percent:.2f}%"
            
            self.emergency_handler.declare_emergency(
                trigger=EmergencyTrigger.RISK_BREACH,
                level=emergency_level,
                description=description,
                recommended_action=emergency_action,
                affected_accounts=[account_id],
                metadata={
                    "circuit_breaker_type": CircuitBreakerType.ACCOUNT_LEVEL.value,
                    "account_id": account_id,
                    "action_level": action_level,
                    "action_name": action_name,
                    "daily_loss_percent": daily_loss_percent,
                    "drawdown_percent": drawdown_percent,
                    "trigger_time": timestamp.isoformat()
                }
            )
            
            logger.warning(description)
        else:
            # Just log a warning for warning level
            logger.warning(
                f"Account {account_id} approaching loss limits: "
                f"Daily loss: {daily_loss_percent:.2f}%, Max drawdown: {drawdown_percent:.2f}%"
            )
    
    def reset_account_circuit_breaker(self, account_id: str) -> bool:
        """Manually reset an account-level circuit breaker.
        
        Args:
            account_id: Account ID
            
        Returns:
            bool: True if reset was successful, False otherwise
        """
        if account_id not in self.account_circuit_breakers:
            logger.error(f"Account {account_id} not found in circuit breakers")
            return False
        
        state = self.account_circuit_breakers[account_id]
        
        # Only reset if triggered
        if state["status"] not in [CircuitBreakerStatus.TRIGGERED.value, CircuitBreakerStatus.MONITORING.value]:
            logger.info(f"Account circuit breaker for {account_id} is not triggered, no need to reset")
            return False
        
        # Reset state
        state["status"] = CircuitBreakerStatus.NORMAL.value
        state["triggered_action"] = None
        state["trigger_time"] = None
        
        # Resolve emergency
        self.emergency_handler.resolve_emergency(
            trigger=EmergencyTrigger.RISK_BREACH,
            metadata={
                "circuit_breaker_type": CircuitBreakerType.ACCOUNT_LEVEL.value,
                "account_id": account_id
            }
        )
        
        logger.info(f"Account circuit breaker reset for {account_id}")
        return True
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics for all circuit breakers.
        
        This should be called at the start of each trading day.
        """
        # Reset market-wide circuit breakers
        for symbol in self.market_circuit_breakers:
            state = self.market_circuit_breakers[symbol]
            state["triggers_today"] = 0
            state["reference_price"] = state["current_price"]
            state["decline_percent"] = 0.0
        
        # Reset account-level circuit breakers
        for account_id in self.account_circuit_breakers:
            state = self.account_circuit_breakers[account_id]
            state["daily_loss_percent"] = 0.0
            state["starting_equity"] = state["current_equity"]
            
            # Only reset status if it's not already triggered
            if state["status"] != CircuitBreakerStatus.TRIGGERED.value:
                state["status"] = CircuitBreakerStatus.NORMAL.value
        
        logger.info("Reset daily metrics for all circuit breakers")
    
    def get_market_circuit_breaker_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of market-wide circuit breakers.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dict[str, Any]: Circuit breaker status information
        """
        if symbol:
            if symbol in self.market_circuit_breakers:
                return {symbol: self.market_circuit_breakers[symbol]}
            return {}
        
        return self.market_circuit_breakers
    
    def get_account_circuit_breaker_status(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of account-level circuit breakers.
        
        Args:
            account_id: Optional account ID to filter by
            
        Returns:
            Dict[str, Any]: Circuit breaker status information
        """
        if account_id:
            if account_id in self.account_circuit_breakers:
                return {account_id: self.account_circuit_breakers[account_id]}
            return {}
        
        return self.account_circuit_breakers


def create_circuit_breaker_manager(emergency_handler: EmergencyHandler, 
                                 config_path: Optional[str] = None) -> CircuitBreakerManager:
    """Create a circuit breaker manager.
    
    Args:
        emergency_handler: Emergency handler for declaring emergencies
        config_path: Path to circuit breaker configuration file
        
    Returns:
        CircuitBreakerManager: Initialized circuit breaker manager
    """
    return CircuitBreakerManager(emergency_handlers=[emergency_handler], config_path=config_path)