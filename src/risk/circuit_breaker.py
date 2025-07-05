"""Circuit breaker implementation for trading risk management.

This module provides circuit breaker functionality to automatically halt or
throttle trading activity based on market conditions or account performance.
It implements both market-wide circuit breakers and account-level circuit breakers.
"""

import datetime
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
import threading
import time
import uuid
from abc import ABC, abstractmethod

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.emergency import EmergencyTrigger, EmergencyLevel, EmergencyAction

# Configure logger
logger = get_logger(__name__)


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    MARKET_WIDE = "market_wide"  # Market-wide circuit breaker
    ACCOUNT_LEVEL = "account_level"  # Account-specific circuit breaker
    SYMBOL_LEVEL = "symbol_level"  # Symbol-specific circuit breaker


class CircuitBreakerLevel(Enum):
    """Circuit breaker severity levels."""
    LEVEL_1 = "LEVEL_1"  # Level 1 circuit breaker (least severe)
    LEVEL_2 = "LEVEL_2"  # Level 2 circuit breaker
    LEVEL_3 = "LEVEL_3"  # Level 3 circuit breaker (most severe)
    WARNING = "warning"  # Warning level (pre-breach)
    SOFT_LIMIT = "soft_limit"  # Soft limit breach
    HARD_LIMIT = "hard_limit"  # Hard limit breach


class CircuitBreakerStatus(Enum):
    """Circuit breaker status."""
    ACTIVE = "active"  # Circuit breaker is active
    TRIGGERED = "triggered"  # Circuit breaker has been triggered
    RESET = "reset"  # Circuit breaker has been reset
    DISABLED = "disabled"  # Circuit breaker is disabled


class CircuitBreakerEvent:
    """Represents a circuit breaker event."""
    
    def __init__(self,
                circuit_type: CircuitBreakerType,
                level: CircuitBreakerLevel,
                description: str,
                market: Optional[str] = None,
                symbol: Optional[str] = None,
                account_id: Optional[str] = None,
                duration_minutes: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """Initialize a circuit breaker event.
        
        Args:
            circuit_type: Type of circuit breaker
            level: Severity level of the circuit breaker
            description: Human-readable description
            market: Affected market
            symbol: Affected symbol
            account_id: Affected account ID
            duration_minutes: Duration of the circuit breaker in minutes
            metadata: Additional metadata
        """
        self.id = str(uuid.uuid4())
        self.circuit_type = circuit_type
        self.level = level
        self.description = description
        self.market = market
        self.symbol = symbol
        self.account_id = account_id
        self.timestamp = datetime.datetime.now()
        self.duration_minutes = duration_minutes
        self.metadata = metadata or {}
        self.expiry_timestamp = None
        if duration_minutes is not None:
            self.expiry_timestamp = self.timestamp + datetime.timedelta(minutes=duration_minutes)
        self.status = CircuitBreakerStatus.TRIGGERED
        self.reset_timestamp = None
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.status = CircuitBreakerStatus.RESET
        self.reset_timestamp = datetime.datetime.now()
    
    def is_expired(self) -> bool:
        """Check if the circuit breaker has expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if self.expiry_timestamp is None:
            return False
        return datetime.datetime.now() > self.expiry_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the circuit breaker event to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the event
        """
        return {
            "id": self.id,
            "circuit_type": self.circuit_type.value,
            "level": self.level.value,
            "description": self.description,
            "market": self.market,
            "symbol": self.symbol,
            "account_id": self.account_id,
            "duration_minutes": self.duration_minutes,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "expiry_timestamp": self.expiry_timestamp.isoformat() if self.expiry_timestamp else None,
            "status": self.status.value,
            "reset_timestamp": self.reset_timestamp.isoformat() if self.reset_timestamp else None
        }


class CircuitBreaker(ABC):
    """Base class for all circuit breakers."""
    
    def __init__(self, enabled: bool = True):
        """Initialize a circuit breaker.
        
        Args:
            enabled: Whether the circuit breaker is enabled
        """
        self.enabled = enabled
        self.status = CircuitBreakerStatus.ACTIVE if enabled else CircuitBreakerStatus.DISABLED
        self.last_triggered = None
    
    @abstractmethod
    def check(self, *args, **kwargs) -> Optional[CircuitBreakerEvent]:
        """Check if the circuit breaker should be triggered.
        
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the circuit breaker to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        pass





class MarketWideCircuitBreaker(CircuitBreaker):
    """Market-wide circuit breaker implementation."""
    
    def __init__(self,
                market: str,
                level_1_percent: float = 7.0,
                level_2_percent: float = 13.0,
                level_3_percent: float = 20.0,
                reference_index: Optional[str] = None,
                level_1_duration_minutes: int = 15,
                level_2_duration_minutes: int = 30,
                level_3_duration_minutes: int = 0,  # 0 means rest of the day
                enabled: bool = True):
        """Initialize a market-wide circuit breaker.
        
        Args:
            market: Market identifier
            level_1_percent: Percentage drop for Level 1 circuit breaker
            level_2_percent: Percentage drop for Level 2 circuit breaker
            level_3_percent: Percentage drop for Level 3 circuit breaker
            reference_index: Reference index for the circuit breaker
            level_1_duration_minutes: Duration in minutes for Level 1 circuit breaker
            level_2_duration_minutes: Duration in minutes for Level 2 circuit breaker
            level_3_duration_minutes: Duration in minutes for Level 3 circuit breaker (0 = rest of day)
            enabled: Whether the circuit breaker is enabled
        """
        super().__init__(enabled=enabled)
        self.market = market
        self.level_1_percent = level_1_percent
        self.level_2_percent = level_2_percent
        self.level_3_percent = level_3_percent
        self.reference_index = reference_index or market
        self.level_1_duration_minutes = level_1_duration_minutes
        self.level_2_duration_minutes = level_2_duration_minutes
        self.level_3_duration_minutes = level_3_duration_minutes
        self.reference_value = None
        self.current_value = None
    
    def update_reference_value(self, value: float) -> None:
        """Update the reference value for the circuit breaker.
        
        Args:
            value: New reference value
        """
        self.reference_value = value
    
    def check(self, current_value: float) -> Optional[CircuitBreakerEvent]:
        """Check if the circuit breaker should be triggered.
        
        Args:
            current_value: Current value of the reference index
            
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        if not self.enabled or self.reference_value is None:
            return None
        
        self.current_value = current_value
        percent_change = ((current_value - self.reference_value) / self.reference_value) * 100.0
        
        # Check for circuit breaker levels (negative percent_change means a drop)
        if percent_change <= -self.level_3_percent:
            # Level 3 circuit breaker
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.LEVEL_3,
                description=f"Level 3 circuit breaker triggered: {abs(percent_change):.2f}% drop in {self.market}",
                market=self.market,
                duration_minutes=self.level_3_duration_minutes,
                metadata={
                    "reference_value": self.reference_value,
                    "current_value": current_value,
                    "percent_change": percent_change,
                    "threshold": -self.level_3_percent
                }
            )
        elif percent_change <= -self.level_2_percent:
            # Level 2 circuit breaker
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.LEVEL_2,
                description=f"Level 2 circuit breaker triggered: {abs(percent_change):.2f}% drop in {self.market}",
                market=self.market,
                duration_minutes=self.level_2_duration_minutes,
                metadata={
                    "reference_value": self.reference_value,
                    "current_value": current_value,
                    "percent_change": percent_change,
                    "threshold": -self.level_2_percent
                }
            )
        elif percent_change <= -self.level_1_percent:
            # Level 1 circuit breaker
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.LEVEL_1,
                description=f"Level 1 circuit breaker triggered: {abs(percent_change):.2f}% drop in {self.market}",
                market=self.market,
                duration_minutes=self.level_1_duration_minutes,
                metadata={
                    "reference_value": self.reference_value,
                    "current_value": current_value,
                    "percent_change": percent_change,
                    "threshold": -self.level_1_percent
                }
            )
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the circuit breaker to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "market": self.market,
            "level_1_percent": self.level_1_percent,
            "level_2_percent": self.level_2_percent,
            "level_3_percent": self.level_3_percent,
            "reference_index": self.reference_index,
            "level_1_duration_minutes": self.level_1_duration_minutes,
            "level_2_duration_minutes": self.level_2_duration_minutes,
            "level_3_duration_minutes": self.level_3_duration_minutes,
            "enabled": self.enabled,
            "status": self.status.value,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "reference_value": self.reference_value,
            "current_value": self.current_value
        }


class AccountCircuitBreaker(CircuitBreaker):
    """Account-level circuit breaker implementation."""
    
    def __init__(self,
                account_id: str,
                daily_loss_percent_warning: float = 5.0,
                daily_loss_percent_soft: float = 10.0,
                daily_loss_percent_hard: float = 15.0,
                daily_loss_amount_warning: Optional[float] = None,
                daily_loss_amount_soft: Optional[float] = None,
                daily_loss_amount_hard: Optional[float] = None,
                enabled: bool = True):
        """Initialize an account-level circuit breaker.
        
        Args:
            account_id: Account identifier
            daily_loss_percent_warning: Daily loss percentage for warning
            daily_loss_percent_soft: Daily loss percentage for soft limit
            daily_loss_percent_hard: Daily loss percentage for hard limit
            daily_loss_amount_warning: Daily loss amount for warning
            daily_loss_amount_soft: Daily loss amount for soft limit
            daily_loss_amount_hard: Daily loss amount for hard limit
            enabled: Whether the circuit breaker is enabled
        """
        super().__init__(enabled=enabled)
        self.account_id = account_id
        self.daily_loss_percent_warning = daily_loss_percent_warning
        self.daily_loss_percent_soft = daily_loss_percent_soft
        self.daily_loss_percent_hard = daily_loss_percent_hard
        self.daily_loss_amount_warning = daily_loss_amount_warning
        self.daily_loss_amount_soft = daily_loss_amount_soft
        self.daily_loss_amount_hard = daily_loss_amount_hard
        self.starting_balance = None
        self.current_balance = None
        self.daily_pnl = 0.0
        self.reset_date = datetime.date.today()
    
    def update_starting_balance(self, balance: float) -> None:
        """Update the starting balance for the account.
        
        Args:
            balance: Starting balance
        """
        self.starting_balance = balance
        self.current_balance = balance
    
    def update_balance(self, balance: float) -> None:
        """Update the current balance for the account.
        
        Args:
            balance: Current balance
        """
        self.current_balance = balance
        if self.starting_balance is not None:
            self.daily_pnl = balance - self.starting_balance
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update the daily P&L directly.
        
        Args:
            pnl: Daily P&L
        """
        self.daily_pnl = pnl
    
    def reset_daily(self) -> None:
        """Reset the daily tracking."""
        self.reset_date = datetime.date.today()
        if self.current_balance is not None:
            self.starting_balance = self.current_balance
        self.daily_pnl = 0.0
    
    def check(self) -> Optional[CircuitBreakerEvent]:
        """Check if the circuit breaker should be triggered.
        
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        if not self.enabled or self.starting_balance is None or self.current_balance is None:
            return None
        
        # Check if we need to reset for a new day
        today = datetime.date.today()
        if today > self.reset_date:
            self.reset_daily()
        
        # Calculate loss percentage
        loss_percent = 0.0
        if self.daily_pnl < 0 and self.starting_balance > 0:
            loss_percent = (abs(self.daily_pnl) / self.starting_balance) * 100.0
        
        # Check for circuit breaker levels
        if self.daily_loss_amount_hard is not None and abs(self.daily_pnl) >= self.daily_loss_amount_hard:
            # Hard limit breach by amount
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.HARD_LIMIT,
                description=f"Account {self.account_id} hard daily loss limit breached: ${abs(self.daily_pnl):.2f}",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_amount": self.daily_loss_amount_hard
                }
            )
        elif loss_percent >= self.daily_loss_percent_hard:
            # Hard limit breach by percentage
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.HARD_LIMIT,
                description=f"Account {self.account_id} hard daily loss limit breached: {loss_percent:.2f}%",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_percent": self.daily_loss_percent_hard
                }
            )
        elif self.daily_loss_amount_soft is not None and abs(self.daily_pnl) >= self.daily_loss_amount_soft:
            # Soft limit breach by amount
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.SOFT_LIMIT,
                description=f"Account {self.account_id} soft daily loss limit breached: ${abs(self.daily_pnl):.2f}",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_amount": self.daily_loss_amount_soft
                }
            )
        elif loss_percent >= self.daily_loss_percent_soft:
            # Soft limit breach by percentage
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.SOFT_LIMIT,
                description=f"Account {self.account_id} soft daily loss limit breached: {loss_percent:.2f}%",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_percent": self.daily_loss_percent_soft
                }
            )
        elif self.daily_loss_amount_warning is not None and abs(self.daily_pnl) >= self.daily_loss_amount_warning:
            # Warning by amount
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.WARNING,
                description=f"Account {self.account_id} approaching daily loss limit: ${abs(self.daily_pnl):.2f}",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_amount": self.daily_loss_amount_warning
                }
            )
        elif loss_percent >= self.daily_loss_percent_warning:
            # Warning by percentage
            return CircuitBreakerEvent(
                circuit_type=CircuitBreakerType.ACCOUNT_LEVEL,
                level=CircuitBreakerLevel.WARNING,
                description=f"Account {self.account_id} approaching daily loss limit: {loss_percent:.2f}%",
                account_id=self.account_id,
                metadata={
                    "starting_balance": self.starting_balance,
                    "current_balance": self.current_balance,
                    "daily_pnl": self.daily_pnl,
                    "loss_percent": loss_percent,
                    "threshold_percent": self.daily_loss_percent_warning
                }
            )
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the circuit breaker to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "account_id": self.account_id,
            "daily_loss_percent_warning": self.daily_loss_percent_warning,
            "daily_loss_percent_soft": self.daily_loss_percent_soft,
            "daily_loss_percent_hard": self.daily_loss_percent_hard,
            "daily_loss_amount_warning": self.daily_loss_amount_warning,
            "daily_loss_amount_soft": self.daily_loss_amount_soft,
            "daily_loss_amount_hard": self.daily_loss_amount_hard,
            "enabled": self.enabled,
            "status": self.status.value,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "daily_pnl": self.daily_pnl,
            "reset_date": self.reset_date.isoformat()
        }


class CircuitBreakerManager:
    """Manager for circuit breakers."""
    
    def __init__(self, circuit_breakers=None, emergency_handlers=None, emergency_handler=None):
        """Initialize the circuit breaker manager.
        
        Args:
            circuit_breakers: Optional list of circuit breakers to add
            emergency_handlers: Optional list of emergency handlers to notify of circuit breaker events
            emergency_handler: Optional single emergency handler (for backward compatibility)
        """
        # Handle both emergency_handlers list and single emergency_handler for backward compatibility
        if emergency_handlers:
            self.emergency_handler = emergency_handlers[0] if emergency_handlers else None
        else:
            self.emergency_handler = emergency_handler
            
        self.market_circuit_breakers: Dict[str, MarketWideCircuitBreaker] = {}
        self.account_circuit_breakers: Dict[str, AccountCircuitBreaker] = {}
        self.active_circuit_breakers: Dict[str, CircuitBreakerEvent] = {}
        self.circuit_breaker_history: List[CircuitBreakerEvent] = []
        self.check_thread = None
        self.running = False
        self.check_interval_seconds = 5
        self.lock = threading.Lock()
        
        # Add circuit breakers if provided
        if circuit_breakers:
            for cb in circuit_breakers:
                self.add_circuit_breaker(cb)
    
    def start(self) -> None:
        """Start the circuit breaker manager."""
        with self.lock:
            if not self.running:
                self.running = True
                self.check_thread = threading.Thread(target=self._check_circuit_breakers, daemon=True)
                self.check_thread.start()
                logger.info("Circuit breaker manager started")
    
    def stop(self) -> None:
        """Stop the circuit breaker manager."""
        with self.lock:
            self.running = False
            if self.check_thread and self.check_thread.is_alive():
                self.check_thread.join(timeout=2.0)
            logger.info("Circuit breaker manager stopped")
    
    def _check_circuit_breakers(self) -> None:
        """Periodically check circuit breakers."""
        while self.running:
            try:
                # Check for expired circuit breakers
                self._check_expired_circuit_breakers()
                
                # Sleep for the check interval
                time.sleep(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error checking circuit breakers: {str(e)}")
    
    def _check_expired_circuit_breakers(self) -> None:
        """Check for expired circuit breakers and reset them."""
        with self.lock:
            expired_ids = []
            for cb_id, cb_event in self.active_circuit_breakers.items():
                if cb_event.is_expired():
                    cb_event.reset()
                    self.circuit_breaker_history.append(cb_event)
                    expired_ids.append(cb_id)
                    logger.info(f"Circuit breaker expired and reset: {cb_event.description}")
            
            # Remove expired circuit breakers from active list
            for cb_id in expired_ids:
                del self.active_circuit_breakers[cb_id]
    
    def add_circuit_breaker(self, circuit_breaker) -> None:
        """Add a circuit breaker.
        
        Args:
            circuit_breaker: Circuit breaker to add
        """
        with self.lock:
            if isinstance(circuit_breaker, MarketWideCircuitBreaker):
                self.add_market_circuit_breaker(circuit_breaker)
            elif isinstance(circuit_breaker, AccountCircuitBreaker):
                self.add_account_circuit_breaker(circuit_breaker)
            else:
                raise ValueError(f"Unsupported circuit breaker type: {type(circuit_breaker)}")
    
    def add_market_circuit_breaker(self, circuit_breaker: MarketWideCircuitBreaker) -> None:
        """Add a market-wide circuit breaker.
        
        Args:
            circuit_breaker: Market-wide circuit breaker to add
        """
        with self.lock:
            self.market_circuit_breakers[circuit_breaker.market] = circuit_breaker
            logger.info(f"Added market-wide circuit breaker for {circuit_breaker.market}")
    
    def add_account_circuit_breaker(self, circuit_breaker: AccountCircuitBreaker) -> None:
        """Add an account-level circuit breaker.
        
        Args:
            circuit_breaker: Account-level circuit breaker to add
        """
        with self.lock:
            self.account_circuit_breakers[circuit_breaker.account_id] = circuit_breaker
            logger.info(f"Added account-level circuit breaker for {circuit_breaker.account_id}")
    
    def remove_market_circuit_breaker(self, market: str) -> bool:
        """Remove a market-wide circuit breaker.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: True if removed, False if not found
        """
        with self.lock:
            if market in self.market_circuit_breakers:
                del self.market_circuit_breakers[market]
                logger.info(f"Removed market-wide circuit breaker for {market}")
                return True
            return False
    
    def remove_account_circuit_breaker(self, account_id: str) -> bool:
        """Remove an account-level circuit breaker.
        
        Args:
            account_id: Account identifier
            
        Returns:
            bool: True if removed, False if not found
        """
        with self.lock:
            if account_id in self.account_circuit_breakers:
                del self.account_circuit_breakers[account_id]
                logger.info(f"Removed account-level circuit breaker for {account_id}")
                return True
            return False
    
    def add_circuit_breaker(self, circuit_breaker: CircuitBreaker) -> None:
        """Add a circuit breaker to the manager.
        
        This method determines the type of circuit breaker and adds it to the appropriate collection.
        
        Args:
            circuit_breaker: Circuit breaker to add
        """
        if isinstance(circuit_breaker, MarketWideCircuitBreaker):
            self.add_market_circuit_breaker(circuit_breaker)
        elif isinstance(circuit_breaker, AccountCircuitBreaker):
            self.add_account_circuit_breaker(circuit_breaker)
        else:
            logger.warning(f"Unknown circuit breaker type: {type(circuit_breaker).__name__}")
            raise ValueError(f"Unsupported circuit breaker type: {type(circuit_breaker).__name__}")
    
    def update_market_data(self, market: str, current_value: float) -> Optional[CircuitBreakerEvent]:
        """Update market data and check for circuit breaker triggers.
        
        Args:
            market: Market identifier
            current_value: Current value of the market index
            
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        with self.lock:
            if market in self.market_circuit_breakers:
                circuit_breaker = self.market_circuit_breakers[market]
                
                # Check if circuit breaker is triggered
                event = circuit_breaker.check(current_value)
                if event:
                    # Update last triggered timestamp
                    circuit_breaker.last_triggered = datetime.datetime.now()
                    
                    # Add to active circuit breakers
                    self.active_circuit_breakers[event.id] = event
                    
                    # Notify emergency handler if available
                    if self.emergency_handler:
                        self._notify_emergency_handler(event)
                    
                    logger.warning(f"Market-wide circuit breaker triggered: {event.description}")
                    return event
            
            return None
    
    def update_account_balance(self, account_id: str, balance: float) -> Optional[CircuitBreakerEvent]:
        """Update account balance and check for circuit breaker triggers.
        
        Args:
            account_id: Account identifier
            balance: Current account balance
            
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        with self.lock:
            if account_id in self.account_circuit_breakers:
                circuit_breaker = self.account_circuit_breakers[account_id]
                
                # Update balance
                circuit_breaker.update_balance(balance)
                
                # Check if circuit breaker is triggered
                event = circuit_breaker.check()
                if event:
                    # Update last triggered timestamp
                    circuit_breaker.last_triggered = datetime.datetime.now()
                    
                    # Add to active circuit breakers
                    self.active_circuit_breakers[event.id] = event
                    
                    # Notify emergency handler if available
                    if self.emergency_handler:
                        self._notify_emergency_handler(event)
                    
                    logger.warning(f"Account-level circuit breaker triggered: {event.description}")
                    return event
            
            return None
    
    def update_account_pnl(self, account_id: str, daily_pnl: float) -> Optional[CircuitBreakerEvent]:
        """Update account P&L directly and check for circuit breaker triggers.
        
        Args:
            account_id: Account identifier
            daily_pnl: Daily P&L for the account
            
        Returns:
            Optional[CircuitBreakerEvent]: Circuit breaker event if triggered, None otherwise
        """
        with self.lock:
            if account_id in self.account_circuit_breakers:
                circuit_breaker = self.account_circuit_breakers[account_id]
                
                # Update P&L
                circuit_breaker.update_daily_pnl(daily_pnl)
                
                # Check if circuit breaker is triggered
                event = circuit_breaker.check()
                if event:
                    # Update last triggered timestamp
                    circuit_breaker.last_triggered = datetime.datetime.now()
                    
                    # Add to active circuit breakers
                    self.active_circuit_breakers[event.id] = event
                    
                    # Notify emergency handler if available
                    if self.emergency_handler:
                        self._notify_emergency_handler(event)
                    
                    logger.warning(f"Account-level circuit breaker triggered: {event.description}")
                    return event
            
            return None
    
    def reset_circuit_breaker(self, circuit_breaker_id: str) -> bool:
        """Reset a circuit breaker.
        
        Args:
            circuit_breaker_id: Circuit breaker ID
            
        Returns:
            bool: True if reset, False if not found
        """
        with self.lock:
            if circuit_breaker_id in self.active_circuit_breakers:
                event = self.active_circuit_breakers[circuit_breaker_id]
                event.reset()
                self.circuit_breaker_history.append(event)
                del self.active_circuit_breakers[circuit_breaker_id]
                logger.info(f"Circuit breaker reset: {event.description}")
                return True
            return False
    
    def get_active_circuit_breakers(self) -> Dict[str, CircuitBreakerEvent]:
        """Get all active circuit breakers.
        
        Returns:
            Dict[str, CircuitBreakerEvent]: Dictionary of active circuit breaker events
        """
        with self.lock:
            return self.active_circuit_breakers.copy()
    
    def get_circuit_breaker_history(self, limit: Optional[int] = None) -> List[CircuitBreakerEvent]:
        """Get circuit breaker history.
        
        Args:
            limit: Optional limit on number of events to return
            
        Returns:
            List[CircuitBreakerEvent]: List of historical circuit breaker events
        """
        with self.lock:
            if limit is not None:
                return self.circuit_breaker_history[-limit:]
            return self.circuit_breaker_history.copy()
    
    def get_market_circuit_breakers(self) -> Dict[str, MarketWideCircuitBreaker]:
        """Get all market-wide circuit breakers.
        
        Returns:
            Dict[str, MarketWideCircuitBreaker]: Dictionary of market-wide circuit breakers
        """
        with self.lock:
            return self.market_circuit_breakers.copy()
    
    def get_account_circuit_breakers(self) -> Dict[str, AccountCircuitBreaker]:
        """Get all account-level circuit breakers.
        
        Returns:
            Dict[str, AccountCircuitBreaker]: Dictionary of account-level circuit breakers
        """
        with self.lock:
            return self.account_circuit_breakers.copy()
    
    def check_circuit_breakers(self, metrics: Dict[str, Any]) -> Dict[str, CircuitBreakerEvent]:
        """Check all circuit breakers against the provided metrics.
        
        Args:
            metrics: Dictionary of risk metrics to check against
            
        Returns:
            Dict[str, CircuitBreakerEvent]: Dictionary of triggered circuit breaker events
        """
        triggered_breakers = {}
        
        # Check market circuit breakers
        for market, cb in self.market_circuit_breakers.items():
            if market in metrics:
                event = cb.check(metrics[market])
                if event:
                    triggered_breakers[event.id] = event
                    self.active_circuit_breakers[event.id] = event
                    
                    # Notify emergency handler if available
                    if self.emergency_handler:
                        self._notify_emergency_handler(event)
        
        # Check account circuit breakers
        for account_id, cb in self.account_circuit_breakers.items():
            event = cb.check()
            if event:
                triggered_breakers[event.id] = event
                self.active_circuit_breakers[event.id] = event
                
                # Notify emergency handler if available
                if self.emergency_handler:
                    self._notify_emergency_handler(event)
        
        return triggered_breakers
    
    def _notify_emergency_handler(self, event: CircuitBreakerEvent) -> None:
        """Notify the emergency handler of a circuit breaker event.
        
        Args:
            event: Circuit breaker event
        """
        if not self.emergency_handler:
            return
        
        try:
            # Map circuit breaker level to emergency level
            emergency_level = EmergencyLevel.MEDIUM
            if event.level == CircuitBreakerLevel.LEVEL_3 or event.level == CircuitBreakerLevel.HARD_LIMIT:
                emergency_level = EmergencyLevel.HIGH
            elif event.level == CircuitBreakerLevel.WARNING:
                emergency_level = EmergencyLevel.LOW
            
            # Notify emergency handler
            self.emergency_handler._handle_circuit_breaker({
                "circuit_breaker_type": event.circuit_type.value,
                "level": event.level.value,
                "market": event.market,
                "symbol": event.symbol,
                "account_id": event.account_id,
                "details": event.description,
                "duration_minutes": event.duration_minutes,
                "metadata": event.metadata
            })
        except Exception as e:
            logger.error(f"Error notifying emergency handler: {str(e)}")


def create_circuit_breaker_manager(emergency_handler=None, circuit_breakers=None, emergency_handlers=None) -> CircuitBreakerManager:
    """Create a circuit breaker manager.
    
    Args:
        emergency_handler: Optional emergency handler to notify of circuit breaker events (deprecated)
        circuit_breakers: Optional list of circuit breakers to add
        emergency_handlers: Optional list of emergency handlers to notify of circuit breaker events
        
    Returns:
        CircuitBreakerManager: Initialized circuit breaker manager
    """
    manager = CircuitBreakerManager(
        circuit_breakers=circuit_breakers,
        emergency_handlers=emergency_handlers,
        emergency_handler=emergency_handler
    )
    manager.start()
    return manager


# Type aliases for backward compatibility
MarketCircuitBreaker = MarketWideCircuitBreaker