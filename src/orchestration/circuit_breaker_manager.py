"""Circuit Breaker Manager for Orchestration Layer.

This module provides a compatibility layer between the risk module's circuit breakers
and the orchestration layer's circuit breaker implementation.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable

# Import the actual implementation from risk module
from src.risk.circuit_breaker import (
    CircuitBreaker as RiskCircuitBreaker,
    MarketWideCircuitBreaker,
    AccountCircuitBreaker,
    CircuitBreakerManager as RiskCircuitBreakerManager,
    CircuitBreakerEvent,
    CircuitBreakerType as RiskCircuitBreakerType,
    CircuitBreakerLevel,
    CircuitBreakerStatus
)


class CircuitBreakerType(Enum):
    """Types of circuit breakers for orchestration layer."""
    MARKET = "market"  # Market-wide circuit breaker
    ACCOUNT = "account"  # Account-level circuit breaker
    STRATEGY = "strategy"  # Strategy-level circuit breaker


class CircuitBreaker:
    """Circuit breaker for orchestration layer.
    
    This class provides a compatibility layer between the orchestration layer's
    circuit breaker interface and the risk module's circuit breaker implementation.
    """
    
    def __init__(self, 
                name: str, 
                cb_type: CircuitBreakerType, 
                threshold: float, 
                description: str, 
                cooldown_minutes: int = 60):
        """Initialize a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            cb_type: Type of circuit breaker
            threshold: Threshold value for triggering
            description: Description of the circuit breaker
            cooldown_minutes: Cooldown period in minutes
        """
        self.name = name
        self.cb_type = cb_type
        self.threshold = threshold
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        
        # Map to the appropriate risk module circuit breaker type
        if cb_type == CircuitBreakerType.MARKET:
            self._risk_cb_type = RiskCircuitBreakerType.MARKET_WIDE
        elif cb_type == CircuitBreakerType.ACCOUNT:
            self._risk_cb_type = RiskCircuitBreakerType.ACCOUNT_LEVEL
        else:
            self._risk_cb_type = RiskCircuitBreakerType.SYMBOL_LEVEL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the circuit breaker to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.cb_type.value,
            "threshold": self.threshold,
            "description": self.description,
            "cooldown_minutes": self.cooldown_minutes
        }


class CircuitBreakerManager:
    """
    Manager for circuit breakers in the orchestration layer.
    
    This class provides a compatibility layer between the orchestration layer's
    circuit breaker manager interface and the risk module's circuit breaker manager.
    """
    
    def __init__(self, circuit_breakers=None, emergency_handlers=None):
        """
        Initialize the circuit breaker manager.
        
        Args:
            circuit_breakers: Optional list of circuit breakers to add
            emergency_handlers: Optional list of emergency handlers to notify of circuit breaker events
        """
        # Create the actual risk module circuit breaker manager
        self._risk_manager = RiskCircuitBreakerManager(circuit_breakers=[], emergency_handlers=emergency_handlers if emergency_handlers else [])
        self._risk_manager.start()
        
        # Add circuit breakers if provided
        if circuit_breakers:
            for cb in circuit_breakers:
                self.add_circuit_breaker(cb)
    
    def add_circuit_breaker(self, circuit_breaker: CircuitBreaker) -> None:
        """Add a circuit breaker to the manager.
        
        Args:
            circuit_breaker: Circuit breaker to add
        """
        if circuit_breaker.cb_type == CircuitBreakerType.MARKET:
            market_cb = MarketWideCircuitBreaker(
                market=circuit_breaker.name,
                level_1_percent=circuit_breaker.threshold,
                level_2_percent=circuit_breaker.threshold * 1.5,  # Scale up for higher levels
                level_3_percent=circuit_breaker.threshold * 2.0,
                level_1_duration_minutes=circuit_breaker.cooldown_minutes,
                level_2_duration_minutes=circuit_breaker.cooldown_minutes * 2,
                level_3_duration_minutes=circuit_breaker.cooldown_minutes * 3
            )
            self._risk_manager.add_market_circuit_breaker(market_cb)
        elif circuit_breaker.cb_type == CircuitBreakerType.ACCOUNT:
            account_cb = AccountCircuitBreaker(
                account_id=circuit_breaker.name,
                daily_loss_limit_percent=circuit_breaker.threshold,
                weekly_loss_limit_percent=circuit_breaker.threshold * 1.5,
                max_drawdown_percent=circuit_breaker.threshold * 2.0,
                initial_balance=100000.0  # Default initial balance
            )
            self._risk_manager.add_account_circuit_breaker(account_cb)
    
    def get_active_circuit_breakers(self) -> List[Dict[str, Any]]:
        """Get active circuit breakers.
        
        Returns:
            List[Dict[str, Any]]: List of active circuit breakers
        """
        return [cb.to_dict() for cb in self._risk_manager.get_active_circuit_breakers().values()]
    
    def reset_circuit_breaker(self, circuit_breaker_id: str) -> bool:
        """Reset a circuit breaker.
        
        Args:
            circuit_breaker_id: ID of the circuit breaker to reset
            
        Returns:
            bool: True if reset successful, False otherwise
        """
        return self._risk_manager.reset_circuit_breaker(circuit_breaker_id)