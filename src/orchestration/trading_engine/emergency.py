"""Emergency handling for the trading engine.

This module provides functionality for handling emergency situations
in the trading engine, including circuit breakers, risk breaches,
and system errors.
"""

import datetime
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
import threading
import time
import uuid

from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    CRITICAL = "critical"  # Severe emergency requiring immediate halt
    HIGH = "high"  # High severity emergency
    MEDIUM = "medium"  # Medium severity emergency
    LOW = "low"  # Low severity emergency


class EmergencyAction(Enum):
    """Actions to take in response to emergencies."""
    HALT_TRADING = "halt_trading"  # Stop all trading activity
    PAUSE_TRADING = "pause_trading"  # Pause new orders but allow existing ones to complete
    THROTTLE = "throttle"  # Reduce trading frequency/volume
    MONITOR = "monitor"  # Continue trading but monitor closely


class EmergencyTrigger(Enum):
    """Types of emergency triggers."""
    MARKET_DATA = "market_data"  # Market data issues
    CONNECTIVITY = "connectivity"  # Connectivity issues
    RISK_BREACH = "risk_breach"  # Risk limit breaches
    SYSTEM_ERROR = "system_error"  # System errors
    MANUAL = "manual"  # Manually triggered
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker triggered


class EmergencyEvent:
    """Represents an emergency event in the trading system."""
    
    def __init__(self, 
                trigger: EmergencyTrigger,
                level: EmergencyLevel,
                description: str,
                recommended_action: EmergencyAction,
                affected_markets: Optional[List[str]] = None,
                affected_symbols: Optional[List[str]] = None,
                affected_accounts: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                source: Optional[str] = None):
        """Initialize an emergency event.
        
        Args:
            trigger: What triggered the emergency
            level: Severity level of the emergency
            description: Human-readable description
            recommended_action: Recommended action to take
            affected_markets: List of affected markets
            affected_symbols: List of affected symbols
            affected_accounts: List of affected accounts
            metadata: Additional metadata about the emergency
            source: Source of the emergency (e.g., component name)
        """
        self.id = str(uuid.uuid4())
        self.trigger = trigger
        self.level = level
        self.description = description
        self.recommended_action = recommended_action
        self.affected_markets = affected_markets or []
        self.affected_symbols = affected_symbols or []
        self.affected_accounts = affected_accounts or []
        self.metadata = metadata or {}
        self.source = source
        self.timestamp = datetime.datetime.now()
        self.resolved = False
        self.resolution_timestamp = None
        self.resolution_description = None
    
    def resolve(self, description: Optional[str] = None) -> None:
        """Mark the emergency as resolved.
        
        Args:
            description: Optional description of the resolution
        """
        self.resolved = True
        self.resolution_timestamp = datetime.datetime.now()
        self.resolution_description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the emergency event to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the event
        """
        return {
            "id": self.id,
            "trigger": self.trigger.value,
            "level": self.level.value,
            "description": self.description,
            "recommended_action": self.recommended_action.value,
            "affected_markets": self.affected_markets,
            "affected_symbols": self.affected_symbols,
            "affected_accounts": self.affected_accounts,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            "resolution_description": self.resolution_description
        }


class EmergencyHandler:
    """Handles emergency situations in the trading engine."""
    
    def __init__(self):
        """Initialize the emergency handler."""
        self.active_emergencies: Dict[str, EmergencyEvent] = {}
        self.emergency_history: List[EmergencyEvent] = []
        self.current_level = EmergencyLevel.LOW
        self.trading_paused = False
        self.trading_halted = False
        
        # Event handlers
        self.market_data_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.connectivity_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.risk_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.system_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.manual_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.circuit_breaker_handlers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Heartbeat monitoring
        self.heartbeat_interval_seconds = 5
        self.heartbeat_timeout_seconds = 15
        self.last_heartbeat = datetime.datetime.now()
        self.heartbeat_thread = None
        self.heartbeat_running = False
    
    def start(self) -> None:
        """Start the emergency handler."""
        # Start heartbeat monitoring
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Emergency handler started")
    
    def stop(self) -> None:
        """Stop the emergency handler."""
        # Stop heartbeat monitoring
        self.heartbeat_running = False
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
        logger.info("Emergency handler stopped")
    
    def _heartbeat_monitor(self) -> None:
        """Monitor system heartbeat to detect issues."""
        while self.heartbeat_running:
            now = datetime.datetime.now()
            time_since_heartbeat = (now - self.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.heartbeat_timeout_seconds:
                # Heartbeat timeout - declare emergency
                logger.error(f"Heartbeat timeout: {time_since_heartbeat:.1f} seconds since last heartbeat")
                self._handle_system_error({
                    "error_type": "heartbeat_timeout",
                    "component": "trading_engine",
                    "severity": "high",
                    "details": f"No heartbeat received for {time_since_heartbeat:.1f} seconds"
                })
            
            time.sleep(1.0)
    
    def update_heartbeat(self) -> None:
        """Update the system heartbeat."""
        self.last_heartbeat = datetime.datetime.now()
    
    def register_market_data_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for market data emergencies.
        
        Args:
            handler: Function to call when a market data emergency occurs
        """
        self.market_data_handlers.append(handler)
    
    def register_connectivity_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for connectivity emergencies.
        
        Args:
            handler: Function to call when a connectivity emergency occurs
        """
        self.connectivity_handlers.append(handler)
    
    def register_risk_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for risk breach emergencies.
        
        Args:
            handler: Function to call when a risk breach emergency occurs
        """
        self.risk_handlers.append(handler)
    
    def register_system_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for system error emergencies.
        
        Args:
            handler: Function to call when a system error emergency occurs
        """
        self.system_handlers.append(handler)
    
    def register_manual_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for manual emergencies.
        
        Args:
            handler: Function to call when a manual emergency occurs
        """
        self.manual_handlers.append(handler)
    
    def register_circuit_breaker_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for circuit breaker emergencies.
        
        Args:
            handler: Function to call when a circuit breaker emergency occurs
        """
        self.circuit_breaker_handlers.append(handler)
    
    def declare_emergency(self, 
                         trigger: EmergencyTrigger,
                         level: EmergencyLevel,
                         description: str,
                         recommended_action: Optional[EmergencyAction] = None,
                         affected_markets: Optional[List[str]] = None,
                         affected_symbols: Optional[List[str]] = None,
                         affected_accounts: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         source: Optional[str] = None) -> EmergencyEvent:
        """Declare an emergency.
        
        Args:
            trigger: What triggered the emergency
            level: Severity level of the emergency
            description: Human-readable description
            recommended_action: Recommended action to take
            affected_markets: List of affected markets
            affected_symbols: List of affected symbols
            affected_accounts: List of affected accounts
            metadata: Additional metadata about the emergency
            source: Source of the emergency (e.g., component name)
            
        Returns:
            EmergencyEvent: The created emergency event
        """
        # Default recommended action based on level if not provided
        if recommended_action is None:
            if level == EmergencyLevel.CRITICAL:
                recommended_action = EmergencyAction.HALT_TRADING
            elif level == EmergencyLevel.HIGH:
                recommended_action = EmergencyAction.PAUSE_TRADING
            elif level == EmergencyLevel.MEDIUM:
                recommended_action = EmergencyAction.THROTTLE
            else:  # LOW
                recommended_action = EmergencyAction.MONITOR
        
        # Create emergency event
        event = EmergencyEvent(
            trigger=trigger,
            level=level,
            description=description,
            recommended_action=recommended_action,
            affected_markets=affected_markets,
            affected_symbols=affected_symbols,
            affected_accounts=affected_accounts,
            metadata=metadata,
            source=source
        )
        
        # Add to active emergencies
        self.active_emergencies[event.id] = event
        
        # Update current emergency level (use highest active level)
        self._update_emergency_level()
        
        # Take action based on recommended action
        if recommended_action == EmergencyAction.HALT_TRADING:
            self.trading_halted = True
            self.trading_paused = True
            logger.critical(f"TRADING HALTED: {description}")
        elif recommended_action == EmergencyAction.PAUSE_TRADING:
            self.trading_paused = True
            logger.error(f"TRADING PAUSED: {description}")
        elif recommended_action == EmergencyAction.THROTTLE:
            logger.warning(f"TRADING THROTTLED: {description}")
        else:  # MONITOR
            logger.info(f"MONITORING: {description}")
        
        # Call appropriate handlers based on trigger
        handler_data = {
            "event": event.to_dict(),
            "action": recommended_action.value,
            "level": level.value
        }
        
        if trigger == EmergencyTrigger.MARKET_DATA:
            for handler in self.market_data_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in market data emergency handler: {str(e)}")
        elif trigger == EmergencyTrigger.CONNECTIVITY:
            for handler in self.connectivity_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in connectivity emergency handler: {str(e)}")
        elif trigger == EmergencyTrigger.RISK_BREACH:
            for handler in self.risk_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in risk breach emergency handler: {str(e)}")
        elif trigger == EmergencyTrigger.SYSTEM_ERROR:
            for handler in self.system_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in system error emergency handler: {str(e)}")
        elif trigger == EmergencyTrigger.MANUAL:
            for handler in self.manual_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in manual emergency handler: {str(e)}")
        elif trigger == EmergencyTrigger.CIRCUIT_BREAKER:
            for handler in self.circuit_breaker_handlers:
                try:
                    handler(handler_data)
                except Exception as e:
                    logger.error(f"Error in circuit breaker emergency handler: {str(e)}")
        
        return event
    
    def resolve_emergency(self, 
                         event_id: Optional[str] = None,
                         trigger: Optional[EmergencyTrigger] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         description: Optional[str] = None) -> List[EmergencyEvent]:
        """Resolve an emergency or emergencies.
        
        Args:
            event_id: ID of the specific emergency to resolve
            trigger: Resolve all emergencies with this trigger
            metadata: Resolve emergencies matching this metadata
            description: Description of the resolution
            
        Returns:
            List[EmergencyEvent]: List of resolved emergency events
        """
        resolved_events = []
        
        # Resolve by ID if provided
        if event_id and event_id in self.active_emergencies:
            event = self.active_emergencies[event_id]
            event.resolve(description)
            resolved_events.append(event)
            self.emergency_history.append(event)
            del self.active_emergencies[event_id]
        
        # Resolve by trigger and/or metadata
        elif trigger or metadata:
            to_resolve = []
            for event_id, event in self.active_emergencies.items():
                if trigger and event.trigger != trigger:
                    continue
                
                if metadata:
                    # Check if all metadata key/value pairs match
                    match = True
                    for key, value in metadata.items():
                        if key not in event.metadata or event.metadata[key] != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                to_resolve.append(event_id)
            
            # Resolve matched events
            for event_id in to_resolve:
                event = self.active_emergencies[event_id]
                event.resolve(description)
                resolved_events.append(event)
                self.emergency_history.append(event)
                del self.active_emergencies[event_id]
        
        # Update current emergency level
        self._update_emergency_level()
        
        # Update trading state based on remaining emergencies
        self._update_trading_state()
        
        return resolved_events
    
    def _update_emergency_level(self) -> None:
        """Update the current emergency level based on active emergencies."""
        if not self.active_emergencies:
            self.current_level = EmergencyLevel.LOW
            return
        
        # Find highest level among active emergencies
        highest_level = EmergencyLevel.LOW
        for event in self.active_emergencies.values():
            if event.level == EmergencyLevel.CRITICAL:
                highest_level = EmergencyLevel.CRITICAL
                break
            elif event.level == EmergencyLevel.HIGH and highest_level != EmergencyLevel.CRITICAL:
                highest_level = EmergencyLevel.HIGH
            elif event.level == EmergencyLevel.MEDIUM and highest_level not in [EmergencyLevel.CRITICAL, EmergencyLevel.HIGH]:
                highest_level = EmergencyLevel.MEDIUM
        
        self.current_level = highest_level
    
    def _update_trading_state(self) -> None:
        """Update trading state based on active emergencies."""
        # Reset trading state
        self.trading_halted = False
        self.trading_paused = False
        
        # Check if any active emergency requires halting or pausing
        for event in self.active_emergencies.values():
            if event.recommended_action == EmergencyAction.HALT_TRADING:
                self.trading_halted = True
                self.trading_paused = True
                break
            elif event.recommended_action == EmergencyAction.PAUSE_TRADING:
                self.trading_paused = True
        
        if self.trading_halted:
            logger.warning("Trading remains HALTED due to active emergencies")
        elif self.trading_paused:
            logger.warning("Trading remains PAUSED due to active emergencies")
        else:
            logger.info("Trading resumed - all critical emergencies resolved")
    
    def get_active_emergencies(self) -> Dict[str, EmergencyEvent]:
        """Get all active emergencies.
        
        Returns:
            Dict[str, EmergencyEvent]: Dictionary of active emergency events
        """
        return self.active_emergencies
    
    def get_emergency_history(self, limit: Optional[int] = None) -> List[EmergencyEvent]:
        """Get emergency history.
        
        Args:
            limit: Optional limit on number of events to return
            
        Returns:
            List[EmergencyEvent]: List of historical emergency events
        """
        if limit is not None:
            return self.emergency_history[-limit:]
        return self.emergency_history
    
    def get_current_emergency_level(self) -> EmergencyLevel:
        """Get the current emergency level.
        
        Returns:
            EmergencyLevel: Current emergency level
        """
        return self.current_level
    
    def is_trading_paused(self) -> bool:
        """Check if trading is paused.
        
        Returns:
            bool: True if trading is paused, False otherwise
        """
        return self.trading_paused
    
    def is_trading_halted(self) -> bool:
        """Check if trading is halted.
        
        Returns:
            bool: True if trading is halted, False otherwise
        """
        return self.trading_halted
    
    def _handle_market_data_issue(self, data: Dict[str, Any]) -> None:
        """Handle market data issues.
        
        Args:
            data: Information about the market data issue
        """
        issue_type = data.get("issue_type")
        market = data.get("market")
        symbols = data.get("symbols", [])
        severity = data.get("severity", "medium").lower()
        details = data.get("details", "Market data issue detected")
        
        # Determine emergency level based on severity
        level = EmergencyLevel.MEDIUM
        if severity == "critical":
            level = EmergencyLevel.CRITICAL
        elif severity == "high":
            level = EmergencyLevel.HIGH
        elif severity == "low":
            level = EmergencyLevel.LOW
        
        # Determine recommended action based on severity
        action = EmergencyAction.THROTTLE
        if severity == "critical":
            action = EmergencyAction.HALT_TRADING
        elif severity == "high":
            action = EmergencyAction.PAUSE_TRADING
        elif severity == "low":
            action = EmergencyAction.MONITOR
        
        # Create description
        description = f"Market data issue: {issue_type}. "
        if market:
            description += f"Market: {market}. "
        if symbols:
            description += f"Affected symbols: {', '.join(symbols)}. "
        description += details
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.MARKET_DATA,
            level=level,
            description=description,
            recommended_action=action,
            affected_markets=[market] if market else None,
            affected_symbols=symbols,
            metadata=data,
            source="market_data_provider"
        )
    
    def _handle_broker_connection(self, data: Dict[str, Any]) -> None:
        """Handle broker connection issues.
        
        Args:
            data: Information about the broker connection issue
        """
        broker_id = data.get("broker_id")
        status = data.get("status")
        details = data.get("details", "Broker connection issue detected")
        
        # Determine emergency level and action based on status
        level = EmergencyLevel.HIGH
        action = EmergencyAction.PAUSE_TRADING
        
        if status == "disconnected":
            description = f"Broker connection lost: {broker_id}. {details}"
        elif status == "degraded":
            description = f"Broker connection degraded: {broker_id}. {details}"
            level = EmergencyLevel.MEDIUM
            action = EmergencyAction.THROTTLE
        elif status == "reconnected":
            # Resolve any existing broker connection emergencies for this broker
            self.resolve_emergency(
                trigger=EmergencyTrigger.CONNECTIVITY,
                metadata={"broker_id": broker_id},
                description=f"Broker connection restored: {broker_id}"
            )
            return
        else:
            description = f"Broker connection issue: {broker_id}, status: {status}. {details}"
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.CONNECTIVITY,
            level=level,
            description=description,
            recommended_action=action,
            metadata={
                "broker_id": broker_id,
                "status": status,
                **data
            },
            source="broker_connector"
        )
    
    def _handle_data_connection(self, data: Dict[str, Any]) -> None:
        """Handle data provider connection issues.
        
        Args:
            data: Information about the data connection issue
        """
        provider_id = data.get("provider_id")
        status = data.get("status")
        details = data.get("details", "Data provider connection issue detected")
        
        # Determine emergency level and action based on status
        level = EmergencyLevel.HIGH
        action = EmergencyAction.PAUSE_TRADING
        
        if status == "disconnected":
            description = f"Data provider connection lost: {provider_id}. {details}"
        elif status == "degraded":
            description = f"Data provider connection degraded: {provider_id}. {details}"
            level = EmergencyLevel.MEDIUM
            action = EmergencyAction.THROTTLE
        elif status == "reconnected":
            # Resolve any existing data connection emergencies for this provider
            self.resolve_emergency(
                trigger=EmergencyTrigger.CONNECTIVITY,
                metadata={"provider_id": provider_id},
                description=f"Data provider connection restored: {provider_id}"
            )
            return
        else:
            description = f"Data provider connection issue: {provider_id}, status: {status}. {details}"
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.CONNECTIVITY,
            level=level,
            description=description,
            recommended_action=action,
            metadata={
                "provider_id": provider_id,
                "status": status,
                **data
            },
            source="data_connector"
        )
    
    def _handle_risk_breach(self, data: Dict[str, Any]) -> None:
        """Handle risk limit breaches.
        
        Args:
            data: Information about the risk breach
        """
        breach_type = data.get("breach_type")
        severity = data.get("severity", "medium").lower()
        account_id = data.get("account_id")
        symbol = data.get("symbol")
        details = data.get("details", "Risk limit breach detected")
        
        # Determine emergency level based on severity
        level = EmergencyLevel.MEDIUM
        if severity == "critical":
            level = EmergencyLevel.CRITICAL
        elif severity == "high":
            level = EmergencyLevel.HIGH
        elif severity == "low":
            level = EmergencyLevel.LOW
        
        # Determine recommended action based on severity
        action = EmergencyAction.THROTTLE
        if severity == "critical":
            action = EmergencyAction.HALT_TRADING
        elif severity == "high":
            action = EmergencyAction.PAUSE_TRADING
        elif severity == "low":
            action = EmergencyAction.MONITOR
        
        # Create description
        description = f"Risk limit breach: {breach_type}. "
        if account_id:
            description += f"Account: {account_id}. "
        if symbol:
            description += f"Symbol: {symbol}. "
        description += details
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.RISK_BREACH,
            level=level,
            description=description,
            recommended_action=action,
            affected_accounts=[account_id] if account_id else None,
            affected_symbols=[symbol] if symbol else None,
            metadata=data,
            source="risk_manager"
        )
    
    def _handle_system_error(self, data: Dict[str, Any]) -> None:
        """Handle system errors.
        
        Args:
            data: Information about the system error
        """
        error_type = data.get("error_type")
        severity = data.get("severity", "medium").lower()
        component = data.get("component")
        details = data.get("details", "System error detected")
        
        # Determine emergency level based on severity
        level = EmergencyLevel.MEDIUM
        if severity == "critical":
            level = EmergencyLevel.CRITICAL
        elif severity == "high":
            level = EmergencyLevel.HIGH
        elif severity == "low":
            level = EmergencyLevel.LOW
        
        # Determine recommended action based on severity
        action = EmergencyAction.THROTTLE
        if severity == "critical":
            action = EmergencyAction.HALT_TRADING
        elif severity == "high":
            action = EmergencyAction.PAUSE_TRADING
        elif severity == "low":
            action = EmergencyAction.MONITOR
        
        # Create description
        description = f"System error: {error_type}. "
        if component:
            description += f"Component: {component}. "
        description += details
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.SYSTEM_ERROR,
            level=level,
            description=description,
            recommended_action=action,
            metadata=data,
            source=component or "system"
        )
    
    def _handle_circuit_breaker(self, data: Dict[str, Any]) -> None:
        """Handle circuit breaker triggers.
        
        Args:
            data: Information about the circuit breaker trigger
        """
        circuit_type = data.get("circuit_breaker_type")
        level = data.get("level")
        market = data.get("market")
        symbol = data.get("symbol")
        account_id = data.get("account_id")
        details = data.get("details", "Circuit breaker triggered")
        
        # Determine emergency level based on circuit breaker level
        emergency_level = EmergencyLevel.MEDIUM
        if level == "LEVEL_3":
            emergency_level = EmergencyLevel.CRITICAL
        elif level == "LEVEL_2" or level == "hard_limit":
            emergency_level = EmergencyLevel.HIGH
        elif level == "warning":
            emergency_level = EmergencyLevel.LOW
        
        # Determine recommended action based on circuit breaker level
        action = EmergencyAction.PAUSE_TRADING
        if level == "LEVEL_3":
            action = EmergencyAction.HALT_TRADING
        elif level == "LEVEL_1" or level == "soft_limit":
            action = EmergencyAction.THROTTLE
        elif level == "warning":
            action = EmergencyAction.MONITOR
        
        # Create description
        description = f"Circuit breaker triggered: {circuit_type}. "
        if market:
            description += f"Market: {market}. "
        if symbol:
            description += f"Symbol: {symbol}. "
        if account_id:
            description += f"Account: {account_id}. "
        description += details
        
        # Declare emergency
        self.declare_emergency(
            trigger=EmergencyTrigger.CIRCUIT_BREAKER,
            level=emergency_level,
            description=description,
            recommended_action=action,
            affected_markets=[market] if market else None,
            affected_symbols=[symbol] if symbol else None,
            affected_accounts=[account_id] if account_id else None,
            metadata=data,
            source="circuit_breaker_manager"
        )


def create_emergency_handler() -> EmergencyHandler:
    """Create an emergency handler.
    
    Returns:
        EmergencyHandler: Initialized emergency handler
    """
    return EmergencyHandler()