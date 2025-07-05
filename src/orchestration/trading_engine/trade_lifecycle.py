from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
import json
import os
import time
import datetime
from pathlib import Path
import logging


class TradeState(Enum):
    """Trade state enum."""
    CREATED = "created"                # Trade has been created but not submitted
    PENDING_SUBMIT = "pending_submit"  # Trade is pending submission to broker
    SUBMITTED = "submitted"            # Trade has been submitted to broker
    ACKNOWLEDGED = "acknowledged"      # Trade has been acknowledged by broker
    PARTIALLY_FILLED = "partially_filled"  # Trade has been partially filled
    FILLED = "filled"                  # Trade has been completely filled
    PENDING_CANCEL = "pending_cancel"  # Trade is pending cancellation
    CANCELLED = "cancelled"            # Trade has been cancelled
    REJECTED = "rejected"              # Trade has been rejected by broker
    EXPIRED = "expired"                # Trade has expired
    ERROR = "error"                    # Trade has encountered an error


class TradeEvent(Enum):
    """Trade event enum."""
    CREATE = "create"                  # Trade has been created
    SUBMIT = "submit"                  # Trade has been submitted to broker
    ACKNOWLEDGE = "acknowledge"        # Trade has been acknowledged by broker
    PARTIAL_FILL = "partial_fill"      # Trade has been partially filled
    FILL = "fill"                      # Trade has been completely filled
    CANCEL_REQUEST = "cancel_request"  # Cancel request has been initiated
    CANCEL = "cancel"                  # Trade has been cancelled
    REJECT = "reject"                  # Trade has been rejected by broker
    EXPIRE = "expire"                  # Trade has expired
    ERROR = "error"                    # Trade has encountered an error


class TradeReportType(Enum):
    """Trade report type enum."""
    EXECUTION = "execution"            # Execution report
    ORDER_STATUS = "order_status"      # Order status report
    POSITION = "position"              # Position report
    ACCOUNT = "account"                # Account report
    CUSTOM = "custom"                  # Custom report


@dataclass
class TradeTimeout:
    """Trade timeout configuration."""
    state: TradeState                   # Trade state
    timeout_seconds: int                # Timeout in seconds
    action: str                         # Action to take when timeout occurs
    escalation_level: int = 1           # Escalation level (1-5, with 5 being highest)
    notify: bool = True                 # Whether to notify when timeout occurs
    retry_count: int = 0                # Number of retries before taking action
    custom_handler: Optional[Callable] = None  # Custom handler function


@dataclass
class TradeTransition:
    """Trade state transition."""
    from_state: TradeState              # From state
    event: TradeEvent                   # Event
    to_state: TradeState                # To state
    validation_func: Optional[Callable] = None  # Validation function
    action_func: Optional[Callable] = None      # Action function
    is_allowed_in_production: bool = True       # Whether transition is allowed in production
    requires_approval: bool = False             # Whether transition requires approval
    timeout_seconds: int = 0                    # Timeout in seconds (0 = no timeout)


@dataclass
class TradeReport:
    """Trade report."""
    trade_id: str                       # Trade ID
    report_type: TradeReportType        # Report type
    timestamp: float                    # Timestamp
    data: Dict[str, Any]                # Report data
    source: str                         # Report source
    is_production: bool = False         # Whether report is from production


class TradeLifecycleManager:
    """Trade lifecycle manager.
    
    This class manages the lifecycle of trades, including state transitions,
    timeouts, and reporting.
    """
    def __init__(self, config_dir: Optional[str] = None, is_production: bool = False):
        """Initialize trade lifecycle manager.
        
        Args:
            config_dir: Directory containing configuration files
            is_production: Whether manager is running in production mode
        """
        self.is_production = is_production
        self.logger = logging.getLogger(__name__)
        
        # Set config directory
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "config")
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize state transitions
        self.transitions: List[TradeTransition] = []
        self._init_transitions()
        
        # Initialize timeouts
        self.timeouts: List[TradeTimeout] = []
        self._init_timeouts()
        
        # Initialize reports
        self.reports: List[TradeReport] = []
        self.report_handlers: Dict[TradeReportType, List[Callable]] = {}
        
        # Load configurations
        self._load_transitions()
        self._load_timeouts()
    
    def _init_transitions(self) -> None:
        """Initialize state transitions."""
        # Define standard transitions
        self.transitions = [
            # From CREATED
            TradeTransition(TradeState.CREATED, TradeEvent.SUBMIT, TradeState.PENDING_SUBMIT),
            TradeTransition(TradeState.CREATED, TradeEvent.CANCEL, TradeState.CANCELLED),
            TradeTransition(TradeState.CREATED, TradeEvent.REJECT, TradeState.REJECTED),
            TradeTransition(TradeState.CREATED, TradeEvent.ERROR, TradeState.ERROR),
            
            # From PENDING_SUBMIT
            TradeTransition(TradeState.PENDING_SUBMIT, TradeEvent.ACKNOWLEDGE, TradeState.ACKNOWLEDGED),
            TradeTransition(TradeState.PENDING_SUBMIT, TradeEvent.REJECT, TradeState.REJECTED),
            TradeTransition(TradeState.PENDING_SUBMIT, TradeEvent.ERROR, TradeState.ERROR),
            TradeTransition(TradeState.PENDING_SUBMIT, TradeEvent.EXPIRE, TradeState.EXPIRED),
            
            # From ACKNOWLEDGED
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.PARTIAL_FILL, TradeState.PARTIALLY_FILLED),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.FILL, TradeState.FILLED),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.CANCEL_REQUEST, TradeState.PENDING_CANCEL),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.CANCEL, TradeState.CANCELLED),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.REJECT, TradeState.REJECTED),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.EXPIRE, TradeState.EXPIRED),
            TradeTransition(TradeState.ACKNOWLEDGED, TradeEvent.ERROR, TradeState.ERROR),
            
            # From PARTIALLY_FILLED
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.PARTIAL_FILL, TradeState.PARTIALLY_FILLED),
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.FILL, TradeState.FILLED),
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.CANCEL_REQUEST, TradeState.PENDING_CANCEL),
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.CANCEL, TradeState.CANCELLED),
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.EXPIRE, TradeState.EXPIRED),
            TradeTransition(TradeState.PARTIALLY_FILLED, TradeEvent.ERROR, TradeState.ERROR),
            
            # From PENDING_CANCEL
            TradeTransition(TradeState.PENDING_CANCEL, TradeEvent.CANCEL, TradeState.CANCELLED),
            TradeTransition(TradeState.PENDING_CANCEL, TradeEvent.REJECT, TradeState.ACKNOWLEDGED),
            TradeTransition(TradeState.PENDING_CANCEL, TradeEvent.PARTIAL_FILL, TradeState.PARTIALLY_FILLED),
            TradeTransition(TradeState.PENDING_CANCEL, TradeEvent.FILL, TradeState.FILLED),
            TradeTransition(TradeState.PENDING_CANCEL, TradeEvent.ERROR, TradeState.ERROR),
            
            # Special transitions for production
            TradeTransition(TradeState.REJECTED, TradeEvent.SUBMIT, TradeState.PENDING_SUBMIT, 
                           is_allowed_in_production=False),
            TradeTransition(TradeState.CANCELLED, TradeEvent.SUBMIT, TradeState.PENDING_SUBMIT, 
                           is_allowed_in_production=False),
            TradeTransition(TradeState.ERROR, TradeEvent.SUBMIT, TradeState.PENDING_SUBMIT, 
                           is_allowed_in_production=False),
            
            # Emergency transitions (require approval in production)
            TradeTransition(TradeState.FILLED, TradeEvent.ERROR, TradeState.ERROR, 
                           requires_approval=True),
            TradeTransition(TradeState.CANCELLED, TradeEvent.ERROR, TradeState.ERROR, 
                           requires_approval=True),
            TradeTransition(TradeState.REJECTED, TradeEvent.ERROR, TradeState.ERROR, 
                           requires_approval=True),
            TradeTransition(TradeState.EXPIRED, TradeEvent.ERROR, TradeState.ERROR, 
                           requires_approval=True),
        ]
    
    def _init_timeouts(self) -> None:
        """Initialize timeouts."""
        # Define standard timeouts
        self.timeouts = [
            # Submission timeouts
            TradeTimeout(TradeState.PENDING_SUBMIT, 30, "retry", escalation_level=1),
            TradeTimeout(TradeState.PENDING_SUBMIT, 120, "error", escalation_level=3),
            
            # Acknowledgement timeouts
            TradeTimeout(TradeState.SUBMITTED, 15, "check_status", escalation_level=1),
            TradeTimeout(TradeState.SUBMITTED, 60, "error", escalation_level=3),
            
            # Cancellation timeouts
            TradeTimeout(TradeState.PENDING_CANCEL, 15, "check_status", escalation_level=1),
            TradeTimeout(TradeState.PENDING_CANCEL, 60, "force_cancel", escalation_level=4),
        ]
    
    def _load_transitions(self) -> None:
        """Load state transitions from configuration file."""
        transitions_file = os.path.join(self.config_dir, "trade_transitions.json")
        if os.path.exists(transitions_file):
            try:
                with open(transitions_file, "r") as f:
                    transitions_data = json.load(f)
                
                # Clear existing transitions
                self.transitions = []
                
                # Add transitions from file
                for t_data in transitions_data:
                    self.transitions.append(TradeTransition(
                        from_state=TradeState(t_data["from_state"]),
                        event=TradeEvent(t_data["event"]),
                        to_state=TradeState(t_data["to_state"]),
                        is_allowed_in_production=t_data.get("is_allowed_in_production", True),
                        requires_approval=t_data.get("requires_approval", False),
                        timeout_seconds=t_data.get("timeout_seconds", 0)
                    ))
                
                self.logger.info(f"Loaded {len(self.transitions)} trade transitions from {transitions_file}")
            except Exception as e:
                self.logger.error(f"Error loading trade transitions: {e}")
                # Initialize with default transitions
                self._init_transitions()
    
    def _load_timeouts(self) -> None:
        """Load timeouts from configuration file."""
        timeouts_file = os.path.join(self.config_dir, "trade_timeouts.json")
        if os.path.exists(timeouts_file):
            try:
                with open(timeouts_file, "r") as f:
                    timeouts_data = json.load(f)
                
                # Clear existing timeouts
                self.timeouts = []
                
                # Add timeouts from file
                for t_data in timeouts_data:
                    self.timeouts.append(TradeTimeout(
                        state=TradeState(t_data["state"]),
                        timeout_seconds=t_data["timeout_seconds"],
                        action=t_data["action"],
                        escalation_level=t_data.get("escalation_level", 1),
                        notify=t_data.get("notify", True),
                        retry_count=t_data.get("retry_count", 0)
                    ))
                
                self.logger.info(f"Loaded {len(self.timeouts)} trade timeouts from {timeouts_file}")
            except Exception as e:
                self.logger.error(f"Error loading trade timeouts: {e}")
                # Initialize with default timeouts
                self._init_timeouts()
    
    def _save_transitions(self) -> None:
        """Save state transitions to configuration file."""
        transitions_file = os.path.join(self.config_dir, "trade_transitions.json")
        try:
            transitions_data = []
            for t in self.transitions:
                transitions_data.append({
                    "from_state": t.from_state.value,
                    "event": t.event.value,
                    "to_state": t.to_state.value,
                    "is_allowed_in_production": t.is_allowed_in_production,
                    "requires_approval": t.requires_approval,
                    "timeout_seconds": t.timeout_seconds
                })
            
            with open(transitions_file, "w") as f:
                json.dump(transitions_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.transitions)} trade transitions to {transitions_file}")
        except Exception as e:
            self.logger.error(f"Error saving trade transitions: {e}")
    
    def _save_timeouts(self) -> None:
        """Save timeouts to configuration file."""
        timeouts_file = os.path.join(self.config_dir, "trade_timeouts.json")
        try:
            timeouts_data = []
            for t in self.timeouts:
                timeouts_data.append({
                    "state": t.state.value,
                    "timeout_seconds": t.timeout_seconds,
                    "action": t.action,
                    "escalation_level": t.escalation_level,
                    "notify": t.notify,
                    "retry_count": t.retry_count
                })
            
            with open(timeouts_file, "w") as f:
                json.dump(timeouts_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.timeouts)} trade timeouts to {timeouts_file}")
        except Exception as e:
            self.logger.error(f"Error saving trade timeouts: {e}")
    
    def get_valid_transitions(self, current_state: TradeState) -> List[Tuple[TradeEvent, TradeState]]:
        """Get valid transitions from current state.
        
        Args:
            current_state: Current trade state
            
        Returns:
            List of tuples containing valid events and resulting states
        """
        valid_transitions = []
        for t in self.transitions:
            if t.from_state == current_state:
                # Check if transition is allowed in production
                if self.is_production and not t.is_allowed_in_production:
                    continue
                
                valid_transitions.append((t.event, t.to_state))
        
        return valid_transitions
    
    def get_transition(self, from_state: TradeState, event: TradeEvent) -> Optional[TradeTransition]:
        """Get transition for given state and event.
        
        Args:
            from_state: Current trade state
            event: Trade event
            
        Returns:
            TradeTransition if found, None otherwise
        """
        for t in self.transitions:
            if t.from_state == from_state and t.event == event:
                # Check if transition is allowed in production
                if self.is_production and not t.is_allowed_in_production:
                    return None
                
                return t
        
        return None
    
    def add_transition(self, transition: TradeTransition) -> None:
        """Add a state transition.
        
        Args:
            transition: Trade state transition to add
        """
        # Remove existing transition if any
        self.remove_transition(transition.from_state, transition.event)
        
        # Add new transition
        self.transitions.append(transition)
        self._save_transitions()
    
    def remove_transition(self, from_state: TradeState, event: TradeEvent) -> bool:
        """Remove a state transition.
        
        Args:
            from_state: From state
            event: Event
            
        Returns:
            True if removed, False if not found
        """
        for i, t in enumerate(self.transitions):
            if t.from_state == from_state and t.event == event:
                del self.transitions[i]
                self._save_transitions()
                return True
        
        return False
    
    def get_timeout(self, state: TradeState) -> Optional[TradeTimeout]:
        """Get timeout for given state.
        
        Args:
            state: Trade state
            
        Returns:
            TradeTimeout if found, None otherwise
        """
        for t in self.timeouts:
            if t.state == state:
                return t
        
        return None
    
    def add_timeout(self, timeout: TradeTimeout) -> None:
        """Add a timeout.
        
        Args:
            timeout: Trade timeout to add
        """
        # Remove existing timeout if any
        self.remove_timeout(timeout.state)
        
        # Add new timeout
        self.timeouts.append(timeout)
        self._save_timeouts()
    
    def remove_timeout(self, state: TradeState) -> bool:
        """Remove a timeout.
        
        Args:
            state: Trade state
            
        Returns:
            True if removed, False if not found
        """
        for i, t in enumerate(self.timeouts):
            if t.state == state:
                del self.timeouts[i]
                self._save_timeouts()
                return True
        
        return False
    
    def add_report_handler(self, report_type: TradeReportType, handler: Callable) -> None:
        """Add a report handler.
        
        Args:
            report_type: Trade report type
            handler: Handler function
        """
        if report_type not in self.report_handlers:
            self.report_handlers[report_type] = []
        
        self.report_handlers[report_type].append(handler)
    
    def remove_report_handler(self, report_type: TradeReportType, handler: Callable) -> bool:
        """Remove a report handler.
        
        Args:
            report_type: Trade report type
            handler: Handler function
            
        Returns:
            True if removed, False if not found
        """
        if report_type not in self.report_handlers:
            return False
        
        try:
            self.report_handlers[report_type].remove(handler)
            return True
        except ValueError:
            return False
    
    def add_report(self, report: TradeReport) -> None:
        """Add a trade report.
        
        Args:
            report: Trade report to add
        """
        self.reports.append(report)
        
        # Call handlers
        if report.report_type in self.report_handlers:
            for handler in self.report_handlers[report.report_type]:
                try:
                    handler(report)
                except Exception as e:
                    self.logger.error(f"Error in report handler: {e}")
    
    def get_reports(self, trade_id: str, report_type: Optional[TradeReportType] = None) -> List[TradeReport]:
        """Get trade reports.
        
        Args:
            trade_id: Trade ID
            report_type: Report type filter (optional)
            
        Returns:
            List of trade reports
        """
        if report_type is None:
            return [r for r in self.reports if r.trade_id == trade_id]
        else:
            return [r for r in self.reports if r.trade_id == trade_id and r.report_type == report_type]
    
    def clear_reports(self, trade_id: Optional[str] = None) -> None:
        """Clear trade reports.
        
        Args:
            trade_id: Trade ID (optional, if None, clear all reports)
        """
        if trade_id is None:
            self.reports = []
        else:
            self.reports = [r for r in self.reports if r.trade_id != trade_id]


def create_trade_lifecycle_manager(config_dir: Optional[str] = None, is_production: bool = False) -> TradeLifecycleManager:
    """Create a trade lifecycle manager.
    
    Args:
        config_dir: Directory containing configuration files
        is_production: Whether manager is running in production mode
        
    Returns:
        TradeLifecycleManager instance
    """
    return TradeLifecycleManager(config_dir, is_production)


def get_production_trade_lifecycle_manager(config_dir: Optional[str] = None) -> TradeLifecycleManager:
    """Get a production trade lifecycle manager.
    
    This function creates a trade lifecycle manager with production settings.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        TradeLifecycleManager instance configured for production
    """
    manager = create_trade_lifecycle_manager(config_dir, is_production=True)
    
    # Add production-specific timeouts
    manager.add_timeout(TradeTimeout(TradeState.PENDING_SUBMIT, 15, "retry", escalation_level=1, retry_count=2))
    manager.add_timeout(TradeTimeout(TradeState.PENDING_SUBMIT, 60, "error", escalation_level=3))
    manager.add_timeout(TradeTimeout(TradeState.SUBMITTED, 10, "check_status", escalation_level=1))
    manager.add_timeout(TradeTimeout(TradeState.SUBMITTED, 30, "error", escalation_level=3))
    manager.add_timeout(TradeTimeout(TradeState.ACKNOWLEDGED, 300, "check_status", escalation_level=1))
    manager.add_timeout(TradeTimeout(TradeState.ACKNOWLEDGED, 1800, "cancel_request", escalation_level=2))
    manager.add_timeout(TradeTimeout(TradeState.PARTIALLY_FILLED, 300, "check_status", escalation_level=1))
    manager.add_timeout(TradeTimeout(TradeState.PARTIALLY_FILLED, 1800, "cancel_request", escalation_level=2))
    manager.add_timeout(TradeTimeout(TradeState.PENDING_CANCEL, 10, "check_status", escalation_level=1, retry_count=3))
    manager.add_timeout(TradeTimeout(TradeState.PENDING_CANCEL, 30, "force_cancel", escalation_level=4))
    
    return manager


def validate_trade_transition(current_state: TradeState, event: TradeEvent, 
                             manager: Optional[TradeLifecycleManager] = None) -> bool:
    """Validate a trade state transition.
    
    Args:
        current_state: Current trade state
        event: Trade event
        manager: Trade lifecycle manager (optional)
        
    Returns:
        True if transition is valid, False otherwise
    """
    if manager is None:
        manager = create_trade_lifecycle_manager()
    
    transition = manager.get_transition(current_state, event)
    return transition is not None


def get_next_trade_state(current_state: TradeState, event: TradeEvent, 
                        manager: Optional[TradeLifecycleManager] = None) -> Optional[TradeState]:
    """Get next trade state.
    
    Args:
        current_state: Current trade state
        event: Trade event
        manager: Trade lifecycle manager (optional)
        
    Returns:
        Next trade state if transition is valid, None otherwise
    """
    if manager is None:
        manager = create_trade_lifecycle_manager()
    
    transition = manager.get_transition(current_state, event)
    if transition is not None:
        return transition.to_state
    
    return None