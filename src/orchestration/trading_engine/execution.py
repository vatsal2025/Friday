"""Execution Strategies for the Friday AI Trading System.

This module implements various execution strategies for order execution,
including TWAP, VWAP, and other algorithms to optimize trade execution.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
import time
import uuid
import threading

from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ExecutionStrategy:
    """Base class for execution strategies.
    
    This abstract class defines the interface for all execution strategies.
    """
    
    def __init__(self, order_manager=None, event_system: Optional[EventSystem] = None):
        """Initialize the execution strategy.
        
        Args:
            order_manager: The order manager for executing trades.
            event_system: The event system for publishing and subscribing to events.
        """
        self.order_manager = order_manager
        self.event_system = event_system
        self.active_executions = {}
    
    def execute(self, order_params: Dict[str, Any]) -> str:
        """Execute an order using this strategy.
        
        Args:
            order_params: The parameters for the order.
            
        Returns:
            str: The execution ID.
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def cancel(self, execution_id: str) -> bool:
        """Cancel an execution.
        
        Args:
            execution_id: The ID of the execution to cancel.
            
        Returns:
            bool: True if the execution was cancelled, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement cancel()")
    
    def get_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the status of an execution.
        
        Args:
            execution_id: The ID of the execution to get status for.
            
        Returns:
            Dict[str, Any]: The execution status.
        """
        return self.active_executions.get(execution_id, {"status": "unknown"})


class ImmediateExecution(ExecutionStrategy):
    """Immediate execution strategy.
    
    This strategy executes the entire order immediately as a market order.
    """
    
    def execute(self, order_params: Dict[str, Any]) -> str:
        """Execute an order immediately.
        
        Args:
            order_params: The parameters for the order.
            
        Returns:
            str: The execution ID.
        """
        if not self.order_manager:
            raise ValueError("Order manager is required for execution")
        
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution = {
            "id": execution_id,
            "strategy": "immediate",
            "status": "executing",
            "created_at": datetime.now().isoformat(),
            "order_params": order_params,
            "orders": []
        }
        
        self.active_executions[execution_id] = execution
        
        # Create market order
        try:
            if order_params.get("order_type", "MARKET").upper() == "MARKET":
                order = self.order_manager.create_market_order(
                    symbol=order_params["symbol"],
                    quantity=order_params["quantity"],
                    side=order_params["side"],
                    metadata={"execution_id": execution_id}
                )
            else:  # LIMIT order
                order = self.order_manager.create_limit_order(
                    symbol=order_params["symbol"],
                    quantity=order_params["quantity"],
                    side=order_params["side"],
                    price=order_params["price"],
                    metadata={"execution_id": execution_id}
                )
            
            execution["orders"].append(order["id"])
            execution["status"] = "completed"
            
            # Emit execution event if event system is available
            if self.event_system:
                self.event_system.emit(
                    event_type="execution_completed",
                    data=execution,
                    source="execution_strategy"
                )
            
            logger.info(f"Immediate execution completed: {execution_id}")
            
        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            logger.error(f"Error in immediate execution: {str(e)}")
            
            # Emit execution event if event system is available
            if self.event_system:
                self.event_system.emit(
                    event_type="execution_failed",
                    data=execution,
                    source="execution_strategy"
                )
        
        return execution_id
    
    def cancel(self, execution_id: str) -> bool:
        """Cancel an execution.
        
        Args:
            execution_id: The ID of the execution to cancel.
            
        Returns:
            bool: True if the execution was cancelled, False otherwise.
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Execution not found: {execution_id}")
            return False
        
        execution = self.active_executions[execution_id]
        
        # If already completed or failed, can't cancel
        if execution["status"] in ["completed", "failed", "cancelled"]:
            return False
        
        # Cancel all orders
        cancelled = True
        for order_id in execution["orders"]:
            if not self.order_manager.cancel_order(order_id):
                cancelled = False
        
        if cancelled:
            execution["status"] = "cancelled"
            execution["cancelled_at"] = datetime.now().isoformat()
            
            # Emit execution event if event system is available
            if self.event_system:
                self.event_system.emit(
                    event_type="execution_cancelled",
                    data=execution,
                    source="execution_strategy"
                )
            
            logger.info(f"Execution cancelled: {execution_id}")
        
        return cancelled


class TWAPExecution(ExecutionStrategy):
    """Time-Weighted Average Price (TWAP) execution strategy.
    
    This strategy splits the order into equal-sized chunks and executes them
    at regular intervals over a specified time period.
    """
    
    def execute(self, order_params: Dict[str, Any]) -> str:
        """Execute an order using TWAP strategy.
        
        Args:
            order_params: The parameters for the order, including:
                - symbol: The symbol to trade
                - quantity: The total quantity to trade
                - side: The side of the trade ("buy" or "sell")
                - duration_minutes: The duration of the execution in minutes
                - num_slices: The number of slices to split the order into
            
        Returns:
            str: The execution ID.
        """
        if not self.order_manager:
            raise ValueError("Order manager is required for execution")
        
        execution_id = str(uuid.uuid4())
        
        # Get TWAP-specific parameters
        duration_minutes = order_params.get("duration_minutes", 60)
        num_slices = order_params.get("num_slices", 10)
        
        # Calculate slice size and interval
        total_quantity = order_params["quantity"]
        slice_quantity = total_quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        # Create execution record
        execution = {
            "id": execution_id,
            "strategy": "twap",
            "status": "executing",
            "created_at": datetime.now().isoformat(),
            "order_params": order_params,
            "total_quantity": total_quantity,
            "executed_quantity": 0,
            "remaining_quantity": total_quantity,
            "num_slices": num_slices,
            "slices_executed": 0,
            "next_slice_time": datetime.now().isoformat(),
            "orders": [],
            "is_cancelled": False
        }
        
        self.active_executions[execution_id] = execution
        
        # Start execution thread
        thread = threading.Thread(
            target=self._execute_twap,
            args=(execution_id, order_params, slice_quantity, interval_seconds, num_slices)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started TWAP execution: {execution_id}")
        return execution_id
    
    def _execute_twap(self, execution_id: str, order_params: Dict[str, Any], 
                     slice_quantity: float, interval_seconds: float, num_slices: int) -> None:
        """Execute a TWAP order in a separate thread.
        
        Args:
            execution_id: The execution ID.
            order_params: The order parameters.
            slice_quantity: The quantity per slice.
            interval_seconds: The interval between slices in seconds.
            num_slices: The total number of slices.
        """
        execution = self.active_executions[execution_id]
        
        for i in range(num_slices):
            # Check if execution was cancelled
            if execution.get("is_cancelled", False):
                logger.info(f"TWAP execution cancelled: {execution_id}")
                break
            
            # Execute slice
            try:
                slice_order = self.order_manager.create_market_order(
                    symbol=order_params["symbol"],
                    quantity=slice_quantity,
                    side=order_params["side"],
                    metadata={
                        "execution_id": execution_id,
                        "slice_number": i + 1,
                        "total_slices": num_slices
                    }
                )
                
                # Update execution record
                execution["orders"].append(slice_order["id"])
                execution["executed_quantity"] += slice_quantity
                execution["remaining_quantity"] -= slice_quantity
                execution["slices_executed"] += 1
                
                # Update next slice time
                next_slice_time = datetime.now() + timedelta(seconds=interval_seconds)
                execution["next_slice_time"] = next_slice_time.isoformat()
                
                logger.info(f"TWAP slice {i+1}/{num_slices} executed: {execution_id}")
                
                # Emit execution update event if event system is available
                if self.event_system:
                    self.event_system.emit(
                        event_type="execution_update",
                        data=execution,
                        source="execution_strategy"
                    )
                
                # Wait for next slice, unless this is the last slice
                if i < num_slices - 1:
                    time.sleep(interval_seconds)
                
            except Exception as e:
                execution["status"] = "failed"
                execution["error"] = str(e)
                logger.error(f"Error in TWAP execution: {str(e)}")
                
                # Emit execution event if event system is available
                if self.event_system:
                    self.event_system.emit(
                        event_type="execution_failed",
                        data=execution,
                        source="execution_strategy"
                    )
                
                return
        
        # Mark execution as completed
        execution["status"] = "completed"
        execution["completed_at"] = datetime.now().isoformat()
        
        # Emit execution event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="execution_completed",
                data=execution,
                source="execution_strategy"
            )
        
        logger.info(f"TWAP execution completed: {execution_id}")
    
    def cancel(self, execution_id: str) -> bool:
        """Cancel a TWAP execution.
        
        Args:
            execution_id: The ID of the execution to cancel.
            
        Returns:
            bool: True if the execution was cancelled, False otherwise.
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Execution not found: {execution_id}")
            return False
        
        execution = self.active_executions[execution_id]
        
        # If already completed or failed, can't cancel
        if execution["status"] in ["completed", "failed", "cancelled"]:
            return False
        
        # Mark as cancelled to stop the execution thread
        execution["is_cancelled"] = True
        execution["status"] = "cancelled"
        execution["cancelled_at"] = datetime.now().isoformat()
        
        # Cancel all active orders
        for order_id in execution["orders"]:
            self.order_manager.cancel_order(order_id)
        
        # Emit execution event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="execution_cancelled",
                data=execution,
                source="execution_strategy"
            )
        
        logger.info(f"TWAP execution cancelled: {execution_id}")
        return True


class ExecutionFactory:
    """Factory for creating execution strategies.
    
    This class provides a centralized way to create execution strategies.
    """
    
    def __init__(self, order_manager=None, event_system: Optional[EventSystem] = None):
        """Initialize the execution factory.
        
        Args:
            order_manager: The order manager for executing trades.
            event_system: The event system for publishing and subscribing to events.
        """
        self.order_manager = order_manager
        self.event_system = event_system
        self.strategies = {}
        
        # Register default strategies
        self.register_strategy("immediate", ImmediateExecution)
        self.register_strategy("twap", TWAPExecution)
    
    def register_strategy(self, name: str, strategy_class) -> None:
        """Register an execution strategy.
        
        Args:
            name: The name of the strategy.
            strategy_class: The strategy class.
        """
        self.strategies[name] = strategy_class
        logger.info(f"Registered execution strategy: {name}")
    
    def create_strategy(self, name: str) -> ExecutionStrategy:
        """Create an execution strategy.
        
        Args:
            name: The name of the strategy to create.
            
        Returns:
            ExecutionStrategy: The created strategy.
        """
        if name not in self.strategies:
            raise ValueError(f"Unknown execution strategy: {name}")
        
        strategy_class = self.strategies[name]
        return strategy_class(self.order_manager, self.event_system)
    
    def get_available_strategies(self) -> List[str]:
        """Get the names of available execution strategies.
        
        Returns:
            List[str]: The names of available strategies.
        """
        return list(self.strategies.keys())


class MarketImpactEstimator:
    """Estimator for market impact of trades.
    
    This class provides methods to estimate the market impact of trades
    based on order size, market liquidity, and other factors.
    """
    
    def __init__(self):
        """Initialize the market impact estimator."""
        pass
    
    def estimate_impact(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Estimate the market impact of a trade.
        
        Args:
            symbol: The symbol to trade.
            quantity: The quantity to trade.
            side: The side of the trade ("buy" or "sell").
            
        Returns:
            Dict[str, Any]: The estimated market impact.
        """
        # This is a simplified implementation - in a real system, this would
        # involve more sophisticated modeling based on market microstructure,
        # order book depth, historical volatility, etc.
        
        # Placeholder implementation
        impact_bps = min(10, quantity / 1000)  # Simple linear model capped at 10 bps
        
        return {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "impact_bps": impact_bps,
            "estimated_slippage": impact_bps / 10000,  # Convert bps to decimal
            "timestamp": datetime.now().isoformat()
        }
    
    def recommend_strategy(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Recommend an execution strategy based on market impact.
        
        Args:
            symbol: The symbol to trade.
            quantity: The quantity to trade.
            side: The side of the trade ("buy" or "sell").
            
        Returns:
            Dict[str, Any]: The recommended strategy.
        """
        impact = self.estimate_impact(symbol, quantity, side)
        
        # Simple decision logic based on impact
        if impact["impact_bps"] < 2:  # Less than 2 bps impact
            strategy = "immediate"
            params = {}
        else:  # More than 2 bps impact
            strategy = "twap"
            # Scale duration and slices based on impact
            duration_minutes = min(60, int(impact["impact_bps"] * 10))  # 10 minutes per bps, max 60 minutes
            num_slices = min(20, int(impact["impact_bps"] * 2))  # 2 slices per bps, max 20 slices
            params = {
                "duration_minutes": duration_minutes,
                "num_slices": num_slices
            }
        
        return {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "recommended_strategy": strategy,
            "strategy_params": params,
            "estimated_impact": impact,
            "timestamp": datetime.now().isoformat()
        }