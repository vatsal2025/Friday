"""Trade Lifecycle Management for the Friday AI Trading System.

This module implements trade lifecycle management, tracking trades from
signal generation through execution and settlement.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import uuid
import json

from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class TradeState:
    """Enumeration of trade states in the lifecycle."""
    SIGNAL_GENERATED = "signal_generated"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TRADE_COMPLETED = "trade_completed"
    TRADE_FAILED = "trade_failed"


class TradeLifecycleManager:
    """Manager for trade lifecycle tracking and reporting.
    
    This class tracks trades from signal generation through execution and settlement,
    providing a complete audit trail and reporting capabilities.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the trade lifecycle manager.
        
        Args:
            event_system: The event system for publishing and subscribing to events.
        """
        self.event_system = event_system
        self.trades = {}
        self.trade_history = {}
        
        # Register event handlers if event system is provided
        if self.event_system:
            self.event_system.register_handler(
                callback=self._handle_trade_signal,
                event_types=["trade_signal"]
            )
            self.event_system.register_handler(
                callback=self._handle_order_event,
                event_types=["order_created", "order_submitted", "order_filled", 
                             "order_cancelled", "order_rejected"]
            )
            self.event_system.register_handler(
                callback=self._handle_execution_event,
                event_types=["execution_completed", "execution_failed", "execution_cancelled"]
            )
    
    def create_trade(self, signal_data: Dict[str, Any]) -> str:
        """Create a new trade from a signal.
        
        Args:
            signal_data: The signal data.
            
        Returns:
            str: The trade ID.
        """
        trade_id = str(uuid.uuid4())
        
        # Create trade record
        trade = {
            "id": trade_id,
            "symbol": signal_data.get("symbol"),
            "direction": signal_data.get("direction"),
            "state": TradeState.SIGNAL_GENERATED,
            "signal_id": signal_data.get("id"),
            "signal_confidence": signal_data.get("confidence"),
            "signal_source": signal_data.get("source"),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "orders": [],
            "executions": [],
            "history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "state": TradeState.SIGNAL_GENERATED,
                    "data": signal_data
                }
            ]
        }
        
        self.trades[trade_id] = trade
        self.trade_history[trade_id] = []
        self._add_history_entry(trade_id, TradeState.SIGNAL_GENERATED, signal_data)
        
        logger.info(f"Created trade {trade_id} from signal {signal_data.get('id')}")
        return trade_id
    
    def update_trade_state(self, trade_id: str, state: str, data: Dict[str, Any]) -> bool:
        """Update the state of a trade.
        
        Args:
            trade_id: The ID of the trade to update.
            state: The new state of the trade.
            data: Additional data for the state update.
            
        Returns:
            bool: True if the trade was updated, False otherwise.
        """
        if trade_id not in self.trades:
            logger.warning(f"Trade not found: {trade_id}")
            return False
        
        trade = self.trades[trade_id]
        
        # Update trade state
        trade["state"] = state
        trade["updated_at"] = datetime.now().isoformat()
        
        # Add history entry
        self._add_history_entry(trade_id, state, data)
        
        # Update trade-specific fields based on state
        if state == TradeState.ORDER_CREATED:
            order_id = data.get("id")
            if order_id and order_id not in trade["orders"]:
                trade["orders"].append(order_id)
        
        elif state == TradeState.ORDER_FILLED:
            trade["fill_price"] = data.get("fill_price")
            trade["fill_quantity"] = data.get("fill_quantity")
            trade["filled_at"] = data.get("filled_at") or datetime.now().isoformat()
        
        elif state in [TradeState.TRADE_COMPLETED, TradeState.TRADE_FAILED]:
            trade["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Updated trade {trade_id} to state {state}")
        return True
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a trade by ID.
        
        Args:
            trade_id: The ID of the trade to get.
            
        Returns:
            Dict[str, Any]: The trade data, or None if not found.
        """
        return self.trades.get(trade_id)
    
    def get_trade_history(self, trade_id: str) -> List[Dict[str, Any]]:
        """Get the history of a trade.
        
        Args:
            trade_id: The ID of the trade to get history for.
            
        Returns:
            List[Dict[str, Any]]: The trade history, or empty list if not found.
        """
        return self.trade_history.get(trade_id, [])
    
    def get_trades_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get trades for a symbol.
        
        Args:
            symbol: The symbol to get trades for.
            
        Returns:
            List[Dict[str, Any]]: The trades for the symbol.
        """
        return [trade for trade in self.trades.values() if trade["symbol"] == symbol]
    
    def get_trades_by_state(self, state: str) -> List[Dict[str, Any]]:
        """Get trades in a specific state.
        
        Args:
            state: The state to get trades for.
            
        Returns:
            List[Dict[str, Any]]: The trades in the state.
        """
        return [trade for trade in self.trades.values() if trade["state"] == state]
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get active trades (not completed or failed).
        
        Returns:
            List[Dict[str, Any]]: The active trades.
        """
        return [trade for trade in self.trades.values() 
                if trade["state"] not in [TradeState.TRADE_COMPLETED, TradeState.TRADE_FAILED]]
    
    def _add_history_entry(self, trade_id: str, state: str, data: Dict[str, Any]) -> None:
        """Add an entry to the trade history.
        
        Args:
            trade_id: The ID of the trade.
            state: The state of the trade.
            data: Additional data for the history entry.
        """
        if trade_id not in self.trade_history:
            self.trade_history[trade_id] = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "data": data
        }
        
        self.trade_history[trade_id].append(entry)
        
        # Also add to the trade record for convenience
        if trade_id in self.trades:
            if "history" not in self.trades[trade_id]:
                self.trades[trade_id]["history"] = []
            self.trades[trade_id]["history"].append(entry)
    
    def _handle_trade_signal(self, event: Event) -> None:
        """Handle a trade signal event.
        
        Args:
            event: The trade signal event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Trade signal event has no data")
            return
        
        signal = event.data
        
        # Create a new trade from the signal
        self.create_trade(signal)
    
    def _handle_order_event(self, event: Event) -> None:
        """Handle an order event.
        
        Args:
            event: The order event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Order event has no data")
            return
        
        order = event.data
        metadata = order.get("metadata", {})
        signal_id = metadata.get("signal_id")
        
        if not signal_id:
            logger.warning(f"Order {order.get('id')} has no signal ID in metadata")
            return
        
        # Find trades associated with this signal
        trades = [trade for trade in self.trades.values() if trade.get("signal_id") == signal_id]
        
        if not trades:
            logger.warning(f"No trades found for signal {signal_id}")
            return
        
        # Update each trade
        for trade in trades:
            trade_id = trade["id"]
            
            if event.event_type == "order_created":
                self.update_trade_state(trade_id, TradeState.ORDER_CREATED, order)
            
            elif event.event_type == "order_submitted":
                self.update_trade_state(trade_id, TradeState.ORDER_SUBMITTED, order)
            
            elif event.event_type == "order_filled":
                self.update_trade_state(trade_id, TradeState.ORDER_FILLED, order)
                
                # Check if all orders are filled to mark trade as completed
                all_filled = True
                for order_id in trade["orders"]:
                    order_data = order if order.get("id") == order_id else None
                    if not order_data or order_data.get("status") != "filled":
                        all_filled = False
                        break
                
                if all_filled:
                    self.update_trade_state(trade_id, TradeState.TRADE_COMPLETED, {
                        "message": "All orders filled",
                        "orders": trade["orders"]
                    })
            
            elif event.event_type == "order_cancelled":
                self.update_trade_state(trade_id, TradeState.ORDER_CANCELLED, order)
            
            elif event.event_type == "order_rejected":
                self.update_trade_state(trade_id, TradeState.ORDER_REJECTED, order)
                
                # Mark trade as failed if order was rejected
                self.update_trade_state(trade_id, TradeState.TRADE_FAILED, {
                    "message": "Order rejected",
                    "reason": order.get("rejection_reason"),
                    "order_id": order.get("id")
                })
    
    def _handle_execution_event(self, event: Event) -> None:
        """Handle an execution event.
        
        Args:
            event: The execution event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Execution event has no data")
            return
        
        execution = event.data
        orders = execution.get("orders", [])
        
        # Find trades associated with these orders
        for order_id in orders:
            trades = [trade for trade in self.trades.values() if order_id in trade.get("orders", [])]
            
            for trade in trades:
                trade_id = trade["id"]
                
                # Add execution to trade
                if "executions" not in trade:
                    trade["executions"] = []
                
                if execution.get("id") not in trade["executions"]:
                    trade["executions"].append(execution.get("id"))
                
                # Update trade state based on execution event
                if event.event_type == "execution_completed":
                    # Only update if all orders in the trade are part of this execution
                    if set(trade["orders"]).issubset(set(orders)):
                        self.update_trade_state(trade_id, TradeState.TRADE_COMPLETED, {
                            "message": "Execution completed",
                            "execution_id": execution.get("id")
                        })
                
                elif event.event_type == "execution_failed":
                    # Mark trade as failed if execution failed
                    self.update_trade_state(trade_id, TradeState.TRADE_FAILED, {
                        "message": "Execution failed",
                        "reason": execution.get("error"),
                        "execution_id": execution.get("id")
                    })
                
                elif event.event_type == "execution_cancelled":
                    # Only update if all orders in the trade are part of this execution
                    if set(trade["orders"]).issubset(set(orders)):
                        self.update_trade_state(trade_id, TradeState.ORDER_CANCELLED, {
                            "message": "Execution cancelled",
                            "execution_id": execution.get("id")
                        })


class TradeReporter:
    """Reporter for generating trade reports.
    
    This class provides methods for generating various reports on trades
    and their performance.
    """
    
    def __init__(self, lifecycle_manager: TradeLifecycleManager):
        """Initialize the trade reporter.
        
        Args:
            lifecycle_manager: The trade lifecycle manager.
        """
        self.lifecycle_manager = lifecycle_manager
    
    def generate_trade_summary(self, trade_id: str) -> Dict[str, Any]:
        """Generate a summary for a specific trade.
        
        Args:
            trade_id: The ID of the trade to summarize.
            
        Returns:
            Dict[str, Any]: The trade summary.
        """
        trade = self.lifecycle_manager.get_trade(trade_id)
        if not trade:
            return {"error": f"Trade not found: {trade_id}"}
        
        history = self.lifecycle_manager.get_trade_history(trade_id)
        
        # Calculate duration
        created_at = datetime.fromisoformat(trade["created_at"])
        completed_at = None
        if "completed_at" in trade:
            completed_at = datetime.fromisoformat(trade["completed_at"])
            duration_seconds = (completed_at - created_at).total_seconds()
        else:
            duration_seconds = (datetime.now() - created_at).total_seconds()
        
        # Calculate performance metrics if trade is completed
        performance = {}
        if trade["state"] == TradeState.TRADE_COMPLETED and "fill_price" in trade:
            # This is a simplified implementation - in a real system, this would
            # involve more sophisticated performance calculation
            signal_price = None
            for entry in history:
                if entry["state"] == TradeState.SIGNAL_GENERATED:
                    signal_data = entry["data"]
                    if "price" in signal_data:
                        signal_price = signal_data["price"]
                    break
            
            if signal_price:
                if trade["direction"] == "buy":
                    performance["price_change"] = trade["fill_price"] - signal_price
                    performance["price_change_pct"] = (trade["fill_price"] / signal_price - 1) * 100
                else:  # sell
                    performance["price_change"] = signal_price - trade["fill_price"]
                    performance["price_change_pct"] = (signal_price / trade["fill_price"] - 1) * 100
        
        return {
            "trade_id": trade_id,
            "symbol": trade["symbol"],
            "direction": trade["direction"],
            "state": trade["state"],
            "created_at": trade["created_at"],
            "completed_at": trade.get("completed_at"),
            "duration_seconds": duration_seconds,
            "signal_confidence": trade["signal_confidence"],
            "signal_source": trade["signal_source"],
            "fill_price": trade.get("fill_price"),
            "fill_quantity": trade.get("fill_quantity"),
            "orders": len(trade["orders"]),
            "performance": performance
        }
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a daily report for a specific date.
        
        Args:
            date: The date to generate the report for. If None, uses today.
            
        Returns:
            Dict[str, Any]: The daily report.
        """
        if date is None:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        next_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        next_date = next_date.replace(day=next_date.day + 1)
        
        # Filter trades for the specified date
        trades = []
        for trade in self.lifecycle_manager.trades.values():
            created_at = datetime.fromisoformat(trade["created_at"])
            if date <= created_at < next_date:
                trades.append(trade)
        
        # Calculate summary statistics
        total_trades = len(trades)
        completed_trades = len([t for t in trades if t["state"] == TradeState.TRADE_COMPLETED])
        failed_trades = len([t for t in trades if t["state"] == TradeState.TRADE_FAILED])
        active_trades = total_trades - completed_trades - failed_trades
        
        buy_trades = len([t for t in trades if t["direction"] == "buy"])
        sell_trades = len([t for t in trades if t["direction"] == "sell"])
        
        symbols = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in symbols:
                symbols[symbol] = 0
            symbols[symbol] += 1
        
        return {
            "date": date.isoformat(),
            "total_trades": total_trades,
            "completed_trades": completed_trades,
            "failed_trades": failed_trades,
            "active_trades": active_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "symbols": symbols,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_performance_report(self, start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a performance report for a date range.
        
        Args:
            start_date: The start date for the report. If None, uses 30 days ago.
            end_date: The end date for the report. If None, uses today.
            
        Returns:
            Dict[str, Any]: The performance report.
        """
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(day=start_date.day - 30)
        
        if end_date is None:
            end_date = datetime.now()
        
        # Filter completed trades for the specified date range
        trades = []
        for trade in self.lifecycle_manager.trades.values():
            if trade["state"] != TradeState.TRADE_COMPLETED or "completed_at" not in trade:
                continue
            
            completed_at = datetime.fromisoformat(trade["completed_at"])
            if start_date <= completed_at <= end_date:
                trades.append(trade)
        
        # Calculate performance metrics
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_trades": 0,
                "message": "No completed trades in the specified date range",
                "generated_at": datetime.now().isoformat()
            }
        
        # Calculate win/loss ratio and other metrics
        # This is a simplified implementation - in a real system, this would
        # involve more sophisticated performance calculation
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        for trade in trades:
            if "performance" in trade and "price_change" in trade["performance"]:
                price_change = trade["performance"]["price_change"]
                if price_change > 0:
                    winning_trades += 1
                    total_pnl += price_change
                else:
                    losing_trades += 1
                    total_pnl += price_change
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else float('inf')
        
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_pnl": total_pnl,
            "generated_at": datetime.now().isoformat()
        }
    
    def export_trades_to_json(self, file_path: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> bool:
        """Export trades to a JSON file.
        
        Args:
            file_path: The path to the output file.
            start_date: The start date for the export. If None, exports all trades.
            end_date: The end date for the export. If None, uses today.
            
        Returns:
            bool: True if the export was successful, False otherwise.
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Filter trades for the specified date range
        trades_to_export = []
        for trade in self.lifecycle_manager.trades.values():
            created_at = datetime.fromisoformat(trade["created_at"])
            if start_date is None or (start_date <= created_at <= end_date):
                trades_to_export.append(trade)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(trades_to_export, f, indent=2)
            logger.info(f"Exported {len(trades_to_export)} trades to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting trades to {file_path}: {str(e)}")
            return False