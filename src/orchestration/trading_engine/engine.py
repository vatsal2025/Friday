"""Trading Engine for the Friday AI Trading System.

This module implements the core trading engine functionality for signal generation
and order management, integrating model predictions with trading rules.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import uuid

from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class SignalGenerator:
    """Signal generation system that integrates model predictions and trading rules.
    
    This class is responsible for generating trading signals based on model predictions
    and applying trading rules to filter and enhance signals.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None):
        """Initialize the signal generator.
        
        Args:
            event_system: The event system for publishing and subscribing to events.
        """
        self.event_system = event_system
        self.rules = []
        self.signal_handlers = []
        
        # Register event handlers if event system is provided
        if self.event_system:
            self.event_system.register_handler(
                callback=self._handle_model_prediction,
                event_types=["model_prediction"]
            )
    
    def add_rule(self, rule_func: Callable[[Dict[str, Any]], bool], name: str = None) -> None:
        """Add a trading rule to the signal generator.
        
        Args:
            rule_func: A function that takes a signal and returns True if the signal
                      passes the rule, False otherwise.
            name: Optional name for the rule.
        """
        rule_id = name or f"rule_{len(self.rules) + 1}"
        self.rules.append({"id": rule_id, "func": rule_func})
        logger.info(f"Added trading rule: {rule_id}")
    
    def add_signal_handler(self, handler_func: Callable[[Dict[str, Any]], None], name: str = None) -> None:
        """Add a signal handler to the signal generator.
        
        Args:
            handler_func: A function that takes a signal and performs an action.
            name: Optional name for the handler.
        """
        handler_id = name or f"handler_{len(self.signal_handlers) + 1}"
        self.signal_handlers.append({"id": handler_id, "func": handler_func})
        logger.info(f"Added signal handler: {handler_id}")
    
    def generate_signal(self, prediction: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate a trading signal from a model prediction.
        
        Args:
            prediction: The model prediction data.
            confidence: The confidence level of the prediction.
            
        Returns:
            Dict[str, Any]: The generated trading signal data, or None if no signal is generated.
        """
        # Extract relevant information from the prediction
        symbol = prediction.get("symbol")
        prediction_type = prediction.get("type")
        predicted_value = prediction.get("value")
        prediction_horizon = prediction.get("horizon", "short_term")
        
        if not symbol or not prediction_type or predicted_value is None:
            logger.warning("Prediction missing required fields")
            return None
        
        # Determine signal direction based on prediction type and value
        signal_direction = None
        if prediction_type == "price_direction":
            signal_direction = "buy" if predicted_value > 0.5 else "sell"
        elif prediction_type == "return":
            signal_direction = "buy" if predicted_value > 0 else "sell"
        elif prediction_type == "volatility":
            signal_direction = "hold" if predicted_value > 0.2 else "buy"
        else:
            logger.warning(f"Unknown prediction type: {prediction_type}")
            return None
        
        # Apply confidence threshold adjustment
        signal_strength = min(1.0, confidence * 1.2)  # Slight boost to confidence
        
        # Create the signal data
        signal_data = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "direction": signal_direction,
            "confidence": signal_strength,
            "source": "model_prediction",
            "prediction_type": prediction_type,
            "prediction_value": predicted_value,
            "prediction_horizon": prediction_horizon,
            "timestamp": prediction.get("timestamp") or datetime.now().isoformat(),
            "metadata": {
                "original_confidence": confidence,
                "prediction_id": prediction.get("id"),
                "model_id": prediction.get("model_id"),
            }
        }
        
        # Apply trading rules
        for rule in self.rules:
            try:
                if not rule["func"](signal_data):
                    logger.info(f"Signal rejected by rule {rule['id']}")
                    return None
            except Exception as e:
                logger.error(f"Error applying rule {rule['id']}: {str(e)}")
                return None
        
        logger.info(f"Generated {signal_direction} signal for {symbol} with confidence {signal_strength}")
        return signal_data
    
    def process_signal(self, signal: Dict[str, Any]) -> None:
        """Process a trading signal by applying handlers and emitting events.
        
        Args:
            signal: The trading signal data.
        """
        # Apply signal handlers
        for handler in self.signal_handlers:
            try:
                handler["func"](signal)
            except Exception as e:
                logger.error(f"Error in signal handler {handler['id']}: {str(e)}")
        
        # Emit signal event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="trade_signal",
                data=signal,
                source="signal_generator"
            )
    
    def _handle_model_prediction(self, event: Event) -> None:
        """Handle a model prediction event.
        
        Args:
            event: The model prediction event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"Model prediction event has no data")
            return
        
        prediction = event.data
        confidence = prediction.get("confidence", 0.5)
        
        # Generate signal from prediction
        signal = self.generate_signal(prediction, confidence)
        
        # Process signal if generated
        if signal:
            self.process_signal(signal)


class OrderManager:
    """Order management system for handling trade execution.
    
    This class is responsible for creating, tracking, and managing orders
    based on trading signals.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None, broker_service=None):
        """Initialize the order manager.
        
        Args:
            event_system: The event system for publishing and subscribing to events.
            broker_service: The broker service for executing trades.
        """
        self.event_system = event_system
        self.broker_service = broker_service
        self.orders = {}
        self.active_orders = {}
        
        # Register event handlers if event system is provided
        if self.event_system:
            self.event_system.register_handler(
                callback=self._handle_trade_signal,
                event_types=["trade_signal"]
            )
            self.event_system.register_handler(
                callback=self._handle_order_update,
                event_types=["order_filled", "order_cancelled", "order_rejected"]
            )
    
    def create_market_order(self, symbol: str, quantity: float, side: str, 
                           metadata: Optional[Dict[str, Any]] = None,
                           asset_class: Optional[str] = None,
                           sector: Optional[str] = None) -> Dict[str, Any]:
        """Create a market order.
        
        Args:
            symbol: The symbol to trade.
            quantity: The quantity to trade.
            side: The side of the trade ("buy" or "sell").
            metadata: Optional metadata for the order.
            asset_class: Optional asset class (e.g., 'equities', 'options', 'futures', 'forex', 'crypto').
            sector: Optional sector of the asset (e.g., 'Technology', 'Healthcare').
            
        Returns:
            Dict[str, Any]: The created order data.
        """
        order_id = str(uuid.uuid4())
        order = {
            "id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": "market",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "asset_class": asset_class,
            "sector": sector,
            "metadata": metadata or {}
        }
        
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        # Execute the order if broker service is available
        if self.broker_service:
            try:
                order_params = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "order_type": "MARKET",
                    "side": side.upper(),
                    "asset_class": asset_class,
                    "sector": sector
                }
                response = self.broker_service.place_order(order_params)
                order["broker_order_id"] = response.get("order_id")
                order["status"] = "submitted"
                logger.info(f"Submitted market order: {order_id}")
            except Exception as e:
                order["status"] = "failed"
                order["error"] = str(e)
                logger.error(f"Error submitting market order: {str(e)}")
        
        # Emit order event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="order_created",
                data=order,
                source="order_manager"
            )
        
        return order
    
    def create_limit_order(self, symbol: str, quantity: float, side: str, price: float,
                          metadata: Optional[Dict[str, Any]] = None,
                          asset_class: Optional[str] = None,
                          sector: Optional[str] = None) -> Dict[str, Any]:
        """Create a limit order.
        
        Args:
            symbol: The symbol to trade.
            quantity: The quantity to trade.
            side: The side of the trade ("buy" or "sell").
            price: The limit price.
            metadata: Optional metadata for the order.
            asset_class: Optional asset class (e.g., 'equities', 'options', 'futures', 'forex', 'crypto').
            sector: Optional sector of the asset (e.g., 'Technology', 'Healthcare').
            
        Returns:
            Dict[str, Any]: The created order data.
        """
        order_id = str(uuid.uuid4())
        order = {
            "id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": "limit",
            "price": price,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "asset_class": asset_class,
            "sector": sector,
            "metadata": metadata or {}
        }
        
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        # Execute the order if broker service is available
        if self.broker_service:
            try:
                order_params = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "order_type": "LIMIT",
                    "side": side.upper(),
                    "price": price,
                    "asset_class": asset_class,
                    "sector": sector
                }
                response = self.broker_service.place_order(order_params)
                order["broker_order_id"] = response.get("order_id")
                order["status"] = "submitted"
                logger.info(f"Submitted limit order: {order_id}")
            except Exception as e:
                order["status"] = "failed"
                order["error"] = str(e)
                logger.error(f"Error submitting limit order: {str(e)}")
        
        # Emit order event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="order_created",
                data=order,
                source="order_manager"
            )
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: The ID of the order to cancel.
            
        Returns:
            bool: True if the order was cancelled, False otherwise.
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order not found or not active: {order_id}")
            return False
        
        order = self.active_orders[order_id]
        
        # Cancel the order if broker service is available
        if self.broker_service and order.get("broker_order_id"):
            try:
                self.broker_service.cancel_order({"order_id": order.get("broker_order_id")})
                order["status"] = "cancelling"
                logger.info(f"Cancelling order: {order_id}")
            except Exception as e:
                logger.error(f"Error cancelling order: {str(e)}")
                return False
        else:
            # If no broker service, just mark as cancelled
            order["status"] = "cancelled"
            order["cancelled_at"] = datetime.now().isoformat()
            del self.active_orders[order_id]
        
        # Emit order event if event system is available
        if self.event_system:
            self.event_system.emit(
                event_type="order_cancelled",
                data=order,
                source="order_manager"
            )
        
        return True
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get an order by ID.
        
        Args:
            order_id: The ID of the order to get.
            
        Returns:
            Dict[str, Any]: The order data, or None if not found.
        """
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by.
            
        Returns:
            List[Dict[str, Any]]: The active orders.
        """
        if symbol:
            return [order for order in self.active_orders.values() if order["symbol"] == symbol]
        return list(self.active_orders.values())
    
    def _handle_trade_signal(self, event: Event) -> None:
        """Handle a trade signal event.
        
        Args:
            event: The trade signal event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"Trade signal event has no data")
            return
        
        signal = event.data
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        confidence = signal.get("confidence", 0.5)
        
        if not symbol or not direction:
            logger.warning(f"Trade signal missing required fields")
            return
        
        # Skip hold signals
        if direction.lower() == "hold":
            logger.info(f"Received hold signal for {symbol}, no action taken")
            return
        
        # Determine order quantity based on confidence
        # This is a simplified implementation - in a real system, this would
        # involve position sizing based on risk management rules
        base_quantity = 100  # Base quantity
        quantity = base_quantity * confidence
        
        # Create market order
        self.create_market_order(
            symbol=symbol,
            quantity=quantity,
            side=direction.lower(),
            asset_class=signal.get("asset_class"),
            sector=signal.get("sector"),
            metadata={
                "signal_id": signal.get("id"),
                "confidence": confidence,
                "source": signal.get("source")
            }
        )
    
    def _handle_order_update(self, event: Event) -> None:
        """Handle an order update event.
        
        Args:
            event: The order update event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning(f"Order update event has no data")
            return
        
        update = event.data
        order_id = update.get("order_id")
        
        if not order_id or order_id not in self.orders:
            logger.warning(f"Order update for unknown order: {order_id}")
            return
        
        order = self.orders[order_id]
        
        # Update order status
        if event.event_type == "order_filled":
            order["status"] = "filled"
            order["filled_at"] = update.get("timestamp") or datetime.now().isoformat()
            order["fill_price"] = update.get("price")
            order["fill_quantity"] = update.get("quantity")
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        elif event.event_type == "order_cancelled":
            order["status"] = "cancelled"
            order["cancelled_at"] = update.get("timestamp") or datetime.now().isoformat()
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        elif event.event_type == "order_rejected":
            order["status"] = "rejected"
            order["rejected_at"] = update.get("timestamp") or datetime.now().isoformat()
            order["rejection_reason"] = update.get("reason")
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]


class TradingEngine:
    """Main trading engine that coordinates signal generation and order management.
    
    This class serves as the central component of the trading system, integrating
    signal generation with order management and risk controls.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None, broker_service=None, risk_manager=None):
        """Initialize the trading engine.
        
        Args:
            event_system: The event system for publishing and subscribing to events.
            broker_service: The broker service for executing trades.
            risk_manager: The risk manager for controlling risk.
        """
        self.event_system = event_system
        self.broker_service = broker_service
        self.risk_manager = risk_manager
        
        # Initialize components
        self.signal_generator = SignalGenerator(event_system)
        self.order_manager = OrderManager(event_system, broker_service)
        
        logger.info("Initialized Trading Engine")
    
    def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting Trading Engine")
        
        # Add default trading rules
        self._add_default_trading_rules()
        
        # Register additional event handlers
        if self.event_system:
            self.event_system.register_handler(
                callback=self._handle_system_event,
                event_types=["system_status", "market_status"]
            )
    
    def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping Trading Engine")
        
        # Cancel all active orders
        active_orders = self.order_manager.get_active_orders()
        for order in active_orders:
            self.order_manager.cancel_order(order["id"])
    
    def add_trading_rule(self, rule_func: Callable[[Dict[str, Any]], bool], name: str = None) -> None:
        """Add a trading rule to the signal generator.
        
        Args:
            rule_func: A function that takes a signal and returns True if the signal
                      passes the rule, False otherwise.
            name: Optional name for the rule.
        """
        self.signal_generator.add_rule(rule_func, name)
    
    def process_model_prediction(self, prediction: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Process a model prediction and generate a trading signal.
        
        Args:
            prediction: The model prediction data.
            confidence: The confidence level of the prediction.
            
        Returns:
            Dict[str, Any]: The generated trading signal data, or None if no signal is generated.
        """
        signal = self.signal_generator.generate_signal(prediction, confidence)
        
        if signal:
            # Apply risk management if available
            if self.risk_manager:
                signal = self.risk_manager.adjust_signal(signal)
            
            # Process the signal
            self.signal_generator.process_signal(signal)
        
        return signal
    
    def _add_default_trading_rules(self) -> None:
        """Add default trading rules to the signal generator."""
        # Minimum confidence rule
        def minimum_confidence_rule(signal: Dict[str, Any]) -> bool:
            return signal.get("confidence", 0) >= 0.6
        
        # Trading hours rule (simplified - would be more sophisticated in production)
        def trading_hours_rule(signal: Dict[str, Any]) -> bool:
            now = datetime.now()
            # Only trade during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
            if now.weekday() >= 5:  # Weekend
                return False
            if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
                return False
            return True
        
        # Add the rules
        self.add_trading_rule(minimum_confidence_rule, "minimum_confidence")
        self.add_trading_rule(trading_hours_rule, "trading_hours")
    
    def _handle_system_event(self, event: Event) -> None:
        """Handle system-level events.
        
        Args:
            event: The system event.
        """
        if event.event_type == "market_status" and hasattr(event, "data"):
            market_status = event.data.get("status")
            
            if market_status == "closed":
                logger.info("Market closed, stopping trading engine")
                self.stop()
            elif market_status == "open":
                logger.info("Market open, starting trading engine")
                self.start()