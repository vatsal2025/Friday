"""Trading Engine Integration for the Friday AI Trading System.

This module integrates the various components of the trading engine,
providing a unified interface for the rest of the system.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import threading
import time

from src.infrastructure.event import EventSystem, Event
from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.engine import TradingEngine, SignalGenerator, OrderManager
from src.orchestration.trading_engine.execution import ExecutionFactory, MarketImpactEstimator
from src.orchestration.trading_engine.lifecycle import TradeLifecycleManager, TradeReporter

# Configure logger
logger = get_logger(__name__)


class TradingEngineIntegrator:
    """Integrator for the trading engine components.
    
    This class integrates the various components of the trading engine,
    providing a unified interface for the rest of the system.
    """
    
    def __init__(self, event_system: Optional[EventSystem] = None, broker_service=None, risk_manager=None):
        """Initialize the trading engine integrator.
        
        Args:
            event_system: The event system for publishing and subscribing to events.
            broker_service: The broker service for executing trades.
            risk_manager: The risk manager for controlling risk.
        """
        self.event_system = event_system
        self.broker_service = broker_service
        self.risk_manager = risk_manager
        
        # Initialize components
        self.trading_engine = TradingEngine(event_system, broker_service, risk_manager)
        self.lifecycle_manager = TradeLifecycleManager(event_system)
        self.trade_reporter = TradeReporter(self.lifecycle_manager)
        self.execution_factory = ExecutionFactory(self.trading_engine.order_manager, event_system)
        self.market_impact_estimator = MarketImpactEstimator()
        
        # Set up continuous trading flag
        self.is_continuous_trading = False
        self.continuous_trading_thread = None
        
        logger.info("Initialized Trading Engine Integrator")
    
    def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting Trading Engine Integrator")
        self.trading_engine.start()
    
    def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping Trading Engine Integrator")
        self.trading_engine.stop()
        
        # Stop continuous trading if active
        self.stop_continuous_trading()
    
    def process_model_prediction(self, prediction: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Process a model prediction and generate a trading signal.
        
        Args:
            prediction: The model prediction data.
            confidence: The confidence level of the prediction.
            
        Returns:
            Dict[str, Any]: The generated trading signal data, or None if no signal is generated.
        """
        return self.trading_engine.process_model_prediction(prediction, confidence)
    
    def execute_order(self, order_params: Dict[str, Any], strategy_name: str = "immediate") -> Dict[str, Any]:
        """Execute an order using a specified execution strategy.
        
        Args:
            order_params: The parameters for the order.
            strategy_name: The name of the execution strategy to use.
            
        Returns:
            Dict[str, Any]: The execution result.
        """
        # Estimate market impact
        impact = self.market_impact_estimator.estimate_impact(
            symbol=order_params["symbol"],
            quantity=order_params["quantity"],
            side=order_params["side"]
        )
        
        # Create execution strategy
        strategy = self.execution_factory.create_strategy(strategy_name)
        
        # Execute order
        execution_id = strategy.execute(order_params)
        
        return {
            "execution_id": execution_id,
            "strategy": strategy_name,
            "order_params": order_params,
            "estimated_impact": impact,
            "timestamp": datetime.now().isoformat()
        }
    
    def recommend_execution_strategy(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Recommend an execution strategy based on market impact.
        
        Args:
            symbol: The symbol to trade.
            quantity: The quantity to trade.
            side: The side of the trade ("buy" or "sell").
            
        Returns:
            Dict[str, Any]: The recommended strategy.
        """
        return self.market_impact_estimator.recommend_strategy(symbol, quantity, side)
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a trade by ID.
        
        Args:
            trade_id: The ID of the trade to get.
            
        Returns:
            Dict[str, Any]: The trade data, or None if not found.
        """
        return self.lifecycle_manager.get_trade(trade_id)
    
    def get_trade_history(self, trade_id: str) -> List[Dict[str, Any]]:
        """Get the history of a trade.
        
        Args:
            trade_id: The ID of the trade to get history for.
            
        Returns:
            List[Dict[str, Any]]: The trade history, or empty list if not found.
        """
        return self.lifecycle_manager.get_trade_history(trade_id)
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get active trades (not completed or failed).
        
        Returns:
            List[Dict[str, Any]]: The active trades.
        """
        return self.lifecycle_manager.get_active_trades()
    
    def generate_trade_summary(self, trade_id: str) -> Dict[str, Any]:
        """Generate a summary for a specific trade.
        
        Args:
            trade_id: The ID of the trade to summarize.
            
        Returns:
            Dict[str, Any]: The trade summary.
        """
        return self.trade_reporter.generate_trade_summary(trade_id)
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a daily report for a specific date.
        
        Args:
            date: The date to generate the report for. If None, uses today.
            
        Returns:
            Dict[str, Any]: The daily report.
        """
        return self.trade_reporter.generate_daily_report(date)
    
    def generate_performance_report(self, start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a performance report for a date range.
        
        Args:
            start_date: The start date for the report. If None, uses 30 days ago.
            end_date: The end date for the report. If None, uses today.
            
        Returns:
            Dict[str, Any]: The performance report.
        """
        return self.trade_reporter.generate_performance_report(start_date, end_date)
    
    def add_trading_rule(self, rule_func: Callable[[Dict[str, Any]], bool], name: str = None) -> None:
        """Add a trading rule to the signal generator.
        
        Args:
            rule_func: A function that takes a signal and returns True if the signal
                      passes the rule, False otherwise.
            name: Optional name for the rule.
        """
        self.trading_engine.add_trading_rule(rule_func, name)
    
    def start_continuous_trading(self, interval_seconds: float = 60.0) -> None:
        """Start continuous trading mode.
        
        In continuous trading mode, the system periodically checks for new model
        predictions and market data to generate trading signals.
        
        Args:
            interval_seconds: The interval between checks in seconds.
        """
        if self.is_continuous_trading:
            logger.warning("Continuous trading is already active")
            return
        
        self.is_continuous_trading = True
        self.continuous_trading_thread = threading.Thread(
            target=self._continuous_trading_loop,
            args=(interval_seconds,)
        )
        self.continuous_trading_thread.daemon = True
        self.continuous_trading_thread.start()
        
        logger.info(f"Started continuous trading with interval {interval_seconds} seconds")
    
    def stop_continuous_trading(self) -> None:
        """Stop continuous trading mode."""
        if not self.is_continuous_trading:
            logger.warning("Continuous trading is not active")
            return
        
        self.is_continuous_trading = False
        if self.continuous_trading_thread and self.continuous_trading_thread.is_alive():
            self.continuous_trading_thread.join(timeout=5.0)
        
        logger.info("Stopped continuous trading")
    
    def _continuous_trading_loop(self, interval_seconds: float) -> None:
        """Continuous trading loop.
        
        Args:
            interval_seconds: The interval between checks in seconds.
        """
        while self.is_continuous_trading:
            try:
                # This is a placeholder for the actual continuous trading logic
                # In a real implementation, this would check for new model predictions,
                # market data, and other inputs to generate trading signals
                
                # Emit a heartbeat event if event system is available
                if self.event_system:
                    self.event_system.emit(
                        event_type="trading_engine_heartbeat",
                        data={
                            "timestamp": datetime.now().isoformat(),
                            "active_trades": len(self.get_active_trades())
                        },
                        source="trading_engine_integrator"
                    )
                
                # Sleep for the specified interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in continuous trading loop: {str(e)}")
                time.sleep(interval_seconds)  # Sleep even on error to avoid tight loop


class ModelTradingBridgeIntegration:
    """Integration of the Model Trading Bridge with the Trading Engine.
    
    This class integrates the Model Trading Bridge from Phase 3 with the
    Trading Engine from Phase 4, providing a seamless transition between
    model predictions and trading signals.
    """
    
    def __init__(self, trading_engine_integrator: TradingEngineIntegrator, event_system: Optional[EventSystem] = None):
        """Initialize the model trading bridge integration.
        
        Args:
            trading_engine_integrator: The trading engine integrator.
            event_system: The event system for publishing and subscribing to events.
        """
        self.trading_engine_integrator = trading_engine_integrator
        self.event_system = event_system
        
        # Register event handlers if event system is provided
        if self.event_system:
            self.event_system.register_handler(
                callback=self._handle_model_prediction,
                event_types=["model_prediction"]
            )
        
        logger.info("Initialized Model Trading Bridge Integration")
    
    def process_prediction(self, prediction: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Process a model prediction and generate a trading signal.
        
        Args:
            prediction: The model prediction data.
            confidence: The confidence level of the prediction.
            
        Returns:
            Dict[str, Any]: The generated trading signal data, or None if no signal is generated.
        """
        return self.trading_engine_integrator.process_model_prediction(prediction, confidence)
    
    def _handle_model_prediction(self, event: Event) -> None:
        """Handle a model prediction event.
        
        Args:
            event: The model prediction event.
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Model prediction event has no data")
            return
        
        prediction = event.data
        confidence = prediction.get("confidence", 0.5)
        
        # Process the prediction
        signal = self.process_prediction(prediction, confidence)
        
        if signal:
            logger.info(f"Generated signal from model prediction: {signal.get('id')}")
        else:
            logger.info("No signal generated from model prediction")


def create_trading_engine(event_system: Optional[EventSystem] = None, 
                         broker_service=None, risk_manager=None) -> TradingEngineIntegrator:
    """Create and initialize a trading engine integrator.
    
    Args:
        event_system: The event system for publishing and subscribing to events.
        broker_service: The broker service for executing trades.
        risk_manager: The risk manager for controlling risk.
        
    Returns:
        TradingEngineIntegrator: The initialized trading engine integrator.
    """
    integrator = TradingEngineIntegrator(event_system, broker_service, risk_manager)
    integrator.start()
    return integrator


def create_model_trading_bridge(trading_engine_integrator: TradingEngineIntegrator, 
                              event_system: Optional[EventSystem] = None) -> ModelTradingBridgeIntegration:
    """Create and initialize a model trading bridge integration.
    
    Args:
        trading_engine_integrator: The trading engine integrator.
        event_system: The event system for publishing and subscribing to events.
        
    Returns:
        ModelTradingBridgeIntegration: The initialized model trading bridge integration.
    """
    return ModelTradingBridgeIntegration(trading_engine_integrator, event_system)