"""Common event handlers for the Friday AI Trading System.

This module provides pre-built event handlers for common scenarios
in the trading system.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from src.infrastructure.event.event_system import Event, EventHandler, EventSystem
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class LoggingEventHandler:
    """Handler that logs all events it receives.
    
    This handler can be used to keep a log of all events or specific event types
    for debugging and monitoring purposes.
    """
    
    def __init__(self, event_types: Optional[List[str]] = None, log_level: str = "INFO"):
        """Initialize the logging event handler.
        
        Args:
            event_types: List of event types to log. If None, all events are logged.
            log_level: The log level to use. Defaults to "INFO".
        """
        self.event_types = event_types
        self.log_level = log_level.upper()
        self.logger = get_logger(f"{__name__}.LoggingEventHandler")
    
    def handle(self, event: Event) -> None:
        """Handle an event by logging it.
        
        Args:
            event: The event to handle.
        """
        if self.event_types is None or event.event_type in self.event_types:
            log_method = getattr(self.logger, self.log_level.lower(), self.logger.info)
            log_method(f"Event: {event.event_type}, ID: {event.id}, Source: {event.source}")
            log_method(f"Data: {json.dumps(event.data, indent=2)}")


class ErrorEventHandler:
    """Handler for error events in the system.
    
    This handler processes error events and can perform actions like
    sending alerts, writing to error logs, or triggering recovery procedures.
    """
    
    def __init__(self, alert_callback: Optional[Callable[[Event], None]] = None):
        """Initialize the error event handler.
        
        Args:
            alert_callback: Optional callback function to call when an error event is received.
                This can be used to send alerts or notifications.
        """
        self.alert_callback = alert_callback
        self.logger = get_logger(f"{__name__}.ErrorEventHandler")
    
    def handle(self, event: Event) -> None:
        """Handle an error event.
        
        Args:
            event: The error event to handle.
        """
        # Log the error
        error_message = event.data.get("message", "Unknown error")
        error_source = event.source or "Unknown source"
        error_details = event.data.get("details", {})
        
        self.logger.error(f"Error from {error_source}: {error_message}")
        if error_details:
            self.logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
        
        # Call the alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(event)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")


class TradeSignalHandler:
    """Handler for trade signal events.
    
    This handler processes trade signals and can perform actions like
    validating signals, executing trades, or forwarding signals to other components.
    """
    
    def __init__(self, 
                 execute_trade_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 validation_callback: Optional[Callable[[Dict[str, Any]], bool]] = None):
        """Initialize the trade signal handler.
        
        Args:
            execute_trade_callback: Optional callback function to execute a trade.
            validation_callback: Optional callback function to validate a trade signal.
        """
        self.execute_trade_callback = execute_trade_callback
        self.validation_callback = validation_callback
        self.logger = get_logger(f"{__name__}.TradeSignalHandler")
    
    def handle(self, event: Event) -> None:
        """Handle a trade signal event.
        
        Args:
            event: The trade signal event to handle.
        """
        signal_data = event.data
        signal_source = event.source or "Unknown source"
        
        self.logger.info(f"Received trade signal from {signal_source}")
        self.logger.info(f"Signal data: {json.dumps(signal_data, indent=2)}")
        
        # Validate the signal if a validation callback is provided
        if self.validation_callback:
            try:
                is_valid = self.validation_callback(signal_data)
                if not is_valid:
                    self.logger.warning(f"Trade signal validation failed: {signal_data}")
                    return
            except Exception as e:
                self.logger.error(f"Error validating trade signal: {str(e)}")
                return
        
        # Execute the trade if an execution callback is provided
        if self.execute_trade_callback:
            try:
                self.execute_trade_callback(signal_data)
                self.logger.info(f"Trade executed for signal: {signal_data}")
            except Exception as e:
                self.logger.error(f"Error executing trade: {str(e)}")


class DataEventHandler:
    """Handler for data events in the system.
    
    This handler processes data events like market data updates,
    indicator calculations, or other data-related events.
    """
    
    def __init__(self, 
                 process_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 forward_processed_data: bool = False,
                 event_system: Optional[EventSystem] = None):
        """Initialize the data event handler.
        
        Args:
            process_callback: Optional callback function to process the data.
            forward_processed_data: Whether to forward processed data as a new event.
            event_system: The event system to use for forwarding processed data.
                Required if forward_processed_data is True.
        """
        self.process_callback = process_callback
        self.forward_processed_data = forward_processed_data
        self.event_system = event_system
        self.logger = get_logger(f"{__name__}.DataEventHandler")
        
        if self.forward_processed_data and self.event_system is None:
            raise ValueError("event_system must be provided if forward_processed_data is True")
    
    def handle(self, event: Event) -> None:
        """Handle a data event.
        
        Args:
            event: The data event to handle.
        """
        data = event.data
        data_source = event.source or "Unknown source"
        
        self.logger.debug(f"Processing data from {data_source}")
        
        # Process the data if a process callback is provided
        processed_data = data
        if self.process_callback:
            try:
                processed_data = self.process_callback(data)
                self.logger.debug(f"Data processed successfully")
            except Exception as e:
                self.logger.error(f"Error processing data: {str(e)}")
                return
        
        # Forward the processed data as a new event if requested
        if self.forward_processed_data and self.event_system:
            try:
                self.event_system.emit(
                    event_type="data_processed",
                    data=processed_data,
                    source=f"data_processor_{data_source}"
                )
                self.logger.debug(f"Processed data forwarded as new event")
            except Exception as e:
                self.logger.error(f"Error forwarding processed data: {str(e)}")


class ModelPredictionHandler:
    """Handler for model prediction events.
    
    This handler processes prediction events from ML models
    and can perform actions like generating trade signals or alerts.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 generate_signal_callback: Optional[Callable[[Dict[str, Any], float], Dict[str, Any]]] = None,
                 event_system: Optional[EventSystem] = None):
        """Initialize the model prediction handler.
        
        Args:
            threshold: The confidence threshold for generating signals.
            generate_signal_callback: Optional callback to generate a trade signal from a prediction.
            event_system: The event system to use for emitting trade signals.
        """
        self.threshold = threshold
        self.generate_signal_callback = generate_signal_callback
        self.event_system = event_system
        self.logger = get_logger(f"{__name__}.ModelPredictionHandler")
    
    def handle(self, event: Event) -> None:
        """Handle a model prediction event.
        
        Args:
            event: The model prediction event to handle.
        """
        prediction_data = event.data
        model_source = event.source or "Unknown model"
        
        # Extract prediction confidence
        confidence = prediction_data.get("confidence", 0.0)
        prediction = prediction_data.get("prediction", {})
        
        self.logger.info(f"Received prediction from {model_source} with confidence {confidence}")
        
        # Check if confidence meets threshold
        if confidence < self.threshold:
            self.logger.info(f"Prediction confidence {confidence} below threshold {self.threshold}, ignoring")
            return
        
        # Generate trade signal if callback is provided
        if self.generate_signal_callback and self.event_system:
            try:
                signal_data = self.generate_signal_callback(prediction, confidence)
                
                if signal_data:
                    self.event_system.emit(
                        event_type="trade_signal",
                        data=signal_data,
                        source=f"model_prediction_{model_source}"
                    )
                    self.logger.info(f"Generated trade signal from prediction: {signal_data}")
            except Exception as e:
                self.logger.error(f"Error generating trade signal from prediction: {str(e)}")


class SystemMonitorHandler:
    """Handler for system monitoring events.
    
    This handler processes system events like resource usage, health checks,
    and can trigger alerts or recovery actions when thresholds are exceeded.
    """
    
    def __init__(self, thresholds: Dict[str, float] = None, alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """Initialize the system monitor handler.
        
        Args:
            thresholds: Dictionary of metric names and their threshold values.
            alert_callback: Optional callback function to call when a threshold is exceeded.
        """
        self.thresholds = thresholds or {
            "cpu_usage": 80.0,  # percent
            "memory_usage": 80.0,  # percent
            "disk_usage": 80.0,  # percent
            "latency": 1000.0,  # milliseconds
        }
        self.alert_callback = alert_callback
        self.logger = get_logger(f"{__name__}.SystemMonitorHandler")
    
    def handle(self, event: Event) -> None:
        """Handle a system monitoring event.
        
        Args:
            event: The system monitoring event to handle.
        """
        metrics = event.data.get("metrics", {})
        system_component = event.source or "Unknown component"
        
        self.logger.debug(f"Received system metrics from {system_component}")
        
        # Check each metric against its threshold
        alerts = []
        for metric_name, metric_value in metrics.items():
            if metric_name in self.thresholds and metric_value > self.thresholds[metric_name]:
                alert_message = f"{metric_name} exceeded threshold: {metric_value} > {self.thresholds[metric_name]}"
                alerts.append((metric_name, metric_value, self.thresholds[metric_name]))
                self.logger.warning(f"{system_component}: {alert_message}")
        
        # Call the alert callback if thresholds were exceeded
        if alerts and self.alert_callback:
            try:
                for metric_name, metric_value, threshold in alerts:
                    self.alert_callback(metric_name, {
                        "component": system_component,
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold,
                        "timestamp": event.timestamp
                    })
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")


def register_common_handlers(event_system: EventSystem) -> Dict[str, Any]:
    """Register common handlers with the event system.
    
    This is a convenience function to set up standard handlers for
    common event types in the trading system.
    
    Args:
        event_system: The event system to register handlers with.
        
    Returns:
        Dict[str, Any]: A dictionary of the registered handlers for reference.
    """
    # Create handlers
    logging_handler = LoggingEventHandler(log_level="DEBUG")
    error_handler = ErrorEventHandler()
    trade_signal_handler = TradeSignalHandler()
    data_handler = DataEventHandler(event_system=event_system, forward_processed_data=True)
    system_monitor_handler = SystemMonitorHandler()
    
    # Register handlers with appropriate event types
    event_system.register_handler(
        callback=logging_handler.handle,
        # Register for all event types by not specifying event_types
    )
    
    event_system.register_handler(
        callback=error_handler.handle,
        event_types=["system_error", "data_error", "trade_error", "model_error"]
    )
    
    event_system.register_handler(
        callback=trade_signal_handler.handle,
        event_types=["trade_signal"]
    )
    
    event_system.register_handler(
        callback=data_handler.handle,
        event_types=["data_received", "market_data"]
    )
    
    event_system.register_handler(
        callback=system_monitor_handler.handle,
        event_types=["system_metrics", "system_health"]
    )
    
    # Return the handlers for reference
    return {
        "logging_handler": logging_handler,
        "error_handler": error_handler,
        "trade_signal_handler": trade_signal_handler,
        "data_handler": data_handler,
        "system_monitor_handler": system_monitor_handler
    }