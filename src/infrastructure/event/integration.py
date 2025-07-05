"""Integration module for the event system.

This module provides utilities and examples for integrating the event system
with other components of the Friday AI Trading System.
"""

import json
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.infrastructure.event import (
    Event, EventSystem, EventSystemConfig,
    get_production_config, setup_event_monitoring
)
from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class EventSystemIntegration:
    """Integration class for the event system.
    
    This class provides utilities for integrating the event system with
    other components of the Friday AI Trading System, including:
    
    - Graceful startup and shutdown
    - Signal handling
    - Component registration
    - Health monitoring and reporting
    - Integration with external systems
    """
    
    def __init__(
        self,
        config: Optional[EventSystemConfig] = None,
        auto_start: bool = False
    ):
        """Initialize the event system integration.
        
        Args:
            config: Configuration for the event system.
                If None, a production configuration will be used.
            auto_start: Whether to automatically start the event system.
        """
        # Use provided config or get production config
        self.config = config or get_production_config()
        
        # Create the event system
        self.event_system = EventSystem(
            max_queue_size=self.config.queue.max_size,
            max_events=self.config.store.max_events,
            persistence_path=self.config.store.persistence_path,
            persistence_enabled=self.config.store.enabled,
            persistence_interval=self.config.store.persistence_interval
        )
        
        # Set up monitoring
        self.monitor, self.health_check, self.dashboard = setup_event_monitoring(
            self.event_system,
            check_interval=self.config.monitoring.check_interval,
            alert_threshold=self.config.monitoring.alert_threshold
        )
        
        # Track registered components
        self.registered_components: Dict[str, Dict[str, Any]] = {}
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        # Start if requested
        if auto_start:
            self.start()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        # Define signal handler
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down event system...")
            self.stop()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # On Windows, SIGBREAK is sent when Ctrl+Break is pressed
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def start(self):
        """Start the event system and monitoring."""
        logger.info("Starting event system integration")
        
        # Start the event system
        if not self.event_system.is_running():
            self.event_system.start()
            logger.info("Event system started")
        
        # Log startup information
        logger.info(f"Event system queue size: {self.config.queue.max_size}")
        logger.info(f"Event persistence: {'enabled' if self.config.store.enabled else 'disabled'}")
        if self.config.store.enabled:
            logger.info(f"Persistence path: {self.config.store.persistence_path}")
        
        # Start health reporting thread
        self._start_health_reporting()
        
        logger.info("Event system integration started successfully")
    
    def stop(self):
        """Stop the event system and monitoring."""
        logger.info("Stopping event system integration")
        
        # Stop the event system
        if self.event_system.is_running():
            self.event_system.stop()
            logger.info("Event system stopped")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
        if self.health_check:
            self.health_check.stop()
        
        logger.info("Event system integration stopped successfully")
    
    def _start_health_reporting(self):
        """Start a thread for periodic health reporting."""
        def report_health():
            while self.event_system.is_running():
                try:
                    # Get health status
                    health_status = self.health_check.get_health_status()
                    
                    # Log health status
                    if health_status["status"] == "healthy":
                        logger.debug(f"Event system health: {health_status['status']}")
                    else:
                        logger.warning(
                            f"Event system health: {health_status['status']}, "
                            f"issues: {health_status.get('issues', [])}")
                    
                    # Emit health event
                    self.event_system.emit(Event(
                        event_type="system_health",
                        data=health_status,
                        source="event_system_integration"
                    ))
                    
                except Exception as e:
                    logger.error(f"Error in health reporting: {e}")
                
                # Sleep until next report
                time.sleep(self.config.monitoring.check_interval)
        
        # Start the reporting thread
        health_thread = threading.Thread(
            target=report_health,
            daemon=True,
            name="event-health-reporter"
        )
        health_thread.start()
    
    def register_component(
        self,
        component_id: str,
        component_type: str,
        event_types: List[str],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a component with the event system.
        
        This method registers a component that will emit or handle events,
        making it easier to track and manage system components.
        
        Args:
            component_id: Unique identifier for the component.
            component_type: Type of the component (e.g., "data_source", "model", "service").
            event_types: List of event types this component will emit or handle.
            description: Human-readable description of the component.
            metadata: Additional metadata about the component.
        """
        # Store component information
        self.registered_components[component_id] = {
            "id": component_id,
            "type": component_type,
            "event_types": event_types,
            "description": description,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        logger.info(f"Registered component: {component_id} ({component_type})")
        
        # Emit component registration event
        self.event_system.emit(Event(
            event_type="component_registered",
            data={
                "component_id": component_id,
                "component_type": component_type,
                "event_types": event_types,
                "description": description
            },
            source="event_system_integration"
        ))
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the event system.
        
        Args:
            component_id: Unique identifier for the component.
            
        Returns:
            bool: True if the component was unregistered, False if not found.
        """
        if component_id in self.registered_components:
            component = self.registered_components.pop(component_id)
            
            logger.info(f"Unregistered component: {component_id} ({component['type']})")
            
            # Emit component unregistration event
            self.event_system.emit(Event(
                event_type="component_unregistered",
                data={"component_id": component_id, "component_type": component['type']},
                source="event_system_integration"
            ))
            
            return True
        
        return False
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered component.
        
        Args:
            component_id: Unique identifier for the component.
            
        Returns:
            Dict or None: Component information, or None if not found.
        """
        return self.registered_components.get(component_id)
    
    def get_components_by_type(self, component_type: str) -> List[Dict[str, Any]]:
        """Get all components of a specific type.
        
        Args:
            component_type: Type of components to retrieve.
            
        Returns:
            List[Dict]: List of component information dictionaries.
        """
        return [comp for comp in self.registered_components.values() 
                if comp["type"] == component_type]
    
    def get_components_by_event_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all components that handle a specific event type.
        
        Args:
            event_type: Event type to match.
            
        Returns:
            List[Dict]: List of component information dictionaries.
        """
        return [comp for comp in self.registered_components.values() 
                if event_type in comp["event_types"]]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the event system.
        
        Returns:
            Dict: Status information including health, metrics, and components.
        """
        # Get health status
        health_status = self.health_check.get_health_status() if self.health_check else {}
        
        # Get metrics
        metrics = self.monitor.get_summary() if self.monitor else {}
        
        # Build status report
        status = {
            "timestamp": time.time(),
            "running": self.event_system.is_running(),
            "health": health_status,
            "metrics": metrics,
            "components": {
                "count": len(self.registered_components),
                "by_type": {}
            }
        }
        
        # Count components by type
        for comp in self.registered_components.values():
            comp_type = comp["type"]
            if comp_type not in status["components"]["by_type"]:
                status["components"]["by_type"][comp_type] = 0
            status["components"]["by_type"][comp_type] += 1
        
        return status
    
    def save_system_status(self, file_path: str) -> None:
        """Save the current system status to a file.
        
        Args:
            file_path: Path to the output file.
        """
        status = self.get_system_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Saved system status to {file_path}")
    
    def integrate_with_external_system(
        self,
        system_name: str,
        event_types: List[str],
        forward_callback: callable,
        receive_callback: Optional[callable] = None
    ) -> Tuple[callable, Optional[callable]]:
        """Integrate with an external system for event exchange.
        
        This method sets up bidirectional event flow between the Friday event system
        and an external system.
        
        Args:
            system_name: Name of the external system.
            event_types: Event types to exchange with the external system.
            forward_callback: Callback function that forwards events to the external system.
                Function signature: forward_callback(event: Event) -> bool
            receive_callback: Optional callback function that receives events from the external system.
                Function signature: receive_callback() -> Optional[Event]
                
        Returns:
            Tuple containing:
                - Handler function registered with the event system
                - Polling function for the external system (or None if receive_callback is None)
        """
        logger.info(f"Integrating with external system: {system_name}")
        
        # Register the external system as a component
        self.register_component(
            component_id=f"external_{system_name}",
            component_type="external_system",
            event_types=event_types,
            description=f"External system integration: {system_name}"
        )
        
        # Create a handler for forwarding events to the external system
        def external_system_handler(event):
            try:
                result = forward_callback(event)
                if not result:
                    logger.warning(f"Failed to forward event {event.event_id} to {system_name}")
                return result
            except Exception as e:
                logger.error(f"Error forwarding event to {system_name}: {e}")
                return False
        
        # Register the handler with the event system
        handler = self.event_system.register_handler(
            callback=external_system_handler,
            event_types=event_types
        )
        
        # Set up polling for incoming events if a receive callback is provided
        polling_function = None
        if receive_callback:
            def poll_external_events():
                while self.event_system.is_running():
                    try:
                        # Get event from external system
                        event = receive_callback()
                        
                        # If an event was received, emit it to our event system
                        if event:
                            # Set source if not already set
                            if not event.source:
                                event.source = f"external_{system_name}"
                            
                            # Emit the event
                            self.event_system.emit(event)
                    
                    except Exception as e:
                        logger.error(f"Error receiving event from {system_name}: {e}")
                    
                    # Sleep briefly to avoid tight polling
                    time.sleep(0.1)
            
            # Start polling in a separate thread
            polling_thread = threading.Thread(
                target=poll_external_events,
                daemon=True,
                name=f"event-{system_name}-poller"
            )
            polling_thread.start()
            
            # Return the polling function for reference
            polling_function = poll_external_events
        
        logger.info(f"Integration with {system_name} established")
        
        return handler, polling_function


# Example usage
def example_integration():
    """Example of how to use the event system integration."""
    # Create the integration
    integration = EventSystemIntegration(auto_start=True)
    
    try:
        # Register some components
        integration.register_component(
            component_id="market_data_service",
            component_type="data_source",
            event_types=["market_data", "market_status"],
            description="Service providing real-time market data"
        )
        
        integration.register_component(
            component_id="trading_model",
            component_type="model",
            event_types=["model_prediction", "model_status"],
            description="ML model for generating trading signals"
        )
        
        integration.register_component(
            component_id="order_manager",
            component_type="service",
            event_types=["order", "execution", "order_status"],
            description="Service for managing trading orders"
        )
        
        # Example of integrating with an external system (e.g., a trading API)
        def forward_to_trading_api(event):
            # In a real implementation, this would send the event to the trading API
            logger.info(f"Forwarding event {event.event_id} to trading API")
            return True
        
        def receive_from_trading_api():
            # In a real implementation, this would poll the trading API for events
            # For this example, we'll just return None (no events)
            return None
        
        # Set up the integration
        integration.integrate_with_external_system(
            system_name="trading_api",
            event_types=["order", "execution"],
            forward_callback=forward_to_trading_api,
            receive_callback=receive_from_trading_api
        )
        
        # Emit some test events
        integration.event_system.emit(Event(
            event_type="market_data",
            data={"symbol": "AAPL", "price": 150.0, "volume": 1000},
            source="example_integration"
        ))
        
        integration.event_system.emit(Event(
            event_type="model_prediction",
            data={"symbol": "AAPL", "prediction": "BUY", "confidence": 0.85},
            source="example_integration"
        ))
        
        integration.event_system.emit(Event(
            event_type="order",
            data={
                "order_id": "ord123",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "price": 150.0
            },
            source="example_integration"
        ))
        
        # Save system status
        integration.save_system_status("event_system_status.json")
        
        # Keep the example running for a while
        logger.info("Example integration running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Example integration stopped by user")
    
    finally:
        # Clean up
        integration.stop()


if __name__ == "__main__":
    example_integration()