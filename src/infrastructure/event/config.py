"""Configuration module for the event system.

This module provides configuration classes and utilities for setting up
the event system in different environments (development, testing, production).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


@dataclass
class EventQueueConfig:
    """Configuration for the event queue."""
    
    max_size: int = 10000
    """Maximum number of events in the queue before blocking."""
    
    block_on_full: bool = True
    """Whether to block when the queue is full or raise an exception."""


@dataclass
class EventStoreConfig:
    """Configuration for the event store."""
    
    enabled: bool = False
    """Whether to enable event persistence."""
    
    max_events: int = 100000
    """Maximum number of events to store in memory."""
    
    persistence_path: Optional[str] = None
    """Path to the directory where events will be persisted."""
    
    persistence_interval: int = 60
    """Interval in seconds between persistence operations."""
    
    file_prefix: str = "events"
    """Prefix for event persistence files."""
    
    compression: bool = True
    """Whether to compress persisted events."""
    
    encryption_enabled: bool = False
    """Whether to encrypt persisted events."""
    
    encryption_key_path: Optional[str] = None
    """Path to the encryption key file."""
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        if self.enabled and not self.persistence_path:
            # Default to a directory in the user's home directory
            self.persistence_path = os.path.expanduser("~/friday/events")
            logger.info(f"Setting default persistence path to {self.persistence_path}")


@dataclass
class EventMonitoringConfig:
    """Configuration for event system monitoring."""
    
    enabled: bool = True
    """Whether to enable monitoring."""
    
    check_interval: int = 30
    """Interval in seconds between health checks."""
    
    metrics_window: int = 3600
    """Time window in seconds for metrics collection."""
    
    alert_threshold: float = 0.8
    """Queue capacity threshold (0.0-1.0) for triggering alerts."""
    
    rate_alert_threshold: float = 0.9
    """Event rate threshold (0.0-1.0) compared to historical average for triggering alerts."""
    
    latency_threshold_ms: float = 100.0
    """Latency threshold in milliseconds for triggering alerts."""
    
    error_rate_threshold: float = 0.01
    """Error rate threshold (0.0-1.0) for triggering alerts."""


@dataclass
class EventHandlerConfig:
    """Configuration for an event handler."""
    
    name: str
    """Name of the handler."""
    
    event_types: List[str]
    """List of event types this handler will process."""
    
    enabled: bool = True
    """Whether this handler is enabled."""
    
    max_retries: int = 3
    """Maximum number of retries for failed event processing."""
    
    retry_delay: float = 1.0
    """Delay in seconds between retries."""
    
    timeout: Optional[float] = None
    """Timeout in seconds for event processing."""
    
    async_processing: bool = False
    """Whether to process events asynchronously in a separate thread."""
    
    max_concurrent: int = 1
    """Maximum number of concurrent event processing tasks if async_processing is True."""
    
    priority: int = 0
    """Handler priority (higher values = higher priority)."""
    
    filter_expression: Optional[str] = None
    """Python expression string for filtering events."""


@dataclass
class EventSystemConfig:
    """Configuration for the event system."""
    
    queue: EventQueueConfig = field(default_factory=EventQueueConfig)
    """Configuration for the event queue."""
    
    store: EventStoreConfig = field(default_factory=EventStoreConfig)
    """Configuration for the event store."""
    
    monitoring: EventMonitoringConfig = field(default_factory=EventMonitoringConfig)
    """Configuration for monitoring."""
    
    handlers: Dict[str, EventHandlerConfig] = field(default_factory=dict)
    """Configuration for event handlers."""
    
    worker_threads: int = 1
    """Number of worker threads for processing events."""
    
    shutdown_timeout: float = 5.0
    """Timeout in seconds for graceful shutdown."""
    
    def add_handler_config(self, handler_config: EventHandlerConfig) -> None:
        """Add a handler configuration.
        
        Args:
            handler_config: The handler configuration to add.
        """
        self.handlers[handler_config.name] = handler_config
    
    def remove_handler_config(self, handler_name: str) -> None:
        """Remove a handler configuration.
        
        Args:
            handler_name: The name of the handler configuration to remove.
        """
        if handler_name in self.handlers:
            del self.handlers[handler_name]
    
    def get_handler_config(self, handler_name: str) -> Optional[EventHandlerConfig]:
        """Get a handler configuration by name.
        
        Args:
            handler_name: The name of the handler configuration to get.
            
        Returns:
            The handler configuration, or None if not found.
        """
        return self.handlers.get(handler_name)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EventSystemConfig':
        """Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            An EventSystemConfig instance.
        """
        config = cls()
        
        # Queue config
        if 'queue' in config_dict:
            queue_dict = config_dict['queue']
            config.queue = EventQueueConfig(
                max_size=queue_dict.get('max_size', config.queue.max_size),
                block_on_full=queue_dict.get('block_on_full', config.queue.block_on_full)
            )
        
        # Store config
        if 'store' in config_dict:
            store_dict = config_dict['store']
            config.store = EventStoreConfig(
                enabled=store_dict.get('enabled', config.store.enabled),
                max_events=store_dict.get('max_events', config.store.max_events),
                persistence_path=store_dict.get('persistence_path', config.store.persistence_path),
                persistence_interval=store_dict.get('persistence_interval', config.store.persistence_interval),
                file_prefix=store_dict.get('file_prefix', config.store.file_prefix),
                compression=store_dict.get('compression', config.store.compression),
                encryption_enabled=store_dict.get('encryption_enabled', config.store.encryption_enabled),
                encryption_key_path=store_dict.get('encryption_key_path', config.store.encryption_key_path)
            )
        
        # Monitoring config
        if 'monitoring' in config_dict:
            monitoring_dict = config_dict['monitoring']
            config.monitoring = EventMonitoringConfig(
                enabled=monitoring_dict.get('enabled', config.monitoring.enabled),
                check_interval=monitoring_dict.get('check_interval', config.monitoring.check_interval),
                metrics_window=monitoring_dict.get('metrics_window', config.monitoring.metrics_window),
                alert_threshold=monitoring_dict.get('alert_threshold', config.monitoring.alert_threshold),
                rate_alert_threshold=monitoring_dict.get('rate_alert_threshold', config.monitoring.rate_alert_threshold),
                latency_threshold_ms=monitoring_dict.get('latency_threshold_ms', config.monitoring.latency_threshold_ms),
                error_rate_threshold=monitoring_dict.get('error_rate_threshold', config.monitoring.error_rate_threshold)
            )
        
        # Handler configs
        if 'handlers' in config_dict:
            for handler_name, handler_dict in config_dict['handlers'].items():
                config.add_handler_config(EventHandlerConfig(
                    name=handler_name,
                    event_types=handler_dict.get('event_types', []),
                    enabled=handler_dict.get('enabled', True),
                    max_retries=handler_dict.get('max_retries', 3),
                    retry_delay=handler_dict.get('retry_delay', 1.0),
                    timeout=handler_dict.get('timeout'),
                    async_processing=handler_dict.get('async_processing', False),
                    max_concurrent=handler_dict.get('max_concurrent', 1),
                    priority=handler_dict.get('priority', 0),
                    filter_expression=handler_dict.get('filter_expression')
                ))
        
        # General config
        config.worker_threads = config_dict.get('worker_threads', config.worker_threads)
        config.shutdown_timeout = config_dict.get('shutdown_timeout', config.shutdown_timeout)
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        result = {
            'queue': {
                'max_size': self.queue.max_size,
                'block_on_full': self.queue.block_on_full
            },
            'store': {
                'enabled': self.store.enabled,
                'max_events': self.store.max_events,
                'persistence_path': self.store.persistence_path,
                'persistence_interval': self.store.persistence_interval,
                'file_prefix': self.store.file_prefix,
                'compression': self.store.compression,
                'encryption_enabled': self.store.encryption_enabled,
                'encryption_key_path': self.store.encryption_key_path
            },
            'monitoring': {
                'enabled': self.monitoring.enabled,
                'check_interval': self.monitoring.check_interval,
                'metrics_window': self.monitoring.metrics_window,
                'alert_threshold': self.monitoring.alert_threshold,
                'rate_alert_threshold': self.monitoring.rate_alert_threshold,
                'latency_threshold_ms': self.monitoring.latency_threshold_ms,
                'error_rate_threshold': self.monitoring.error_rate_threshold
            },
            'handlers': {},
            'worker_threads': self.worker_threads,
            'shutdown_timeout': self.shutdown_timeout
        }
        
        # Add handler configs
        for name, handler in self.handlers.items():
            result['handlers'][name] = {
                'event_types': handler.event_types,
                'enabled': handler.enabled,
                'max_retries': handler.max_retries,
                'retry_delay': handler.retry_delay,
                'timeout': handler.timeout,
                'async_processing': handler.async_processing,
                'max_concurrent': handler.max_concurrent,
                'priority': handler.priority,
                'filter_expression': handler.filter_expression
            }
        
        return result


# Predefined configurations

def get_development_config() -> EventSystemConfig:
    """Get a configuration suitable for development environments.
    
    Returns:
        EventSystemConfig: A configuration for development.
    """
    config = EventSystemConfig()
    
    # Smaller queue size for development
    config.queue.max_size = 1000
    
    # Enable event store with default settings
    config.store.enabled = True
    config.store.max_events = 10000
    config.store.persistence_path = os.path.expanduser("~/friday/events/dev")
    config.store.persistence_interval = 30  # More frequent persistence for development
    
    # Enable monitoring with development settings
    config.monitoring.enabled = True
    config.monitoring.check_interval = 15  # More frequent checks for development
    
    # Add some common handler configurations
    config.add_handler_config(EventHandlerConfig(
        name="logging_handler",
        event_types=["*"],  # All event types
        priority=100  # High priority for logging
    ))
    
    config.add_handler_config(EventHandlerConfig(
        name="error_handler",
        event_types=["error", "exception"],
        max_retries=1  # Don't retry error events in development
    ))
    
    return config


def get_testing_config() -> EventSystemConfig:
    """Get a configuration suitable for testing environments.
    
    Returns:
        EventSystemConfig: A configuration for testing.
    """
    config = EventSystemConfig()
    
    # Small queue for testing
    config.queue.max_size = 100
    
    # Disable persistence for testing
    config.store.enabled = False
    
    # Disable monitoring for testing
    config.monitoring.enabled = False
    
    # Single worker thread for deterministic testing
    config.worker_threads = 1
    
    return config


def get_production_config() -> EventSystemConfig:
    """Get a configuration suitable for production environments.
    
    Returns:
        EventSystemConfig: A configuration for production.
    """
    config = EventSystemConfig()
    
    # Large queue for production
    config.queue.max_size = 50000
    
    # Enable event store with production settings
    config.store.enabled = True
    config.store.max_events = 1000000
    config.store.persistence_path = "/var/friday/events"
    config.store.persistence_interval = 300  # Less frequent persistence to reduce I/O
    config.store.compression = True
    config.store.encryption_enabled = True
    config.store.encryption_key_path = "/etc/friday/keys/event_store.key"
    
    # Enable monitoring with production settings
    config.monitoring.enabled = True
    config.monitoring.check_interval = 60  # Check every minute
    config.monitoring.metrics_window = 86400  # 24 hours of metrics
    
    # Multiple worker threads for production
    config.worker_threads = 4
    
    # Add production handler configurations
    config.add_handler_config(EventHandlerConfig(
        name="logging_handler",
        event_types=["*"],
        priority=100,
        async_processing=True,
        max_concurrent=2
    ))
    
    config.add_handler_config(EventHandlerConfig(
        name="error_handler",
        event_types=["error", "exception"],
        max_retries=5,
        retry_delay=5.0,
        priority=90
    ))
    
    config.add_handler_config(EventHandlerConfig(
        name="trade_signal_handler",
        event_types=["trade_signal"],
        priority=80,
        timeout=2.0,  # Timeout for trade signals
        async_processing=True,
        max_concurrent=4
    ))
    
    config.add_handler_config(EventHandlerConfig(
        name="market_data_handler",
        event_types=["market_data"],
        priority=70,
        async_processing=True,
        max_concurrent=4
    ))
    
    config.add_handler_config(EventHandlerConfig(
        name="model_prediction_handler",
        event_types=["model_prediction"],
        priority=60,
        async_processing=True,
        max_concurrent=2
    ))
    
    return config


def load_config_from_file(file_path: str) -> EventSystemConfig:
    """Load configuration from a JSON or YAML file.
    
    Args:
        file_path: Path to the configuration file.
        
    Returns:
        EventSystemConfig: The loaded configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported or the file is invalid.
    """
    import json
    import yaml
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Determine file format from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        with open(file_path, 'r') as f:
            if ext == '.json':
                config_dict = json.load(f)
            elif ext in ('.yaml', '.yml'):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
        
        return EventSystemConfig.from_dict(config_dict)
    
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid configuration file: {e}")


def save_config_to_file(config: EventSystemConfig, file_path: str) -> None:
    """Save configuration to a JSON or YAML file.
    
    Args:
        config: The configuration to save.
        file_path: Path to the output file.
        
    Raises:
        ValueError: If the file format is not supported.
    """
    import json
    import yaml
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Determine file format from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    config_dict = config.to_dict()
    
    with open(file_path, 'w') as f:
        if ext == '.json':
            json.dump(config_dict, f, indent=2)
        elif ext in ('.yaml', '.yml'):
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")


def get_config_from_env() -> EventSystemConfig:
    """Get configuration based on the current environment.
    
    Returns:
        EventSystemConfig: A configuration for the current environment.
    """
    env = os.environ.get('FRIDAY_ENV', 'development').lower()
    
    if env == 'production':
        return get_production_config()
    elif env == 'testing':
        return get_testing_config()
    else:  # development is the default
        return get_development_config()