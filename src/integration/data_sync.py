"""Data synchronization for external system integrations.

This module provides utilities for synchronizing data between external systems and the Friday platform,
including incremental synchronization, conflict resolution, and data validation.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
import time
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import hashlib

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.config import ConfigManager
from src.data.integration import DataPipeline

# Create logger
logger = get_logger(__name__)


class SyncError(FridayError):
    """Exception raised for errors in data synchronization."""
    pass


class SyncDirection(Enum):
    """Direction of data synchronization."""
    PULL = 'pull'  # From external system to Friday
    PUSH = 'push'  # From Friday to external system
    BIDIRECTIONAL = 'bidirectional'  # Both directions


class SyncStrategy(Enum):
    """Strategy for data synchronization."""
    FULL = 'full'  # Full synchronization (all data)
    INCREMENTAL = 'incremental'  # Incremental synchronization (only changed data)
    DIFFERENTIAL = 'differential'  # Differential synchronization (compare and sync differences)


class ConflictResolutionStrategy(Enum):
    """Strategy for resolving conflicts during synchronization."""
    EXTERNAL_WINS = 'external_wins'  # External system data takes precedence
    FRIDAY_WINS = 'friday_wins'  # Friday data takes precedence
    NEWEST_WINS = 'newest_wins'  # Newest data takes precedence
    MANUAL = 'manual'  # Manual resolution required


class SyncStatus(Enum):
    """Status of a synchronization operation."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    PARTIAL = 'partial'  # Completed with some errors


class SyncRecord:
    """Record of a synchronization operation."""
    
    def __init__(self, sync_id: str, system_id: str, data_type: str,
                 direction: SyncDirection, strategy: SyncStrategy,
                 start_time: datetime):
        """Initialize a synchronization record.
        
        Args:
            sync_id: The ID of the synchronization operation.
            system_id: The ID of the external system.
            data_type: The type of data being synchronized.
            direction: The direction of synchronization.
            strategy: The synchronization strategy.
            start_time: The start time of the synchronization.
        """
        self.sync_id = sync_id
        self.system_id = system_id
        self.data_type = data_type
        self.direction = direction
        self.strategy = strategy
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.status = SyncStatus.PENDING
        self.items_processed = 0
        self.items_succeeded = 0
        self.items_failed = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.last_sync_key: Optional[str] = None
        
    def update(self, status: SyncStatus, items_processed: int = 0,
               items_succeeded: int = 0, items_failed: int = 0):
        """Update the synchronization record.
        
        Args:
            status: The new status.
            items_processed: The number of items processed.
            items_succeeded: The number of items successfully synchronized.
            items_failed: The number of items that failed to synchronize.
        """
        self.status = status
        self.items_processed += items_processed
        self.items_succeeded += items_succeeded
        self.items_failed += items_failed
        
        if status in (SyncStatus.COMPLETED, SyncStatus.FAILED, SyncStatus.PARTIAL):
            self.end_time = datetime.now()
            
    def add_error(self, error: str):
        """Add an error to the synchronization record.
        
        Args:
            error: The error message.
        """
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning to the synchronization record.
        
        Args:
            warning: The warning message.
        """
        self.warnings.append(warning)
        
    def set_last_sync_key(self, sync_key: str):
        """Set the last synchronization key.
        
        Args:
            sync_key: The synchronization key.
        """
        self.last_sync_key = sync_key
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the synchronization record to a dictionary.
        
        Returns:
            Dict[str, Any]: The synchronization record as a dictionary.
        """
        return {
            'sync_id': self.sync_id,
            'system_id': self.system_id,
            'data_type': self.data_type,
            'direction': self.direction.value,
            'strategy': self.strategy.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'items_processed': self.items_processed,
            'items_succeeded': self.items_succeeded,
            'items_failed': self.items_failed,
            'errors': self.errors,
            'warnings': self.warnings,
            'last_sync_key': self.last_sync_key
        }
        
    @classmethod
    def from_dict(cls, record_dict: Dict[str, Any]) -> 'SyncRecord':
        """Create a synchronization record from a dictionary.
        
        Args:
            record_dict: The dictionary containing the record data.
            
        Returns:
            SyncRecord: The created synchronization record.
        """
        record = cls(
            sync_id=record_dict['sync_id'],
            system_id=record_dict['system_id'],
            data_type=record_dict['data_type'],
            direction=SyncDirection(record_dict['direction']),
            strategy=SyncStrategy(record_dict['strategy']),
            start_time=datetime.fromisoformat(record_dict['start_time'])
        )
        
        if record_dict.get('end_time'):
            record.end_time = datetime.fromisoformat(record_dict['end_time'])
            
        record.status = SyncStatus(record_dict['status'])
        record.items_processed = record_dict['items_processed']
        record.items_succeeded = record_dict['items_succeeded']
        record.items_failed = record_dict['items_failed']
        record.errors = record_dict['errors']
        record.warnings = record_dict['warnings']
        record.last_sync_key = record_dict.get('last_sync_key')
        
        return record


class SyncEvent(Event):
    """Event for data synchronization."""
    
    def __init__(self, sync_record: SyncRecord):
        """Initialize a synchronization event.
        
        Args:
            sync_record: The synchronization record.
        """
        super().__init__(f"sync.{sync_record.system_id}.{sync_record.data_type}")
        self.sync_record = sync_record
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['sync_record'] = self.sync_record.to_dict()
        return event_dict


class SyncStartEvent(SyncEvent):
    """Event for the start of a synchronization operation."""
    
    def __init__(self, sync_record: SyncRecord):
        """Initialize a synchronization start event.
        
        Args:
            sync_record: The synchronization record.
        """
        super().__init__(sync_record)
        self.event_type = f"sync.start.{sync_record.system_id}.{sync_record.data_type}"


class SyncCompleteEvent(SyncEvent):
    """Event for the completion of a synchronization operation."""
    
    def __init__(self, sync_record: SyncRecord):
        """Initialize a synchronization complete event.
        
        Args:
            sync_record: The synchronization record.
        """
        super().__init__(sync_record)
        self.event_type = f"sync.complete.{sync_record.system_id}.{sync_record.data_type}"


class SyncFailedEvent(SyncEvent):
    """Event for a failed synchronization operation."""
    
    def __init__(self, sync_record: SyncRecord):
        """Initialize a synchronization failed event.
        
        Args:
            sync_record: The synchronization record.
        """
        super().__init__(sync_record)
        self.event_type = f"sync.failed.{sync_record.system_id}.{sync_record.data_type}"


class SyncConflictEvent(SyncEvent):
    """Event for a synchronization conflict."""
    
    def __init__(self, sync_record: SyncRecord, item_id: str,
                 external_data: Dict[str, Any], friday_data: Dict[str, Any]):
        """Initialize a synchronization conflict event.
        
        Args:
            sync_record: The synchronization record.
            item_id: The ID of the item with a conflict.
            external_data: The data from the external system.
            friday_data: The data from Friday.
        """
        super().__init__(sync_record)
        self.event_type = f"sync.conflict.{sync_record.system_id}.{sync_record.data_type}"
        self.item_id = item_id
        self.external_data = external_data
        self.friday_data = friday_data
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['item_id'] = self.item_id
        event_dict['external_data'] = self.external_data
        event_dict['friday_data'] = self.friday_data
        return event_dict


class DataValidator:
    """Base class for data validators."""
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data.
        
        Args:
            data: The data to validate.
            
        Returns:
            Tuple[bool, List[str]]: A tuple of (is_valid, error_messages).
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate()")


class SchemaValidator(DataValidator):
    """Validator for JSON Schema validation."""
    
    def __init__(self, schema: Dict[str, Any]):
        """Initialize a schema validator.
        
        Args:
            schema: The JSON Schema to validate against.
        """
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against a JSON Schema.
        
        Args:
            data: The data to validate.
            
        Returns:
            Tuple[bool, List[str]]: A tuple of (is_valid, error_messages).
        """
        try:
            import jsonschema
            
            try:
                jsonschema.validate(instance=data, schema=self.schema)
                return True, []
            except jsonschema.exceptions.ValidationError as e:
                return False, [str(e)]
        except ImportError:
            logger.warning("jsonschema package not found, validation skipped")
            return True, ["jsonschema package not found, validation skipped"]


class CustomValidator(DataValidator):
    """Validator using a custom validation function."""
    
    def __init__(self, validate_func: Callable[[Dict[str, Any]], Tuple[bool, List[str]]]):
        """Initialize a custom validator.
        
        Args:
            validate_func: A function that validates data and returns a tuple of (is_valid, error_messages).
        """
        self.validate_func = validate_func
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data using a custom function.
        
        Args:
            data: The data to validate.
            
        Returns:
            Tuple[bool, List[str]]: A tuple of (is_valid, error_messages).
        """
        return self.validate_func(data)


class CompositeValidator(DataValidator):
    """Validator that combines multiple validators."""
    
    def __init__(self, validators: List[DataValidator]):
        """Initialize a composite validator.
        
        Args:
            validators: The validators to use.
        """
        self.validators = validators
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data using multiple validators.
        
        Args:
            data: The data to validate.
            
        Returns:
            Tuple[bool, List[str]]: A tuple of (is_valid, error_messages).
        """
        is_valid = True
        error_messages = []
        
        for validator in self.validators:
            valid, errors = validator.validate(data)
            if not valid:
                is_valid = False
                error_messages.extend(errors)
                
        return is_valid, error_messages


class SyncConfig:
    """Configuration for data synchronization."""
    
    def __init__(self, system_id: str, data_type: str,
                 direction: SyncDirection = SyncDirection.PULL,
                 strategy: SyncStrategy = SyncStrategy.INCREMENTAL,
                 conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.NEWEST_WINS,
                 sync_interval: int = 3600,
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 validators: Optional[List[DataValidator]] = None,
                 transform_config: Optional[Dict[str, Any]] = None):
        """Initialize a synchronization configuration.
        
        Args:
            system_id: The ID of the external system.
            data_type: The type of data to synchronize.
            direction: The direction of synchronization.
            strategy: The synchronization strategy.
            conflict_resolution: The conflict resolution strategy.
            sync_interval: The interval between synchronizations in seconds.
            batch_size: The number of items to process in a batch.
            max_retries: The maximum number of retries for failed operations.
            retry_delay: The delay between retries in seconds.
            validators: The validators to use for data validation.
            transform_config: Configuration for data transformation.
        """
        self.system_id = system_id
        self.data_type = data_type
        self.direction = direction
        self.strategy = strategy
        self.conflict_resolution = conflict_resolution
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validators = validators or []
        self.transform_config = transform_config or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return {
            'system_id': self.system_id,
            'data_type': self.data_type,
            'direction': self.direction.value,
            'strategy': self.strategy.value,
            'conflict_resolution': self.conflict_resolution.value,
            'sync_interval': self.sync_interval,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'transform_config': self.transform_config
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SyncConfig':
        """Create a synchronization configuration from a dictionary.
        
        Args:
            config_dict: The dictionary containing the configuration.
            
        Returns:
            SyncConfig: The created synchronization configuration.
        """
        return cls(
            system_id=config_dict['system_id'],
            data_type=config_dict['data_type'],
            direction=SyncDirection(config_dict['direction']),
            strategy=SyncStrategy(config_dict['strategy']),
            conflict_resolution=ConflictResolutionStrategy(config_dict['conflict_resolution']),
            sync_interval=config_dict.get('sync_interval', 3600),
            batch_size=config_dict.get('batch_size', 100),
            max_retries=config_dict.get('max_retries', 3),
            retry_delay=config_dict.get('retry_delay', 5),
            transform_config=config_dict.get('transform_config', {})
        )


class SyncTask:
    """Task for data synchronization."""
    
    def __init__(self, config: SyncConfig, external_client: Any,
                 friday_repository: Any, event_system: Optional[EventSystem] = None,
                 data_pipeline: Optional[DataPipeline] = None):
        """Initialize a synchronization task.
        
        Args:
            config: The synchronization configuration.
            external_client: The client for the external system.
            friday_repository: The repository for Friday data.
            event_system: The event system for publishing events.
            data_pipeline: The data pipeline for processing data.
        """
        self.config = config
        self.external_client = external_client
        self.friday_repository = friday_repository
        self.event_system = event_system
        self.data_pipeline = data_pipeline
        self.last_sync_time: Optional[datetime] = None
        self.last_sync_key: Optional[str] = None
        self.sync_records: List[SyncRecord] = []
        self.lock = threading.RLock()
        
    def generate_sync_id(self) -> str:
        """Generate a unique synchronization ID.
        
        Returns:
            str: The generated synchronization ID.
        """
        timestamp = int(time.time())
        random_part = os.urandom(4).hex()
        return f"{self.config.system_id}_{self.config.data_type}_{timestamp}_{random_part}"
        
    def start_sync(self) -> SyncRecord:
        """Start a synchronization operation.
        
        Returns:
            SyncRecord: The synchronization record.
        """
        sync_id = self.generate_sync_id()
        sync_record = SyncRecord(
            sync_id=sync_id,
            system_id=self.config.system_id,
            data_type=self.config.data_type,
            direction=self.config.direction,
            strategy=self.config.strategy,
            start_time=datetime.now()
        )
        
        with self.lock:
            self.sync_records.append(sync_record)
            
        # Publish start event
        if self.event_system:
            self.event_system.publish(SyncStartEvent(sync_record))
            
        return sync_record
        
    def complete_sync(self, sync_record: SyncRecord, status: SyncStatus,
                     items_processed: int, items_succeeded: int, items_failed: int):
        """Complete a synchronization operation.
        
        Args:
            sync_record: The synchronization record.
            status: The final status.
            items_processed: The number of items processed.
            items_succeeded: The number of items successfully synchronized.
            items_failed: The number of items that failed to synchronize.
        """
        sync_record.update(status, items_processed, items_succeeded, items_failed)
        
        # Update last sync time and key
        if status in (SyncStatus.COMPLETED, SyncStatus.PARTIAL):
            self.last_sync_time = datetime.now()
            if sync_record.last_sync_key:
                self.last_sync_key = sync_record.last_sync_key
                
        # Publish event
        if self.event_system:
            if status == SyncStatus.FAILED:
                self.event_system.publish(SyncFailedEvent(sync_record))
            else:
                self.event_system.publish(SyncCompleteEvent(sync_record))
                
    def handle_conflict(self, sync_record: SyncRecord, item_id: str,
                       external_data: Dict[str, Any], friday_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a synchronization conflict.
        
        Args:
            sync_record: The synchronization record.
            item_id: The ID of the item with a conflict.
            external_data: The data from the external system.
            friday_data: The data from Friday.
            
        Returns:
            Dict[str, Any]: The resolved data.
        """
        # Publish conflict event
        if self.event_system:
            self.event_system.publish(SyncConflictEvent(
                sync_record, item_id, external_data, friday_data
            ))
            
        # Resolve conflict based on strategy
        if self.config.conflict_resolution == ConflictResolutionStrategy.EXTERNAL_WINS:
            return external_data
        elif self.config.conflict_resolution == ConflictResolutionStrategy.FRIDAY_WINS:
            return friday_data
        elif self.config.conflict_resolution == ConflictResolutionStrategy.NEWEST_WINS:
            # Compare timestamps if available
            external_timestamp = external_data.get('updated_at') or external_data.get('timestamp')
            friday_timestamp = friday_data.get('updated_at') or friday_data.get('timestamp')
            
            if external_timestamp and friday_timestamp:
                if isinstance(external_timestamp, str):
                    external_timestamp = datetime.fromisoformat(external_timestamp)
                if isinstance(friday_timestamp, str):
                    friday_timestamp = datetime.fromisoformat(friday_timestamp)
                    
                if external_timestamp > friday_timestamp:
                    return external_data
                else:
                    return friday_data
            else:
                # Default to external wins if timestamps not available
                return external_data
        elif self.config.conflict_resolution == ConflictResolutionStrategy.MANUAL:
            # Add to conflicts for manual resolution
            sync_record.add_warning(f"Manual conflict resolution required for item {item_id}")
            # Return None to skip this item for now
            return None
            
        # Default to external wins
        return external_data
        
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data.
        
        Args:
            data: The data to validate.
            
        Returns:
            Tuple[bool, List[str]]: A tuple of (is_valid, error_messages).
        """
        if not self.config.validators:
            return True, []
            
        # Create a composite validator
        validator = CompositeValidator(self.config.validators)
        return validator.validate(data)
        
    def transform_data(self, data: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """Transform data.
        
        Args:
            data: The data to transform.
            direction: The direction of transformation ('to_friday' or 'to_external').
            
        Returns:
            Dict[str, Any]: The transformed data.
        """
        if not self.data_pipeline:
            return data
            
        # Add transformation configuration
        context = {
            'direction': direction,
            'config': self.config.transform_config
        }
        
        # Process data through pipeline
        result = self.data_pipeline.process(data, context)
        return result
        
    def calculate_sync_key(self, data: Dict[str, Any]) -> str:
        """Calculate a synchronization key for data.
        
        Args:
            data: The data to calculate a key for.
            
        Returns:
            str: The calculated synchronization key.
        """
        # Use timestamp if available
        timestamp = data.get('updated_at') or data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, datetime):
                return timestamp.isoformat()
            return str(timestamp)
            
        # Otherwise, hash the data
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
        
    def pull_data(self, sync_record: SyncRecord) -> Tuple[int, int, int]:
        """Pull data from the external system to Friday.
        
        Args:
            sync_record: The synchronization record.
            
        Returns:
            Tuple[int, int, int]: A tuple of (items_processed, items_succeeded, items_failed).
        """
        items_processed = 0
        items_succeeded = 0
        items_failed = 0
        last_key = None
        
        try:
            # Get data from external system
            if self.config.strategy == SyncStrategy.INCREMENTAL and self.last_sync_key:
                external_data = self.external_client.get_data_since(
                    self.config.data_type, self.last_sync_key, self.config.batch_size
                )
            else:
                external_data = self.external_client.get_data(
                    self.config.data_type, self.config.batch_size
                )
                
            # Process each item
            for item in external_data:
                items_processed += 1
                
                try:
                    # Calculate sync key
                    sync_key = self.calculate_sync_key(item)
                    if not last_key or sync_key > last_key:
                        last_key = sync_key
                        
                    # Transform data
                    transformed_item = self.transform_data(item, 'to_friday')
                    
                    # Validate data
                    is_valid, errors = self.validate_data(transformed_item)
                    if not is_valid:
                        sync_record.add_error(f"Validation failed for item {item.get('id')}: {', '.join(errors)}")
                        items_failed += 1
                        continue
                        
                    # Check for conflicts
                    item_id = transformed_item.get('id')
                    if item_id:
                        friday_item = self.friday_repository.get_by_id(item_id)
                        if friday_item:
                            # Handle conflict
                            resolved_item = self.handle_conflict(
                                sync_record, item_id, transformed_item, friday_item
                            )
                            if resolved_item is None:
                                # Skip this item for now
                                continue
                            transformed_item = resolved_item
                            
                    # Save to Friday
                    self.friday_repository.save(transformed_item)
                    items_succeeded += 1
                except Exception as e:
                    sync_record.add_error(f"Failed to process item {item.get('id')}: {str(e)}")
                    items_failed += 1
                    
            # Update last sync key
            if last_key:
                sync_record.set_last_sync_key(last_key)
                
            return items_processed, items_succeeded, items_failed
        except Exception as e:
            sync_record.add_error(f"Failed to pull data: {str(e)}")
            return items_processed, items_succeeded, items_failed
            
    def push_data(self, sync_record: SyncRecord) -> Tuple[int, int, int]:
        """Push data from Friday to the external system.
        
        Args:
            sync_record: The synchronization record.
            
        Returns:
            Tuple[int, int, int]: A tuple of (items_processed, items_succeeded, items_failed).
        """
        items_processed = 0
        items_succeeded = 0
        items_failed = 0
        last_key = None
        
        try:
            # Get data from Friday
            if self.config.strategy == SyncStrategy.INCREMENTAL and self.last_sync_time:
                friday_data = self.friday_repository.get_updated_since(
                    self.last_sync_time, self.config.batch_size
                )
            else:
                friday_data = self.friday_repository.get_all(
                    self.config.batch_size
                )
                
            # Process each item
            for item in friday_data:
                items_processed += 1
                
                try:
                    # Calculate sync key
                    sync_key = self.calculate_sync_key(item)
                    if not last_key or sync_key > last_key:
                        last_key = sync_key
                        
                    # Transform data
                    transformed_item = self.transform_data(item, 'to_external')
                    
                    # Validate data
                    is_valid, errors = self.validate_data(transformed_item)
                    if not is_valid:
                        sync_record.add_error(f"Validation failed for item {item.get('id')}: {', '.join(errors)}")
                        items_failed += 1
                        continue
                        
                    # Check for conflicts
                    item_id = transformed_item.get('id')
                    if item_id:
                        external_item = self.external_client.get_by_id(self.config.data_type, item_id)
                        if external_item:
                            # Handle conflict
                            resolved_item = self.handle_conflict(
                                sync_record, item_id, external_item, transformed_item
                            )
                            if resolved_item is None:
                                # Skip this item for now
                                continue
                            transformed_item = resolved_item
                            
                    # Save to external system
                    self.external_client.save(self.config.data_type, transformed_item)
                    items_succeeded += 1
                except Exception as e:
                    sync_record.add_error(f"Failed to process item {item.get('id')}: {str(e)}")
                    items_failed += 1
                    
            # Update last sync key
            if last_key:
                sync_record.set_last_sync_key(last_key)
                
            return items_processed, items_succeeded, items_failed
        except Exception as e:
            sync_record.add_error(f"Failed to push data: {str(e)}")
            return items_processed, items_succeeded, items_failed
            
    def sync_bidirectional(self, sync_record: SyncRecord) -> Tuple[int, int, int]:
        """Synchronize data in both directions.
        
        Args:
            sync_record: The synchronization record.
            
        Returns:
            Tuple[int, int, int]: A tuple of (items_processed, items_succeeded, items_failed).
        """
        # Pull data first
        pull_processed, pull_succeeded, pull_failed = self.pull_data(sync_record)
        
        # Then push data
        push_processed, push_succeeded, push_failed = self.push_data(sync_record)
        
        return (
            pull_processed + push_processed,
            pull_succeeded + push_succeeded,
            pull_failed + push_failed
        )
        
    def sync(self) -> SyncRecord:
        """Synchronize data.
        
        Returns:
            SyncRecord: The synchronization record.
        """
        # Start synchronization
        sync_record = self.start_sync()
        sync_record.update(SyncStatus.RUNNING)
        
        try:
            # Synchronize data based on direction
            if self.config.direction == SyncDirection.PULL:
                items_processed, items_succeeded, items_failed = self.pull_data(sync_record)
            elif self.config.direction == SyncDirection.PUSH:
                items_processed, items_succeeded, items_failed = self.push_data(sync_record)
            else:  # BIDIRECTIONAL
                items_processed, items_succeeded, items_failed = self.sync_bidirectional(sync_record)
                
            # Determine final status
            if items_failed == 0:
                status = SyncStatus.COMPLETED
            elif items_succeeded > 0:
                status = SyncStatus.PARTIAL
            else:
                status = SyncStatus.FAILED
                
            # Complete synchronization
            self.complete_sync(
                sync_record, status, items_processed, items_succeeded, items_failed
            )
            
            return sync_record
        except Exception as e:
            logger.error(f"Synchronization failed: {str(e)}")
            sync_record.add_error(f"Synchronization failed: {str(e)}")
            self.complete_sync(sync_record, SyncStatus.FAILED, 0, 0, 0)
            return sync_record


class SyncManager:
    """Manager for data synchronization."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 event_system: Optional[EventSystem] = None):
        """Initialize a synchronization manager.
        
        Args:
            config_manager: The configuration manager.
            event_system: The event system.
        """
        self.config_manager = config_manager
        self.event_system = event_system
        self.tasks: Dict[str, SyncTask] = {}
        self.scheduled_tasks: Dict[str, Tuple[SyncTask, datetime]] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
    def add_task(self, task: SyncTask) -> str:
        """Add a synchronization task.
        
        Args:
            task: The synchronization task.
            
        Returns:
            str: The task ID.
        """
        task_id = f"{task.config.system_id}_{task.config.data_type}"
        
        with self.lock:
            self.tasks[task_id] = task
            
            # Schedule the task if interval is set
            if task.config.sync_interval > 0:
                next_run = datetime.now() + timedelta(seconds=task.config.sync_interval)
                self.scheduled_tasks[task_id] = (task, next_run)
                
        return task_id
        
    def remove_task(self, task_id: str) -> bool:
        """Remove a synchronization task.
        
        Args:
            task_id: The ID of the task to remove.
            
        Returns:
            bool: True if the task was removed, False otherwise.
        """
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                if task_id in self.scheduled_tasks:
                    del self.scheduled_tasks[task_id]
                return True
            return False
            
    def get_task(self, task_id: str) -> Optional[SyncTask]:
        """Get a synchronization task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            Optional[SyncTask]: The task, or None if not found.
        """
        with self.lock:
            return self.tasks.get(task_id)
            
    def get_all_tasks(self) -> Dict[str, SyncTask]:
        """Get all synchronization tasks.
        
        Returns:
            Dict[str, SyncTask]: All tasks.
        """
        with self.lock:
            return self.tasks.copy()
            
    def run_task(self, task_id: str) -> Optional[SyncRecord]:
        """Run a synchronization task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            Optional[SyncRecord]: The synchronization record, or None if the task was not found.
        """
        task = self.get_task(task_id)
        if task is None:
            return None
            
        # Run the task
        sync_record = task.sync()
        
        # Update next run time if scheduled
        with self.lock:
            if task_id in self.scheduled_tasks:
                next_run = datetime.now() + timedelta(seconds=task.config.sync_interval)
                self.scheduled_tasks[task_id] = (task, next_run)
                
        return sync_record
        
    def start_scheduler(self):
        """Start the scheduler thread."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_thread, daemon=True)
            self.scheduler_thread.start()
            
    def stop_scheduler(self):
        """Stop the scheduler thread."""
        with self.lock:
            self.running = False
            if self.scheduler_thread is not None:
                self.scheduler_thread.join(timeout=5.0)
                self.scheduler_thread = None
                
    def _scheduler_thread(self):
        """Thread for scheduling synchronization tasks."""
        while self.running:
            try:
                # Get tasks that should run
                tasks_to_run = []
                with self.lock:
                    now = datetime.now()
                    for task_id, (task, next_run) in list(self.scheduled_tasks.items()):
                        if now >= next_run:
                            tasks_to_run.append(task_id)
                            
                # Run the tasks
                for task_id in tasks_to_run:
                    try:
                        self.run_task(task_id)
                    except Exception as e:
                        logger.error(f"Failed to run scheduled task {task_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Error in scheduler thread: {str(e)}")
                
            # Sleep for a short time
            time.sleep(1.0)
            
    def create_task_from_config(self, config: Dict[str, Any], external_client: Any,
                              friday_repository: Any, data_pipeline: Optional[DataPipeline] = None) -> SyncTask:
        """Create a synchronization task from a configuration dictionary.
        
        Args:
            config: The configuration dictionary.
            external_client: The client for the external system.
            friday_repository: The repository for Friday data.
            data_pipeline: The data pipeline for processing data.
            
        Returns:
            SyncTask: The created synchronization task.
            
        Raises:
            SyncError: If the configuration is invalid.
        """
        try:
            # Create sync config
            sync_config = SyncConfig.from_dict(config)
            
            # Create validators if specified
            validators = []
            if 'validators' in config:
                for validator_config in config['validators']:
                    validator_type = validator_config.get('type')
                    if validator_type == 'schema':
                        schema = validator_config.get('schema')
                        if schema:
                            validators.append(SchemaValidator(schema))
                    elif validator_type == 'custom':
                        # Custom validators must be added programmatically
                        pass
                        
            sync_config.validators = validators
            
            # Create and return the task
            task = SyncTask(
                config=sync_config,
                external_client=external_client,
                friday_repository=friday_repository,
                event_system=self.event_system,
                data_pipeline=data_pipeline
            )
            
            return task
        except Exception as e:
            raise SyncError(f"Failed to create synchronization task: {str(e)}") from e
            
    def load_tasks_from_config(self, external_clients: Dict[str, Any],
                             friday_repositories: Dict[str, Any],
                             data_pipelines: Optional[Dict[str, DataPipeline]] = None):
        """Load synchronization tasks from configuration.
        
        Args:
            external_clients: A dictionary of external clients.
            friday_repositories: A dictionary of Friday repositories.
            data_pipelines: A dictionary of data pipelines.
            
        Raises:
            SyncError: If the configuration is invalid or a task cannot be created.
        """
        if self.config_manager is None:
            raise SyncError("Configuration manager is required to load tasks from configuration")
            
        try:
            # Get sync configurations
            sync_configs = self.config_manager.get('integration.sync', {})
            
            # Create tasks
            for config_id, config in sync_configs.items():
                system_id = config.get('system_id')
                data_type = config.get('data_type')
                
                if not system_id or not data_type:
                    logger.warning(f"Invalid sync configuration {config_id}: missing system_id or data_type")
                    continue
                    
                # Get external client
                external_client = external_clients.get(system_id)
                if external_client is None:
                    logger.warning(f"External client not found for system {system_id}")
                    continue
                    
                # Get Friday repository
                friday_repository = friday_repositories.get(data_type)
                if friday_repository is None:
                    logger.warning(f"Friday repository not found for data type {data_type}")
                    continue
                    
                # Get data pipeline if specified
                data_pipeline = None
                if data_pipelines and 'pipeline' in config:
                    pipeline_id = config['pipeline']
                    data_pipeline = data_pipelines.get(pipeline_id)
                    
                # Create and add the task
                task = self.create_task_from_config(
                    config, external_client, friday_repository, data_pipeline
                )
                self.add_task(task)
                
            logger.info(f"Loaded {len(self.tasks)} synchronization tasks from configuration")
        except Exception as e:
            raise SyncError(f"Failed to load synchronization tasks: {str(e)}") from e
            
    def save_sync_records(self, file_path: str):
        """Save synchronization records to a file.
        
        Args:
            file_path: The path to save the records to.
            
        Raises:
            SyncError: If the records cannot be saved.
        """
        try:
            # Get all sync records
            records = []
            with self.lock:
                for task in self.tasks.values():
                    records.extend([record.to_dict() for record in task.sync_records])
                    
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the records
            with open(file_path, 'w') as f:
                json.dump(records, f, indent=2)
                
            logger.info(f"Saved {len(records)} synchronization records to {file_path}")
        except Exception as e:
            raise SyncError(f"Failed to save synchronization records: {str(e)}") from e
            
    def load_sync_records(self, file_path: str):
        """Load synchronization records from a file.
        
        Args:
            file_path: The path to load the records from.
            
        Raises:
            SyncError: If the records cannot be loaded.
        """
        try:
            # Load the records
            with open(file_path, 'r') as f:
                records_data = json.load(f)
                
            # Add records to tasks
            for record_data in records_data:
                system_id = record_data.get('system_id')
                data_type = record_data.get('data_type')
                
                if not system_id or not data_type:
                    continue
                    
                task_id = f"{system_id}_{data_type}"
                task = self.get_task(task_id)
                
                if task is not None:
                    record = SyncRecord.from_dict(record_data)
                    task.sync_records.append(record)
                    
                    # Update last sync time and key if needed
                    if record.status in (SyncStatus.COMPLETED, SyncStatus.PARTIAL):
                        if record.end_time and (task.last_sync_time is None or record.end_time > task.last_sync_time):
                            task.last_sync_time = record.end_time
                            
                        if record.last_sync_key:
                            task.last_sync_key = record.last_sync_key
                            
            logger.info(f"Loaded synchronization records from {file_path}")
        except Exception as e:
            raise SyncError(f"Failed to load synchronization records: {str(e)}") from e


def create_sync_config(system_id: str, data_type: str, **kwargs) -> SyncConfig:
    """Create a synchronization configuration.
    
    Args:
        system_id: The ID of the external system.
        data_type: The type of data to synchronize.
        **kwargs: Additional configuration options.
        
    Returns:
        SyncConfig: The created synchronization configuration.
    """
    return SyncConfig(system_id, data_type, **kwargs)


def create_sync_task(config: SyncConfig, external_client: Any, friday_repository: Any,
                   event_system: Optional[EventSystem] = None,
                   data_pipeline: Optional[DataPipeline] = None) -> SyncTask:
    """Create a synchronization task.
    
    Args:
        config: The synchronization configuration.
        external_client: The client for the external system.
        friday_repository: The repository for Friday data.
        event_system: The event system for publishing events.
        data_pipeline: The data pipeline for processing data.
        
    Returns:
        SyncTask: The created synchronization task.
    """
    return SyncTask(config, external_client, friday_repository, event_system, data_pipeline)


def get_sync_manager() -> SyncManager:
    """Get the global synchronization manager instance.
    
    Returns:
        SyncManager: The global synchronization manager instance.
    """
    # Use a global variable to store the sync manager instance
    global _sync_manager
    
    # Create the sync manager if it doesn't exist
    if '_sync_manager' not in globals():
        from src.infrastructure.config import get_config_manager
        from src.infrastructure.event import get_event_system
        _sync_manager = SyncManager(get_config_manager(), get_event_system())
        
    return _sync_manager