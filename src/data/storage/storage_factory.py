"""Data Storage Factory for the Friday AI Trading System.

This module provides a factory class for creating and managing different
data storage backends based on configuration settings.
"""

from typing import Any, Dict, Optional, Type, Union
from pathlib import Path

from src.data.storage.data_storage import DataStorage, StorageError
from src.data.storage.local_parquet_storage import LocalParquetStorage
from src.data.storage.mongodb_storage import MongoDBStorage
from src.data.storage.sql_storage import SQLStorage
from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class DataStorageFactory:
    """Factory class for creating data storage instances.
    
    This factory provides methods to create and configure different storage
    backends based on configuration settings and requirements.
    
    Attributes:
        _storage_classes: Dictionary mapping storage type names to their classes.
        _instances: Dictionary of cached storage instances.
    """
    
    _storage_classes = {
        'local_parquet': LocalParquetStorage,
        'mongodb': MongoDBStorage,
        'postgresql': SQLStorage,
        'sql': SQLStorage,  # Alias for SQL storage
    }
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the DataStorageFactory.
        
        Args:
            config: Configuration manager instance.
        """
        self.config = config or ConfigManager()
        self._instances = {}
    
    def create_storage(
        self, 
        storage_type: Optional[str] = None,
        **kwargs
    ) -> DataStorage:
        """Create a storage instance of the specified type.
        
        Args:
            storage_type: Type of storage to create. If None, uses default from config.
            **kwargs: Additional arguments passed to the storage constructor.
            
        Returns:
            DataStorage: Configured storage instance.
            
        Raises:
            StorageError: If storage type is not supported or creation fails.
        """
        try:
            # Get storage type from config if not specified
            if storage_type is None:
                storage_type = self.config.get(
                    "data.storage.default_backend", 
                    "local_parquet"
                )
            
            # Normalize storage type
            storage_type = storage_type.lower()
            
            # Check if storage type is supported
            if storage_type not in self._storage_classes:
                supported_types = list(self._storage_classes.keys())
                raise StorageError(
                    f"Unsupported storage type: {storage_type}. "
                    f"Supported types: {supported_types}"
                )
            
            # Get storage class
            storage_class = self._storage_classes[storage_type]
            
            # Create storage instance with configuration
            storage_kwargs = self._get_storage_config(storage_type)
            storage_kwargs.update(kwargs)
            
            instance = storage_class(config=self.config, **storage_kwargs)
            
            logger.info(f"Created {storage_type} storage instance")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create {storage_type} storage: {str(e)}")
            raise StorageError(f"Failed to create {storage_type} storage: {str(e)}")
    
    def get_default_storage(self) -> DataStorage:
        """Get the default storage instance based on configuration.
        
        Returns:
            DataStorage: Default configured storage instance.
        """
        default_type = self.config.get("data.storage.default_backend", "local_parquet")
        
        # Use cached instance if available
        if default_type in self._instances:
            return self._instances[default_type]
        
        # Create new instance and cache it
        instance = self.create_storage(default_type)
        self._instances[default_type] = instance
        
        return instance
    
    def get_storage_for_training(self) -> DataStorage:
        """Get storage instance optimized for model training data retrieval.
        
        Returns:
            DataStorage: Storage instance optimized for training workloads.
        """
        # For training, we prefer local parquet for fast reading
        training_storage_type = self.config.get(
            "data.storage.training_backend", 
            "local_parquet"
        )
        
        cache_key = f"{training_storage_type}_training"
        
        if cache_key in self._instances:
            return self._instances[cache_key]
        
        # Create storage with training-optimized configuration
        training_kwargs = self._get_training_optimized_config(training_storage_type)
        instance = self.create_storage(training_storage_type, **training_kwargs)
        
        self._instances[cache_key] = instance
        return instance
    
    def _get_storage_config(self, storage_type: str) -> Dict[str, Any]:
        """Get configuration for a specific storage type.
        
        Args:
            storage_type: Type of storage.
            
        Returns:
            Dict containing storage-specific configuration.
        """
        config_map = {
            'local_parquet': self._get_parquet_config,
            'mongodb': self._get_mongodb_config,
            'postgresql': self._get_postgresql_config,
            'sql': self._get_postgresql_config,
        }
        
        config_func = config_map.get(storage_type, lambda: {})
        return config_func()
    
    def _get_parquet_config(self) -> Dict[str, Any]:
        """Get LocalParquetStorage configuration."""
        parquet_config = self.config.get("data.storage.local_parquet", {})
        
        return {
            'base_dir': parquet_config.get('base_dir', 'data/market_data')
        }
    
    def _get_mongodb_config(self) -> Dict[str, Any]:
        """Get MongoDB storage configuration."""
        mongodb_config = self.config.get("data.storage.mongodb", {})
        
        return {
            'connection_string': mongodb_config.get('connection_string'),
            'database_name': mongodb_config.get('database_name')
        }
    
    def _get_postgresql_config(self) -> Dict[str, Any]:
        """Get PostgreSQL storage configuration."""
        pg_config = self.config.get("data.storage.postgresql", {})
        
        return {
            'connection_string': pg_config.get('connection_string'),
            'schema': pg_config.get('schema')
        }
    
    def _get_training_optimized_config(self, storage_type: str) -> Dict[str, Any]:
        """Get training-optimized configuration for storage type.
        
        Args:
            storage_type: Type of storage.
            
        Returns:
            Dict containing training-optimized configuration.
        """
        retrieval_config = self.config.get("data.storage.retrieval", {})
        
        if storage_type == 'local_parquet':
            return {
                'base_dir': self.config.get("data.storage.local_parquet.base_dir", 'data/market_data')
            }
        
        elif storage_type == 'mongodb':
            mongodb_config = self.config.get("data.storage.mongodb", {})
            return {
                'connection_string': mongodb_config.get('connection_string'),
                'database_name': mongodb_config.get('database_name')
            }
        
        elif storage_type in ['postgresql', 'sql']:
            pg_config = self.config.get("data.storage.postgresql", {})
            return {
                'connection_string': pg_config.get('connection_string'),
                'schema': pg_config.get('schema')
            }
        
        return {}
    
    def register_storage_class(
        self, 
        storage_type: str, 
        storage_class: Type[DataStorage]
    ) -> None:
        """Register a custom storage class.
        
        Args:
            storage_type: Name for the storage type.
            storage_class: Storage class that inherits from DataStorage.
        """
        if not issubclass(storage_class, DataStorage):
            raise ValueError(f"Storage class must inherit from DataStorage")
        
        self._storage_classes[storage_type.lower()] = storage_class
        logger.info(f"Registered custom storage class: {storage_type}")
    
    def list_supported_types(self) -> list:
        """List all supported storage types.
        
        Returns:
            List of supported storage type names.
        """
        return list(self._storage_classes.keys())
    
    def validate_storage_config(self, storage_type: str = None) -> Dict[str, Any]:
        """Validate storage configuration.
        
        Args:
            storage_type: Storage type to validate. If None, validates all.
            
        Returns:
            Dict containing validation results.
        """
        results = {}
        
        if storage_type:
            storage_types = [storage_type]
        else:
            storage_types = self.list_supported_types()
        
        for stype in storage_types:
            try:
                config = self._get_storage_config(stype)
                
                # Basic validation
                validation_result = {
                    'type': stype,
                    'valid': True,
                    'config': config,
                    'issues': []
                }
                
                # Type-specific validations
                if stype == 'local_parquet':
                    base_dir = config.get('base_dir')
                    if base_dir:
                        base_path = Path(base_dir)
                        if not base_path.parent.exists():
                            validation_result['issues'].append(
                                f"Parent directory does not exist: {base_path.parent}"
                            )
                
                elif stype == 'mongodb':
                    if not config.get('connection_string'):
                        validation_result['issues'].append("Missing MongoDB connection string")
                    if not config.get('database_name'):
                        validation_result['issues'].append("Missing MongoDB database name")
                
                elif stype in ['postgresql', 'sql']:
                    if not config.get('connection_string'):
                        validation_result['issues'].append("Missing PostgreSQL connection string")
                
                if validation_result['issues']:
                    validation_result['valid'] = False
                
                results[stype] = validation_result
                
            except Exception as e:
                results[stype] = {
                    'type': stype,
                    'valid': False,
                    'error': str(e),
                    'issues': [f"Configuration error: {str(e)}"]
                }
        
        return results
    
    def clear_cache(self) -> None:
        """Clear cached storage instances."""
        self._instances.clear()
        logger.info("Cleared storage instance cache")


# Global factory instance
_factory_instance = None


def get_storage_factory(config: Optional[ConfigManager] = None) -> DataStorageFactory:
    """Get the global storage factory instance.
    
    Args:
        config: Configuration manager instance.
        
    Returns:
        DataStorageFactory: Global factory instance.
    """
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = DataStorageFactory(config)
    
    return _factory_instance


def get_default_storage() -> DataStorage:
    """Get the default storage instance.
    
    Returns:
        DataStorage: Default storage instance.
    """
    factory = get_storage_factory()
    return factory.get_default_storage()


def get_training_storage() -> DataStorage:
    """Get storage instance optimized for model training.
    
    Returns:
        DataStorage: Training-optimized storage instance.
    """
    factory = get_storage_factory()
    return factory.get_storage_for_training()


def create_storage(storage_type: str, **kwargs) -> DataStorage:
    """Create a storage instance of the specified type.
    
    Args:
        storage_type: Type of storage to create.
        **kwargs: Additional arguments passed to storage constructor.
        
    Returns:
        DataStorage: Configured storage instance.
    """
    factory = get_storage_factory()
    return factory.create_storage(storage_type, **kwargs)
