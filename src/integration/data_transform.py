"""Data transformation module for external system integration.

This module provides utilities for transforming data between external systems and the Friday platform.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar
import json
import logging
from datetime import datetime
from decimal import Decimal

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError

# Create logger
logger = get_logger(__name__)

# Type variable for generic functions
T = TypeVar('T')


class TransformationError(FridayError):
    """Exception raised for errors in data transformation."""
    pass


class DataTransformer:
    """Base class for data transformers.
    
    Data transformers convert data between external system formats and internal Friday formats.
    """
    
    def __init__(self, name: str):
        """Initialize a data transformer.
        
        Args:
            name: The name of the transformer.
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        
    def transform_to_external(self, data: Any) -> Any:
        """Transform data from internal format to external format.
        
        Args:
            data: The data to transform.
            
        Returns:
            Any: The transformed data.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        raise NotImplementedError("Subclasses must implement transform_to_external")
        
    def transform_from_external(self, data: Any) -> Any:
        """Transform data from external format to internal format.
        
        Args:
            data: The data to transform.
            
        Returns:
            Any: The transformed data.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        raise NotImplementedError("Subclasses must implement transform_from_external")


class OrderTransformer(DataTransformer):
    """Transformer for order data."""
    
    def __init__(self, name: str, mapping_to_external: Dict[str, str], mapping_from_external: Dict[str, str]):
        """Initialize an order transformer.
        
        Args:
            name: The name of the transformer.
            mapping_to_external: Mapping of internal field names to external field names.
            mapping_from_external: Mapping of external field names to internal field names.
        """
        super().__init__(name)
        self.mapping_to_external = mapping_to_external
        self.mapping_from_external = mapping_from_external
        
    def transform_to_external(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Transform an order from internal format to external format.
        
        Args:
            order: The order to transform.
            
        Returns:
            Dict[str, Any]: The transformed order.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        try:
            external_order = {}
            
            # Map fields according to the mapping
            for internal_field, external_field in self.mapping_to_external.items():
                if internal_field in order:
                    external_order[external_field] = order[internal_field]
                    
            # Apply any custom transformations
            external_order = self._apply_custom_to_external(external_order, order)
            
            return external_order
        except Exception as e:
            raise TransformationError(f"Failed to transform order to external format: {str(e)}") from e
            
    def transform_from_external(self, external_order: Dict[str, Any]) -> Dict[str, Any]:
        """Transform an order from external format to internal format.
        
        Args:
            external_order: The external order to transform.
            
        Returns:
            Dict[str, Any]: The transformed order.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        try:
            internal_order = {}
            
            # Map fields according to the mapping
            for external_field, internal_field in self.mapping_from_external.items():
                if external_field in external_order:
                    internal_order[internal_field] = external_order[external_field]
                    
            # Apply any custom transformations
            internal_order = self._apply_custom_from_external(internal_order, external_order)
            
            return internal_order
        except Exception as e:
            raise TransformationError(f"Failed to transform order from external format: {str(e)}") from e
            
    def _apply_custom_to_external(self, external_order: Dict[str, Any], 
                                 internal_order: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom transformations when converting to external format.
        
        Override this method in subclasses to apply custom transformations.
        
        Args:
            external_order: The partially transformed external order.
            internal_order: The original internal order.
            
        Returns:
            Dict[str, Any]: The fully transformed external order.
        """
        return external_order
        
    def _apply_custom_from_external(self, internal_order: Dict[str, Any], 
                                   external_order: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom transformations when converting from external format.
        
        Override this method in subclasses to apply custom transformations.
        
        Args:
            internal_order: The partially transformed internal order.
            external_order: The original external order.
            
        Returns:
            Dict[str, Any]: The fully transformed internal order.
        """
        return internal_order


class MarketDataTransformer(DataTransformer):
    """Transformer for market data."""
    
    def __init__(self, name: str, mapping_to_external: Dict[str, str], mapping_from_external: Dict[str, str]):
        """Initialize a market data transformer.
        
        Args:
            name: The name of the transformer.
            mapping_to_external: Mapping of internal field names to external field names.
            mapping_from_external: Mapping of external field names to internal field names.
        """
        super().__init__(name)
        self.mapping_to_external = mapping_to_external
        self.mapping_from_external = mapping_from_external
        
    def transform_to_external(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform market data from internal format to external format.
        
        Args:
            market_data: The market data to transform.
            
        Returns:
            Dict[str, Any]: The transformed market data.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        try:
            external_data = {}
            
            # Map fields according to the mapping
            for internal_field, external_field in self.mapping_to_external.items():
                if internal_field in market_data:
                    external_data[external_field] = market_data[internal_field]
                    
            # Apply any custom transformations
            external_data = self._apply_custom_to_external(external_data, market_data)
            
            return external_data
        except Exception as e:
            raise TransformationError(f"Failed to transform market data to external format: {str(e)}") from e
            
    def transform_from_external(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform market data from external format to internal format.
        
        Args:
            external_data: The external market data to transform.
            
        Returns:
            Dict[str, Any]: The transformed market data.
            
        Raises:
            TransformationError: If the transformation fails.
        """
        try:
            internal_data = {}
            
            # Map fields according to the mapping
            for external_field, internal_field in self.mapping_from_external.items():
                if external_field in external_data:
                    internal_data[internal_field] = external_data[external_field]
                    
            # Apply any custom transformations
            internal_data = self._apply_custom_from_external(internal_data, external_data)
            
            return internal_data
        except Exception as e:
            raise TransformationError(f"Failed to transform market data from external format: {str(e)}") from e
            
    def _apply_custom_to_external(self, external_data: Dict[str, Any], 
                                 internal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom transformations when converting to external format.
        
        Override this method in subclasses to apply custom transformations.
        
        Args:
            external_data: The partially transformed external data.
            internal_data: The original internal data.
            
        Returns:
            Dict[str, Any]: The fully transformed external data.
        """
        return external_data
        
    def _apply_custom_from_external(self, internal_data: Dict[str, Any], 
                                   external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom transformations when converting from external format.
        
        Override this method in subclasses to apply custom transformations.
        
        Args:
            internal_data: The partially transformed internal data.
            external_data: The original external data.
            
        Returns:
            Dict[str, Any]: The fully transformed internal data.
        """
        return internal_data


class TransformerFactory:
    """Factory for creating data transformers."""
    
    _transformers: Dict[str, Type[DataTransformer]] = {}
    
    @classmethod
    def register_transformer(cls, transformer_type: str, transformer_class: Type[DataTransformer]):
        """Register a transformer class.
        
        Args:
            transformer_type: The type of transformer.
            transformer_class: The transformer class.
        """
        cls._transformers[transformer_type] = transformer_class
        
    @classmethod
    def create_transformer(cls, transformer_type: str, name: str, **kwargs) -> DataTransformer:
        """Create a transformer.
        
        Args:
            transformer_type: The type of transformer.
            name: The name of the transformer.
            **kwargs: Additional arguments for the transformer.
            
        Returns:
            DataTransformer: The created transformer.
            
        Raises:
            ValueError: If the transformer type is not registered.
        """
        if transformer_type not in cls._transformers:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
            
        transformer_class = cls._transformers[transformer_type]
        return transformer_class(name, **kwargs)


# Register built-in transformers
TransformerFactory.register_transformer('order', OrderTransformer)
TransformerFactory.register_transformer('market_data', MarketDataTransformer)


# Utility functions for common transformations

def transform_datetime(dt_str: str, format_str: Optional[str] = None) -> datetime:
    """Transform a datetime string to a datetime object.
    
    Args:
        dt_str: The datetime string.
        format_str: The format string for parsing the datetime.
        
    Returns:
        datetime: The parsed datetime.
        
    Raises:
        TransformationError: If the transformation fails.
    """
    try:
        if format_str:
            return datetime.strptime(dt_str, format_str)
        else:
            # Try common formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format with microseconds and Z
                '%Y-%m-%dT%H:%M:%SZ',      # ISO format with Z
                '%Y-%m-%dT%H:%M:%S.%f',    # ISO format with microseconds
                '%Y-%m-%dT%H:%M:%S',       # ISO format
                '%Y-%m-%d %H:%M:%S.%f',    # Standard format with microseconds
                '%Y-%m-%d %H:%M:%S',       # Standard format
                '%Y-%m-%d',                # Date only
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
                    
            raise ValueError(f"Could not parse datetime string: {dt_str}")
    except Exception as e:
        raise TransformationError(f"Failed to transform datetime: {str(e)}") from e


def transform_decimal(value: Union[str, int, float]) -> Decimal:
    """Transform a value to a Decimal.
    
    Args:
        value: The value to transform.
        
    Returns:
        Decimal: The transformed value.
        
    Raises:
        TransformationError: If the transformation fails.
    """
    try:
        return Decimal(str(value))
    except Exception as e:
        raise TransformationError(f"Failed to transform to Decimal: {str(e)}") from e


def apply_mapping(data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Apply a field mapping to data.
    
    Args:
        data: The data to transform.
        mapping: The field mapping (source_field -> target_field).
        
    Returns:
        Dict[str, Any]: The transformed data.
    """
    result = {}
    for source_field, target_field in mapping.items():
        if source_field in data:
            result[target_field] = data[source_field]
    return result


def create_transformer_from_config(config: Dict[str, Any]) -> DataTransformer:
    """Create a transformer from a configuration.
    
    Args:
        config: The transformer configuration.
        
    Returns:
        DataTransformer: The created transformer.
        
    Raises:
        ValueError: If the configuration is invalid.
    """
    if 'type' not in config:
        raise ValueError("Transformer configuration must include 'type'")
        
    if 'name' not in config:
        raise ValueError("Transformer configuration must include 'name'")
        
    transformer_type = config['type']
    name = config['name']
    
    # Extract additional arguments
    kwargs = {k: v for k, v in config.items() if k not in ('type', 'name')}
    
    return TransformerFactory.create_transformer(transformer_type, name, **kwargs)