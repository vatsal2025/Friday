"""Data transformer module for the Friday AI Trading System."""

from enum import Enum
from src.data.processing.data_processor import DataProcessor


class TransformationType(Enum):
    """Enumeration of data transformation types."""
    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
    LOG_TRANSFORM = "log_transform"


class DataTransformer(DataProcessor):
    """Class for transforming data."""
    
    def __init__(self, config=None):
        """Initialize a data transformer."""
        super().__init__(config)
        self.transformation_params = {}