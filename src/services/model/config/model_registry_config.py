"""Configuration settings for the model registry system.

This module defines default configuration settings for the model registry,
including storage paths, serialization formats, and caching behavior.
"""

import os
from pathlib import Path
from enum import Enum, auto


class SerializationFormat(Enum):
    """Enumeration of supported serialization formats."""
    JOBLIB = auto()
    PICKLE = auto()
    JSON = auto()
    ONNX = auto()
    CUSTOM = auto()


class ModelStatus(Enum):
    """Enumeration of model status values."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()
    DEPRECATED = auto()
    ARCHIVED = auto()


class ModelType(Enum):
    """Enumeration of supported model types."""
    LINEAR = auto()
    RIDGE = auto()
    LASSO = auto()
    RANDOM_FOREST = auto()
    GRADIENT_BOOSTING = auto()
    SVR = auto()
    LSTM = auto()
    TRANSFORMER = auto()
    CUSTOM = auto()


class PredictionTarget(Enum):
    """Enumeration of prediction target types."""
    PRICE = auto()
    RETURN = auto()
    VOLATILITY = auto()
    TREND = auto()
    CUSTOM = auto()


class ModelRegistryConfig:
    """Configuration settings for the model registry."""
    
    # Default base directory for model storage
    DEFAULT_MODELS_DIR = os.path.join(str(Path.home()), ".friday", "models")
    
    # Default serialization format
    DEFAULT_SERIALIZATION_FORMAT = SerializationFormat.JOBLIB
    
    # Default model file name
    DEFAULT_MODEL_FILENAME = "model.joblib"
    
    # Default metadata file name
    DEFAULT_METADATA_FILENAME = "metadata.json"
    
    # Maximum number of models to cache in memory
    MAX_CACHE_SIZE = 10
    
    # Default model status for newly registered models
    DEFAULT_MODEL_STATUS = ModelStatus.DEVELOPMENT
    
    # Required metadata fields for model registration
    REQUIRED_METADATA_FIELDS = [
        "model_type",
        "created_at",
    ]
    
    # Optional metadata fields for model registration
    OPTIONAL_METADATA_FIELDS = [
        "metrics",
        "hyperparameters",
        "training_data_info",
        "feature_importance",
        "pipeline_steps",
        "dependencies",
        "status",
        "updated_at",
        "updated_by",
    ]
    
    # Database configuration
    DATABASE_URI = os.environ.get(
        "MODEL_REGISTRY_DB_URI",
        "sqlite:///" + os.path.join(DEFAULT_MODELS_DIR, "model_registry.db")
    )
    
    # Enable or disable database integration
    USE_DATABASE = True
    
    # Enable or disable model versioning
    ENABLE_VERSIONING = True
    
    # Enable or disable model caching
    ENABLE_CACHING = True
    
    # Enable or disable model validation on registration
    ENABLE_VALIDATION = True
    
    # Enable or disable automatic model evaluation on registration
    ENABLE_AUTO_EVALUATION = False
    
    # Default evaluation metrics to track
    DEFAULT_EVALUATION_METRICS = [
        "mse",
        "rmse",
        "mae",
        "r2",
    ]
    
    # Default tags for model categorization
    DEFAULT_TAGS = [
        "friday",
    ]
    
    # Model registry version
    VERSION = "1.0.0"


# Default configuration instance
default_config = ModelRegistryConfig()