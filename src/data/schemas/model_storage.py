"""Model storage schemas for the Friday AI Trading System.

This module defines the schemas for storing and retrieving machine learning models.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator


class ModelType(str, Enum):
    """Types of machine learning models."""
    
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    REINFORCEMENT = "reinforcement"
    TIME_SERIES = "time_series"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """Machine learning frameworks."""
    
    SCIKIT_LEARN = "scikit-learn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    KERAS = "keras"
    CUSTOM = "custom"


class ModelMetrics(BaseModel):
    """Performance metrics for a model."""
    
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    roc_auc: Optional[float] = Field(None, description="ROC AUC score")
    mse: Optional[float] = Field(None, description="Mean squared error")
    rmse: Optional[float] = Field(None, description="Root mean squared error")
    mae: Optional[float] = Field(None, description="Mean absolute error")
    r2: Optional[float] = Field(None, description="R-squared score")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Custom metrics")


class ModelVersion(BaseModel):
    """Version information for a model."""
    
    version: str = Field(..., description="Version number")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    created_by: str = Field(..., description="Creator of the model")
    description: Optional[str] = Field(None, description="Version description")
    metrics: Optional[ModelMetrics] = Field(None, description="Performance metrics")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    file_path: str = Field(..., description="Path to the model file")
    is_active: bool = Field(default=False, description="Whether this version is active")
    
    @validator('created_at')
    def created_at_must_be_utc(cls, v):
        """Validate that created_at is in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v


class ModelSchema(BaseModel):
    """Schema for a machine learning model."""
    
    name: str = Field(..., description="Model name")
    model_id: str = Field(..., description="Unique model identifier")
    description: Optional[str] = Field(None, description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    framework: ModelFramework = Field(..., description="Framework used")
    target_variable: str = Field(..., description="Target variable for prediction")
    features: List[str] = Field(..., description="Features used by the model")
    versions: List[ModelVersion] = Field(default_factory=list, description="Model versions")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    @validator('created_at', 'updated_at')
    def timestamps_must_be_utc(cls, v):
        """Validate that timestamps are in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ModelTrainingConfig(BaseModel):
    """Configuration for model training."""
    
    model_id: str = Field(..., description="Model identifier")
    training_data_path: str = Field(..., description="Path to training data")
    validation_data_path: Optional[str] = Field(None, description="Path to validation data")
    test_data_path: Optional[str] = Field(None, description="Path to test data")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    cross_validation: bool = Field(default=True, description="Whether to use cross-validation")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    random_state: Optional[int] = Field(None, description="Random state for reproducibility")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping")
    early_stopping_patience: int = Field(default=10, description="Patience for early stopping")
    early_stopping_metric: str = Field(default="val_loss", description="Metric for early stopping")
    max_epochs: int = Field(default=100, description="Maximum number of epochs")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    optimizer: Optional[str] = Field(None, description="Optimizer for training")
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    loss_function: Optional[str] = Field(None, description="Loss function")
    metrics: List[str] = Field(default_factory=list, description="Metrics to track")
    callbacks: List[str] = Field(default_factory=list, description="Callbacks to use")
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration")