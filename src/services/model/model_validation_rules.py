"""Model-specific validation rules for Friday AI Trading System.

This module provides specialized validation rules for different types of models
used in the trading system.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from enum import Enum

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ValidationResult

# Create logger
logger = get_logger(__name__)


class ModelType(Enum):
    """Types of models used in the trading system."""
    REGRESSION = "regression"  # Regression models (predict continuous values)
    CLASSIFICATION = "classification"  # Classification models (predict classes)
    TIME_SERIES = "time_series"  # Time series forecasting models
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # RL models
    ENSEMBLE = "ensemble"  # Ensemble models
    CUSTOM = "custom"  # Custom model types


class ValidationRuleType(Enum):
    """Types of validation rules."""
    INTERFACE = "interface"  # Model interface validation
    DATA_COMPATIBILITY = "data_compatibility"  # Data compatibility validation
    PERFORMANCE = "performance"  # Performance validation
    RESOURCE = "resource"  # Resource usage validation
    SECURITY = "security"  # Security validation
    CUSTOM = "custom"  # Custom validation


class ValidationRule:
    """Base class for model validation rules."""
    
    def __init__(self, name: str, rule_type: ValidationRuleType, description: str = ""):
        """Initialize a validation rule.
        
        Args:
            name: Name of the validation rule
            rule_type: Type of validation rule
            description: Description of what the rule validates
        """
        self.name = name
        self.rule_type = rule_type
        self.description = description
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Validate a model against this rule.
        
        Args:
            model: The model to validate
            metadata: Model metadata
            
        Returns:
            Tuple[ValidationResult, str]: Validation result and message
        """
        raise NotImplementedError("Subclasses must implement validate()")


# Interface Validation Rules

class HasPredictMethod(ValidationRule):
    """Validates that a model has a predict method."""
    
    def __init__(self):
        super().__init__(
            name="has_predict_method",
            rule_type=ValidationRuleType.INTERFACE,
            description="Validates that a model has a callable predict method"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if not hasattr(model, 'predict'):
            return ValidationResult.FAIL, "Model does not have a 'predict' method"
        if not callable(getattr(model, 'predict')):
            return ValidationResult.FAIL, "Model has 'predict' attribute but it is not callable"
        return ValidationResult.PASS, "Model has required 'predict' method"


class HasFitMethod(ValidationRule):
    """Validates that a model has a fit method."""
    
    def __init__(self):
        super().__init__(
            name="has_fit_method",
            rule_type=ValidationRuleType.INTERFACE,
            description="Validates that a model has a callable fit method"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if not hasattr(model, 'fit'):
            return ValidationResult.FAIL, "Model does not have a 'fit' method"
        if not callable(getattr(model, 'fit')):
            return ValidationResult.FAIL, "Model has 'fit' attribute but it is not callable"
        return ValidationResult.PASS, "Model has required 'fit' method"


# Resource Validation Rules

class ModelSizeRule(ValidationRule):
    """Validates that a model is not too large."""
    
    def __init__(self, max_size_mb: float = 100.0, warning_threshold: float = 0.8):
        """Initialize the model size rule.
        
        Args:
            max_size_mb: Maximum allowed size in MB
            warning_threshold: Threshold for warning as a fraction of max_size_mb
        """
        super().__init__(
            name="model_size",
            rule_type=ValidationRuleType.RESOURCE,
            description=f"Validates that model size is below {max_size_mb} MB"
        )
        self.max_size_mb = max_size_mb
        self.warning_threshold = warning_threshold
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        model_path = metadata.get("model_path")
        if not model_path or not os.path.exists(model_path):
            return ValidationResult.ERROR, "Model file path not found in metadata"
        
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > self.max_size_mb:
            return ValidationResult.FAIL, f"Model size ({size_mb:.2f} MB) exceeds maximum allowed size ({self.max_size_mb} MB)"
        elif size_mb > self.max_size_mb * self.warning_threshold:
            return ValidationResult.WARNING, f"Model size ({size_mb:.2f} MB) is approaching maximum allowed size ({self.max_size_mb} MB)"
        return ValidationResult.PASS, f"Model size ({size_mb:.2f} MB) is within limits"


# Performance Validation Rules

class MinimumAccuracyRule(ValidationRule):
    """Validates that a classification model meets minimum accuracy requirements."""
    
    def __init__(self, min_accuracy: float = 0.7, test_data: Optional[Any] = None, test_labels: Optional[Any] = None):
        """Initialize the minimum accuracy rule.
        
        Args:
            min_accuracy: Minimum required accuracy (0.0 to 1.0)
            test_data: Test data for validation
            test_labels: True labels for test data
        """
        super().__init__(
            name="minimum_accuracy",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that model accuracy is at least {min_accuracy * 100:.1f}%"
        )
        self.min_accuracy = min_accuracy
        self.test_data = test_data
        self.test_labels = test_labels
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for accuracy validation"
        
        try:
            # Make predictions
            predictions = model.predict(self.test_data)
            
            # Calculate accuracy
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(self.test_labels, 'values'):
                test_labels = self.test_labels.values
            else:
                test_labels = self.test_labels
            
            # For binary/multiclass classification
            accuracy = np.mean(predictions == test_labels)
            
            if accuracy < self.min_accuracy:
                return ValidationResult.FAIL, f"Model accuracy ({accuracy:.4f}) is below minimum required ({self.min_accuracy:.4f})"
            return ValidationResult.PASS, f"Model accuracy ({accuracy:.4f}) meets minimum requirement"
        except Exception as e:
            return ValidationResult.ERROR, f"Error calculating model accuracy: {str(e)}"


class RegressionPerformanceRule(ValidationRule):
    """Validates that a regression model meets performance requirements."""
    
    def __init__(self, 
                 max_mse: float = 1.0, 
                 min_r2: float = 0.7,
                 test_data: Optional[Any] = None, 
                 test_labels: Optional[Any] = None):
        """Initialize the regression performance rule.
        
        Args:
            max_mse: Maximum allowed mean squared error
            min_r2: Minimum required R² score
            test_data: Test data for validation
            test_labels: True values for test data
        """
        super().__init__(
            name="regression_performance",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that regression model has MSE <= {max_mse} and R² >= {min_r2}"
        )
        self.max_mse = max_mse
        self.min_r2 = min_r2
        self.test_data = test_data
        self.test_labels = test_labels
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for regression performance validation"
        
        try:
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Make predictions
            predictions = model.predict(self.test_data)
            
            # Calculate metrics
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(self.test_labels, 'values'):
                test_labels = self.test_labels.values
            else:
                test_labels = self.test_labels
            
            mse = mean_squared_error(test_labels, predictions)
            r2 = r2_score(test_labels, predictions)
            
            if mse > self.max_mse and r2 < self.min_r2:
                return ValidationResult.FAIL, f"Model performance is poor: MSE={mse:.4f} (max {self.max_mse}), R²={r2:.4f} (min {self.min_r2})"
            elif mse > self.max_mse:
                return ValidationResult.WARNING, f"Model MSE={mse:.4f} exceeds maximum allowed ({self.max_mse}), but R²={r2:.4f} is acceptable"
            elif r2 < self.min_r2:
                return ValidationResult.WARNING, f"Model R²={r2:.4f} is below minimum required ({self.min_r2}), but MSE={mse:.4f} is acceptable"
            return ValidationResult.PASS, f"Model performance is good: MSE={mse:.4f}, R²={r2:.4f}"
        except Exception as e:
            return ValidationResult.ERROR, f"Error calculating regression performance metrics: {str(e)}"


class BenchmarkComparisonRule(ValidationRule):
    """Validates that a model performs at least as well as a benchmark model."""
    
    def __init__(self, 
                 benchmark_model: Any,
                 test_data: Optional[Any] = None, 
                 test_labels: Optional[Any] = None,
                 metric_name: str = "accuracy",
                 min_improvement: float = 0.0):
        """Initialize the benchmark comparison rule.
        
        Args:
            benchmark_model: The benchmark model to compare against
            test_data: Test data for validation
            test_labels: True values/labels for test data
            metric_name: Name of the metric to compare ("accuracy", "mse", "r2", etc.)
            min_improvement: Minimum required improvement over benchmark (0.0 for equal performance)
        """
        super().__init__(
            name="benchmark_comparison",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that model {metric_name} is at least {min_improvement:.1%} better than benchmark"
        )
        self.benchmark_model = benchmark_model
        self.test_data = test_data
        self.test_labels = test_labels
        self.metric_name = metric_name
        self.min_improvement = min_improvement
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for benchmark comparison"
        
        try:
            # Calculate metrics based on metric name
            if self.metric_name == "accuracy":
                from sklearn.metrics import accuracy_score
                
                # Benchmark predictions
                benchmark_preds = self.benchmark_model.predict(self.test_data)
                benchmark_score = accuracy_score(self.test_labels, benchmark_preds)
                
                # Model predictions
                model_preds = model.predict(self.test_data)
                model_score = accuracy_score(self.test_labels, model_preds)
                
                # Higher is better for accuracy
                improvement = model_score - benchmark_score
                better = improvement >= self.min_improvement
                
            elif self.metric_name == "mse":
                from sklearn.metrics import mean_squared_error
                
                # Benchmark predictions
                benchmark_preds = self.benchmark_model.predict(self.test_data)
                benchmark_score = mean_squared_error(self.test_labels, benchmark_preds)
                
                # Model predictions
                model_preds = model.predict(self.test_data)
                model_score = mean_squared_error(self.test_labels, model_preds)
                
                # Lower is better for MSE
                improvement = benchmark_score - model_score
                better = improvement >= self.min_improvement
                
            elif self.metric_name == "r2":
                from sklearn.metrics import r2_score
                
                # Benchmark predictions
                benchmark_preds = self.benchmark_model.predict(self.test_data)
                benchmark_score = r2_score(self.test_labels, benchmark_preds)
                
                # Model predictions
                model_preds = model.predict(self.test_data)
                model_score = r2_score(self.test_labels, model_preds)
                
                # Higher is better for R²
                improvement = model_score - benchmark_score
                better = improvement >= self.min_improvement
                
            else:
                return ValidationResult.ERROR, f"Unsupported metric: {self.metric_name}"
            
            if better:
                return ValidationResult.PASS, f"Model {self.metric_name} ({model_score:.4f}) is better than benchmark ({benchmark_score:.4f}) by {improvement:.4f}"
            else:
                return ValidationResult.FAIL, f"Model {self.metric_name} ({model_score:.4f}) is not sufficiently better than benchmark ({benchmark_score:.4f}), improvement: {improvement:.4f}, required: {self.min_improvement:.4f}"
        
        except Exception as e:
            return ValidationResult.ERROR, f"Error comparing to benchmark: {str(e)}"


# Model Type-Specific Validation Rules

def get_validation_rules_for_model_type(model_type: ModelType) -> List[ValidationRule]:
    """Get a list of validation rules appropriate for a specific model type.
    
    Args:
        model_type: Type of model to get validation rules for
        
    Returns:
        List of validation rules appropriate for the model type
    """
    # Common rules for all model types
    common_rules = [
        HasPredictMethod(),
        ModelSizeRule()
    ]
    
    # Model type-specific rules
    if model_type == ModelType.REGRESSION:
        return common_rules + [
            HasFitMethod()
            # Note: Performance rules require test data and are added separately
        ]
    
    elif model_type == ModelType.CLASSIFICATION:
        return common_rules + [
            HasFitMethod()
            # Note: Performance rules require test data and are added separately
        ]
    
    elif model_type == ModelType.TIME_SERIES:
        return common_rules + [
            HasFitMethod()
            # Note: Performance rules require test data and are added separately
        ]
    
    elif model_type == ModelType.REINFORCEMENT_LEARNING:
        # RL models might have different interfaces
        return common_rules
    
    elif model_type == ModelType.ENSEMBLE:
        return common_rules
    
    else:  # ModelType.CUSTOM or unknown
        return common_rules