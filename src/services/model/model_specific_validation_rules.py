"""Model-specific validation rules for Friday AI Trading System.

This module provides specialized validation rules for different types of models,
including regression, classification, time series, and reinforcement learning models.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

from src.services.model.model_validation_rules import (
    ValidationRule,
    ValidationRuleType,
    ValidationResult,
    ModelType
)

logger = logging.getLogger(__name__)


class ClassificationAccuracyRule(ValidationRule):
    """Validates that a classification model meets accuracy requirements."""
    
    def __init__(self, 
                 min_accuracy: float,
                 test_data: Optional[Any] = None, 
                 test_labels: Optional[Any] = None):
        """Initialize the classification accuracy rule.
        
        Args:
            min_accuracy: Minimum required accuracy
            test_data: Test data for validation
            test_labels: True labels for test data
        """
        super().__init__(
            name="classification_accuracy",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that classification model has accuracy >= {min_accuracy}"
        )
        self.min_accuracy = min_accuracy
        self.test_data = test_data
        self.test_labels = test_labels
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for accuracy validation"
        
        try:
            from sklearn.metrics import accuracy_score
            
            # Make predictions
            predictions = model.predict(self.test_data)
            
            # Calculate accuracy
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(self.test_labels, 'values'):
                test_labels = self.test_labels.values
            else:
                test_labels = self.test_labels
            
            accuracy = accuracy_score(test_labels, predictions)
            
            if accuracy >= self.min_accuracy:
                return ValidationResult.PASS, f"Model accuracy is {accuracy:.4f}, which meets the minimum requirement of {self.min_accuracy}"
            else:
                return ValidationResult.FAIL, f"Model accuracy is {accuracy:.4f}, which is below the minimum requirement of {self.min_accuracy}"
        except Exception as e:
            return ValidationResult.ERROR, f"Error calculating classification accuracy: {str(e)}"


class ClassificationPrecisionRecallRule(ValidationRule):
    """Validates that a classification model meets precision and recall requirements."""
    
    def __init__(self, 
                 min_precision: float,
                 min_recall: float,
                 test_data: Optional[Any] = None, 
                 test_labels: Optional[Any] = None,
                 average: str = 'weighted'):
        """Initialize the precision-recall rule.
        
        Args:
            min_precision: Minimum required precision
            min_recall: Minimum required recall
            test_data: Test data for validation
            test_labels: True labels for test data
            average: Averaging method for multi-class metrics ('micro', 'macro', 'weighted')
        """
        super().__init__(
            name="classification_precision_recall",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that classification model has precision >= {min_precision} and recall >= {min_recall}"
        )
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.test_data = test_data
        self.test_labels = test_labels
        self.average = average
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for precision-recall validation"
        
        try:
            from sklearn.metrics import precision_score, recall_score
            
            # Make predictions
            predictions = model.predict(self.test_data)
            
            # Calculate metrics
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(self.test_labels, 'values'):
                test_labels = self.test_labels.values
            else:
                test_labels = self.test_labels
            
            precision = precision_score(test_labels, predictions, average=self.average, zero_division=0)
            recall = recall_score(test_labels, predictions, average=self.average, zero_division=0)
            
            if precision < self.min_precision and recall < self.min_recall:
                return ValidationResult.FAIL, f"Model performance is poor: precision={precision:.4f} (min {self.min_precision}), recall={recall:.4f} (min {self.min_recall})"
            elif precision < self.min_precision:
                return ValidationResult.WARNING, f"Model precision={precision:.4f} is below minimum required ({self.min_precision}), but recall={recall:.4f} is acceptable"
            elif recall < self.min_recall:
                return ValidationResult.WARNING, f"Model recall={recall:.4f} is below minimum required ({self.min_recall}), but precision={precision:.4f} is acceptable"
            return ValidationResult.PASS, f"Model performance is good: precision={precision:.4f}, recall={recall:.4f}"
        except Exception as e:
            return ValidationResult.ERROR, f"Error calculating precision-recall metrics: {str(e)}"


class TimeSeriesPerformanceRule(ValidationRule):
    """Validates that a time series model meets performance requirements."""
    
    def __init__(self, 
                 max_mse: float,
                 min_r2: float,
                 test_data: Optional[Any] = None, 
                 test_labels: Optional[Any] = None):
        """Initialize the time series performance rule.
        
        Args:
            max_mse: Maximum allowed mean squared error
            min_r2: Minimum required R² score
            test_data: Test data for validation
            test_labels: True values for test data
        """
        super().__init__(
            name="time_series_performance",
            rule_type=ValidationRuleType.PERFORMANCE,
            description=f"Validates that time series model has MSE <= {max_mse} and R² >= {min_r2}"
        )
        self.max_mse = max_mse
        self.min_r2 = min_r2
        self.test_data = test_data
        self.test_labels = test_labels
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if self.test_data is None or self.test_labels is None:
            return ValidationResult.ERROR, "Test data or labels not provided for time series performance validation"
        
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
            return ValidationResult.ERROR, f"Error calculating time series performance metrics: {str(e)}"


class HasPredictWithConfidenceMethod(ValidationRule):
    """Validates that a model has a predict_with_confidence method."""
    
    def __init__(self):
        """Initialize the has_predict_with_confidence rule."""
        super().__init__(
            name="has_predict_with_confidence",
            rule_type=ValidationRuleType.INTERFACE,
            description="Validates that model has a predict_with_confidence method"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        if hasattr(model, 'predict_with_confidence') and callable(getattr(model, 'predict_with_confidence')):
            return ValidationResult.PASS, "Model has a callable predict_with_confidence method"
        else:
            return ValidationResult.FAIL, "Model does not have a callable predict_with_confidence method"


class EnsembleModelConsistencyRule(ValidationRule):
    """Validates that an ensemble model has consistent base models."""
    
    def __init__(self):
        """Initialize the ensemble model consistency rule."""
        super().__init__(
            name="ensemble_model_consistency",
            rule_type=ValidationRuleType.STRUCTURE,
            description="Validates that ensemble model has consistent base models"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        # Check if the model has base_models attribute
        if not hasattr(model, 'base_models') or not model.base_models:
            return ValidationResult.FAIL, "Ensemble model does not have base_models attribute or it is empty"
        
        # Check if all base models have predict method
        for i, base_model in enumerate(model.base_models):
            if not hasattr(base_model, 'predict') or not callable(getattr(base_model, 'predict')):
                return ValidationResult.FAIL, f"Base model {i} does not have a callable predict method"
        
        return ValidationResult.PASS, f"Ensemble model has {len(model.base_models)} consistent base models"


class ReinforcementLearningInterfaceRule(ValidationRule):
    """Validates that a reinforcement learning model has the required interface."""
    
    def __init__(self):
        """Initialize the reinforcement learning interface rule."""
        super().__init__(
            name="reinforcement_learning_interface",
            rule_type=ValidationRuleType.INTERFACE,
            description="Validates that reinforcement learning model has the required interface"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        required_methods = ['act', 'learn']
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                missing_methods.append(method)
        
        if missing_methods:
            return ValidationResult.FAIL, f"Model is missing required methods: {', '.join(missing_methods)}"
        else:
            return ValidationResult.PASS, "Model has all required reinforcement learning interface methods"


def get_model_specific_validation_rules(model_type: ModelType, test_data=None, test_labels=None) -> List[ValidationRule]:
    """Get model-specific validation rules for a given model type.
    
    Args:
        model_type: Type of model to get validation rules for
        test_data: Test data for validation rules that require it
        test_labels: Test labels for validation rules that require them
        
    Returns:
        List of validation rules specific to the model type
    """
    if model_type == ModelType.REGRESSION:
        return [
            # Add regression-specific rules with test data if provided
            RegressionPerformanceRule(max_mse=1.0, min_r2=0.7, test_data=test_data, test_labels=test_labels) 
            if test_data is not None and test_labels is not None else None
        ]
    
    elif model_type == ModelType.CLASSIFICATION:
        return [
            # Add classification-specific rules with test data if provided
            ClassificationAccuracyRule(min_accuracy=0.8, test_data=test_data, test_labels=test_labels),
            ClassificationPrecisionRecallRule(min_precision=0.7, min_recall=0.7, 
                                             test_data=test_data, test_labels=test_labels)
            if test_data is not None and test_labels is not None else None
        ]
    
    elif model_type == ModelType.TIME_SERIES:
        return [
            # Add time series-specific rules with test data if provided
            TimeSeriesPerformanceRule(max_mse=1.0, min_r2=0.7, test_data=test_data, test_labels=test_labels),
            HasPredictWithConfidenceMethod()
            if test_data is not None and test_labels is not None else None
        ]
    
    elif model_type == ModelType.REINFORCEMENT_LEARNING:
        return [
            # Add reinforcement learning-specific rules
            ReinforcementLearningInterfaceRule()
        ]
    
    elif model_type == ModelType.ENSEMBLE:
        return [
            # Add ensemble-specific rules
            EnsembleModelConsistencyRule()
        ]
    
    else:  # ModelType.CUSTOM or unknown
        return []