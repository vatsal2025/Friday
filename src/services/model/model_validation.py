"""Model Validation for Friday AI Trading System.

This module provides functionality for validating machine learning models
to ensure they meet quality standards before deployment.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ValidationResult
from src.services.model.model_validation_rules import ModelType, get_validation_rules_for_model_type
from src.services.model.validation_pipeline import ValidationPipeline, ValidationStage, create_default_pipeline
from src.services.model.benchmark_validation import BenchmarkValidator, BenchmarkDataset, BenchmarkMetric
from src.services.model.validation_report import ValidationReportGenerator, ReportFormat

# Create logger
logger = get_logger(__name__)


class ModelValidator:
    """Validator for machine learning models."""
    
    def __init__(self):
        """Initialize a model validator."""
        self.custom_validators = []
        logger.info("Initialized model validator")
    
    def register_validator(self, validator_func: Callable[[Any], bool]) -> None:
        """Register a custom validator function.
        
        Args:
            validator_func: Function that takes a model and returns True if valid, False otherwise
        """
        self.custom_validators.append(validator_func)
        logger.info(f"Registered custom validator: {validator_func.__name__}")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate a model using basic validation checks.
        
        Args:
            model: The model to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Basic validation
        is_valid, message = self._validate_basic(model)
        if not is_valid:
            return False, message
        
        # Run custom validators
        for validator in self.custom_validators:
            try:
                if not validator(model):
                    return False, f"Failed custom validation: {validator.__name__}"
            except Exception as e:
                return False, f"Error in custom validator {validator.__name__}: {str(e)}"
        
        return True, "Model passed basic validation"
    
    def _validate_basic(self, model: Any) -> Tuple[bool, str]:
        """Perform basic validation checks on a model.
        
        Args:
            model: The model to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if model is None
        if model is None:
            return False, "Model is None"
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            return False, "Model does not have a predict method"
        
        # Check if predict method is callable
        if not callable(getattr(model, 'predict')):
            return False, "Model predict method is not callable"
        
        return True, "Model passed basic validation"
    
    def validate_with_data(self, model: Any, test_data: np.ndarray, 
                         expected_output: np.ndarray = None) -> Tuple[bool, str]:
        """Validate a model using test data.
        
        Args:
            model: The model to validate
            test_data: Test data to validate with
            expected_output: Expected output for the test data
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Basic validation
        is_valid, message = self._validate_basic(model)
        if not is_valid:
            return False, message
        
        try:
            # Run prediction
            predictions = model.predict(test_data)
            
            # Check if predictions are valid
            if predictions is None:
                return False, "Model returned None predictions"
            
            # If expected output is provided, compare with predictions
            if expected_output is not None:
                # Check shapes
                if predictions.shape != expected_output.shape:
                    return False, f"Prediction shape {predictions.shape} does not match expected shape {expected_output.shape}"
                
                # Check values (simple equality check)
                if not np.array_equal(predictions, expected_output):
                    # Check if values are close (for floating point values)
                    if np.issubdtype(predictions.dtype, np.floating) and np.issubdtype(expected_output.dtype, np.floating):
                        if not np.allclose(predictions, expected_output):
                            return False, "Predictions do not match expected output"
                    else:
                        return False, "Predictions do not match expected output"
            
            return True, "Model passed data validation"
        
        except Exception as e:
            return False, f"Error validating model with data: {str(e)}"
    
    def create_validator_from_test_data(self, test_data: np.ndarray, 
                                      expected_output: np.ndarray) -> Callable[[Any], bool]:
        """Create a validator function from test data.
        
        Args:
            test_data: Test data to validate with
            expected_output: Expected output for the test data
            
        Returns:
            Callable[[Any], bool]: Validator function
        """
        def validator(model: Any) -> bool:
            is_valid, _ = self.validate_with_data(model, test_data, expected_output)
            return is_valid
        
        validator.__name__ = "test_data_validator"
        return validator
    
    def create_validator_from_function(self, func: Callable[[Any], bool]) -> Callable[[Any], bool]:
        """Create a validator function from a custom function.
        
        Args:
            func: Function that takes a model and returns True if valid, False otherwise
            
        Returns:
            Callable[[Any], bool]: Validator function
        """
        def validator(model: Any) -> bool:
            try:
                return func(model)
            except Exception as e:
                logger.error(f"Error in validator function: {str(e)}")
                return False
        
        validator.__name__ = func.__name__
        return validator
    
    def validate_model_with_pipeline(self, model: Any, model_type: ModelType, 
                                   metadata: Dict[str, Any] = None,
                                   pipeline_name: str = None,
                                   benchmark_datasets: List[BenchmarkDataset] = None,
                                   benchmark_metrics: List[BenchmarkMetric] = None,
                                   benchmark_thresholds: Dict[str, Dict[str, Any]] = None,
                                   generate_report: bool = True,
                                   report_format: ReportFormat = ReportFormat.HTML) -> Dict[str, Any]:
        """Validate a model using a validation pipeline.
        
        Args:
            model: The model to validate
            model_type: Type of the model
            metadata: Model metadata
            pipeline_name: Name of the validation pipeline to use
            benchmark_datasets: List of benchmark datasets to validate against
            benchmark_metrics: List of metrics to calculate for benchmark validation
            benchmark_thresholds: Dictionary of metric thresholds for benchmark validation
            generate_report: Whether to generate a validation report
            report_format: Format of the validation report
            
        Returns:
            Dict[str, Any]: Validation results
        """
        from src.services.model.validation_pipeline import get_pipeline
        
        # Use default pipeline if not specified
        if pipeline_name is None:
            pipeline_name = f"default_{model_type.value}"
        
        # Get the validation pipeline
        pipeline = get_pipeline(pipeline_name)
        if pipeline is None:
            # Create a default pipeline for the model type
            pipeline = create_default_pipeline(model_type)
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add model type to metadata
        metadata["model_type"] = model_type.value
        
        # Run the validation pipeline
        logger.info(f"Validating model using pipeline: {pipeline_name}")
        pipeline_result = pipeline.validate(model, metadata)
        
        # Run benchmark validation if datasets are provided
        benchmark_results = []
        if benchmark_datasets and benchmark_metrics:
            from src.services.model.benchmark_validation import benchmark_validator
            
            logger.info(f"Running benchmark validation with {len(benchmark_datasets)} datasets")
            for dataset in benchmark_datasets:
                # Validate against benchmark dataset
                benchmark_result = benchmark_validator.validate_model(
                    model, dataset.name, benchmark_metrics, benchmark_thresholds
                )
                benchmark_results.append(benchmark_result)
        
        # Generate validation report if requested
        report_path = None
        if generate_report:
            from src.services.model.validation_report import validation_report_generator
            
            # Get model name and version from metadata
            model_name = metadata.get("model_name", "unknown")
            model_version = metadata.get("model_version", "unknown")
            
            logger.info(f"Generating validation report for model: {model_name} v{model_version}")
            report_path = validation_report_generator.generate_report(
                model_name, model_version, pipeline_result, benchmark_results, metadata, report_format
            )
        
        # Create result dictionary
        result = {
            "overall_result": pipeline_result.overall_result,
            "pipeline_result": pipeline_result,
            "benchmark_results": benchmark_results,
            "report_path": report_path
        }
        
        return result


# Create a singleton instance
model_validator = ModelValidator()