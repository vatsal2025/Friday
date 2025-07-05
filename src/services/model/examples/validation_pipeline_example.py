"""Example of using validation pipelines with multiple stages and model-specific rules.

This module demonstrates how to create and use validation pipelines with multiple stages,
including model-specific validation rules and benchmark validation.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# Import validation framework components
from src.services.model.model_validation import model_validator
from src.services.model.validation_pipeline import (
    ValidationPipeline,
    ValidationStage,
    ValidationPipelineResult,
    create_default_pipeline,
    validation_pipeline_registry
)
from src.services.model.model_validation_rules import (
    ValidationRule,
    ValidationRuleType,
    ValidationResult,
    ModelType,
    HasPredictMethod,
    HasFitMethod,
    ModelSizeRule,
    MinimumAccuracyRule,
    RegressionPerformanceRule,
    BenchmarkComparisonRule
)
from src.services.model.model_specific_validation_rules import (
    ClassificationAccuracyRule,
    ClassificationPrecisionRecallRule,
    TimeSeriesPerformanceRule,
    get_model_specific_validation_rules
)

logger = logging.getLogger(__name__)


def create_sample_classification_model() -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Create a sample classification model with test data.
    
    Returns:
        Tuple containing the trained model, test data, and test labels
    """
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


def create_sample_regression_model() -> Tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    """Create a sample regression model with test data.
    
    Returns:
        Tuple containing the trained model, test data, and test labels
    """
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


def create_benchmark_model(model_type: ModelType) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Create a benchmark model of the specified type with test data.
    
    Args:
        model_type: Type of model to create (REGRESSION or CLASSIFICATION)
        
    Returns:
        Tuple containing the trained benchmark model, test data, and test labels
    """
    if model_type == ModelType.REGRESSION:
        # Generate synthetic regression data
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train a linear regression model as benchmark
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    elif model_type == ModelType.CLASSIFICATION:
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_classes=2,
            random_state=42
        )
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train a logistic regression model as benchmark
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    else:
        raise ValueError(f"Unsupported model type for benchmark: {model_type}")


def create_custom_validation_pipeline() -> ValidationPipeline:
    """Create a custom validation pipeline with multiple stages.
    
    Returns:
        A configured ValidationPipeline instance
    """
    # Create a new validation pipeline with all stages
    pipeline = ValidationPipeline(
        name="custom_comprehensive_pipeline",
        stages=[
            ValidationStage.BASIC,
            ValidationStage.DATA_COMPATIBILITY,
            ValidationStage.PERFORMANCE,
            ValidationStage.RESOURCE,
            ValidationStage.SECURITY,
            ValidationStage.CUSTOM
        ]
    )
    
    # Add basic interface rules
    pipeline.add_rule(ValidationStage.BASIC, HasPredictMethod())
    pipeline.add_rule(ValidationStage.BASIC, HasFitMethod())
    
    # Add resource rules
    pipeline.add_rule(ValidationStage.RESOURCE, ModelSizeRule(max_size_mb=100, warning_threshold_mb=50))
    
    # Add custom rules
    pipeline.add_rule(ValidationStage.CUSTOM, CustomValidationRule())
    
    return pipeline


class CustomValidationRule(ValidationRule):
    """A custom validation rule for demonstration purposes."""
    
    def __init__(self):
        """Initialize the custom validation rule."""
        super().__init__(
            name="custom_rule",
            rule_type=ValidationRuleType.CUSTOM,
            description="A custom validation rule for demonstration"
        )
    
    def validate(self, model: Any, metadata: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        # This is just a placeholder custom rule that always passes
        # In a real scenario, you would implement specific validation logic
        return ValidationResult.PASS, "Custom validation rule passed"


def example_validate_with_default_pipeline():
    """Example of validating a model with the default pipeline."""
    print("\n=== Example: Validating with Default Pipeline ===")
    
    # Create a sample classification model
    model, X_test, y_test = create_sample_classification_model()
    
    # Validate the model using the default pipeline for classification models
    result = model_validator.validate_model_with_pipeline(
        model=model,
        model_type=ModelType.CLASSIFICATION,
        metadata={
            "name": "sample_classification_model",
            "version": "1.0.0",
            "description": "A sample classification model for validation"
        },
        test_data=X_test,
        test_labels=y_test,
        generate_report=True,
        report_format="markdown"
    )
    
    # Print the validation results
    print(f"Overall validation result: {result['overall_result']}")
    print(f"Report path: {result.get('report_path', 'No report generated')}")
    
    # Print failed rules if any
    pipeline_result = result['pipeline_result']
    if isinstance(pipeline_result, dict):
        pipeline_result = ValidationPipelineResult.from_dict(pipeline_result)
    
    failed_rules = pipeline_result.get_failed_rules()
    if failed_rules:
        print("\nFailed rules:")
        for rule_result in failed_rules:
            print(f"- {rule_result.rule_name}: {rule_result.message}")
    else:
        print("\nAll rules passed!")


def example_validate_with_custom_pipeline():
    """Example of validating a model with a custom pipeline."""
    print("\n=== Example: Validating with Custom Pipeline ===")
    
    # Create a sample regression model
    model, X_test, y_test = create_sample_regression_model()
    
    # Create a custom validation pipeline
    pipeline = create_custom_validation_pipeline()
    
    # Add model-specific rules to the pipeline
    model_specific_rules = get_model_specific_validation_rules(
        model_type=ModelType.REGRESSION,
        test_data=X_test,
        test_labels=y_test
    )
    
    for rule in model_specific_rules:
        if rule is not None:  # Some rules might be None if test data is not provided
            pipeline.add_rule(ValidationStage.PERFORMANCE, rule)
    
    # Register the custom pipeline
    validation_pipeline_registry.register_pipeline("custom_regression_pipeline", pipeline)
    
    # Validate the model using the custom pipeline
    result = model_validator.validate_model_with_pipeline(
        model=model,
        pipeline_name="custom_regression_pipeline",
        metadata={
            "name": "sample_regression_model",
            "version": "1.0.0",
            "description": "A sample regression model for validation"
        },
        test_data=X_test,
        test_labels=y_test,
        generate_report=True,
        report_format="json"
    )
    
    # Print the validation results
    print(f"Overall validation result: {result['overall_result']}")
    print(f"Report path: {result.get('report_path', 'No report generated')}")
    
    # Print warnings if any
    pipeline_result = result['pipeline_result']
    if isinstance(pipeline_result, dict):
        pipeline_result = ValidationPipelineResult.from_dict(pipeline_result)
    
    warnings = pipeline_result.get_warnings()
    if warnings:
        print("\nWarnings:")
        for rule_result in warnings:
            print(f"- {rule_result.rule_name}: {rule_result.message}")


def example_validate_with_benchmark_comparison():
    """Example of validating a model with benchmark comparison."""
    print("\n=== Example: Validating with Benchmark Comparison ===")
    
    # Create a sample classification model
    model, X_test, y_test = create_sample_classification_model()
    
    # Create a benchmark model
    benchmark_model, _, _ = create_benchmark_model(ModelType.CLASSIFICATION)
    
    # Create a benchmark comparison rule
    benchmark_rule = BenchmarkComparisonRule(
        benchmark_model=benchmark_model,
        min_improvement=0.05,  # Require 5% improvement over benchmark
        test_data=X_test,
        test_labels=y_test,
        metric="accuracy"  # Use accuracy as the comparison metric
    )
    
    # Create a custom pipeline with the benchmark rule
    pipeline = ValidationPipeline(
        name="benchmark_comparison_pipeline",
        stages=[ValidationStage.BASIC, ValidationStage.PERFORMANCE]
    )
    
    pipeline.add_rule(ValidationStage.BASIC, HasPredictMethod())
    pipeline.add_rule(ValidationStage.PERFORMANCE, benchmark_rule)
    
    # Add a standard accuracy rule for comparison
    pipeline.add_rule(
        ValidationStage.PERFORMANCE,
        ClassificationAccuracyRule(min_accuracy=0.8, test_data=X_test, test_labels=y_test)
    )
    
    # Register the pipeline
    validation_pipeline_registry.register_pipeline("benchmark_comparison_pipeline", pipeline)
    
    # Validate the model using the benchmark comparison pipeline
    result = model_validator.validate_model_with_pipeline(
        model=model,
        pipeline_name="benchmark_comparison_pipeline",
        metadata={
            "name": "sample_classification_model",
            "version": "1.0.0",
            "description": "A sample classification model for benchmark comparison"
        },
        generate_report=True,
        report_format="html"
    )
    
    # Print the validation results
    print(f"Overall validation result: {result['overall_result']}")
    print(f"Report path: {result.get('report_path', 'No report generated')}")
    
    # Print all rule results
    pipeline_result = result['pipeline_result']
    if isinstance(pipeline_result, dict):
        pipeline_result = ValidationPipelineResult.from_dict(pipeline_result)
    
    print("\nRule results:")
    for rule_result in pipeline_result.rule_results:
        print(f"- {rule_result.rule_name}: {rule_result.result} - {rule_result.message}")


def main():
    """Run all validation pipeline examples."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    example_validate_with_default_pipeline()
    example_validate_with_custom_pipeline()
    example_validate_with_benchmark_comparison()


if __name__ == "__main__":
    main()