"""Example script demonstrating the use of the benchmark validation system.

This script shows how to create benchmark datasets, register benchmark models,
and validate models against benchmarks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from src.services.model.benchmark_validation import (
    BenchmarkDataset,
    BenchmarkMetric,
    BenchmarkValidator,
    benchmark_validator
)


def create_regression_benchmark_dataset():
    """Create a sample regression benchmark dataset.
    
    Returns:
        BenchmarkDataset: The created benchmark dataset
    """
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["target"])
    
    # Create benchmark dataset
    dataset = BenchmarkDataset(
        name="regression_benchmark",
        description="Benchmark dataset for regression models",
        features=X_df,
        targets=y_df,
        metadata={
            "created_by": "benchmark_validation_example.py",
            "feature_count": X.shape[1],
            "sample_count": X.shape[0]
        }
    )
    
    return dataset


def create_classification_benchmark_dataset():
    """Create a sample classification benchmark dataset.
    
    Returns:
        BenchmarkDataset: The created benchmark dataset
    """
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["target"])
    
    # Create benchmark dataset
    dataset = BenchmarkDataset(
        name="classification_benchmark",
        description="Benchmark dataset for classification models",
        features=X_df,
        targets=y_df,
        metadata={
            "created_by": "benchmark_validation_example.py",
            "feature_count": X.shape[1],
            "sample_count": X.shape[0],
            "class_count": 2
        }
    )
    
    return dataset


def create_benchmark_models():
    """Create and register benchmark models.
    
    Returns:
        tuple: The regression and classification benchmark models
    """
    # Create regression benchmark model
    reg_X, reg_y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    reg_X_train, _, reg_y_train, _ = train_test_split(reg_X, reg_y, test_size=0.2, random_state=42)
    reg_benchmark = LinearRegression()
    reg_benchmark.fit(reg_X_train, reg_y_train)
    
    # Create classification benchmark model
    cls_X, cls_y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    cls_X_train, _, cls_y_train, _ = train_test_split(cls_X, cls_y, test_size=0.2, random_state=42)
    cls_benchmark = LogisticRegression(random_state=42)
    cls_benchmark.fit(cls_X_train, cls_y_train)
    
    return reg_benchmark, cls_benchmark


def example_register_benchmark_datasets():
    """Example of registering benchmark datasets."""
    print("\n=== Example: Register Benchmark Datasets ===")
    
    # Create benchmark datasets
    reg_dataset = create_regression_benchmark_dataset()
    cls_dataset = create_classification_benchmark_dataset()
    
    # Register datasets with the benchmark validator
    benchmark_validator.register_dataset(reg_dataset)
    benchmark_validator.register_dataset(cls_dataset)
    
    print(f"Registered benchmark datasets: {list(benchmark_validator.datasets.keys())}")


def example_register_benchmark_models():
    """Example of registering benchmark models."""
    print("\n=== Example: Register Benchmark Models ===")
    
    # Create benchmark models
    reg_benchmark, cls_benchmark = create_benchmark_models()
    
    # Register models with the benchmark validator
    benchmark_validator.register_benchmark_model(
        name="linear_regression_benchmark",
        model=reg_benchmark,
        description="Linear regression benchmark model"
    )
    
    benchmark_validator.register_benchmark_model(
        name="logistic_regression_benchmark",
        model=cls_benchmark,
        description="Logistic regression benchmark model"
    )
    
    print(f"Registered benchmark models: {list(benchmark_validator.benchmark_models.keys())}")


def example_validate_against_benchmark_dataset():
    """Example of validating a model against a benchmark dataset."""
    print("\n=== Example: Validate Against Benchmark Dataset ===")
    
    # Create a model to validate
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Define metrics and thresholds
    metrics = [BenchmarkMetric.MSE, BenchmarkMetric.R_SQUARED]
    thresholds = {
        "mse": {"threshold": 0.5, "comparison": "<="},  # MSE should be <= 0.5
        "r_squared": {"threshold": 0.7, "comparison": ">="}  # R² should be >= 0.7
    }
    
    # Validate the model
    result = benchmark_validator.validate_model(
        model=model,
        dataset_name="regression_benchmark",
        metrics=metrics,
        thresholds=thresholds
    )
    
    # Print results
    print(f"Validation result: {result.validation_result.name}")
    print(f"Message: {result.message}")
    print("Metrics:")
    for metric_name, metric_data in result.metrics.items():
        print(f"  {metric_name}: {metric_data['value']:.4f}")
        if 'threshold' in metric_data:
            print(f"    Threshold: {metric_data['threshold']} ({metric_data['comparison']})")
            print(f"    Passed: {metric_data['passed']}")


def example_compare_with_benchmark_model():
    """Example of comparing a model with a benchmark model."""
    print("\n=== Example: Compare With Benchmark Model ===")
    
    # Create a model to compare
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Define metrics
    metrics = [BenchmarkMetric.MSE, BenchmarkMetric.R_SQUARED]
    
    # Compare the model with the benchmark
    result = benchmark_validator.compare_with_benchmark(
        model=model,
        benchmark_name="linear_regression_benchmark",
        dataset_name="regression_benchmark",
        metrics=metrics
    )
    
    # Print results
    print(f"Comparison result: {result.validation_result.name}")
    print(f"Message: {result.message}")
    print("Metrics:")
    for metric_name, metric_data in result.metrics.items():
        print(f"  {metric_name}:")
        print(f"    Model: {metric_data['value']:.4f}")
        print(f"    Benchmark: {metric_data['benchmark_value']:.4f}")
        print(f"    Difference: {metric_data['difference']:.4f}")
        print(f"    Percent Difference: {metric_data['percent_difference']:.2f}%")
        print(f"    Better: {metric_data['better']}")


def main():
    """Run all examples."""
    # Register benchmark datasets
    example_register_benchmark_datasets()
    
    # Register benchmark models
    example_register_benchmark_models()
    
    # Validate against benchmark dataset
    example_validate_against_benchmark_dataset()
    
    # Compare with benchmark model
    example_compare_with_benchmark_model()


if __name__ == "__main__":
    main()