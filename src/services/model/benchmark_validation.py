"""Benchmark Validation for Friday AI Trading System.

This module provides functionality for validating model performance against benchmark datasets.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ValidationResult
from src.backtesting.performance import PerformanceMetrics, PerformanceAnalytics, BenchmarkComparison

# Create logger
logger = get_logger(__name__)


class BenchmarkMetric(Enum):
    """Metrics for benchmark comparison."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"  # Mean Squared Error
    RMSE = "rmse"  # Root Mean Squared Error
    MAE = "mae"  # Mean Absolute Error
    R_SQUARED = "r_squared"
    ALPHA = "alpha"  # Jensen's Alpha
    BETA = "beta"  # Market Beta
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    CORRELATION = "correlation"
    UP_CAPTURE = "up_capture"
    DOWN_CAPTURE = "down_capture"
    OUTPERFORMANCE_PCT = "outperformance_pct"


class BenchmarkDataset:
    """Benchmark dataset for model validation."""
    
    def __init__(self, name: str, description: str, features: pd.DataFrame, 
                 targets: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Initialize a benchmark dataset.
        
        Args:
            name: Name of the benchmark dataset
            description: Description of the benchmark dataset
            features: Feature data (X)
            targets: Target data (y)
            metadata: Additional metadata about the dataset
        """
        self.name = name
        self.description = description
        self.features = features
        self.targets = targets
        self.metadata = metadata or {}
        
        # Validate inputs
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        
        logger.info(f"Initialized benchmark dataset '{name}' with {len(features)} samples")
    
    @classmethod
    def from_file(cls, file_path: str, name: str = None, description: str = None) -> 'BenchmarkDataset':
        """Load a benchmark dataset from a file.
        
        Args:
            file_path: Path to the dataset file (CSV or JSON)
            name: Name of the benchmark dataset (defaults to filename)
            description: Description of the benchmark dataset
            
        Returns:
            BenchmarkDataset: Loaded benchmark dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmark dataset file not found: {file_path}")
        
        # Default name to filename if not provided
        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load data based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            data = pd.read_csv(file_path)
        elif file_ext == '.json':
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Check if the JSON has a specific structure
            if 'features' in json_data and 'targets' in json_data:
                features = pd.DataFrame(json_data['features'])
                targets = pd.DataFrame(json_data['targets'])
                metadata = json_data.get('metadata', {})
                return cls(name, description or json_data.get('description', ''), 
                           features, targets, metadata)
            else:
                # Assume it's a flat structure with target columns specified in metadata
                data = pd.DataFrame(json_data)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # For CSV and flat JSON, we need to determine which columns are targets
        # This should be specified in the metadata or we use the last column as target
        if 'metadata' in data.columns:
            metadata = json.loads(data['metadata'].iloc[0])
            data = data.drop(columns=['metadata'])
        else:
            metadata = {}
        
        target_columns = metadata.get('target_columns', [data.columns[-1]])
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        features = data[feature_columns]
        targets = data[target_columns]
        
        return cls(name, description or '', features, targets, metadata)


class BenchmarkResult:
    """Result of a benchmark validation."""
    
    def __init__(self, dataset_name: str, model_name: str):
        """Initialize a benchmark result.
        
        Args:
            dataset_name: Name of the benchmark dataset
            model_name: Name of the model being validated
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.metrics = {}
        self.predictions = None
        self.benchmark_predictions = None
        self.validation_result = ValidationResult.PASS
        self.message = ""
    
    def add_metric(self, metric: Union[str, BenchmarkMetric], value: float, 
                  threshold: float = None, comparison: str = None) -> None:
        """Add a benchmark metric result.
        
        Args:
            metric: Metric name or BenchmarkMetric enum
            value: Metric value
            threshold: Threshold value for validation
            comparison: Comparison operator ('>', '<', '>=', '<=', '==')
        """
        metric_name = metric.value if isinstance(metric, BenchmarkMetric) else metric
        
        metric_result = {
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "passed": True
        }
        
        # Validate against threshold if provided
        if threshold is not None and comparison is not None:
            if comparison == '>' and not value > threshold:
                metric_result["passed"] = False
            elif comparison == '<' and not value < threshold:
                metric_result["passed"] = False
            elif comparison == '>=' and not value >= threshold:
                metric_result["passed"] = False
            elif comparison == '<=' and not value <= threshold:
                metric_result["passed"] = False
            elif comparison == '==' and not value == threshold:
                metric_result["passed"] = False
        
        self.metrics[metric_name] = metric_result
    
    def set_validation_result(self, result: ValidationResult, message: str) -> None:
        """Set the overall validation result.
        
        Args:
            result: Validation result
            message: Validation message
        """
        self.validation_result = result
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark result to a dictionary.
        
        Returns:
            Dictionary representation of the benchmark result
        """
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "validation_result": self.validation_result.value,
            "message": self.message
        }


class BenchmarkValidator:
    """Validator for benchmarking model performance."""
    
    def __init__(self):
        """Initialize a benchmark validator."""
        self.datasets = {}
        self.benchmark_models = {}
        logger.info("Initialized benchmark validator")
    
    def register_dataset(self, dataset: BenchmarkDataset) -> None:
        """Register a benchmark dataset.
        
        Args:
            dataset: Benchmark dataset to register
        """
        self.datasets[dataset.name] = dataset
        logger.info(f"Registered benchmark dataset: {dataset.name}")
    
    def register_benchmark_model(self, name: str, model: Any, 
                               description: str = None) -> None:
        """Register a benchmark model.
        
        Args:
            name: Name of the benchmark model
            model: The benchmark model
            description: Description of the benchmark model
        """
        self.benchmark_models[name] = {
            "model": model,
            "description": description or ""
        }
        logger.info(f"Registered benchmark model: {name}")
    
    def validate_model(self, model: Any, dataset_name: str, 
                      metrics: List[BenchmarkMetric], 
                      thresholds: Dict[str, Dict[str, Any]] = None) -> BenchmarkResult:
        """Validate a model against a benchmark dataset.
        
        Args:
            model: The model to validate
            dataset_name: Name of the benchmark dataset
            metrics: List of metrics to calculate
            thresholds: Dictionary of metric thresholds for validation
                        {metric_name: {"threshold": value, "comparison": operator}}
            
        Returns:
            BenchmarkResult: Result of the benchmark validation
        """
        # Get the benchmark dataset
        if dataset_name not in self.datasets:
            raise ValueError(f"Benchmark dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        model_name = getattr(model, "__name__", str(model.__class__.__name__))
        result = BenchmarkResult(dataset_name, model_name)
        
        try:
            # Make predictions
            predictions = model.predict(dataset.features)
            result.predictions = predictions
            
            # Calculate metrics
            for metric in metrics:
                metric_name = metric.value if isinstance(metric, BenchmarkMetric) else metric
                metric_value = self._calculate_metric(metric, predictions, dataset.targets)
                
                # Get threshold for this metric if provided
                threshold_info = thresholds.get(metric_name, {}) if thresholds else {}
                threshold = threshold_info.get("threshold")
                comparison = threshold_info.get("comparison")
                
                result.add_metric(metric, metric_value, threshold, comparison)
            
            # Determine overall validation result
            failed_metrics = [name for name, info in result.metrics.items() if not info["passed"]]
            
            if failed_metrics:
                result.set_validation_result(
                    ValidationResult.FAIL,
                    f"Failed metrics: {', '.join(failed_metrics)}"
                )
            else:
                result.set_validation_result(
                    ValidationResult.PASS,
                    "All metrics passed validation thresholds"
                )
        
        except Exception as e:
            logger.error(f"Error validating model against benchmark dataset: {str(e)}")
            result.set_validation_result(
                ValidationResult.ERROR,
                f"Error validating model: {str(e)}"
            )
        
        return result
    
    def compare_with_benchmark(self, model: Any, benchmark_name: str, 
                             dataset_name: str, metrics: List[BenchmarkMetric]) -> BenchmarkResult:
        """Compare a model with a benchmark model.
        
        Args:
            model: The model to validate
            benchmark_name: Name of the benchmark model to compare with
            dataset_name: Name of the benchmark dataset
            metrics: List of metrics to calculate
            
        Returns:
            BenchmarkResult: Result of the benchmark comparison
        """
        # Get the benchmark dataset and model
        if dataset_name not in self.datasets:
            raise ValueError(f"Benchmark dataset not found: {dataset_name}")
        
        if benchmark_name not in self.benchmark_models:
            raise ValueError(f"Benchmark model not found: {benchmark_name}")
        
        dataset = self.datasets[dataset_name]
        benchmark_model = self.benchmark_models[benchmark_name]["model"]
        model_name = getattr(model, "__name__", str(model.__class__.__name__))
        result = BenchmarkResult(dataset_name, model_name)
        
        try:
            # Make predictions with both models
            predictions = model.predict(dataset.features)
            benchmark_predictions = benchmark_model.predict(dataset.features)
            
            result.predictions = predictions
            result.benchmark_predictions = benchmark_predictions
            
            # Calculate metrics for both models and compare
            for metric in metrics:
                metric_name = metric.value if isinstance(metric, BenchmarkMetric) else metric
                
                # Calculate metric for the model being validated
                model_metric = self._calculate_metric(metric, predictions, dataset.targets)
                
                # Calculate metric for the benchmark model
                benchmark_metric = self._calculate_metric(metric, benchmark_predictions, dataset.targets)
                
                # Determine if higher or lower is better for this metric
                higher_is_better = self._is_higher_better(metric)
                
                # Compare with benchmark
                if higher_is_better:
                    passed = model_metric >= benchmark_metric
                    comparison = ">="
                else:
                    passed = model_metric <= benchmark_metric
                    comparison = "<="
                
                # Add metric result
                result.metrics[metric_name] = {
                    "value": model_metric,
                    "benchmark_value": benchmark_metric,
                    "difference": model_metric - benchmark_metric,
                    "percent_difference": ((model_metric / benchmark_metric) - 1) * 100 if benchmark_metric != 0 else float('inf'),
                    "comparison": comparison,
                    "passed": passed
                }
            
            # Determine overall validation result
            failed_metrics = [name for name, info in result.metrics.items() if not info["passed"]]
            
            if failed_metrics:
                result.set_validation_result(
                    ValidationResult.FAIL,
                    f"Model underperforms benchmark on metrics: {', '.join(failed_metrics)}"
                )
            else:
                result.set_validation_result(
                    ValidationResult.PASS,
                    "Model meets or exceeds benchmark performance on all metrics"
                )
        
        except Exception as e:
            logger.error(f"Error comparing model with benchmark: {str(e)}")
            result.set_validation_result(
                ValidationResult.ERROR,
                f"Error comparing model with benchmark: {str(e)}"
            )
        
        return result
    
    def _calculate_metric(self, metric: Union[str, BenchmarkMetric], 
                         predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate a metric value.
        
        Args:
            metric: Metric to calculate
            predictions: Model predictions
            targets: True target values
            
        Returns:
            float: Calculated metric value
        """
        metric_name = metric.value if isinstance(metric, BenchmarkMetric) else metric
        
        # Classification metrics
        if metric_name == BenchmarkMetric.ACCURACY.value:
            from sklearn.metrics import accuracy_score
            return accuracy_score(targets, predictions)
        
        elif metric_name == BenchmarkMetric.PRECISION.value:
            from sklearn.metrics import precision_score
            return precision_score(targets, predictions, average='weighted')
        
        elif metric_name == BenchmarkMetric.RECALL.value:
            from sklearn.metrics import recall_score
            return recall_score(targets, predictions, average='weighted')
        
        elif metric_name == BenchmarkMetric.F1_SCORE.value:
            from sklearn.metrics import f1_score
            return f1_score(targets, predictions, average='weighted')
        
        elif metric_name == BenchmarkMetric.ROC_AUC.value:
            from sklearn.metrics import roc_auc_score
            # Handle multi-class case
            if len(np.unique(targets)) > 2:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(targets)
                targets_bin = label_binarize(targets, classes=classes)
                if len(classes) == 2:
                    return roc_auc_score(targets, predictions)
                else:
                    return roc_auc_score(targets_bin, predictions, multi_class='ovr')
            else:
                return roc_auc_score(targets, predictions)
        
        # Regression metrics
        elif metric_name == BenchmarkMetric.MSE.value:
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(targets, predictions)
        
        elif metric_name == BenchmarkMetric.RMSE.value:
            from sklearn.metrics import mean_squared_error
            return np.sqrt(mean_squared_error(targets, predictions))
        
        elif metric_name == BenchmarkMetric.MAE.value:
            from sklearn.metrics import mean_absolute_error
            return mean_absolute_error(targets, predictions)
        
        elif metric_name == BenchmarkMetric.R_SQUARED.value:
            from sklearn.metrics import r2_score
            return r2_score(targets, predictions)
        
        # Trading strategy metrics
        elif metric_name in [m.value for m in BenchmarkMetric if m.value in [
            'alpha', 'beta', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'information_ratio', 'tracking_error', 'correlation',
            'up_capture', 'down_capture', 'outperformance_pct'
        ]]:
            # These metrics require equity curves or returns
            # We assume predictions and targets are returns series
            if metric_name == BenchmarkMetric.ALPHA.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_alpha()
            
            elif metric_name == BenchmarkMetric.BETA.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_beta()
            
            elif metric_name == BenchmarkMetric.SHARPE_RATIO.value:
                # Assuming predictions are returns
                return np.mean(predictions) / np.std(predictions) * np.sqrt(252)  # Annualized
            
            elif metric_name == BenchmarkMetric.SORTINO_RATIO.value:
                # Assuming predictions are returns
                negative_returns = predictions[predictions < 0]
                downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
                return np.mean(predictions) / downside_deviation * np.sqrt(252)  # Annualized
            
            elif metric_name == BenchmarkMetric.MAX_DRAWDOWN.value:
                # Calculate cumulative returns
                cum_returns = (1 + predictions).cumprod()
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns / running_max) - 1
                return np.min(drawdown)
            
            elif metric_name == BenchmarkMetric.INFORMATION_RATIO.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_information_ratio()
            
            elif metric_name == BenchmarkMetric.TRACKING_ERROR.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_tracking_error()
            
            elif metric_name == BenchmarkMetric.CORRELATION.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_correlation()
            
            elif metric_name == BenchmarkMetric.UP_CAPTURE.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_up_capture()
            
            elif metric_name == BenchmarkMetric.DOWN_CAPTURE.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_down_capture()
            
            elif metric_name == BenchmarkMetric.OUTPERFORMANCE_PCT.value:
                benchmark_comparison = BenchmarkComparison(predictions, targets)
                return benchmark_comparison.calculate_outperformance_percentage()
        
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    
    def _is_higher_better(self, metric: Union[str, BenchmarkMetric]) -> bool:
        """Determine if higher values are better for a metric.
        
        Args:
            metric: Metric to check
            
        Returns:
            bool: True if higher values are better, False otherwise
        """
        metric_name = metric.value if isinstance(metric, BenchmarkMetric) else metric
        
        # Metrics where higher is better
        higher_better_metrics = [
            BenchmarkMetric.ACCURACY.value,
            BenchmarkMetric.PRECISION.value,
            BenchmarkMetric.RECALL.value,
            BenchmarkMetric.F1_SCORE.value,
            BenchmarkMetric.ROC_AUC.value,
            BenchmarkMetric.R_SQUARED.value,
            BenchmarkMetric.ALPHA.value,
            BenchmarkMetric.SHARPE_RATIO.value,
            BenchmarkMetric.SORTINO_RATIO.value,
            BenchmarkMetric.INFORMATION_RATIO.value,
            BenchmarkMetric.CORRELATION.value,
            BenchmarkMetric.OUTPERFORMANCE_PCT.value
        ]
        
        # Metrics where lower is better
        lower_better_metrics = [
            BenchmarkMetric.MSE.value,
            BenchmarkMetric.RMSE.value,
            BenchmarkMetric.MAE.value,
            BenchmarkMetric.MAX_DRAWDOWN.value,
            BenchmarkMetric.TRACKING_ERROR.value
        ]
        
        # Special cases
        special_cases = {
            BenchmarkMetric.BETA.value: None,  # Depends on strategy
            BenchmarkMetric.UP_CAPTURE.value: True,  # Higher is better
            BenchmarkMetric.DOWN_CAPTURE.value: False  # Lower is better
        }
        
        if metric_name in higher_better_metrics:
            return True
        elif metric_name in lower_better_metrics:
            return False
        elif metric_name in special_cases:
            return special_cases[metric_name]
        else:
            # Default to higher is better
            return True


# Create a singleton instance
benchmark_validator = BenchmarkValidator()