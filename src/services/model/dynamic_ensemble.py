"""Dynamic Ensemble Models for Friday AI Trading System.

This module extends the ensemble methods with dynamic weight adjustment capabilities
based on model performance monitoring.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.services.model.ensemble_methods import (
    EnsembleModel, 
    WeightedEnsemble, 
    EnsembleWeightingStrategy
)
from src.services.model.model_monitoring import DynamicWeightAdjuster, ModelMonitor
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_database import ModelMetadata, ModelMetric

# Set up logger
logger = logging.getLogger(__name__)


class DynamicWeightedEnsemble(WeightedEnsemble):
    """Ensemble model with dynamic weight adjustment based on performance monitoring.
    
    This class extends the WeightedEnsemble to automatically adjust weights based on
    the performance of component models over time.
    """
    
    def __init__(self, 
                 models: List[Dict[str, Any]] = None,
                 optimization_metric: str = None,
                 performance_window: int = 10,
                 adjustment_threshold: float = 0.05,
                 registry: ModelRegistry = None):
        """Initialize a DynamicWeightedEnsemble.
        
        Args:
            models: List of model dictionaries with keys 'model', 'model_id', and 'weight'.
            optimization_metric: Metric to optimize weights for (e.g., 'accuracy', 'mse').
            performance_window: Number of recent performance samples to consider.
            adjustment_threshold: Minimum relative change in performance to trigger adjustment.
            registry: Optional model registry for loading/saving models.
        """
        # Initialize parent class with DYNAMIC weighting strategy
        super().__init__(models=models, 
                         optimization_metric=optimization_metric,
                         weighting_strategy=EnsembleWeightingStrategy.DYNAMIC)
        
        self.performance_window = performance_window
        self.adjustment_threshold = adjustment_threshold
        self.registry = registry
        
        # Performance history for component models
        self.performance_history = {}
        
        # Last adjustment timestamp
        self.last_adjustment = None
        
        logger.info("Initialized DynamicWeightedEnsemble with %d models", 
                   len(models) if models else 0)
    
    def record_performance(self, 
                          model_id: str, 
                          metric_name: str, 
                          value: float,
                          timestamp=None) -> None:
        """Record performance for a component model.
        
        Args:
            model_id: ID of the model.
            metric_name: Name of the performance metric.
            value: Value of the metric.
            timestamp: Optional timestamp for the performance record.
        """
        import datetime
        if not timestamp:
            timestamp = datetime.datetime.utcnow()
        
        # Initialize history for this model if needed
        if model_id not in self.performance_history:
            self.performance_history[model_id] = {}
        
        # Initialize history for this metric if needed
        if metric_name not in self.performance_history[model_id]:
            self.performance_history[model_id][metric_name] = []
        
        # Add performance record
        self.performance_history[model_id][metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Trim to window size
        if len(self.performance_history[model_id][metric_name]) > self.performance_window:
            self.performance_history[model_id][metric_name] = \
                self.performance_history[model_id][metric_name][-self.performance_window:]
        
        # Also record in model metrics database if available
        try:
            ModelMetric.record_metric(model_id, metric_name, value, timestamp)
        except Exception as e:
            logger.warning(f"Failed to record metric in database: {str(e)}")
    
    def adjust_weights(self, force: bool = False) -> bool:
        """Adjust weights based on recent performance.
        
        Args:
            force: Force adjustment even if threshold not met.
            
        Returns:
            bool: True if weights were adjusted, False otherwise.
        """
        if not self.optimization_metric:
            logger.warning("No optimization metric specified for weight adjustment")
            return False
        
        # Check if we have performance data for all models
        model_performances = {}
        for model_info in self.models:
            model_id = model_info.get("model_id")
            if not model_id:
                continue
            
            # Check if we have performance history for this model and metric
            if (model_id in self.performance_history and 
                self.optimization_metric in self.performance_history[model_id]):
                
                # Get average of recent performance
                history = self.performance_history[model_id][self.optimization_metric]
                if len(history) >= 3:  # Require at least 3 samples
                    avg_performance = sum(h["value"] for h in history) / len(history)
                    model_performances[model_id] = avg_performance
        
        # Check if we have enough performance data
        if len(model_performances) < len(self.models) / 2:  # At least half of models
            logger.warning("Insufficient performance data for weight adjustment")
            return False
        
        # Calculate new weights based on performance
        new_weights = self._calculate_weights(model_performances)
        
        # Check if adjustment threshold is met
        if not force:
            significant_change = False
            for model_info in self.models:
                model_id = model_info.get("model_id")
                if not model_id or model_id not in new_weights:
                    continue
                
                current_weight = model_info.get("weight", 0)
                new_weight = new_weights[model_id]
                
                # Check if relative change exceeds threshold
                if current_weight > 0:
                    relative_change = abs(new_weight - current_weight) / current_weight
                    if relative_change > self.adjustment_threshold:
                        significant_change = True
                        break
            
            if not significant_change:
                logger.info("No significant performance changes, skipping weight adjustment")
                return False
        
        # Update weights
        for model_id, weight in new_weights.items():
            self.update_weight(model_id, weight)
        
        # Update last adjustment timestamp
        import datetime
        self.last_adjustment = datetime.datetime.utcnow()
        
        logger.info("Adjusted weights based on %s performance", self.optimization_metric)
        return True
    
    def _calculate_weights(self, performances: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on model performances.
        
        Args:
            performances: Dictionary of model IDs to performance values.
            
        Returns:
            Dict[str, float]: Dictionary of model IDs to weights.
        """
        weights = {}
        
        # Determine if higher is better for this metric
        higher_is_better = self.optimization_metric in ["accuracy", "precision", "recall", "f1"]
        
        if higher_is_better:
            # For metrics where higher is better, weight proportional to value
            total = sum(performances.values())
            if total > 0:
                for model_id, value in performances.items():
                    weights[model_id] = value / total
            else:
                # Equal weights if total is zero
                equal_weight = 1.0 / len(performances)
                weights = {model_id: equal_weight for model_id in performances}
        else:
            # For metrics where lower is better (like error metrics),
            # weight inversely proportional to value
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            inverse_values = {model_id: 1.0 / (value + epsilon) for model_id, value in performances.items()}
            total = sum(inverse_values.values())
            if total > 0:
                for model_id, inv_value in inverse_values.items():
                    weights[model_id] = inv_value / total
            else:
                # Equal weights if total is zero
                equal_weight = 1.0 / len(performances)
                weights = {model_id: equal_weight for model_id in performances}
        
        return weights
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the ensemble.
        
        This method first checks if weights should be adjusted based on recent
        performance before making predictions.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Weighted average predictions.
        """
        # Optionally adjust weights before prediction
        # In a real system, you might want to do this less frequently
        # or in a separate process to avoid prediction latency
        try:
            self.adjust_weights(force=False)
        except Exception as e:
            logger.warning(f"Failed to adjust weights: {str(e)}")
        
        # Call parent class predict method
        return super().predict(X)
    
    def save(self, path: str) -> Dict[str, Any]:
        """Save the ensemble model to disk.
        
        Args:
            path: Directory path to save the model.
            
        Returns:
            Dict[str, Any]: Metadata about the saved model.
        """
        # Save additional metadata
        metadata = super().save(path)
        
        # Add dynamic ensemble specific metadata
        metadata["performance_window"] = self.performance_window
        metadata["adjustment_threshold"] = self.adjustment_threshold
        metadata["last_adjustment"] = self.last_adjustment.isoformat() if self.last_adjustment else None
        
        return metadata
    
    @classmethod
    def load(cls, path: str, registry: ModelRegistry = None) -> 'DynamicWeightedEnsemble':
        """Load a dynamic weighted ensemble from disk.
        
        Args:
            path: Path to the saved ensemble.
            registry: Optional model registry for loading component models.
            
        Returns:
            DynamicWeightedEnsemble: Loaded ensemble model.
        """
        # Load base ensemble
        ensemble = super().load(path, registry)
        
        # Create dynamic ensemble with same parameters
        dynamic_ensemble = cls(
            models=ensemble.models,
            optimization_metric=ensemble.optimization_metric,
            registry=registry
        )
        
        # Load additional metadata if available
        try:
            import json
            import os
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                if "performance_window" in metadata:
                    dynamic_ensemble.performance_window = metadata["performance_window"]
                
                if "adjustment_threshold" in metadata:
                    dynamic_ensemble.adjustment_threshold = metadata["adjustment_threshold"]
                
                if "last_adjustment" in metadata and metadata["last_adjustment"]:
                    import datetime
                    dynamic_ensemble.last_adjustment = datetime.datetime.fromisoformat(
                        metadata["last_adjustment"])
        except Exception as e:
            logger.warning(f"Failed to load dynamic ensemble metadata: {str(e)}")
        
        return dynamic_ensemble


class ABTestingEnsemble(EnsembleModel):
    """Ensemble model that performs A/B testing between component models.
    
    This ensemble routes predictions to different component models based on
    configured traffic splits, and tracks performance to determine the best model.
    """
    
    def __init__(self, 
                 models: List[Dict[str, Any]] = None,
                 traffic_splits: Dict[str, float] = None,
                 metrics: List[str] = None,
                 min_samples: int = 100):
        """Initialize an A/B testing ensemble.
        
        Args:
            models: List of model dictionaries with keys 'model', 'model_id'.
            traffic_splits: Dictionary mapping model IDs to traffic fractions (0-1).
            metrics: List of metrics to track for the test.
            min_samples: Minimum number of samples required for each model.
        """
        super().__init__(models=models, weighting_strategy=EnsembleWeightingStrategy.DYNAMIC)
        
        # Set default traffic splits if not provided
        if not traffic_splits and models:
            # Equal split by default
            equal_split = 1.0 / len(models)
            traffic_splits = {model_info.get("model_id"): equal_split 
                             for model_info in models if model_info.get("model_id")}
        
        self.traffic_splits = traffic_splits or {}
        
        # Set default metrics if not provided
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1", "latency"]
        self.min_samples = min_samples
        
        # Performance tracking
        self.performance_records = {}
        self.sample_counts = {}
        
        # Current winner
        self.winner = None
        
        logger.info("Initialized ABTestingEnsemble with %d models", 
                   len(models) if models else 0)
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the A/B testing ensemble.
        
        This method routes the prediction to a model based on the traffic splits.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Predictions from the selected model.
        """
        # If we have a clear winner and enough samples, use it exclusively
        if self.winner and all(count >= self.min_samples * 2 for count in self.sample_counts.values()):
            for model_info in self.models:
                if model_info.get("model_id") == self.winner:
                    return model_info["model"].predict(X)
        
        # Otherwise, select a model based on traffic splits
        model_ids = list(self.traffic_splits.keys())
        probabilities = list(self.traffic_splits.values())
        
        # Normalize probabilities if needed
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        
        # Select a model based on probabilities
        import random
        selected_id = random.choices(model_ids, weights=probabilities, k=1)[0]
        
        # Find the selected model
        for model_info in self.models:
            if model_info.get("model_id") == selected_id:
                # Increment sample count
                self.sample_counts[selected_id] = self.sample_counts.get(selected_id, 0) + 1
                
                # Make prediction
                return model_info["model"].predict(X)
        
        # Fallback to first model if selected not found
        logger.warning(f"Selected model {selected_id} not found, using first model")
        return self.models[0]["model"].predict(X)
    
    def record_performance(self, 
                          model_id: str, 
                          y_true: Any, 
                          y_pred: Any,
                          additional_metrics: Dict[str, float] = None) -> None:
        """Record performance for a component model.
        
        Args:
            model_id: ID of the model.
            y_true: True values.
            y_pred: Predicted values.
            additional_metrics: Additional pre-calculated metrics.
        """
        # Initialize records for this model if needed
        if model_id not in self.performance_records:
            self.performance_records[model_id] = {metric: [] for metric in self.metrics}
        
        # Calculate metrics if not provided
        metrics_to_record = additional_metrics or {}
        
        if not additional_metrics:
            try:
                # Convert to numpy arrays if needed
                if not isinstance(y_true, np.ndarray):
                    y_true = np.array(y_true)
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.array(y_pred)
                
                # Calculate requested metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score, 
                    mean_squared_error, mean_absolute_error
                )
                
                for metric in self.metrics:
                    if metric == "accuracy" and len(y_pred.shape) <= 1:
                        metrics_to_record[metric] = accuracy_score(y_true, y_pred)
                    elif metric == "precision" and len(y_pred.shape) <= 1:
                        metrics_to_record[metric] = precision_score(y_true, y_pred, average="weighted")
                    elif metric == "recall" and len(y_pred.shape) <= 1:
                        metrics_to_record[metric] = recall_score(y_true, y_pred, average="weighted")
                    elif metric == "f1" and len(y_pred.shape) <= 1:
                        metrics_to_record[metric] = f1_score(y_true, y_pred, average="weighted")
                    elif metric == "mse":
                        metrics_to_record[metric] = mean_squared_error(y_true, y_pred)
                    elif metric == "rmse":
                        metrics_to_record[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
                    elif metric == "mae":
                        metrics_to_record[metric] = mean_absolute_error(y_true, y_pred)
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
        
        # Record metrics
        for metric_name, value in metrics_to_record.items():
            if metric_name in self.performance_records[model_id]:
                self.performance_records[model_id][metric_name].append(value)
        
        # Check if we should evaluate the test
        self._check_test_completion()
    
    def _check_test_completion(self) -> None:
        """Check if the A/B test should be evaluated."""
        # Check if all models have reached minimum sample size
        if all(self.sample_counts.get(model_info.get("model_id"), 0) >= self.min_samples 
               for model_info in self.models):
            self._evaluate_test()
    
    def _evaluate_test(self) -> None:
        """Evaluate the A/B test and determine the winner."""
        results = {}
        
        # Calculate aggregate metrics for each model
        for model_info in self.models:
            model_id = model_info.get("model_id")
            if not model_id or model_id not in self.performance_records:
                continue
            
            results[model_id] = {}
            
            for metric_name, values in self.performance_records[model_id].items():
                if values:  # Only calculate if we have values
                    results[model_id][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }
        
        # Determine winner based on primary metric (first in the list)
        primary_metric = self.metrics[0] if self.metrics else None
        if primary_metric:
            # Find models with this metric
            candidate_models = [
                model_id for model_id in results 
                if primary_metric in results[model_id]
            ]
            
            if candidate_models:
                # For metrics where higher is better (accuracy, precision, recall, f1)
                higher_is_better = primary_metric in ["accuracy", "precision", "recall", "f1"]
                
                if higher_is_better:
                    # Find model with highest value
                    winner = max(
                        candidate_models,
                        key=lambda m: results[m][primary_metric]["mean"]
                    )
                else:
                    # Find model with lowest value
                    winner = min(
                        candidate_models,
                        key=lambda m: results[m][primary_metric]["mean"]
                    )
                
                # Set winner
                self.winner = winner
                logger.info(f"A/B test completed. Winner: {winner} based on {primary_metric}")
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get the results of the A/B test.
        
        Returns:
            Dict[str, Any]: Test results.
        """
        results = {
            "winner": self.winner,
            "primary_metric": self.metrics[0] if self.metrics else None,
            "sample_counts": self.sample_counts,
            "model_results": {}
        }
        
        # Calculate aggregate metrics for each model
        for model_info in self.models:
            model_id = model_info.get("model_id")
            if not model_id or model_id not in self.performance_records:
                continue
            
            results["model_results"][model_id] = {}
            
            for metric_name, values in self.performance_records[model_id].items():
                if values:  # Only calculate if we have values
                    results["model_results"][model_id][metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "count": len(values)
                    }
        
        return results
    
    def save(self, path: str) -> Dict[str, Any]:
        """Save the A/B testing ensemble to disk.
        
        Args:
            path: Directory path to save the model.
            
        Returns:
            Dict[str, Any]: Metadata about the saved model.
        """
        # Save base ensemble
        metadata = super().save(path)
        
        # Add A/B testing specific metadata
        metadata["traffic_splits"] = self.traffic_splits
        metadata["metrics"] = self.metrics
        metadata["min_samples"] = self.min_samples
        metadata["winner"] = self.winner
        metadata["sample_counts"] = self.sample_counts
        
        # Save test results
        import json
        import os
        results_path = os.path.join(path, "ab_test_results.json")
        with open(results_path, "w") as f:
            json.dump(self.get_test_results(), f, indent=2)
        
        return metadata
    
    @classmethod
    def load(cls, path: str, registry: ModelRegistry = None) -> 'ABTestingEnsemble':
        """Load an A/B testing ensemble from disk.
        
        Args:
            path: Path to the saved ensemble.
            registry: Optional model registry for loading component models.
            
        Returns:
            ABTestingEnsemble: Loaded ensemble model.
        """
        # Load base ensemble
        ensemble = super().load(path, registry)
        
        # Load metadata
        import json
        import os
        metadata_path = os.path.join(path, "metadata.json")
        
        traffic_splits = None
        metrics = None
        min_samples = 100
        winner = None
        sample_counts = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            traffic_splits = metadata.get("traffic_splits")
            metrics = metadata.get("metrics")
            min_samples = metadata.get("min_samples", 100)
            winner = metadata.get("winner")
            sample_counts = metadata.get("sample_counts", {})
        
        # Create A/B testing ensemble
        ab_ensemble = cls(
            models=ensemble.models,
            traffic_splits=traffic_splits,
            metrics=metrics,
            min_samples=min_samples
        )
        
        ab_ensemble.winner = winner
        ab_ensemble.sample_counts = sample_counts
        
        # Load test results if available
        results_path = os.path.join(path, "ab_test_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                
                # Reconstruct performance records
                for model_id, metrics in results.get("model_results", {}).items():
                    ab_ensemble.performance_records[model_id] = {}
                    for metric_name, stats in metrics.items():
                        # Create a synthetic list with the mean value repeated count times
                        count = stats.get("count", 0)
                        mean = stats.get("mean", 0)
                        ab_ensemble.performance_records[model_id][metric_name] = [mean] * count
            except Exception as e:
                logger.warning(f"Failed to load A/B test results: {str(e)}")
        
        return ab_ensemble