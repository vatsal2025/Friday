import datetime
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import ttest_ind

from src.services.model.model_registry import ModelRegistry
from src.services.model.model_database import ModelMetadata
from src.services.model.dynamic_ensemble import ABTestingEnsemble

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Class for monitoring model performance and drift."""
    
    def __init__(self, registry: ModelRegistry):
        """Initialize a model monitor.
        
        Args:
            registry: Model registry to use for loading and saving models.
        """
        self.registry = registry
        self.performance_records = {}
        logger.info("Initialized ModelMonitor with registry %s", registry)
    
    def record_prediction(self, 
                          model_id: str, 
                          input_data: Any, 
                          prediction: Any, 
                          actual: Any = None,
                          metadata: Dict[str, Any] = None) -> None:
        """Record a prediction for monitoring.
        
        Args:
            model_id: ID of the model that made the prediction.
            input_data: Input data for the prediction.
            prediction: The prediction made by the model.
            actual: The actual value, if available.
            metadata: Additional metadata about the prediction.
        """
        if model_id not in self.performance_records:
            self.performance_records[model_id] = []
        
        record = {
            "timestamp": datetime.datetime.utcnow(),
            "input": input_data,
            "prediction": prediction,
            "metadata": metadata or {}
        }
        
        if actual is not None:
            record["actual"] = actual
        
        self.performance_records[model_id].append(record)
        logger.debug("Recorded prediction for model %s", model_id)
    
    def calculate_metrics(self, model_id: str) -> Dict[str, float]:
        """Calculate performance metrics for a model.
        
        Args:
            model_id: ID of the model to calculate metrics for.
            
        Returns:
            Dict[str, float]: Dictionary of metric names to values.
            
        Raises:
            ValueError: If the model ID doesn't exist or has no records with actuals.
        """
        if model_id not in self.performance_records:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Filter records that have actual values
        records_with_actuals = [
            record for record in self.performance_records[model_id]
            if "actual" in record
        ]
        
        if not records_with_actuals:
            raise ValueError(f"Model ID {model_id} has no records with actual values")
        
        # Extract predictions and actuals
        predictions = [record["prediction"] for record in records_with_actuals]
        actuals = [record["actual"] for record in records_with_actuals]
        
        # Calculate metrics based on data type
        metrics = {}
        
        # Check if data is numeric
        if all(isinstance(p, (int, float)) for p in predictions) and \
           all(isinstance(a, (int, float)) for a in actuals):
            # Regression metrics
            metrics["mse"] = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        else:
            # Classification metrics
            correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
            total = len(predictions)
            metrics["accuracy"] = correct / total if total > 0 else 0
            
            # Calculate precision, recall, f1 for each class
            classes = set(actuals)
            for cls in classes:
                # True positives: predicted cls and actual is cls
                tp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a == cls)
                # False positives: predicted cls but actual is not cls
                fp = sum(1 for p, a in zip(predictions, actuals) if p == cls and a != cls)
                # False negatives: predicted not cls but actual is cls
                fn = sum(1 for p, a in zip(predictions, actuals) if p != cls and a == cls)
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[f"precision_{cls}"] = precision
                metrics[f"recall_{cls}"] = recall
                metrics[f"f1_{cls}"] = f1
        
        logger.info("Calculated metrics for model %s: %s", model_id, metrics)
        return metrics
    
    def detect_drift(self, model_id: str, window_size: int = 100) -> Dict[str, Any]:
        """Detect drift in model inputs and performance.
        
        Args:
            model_id: ID of the model to detect drift for.
            window_size: Size of the window to use for drift detection.
            
        Returns:
            Dict[str, Any]: Drift detection results.
            
        Raises:
            ValueError: If the model ID doesn't exist or has insufficient records.
        """
        if model_id not in self.performance_records:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        records = self.performance_records[model_id]
        if len(records) < window_size * 2:
            raise ValueError(f"Insufficient records for model {model_id}. Need at least {window_size * 2}.")
        
        # Split records into two windows
        recent_window = records[-window_size:]
        previous_window = records[-2 * window_size:-window_size]
        
        # Check for input drift
        # This is a simplified implementation and would need to be adapted
        # based on the specific input data structure
        input_drift = {
            "detected": False,
            "details": "Input drift detection not implemented"
        }
        
        # Check for performance drift
        performance_drift = {
            "detected": False,
            "details": {}
        }
        
        # Calculate metrics for both windows if actuals are available
        recent_with_actuals = [r for r in recent_window if "actual" in r]
        previous_with_actuals = [r for r in previous_window if "actual" in r]
        
        if recent_with_actuals and previous_with_actuals:
            # Extract predictions and actuals
            recent_predictions = [r["prediction"] for r in recent_with_actuals]
            recent_actuals = [r["actual"] for r in recent_with_actuals]
            previous_predictions = [r["prediction"] for r in previous_with_actuals]
            previous_actuals = [r["actual"] for r in previous_with_actuals]
            
            # Calculate error for both windows
            if all(isinstance(p, (int, float)) for p in recent_predictions + previous_predictions) and \
               all(isinstance(a, (int, float)) for a in recent_actuals + previous_actuals):
                # Regression errors
                recent_errors = [abs(p - a) for p, a in zip(recent_predictions, recent_actuals)]
                previous_errors = [abs(p - a) for p, a in zip(previous_predictions, previous_actuals)]
                
                # Check if mean error has increased significantly
                recent_mean_error = np.mean(recent_errors)
                previous_mean_error = np.mean(previous_errors)
                error_change_pct = (recent_mean_error - previous_mean_error) / previous_mean_error \
                    if previous_mean_error > 0 else 0
                
                performance_drift["details"]["error_change_pct"] = error_change_pct
                performance_drift["detected"] = error_change_pct > 0.1  # 10% increase in error
            else:
                # Classification accuracy
                recent_correct = sum(1 for p, a in zip(recent_predictions, recent_actuals) if p == a)
                recent_accuracy = recent_correct / len(recent_predictions) if recent_predictions else 0
                
                previous_correct = sum(1 for p, a in zip(previous_predictions, previous_actuals) if p == a)
                previous_accuracy = previous_correct / len(previous_predictions) if previous_predictions else 0
                
                accuracy_change_pct = (previous_accuracy - recent_accuracy) / previous_accuracy \
                    if previous_accuracy > 0 else 0
                
                performance_drift["details"]["accuracy_change_pct"] = accuracy_change_pct
                performance_drift["detected"] = accuracy_change_pct > 0.05  # 5% decrease in accuracy
        
        result = {
            "input_drift": input_drift,
            "performance_drift": performance_drift,
            "timestamp": datetime.datetime.utcnow()
        }
        
        logger.info("Drift detection for model %s: %s", model_id, result)
        return result
    
    def save_monitoring_data(self, output_dir: str) -> None:
        """Save monitoring data to disk.
        
        Args:
            output_dir: Directory to save monitoring data to.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance records
        for model_id, records in self.performance_records.items():
            # Convert records to serializable format
            serializable_records = []
            for record in records:
                serializable_record = {}
                for key, value in record.items():
                    if key == "timestamp":
                        serializable_record[key] = value.isoformat()
                    elif key in ["input", "prediction", "actual"]:
                        # Convert numpy arrays to lists
                        if hasattr(value, "tolist"):
                            serializable_record[key] = value.tolist()
                        else:
                            serializable_record[key] = value
                    else:
                        serializable_record[key] = value
                serializable_records.append(serializable_record)
            
            # Save to file
            output_path = os.path.join(output_dir, f"{model_id}_monitoring.json")
            with open(output_path, "w") as f:
                json.dump(serializable_records, f, indent=2)
        
        logger.info("Saved monitoring data to %s", output_dir)
    
    def load_monitoring_data(self, input_dir: str) -> None:
        """Load monitoring data from disk.
        
        Args:
            input_dir: Directory to load monitoring data from.
        """
        if not os.path.exists(input_dir):
            logger.warning("Input directory %s does not exist", input_dir)
            return
        
        # Clear existing records
        self.performance_records = {}
        
        # Load performance records
        for filename in os.listdir(input_dir):
            if filename.endswith("_monitoring.json"):
                model_id = filename.replace("_monitoring.json", "")
                input_path = os.path.join(input_dir, filename)
                
                try:
                    with open(input_path, "r") as f:
                        serialized_records = json.load(f)
                    
                    # Convert records back to original format
                    records = []
                    for serialized_record in serialized_records:
                        record = {}
                        for key, value in serialized_record.items():
                            if key == "timestamp":
                                record[key] = datetime.datetime.fromisoformat(value)
                            else:
                                record[key] = value
                        records.append(record)
                    
                    self.performance_records[model_id] = records
                except Exception as e:
                    logger.error("Error loading monitoring data for model %s: %s", model_id, e)
        
        logger.info("Loaded monitoring data from %s", input_dir)


class ABTestingFramework:
    """Framework for A/B testing models and ensembles."""
    
    def __init__(self, registry: ModelRegistry):
        """Initialize the A/B testing framework.
        
        Args:
            registry: Model registry to use for loading and saving models.
        """
        self.registry = registry
        self.active_tests = {}
        logger.info("Initialized ABTestingFramework with registry %s", registry)
    
    def create_test(self, 
                   test_id: str,
                   model_a_id: str,
                   model_b_id: str,
                   traffic_split: float = 0.5,
                   metrics: List[str] = None,
                   duration_days: int = 7,
                   min_samples: int = 100) -> Dict[str, Any]:
        """Create a new A/B test.
        
        Args:
            test_id: ID for the test.
            model_a_id: ID of the first model (control).
            model_b_id: ID of the second model (variant).
            traffic_split: Fraction of traffic to route to model B (0-1).
            metrics: List of metrics to track for the test.
            duration_days: Duration of the test in days.
            min_samples: Minimum number of samples required for each model.
            
        Returns:
            Dict[str, Any]: Test configuration.
            
        Raises:
            ValueError: If the test ID already exists or if the models don't exist.
        """
        if test_id in self.active_tests:
            raise ValueError(f"Test ID {test_id} already exists")
        
        # Verify models exist
        model_a_metadata = ModelMetadata.get_by_id(model_a_id)
        model_b_metadata = ModelMetadata.get_by_id(model_b_id)
        
        if not model_a_metadata:
            raise ValueError(f"Model A ID {model_a_id} does not exist")
        if not model_b_metadata:
            raise ValueError(f"Model B ID {model_b_id} does not exist")
        
        # Set default metrics if not provided
        if metrics is None:
            # Use appropriate metrics based on model type
            if model_a_metadata.model_type == model_b_metadata.model_type:
                if model_a_metadata.model_type == "classification":
                    metrics = ["accuracy", "precision", "recall", "f1"]
                else:  # regression
                    metrics = ["mse", "rmse", "mae"]
            else:
                # If model types differ, use generic metrics
                metrics = ["accuracy"]
        
        # Create test configuration
        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(days=duration_days)
        
        test_config = {
            "test_id": test_id,
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "traffic_split": traffic_split,
            "metrics": metrics,
            "min_samples": min_samples,
            "start_time": start_time,
            "end_time": end_time,
            "status": "active",
            "results": {
                "model_a": {
                    "samples": 0,
                    "metrics": {metric: [] for metric in metrics}
                },
                "model_b": {
                    "samples": 0,
                    "metrics": {metric: [] for metric in metrics}
                }
            }
        }
        
        self.active_tests[test_id] = test_config
        logger.info("Created A/B test %s between models %s and %s", 
                   test_id, model_a_id, model_b_id)
        
        return test_config
    
    def record_performance(self, 
                          test_id: str,
                          model_key: str,
                          input_data: Any,
                          actual: Any,
                          metrics: Dict[str, float]) -> None:
        """Record performance metrics for a model in an A/B test.
        
        Args:
            test_id: ID of the test.
            model_key: Key of the model ("model_a" or "model_b").
            input_data: Input data for the prediction.
            actual: The actual value.
            metrics: Dictionary of metric names to values.
            
        Raises:
            ValueError: If the test ID doesn't exist or the model key is invalid.
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test ID {test_id} does not exist")
        
        if model_key not in ["model_a", "model_b"]:
            raise ValueError(f"Invalid model key {model_key}. Must be 'model_a' or 'model_b'.")
        
        test = self.active_tests[test_id]
        
        # Only record if test is active
        if test["status"] != "active":
            logger.warning("Test %s is not active. Performance not recorded.", test_id)
            return
        
        # Increment sample count
        test["results"][model_key]["samples"] += 1
        
        # Record metrics
        for metric_name, value in metrics.items():
            if metric_name in test["results"][model_key]["metrics"]:
                test["results"][model_key]["metrics"][metric_name].append(value)
        
        # Check if test should be completed
        self._check_test_completion(test_id)
    
    def _check_test_completion(self, test_id: str) -> None:
        """Check if a test should be completed and evaluated.
        
        Args:
            test_id: ID of the test to check.
        """
        test = self.active_tests[test_id]
        
        # Check if test has reached end time
        now = datetime.datetime.utcnow()
        if now >= test["end_time"]:
            self.evaluate_test(test_id)
            return
        
        # Check if both models have reached minimum sample size
        if (test["results"]["model_a"]["samples"] >= test["min_samples"] and
                test["results"]["model_b"]["samples"] >= test["min_samples"]):
            # Check if there's a clear winner with statistical significance
            primary_metric = test["metrics"][0] if test["metrics"] else None
            if primary_metric:
                # Get metric values for both models
                values_a = test["results"]["model_a"]["metrics"].get(primary_metric, [])
                values_b = test["results"]["model_b"]["metrics"].get(primary_metric, [])
                
                if values_a and values_b:
                    # Perform t-test to check for statistical significance
                    t_stat, p_value = ttest_ind(values_a, values_b, equal_var=False)
                    
                    # If p-value is less than significance level (0.05), there is a statistically significant difference
                    if p_value < 0.05:
                        logger.info(f"A/B test {test_id} has statistically significant results (p-value: {p_value:.4f})")
                        self.evaluate_test(test_id)
                        return
                    else:
                        logger.debug(f"A/B test {test_id} does not yet have statistically significant results (p-value: {p_value:.4f})")
    
    def evaluate_test(self, test_id: str) -> Dict[str, Any]:
        """Evaluate an A/B test and determine the winner.
        
        Args:
            test_id: ID of the test to evaluate.
            
        Returns:
            Dict[str, Any]: Test results with winner.
            
        Raises:
            ValueError: If the test ID doesn't exist.
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test ID {test_id} does not exist")
        
        test = self.active_tests[test_id]
        
        # Mark test as completed
        test["status"] = "completed"
        test["completion_time"] = datetime.datetime.utcnow()
        
        # Calculate aggregate metrics for each model
        results = {"model_a": {}, "model_b": {}}
        
        for model_key in ["model_a", "model_b"]:
            for metric_name, values in test["results"][model_key]["metrics"].items():
                if values:  # Only calculate if we have values
                    results[model_key][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }
        
        test["aggregate_results"] = results
        
        # Determine winner based on primary metric (first in the list)
        primary_metric = test["metrics"][0] if test["metrics"] else None
        if primary_metric and primary_metric in results["model_a"] and primary_metric in results["model_b"]:
            # For metrics where higher is better (accuracy, precision, recall, f1)
            higher_is_better = primary_metric in ["accuracy", "precision", "recall", "f1"]
            
            a_value = results["model_a"][primary_metric]["mean"]
            b_value = results["model_b"][primary_metric]["mean"]
            
            if higher_is_better:
                winner = "model_a" if a_value > b_value else "model_b"
                margin = abs(a_value - b_value) / max(a_value, b_value) if max(a_value, b_value) > 0 else 0
            else:  # Lower is better (mse, rmse, mae)
                winner = "model_a" if a_value < b_value else "model_b"
                margin = abs(a_value - b_value) / max(a_value, b_value) if max(a_value, b_value) > 0 else 0
            
            test["winner"] = {
                "model_id": test[f"{winner}_id"],
                "margin": margin,
                "primary_metric": primary_metric,
                "a_value": a_value,
                "b_value": b_value
            }
            
            logger.info(f"A/B test {test_id} completed. Winner: {winner} with {primary_metric} margin of {margin:.2%}")
        
        return test
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get the results of an A/B test.
        
        Args:
            test_id: ID of the test.
            
        Returns:
            Dict[str, Any]: Test results.
            
        Raises:
            ValueError: If the test ID doesn't exist.
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test ID {test_id} does not exist")
        
        return self.active_tests[test_id]
    
    def list_tests(self, status: str = None) -> List[Dict[str, Any]]:
        """List all A/B tests, optionally filtered by status.
        
        Args:
            status: Optional status to filter by ("active", "completed").
            
        Returns:
            List[Dict[str, Any]]: List of test configurations.
        """
        if status:
            return [test for test in self.active_tests.values() if test["status"] == status]
        else:
            return list(self.active_tests.values())


class DynamicWeightAdjuster:
    """Class for dynamically adjusting weights in ensemble models based on performance."""
    
    def __init__(self, 
                 registry: ModelRegistry,
                 performance_window: int = 10,
                 adjustment_interval: int = 86400,  # Default to daily adjustment
                 min_performance_samples: int = 5):
        """Initialize a dynamic weight adjuster.
        
        Args:
            registry: Model registry to use for loading and saving models.
            performance_window: Number of performance records to consider for adjustment.
            adjustment_interval: Interval between adjustments in seconds.
            min_performance_samples: Minimum number of performance samples required for adjustment.
        """
        self.registry = registry
        self.performance_window = performance_window
        self.adjustment_interval = adjustment_interval
        self.min_performance_samples = min_performance_samples
        self.last_adjustment_time = {}
        self.performance_records = {}
        logger.info("Initialized DynamicWeightAdjuster with registry %s", registry)
    
    def record_performance(self, 
                          ensemble_id: str,
                          model_id: str,
                          input_data: Any,
                          actual: Any,
                          metrics: Dict[str, float]) -> None:
        """Record performance metrics for a model in an ensemble.
        
        Args:
            ensemble_id: ID of the ensemble.
            model_id: ID of the model within the ensemble.
            input_data: Input data for the prediction.
            actual: The actual value.
            metrics: Dictionary of metric names to values.
        """
        if ensemble_id not in self.performance_records:
            self.performance_records[ensemble_id] = {}
        
        if model_id not in self.performance_records[ensemble_id]:
            self.performance_records[ensemble_id][model_id] = []
        
        record = {
            "timestamp": datetime.datetime.utcnow(),
            "input": input_data,
            "actual": actual,
            "metrics": metrics
        }
        
        self.performance_records[ensemble_id][model_id].append(record)
        logger.debug("Recorded performance for model %s in ensemble %s", model_id, ensemble_id)
        
        # Check if we should adjust weights
        self._check_adjustment(ensemble_id)
    
    def _check_adjustment(self, ensemble_id: str) -> None:
        """Check if weights should be adjusted for an ensemble.
        
        Args:
            ensemble_id: ID of the ensemble to check.
        """
        # Check if enough time has passed since last adjustment
        now = datetime.datetime.utcnow()
        last_time = self.last_adjustment_time.get(ensemble_id)
        
        if last_time and (now - last_time).total_seconds() < self.adjustment_interval:
            return
        
        # Check if we have enough performance records for all models
        if ensemble_id not in self.performance_records:
            return
        
        model_records = self.performance_records[ensemble_id]
        if not all(len(records) >= self.min_performance_samples for records in model_records.values()):
            return
        
        # Adjust weights
        self._adjust_weights(ensemble_id)
        self.last_adjustment_time[ensemble_id] = now
    
    def _adjust_weights(self, ensemble_id: str) -> None:
        """Adjust weights for an ensemble based on performance.
        
        Args:
            ensemble_id: ID of the ensemble to adjust weights for.
        """
        try:
            # Load the ensemble
            ensemble = self.registry.load_model(ensemble_id)
            
            # Get recent performance records for each model
            model_records = self.performance_records[ensemble_id]
            recent_records = {}
            
            for model_id, records in model_records.items():
                recent_records[model_id] = records[-self.performance_window:]
            
            # Calculate average performance for each model
            model_performance = {}
            
            for model_id, records in recent_records.items():
                # Get the primary metric (first metric in the first record)
                if not records or "metrics" not in records[0]:
                    continue
                
                metrics = records[0]["metrics"]
                if not metrics:
                    continue
                
                primary_metric = list(metrics.keys())[0]
                
                # Calculate average for the primary metric
                values = [record["metrics"].get(primary_metric, 0) for record in records 
                          if "metrics" in record and primary_metric in record["metrics"]]
                
                if values:
                    model_performance[model_id] = np.mean(values)
            
            # Adjust weights based on performance
            if hasattr(ensemble, "models") and model_performance:
                # Normalize performance to sum to 1
                total_performance = sum(model_performance.values())
                normalized_performance = {}
                
                for model_id, performance in model_performance.items():
                    normalized_performance[model_id] = performance / total_performance if total_performance > 0 else 0
                
                # Update weights
                for model_info in ensemble.models:
                    model_id = model_info.get("model_id")
                    if model_id in normalized_performance:
                        model_info["weight"] = normalized_performance[model_id]
                
                # Save the updated ensemble
                self.registry.save_model(ensemble, ensemble_id)
                logger.info("Adjusted weights for ensemble %s: %s", 
                           ensemble_id, {m.get("model_id"): m.get("weight") for m in ensemble.models})
        except Exception as e:
            logger.error("Error adjusting weights for ensemble %s: %s", ensemble_id, e)
    
    def save_performance_data(self, output_dir: str) -> None:
        """Save performance data to disk.
        
        Args:
            output_dir: Directory to save performance data to.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance records
        for ensemble_id, model_records in self.performance_records.items():
            ensemble_dir = os.path.join(output_dir, ensemble_id)
            os.makedirs(ensemble_dir, exist_ok=True)
            
            for model_id, records in model_records.items():
                # Convert records to serializable format
                serializable_records = []
                for record in records:
                    serializable_record = {}
                    for key, value in record.items():
                        if key == "timestamp":
                            serializable_record[key] = value.isoformat()
                        elif key in ["input", "actual"]:
                            # Convert numpy arrays to lists
                            if hasattr(value, "tolist"):
                                serializable_record[key] = value.tolist()
                            else:
                                serializable_record[key] = value
                        else:
                            serializable_record[key] = value
                    serializable_records.append(serializable_record)
                
                # Save to file
                output_path = os.path.join(ensemble_dir, f"{model_id}_performance.json")
                with open(output_path, "w") as f:
                    json.dump(serializable_records, f, indent=2)
        
        # Save last adjustment times
        adjustment_times = {}
        for ensemble_id, timestamp in self.last_adjustment_time.items():
            adjustment_times[ensemble_id] = timestamp.isoformat()
        
        output_path = os.path.join(output_dir, "adjustment_times.json")
        with open(output_path, "w") as f:
            json.dump(adjustment_times, f, indent=2)
        
        logger.info("Saved performance data to %s", output_dir)
    
    def load_performance_data(self, input_dir: str) -> None:
        """Load performance data from disk.
        
        Args:
            input_dir: Directory to load performance data from.
        """
        if not os.path.exists(input_dir):
            logger.warning("Input directory %s does not exist", input_dir)
            return
        
        # Clear existing records
        self.performance_records = {}
        self.last_adjustment_time = {}
        
        # Load performance records
        for ensemble_dir in os.listdir(input_dir):
            ensemble_path = os.path.join(input_dir, ensemble_dir)
            if os.path.isdir(ensemble_path) and ensemble_dir != "__pycache__":
                ensemble_id = ensemble_dir
                self.performance_records[ensemble_id] = {}
                
                for filename in os.listdir(ensemble_path):
                    if filename.endswith("_performance.json"):
                        model_id = filename.replace("_performance.json", "")
                        input_path = os.path.join(ensemble_path, filename)
                        
                        try:
                            with open(input_path, "r") as f:
                                serialized_records = json.load(f)
                            
                            # Convert records back to original format
                            records = []
                            for serialized_record in serialized_records:
                                record = {}
                                for key, value in serialized_record.items():
                                    if key == "timestamp":
                                        record[key] = datetime.datetime.fromisoformat(value)
                                    else:
                                        record[key] = value
                                records.append(record)
                            
                            self.performance_records[ensemble_id][model_id] = records
                        except Exception as e:
                            logger.error("Error loading performance data for model %s in ensemble %s: %s", 
                                       model_id, ensemble_id, e)
        
        # Load last adjustment times
        adjustment_times_path = os.path.join(input_dir, "adjustment_times.json")
        if os.path.exists(adjustment_times_path):
            try:
                with open(adjustment_times_path, "r") as f:
                    serialized_times = json.load(f)
                
                for ensemble_id, timestamp_str in serialized_times.items():
                    self.last_adjustment_time[ensemble_id] = datetime.datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.error("Error loading adjustment times: %s", e)
        
        logger.info("Loaded performance data from %s", input_dir)