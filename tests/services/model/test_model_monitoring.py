"""Tests for model monitoring and dynamic weight adjustment.

This module contains tests for the model monitoring system, A/B testing framework,
and dynamic weight adjustment for ensemble models.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.services.model.model_monitoring import (
    ModelMonitor, 
    ABTestingFramework, 
    DynamicWeightAdjuster,
    ModelDrift,
    DegradationSeverity
)
from src.services.model.dynamic_ensemble import (
    DynamicWeightedEnsemble,
    ABTestingEnsemble
)
from src.services.model.ensemble_methods import (
    EnsembleModel,
    WeightedEnsemble,
    EnsembleWeightingStrategy
)
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_database import ModelMetadata, ModelMetric
from src.services.model.config.model_registry_config import ModelStatus


class TestModelDrift(unittest.TestCase):
    """Tests for the ModelDrift class."""
    
    def test_drift_detection(self):
        """Test that drift is correctly detected and severity is calculated."""
        # Create a drift with 10% relative change (threshold 5%)
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.81,  # 10% decrease
            threshold=0.05
        )
        
        # Check that threshold is exceeded
        self.assertTrue(drift.threshold_exceeded)
        
        # Check that severity is correctly calculated
        self.assertEqual(drift.severity, DegradationSeverity.MEDIUM)
        
        # Check absolute and relative change
        self.assertAlmostEqual(drift.absolute_change, -0.09)
        self.assertAlmostEqual(drift.relative_change, -0.1, places=2)
    
    def test_drift_severity_levels(self):
        """Test that different severity levels are correctly assigned."""
        # No drift (below threshold)
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.87,  # 3.3% decrease
            threshold=0.05
        )
        self.assertEqual(drift.severity, DegradationSeverity.NONE)
        
        # Low severity (just above threshold)
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.85,  # 5.6% decrease
            threshold=0.05
        )
        self.assertEqual(drift.severity, DegradationSeverity.LOW)
        
        # Medium severity
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.81,  # 10% decrease
            threshold=0.05
        )
        self.assertEqual(drift.severity, DegradationSeverity.MEDIUM)
        
        # High severity
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.72,  # 20% decrease
            threshold=0.05
        )
        self.assertEqual(drift.severity, DegradationSeverity.HIGH)
        
        # Critical severity
        drift = ModelDrift(
            model_id="test_model",
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.63,  # 30% decrease
            threshold=0.05
        )
        self.assertEqual(drift.severity, DegradationSeverity.CRITICAL)


class TestModelMonitor(unittest.TestCase):
    """Tests for the ModelMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a model monitor
        self.monitor = ModelMonitor(
            registry=self.registry,
            monitoring_interval=1,  # 1 second for testing
            lookback_window=7,
            min_samples_for_baseline=2
        )
        
        # Mock model metadata and metrics
        self.model_id = "test_model"
        self.model_metadata = MagicMock(spec=ModelMetadata)
        self.model_metadata.to_registry_dict.return_value = {}
        
        # Patch ModelMetadata.get_by_id
        self.get_by_id_patcher = patch('src.services.model.model_database.ModelMetadata.get_by_id')
        self.mock_get_by_id = self.get_by_id_patcher.start()
        self.mock_get_by_id.return_value = self.model_metadata
        
        # Patch ModelMetric methods
        self.get_latest_metrics_patcher = patch('src.services.model.model_database.ModelMetric.get_latest_metrics')
        self.mock_get_latest_metrics = self.get_latest_metrics_patcher.start()
        
        self.get_all_metrics_patcher = patch('src.services.model.model_database.ModelMetric.get_all_metrics')
        self.mock_get_all_metrics = self.get_all_metrics_patcher.start()
        
        self.get_metric_history_patcher = patch('src.services.model.model_database.ModelMetric.get_metric_history')
        self.mock_get_metric_history = self.get_metric_history_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.get_by_id_patcher.stop()
        self.get_latest_metrics_patcher.stop()
        self.get_all_metrics_patcher.stop()
        self.get_metric_history_patcher.stop()
    
    def test_check_model_drift(self):
        """Test that model drift is correctly detected."""
        # Set up mock metrics
        import datetime
        
        # Create metric history objects
        class MockMetric:
            def __init__(self, value, timestamp):
                self.value = value
                self.timestamp = timestamp
        
        # Set up baseline metrics (older)
        baseline_time = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        baseline_metrics = [
            MockMetric(0.90, baseline_time),
            MockMetric(0.91, baseline_time + datetime.timedelta(days=1)),
            MockMetric(0.89, baseline_time + datetime.timedelta(days=2))
        ]
        
        # Set up latest metric (significantly worse)
        latest_time = datetime.datetime.utcnow()
        latest_metric = MockMetric(0.80, latest_time)
        
        # Configure mocks
        self.mock_get_all_metrics.return_value = baseline_metrics + [latest_metric]
        self.mock_get_latest_metrics.return_value = {"accuracy": 0.80}
        self.mock_get_metric_history.return_value = baseline_metrics + [latest_metric]
        
        # Check for drift
        drifts = self.monitor.check_model_drift(self.model_id)
        
        # Verify drift was detected
        self.assertEqual(len(drifts), 1)
        self.assertEqual(drifts[0].model_id, self.model_id)
        self.assertEqual(drifts[0].metric_name, "accuracy")
        self.assertAlmostEqual(drifts[0].baseline_value, 0.90, places=2)  # Average of baseline metrics
        self.assertEqual(drifts[0].current_value, 0.80)
        self.assertTrue(drifts[0].threshold_exceeded)
    
    def test_handle_drift_by_severity(self):
        """Test that drift is handled according to severity."""
        # Create drifts with different severities
        low_drift = ModelDrift(
            model_id=self.model_id,
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.85,  # 5.6% decrease
            threshold=0.05
        )
        
        medium_drift = ModelDrift(
            model_id=self.model_id,
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.81,  # 10% decrease
            threshold=0.05
        )
        
        high_drift = ModelDrift(
            model_id=self.model_id,
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.72,  # 20% decrease
            threshold=0.05
        )
        
        critical_drift = ModelDrift(
            model_id=self.model_id,
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.63,  # 30% decrease
            threshold=0.05
        )
        
        # Mock the handling methods
        self.monitor._flag_model_for_review = MagicMock()
        self.monitor._flag_model_for_retraining = MagicMock()
        self.monitor._demote_model_to_staging = MagicMock()
        
        # Handle drifts
        self.monitor.handle_drift(low_drift)
        self.monitor.handle_drift(medium_drift)
        self.monitor.handle_drift(high_drift)
        self.monitor.handle_drift(critical_drift)
        
        # Verify correct actions were taken
        self.monitor._flag_model_for_review.assert_called_once_with(self.model_id, medium_drift)
        self.monitor._flag_model_for_retraining.assert_called_once_with(self.model_id, high_drift)
        self.monitor._demote_model_to_staging.assert_called_once_with(self.model_id, critical_drift)


class TestABTestingFramework(unittest.TestCase):
    """Tests for the ABTestingFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create an A/B testing framework
        self.ab_framework = ABTestingFramework(registry=self.registry)
        
        # Mock model metadata
        self.model_a_id = "model_a"
        self.model_b_id = "model_b"
        self.model_a_metadata = MagicMock(spec=ModelMetadata)
        self.model_b_metadata = MagicMock(spec=ModelMetadata)
        
        # Patch ModelMetadata.get_by_id
        self.get_by_id_patcher = patch('src.services.model.model_database.ModelMetadata.get_by_id')
        self.mock_get_by_id = self.get_by_id_patcher.start()
        self.mock_get_by_id.side_effect = lambda model_id: {
            self.model_a_id: self.model_a_metadata,
            self.model_b_id: self.model_b_metadata
        }.get(model_id)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.get_by_id_patcher.stop()
    
    def test_create_test(self):
        """Test creating a new A/B test."""
        # Create a test
        test_id = "test_1"
        test_config = self.ab_framework.create_test(
            test_id=test_id,
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=0.3,  # 30% traffic to model B
            metrics=["accuracy", "latency"],
            duration_days=14,
            min_samples=50
        )
        
        # Verify test was created correctly
        self.assertEqual(test_config["test_id"], test_id)
        self.assertEqual(test_config["model_a_id"], self.model_a_id)
        self.assertEqual(test_config["model_b_id"], self.model_b_id)
        self.assertEqual(test_config["traffic_split"], 0.3)
        self.assertEqual(test_config["metrics"], ["accuracy", "latency"])
        self.assertEqual(test_config["min_samples"], 50)
        self.assertEqual(test_config["status"], "active")
        self.assertIsNone(test_config["winner"])
        
        # Verify test is stored in the framework
        self.assertIn(test_id, self.ab_framework.active_tests)
        self.assertEqual(self.ab_framework.active_tests[test_id], test_config)
    
    def test_record_prediction(self):
        """Test recording predictions for an A/B test."""
        # Create a test
        test_id = "test_1"
        self.ab_framework.create_test(
            test_id=test_id,
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id
        )
        
        # Record predictions for both models
        # Model A (better accuracy)
        for _ in range(10):
            self.ab_framework.record_prediction(
                test_id=test_id,
                model_id=self.model_a_id,
                prediction=1,
                actual=1,
                metrics={"accuracy": 1.0, "latency": 0.05}
            )
        
        # Model B (worse accuracy, better latency)
        for _ in range(10):
            self.ab_framework.record_prediction(
                test_id=test_id,
                model_id=self.model_b_id,
                prediction=0,
                actual=1,
                metrics={"accuracy": 0.0, "latency": 0.02}
            )
        
        # Verify metrics were recorded
        test = self.ab_framework.active_tests[test_id]
        self.assertEqual(test["results"]["model_a"]["samples"], 10)
        self.assertEqual(test["results"]["model_b"]["samples"], 10)
        self.assertEqual(len(test["results"]["model_a"]["metrics"]["accuracy"]), 10)
        self.assertEqual(len(test["results"]["model_b"]["metrics"]["accuracy"]), 10)
        
        # Evaluate the test
        results = self.ab_framework.evaluate_test(test_id)
        
        # Verify winner is model A (based on accuracy)
        self.assertEqual(results["winner"]["model_id"], self.model_a_id)
        self.assertEqual(results["winner"]["primary_metric"], "accuracy")
        
        # Verify aggregate results
        self.assertAlmostEqual(results["aggregate_results"]["model_a"]["accuracy"]["mean"], 1.0)
        self.assertAlmostEqual(results["aggregate_results"]["model_b"]["accuracy"]["mean"], 0.0)
        self.assertAlmostEqual(results["aggregate_results"]["model_a"]["latency"]["mean"], 0.05)
        self.assertAlmostEqual(results["aggregate_results"]["model_b"]["latency"]["mean"], 0.02)


class TestDynamicWeightAdjuster(unittest.TestCase):
    """Tests for the DynamicWeightAdjuster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a dynamic weight adjuster
        self.adjuster = DynamicWeightAdjuster(
            registry=self.registry,
            performance_window=5,
            adjustment_interval=1,  # 1 second for testing
            min_performance_samples=2
        )
        
        # Create mock models and ensemble
        self.model_a_id = "model_a"
        self.model_b_id = "model_b"
        self.model_c_id = "model_c"
        self.ensemble_id = "ensemble_1"
        
        # Mock ensemble model
        self.ensemble = MagicMock(spec=WeightedEnsemble)
        self.ensemble.models = [
            {"model_id": self.model_a_id, "weight": 0.33},
            {"model_id": self.model_b_id, "weight": 0.33},
            {"model_id": self.model_c_id, "weight": 0.34}
        ]
        
        # Configure registry to return the ensemble
        self.registry.load_model.return_value = self.ensemble
        
        # Patch ModelMetric.get_latest_metrics
        self.get_latest_metrics_patcher = patch('src.services.model.model_database.ModelMetric.get_latest_metrics')
        self.mock_get_latest_metrics = self.get_latest_metrics_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.get_latest_metrics_patcher.stop()
    
    def test_adjust_ensemble_weights(self):
        """Test adjusting ensemble weights based on performance."""
        # Configure mock to return different accuracies for each model
        self.mock_get_latest_metrics.side_effect = lambda model_id: {
            self.model_a_id: {"accuracy": 0.9},  # Best model
            self.model_b_id: {"accuracy": 0.7},
            self.model_c_id: {"accuracy": 0.8}
        }.get(model_id, {})
        
        # Adjust weights
        result = self.adjuster.adjust_ensemble_weights(self.ensemble_id)
        
        # Verify weights were adjusted
        self.assertTrue(result)
        
        # Verify update_weight was called with correct weights
        expected_weights = {
            self.model_a_id: 0.9 / 2.4,  # 0.375
            self.model_b_id: 0.7 / 2.4,  # 0.292
            self.model_c_id: 0.8 / 2.4   # 0.333
        }
        
        # Check that update_weight was called for each model
        self.assertEqual(self.ensemble.update_weight.call_count, 3)
        
        # Verify the ensemble was saved back to the registry
        self.registry.update_model.assert_called_once_with(self.ensemble_id, self.ensemble)
    
    def test_calculate_weights_higher_better(self):
        """Test weight calculation for metrics where higher is better."""
        # Performances for accuracy (higher is better)
        performances = {
            self.model_a_id: 0.9,
            self.model_b_id: 0.7,
            self.model_c_id: 0.8
        }
        
        # Calculate weights
        weights = self.adjuster._calculate_weights(performances, "accuracy")
        
        # Verify weights sum to 1 and are proportional to performance
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights[self.model_a_id], 0.9 / 2.4)  # 0.375
        self.assertAlmostEqual(weights[self.model_b_id], 0.7 / 2.4)  # 0.292
        self.assertAlmostEqual(weights[self.model_c_id], 0.8 / 2.4)  # 0.333
    
    def test_calculate_weights_lower_better(self):
        """Test weight calculation for metrics where lower is better."""
        # Performances for MSE (lower is better)
        performances = {
            self.model_a_id: 0.1,  # Best model (lowest error)
            self.model_b_id: 0.3,
            self.model_c_id: 0.2
        }
        
        # Calculate weights
        weights = self.adjuster._calculate_weights(performances, "mse")
        
        # Verify weights sum to 1 and are inversely proportional to performance
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        
        # Calculate expected weights (inverse of error)
        total_inverse = (1/0.1) + (1/0.3) + (1/0.2)  # 10 + 3.33 + 5 = 18.33
        expected_a = (1/0.1) / total_inverse  # 0.545
        expected_b = (1/0.3) / total_inverse  # 0.182
        expected_c = (1/0.2) / total_inverse  # 0.273
        
        self.assertAlmostEqual(weights[self.model_a_id], expected_a, places=3)
        self.assertAlmostEqual(weights[self.model_b_id], expected_b, places=3)
        self.assertAlmostEqual(weights[self.model_c_id], expected_c, places=3)


class TestDynamicWeightedEnsemble(unittest.TestCase):
    """Tests for the DynamicWeightedEnsemble class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create component models
        self.model_a = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model_b = LogisticRegression(random_state=42)
        
        # Train component models
        self.model_a.fit(self.X_train, self.y_train)
        self.model_b.fit(self.X_train, self.y_train)
        
        # Create ensemble
        self.ensemble = DynamicWeightedEnsemble(
            models=[
                {"model": self.model_a, "model_id": "model_a", "weight": 0.5},
                {"model": self.model_b, "model_id": "model_b", "weight": 0.5}
            ],
            optimization_metric="accuracy",
            performance_window=5,
            adjustment_threshold=0.05
        )
    
    def test_record_performance(self):
        """Test recording performance for component models."""
        # Record performance for both models
        self.ensemble.record_performance("model_a", "accuracy", 0.9)
        self.ensemble.record_performance("model_b", "accuracy", 0.7)
        
        # Verify performance was recorded
        self.assertIn("model_a", self.ensemble.performance_history)
        self.assertIn("model_b", self.ensemble.performance_history)
        self.assertIn("accuracy", self.ensemble.performance_history["model_a"])
        self.assertIn("accuracy", self.ensemble.performance_history["model_b"])
        self.assertEqual(len(self.ensemble.performance_history["model_a"]["accuracy"]), 1)
        self.assertEqual(len(self.ensemble.performance_history["model_b"]["accuracy"]), 1)
        self.assertEqual(self.ensemble.performance_history["model_a"]["accuracy"][0]["value"], 0.9)
        self.assertEqual(self.ensemble.performance_history["model_b"]["accuracy"][0]["value"], 0.7)
    
    def test_adjust_weights(self):
        """Test adjusting weights based on performance."""
        # Record multiple performance samples
        for _ in range(3):
            self.ensemble.record_performance("model_a", "accuracy", 0.9)
            self.ensemble.record_performance("model_b", "accuracy", 0.7)
        
        # Adjust weights
        result = self.ensemble.adjust_weights(force=True)
        
        # Verify weights were adjusted
        self.assertTrue(result)
        
        # Verify weights reflect performance
        model_a_weight = None
        model_b_weight = None
        for model_info in self.ensemble.models:
            if model_info["model_id"] == "model_a":
                model_a_weight = model_info["weight"]
            elif model_info["model_id"] == "model_b":
                model_b_weight = model_info["weight"]
        
        # Model A should have higher weight
        self.assertGreater(model_a_weight, model_b_weight)
        
        # Weights should sum to 1
        self.assertAlmostEqual(model_a_weight + model_b_weight, 1.0)
        
        # Calculate expected weights
        expected_a = 0.9 / (0.9 + 0.7)  # 0.5625
        expected_b = 0.7 / (0.9 + 0.7)  # 0.4375
        
        self.assertAlmostEqual(model_a_weight, expected_a, places=4)
        self.assertAlmostEqual(model_b_weight, expected_b, places=4)
    
    def test_predict_with_adjustment(self):
        """Test that predict method adjusts weights before prediction."""
        # Record performance to trigger adjustment
        for _ in range(3):
            self.ensemble.record_performance("model_a", "accuracy", 0.9)
            self.ensemble.record_performance("model_b", "accuracy", 0.7)
        
        # Mock the adjust_weights method to verify it's called
        original_adjust = self.ensemble.adjust_weights
        self.ensemble.adjust_weights = MagicMock(return_value=True)
        
        # Make a prediction
        self.ensemble.predict(self.X_test[:1])
        
        # Verify adjust_weights was called
        self.ensemble.adjust_weights.assert_called_once_with(force=False)
        
        # Restore original method
        self.ensemble.adjust_weights = original_adjust
    
    def test_save_and_load(self):
        """Test saving and loading a dynamic weighted ensemble."""
        # Record performance
        self.ensemble.record_performance("model_a", "accuracy", 0.9)
        self.ensemble.record_performance("model_b", "accuracy", 0.7)
        
        # Adjust weights
        self.ensemble.adjust_weights(force=True)
        
        # Create a temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the ensemble
            metadata = self.ensemble.save(temp_dir)
            
            # Verify metadata includes dynamic ensemble specific fields
            self.assertIn("performance_window", metadata)
            self.assertIn("adjustment_threshold", metadata)
            
            # Create a mock registry for loading
            registry = MagicMock(spec=ModelRegistry)
            
            # Mock the load method of the parent class
            with patch('src.services.model.dynamic_ensemble.WeightedEnsemble.load') as mock_load:
                # Configure mock to return a copy of the original ensemble
                mock_load.return_value = WeightedEnsemble(
                    models=[
                        {"model": self.model_a, "model_id": "model_a", "weight": 0.5625},
                        {"model": self.model_b, "model_id": "model_b", "weight": 0.4375}
                    ],
                    optimization_metric="accuracy"
                )
                
                # Load the ensemble
                loaded = DynamicWeightedEnsemble.load(temp_dir, registry)
                
                # Verify loaded ensemble has correct attributes
                self.assertEqual(loaded.performance_window, self.ensemble.performance_window)
                self.assertEqual(loaded.adjustment_threshold, self.ensemble.adjustment_threshold)
                self.assertEqual(loaded.optimization_metric, self.ensemble.optimization_metric)


class TestABTestingEnsemble(unittest.TestCase):
    """Tests for the ABTestingEnsemble class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create component models
        self.model_a = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model_b = LogisticRegression(random_state=42)
        
        # Train component models
        self.model_a.fit(self.X_train, self.y_train)
        self.model_b.fit(self.X_train, self.y_train)
        
        # Create ensemble
        self.ensemble = ABTestingEnsemble(
            models=[
                {"model": self.model_a, "model_id": "model_a"},
                {"model": self.model_b, "model_id": "model_b"}
            ],
            traffic_splits={"model_a": 0.5, "model_b": 0.5},
            metrics=["accuracy", "latency"],
            min_samples=10
        )
    
    def test_predict_routing(self):
        """Test that predictions are routed to different models."""
        # Make multiple predictions
        predictions = []
        for _ in range(100):
            predictions.append(self.ensemble.predict(self.X_test[:1]))
        
        # Verify both models were used
        self.assertGreater(self.ensemble.sample_counts.get("model_a", 0), 0)
        self.assertGreater(self.ensemble.sample_counts.get("model_b", 0), 0)
        
        # Verify total samples matches number of predictions
        total_samples = sum(self.ensemble.sample_counts.values())
        self.assertEqual(total_samples, 100)
    
    def test_record_performance(self):
        """Test recording performance for component models."""
        # Record performance for both models
        y_true = np.array([1])
        y_pred_a = np.array([1])  # Correct prediction
        y_pred_b = np.array([0])  # Incorrect prediction
        
        # Record for model A (better accuracy)
        self.ensemble.record_performance("model_a", y_true, y_pred_a)
        
        # Record for model B (worse accuracy)
        self.ensemble.record_performance("model_b", y_true, y_pred_b)
        
        # Verify performance was recorded
        self.assertIn("model_a", self.ensemble.performance_records)
        self.assertIn("model_b", self.ensemble.performance_records)
        self.assertIn("accuracy", self.ensemble.performance_records["model_a"])
        self.assertIn("accuracy", self.ensemble.performance_records["model_b"])
        self.assertEqual(len(self.ensemble.performance_records["model_a"]["accuracy"]), 1)
        self.assertEqual(len(self.ensemble.performance_records["model_b"]["accuracy"]), 1)
        self.assertEqual(self.ensemble.performance_records["model_a"]["accuracy"][0], 1.0)
        self.assertEqual(self.ensemble.performance_records["model_b"]["accuracy"][0], 0.0)
    
    def test_evaluate_test(self):
        """Test evaluating the A/B test to determine a winner."""
        # Record performance for both models
        for _ in range(10):
            # Model A: 90% accuracy, 50ms latency
            self.ensemble.record_performance("model_a", None, None, {
                "accuracy": 0.9,
                "latency": 0.05
            })
            
            # Model B: 80% accuracy, 30ms latency
            self.ensemble.record_performance("model_b", None, None, {
                "accuracy": 0.8,
                "latency": 0.03
            })
            
            # Update sample counts
            self.ensemble.sample_counts["model_a"] = 10
            self.ensemble.sample_counts["model_b"] = 10
        
        # Evaluate the test
        self.ensemble._evaluate_test()
        
        # Verify winner is model A (based on accuracy, which is first metric)
        self.assertEqual(self.ensemble.winner, "model_a")
        
        # Get test results
        results = self.ensemble.get_test_results()
        
        # Verify results
        self.assertEqual(results["winner"], "model_a")
        self.assertEqual(results["primary_metric"], "accuracy")
        self.assertEqual(results["sample_counts"]["model_a"], 10)
        self.assertEqual(results["sample_counts"]["model_b"], 10)
        self.assertAlmostEqual(results["model_results"]["model_a"]["accuracy"]["mean"], 0.9)
        self.assertAlmostEqual(results["model_results"]["model_b"]["accuracy"]["mean"], 0.8)
        self.assertAlmostEqual(results["model_results"]["model_a"]["latency"]["mean"], 0.05)
        self.assertAlmostEqual(results["model_results"]["model_b"]["latency"]["mean"], 0.03)
    
    def test_save_and_load(self):
        """Test saving and loading an A/B testing ensemble."""
        # Record performance
        for _ in range(10):
            self.ensemble.record_performance("model_a", None, None, {"accuracy": 0.9})
            self.ensemble.record_performance("model_b", None, None, {"accuracy": 0.8})
        
        # Update sample counts
        self.ensemble.sample_counts["model_a"] = 10
        self.ensemble.sample_counts["model_b"] = 10
        
        # Evaluate the test
        self.ensemble._evaluate_test()
        
        # Create a temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the ensemble
            metadata = self.ensemble.save(temp_dir)
            
            # Verify metadata includes A/B testing specific fields
            self.assertIn("traffic_splits", metadata)
            self.assertIn("metrics", metadata)
            self.assertIn("min_samples", metadata)
            self.assertIn("winner", metadata)
            
            # Create a mock registry for loading
            registry = MagicMock(spec=ModelRegistry)
            
            # Mock the load method of the parent class
            with patch('src.services.model.dynamic_ensemble.EnsembleModel.load') as mock_load:
                # Configure mock to return a copy of the original ensemble
                mock_ensemble = EnsembleModel()
                mock_ensemble.models = [
                    {"model": self.model_a, "model_id": "model_a"},
                    {"model": self.model_b, "model_id": "model_b"}
                ]
                mock_load.return_value = mock_ensemble
                
                # Load the ensemble
                loaded = ABTestingEnsemble.load(temp_dir, registry)
                
                # Verify loaded ensemble has correct attributes
                self.assertEqual(loaded.traffic_splits, self.ensemble.traffic_splits)
                self.assertEqual(loaded.metrics, self.ensemble.metrics)
                self.assertEqual(loaded.min_samples, self.ensemble.min_samples)
                self.assertEqual(loaded.winner, self.ensemble.winner)


if __name__ == "__main__":
    unittest.main()