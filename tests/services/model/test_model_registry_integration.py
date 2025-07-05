"""Integration tests for the model registry system.

This module contains integration tests that demonstrate how all the components
of the model registry system work together, including deep learning models,
classification models, pattern recognition, and ensemble methods.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
import json
import logging

# Configure logging to suppress warnings during tests
logging.basicConfig(level=logging.ERROR)

# Import the model registry components
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_registry_config import ModelType, PredictionTarget, SerializationFormat
from src.services.model.model_registry_extensions import (
    ExtendedModelRegistry,
    DeepLearningModelRegistry,
    ClassificationModelRegistry,
    PatternRecognitionModelRegistry,
    EnsembleModelRegistry
)
from src.services.model.classification_models import (
    ClassificationType,
    ClassificationMetrics,
    ClassificationModelFactory
)
from src.services.model.pattern_recognition import (
    PatternType,
    PatternRecognitionSystem,
    CandlestickPatternDetector
)
from src.services.model.ensemble_methods import (
    EnsembleMethod,
    EnsembleModelTrainer,
    VotingEnsemble
)


# Create a simple sklearn-like model for testing
class SimpleModel:
    """A simple model that implements the sklearn interface."""
    
    def __init__(self, name="simple_model", prediction_value=0):
        self.name = name
        self.prediction_value = prediction_value
    
    def fit(self, X, y):
        """Fit the model to the data."""
        return self
    
    def predict(self, X):
        """Predict using the model."""
        return np.full(len(X), self.prediction_value)
    
    def predict_proba(self, X):
        """Predict probabilities using the model."""
        probs = np.zeros((len(X), 2))
        probs[:, self.prediction_value] = 0.8
        probs[:, 1 - self.prediction_value] = 0.2
        return probs


# Create a simple PyTorch-like model for testing
class SimplePyTorchModel:
    """A simple model that mimics a PyTorch model."""
    
    def __init__(self, name="pytorch_model"):
        self.name = name
        self.__class__.__module__ = "torch.nn.modules"
    
    def __str__(self):
        return f"SimplePyTorchModel({self.name})"
    
    def parameters(self):
        """Return model parameters."""
        return []
    
    def state_dict(self):
        """Return model state dict."""
        return {"weights": np.zeros(10)}
    
    def eval(self):
        """Set model to evaluation mode."""
        return self


# Create a simple TensorFlow-like model for testing
class SimpleTensorFlowModel:
    """A simple model that mimics a TensorFlow model."""
    
    def __init__(self, name="tensorflow_model"):
        self.name = name
        self.__class__.__module__ = "tensorflow.keras.models"
    
    def __str__(self):
        return f"SimpleTensorFlowModel({self.name})"
    
    def summary(self):
        """Print model summary."""
        print(f"Model: {self.name}")
        print("Layers: None")
    
    def get_weights(self):
        """Return model weights."""
        return [np.zeros(10)]
    
    def predict(self, X):
        """Predict using the model."""
        return np.zeros(len(X))


class TestModelRegistryIntegration(unittest.TestCase):
    """Integration tests for the model registry system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a ModelRegistry with the temporary directory
        self.registry = ModelRegistry(storage_dir=self.temp_dir)
        
        # Create an ExtendedModelRegistry
        self.extended_registry = ExtendedModelRegistry(self.registry)
        
        # Create test data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch("torch.nn.Module")
    @patch("torch.__version__", "1.9.0")
    def test_deep_learning_workflow(self, mock_module):
        """Test the deep learning workflow."""
        # Skip if torch is not available
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not available")
        
        # Create a mock PyTorch model
        model = SimplePyTorchModel()
        
        # Register the model
        with patch.object(model, "__class__", spec=mock_module):
            model_id = self.extended_registry.deep_learning.register_pytorch_model(
                model=model,
                model_type=ModelType.LSTM,
                prediction_target=PredictionTarget.PRICE,
                metadata={"test": True},
                tags=["test"]
            )
        
        # Check that the model was registered
        self.assertIsNotNone(model_id)
        
        # Get the model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Check that the metadata contains the right information
        self.assertEqual(metadata["model_type"], ModelType.LSTM.name)
        self.assertEqual(metadata["prediction_target"], PredictionTarget.PRICE.name)
        self.assertEqual(metadata["test"], True)
        self.assertEqual(metadata["framework"], "pytorch")
        
        # Check that the tags are correct
        tags = self.registry.get_model_tags(model_id)
        self.assertIn("test", tags)
        self.assertIn("pytorch", tags)
        self.assertIn("deep_learning", tags)
    
    @patch("tensorflow.keras.Model")
    @patch("tensorflow.__version__", "2.5.0")
    def test_tensorflow_workflow(self, mock_model):
        """Test the TensorFlow workflow."""
        # Skip if tensorflow is not available
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not available")
        
        # Create a mock TensorFlow model
        model = SimpleTensorFlowModel()
        
        # Register the model
        with patch.object(model, "__class__", spec=mock_model):
            model_id = self.extended_registry.deep_learning.register_tensorflow_model(
                model=model,
                model_type=ModelType.LSTM,
                prediction_target=PredictionTarget.PRICE,
                metadata={"test": True},
                tags=["test"]
            )
        
        # Check that the model was registered
        self.assertIsNotNone(model_id)
        
        # Get the model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Check that the metadata contains the right information
        self.assertEqual(metadata["model_type"], ModelType.LSTM.name)
        self.assertEqual(metadata["prediction_target"], PredictionTarget.PRICE.name)
        self.assertEqual(metadata["test"], True)
        self.assertEqual(metadata["framework"], "tensorflow")
        
        # Check that the tags are correct
        tags = self.registry.get_model_tags(model_id)
        self.assertIn("test", tags)
        self.assertIn("tensorflow", tags)
        self.assertIn("deep_learning", tags)
    
    def test_classification_workflow(self):
        """Test the classification workflow."""
        # Create a simple classification model
        model = SimpleModel(prediction_value=1)
        
        # Register the model
        model_id = self.extended_registry.classification.register_classification_model(
            model=model,
            classification_type=ClassificationType.BINARY,
            metadata={"test": True},
            tags=["test"],
            evaluation_data=(self.X_test, self.y_test)
        )
        
        # Check that the model was registered
        self.assertIsNotNone(model_id)
        
        # Get the model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Check that the metadata contains the right information
        self.assertEqual(metadata["model_type"], ModelType.CLASSIFICATION.name)
        self.assertEqual(metadata["prediction_target"], PredictionTarget.BINARY_CLASSIFICATION.name)
        self.assertEqual(metadata["test"], True)
        self.assertEqual(metadata["classification_type"], ClassificationType.BINARY.name)
        
        # Check that the evaluation metrics are present
        self.assertIn("evaluation_metrics", metadata)
        metrics = metadata["evaluation_metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        
        # Check that the tags are correct
        tags = self.registry.get_model_tags(model_id)
        self.assertIn("test", tags)
        self.assertIn("classification", tags)
        self.assertIn("binary", tags)
        
        # Load the model
        loaded_model = self.extended_registry.classification.load_classification_model(model_id)
        
        # Check that the model was loaded correctly
        self.assertEqual(loaded_model.name, model.name)
        
        # Get classification models
        models = self.extended_registry.classification.get_classification_models(
            classification_type=ClassificationType.BINARY
        )
        
        # Check that the model is in the list
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], model_id)
    
    def test_pattern_recognition_workflow(self):
        """Test the pattern recognition workflow."""
        # Create a pattern detector
        detector = CandlestickPatternDetector()
        
        # Register the detector
        detector_id = self.extended_registry.pattern_recognition.register_pattern_detector(
            detector=detector,
            detector_type="candlestick",
            metadata={"test": True},
            tags=["test"]
        )
        
        # Check that the detector was registered
        self.assertIsNotNone(detector_id)
        
        # Get the detector metadata
        metadata = self.registry.get_model_metadata(detector_id)
        
        # Check that the metadata contains the right information
        self.assertEqual(metadata["model_type"], ModelType.PATTERN_RECOGNITION.name)
        self.assertEqual(metadata["prediction_target"], PredictionTarget.PATTERN.name)
        self.assertEqual(metadata["test"], True)
        self.assertEqual(metadata["detector_type"], "candlestick")
        
        # Check that the tags are correct
        tags = self.registry.get_model_tags(detector_id)
        self.assertIn("test", tags)
        self.assertIn("pattern_detector", tags)
        self.assertIn("candlestick", tags)
        
        # Load the detector
        loaded_detector = self.extended_registry.pattern_recognition.load_pattern_detector(detector_id)
        
        # Check that the detector was loaded correctly
        self.assertIsInstance(loaded_detector, CandlestickPatternDetector)
        
        # Get pattern detectors
        detectors = self.extended_registry.pattern_recognition.get_pattern_detectors(
            detector_type="candlestick"
        )
        
        # Check that the detector is in the list
        self.assertEqual(len(detectors), 1)
        self.assertEqual(detectors[0]["id"], detector_id)
    
    def test_ensemble_workflow(self):
        """Test the ensemble workflow."""
        # Create and register base models
        model1 = SimpleModel(name="model1", prediction_value=0)
        model2 = SimpleModel(name="model2", prediction_value=1)
        
        model1_id = self.registry.register_model(
            model=model1,
            model_type=ModelType.REGRESSION,
            prediction_target=PredictionTarget.PRICE,
            metadata={"name": "model1"},
            tags=["base_model"]
        )
        
        model2_id = self.registry.register_model(
            model=model2,
            model_type=ModelType.REGRESSION,
            prediction_target=PredictionTarget.PRICE,
            metadata={"name": "model2"},
            tags=["base_model"]
        )
        
        # Create an ensemble from the base models
        ensemble_id = self.extended_registry.ensemble.create_ensemble_from_models(
            name="test_ensemble",
            ensemble_method=EnsembleMethod.VOTING,
            model_ids=[model1_id, model2_id],
            weights=[0.7, 0.3]
        )
        
        # Check that the ensemble was registered
        self.assertIsNotNone(ensemble_id)
        
        # Get the ensemble metadata
        metadata = self.registry.get_model_metadata(ensemble_id)
        
        # Check that the metadata contains the right information
        self.assertEqual(metadata["ensemble_method"], EnsembleMethod.VOTING.name)
        self.assertEqual(metadata["model_count"], 2)
        self.assertEqual(metadata["component_model_ids"], [model1_id, model2_id])
        
        # Check that the tags are correct
        tags = self.registry.get_model_tags(ensemble_id)
        self.assertIn("ensemble", tags)
        self.assertIn("voting", tags)
        
        # Load the ensemble
        loaded_ensemble = self.extended_registry.ensemble.load_ensemble(ensemble_id)
        
        # Check that the ensemble was loaded correctly
        self.assertIsInstance(loaded_ensemble, VotingEnsemble)
        
        # Get ensembles
        ensembles = self.extended_registry.ensemble.get_ensembles(
            ensemble_method=EnsembleMethod.VOTING
        )
        
        # Check that the ensemble is in the list
        self.assertEqual(len(ensembles), 1)
        self.assertEqual(ensembles[0]["id"], ensemble_id)
    
    def test_end_to_end_workflow(self):
        """Test an end-to-end workflow with all components."""
        # 1. Create and register classification models
        model1 = SimpleModel(name="classifier1", prediction_value=0)
        model2 = SimpleModel(name="classifier2", prediction_value=1)
        
        model1_id = self.extended_registry.classification.register_classification_model(
            model=model1,
            classification_type=ClassificationType.BINARY,
            metadata={"name": "classifier1"},
            tags=["base_model"],
            evaluation_data=(self.X_test, self.y_test)
        )
        
        model2_id = self.extended_registry.classification.register_classification_model(
            model=model2,
            classification_type=ClassificationType.BINARY,
            metadata={"name": "classifier2"},
            tags=["base_model"],
            evaluation_data=(self.X_test, self.y_test)
        )
        
        # 2. Create a pattern detector
        detector = CandlestickPatternDetector()
        
        detector_id = self.extended_registry.pattern_recognition.register_pattern_detector(
            detector=detector,
            detector_type="candlestick",
            metadata={"name": "candlestick_detector"},
            tags=["pattern"]
        )
        
        # 3. Create an ensemble from the classification models
        ensemble_id = self.extended_registry.ensemble.create_ensemble_from_models(
            name="classification_ensemble",
            ensemble_method=EnsembleMethod.VOTING,
            model_ids=[model1_id, model2_id]
        )
        
        # 4. Query models by type
        classification_models = self.extended_registry.classification.get_classification_models(
            classification_type=ClassificationType.BINARY
        )
        
        pattern_detectors = self.extended_registry.pattern_recognition.get_pattern_detectors(
            detector_type="candlestick"
        )
        
        ensembles = self.extended_registry.ensemble.get_ensembles(
            ensemble_method=EnsembleMethod.VOTING
        )
        
        # 5. Check that all models are found
        self.assertEqual(len(classification_models), 2)
        self.assertEqual(len(pattern_detectors), 1)
        self.assertEqual(len(ensembles), 1)
        
        # 6. Load and use the ensemble
        ensemble = self.extended_registry.ensemble.load_ensemble(ensemble_id)
        
        # Make predictions with the ensemble
        predictions = ensemble.predict(self.X_test)
        
        # Check that predictions are returned
        self.assertEqual(len(predictions), len(self.X_test))


if __name__ == "__main__":
    unittest.main()