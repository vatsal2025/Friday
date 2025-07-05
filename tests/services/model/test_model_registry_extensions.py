"""Tests for the model registry extensions.

This module contains tests for the deep learning, classification, pattern recognition,
and ensemble model registry extensions.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
import json

from src.services.model.model_registry import ModelRegistry
from src.services.model.model_registry_config import ModelType, PredictionTarget, SerializationFormat
from src.services.model.model_registry_extensions import (
    DeepLearningModelRegistry,
    ClassificationModelRegistry,
    PatternRecognitionModelRegistry,
    EnsembleModelRegistry,
    ExtendedModelRegistry
)
from src.services.model.classification_models import ClassificationType
from src.services.model.ensemble_methods import EnsembleMethod


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name="mock_model"):
        self.name = name
        
    def predict(self, X):
        """Mock predict method."""
        return np.zeros(len(X))
    
    def predict_proba(self, X):
        """Mock predict_proba method."""
        return np.zeros((len(X), 2))


class TestDeepLearningModelRegistry(unittest.TestCase):
    """Tests for the DeepLearningModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.register_model.return_value = "mock_model_id"
        self.mock_registry.get_model_metadata.return_value = {
            "framework": "pytorch",
            "model_type": ModelType.LSTM.name,
            "prediction_target": PredictionTarget.PRICE.name
        }
        
        # Create the DeepLearningModelRegistry
        self.dl_registry = DeepLearningModelRegistry(self.mock_registry)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch("torch.nn.Module")
    @patch("torch.__version__", "1.9.0")
    def test_register_pytorch_model(self, mock_module):
        """Test registering a PyTorch model."""
        # Create a mock PyTorch model
        mock_model = MagicMock(spec=mock_module)
        
        # Register the model
        model_id = self.dl_registry.register_pytorch_model(
            model=mock_model,
            model_type=ModelType.LSTM,
            prediction_target=PredictionTarget.PRICE
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(call_args["model"], mock_model)
        self.assertEqual(call_args["model_type"], ModelType.LSTM)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.PRICE)
        self.assertEqual(call_args["serialization_format"], SerializationFormat.CUSTOM)
        
        # Check that the metadata contains the right information
        metadata = call_args["metadata"]
        self.assertEqual(metadata["framework"], "pytorch")
        self.assertEqual(metadata["save_format"], "pt")
        self.assertEqual(metadata["pytorch_version"], "1.9.0")
        
        # Check that the tags are correct
        tags = call_args["tags"]
        self.assertIn("pytorch", tags)
        self.assertIn("deep_learning", tags)
        
        # Check that the model ID is returned
        self.assertEqual(model_id, "mock_model_id")
    
    @patch("tensorflow.keras.Model")
    @patch("tensorflow.__version__", "2.5.0")
    def test_register_tensorflow_model(self, mock_model):
        """Test registering a TensorFlow model."""
        # Create a mock TensorFlow model
        mock_tf_model = MagicMock(spec=mock_model)
        
        # Register the model
        model_id = self.dl_registry.register_tensorflow_model(
            model=mock_tf_model,
            model_type=ModelType.LSTM,
            prediction_target=PredictionTarget.PRICE
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(call_args["model"], mock_tf_model)
        self.assertEqual(call_args["model_type"], ModelType.LSTM)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.PRICE)
        self.assertEqual(call_args["serialization_format"], SerializationFormat.CUSTOM)
        
        # Check that the metadata contains the right information
        metadata = call_args["metadata"]
        self.assertEqual(metadata["framework"], "tensorflow")
        self.assertEqual(metadata["save_format"], "tf")
        self.assertEqual(metadata["tensorflow_version"], "2.5.0")
        
        # Check that the tags are correct
        tags = call_args["tags"]
        self.assertIn("tensorflow", tags)
        self.assertIn("deep_learning", tags)
        
        # Check that the model ID is returned
        self.assertEqual(model_id, "mock_model_id")
    
    def test_load_deep_learning_model(self):
        """Test loading a deep learning model."""
        # Mock the load_model method
        mock_model = MockModel()
        self.mock_registry.load_model.return_value = mock_model
        
        # Load the model
        model = self.dl_registry.load_deep_learning_model("mock_model_id")
        
        # Check that load_model was called with the right arguments
        self.mock_registry.load_model.assert_called_once_with("mock_model_id")
        
        # Check that the model is returned
        self.assertEqual(model, mock_model)
    
    def test_load_deep_learning_model_not_found(self):
        """Test loading a deep learning model that doesn't exist."""
        # Mock the get_model_metadata method to return None
        self.mock_registry.get_model_metadata.return_value = None
        
        # Check that loading a non-existent model raises an error
        with self.assertRaises(ValueError):
            self.dl_registry.load_deep_learning_model("non_existent_model_id")
    
    def test_load_deep_learning_model_wrong_type(self):
        """Test loading a model that is not a deep learning model."""
        # Mock the get_model_metadata method to return a non-deep learning model
        self.mock_registry.get_model_metadata.return_value = {
            "framework": "scikit-learn"
        }
        
        # Check that loading a non-deep learning model raises an error
        with self.assertRaises(ValueError):
            self.dl_registry.load_deep_learning_model("wrong_type_model_id")


class TestClassificationModelRegistry(unittest.TestCase):
    """Tests for the ClassificationModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.register_model.return_value = "mock_model_id"
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": ModelType.CLASSIFICATION.name,
            "prediction_target": PredictionTarget.BINARY_CLASSIFICATION.name
        }
        
        # Create the ClassificationModelRegistry
        self.clf_registry = ClassificationModelRegistry(self.mock_registry)
    
    def test_register_classification_model(self):
        """Test registering a classification model."""
        # Create a mock classification model
        mock_model = MockModel()
        
        # Register the model
        model_id = self.clf_registry.register_classification_model(
            model=mock_model,
            classification_type=ClassificationType.BINARY
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(call_args["model"], mock_model)
        self.assertEqual(call_args["model_type"], ModelType.CLASSIFICATION)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.BINARY_CLASSIFICATION)
        
        # Check that the metadata contains the right information
        metadata = call_args["metadata"]
        self.assertEqual(metadata["classification_type"], ClassificationType.BINARY.name)
        
        # Check that the tags are correct
        tags = call_args["tags"]
        self.assertIn("classification", tags)
        self.assertIn("binary", tags)
        
        # Check that the model ID is returned
        self.assertEqual(model_id, "mock_model_id")
    
    def test_register_classification_model_with_evaluation(self):
        """Test registering a classification model with evaluation data."""
        # Create a mock classification model
        mock_model = MockModel()
        
        # Create mock evaluation data
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        
        # Register the model with evaluation data
        model_id = self.clf_registry.register_classification_model(
            model=mock_model,
            classification_type=ClassificationType.BINARY,
            evaluation_data=(X_test, y_test)
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        
        # Check that the metadata contains evaluation metrics
        metadata = call_args["metadata"]
        self.assertIn("evaluation_metrics", metadata)
        metrics = metadata["evaluation_metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
    
    def test_detect_model_library(self):
        """Test detecting the library of a model."""
        # Create mock models with different modules
        sklearn_model = MagicMock()
        sklearn_model.__class__.__module__ = "sklearn.linear_model"
        
        xgboost_model = MagicMock()
        xgboost_model.__class__.__module__ = "xgboost.sklearn"
        
        pytorch_model = MagicMock()
        pytorch_model.__class__.__module__ = "torch.nn.modules"
        
        # Detect the libraries
        sklearn_lib = self.clf_registry._detect_model_library(sklearn_model)
        xgboost_lib = self.clf_registry._detect_model_library(xgboost_model)
        pytorch_lib = self.clf_registry._detect_model_library(pytorch_model)
        
        # Check that the libraries are detected correctly
        self.assertEqual(sklearn_lib, "scikit-learn")
        self.assertEqual(xgboost_lib, "xgboost")
        self.assertEqual(pytorch_lib, "pytorch")
    
    def test_load_classification_model(self):
        """Test loading a classification model."""
        # Mock the load_model method
        mock_model = MockModel()
        self.mock_registry.load_model.return_value = mock_model
        
        # Load the model
        model = self.clf_registry.load_classification_model("mock_model_id")
        
        # Check that load_model was called with the right arguments
        self.mock_registry.load_model.assert_called_once_with("mock_model_id")
        
        # Check that the model is returned
        self.assertEqual(model, mock_model)
    
    def test_load_classification_model_not_found(self):
        """Test loading a classification model that doesn't exist."""
        # Mock the get_model_metadata method to return None
        self.mock_registry.get_model_metadata.return_value = None
        
        # Check that loading a non-existent model raises an error
        with self.assertRaises(ValueError):
            self.clf_registry.load_classification_model("non_existent_model_id")
    
    def test_load_classification_model_wrong_type(self):
        """Test loading a model that is not a classification model."""
        # Mock the get_model_metadata method to return a non-classification model
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": ModelType.REGRESSION.name
        }
        
        # Check that loading a non-classification model raises an error
        with self.assertRaises(ValueError):
            self.clf_registry.load_classification_model("wrong_type_model_id")
    
    def test_get_classification_models(self):
        """Test getting classification models."""
        # Mock the get_models method
        mock_models = [
            {"id": "model1", "metadata": {"evaluation_metrics": {"accuracy": 0.9}}},
            {"id": "model2", "metadata": {"evaluation_metrics": {"accuracy": 0.8}}}
        ]
        self.mock_registry.get_models.return_value = mock_models
        
        # Get models with no filters
        models = self.clf_registry.get_classification_models()
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[0]["model_type"], ModelType.CLASSIFICATION.name)
        
        # Check that the models are returned
        self.assertEqual(models, mock_models)
        
        # Reset the mock
        self.mock_registry.get_models.reset_mock()
        
        # Get models with filters
        models = self.clf_registry.get_classification_models(
            classification_type=ClassificationType.BINARY,
            min_accuracy=0.85,
            tags=["test"]
        )
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[0]["model_type"], ModelType.CLASSIFICATION.name)
        self.assertEqual(call_args[0]["prediction_target"], PredictionTarget.BINARY_CLASSIFICATION.name)
        self.assertEqual(call_args[1], ["test"])
        
        # Check that the models are filtered by accuracy
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "model1")


class TestPatternRecognitionModelRegistry(unittest.TestCase):
    """Tests for the PatternRecognitionModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.register_model.return_value = "mock_detector_id"
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": ModelType.PATTERN_RECOGNITION.name,
            "prediction_target": PredictionTarget.PATTERN.name
        }
        
        # Create the PatternRecognitionModelRegistry
        self.pattern_registry = PatternRecognitionModelRegistry(self.mock_registry)
    
    def test_register_pattern_detector(self):
        """Test registering a pattern detector."""
        # Create a mock pattern detector
        mock_detector = MagicMock()
        mock_detector.supported_patterns = ["doji", "hammer"]
        
        # Register the detector
        detector_id = self.pattern_registry.register_pattern_detector(
            detector=mock_detector,
            detector_type="candlestick"
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(call_args["model"], mock_detector)
        self.assertEqual(call_args["model_type"], ModelType.PATTERN_RECOGNITION)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.PATTERN)
        
        # Check that the metadata contains the right information
        metadata = call_args["metadata"]
        self.assertEqual(metadata["detector_type"], "candlestick")
        self.assertEqual(metadata["patterns_supported"], ["doji", "hammer"])
        
        # Check that the tags are correct
        tags = call_args["tags"]
        self.assertIn("pattern_detector", tags)
        self.assertIn("candlestick", tags)
        
        # Check that the detector ID is returned
        self.assertEqual(detector_id, "mock_detector_id")
    
    def test_load_pattern_detector(self):
        """Test loading a pattern detector."""
        # Mock the load_model method
        mock_detector = MagicMock()
        self.mock_registry.load_model.return_value = mock_detector
        
        # Load the detector
        detector = self.pattern_registry.load_pattern_detector("mock_detector_id")
        
        # Check that load_model was called with the right arguments
        self.mock_registry.load_model.assert_called_once_with("mock_detector_id")
        
        # Check that the detector is returned
        self.assertEqual(detector, mock_detector)
    
    def test_load_pattern_detector_not_found(self):
        """Test loading a pattern detector that doesn't exist."""
        # Mock the get_model_metadata method to return None
        self.mock_registry.get_model_metadata.return_value = None
        
        # Check that loading a non-existent detector raises an error
        with self.assertRaises(ValueError):
            self.pattern_registry.load_pattern_detector("non_existent_detector_id")
    
    def test_load_pattern_detector_wrong_type(self):
        """Test loading a model that is not a pattern detector."""
        # Mock the get_model_metadata method to return a non-pattern detector
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": ModelType.REGRESSION.name
        }
        
        # Check that loading a non-pattern detector raises an error
        with self.assertRaises(ValueError):
            self.pattern_registry.load_pattern_detector("wrong_type_detector_id")
    
    def test_get_pattern_detectors(self):
        """Test getting pattern detectors."""
        # Mock the get_models method
        mock_detectors = [
            {"id": "detector1", "metadata": {"detector_type": "candlestick", "patterns_supported": ["doji", "hammer"]}},
            {"id": "detector2", "metadata": {"detector_type": "chart", "patterns_supported": ["head_and_shoulders"]}}
        ]
        self.mock_registry.get_models.return_value = mock_detectors
        
        # Get detectors with no filters
        detectors = self.pattern_registry.get_pattern_detectors()
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[0]["model_type"], ModelType.PATTERN_RECOGNITION.name)
        self.assertEqual(call_args[0]["prediction_target"], PredictionTarget.PATTERN.name)
        
        # Check that the detectors are returned
        self.assertEqual(detectors, mock_detectors)
        
        # Reset the mock
        self.mock_registry.get_models.reset_mock()
        
        # Get detectors with filters
        detectors = self.pattern_registry.get_pattern_detectors(
            detector_type="candlestick",
            pattern="doji",
            tags=["test"]
        )
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[0]["model_type"], ModelType.PATTERN_RECOGNITION.name)
        self.assertEqual(call_args[0]["prediction_target"], PredictionTarget.PATTERN.name)
        self.assertEqual(call_args[0]["metadata.detector_type"], "candlestick")
        self.assertEqual(call_args[1], ["test"])
        
        # Check that the detectors are filtered by pattern
        self.assertEqual(len(detectors), 1)
        self.assertEqual(detectors[0]["id"], "detector1")


class TestEnsembleModelRegistry(unittest.TestCase):
    """Tests for the EnsembleModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.register_model.return_value = "mock_ensemble_id"
        self.mock_registry.get_model_metadata.return_value = {
            "ensemble_method": EnsembleMethod.VOTING.name,
            "model_type": ModelType.ENSEMBLE.name,
            "prediction_target": PredictionTarget.PRICE.name
        }
        
        # Create the EnsembleModelRegistry
        self.ensemble_registry = EnsembleModelRegistry(self.mock_registry)
    
    def test_register_ensemble(self):
        """Test registering an ensemble model."""
        # Create a mock ensemble model
        mock_ensemble = MagicMock()
        mock_ensemble.ensemble_method = EnsembleMethod.VOTING
        mock_ensemble.weighting_strategy = MagicMock(name="EQUAL")
        mock_ensemble.models = [{"model": MagicMock(), "weight": 1.0, "model_id": "model1"}]
        mock_ensemble.metadata = {"name": "test_ensemble"}
        mock_ensemble.model_type = ModelType.ENSEMBLE
        mock_ensemble.prediction_target = PredictionTarget.PRICE
        
        # Register the ensemble
        ensemble_id = self.ensemble_registry.register_ensemble(
            ensemble=mock_ensemble
        )
        
        # Check that register_model was called with the right arguments
        self.mock_registry.register_model.assert_called_once()
        call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(call_args["model"], mock_ensemble)
        self.assertEqual(call_args["model_type"], ModelType.ENSEMBLE)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.PRICE)
        
        # Check that the metadata contains the right information
        metadata = call_args["metadata"]
        self.assertEqual(metadata["name"], "test_ensemble")
        self.assertEqual(metadata["ensemble_method"], EnsembleMethod.VOTING.name)
        self.assertEqual(metadata["model_count"], 1)
        self.assertEqual(metadata["component_model_ids"], ["model1"])
        
        # Check that the tags are correct
        tags = call_args["tags"]
        self.assertIn("ensemble", tags)
        self.assertIn("voting", tags)
        
        # Check that the ensemble ID is returned
        self.assertEqual(ensemble_id, "mock_ensemble_id")
    
    def test_load_ensemble(self):
        """Test loading an ensemble model."""
        # Mock the load_model method
        mock_ensemble = MagicMock()
        self.mock_registry.load_model.return_value = mock_ensemble
        
        # Load the ensemble
        ensemble = self.ensemble_registry.load_ensemble("mock_ensemble_id")
        
        # Check that load_model was called with the right arguments
        self.mock_registry.load_model.assert_called_once_with("mock_ensemble_id")
        
        # Check that the ensemble is returned
        self.assertEqual(ensemble, mock_ensemble)
    
    def test_load_ensemble_not_found(self):
        """Test loading an ensemble that doesn't exist."""
        # Mock the get_model_metadata method to return None
        self.mock_registry.get_model_metadata.return_value = None
        
        # Check that loading a non-existent ensemble raises an error
        with self.assertRaises(ValueError):
            self.ensemble_registry.load_ensemble("non_existent_ensemble_id")
    
    def test_load_ensemble_wrong_type(self):
        """Test loading a model that is not an ensemble."""
        # Mock the get_model_metadata method to return a non-ensemble model
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": ModelType.REGRESSION.name
        }
        
        # Check that loading a non-ensemble model raises an error
        with self.assertRaises(ValueError):
            self.ensemble_registry.load_ensemble("wrong_type_ensemble_id")
    
    @patch("src.services.model.ensemble_methods.EnsembleModelTrainer")
    def test_create_ensemble_from_models(self, mock_trainer_class):
        """Test creating an ensemble from existing models."""
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock the create_voting_ensemble method
        mock_ensemble = MagicMock()
        mock_trainer.create_voting_ensemble.return_value = mock_ensemble
        
        # Mock the load_model method
        mock_models = [MagicMock(), MagicMock()]
        self.mock_registry.load_model.side_effect = mock_models
        
        # Mock the get_model_metadata method
        self.mock_registry.get_model_metadata.side_effect = [
            {"model_type": ModelType.REGRESSION.name, "prediction_target": PredictionTarget.PRICE.name},
            {"model_type": ModelType.REGRESSION.name, "prediction_target": PredictionTarget.PRICE.name}
        ]
        
        # Create the ensemble
        ensemble_id = self.ensemble_registry.create_ensemble_from_models(
            name="test_ensemble",
            ensemble_method=EnsembleMethod.VOTING,
            model_ids=["model1", "model2"]
        )
        
        # Check that load_model was called for each model ID
        self.assertEqual(self.mock_registry.load_model.call_count, 2)
        
        # Check that create_voting_ensemble was called with the right arguments
        mock_trainer.create_voting_ensemble.assert_called_once()
        call_args = mock_trainer.create_voting_ensemble.call_args[1]
        self.assertEqual(call_args["name"], "test_ensemble")
        self.assertEqual(call_args["model_type"], ModelType.REGRESSION)
        self.assertEqual(call_args["prediction_target"], PredictionTarget.PRICE)
        self.assertEqual(call_args["models"], mock_models)
        self.assertEqual(call_args["model_ids"], ["model1", "model2"])
        
        # Check that register_ensemble was called with the right arguments
        # This is indirectly tested through the mock_registry.register_model call
        
        # Check that the ensemble ID is returned
        self.assertEqual(ensemble_id, "mock_ensemble_id")
    
    def test_get_ensembles(self):
        """Test getting ensemble models."""
        # Mock the get_models method
        mock_ensembles = [
            {"id": "ensemble1", "metadata": {"ensemble_method": EnsembleMethod.VOTING.name}},
            {"id": "ensemble2", "metadata": {"ensemble_method": EnsembleMethod.STACKING.name}}
        ]
        self.mock_registry.get_models.return_value = mock_ensembles
        
        # Get ensembles with no filters
        ensembles = self.ensemble_registry.get_ensembles()
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[1], ["ensemble"])
        
        # Check that the ensembles are returned
        self.assertEqual(ensembles, mock_ensembles)
        
        # Reset the mock
        self.mock_registry.get_models.reset_mock()
        
        # Get ensembles with filters
        ensembles = self.ensemble_registry.get_ensembles(
            ensemble_method=EnsembleMethod.VOTING,
            model_type=ModelType.REGRESSION,
            prediction_target=PredictionTarget.PRICE,
            tags=["test"]
        )
        
        # Check that get_models was called with the right arguments
        self.mock_registry.get_models.assert_called_once()
        call_args = self.mock_registry.get_models.call_args[0]
        self.assertEqual(call_args[0]["metadata.ensemble_method"], EnsembleMethod.VOTING.name)
        self.assertEqual(call_args[0]["model_type"], ModelType.REGRESSION.name)
        self.assertEqual(call_args[0]["prediction_target"], PredictionTarget.PRICE.name)
        self.assertEqual(call_args[1], ["test"])


class TestExtendedModelRegistry(unittest.TestCase):
    """Tests for the ExtendedModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        
        # Create the ExtendedModelRegistry
        self.extended_registry = ExtendedModelRegistry(self.mock_registry)
    
    def test_initialization(self):
        """Test initialization of the ExtendedModelRegistry."""
        # Check that all extensions are initialized
        self.assertIsInstance(self.extended_registry.deep_learning, DeepLearningModelRegistry)
        self.assertIsInstance(self.extended_registry.classification, ClassificationModelRegistry)
        self.assertIsInstance(self.extended_registry.pattern_recognition, PatternRecognitionModelRegistry)
        self.assertIsInstance(self.extended_registry.ensemble, EnsembleModelRegistry)
        
        # Check that all extensions use the same base registry
        self.assertEqual(self.extended_registry.deep_learning.registry, self.mock_registry)
        self.assertEqual(self.extended_registry.classification.registry, self.mock_registry)
        self.assertEqual(self.extended_registry.pattern_recognition.registry, self.mock_registry)
        self.assertEqual(self.extended_registry.ensemble.registry, self.mock_registry)
    
    def test_get_base_registry(self):
        """Test getting the base registry."""
        # Get the base registry
        base_registry = self.extended_registry.get_base_registry()
        
        # Check that the base registry is returned
        self.assertEqual(base_registry, self.mock_registry)
    
    def test_initialization_without_base_registry(self):
        """Test initialization of the ExtendedModelRegistry without a base registry."""
        # Create the ExtendedModelRegistry without a base registry
        with patch("src.services.model.model_registry.ModelRegistry") as mock_registry_class:
            # Mock the ModelRegistry constructor
            mock_registry_instance = MagicMock(spec=ModelRegistry)
            mock_registry_class.return_value = mock_registry_instance
            
            # Create the ExtendedModelRegistry
            extended_registry = ExtendedModelRegistry()
            
            # Check that a new ModelRegistry was created
            mock_registry_class.assert_called_once()
            
            # Check that all extensions use the new registry
            self.assertEqual(extended_registry.deep_learning.registry, mock_registry_instance)
            self.assertEqual(extended_registry.classification.registry, mock_registry_instance)
            self.assertEqual(extended_registry.pattern_recognition.registry, mock_registry_instance)
            self.assertEqual(extended_registry.ensemble.registry, mock_registry_instance)


if __name__ == "__main__":
    unittest.main()