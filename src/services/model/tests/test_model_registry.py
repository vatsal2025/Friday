"""Unit tests for the model registry system.

This module contains tests for the ModelRegistry, ModelOperations, ModelVersioning,
ModelSerializer, and ModelLoader classes.
"""

import os
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from src.services.model.model_registry import ModelRegistry
from src.services.model.model_operations import ModelOperations
from src.services.model.model_versioning import ModelVersioning
from src.services.model.model_serialization import ModelSerializer
from src.services.model.model_loader import ModelLoader
from src.services.model.model_trainer_integration import ModelTrainerIntegration


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(models_dir=self.temp_dir)
        
        # Create a sample model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Sample metadata
        self.metadata = {
            "model_type": "random_forest",
            "metrics": {"mse": 0.5, "r2": 0.8},
            "hyperparameters": {"n_estimators": 10}
        }

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_register_model(self):
        """Test registering a model."""
        model_id = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model",
            tags=["test", "regression"],
            metadata=self.metadata
        )
        
        self.assertIsNotNone(model_id)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_model", "1.0.0")))
        
        # Check metadata file exists
        metadata_file = os.path.join(self.temp_dir, "test_model", "1.0.0", "metadata.json")
        self.assertTrue(os.path.exists(metadata_file))

    def test_load_model(self):
        """Test loading a model."""
        # Register a model first
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Load the model
        loaded_model = self.registry.load_model("test_model", "1.0.0")
        
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, RandomForestRegressor)

    def test_get_model_metadata(self):
        """Test getting model metadata."""
        # Register a model first
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Get the metadata
        metadata = self.registry.get_model_metadata("test_model", "1.0.0")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["model_type"], "random_forest")
        self.assertEqual(metadata["metrics"]["mse"], 0.5)
        self.assertEqual(metadata["metrics"]["r2"], 0.8)
        self.assertEqual(metadata["hyperparameters"]["n_estimators"], 10)

    def test_update_model_metadata(self):
        """Test updating model metadata."""
        # Register a model first
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Update the metadata
        updated_metadata = {"status": "production", "updated_by": "test"}
        self.registry.update_model_metadata("test_model", "1.0.0", updated_metadata)
        
        # Get the updated metadata
        metadata = self.registry.get_model_metadata("test_model", "1.0.0")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["status"], "production")
        self.assertEqual(metadata["updated_by"], "test")

    def test_list_models(self):
        """Test listing models."""
        # Register two models
        self.registry.register_model(
            model=self.model,
            model_name="test_model_1",
            model_version="1.0.0",
            description="Test model 1",
            tags=["test", "regression"],
            metadata=self.metadata
        )
        
        self.registry.register_model(
            model=self.model,
            model_name="test_model_2",
            model_version="1.0.0",
            description="Test model 2",
            tags=["test", "classification"],
            metadata=self.metadata
        )
        
        # List all models
        models = self.registry.list_models()
        
        self.assertEqual(len(models), 2)
        self.assertIn("test_model_1", models)
        self.assertIn("test_model_2", models)

    def test_get_model_versions(self):
        """Test getting model versions."""
        # Register two versions of the same model
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model v1",
            tags=["test"],
            metadata=self.metadata
        )
        
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="2.0.0",
            description="Test model v2",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Get versions
        versions = self.registry.get_model_versions("test_model")
        
        self.assertEqual(len(versions), 2)
        self.assertIn("1.0.0", versions)
        self.assertIn("2.0.0", versions)

    def test_get_latest_version(self):
        """Test getting the latest model version."""
        # Register two versions of the same model
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model v1",
            tags=["test"],
            metadata=self.metadata
        )
        
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="2.0.0",
            description="Test model v2",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Get the latest version
        latest_version = self.registry.get_latest_version("test_model")
        
        self.assertEqual(latest_version, "2.0.0")

    def test_delete_model(self):
        """Test deleting a model."""
        # Register a model
        self.registry.register_model(
            model=self.model,
            model_name="test_model",
            model_version="1.0.0",
            description="Test model",
            tags=["test"],
            metadata=self.metadata
        )
        
        # Delete the model
        self.registry.delete_model("test_model", "1.0.0")
        
        # Check if the model directory exists
        model_dir = os.path.join(self.temp_dir, "test_model", "1.0.0")
        self.assertFalse(os.path.exists(model_dir))


class TestModelOperations(unittest.TestCase):
    """Test cases for the ModelOperations class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.operations = ModelOperations(model_registry=self.mock_registry)

    def test_list_models(self):
        """Test listing models."""
        # Set up mock return value
        self.mock_registry.list_models.return_value = ["model1", "model2"]
        
        # Call the method
        models = self.operations.list_models()
        
        # Assertions
        self.assertEqual(len(models), 2)
        self.assertIn("model1", models)
        self.assertIn("model2", models)
        self.mock_registry.list_models.assert_called_once()

    def test_get_model_versions(self):
        """Test getting model versions."""
        # Set up mock return value
        self.mock_registry.get_model_versions.return_value = ["1.0.0", "2.0.0"]
        
        # Call the method
        versions = self.operations.get_model_versions("test_model")
        
        # Assertions
        self.assertEqual(len(versions), 2)
        self.assertIn("1.0.0", versions)
        self.assertIn("2.0.0", versions)
        self.mock_registry.get_model_versions.assert_called_once_with("test_model")

    @patch('src.services.model.model_operations.ModelOperations._get_model_metrics')
    def test_find_best_model(self, mock_get_metrics):
        """Test finding the best model based on a metric."""
        # Set up mock return values
        self.mock_registry.get_model_versions.return_value = ["1.0.0", "2.0.0"]
        mock_get_metrics.side_effect = [
            {"r2": 0.8},  # Version 1.0.0
            {"r2": 0.9}   # Version 2.0.0
        ]
        
        # Call the method
        best_model = self.operations.find_best_model("test_model", "r2", higher_is_better=True)
        
        # Assertions
        self.assertEqual(best_model, "2.0.0")
        self.mock_registry.get_model_versions.assert_called_once_with("test_model")
        self.assertEqual(mock_get_metrics.call_count, 2)


class TestModelVersioning(unittest.TestCase):
    """Test cases for the ModelVersioning class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.versioning = ModelVersioning(model_registry=self.mock_registry)

    def test_get_model_lineage(self):
        """Test getting model lineage."""
        # Set up mock return values
        self.mock_registry.get_model_versions.return_value = ["1.0.0", "2.0.0"]
        self.mock_registry.get_model_metadata.side_effect = [
            {"created_at": "2023-01-01", "metrics": {"r2": 0.8}},
            {"created_at": "2023-01-02", "metrics": {"r2": 0.9}}
        ]
        
        # Call the method
        lineage = self.versioning.get_model_lineage("test_model")
        
        # Assertions
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0]["version"], "1.0.0")
        self.assertEqual(lineage[1]["version"], "2.0.0")
        self.mock_registry.get_model_versions.assert_called_once_with("test_model")
        self.assertEqual(self.mock_registry.get_model_metadata.call_count, 2)

    def test_tag_version(self):
        """Test tagging a model version."""
        # Set up mock return value
        self.mock_registry.get_model_metadata.return_value = {"tags": ["test"]}
        
        # Call the method
        self.versioning.tag_version("test_model", "1.0.0", "production")
        
        # Assertions
        self.mock_registry.get_model_metadata.assert_called_once_with("test_model", "1.0.0")
        self.mock_registry.update_model_metadata.assert_called_once()
        # Check that the tags were updated correctly
        update_call_args = self.mock_registry.update_model_metadata.call_args[0]
        self.assertEqual(update_call_args[0], "test_model")
        self.assertEqual(update_call_args[1], "1.0.0")
        self.assertEqual(set(update_call_args[2]["tags"]), {"test", "production"})


class TestModelSerializer(unittest.TestCase):
    """Test cases for the ModelSerializer class."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.serializer = ModelSerializer()
        
        # Create a sample model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_serialize_deserialize_model(self):
        """Test serializing and deserializing a model."""
        # Serialize the model
        file_path = os.path.join(self.temp_dir, "test_model.joblib")
        self.serializer.serialize_model(self.model, file_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Deserialize the model
        loaded_model = self.serializer.deserialize_model(file_path)
        
        # Check if the loaded model is the same type
        self.assertIsInstance(loaded_model, RandomForestRegressor)
        
        # Check if the model parameters are the same
        self.assertEqual(loaded_model.n_estimators, self.model.n_estimators)
        self.assertEqual(loaded_model.random_state, self.model.random_state)

    def test_compute_model_hash(self):
        """Test computing a model hash."""
        # Compute the hash
        hash1 = self.serializer.compute_model_hash(self.model)
        
        # Create a different model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model2 = RandomForestRegressor(n_estimators=20, random_state=42)
        model2.fit(X, y)
        hash2 = self.serializer.compute_model_hash(model2)
        
        # Check if the hashes are different
        self.assertNotEqual(hash1, hash2)

    def test_get_serialized_size(self):
        """Test getting the serialized size of a model."""
        # Get the size
        size = self.serializer.get_serialized_size(self.model)
        
        # Check if the size is positive
        self.assertGreater(size, 0)


class TestModelLoader(unittest.TestCase):
    """Test cases for the ModelLoader class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a mock ModelRegistry and ModelOperations
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_operations = MagicMock(spec=ModelOperations)
        
        # Configure the loader to use the mocks
        self.loader = ModelLoader(
            model_registry=self.mock_registry,
            model_operations=self.mock_operations
        )
        
        # Create a sample model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)

    def test_load_model(self):
        """Test loading a model."""
        # Set up mock return values
        self.mock_registry.get_latest_version.return_value = "1.0.0"
        self.mock_registry.load_model.return_value = self.model
        
        # Call the method
        loaded_model = self.loader.load_model("test_model")
        
        # Assertions
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, RandomForestRegressor)
        self.mock_registry.get_latest_version.assert_called_once_with("test_model")
        self.mock_registry.load_model.assert_called_once_with("test_model", "1.0.0")

    def test_load_model_by_tags(self):
        """Test loading models by tags."""
        # Set up mock return values
        self.mock_operations.list_models.return_value = ["model1", "model2"]
        self.mock_registry.get_model_metadata.side_effect = [
            {"tags": ["test", "regression"]},
            {"tags": ["test", "classification"]}
        ]
        self.mock_registry.get_latest_version.side_effect = ["1.0.0", "1.0.0"]
        self.mock_registry.load_model.side_effect = [self.model, self.model]
        
        # Call the method
        loaded_models = self.loader.load_model_by_tags(["test"])
        
        # Assertions
        self.assertEqual(len(loaded_models), 2)
        self.mock_operations.list_models.assert_called_once()
        self.assertEqual(self.mock_registry.get_model_metadata.call_count, 2)
        self.assertEqual(self.mock_registry.get_latest_version.call_count, 2)
        self.assertEqual(self.mock_registry.load_model.call_count, 2)

    def test_load_best_model(self):
        """Test loading the best model based on a metric."""
        # Set up mock return values
        self.mock_operations.find_best_model.return_value = "2.0.0"
        self.mock_registry.load_model.return_value = self.model
        
        # Call the method
        loaded_model = self.loader.load_best_model("test_model", "r2")
        
        # Assertions
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, RandomForestRegressor)
        self.mock_operations.find_best_model.assert_called_once_with(
            "test_model", "r2", higher_is_better=True
        )
        self.mock_registry.load_model.assert_called_once_with("test_model", "2.0.0")

    def test_get_model_metadata(self):
        """Test getting model metadata."""
        # Set up mock return values
        self.mock_registry.get_latest_version.return_value = "1.0.0"
        self.mock_registry.get_model_metadata.return_value = {
            "model_type": "random_forest",
            "metrics": {"r2": 0.8}
        }
        
        # Call the method
        metadata = self.loader.get_model_metadata("test_model")
        
        # Assertions
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["model_type"], "random_forest")
        self.assertEqual(metadata["metrics"]["r2"], 0.8)
        self.mock_registry.get_latest_version.assert_called_once_with("test_model")
        self.mock_registry.get_model_metadata.assert_called_once_with("test_model", "1.0.0")

    def test_clear_cache(self):
        """Test clearing the model cache."""
        # Set up the cache
        self.loader._model_cache = {"test_model_1.0.0": self.model}
        
        # Call the method
        self.loader.clear_cache()
        
        # Assertions
        self.assertEqual(len(self.loader._model_cache), 0)


class TestModelTrainerIntegration(unittest.TestCase):
    """Test cases for the ModelTrainerIntegration class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a mock ModelRegistry and ModelSerializer
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_serializer = MagicMock(spec=ModelSerializer)
        
        # Configure the integration to use the mocks
        self.integration = ModelTrainerIntegration(
            model_registry=self.mock_registry,
            model_serializer=self.mock_serializer
        )
        
        # Create a sample model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)

    def test_register_model_from_trainer(self):
        """Test registering a model from a trainer."""
        # Set up mock return value
        self.mock_registry.register_model.return_value = "model_id_123"
        
        # Call the method
        model_id = self.integration.register_model_from_trainer(
            model=self.model,
            model_name="test_model",
            model_type="random_forest",
            evaluation_results={
                "metrics": {"r2": 0.8},
                "details": {"test_size": 20}
            },
            training_data_info={"train_size": 80, "test_size": 20},
            hyperparameters={"n_estimators": 10},
            feature_importance={"feature_1": 0.5, "feature_2": 0.3},
            tags=["test", "regression"],
            description="Test model"
        )
        
        # Assertions
        self.assertEqual(model_id, "model_id_123")
        self.mock_registry.register_model.assert_called_once()
        # Check that the metadata was constructed correctly
        register_call_args = self.mock_registry.register_model.call_args[1]
        self.assertEqual(register_call_args["model_name"], "test_model")
        self.assertEqual(register_call_args["tags"], ["test", "regression"])
        self.assertEqual(register_call_args["description"], "Test model")
        self.assertEqual(register_call_args["metadata"]["model_type"], "random_forest")
        self.assertEqual(register_call_args["metadata"]["metrics"]["r2"], 0.8)
        self.assertEqual(register_call_args["metadata"]["hyperparameters"]["n_estimators"], 10)


if __name__ == "__main__":
    unittest.main()