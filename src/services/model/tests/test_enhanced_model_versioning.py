"""Test cases for the enhanced model versioning system.

This module contains tests for semantic versioning, automatic version incrementation,
and migration tools for backward compatibility.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.services.model import ModelRegistry
from src.services.model.enhanced_model_versioning import EnhancedModelVersioning
from src.services.model.utils.semantic_versioning import (
    SemanticVersioning, 
    VersionChangeType,
    ModelVersionMigrator
)


class TestSemanticVersioning(unittest.TestCase):
    """Test cases for the SemanticVersioning class."""

    def test_validate_version(self):
        """Test version validation."""
        # Valid versions
        self.assertTrue(SemanticVersioning.validate_version("1.0.0"))
        self.assertTrue(SemanticVersioning.validate_version("0.1.0"))
        self.assertTrue(SemanticVersioning.validate_version("10.20.30"))
        
        # Invalid versions
        self.assertFalse(SemanticVersioning.validate_version("1.0"))
        self.assertFalse(SemanticVersioning.validate_version("1"))
        self.assertFalse(SemanticVersioning.validate_version("1.0.0.0"))
        self.assertFalse(SemanticVersioning.validate_version("v1.0.0"))
        self.assertFalse(SemanticVersioning.validate_version("1.0.0-alpha"))

    def test_compare_versions(self):
        """Test version comparison."""
        # Equal versions
        self.assertEqual(SemanticVersioning.compare_versions("1.0.0", "1.0.0"), 0)
        
        # First version is lower
        self.assertEqual(SemanticVersioning.compare_versions("1.0.0", "1.0.1"), -1)
        self.assertEqual(SemanticVersioning.compare_versions("1.0.0", "1.1.0"), -1)
        self.assertEqual(SemanticVersioning.compare_versions("1.0.0", "2.0.0"), -1)
        
        # First version is higher
        self.assertEqual(SemanticVersioning.compare_versions("1.0.1", "1.0.0"), 1)
        self.assertEqual(SemanticVersioning.compare_versions("1.1.0", "1.0.0"), 1)
        self.assertEqual(SemanticVersioning.compare_versions("2.0.0", "1.0.0"), 1)

    def test_increment_version(self):
        """Test version incrementation."""
        # Patch increment
        self.assertEqual(
            SemanticVersioning.increment_version("1.0.0", VersionChangeType.PATCH),
            "1.0.1"
        )
        
        # Minor increment
        self.assertEqual(
            SemanticVersioning.increment_version("1.0.0", VersionChangeType.MINOR),
            "1.1.0"
        )
        
        # Major increment
        self.assertEqual(
            SemanticVersioning.increment_version("1.0.0", VersionChangeType.MAJOR),
            "2.0.0"
        )

    def test_determine_version_change_type(self):
        """Test determining version change type based on metadata changes."""
        # Setup test metadata
        metadata1 = {
            "metrics": {"accuracy": 0.8, "f1": 0.75},
            "hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "input_schema": {"features": 10},
            "output_schema": {"classes": 2}
        }
        
        # No changes (should be PATCH)
        metadata2 = metadata1.copy()
        self.assertEqual(
            SemanticVersioning.determine_version_change_type(metadata2, metadata1),
            VersionChangeType.PATCH
        )
        
        # Metric changes only (should be PATCH)
        metadata2 = metadata1.copy()
        metadata2["metrics"] = {"accuracy": 0.85, "f1": 0.78}
        self.assertEqual(
            SemanticVersioning.determine_version_change_type(metadata2, metadata1),
            VersionChangeType.PATCH
        )
        
        # Hyperparameter changes (should be MINOR)
        metadata2 = metadata1.copy()
        metadata2["hyperparameters"] = {"n_estimators": 200, "max_depth": 5}
        self.assertEqual(
            SemanticVersioning.determine_version_change_type(metadata2, metadata1),
            VersionChangeType.MINOR
        )
        
        # Interface changes (should be MAJOR)
        metadata2 = metadata1.copy()
        interface_changes = {"input_schema_changed": True}
        self.assertEqual(
            SemanticVersioning.determine_version_change_type(metadata2, metadata1, interface_changes),
            VersionChangeType.MAJOR
        )


class TestModelVersionMigrator(unittest.TestCase):
    """Test cases for the ModelVersionMigrator class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.migrations_dir = os.path.join(self.temp_dir, "migrations")
        os.makedirs(self.migrations_dir, exist_ok=True)
        
        # Initialize the migrator
        self.migrator = ModelVersionMigrator(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_register_and_get_adapter(self):
        """Test registering and retrieving migration adapters."""
        # Define a simple adapter function
        def test_adapter(model_data):
            return model_data
        
        # Register the adapter
        self.migrator.register_adapter("test_model", "1.0.0", "2.0.0", test_adapter)
        
        # Get the adapter
        adapter = self.migrator.get_adapter("test_model", "1.0.0", "2.0.0")
        
        # Verify the adapter is the same function
        self.assertEqual(adapter, test_adapter)
        
        # Try to get a non-existent adapter
        adapter = self.migrator.get_adapter("test_model", "1.0.0", "3.0.0")
        self.assertIsNone(adapter)

    def test_create_migration_script(self):
        """Test creating migration scripts."""
        # Create a migration script
        script_path = self.migrator.create_migration_script(
            "test_model",
            "1.0.0",
            "2.0.0",
            "def migrate(model_data):\n    return model_data"
        )
        
        # Verify the script was created
        self.assertTrue(os.path.exists(script_path))
        
        # Verify the script content
        with open(script_path, "r") as f:
            content = f.read()
        
        self.assertIn("def migrate(model_data):", content)

    def test_apply_migration_with_adapter(self):
        """Test applying migration using a registered adapter."""
        # Define a test adapter function
        def test_adapter(model_data):
            model_data["metadata"]["version"] = "2.0.0"
            return model_data
        
        # Register the adapter
        self.migrator.register_adapter("test_model", "1.0.0", "2.0.0", test_adapter)
        
        # Apply the migration
        model_data = {
            "model": MagicMock(),
            "metadata": {"version": "1.0.0"}
        }
        
        migrated_data = self.migrator.apply_migration(
            "test_model",
            "1.0.0",
            "2.0.0",
            model_data
        )
        
        # Verify the migration was applied
        self.assertEqual(migrated_data["metadata"]["version"], "2.0.0")

    def test_create_backward_compatibility_wrapper(self):
        """Test creating a backward compatibility wrapper."""
        # Define transformation functions
        def transform_inputs(*args, **kwargs):
            return args, kwargs
        
        def transform_outputs(output):
            return output
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0])
        
        # Create the wrapper
        wrapped_model = self.migrator.create_backward_compatibility_wrapper(
            mock_model,
            transform_inputs,
            transform_outputs
        )
        
        # Test the wrapper
        result = wrapped_model.predict(np.array([[1, 2, 3]]))
        
        # Verify the result
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))
        mock_model.predict.assert_called_once()


class TestEnhancedModelVersioning(unittest.TestCase):
    """Test cases for the EnhancedModelVersioning class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock ModelRegistry
        self.mock_registry = MagicMock(spec=ModelRegistry)
        self.mock_registry.base_dir = self.temp_dir
        self.mock_registry.get_model_metadata.return_value = {
            "metrics": {"accuracy": 0.8},
            "hyperparameters": {"n_estimators": 100}
        }
        
        # Initialize the enhanced versioning
        self.versioning = EnhancedModelVersioning(registry=self.mock_registry)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_auto_increment_version_patch(self):
        """Test automatic version incrementation with patch changes."""
        # Setup test metadata
        current_version = "1.0.0"
        metadata = {
            "metrics": {"accuracy": 0.85},  # Changed metrics
            "hyperparameters": {"n_estimators": 100}  # Same hyperparameters
        }
        previous_metadata = {
            "metrics": {"accuracy": 0.8},
            "hyperparameters": {"n_estimators": 100}
        }
        
        # Auto-increment version
        new_version = self.versioning.auto_increment_version(
            "test_model",
            current_version,
            metadata,
            previous_metadata
        )
        
        # Verify the new version
        self.assertEqual(new_version, "1.0.1")

    def test_auto_increment_version_minor(self):
        """Test automatic version incrementation with minor changes."""
        # Setup test metadata
        current_version = "1.0.0"
        metadata = {
            "metrics": {"accuracy": 0.85},
            "hyperparameters": {"n_estimators": 200}  # Changed hyperparameters
        }
        previous_metadata = {
            "metrics": {"accuracy": 0.8},
            "hyperparameters": {"n_estimators": 100}
        }
        
        # Auto-increment version
        new_version = self.versioning.auto_increment_version(
            "test_model",
            current_version,
            metadata,
            previous_metadata
        )
        
        # Verify the new version
        self.assertEqual(new_version, "1.1.0")

    def test_auto_increment_version_major(self):
        """Test automatic version incrementation with major changes."""
        # Setup test metadata
        current_version = "1.0.0"
        metadata = {
            "metrics": {"accuracy": 0.85},
            "hyperparameters": {"n_estimators": 100}
        }
        previous_metadata = {
            "metrics": {"accuracy": 0.8},
            "hyperparameters": {"n_estimators": 100}
        }
        
        # Define interface changes
        interface_changes = {"input_schema_changed": True}
        
        # Auto-increment version
        new_version = self.versioning.auto_increment_version(
            "test_model",
            current_version,
            metadata,
            previous_metadata,
            interface_changes
        )
        
        # Verify the new version
        self.assertEqual(new_version, "2.0.0")

    @patch('src.services.model.enhanced_model_versioning.SecurityAuditLogger')
    def test_version_history_and_lineage(self, mock_audit_logger):
        """Test getting version history and lineage."""
        # Setup mock return values
        self.mock_registry.get_model_versions.return_value = ["1.0.0", "1.1.0", "2.0.0"]
        self.mock_registry.get_model_metadata.side_effect = [
            {"created_at": "2023-01-01", "metrics": {"accuracy": 0.8}},
            {"created_at": "2023-01-02", "metrics": {"accuracy": 0.85}},
            {"created_at": "2023-01-03", "metrics": {"accuracy": 0.9}}
        ]
        
        # Get version history
        history = self.versioning.get_version_history("test_model")
        
        # Verify the history
        self.assertEqual(len(history), 3)
        
        # Get version lineage
        lineage = self.versioning.get_version_lineage("test_model", "2.0.0")
        
        # Verify the lineage
        self.assertEqual(len(lineage), 3)

    def test_register_migration_adapter(self):
        """Test registering a migration adapter."""
        # Define a test adapter function
        def test_adapter(model_data):
            return model_data
        
        # Register the adapter
        self.versioning.register_migration_adapter(
            "test_model",
            "1.0.0",
            "2.0.0",
            test_adapter
        )
        
        # Verify the adapter was registered
        adapter = self.versioning.migrator.get_adapter("test_model", "1.0.0", "2.0.0")
        self.assertEqual(adapter, test_adapter)

    def test_find_migration_path(self):
        """Test finding a migration path between versions."""
        # Setup mock return values
        self.mock_registry.get_model_versions.return_value = ["1.0.0", "1.1.0", "2.0.0"]
        
        # Register adapters
        def test_adapter(model_data):
            return model_data
        
        self.versioning.register_migration_adapter("test_model", "1.0.0", "1.1.0", test_adapter)
        self.versioning.register_migration_adapter("test_model", "1.1.0", "2.0.0", test_adapter)
        
        # Find migration path
        path = self.versioning.find_migration_path("test_model", "1.0.0", "2.0.0")
        
        # Verify the path
        self.assertEqual(path, ["1.0.0", "1.1.0", "2.0.0"])


if __name__ == "__main__":
    unittest.main()