"""Semantic Versioning Utilities for Model Management.

This module provides utilities for semantic versioning of machine learning models,
including version validation, comparison, incrementation, and migration tools.
"""

import re
import json
import os
import shutil
import importlib
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from src.infrastructure.logging import get_logger
from src.services.model.utils.model_registry_utils import validate_model_version, compare_versions

# Create logger
logger = get_logger(__name__)


class VersionChangeType(Enum):
    """Type of version change."""
    MAJOR = "major"  # Breaking changes, incompatible API changes
    MINOR = "minor"  # New functionality in a backward compatible manner
    PATCH = "patch"  # Backward compatible bug fixes


class SemanticVersioning:
    """Semantic Versioning utilities for model versioning.

    This class provides functionality for working with semantic versions,
    including validation, comparison, and incrementation.
    """

    @staticmethod
    def validate_version(version: str) -> bool:
        """Validate that a version string follows semantic versioning format.

        Args:
            version: The version string to validate.

        Returns:
            bool: True if the version is valid, False otherwise.
        """
        return validate_model_version(version)

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """Compare two semantic version strings.

        Args:
            version1: The first version string.
            version2: The second version string.

        Returns:
            int: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
        """
        return compare_versions(version1, version2)

    @staticmethod
    def increment_version(version: str, change_type: VersionChangeType) -> str:
        """Increment a version string based on the type of change.

        Args:
            version: The current version string (MAJOR.MINOR.PATCH).
            change_type: The type of change (MAJOR, MINOR, or PATCH).

        Returns:
            str: The incremented version string.

        Raises:
            ValueError: If the version string is invalid.
        """
        if not SemanticVersioning.validate_version(version):
            raise ValueError(f"Invalid version format: {version}. Expected format: MAJOR.MINOR.PATCH")

        major, minor, patch = map(int, version.split('.'))

        if change_type == VersionChangeType.MAJOR:
            return f"{major + 1}.0.0"
        elif change_type == VersionChangeType.MINOR:
            return f"{major}.{minor + 1}.0"
        elif change_type == VersionChangeType.PATCH:
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid change type: {change_type}")

    @staticmethod
    def determine_version_change_type(
        model_metadata: Dict[str, Any],
        previous_metadata: Dict[str, Any],
        model_interface_changes: Optional[Dict[str, Any]] = None
    ) -> VersionChangeType:
        """Determine the type of version change based on metadata and interface changes.

        Args:
            model_metadata: The current model metadata.
            previous_metadata: The previous model metadata.
            model_interface_changes: Optional dictionary describing interface changes.

        Returns:
            VersionChangeType: The type of version change (MAJOR, MINOR, or PATCH).
        """
        # Check for breaking changes (MAJOR)
        if model_interface_changes and model_interface_changes.get("breaking_changes", False):
            return VersionChangeType.MAJOR

        # Check for algorithm changes (potentially MAJOR)
        if model_metadata.get("algorithm") != previous_metadata.get("algorithm"):
            return VersionChangeType.MAJOR

        # Check for feature changes (potentially MAJOR)
        current_features = set(model_metadata.get("features", []))
        previous_features = set(previous_metadata.get("features", []))
        if current_features != previous_features:
            # If features were removed, it's a breaking change
            if previous_features - current_features:
                return VersionChangeType.MAJOR

        # Check for new functionality (MINOR)
        if model_interface_changes and model_interface_changes.get("new_functionality", False):
            return VersionChangeType.MINOR

        # Check for added features (MINOR)
        if current_features - previous_features:
            return VersionChangeType.MINOR

        # Check for hyperparameter changes (MINOR)
        current_hyperparams = model_metadata.get("hyperparameters", {})
        previous_hyperparams = previous_metadata.get("hyperparameters", {})
        if current_hyperparams != previous_hyperparams:
            return VersionChangeType.MINOR

        # Default to PATCH for bug fixes, performance improvements, etc.
        return VersionChangeType.PATCH


class ModelVersionMigrator:
    """Model Version Migrator for ensuring backward compatibility.

    This class provides functionality for migrating models between versions,
    including adapters for backward compatibility.
    """

    def __init__(self, models_dir: str):
        """Initialize the model version migrator.

        Args:
            models_dir: The base directory for model storage.
        """
        self.models_dir = models_dir
        self.adapters = {}
        self.migrations_dir = os.path.join(models_dir, "_migrations")
        os.makedirs(self.migrations_dir, exist_ok=True)

    def register_adapter(self, model_name: str, from_version: str, to_version: str, adapter_fn: Callable) -> None:
        """Register an adapter function for migrating between versions.

        Args:
            model_name: The name of the model.
            from_version: The source version.
            to_version: The target version.
            adapter_fn: The adapter function that transforms model inputs/outputs.
        """
        key = f"{model_name}:{from_version}:{to_version}"
        self.adapters[key] = adapter_fn
        logger.info(f"Registered adapter for {model_name} from v{from_version} to v{to_version}")

    def get_adapter(self, model_name: str, from_version: str, to_version: str) -> Optional[Callable]:
        """Get an adapter function for migrating between versions.

        Args:
            model_name: The name of the model.
            from_version: The source version.
            to_version: The target version.

        Returns:
            Optional[Callable]: The adapter function, or None if no adapter is registered.
        """
        key = f"{model_name}:{from_version}:{to_version}"
        return self.adapters.get(key)

    def create_migration_script(self, model_name: str, from_version: str, to_version: str, 
                               migration_code: str) -> str:
        """Create a migration script for migrating between versions.

        Args:
            model_name: The name of the model.
            from_version: The source version.
            to_version: The target version.
            migration_code: The Python code for the migration script.

        Returns:
            str: The path to the created migration script.
        """
        # Create a directory for this model's migrations if it doesn't exist
        model_migrations_dir = os.path.join(self.migrations_dir, model_name)
        os.makedirs(model_migrations_dir, exist_ok=True)

        # Create the migration script file
        script_name = f"migrate_{from_version}_to_{to_version}.py"
        script_path = os.path.join(model_migrations_dir, script_name)

        with open(script_path, 'w') as f:
            f.write(migration_code)

        logger.info(f"Created migration script for {model_name} from v{from_version} to v{to_version}")
        return script_path

    def apply_migration(self, model_name: str, from_version: str, to_version: str, 
                       model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a migration to model data.

        Args:
            model_name: The name of the model.
            from_version: The source version.
            to_version: The target version.
            model_data: The model data to migrate.

        Returns:
            Dict[str, Any]: The migrated model data.

        Raises:
            ValueError: If no migration path is available.
        """
        # Check if we have a direct adapter
        adapter = self.get_adapter(model_name, from_version, to_version)
        if adapter:
            return adapter(model_data)

        # Check if we have a migration script
        script_name = f"migrate_{from_version}_to_{to_version}.py"
        script_path = os.path.join(self.migrations_dir, model_name, script_name)

        if os.path.exists(script_path):
            # Dynamically import the migration script
            spec = importlib.util.spec_from_file_location("migration_module", script_path)
            migration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration_module)

            # Call the migrate function
            if hasattr(migration_module, "migrate"):
                return migration_module.migrate(model_data)

        # If we get here, no migration path was found
        raise ValueError(f"No migration path available for {model_name} from v{from_version} to v{to_version}")

    def create_backward_compatibility_wrapper(self, model_name: str, current_version: str, 
                                            target_version: str) -> Callable:
        """Create a wrapper function for backward compatibility.

        Args:
            model_name: The name of the model.
            current_version: The current model version.
            target_version: The target version to be compatible with.

        Returns:
            Callable: A wrapper function that provides backward compatibility.

        Raises:
            ValueError: If no migration path is available.
        """
        adapter = self.get_adapter(model_name, current_version, target_version)
        if not adapter:
            raise ValueError(f"No adapter available for {model_name} from v{current_version} to v{target_version}")

        def wrapper(model_fn):
            def wrapped_model(*args, **kwargs):
                # Transform inputs using the adapter
                adapted_args, adapted_kwargs = adapter.transform_inputs(*args, **kwargs)
                
                # Call the model function with adapted inputs
                result = model_fn(*adapted_args, **adapted_kwargs)
                
                # Transform outputs using the adapter
                return adapter.transform_outputs(result)
            
            return wrapped_model

        return wrapper