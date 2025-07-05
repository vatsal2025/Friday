"""Enhanced Model Versioning Tools for Trading Engine.

This module extends the base model versioning functionality with semantic versioning,
automatic version incrementation, and migration tools for backward compatibility.
"""

import os
import json
import datetime
import shutil
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from enum import Enum

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ModelVersionManager, ModelVersionStatus, ValidationResult
from src.services.model.utils.semantic_versioning import SemanticVersioning, VersionChangeType, ModelVersionMigrator

# Create logger
logger = get_logger(__name__)


class EnhancedModelVersionManager(ModelVersionManager):
    """Enhanced Model Version Manager with semantic versioning and migration tools.

    This class extends the base ModelVersionManager with additional functionality for
    semantic versioning, automatic version incrementation, and backward compatibility.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the enhanced model version manager.

        Args:
            base_dir: Base directory for model storage.
        """
        super().__init__(base_dir)
        self.migrator = ModelVersionMigrator(self.base_dir)
        self.version_history_file = os.path.join(os.path.dirname(self.base_dir), "version_history.json")
        self._load_version_history()

    def _load_version_history(self) -> None:
        """Load the version history from file."""
        if os.path.exists(self.version_history_file):
            try:
                with open(self.version_history_file, 'r') as f:
                    self.version_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading version history: {str(e)}")
                self.version_history = {}
        else:
            self.version_history = {}

    def _save_version_history(self) -> None:
        """Save the version history to file."""
        try:
            with open(self.version_history_file, 'w') as f:
                json.dump(self.version_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version history: {str(e)}")

    def create_new_version_with_auto_increment(
        self,
        model: Any,
        model_name: str,
        previous_version: Optional[str] = None,
        format: str = "joblib",
        metadata: Optional[Dict[str, Any]] = None,
        custom_save_fn: Optional[Callable[[Any, str], None]] = None,
        interface_changes: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Create a new version of a model with automatic version incrementation.

        Args:
            model: The model object
            model_name: Name of the model
            previous_version: Previous version (if None, will try to find latest)
            format: Model serialization format
            metadata: Additional metadata
            custom_save_fn: Custom function for saving the model
            interface_changes: Dictionary describing interface changes

        Returns:
            Tuple[str, str]: Model ID and new version
        """
        # Get the latest version if previous_version is not provided
        if previous_version is None:
            model_dir = os.path.join(self.base_dir, model_name)
            if os.path.exists(model_dir):
                versions = [v for v in os.listdir(model_dir) 
                           if os.path.isdir(os.path.join(model_dir, v)) and 
                           SemanticVersioning.validate_version(v)]
                if versions:
                    # Sort versions using semantic versioning comparison
                    versions.sort(key=lambda v: [int(x) for x in v.split('.')], reverse=True)
                    previous_version = versions[0]
                else:
                    previous_version = "0.0.0"
            else:
                previous_version = "0.0.0"

        # If this is the first version, start at 1.0.0
        if previous_version == "0.0.0":
            new_version = "1.0.0"
        else:
            # Get previous metadata to determine change type
            previous_metadata = None
            previous_model_path = os.path.join(self.base_dir, model_name, previous_version)
            previous_metadata_path = os.path.join(previous_model_path, "metadata.json")
            
            if os.path.exists(previous_metadata_path):
                try:
                    with open(previous_metadata_path, 'r') as f:
                        previous_metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading previous metadata: {str(e)}")
                    previous_metadata = {}
            else:
                previous_metadata = {}

            # Determine the type of version change
            if metadata and previous_metadata:
                change_type = SemanticVersioning.determine_version_change_type(
                    metadata, previous_metadata, interface_changes
                )
            else:
                # Default to patch version increment if we can't determine the change type
                change_type = VersionChangeType.PATCH

            # Increment the version based on the change type
            new_version = SemanticVersioning.increment_version(previous_version, change_type)
            
            # Log the version change
            logger.info(f"Incrementing {model_name} version from {previous_version} to {new_version} ({change_type.value})")

        # Create the new version
        model_id, _ = super().create_new_version(
            model=model,
            model_name=model_name,
            previous_version=previous_version,
            format=format,
            metadata=metadata,
            custom_save_fn=custom_save_fn
        )

        # Update the version history
        if model_name not in self.version_history:
            self.version_history[model_name] = []

        self.version_history[model_name].append({
            "version": new_version,
            "previous_version": previous_version if previous_version != "0.0.0" else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "change_type": change_type.value if previous_version != "0.0.0" else "initial",
            "model_id": model_id
        })

        self._save_version_history()

        return model_id, new_version

    def register_migration_adapter(self, model_name: str, from_version: str, to_version: str, 
                                 adapter_fn: Callable) -> None:
        """Register a migration adapter for backward compatibility.

        Args:
            model_name: Name of the model
            from_version: Source version
            to_version: Target version
            adapter_fn: Adapter function for migration
        """
        self.migrator.register_adapter(model_name, from_version, to_version, adapter_fn)

    def create_migration_script(self, model_name: str, from_version: str, to_version: str, 
                              migration_code: str) -> str:
        """Create a migration script for backward compatibility.

        Args:
            model_name: Name of the model
            from_version: Source version
            to_version: Target version
            migration_code: Python code for migration

        Returns:
            str: Path to the created migration script
        """
        return self.migrator.create_migration_script(model_name, from_version, to_version, migration_code)

    def apply_migration(self, model_name: str, from_version: str, to_version: str, 
                      model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a migration to model data.

        Args:
            model_name: Name of the model
            from_version: Source version
            to_version: Target version
            model_data: Model data to migrate

        Returns:
            Dict[str, Any]: Migrated model data
        """
        return self.migrator.apply_migration(model_name, from_version, to_version, model_data)

    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get the version history for a model.

        Args:
            model_name: Name of the model

        Returns:
            List[Dict[str, Any]]: List of version history entries
        """
        return self.version_history.get(model_name, [])

    def get_version_lineage(self, model_name: str, version: str) -> List[Dict[str, Any]]:
        """Get the lineage of a specific model version.

        Args:
            model_name: Name of the model
            version: Version to get lineage for

        Returns:
            List[Dict[str, Any]]: List of version history entries in the lineage
        """
        history = self.get_version_history(model_name)
        if not history:
            return []

        # Build a version map for quick lookup
        version_map = {entry["version"]: entry for entry in history}

        # Build the lineage by following the previous_version links
        lineage = []
        current = version_map.get(version)

        while current:
            lineage.append(current)
            prev_version = current.get("previous_version")
            if not prev_version:
                break
            current = version_map.get(prev_version)

        return lineage

    def find_migration_path(self, model_name: str, from_version: str, to_version: str) -> List[Tuple[str, str]]:
        """Find a migration path between two versions.

        Args:
            model_name: Name of the model
            from_version: Source version
            to_version: Target version

        Returns:
            List[Tuple[str, str]]: List of version pairs representing the migration path
        """
        # If versions are the same, no migration needed
        if from_version == to_version:
            return []

        # Get the version history
        history = self.get_version_history(model_name)
        if not history:
            return []

        # Build a version map for quick lookup
        version_map = {entry["version"]: entry for entry in history}

        # Check if we have a direct migration
        direct_key = f"{model_name}:{from_version}:{to_version}"
        if direct_key in self.migrator.adapters:
            return [(from_version, to_version)]

        # Build a graph of version relationships
        graph = {}
        for entry in history:
            version = entry["version"]
            prev_version = entry.get("previous_version")
            if version not in graph:
                graph[version] = []
            if prev_version:
                if prev_version not in graph:
                    graph[prev_version] = []
                # Add edges in both directions
                graph[version].append(prev_version)
                graph[prev_version].append(version)

        # Use breadth-first search to find the shortest path
        queue = [(from_version, [])]
        visited = set([from_version])

        while queue:
            current, path = queue.pop(0)
            if current == to_version:
                # Convert path to pairs
                pairs = []
                for i in range(len(path)):
                    if i > 0:
                        pairs.append((path[i-1], path[i]))
                return pairs

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        return []

    def migrate_model(self, model_name: str, from_version: str, to_version: str) -> bool:
        """Migrate a model from one version to another.

        Args:
            model_name: Name of the model
            from_version: Source version
            to_version: Target version

        Returns:
            bool: True if migration was successful, False otherwise
        """
        try:
            # Find a migration path
            migration_path = self.find_migration_path(model_name, from_version, to_version)
            if not migration_path:
                logger.error(f"No migration path found for {model_name} from v{from_version} to v{to_version}")
                return False

            # Load the source model
            source_path = os.path.join(self.base_dir, model_name, from_version)
            if not os.path.exists(source_path):
                logger.error(f"Source model {model_name} v{from_version} not found")
                return False

            # Load the model and metadata
            model, metadata = self.serializer.load_model(model_name, from_version)
            model_data = {
                "model": model,
                "metadata": metadata.to_dict()
            }

            # Apply migrations along the path
            for source, target in migration_path:
                logger.info(f"Migrating {model_name} from v{source} to v{target}")
                model_data = self.apply_migration(model_name, source, target, model_data)

            # Save the migrated model
            migrated_model = model_data["model"]
            migrated_metadata = model_data["metadata"]

            # Create a new version with the migrated model
            self.create_new_version(
                model=migrated_model,
                model_name=model_name,
                previous_version=from_version,
                format=migrated_metadata.get("format", "joblib"),
                metadata=migrated_metadata,
                custom_save_fn=None
            )

            logger.info(f"Successfully migrated {model_name} from v{from_version} to v{to_version}")
            return True

        except Exception as e:
            logger.error(f"Error migrating {model_name} from v{from_version} to v{to_version}: {str(e)}")
            return False