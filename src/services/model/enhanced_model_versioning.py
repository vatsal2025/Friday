"""Enhanced Model Versioning for Model Management.

This module extends the base model versioning functionality with semantic versioning,
automatic version incrementation, and migration tools for backward compatibility.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from src.infrastructure.logging import get_logger
from src.infrastructure.security.audit_logging import SecurityAuditLogger, AuditEventType
from src.services.model.model_versioning import ModelVersioning
from src.services.model.utils.semantic_versioning import (
    SemanticVersioning, VersionChangeType, ModelVersionMigrator
)

# Create logger
logger = get_logger(__name__)


class EnhancedModelVersioning(ModelVersioning):
    """Enhanced Model Versioning with semantic versioning and migration tools.

    This class extends the base ModelVersioning with additional functionality for
    semantic versioning, automatic version incrementation, and backward compatibility.
    """

    def __init__(self, registry=None):
        """Initialize the enhanced model versioning.

        Args:
            registry: Optional model registry instance.
        """
        super().__init__(registry)
        self.audit_logger = SecurityAuditLogger()
        self.migrator = ModelVersionMigrator(self.registry.base_dir)
        self.version_history_file = os.path.join(self.registry.base_dir, "version_history.json")
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

    def auto_increment_version(self, model_name: str, current_version: str, 
                             metadata: Dict[str, Any], 
                             previous_metadata: Dict[str, Any],
                             interface_changes: Optional[Dict[str, Any]] = None) -> str:
        """Automatically increment a model version based on changes.

        Args:
            model_name: Name of the model
            current_version: Current version string
            metadata: New model metadata
            previous_metadata: Previous model metadata
            interface_changes: Optional dictionary describing interface changes

        Returns:
            str: New version string
        """
        # Determine the type of version change
        change_type = SemanticVersioning.determine_version_change_type(
            metadata, previous_metadata, interface_changes
        )
        
        # Increment the version
        new_version = SemanticVersioning.increment_version(current_version, change_type)
        
        # Log the version change
        logger.info(f"Auto-incrementing {model_name} version from {current_version} to {new_version} ({change_type.value})")
        
        # Audit log the version change
        self.audit_logger.log_model_version_change(
            model_name=model_name,
            old_version=current_version,
            new_version=new_version,
            change_type=change_type.value,
            user_id=metadata.get("created_by", "system")
        )
        
        # Update version history
        if model_name not in self.version_history:
            self.version_history[model_name] = []
            
        self.version_history[model_name].append({
            "version": new_version,
            "previous_version": current_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "change_type": change_type.value,
            "changes": self._summarize_changes(metadata, previous_metadata, interface_changes)
        })
        
        self._save_version_history()
        
        return new_version

    def _summarize_changes(self, metadata: Dict[str, Any], previous_metadata: Dict[str, Any],
                         interface_changes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Summarize the changes between two model versions.

        Args:
            metadata: New model metadata
            previous_metadata: Previous model metadata
            interface_changes: Optional dictionary describing interface changes

        Returns:
            Dict[str, Any]: Summary of changes
        """
        changes = {}
        
        # Check for algorithm changes
        if metadata.get("algorithm") != previous_metadata.get("algorithm"):
            changes["algorithm"] = {
                "from": previous_metadata.get("algorithm"),
                "to": metadata.get("algorithm")
            }
        
        # Check for feature changes
        current_features = set(metadata.get("features", []))
        previous_features = set(previous_metadata.get("features", []))
        
        if current_features != previous_features:
            changes["features"] = {
                "added": list(current_features - previous_features),
                "removed": list(previous_features - current_features)
            }
        
        # Check for hyperparameter changes
        current_hyperparams = metadata.get("hyperparameters", {})
        previous_hyperparams = previous_metadata.get("hyperparameters", {})
        
        if current_hyperparams != previous_hyperparams:
            hyperparameter_changes = {}
            
            # Find added, removed, and modified hyperparameters
            for key in set(current_hyperparams.keys()) | set(previous_hyperparams.keys()):
                if key not in previous_hyperparams:
                    hyperparameter_changes[key] = {"added": current_hyperparams[key]}
                elif key not in current_hyperparams:
                    hyperparameter_changes[key] = {"removed": previous_hyperparams[key]}
                elif current_hyperparams[key] != previous_hyperparams[key]:
                    hyperparameter_changes[key] = {
                        "from": previous_hyperparams[key],
                        "to": current_hyperparams[key]
                    }
            
            if hyperparameter_changes:
                changes["hyperparameters"] = hyperparameter_changes
        
        # Check for performance changes
        current_metrics = metadata.get("metrics", {})
        previous_metrics = previous_metadata.get("metrics", {})
        
        if current_metrics and previous_metrics:
            metric_changes = {}
            
            for key in set(current_metrics.keys()) & set(previous_metrics.keys()):
                if isinstance(current_metrics[key], (int, float)) and isinstance(previous_metrics[key], (int, float)):
                    diff = current_metrics[key] - previous_metrics[key]
                    if diff != 0:
                        metric_changes[key] = {
                            "from": previous_metrics[key],
                            "to": current_metrics[key],
                            "diff": diff,
                            "percent_change": (diff / previous_metrics[key]) * 100 if previous_metrics[key] != 0 else float('inf')
                        }
            
            if metric_changes:
                changes["metrics"] = metric_changes
        
        # Include interface changes if provided
        if interface_changes:
            changes["interface"] = interface_changes
        
        return changes

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

    def create_backward_compatibility_wrapper(self, model_name: str, current_version: str, 
                                            target_version: str) -> Callable:
        """Create a wrapper function for backward compatibility.

        Args:
            model_name: Name of the model
            current_version: Current model version
            target_version: Target version to be compatible with

        Returns:
            Callable: Wrapper function for backward compatibility
        """
        return self.migrator.create_backward_compatibility_wrapper(model_name, current_version, target_version)

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
        if self.migrator.get_adapter(model_name, from_version, to_version):
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
        queue = [(from_version, [from_version])]
        visited = set([from_version])

        while queue:
            current, path = queue.pop(0)
            if current == to_version:
                # Convert path to pairs
                pairs = []
                for i in range(len(path) - 1):
                    pairs.append((path[i], path[i+1]))
                return pairs

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        return []