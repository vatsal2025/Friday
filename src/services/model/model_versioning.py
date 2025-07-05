"""Model Versioning for Friday AI Trading System.

This module provides functionality for tracking and comparing model versions.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
import difflib

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry

# Create logger
logger = get_logger(__name__)


class ModelVersioning:
    """Model Versioning for tracking and comparing model versions.

    This class provides functionality for tracking model versions, comparing versions,
    and managing model lineage.

    Attributes:
        registry: The model registry instance.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize the model versioning.

        Args:
            registry: The model registry. If None, a new one will be created.
        """
        self.registry = registry or ModelRegistry()
        logger.info("Initialized ModelVersioning")

    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """Get the lineage of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[Dict[str, Any]]: List of model metadata for all versions, sorted by creation time.

        Raises:
            ValueError: If no models with the given name are found in the registry.
        """
        # Get all versions of the model
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            raise ValueError(f"No models with name {model_name} found in registry")
        
        # Sort by creation time
        versions.sort(key=lambda x: x.get("created_at", ""))
        
        return versions

    def compare_model_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model.

        Args:
            model_name: Name of the model.
            version1: First version to compare.
            version2: Second version to compare.

        Returns:
            Dict[str, Any]: Comparison results.

        Raises:
            ValueError: If either version is not found in the registry.
        """
        # Get model IDs for both versions
        model_id1 = None
        model_id2 = None
        
        for model_id, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name:
                if metadata.get("version") == version1:
                    model_id1 = model_id
                elif metadata.get("version") == version2:
                    model_id2 = model_id
        
        if model_id1 is None:
            raise ValueError(f"Model with name {model_name} and version {version1} not found in registry")
        if model_id2 is None:
            raise ValueError(f"Model with name {model_name} and version {version2} not found in registry")
        
        # Get metadata for both versions
        metadata1 = self.registry.get_model_metadata(model_id1)
        metadata2 = self.registry.get_model_metadata(model_id2)
        
        # Compare metrics
        metrics1 = metadata1.get("metrics", {})
        metrics2 = metadata2.get("metrics", {})
        
        metric_comparison = {}
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            value1 = metrics1.get(metric, None)
            value2 = metrics2.get(metric, None)
            
            if value1 is not None and value2 is not None:
                diff = value2 - value1
                pct_change = (diff / value1) * 100 if value1 != 0 else float('inf')
                metric_comparison[metric] = {
                    "version1": value1,
                    "version2": value2,
                    "difference": diff,
                    "percent_change": pct_change
                }
            else:
                metric_comparison[metric] = {
                    "version1": value1,
                    "version2": value2,
                    "difference": None,
                    "percent_change": None
                }
        
        # Compare other metadata
        metadata_comparison = {}
        all_metadata_keys = set(metadata1.get("additional_metadata", {}).keys()) | set(metadata2.get("additional_metadata", {}).keys())
        
        for key in all_metadata_keys:
            value1 = metadata1.get("additional_metadata", {}).get(key, None)
            value2 = metadata2.get("additional_metadata", {}).get(key, None)
            
            metadata_comparison[key] = {
                "version1": value1,
                "version2": value2,
                "changed": value1 != value2
            }
        
        # Compare tags
        tags1 = set(metadata1.get("tags", []))
        tags2 = set(metadata2.get("tags", []))
        
        tags_comparison = {
            "common": list(tags1 & tags2),
            "only_in_version1": list(tags1 - tags2),
            "only_in_version2": list(tags2 - tags1)
        }
        
        # Create comparison result
        comparison = {
            "model_name": model_name,
            "version1": {
                "version": version1,
                "id": model_id1,
                "created_at": metadata1.get("created_at"),
                "status": metadata1.get("status")
            },
            "version2": {
                "version": version2,
                "id": model_id2,
                "created_at": metadata2.get("created_at"),
                "status": metadata2.get("status")
            },
            "metrics_comparison": metric_comparison,
            "metadata_comparison": metadata_comparison,
            "tags_comparison": tags_comparison,
            "time_difference": self._calculate_time_difference(metadata1.get("created_at"), metadata2.get("created_at"))
        }
        
        return comparison

    def _calculate_time_difference(self, time1_str: str, time2_str: str) -> Dict[str, Any]:
        """Calculate the time difference between two ISO format timestamps.

        Args:
            time1_str: First timestamp in ISO format.
            time2_str: Second timestamp in ISO format.

        Returns:
            Dict[str, Any]: Time difference in various units.
        """
        try:
            time1 = datetime.datetime.fromisoformat(time1_str)
            time2 = datetime.datetime.fromisoformat(time2_str)
            
            diff = time2 - time1
            
            return {
                "total_seconds": diff.total_seconds(),
                "days": diff.days,
                "hours": diff.seconds // 3600,
                "minutes": (diff.seconds % 3600) // 60,
                "seconds": diff.seconds % 60
            }
        except Exception as e:
            logger.error(f"Error calculating time difference: {str(e)}")
            return {
                "total_seconds": None,
                "days": None,
                "hours": None,
                "minutes": None,
                "seconds": None
            }

    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get the version history of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[Dict[str, Any]]: List of model metadata for all versions, with additional history information.

        Raises:
            ValueError: If no models with the given name are found in the registry.
        """
        # Get all versions of the model
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            raise ValueError(f"No models with name {model_name} found in registry")
        
        # Sort by creation time
        versions.sort(key=lambda x: x.get("created_at", ""))
        
        # Add history information
        history = []
        prev_version = None
        
        for i, version in enumerate(versions):
            history_entry = {
                "id": version.get("id"),
                "version": version.get("version"),
                "created_at": version.get("created_at"),
                "status": version.get("status"),
                "metrics": version.get("metrics", {}),
                "tags": version.get("tags", []),
                "description": version.get("description", ""),
                "is_first_version": i == 0,
                "is_latest_version": i == len(versions) - 1
            }
            
            # Add comparison with previous version if not first version
            if prev_version is not None:
                # Compare metrics
                metrics_prev = prev_version.get("metrics", {})
                metrics_curr = version.get("metrics", {})
                
                metric_changes = {}
                all_metrics = set(metrics_prev.keys()) | set(metrics_curr.keys())
                
                for metric in all_metrics:
                    value_prev = metrics_prev.get(metric, None)
                    value_curr = metrics_curr.get(metric, None)
                    
                    if value_prev is not None and value_curr is not None:
                        diff = value_curr - value_prev
                        pct_change = (diff / value_prev) * 100 if value_prev != 0 else float('inf')
                        metric_changes[metric] = {
                            "previous": value_prev,
                            "current": value_curr,
                            "difference": diff,
                            "percent_change": pct_change
                        }
                    else:
                        metric_changes[metric] = {
                            "previous": value_prev,
                            "current": value_curr,
                            "difference": None,
                            "percent_change": None
                        }
                
                history_entry["metric_changes"] = metric_changes
                
                # Calculate time since previous version
                history_entry["time_since_previous"] = self._calculate_time_difference(
                    prev_version.get("created_at"), version.get("created_at")
                )
            
            history.append(history_entry)
            prev_version = version
        
        return history

    def get_latest_version(self, model_name: str) -> Dict[str, Any]:
        """Get the latest version of a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dict[str, Any]: Metadata for the latest version.

        Raises:
            ValueError: If no models with the given name are found in the registry.
        """
        # Get all versions of the model
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            raise ValueError(f"No models with name {model_name} found in registry")
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return versions[0]

    def create_version_diff(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Create a diff between two versions of a model.

        Args:
            model_name: Name of the model.
            version1: First version to compare.
            version2: Second version to compare.

        Returns:
            Dict[str, Any]: Diff results.

        Raises:
            ValueError: If either version is not found in the registry.
        """
        # Get model IDs for both versions
        model_id1 = None
        model_id2 = None
        
        for model_id, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name:
                if metadata.get("version") == version1:
                    model_id1 = model_id
                elif metadata.get("version") == version2:
                    model_id2 = model_id
        
        if model_id1 is None:
            raise ValueError(f"Model with name {model_name} and version {version1} not found in registry")
        if model_id2 is None:
            raise ValueError(f"Model with name {model_name} and version {version2} not found in registry")
        
        # Get metadata for both versions
        metadata1 = self.registry.get_model_metadata(model_id1)
        metadata2 = self.registry.get_model_metadata(model_id2)
        
        # Convert metadata to JSON strings for diffing
        json1 = json.dumps(metadata1, sort_keys=True, indent=2)
        json2 = json.dumps(metadata2, sort_keys=True, indent=2)
        
        # Create diff
        diff = difflib.unified_diff(
            json1.splitlines(),
            json2.splitlines(),
            fromfile=f"{model_name} {version1}",
            tofile=f"{model_name} {version2}",
            lineterm=""
        )
        
        return {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "diff": "\n".join(diff)
        }

    def tag_version(self, model_name: str, version: str, tag: str) -> None:
        """Add a tag to a model version.

        Args:
            model_name: Name of the model.
            version: Version of the model.
            tag: Tag to add.

        Raises:
            ValueError: If the model version is not found in the registry.
        """
        # Get model ID
        model_id = None
        
        for mid, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name and metadata.get("version") == version:
                model_id = mid
                break
        
        if model_id is None:
            raise ValueError(f"Model with name {model_name} and version {version} not found in registry")
        
        # Get current tags
        metadata = self.registry.get_model_metadata(model_id)
        tags = metadata.get("tags", [])
        
        # Add tag if not already present
        if tag not in tags:
            tags.append(tag)
            
            # Update metadata
            self.registry.update_model_metadata(model_id, {"tags": tags})
            
            logger.info(f"Added tag '{tag}' to model {model_name} version {version}")

    def untag_version(self, model_name: str, version: str, tag: str) -> None:
        """Remove a tag from a model version.

        Args:
            model_name: Name of the model.
            version: Version of the model.
            tag: Tag to remove.

        Raises:
            ValueError: If the model version is not found in the registry.
        """
        # Get model ID
        model_id = None
        
        for mid, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name and metadata.get("version") == version:
                model_id = mid
                break
        
        if model_id is None:
            raise ValueError(f"Model with name {model_name} and version {version} not found in registry")
        
        # Get current tags
        metadata = self.registry.get_model_metadata(model_id)
        tags = metadata.get("tags", [])
        
        # Remove tag if present
        if tag in tags:
            tags.remove(tag)
            
            # Update metadata
            self.registry.update_model_metadata(model_id, {"tags": tags})
            
            logger.info(f"Removed tag '{tag}' from model {model_name} version {version}")

    def get_versions_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get all model versions with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List[Dict[str, Any]]: List of model metadata for versions with the tag.
        """
        versions = []
        
        for model_id, metadata in self.registry.models_metadata.get("models", {}).items():
            if tag in metadata.get("tags", []):
                # Add model ID to metadata
                model_info = metadata.copy()
                model_info["id"] = model_id
                versions.append(model_info)
        
        return versions