"""Model Operations for Friday AI Trading System.

This module provides operations for loading, retrieving, and managing machine learning models.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
import joblib

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry

# Create logger
logger = get_logger(__name__)


class ModelOperations:
    """Operations for managing machine learning models.

    This class provides high-level operations for working with models in the registry.

    Attributes:
        registry: The model registry instance.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize the model operations.

        Args:
            registry: The model registry. If None, a new one will be created.
        """
        self.registry = registry or ModelRegistry()
        logger.info("Initialized ModelOperations")

    def load_model(self, model_id: str) -> Any:
        """Load a model from the registry.

        Args:
            model_id: ID of the model to load.

        Returns:
            Any: The loaded model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        return self.registry.load_model(model_id)

    def get_model_by_name_and_version(self, model_name: str, version: str) -> Tuple[str, Any]:
        """Get a model by name and version.

        Args:
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            Tuple[str, Any]: The model ID and the loaded model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Find model ID by name and version
        model_id = None
        for mid, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name and metadata.get("version") == version:
                model_id = mid
                break
        
        if model_id is None:
            raise ValueError(f"Model with name {model_name} and version {version} not found in registry")
        
        # Load and return model
        model = self.registry.load_model(model_id)
        return model_id, model

    def get_latest_model_by_name(self, model_name: str) -> Tuple[str, Any]:
        """Get the latest version of a model by name.

        Args:
            model_name: Name of the model.

        Returns:
            Tuple[str, Any]: The model ID and the loaded model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Find all versions of the model
        model_versions = []
        for mid, metadata in self.registry.models_metadata.get("models", {}).items():
            if metadata.get("name") == model_name:
                model_versions.append((mid, metadata.get("version"), metadata.get("created_at")))
        
        if not model_versions:
            raise ValueError(f"No models with name {model_name} found in registry")
        
        # Sort by created_at timestamp (newest first)
        model_versions.sort(key=lambda x: x[2], reverse=True)
        
        # Get latest model ID
        latest_model_id = model_versions[0][0]
        
        # Load and return model
        model = self.registry.load_model(latest_model_id)
        return latest_model_id, model

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get metadata for a model.

        Args:
            model_id: ID of the model.

        Returns:
            Dict[str, Any]: The model metadata.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        return self.registry.get_model_metadata(model_id)

    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a model.

        Args:
            model_id: ID of the model.
            metadata: New metadata to update.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        self.registry.update_model_metadata(model_id, metadata)

    def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry.

        Args:
            model_id: ID of the model to delete.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        self.registry.delete_model(model_id)

    def list_models(self, name_filter: Optional[str] = None, tag_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List models in the registry.

        Args:
            name_filter: Optional filter by model name.
            tag_filter: Optional filter by tags.

        Returns:
            List[Dict[str, Any]]: List of model metadata.
        """
        return self.registry.list_models(name_filter, tag_filter)

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[Dict[str, Any]]: List of model metadata for all versions.
        """
        return self.registry.get_model_versions(model_name)

    def set_model_status(self, model_id: str, status: str) -> None:
        """Set the status of a model.

        Args:
            model_id: ID of the model.
            status: New status for the model (e.g., 'active', 'archived', 'deprecated').

        Raises:
            ValueError: If the model is not found in the registry.
        """
        self.registry.set_model_status(model_id, status)

    def export_model(self, model_id: str, export_dir: str) -> str:
        """Export a model to a directory.

        Args:
            model_id: ID of the model to export.
            export_dir: Directory to export the model to.

        Returns:
            str: Path to the exported model file.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        return self.registry.export_model(model_id, export_dir)

    def import_model(self, model_file: str, metadata_file: Optional[str] = None) -> str:
        """Import a model from a file.

        Args:
            model_file: Path to the model file to import.
            metadata_file: Optional path to the metadata file.

        Returns:
            str: The model ID.

        Raises:
            ValueError: If the model file does not exist.
        """
        return self.registry.import_model(model_file, metadata_file)

    def compare_models(self, model_id_1: str, model_id_2: str) -> Dict[str, Any]:
        """Compare two models.

        Args:
            model_id_1: ID of the first model.
            model_id_2: ID of the second model.

        Returns:
            Dict[str, Any]: Comparison results.

        Raises:
            ValueError: If either model is not found in the registry.
        """
        # Get metadata for both models
        metadata_1 = self.registry.get_model_metadata(model_id_1)
        metadata_2 = self.registry.get_model_metadata(model_id_2)
        
        # Compare metrics
        metrics_1 = metadata_1.get("metrics", {})
        metrics_2 = metadata_2.get("metrics", {})
        
        metric_comparison = {}
        all_metrics = set(metrics_1.keys()) | set(metrics_2.keys())
        
        for metric in all_metrics:
            value_1 = metrics_1.get(metric, None)
            value_2 = metrics_2.get(metric, None)
            
            if value_1 is not None and value_2 is not None:
                diff = value_2 - value_1
                pct_change = (diff / value_1) * 100 if value_1 != 0 else float('inf')
                metric_comparison[metric] = {
                    "model_1": value_1,
                    "model_2": value_2,
                    "difference": diff,
                    "percent_change": pct_change
                }
            else:
                metric_comparison[metric] = {
                    "model_1": value_1,
                    "model_2": value_2,
                    "difference": None,
                    "percent_change": None
                }
        
        # Compare other metadata
        comparison = {
            "model_1": {
                "id": model_id_1,
                "name": metadata_1.get("name"),
                "version": metadata_1.get("version"),
                "type": metadata_1.get("type"),
                "created_at": metadata_1.get("created_at")
            },
            "model_2": {
                "id": model_id_2,
                "name": metadata_2.get("name"),
                "version": metadata_2.get("version"),
                "type": metadata_2.get("type"),
                "created_at": metadata_2.get("created_at")
            },
            "metrics_comparison": metric_comparison
        }
        
        return comparison

    def find_best_model(self, model_name: str, metric: str, higher_is_better: bool = True) -> Tuple[str, Any]:
        """Find the best model by a specific metric.

        Args:
            model_name: Name of the model.
            metric: Metric to compare models by.
            higher_is_better: Whether higher values of the metric are better.

        Returns:
            Tuple[str, Any]: The model ID and the loaded model.

        Raises:
            ValueError: If no models with the given name are found in the registry.
        """
        # Get all versions of the model
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            raise ValueError(f"No models with name {model_name} found in registry")
        
        # Filter versions that have the metric
        versions_with_metric = [v for v in versions if metric in v.get("metrics", {})]
        
        if not versions_with_metric:
            raise ValueError(f"No models with name {model_name} have metric {metric}")
        
        # Sort by metric
        if higher_is_better:
            versions_with_metric.sort(key=lambda x: x.get("metrics", {}).get(metric, float('-inf')), reverse=True)
        else:
            versions_with_metric.sort(key=lambda x: x.get("metrics", {}).get(metric, float('inf')))
        
        # Get best model ID
        best_model_id = versions_with_metric[0].get("id")
        
        # Load and return model
        model = self.registry.load_model(best_model_id)
        return best_model_id, model