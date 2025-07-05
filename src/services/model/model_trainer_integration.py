"""Model Trainer Integration for Friday AI Trading System.

This module provides integration between the model trainer and the model registry,
allowing trained models to be automatically registered and versioned.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_serialization import ModelSerializer

# Create logger
logger = get_logger(__name__)


class ModelTrainerIntegration:
    """Integration between model trainer and model registry.

    This class provides functionality for integrating the model trainer with the model registry,
    allowing trained models to be automatically registered and versioned.

    Attributes:
        registry: The model registry instance.
        serializer: The model serializer instance.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        serializer: Optional[ModelSerializer] = None,
    ):
        """Initialize the model trainer integration.

        Args:
            registry: The model registry. If None, a new one will be created.
            serializer: The model serializer. If None, a new one will be created.
        """
        self.registry = registry or ModelRegistry()
        self.serializer = serializer or ModelSerializer()
        logger.info("Initialized ModelTrainerIntegration")

    def register_trained_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Register a trained model in the registry.

        Args:
            model: The trained model object.
            model_name: Name of the model.
            model_type: Type of the model (e.g., 'random_forest', 'gradient_boosting').
            metrics: Performance metrics for the model.
            metadata: Additional metadata for the model.
            tags: Tags for the model.
            description: Description of the model.
            version: Version of the model. If None, a new version will be generated.

        Returns:
            str: The model ID.
        """
        # Generate version if not provided
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add model hash to metadata
        if metadata is None:
            metadata = {}
        metadata["model_hash"] = self.serializer.compute_model_hash(model)
        
        # Register model
        model_id = self.registry.register_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            version=version,
            metadata=metadata,
            metrics=metrics,
            tags=tags,
            description=description
        )
        
        logger.info(f"Registered trained model {model_name} with version {version} and ID {model_id}")
        
        return model_id

    def register_model_from_file(
        self,
        model_file: str,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Register a model from a file in the registry.

        Args:
            model_file: Path to the model file.
            model_name: Name of the model.
            model_type: Type of the model (e.g., 'random_forest', 'gradient_boosting').
            metrics: Performance metrics for the model.
            metadata: Additional metadata for the model.
            tags: Tags for the model.
            description: Description of the model.
            version: Version of the model. If None, a new version will be generated.

        Returns:
            str: The model ID.

        Raises:
            ValueError: If the model file does not exist.
        """
        # Check if model file exists
        if not os.path.exists(model_file):
            raise ValueError(f"Model file {model_file} not found")
        
        # Load model
        model = self.serializer.deserialize(model_file)
        
        # Add file info to metadata
        if metadata is None:
            metadata = {}
        metadata["original_file"] = model_file
        metadata["file_info"] = self.serializer.get_model_info(model_file)
        
        # Register model
        return self.register_trained_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            metadata=metadata,
            tags=tags,
            description=description,
            version=version
        )

    def register_evaluation_results(
        self,
        model_id: str,
        evaluation_results: Dict[str, Any],
    ) -> None:
        """Register evaluation results for a model.

        Args:
            model_id: ID of the model.
            evaluation_results: Evaluation results for the model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with evaluation results
        metadata["evaluation_results"] = evaluation_results
        
        # Update metrics if present in evaluation results
        if "metrics" in evaluation_results:
            metadata["metrics"].update(evaluation_results["metrics"])
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered evaluation results for model {model_id}")

    def register_training_data_info(
        self,
        model_id: str,
        data_info: Dict[str, Any],
    ) -> None:
        """Register information about the training data for a model.

        Args:
            model_id: ID of the model.
            data_info: Information about the training data.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with training data info
        if "additional_metadata" not in metadata:
            metadata["additional_metadata"] = {}
        metadata["additional_metadata"]["training_data"] = data_info
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered training data info for model {model_id}")

    def register_hyperparameters(
        self,
        model_id: str,
        hyperparameters: Dict[str, Any],
    ) -> None:
        """Register hyperparameters for a model.

        Args:
            model_id: ID of the model.
            hyperparameters: Hyperparameters for the model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with hyperparameters
        if "additional_metadata" not in metadata:
            metadata["additional_metadata"] = {}
        metadata["additional_metadata"]["hyperparameters"] = hyperparameters
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered hyperparameters for model {model_id}")

    def register_feature_importance(
        self,
        model_id: str,
        feature_importance: Dict[str, float],
    ) -> None:
        """Register feature importance for a model.

        Args:
            model_id: ID of the model.
            feature_importance: Feature importance for the model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with feature importance
        if "additional_metadata" not in metadata:
            metadata["additional_metadata"] = {}
        metadata["additional_metadata"]["feature_importance"] = feature_importance
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered feature importance for model {model_id}")

    def register_model_pipeline(
        self,
        model_id: str,
        pipeline_steps: List[Dict[str, Any]],
    ) -> None:
        """Register model pipeline steps for a model.

        Args:
            model_id: ID of the model.
            pipeline_steps: Pipeline steps for the model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with pipeline steps
        if "additional_metadata" not in metadata:
            metadata["additional_metadata"] = {}
        metadata["additional_metadata"]["pipeline_steps"] = pipeline_steps
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered model pipeline steps for model {model_id}")

    def register_model_dependencies(
        self,
        model_id: str,
        dependencies: Dict[str, str],
    ) -> None:
        """Register model dependencies for a model.

        Args:
            model_id: ID of the model.
            dependencies: Dependencies for the model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        
        # Update metadata with dependencies
        if "additional_metadata" not in metadata:
            metadata["additional_metadata"] = {}
        metadata["additional_metadata"]["dependencies"] = dependencies
        
        # Update model metadata
        self.registry.update_model_metadata(model_id, metadata)
        
        logger.info(f"Registered model dependencies for model {model_id}")

    def register_model_from_trainer(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        evaluation_results: Dict[str, Any],
        training_data_info: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None,
        pipeline_steps: Optional[List[Dict[str, Any]]] = None,
        dependencies: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Register a model from a trainer with all associated information.

        Args:
            model: The trained model object.
            model_name: Name of the model.
            model_type: Type of the model (e.g., 'random_forest', 'gradient_boosting').
            evaluation_results: Evaluation results for the model.
            training_data_info: Information about the training data.
            hyperparameters: Hyperparameters for the model.
            feature_importance: Feature importance for the model.
            pipeline_steps: Pipeline steps for the model.
            dependencies: Dependencies for the model.
            tags: Tags for the model.
            description: Description of the model.
            version: Version of the model. If None, a new version will be generated.

        Returns:
            str: The model ID.
        """
        # Extract metrics from evaluation results
        metrics = evaluation_results.get("metrics", {})
        
        # Create metadata
        metadata = {
            "training_data": training_data_info,
            "hyperparameters": hyperparameters,
        }
        
        if feature_importance is not None:
            metadata["feature_importance"] = feature_importance
        
        if pipeline_steps is not None:
            metadata["pipeline_steps"] = pipeline_steps
        
        if dependencies is not None:
            metadata["dependencies"] = dependencies
        
        # Register model
        model_id = self.register_trained_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            metadata=metadata,
            tags=tags,
            description=description,
            version=version
        )
        
        # Register evaluation results
        self.register_evaluation_results(model_id, evaluation_results)
        
        return model_id