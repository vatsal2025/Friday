"""Extensions for the Model Registry to support deep learning and classification models.

This module extends the core ModelRegistry functionality to handle deep learning models
(PyTorch, TensorFlow, Keras) and classification models, integrating with the ensemble methods
and pattern recognition systems.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
import os
import uuid
from datetime import datetime
import json

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_registry_config import ModelType, PredictionTarget, SerializationFormat
from src.services.model.deep_learning_serialization import DeepLearningSerializer
from src.services.model.classification_models import ClassificationType, ClassificationMetrics
from src.services.model.pattern_recognition import PatternRecognitionSystem
from src.services.model.ensemble_methods import EnsembleModel, EnsembleMethod, EnsembleModelTrainer

# Create logger
logger = get_logger(__name__)


class DeepLearningModelRegistry:
    """Extension of ModelRegistry for deep learning models."""
    
    def __init__(self, base_registry: ModelRegistry):
        """Initialize the deep learning model registry extension.
        
        Args:
            base_registry: The base ModelRegistry instance to extend.
        """
        self.registry = base_registry
        self.dl_serializer = DeepLearningSerializer()
        
        # Register the deep learning serializer with the base registry
        self.registry.register_custom_serializer(
            SerializationFormat.ONNX, self.dl_serializer
        )
        self.registry.register_custom_serializer(
            SerializationFormat.CUSTOM, self.dl_serializer
        )
        
        logger.info("Initialized DeepLearningModelRegistry extension")
    
    def register_pytorch_model(self, 
                              model: Any, 
                              model_type: ModelType,
                              prediction_target: PredictionTarget,
                              metadata: Optional[Dict[str, Any]] = None,
                              tags: Optional[List[str]] = None,
                              save_format: str = "pt",
                              include_architecture: bool = True) -> str:
        """Register a PyTorch model in the registry.
        
        Args:
            model: The PyTorch model to register.
            model_type: The type of the model.
            prediction_target: The prediction target of the model.
            metadata: Optional metadata to associate with the model.
            tags: Optional tags to associate with the model.
            save_format: Format to save the model in ('pt', 'onnx', or 'script').
            include_architecture: Whether to save the model architecture.
            
        Returns:
            str: The ID of the registered model.
        """
        # Validate the model is a PyTorch model
        try:
            import torch
            if not isinstance(model, torch.nn.Module):
                raise ValueError("Model is not a PyTorch nn.Module instance")
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it to use this feature.")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "framework": "pytorch",
            "save_format": save_format,
            "include_architecture": include_architecture,
            "pytorch_version": torch.__version__
        })
        
        # Add architecture information if requested
        if include_architecture:
            try:
                # Get model summary as string
                from io import StringIO
                import sys
                
                # Redirect stdout to capture summary
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                # Print model architecture
                print(model)
                
                # Restore stdout and get the captured output
                sys.stdout = old_stdout
                model_summary = mystdout.getvalue()
                
                metadata["architecture_summary"] = model_summary
            except Exception as e:
                logger.warning(f"Could not capture model architecture: {str(e)}")
        
        # Add default tags if none provided
        if tags is None:
            tags = []
        
        tags.extend(["pytorch", "deep_learning"])
        
        # Register the model using the deep learning serializer
        model_id = self.registry.register_model(
            model=model,
            model_type=model_type,
            prediction_target=prediction_target,
            metadata=metadata,
            tags=tags,
            serialization_format=SerializationFormat.CUSTOM
        )
        
        logger.info(f"Registered PyTorch model with ID {model_id}")
        return model_id
    
    def register_tensorflow_model(self,
                                 model: Any,
                                 model_type: ModelType,
                                 prediction_target: PredictionTarget,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None,
                                 save_format: str = "tf",
                                 include_architecture: bool = True) -> str:
        """Register a TensorFlow model in the registry.
        
        Args:
            model: The TensorFlow model to register.
            model_type: The type of the model.
            prediction_target: The prediction target of the model.
            metadata: Optional metadata to associate with the model.
            tags: Optional tags to associate with the model.
            save_format: Format to save the model in ('tf', 'h5', 'onnx', or 'saved_model').
            include_architecture: Whether to save the model architecture.
            
        Returns:
            str: The ID of the registered model.
        """
        # Validate the model is a TensorFlow model
        try:
            import tensorflow as tf
            if not isinstance(model, (tf.keras.Model, tf.Module)):
                raise ValueError("Model is not a TensorFlow/Keras model instance")
        except ImportError:
            raise ImportError("TensorFlow is not installed. Please install it to use this feature.")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "framework": "tensorflow",
            "save_format": save_format,
            "include_architecture": include_architecture,
            "tensorflow_version": tf.__version__
        })
        
        # Add architecture information if requested
        if include_architecture and isinstance(model, tf.keras.Model):
            try:
                # Get model summary as string
                from io import StringIO
                import sys
                
                # Redirect stdout to capture summary
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                # Print model architecture
                model.summary()
                
                # Restore stdout and get the captured output
                sys.stdout = old_stdout
                model_summary = mystdout.getvalue()
                
                metadata["architecture_summary"] = model_summary
            except Exception as e:
                logger.warning(f"Could not capture model architecture: {str(e)}")
        
        # Add default tags if none provided
        if tags is None:
            tags = []
        
        tags.extend(["tensorflow", "deep_learning"])
        
        # Register the model using the deep learning serializer
        model_id = self.registry.register_model(
            model=model,
            model_type=model_type,
            prediction_target=prediction_target,
            metadata=metadata,
            tags=tags,
            serialization_format=SerializationFormat.CUSTOM
        )
        
        logger.info(f"Registered TensorFlow model with ID {model_id}")
        return model_id
    
    def load_deep_learning_model(self, model_id: str) -> Any:
        """Load a deep learning model from the registry.
        
        Args:
            model_id: The ID of the model to load.
            
        Returns:
            Any: The loaded deep learning model.
            
        Raises:
            ValueError: If the model is not a deep learning model or cannot be loaded.
        """
        # Load model metadata first to check framework
        metadata = self.registry.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        framework = metadata.get("framework")
        if framework not in ["pytorch", "tensorflow"]:
            raise ValueError(f"Model with ID {model_id} is not a deep learning model")
        
        # Load the model using the base registry
        model = self.registry.load_model(model_id)
        
        logger.info(f"Loaded {framework} model with ID {model_id}")
        return model
    
    def convert_to_onnx(self, model_id: str, input_shape: Optional[Tuple] = None) -> str:
        """Convert a deep learning model to ONNX format and register it.
        
        Args:
            model_id: The ID of the model to convert.
            input_shape: Optional input shape for the model.
            
        Returns:
            str: The ID of the registered ONNX model.
            
        Raises:
            ValueError: If the model cannot be converted to ONNX.
        """
        # Load model metadata first to check framework
        metadata = self.registry.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        framework = metadata.get("framework")
        if framework not in ["pytorch", "tensorflow"]:
            raise ValueError(f"Model with ID {model_id} is not a deep learning model")
        
        # Load the original model
        model = self.registry.load_model(model_id)
        
        # Convert to ONNX
        try:
            import onnx
            
            # Create a new metadata dict for the ONNX model
            onnx_metadata = metadata.copy()
            onnx_metadata.update({
                "original_model_id": model_id,
                "original_framework": framework,
                "framework": "onnx",
                "onnx_version": onnx.__version__,
                "converted_at": datetime.now().isoformat()
            })
            
            # Convert the model to ONNX format
            if framework == "pytorch":
                onnx_model = self.dl_serializer.convert_pytorch_to_onnx(model, input_shape)
            else:  # tensorflow
                onnx_model = self.dl_serializer.convert_tensorflow_to_onnx(model, input_shape)
            
            # Register the ONNX model
            tags = metadata.get("tags", [])
            if "onnx" not in tags:
                tags.append("onnx")
            
            onnx_model_id = self.registry.register_model(
                model=onnx_model,
                model_type=ModelType[metadata.get("model_type")],
                prediction_target=PredictionTarget[metadata.get("prediction_target")],
                metadata=onnx_metadata,
                tags=tags,
                serialization_format=SerializationFormat.ONNX
            )
            
            logger.info(f"Converted model {model_id} to ONNX format with ID {onnx_model_id}")
            return onnx_model_id
        
        except Exception as e:
            error_msg = f"Error converting model {model_id} to ONNX format: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class ClassificationModelRegistry:
    """Extension of ModelRegistry for classification models."""
    
    def __init__(self, base_registry: ModelRegistry):
        """Initialize the classification model registry extension.
        
        Args:
            base_registry: The base ModelRegistry instance to extend.
        """
        self.registry = base_registry
        self.metrics = ClassificationMetrics()
        
        logger.info("Initialized ClassificationModelRegistry extension")
    
    def register_classification_model(self,
                                     model: Any,
                                     classification_type: ClassificationType,
                                     metadata: Optional[Dict[str, Any]] = None,
                                     tags: Optional[List[str]] = None,
                                     evaluation_data: Optional[Tuple[Any, Any]] = None) -> str:
        """Register a classification model in the registry.
        
        Args:
            model: The classification model to register.
            classification_type: The type of classification.
            metadata: Optional metadata to associate with the model.
            tags: Optional tags to associate with the model.
            evaluation_data: Optional tuple of (X_test, y_test) for evaluation.
            
        Returns:
            str: The ID of the registered model.
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "classification_type": classification_type.name,
            "model_library": self._detect_model_library(model)
        })
        
        # Add default tags if none provided
        if tags is None:
            tags = []
        
        tags.extend(["classification", classification_type.name.lower()])
        
        # Evaluate the model if evaluation data is provided
        if evaluation_data is not None:
            X_test, y_test = evaluation_data
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics based on classification type
                if classification_type == ClassificationType.BINARY:
                    metrics = {
                        "accuracy": self.metrics.accuracy(y_test, y_pred),
                        "precision": self.metrics.precision(y_test, y_pred),
                        "recall": self.metrics.recall(y_test, y_pred),
                        "f1_score": self.metrics.f1_score(y_test, y_pred)
                    }
                    
                    # Try to get probability predictions for ROC AUC
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = self.metrics.roc_auc(y_test, y_prob)
                    except (AttributeError, IndexError):
                        logger.warning("Could not calculate ROC AUC: model does not support predict_proba")
                
                elif classification_type == ClassificationType.MULTICLASS:
                    metrics = {
                        "accuracy": self.metrics.accuracy(y_test, y_pred),
                        "precision_macro": self.metrics.precision(y_test, y_pred, average="macro"),
                        "recall_macro": self.metrics.recall(y_test, y_pred, average="macro"),
                        "f1_score_macro": self.metrics.f1_score(y_test, y_pred, average="macro")
                    }
                
                else:  # MULTILABEL
                    metrics = {
                        "accuracy": self.metrics.accuracy(y_test, y_pred),
                        "precision_samples": self.metrics.precision(y_test, y_pred, average="samples"),
                        "recall_samples": self.metrics.recall(y_test, y_pred, average="samples"),
                        "f1_score_samples": self.metrics.f1_score(y_test, y_pred, average="samples")
                    }
                
                # Add metrics to metadata
                metadata["evaluation_metrics"] = metrics
                
            except Exception as e:
                logger.warning(f"Could not evaluate model: {str(e)}")
        
        # Map classification type to model type and prediction target
        model_type = ModelType.CLASSIFICATION
        if classification_type == ClassificationType.BINARY:
            prediction_target = PredictionTarget.BINARY_CLASSIFICATION
        elif classification_type == ClassificationType.MULTICLASS:
            prediction_target = PredictionTarget.MULTICLASS_CLASSIFICATION
        else:  # MULTILABEL
            prediction_target = PredictionTarget.MULTILABEL_CLASSIFICATION
        
        # Register the model using the base registry
        model_id = self.registry.register_model(
            model=model,
            model_type=model_type,
            prediction_target=prediction_target,
            metadata=metadata,
            tags=tags
        )
        
        logger.info(f"Registered {classification_type.name} classification model with ID {model_id}")
        return model_id
    
    def _detect_model_library(self, model: Any) -> str:
        """Detect the library of a classification model.
        
        Args:
            model: The model to detect the library of.
            
        Returns:
            str: The detected library name.
        """
        model_class = model.__class__.__name__
        model_module = model.__class__.__module__
        
        if "sklearn" in model_module:
            return "scikit-learn"
        elif "xgboost" in model_module:
            return "xgboost"
        elif "lightgbm" in model_module:
            return "lightgbm"
        elif "catboost" in model_module:
            return "catboost"
        elif "torch" in model_module:
            return "pytorch"
        elif "tensorflow" in model_module or "keras" in model_module:
            return "tensorflow"
        else:
            return "unknown"
    
    def load_classification_model(self, model_id: str) -> Any:
        """Load a classification model from the registry.
        
        Args:
            model_id: The ID of the model to load.
            
        Returns:
            Any: The loaded classification model.
            
        Raises:
            ValueError: If the model is not a classification model.
        """
        # Load model metadata first to check model type
        metadata = self.registry.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_type = metadata.get("model_type")
        if model_type != ModelType.CLASSIFICATION.name:
            raise ValueError(f"Model with ID {model_id} is not a classification model")
        
        # Load the model using the base registry
        model = self.registry.load_model(model_id)
        
        logger.info(f"Loaded classification model with ID {model_id}")
        return model
    
    def get_classification_models(self, 
                                 classification_type: Optional[ClassificationType] = None,
                                 min_accuracy: Optional[float] = None,
                                 tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get classification models from the registry.
        
        Args:
            classification_type: Optional type of classification to filter by.
            min_accuracy: Optional minimum accuracy to filter by.
            tags: Optional tags to filter by.
            
        Returns:
            List[Dict[str, Any]]: List of model metadata dictionaries.
        """
        # Start with base filter for classification models
        filter_criteria = {
            "model_type": ModelType.CLASSIFICATION.name
        }
        
        # Add classification type filter if provided
        if classification_type is not None:
            if classification_type == ClassificationType.BINARY:
                filter_criteria["prediction_target"] = PredictionTarget.BINARY_CLASSIFICATION.name
            elif classification_type == ClassificationType.MULTICLASS:
                filter_criteria["prediction_target"] = PredictionTarget.MULTICLASS_CLASSIFICATION.name
            else:  # MULTILABEL
                filter_criteria["prediction_target"] = PredictionTarget.MULTILABEL_CLASSIFICATION.name
        
        # Get models matching the filter criteria
        models = self.registry.get_models(filter_criteria, tags)
        
        # Filter by minimum accuracy if provided
        if min_accuracy is not None:
            filtered_models = []
            for model in models:
                metrics = model.get("metadata", {}).get("evaluation_metrics", {})
                accuracy = metrics.get("accuracy")
                if accuracy is not None and accuracy >= min_accuracy:
                    filtered_models.append(model)
            models = filtered_models
        
        return models


class PatternRecognitionModelRegistry:
    """Extension of ModelRegistry for pattern recognition models."""
    
    def __init__(self, base_registry: ModelRegistry):
        """Initialize the pattern recognition model registry extension.
        
        Args:
            base_registry: The base ModelRegistry instance to extend.
        """
        self.registry = base_registry
        self.pattern_system = PatternRecognitionSystem()
        
        logger.info("Initialized PatternRecognitionModelRegistry extension")
    
    def register_pattern_detector(self,
                                 detector: Any,
                                 detector_type: str,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 tags: Optional[List[str]] = None) -> str:
        """Register a pattern detector in the registry.
        
        Args:
            detector: The pattern detector to register.
            detector_type: The type of the detector (e.g., 'candlestick', 'chart', 'indicator').
            metadata: Optional metadata to associate with the detector.
            tags: Optional tags to associate with the detector.
            
        Returns:
            str: The ID of the registered detector.
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "detector_type": detector_type,
            "patterns_supported": getattr(detector, "supported_patterns", [])
        })
        
        # Add default tags if none provided
        if tags is None:
            tags = []
        
        tags.extend(["pattern_detector", detector_type])
        
        # Register the detector using the base registry
        detector_id = self.registry.register_model(
            model=detector,
            model_type=ModelType.PATTERN_RECOGNITION,
            prediction_target=PredictionTarget.PATTERN,
            metadata=metadata,
            tags=tags
        )
        
        logger.info(f"Registered {detector_type} pattern detector with ID {detector_id}")
        return detector_id
    
    def load_pattern_detector(self, detector_id: str) -> Any:
        """Load a pattern detector from the registry.
        
        Args:
            detector_id: The ID of the detector to load.
            
        Returns:
            Any: The loaded pattern detector.
            
        Raises:
            ValueError: If the model is not a pattern detector.
        """
        # Load model metadata first to check model type
        metadata = self.registry.get_model_metadata(detector_id)
        if metadata is None:
            raise ValueError(f"Detector with ID {detector_id} not found in registry")
        
        model_type = metadata.get("model_type")
        if model_type != ModelType.PATTERN_RECOGNITION.name:
            raise ValueError(f"Model with ID {detector_id} is not a pattern detector")
        
        # Load the detector using the base registry
        detector = self.registry.load_model(detector_id)
        
        logger.info(f"Loaded pattern detector with ID {detector_id}")
        return detector
    
    def get_pattern_detectors(self, 
                             detector_type: Optional[str] = None,
                             pattern: Optional[str] = None,
                             tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get pattern detectors from the registry.
        
        Args:
            detector_type: Optional type of detector to filter by.
            pattern: Optional pattern name to filter by.
            tags: Optional tags to filter by.
            
        Returns:
            List[Dict[str, Any]]: List of detector metadata dictionaries.
        """
        # Start with base filter for pattern recognition models
        filter_criteria = {
            "model_type": ModelType.PATTERN_RECOGNITION.name,
            "prediction_target": PredictionTarget.PATTERN.name
        }
        
        # Add detector type filter if provided
        if detector_type is not None:
            filter_criteria["metadata.detector_type"] = detector_type
        
        # Get detectors matching the filter criteria
        detectors = self.registry.get_models(filter_criteria, tags)
        
        # Filter by pattern if provided
        if pattern is not None:
            filtered_detectors = []
            for detector in detectors:
                patterns_supported = detector.get("metadata", {}).get("patterns_supported", [])
                if pattern in patterns_supported:
                    filtered_detectors.append(detector)
            detectors = filtered_detectors
        
        return detectors


class EnsembleModelRegistry:
    """Extension of ModelRegistry for ensemble models."""
    
    def __init__(self, base_registry: ModelRegistry):
        """Initialize the ensemble model registry extension.
        
        Args:
            base_registry: The base ModelRegistry instance to extend.
        """
        self.registry = base_registry
        self.trainer = EnsembleModelTrainer(base_registry)
        
        logger.info("Initialized EnsembleModelRegistry extension")
    
    def register_ensemble(self,
                         ensemble: EnsembleModel,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> str:
        """Register an ensemble model in the registry.
        
        Args:
            ensemble: The ensemble model to register.
            metadata: Optional metadata to associate with the ensemble.
            tags: Optional tags to associate with the ensemble.
            
        Returns:
            str: The ID of the registered ensemble.
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Combine with ensemble metadata
        metadata.update(ensemble.metadata)
        metadata.update({
            "ensemble_method": ensemble.ensemble_method.name,
            "weighting_strategy": ensemble.weighting_strategy.name,
            "model_count": len(ensemble.models)
        })
        
        # Add model IDs if available
        model_ids = []
        for model_info in ensemble.models:
            if "model_id" in model_info and model_info["model_id"] is not None:
                model_ids.append(model_info["model_id"])
        
        if model_ids:
            metadata["component_model_ids"] = model_ids
        
        # Add default tags if none provided
        if tags is None:
            tags = []
        
        tags.extend(["ensemble", ensemble.ensemble_method.name.lower()])
        
        # Register the ensemble using the base registry
        ensemble_id = self.registry.register_model(
            model=ensemble,
            model_type=ensemble.model_type,
            prediction_target=ensemble.prediction_target,
            metadata=metadata,
            tags=tags
        )
        
        logger.info(f"Registered {ensemble.ensemble_method.name} ensemble with ID {ensemble_id}")
        return ensemble_id
    
    def load_ensemble(self, ensemble_id: str) -> EnsembleModel:
        """Load an ensemble model from the registry.
        
        Args:
            ensemble_id: The ID of the ensemble to load.
            
        Returns:
            EnsembleModel: The loaded ensemble model.
            
        Raises:
            ValueError: If the model is not an ensemble.
        """
        # Load model metadata first to check if it's an ensemble
        metadata = self.registry.get_model_metadata(ensemble_id)
        if metadata is None:
            raise ValueError(f"Ensemble with ID {ensemble_id} not found in registry")
        
        if "ensemble_method" not in metadata:
            raise ValueError(f"Model with ID {ensemble_id} is not an ensemble")
        
        # Load the ensemble using the base registry
        ensemble = self.registry.load_model(ensemble_id)
        
        logger.info(f"Loaded {metadata['ensemble_method']} ensemble with ID {ensemble_id}")
        return ensemble
    
    def create_ensemble_from_models(self,
                                   name: str,
                                   ensemble_method: EnsembleMethod,
                                   model_ids: List[str],
                                   weights: Optional[List[float]] = None,
                                   meta_model_id: Optional[str] = None) -> str:
        """Create and register an ensemble from existing models in the registry.
        
        Args:
            name: Name of the ensemble model.
            ensemble_method: Method used for ensembling.
            model_ids: List of model IDs to include in the ensemble.
            weights: Optional list of weights for the models.
            meta_model_id: Optional ID of a meta-model for stacking ensemble.
            
        Returns:
            str: The ID of the registered ensemble.
            
        Raises:
            ValueError: If the models cannot be loaded or are incompatible.
        """
        # Load all models
        models = []
        model_types = set()
        prediction_targets = set()
        
        for model_id in model_ids:
            try:
                model = self.registry.load_model(model_id)
                metadata = self.registry.get_model_metadata(model_id)
                
                models.append(model)
                model_types.add(ModelType[metadata.get("model_type")])
                prediction_targets.add(PredictionTarget[metadata.get("prediction_target")])
            
            except Exception as e:
                error_msg = f"Error loading model {model_id}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Check if models are compatible
        if len(model_types) > 1:
            logger.warning(f"Ensemble contains models of different types: {model_types}")
        
        if len(prediction_targets) > 1:
            logger.warning(f"Ensemble contains models with different prediction targets: {prediction_targets}")
        
        # Determine model type and prediction target for the ensemble
        model_type = list(model_types)[0] if len(model_types) == 1 else ModelType.ENSEMBLE
        prediction_target = list(prediction_targets)[0] if len(prediction_targets) == 1 else None
        
        if prediction_target is None:
            # Try to infer a common prediction target
            if all(pt in [PredictionTarget.BINARY_CLASSIFICATION, 
                          PredictionTarget.MULTICLASS_CLASSIFICATION, 
                          PredictionTarget.MULTILABEL_CLASSIFICATION] 
                   for pt in prediction_targets):
                prediction_target = PredictionTarget.BINARY_CLASSIFICATION
            else:
                # Default to the first one
                prediction_target = list(prediction_targets)[0]
        
        # Create the ensemble based on the method
        if ensemble_method == EnsembleMethod.VOTING:
            # Determine if this is a classification ensemble
            is_classification = any(pt in [PredictionTarget.BINARY_CLASSIFICATION, 
                                         PredictionTarget.MULTICLASS_CLASSIFICATION, 
                                         PredictionTarget.MULTILABEL_CLASSIFICATION] 
                                  for pt in prediction_targets)
            
            ensemble = self.trainer.create_voting_ensemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                models=models,
                model_ids=model_ids,
                weights=weights,
                is_classification=is_classification
            )
        
        elif ensemble_method == EnsembleMethod.STACKING:
            if meta_model_id is None:
                raise ValueError("meta_model_id is required for stacking ensemble")
            
            # Load meta-model
            try:
                meta_model = self.registry.load_model(meta_model_id)
            except Exception as e:
                error_msg = f"Error loading meta-model {meta_model_id}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            ensemble = self.trainer.create_stacking_ensemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                base_models=models,
                meta_model=meta_model,
                model_ids=model_ids
            )
        
        elif ensemble_method == EnsembleMethod.WEIGHTED:
            ensemble = self.trainer.create_weighted_ensemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                models=models,
                model_ids=model_ids
            )
        
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
        
        # Register the ensemble
        ensemble_id = self.register_ensemble(
            ensemble=ensemble,
            metadata={
                "created_from_models": model_ids,
                "meta_model_id": meta_model_id if ensemble_method == EnsembleMethod.STACKING else None
            }
        )
        
        return ensemble_id
    
    def get_ensembles(self,
                     ensemble_method: Optional[EnsembleMethod] = None,
                     model_type: Optional[ModelType] = None,
                     prediction_target: Optional[PredictionTarget] = None,
                     tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ensemble models from the registry.
        
        Args:
            ensemble_method: Optional ensemble method to filter by.
            model_type: Optional model type to filter by.
            prediction_target: Optional prediction target to filter by.
            tags: Optional tags to filter by.
            
        Returns:
            List[Dict[str, Any]]: List of ensemble metadata dictionaries.
        """
        # Start with base filter for ensembles
        filter_criteria = {}
        
        # Add ensemble method filter if provided
        if ensemble_method is not None:
            filter_criteria["metadata.ensemble_method"] = ensemble_method.name
        else:
            # Ensure we only get ensembles
            if "ensemble" not in (tags or []):
                if tags is None:
                    tags = ["ensemble"]
                else:
                    tags.append("ensemble")
        
        # Add model type filter if provided
        if model_type is not None:
            filter_criteria["model_type"] = model_type.name
        
        # Add prediction target filter if provided
        if prediction_target is not None:
            filter_criteria["prediction_target"] = prediction_target.name
        
        # Get ensembles matching the filter criteria
        ensembles = self.registry.get_models(filter_criteria, tags)
        
        return ensembles


class ExtendedModelRegistry:
    """Comprehensive extension of ModelRegistry with all specialized registry extensions."""
    
    def __init__(self, base_registry: Optional[ModelRegistry] = None):
        """Initialize the extended model registry.
        
        Args:
            base_registry: Optional base ModelRegistry instance to extend.
                If not provided, a new ModelRegistry instance will be created.
        """
        self.registry = base_registry or ModelRegistry()
        
        # Initialize all extensions
        self.deep_learning = DeepLearningModelRegistry(self.registry)
        self.classification = ClassificationModelRegistry(self.registry)
        self.pattern_recognition = PatternRecognitionModelRegistry(self.registry)
        self.ensemble = EnsembleModelRegistry(self.registry)
        
        logger.info("Initialized ExtendedModelRegistry with all extensions")
    
    def get_base_registry(self) -> ModelRegistry:
        """Get the base ModelRegistry instance.
        
        Returns:
            ModelRegistry: The base ModelRegistry instance.
        """
        return self.registry