"""Ensemble Methods for Model Performance Optimization.

This module provides functionality for creating and managing ensemble models
to improve prediction accuracy and robustness in the Friday AI Trading System.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from enum import Enum, auto
import joblib
import os
import uuid
from datetime import datetime

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_registry_config import ModelType, PredictionTarget

# Create logger
logger = get_logger(__name__)


class EnsembleMethod(Enum):
    """Enum for different ensemble methods."""
    VOTING = auto()       # Simple voting (classification) or averaging (regression)
    BAGGING = auto()      # Bootstrap aggregating
    BOOSTING = auto()     # Sequential ensemble method
    STACKING = auto()     # Meta-model approach
    BLENDING = auto()     # Similar to stacking but with a validation set
    WEIGHTED = auto()     # Weighted average based on model performance
    CUSTOM = auto()       # Custom ensemble method


class EnsembleWeightingStrategy(Enum):
    """Enum for different strategies to weight models in an ensemble."""
    EQUAL = auto()            # Equal weights for all models
    PERFORMANCE = auto()      # Weights based on individual model performance
    OPTIMIZATION = auto()     # Weights determined through optimization
    DYNAMIC = auto()          # Weights that change based on recent performance
    CONFIDENCE = auto()       # Weights based on model confidence/uncertainty
    CUSTOM = auto()           # Custom weighting strategy


class EnsembleModel:
    """Base class for ensemble models."""
    
    def __init__(self, 
                 name: str,
                 ensemble_method: EnsembleMethod,
                 model_type: ModelType,
                 prediction_target: PredictionTarget,
                 weighting_strategy: EnsembleWeightingStrategy = EnsembleWeightingStrategy.EQUAL):
        """Initialize an ensemble model.
        
        Args:
            name: Name of the ensemble model.
            ensemble_method: Method used for ensembling.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            weighting_strategy: Strategy for weighting models in the ensemble.
        """
        self.name = name
        self.ensemble_method = ensemble_method
        self.model_type = model_type
        self.prediction_target = prediction_target
        self.weighting_strategy = weighting_strategy
        self.models = []
        self.weights = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "ensemble_id": str(uuid.uuid4()),
            "ensemble_method": ensemble_method.name,
            "model_type": model_type.name,
            "prediction_target": prediction_target.name,
            "weighting_strategy": weighting_strategy.name
        }
        
        logger.info(f"Initialized {name} ensemble model with {ensemble_method.name} method")
    
    def add_model(self, model: Any, weight: float = 1.0, model_id: Optional[str] = None) -> None:
        """Add a model to the ensemble.
        
        Args:
            model: The model to add.
            weight: Weight of the model in the ensemble.
            model_id: Optional ID of the model.
        """
        model_info = {
            "model": model,
            "weight": weight,
            "model_id": model_id or str(uuid.uuid4()),
            "added_at": datetime.now().isoformat()
        }
        self.models.append(model_info)
        self.weights.append(weight)
        self._normalize_weights()
        
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["model_count"] = len(self.models)
        
        logger.info(f"Added model {model_info['model_id']} to {self.name} ensemble with weight {weight}")
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the ensemble.
        
        Args:
            model_id: ID of the model to remove.
            
        Returns:
            bool: True if the model was removed, False otherwise.
        """
        for i, model_info in enumerate(self.models):
            if model_info["model_id"] == model_id:
                self.models.pop(i)
                self.weights.pop(i)
                self._normalize_weights()
                
                self.metadata["updated_at"] = datetime.now().isoformat()
                self.metadata["model_count"] = len(self.models)
                
                logger.info(f"Removed model {model_id} from {self.name} ensemble")
                return True
        
        logger.warning(f"Model {model_id} not found in {self.name} ensemble")
        return False
    
    def update_weight(self, model_id: str, weight: float) -> bool:
        """Update the weight of a model in the ensemble.
        
        Args:
            model_id: ID of the model to update.
            weight: New weight of the model.
            
        Returns:
            bool: True if the weight was updated, False otherwise.
        """
        for i, model_info in enumerate(self.models):
            if model_info["model_id"] == model_id:
                model_info["weight"] = weight
                self.weights[i] = weight
                self._normalize_weights()
                
                self.metadata["updated_at"] = datetime.now().isoformat()
                
                logger.info(f"Updated weight of model {model_id} to {weight} in {self.name} ensemble")
                return True
        
        logger.warning(f"Model {model_id} not found in {self.name} ensemble")
        return False
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        if self.weights:
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]
                for i, model_info in enumerate(self.models):
                    model_info["weight"] = self.weights[i]
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions with the ensemble model.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Ensemble predictions.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the predict method")
    
    def save(self, path: str) -> str:
        """Save the ensemble model to disk.
        
        Args:
            path: Directory path to save the model.
            
        Returns:
            str: Path to the saved model file.
        """
        os.makedirs(path, exist_ok=True)
        
        # Create a serializable version of the model info
        serializable_models = []
        for model_info in self.models:
            serializable_info = {
                "weight": model_info["weight"],
                "model_id": model_info["model_id"],
                "added_at": model_info["added_at"]
            }
            serializable_models.append(serializable_info)
        
        # Create a serializable ensemble object
        ensemble_data = {
            "name": self.name,
            "ensemble_method": self.ensemble_method.name,
            "model_type": self.model_type.name,
            "prediction_target": self.prediction_target.name,
            "weighting_strategy": self.weighting_strategy.name,
            "models": serializable_models,
            "weights": self.weights,
            "metadata": self.metadata
        }
        
        # Save the ensemble metadata
        metadata_path = os.path.join(path, f"{self.name}_metadata.joblib")
        joblib.dump(ensemble_data, metadata_path)
        
        # Save individual models if they're not already in the registry
        for i, model_info in enumerate(self.models):
            if not model_info.get("in_registry", False):
                model_path = os.path.join(path, f"{self.name}_model_{i}.joblib")
                joblib.dump(model_info["model"], model_path)
                model_info["model_path"] = model_path
        
        logger.info(f"Saved {self.name} ensemble model to {path}")
        return metadata_path
    
    @classmethod
    def load(cls, path: str, registry: Optional[ModelRegistry] = None) -> 'EnsembleModel':
        """Load an ensemble model from disk.
        
        Args:
            path: Path to the saved model metadata file.
            registry: Optional model registry to load models from.
            
        Returns:
            EnsembleModel: Loaded ensemble model.
            
        Raises:
            ValueError: If the model cannot be loaded.
        """
        try:
            # Load ensemble metadata
            ensemble_data = joblib.load(path)
            
            # Create ensemble instance
            ensemble = cls(
                name=ensemble_data["name"],
                ensemble_method=EnsembleMethod[ensemble_data["ensemble_method"]],
                model_type=ModelType[ensemble_data["model_type"]],
                prediction_target=PredictionTarget[ensemble_data["prediction_target"]],
                weighting_strategy=EnsembleWeightingStrategy[ensemble_data["weighting_strategy"]]
            )
            
            # Restore metadata
            ensemble.metadata = ensemble_data["metadata"]
            
            # Load individual models
            for i, model_info in enumerate(ensemble_data["models"]):
                model = None
                model_id = model_info["model_id"]
                
                # Try to load from registry first if available
                if registry is not None:
                    try:
                        model = registry.load_model(model_id)
                        model_info["in_registry"] = True
                    except Exception as e:
                        logger.warning(f"Could not load model {model_id} from registry: {str(e)}")
                
                # If not in registry or registry not available, try to load from file
                if model is None and "model_path" in model_info:
                    model_path = model_info["model_path"]
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                    else:
                        # Try to find the model in the same directory as the metadata
                        dir_path = os.path.dirname(path)
                        model_path = os.path.join(dir_path, f"{ensemble.name}_model_{i}.joblib")
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                
                if model is not None:
                    ensemble.add_model(model, model_info["weight"], model_id)
                else:
                    logger.error(f"Could not load model {model_id} for ensemble {ensemble.name}")
            
            logger.info(f"Loaded {ensemble.name} ensemble model from {path}")
            return ensemble
        
        except Exception as e:
            error_msg = f"Error loading ensemble model from {path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class VotingEnsemble(EnsembleModel):
    """Ensemble model that uses voting or averaging for predictions."""
    
    def __init__(self, 
                 name: str,
                 model_type: ModelType,
                 prediction_target: PredictionTarget,
                 weighting_strategy: EnsembleWeightingStrategy = EnsembleWeightingStrategy.EQUAL,
                 is_classification: bool = False):
        """Initialize a voting ensemble model.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            weighting_strategy: Strategy for weighting models in the ensemble.
            is_classification: Whether this is a classification ensemble.
        """
        super().__init__(
            name=name,
            ensemble_method=EnsembleMethod.VOTING,
            model_type=model_type,
            prediction_target=prediction_target,
            weighting_strategy=weighting_strategy
        )
        self.is_classification = is_classification
        self.metadata["is_classification"] = is_classification
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions with the voting ensemble.
        
        For classification, uses weighted voting.
        For regression, uses weighted averaging.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Ensemble predictions.
            
        Raises:
            ValueError: If no models are in the ensemble.
        """
        if not self.models:
            raise ValueError(f"No models in {self.name} ensemble")
        
        predictions = []
        weights = []
        
        for model_info in self.models:
            model = model_info["model"]
            weight = model_info["weight"]
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.error(f"Error predicting with model {model_info['model_id']}: {str(e)}")
        
        if not predictions:
            raise ValueError(f"No successful predictions from models in {self.name} ensemble")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights).reshape(-1, 1)
        
        if self.is_classification:
            # For classification, get the weighted vote counts for each class
            if len(predictions.shape) > 2:  # Multi-class with probabilities
                # Weighted average of probabilities
                weighted_probs = np.sum(predictions * weights[:, np.newaxis, :], axis=0)
                return np.argmax(weighted_probs, axis=1)
            else:  # Binary or multi-class with class labels
                # Get unique classes
                unique_classes = np.unique(predictions.flatten())
                
                # Count weighted votes for each class
                class_votes = {}
                for cls in unique_classes:
                    class_votes[cls] = np.sum(weights[predictions == cls])
                
                # Get class with highest weighted votes for each sample
                result = np.zeros(predictions.shape[1])
                for i in range(predictions.shape[1]):
                    sample_votes = {}
                    for j in range(predictions.shape[0]):
                        cls = predictions[j, i]
                        if cls not in sample_votes:
                            sample_votes[cls] = 0
                        sample_votes[cls] += weights[j]
                    
                    # Get class with highest weighted votes
                    result[i] = max(sample_votes.items(), key=lambda x: x[1])[0]
                
                return result
        else:
            # For regression, compute weighted average
            return np.average(predictions, axis=0, weights=weights.flatten())


class StackingEnsemble(EnsembleModel):
    """Ensemble model that uses stacking for predictions."""
    
    def __init__(self, 
                 name: str,
                 model_type: ModelType,
                 prediction_target: PredictionTarget,
                 meta_model: Any,
                 use_features: bool = True):
        """Initialize a stacking ensemble model.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            meta_model: Model to use for combining base model predictions.
            use_features: Whether to include original features in meta-model input.
        """
        super().__init__(
            name=name,
            ensemble_method=EnsembleMethod.STACKING,
            model_type=model_type,
            prediction_target=prediction_target,
            weighting_strategy=EnsembleWeightingStrategy.CUSTOM
        )
        self.meta_model = meta_model
        self.use_features = use_features
        self.metadata["use_features"] = use_features
        self.is_fitted = False
    
    def fit(self, X: Any, y: Any) -> 'StackingEnsemble':
        """Fit the stacking ensemble.
        
        Args:
            X: Training data features.
            y: Training data targets.
            
        Returns:
            StackingEnsemble: Fitted ensemble.
            
        Raises:
            ValueError: If no models are in the ensemble.
        """
        if not self.models:
            raise ValueError(f"No models in {self.name} ensemble")
        
        # Get predictions from base models
        base_predictions = self._get_base_predictions(X)
        
        # Prepare meta-model input
        if self.use_features:
            if isinstance(X, pd.DataFrame):
                X_meta = np.hstack([base_predictions, X.values])
            else:
                X_meta = np.hstack([base_predictions, X])
        else:
            X_meta = base_predictions
        
        # Fit meta-model
        self.meta_model.fit(X_meta, y)
        self.is_fitted = True
        
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["is_fitted"] = True
        
        logger.info(f"Fitted {self.name} stacking ensemble")
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions with the stacking ensemble.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Ensemble predictions.
            
        Raises:
            ValueError: If the ensemble is not fitted or no models are in the ensemble.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} stacking ensemble is not fitted")
        
        if not self.models:
            raise ValueError(f"No models in {self.name} ensemble")
        
        # Get predictions from base models
        base_predictions = self._get_base_predictions(X)
        
        # Prepare meta-model input
        if self.use_features:
            if isinstance(X, pd.DataFrame):
                X_meta = np.hstack([base_predictions, X.values])
            else:
                X_meta = np.hstack([base_predictions, X])
        else:
            X_meta = base_predictions
        
        # Make predictions with meta-model
        return self.meta_model.predict(X_meta)
    
    def _get_base_predictions(self, X: Any) -> np.ndarray:
        """Get predictions from all base models.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Array of base model predictions.
            
        Raises:
            ValueError: If no successful predictions are made.
        """
        predictions = []
        
        for model_info in self.models:
            model = model_info["model"]
            
            try:
                pred = model.predict(X)
                # Ensure predictions are 2D
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting with model {model_info['model_id']}: {str(e)}")
        
        if not predictions:
            raise ValueError(f"No successful predictions from models in {self.name} ensemble")
        
        # Concatenate predictions horizontally
        return np.hstack(predictions)
    
    def save(self, path: str) -> str:
        """Save the stacking ensemble to disk.
        
        Args:
            path: Directory path to save the model.
            
        Returns:
            str: Path to the saved model file.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save meta-model
        meta_model_path = os.path.join(path, f"{self.name}_meta_model.joblib")
        joblib.dump(self.meta_model, meta_model_path)
        
        # Save ensemble metadata and models
        metadata_path = super().save(path)
        
        # Update metadata with meta-model path
        ensemble_data = joblib.load(metadata_path)
        ensemble_data["meta_model_path"] = meta_model_path
        ensemble_data["is_fitted"] = self.is_fitted
        joblib.dump(ensemble_data, metadata_path)
        
        return metadata_path
    
    @classmethod
    def load(cls, path: str, registry: Optional[ModelRegistry] = None) -> 'StackingEnsemble':
        """Load a stacking ensemble from disk.
        
        Args:
            path: Path to the saved model metadata file.
            registry: Optional model registry to load models from.
            
        Returns:
            StackingEnsemble: Loaded stacking ensemble.
            
        Raises:
            ValueError: If the model cannot be loaded.
        """
        try:
            # Load ensemble metadata
            ensemble_data = joblib.load(path)
            
            # Load meta-model
            meta_model_path = ensemble_data.get("meta_model_path")
            if not meta_model_path or not os.path.exists(meta_model_path):
                # Try to find meta-model in the same directory
                dir_path = os.path.dirname(path)
                meta_model_path = os.path.join(dir_path, f"{ensemble_data['name']}_meta_model.joblib")
            
            if not os.path.exists(meta_model_path):
                raise ValueError(f"Meta-model not found at {meta_model_path}")
            
            meta_model = joblib.load(meta_model_path)
            
            # Create ensemble instance
            ensemble = cls(
                name=ensemble_data["name"],
                model_type=ModelType[ensemble_data["model_type"]],
                prediction_target=PredictionTarget[ensemble_data["prediction_target"]],
                meta_model=meta_model,
                use_features=ensemble_data.get("use_features", True)
            )
            
            # Restore metadata
            ensemble.metadata = ensemble_data["metadata"]
            ensemble.is_fitted = ensemble_data.get("is_fitted", False)
            
            # Load base models
            for i, model_info in enumerate(ensemble_data["models"]):
                model = None
                model_id = model_info["model_id"]
                
                # Try to load from registry first if available
                if registry is not None:
                    try:
                        model = registry.load_model(model_id)
                        model_info["in_registry"] = True
                    except Exception as e:
                        logger.warning(f"Could not load model {model_id} from registry: {str(e)}")
                
                # If not in registry or registry not available, try to load from file
                if model is None and "model_path" in model_info:
                    model_path = model_info["model_path"]
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                    else:
                        # Try to find the model in the same directory as the metadata
                        dir_path = os.path.dirname(path)
                        model_path = os.path.join(dir_path, f"{ensemble.name}_model_{i}.joblib")
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                
                if model is not None:
                    ensemble.add_model(model, model_info["weight"], model_id)
                else:
                    logger.error(f"Could not load model {model_id} for ensemble {ensemble.name}")
            
            logger.info(f"Loaded {ensemble.name} stacking ensemble from {path}")
            return ensemble
        
        except Exception as e:
            error_msg = f"Error loading stacking ensemble from {path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class WeightedEnsemble(EnsembleModel):
    """Ensemble model that uses weighted averaging for predictions."""
    
    def __init__(self, 
                 name: str,
                 model_type: ModelType,
                 prediction_target: PredictionTarget,
                 optimization_metric: Optional[Callable] = None):
        """Initialize a weighted ensemble model.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            optimization_metric: Metric to optimize when determining weights.
                Should be a function that takes (y_true, y_pred) and returns a score.
                Higher values should be better.
        """
        super().__init__(
            name=name,
            ensemble_method=EnsembleMethod.WEIGHTED,
            model_type=model_type,
            prediction_target=prediction_target,
            weighting_strategy=EnsembleWeightingStrategy.PERFORMANCE
        )
        self.optimization_metric = optimization_metric
        self.is_fitted = False
    
    def fit(self, X: Any, y: Any) -> 'WeightedEnsemble':
        """Fit the weighted ensemble by determining optimal weights.
        
        Args:
            X: Training data features.
            y: Training data targets.
            
        Returns:
            WeightedEnsemble: Fitted ensemble.
            
        Raises:
            ValueError: If no models are in the ensemble.
        """
        if not self.models:
            raise ValueError(f"No models in {self.name} ensemble")
        
        # Get predictions from all models
        predictions = []
        model_ids = []
        
        for model_info in self.models:
            model = model_info["model"]
            model_id = model_info["model_id"]
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
                model_ids.append(model_id)
            except Exception as e:
                logger.error(f"Error predicting with model {model_id}: {str(e)}")
        
        if not predictions:
            raise ValueError(f"No successful predictions from models in {self.name} ensemble")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        
        # Determine weights based on individual model performance
        if self.optimization_metric is not None:
            # Calculate performance for each model
            performances = []
            for i, pred in enumerate(predictions):
                try:
                    score = self.optimization_metric(y, pred)
                    performances.append(max(0, score))  # Ensure non-negative
                except Exception as e:
                    logger.error(f"Error calculating performance for model {model_ids[i]}: {str(e)}")
                    performances.append(0)  # Assign zero weight if error
            
            # Set weights based on performance
            if sum(performances) > 0:
                weights = [p / sum(performances) for p in performances]
            else:
                # If all performances are zero or negative, use equal weights
                weights = [1.0 / len(predictions) for _ in predictions]
        else:
            # Use equal weights if no optimization metric is provided
            weights = [1.0 / len(predictions) for _ in predictions]
        
        # Update model weights
        for i, model_id in enumerate(model_ids):
            self.update_weight(model_id, weights[i])
        
        self.is_fitted = True
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["is_fitted"] = True
        
        logger.info(f"Fitted {self.name} weighted ensemble with weights: {weights}")
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions with the weighted ensemble.
        
        Args:
            X: Input data for prediction.
            
        Returns:
            np.ndarray: Ensemble predictions.
            
        Raises:
            ValueError: If no models are in the ensemble.
        """
        if not self.models:
            raise ValueError(f"No models in {self.name} ensemble")
        
        predictions = []
        weights = []
        
        for model_info in self.models:
            model = model_info["model"]
            weight = model_info["weight"]
            
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.error(f"Error predicting with model {model_info['model_id']}: {str(e)}")
        
        if not predictions:
            raise ValueError(f"No successful predictions from models in {self.name} ensemble")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights).reshape(-1, 1)
        
        # Compute weighted average
        return np.average(predictions, axis=0, weights=weights.flatten())


class EnsembleFactory:
    """Factory class for creating ensemble models."""
    
    @staticmethod
    def create_ensemble(ensemble_type: str, 
                        name: str,
                        model_type: ModelType,
                        prediction_target: PredictionTarget,
                        **kwargs) -> EnsembleModel:
        """Create an ensemble model of the specified type.
        
        Args:
            ensemble_type: Type of ensemble to create.
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            **kwargs: Additional arguments for the specific ensemble type.
            
        Returns:
            EnsembleModel: Created ensemble model.
            
        Raises:
            ValueError: If the ensemble type is not supported.
        """
        if ensemble_type.upper() == "VOTING":
            return VotingEnsemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                weighting_strategy=kwargs.get("weighting_strategy", EnsembleWeightingStrategy.EQUAL),
                is_classification=kwargs.get("is_classification", False)
            )
        elif ensemble_type.upper() == "STACKING":
            if "meta_model" not in kwargs:
                raise ValueError("meta_model is required for stacking ensemble")
            
            return StackingEnsemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                meta_model=kwargs["meta_model"],
                use_features=kwargs.get("use_features", True)
            )
        elif ensemble_type.upper() == "WEIGHTED":
            return WeightedEnsemble(
                name=name,
                model_type=model_type,
                prediction_target=prediction_target,
                optimization_metric=kwargs.get("optimization_metric", None)
            )
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")


class EnsembleModelTrainer:
    """Trainer for ensemble models."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize an ensemble model trainer.
        
        Args:
            registry: Optional model registry to load models from and save models to.
        """
        self.registry = registry
        logger.info("Initialized EnsembleModelTrainer")
    
    def create_voting_ensemble(self, 
                              name: str,
                              model_type: ModelType,
                              prediction_target: PredictionTarget,
                              models: List[Any],
                              model_ids: Optional[List[str]] = None,
                              weights: Optional[List[float]] = None,
                              is_classification: bool = False) -> VotingEnsemble:
        """Create a voting ensemble from a list of models.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            models: List of models to include in the ensemble.
            model_ids: Optional list of model IDs corresponding to the models.
            weights: Optional list of weights for the models.
            is_classification: Whether this is a classification ensemble.
            
        Returns:
            VotingEnsemble: Created voting ensemble.
            
        Raises:
            ValueError: If the lengths of models, model_ids, and weights don't match.
        """
        if model_ids is not None and len(models) != len(model_ids):
            raise ValueError("Length of models and model_ids must match")
        
        if weights is not None and len(models) != len(weights):
            raise ValueError("Length of models and weights must match")
        
        # Create ensemble
        ensemble = VotingEnsemble(
            name=name,
            model_type=model_type,
            prediction_target=prediction_target,
            weighting_strategy=EnsembleWeightingStrategy.EQUAL if weights is None else EnsembleWeightingStrategy.CUSTOM,
            is_classification=is_classification
        )
        
        # Add models to ensemble
        for i, model in enumerate(models):
            model_id = model_ids[i] if model_ids is not None else None
            weight = weights[i] if weights is not None else 1.0
            ensemble.add_model(model, weight, model_id)
        
        logger.info(f"Created voting ensemble {name} with {len(models)} models")
        return ensemble
    
    def create_stacking_ensemble(self,
                                name: str,
                                model_type: ModelType,
                                prediction_target: PredictionTarget,
                                base_models: List[Any],
                                meta_model: Any,
                                model_ids: Optional[List[str]] = None,
                                use_features: bool = True) -> StackingEnsemble:
        """Create a stacking ensemble from a list of base models and a meta-model.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            base_models: List of base models to include in the ensemble.
            meta_model: Model to use for combining base model predictions.
            model_ids: Optional list of model IDs corresponding to the base models.
            use_features: Whether to include original features in meta-model input.
            
        Returns:
            StackingEnsemble: Created stacking ensemble.
            
        Raises:
            ValueError: If the lengths of base_models and model_ids don't match.
        """
        if model_ids is not None and len(base_models) != len(model_ids):
            raise ValueError("Length of base_models and model_ids must match")
        
        # Create ensemble
        ensemble = StackingEnsemble(
            name=name,
            model_type=model_type,
            prediction_target=prediction_target,
            meta_model=meta_model,
            use_features=use_features
        )
        
        # Add base models to ensemble
        for i, model in enumerate(base_models):
            model_id = model_ids[i] if model_ids is not None else None
            ensemble.add_model(model, 1.0, model_id)
        
        logger.info(f"Created stacking ensemble {name} with {len(base_models)} base models")
        return ensemble
    
    def create_weighted_ensemble(self,
                                name: str,
                                model_type: ModelType,
                                prediction_target: PredictionTarget,
                                models: List[Any],
                                model_ids: Optional[List[str]] = None,
                                optimization_metric: Optional[Callable] = None) -> WeightedEnsemble:
        """Create a weighted ensemble from a list of models.
        
        Args:
            name: Name of the ensemble model.
            model_type: Type of the models in the ensemble.
            prediction_target: Target of the predictions.
            models: List of models to include in the ensemble.
            model_ids: Optional list of model IDs corresponding to the models.
            optimization_metric: Metric to optimize when determining weights.
            
        Returns:
            WeightedEnsemble: Created weighted ensemble.
            
        Raises:
            ValueError: If the lengths of models and model_ids don't match.
        """
        if model_ids is not None and len(models) != len(model_ids):
            raise ValueError("Length of models and model_ids must match")
        
        # Create ensemble
        ensemble = WeightedEnsemble(
            name=name,
            model_type=model_type,
            prediction_target=prediction_target,
            optimization_metric=optimization_metric
        )
        
        # Add models to ensemble
        for i, model in enumerate(models):
            model_id = model_ids[i] if model_ids is not None else None
            ensemble.add_model(model, 1.0, model_id)
        
        logger.info(f"Created weighted ensemble {name} with {len(models)} models")
        return ensemble
    
    def train_ensemble(self, 
                      ensemble: EnsembleModel,
                      X_train: Any,
                      y_train: Any) -> EnsembleModel:
        """Train an ensemble model.
        
        Args:
            ensemble: Ensemble model to train.
            X_train: Training data features.
            y_train: Training data targets.
            
        Returns:
            EnsembleModel: Trained ensemble model.
            
        Raises:
            ValueError: If the ensemble type is not supported for training.
        """
        if isinstance(ensemble, StackingEnsemble):
            ensemble.fit(X_train, y_train)
        elif isinstance(ensemble, WeightedEnsemble):
            ensemble.fit(X_train, y_train)
        elif isinstance(ensemble, VotingEnsemble):
            # VotingEnsemble doesn't need training, but we can optimize weights
            if ensemble.weighting_strategy == EnsembleWeightingStrategy.PERFORMANCE:
                # Create a temporary WeightedEnsemble to determine optimal weights
                weighted = WeightedEnsemble(
                    name=f"{ensemble.name}_temp",
                    model_type=ensemble.model_type,
                    prediction_target=ensemble.prediction_target
                )
                
                # Add the same models
                for model_info in ensemble.models:
                    weighted.add_model(
                        model_info["model"],
                        model_info["weight"],
                        model_info["model_id"]
                    )
                
                # Fit to determine optimal weights
                weighted.fit(X_train, y_train)
                
                # Copy weights to original ensemble
                for i, model_info in enumerate(weighted.models):
                    ensemble.update_weight(model_info["model_id"], model_info["weight"])
        else:
            raise ValueError(f"Unsupported ensemble type for training: {type(ensemble).__name__}")
        
        logger.info(f"Trained {ensemble.name} ensemble")
        return ensemble
    
    def evaluate_ensemble(self,
                         ensemble: EnsembleModel,
                         X_test: Any,
                         y_test: Any,
                         metrics: List[Callable]) -> Dict[str, float]:
        """Evaluate an ensemble model.
        
        Args:
            ensemble: Ensemble model to evaluate.
            X_test: Test data features.
            y_test: Test data targets.
            metrics: List of metric functions to use for evaluation.
                Each function should take (y_true, y_pred) and return a score.
            
        Returns:
            Dict[str, float]: Dictionary of metric names and scores.
        """
        # Make predictions
        try:
            y_pred = ensemble.predict(X_test)
        except Exception as e:
            error_msg = f"Error predicting with {ensemble.name} ensemble: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate metrics
        results = {}
        for i, metric_func in enumerate(metrics):
            try:
                metric_name = getattr(metric_func, "__name__", f"metric_{i}")
                score = metric_func(y_test, y_pred)
                results[metric_name] = score
            except Exception as e:
                logger.error(f"Error calculating {metric_name} for {ensemble.name} ensemble: {str(e)}")
        
        logger.info(f"Evaluated {ensemble.name} ensemble: {results}")
        return results
    
    def save_ensemble(self, ensemble: EnsembleModel, path: str) -> str:
        """Save an ensemble model.
        
        Args:
            ensemble: Ensemble model to save.
            path: Directory path to save the model.
            
        Returns:
            str: Path to the saved model file.
        """
        return ensemble.save(path)
    
    def load_ensemble(self, path: str, ensemble_type: Type[EnsembleModel] = EnsembleModel) -> EnsembleModel:
        """Load an ensemble model.
        
        Args:
            path: Path to the saved model file.
            ensemble_type: Type of ensemble to load.
            
        Returns:
            EnsembleModel: Loaded ensemble model.
        """
        return ensemble_type.load(path, self.registry)
    
    def register_ensemble(self, ensemble: EnsembleModel, tags: Optional[List[str]] = None) -> Optional[str]:
        """Register an ensemble model in the model registry.
        
        Args:
            ensemble: Ensemble model to register.
            tags: Optional list of tags to associate with the model.
            
        Returns:
            Optional[str]: Model ID if registration was successful, None otherwise.
        """
        if self.registry is None:
            logger.warning(f"Cannot register {ensemble.name} ensemble: No registry provided")
            return None
        
        try:
            # Save ensemble to temporary directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                ensemble_path = ensemble.save(temp_dir)
                
                # Register ensemble in registry
                model_id = self.registry.register_model(
                    model=ensemble,
                    model_type=ensemble.model_type,
                    prediction_target=ensemble.prediction_target,
                    metadata={
                        "ensemble_method": ensemble.ensemble_method.name,
                        "weighting_strategy": ensemble.weighting_strategy.name,
                        "model_count": len(ensemble.models),
                        "ensemble_id": ensemble.metadata.get("ensemble_id", str(uuid.uuid4()))
                    },
                    tags=tags or ["ensemble"],
                    model_path=ensemble_path
                )
                
                logger.info(f"Registered {ensemble.name} ensemble in registry with ID {model_id}")
                return model_id
        
        except Exception as e:
            logger.error(f"Error registering {ensemble.name} ensemble in registry: {str(e)}")
            return None