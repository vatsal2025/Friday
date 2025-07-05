"""Model Loader for Friday AI Trading System.

This module provides functionality for loading machine learning models from the model registry.
"""

import os
import joblib
from typing import Dict, List, Optional, Any, Union, Tuple

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_operations import ModelOperations

# Create logger
logger = get_logger(__name__)


class ModelLoader:
    """Loader for machine learning models.

    This class provides functionality for loading machine learning models from the model registry.

    Attributes:
        model_registry: The model registry instance.
        model_operations: The model operations instance.
        models_cache: Cache of loaded models.
    """

    def __init__(self, registry_dir: Optional[str] = None):
        """Initialize the model loader.

        Args:
            registry_dir: Directory for the model registry. If None, the default directory will be used.
        """
        self.model_registry = ModelRegistry(registry_dir=registry_dir)
        self.model_operations = ModelOperations(model_registry=self.model_registry)
        self.models_cache = {}
        
        logger.info("Initialized ModelLoader")

    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a model from the registry.

        Args:
            model_name: Name of the model to load.
            version: Version of the model to load. If None, the latest version will be used.

        Returns:
            Any: The loaded model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Check if model is already in cache
        cache_key = f"{model_name}_{version if version else 'latest'}"
        if cache_key in self.models_cache:
            logger.info(f"Loading model {model_name} (version: {version if version else 'latest'}) from cache")
            return self.models_cache[cache_key]
        
        # Load model from registry
        try:
            if version:
                logger.info(f"Loading model {model_name} (version: {version}) from registry")
                model = self.model_operations.get_model_by_name_and_version(model_name, version)
            else:
                logger.info(f"Loading latest version of model {model_name} from registry")
                model = self.model_operations.get_latest_model_version(model_name)
            
            # Cache model
            self.models_cache[cache_key] = model
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name} (version: {version if version else 'latest'}): {str(e)}")
            raise ValueError(f"Model {model_name} (version: {version if version else 'latest'}) not found in registry")

    def load_model_by_tags(self, tags: List[str], require_all_tags: bool = True) -> List[Any]:
        """Load models from the registry that match the given tags.

        Args:
            tags: List of tags to match.
            require_all_tags: If True, models must have all tags to match. If False, models with any of the tags will match.

        Returns:
            List[Any]: List of loaded models.
        """
        # Get models with matching tags
        try:
            model_infos = self.model_registry.list_models(tags=tags, require_all_tags=require_all_tags)
            
            # Load models
            models = []
            for model_info in model_infos:
                model_name = model_info.get("name")
                model_version = model_info.get("version")
                
                if model_name and model_version:
                    model = self.load_model(model_name, model_version)
                    models.append(model)
            
            logger.info(f"Loaded {len(models)} models with tags {tags}")
            
            return models
        except Exception as e:
            logger.error(f"Error loading models with tags {tags}: {str(e)}")
            return []

    def load_best_model(self, model_name: str, metric: str = "r2", higher_is_better: bool = True) -> Any:
        """Load the best model version based on a metric.

        Args:
            model_name: Name of the model.
            metric: Metric to use for comparison.
            higher_is_better: If True, higher metric values are better. If False, lower metric values are better.

        Returns:
            Any: The best model.

        Raises:
            ValueError: If no models are found with the given name.
        """
        try:
            # Find the best model version
            best_model = self.model_operations.find_best_model(model_name, metric, higher_is_better)
            
            if best_model:
                logger.info(f"Loaded best model {model_name} based on {metric}")
                return best_model
            else:
                raise ValueError(f"No models found with name {model_name}")
        except Exception as e:
            logger.error(f"Error loading best model {model_name} based on {metric}: {str(e)}")
            raise

    def load_model_for_prediction(
        self,
        symbol: str,
        timeframe: str,
        target_type: str,
        forecast_horizon: int,
        model_type: Optional[str] = None,
    ) -> Any:
        """Load a model for prediction.

        Args:
            symbol: The symbol for which to load the model.
            timeframe: The timeframe for which to load the model.
            target_type: The type of prediction target.
            forecast_horizon: The forecast horizon in periods.
            model_type: The type of model to load. If None, the best model will be loaded.

        Returns:
            Any: The loaded model.

        Raises:
            ValueError: If no matching models are found.
        """
        # Create tags for filtering
        tags = [symbol, timeframe, target_type, f"horizon_{forecast_horizon}"]
        if model_type:
            tags.append(model_type)
        
        try:
            # Get models with matching tags
            model_infos = self.model_registry.list_models(tags=tags, require_all_tags=True)
            
            if not model_infos:
                raise ValueError(f"No models found for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon")
            
            # If model type is specified, filter by model type
            if model_type:
                model_infos = [info for info in model_infos if info.get("metadata", {}).get("model_type") == model_type]
            
            if not model_infos:
                raise ValueError(f"No models found for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon using {model_type}")
            
            # If multiple models are found, load the best one based on R2 score
            if len(model_infos) > 1:
                # Extract model names
                model_names = [info.get("name") for info in model_infos if info.get("name")]
                
                # Find the best model
                best_model = None
                best_r2 = -float("inf")
                
                for model_name in model_names:
                    model = self.load_best_model(model_name, "r2", True)
                    model_metadata = self.model_registry.get_model_metadata(model_name, model.metadata.get("version"))
                    r2 = model_metadata.get("metrics", {}).get("r2", -float("inf"))
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                
                if best_model:
                    logger.info(f"Loaded best model for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon")
                    return best_model
                else:
                    raise ValueError(f"No valid models found for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon")
            else:
                # Load the single model found
                model_info = model_infos[0]
                model_name = model_info.get("name")
                model_version = model_info.get("version")
                
                if model_name and model_version:
                    model = self.load_model(model_name, model_version)
                    logger.info(f"Loaded model {model_name} (version: {model_version}) for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon")
                    return model
                else:
                    raise ValueError(f"Invalid model information for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon")
        except Exception as e:
            logger.error(f"Error loading model for {symbol} on {timeframe} timeframe with {target_type} target and {forecast_horizon} horizon: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the models cache."""
        self.models_cache = {}
        logger.info("Cleared models cache")

    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a model.

        Args:
            model_name: Name of the model.
            version: Version of the model. If None, the latest version will be used.

        Returns:
            Dict[str, Any]: The model metadata.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        try:
            if version:
                metadata = self.model_registry.get_model_metadata(model_name, version)
            else:
                latest_version = self.model_registry.get_latest_version(model_name)
                metadata = self.model_registry.get_model_metadata(model_name, latest_version)
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata for model {model_name} (version: {version if version else 'latest'}): {str(e)}")
            raise ValueError(f"Model {model_name} (version: {version if version else 'latest'}) not found in registry")


def load_model(model_name: str, version: Optional[str] = None) -> Any:
    """Convenience function to load a model from the registry.

    Args:
        model_name: Name of the model to load.
        version: Version of the model to load. If None, the latest version will be used.

    Returns:
        Any: The loaded model.

    Raises:
        ValueError: If the model is not found in the registry.
    """
    loader = ModelLoader()
    return loader.load_model(model_name, version)


def load_model_for_prediction(
    symbol: str,
    timeframe: str,
    target_type: str,
    forecast_horizon: int,
    model_type: Optional[str] = None,
) -> Any:
    """Convenience function to load a model for prediction.

    Args:
        symbol: The symbol for which to load the model.
        timeframe: The timeframe for which to load the model.
        target_type: The type of prediction target.
        forecast_horizon: The forecast horizon in periods.
        model_type: The type of model to load. If None, the best model will be loaded.

    Returns:
        Any: The loaded model.

    Raises:
        ValueError: If no matching models are found.
    """
    loader = ModelLoader()
    return loader.load_model_for_prediction(symbol, timeframe, target_type, forecast_horizon, model_type)