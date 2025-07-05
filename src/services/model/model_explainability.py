import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import shap
from typing import Dict, List, Any, Optional, Union, Tuple

from src.services.model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Class for explaining model predictions and behavior."""
    
    def __init__(self, registry: ModelRegistry):
        """Initialize a model explainer.
        
        Args:
            registry: Model registry to use for loading models.
        """
        self.registry = registry
        self.explainers = {}
        logger.info("Initialized ModelExplainer with registry %s", registry)
    
    def explain_prediction(self, 
                           model_id: str, 
                           input_data: Any, 
                           prediction: Any = None,
                           num_features: int = 10) -> Dict[str, Any]:
        """Explain a specific prediction made by a model.
        
        Args:
            model_id: ID of the model to explain.
            input_data: Input data for the prediction.
            prediction: The prediction to explain (optional, will be computed if not provided).
            num_features: Number of top features to include in the explanation.
            
        Returns:
            Dict[str, Any]: Explanation of the prediction.
            
        Raises:
            ValueError: If the model ID doesn't exist or the model type is not supported.
        """
        # Load the model
        model = self.registry.load_model(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Make prediction if not provided
        if prediction is None:
            prediction = model.predict(input_data)
        
        # Get or create explainer for this model
        if model_id not in self.explainers:
            self._create_explainer(model_id, model)
        
        explainer = self.explainers.get(model_id)
        if explainer is None:
            raise ValueError(f"Could not create explainer for model {model_id}")
        
        # Generate explanation
        explanation = {}
        
        try:
            # Convert input data to appropriate format if needed
            if isinstance(input_data, pd.DataFrame):
                # For SHAP explainers that expect numpy arrays
                if hasattr(explainer, "shap_values"):
                    shap_values = explainer.shap_values(input_data.values)
                else:
                    shap_values = explainer(input_data.values)
                
                # Get feature names
                feature_names = input_data.columns.tolist()
                
                # Create feature importance dictionary
                feature_importance = {}
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # For multi-class models
                    for i, class_shap_values in enumerate(shap_values):
                        if len(class_shap_values.shape) > 1:
                            # For a batch of inputs, take the first one
                            values = class_shap_values[0]
                        else:
                            values = class_shap_values
                        
                        # Sort features by importance
                        sorted_idx = np.argsort(np.abs(values))[::-1][:num_features]
                        class_importance = {feature_names[i]: float(values[i]) for i in sorted_idx}
                        feature_importance[f"class_{i}"] = class_importance
                else:
                    # For regression or binary classification
                    if len(shap_values.shape) > 1:
                        # For a batch of inputs, take the first one
                        values = shap_values[0]
                    else:
                        values = shap_values
                    
                    # Sort features by importance
                    sorted_idx = np.argsort(np.abs(values))[::-1][:num_features]
                    feature_importance = {feature_names[i]: float(values[i]) for i in sorted_idx}
                
                explanation["feature_importance"] = feature_importance
                explanation["base_value"] = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
            else:
                # For non-DataFrame inputs, provide a simpler explanation
                explanation["message"] = "Detailed feature importance requires DataFrame input"
        
        except Exception as e:
            logger.error("Error generating explanation for model %s: %s", model_id, e)
            explanation["error"] = str(e)
        
        # Add prediction to explanation
        explanation["prediction"] = prediction
        
        return explanation
    
    def _create_explainer(self, model_id: str, model: Any) -> None:
        """Create an appropriate explainer for the model.
        
        Args:
            model_id: ID of the model.
            model: The model object.
            
        Raises:
            ValueError: If the model type is not supported for explanation.
        """
        try:
            # Try to create a SHAP explainer based on model type
            if hasattr(model, "feature_names"):
                # For models with feature names (like XGBoost)
                self.explainers[model_id] = shap.TreeExplainer(model)
            elif hasattr(model, "predict_proba"):
                # For scikit-learn models with predict_proba
                self.explainers[model_id] = shap.KernelExplainer(
                    model.predict_proba, 
                    shap.sample(np.zeros((1, model.n_features_in_)), 100)
                )
            elif hasattr(model, "predict"):
                # For scikit-learn models without predict_proba
                self.explainers[model_id] = shap.KernelExplainer(
                    model.predict, 
                    shap.sample(np.zeros((1, model.n_features_in_)), 100)
                )
            else:
                # For other model types, use a generic approach
                logger.warning("Using generic explainer for model %s", model_id)
                self.explainers[model_id] = shap.Explainer(model)
            
            logger.info("Created explainer for model %s", model_id)
        
        except Exception as e:
            logger.error("Error creating explainer for model %s: %s", model_id, e)
            self.explainers[model_id] = None
    
    def generate_feature_importance(self, 
                                   model_id: str, 
                                   data: pd.DataFrame,
                                   num_features: int = 10,
                                   output_path: str = None) -> Dict[str, Any]:
        """Generate global feature importance for a model.
        
        Args:
            model_id: ID of the model to explain.
            data: Sample data to use for generating feature importance.
            num_features: Number of top features to include.
            output_path: Path to save the feature importance plot (optional).
            
        Returns:
            Dict[str, Any]: Feature importance information.
            
        Raises:
            ValueError: If the model ID doesn't exist or the model type is not supported.
        """
        # Load the model
        model = self.registry.load_model(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Get or create explainer for this model
        if model_id not in self.explainers:
            self._create_explainer(model_id, model)
        
        explainer = self.explainers.get(model_id)
        if explainer is None:
            raise ValueError(f"Could not create explainer for model {model_id}")
        
        # Generate feature importance
        result = {}
        
        try:
            # Calculate SHAP values
            shap_values = explainer.shap_values(data) if hasattr(explainer, "shap_values") else explainer(data)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # For multi-class models
                feature_importance = {}
                for i, class_shap_values in enumerate(shap_values):
                    # Calculate mean absolute SHAP value for each feature
                    mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)
                    
                    # Sort features by importance
                    sorted_idx = np.argsort(mean_abs_shap)[::-1][:num_features]
                    class_importance = {data.columns[i]: float(mean_abs_shap[i]) for i in sorted_idx}
                    feature_importance[f"class_{i}"] = class_importance
                
                result["feature_importance"] = feature_importance
            else:
                # For regression or binary classification
                # Calculate mean absolute SHAP value for each feature
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                
                # Sort features by importance
                sorted_idx = np.argsort(mean_abs_shap)[::-1][:num_features]
                feature_importance = {data.columns[i]: float(mean_abs_shap[i]) for i in sorted_idx}
                
                result["feature_importance"] = feature_importance
            
            # Generate plot if output path is provided
            if output_path:
                plt.figure(figsize=(10, 6))
                
                if isinstance(shap_values, list):
                    # For multi-class, use the first class for visualization
                    shap.summary_plot(shap_values[0], data, show=False)
                else:
                    shap.summary_plot(shap_values, data, show=False)
                
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                result["plot_path"] = output_path
        
        except Exception as e:
            logger.error("Error generating feature importance for model %s: %s", model_id, e)
            result["error"] = str(e)
        
        return result
    
    def generate_partial_dependence(self,
                                   model_id: str,
                                   data: pd.DataFrame,
                                   feature_name: str,
                                   output_path: str = None) -> Dict[str, Any]:
        """Generate partial dependence plot for a specific feature.
        
        Args:
            model_id: ID of the model to explain.
            data: Sample data to use for generating the plot.
            feature_name: Name of the feature to analyze.
            output_path: Path to save the partial dependence plot (optional).
            
        Returns:
            Dict[str, Any]: Partial dependence information.
            
        Raises:
            ValueError: If the model ID doesn't exist or the feature is not found.
        """
        # Load the model
        model = self.registry.load_model(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Check if feature exists in data
        if feature_name not in data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        # Generate partial dependence
        result = {}
        
        try:
            # Get feature values
            feature_values = data[feature_name].unique()
            feature_values.sort()
            
            # Create a grid of values to evaluate
            grid = np.linspace(
                np.percentile(data[feature_name], 1),
                np.percentile(data[feature_name], 99),
                num=20
            )
            
            # Create copies of the data with different feature values
            predictions = []
            for value in grid:
                data_copy = data.copy()
                data_copy[feature_name] = value
                pred = model.predict(data_copy)
                predictions.append(np.mean(pred))
            
            # Store results
            result["feature_name"] = feature_name
            result["grid_values"] = grid.tolist()
            result["predictions"] = predictions
            
            # Generate plot if output path is provided
            if output_path:
                plt.figure(figsize=(10, 6))
                plt.plot(grid, predictions)
                plt.xlabel(feature_name)
                plt.ylabel('Predicted Value')
                plt.title(f'Partial Dependence Plot for {feature_name}')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                result["plot_path"] = output_path
        
        except Exception as e:
            logger.error("Error generating partial dependence for model %s, feature %s: %s", 
                       model_id, feature_name, e)
            result["error"] = str(e)
        
        return result
    
    def generate_ice_curves(self,
                           model_id: str,
                           data: pd.DataFrame,
                           feature_name: str,
                           num_samples: int = 10,
                           output_path: str = None) -> Dict[str, Any]:
        """Generate Individual Conditional Expectation (ICE) curves for a feature.
        
        Args:
            model_id: ID of the model to explain.
            data: Sample data to use for generating the curves.
            feature_name: Name of the feature to analyze.
            num_samples: Number of individual samples to plot.
            output_path: Path to save the ICE curves plot (optional).
            
        Returns:
            Dict[str, Any]: ICE curves information.
            
        Raises:
            ValueError: If the model ID doesn't exist or the feature is not found.
        """
        # Load the model
        model = self.registry.load_model(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Check if feature exists in data
        if feature_name not in data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")
        
        # Generate ICE curves
        result = {}
        
        try:
            # Create a grid of values to evaluate
            grid = np.linspace(
                np.percentile(data[feature_name], 1),
                np.percentile(data[feature_name], 99),
                num=20
            )
            
            # Sample rows from the data
            if len(data) > num_samples:
                sample_indices = np.random.choice(len(data), num_samples, replace=False)
                samples = data.iloc[sample_indices]
            else:
                samples = data
            
            # Generate ICE curves for each sample
            ice_curves = []
            for i, (_, row) in enumerate(samples.iterrows()):
                curve = []
                for value in grid:
                    row_copy = row.copy()
                    row_copy[feature_name] = value
                    pred = model.predict(pd.DataFrame([row_copy]))[0]
                    curve.append(float(pred) if hasattr(pred, "__iter__") else float(pred))
                
                ice_curves.append({
                    "sample_id": i,
                    "grid_values": grid.tolist(),
                    "predictions": curve
                })
            
            result["feature_name"] = feature_name
            result["ice_curves"] = ice_curves
            
            # Calculate partial dependence (average of ICE curves)
            pd_values = np.mean([curve["predictions"] for curve in ice_curves], axis=0)
            result["partial_dependence"] = pd_values.tolist()
            
            # Generate plot if output path is provided
            if output_path:
                plt.figure(figsize=(10, 6))
                
                # Plot individual ICE curves
                for curve in ice_curves:
                    plt.plot(curve["grid_values"], curve["predictions"], 
                             color='lightblue', alpha=0.5, linewidth=1)
                
                # Plot partial dependence (average)
                plt.plot(grid, pd_values, color='red', linewidth=2, label='Average (PD)')
                
                plt.xlabel(feature_name)
                plt.ylabel('Predicted Value')
                plt.title(f'ICE Curves for {feature_name}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                result["plot_path"] = output_path
        
        except Exception as e:
            logger.error("Error generating ICE curves for model %s, feature %s: %s", 
                       model_id, feature_name, e)
            result["error"] = str(e)
        
        return result
    
    def generate_shap_interaction(self,
                                 model_id: str,
                                 data: pd.DataFrame,
                                 feature1: str,
                                 feature2: str,
                                 output_path: str = None) -> Dict[str, Any]:
        """Generate SHAP interaction values between two features.
        
        Args:
            model_id: ID of the model to explain.
            data: Sample data to use for generating interaction values.
            feature1: Name of the first feature.
            feature2: Name of the second feature.
            output_path: Path to save the interaction plot (optional).
            
        Returns:
            Dict[str, Any]: Interaction information.
            
        Raises:
            ValueError: If the model ID doesn't exist or features are not found.
        """
        # Load the model
        model = self.registry.load_model(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        # Check if features exist in data
        if feature1 not in data.columns:
            raise ValueError(f"Feature '{feature1}' not found in data")
        if feature2 not in data.columns:
            raise ValueError(f"Feature '{feature2}' not found in data")
        
        # Generate interaction values
        result = {}
        
        try:
            # Get or create explainer for this model
            if model_id not in self.explainers:
                self._create_explainer(model_id, model)
            
            explainer = self.explainers.get(model_id)
            if explainer is None:
                raise ValueError(f"Could not create explainer for model {model_id}")
            
            # Check if explainer supports interaction values
            if not hasattr(explainer, "shap_interaction_values"):
                raise ValueError("This model explainer does not support interaction values")
            
            # Calculate interaction values
            interaction_values = explainer.shap_interaction_values(data)
            
            # Get feature indices
            feature1_idx = data.columns.get_loc(feature1)
            feature2_idx = data.columns.get_loc(feature2)
            
            # Extract interaction values for the two features
            if isinstance(interaction_values, list):
                # For multi-class models, use the first class
                interaction = interaction_values[0][:, feature1_idx, feature2_idx]
            else:
                interaction = interaction_values[:, feature1_idx, feature2_idx]
            
            # Store results
            result["feature1"] = feature1
            result["feature2"] = feature2
            result["interaction_values"] = interaction.tolist()
            result["mean_interaction"] = float(np.mean(np.abs(interaction)))
            
            # Generate plot if output path is provided
            if output_path:
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot
                plt.scatter(data[feature1], data[feature2], c=interaction, 
                           cmap='coolwarm', s=50, alpha=0.8)
                
                plt.colorbar(label='Interaction Value')
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                plt.title(f'SHAP Interaction: {feature1} vs {feature2}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                result["plot_path"] = output_path
        
        except Exception as e:
            logger.error("Error generating SHAP interaction for model %s, features %s and %s: %s", 
                       model_id, feature1, feature2, e)
            result["error"] = str(e)
        
        return result
    
    def save_explanation(self, explanation: Dict[str, Any], output_path: str) -> None:
        """Save an explanation to disk.
        
        Args:
            explanation: The explanation to save.
            output_path: Path to save the explanation to.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert explanation to serializable format
            serializable_explanation = {}
            for key, value in explanation.items():
                if isinstance(value, np.ndarray):
                    serializable_explanation[key] = value.tolist()
                elif isinstance(value, (np.int64, np.float64)):
                    serializable_explanation[key] = float(value)
                else:
                    serializable_explanation[key] = value
            
            # Save to file
            with open(output_path, "w") as f:
                json.dump(serializable_explanation, f, indent=2)
            
            logger.info("Saved explanation to %s", output_path)
        
        except Exception as e:
            logger.error("Error saving explanation to %s: %s", output_path, e)
    
    def load_explanation(self, input_path: str) -> Dict[str, Any]:
        """Load an explanation from disk.
        
        Args:
            input_path: Path to load the explanation from.
            
        Returns:
            Dict[str, Any]: The loaded explanation.
            
        Raises:
            FileNotFoundError: If the input path doesn't exist.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist")
        
        try:
            # Load from file
            with open(input_path, "r") as f:
                explanation = json.load(f)
            
            logger.info("Loaded explanation from %s", input_path)
            return explanation
        
        except Exception as e:
            logger.error("Error loading explanation from %s: %s", input_path, e)
            raise