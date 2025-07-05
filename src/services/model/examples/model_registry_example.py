"""Example script demonstrating the use of the model registry system.

This script shows how to register, load, and manage models using the model registry.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.services.model import (
    ModelRegistry,
    ModelOperations,
    ModelVersioning,
    ModelSerializer,
    ModelTrainerIntegration,
    ModelLoader,
    load_model,
    load_model_for_prediction
)


def create_sample_model():
    """Create a sample model for demonstration.

    Returns:
        tuple: The model, X_test, y_test, and evaluation results.
    """
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Create evaluation results
    eval_results = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "test_size": len(y_test),
        "train_size": len(y_train)
    }
    
    return model, X_test, y_test, eval_results


def example_register_model():
    """Example of registering a model in the registry."""
    print("\n=== Example: Register Model ===")
    
    # Create a sample model
    model, X_test, y_test, eval_results = create_sample_model()
    
    # Initialize the model registry
    registry = ModelRegistry()
    
    # Register the model
    model_id = registry.register_model(
        model=model,
        model_name="example_model",
        model_version="1.0.0",
        description="Example model for demonstration",
        tags=["example", "regression", "random_forest"],
        metadata={
            "model_type": "random_forest",
            "metrics": {
                "mse": eval_results["mse"],
                "rmse": eval_results["rmse"],
                "r2": eval_results["r2"]
            },
            "hyperparameters": {
                "n_estimators": 100,
                "random_state": 42
            }
        }
    )
    
    print(f"Registered model with ID: {model_id}")
    
    # Register another version of the model
    model, _, _, eval_results = create_sample_model()
    model_id = registry.register_model(
        model=model,
        model_name="example_model",
        model_version="1.1.0",
        description="Updated example model for demonstration",
        tags=["example", "regression", "random_forest", "updated"],
        metadata={
            "model_type": "random_forest",
            "metrics": {
                "mse": eval_results["mse"],
                "rmse": eval_results["rmse"],
                "r2": eval_results["r2"]
            },
            "hyperparameters": {
                "n_estimators": 200,
                "random_state": 42
            }
        }
    )
    
    print(f"Registered updated model with ID: {model_id}")
    
    return registry


def example_load_model(registry):
    """Example of loading a model from the registry.
    
    Args:
        registry: The model registry instance.
    """
    print("\n=== Example: Load Model ===")
    
    # Load the model by name and version
    model = registry.load_model("example_model", "1.0.0")
    print(f"Loaded model: {model}")
    
    # Load the latest version of the model
    latest_version = registry.get_latest_version("example_model")
    model = registry.load_model("example_model", latest_version)
    print(f"Loaded latest version ({latest_version}) of model: {model}")
    
    # Use the convenience function to load the model
    model = load_model("example_model")
    print(f"Loaded model using convenience function: {model}")
    
    return model


def example_model_metadata(registry):
    """Example of working with model metadata.
    
    Args:
        registry: The model registry instance.
    """
    print("\n=== Example: Model Metadata ===")
    
    # Get model metadata
    metadata = registry.get_model_metadata("example_model", "1.0.0")
    print(f"Model metadata: {metadata}")
    
    # Update model metadata
    registry.update_model_metadata(
        "example_model",
        "1.0.0",
        {"status": "production", "updated_by": "example_script"}
    )
    
    # Get updated metadata
    metadata = registry.get_model_metadata("example_model", "1.0.0")
    print(f"Updated model metadata: {metadata}")
    
    return metadata


def example_model_operations():
    """Example of using model operations."""
    print("\n=== Example: Model Operations ===")
    
    # Initialize model operations
    registry = ModelRegistry()
    operations = ModelOperations(model_registry=registry)
    
    # List all models
    models = operations.list_models()
    print(f"All models: {models}")
    
    # List models with a specific tag
    models = operations.list_models(tags=["updated"])
    print(f"Models with 'updated' tag: {models}")
    
    # Get model versions
    versions = operations.get_model_versions("example_model")
    print(f"Versions of 'example_model': {versions}")
    
    # Find the best model based on a metric
    best_model = operations.find_best_model("example_model", "r2", higher_is_better=True)
    print(f"Best model based on R2 score: {best_model}")
    
    # Compare models
    comparison = operations.compare_models(
        "example_model",
        "1.0.0",
        "1.1.0",
        metrics=["mse", "rmse", "r2"]
    )
    print(f"Model comparison: {comparison}")
    
    return operations


def example_model_versioning():
    """Example of using model versioning."""
    print("\n=== Example: Model Versioning ===")
    
    # Initialize model versioning
    registry = ModelRegistry()
    versioning = ModelVersioning(model_registry=registry)
    
    # Get model lineage
    lineage = versioning.get_model_lineage("example_model")
    print(f"Model lineage: {lineage}")
    
    # Get version history
    history = versioning.get_version_history("example_model")
    print(f"Version history: {history}")
    
    # Tag a version
    versioning.tag_version("example_model", "1.0.0", "baseline")
    versioning.tag_version("example_model", "1.1.0", "improved")
    
    # Get versions by tag
    versions = versioning.get_versions_by_tag("example_model", "baseline")
    print(f"Versions with 'baseline' tag: {versions}")
    
    return versioning


def example_model_serialization():
    """Example of using model serialization."""
    print("\n=== Example: Model Serialization ===")
    
    # Create a sample model
    model, _, _, _ = create_sample_model()
    
    # Initialize model serialization
    serializer = ModelSerializer()
    
    # Serialize model to file
    file_path = os.path.join(os.getcwd(), "example_model.joblib")
    serializer.serialize_model(model, file_path)
    print(f"Serialized model to: {file_path}")
    
    # Get model hash
    model_hash = serializer.compute_model_hash(model)
    print(f"Model hash: {model_hash}")
    
    # Get serialized size
    size = serializer.get_serialized_size(model)
    print(f"Serialized size: {size} bytes")
    
    # Deserialize model from file
    deserialized_model = serializer.deserialize_model(file_path)
    print(f"Deserialized model: {deserialized_model}")
    
    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return serializer


def example_model_trainer_integration():
    """Example of using model trainer integration."""
    print("\n=== Example: Model Trainer Integration ===")
    
    # Create a sample model
    model, _, _, eval_results = create_sample_model()
    
    # Initialize model trainer integration
    integration = ModelTrainerIntegration()
    
    # Register model from trainer
    model_id = integration.register_model_from_trainer(
        model=model,
        model_name="trainer_example_model",
        model_type="random_forest",
        evaluation_results={
            "metrics": {
                "mse": eval_results["mse"],
                "rmse": eval_results["rmse"],
                "r2": eval_results["r2"]
            },
            "details": eval_results
        },
        training_data_info={
            "train_size": eval_results["train_size"],
            "test_size": eval_results["test_size"]
        },
        hyperparameters={
            "n_estimators": 100,
            "random_state": 42
        },
        feature_importance={
            f"feature_{i}": importance
            for i, importance in enumerate(model.feature_importances_)
        },
        tags=["example", "trainer", "random_forest"],
        description="Example model registered from trainer"
    )
    
    print(f"Registered model from trainer with ID: {model_id}")
    
    return integration


def example_model_loader():
    """Example of using model loader."""
    print("\n=== Example: Model Loader ===")
    
    # Initialize model loader
    loader = ModelLoader()
    
    # Load model by name
    model = loader.load_model("example_model")
    print(f"Loaded model: {model}")
    
    # Load model by tags
    models = loader.load_model_by_tags(["example", "regression"])
    print(f"Loaded {len(models)} models with tags 'example' and 'regression'")
    
    # Load best model
    best_model = loader.load_best_model("example_model", "r2")
    print(f"Loaded best model based on R2 score: {best_model}")
    
    # Get model metadata
    metadata = loader.get_model_metadata("example_model")
    print(f"Model metadata: {metadata}")
    
    # Clear cache
    loader.clear_cache()
    print("Cleared model cache")
    
    return loader


def main():
    """Main function to run all examples."""
    print("=== Model Registry Examples ===")
    
    # Run examples
    registry = example_register_model()
    model = example_load_model(registry)
    metadata = example_model_metadata(registry)
    operations = example_model_operations()
    versioning = example_model_versioning()
    serializer = example_model_serialization()
    integration = example_model_trainer_integration()
    loader = example_model_loader()
    
    print("\n=== Examples Completed ===")


if __name__ == "__main__":
    main()