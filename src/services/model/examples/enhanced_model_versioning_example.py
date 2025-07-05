"""Example script demonstrating the use of the enhanced model versioning system.

This script shows how to use semantic versioning, automatic version incrementation,
and migration tools for backward compatibility in the model versioning system.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple, Callable

from src.services.model import (
    ModelRegistry,
    ModelOperations,
    load_model
)
from src.services.model.enhanced_model_versioning import EnhancedModelVersioning
from src.services.model.utils.semantic_versioning import SemanticVersioning, VersionChangeType


def create_sample_model(n_estimators=100, random_state=42, model_type="random_forest"):
    """Create a sample model for demonstration.

    Args:
        n_estimators: Number of estimators for the model
        random_state: Random state for reproducibility
        model_type: Type of model to create (random_forest or gradient_boosting)

    Returns:
        tuple: The model, X_test, y_test, and evaluation results.
    """
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    else:  # gradient_boosting
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
    
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


def example_semantic_versioning():
    """Example of using semantic versioning utilities."""
    print("\n=== Example: Semantic Versioning Utilities ===\n")
    
    # Validate version strings
    valid_version = "1.0.0"
    invalid_version = "1.0"
    
    print(f"Is '{valid_version}' valid? {SemanticVersioning.validate_version(valid_version)}")
    print(f"Is '{invalid_version}' valid? {SemanticVersioning.validate_version(invalid_version)}")
    
    # Compare versions
    version1 = "1.0.0"
    version2 = "1.1.0"
    version3 = "2.0.0"
    
    print(f"Compare '{version1}' and '{version2}': {SemanticVersioning.compare_versions(version1, version2)}")
    print(f"Compare '{version2}' and '{version3}': {SemanticVersioning.compare_versions(version2, version3)}")
    print(f"Compare '{version1}' and '{version1}': {SemanticVersioning.compare_versions(version1, version1)}")
    
    # Increment versions
    print(f"Increment '{version1}' with PATCH change: {SemanticVersioning.increment_version(version1, VersionChangeType.PATCH)}")
    print(f"Increment '{version1}' with MINOR change: {SemanticVersioning.increment_version(version1, VersionChangeType.MINOR)}")
    print(f"Increment '{version1}' with MAJOR change: {SemanticVersioning.increment_version(version1, VersionChangeType.MAJOR)}")


def example_auto_version_incrementation():
    """Example of using automatic version incrementation."""
    print("\n=== Example: Automatic Version Incrementation ===\n")
    
    # Initialize the model registry and enhanced versioning
    registry = ModelRegistry()
    versioning = EnhancedModelVersioning(registry=registry)
    
    # Create and register the initial model (v1.0.0)
    model_v1, _, _, eval_results = create_sample_model(n_estimators=100)
    
    metadata_v1 = {
        "model_type": "random_forest",
        "metrics": {
            "mse": eval_results["mse"],
            "rmse": eval_results["rmse"],
            "r2": eval_results["r2"]
        },
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    model_id = registry.register_model(
        model=model_v1,
        model_name="enhanced_example_model",
        model_version="1.0.0",
        description="Initial model version",
        tags=["example", "enhanced", "regression"],
        metadata=metadata_v1
    )
    
    print(f"Registered initial model with ID: {model_id}")
    
    # Create a model with patch changes (only metrics changed)
    model_v2, _, _, eval_results = create_sample_model(n_estimators=100)
    
    metadata_v2 = {
        "model_type": "random_forest",
        "metrics": {
            "mse": eval_results["mse"] * 0.95,  # Slightly better metrics
            "rmse": np.sqrt(eval_results["mse"] * 0.95),
            "r2": eval_results["r2"] * 1.02
        },
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    # Auto-increment version (should be a PATCH increment to 1.0.1)
    new_version = versioning.auto_increment_version(
        model_name="enhanced_example_model",
        current_version="1.0.0",
        metadata=metadata_v2,
        previous_metadata=metadata_v1
    )
    
    model_id = registry.register_model(
        model=model_v2,
        model_name="enhanced_example_model",
        model_version=new_version,
        description="Model with improved metrics",
        tags=["example", "enhanced", "regression"],
        metadata=metadata_v2
    )
    
    print(f"Registered model with patch changes, new version: {new_version}, ID: {model_id}")
    
    # Create a model with minor changes (hyperparameters changed)
    model_v3, _, _, eval_results = create_sample_model(n_estimators=200)
    
    metadata_v3 = {
        "model_type": "random_forest",
        "metrics": {
            "mse": eval_results["mse"],
            "rmse": eval_results["rmse"],
            "r2": eval_results["r2"]
        },
        "hyperparameters": {
            "n_estimators": 200,  # Changed hyperparameter
            "random_state": 42
        },
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    # Auto-increment version (should be a MINOR increment to 1.1.0)
    new_version = versioning.auto_increment_version(
        model_name="enhanced_example_model",
        current_version=new_version,  # Use the last version
        metadata=metadata_v3,
        previous_metadata=metadata_v2
    )
    
    model_id = registry.register_model(
        model=model_v3,
        model_name="enhanced_example_model",
        model_version=new_version,
        description="Model with changed hyperparameters",
        tags=["example", "enhanced", "regression"],
        metadata=metadata_v3
    )
    
    print(f"Registered model with minor changes, new version: {new_version}, ID: {model_id}")
    
    # Create a model with major changes (interface changed)
    model_v4, _, _, eval_results = create_sample_model(model_type="gradient_boosting")
    
    metadata_v4 = {
        "model_type": "gradient_boosting",  # Changed model type
        "metrics": {
            "mse": eval_results["mse"],
            "rmse": eval_results["rmse"],
            "r2": eval_results["r2"]
        },
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    # Define interface changes
    interface_changes = {
        "model_type_changed": True,
        "prediction_method_signature_changed": True
    }
    
    # Auto-increment version (should be a MAJOR increment to 2.0.0)
    new_version = versioning.auto_increment_version(
        model_name="enhanced_example_model",
        current_version=new_version,  # Use the last version
        metadata=metadata_v4,
        previous_metadata=metadata_v3,
        interface_changes=interface_changes
    )
    
    model_id = registry.register_model(
        model=model_v4,
        model_name="enhanced_example_model",
        model_version=new_version,
        description="Model with changed interface",
        tags=["example", "enhanced", "regression"],
        metadata=metadata_v4
    )
    
    print(f"Registered model with major changes, new version: {new_version}, ID: {model_id}")
    
    # Get version history
    history = versioning.get_version_history("enhanced_example_model")
    print(f"\nVersion history: {history}")
    
    return versioning


def example_version_migration():
    """Example of using version migration tools."""
    print("\n=== Example: Version Migration Tools ===\n")
    
    # Initialize the model registry and enhanced versioning
    registry = ModelRegistry()
    versioning = EnhancedModelVersioning(registry=registry)
    
    # Create and register models with different interfaces
    # Model v1: RandomForestRegressor with 10 features
    model_v1, X_test_v1, _, _ = create_sample_model(model_type="random_forest")
    
    metadata_v1 = {
        "model_type": "random_forest",
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    registry.register_model(
        model=model_v1,
        model_name="migration_example_model",
        model_version="1.0.0",
        description="Initial model version",
        metadata=metadata_v1
    )
    
    # Model v2: GradientBoostingRegressor with different interface
    model_v2, _, _, _ = create_sample_model(model_type="gradient_boosting")
    
    metadata_v2 = {
        "model_type": "gradient_boosting",
        "input_schema": {"features": 10},
        "output_schema": {"target": 1}
    }
    
    registry.register_model(
        model=model_v2,
        model_name="migration_example_model",
        model_version="2.0.0",
        description="Model with changed interface",
        metadata=metadata_v2
    )
    
    # Define a migration adapter function
    def random_forest_to_gradient_boosting_adapter(model_data):
        """Adapter function to migrate from RandomForest to GradientBoosting."""
        # In a real scenario, this would handle more complex transformations
        # For this example, we'll just update the model_type in metadata
        model_data["metadata"]["model_type"] = "gradient_boosting"
        return model_data
    
    # Define an input transformation function for backward compatibility
    def transform_inputs(*args, **kwargs):
        """Transform inputs for backward compatibility."""
        # In a real scenario, this might transform feature formats, etc.
        # For this example, we'll just pass through the inputs
        return args, kwargs
    
    # Define an output transformation function for backward compatibility
    def transform_outputs(output):
        """Transform outputs for backward compatibility."""
        # In a real scenario, this might transform prediction formats, etc.
        # For this example, we'll just pass through the output
        return output
    
    # Register the migration adapter
    versioning.register_migration_adapter(
        model_name="migration_example_model",
        from_version="1.0.0",
        to_version="2.0.0",
        adapter_fn=random_forest_to_gradient_boosting_adapter
    )
    
    # Create a backward compatibility wrapper
    wrapped_model = versioning.create_backward_compatibility_wrapper(
        model_name="migration_example_model",
        model_version="2.0.0",
        target_version="1.0.0",
        transform_inputs_fn=transform_inputs,
        transform_outputs_fn=transform_outputs
    )
    
    # Test the wrapped model
    print("Testing backward compatibility wrapper:")
    original_prediction = model_v1.predict(X_test_v1[0:1])
    wrapped_prediction = wrapped_model.predict(X_test_v1[0:1])
    print(f"Original model prediction: {original_prediction}")
    print(f"Wrapped model prediction: {wrapped_prediction}")
    
    # Find migration path
    migration_path = versioning.find_migration_path(
        model_name="migration_example_model",
        from_version="1.0.0",
        to_version="2.0.0"
    )
    print(f"\nMigration path: {migration_path}")
    
    # Migrate model
    migrated_model_data = versioning.apply_migration(
        model_name="migration_example_model",
        from_version="1.0.0",
        to_version="2.0.0",
        model_data={
            "model": model_v1,
            "metadata": metadata_v1
        }
    )
    
    print(f"\nMigrated model metadata: {migrated_model_data['metadata']}")
    
    return versioning


def main():
    """Main function to run all examples."""
    print("=== Enhanced Model Versioning Examples ===\n")
    
    # Run examples
    example_semantic_versioning()
    versioning = example_auto_version_incrementation()
    example_version_migration()
    
    print("\n=== Examples Completed ===")


if __name__ == "__main__":
    main()