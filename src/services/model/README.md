# Friday AI Trading System - Model Registry

## Overview

The Model Registry is a comprehensive system for managing machine learning models in the Friday AI Trading System. It provides functionality for model registration, versioning, serialization, loading, and metadata management.

## Components

### ModelRegistry

The core component that manages model storage, retrieval, and metadata. It provides methods for:

- Registering models
- Loading models
- Retrieving models by name and version
- Getting the latest version of a model
- Managing model metadata
- Deleting models
- Listing models
- Getting model versions
- Setting model status
- Exporting and importing models

### ModelOperations

Provides higher-level operations on models, including:

- Loading models
- Retrieving models by name and version
- Getting the latest version of a model
- Managing model metadata
- Deleting models
- Listing models
- Getting model versions
- Setting model status
- Exporting and importing models
- Comparing models
- Finding the best model based on a metric

### ModelVersioning

Manages model versioning, including:

- Getting model lineage
- Comparing model versions
- Calculating time differences between versions
- Getting version history
- Getting the latest version
- Creating a diff between two versions
- Tagging and untagging versions
- Getting versions by tag

### ModelSerializer

Handles model serialization and deserialization, including:

- Serializing models to files
- Deserializing models from files
- Computing model hashes
- Checking model compatibility
- Getting serialized size
- Serializing and deserializing to/from bytes
- Getting model file information

### ModelTrainerIntegration

Integrates model training with the registry, including:

- Registering trained models
- Registering models from files
- Registering associated information such as evaluation results, training data info, hyperparameters, feature importance, pipeline steps, and dependencies

### ModelLoader

Provides a simple interface for loading models from the registry, including:

- Loading models by name and version
- Loading models by tags
- Loading the best model based on a metric
- Loading models for prediction

### MarketDataModelTrainer

Trains machine learning models on market data, including:

- Discovering processed data files
- Loading processed data
- Preparing features and targets
- Training models
- Saving models and evaluation results
- Training models for a symbol
- Training models for all symbols

## Database Integration

The Model Registry is integrated with the database using SQLAlchemy models:

- `ModelMetadata`: Stores model metadata
- `ModelTag`: Stores model tags
- `ModelMetric`: Stores model metric history

## Usage Examples

### Registering a Model

```python
from src.services.model import ModelRegistry

# Initialize the registry
registry = ModelRegistry()

# Register a model
model_id = registry.register_model(
    model=my_model,
    model_name="my_model",
    model_version="1.0.0",
    description="My first model",
    tags=["tag1", "tag2"]
)
```

### Loading a Model

```python
from src.services.model import load_model

# Load the latest version of a model
model = load_model("my_model")

# Load a specific version of a model
model = load_model("my_model", "1.0.0")
```

### Loading a Model for Prediction

```python
from src.services.model import load_model_for_prediction

# Load a model for prediction
model = load_model_for_prediction(
    symbol="AAPL",
    timeframe="1d",
    target_type="next_close",
    forecast_horizon=1,
    model_type="random_forest"
)
```

### Training Models

```python
from src.services.model import MarketDataModelTrainer, ModelType, PredictionTarget

# Initialize the trainer
trainer = MarketDataModelTrainer(use_model_registry=True)

# Train models for a symbol
results = trainer.train_models_for_symbol(
    symbol="AAPL",
    timeframes=["1d", "1h"],
    target_types=[PredictionTarget.NEXT_CLOSE, PredictionTarget.PRICE_DIRECTION],
    forecast_horizons=[1, 5, 10],
    model_types=[ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING],
    tune_hyperparams=True
)
```

## Best Practices

1. **Use Semantic Versioning**: Follow semantic versioning (MAJOR.MINOR.PATCH) for model versions.
2. **Add Descriptive Tags**: Use tags to categorize models and make them easier to find.
3. **Include Comprehensive Metadata**: Add detailed metadata to models, including evaluation results, training data info, hyperparameters, and dependencies.
4. **Use the Model Registry for All Models**: Register all models in the registry, even if they are not used in production.
5. **Compare Models Before Deployment**: Use the model comparison functionality to compare models before deploying them to production.
6. **Monitor Model Performance**: Use the model metrics functionality to monitor model performance over time.
7. **Export Models for Backup**: Regularly export models for backup and disaster recovery.
8. **Use the Model Loader for Prediction**: Use the model loader to load models for prediction, as it provides caching and error handling.
9. **Document Model Changes**: Add detailed descriptions when updating model versions to document changes.
10. **Clean Up Old Models**: Regularly delete old and unused models to save storage space.