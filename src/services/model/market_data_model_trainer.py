"""Market Data Model Trainer for Friday AI Trading System.

This module provides functionality for training machine learning models on market data.
"""

import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.infrastructure.logging import get_logger
from src.services.model.model_trainer_integration import ModelTrainerIntegration

# Create logger
logger = get_logger(__name__)


class ModelType(Enum):
    """Enum for model types."""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "svr"


class PredictionTarget(Enum):
    """Enum for prediction targets."""
    NEXT_CLOSE = "next_close"
    PRICE_DIRECTION = "price_direction"
    RETURN = "return"
    VOLATILITY = "volatility"


class MarketDataModelTrainer:
    """Trainer for market data models.

    This class provides functionality for training machine learning models on market data.

    Attributes:
        processed_data_dir: Directory containing processed market data.
        models_dir: Directory to store trained models.
        evaluation_dir: Directory to store model evaluation results.
        test_size: Proportion of data to use for testing.
        random_state: Random state for reproducibility.
        model_integration: Integration with model registry.
    """

    def __init__(
        self,
        processed_data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        evaluation_dir: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_model_registry: bool = True,
    ):
        """Initialize the market data model trainer.

        Args:
            processed_data_dir: Directory containing processed market data.
            models_dir: Directory to store trained models.
            evaluation_dir: Directory to store model evaluation results.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.
            use_model_registry: Whether to use the model registry.
        """
        self.processed_data_dir = processed_data_dir or os.path.join(os.getcwd(), "data", "processed")
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models")
        self.evaluation_dir = evaluation_dir or os.path.join(os.getcwd(), "evaluation")
        self.test_size = test_size
        self.random_state = random_state
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Initialize model registry integration if enabled
        self.use_model_registry = use_model_registry
        self.model_integration = ModelTrainerIntegration() if use_model_registry else None
        
        logger.info(f"Initialized MarketDataModelTrainer with processed data directory: {self.processed_data_dir}")

    def discover_processed_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover processed data files.

        Returns:
            Dict[str, Dict[str, List[str]]]: Dictionary of processed data files grouped by symbol and timeframe.
        """
        data_files = glob.glob(os.path.join(self.processed_data_dir, "**", "*.csv"), recursive=True)
        
        # Group files by symbol and timeframe
        grouped_files = {}
        for file_path in data_files:
            file_name = os.path.basename(file_path)
            parts = file_name.split("_")
            
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                
                if symbol not in grouped_files:
                    grouped_files[symbol] = {}
                
                if timeframe not in grouped_files[symbol]:
                    grouped_files[symbol][timeframe] = []
                
                grouped_files[symbol][timeframe].append(file_path)
        
        return grouped_files

    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """Load processed data from a file.

        Args:
            file_path: Path to the processed data file.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        data = pd.read_csv(file_path, parse_dates=["timestamp"])
        data.set_index("timestamp", inplace=True)
        
        return data

    def prepare_features_and_target(
        self,
        data: pd.DataFrame,
        target_type: PredictionTarget,
        forecast_horizon: int = 1,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training.

        Args:
            data: The processed data.
            target_type: The type of prediction target.
            forecast_horizon: The forecast horizon in periods.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: The features and target.

        Raises:
            ValueError: If the target type is not supported.
        """
        # Create target variable based on target type
        if target_type == PredictionTarget.NEXT_CLOSE:
            target = data["close"].shift(-forecast_horizon)
            target_name = f"next_close_{forecast_horizon}"
        elif target_type == PredictionTarget.PRICE_DIRECTION:
            target = (data["close"].shift(-forecast_horizon) > data["close"]).astype(int)
            target_name = f"price_direction_{forecast_horizon}"
        elif target_type == PredictionTarget.RETURN:
            target = data["close"].pct_change(forecast_horizon).shift(-forecast_horizon)
            target_name = f"return_{forecast_horizon}"
        elif target_type == PredictionTarget.VOLATILITY:
            target = data["close"].rolling(forecast_horizon).std().shift(-forecast_horizon)
            target_name = f"volatility_{forecast_horizon}"
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Select feature columns (exclude target and other non-feature columns)
        exclude_cols = ["open", "high", "low", "close", "volume", "vwap"]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Create features DataFrame
        features = data[feature_cols].copy()
        
        # Drop rows with NaN values in target or features
        valid_idx = ~(target.isna() | features.isna().any(axis=1))
        features = features[valid_idx]
        target = target[valid_idx]
        
        # Set target name
        target.name = target_name
        
        return features, target

    def train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: ModelType,
        tune_hyperparams: bool = False,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        """Train a model on the given features and target.

        Args:
            features: The features for training.
            target: The target for training.
            model_type: The type of model to train.
            tune_hyperparams: Whether to tune hyperparameters.

        Returns:
            Tuple[Pipeline, Dict[str, float]]: The trained model pipeline and evaluation results.

        Raises:
            ValueError: If the model type is not supported.
        """
        # Split data into training and testing sets
        train_size = int(len(features) * (1 - self.test_size))
        X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
        y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]
        
        # Create pipeline with preprocessing and model
        if model_type == ModelType.LINEAR:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])
            param_grid = {}
        elif model_type == ModelType.RIDGE:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(random_state=self.random_state))
            ])
            param_grid = {
                "model__alpha": [0.1, 1.0, 10.0, 100.0]
            }
        elif model_type == ModelType.LASSO:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(random_state=self.random_state))
            ])
            param_grid = {
                "model__alpha": [0.1, 1.0, 10.0, 100.0]
            }
        elif model_type == ModelType.RANDOM_FOREST:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(random_state=self.random_state))
            ])
            param_grid = {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20, 30]
            }
        elif model_type == ModelType.GRADIENT_BOOSTING:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(random_state=self.random_state))
            ])
            param_grid = {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2]
            }
        elif model_type == ModelType.SVR:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR())
            ])
            param_grid = {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto", 0.1, 0.01]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Tune hyperparameters if requested
        if tune_hyperparams and param_grid:
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_params = {}
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create evaluation results
        eval_results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "test_size": len(y_test),
            "train_size": len(y_train)
        }
        
        # Add hyperparameters to evaluation results
        if best_params:
            eval_results["best_params"] = best_params
        
        return model, eval_results

    def save_model(
        self,
        model: Pipeline,
        eval_results: Dict[str, float],
        symbol: str,
        timeframe: str,
        target_type: PredictionTarget,
        forecast_horizon: int,
        model_type: ModelType,
    ) -> str:
        """Save a trained model and evaluation results.

        Args:
            model: The trained model pipeline.
            eval_results: The evaluation results.
            symbol: The symbol for which the model was trained.
            timeframe: The timeframe for which the model was trained.
            target_type: The type of prediction target.
            forecast_horizon: The forecast horizon in periods.
            model_type: The type of model.

        Returns:
            str: The path to the saved model file.
        """
        # Create model directory
        model_dir = os.path.join(self.models_dir, symbol, timeframe)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create evaluation directory
        eval_dir = os.path.join(self.evaluation_dir, symbol, timeframe)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create file names
        model_name = f"{symbol}_{timeframe}_{target_type.value}_{forecast_horizon}_{model_type.value}"
        model_file = os.path.join(model_dir, f"{model_name}.joblib")
        eval_file = os.path.join(eval_dir, f"{model_name}_eval.json")
        
        # Save model
        joblib.dump(model, model_file)
        
        # Save evaluation results
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Register model in registry if enabled
        if self.use_model_registry and self.model_integration is not None:
            # Extract feature importance if available
            feature_importance = None
            if hasattr(model.named_steps["model"], "feature_importances_"):
                feature_names = list(model.named_steps["model"].feature_names_in_)
                importance_values = model.named_steps["model"].feature_importances_
                feature_importance = dict(zip(feature_names, importance_values.tolist()))
            
            # Extract pipeline steps
            pipeline_steps = []
            for name, step in model.named_steps.items():
                step_info = {"name": name, "type": type(step).__name__}
                if hasattr(step, "get_params"):
                    step_info["params"] = step.get_params()
                pipeline_steps.append(step_info)
            
            # Create training data info
            training_data_info = {
                "symbol": symbol,
                "timeframe": timeframe,
                "target_type": target_type.value,
                "forecast_horizon": forecast_horizon,
                "train_size": eval_results.get("train_size", 0),
                "test_size": eval_results.get("test_size", 0)
            }
            
            # Extract hyperparameters
            hyperparameters = {}
            for name, step in model.named_steps.items():
                if hasattr(step, "get_params"):
                    hyperparameters[name] = step.get_params()
            
            # Register model in registry
            model_id = self.model_integration.register_model_from_trainer(
                model=model,
                model_name=model_name,
                model_type=model_type.value,
                evaluation_results={
                    "metrics": {
                        "mse": eval_results.get("mse", 0),
                        "rmse": eval_results.get("rmse", 0),
                        "mae": eval_results.get("mae", 0),
                        "r2": eval_results.get("r2", 0)
                    },
                    "details": eval_results
                },
                training_data_info=training_data_info,
                hyperparameters=hyperparameters,
                feature_importance=feature_importance,
                pipeline_steps=pipeline_steps,
                dependencies={
                    "scikit-learn": "1.0.2",
                    "numpy": "1.21.5",
                    "pandas": "1.3.5"
                },
                tags=[symbol, timeframe, target_type.value, f"horizon_{forecast_horizon}", model_type.value],
                description=f"Model for predicting {target_type.value} with {forecast_horizon} period horizon for {symbol} on {timeframe} timeframe using {model_type.value}"
            )
            
            logger.info(f"Registered model in registry with ID: {model_id}")
        
        logger.info(f"Saved model to {model_file} and evaluation results to {eval_file}")
        
        return model_file

    def train_models_for_symbol(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        target_types: Optional[List[PredictionTarget]] = None,
        forecast_horizons: Optional[List[int]] = None,
        model_types: Optional[List[ModelType]] = None,
        tune_hyperparams: bool = False,
    ) -> Dict[str, Any]:
        """Train models for a symbol.

        Args:
            symbol: The symbol to train models for.
            timeframes: The timeframes to train models for. If None, all available timeframes will be used.
            target_types: The target types to train models for. If None, all target types will be used.
            forecast_horizons: The forecast horizons to train models for. If None, [1, 5, 10] will be used.
            model_types: The model types to train. If None, [RANDOM_FOREST, GRADIENT_BOOSTING] will be used.
            tune_hyperparams: Whether to tune hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary of trained models and their file paths.
        """
        # Discover processed data files for the symbol
        all_data_files = self.discover_processed_data()
        if symbol not in all_data_files:
            logger.warning(f"No processed data found for symbol: {symbol}")
            return {}
        
        # Get available timeframes for the symbol
        available_timeframes = list(all_data_files[symbol].keys())
        if not timeframes:
            timeframes = available_timeframes
        else:
            timeframes = [tf for tf in timeframes if tf in available_timeframes]
        
        if not timeframes:
            logger.warning(f"No matching timeframes found for symbol: {symbol}")
            return {}
        
        # Set default values if not provided
        if not target_types:
            target_types = list(PredictionTarget)
        
        if not forecast_horizons:
            forecast_horizons = [1, 5, 10]
        
        if not model_types:
            model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]
        
        # Initialize results dictionary
        results = {}
        
        # Train models for each timeframe, target type, forecast horizon, and model type
        for timeframe in timeframes:
            logger.info(f"Training models for {symbol} on {timeframe} timeframe")
            
            # Load data for the timeframe
            data_files = all_data_files[symbol][timeframe]
            if not data_files:
                logger.warning(f"No data files found for {symbol} on {timeframe} timeframe")
                continue
            
            # Use the most recent data file
            data_file = sorted(data_files)[-1]
            data = self.load_processed_data(data_file)
            
            # Train models for each target type, forecast horizon, and model type
            for target_type in target_types:
                for forecast_horizon in forecast_horizons:
                    # Prepare features and target
                    try:
                        features, target = self.prepare_features_and_target(
                            data,
                            target_type,
                            forecast_horizon
                        )
                    except Exception as e:
                        logger.error(f"Error preparing features and target for {symbol} on {timeframe} timeframe with {target_type.value} target and {forecast_horizon} horizon: {str(e)}")
                        continue
                    
                    # Skip if not enough data
                    if len(features) < 100:
                        logger.warning(f"Not enough data for {symbol} on {timeframe} timeframe with {target_type.value} target and {forecast_horizon} horizon")
                        continue
                    
                    for model_type in model_types:
                        # Train model
                        try:
                            model, eval_results = self.train_model(
                                features,
                                target,
                                model_type,
                                tune_hyperparams
                            )
                        except Exception as e:
                            logger.error(f"Error training {model_type.value} model for {symbol} on {timeframe} timeframe with {target_type.value} target and {forecast_horizon} horizon: {str(e)}")
                            continue
                        
                        # Save model and evaluation results
                        try:
                            model_file = self.save_model(
                                model,
                                eval_results,
                                symbol,
                                timeframe,
                                target_type,
                                forecast_horizon,
                                model_type
                            )
                        except Exception as e:
                            logger.error(f"Error saving {model_type.value} model for {symbol} on {timeframe} timeframe with {target_type.value} target and {forecast_horizon} horizon: {str(e)}")
                            continue
                        
                        # Store model path and model in results
                        model_key = f"{symbol}_{timeframe}_{target_type.value}_{forecast_horizon}_{model_type.value}"
                        results[model_key] = {
                            "model": model,
                            "model_file": model_file,
                            "eval_results": eval_results
                        }
                        
                        logger.info(f"Trained and saved {model_type.value} model for {symbol} on {timeframe} timeframe with {target_type.value} target and {forecast_horizon} horizon")
        
        return results

    def train_all_models(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        target_types: Optional[List[PredictionTarget]] = None,
        forecast_horizons: Optional[List[int]] = None,
        model_types: Optional[List[ModelType]] = None,
        tune_hyperparams: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Train models for all symbols.

        Args:
            symbols: The symbols to train models for. If None, all available symbols will be used.
            timeframes: The timeframes to train models for. If None, all available timeframes will be used.
            target_types: The target types to train models for. If None, all target types will be used.
            forecast_horizons: The forecast horizons to train models for. If None, [1, 5, 10] will be used.
            model_types: The model types to train. If None, [RANDOM_FOREST, GRADIENT_BOOSTING] will be used.
            tune_hyperparams: Whether to tune hyperparameters.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of trained models and their file paths for each symbol.
        """
        # Discover processed data files
        all_data_files = self.discover_processed_data()
        
        # Get available symbols
        available_symbols = list(all_data_files.keys())
        if not symbols:
            symbols = available_symbols
        else:
            symbols = [sym for sym in symbols if sym in available_symbols]
        
        if not symbols:
            logger.warning("No matching symbols found")
            return {}
        
        # Initialize results dictionary
        all_results = {}
        
        # Train models for each symbol
        for symbol in symbols:
            try:
                results = self.train_models_for_symbol(
                    symbol,
                    timeframes,
                    target_types,
                    forecast_horizons,
                    model_types,
                    tune_hyperparams
                )
                all_results[symbol] = results
            except Exception as e:
                logger.error(f"Error training models for symbol {symbol}: {str(e)}")
        
        return all_results


def main():
    """Main function for training market data models."""
    # Initialize trainer
    trainer = MarketDataModelTrainer(use_model_registry=True)
    
    # Train models for all symbols
    trainer.train_all_models()


if __name__ == "__main__":
    main()