"""Market data model trainer module for the Friday AI Trading System.

This module provides functionality for training machine learning models
using processed market data for trading strategies.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class ModelType:
    """Class for model types."""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "svr"


class PredictionTarget:
    """Class for prediction targets."""
    NEXT_CLOSE = "next_close"
    PRICE_DIRECTION = "price_direction"
    RETURN = "return"
    VOLATILITY = "volatility"


class MarketDataModelTrainer:
    """Class for training machine learning models using processed market data.

    This class provides methods for loading processed market data, preparing features
    and targets, training models, and evaluating their performance.

    Attributes:
        config: Configuration manager.
        processed_data_dir: Directory containing processed market data files.
        models_dir: Directory to save trained models.
        evaluation_dir: Directory to save model evaluation results.
        test_size: Fraction of data to use for testing.
        random_state: Random state for reproducibility.
        models: Dictionary of trained models.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        processed_data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        evaluation_dir: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize a market data model trainer.

        Args:
            config: Configuration manager. If None, a new one will be created.
            processed_data_dir: Directory containing processed market data files. If None, will use config value.
            models_dir: Directory to save trained models. If None, will use config value.
            evaluation_dir: Directory to save model evaluation results. If None, will use config value.
            test_size: Fraction of data to use for testing. Defaults to 0.2.
            random_state: Random state for reproducibility. Defaults to 42.
        """
        self.config = config or ConfigManager()
        
        # Set directories
        self.processed_data_dir = processed_data_dir or self.config.get("data.processed.directory", "src/data/processed")
        self.models_dir = models_dir or self.config.get("models.directory", "src/models")
        self.evaluation_dir = evaluation_dir or self.config.get("models.evaluation.directory", "src/models/evaluation")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Set parameters
        self.test_size = test_size
        self.random_state = random_state
        
        # Dictionary to store trained models
        self.models: Dict[str, Any] = {}
        
        logger.info(f"Initialized MarketDataModelTrainer with processed data directory: {self.processed_data_dir}")

    def discover_processed_data(self) -> Dict[str, Dict[str, str]]:
        """Discover processed data files.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping symbols to dictionaries mapping timeframes to file paths.
        """
        logger.info(f"Discovering processed data files in {self.processed_data_dir}")
        
        # Find all processed CSV files
        csv_files = glob.glob(os.path.join(self.processed_data_dir, "**/*_processed.csv"), recursive=True)
        logger.info(f"Found {len(csv_files)} processed CSV files")
        
        # Group files by symbol and timeframe
        symbol_timeframe_files: Dict[str, Dict[str, str]] = {}
        for file_path in csv_files:
            # Extract symbol and timeframe from filename
            filename = os.path.basename(file_path)
            parts = filename.split("_")
            if len(parts) >= 3 and parts[-1] == "processed.csv":
                symbol = parts[0]
                timeframe = parts[1]
                
                if symbol not in symbol_timeframe_files:
                    symbol_timeframe_files[symbol] = {}
                symbol_timeframe_files[symbol][timeframe] = file_path
        
        logger.info(f"Grouped files for {len(symbol_timeframe_files)} symbols")
        return symbol_timeframe_files

    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """Load processed data from a CSV file.

        Args:
            file_path: Path to the processed CSV file.

        Returns:
            pd.DataFrame: The loaded data.
        """
        logger.debug(f"Loading processed data from {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        return df

    def prepare_features_and_target(
        self,
        data: pd.DataFrame,
        target_type: str = PredictionTarget.NEXT_CLOSE,
        forecast_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training.

        Args:
            data: The processed data.
            target_type: The type of prediction target. Defaults to PredictionTarget.NEXT_CLOSE.
            forecast_horizon: The number of periods ahead to forecast. Defaults to 1.
            feature_columns: Optional list of feature columns to use. If None, will use all columns except target.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: The features and target.
        """
        # Create target variable based on target_type
        if target_type == PredictionTarget.NEXT_CLOSE:
            # Predict the close price n periods ahead
            target = data['close'].shift(-forecast_horizon)
            target_name = f'close_t+{forecast_horizon}'
        
        elif target_type == PredictionTarget.PRICE_DIRECTION:
            # Predict the direction of price movement (1 for up, 0 for down)
            future_close = data['close'].shift(-forecast_horizon)
            target = (future_close > data['close']).astype(int)
            target_name = f'direction_t+{forecast_horizon}'
        
        elif target_type == PredictionTarget.RETURN:
            # Predict the return n periods ahead
            future_close = data['close'].shift(-forecast_horizon)
            target = (future_close - data['close']) / data['close']
            target_name = f'return_t+{forecast_horizon}'
        
        elif target_type == PredictionTarget.VOLATILITY:
            # Predict the volatility (high-low range) n periods ahead
            future_high = data['high'].shift(-forecast_horizon)
            future_low = data['low'].shift(-forecast_horizon)
            target = (future_high - future_low) / data['close']
            target_name = f'volatility_t+{forecast_horizon}'
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Select feature columns
        if feature_columns is None:
            # Use all columns except close if predicting close
            if target_type == PredictionTarget.NEXT_CLOSE:
                feature_columns = [col for col in data.columns if col != 'close']
            else:
                feature_columns = list(data.columns)
        
        # Create features DataFrame
        features = data[feature_columns].copy()
        
        # Remove rows with NaN in target
        valid_indices = ~target.isna()
        features = features.loc[valid_indices]
        target = target.loc[valid_indices]
        
        # Name the target series
        target.name = target_name
        
        return features, target

    def train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str = ModelType.RANDOM_FOREST,
        hyperparameter_tuning: bool = True,
        cv_folds: int = 5,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a machine learning model.

        Args:
            features: The feature data.
            target: The target data.
            model_type: The type of model to train. Defaults to ModelType.RANDOM_FOREST.
            hyperparameter_tuning: Whether to perform hyperparameter tuning. Defaults to True.
            cv_folds: Number of cross-validation folds for hyperparameter tuning. Defaults to 5.

        Returns:
            Tuple[Any, Dict[str, Any]]: The trained model and training results.
        """
        logger.info(f"Training {model_type} model with {len(features)} samples")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create pipeline with preprocessing and model
        if model_type == ModelType.LINEAR:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            param_grid = {}
        
        elif model_type == ModelType.RIDGE:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=self.random_state))
            ])
            param_grid = {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        
        elif model_type == ModelType.LASSO:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(random_state=self.random_state))
            ])
            param_grid = {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        
        elif model_type == ModelType.RANDOM_FOREST:
            model = Pipeline([
                ('model', RandomForestRegressor(random_state=self.random_state))
            ])
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        
        elif model_type == ModelType.GRADIENT_BOOSTING:
            model = Pipeline([
                ('model', GradientBoostingRegressor(random_state=self.random_state))
            ])
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        
        elif model_type == ModelType.SVR:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR())
            ])
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__gamma': ['scale', 'auto', 0.1, 0.01],
                'model__kernel': ['rbf', 'linear']
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning and param_grid:
            logger.info(f"Performing hyperparameter tuning with {cv_folds}-fold cross-validation")
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Create grid search
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            # Fit model without hyperparameter tuning
            model.fit(X_train, y_train)
            best_params = {}
        
        # Evaluate model on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create results dictionary
        results = {
            'model_type': model_type,
            'target_name': target.name,
            'n_samples': len(features),
            'n_features': features.shape[1],
            'feature_names': list(features.columns),
            'best_params': best_params,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return model, results

    def save_model(self, model: Any, symbol: str, timeframe: str, target_type: str, results: Dict[str, Any]) -> str:
        """Save a trained model and its evaluation results.

        Args:
            model: The trained model to save.
            symbol: The symbol the model was trained on.
            timeframe: The timeframe the model was trained on.
            target_type: The type of prediction target.
            results: The model evaluation results.

        Returns:
            str: The path to the saved model file.
        """
        # Create model filename
        model_filename = f"{symbol}_{timeframe}_{target_type}_{results['model_type']}_model.joblib"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Create results filename
        results_filename = f"{symbol}_{timeframe}_{target_type}_{results['model_type']}_results.json"
        results_path = os.path.join(self.evaluation_dir, results_filename)
        
        # Save results
        pd.Series(results).to_json(results_path)
        logger.info(f"Saved results to {results_path}")
        
        return model_path

    def train_models_for_symbol(
        self,
        symbol: str,
        timeframe_files: Dict[str, str],
        model_types: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        forecast_horizons: Optional[List[int]] = None,
    ) -> Dict[str, str]:
        """Train models for a single symbol across different timeframes.

        Args:
            symbol: The symbol to train models for.
            timeframe_files: Dictionary mapping timeframes to file paths.
            model_types: Optional list of model types to train. If None, will use default types.
            target_types: Optional list of target types to predict. If None, will use default types.
            forecast_horizons: Optional list of forecast horizons. If None, will use default horizons.

        Returns:
            Dict[str, str]: Dictionary mapping model identifiers to model file paths.
        """
        logger.info(f"Training models for symbol: {symbol} with {len(timeframe_files)} timeframes")
        
        # Set defaults if not provided
        if model_types is None:
            model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]
        
        if target_types is None:
            target_types = [PredictionTarget.NEXT_CLOSE, PredictionTarget.PRICE_DIRECTION]
        
        if forecast_horizons is None:
            forecast_horizons = [1, 5, 10]
        
        # Dictionary to store model file paths
        model_paths: Dict[str, str] = {}
        
        # Train models for each timeframe
        for timeframe, file_path in timeframe_files.items():
            logger.info(f"Processing timeframe: {timeframe} for symbol: {symbol}")
            
            # Load processed data
            data = self.load_processed_data(file_path)
            
            # Skip if not enough data
            if len(data) < 100:  # Arbitrary minimum, adjust as needed
                logger.warning(f"Not enough data for {symbol} {timeframe}, skipping")
                continue
            
            # Train models for each target type and forecast horizon
            for target_type in target_types:
                for horizon in forecast_horizons:
                    # Skip long horizons for short timeframes
                    if (timeframe in ['1m', '5m'] and horizon > 5) or \
                       (timeframe in ['15m', '30m'] and horizon > 10):
                        continue
                    
                    logger.info(f"Preparing features for {target_type} with horizon {horizon}")
                    
                    try:
                        # Prepare features and target
                        features, target = self.prepare_features_and_target(
                            data, target_type=target_type, forecast_horizon=horizon
                        )
                        
                        # Skip if not enough valid samples
                        if len(features) < 100:  # Arbitrary minimum, adjust as needed
                            logger.warning(f"Not enough valid samples for {symbol} {timeframe} {target_type} h={horizon}, skipping")
                            continue
                        
                        # Train models for each model type
                        for model_type in model_types:
                            model_id = f"{symbol}_{timeframe}_{target_type}_h{horizon}_{model_type}"
                            logger.info(f"Training model: {model_id}")
                            
                            try:
                                # Train model
                                model, results = self.train_model(
                                    features, target, model_type=model_type
                                )
                                
                                # Save model
                                model_path = self.save_model(
                                    model, symbol, timeframe, f"{target_type}_h{horizon}", results
                                )
                                
                                # Store model path
                                model_paths[model_id] = model_path
                                
                                # Store model in memory
                                self.models[model_id] = model
                                
                            except Exception as e:
                                logger.error(f"Error training model {model_id}: {str(e)}")
                                logger.error(traceback.format_exc())
                                continue
                    
                    except Exception as e:
                        logger.error(f"Error preparing features for {symbol} {timeframe} {target_type} h={horizon}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
        
        return model_paths

    def train_all_models(self) -> Dict[str, Dict[str, str]]:
        """Train models for all symbols and timeframes.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping symbols to dictionaries mapping model identifiers to model file paths.
        """
        # Discover processed data files
        symbol_timeframe_files = self.discover_processed_data()
        
        # Dictionary to store model file paths by symbol
        all_model_paths: Dict[str, Dict[str, str]] = {}
        
        # Train models for each symbol
        for symbol, timeframe_files in symbol_timeframe_files.items():
            try:
                model_paths = self.train_models_for_symbol(symbol, timeframe_files)
                all_model_paths[symbol] = model_paths
            except Exception as e:
                logger.error(f"Error training models for symbol {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        return all_model_paths


def main():
    """Main function to run the market data model trainer."""
    # Initialize the market data model trainer
    trainer = MarketDataModelTrainer()
    
    # Train all models
    trainer.train_all_models()
    
    logger.info("Model training completed")


if __name__ == "__main__":
    main()