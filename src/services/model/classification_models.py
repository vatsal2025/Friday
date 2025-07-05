"""Classification Models Support for Friday AI Trading System.

This module provides functionality for handling classification models including
model creation, evaluation, and specialized metrics for classification tasks.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum, auto

from src.infrastructure.logging import get_logger
from src.services.model.model_registry_config import ModelType, PredictionTarget

# Create logger
logger = get_logger(__name__)


class ClassificationType(Enum):
    """Enum for different types of classification problems."""
    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()


class ClassificationMetrics:
    """Utility class for computing classification-specific metrics.
    
    This class provides methods for calculating common classification metrics
    such as accuracy, precision, recall, F1 score, and confusion matrix.
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            
        Returns:
            float: Accuracy score between 0 and 1.
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
        """Calculate precision score.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            average: Averaging strategy for multiclass/multilabel classification.
                     Options: 'binary', 'micro', 'macro', 'weighted', None (returns per-class scores).
            
        Returns:
            Union[float, np.ndarray]: Precision score(s).
        """
        try:
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average=average)
        except ImportError:
            logger.warning("scikit-learn not installed. Using basic precision implementation.")
            # Basic implementation for binary classification
            if average != 'binary':
                logger.warning(f"Average type '{average}' not supported in basic implementation. Using binary.")
            
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            predicted_positives = np.sum(y_pred == 1)
            
            if predicted_positives == 0:
                return 0.0
            
            return true_positives / predicted_positives
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
        """Calculate recall score.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            average: Averaging strategy for multiclass/multilabel classification.
                     Options: 'binary', 'micro', 'macro', 'weighted', None (returns per-class scores).
            
        Returns:
            Union[float, np.ndarray]: Recall score(s).
        """
        try:
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average=average)
        except ImportError:
            logger.warning("scikit-learn not installed. Using basic recall implementation.")
            # Basic implementation for binary classification
            if average != 'binary':
                logger.warning(f"Average type '{average}' not supported in basic implementation. Using binary.")
            
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            actual_positives = np.sum(y_true == 1)
            
            if actual_positives == 0:
                return 0.0
            
            return true_positives / actual_positives
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
        """Calculate F1 score.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            average: Averaging strategy for multiclass/multilabel classification.
                     Options: 'binary', 'micro', 'macro', 'weighted', None (returns per-class scores).
            
        Returns:
            Union[float, np.ndarray]: F1 score(s).
        """
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average=average)
        except ImportError:
            logger.warning("scikit-learn not installed. Using basic F1 implementation.")
            # Basic implementation for binary classification
            if average != 'binary':
                logger.warning(f"Average type '{average}' not supported in basic implementation. Using binary.")
            
            precision = ClassificationMetrics.precision(y_true, y_pred)
            recall = ClassificationMetrics.recall(y_true, y_pred)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            
        Returns:
            np.ndarray: Confusion matrix.
        """
        try:
            from sklearn.metrics import confusion_matrix
            return confusion_matrix(y_true, y_pred)
        except ImportError:
            logger.warning("scikit-learn not installed. Using basic confusion matrix implementation.")
            # Basic implementation for binary classification
            classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(classes)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            
            for i, true_class in enumerate(classes):
                for j, pred_class in enumerate(classes):
                    cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
            
            return cm
    
    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro') -> float:
        """Calculate ROC AUC score.
        
        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities or decision function scores.
            average: Averaging strategy for multiclass classification.
                     Options: 'macro', 'weighted', 'micro'.
            
        Returns:
            float: ROC AUC score.
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_score, average=average)
        except ImportError:
            logger.error("scikit-learn not installed. ROC AUC calculation requires scikit-learn.")
            raise NotImplementedError("ROC AUC calculation requires scikit-learn.")


class ClassificationModelEvaluator:
    """Evaluator for classification models.
    
    This class provides methods for evaluating classification models using
    appropriate metrics based on the classification type.
    """
    
    def __init__(self, classification_type: ClassificationType = ClassificationType.BINARY):
        """Initialize the classification model evaluator.
        
        Args:
            classification_type: Type of classification problem.
        """
        self.classification_type = classification_type
        logger.info(f"Initialized ClassificationModelEvaluator with type: {classification_type.name}")
    
    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate a classification model.
        
        Args:
            model: The classification model with a predict method.
            X: Feature data.
            y: Ground truth labels.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Get probability predictions if available
            try:
                y_prob = model.predict_proba(X)
                has_probabilities = True
            except (AttributeError, NotImplementedError):
                has_probabilities = False
            
            # Calculate metrics
            metrics = {}
            
            # Basic metrics for all classification types
            metrics['accuracy'] = ClassificationMetrics.accuracy(y, y_pred)
            
            # Metrics based on classification type
            if self.classification_type == ClassificationType.BINARY:
                metrics['precision'] = ClassificationMetrics.precision(y, y_pred)
                metrics['recall'] = ClassificationMetrics.recall(y, y_pred)
                metrics['f1'] = ClassificationMetrics.f1_score(y, y_pred)
                
                # ROC AUC if probabilities are available
                if has_probabilities:
                    try:
                        # For binary classification, we need the probability of the positive class
                        pos_class_idx = 1 if y_prob.shape[1] > 1 else 0
                        metrics['roc_auc'] = ClassificationMetrics.roc_auc_score(y, y_prob[:, pos_class_idx])
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            
            elif self.classification_type == ClassificationType.MULTICLASS:
                metrics['precision_macro'] = ClassificationMetrics.precision(y, y_pred, average='macro')
                metrics['recall_macro'] = ClassificationMetrics.recall(y, y_pred, average='macro')
                metrics['f1_macro'] = ClassificationMetrics.f1_score(y, y_pred, average='macro')
                
                metrics['precision_weighted'] = ClassificationMetrics.precision(y, y_pred, average='weighted')
                metrics['recall_weighted'] = ClassificationMetrics.recall(y, y_pred, average='weighted')
                metrics['f1_weighted'] = ClassificationMetrics.f1_score(y, y_pred, average='weighted')
                
                # ROC AUC for multiclass if probabilities are available
                if has_probabilities:
                    try:
                        metrics['roc_auc_macro'] = ClassificationMetrics.roc_auc_score(y, y_prob, average='macro')
                        metrics['roc_auc_weighted'] = ClassificationMetrics.roc_auc_score(y, y_prob, average='weighted')
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            
            elif self.classification_type == ClassificationType.MULTILABEL:
                metrics['precision_samples'] = ClassificationMetrics.precision(y, y_pred, average='samples')
                metrics['recall_samples'] = ClassificationMetrics.recall(y, y_pred, average='samples')
                metrics['f1_samples'] = ClassificationMetrics.f1_score(y, y_pred, average='samples')
                
                metrics['precision_macro'] = ClassificationMetrics.precision(y, y_pred, average='macro')
                metrics['recall_macro'] = ClassificationMetrics.recall(y, y_pred, average='macro')
                metrics['f1_macro'] = ClassificationMetrics.f1_score(y, y_pred, average='macro')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {str(e)}")
            raise


class ClassificationModelFactory:
    """Factory for creating classification models.
    
    This class provides methods for creating various types of classification models
    with appropriate hyperparameters based on the data characteristics.
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """Create a classification model of the specified type.
        
        Args:
            model_type: Type of classification model to create.
            **kwargs: Additional arguments to pass to the model constructor.
            
        Returns:
            Any: The created classification model.
            
        Raises:
            ValueError: If the model type is not supported.
            ImportError: If the required library is not installed.
        """
        model_type = model_type.lower()
        
        try:
            if model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**kwargs)
                
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**kwargs)
                
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**kwargs)
                
            elif model_type == 'svm':
                from sklearn.svm import SVC
                return SVC(**kwargs)
                
            elif model_type == 'naive_bayes':
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB(**kwargs)
                
            elif model_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(**kwargs)
                
            elif model_type == 'decision_tree':
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(**kwargs)
                
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    return xgb.XGBClassifier(**kwargs)
                except ImportError:
                    logger.error("XGBoost is not installed. Please install it to use XGBoost models.")
                    raise ImportError("XGBoost is not installed. Please install it to use XGBoost models.")
                    
            elif model_type == 'lightgbm':
                try:
                    import lightgbm as lgb
                    return lgb.LGBMClassifier(**kwargs)
                except ImportError:
                    logger.error("LightGBM is not installed. Please install it to use LightGBM models.")
                    raise ImportError("LightGBM is not installed. Please install it to use LightGBM models.")
                    
            elif model_type == 'catboost':
                try:
                    import catboost as cb
                    return cb.CatBoostClassifier(**kwargs)
                except ImportError:
                    logger.error("CatBoost is not installed. Please install it to use CatBoost models.")
                    raise ImportError("CatBoost is not installed. Please install it to use CatBoost models.")
                    
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating classification model: {str(e)}")
            raise
    
    @staticmethod
    def get_default_hyperparameters(model_type: str, n_samples: int, n_features: int, n_classes: int) -> Dict[str, Any]:
        """Get default hyperparameters for a classification model based on data characteristics.
        
        Args:
            model_type: Type of classification model.
            n_samples: Number of samples in the dataset.
            n_features: Number of features in the dataset.
            n_classes: Number of classes in the dataset.
            
        Returns:
            Dict[str, Any]: Dictionary of default hyperparameters.
        """
        model_type = model_type.lower()
        
        # Determine dataset size category
        if n_samples < 1000:
            size_category = 'small'
        elif n_samples < 10000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # Determine feature dimensionality category
        if n_features < 10:
            feature_category = 'low'
        elif n_features < 100:
            feature_category = 'medium'
        else:
            feature_category = 'high'
        
        # Default hyperparameters based on model type and data characteristics
        if model_type == 'logistic_regression':
            params = {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'multi_class': 'auto',
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'large' and feature_category == 'high':
                params['solver'] = 'saga'
                params['penalty'] = 'l1'
            
            if n_classes > 2:
                params['multi_class'] = 'multinomial'
            
        elif model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['n_estimators'] = 50
            elif size_category == 'large':
                params['n_estimators'] = 200
            
            if feature_category == 'high':
                params['max_features'] = 'sqrt'
            
        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['n_estimators'] = 50
            elif size_category == 'large':
                params['n_estimators'] = 200
                params['learning_rate'] = 0.05
            
        elif model_type == 'svm':
            params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'large':
                params['kernel'] = 'linear'  # Linear kernel is faster for large datasets
            
            if feature_category == 'high':
                params['kernel'] = 'linear'  # Linear kernel is better for high-dimensional data
            
        elif model_type == 'naive_bayes':
            # GaussianNB has few hyperparameters to tune
            params = {}
            
        elif model_type == 'knn':
            params = {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['n_neighbors'] = 3
            elif size_category == 'large':
                params['n_neighbors'] = 10
                params['algorithm'] = 'kd_tree'  # kd_tree is faster for large datasets
            
        elif model_type == 'decision_tree':
            params = {
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'large':
                params['max_depth'] = 10  # Prevent overfitting on large datasets
            
        elif model_type == 'xgboost':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic' if n_classes == 2 else 'multi:softprob',
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['n_estimators'] = 50
            elif size_category == 'large':
                params['n_estimators'] = 200
                params['learning_rate'] = 0.05
            
            if n_classes > 2:
                params['num_class'] = n_classes
            
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,  # -1 means no limit
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary' if n_classes == 2 else 'multiclass',
                'random_state': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['n_estimators'] = 50
            elif size_category == 'large':
                params['n_estimators'] = 200
                params['learning_rate'] = 0.05
            
            if n_classes > 2:
                params['num_class'] = n_classes
            
        elif model_type == 'catboost':
            params = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'Logloss' if n_classes == 2 else 'MultiClass',
                'random_seed': 42
            }
            
            # Adjust for dataset characteristics
            if size_category == 'small':
                params['iterations'] = 50
            elif size_category == 'large':
                params['iterations'] = 200
                params['learning_rate'] = 0.05
            
        else:
            params = {}
        
        return params


class ClassificationModelTrainer:
    """Trainer for classification models.
    
    This class provides methods for training classification models with
    appropriate hyperparameters and evaluation metrics.
    """
    
    def __init__(self, classification_type: ClassificationType = ClassificationType.BINARY):
        """Initialize the classification model trainer.
        
        Args:
            classification_type: Type of classification problem.
        """
        self.classification_type = classification_type
        self.evaluator = ClassificationModelEvaluator(classification_type)
        logger.info(f"Initialized ClassificationModelTrainer with type: {classification_type.name}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str, 
              hyperparameters: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, float]]:
        """Train a classification model.
        
        Args:
            X_train: Training feature data.
            y_train: Training labels.
            model_type: Type of classification model to train.
            hyperparameters: Hyperparameters for the model. If None, default hyperparameters are used.
            
        Returns:
            Tuple[Any, Dict[str, float]]: Trained model and training metrics.
            
        Raises:
            ValueError: If the model type is not supported.
        """
        try:
            # Get default hyperparameters if not provided
            if hyperparameters is None:
                n_samples, n_features = X_train.shape
                n_classes = len(np.unique(y_train))
                hyperparameters = ClassificationModelFactory.get_default_hyperparameters(
                    model_type, n_samples, n_features, n_classes)
            
            # Create and train the model
            model = ClassificationModelFactory.create_model(model_type, **hyperparameters)
            model.fit(X_train, y_train)
            
            # Evaluate the model on training data
            metrics = self.evaluator.evaluate(model, X_train, y_train)
            
            logger.info(f"Trained {model_type} model with {self.classification_type.name} classification type")
            logger.info(f"Training metrics: {metrics}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training classification model: {str(e)}")
            raise
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained classification model on test data.
        
        Args:
            model: Trained classification model.
            X_test: Test feature data.
            y_test: Test labels.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        return self.evaluator.evaluate(model, X_test, y_test)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, model_type: str, 
                       hyperparameters: Optional[Dict[str, Any]] = None, 
                       n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation for a classification model.
        
        Args:
            X: Feature data.
            y: Labels.
            model_type: Type of classification model to train.
            hyperparameters: Hyperparameters for the model. If None, default hyperparameters are used.
            n_splits: Number of cross-validation splits.
            
        Returns:
            Dict[str, List[float]]: Cross-validation metrics for each fold.
            
        Raises:
            ValueError: If the model type is not supported.
            ImportError: If scikit-learn is not installed.
        """
        try:
            from sklearn.model_selection import KFold
            
            # Get default hyperparameters if not provided
            if hyperparameters is None:
                n_samples, n_features = X.shape
                n_classes = len(np.unique(y))
                hyperparameters = ClassificationModelFactory.get_default_hyperparameters(
                    model_type, n_samples, n_features, n_classes)
            
            # Initialize cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Initialize metrics dictionary
            cv_metrics = {}
            
            # Perform cross-validation
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                logger.info(f"Training fold {fold+1}/{n_splits}")
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model = ClassificationModelFactory.create_model(model_type, **hyperparameters)
                model.fit(X_train, y_train)
                
                # Evaluate model
                fold_metrics = self.evaluator.evaluate(model, X_test, y_test)
                
                # Store metrics
                for metric_name, metric_value in fold_metrics.items():
                    if metric_name not in cv_metrics:
                        cv_metrics[metric_name] = []
                    cv_metrics[metric_name].append(metric_value)
            
            logger.info(f"Completed {n_splits}-fold cross-validation for {model_type} model")
            
            return cv_metrics
            
        except ImportError:
            logger.error("scikit-learn is not installed. Cross-validation requires scikit-learn.")
            raise ImportError("scikit-learn is not installed. Cross-validation requires scikit-learn.")
        except Exception as e:
            logger.error(f"Error performing cross-validation: {str(e)}")
            raise