import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import os

from src.services.model.model_registry import ModelRegistry
from src.services.model.model_monitoring import ABTestingFramework

logger = logging.getLogger(__name__)


class FeedbackSource:
    """Enum-like class for feedback sources"""
    USER = "user"
    SYSTEM = "system"
    AB_TEST = "ab_test"
    SIMULATION = "simulation"
    BACKTEST = "backtest"


class FeedbackType:
    """Enum-like class for feedback types"""
    PREDICTION_ACCURACY = "prediction_accuracy"
    FEATURE_IMPORTANCE = "feature_importance"
    MODEL_DRIFT = "model_drift"
    TRADING_PERFORMANCE = "trading_performance"
    USER_SATISFACTION = "user_satisfaction"
    EXECUTION_SPEED = "execution_speed"
    RESOURCE_USAGE = "resource_usage"


class FeedbackEntry:
    """Class representing a single feedback entry for model improvement."""
    
    def __init__(self,
                 model_id: str,
                 timestamp: datetime,
                 source: str,
                 feedback_type: str,
                 score: float,
                 metadata: Dict[str, Any] = None,
                 description: str = None):
        """Initialize a feedback entry.
        
        Args:
            model_id: ID of the model receiving feedback.
            timestamp: Time when the feedback was generated.
            source: Source of the feedback (e.g., user, system, A/B test).
            feedback_type: Type of feedback (e.g., accuracy, performance).
            score: Numerical score or rating (-1.0 to 1.0 or 0.0 to 1.0 depending on type).
            metadata: Additional structured information about the feedback.
            description: Human-readable description of the feedback.
        """
        self.model_id = model_id
        self.timestamp = timestamp
        self.source = source
        self.feedback_type = feedback_type
        self.score = score
        self.metadata = metadata or {}
        self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback entry to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the feedback entry.
        """
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "feedback_type": self.feedback_type,
            "score": self.score,
            "metadata": self.metadata,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create a feedback entry from a dictionary.
        
        Args:
            data: Dictionary containing feedback entry data.
            
        Returns:
            FeedbackEntry: A new feedback entry instance.
        """
        return cls(
            model_id=data["model_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            feedback_type=data["feedback_type"],
            score=data["score"],
            metadata=data.get("metadata", {}),
            description=data.get("description")
        )


class ImprovementAction:
    """Class representing a recommended action for model improvement."""
    
    def __init__(self,
                 model_id: str,
                 timestamp: datetime,
                 action_type: str,
                 priority: float,  # 0.0 to 1.0 (low to high)
                 description: str,
                 parameters: Dict[str, Any] = None,
                 status: str = "pending",
                 feedback_ids: List[str] = None):
        """Initialize an improvement action.
        
        Args:
            model_id: ID of the model to improve.
            timestamp: Time when the action was generated.
            action_type: Type of improvement action.
            priority: Priority level (0.0 to 1.0).
            description: Human-readable description of the action.
            parameters: Parameters for executing the action.
            status: Current status of the action (pending, in_progress, completed, failed).
            feedback_ids: IDs of feedback entries that led to this action.
        """
        self.model_id = model_id
        self.timestamp = timestamp
        self.action_type = action_type
        self.priority = priority
        self.description = description
        self.parameters = parameters or {}
        self.status = status
        self.feedback_ids = feedback_ids or []
        self.id = f"{model_id}_{timestamp.strftime('%Y%m%d%H%M%S')}_{action_type}"
        
        # Execution tracking
        self.execution_start = None
        self.execution_end = None
        self.execution_result = None
        self.execution_error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert improvement action to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the improvement action.
        """
        return {
            "id": self.id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "priority": self.priority,
            "description": self.description,
            "parameters": self.parameters,
            "status": self.status,
            "feedback_ids": self.feedback_ids,
            "execution_start": self.execution_start.isoformat() if self.execution_start else None,
            "execution_end": self.execution_end.isoformat() if self.execution_end else None,
            "execution_result": self.execution_result,
            "execution_error": self.execution_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementAction':
        """Create an improvement action from a dictionary.
        
        Args:
            data: Dictionary containing improvement action data.
            
        Returns:
            ImprovementAction: A new improvement action instance.
        """
        action = cls(
            model_id=data["model_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action_type=data["action_type"],
            priority=data["priority"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            status=data.get("status", "pending"),
            feedback_ids=data.get("feedback_ids", [])
        )
        
        action.id = data.get("id", action.id)
        
        if data.get("execution_start"):
            action.execution_start = datetime.fromisoformat(data["execution_start"])
        if data.get("execution_end"):
            action.execution_end = datetime.fromisoformat(data["execution_end"])
            
        action.execution_result = data.get("execution_result")
        action.execution_error = data.get("execution_error")
        
        return action


class SelfImprovementFeedbackLoop:
    """Class for implementing a self-improvement feedback loop for models."""
    
    def __init__(self, model_registry: ModelRegistry, storage_dir: str = None):
        """Initialize a self-improvement feedback loop.
        
        Args:
            model_registry: Model registry for accessing and updating models.
            storage_dir: Directory for storing feedback and action data.
        """
        self.model_registry = model_registry
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "feedback_data")
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize feedback and action storage
        self.feedback_entries = {}
        self.improvement_actions = {}
        
        # Load existing data if available
        self._load_data()
        
        logger.info("Initialized SelfImprovementFeedbackLoop with storage at %s", self.storage_dir)
    
    def _load_data(self) -> None:
        """Load feedback entries and improvement actions from storage."""
        feedback_file = os.path.join(self.storage_dir, "feedback_entries.json")
        actions_file = os.path.join(self.storage_dir, "improvement_actions.json")
        
        try:
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    for entry_id, entry_dict in feedback_data.items():
                        self.feedback_entries[entry_id] = FeedbackEntry.from_dict(entry_dict)
                logger.info("Loaded %d feedback entries", len(self.feedback_entries))
        except Exception as e:
            logger.error("Error loading feedback entries: %s", e)
        
        try:
            if os.path.exists(actions_file):
                with open(actions_file, 'r') as f:
                    actions_data = json.load(f)
                    for action_id, action_dict in actions_data.items():
                        self.improvement_actions[action_id] = ImprovementAction.from_dict(action_dict)
                logger.info("Loaded %d improvement actions", len(self.improvement_actions))
        except Exception as e:
            logger.error("Error loading improvement actions: %s", e)
    
    def _save_data(self) -> None:
        """Save feedback entries and improvement actions to storage."""
        feedback_file = os.path.join(self.storage_dir, "feedback_entries.json")
        actions_file = os.path.join(self.storage_dir, "improvement_actions.json")
        
        try:
            feedback_data = {entry_id: entry.to_dict() for entry_id, entry in self.feedback_entries.items()}
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            logger.debug("Saved %d feedback entries", len(self.feedback_entries))
        except Exception as e:
            logger.error("Error saving feedback entries: %s", e)
        
        try:
            actions_data = {action.id: action.to_dict() for action in self.improvement_actions.values()}
            with open(actions_file, 'w') as f:
                json.dump(actions_data, f, indent=2)
            logger.debug("Saved %d improvement actions", len(self.improvement_actions))
        except Exception as e:
            logger.error("Error saving improvement actions: %s", e)
    
    def add_feedback(self, feedback: FeedbackEntry) -> str:
        """Add a new feedback entry.
        
        Args:
            feedback: Feedback entry to add.
            
        Returns:
            str: ID of the added feedback entry.
        """
        # Generate a unique ID for the feedback entry
        feedback_id = f"{feedback.model_id}_{feedback.timestamp.strftime('%Y%m%d%H%M%S')}_{feedback.source}_{feedback.feedback_type}"
        
        # Store the feedback entry
        self.feedback_entries[feedback_id] = feedback
        
        # Save to storage
        self._save_data()
        
        logger.info("Added feedback for model %s: %s (score: %.2f)", 
                   feedback.model_id, feedback.feedback_type, feedback.score)
        
        return feedback_id
    
    def add_user_feedback(self, 
                         model_id: str, 
                         feedback_type: str, 
                         score: float, 
                         description: str = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Add user feedback for a model.
        
        Args:
            model_id: ID of the model receiving feedback.
            feedback_type: Type of feedback.
            score: Numerical score or rating.
            description: Human-readable description of the feedback.
            metadata: Additional structured information about the feedback.
            
        Returns:
            str: ID of the added feedback entry.
        """
        feedback = FeedbackEntry(
            model_id=model_id,
            timestamp=datetime.now(),
            source=FeedbackSource.USER,
            feedback_type=feedback_type,
            score=score,
            description=description,
            metadata=metadata
        )
        
        return self.add_feedback(feedback)
    
    def add_system_feedback(self, 
                           model_id: str, 
                           feedback_type: str, 
                           score: float, 
                           description: str = None,
                           metadata: Dict[str, Any] = None) -> str:
        """Add system-generated feedback for a model.
        
        Args:
            model_id: ID of the model receiving feedback.
            feedback_type: Type of feedback.
            score: Numerical score or rating.
            description: Human-readable description of the feedback.
            metadata: Additional structured information about the feedback.
            
        Returns:
            str: ID of the added feedback entry.
        """
        feedback = FeedbackEntry(
            model_id=model_id,
            timestamp=datetime.now(),
            source=FeedbackSource.SYSTEM,
            feedback_type=feedback_type,
            score=score,
            description=description,
            metadata=metadata
        )
        
        return self.add_feedback(feedback)
    
    def add_ab_test_feedback(self, 
                            ab_test_framework: ABTestingFramework, 
                            test_id: str) -> List[str]:
        """Add feedback from A/B test results.
        
        Args:
            ab_test_framework: A/B testing framework instance.
            test_id: ID of the completed A/B test.
            
        Returns:
            List[str]: IDs of the added feedback entries.
            
        Raises:
            ValueError: If the test is not completed or doesn't exist.
        """
        # Get test results
        test_results = ab_test_framework.get_test_results(test_id)
        
        if test_results["status"] != "completed":
            raise ValueError(f"A/B test {test_id} is not completed")
        
        feedback_ids = []
        
        # Extract model IDs
        model_a_id = test_results["model_a_id"]
        model_b_id = test_results["model_b_id"]
        winner_id = test_results.get("winner_id")
        
        # Add feedback for primary metric
        primary_metric = test_results["primary_metric"]
        model_a_score = test_results["model_a_metrics"][primary_metric]["mean"]
        model_b_score = test_results["model_b_metrics"][primary_metric]["mean"]
        
        # Normalize scores to -1.0 to 1.0 range for comparison feedback
        max_score = max(abs(model_a_score), abs(model_b_score))
        if max_score > 0:
            norm_a_score = model_a_score / max_score
            norm_b_score = model_b_score / max_score
        else:
            norm_a_score = 0.0
            norm_b_score = 0.0
        
        # Add feedback for model A
        feedback_a = FeedbackEntry(
            model_id=model_a_id,
            timestamp=datetime.now(),
            source=FeedbackSource.AB_TEST,
            feedback_type=FeedbackType.PREDICTION_ACCURACY,
            score=norm_a_score,
            description=f"A/B test {test_id} result for {primary_metric}",
            metadata={
                "test_id": test_id,
                "primary_metric": primary_metric,
                "raw_score": model_a_score,
                "is_winner": model_a_id == winner_id,
                "compared_to": model_b_id
            }
        )
        feedback_ids.append(self.add_feedback(feedback_a))
        
        # Add feedback for model B
        feedback_b = FeedbackEntry(
            model_id=model_b_id,
            timestamp=datetime.now(),
            source=FeedbackSource.AB_TEST,
            feedback_type=FeedbackType.PREDICTION_ACCURACY,
            score=norm_b_score,
            description=f"A/B test {test_id} result for {primary_metric}",
            metadata={
                "test_id": test_id,
                "primary_metric": primary_metric,
                "raw_score": model_b_score,
                "is_winner": model_b_id == winner_id,
                "compared_to": model_a_id
            }
        )
        feedback_ids.append(self.add_feedback(feedback_b))
        
        # Add additional feedback for secondary metrics
        for metric, metric_data in test_results["model_a_metrics"].items():
            if metric != primary_metric:
                model_a_score = metric_data["mean"]
                model_b_score = test_results["model_b_metrics"][metric]["mean"]
                
                # Normalize scores
                max_score = max(abs(model_a_score), abs(model_b_score))
                if max_score > 0:
                    norm_a_score = model_a_score / max_score
                    norm_b_score = model_b_score / max_score
                else:
                    norm_a_score = 0.0
                    norm_b_score = 0.0
                
                # Add feedback for model A (secondary metric)
                feedback_a_sec = FeedbackEntry(
                    model_id=model_a_id,
                    timestamp=datetime.now(),
                    source=FeedbackSource.AB_TEST,
                    feedback_type=FeedbackType.PREDICTION_ACCURACY,
                    score=norm_a_score,
                    description=f"A/B test {test_id} result for {metric} (secondary)",
                    metadata={
                        "test_id": test_id,
                        "metric": metric,
                        "is_primary": False,
                        "raw_score": model_a_score,
                        "compared_to": model_b_id
                    }
                )
                feedback_ids.append(self.add_feedback(feedback_a_sec))
                
                # Add feedback for model B (secondary metric)
                feedback_b_sec = FeedbackEntry(
                    model_id=model_b_id,
                    timestamp=datetime.now(),
                    source=FeedbackSource.AB_TEST,
                    feedback_type=FeedbackType.PREDICTION_ACCURACY,
                    score=norm_b_score,
                    description=f"A/B test {test_id} result for {metric} (secondary)",
                    metadata={
                        "test_id": test_id,
                        "metric": metric,
                        "is_primary": False,
                        "raw_score": model_b_score,
                        "compared_to": model_a_id
                    }
                )
                feedback_ids.append(self.add_feedback(feedback_b_sec))
        
        logger.info("Added %d feedback entries from A/B test %s", len(feedback_ids), test_id)
        
        return feedback_ids
    
    def add_simulation_feedback(self, 
                              model_id: str, 
                              simulation_results: Dict[str, Any]) -> List[str]:
        """Add feedback from simulation results.
        
        Args:
            model_id: ID of the model being evaluated.
            simulation_results: Results from the simulation.
            
        Returns:
            List[str]: IDs of the added feedback entries.
        """
        feedback_ids = []
        
        # Extract performance metrics from simulation results
        total_return = simulation_results.get("total_return", 0.0)
        win_rate = simulation_results.get("win_rate", 0.0)
        avg_profit_loss = simulation_results.get("avg_profit_loss", 0.0)
        
        # Add trading performance feedback
        trading_feedback = FeedbackEntry(
            model_id=model_id,
            timestamp=datetime.now(),
            source=FeedbackSource.SIMULATION,
            feedback_type=FeedbackType.TRADING_PERFORMANCE,
            score=total_return,  # Use total return as the score
            description=f"Trading performance from simulation",
            metadata={
                "simulation_id": simulation_results.get("simulation_id"),
                "total_return": total_return,
                "win_rate": win_rate,
                "avg_profit_loss": avg_profit_loss,
                "total_trades": simulation_results.get("total_trades", 0)
            }
        )
        feedback_ids.append(self.add_feedback(trading_feedback))
        
        # Add prediction accuracy feedback based on win rate
        accuracy_feedback = FeedbackEntry(
            model_id=model_id,
            timestamp=datetime.now(),
            source=FeedbackSource.SIMULATION,
            feedback_type=FeedbackType.PREDICTION_ACCURACY,
            score=win_rate * 2 - 1,  # Convert 0-1 to -1 to 1 range
            description=f"Prediction accuracy from simulation win rate",
            metadata={
                "simulation_id": simulation_results.get("simulation_id"),
                "win_rate": win_rate,
                "total_trades": simulation_results.get("total_trades", 0)
            }
        )
        feedback_ids.append(self.add_feedback(accuracy_feedback))
        
        logger.info("Added %d feedback entries from simulation for model %s", 
                   len(feedback_ids), model_id)
        
        return feedback_ids
    
    def analyze_feedback(self, 
                        model_id: str, 
                        feedback_types: List[str] = None,
                        sources: List[str] = None,
                        start_date: datetime = None,
                        end_date: datetime = None) -> Dict[str, Any]:
        """Analyze feedback for a specific model.
        
        Args:
            model_id: ID of the model to analyze.
            feedback_types: Types of feedback to include (None for all).
            sources: Sources of feedback to include (None for all).
            start_date: Start date for feedback range (None for all).
            end_date: End date for feedback range (None for all).
            
        Returns:
            Dict[str, Any]: Analysis results.
        """
        # Filter feedback entries for the specified model
        filtered_entries = []
        
        for entry_id, entry in self.feedback_entries.items():
            if entry.model_id != model_id:
                continue
                
            if feedback_types and entry.feedback_type not in feedback_types:
                continue
                
            if sources and entry.source not in sources:
                continue
                
            if start_date and entry.timestamp < start_date:
                continue
                
            if end_date and entry.timestamp > end_date:
                continue
                
            filtered_entries.append(entry)
        
        if not filtered_entries:
            return {
                "model_id": model_id,
                "feedback_count": 0,
                "message": "No feedback entries found matching the criteria"
            }
        
        # Group feedback by type and source
        feedback_by_type = {}
        feedback_by_source = {}
        
        for entry in filtered_entries:
            # Group by type
            if entry.feedback_type not in feedback_by_type:
                feedback_by_type[entry.feedback_type] = []
            feedback_by_type[entry.feedback_type].append(entry)
            
            # Group by source
            if entry.source not in feedback_by_source:
                feedback_by_source[entry.source] = []
            feedback_by_source[entry.source].append(entry)
        
        # Calculate average scores by type and source
        avg_scores_by_type = {}
        for feedback_type, entries in feedback_by_type.items():
            scores = [entry.score for entry in entries]
            avg_scores_by_type[feedback_type] = sum(scores) / len(scores)
        
        avg_scores_by_source = {}
        for source, entries in feedback_by_source.items():
            scores = [entry.score for entry in entries]
            avg_scores_by_source[source] = sum(scores) / len(scores)
        
        # Calculate overall score
        all_scores = [entry.score for entry in filtered_entries]
        overall_score = sum(all_scores) / len(all_scores)
        
        # Identify trends over time
        entries_by_time = sorted(filtered_entries, key=lambda x: x.timestamp)
        time_series_data = []
        
        for entry in entries_by_time:
            time_series_data.append({
                "timestamp": entry.timestamp.isoformat(),
                "score": entry.score,
                "feedback_type": entry.feedback_type,
                "source": entry.source
            })
        
        # Compile analysis results
        analysis = {
            "model_id": model_id,
            "feedback_count": len(filtered_entries),
            "overall_score": overall_score,
            "scores_by_type": avg_scores_by_type,
            "scores_by_source": avg_scores_by_source,
            "time_series": time_series_data,
            "start_date": entries_by_time[0].timestamp.isoformat() if entries_by_time else None,
            "end_date": entries_by_time[-1].timestamp.isoformat() if entries_by_time else None
        }
        
        return analysis
    
    def generate_improvement_actions(self, 
                                    model_id: str, 
                                    min_feedback_count: int = 5) -> List[ImprovementAction]:
        """Generate improvement actions based on feedback analysis.
        
        Args:
            model_id: ID of the model to generate actions for.
            min_feedback_count: Minimum number of feedback entries required.
            
        Returns:
            List[ImprovementAction]: Generated improvement actions.
        """
        # Analyze feedback for the model
        analysis = self.analyze_feedback(model_id)
        
        if analysis["feedback_count"] < min_feedback_count:
            logger.info("Not enough feedback for model %s to generate actions (got %d, need %d)", 
                       model_id, analysis["feedback_count"], min_feedback_count)
            return []
        
        # Get model metadata from registry
        try:
            model_info = self.model_registry.get_model_info(model_id)
            if model_info is None:
                logger.warning("Model %s not found in registry", model_id)
                return []
        except Exception as e:
            logger.error("Error getting model info for %s: %s", model_id, e)
            return []
        
        # Initialize list of actions
        actions = []
        
        # Check for poor prediction accuracy
        if FeedbackType.PREDICTION_ACCURACY in analysis["scores_by_type"]:
            accuracy_score = analysis["scores_by_type"][FeedbackType.PREDICTION_ACCURACY]
            
            if accuracy_score < -0.3:  # Significantly below average
                # Recommend retraining with more data
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="retrain_with_more_data",
                    priority=min(1.0, 0.5 + abs(accuracy_score)),
                    description="Retrain model with more recent data to improve prediction accuracy",
                    parameters={
                        "suggested_data_increase": "50%",
                        "include_recent_data": True
                    }
                ))
                
                # Recommend feature engineering
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="feature_engineering",
                    priority=min(1.0, 0.4 + abs(accuracy_score)),
                    description="Review and improve feature engineering to enhance prediction accuracy",
                    parameters={
                        "suggested_approaches": [
                            "Add technical indicators",
                            "Improve feature normalization",
                            "Consider feature interactions"
                        ]
                    }
                ))
        
        # Check for poor trading performance
        if FeedbackType.TRADING_PERFORMANCE in analysis["scores_by_type"]:
            trading_score = analysis["scores_by_type"][FeedbackType.TRADING_PERFORMANCE]
            
            if trading_score < 0:  # Negative returns
                # Recommend risk management improvements
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="improve_risk_management",
                    priority=min(1.0, 0.6 + abs(trading_score)),
                    description="Improve risk management parameters to reduce losses",
                    parameters={
                        "suggested_changes": [
                            "Tighten stop-loss thresholds",
                            "Reduce position sizing",
                            "Implement volatility-based position sizing"
                        ]
                    }
                ))
                
                # Recommend signal filtering
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="improve_signal_filtering",
                    priority=min(1.0, 0.5 + abs(trading_score)),
                    description="Implement better signal filtering to reduce false positives",
                    parameters={
                        "suggested_approaches": [
                            "Add confirmation indicators",
                            "Increase signal threshold",
                            "Add market regime detection"
                        ]
                    }
                ))
        
        # Check for model drift
        if FeedbackType.MODEL_DRIFT in analysis["scores_by_type"]:
            drift_score = analysis["scores_by_type"][FeedbackType.MODEL_DRIFT]
            
            if drift_score < -0.2:  # Significant drift detected
                # Recommend model retraining
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="retrain_model",
                    priority=min(1.0, 0.7 + abs(drift_score)),
                    description="Retrain model to address detected model drift",
                    parameters={
                        "use_recent_data": True,
                        "preserve_architecture": True
                    }
                ))
        
        # Check for poor user satisfaction
        if FeedbackType.USER_SATISFACTION in analysis["scores_by_type"]:
            satisfaction_score = analysis["scores_by_type"][FeedbackType.USER_SATISFACTION]
            
            if satisfaction_score < -0.2:  # Users are dissatisfied
                # Recommend UX improvements
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="improve_user_experience",
                    priority=min(1.0, 0.5 + abs(satisfaction_score)),
                    description="Improve user experience based on feedback",
                    parameters={
                        "review_user_comments": True,
                        "focus_areas": [
                            "Prediction explanation",
                            "Confidence indicators",
                            "Visualization"
                        ]
                    }
                ))
        
        # Check for resource usage issues
        if FeedbackType.RESOURCE_USAGE in analysis["scores_by_type"]:
            resource_score = analysis["scores_by_type"][FeedbackType.RESOURCE_USAGE]
            
            if resource_score < -0.3:  # High resource usage
                # Recommend model optimization
                actions.append(ImprovementAction(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    action_type="optimize_model",
                    priority=min(1.0, 0.4 + abs(resource_score)),
                    description="Optimize model to reduce resource usage",
                    parameters={
                        "suggested_approaches": [
                            "Model pruning",
                            "Quantization",
                            "Feature selection"
                        ]
                    }
                ))
        
        # Store generated actions
        for action in actions:
            self.improvement_actions[action.id] = action
        
        # Save to storage
        self._save_data()
        
        logger.info("Generated %d improvement actions for model %s", len(actions), model_id)
        
        return actions
    
    def get_pending_actions(self, 
                           model_id: str = None, 
                           min_priority: float = 0.0,
                           action_types: List[str] = None) -> List[ImprovementAction]:
        """Get pending improvement actions.
        
        Args:
            model_id: Filter by model ID (None for all models).
            min_priority: Minimum priority level.
            action_types: Filter by action types (None for all types).
            
        Returns:
            List[ImprovementAction]: Pending improvement actions.
        """
        pending_actions = []
        
        for action in self.improvement_actions.values():
            if action.status != "pending":
                continue
                
            if model_id and action.model_id != model_id:
                continue
                
            if action.priority < min_priority:
                continue
                
            if action_types and action.action_type not in action_types:
                continue
                
            pending_actions.append(action)
        
        # Sort by priority (highest first)
        pending_actions.sort(key=lambda x: x.priority, reverse=True)
        
        return pending_actions
    
    def update_action_status(self, 
                            action_id: str, 
                            status: str,
                            execution_result: Any = None,
                            execution_error: str = None) -> bool:
        """Update the status of an improvement action.
        
        Args:
            action_id: ID of the action to update.
            status: New status (in_progress, completed, failed).
            execution_result: Result of the action execution.
            execution_error: Error message if the action failed.
            
        Returns:
            bool: True if the action was updated, False otherwise.
            
        Raises:
            ValueError: If the action ID doesn't exist.
        """
        if action_id not in self.improvement_actions:
            raise ValueError(f"Action ID '{action_id}' does not exist")
        
        action = self.improvement_actions[action_id]
        
        # Update status
        old_status = action.status
        action.status = status
        
        # Update execution tracking
        if status == "in_progress" and old_status == "pending":
            action.execution_start = datetime.now()
        
        if status in ["completed", "failed"] and old_status in ["pending", "in_progress"]:
            action.execution_end = datetime.now()
            action.execution_result = execution_result
            action.execution_error = execution_error
        
        # Save to storage
        self._save_data()
        
        logger.info("Updated action %s status from %s to %s", action_id, old_status, status)
        
        return True
    
    def execute_action(self, action_id: str) -> Dict[str, Any]:
        """Execute an improvement action.
        
        Args:
            action_id: ID of the action to execute.
            
        Returns:
            Dict[str, Any]: Execution result.
            
        Raises:
            ValueError: If the action ID doesn't exist or is not pending.
        """
        if action_id not in self.improvement_actions:
            raise ValueError(f"Action ID '{action_id}' does not exist")
        
        action = self.improvement_actions[action_id]
        
        if action.status != "pending":
            raise ValueError(f"Action {action_id} is not pending (current status: {action.status})")
        
        # Update status to in_progress
        self.update_action_status(action_id, "in_progress")
        
        try:
            # Execute action based on action_type
            result = None
            error = None
            
            if action.action_type == "retrain_model" or action.action_type == "retrain_with_more_data":
                # This is a placeholder for actual model retraining logic
                # In a real implementation, this would call the model training pipeline
                logger.info("Would retrain model %s with parameters: %s", 
                           action.model_id, action.parameters)
                
                result = {
                    "message": f"Model {action.model_id} would be retrained",
                    "parameters": action.parameters
                }
            
            elif action.action_type == "feature_engineering":
                # This is a placeholder for feature engineering logic
                logger.info("Would improve feature engineering for model %s with approaches: %s", 
                           action.model_id, action.parameters.get("suggested_approaches"))
                
                result = {
                    "message": f"Feature engineering would be improved for model {action.model_id}",
                    "suggested_approaches": action.parameters.get("suggested_approaches")
                }
            
            elif action.action_type == "improve_risk_management":
                # This is a placeholder for risk management improvement logic
                logger.info("Would improve risk management for model %s with changes: %s", 
                           action.model_id, action.parameters.get("suggested_changes"))
                
                result = {
                    "message": f"Risk management would be improved for model {action.model_id}",
                    "suggested_changes": action.parameters.get("suggested_changes")
                }
            
            elif action.action_type == "improve_signal_filtering":
                # This is a placeholder for signal filtering improvement logic
                logger.info("Would improve signal filtering for model %s with approaches: %s", 
                           action.model_id, action.parameters.get("suggested_approaches"))
                
                result = {
                    "message": f"Signal filtering would be improved for model {action.model_id}",
                    "suggested_approaches": action.parameters.get("suggested_approaches")
                }
            
            elif action.action_type == "optimize_model":
                # This is a placeholder for model optimization logic
                logger.info("Would optimize model %s with approaches: %s", 
                           action.model_id, action.parameters.get("suggested_approaches"))
                
                result = {
                    "message": f"Model {action.model_id} would be optimized",
                    "suggested_approaches": action.parameters.get("suggested_approaches")
                }
            
            elif action.action_type == "improve_user_experience":
                # This is a placeholder for UX improvement logic
                logger.info("Would improve user experience for model %s with focus on: %s", 
                           action.model_id, action.parameters.get("focus_areas"))
                
                result = {
                    "message": f"User experience would be improved for model {action.model_id}",
                    "focus_areas": action.parameters.get("focus_areas")
                }
            
            else:
                # Unknown action type
                error = f"Unknown action type: {action.action_type}"
                logger.warning(error)
                self.update_action_status(action_id, "failed", None, error)
                return {"success": False, "error": error}
            
            # Update action status to completed
            self.update_action_status(action_id, "completed", result)
            
            return {
                "success": True,
                "action_id": action_id,
                "result": result
            }
            
        except Exception as e:
            error = f"Error executing action: {str(e)}"
            logger.error(error)
            self.update_action_status(action_id, "failed", None, error)
            
            return {
                "success": False,
                "action_id": action_id,
                "error": error
            }
    
    def get_model_improvement_history(self, model_id: str) -> Dict[str, Any]:
        """Get the improvement history for a specific model.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            Dict[str, Any]: Improvement history.
        """
        # Get all feedback for the model
        feedback_entries = []
        for entry_id, entry in self.feedback_entries.items():
            if entry.model_id == model_id:
                feedback_entries.append(entry)
        
        # Get all actions for the model
        actions = []
        for action_id, action in self.improvement_actions.items():
            if action.model_id == model_id:
                actions.append(action)
        
        # Sort by timestamp
        feedback_entries.sort(key=lambda x: x.timestamp)
        actions.sort(key=lambda x: x.timestamp)
        
        # Compile history
        history = {
            "model_id": model_id,
            "feedback_count": len(feedback_entries),
            "action_count": len(actions),
            "feedback_history": [entry.to_dict() for entry in feedback_entries],
            "action_history": [action.to_dict() for action in actions],
            "timeline": []
        }
        
        # Create combined timeline
        for entry in feedback_entries:
            history["timeline"].append({
                "timestamp": entry.timestamp.isoformat(),
                "type": "feedback",
                "source": entry.source,
                "feedback_type": entry.feedback_type,
                "score": entry.score,
                "description": entry.description
            })
        
        for action in actions:
            history["timeline"].append({
                "timestamp": action.timestamp.isoformat(),
                "type": "action",
                "action_type": action.action_type,
                "priority": action.priority,
                "status": action.status,
                "description": action.description
            })
        
        # Sort timeline by timestamp
        history["timeline"].sort(key=lambda x: x["timestamp"])
        
        return history