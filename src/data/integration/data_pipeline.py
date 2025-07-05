"""Data pipeline module for the Friday AI Trading System.

This module provides the DataPipeline class for creating and executing
data processing pipelines that connect acquisition, processing, and storage components.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from enum import Enum, auto
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import time

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem, Event
from src.data.acquisition import DataFetcher, HistoricalDataFetcher, RealTimeDataStream
from src.data.processing import DataProcessor, DataCleaner, FeatureEngineer, MultiTimeframeProcessor, DataValidator
from src.data.storage import DataStorage

# Create logger
logger = get_logger(__name__)


class PipelineError(Exception):
    """Exception raised for errors in the data pipeline.

    Attributes:
        message: Explanation of the error.
        stage: The pipeline stage where the error occurred.
        details: Additional details about the error.
    """

    def __init__(self, message: str, stage: str = None, details: Any = None):
        self.message = message
        self.stage = stage
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.stage:
            result += f" (stage: {self.stage})"
        return result


class PipelineStage(Enum):
    """Enum for pipeline stages."""
    ACQUISITION = auto()
    VALIDATION = auto()
    CLEANING = auto()
    FEATURE_ENGINEERING = auto()
    STORAGE = auto()
    CUSTOM = auto()


class DataPipeline:
    """Class for creating and executing data processing pipelines.

    This class provides methods for building data pipelines that connect
    acquisition, processing, and storage components, and executing them
    to process data.

    Attributes:
        name: Name of the pipeline.
        config: Configuration manager.
        event_system: Event system for emitting events.
        stages: List of pipeline stages.
        metadata: Dictionary for storing metadata about the pipeline execution.
    """

    def __init__(
        self,
        name: str,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None,
    ):
        """Initialize a data pipeline.

        Args:
            name: Name of the pipeline.
            config: Configuration manager. If None, a new one will be created.
            event_system: Event system for emitting events. If None, events won't be emitted.
        """
        self.name = name
        self.config = config or ConfigManager()
        self.event_system = event_system
        self.stages: List[Dict[str, Any]] = []
        self.metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "last_run_duration": None,
            "last_run_status": None,
            "last_error": None,
            "runs": [],
        }

    def add_acquisition_stage(
        self,
        data_fetcher: Union[DataFetcher, HistoricalDataFetcher, RealTimeDataStream],
        stage_name: Optional[str] = None,
        **kwargs
    ) -> 'DataPipeline':
        """Add a data acquisition stage to the pipeline.

        Args:
            data_fetcher: The data fetcher to use for acquisition.
            stage_name: Name of the stage. If None, a name will be generated.
            **kwargs: Additional keyword arguments for the data fetcher.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        stage_name = stage_name or f"acquisition_{len(self.stages)}"
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.ACQUISITION,
            "component": data_fetcher,
            "params": kwargs,
        })
        logger.info(f"Added acquisition stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def add_validation_stage(
        self,
        data_processor: DataProcessor,
        stage_name: Optional[str] = None,
        **kwargs
    ) -> 'DataPipeline':
        """Add a data validation stage to the pipeline.

        Args:
            data_processor: The data processor to use for validation.
            stage_name: Name of the stage. If None, a name will be generated.
            **kwargs: Additional keyword arguments for the data processor.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        stage_name = stage_name or f"validation_{len(self.stages)}"
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.VALIDATION,
            "component": data_processor,
            "params": kwargs,
        })
        logger.info(f"Added validation stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def add_enhanced_validation_stage(
        self,
        validation_rules: Optional[List[str]] = None,
        warn_only: bool = False,
        stage_name: Optional[str] = None,
        **validator_kwargs
    ) -> 'DataPipeline':
        """Add an enhanced validation stage with comprehensive market data validation.
        
        This is a convenience method that creates a DataValidator with pre-built
        validation rules suitable for market data.
        
        Args:
            validation_rules: List of specific validation rule names to use. If None,
                all available rules will be used.
            warn_only: If True, validation failures will be logged as warnings but
                won't stop pipeline execution (useful for live streams).
            stage_name: Name of the stage. If None, a name will be generated.
            **validator_kwargs: Additional arguments passed to build_default_market_validator.
        
        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        from src.data.processing.data_validator import build_default_market_validator
        
        # Create enhanced validator with default market rules
        validator = build_default_market_validator(**validator_kwargs)
        
        # Add the validation stage
        stage_name = stage_name or f"enhanced_validation_{len(self.stages)}"
        return self.add_validation_stage(
            validator,
            stage_name=stage_name,
            rules=validation_rules,
            warn_only=warn_only
        )

    def add_cleaning_stage(
        self,
        data_cleaner: DataCleaner,
        stage_name: Optional[str] = None,
        **kwargs
    ) -> 'DataPipeline':
        """Add a data cleaning stage to the pipeline.

        Args:
            data_cleaner: The data cleaner to use for cleaning.
            stage_name: Name of the stage. If None, a name will be generated.
            **kwargs: Additional keyword arguments for the data cleaner.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        stage_name = stage_name or f"cleaning_{len(self.stages)}"
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.CLEANING,
            "component": data_cleaner,
            "params": kwargs,
        })
        logger.info(f"Added cleaning stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def add_feature_engineering_stage(
        self,
        feature_engineer: Union[FeatureEngineer, MultiTimeframeProcessor],
        stage_name: Optional[str] = None,
        **kwargs
    ) -> 'DataPipeline':
        """Add a feature engineering stage to the pipeline.

        Args:
            feature_engineer: The feature engineer to use for feature engineering.
            stage_name: Name of the stage. If None, a name will be generated.
            **kwargs: Additional keyword arguments for the feature engineer.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        stage_name = stage_name or f"feature_engineering_{len(self.stages)}"
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.FEATURE_ENGINEERING,
            "component": feature_engineer,
            "params": kwargs,
        })
        logger.info(f"Added feature engineering stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def add_storage_stage(
        self,
        data_storage: DataStorage,
        stage_name: Optional[str] = None,
        **kwargs
    ) -> 'DataPipeline':
        """Add a data storage stage to the pipeline.

        Args:
            data_storage: The data storage to use for storing data.
            stage_name: Name of the stage. If None, a name will be generated.
            **kwargs: Additional keyword arguments for the data storage.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        stage_name = stage_name or f"storage_{len(self.stages)}"
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.STORAGE,
            "component": data_storage,
            "params": kwargs,
        })
        logger.info(f"Added storage stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def add_custom_stage(
        self,
        stage_name: str,
        processor_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
        **kwargs
    ) -> 'DataPipeline':
        """Add a custom processing stage to the pipeline.

        Args:
            stage_name: Name of the stage.
            processor_func: Function that takes a DataFrame and parameters and returns a processed DataFrame.
            **kwargs: Additional keyword arguments for the processor function.

        Returns:
            DataPipeline: The pipeline instance for method chaining.
        """
        self.stages.append({
            "name": stage_name,
            "type": PipelineStage.CUSTOM,
            "component": processor_func,
            "params": kwargs,
        })
        logger.info(f"Added custom stage '{stage_name}' to pipeline '{self.name}'")
        return self

    def execute(
        self,
        input_data: Optional[pd.DataFrame] = None,
        start_stage: Optional[int] = None,
        end_stage: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Execute the pipeline.

        Args:
            input_data: Input data for the pipeline. If None and the first stage is
                an acquisition stage, data will be fetched from there.
            start_stage: Index of the first stage to execute. If None, execution
                starts from the first stage.
            end_stage: Index of the last stage to execute. If None, execution
                continues until the last stage.
            **kwargs: Additional keyword arguments for the pipeline execution.

        Returns:
            pd.DataFrame: The processed data.

        Raises:
            PipelineError: If pipeline execution fails.
        """
        try:
            # Record execution start
            start_time = time.time()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_metadata = {
                "run_id": run_id,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration": None,
                "status": "running",
                "stages": [],
                "error": None,
            }
            self.metadata["runs"].append(run_metadata)
            self.metadata["last_run"] = run_id

            # Emit start event
            self._emit_event(
                "pipeline.start",
                {
                    "pipeline_name": self.name,
                    "run_id": run_id,
                    "start_time": run_metadata["start_time"],
                }
            )

            # Determine stage range
            start_stage = start_stage or 0
            end_stage = end_stage if end_stage is not None else len(self.stages) - 1

            # Validate stage range
            if start_stage < 0 or start_stage >= len(self.stages):
                raise ValueError(f"Invalid start_stage: {start_stage}")
            if end_stage < start_stage or end_stage >= len(self.stages):
                raise ValueError(f"Invalid end_stage: {end_stage}")

            # Initialize data
            data = input_data

            # Execute stages
            for i in range(start_stage, end_stage + 1):
                stage = self.stages[i]
                stage_name = stage["name"]
                stage_type = stage["type"]
                component = stage["component"]
                params = stage["params"]

                # Record stage start
                stage_start_time = time.time()
                stage_metadata = {
                    "name": stage_name,
                    "type": stage_type.name,
                    "start_time": datetime.now().isoformat(),
                    "end_time": None,
                    "duration": None,
                    "status": "running",
                    "error": None,
                }
                run_metadata["stages"].append(stage_metadata)

                # Emit stage start event
                self._emit_event(
                    "pipeline.stage.start",
                    {
                        "pipeline_name": self.name,
                        "run_id": run_id,
                        "stage_name": stage_name,
                        "stage_type": stage_type.name,
                        "start_time": stage_metadata["start_time"],
                    }
                )

                try:
                    logger.info(f"Executing pipeline stage '{stage_name}'")

                    # Execute stage based on type
                    if stage_type == PipelineStage.ACQUISITION:
                        if data is None:
                            # Fetch data if not provided
                            data = component.fetch_data(**params)
                        else:
                            # Skip acquisition if data is already provided
                            logger.info(f"Skipping acquisition stage '{stage_name}' as data is already provided")

                    elif stage_type == PipelineStage.VALIDATION:
                        # Enhanced validation with detailed metrics
                        if isinstance(component, DataValidator):
                            # Use enhanced validation with metrics and warn_only support
                            warn_only = params.get('warn_only', False)
                            validation_rules = params.get('rules', None)
                            
                            validation_passed, error_messages, validation_metrics = component.validate(
                                data, rules=validation_rules, warn_only=warn_only
                            )
                            
                            # Emit detailed validation metrics to event system
                            self._emit_event(
                                "pipeline.validation.metrics",
                                {
                                    "pipeline_name": self.name,
                                    "run_id": run_id,
                                    "stage_name": stage_name,
                                    "validation_metrics": validation_metrics,
                                    "warn_only_mode": warn_only
                                }
                            )
                            
                            # Log validation results
                            if validation_passed:
                                logger.info(f"Validation stage '{stage_name}' passed: {validation_metrics['rules_passed']}/{validation_metrics['rules_tested']} rules passed")
                                if validation_metrics.get('warnings'):
                                    logger.warning(f"Validation stage '{stage_name}' had warnings: {len(validation_metrics['warnings'])} issues found (warn-only mode)")
                            else:
                                # Validation failed - raise PipelineError unless in warn_only mode
                                if warn_only:
                                    logger.warning(f"Validation stage '{stage_name}' failed but continuing due to warn_only mode: {error_messages}")
                                else:
                                    # Emit validation failure event with detailed metrics
                                    self._emit_event(
                                        "pipeline.validation.failed",
                                        {
                                            "pipeline_name": self.name,
                                            "run_id": run_id,
                                            "stage_name": stage_name,
                                            "error_messages": error_messages,
                                            "validation_metrics": validation_metrics
                                        }
                                    )
                                    raise PipelineError(
                                        f"Validation failed in stage '{stage_name}': {'; '.join(error_messages)}",
                                        stage=stage_name,
                                        details={
                                            "error_messages": error_messages,
                                            "validation_metrics": validation_metrics
                                        }
                                    )
                        else:
                            # Fallback to standard validate_data method for other processors
                            component.validate_data(data, **params)

                    elif stage_type == PipelineStage.CLEANING:
                        # Clean data
                        data = component.clean_data(data, **params)

                    elif stage_type == PipelineStage.FEATURE_ENGINEERING:
                        # Generate features
                        data = component.generate_features(data, **params)

                    elif stage_type == PipelineStage.STORAGE:
                        # Store data
                        component.store_data(data, **params)

                    elif stage_type == PipelineStage.CUSTOM:
                        # Execute custom processor function
                        data = component(data, params)

                    # Record stage success
                    stage_end_time = time.time()
                    stage_duration = stage_end_time - stage_start_time
                    stage_metadata["end_time"] = datetime.now().isoformat()
                    stage_metadata["duration"] = stage_duration
                    stage_metadata["status"] = "success"

                    # Emit stage success event
                    self._emit_event(
                        "pipeline.stage.success",
                        {
                            "pipeline_name": self.name,
                            "run_id": run_id,
                            "stage_name": stage_name,
                            "stage_type": stage_type.name,
                            "duration": stage_duration,
                        }
                    )

                except Exception as e:
                    # Record stage error
                    stage_end_time = time.time()
                    stage_duration = stage_end_time - stage_start_time
                    stage_metadata["end_time"] = datetime.now().isoformat()
                    stage_metadata["duration"] = stage_duration
                    stage_metadata["status"] = "error"
                    stage_metadata["error"] = str(e)

                    # Emit stage error event
                    self._emit_event(
                        "pipeline.stage.error",
                        {
                            "pipeline_name": self.name,
                            "run_id": run_id,
                            "stage_name": stage_name,
                            "stage_type": stage_type.name,
                            "duration": stage_duration,
                            "error": str(e),
                        }
                    )

                    # Raise pipeline error
                    raise PipelineError(f"Error in stage '{stage_name}': {str(e)}", stage_name, e)

            # Record execution success
            end_time = time.time()
            duration = end_time - start_time
            run_metadata["end_time"] = datetime.now().isoformat()
            run_metadata["duration"] = duration
            run_metadata["status"] = "success"
            self.metadata["last_run_duration"] = duration
            self.metadata["last_run_status"] = "success"

            # Emit success event
            self._emit_event(
                "pipeline.success",
                {
                    "pipeline_name": self.name,
                    "run_id": run_id,
                    "duration": duration,
                }
            )

            logger.info(f"Pipeline '{self.name}' executed successfully in {duration:.2f} seconds")
            return data

        except Exception as e:
            # Record execution error
            end_time = time.time()
            duration = end_time - start_time
            run_metadata["end_time"] = datetime.now().isoformat()
            run_metadata["duration"] = duration
            run_metadata["status"] = "error"
            run_metadata["error"] = str(e)
            self.metadata["last_run_duration"] = duration
            self.metadata["last_run_status"] = "error"
            self.metadata["last_error"] = str(e)

            # Emit error event
            self._emit_event(
                "pipeline.error",
                {
                    "pipeline_name": self.name,
                    "run_id": run_id,
                    "duration": duration,
                    "error": str(e),
                }
            )

            logger.error(f"Pipeline '{self.name}' failed: {str(e)}")
            logger.debug(traceback.format_exc())

            # Re-raise the error
            if isinstance(e, PipelineError):
                raise
            else:
                raise PipelineError(f"Pipeline '{self.name}' failed: {str(e)}", details=e)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event using the event system.

        Args:
            event_type: The type of the event.
            data: The data to include in the event.
        """
        if self.event_system:
            self.event_system.emit(
                event_type=event_type,
                data=data,
                source=f"pipeline:{self.name}"
            )

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata about the pipeline execution.

        Returns:
            Dict[str, Any]: The metadata.
        """
        return self.metadata

    def clear_metadata(self) -> None:
        """Clear the metadata about the pipeline execution."""
        self.metadata = {
            "name": self.name,
            "created_at": self.metadata["created_at"],
            "last_run": None,
            "last_run_duration": None,
            "last_run_status": None,
            "last_error": None,
            "runs": [],
        }

    def get_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get a stage by name.

        Args:
            stage_name: The name of the stage to get.

        Returns:
            Optional[Dict[str, Any]]: The stage, or None if not found.
        """
        for stage in self.stages:
            if stage["name"] == stage_name:
                return stage
        return None

    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage by name.

        Args:
            stage_name: The name of the stage to remove.

        Returns:
            bool: True if the stage was removed, False if not found.
        """
        for i, stage in enumerate(self.stages):
            if stage["name"] == stage_name:
                self.stages.pop(i)
                logger.info(f"Removed stage '{stage_name}' from pipeline '{self.name}'")
                return True
        return False

    def __str__(self) -> str:
        """Get a string representation of the pipeline.

        Returns:
            str: The string representation.
        """
        return f"DataPipeline(name='{self.name}', stages={len(self.stages)})"

    def __repr__(self) -> str:
        """Get a string representation of the pipeline.

        Returns:
            str: The string representation.
        """
        return self.__str__()