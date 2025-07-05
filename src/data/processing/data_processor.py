"""Data processor module for the Friday AI Trading System.

This module provides the base DataProcessor class and related components for
processing market data.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import Event, EventSystem

# Create logger
logger = get_logger(__name__)


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    pass


class DataProcessingError(Exception):
    """Exception raised for data processing errors."""

    pass


class ProcessingStep(Enum):
    """Enum for data processing steps."""

    VALIDATION = auto()
    CLEANING = auto()
    NORMALIZATION = auto()
    FEATURE_ENGINEERING = auto()
    TRANSFORMATION = auto()
    AGGREGATION = auto()
    RESAMPLING = auto()
    CUSTOM = auto()


class DataProcessor:
    """Base class for data processing operations.

    This class provides the foundation for all data processing operations,
    including validation, cleaning, transformation, and feature engineering.

    Attributes:
        config: Configuration manager.
        event_system: Event system for publishing processing events.
        processing_steps: Dictionary of processing steps and their handlers.
        validation_rules: List of validation rules to apply to data.
        metadata: Dictionary for storing metadata about the processed data.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None,
    ):
        """Initialize a data processor.

        Args:
            config: Configuration manager. If None, a new one will be created.
            event_system: Event system for publishing processing events.
                If None, events won't be published.
        """
        self.config = config or ConfigManager()
        self.event_system = event_system
        self.processing_steps: Dict[ProcessingStep, List[Callable]] = {}
        self.validation_rules: List[Callable] = []
        self.metadata: Dict[str, Any] = {}

    def add_processing_step(
        self, step_type: ProcessingStep, handler: Callable, position: int = -1
    ) -> None:
        """Add a processing step handler.

        Args:
            step_type: The type of processing step.
            handler: The handler function for the step.
            position: The position to insert the handler in the step's list.
                Default is -1 (append to end).
        """
        if step_type not in self.processing_steps:
            self.processing_steps[step_type] = []

        if position < 0 or position >= len(self.processing_steps[step_type]):
            self.processing_steps[step_type].append(handler)
        else:
            self.processing_steps[step_type].insert(position, handler)

    def remove_processing_step(self, step_type: ProcessingStep, handler: Callable) -> bool:
        """Remove a processing step handler.

        Args:
            step_type: The type of processing step.
            handler: The handler function to remove.

        Returns:
            bool: True if the handler was removed, False otherwise.
        """
        if step_type in self.processing_steps and handler in self.processing_steps[step_type]:
            self.processing_steps[step_type].remove(handler)
            return True
        return False

    def add_validation_rule(self, rule: Callable, position: int = -1) -> None:
        """Add a validation rule.

        Args:
            rule: The validation rule function.
            position: The position to insert the rule in the list.
                Default is -1 (append to end).
        """
        if position < 0 or position >= len(self.validation_rules):
            self.validation_rules.append(rule)
        else:
            self.validation_rules.insert(position, rule)

    def remove_validation_rule(self, rule: Callable) -> bool:
        """Remove a validation rule.

        Args:
            rule: The validation rule function to remove.

        Returns:
            bool: True if the rule was removed, False otherwise.
        """
        if rule in self.validation_rules:
            self.validation_rules.remove(rule)
            return True
        return False

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data using the registered validation rules.

        Args:
            data: The data to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating if the
                data is valid, and a list of error messages if not.
        """
        is_valid = True
        error_messages = []

        for rule in self.validation_rules:
            try:
                rule_result, rule_message = rule(data)
                if not rule_result:
                    is_valid = False
                    error_messages.append(rule_message)
            except Exception as e:
                is_valid = False
                error_messages.append(f"Validation rule error: {str(e)}")
                logger.error(f"Validation rule error: {str(e)}\n{traceback.format_exc()}")

        return is_valid, error_messages

    def process_data(
        self, data: pd.DataFrame, steps: Optional[List[ProcessingStep]] = None
    ) -> pd.DataFrame:
        """Process data using the registered processing steps.

        Args:
            data: The data to process.
            steps: Optional list of processing steps to apply. If None, all steps will be applied.

        Returns:
            pd.DataFrame: The processed data.

        Raises:
            DataValidationError: If the data fails validation.
            DataProcessingError: If an error occurs during processing.
        """
        # Make a copy of the input data to avoid modifying the original
        processed_data = data.copy()

        # Record start time for performance tracking
        start_time = datetime.now()
        self.metadata["processing_start_time"] = start_time
        self.metadata["input_shape"] = processed_data.shape

        # Validate data first
        is_valid, error_messages = self.validate_data(processed_data)
        if not is_valid:
            error_msg = "\n".join(error_messages)
            logger.error(f"Data validation failed: {error_msg}")
            self._emit_event("data_validation_failed", {"errors": error_messages})
            raise DataValidationError(f"Data validation failed: {error_msg}")

        # Determine which steps to apply
        if steps is None:
            steps_to_apply = list(self.processing_steps.keys())
        else:
            steps_to_apply = steps

        # Apply processing steps
        for step in steps_to_apply:
            if step in self.processing_steps:
                try:
                    step_start_time = datetime.now()
                    for handler in self.processing_steps[step]:
                        processed_data = handler(processed_data)
                    step_duration = (datetime.now() - step_start_time).total_seconds()
                    self.metadata[f"{step.name.lower()}_duration"] = step_duration
                    self._emit_event(f"data_processing_step_completed", {
                        "step": step.name,
                        "duration": step_duration,
                        "output_shape": processed_data.shape
                    })
                except Exception as e:
                    error_msg = f"Error in processing step {step.name}: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    self._emit_event("data_processing_error", {
                        "step": step.name,
                        "error": str(e)
                    })
                    raise DataProcessingError(error_msg)

        # Record end time and total duration
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        self.metadata["processing_end_time"] = end_time
        self.metadata["total_processing_duration"] = total_duration
        self.metadata["output_shape"] = processed_data.shape

        # Emit completion event
        self._emit_event("data_processing_completed", {
            "total_duration": total_duration,
            "input_shape": self.metadata["input_shape"],
            "output_shape": self.metadata["output_shape"]
        })

        return processed_data

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event using the event system.

        Args:
            event_type: The type of event.
            data: The event data.
        """
        if self.event_system:
            event = Event(
                event_type=f"data.processing.{event_type}",
                data=data,
                source="data_processor"
            )
            self.event_system.emit(event.event_type, event.data, event.source)

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata about the processed data.

        Returns:
            Dict[str, Any]: The metadata dictionary.
        """
        return self.metadata

    def reset_metadata(self) -> None:
        """Reset the metadata dictionary."""
        self.metadata = {}

    @staticmethod
    def check_required_columns(data: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
        """Check if the data contains all required columns.

        Args:
            data: The data to check.
            required_columns: List of required column names.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if all required
                columns are present, and an error message if not.
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        return True, ""

    @staticmethod
    def check_data_types(data: pd.DataFrame, column_types: Dict[str, type]) -> Tuple[bool, str]:
        """Check if the data columns have the expected types.

        Args:
            data: The data to check.
            column_types: Dictionary mapping column names to expected types.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if all columns
                have the expected types, and an error message if not.
        """
        type_errors = []
        for col, expected_type in column_types.items():
            if col in data.columns:
                # Check if column type matches expected type
                if expected_type == float:
                    if not pd.api.types.is_float_dtype(data[col]):
                        type_errors.append(f"{col} (expected: float, got: {data[col].dtype})")
                elif expected_type == int:
                    if not pd.api.types.is_integer_dtype(data[col]):
                        type_errors.append(f"{col} (expected: int, got: {data[col].dtype})")
                elif expected_type == str:
                    if not pd.api.types.is_string_dtype(data[col]):
                        type_errors.append(f"{col} (expected: str, got: {data[col].dtype})")
                elif expected_type == bool:
                    if not pd.api.types.is_bool_dtype(data[col]):
                        type_errors.append(f"{col} (expected: bool, got: {data[col].dtype})")
                elif expected_type == datetime:
                    if not pd.api.types.is_datetime64_dtype(data[col]):
                        type_errors.append(f"{col} (expected: datetime, got: {data[col].dtype})")

        if type_errors:
            return False, f"Column type errors: {', '.join(type_errors)}"
        return True, ""

    @staticmethod
    def check_value_ranges(data: pd.DataFrame, range_checks: Dict[str, Tuple[float, float]]) -> Tuple[bool, str]:
        """Check if the data values are within expected ranges.

        Args:
            data: The data to check.
            range_checks: Dictionary mapping column names to (min, max) tuples.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if all values
                are within expected ranges, and an error message if not.
        """
        range_errors = []
        for col, (min_val, max_val) in range_checks.items():
            if col in data.columns:
                if data[col].min() < min_val:
                    range_errors.append(f"{col} has values below minimum {min_val}")
                if data[col].max() > max_val:
                    range_errors.append(f"{col} has values above maximum {max_val}")

        if range_errors:
            return False, f"Value range errors: {', '.join(range_errors)}"
        return True, ""

    @staticmethod
    def check_missing_values(data: pd.DataFrame, threshold: float = 0.1) -> Tuple[bool, str]:
        """Check if the data has an acceptable level of missing values.

        Args:
            data: The data to check.
            threshold: Maximum acceptable fraction of missing values per column.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if missing values
                are within acceptable limits, and an error message if not.
        """
        missing_fractions = data.isna().mean()
        problem_columns = missing_fractions[missing_fractions > threshold].index.tolist()

        if problem_columns:
            details = ", ".join([f"{col} ({missing_fractions[col]:.2%})" for col in problem_columns])
            return False, f"Excessive missing values in columns: {details}"
        return True, ""

    @staticmethod
    def check_duplicates(data: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Check if the data contains duplicate rows.

        Args:
            data: The data to check.
            subset: Optional list of column names to check for duplicates.
                If None, all columns will be used.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if there
                are no duplicates, and an error message if there are.
        """
        duplicate_count = data.duplicated(subset=subset).sum()
        if duplicate_count > 0:
            return False, f"Found {duplicate_count} duplicate rows"
        return True, ""

    @staticmethod
    def check_index_continuity(data: pd.DataFrame, freq: str = None) -> Tuple[bool, str]:
        """Check if the DataFrame index is continuous (for time series data).

        Args:
            data: The data to check.
            freq: Expected frequency of the time series (e.g., '1D', '1H', '1min').
                If None, will try to infer from the data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if the index
                is continuous, and an error message if not.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return False, "Index is not a DatetimeIndex"

        if freq is None:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is None:
                return False, "Could not infer frequency from index"
            freq = inferred_freq

        # Create a continuous index with the same start, end, and frequency
        expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)

        # Check if the actual index matches the expected index
        missing_dates = expected_index.difference(data.index)
        if len(missing_dates) > 0:
            return False, f"Missing {len(missing_dates)} expected timestamps in index"

        return True, ""