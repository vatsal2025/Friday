"""Data orchestrator module for the Friday AI Trading System.

This module provides the DataOrchestrator class for managing and coordinating
multiple data pipelines and their execution.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
from enum import Enum, auto
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import time
import threading
import asyncio
import concurrent.futures
import uuid
import json
import os

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem, Event
from src.data.integration.data_pipeline import DataPipeline, PipelineError

# Create logger
logger = get_logger(__name__)


class OrchestratorError(Exception):
    """Exception raised for errors in the data orchestrator.

    Attributes:
        message: Explanation of the error.
        pipeline_name: The name of the pipeline where the error occurred.
        details: Additional details about the error.
    """

    def __init__(self, message: str, pipeline_name: str = None, details: Any = None):
        self.message = message
        self.pipeline_name = pipeline_name
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.pipeline_name:
            result += f" (pipeline: {self.pipeline_name})"
        return result


class ExecutionMode(Enum):
    """Enum for pipeline execution modes."""
    SEQUENTIAL = auto()  # Execute pipelines one after another
    PARALLEL = auto()    # Execute pipelines in parallel
    SCHEDULED = auto()   # Execute pipelines according to a schedule


class DataOrchestrator:
    """Class for managing and coordinating multiple data pipelines.

    This class provides methods for registering, executing, and monitoring
    data pipelines, as well as coordinating data flow between them.

    Attributes:
        name: Name of the orchestrator.
        config: Configuration manager.
        event_system: Event system for emitting events.
        pipelines: Dictionary of registered pipelines.
        execution_history: List of execution history records.
        running_pipelines: Set of currently running pipeline names.
        _executor: ThreadPoolExecutor for parallel execution.
        _scheduler: Dictionary of scheduled pipeline executions.
    """

    def __init__(
        self,
        name: str,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None,
        max_workers: int = 5,
    ):
        """Initialize a data orchestrator.

        Args:
            name: Name of the orchestrator.
            config: Configuration manager. If None, a new one will be created.
            event_system: Event system for emitting events. If None, events won't be emitted.
            max_workers: Maximum number of worker threads for parallel execution.
        """
        self.name = name
        self.config = config or ConfigManager()
        self.event_system = event_system
        self.pipelines: Dict[str, DataPipeline] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.running_pipelines: Set[str] = set()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._scheduler: Dict[str, Dict[str, Any]] = {}
        self._stop_scheduler = threading.Event()
        self._scheduler_thread = None

        # Register event handlers if event system is provided
        if self.event_system:
            self._register_event_handlers()

        logger.info(f"Initialized DataOrchestrator '{name}' with max_workers={max_workers}")

    def _register_event_handlers(self) -> None:
        """Register event handlers for pipeline events."""
        self.event_system.register_handler("pipeline.start", self._handle_pipeline_start)
        self.event_system.register_handler("pipeline.success", self._handle_pipeline_success)
        self.event_system.register_handler("pipeline.error", self._handle_pipeline_error)

    def _handle_pipeline_start(self, event: Event) -> None:
        """Handle pipeline start events.

        Args:
            event: The pipeline start event.
        """
        pipeline_name = event.data.get("pipeline_name")
        if pipeline_name:
            self.running_pipelines.add(pipeline_name)

    def _handle_pipeline_success(self, event: Event) -> None:
        """Handle pipeline success events.

        Args:
            event: The pipeline success event.
        """
        pipeline_name = event.data.get("pipeline_name")
        if pipeline_name and pipeline_name in self.running_pipelines:
            self.running_pipelines.remove(pipeline_name)

    def _handle_pipeline_error(self, event: Event) -> None:
        """Handle pipeline error events.

        Args:
            event: The pipeline error event.
        """
        pipeline_name = event.data.get("pipeline_name")
        if pipeline_name and pipeline_name in self.running_pipelines:
            self.running_pipelines.remove(pipeline_name)

    def register_pipeline(self, pipeline: DataPipeline) -> None:
        """Register a pipeline with the orchestrator.

        Args:
            pipeline: The pipeline to register.

        Raises:
            ValueError: If a pipeline with the same name is already registered.
        """
        if pipeline.name in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline.name}' is already registered")

        self.pipelines[pipeline.name] = pipeline
        logger.info(f"Registered pipeline '{pipeline.name}' with orchestrator '{self.name}'")

    def unregister_pipeline(self, pipeline_name: str) -> bool:
        """Unregister a pipeline from the orchestrator.

        Args:
            pipeline_name: The name of the pipeline to unregister.

        Returns:
            bool: True if the pipeline was unregistered, False if not found.

        Raises:
            OrchestratorError: If the pipeline is currently running.
        """
        if pipeline_name in self.running_pipelines:
            raise OrchestratorError(
                f"Cannot unregister pipeline '{pipeline_name}' while it is running",
                pipeline_name
            )

        if pipeline_name in self.pipelines:
            del self.pipelines[pipeline_name]
            logger.info(f"Unregistered pipeline '{pipeline_name}' from orchestrator '{self.name}'")
            return True
        return False

    def get_pipeline(self, pipeline_name: str) -> Optional[DataPipeline]:
        """Get a registered pipeline by name.

        Args:
            pipeline_name: The name of the pipeline to get.

        Returns:
            Optional[DataPipeline]: The pipeline, or None if not found.
        """
        return self.pipelines.get(pipeline_name)

    def list_pipelines(self) -> List[str]:
        """Get a list of registered pipeline names.

        Returns:
            List[str]: The list of pipeline names.
        """
        return list(self.pipelines.keys())

    def execute_pipeline(
        self,
        pipeline_name: str,
        input_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Execute a single pipeline.

        Args:
            pipeline_name: The name of the pipeline to execute.
            input_data: Input data for the pipeline.
            **kwargs: Additional keyword arguments for the pipeline execution.

        Returns:
            pd.DataFrame: The processed data.

        Raises:
            OrchestratorError: If the pipeline is not found or execution fails.
        """
        pipeline = self.get_pipeline(pipeline_name)
        if not pipeline:
            raise OrchestratorError(f"Pipeline '{pipeline_name}' not found", pipeline_name)

        try:
            # Record execution start
            execution_id = str(uuid.uuid4())
            start_time = time.time()
            execution_record = {
                "execution_id": execution_id,
                "pipeline_name": pipeline_name,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration": None,
                "status": "running",
                "error": None,
            }
            self.execution_history.append(execution_record)

            # Emit start event
            self._emit_event(
                "orchestrator.pipeline.start",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "execution_id": execution_id,
                    "start_time": execution_record["start_time"],
                }
            )

            # Execute the pipeline
            logger.info(f"Executing pipeline '{pipeline_name}'")
            result = pipeline.execute(input_data=input_data, **kwargs)

            # Record execution success
            end_time = time.time()
            duration = end_time - start_time
            execution_record["end_time"] = datetime.now().isoformat()
            execution_record["duration"] = duration
            execution_record["status"] = "success"

            # Emit success event
            self._emit_event(
                "orchestrator.pipeline.success",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "execution_id": execution_id,
                    "duration": duration,
                }
            )

            logger.info(f"Pipeline '{pipeline_name}' executed successfully in {duration:.2f} seconds")
            return result

        except Exception as e:
            # Record execution error
            end_time = time.time()
            duration = end_time - start_time
            execution_record["end_time"] = datetime.now().isoformat()
            execution_record["duration"] = duration
            execution_record["status"] = "error"
            execution_record["error"] = str(e)

            # Emit error event
            self._emit_event(
                "orchestrator.pipeline.error",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "execution_id": execution_id,
                    "duration": duration,
                    "error": str(e),
                }
            )

            logger.error(f"Pipeline '{pipeline_name}' execution failed: {str(e)}")
            logger.debug(traceback.format_exc())

            # Re-raise the error
            if isinstance(e, PipelineError):
                raise OrchestratorError(
                    f"Pipeline '{pipeline_name}' execution failed: {str(e)}",
                    pipeline_name,
                    e
                )
            else:
                raise OrchestratorError(
                    f"Pipeline '{pipeline_name}' execution failed: {str(e)}",
                    pipeline_name,
                    e
                )

    def execute_pipelines(
        self,
        pipeline_names: List[str],
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        input_data: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, Exception]]:
        """Execute multiple pipelines.

        Args:
            pipeline_names: The names of the pipelines to execute.
            mode: The execution mode (SEQUENTIAL or PARALLEL).
            input_data: Dictionary of input data for each pipeline, keyed by pipeline name.
            **kwargs: Additional keyword arguments for the pipeline executions.

        Returns:
            Dict[str, Union[pd.DataFrame, Exception]]: Dictionary of results or exceptions,
                keyed by pipeline name.

        Raises:
            OrchestratorError: If any pipeline is not found.
        """
        # Validate pipeline names
        for name in pipeline_names:
            if name not in self.pipelines:
                raise OrchestratorError(f"Pipeline '{name}' not found", name)

        # Initialize results dictionary
        results: Dict[str, Union[pd.DataFrame, Exception]] = {}

        # Execute pipelines based on mode
        if mode == ExecutionMode.SEQUENTIAL:
            for name in pipeline_names:
                try:
                    pipeline_input = input_data.get(name) if input_data else None
                    results[name] = self.execute_pipeline(name, pipeline_input, **kwargs)
                except Exception as e:
                    results[name] = e

        elif mode == ExecutionMode.PARALLEL:
            # Submit all pipeline executions to the thread pool
            futures = {}
            for name in pipeline_names:
                pipeline_input = input_data.get(name) if input_data else None
                futures[name] = self._executor.submit(
                    self.execute_pipeline, name, pipeline_input, **kwargs
                )

            # Collect results as they complete
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = e

        return results

    def schedule_pipeline(
        self,
        pipeline_name: str,
        schedule: Union[str, timedelta, datetime],
        input_data_provider: Optional[Callable[[], pd.DataFrame]] = None,
        **kwargs
    ) -> str:
        """Schedule a pipeline for execution.

        Args:
            pipeline_name: The name of the pipeline to schedule.
            schedule: The schedule for execution. Can be:
                - A cron-like string (e.g., "0 9 * * 1-5" for weekdays at 9am)
                - A timedelta for interval-based execution
                - A datetime for one-time execution
            input_data_provider: Function that provides input data for the pipeline.
            **kwargs: Additional keyword arguments for the pipeline execution.

        Returns:
            str: The schedule ID.

        Raises:
            OrchestratorError: If the pipeline is not found.
        """
        # Validate pipeline
        if pipeline_name not in self.pipelines:
            raise OrchestratorError(f"Pipeline '{pipeline_name}' not found", pipeline_name)

        # Generate schedule ID
        schedule_id = str(uuid.uuid4())

        # Create schedule entry
        schedule_entry = {
            "id": schedule_id,
            "pipeline_name": pipeline_name,
            "schedule": schedule,
            "input_data_provider": input_data_provider,
            "kwargs": kwargs,
            "created_at": datetime.now().isoformat(),
            "last_execution": None,
            "next_execution": self._calculate_next_execution(schedule),
            "enabled": True,
        }

        # Add to scheduler
        self._scheduler[schedule_id] = schedule_entry

        # Start scheduler thread if not already running
        self._start_scheduler_thread()

        logger.info(
            f"Scheduled pipeline '{pipeline_name}' with ID '{schedule_id}', "
            f"next execution: {schedule_entry['next_execution']}"
        )

        return schedule_id

    def _calculate_next_execution(self, schedule: Union[str, timedelta, datetime]) -> datetime:
        """Calculate the next execution time based on the schedule.

        Args:
            schedule: The schedule specification.

        Returns:
            datetime: The next execution time.

        Raises:
            ValueError: If the schedule format is invalid.
        """
        now = datetime.now()

        if isinstance(schedule, datetime):
            # One-time execution
            return schedule

        elif isinstance(schedule, timedelta):
            # Interval-based execution
            return now + schedule

        elif isinstance(schedule, str):
            # Cron-like string (simplified implementation)
            # Format: "minute hour day_of_month month day_of_week"
            # For simplicity, we'll just add a day for now
            # In a real implementation, this would parse the cron string properly
            return now + timedelta(days=1)

        else:
            raise ValueError(f"Invalid schedule format: {schedule}")

    def _start_scheduler_thread(self) -> None:
        """Start the scheduler thread if not already running."""
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._stop_scheduler.clear()
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name=f"orchestrator-{self.name}-scheduler"
            )
            self._scheduler_thread.start()
            logger.info(f"Started scheduler thread for orchestrator '{self.name}'")

    def _scheduler_loop(self) -> None:
        """Main loop for the scheduler thread."""
        logger.info(f"Scheduler thread started for orchestrator '{self.name}'")

        while not self._stop_scheduler.is_set():
            try:
                now = datetime.now()

                # Check for schedules that need to be executed
                for schedule_id, entry in list(self._scheduler.items()):
                    if not entry["enabled"]:
                        continue

                    next_execution = entry["next_execution"]
                    if next_execution and now >= next_execution:
                        # Get input data if provider is available
                        input_data = None
                        if entry["input_data_provider"]:
                            try:
                                input_data = entry["input_data_provider"]()
                            except Exception as e:
                                logger.error(
                                    f"Error getting input data for scheduled pipeline '"
                                    f"{entry['pipeline_name']}': {str(e)}"
                                )

                        # Execute pipeline in a separate thread
                        self._executor.submit(
                            self._execute_scheduled_pipeline,
                            schedule_id,
                            entry["pipeline_name"],
                            input_data,
                            entry["kwargs"]
                        )

                        # Update last execution and calculate next execution
                        entry["last_execution"] = now.isoformat()
                        if isinstance(entry["schedule"], datetime):
                            # One-time execution, disable after execution
                            entry["enabled"] = False
                            entry["next_execution"] = None
                        else:
                            # Calculate next execution time
                            entry["next_execution"] = self._calculate_next_execution(entry["schedule"])

            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                logger.debug(traceback.format_exc())

            # Sleep for a short time to avoid high CPU usage
            time.sleep(1)

        logger.info(f"Scheduler thread stopped for orchestrator '{self.name}'")

    def _execute_scheduled_pipeline(
        self,
        schedule_id: str,
        pipeline_name: str,
        input_data: Optional[pd.DataFrame],
        kwargs: Dict[str, Any]
    ) -> None:
        """Execute a scheduled pipeline.

        Args:
            schedule_id: The schedule ID.
            pipeline_name: The name of the pipeline to execute.
            input_data: Input data for the pipeline.
            kwargs: Additional keyword arguments for the pipeline execution.
        """
        try:
            # Emit scheduled execution start event
            self._emit_event(
                "orchestrator.scheduled.start",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "schedule_id": schedule_id,
                    "start_time": datetime.now().isoformat(),
                }
            )

            # Execute the pipeline
            logger.info(f"Executing scheduled pipeline '{pipeline_name}' (schedule ID: {schedule_id})")
            result = self.execute_pipeline(pipeline_name, input_data, **kwargs)

            # Emit scheduled execution success event
            self._emit_event(
                "orchestrator.scheduled.success",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "schedule_id": schedule_id,
                    "end_time": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            # Emit scheduled execution error event
            self._emit_event(
                "orchestrator.scheduled.error",
                {
                    "orchestrator_name": self.name,
                    "pipeline_name": pipeline_name,
                    "schedule_id": schedule_id,
                    "end_time": datetime.now().isoformat(),
                    "error": str(e),
                }
            )

            logger.error(
                f"Scheduled pipeline '{pipeline_name}' execution failed "
                f"(schedule ID: {schedule_id}): {str(e)}"
            )
            logger.debug(traceback.format_exc())

    def unschedule_pipeline(self, schedule_id: str) -> bool:
        """Unschedule a pipeline.

        Args:
            schedule_id: The schedule ID to remove.

        Returns:
            bool: True if the schedule was removed, False if not found.
        """
        if schedule_id in self._scheduler:
            del self._scheduler[schedule_id]
            logger.info(f"Unscheduled pipeline with ID '{schedule_id}'")
            return True
        return False

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule.

        Args:
            schedule_id: The schedule ID to enable.

        Returns:
            bool: True if the schedule was enabled, False if not found.
        """
        if schedule_id in self._scheduler:
            self._scheduler[schedule_id]["enabled"] = True
            logger.info(f"Enabled schedule with ID '{schedule_id}'")
            return True
        return False

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule.

        Args:
            schedule_id: The schedule ID to disable.

        Returns:
            bool: True if the schedule was disabled, False if not found.
        """
        if schedule_id in self._scheduler:
            self._scheduler[schedule_id]["enabled"] = False
            logger.info(f"Disabled schedule with ID '{schedule_id}'")
            return True
        return False

    def list_schedules(self) -> List[Dict[str, Any]]:
        """Get a list of all schedules.

        Returns:
            List[Dict[str, Any]]: The list of schedule entries.
        """
        return [
            {
                "id": schedule_id,
                "pipeline_name": entry["pipeline_name"],
                "created_at": entry["created_at"],
                "last_execution": entry["last_execution"],
                "next_execution": entry["next_execution"],
                "enabled": entry["enabled"],
            }
            for schedule_id, entry in self._scheduler.items()
        ]

    def get_execution_history(
        self,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get the execution history.

        Args:
            pipeline_name: Filter by pipeline name.
            limit: Maximum number of records to return.
            status: Filter by execution status.

        Returns:
            List[Dict[str, Any]]: The filtered execution history.
        """
        # Filter by pipeline name and status
        filtered_history = self.execution_history
        if pipeline_name:
            filtered_history = [
                record for record in filtered_history
                if record["pipeline_name"] == pipeline_name
            ]
        if status:
            filtered_history = [
                record for record in filtered_history
                if record["status"] == status
            ]

        # Sort by start time (newest first)
        sorted_history = sorted(
            filtered_history,
            key=lambda x: x["start_time"],
            reverse=True
        )

        # Apply limit
        if limit:
            sorted_history = sorted_history[:limit]

        return sorted_history

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history = []
        logger.info(f"Cleared execution history for orchestrator '{self.name}'")

    def save_execution_history(self, file_path: str) -> None:
        """Save the execution history to a file.

        Args:
            file_path: The path to save the history to.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save history to file
            with open(file_path, "w") as f:
                json.dump(self.execution_history, f, indent=2)

            logger.info(f"Saved execution history to '{file_path}'")

        except Exception as e:
            logger.error(f"Error saving execution history: {str(e)}")
            raise OrchestratorError(f"Error saving execution history: {str(e)}", details=e)

    def load_execution_history(self, file_path: str) -> None:
        """Load the execution history from a file.

        Args:
            file_path: The path to load the history from.

        Raises:
            OrchestratorError: If loading fails.
        """
        try:
            # Load history from file
            with open(file_path, "r") as f:
                self.execution_history = json.load(f)

            logger.info(f"Loaded execution history from '{file_path}'")

        except Exception as e:
            logger.error(f"Error loading execution history: {str(e)}")
            raise OrchestratorError(f"Error loading execution history: {str(e)}", details=e)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event using the event system.

        Args:
            event_type: The type of the event.
            data: The data to include in the event.
        """
        if self.event_system:
            event = Event(
                event_type=event_type,
                data=data,
                source=f"orchestrator:{self.name}"
            )
            self.event_system.emit(event)

    def shutdown(self) -> None:
        """Shutdown the orchestrator and release resources."""
        # Stop the scheduler thread
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=5)

        # Shutdown the executor
        self._executor.shutdown(wait=True)

        logger.info(f"Shutdown orchestrator '{self.name}'")

    def __str__(self) -> str:
        """Get a string representation of the orchestrator.

        Returns:
            str: The string representation.
        """
        return f"DataOrchestrator(name='{self.name}', pipelines={len(self.pipelines)})"

    def __repr__(self) -> str:
        """Get a string representation of the orchestrator.

        Returns:
            str: The string representation.
        """
        return self.__str__()