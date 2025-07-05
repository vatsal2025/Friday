"""Model Versioning for Friday AI Trading System.

This module provides functionality for managing model versions,
including validation, promotion to production, and backup.
"""

import os
import json
import shutil
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum

from src.infrastructure.logging import get_logger
from src.services.model.model_validation_rules import ModelType
from src.services.model.validation_pipeline import ValidationStage
from src.services.model.benchmark_validation import BenchmarkMetric
from src.services.model.validation_report import ReportFormat

# Create logger
logger = get_logger(__name__)


class ModelStatus(str, Enum):
    """Enum for model status."""
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class ValidationSeverity(str, Enum):
    """Enum for validation severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationResult:
    """Class for storing validation results."""
    
    def __init__(self, is_valid: bool, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Initialize a validation result.
        
        Args:
            is_valid: Whether the validation passed
            message: Validation message
            severity: Severity level of the validation
        """
        self.is_valid = is_valid
        self.message = message
        self.severity = severity
        self.timestamp = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ValidationResult: Validation result
        """
        severity = ValidationSeverity(data.get("severity", ValidationSeverity.ERROR.value))
        result = cls(data["is_valid"], data["message"], severity)
        result.timestamp = data.get("timestamp", datetime.datetime.now().isoformat())
        return result


class ModelVersionManager:
    """Manager for model versions."""
    
    def __init__(self, model_dir: str, model_name: str):
        """Initialize a model version manager.
        
        Args:
            model_dir: Directory for storing models
            model_name: Name of the model
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.versions_dir = os.path.join(model_dir, model_name, "versions")
        self.production_dir = os.path.join(model_dir, model_name, "production")
        self.backups_dir = os.path.join(model_dir, model_name, "backups")
        
        # Create directories if they don't exist
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.production_dir, exist_ok=True)
        os.makedirs(self.backups_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelVersionManager for {model_name}")
    
    def create_version(self, version: str, model_path: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new model version.
        
        Args:
            version: Version string
            model_path: Path to the model file
            metadata: Additional metadata
            
        Returns:
            str: Path to the new version directory
        """
        # Create version directory
        version_dir = os.path.join(self.versions_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model file to version directory
        model_filename = os.path.basename(model_path)
        dest_path = os.path.join(version_dir, model_filename)
        shutil.copy2(model_path, dest_path)
        
        # Create metadata file
        metadata = metadata or {}
        metadata.update({
            "model_name": self.model_name,
            "version": version,
            "created_at": datetime.datetime.now().isoformat(),
            "status": ModelStatus.DRAFT.value,
            "model_file": model_filename,
            "version_history": [
                {
                    "status": ModelStatus.DRAFT.value,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": "Version created"
                }
            ]
        })
        
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created version {version} for model {self.model_name}")
        return version_dir
    
    def get_version(self, version: str) -> Dict[str, Any]:
        """Get a model version.
        
        Args:
            version: Version string
            
        Returns:
            Dict[str, Any]: Model version metadata
        """
        version_dir = os.path.join(self.versions_dir, version)
        metadata_path = os.path.join(version_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.error(f"Version {version} not found for model {self.model_name}")
            return None
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_model_path(self, version: str) -> str:
        """Get the path to a model version file.
        
        Args:
            version: Version string
            
        Returns:
            str: Path to the model file
        """
        metadata = self.get_version(version)
        if metadata is None:
            return None
        
        model_filename = metadata.get("model_file")
        if model_filename is None:
            logger.error(f"Model file not found in metadata for version {version}")
            return None
        
        version_dir = os.path.join(self.versions_dir, version)
        model_path = os.path.join(version_dir, model_filename)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_filename} not found for version {version}")
            return None
        
        return model_path
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all model versions.
        
        Returns:
            List[Dict[str, Any]]: List of model version metadata
        """
        versions = []
        for version_dir in os.listdir(self.versions_dir):
            metadata_path = os.path.join(self.versions_dir, version_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                versions.append(metadata)
        
        return versions
    
    def get_production_version(self) -> Dict[str, Any]:
        """Get the production model version.
        
        Returns:
            Dict[str, Any]: Production model version metadata
        """
        metadata_path = os.path.join(self.production_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.info(f"No production version found for model {self.model_name}")
            return None
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_production_model_path(self) -> str:
        """Get the path to the production model file.
        
        Returns:
            str: Path to the production model file
        """
        metadata = self.get_production_version()
        if metadata is None:
            return None
        
        model_filename = metadata.get("model_file")
        if model_filename is None:
            logger.error("Model file not found in production metadata")
            return None
        
        model_path = os.path.join(self.production_dir, model_filename)
        if not os.path.exists(model_path):
            logger.error(f"Production model file {model_filename} not found")
            return None
        
        return model_path
    
    def validate_model(self, version: str, validators: List[Callable[[Any], ValidationResult]] = None,
                      model_type: ModelType = None, generate_report: bool = True,
                      report_format: ReportFormat = ReportFormat.HTML,
                      benchmark_dataset_names: List[str] = None,
                      benchmark_metrics: List[BenchmarkMetric] = None) -> bool:
        """Validate a model version.
        
        Args:
            version: Version string
            validators: List of validator functions
            model_type: Type of the model for validation pipeline
            generate_report: Whether to generate a validation report
            report_format: Format of the validation report
            benchmark_dataset_names: Names of benchmark datasets to validate against
            benchmark_metrics: List of metrics to calculate for benchmark validation
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        from src.services.model.model_validation import model_validator
        from src.services.model.benchmark_validation import benchmark_validator
        
        # Get model path
        model_path = self.get_model_path(version)
        if model_path is None:
            logger.error(f"Model path not found for version {version}")
            return False
        
        # Get metadata
        metadata = self.get_version(version)
        if metadata is None:
            logger.error(f"Metadata not found for version {version}")
            return False
        
        # Load model
        try:
            import joblib
            model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._update_validation_results(version, [
                ValidationResult(False, f"Error loading model: {str(e)}", ValidationSeverity.CRITICAL)
            ])
            self._update_model_status(version, ModelStatus.FAILED_VALIDATION, f"Error loading model: {str(e)}")
            return False
        
        # Initialize validation results
        validation_results = []
        
        # Run custom validators if provided
        if validators:
            for validator in validators:
                try:
                    result = validator(model)
                    validation_results.append(result)
                except Exception as e:
                    logger.error(f"Error in validator: {str(e)}")
                    validation_results.append(
                        ValidationResult(False, f"Error in validator: {str(e)}", ValidationSeverity.ERROR)
                    )
        
        # Run validation pipeline if model type is provided
        pipeline_result = None
        benchmark_results = []
        if model_type:
            # Get benchmark datasets if provided
            benchmark_datasets = []
            if benchmark_dataset_names:
                for dataset_name in benchmark_dataset_names:
                    dataset = benchmark_validator.get_dataset(dataset_name)
                    if dataset:
                        benchmark_datasets.append(dataset)
                    else:
                        logger.warning(f"Benchmark dataset {dataset_name} not found")
            
            # Run validation pipeline
            validation_data = model_validator.validate_model_with_pipeline(
                model, model_type, metadata, None, benchmark_datasets, benchmark_metrics,
                None, generate_report, report_format
            )
            
            pipeline_result = validation_data.get("pipeline_result")
            benchmark_results = validation_data.get("benchmark_results", [])
            report_path = validation_data.get("report_path")
            
            # Add report path to metadata
            if report_path:
                metadata["validation_report"] = report_path
            
            # Add pipeline validation results
            if pipeline_result:
                for stage, rules in pipeline_result.stage_results.items():
                    for rule_name, rule_result in rules.items():
                        severity = ValidationSeverity.ERROR
                        if stage == ValidationStage.INFO.value:
                            severity = ValidationSeverity.INFO
                        elif stage == ValidationStage.WARNING.value:
                            severity = ValidationSeverity.WARNING
                        
                        validation_results.append(
                            ValidationResult(rule_result.passed, rule_result.message, severity)
                        )
            
            # Add benchmark validation results
            for benchmark_result in benchmark_results:
                for metric_name, metric_result in benchmark_result.metrics.items():
                    passed = metric_result.get("passed", True)
                    message = f"Benchmark {benchmark_result.dataset_name} - {metric_name}: {metric_result.get('value')}"
                    if not passed:
                        message += f" (threshold: {metric_result.get('threshold')})"
                    
                    severity = ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO
                    validation_results.append(ValidationResult(passed, message, severity))
        
        # Update validation results in metadata
        self._update_validation_results(version, validation_results)
        
        # Check if any critical or error validations failed
        has_critical_error = any(
            not result.is_valid and result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for result in validation_results
        )
        
        # Update model status based on validation results
        if has_critical_error:
            self._update_model_status(version, ModelStatus.FAILED_VALIDATION, "Model failed validation")
            return False
        else:
            self._update_model_status(version, ModelStatus.VALIDATED, "Model passed validation")
            return True
    
    def _update_validation_results(self, version: str, results: List[ValidationResult]) -> None:
        """Update validation results in metadata.
        
        Args:
            version: Version string
            results: List of validation results
        """
        metadata = self.get_version(version)
        if metadata is None:
            logger.error(f"Metadata not found for version {version}")
            return
        
        # Convert validation results to dictionaries
        results_dict = [result.to_dict() for result in results]
        
        # Update metadata
        metadata["validation_results"] = results_dict
        metadata["last_validated"] = datetime.datetime.now().isoformat()
        
        # Write updated metadata
        metadata_path = os.path.join(self.versions_dir, version, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _update_model_status(self, version: str, status: ModelStatus, message: str = None) -> None:
        """Update model status in metadata.
        
        Args:
            version: Version string
            status: New status
            message: Status change message
        """
        metadata = self.get_version(version)
        if metadata is None:
            logger.error(f"Metadata not found for version {version}")
            return
        
        # Update status
        metadata["status"] = status.value
        
        # Add to version history
        history_entry = {
            "status": status.value,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message or f"Status changed to {status.value}"
        }
        metadata["version_history"].append(history_entry)
        
        # Write updated metadata
        metadata_path = os.path.join(self.versions_dir, version, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Updated status of version {version} to {status.value}")
    
    def promote_to_production(self, version: str) -> bool:
        """Promote a model version to production.
        
        Args:
            version: Version string
            
        Returns:
            bool: True if promotion succeeded, False otherwise
        """
        # Get metadata
        metadata = self.get_version(version)
        if metadata is None:
            logger.error(f"Metadata not found for version {version}")
            return False
        
        # Check if model is validated
        if metadata.get("status") != ModelStatus.VALIDATED.value:
            logger.error(f"Cannot promote version {version} to production: not validated")
            return False
        
        # Get model path
        model_path = self.get_model_path(version)
        if model_path is None:
            logger.error(f"Model path not found for version {version}")
            return False
        
        # Backup current production model if it exists
        current_prod = self.get_production_version()
        if current_prod is not None:
            self._backup_production_model()
        
        # Copy model file to production directory
        model_filename = os.path.basename(model_path)
        prod_model_path = os.path.join(self.production_dir, model_filename)
        shutil.copy2(model_path, prod_model_path)
        
        # Update metadata for production
        prod_metadata = metadata.copy()
        prod_metadata["status"] = ModelStatus.PRODUCTION.value
        prod_metadata["promoted_at"] = datetime.datetime.now().isoformat()
        
        # Add to version history
        history_entry = {
            "status": ModelStatus.PRODUCTION.value,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Promoted to production"
        }
        prod_metadata["version_history"].append(history_entry)
        
        # Write production metadata
        prod_metadata_path = os.path.join(self.production_dir, "metadata.json")
        with open(prod_metadata_path, "w") as f:
            json.dump(prod_metadata, f, indent=2)
        
        # Update version metadata status
        self._update_model_status(version, ModelStatus.PRODUCTION, "Promoted to production")
        
        logger.info(f"Promoted version {version} to production")
        return True
    
    def _backup_production_model(self) -> str:
        """Backup the current production model.
        
        Returns:
            str: Path to the backup directory
        """
        # Get production metadata
        metadata = self.get_production_version()
        if metadata is None:
            logger.warning("No production model to backup")
            return None
        
        # Create backup directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.backups_dir, timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy production files to backup directory
        for filename in os.listdir(self.production_dir):
            src_path = os.path.join(self.production_dir, filename)
            dst_path = os.path.join(backup_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
        
        # Update metadata status to DEPRECATED
        backup_metadata = metadata.copy()
        backup_metadata["status"] = ModelStatus.DEPRECATED.value
        backup_metadata["deprecated_at"] = datetime.datetime.now().isoformat()
        
        # Add to version history
        history_entry = {
            "status": ModelStatus.DEPRECATED.value,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": "Deprecated from production"
        }
        backup_metadata["version_history"].append(history_entry)
        
        # Write backup metadata
        backup_metadata_path = os.path.join(backup_dir, "metadata.json")
        with open(backup_metadata_path, "w") as f:
            json.dump(backup_metadata, f, indent=2)
        
        # Update version metadata if it exists
        version = metadata.get("version")
        if version:
            version_metadata = self.get_version(version)
            if version_metadata is not None:
                self._update_model_status(version, ModelStatus.DEPRECATED, "Deprecated from production")
        
        logger.info(f"Backed up production model to {backup_dir}")
        return backup_dir
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all model backups.
        
        Returns:
            List[Dict[str, Any]]: List of model backup metadata
        """
        backups = []
        for backup_dir in os.listdir(self.backups_dir):
            metadata_path = os.path.join(self.backups_dir, backup_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                backups.append(metadata)
        
        return backups


# Common validation functions
def validate_model_interface(model: Any) -> ValidationResult:
    """Validate that a model has the required interface.
    
    Args:
        model: The model to validate
        
    Returns:
        ValidationResult: Validation result
    """
    # Check if model has predict method
    if not hasattr(model, 'predict'):
        return ValidationResult(False, "Model does not have a predict method", ValidationSeverity.ERROR)
    
    # Check if predict method is callable
    if not callable(getattr(model, 'predict')):
        return ValidationResult(False, "Model predict method is not callable", ValidationSeverity.ERROR)
    
    return ValidationResult(True, "Model has required interface", ValidationSeverity.INFO)


def validate_model_size(model_path: str, max_size_mb: float = 100.0, warning_threshold_mb: float = 80.0) -> ValidationResult:
    """Validate that a model file is not too large.
    
    Args:
        model_path: Path to the model file
        max_size_mb: Maximum allowed size in MB
        warning_threshold_mb: Size in MB at which to issue a warning
        
    Returns:
        ValidationResult: Validation result
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        return ValidationResult(False, f"Model file not found: {model_path}", ValidationSeverity.ERROR)
    
    # Get file size in MB
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Check if file is too large
    if size_mb > max_size_mb:
        return ValidationResult(False, f"Model file is too large: {size_mb:.2f} MB (max: {max_size_mb} MB)", 
                              ValidationSeverity.ERROR)
    
    # Issue warning if file is approaching max size
    if size_mb > warning_threshold_mb:
        return ValidationResult(True, f"Model file is large: {size_mb:.2f} MB (warning threshold: {warning_threshold_mb} MB)", 
                              ValidationSeverity.WARNING)
    
    return ValidationResult(True, f"Model file size is acceptable: {size_mb:.2f} MB", ValidationSeverity.INFO)


def create_model_version_manager(model_name: str, model_dir: str = None) -> ModelVersionManager:
    """Create a model version manager.
    
    Args:
        model_name: Name of the model
        model_dir: Directory for storing models
        
    Returns:
        ModelVersionManager: Model version manager
    """
    if model_dir is None:
        # Use default model directory
        model_dir = os.path.join(os.getcwd(), "models")
    
    return ModelVersionManager(model_dir, model_name)