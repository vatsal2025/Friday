"""Model Serialization for Friday AI Trading System.

This module provides functionality for serializing and deserializing machine learning models
with support for different formats, versioning, validation, and backup/rollback mechanisms.
"""

import os
import json
import pickle
import joblib
import datetime
import shutil
import hashlib
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO, Callable, Set
from pathlib import Path

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class ModelSerializationError(Exception):
    """Exception raised for errors during model serialization or deserialization."""
    pass


class ModelValidationError(Exception):
    """Exception raised for errors during model validation."""
    pass


class ModelVersionInfo:
    """Information about a model version.
    
    Attributes:
        version: The version identifier.
        created_at: Timestamp when the version was created.
        model_hash: Hash of the model for integrity verification.
        metadata: Additional metadata about the version.
    """
    
    def __init__(self, version: str, created_at: str, model_hash: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize model version information.
        
        Args:
            version: The version identifier.
            created_at: Timestamp when the version was created.
            model_hash: Hash of the model for integrity verification.
            metadata: Additional metadata about the version.
        """
        self.version = version
        self.created_at = created_at
        self.model_hash = model_hash
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the version info.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_hash": self.model_hash,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersionInfo':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary representation of the version info.
            
        Returns:
            ModelVersionInfo: New instance created from the dictionary.
        """
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            model_hash=data["model_hash"],
            metadata=data.get("metadata", {})
        )


class ModelSerializer:
    """Model Serializer for serializing and deserializing machine learning models.

    This class provides functionality for serializing models in different formats,
    checking compatibility, handling versioning, validation, and backup/rollback mechanisms.

    Attributes:
        default_format: The default serialization format.
        storage_dir: Directory for storing serialized models.
        backup_dir: Directory for storing model backups.
        version_file: Path to the version information file.
        validators: Dictionary of model validators.
    """

    # Supported serialization formats
    FORMATS = ["joblib", "pickle", "json"]

    def __init__(self, 
                 default_format: str = "joblib", 
                 storage_dir: Optional[str] = None,
                 backup_dir: Optional[str] = None,
                 max_backups: int = 5,
                 compression_level: Optional[int] = None):
        """Initialize the model serializer.

        Args:
            default_format: The default serialization format. Must be one of "joblib", "pickle", or "json".
            storage_dir: Directory for storing serialized models. If None, uses a default directory.
            backup_dir: Directory for storing model backups. If None, uses a subdirectory of storage_dir.
            max_backups: Maximum number of backups to keep per model.
            compression_level: Compression level for model serialization (0-9, None for no compression).
                Only applies to joblib and pickle formats.

        Raises:
            ValueError: If the default format is not supported or if compression_level is not in range 0-9.
        """
        if default_format not in self.FORMATS:
            raise ValueError(f"Unsupported serialization format: {default_format}. Must be one of {self.FORMATS}")
        
        # Validate compression level
        if compression_level is not None and (compression_level < 0 or compression_level > 9):
            raise ValueError(f"Compression level must be between 0 and 9, got {compression_level}")
        
        self.default_format = default_format
        self.compression_level = compression_level
        
        # Set up storage directories
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "storage", "models")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.backup_dir = backup_dir or os.path.join(self.storage_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        self.version_file = os.path.join(self.storage_dir, "versions.json")
        self.max_backups = max_backups
        
        # Initialize version information if it doesn't exist
        if not os.path.exists(self.version_file):
            with open(self.version_file, 'w') as f:
                json.dump({"models": {}}, f)
        
        # Dictionary of model validators
        self.validators = {}
        
        logger.info(f"Initialized ModelSerializer with default format: {default_format}")
        logger.info(f"Model storage directory: {self.storage_dir}")
        logger.info(f"Model backup directory: {self.backup_dir}")

    def serialize(self, 
                 model: Any, 
                 model_name: str, 
                 version: Optional[str] = None, 
                 format: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Serialize a model to a file with versioning.

        Args:
            model: The model to serialize.
            model_name: Name of the model.
            version: Version of the model. If None, a new version will be generated.
            format: Serialization format. If None, uses the default format.
            metadata: Additional metadata to store with the model.

        Returns:
            Tuple[str, str]: Path to the serialized model file and the version.

        Raises:
            ModelSerializationError: If there is an error during serialization.
        """
        # Add model class information to metadata for JSON deserialization
        if metadata is None:
            metadata = {}
        
        # Store model class information for later deserialization
        if format == "json" or (format is None and self.default_format == "json"):
            model_class = type(model)
            metadata["model_class"] = f"{model_class.__module__}.{model_class.__name__}"
        # Use default format if not specified
        format = format or self.default_format
        
        # Check if format is supported
        if format not in self.FORMATS:
            raise ModelSerializationError(f"Unsupported serialization format: {format}. Must be one of {self.FORMATS}")
        
        # Generate version if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            version = f"v{timestamp}"
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join(self.storage_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine file extension based on format
        if format == "joblib":
            ext = ".joblib"
        elif format == "pickle":
            ext = ".pkl"
        elif format == "json":
            ext = ".json"
        else:
            ext = ".bin"
        
        # Create file path
        file_path = os.path.join(model_dir, f"{model_name}_{version}{ext}")
        
        try:
            # Compute model hash for integrity verification
            model_hash = self._compute_model_hash(model)
            
            # Create a temporary file for atomic write
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Serialize model based on format
            if format == "joblib":
                if self.compression_level is not None:
                    joblib.dump(model, temp_path, compress=self.compression_level)
                else:
                    joblib.dump(model, temp_path)
            elif format == "pickle":
                with open(temp_path, 'wb') as f:
                    if self.compression_level is not None:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        pickle.dump(model, f)
            elif format == "json":
                # Check if model supports to_json method
                if hasattr(model, 'to_json') and callable(getattr(model, 'to_json')):
                    with open(temp_path, 'w') as f:
                        f.write(model.to_json())
                    
                    # For Keras models, also save weights separately
                    if "keras" in str(type(model)).lower() or "tensorflow" in str(type(model)).lower():
                        try:
                            weights_file = os.path.splitext(temp_path)[0] + ".weights.h5"
                            model.save_weights(weights_file)
                            logger.info(f"Saved model weights to {weights_file}")
                        except Exception as e:
                            logger.warning(f"Could not save weights separately: {str(e)}")
                else:
                    # Try to use json.dumps for simple Python objects
                    try:
                        with open(temp_path, 'w') as f:
                            json.dump(model, f, indent=2)
                    except Exception as e:
                        raise ModelSerializationError(f"Model of type {type(model).__name__} does not support JSON serialization: {str(e)}")
            
            # Move temporary file to final location (atomic operation)
            shutil.move(temp_path, file_path)
            
            # Update version information
            self._update_version_info(model_name, version, model_hash, metadata)
            
            logger.info(f"Serialized model {model_name} (version: {version}) to {file_path} using {format} format")
            return file_path, version
        except Exception as e:
            logger.error(f"Error serializing model {model_name} (version: {version}): {str(e)}")
            # Clean up temporary file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise ModelSerializationError(f"Error serializing model: {str(e)}")

    def deserialize(self, 
                   model_name: str, 
                   version: Optional[str] = None, 
                   format: Optional[str] = None, 
                   validate: bool = True,
                   model_class: Optional[Any] = None) -> Any:
        """Deserialize a model from a file with validation.

        Args:
            model_name: Name of the model.
            version: Version of the model. If None, the latest version will be used.
            format: Serialization format. If None, inferred from file extension or uses default format.
            validate: Whether to validate the model after deserialization.
            model_class: Optional class to use for deserialization (especially for JSON format).

        Returns:
            Any: The deserialized model.

        Raises:
            ModelSerializationError: If there is an error during deserialization.
            ModelValidationError: If the model fails validation.
        """
        try:
            # Get the latest version if not specified
            if version is None:
                version = self.get_latest_version(model_name)
                if version is None:
                    raise ModelSerializationError(f"No versions found for model {model_name}")
            
            # Get version info
            version_info = self.get_version_info(model_name, version)
            if version_info is None:
                raise ModelSerializationError(f"Version {version} not found for model {model_name}")
            
            # Determine file path
            model_dir = os.path.join(self.storage_dir, model_name)
            
            # Try different extensions if format is not specified
            if format is None:
                for ext in [".joblib", ".pkl", ".json"]:
                    file_path = os.path.join(model_dir, f"{model_name}_{version}{ext}")
                    if os.path.exists(file_path):
                        break
                else:
                    raise ModelSerializationError(f"Model file for {model_name} (version: {version}) not found")
            else:
                # Determine file extension based on format
                if format == "joblib":
                    ext = ".joblib"
                elif format == "pickle":
                    ext = ".pkl"
                elif format == "json":
                    ext = ".json"
                else:
                    ext = ".bin"
                
                file_path = os.path.join(model_dir, f"{model_name}_{version}{ext}")
                if not os.path.exists(file_path):
                    raise ModelSerializationError(f"Model file {file_path} not found")
            
            # Infer format from file extension if not specified
            if format is None:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".joblib":
                    format = "joblib"
                elif ext == ".pkl":
                    format = "pickle"
                elif ext == ".json":
                    format = "json"
                else:
                    format = self.default_format
            
            # Check if format is supported
            if format not in self.FORMATS:
                raise ModelSerializationError(f"Unsupported serialization format: {format}. Must be one of {self.FORMATS}")
            
            # Deserialize model based on format
            if format == "joblib":
                model = joblib.load(file_path)
            elif format == "pickle":
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            elif format == "json":
                # Load the JSON string from file
                with open(file_path, 'r') as f:
                    json_str = f.read()
                
                # Check if model_class is provided directly or in version_info metadata
                if model_class is None and version_info and "metadata" in version_info and "model_class" in version_info["metadata"]:
                    model_class_name = version_info["metadata"]["model_class"]
                    # Try to import the model class dynamically
                    try:
                        module_path, class_name = model_class_name.rsplit('.', 1)
                        module = __import__(module_path, fromlist=[class_name])
                        model_class = getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Could not import model class {model_class_name}: {str(e)}")
                
                # Try different approaches to deserialize JSON
                # 1. Try using from_json class method if available on model_class
                if model_class and hasattr(model_class, 'from_json') and callable(getattr(model_class, 'from_json')):
                    model = model_class.from_json(json_str)
                    logger.info(f"Deserialized model using {model_class.__name__}.from_json()")
                # 2. Try using keras.models.model_from_json for Keras models
                elif "keras" in json_str.lower() or "tensorflow" in json_str.lower():
                    try:
                        # Try to import keras
                        try:
                            from tensorflow.keras.models import model_from_json
                        except ImportError:
                            try:
                                from keras.models import model_from_json
                            except ImportError:
                                raise ImportError("Neither tensorflow.keras nor keras is available")
                        
                        model = model_from_json(json_str)
                        
                        # Check if weights file exists separately
                        weights_file = os.path.splitext(file_path)[0] + ".weights.h5"
                        if os.path.exists(weights_file):
                            model.load_weights(weights_file)
                            logger.info(f"Loaded weights from {weights_file}")
                        
                        logger.info("Deserialized Keras model from JSON")
                    except Exception as e:
                        logger.error(f"Error deserializing Keras model from JSON: {str(e)}")
                        raise ModelSerializationError(f"Error deserializing Keras model: {str(e)}")
                # 3. Try using json.loads for simple JSON structures
                else:
                    try:
                        model = json.loads(json_str)
                        logger.info("Deserialized model using json.loads()")
                    except Exception as e:
                        logger.error(f"Error deserializing JSON: {str(e)}")
                        raise ModelSerializationError(f"JSON deserialization failed: {str(e)}")
            
            # Validate model if requested
            if validate:
                self._validate_model(model, model_name, version_info)
            
            logger.info(f"Deserialized model {model_name} (version: {version}) from {file_path} using {format} format")
            return model
        except ModelValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error deserializing model {model_name} (version: {version}): {str(e)}")
            raise ModelSerializationError(f"Error deserializing model: {str(e)}")

    def _compute_model_hash(self, model: Any) -> str:
        """Compute a hash of the model for versioning and integrity verification.

        Args:
            model: The model to hash.

        Returns:
            str: The hash of the model.
        """
        try:
            # Serialize model to bytes using pickle
            model_bytes = pickle.dumps(model)
            
            # Compute SHA-256 hash
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            return model_hash
        except Exception as e:
            logger.error(f"Error computing model hash: {str(e)}")
            return f"error_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(model))}"

    def _update_version_info(self, 
                           model_name: str, 
                           version: str, 
                           model_hash: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update version information for a model.

        Args:
            model_name: Name of the model.
            version: Version of the model.
            model_hash: Hash of the model for integrity verification.
            metadata: Additional metadata to store with the model.
        """
        try:
            # Load existing version information
            with open(self.version_file, 'r') as f:
                version_data = json.load(f)
            
            # Initialize model entry if it doesn't exist
            if model_name not in version_data["models"]:
                version_data["models"][model_name] = {}
            
            # Create version info
            version_info = ModelVersionInfo(
                version=version,
                created_at=datetime.datetime.now().isoformat(),
                model_hash=model_hash,
                metadata=metadata
            )
            
            # Add version info
            version_data["models"][model_name][version] = version_info.to_dict()
            
            # Write updated version information
            with open(self.version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating version information: {str(e)}")
            raise ModelSerializationError(f"Error updating version information: {str(e)}")

    def get_version_info(self, model_name: str, version: str) -> Optional[ModelVersionInfo]:
        """Get version information for a model.

        Args:
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            Optional[ModelVersionInfo]: Version information, or None if not found.
        """
        try:
            # Load version information
            with open(self.version_file, 'r') as f:
                version_data = json.load(f)
            
            # Check if model and version exist
            if model_name in version_data["models"] and version in version_data["models"][model_name]:
                version_info_dict = version_data["models"][model_name][version]
                return ModelVersionInfo.from_dict(version_info_dict)
            
            return None
        except Exception as e:
            logger.error(f"Error getting version information: {str(e)}")
            return None

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model.

        Args:
            model_name: Name of the model.

        Returns:
            Optional[str]: Latest version, or None if no versions are found.
        """
        try:
            # Load version information
            with open(self.version_file, 'r') as f:
                version_data = json.load(f)
            
            # Check if model exists
            if model_name not in version_data["models"] or not version_data["models"][model_name]:
                return None
            
            # Get all versions with their creation timestamps
            versions = []
            for version, info in version_data["models"][model_name].items():
                versions.append((version, info["created_at"]))
            
            # Sort by creation timestamp (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)
            
            # Return the latest version
            return versions[0][0] if versions else None
        except Exception as e:
            logger.error(f"Error getting latest version: {str(e)}")
            return None

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[str]: List of versions, sorted by creation time (newest first).
        """
        try:
            # Load version information
            with open(self.version_file, 'r') as f:
                version_data = json.load(f)
            
            # Check if model exists
            if model_name not in version_data["models"]:
                return []
            
            # Get all versions with their creation timestamps
            versions = []
            for version, info in version_data["models"][model_name].items():
                versions.append((version, info["created_at"]))
            
            # Sort by creation timestamp (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)
            
            # Return the versions
            return [v[0] for v in versions]
        except Exception as e:
            logger.error(f"Error listing versions: {str(e)}")
            return []

    def register_validator(self, model_name: str, validator: Callable[[Any, Dict[str, Any]], bool]) -> None:
        """Register a validator for a model.

        Args:
            model_name: Name of the model.
            validator: Validator function that takes a model and version info and returns a boolean.
        """
        self.validators[model_name] = validator
        logger.info(f"Registered validator for model {model_name}")

    def _validate_model(self, model: Any, model_name: str, version_info: ModelVersionInfo) -> None:
        """Validate a model after deserialization.

        Args:
            model: The deserialized model.
            model_name: Name of the model.
            version_info: Version information.

        Raises:
            ModelValidationError: If the model fails validation.
        """
        # Check model hash for integrity
        computed_hash = self._compute_model_hash(model)
        if computed_hash != version_info.model_hash:
            error_msg = f"Model hash mismatch for {model_name} (version: {version_info.version}). Expected {version_info.model_hash}, got {computed_hash}"
            logger.error(error_msg)
            raise ModelValidationError(error_msg)
        
        # Run custom validator if registered
        if model_name in self.validators:
            validator = self.validators[model_name]
            try:
                if not validator(model, version_info.to_dict()):
                    error_msg = f"Model {model_name} (version: {version_info.version}) failed validation"
                    logger.error(error_msg)
                    raise ModelValidationError(error_msg)
            except Exception as e:
                error_msg = f"Error validating model {model_name} (version: {version_info.version}): {str(e)}"
                logger.error(error_msg)
                raise ModelValidationError(error_msg)
        
        logger.info(f"Model {model_name} (version: {version_info.version}) passed validation")

    def create_backup(self, model_name: str, version: str) -> Optional[str]:
        """Create a backup of a model version.

        Args:
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            Optional[str]: Path to the backup file, or None if backup failed.
        """
        try:
            # Get version info
            version_info = self.get_version_info(model_name, version)
            if version_info is None:
                logger.error(f"Version {version} not found for model {model_name}")
                return None
            
            # Create backup directory for the model if it doesn't exist
            model_backup_dir = os.path.join(self.backup_dir, model_name)
            os.makedirs(model_backup_dir, exist_ok=True)
            
            # Create backup timestamp
            backup_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Find the model file
            model_dir = os.path.join(self.storage_dir, model_name)
            model_file = None
            for ext in [".joblib", ".pkl", ".json", ".bin"]:
                file_path = os.path.join(model_dir, f"{model_name}_{version}{ext}")
                if os.path.exists(file_path):
                    model_file = file_path
                    break
            
            if model_file is None:
                logger.error(f"Model file for {model_name} (version: {version}) not found")
                return None
            
            # Create backup file path
            backup_file = os.path.join(model_backup_dir, f"{model_name}_{version}_backup_{backup_timestamp}{os.path.splitext(model_file)[1]}")
            
            # Copy model file to backup location
            shutil.copy2(model_file, backup_file)
            
            # Create backup metadata file
            backup_metadata = {
                "model_name": model_name,
                "version": version,
                "created_at": backup_timestamp,
                "original_file": model_file,
                "version_info": version_info.to_dict()
            }
            
            backup_metadata_file = os.path.join(model_backup_dir, f"{model_name}_{version}_backup_{backup_timestamp}.json")
            with open(backup_metadata_file, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            # Prune old backups if necessary
            self._prune_backups(model_name)
            
            logger.info(f"Created backup of model {model_name} (version: {version}) at {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return None

    def _prune_backups(self, model_name: str) -> None:
        """Prune old backups to keep only the maximum number of backups.

        Args:
            model_name: Name of the model.
        """
        try:
            # Get backup directory for the model
            model_backup_dir = os.path.join(self.backup_dir, model_name)
            if not os.path.exists(model_backup_dir):
                return
            
            # Get all backup metadata files
            backup_metadata_files = [f for f in os.listdir(model_backup_dir) if f.endswith(".json")]
            
            # If number of backups is within limit, do nothing
            if len(backup_metadata_files) <= self.max_backups:
                return
            
            # Get creation timestamps for all backups
            backups = []
            for metadata_file in backup_metadata_files:
                try:
                    with open(os.path.join(model_backup_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                    backups.append((metadata_file, metadata["created_at"]))
                except Exception:
                    # Skip invalid metadata files
                    continue
            
            # Sort by creation timestamp (oldest first)
            backups.sort(key=lambda x: x[1])
            
            # Remove oldest backups
            for metadata_file, _ in backups[:-self.max_backups]:
                # Get backup file path from metadata
                try:
                    with open(os.path.join(model_backup_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                    
                    # Remove backup file
                    backup_file = os.path.join(model_backup_dir, os.path.basename(metadata["original_file"]).replace(f"_{metadata['version']}", f"_{metadata['version']}_backup_{metadata['created_at']}"))
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    
                    # Remove metadata file
                    os.remove(os.path.join(model_backup_dir, metadata_file))
                    
                    logger.info(f"Pruned old backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Error pruning backup {metadata_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error pruning backups: {str(e)}")

    def rollback(self, model_name: str, version: str, backup_timestamp: Optional[str] = None) -> bool:
        """Rollback a model to a previous version or backup.

        Args:
            model_name: Name of the model.
            version: Version of the model to rollback.
            backup_timestamp: Timestamp of the backup to rollback to. If None, uses the latest backup.

        Returns:
            bool: True if rollback was successful, False otherwise.
        """
        try:
            # Get backup directory for the model
            model_backup_dir = os.path.join(self.backup_dir, model_name)
            if not os.path.exists(model_backup_dir):
                logger.error(f"No backups found for model {model_name}")
                return False
            
            # Find backup metadata files for the version
            backup_metadata_files = [f for f in os.listdir(model_backup_dir) if f.startswith(f"{model_name}_{version}_backup_") and f.endswith(".json")]
            
            if not backup_metadata_files:
                logger.error(f"No backups found for model {model_name} (version: {version})")
                return False
            
            # If backup timestamp is provided, find the specific backup
            if backup_timestamp:
                backup_metadata_file = f"{model_name}_{version}_backup_{backup_timestamp}.json"
                if backup_metadata_file not in backup_metadata_files:
                    logger.error(f"Backup with timestamp {backup_timestamp} not found for model {model_name} (version: {version})")
                    return False
            else:
                # Otherwise, use the latest backup
                backup_metadata_files.sort(reverse=True)  # Sort by timestamp (newest first)
                backup_metadata_file = backup_metadata_files[0]
                backup_timestamp = backup_metadata_file.split("_backup_")[1].split(".")[0]
            
            # Load backup metadata
            with open(os.path.join(model_backup_dir, backup_metadata_file), 'r') as f:
                backup_metadata = json.load(f)
            
            # Get backup file path
            original_file_ext = os.path.splitext(backup_metadata["original_file"])[1]
            backup_file = os.path.join(model_backup_dir, f"{model_name}_{version}_backup_{backup_timestamp}{original_file_ext}")
            
            if not os.path.exists(backup_file):
                logger.error(f"Backup file {backup_file} not found")
                return False
            
            # Get original file path
            original_file = backup_metadata["original_file"]
            
            # Create a backup of the current file before rollback
            if os.path.exists(original_file):
                current_backup = self.create_backup(model_name, version)
                if current_backup is None:
                    logger.warning(f"Failed to create backup of current model before rollback")
            
            # Copy backup file to original location
            shutil.copy2(backup_file, original_file)
            
            logger.info(f"Rolled back model {model_name} (version: {version}) to backup from {backup_timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error rolling back model: {str(e)}")
            return False

    def is_compatible(self, model: Any, model_name: str, version: str) -> bool:
        """Check if a model is compatible with a serialized model version.

        Args:
            model: The model to check compatibility for.
            model_name: Name of the model.
            version: Version of the model.

        Returns:
            bool: True if the model is compatible, False otherwise.
        """
        try:
            # Try to deserialize the model
            deserialized_model = self.deserialize(model_name, version, validate=False)
            
            # Check if the models are of the same type
            if type(model) != type(deserialized_model):
                logger.warning(f"Model types do not match: {type(model).__name__} vs {type(deserialized_model).__name__}")
                return False
            
            # Check if the models have the same attributes
            # This is a simple check and may not work for all model types
            model_attrs = set(dir(model))
            deserialized_attrs = set(dir(deserialized_model))
            
            # Check if all attributes in the model are in the deserialized model
            if not model_attrs.issubset(deserialized_attrs):
                missing_attrs = model_attrs - deserialized_attrs
                logger.warning(f"Deserialized model is missing attributes: {missing_attrs}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking model compatibility: {str(e)}")
            return False