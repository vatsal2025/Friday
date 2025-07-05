"""Model Serialization Tools for Trading Engine.

This module provides utilities for serializing, deserializing, versioning,
and managing machine learning models used in the trading engine.
"""

import os
import json
import pickle
import hashlib
import datetime
import shutil
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import logging
from pathlib import Path
import uuid

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                 "storage", "models")


class ModelFormat:
    """Enumeration of supported model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    CLOUDPICKLE = "cloudpickle"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    ONNX = "onnx"
    CUSTOM = "custom"  # For user-defined serialization formats


class ModelMetadata:
    """Container for model metadata."""
    def __init__(
        self,
        model_id: str,
        model_name: str,
        version: str,
        format: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        description: Optional[str] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        python_version: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        training_data_hash: Optional[str] = None,
        model_hash: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        author: Optional[str] = None,
        environment: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize model metadata.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Name of the model
            version: Version string (semantic versioning recommended)
            format: Serialization format (from ModelFormat)
            created_at: ISO format timestamp of creation
            updated_at: ISO format timestamp of last update
            description: Description of the model
            framework: ML framework used (e.g., "scikit-learn", "pytorch")
            framework_version: Version of the ML framework
            python_version: Python version used
            input_schema: Schema describing expected input format
            output_schema: Schema describing expected output format
            performance_metrics: Dictionary of performance metrics
            training_data_hash: Hash of training data for reproducibility
            model_hash: Hash of model file for integrity verification
            tags: List of tags for categorization
            parameters: Dictionary of model hyperparameters
            author: Author of the model
            environment: Environment (dev, test, prod)
            dependencies: Dictionary of dependencies and versions
            custom_metadata: Any additional custom metadata
        """
        self.model_id = model_id
        self.model_name = model_name
        self.version = version
        self.format = format
        self.created_at = created_at or datetime.datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at
        self.description = description
        self.framework = framework
        self.framework_version = framework_version
        self.python_version = python_version
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.performance_metrics = performance_metrics or {}
        self.training_data_hash = training_data_hash
        self.model_hash = model_hash
        self.tags = tags or []
        self.parameters = parameters or {}
        self.author = author
        self.environment = environment
        self.dependencies = dependencies or {}
        self.custom_metadata = custom_metadata or {}
    
    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> 'ModelMetadata':
        """Create a ModelMetadata instance from a dictionary.
        
        Args:
            metadata_dict: Dictionary containing metadata
            
        Returns:
            ModelMetadata: Instance created from dictionary
        """
        return cls(**metadata_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metadata
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "format": self.format,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "python_version": self.python_version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "performance_metrics": self.performance_metrics,
            "training_data_hash": self.training_data_hash,
            "model_hash": self.model_hash,
            "tags": self.tags,
            "parameters": self.parameters,
            "author": self.author,
            "environment": self.environment,
            "dependencies": self.dependencies,
            "custom_metadata": self.custom_metadata
        }
    
    def save(self, filepath: str) -> None:
        """Save metadata to a JSON file.
        
        Args:
            filepath: Path to save the metadata JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelMetadata':
        """Load metadata from a JSON file.
        
        Args:
            filepath: Path to the metadata JSON file
            
        Returns:
            ModelMetadata: Loaded metadata
        """
        with open(filepath, 'r') as f:
            metadata_dict = json.load(f)
        return cls.from_dict(metadata_dict)
    
    def update(self, **kwargs) -> None:
        """Update metadata fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Always update the updated_at timestamp
        self.updated_at = datetime.datetime.now().isoformat()


class ModelSerializer:
    """Handles serialization and deserialization of models."""
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the model serializer.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir or DEFAULT_MODEL_DIR
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _get_model_path(self, model_id: str, version: str, format: str) -> str:
        """Get the path for a model file.
        
        Args:
            model_id: Model ID
            version: Model version
            format: Model format
            
        Returns:
            str: Path to the model file
        """
        # Create directory structure: base_dir/model_id/version/
        model_dir = os.path.join(self.base_dir, model_id, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine file extension based on format
        extension = self._get_extension_for_format(format)
        
        return os.path.join(model_dir, f"model{extension}")
    
    def _get_metadata_path(self, model_id: str, version: str) -> str:
        """Get the path for a model's metadata file.
        
        Args:
            model_id: Model ID
            version: Model version
            
        Returns:
            str: Path to the metadata file
        """
        model_dir = os.path.join(self.base_dir, model_id, version)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, "metadata.json")
    
    def _get_extension_for_format(self, format: str) -> str:
        """Get the file extension for a model format.
        
        Args:
            format: Model format
            
        Returns:
            str: File extension
        """
        format_extensions = {
            ModelFormat.PICKLE: ".pkl",
            ModelFormat.JOBLIB: ".joblib",
            ModelFormat.CLOUDPICKLE: ".cloudpickle",
            ModelFormat.TORCH: ".pt",
            ModelFormat.TENSORFLOW: ".h5",
            ModelFormat.TENSORFLOW_SAVED_MODEL: "",  # Directory-based format
            ModelFormat.ONNX: ".onnx",
            ModelFormat.CUSTOM: ".model"
        }
        return format_extensions.get(format, ".model")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            str: Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read and update hash in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        format: str,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_save_fn: Optional[Callable[[Any, str], None]] = None
    ) -> Tuple[str, ModelMetadata]:
        """Save a model with metadata.
        
        Args:
            model: The model object to save
            model_name: Name of the model
            version: Version string
            format: Serialization format
            model_id: Optional model ID (generated if not provided)
            metadata: Optional additional metadata
            custom_save_fn: Optional custom save function for custom formats
            
        Returns:
            Tuple[str, ModelMetadata]: Model ID and metadata
        """
        # Generate model_id if not provided
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        # Get paths
        model_path = self._get_model_path(model_id, version, format)
        metadata_path = self._get_metadata_path(model_id, version)
        
        # Save the model based on format
        if format == ModelFormat.PICKLE:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        elif format == ModelFormat.JOBLIB:
            if not JOBLIB_AVAILABLE:
                raise ImportError("joblib is not installed. Install with 'pip install joblib'.")
            joblib.dump(model, model_path)
        
        elif format == ModelFormat.CLOUDPICKLE:
            if not CLOUDPICKLE_AVAILABLE:
                raise ImportError("cloudpickle is not installed. Install with 'pip install cloudpickle'.")
            with open(model_path, 'wb') as f:
                cloudpickle.dump(model, f)
        
        elif format == ModelFormat.TORCH:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Install with 'pip install torch'.")
            torch.save(model, model_path)
        
        elif format == ModelFormat.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Install with 'pip install tensorflow'.")
            model.save(model_path)
        
        elif format == ModelFormat.TENSORFLOW_SAVED_MODEL:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Install with 'pip install tensorflow'.")
            # For SavedModel format, model_path should be a directory
            model_path = os.path.join(os.path.dirname(model_path), "saved_model")
            tf.saved_model.save(model, model_path)
        
        elif format == ModelFormat.ONNX:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX is not installed. Install with 'pip install onnx onnxruntime'.")
            # ONNX export typically requires specific export functions based on the framework
            # This is a placeholder - actual implementation depends on the source framework
            if hasattr(model, "to_onnx"):
                model.to_onnx(model_path)
            else:
                raise ValueError("Model does not support direct ONNX export. Use a framework-specific converter.")
        
        elif format == ModelFormat.CUSTOM:
            if custom_save_fn is None:
                raise ValueError("Custom format requires a custom_save_fn")
            custom_save_fn(model, model_path)
        
        else:
            raise ValueError(f"Unsupported model format: {format}")
        
        # Calculate model hash for integrity verification
        if os.path.isfile(model_path):
            model_hash = self._calculate_file_hash(model_path)
        else:
            # For directory-based formats like TensorFlow SavedModel
            model_hash = None
        
        # Create and save metadata
        metadata_dict = metadata or {}
        metadata_obj = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            format=format,
            model_hash=model_hash,
            **metadata_dict
        )
        metadata_obj.save(metadata_path)
        
        logger.info(f"Model saved: {model_id} (version {version}) in {format} format")
        return model_id, metadata_obj
    
    def load_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        custom_load_fn: Optional[Callable[[str], Any]] = None
    ) -> Tuple[Any, ModelMetadata]:
        """Load a model and its metadata.
        
        Args:
            model_id: Model ID
            version: Model version (latest if not specified)
            custom_load_fn: Optional custom load function for custom formats
            
        Returns:
            Tuple[Any, ModelMetadata]: Loaded model and metadata
        """
        # If version is not specified, find the latest version
        if version is None:
            version = self.get_latest_version(model_id)
            if version is None:
                raise ValueError(f"No versions found for model {model_id}")
        
        # Get paths
        metadata_path = self._get_metadata_path(model_id, version)
        
        # Load metadata first to determine format
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found for model {model_id} version {version}")
        
        metadata = ModelMetadata.load(metadata_path)
        format = metadata.format
        
        # Get model path based on format
        model_path = self._get_model_path(model_id, version, format)
        
        # For directory-based formats like TensorFlow SavedModel
        if format == ModelFormat.TENSORFLOW_SAVED_MODEL:
            model_path = os.path.join(os.path.dirname(model_path), "saved_model")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Verify model integrity if hash is available
        if metadata.model_hash and os.path.isfile(model_path):
            current_hash = self._calculate_file_hash(model_path)
            if current_hash != metadata.model_hash:
                logger.warning(f"Model hash mismatch for {model_id} version {version}. File may be corrupted.")
        
        # Load the model based on format
        if format == ModelFormat.PICKLE:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        elif format == ModelFormat.JOBLIB:
            if not JOBLIB_AVAILABLE:
                raise ImportError("joblib is not installed. Install with 'pip install joblib'.")
            model = joblib.load(model_path)
        
        elif format == ModelFormat.CLOUDPICKLE:
            if not CLOUDPICKLE_AVAILABLE:
                raise ImportError("cloudpickle is not installed. Install with 'pip install cloudpickle'.")
            with open(model_path, 'rb') as f:
                model = cloudpickle.load(f)
        
        elif format == ModelFormat.TORCH:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Install with 'pip install torch'.")
            model = torch.load(model_path)
        
        elif format == ModelFormat.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Install with 'pip install tensorflow'.")
            model = tf.keras.models.load_model(model_path)
        
        elif format == ModelFormat.TENSORFLOW_SAVED_MODEL:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not installed. Install with 'pip install tensorflow'.")
            model = tf.saved_model.load(model_path)
        
        elif format == ModelFormat.ONNX:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX is not installed. Install with 'pip install onnx onnxruntime'.")
            # For ONNX, we typically return an inference session
            model = ort.InferenceSession(model_path)
        
        elif format == ModelFormat.CUSTOM:
            if custom_load_fn is None:
                raise ValueError("Custom format requires a custom_load_fn")
            model = custom_load_fn(model_path)
        
        else:
            raise ValueError(f"Unsupported model format: {format}")
        
        logger.info(f"Model loaded: {model_id} (version {version}) in {format} format")
        return model, metadata
    
    def get_latest_version(self, model_id: str) -> Optional[str]:
        """Get the latest version of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Optional[str]: Latest version or None if no versions exist
        """
        model_dir = os.path.join(self.base_dir, model_id)
        if not os.path.exists(model_dir):
            return None
        
        versions = [v for v in os.listdir(model_dir) 
                   if os.path.isdir(os.path.join(model_dir, v))]
        
        if not versions:
            return None
        
        # Try to sort semantically (assuming semantic versioning)
        try:
            from packaging import version
            versions.sort(key=lambda v: version.parse(v), reverse=True)
        except ImportError:
            # Fall back to string sorting if packaging is not available
            versions.sort(reverse=True)
        
        return versions[0]
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List[str]: List of available versions
        """
        model_dir = os.path.join(self.base_dir, model_id)
        if not os.path.exists(model_dir):
            return []
        
        versions = [v for v in os.listdir(model_dir) 
                   if os.path.isdir(os.path.join(model_dir, v))]
        
        # Try to sort semantically (assuming semantic versioning)
        try:
            from packaging import version
            versions.sort(key=lambda v: version.parse(v), reverse=True)
        except ImportError:
            # Fall back to string sorting if packaging is not available
            versions.sort(reverse=True)
        
        return versions
    
    def list_models(self) -> List[str]:
        """List all available models.
        
        Returns:
            List[str]: List of model IDs
        """
        if not os.path.exists(self.base_dir):
            return []
        
        return [m for m in os.listdir(self.base_dir) 
                if os.path.isdir(os.path.join(self.base_dir, m))]
    
    def get_model_info(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model.
        
        Args:
            model_id: Model ID
            version: Model version (latest if not specified)
            
        Returns:
            Dict[str, Any]: Model information
        """
        if version is None:
            version = self.get_latest_version(model_id)
            if version is None:
                raise ValueError(f"No versions found for model {model_id}")
        
        metadata_path = self._get_metadata_path(model_id, version)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found for model {model_id} version {version}")
        
        metadata = ModelMetadata.load(metadata_path)
        
        # Get all available versions
        all_versions = self.list_versions(model_id)
        
        # Get file size
        model_path = self._get_model_path(model_id, version, metadata.format)
        if os.path.isfile(model_path):
            file_size = os.path.getsize(model_path)
        elif os.path.isdir(model_path):
            # For directory-based formats like TensorFlow SavedModel
            file_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                           for dirpath, _, filenames in os.walk(model_path) 
                           for filename in filenames)
        else:
            file_size = None
        
        return {
            "model_id": model_id,
            "current_version": version,
            "all_versions": all_versions,
            "latest_version": all_versions[0] if all_versions else None,
            "metadata": metadata.to_dict(),
            "file_size": file_size,
            "file_path": model_path
        }
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a model and its metadata.
        
        Args:
            model_id: Model ID
            version: Specific version to delete (all versions if None)
            
        Returns:
            bool: True if successful
        """
        model_dir = os.path.join(self.base_dir, model_id)
        if not os.path.exists(model_dir):
            logger.warning(f"Model {model_id} not found")
            return False
        
        if version is None:
            # Delete all versions (entire model directory)
            shutil.rmtree(model_dir)
            logger.info(f"Deleted all versions of model {model_id}")
            return True
        else:
            # Delete specific version
            version_dir = os.path.join(model_dir, version)
            if not os.path.exists(version_dir):
                logger.warning(f"Version {version} of model {model_id} not found")
                return False
            
            shutil.rmtree(version_dir)
            logger.info(f"Deleted version {version} of model {model_id}")
            
            # Check if there are any versions left
            remaining_versions = [v for v in os.listdir(model_dir) 
                               if os.path.isdir(os.path.join(model_dir, v))]
            
            if not remaining_versions:
                # No versions left, remove the model directory
                os.rmdir(model_dir)
                logger.info(f"Removed empty model directory for {model_id}")
            
            return True
    
    def export_model(
        self,
        model_id: str,
        export_dir: str,
        version: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Export a model to a specified directory.
        
        Args:
            model_id: Model ID
            export_dir: Directory to export to
            version: Model version (latest if not specified)
            include_metadata: Whether to include metadata
            
        Returns:
            str: Path to the exported model
        """
        if version is None:
            version = self.get_latest_version(model_id)
            if version is None:
                raise ValueError(f"No versions found for model {model_id}")
        
        # Get metadata to determine format
        metadata_path = self._get_metadata_path(model_id, version)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found for model {model_id} version {version}")
        
        metadata = ModelMetadata.load(metadata_path)
        format = metadata.format
        
        # Get model path
        model_path = self._get_model_path(model_id, version, format)
        
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Export model file or directory
        if os.path.isfile(model_path):
            export_model_path = os.path.join(export_dir, f"{model_id}_v{version}{self._get_extension_for_format(format)}")
            shutil.copy2(model_path, export_model_path)
        elif os.path.isdir(model_path):
            # For directory-based formats like TensorFlow SavedModel
            export_model_path = os.path.join(export_dir, f"{model_id}_v{version}")
            if os.path.exists(export_model_path):
                shutil.rmtree(export_model_path)
            shutil.copytree(model_path, export_model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Export metadata if requested
        if include_metadata:
            export_metadata_path = os.path.join(export_dir, f"{model_id}_v{version}_metadata.json")
            shutil.copy2(metadata_path, export_metadata_path)
        
        logger.info(f"Exported model {model_id} (version {version}) to {export_dir}")
        return export_model_path
    
    def import_model(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        model_name: Optional[str] = None,
        format: Optional[str] = None
    ) -> Tuple[str, str]:
        """Import a model from a file or directory.
        
        Args:
            model_path: Path to the model file or directory
            metadata_path: Path to metadata file (optional)
            model_id: Model ID (generated if not provided)
            version: Version (default: "1.0.0")
            model_name: Model name (required if metadata not provided)
            format: Model format (required if metadata not provided)
            
        Returns:
            Tuple[str, str]: Model ID and version
        """
        # Generate model_id if not provided
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        # Default version
        if version is None:
            version = "1.0.0"
        
        # Load metadata if provided
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata = ModelMetadata.load(metadata_path)
                
                # Use values from metadata if not explicitly provided
                if model_name is None:
                    model_name = metadata.model_name
                if format is None:
                    format = metadata.format
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
                metadata = None
        
        # Ensure required parameters are provided
        if model_name is None:
            raise ValueError("model_name is required when metadata is not provided or invalid")
        if format is None:
            # Try to infer format from file extension
            if os.path.isfile(model_path):
                ext = os.path.splitext(model_path)[1].lower()
                format_map = {
                    ".pkl": ModelFormat.PICKLE,
                    ".joblib": ModelFormat.JOBLIB,
                    ".cloudpickle": ModelFormat.CLOUDPICKLE,
                    ".pt": ModelFormat.TORCH,
                    ".pth": ModelFormat.TORCH,
                    ".h5": ModelFormat.TENSORFLOW,
                    ".onnx": ModelFormat.ONNX
                }
                format = format_map.get(ext)
            
            if format is None:
                # Check if it's a TensorFlow SavedModel directory
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
                    format = ModelFormat.TENSORFLOW_SAVED_MODEL
                else:
                    raise ValueError("format is required when metadata is not provided or invalid and cannot be inferred")
        
        # Get destination paths
        dest_model_path = self._get_model_path(model_id, version, format)
        dest_metadata_path = self._get_metadata_path(model_id, version)
        
        # Copy model file or directory
        if os.path.isfile(model_path):
            os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)
            shutil.copy2(model_path, dest_model_path)
            model_hash = self._calculate_file_hash(dest_model_path)
        elif os.path.isdir(model_path):
            # For directory-based formats like TensorFlow SavedModel
            if format == ModelFormat.TENSORFLOW_SAVED_MODEL:
                dest_model_path = os.path.join(os.path.dirname(dest_model_path), "saved_model")
            
            if os.path.exists(dest_model_path):
                shutil.rmtree(dest_model_path)
            
            shutil.copytree(model_path, dest_model_path)
            model_hash = None  # Hash not calculated for directories
        else:
            raise FileNotFoundError(f"Model file or directory not found at {model_path}")
        
        # Create or update metadata
        if metadata is None:
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                format=format,
                model_hash=model_hash
            )
        else:
            # Update metadata with new values
            metadata.model_id = model_id
            metadata.version = version
            metadata.model_hash = model_hash
            metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save metadata
        metadata.save(dest_metadata_path)
        
        logger.info(f"Imported model as {model_id} (version {version})")
        return model_id, version
    
    def convert_model(
        self,
        model_id: str,
        target_format: str,
        version: Optional[str] = None,
        new_version: Optional[str] = None,
        converter_fn: Optional[Callable[[Any], Any]] = None
    ) -> Tuple[str, str]:
        """Convert a model to a different format.
        
        Args:
            model_id: Model ID
            target_format: Target format to convert to
            version: Source version (latest if not specified)
            new_version: New version for converted model (incremented if not specified)
            converter_fn: Custom converter function for complex conversions
            
        Returns:
            Tuple[str, str]: Model ID and new version
        """
        # Load the source model
        model, metadata = self.load_model(model_id, version)
        
        source_format = metadata.format
        source_version = metadata.version
        
        # Generate new version if not specified
        if new_version is None:
            try:
                from packaging import version as pkg_version
                v = pkg_version.parse(source_version)
                if hasattr(v, 'major') and hasattr(v, 'minor') and hasattr(v, 'micro'):
                    new_version = f"{v.major}.{v.minor}.{v.micro + 1}"
                else:
                    new_version = f"{source_version}.1"
            except (ImportError, ValueError):
                # Fall back to simple version increment
                new_version = f"{source_version}.1"
        
        # Handle conversion based on source and target formats
        converted_model = None
        
        # If custom converter is provided, use it
        if converter_fn is not None:
            converted_model = converter_fn(model)
        else:
            # Handle common conversion paths
            if source_format == target_format:
                # No conversion needed
                converted_model = model
            
            elif target_format == ModelFormat.ONNX:
                # Convert to ONNX (framework-specific)
                if not ONNX_AVAILABLE:
                    raise ImportError("ONNX is not installed. Install with 'pip install onnx onnxruntime'.")
                
                if source_format == ModelFormat.TORCH:
                    if not TORCH_AVAILABLE:
                        raise ImportError("PyTorch is not installed. Install with 'pip install torch'.")
                    
                    # This is a simplified example - actual conversion depends on model architecture
                    try:
                        import torch.onnx
                        # Need dummy input of correct shape for tracing
                        dummy_input = None  # This should be set based on model requirements
                        if dummy_input is None:
                            raise ValueError("ONNX conversion from PyTorch requires dummy_input. Use converter_fn.")
                        
                        # Temporary path for ONNX model
                        temp_path = os.path.join(self.base_dir, f"temp_{uuid.uuid4()}.onnx")
                        torch.onnx.export(model, dummy_input, temp_path)
                        
                        # Load the ONNX model to return
                        converted_model = onnx.load(temp_path)
                        os.remove(temp_path)  # Clean up
                    except Exception as e:
                        raise ValueError(f"Failed to convert PyTorch model to ONNX: {e}")
                
                elif source_format in [ModelFormat.TENSORFLOW, ModelFormat.TENSORFLOW_SAVED_MODEL]:
                    if not TF_AVAILABLE:
                        raise ImportError("TensorFlow is not installed. Install with 'pip install tensorflow'.")
                    
                    try:
                        import tf2onnx
                        # This is a simplified example - actual conversion depends on model architecture
                        # Temporary path for ONNX model
                        temp_path = os.path.join(self.base_dir, f"temp_{uuid.uuid4()}.onnx")
                        
                        if source_format == ModelFormat.TENSORFLOW_SAVED_MODEL:
                            # Convert SavedModel
                            tf2onnx.convert.from_saved_model(model, output_path=temp_path)
                        else:
                            # Convert Keras model
                            tf2onnx.convert.from_keras(model, output_path=temp_path)
                        
                        # Load the ONNX model to return
                        converted_model = onnx.load(temp_path)
                        os.remove(temp_path)  # Clean up
                    except Exception as e:
                        raise ValueError(f"Failed to convert TensorFlow model to ONNX: {e}")
                
                else:
                    raise ValueError(f"Conversion from {source_format} to ONNX not supported. Use converter_fn.")
            
            elif source_format == ModelFormat.PICKLE and target_format == ModelFormat.JOBLIB:
                if not JOBLIB_AVAILABLE:
                    raise ImportError("joblib is not installed. Install with 'pip install joblib'.")
                # Direct conversion (same Python object)
                converted_model = model
            
            elif source_format == ModelFormat.JOBLIB and target_format == ModelFormat.PICKLE:
                # Direct conversion (same Python object)
                converted_model = model
            
            else:
                raise ValueError(f"Conversion from {source_format} to {target_format} not supported. Use converter_fn.")
        
        if converted_model is None:
            raise ValueError("Model conversion failed")
        
        # Save the converted model
        # Update metadata for the new version
        metadata_dict = metadata.to_dict()
        metadata_dict.update({
            "format": target_format,
            "version": new_version,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "custom_metadata": {
                **(metadata_dict.get("custom_metadata", {}) or {}),
                "converted_from": {
                    "format": source_format,
                    "version": source_version,
                    "conversion_date": datetime.datetime.now().isoformat()
                }
            }
        })
        
        # Remove model_hash as it will be recalculated
        if "model_hash" in metadata_dict:
            del metadata_dict["model_hash"]
        
        # Save the converted model
        _, new_metadata = self.save_model(
            model=converted_model,
            model_name=metadata.model_name,
            version=new_version,
            format=target_format,
            model_id=model_id,
            metadata=metadata_dict
        )
        
        logger.info(f"Converted model {model_id} from {source_format} (v{source_version}) to {target_format} (v{new_version})")
        return model_id, new_version


# Factory function to create model serializer
def create_model_serializer(base_dir: Optional[str] = None) -> ModelSerializer:
    """Create a model serializer with the specified base directory.
    
    Args:
        base_dir: Base directory for model storage
        
    Returns:
        ModelSerializer: Configured model serializer
    """
    return ModelSerializer(base_dir)


# Utility functions

def get_model_registry_info() -> Dict[str, Any]:
    """Get information about the model registry.
    
    Returns:
        Dict[str, Any]: Registry information
    """
    serializer = ModelSerializer()
    models = serializer.list_models()
    
    registry_info = {
        "base_dir": serializer.base_dir,
        "model_count": len(models),
        "models": {}
    }
    
    for model_id in models:
        try:
            latest_version = serializer.get_latest_version(model_id)
            if latest_version:
                model_info = serializer.get_model_info(model_id, latest_version)
                registry_info["models"][model_id] = {
                    "name": model_info["metadata"].get("model_name", "Unknown"),
                    "latest_version": latest_version,
                    "format": model_info["metadata"].get("format", "Unknown"),
                    "versions": len(serializer.list_versions(model_id))
                }
        except Exception as e:
            logger.warning(f"Error getting info for model {model_id}: {e}")
    
    return registry_info


def find_models_by_tag(tag: str) -> List[Dict[str, Any]]:
    """Find models with a specific tag.
    
    Args:
        tag: Tag to search for
        
    Returns:
        List[Dict[str, Any]]: List of matching models
    """
    serializer = ModelSerializer()
    models = serializer.list_models()
    matching_models = []
    
    for model_id in models:
        try:
            versions = serializer.list_versions(model_id)
            for version in versions:
                metadata_path = serializer._get_metadata_path(model_id, version)
                if os.path.exists(metadata_path):
                    metadata = ModelMetadata.load(metadata_path)
                    if tag in metadata.tags:
                        matching_models.append({
                            "model_id": model_id,
                            "version": version,
                            "name": metadata.model_name,
                            "format": metadata.format,
                            "tags": metadata.tags
                        })
        except Exception as e:
            logger.warning(f"Error checking tags for model {model_id}: {e}")
    
    return matching_models


def find_models_by_metadata(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find models matching metadata criteria.
    
    Args:
        query: Dictionary of metadata fields to match
        
    Returns:
        List[Dict[str, Any]]: List of matching models
    """
    serializer = ModelSerializer()
    models = serializer.list_models()
    matching_models = []
    
    for model_id in models:
        try:
            versions = serializer.list_versions(model_id)
            for version in versions:
                metadata_path = serializer._get_metadata_path(model_id, version)
                if os.path.exists(metadata_path):
                    metadata = ModelMetadata.load(metadata_path)
                    metadata_dict = metadata.to_dict()
                    
                    # Check if all query criteria match
                    match = True
                    for key, value in query.items():
                        # Handle nested keys with dot notation (e.g., "custom_metadata.source")
                        if '.' in key:
                            parts = key.split('.')
                            current = metadata_dict
                            for part in parts[:-1]:
                                if part not in current or not isinstance(current[part], dict):
                                    match = False
                                    break
                                current = current[part]
                            
                            if match and (parts[-1] not in current or current[parts[-1]] != value):
                                match = False
                        elif key not in metadata_dict or metadata_dict[key] != value:
                            match = False
                            break
                    
                    if match:
                        matching_models.append({
                            "model_id": model_id,
                            "version": version,
                            "name": metadata.model_name,
                            "format": metadata.format,
                            "metadata": metadata_dict
                        })
        except Exception as e:
            logger.warning(f"Error checking metadata for model {model_id}: {e}")
    
    return matching_models


def backup_model_registry(backup_dir: str) -> str:
    """Backup the entire model registry.
    
    Args:
        backup_dir: Directory to store the backup
        
    Returns:
        str: Path to the backup directory
    """
    serializer = ModelSerializer()
    source_dir = serializer.base_dir
    
    # Create backup directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"model_registry_backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)
    
    # Copy all files
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            dest_item = os.path.join(backup_path, item)
            
            if os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item)
            else:
                shutil.copy2(source_item, dest_item)
    
    # Create backup info file
    backup_info = {
        "timestamp": timestamp,
        "source_directory": source_dir,
        "backup_directory": backup_path,
        "registry_info": get_model_registry_info()
    }
    
    with open(os.path.join(backup_path, "backup_info.json"), 'w') as f:
        json.dump(backup_info, f, indent=2)
    
    logger.info(f"Model registry backed up to {backup_path}")
    return backup_path


def restore_model_registry(backup_path: str, target_dir: Optional[str] = None) -> bool:
    """Restore the model registry from a backup.
    
    Args:
        backup_path: Path to the backup directory
        target_dir: Target directory to restore to (default: original location)
        
    Returns:
        bool: True if successful
    """
    # Check if backup exists
    if not os.path.exists(backup_path) or not os.path.isdir(backup_path):
        logger.error(f"Backup directory not found: {backup_path}")
        return False
    
    # Check for backup info file
    backup_info_path = os.path.join(backup_path, "backup_info.json")
    if os.path.exists(backup_info_path):
        with open(backup_info_path, 'r') as f:
            backup_info = json.load(f)
        
        # Use original location if target_dir not specified
        if target_dir is None:
            target_dir = backup_info.get("source_directory")
            if not target_dir:
                target_dir = DEFAULT_MODEL_DIR
    else:
        # No backup info, use default if target not specified
        if target_dir is None:
            target_dir = DEFAULT_MODEL_DIR
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all files except backup_info.json
    for item in os.listdir(backup_path):
        if item == "backup_info.json":
            continue
        
        source_item = os.path.join(backup_path, item)
        dest_item = os.path.join(target_dir, item)
        
        # Remove existing item if it exists
        if os.path.exists(dest_item):
            if os.path.isdir(dest_item):
                shutil.rmtree(dest_item)
            else:
                os.remove(dest_item)
        
        # Copy item
        if os.path.isdir(source_item):
            shutil.copytree(source_item, dest_item)
        else:
            shutil.copy2(source_item, dest_item)
    
    logger.info(f"Model registry restored to {target_dir}")
    return True