"""Deep Learning Model Serialization for Friday AI Trading System.

This module provides functionality for serializing and deserializing deep learning models
from frameworks like PyTorch and TensorFlow with specialized handling requirements.
"""

import os
import json
import pickle
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO

from src.infrastructure.logging import get_logger
from src.services.model.model_serialization import ModelSerializer

# Create logger
logger = get_logger(__name__)


class DeepLearningSerializer(ModelSerializer):
    """Serializer for deep learning models from frameworks like PyTorch and TensorFlow.

    This class extends the base ModelSerializer to provide specialized functionality
    for handling deep learning models with their unique serialization requirements.

    Attributes:
        default_format: The default serialization format.
        framework_handlers: Dictionary of handlers for different frameworks.
    """

    # Supported deep learning frameworks
    FRAMEWORKS = ["pytorch", "tensorflow", "keras"]

    # Extended formats including deep learning specific ones
    FORMATS = ModelSerializer.FORMATS + ["pytorch", "tensorflow", "onnx"]

    def __init__(self, default_format: str = "joblib"):
        """Initialize the deep learning model serializer.

        Args:
            default_format: The default serialization format. 
                            Can be one of "joblib", "pickle", "json", "pytorch", "tensorflow", or "onnx".

        Raises:
            ValueError: If the default format is not supported.
        """
        # Initialize the parent class
        super().__init__(default_format=default_format)
        
        # Initialize framework handlers
        self.framework_handlers = {
            "pytorch": self._handle_pytorch,
            "tensorflow": self._handle_tensorflow,
            "keras": self._handle_keras,
        }
        
        logger.info(f"Initialized DeepLearningSerializer with default format: {default_format}")

    def detect_framework(self, model: Any) -> str:
        """Detect the deep learning framework of a model.

        Args:
            model: The model to detect the framework for.

        Returns:
            str: The detected framework name or "unknown".
        """
        # Check for PyTorch models
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return "pytorch"
        except ImportError:
            pass

        # Check for TensorFlow/Keras models
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model) or \
               isinstance(model, tf.Module) or \
               hasattr(model, 'keras_api'):
                return "tensorflow"
        except ImportError:
            pass

        # Check for standalone Keras models
        try:
            import keras
            if isinstance(model, keras.Model):
                return "keras"
        except ImportError:
            pass

        return "unknown"

    def serialize(self, model: Any, file_path: str, format: Optional[str] = None) -> str:
        """Serialize a deep learning model to a file.

        Args:
            model: The model to serialize.
            file_path: Path to save the serialized model.
            format: Serialization format. If None, auto-detected or uses default format.

        Returns:
            str: Path to the serialized model file.

        Raises:
            ValueError: If the format is not supported or framework detection fails.
            IOError: If there is an error writing to the file.
        """
        # Auto-detect framework if format is None
        if format is None:
            framework = self.detect_framework(model)
            if framework != "unknown":
                format = framework
            else:
                format = self.default_format

        # Check if format is supported
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported serialization format: {format}. Must be one of {self.FORMATS}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            # Handle deep learning specific formats
            if format in self.FRAMEWORKS:
                if format in self.framework_handlers:
                    return self.framework_handlers[format](model, file_path, "save")
                else:
                    raise ValueError(f"No handler available for framework: {format}")
            elif format == "onnx":
                return self._handle_onnx(model, file_path, "save")
            else:
                # Use parent class for standard formats
                return super().serialize(model, file_path, format)
        except Exception as e:
            logger.error(f"Error serializing deep learning model: {str(e)}")
            raise

    def deserialize(self, file_path: str, format: Optional[str] = None) -> Any:
        """Deserialize a deep learning model from a file.

        Args:
            file_path: Path to the serialized model file.
            format: Serialization format. If None, inferred from file extension or uses default format.

        Returns:
            Any: The deserialized model.

        Raises:
            ValueError: If the format is not supported or the file does not exist.
            IOError: If there is an error reading from the file.
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"Model file {file_path} not found")

        # Infer format from file extension if not specified
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pt" or ext == ".pth":
                format = "pytorch"
            elif ext == ".h5" or ext == ".keras":
                format = "tensorflow"
            elif ext == ".onnx":
                format = "onnx"
            elif ext == ".joblib":
                format = "joblib"
            elif ext == ".pkl" or ext == ".pickle":
                format = "pickle"
            elif ext == ".json":
                format = "json"
            else:
                format = self.default_format

        # Check if format is supported
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported serialization format: {format}. Must be one of {self.FORMATS}")

        try:
            # Handle deep learning specific formats
            if format in self.FRAMEWORKS:
                if format in self.framework_handlers:
                    return self.framework_handlers[format](None, file_path, "load")
                else:
                    raise ValueError(f"No handler available for framework: {format}")
            elif format == "onnx":
                return self._handle_onnx(None, file_path, "load")
            else:
                # Use parent class for standard formats
                return super().deserialize(file_path, format)
        except Exception as e:
            logger.error(f"Error deserializing deep learning model: {str(e)}")
            raise

    def _handle_pytorch(self, model: Any, file_path: str, operation: str) -> Any:
        """Handle PyTorch model serialization/deserialization.

        Args:
            model: The PyTorch model (for save operation).
            file_path: Path to the model file.
            operation: Either "save" or "load".

        Returns:
            str or Any: File path for save operation, loaded model for load operation.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If the operation is invalid.
        """
        try:
            import torch

            if operation == "save":
                torch.save(model.state_dict(), file_path)
                logger.info(f"Saved PyTorch model to {file_path}")
                return file_path
            elif operation == "load":
                # Note: This requires the model class to be defined and imported
                # In a real implementation, you might need to store and load the model architecture too
                # or use a more sophisticated approach like TorchScript
                raise NotImplementedError(
                    "Loading PyTorch models requires model architecture information. "
                    "Consider using TorchScript or storing architecture information separately."
                )
            else:
                raise ValueError(f"Invalid operation: {operation}. Must be 'save' or 'load'.")
        except ImportError:
            logger.error("PyTorch is not installed. Please install it to use PyTorch serialization.")
            raise

    def _handle_tensorflow(self, model: Any, file_path: str, operation: str) -> Any:
        """Handle TensorFlow/Keras model serialization/deserialization.

        Args:
            model: The TensorFlow/Keras model (for save operation).
            file_path: Path to the model file.
            operation: Either "save" or "load".

        Returns:
            str or Any: File path for save operation, loaded model for load operation.

        Raises:
            ImportError: If TensorFlow is not installed.
            ValueError: If the operation is invalid.
        """
        try:
            import tensorflow as tf

            if operation == "save":
                model.save(file_path)
                logger.info(f"Saved TensorFlow model to {file_path}")
                return file_path
            elif operation == "load":
                loaded_model = tf.keras.models.load_model(file_path)
                logger.info(f"Loaded TensorFlow model from {file_path}")
                return loaded_model
            else:
                raise ValueError(f"Invalid operation: {operation}. Must be 'save' or 'load'.")
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it to use TensorFlow serialization.")
            raise

    def _handle_keras(self, model: Any, file_path: str, operation: str) -> Any:
        """Handle standalone Keras model serialization/deserialization.

        Args:
            model: The Keras model (for save operation).
            file_path: Path to the model file.
            operation: Either "save" or "load".

        Returns:
            str or Any: File path for save operation, loaded model for load operation.

        Raises:
            ImportError: If Keras is not installed.
            ValueError: If the operation is invalid.
        """
        try:
            import keras

            if operation == "save":
                model.save(file_path)
                logger.info(f"Saved Keras model to {file_path}")
                return file_path
            elif operation == "load":
                loaded_model = keras.models.load_model(file_path)
                logger.info(f"Loaded Keras model from {file_path}")
                return loaded_model
            else:
                raise ValueError(f"Invalid operation: {operation}. Must be 'save' or 'load'.")
        except ImportError:
            logger.error("Keras is not installed. Please install it to use Keras serialization.")
            raise

    def _handle_onnx(self, model: Any, file_path: str, operation: str) -> Any:
        """Handle ONNX model serialization/deserialization.

        Args:
            model: The model to convert to ONNX (for save operation).
            file_path: Path to the ONNX model file.
            operation: Either "save" or "load".

        Returns:
            str or Any: File path for save operation, loaded model for load operation.

        Raises:
            ImportError: If ONNX is not installed.
            ValueError: If the operation is invalid.
        """
        try:
            import onnx

            if operation == "save":
                # This is a simplified example - actual ONNX conversion depends on the framework
                framework = self.detect_framework(model)
                
                if framework == "pytorch":
                    try:
                        import torch.onnx
                        # Dummy input for tracing - this would need to be customized
                        dummy_input = torch.randn(1, model.input_size)
                        torch.onnx.export(model, dummy_input, file_path)
                        logger.info(f"Saved PyTorch model to ONNX format at {file_path}")
                        return file_path
                    except Exception as e:
                        logger.error(f"Error converting PyTorch model to ONNX: {str(e)}")
                        raise
                        
                elif framework == "tensorflow":
                    try:
                        import tf2onnx
                        import tensorflow as tf
                        # Convert model to ONNX
                        model_proto, _ = tf2onnx.convert.from_keras(model)
                        onnx.save(model_proto, file_path)
                        logger.info(f"Saved TensorFlow model to ONNX format at {file_path}")
                        return file_path
                    except Exception as e:
                        logger.error(f"Error converting TensorFlow model to ONNX: {str(e)}")
                        raise
                        
                else:
                    raise ValueError(f"ONNX conversion not supported for framework: {framework}")
                    
            elif operation == "load":
                # Load ONNX model
                onnx_model = onnx.load(file_path)
                logger.info(f"Loaded ONNX model from {file_path}")
                
                # Note: To actually use the ONNX model for inference, you would need an ONNX runtime
                # This is just loading the model structure
                return onnx_model
                
            else:
                raise ValueError(f"Invalid operation: {operation}. Must be 'save' or 'load'.")
                
        except ImportError:
            logger.error("ONNX is not installed. Please install it to use ONNX serialization.")
            raise

    def compute_model_hash(self, model: Any) -> str:
        """Compute a hash of the deep learning model for versioning and comparison.

        Args:
            model: The model to hash.

        Returns:
            str: The hash of the model.
        """
        try:
            framework = self.detect_framework(model)
            
            if framework == "pytorch":
                try:
                    import torch
                    # Get model state dict and convert to bytes
                    state_dict = model.state_dict()
                    buffer = pickle.dumps(state_dict)
                    model_hash = hashlib.sha256(buffer).hexdigest()
                    return model_hash
                except Exception as e:
                    logger.error(f"Error computing PyTorch model hash: {str(e)}")
                    return super().compute_model_hash(model)
                    
            elif framework == "tensorflow" or framework == "keras":
                try:
                    # Get model weights as numpy arrays and convert to bytes
                    weights = model.get_weights()
                    buffer = pickle.dumps(weights)
                    model_hash = hashlib.sha256(buffer).hexdigest()
                    return model_hash
                except Exception as e:
                    logger.error(f"Error computing TensorFlow/Keras model hash: {str(e)}")
                    return super().compute_model_hash(model)
            else:
                # Fall back to parent class implementation
                return super().compute_model_hash(model)
                
        except Exception as e:
            logger.error(f"Error computing deep learning model hash: {str(e)}")
            return f"error_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(model))}"