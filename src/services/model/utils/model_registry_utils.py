"""Utility functions for the model registry system.

This module provides utility functions for working with the model registry,
including version comparison, path handling, and metadata validation.
"""

import os
import re
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from src.services.model.config.model_registry_config import ModelRegistryConfig


def validate_model_name(model_name: str) -> bool:
    """Validate that a model name follows the required format.
    
    Args:
        model_name: The model name to validate.
        
    Returns:
        bool: True if the model name is valid, False otherwise.
    """
    # Model names should be alphanumeric with underscores and hyphens
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, model_name))


def validate_model_version(version: str) -> bool:
    """Validate that a model version follows semantic versioning format.
    
    Args:
        version: The version string to validate.
        
    Returns:
        bool: True if the version is valid, False otherwise.
    """
    # Semantic versioning pattern: MAJOR.MINOR.PATCH
    pattern = r'^\d+\.\d+\.\d+$'
    return bool(re.match(pattern, version))


def compare_versions(version1: str, version2: str) -> int:
    """Compare two semantic version strings.
    
    Args:
        version1: The first version string.
        version2: The second version string.
        
    Returns:
        int: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    if not validate_model_version(version1) or not validate_model_version(version2):
        raise ValueError("Invalid version format. Expected format: MAJOR.MINOR.PATCH")
    
    v1_parts = list(map(int, version1.split('.')))
    v2_parts = list(map(int, version2.split('.')))
    
    for i in range(3):  # Compare MAJOR, MINOR, PATCH
        if v1_parts[i] < v2_parts[i]:
            return -1
        elif v1_parts[i] > v2_parts[i]:
            return 1
    
    return 0  # Versions are equal


def get_model_path(models_dir: str, model_name: str, model_version: str) -> str:
    """Get the full path to a model directory.
    
    Args:
        models_dir: The base directory for model storage.
        model_name: The name of the model.
        model_version: The version of the model.
        
    Returns:
        str: The full path to the model directory.
    """
    return os.path.join(models_dir, model_name, model_version)


def get_model_file_path(models_dir: str, model_name: str, model_version: str, filename: str = None) -> str:
    """Get the full path to a model file.
    
    Args:
        models_dir: The base directory for model storage.
        model_name: The name of the model.
        model_version: The version of the model.
        filename: The name of the model file. If None, uses the default model filename.
        
    Returns:
        str: The full path to the model file.
    """
    if filename is None:
        filename = ModelRegistryConfig.DEFAULT_MODEL_FILENAME
    
    return os.path.join(get_model_path(models_dir, model_name, model_version), filename)


def get_metadata_file_path(models_dir: str, model_name: str, model_version: str) -> str:
    """Get the full path to a model's metadata file.
    
    Args:
        models_dir: The base directory for model storage.
        model_name: The name of the model.
        model_version: The version of the model.
        
    Returns:
        str: The full path to the metadata file.
    """
    return os.path.join(
        get_model_path(models_dir, model_name, model_version),
        ModelRegistryConfig.DEFAULT_METADATA_FILENAME
    )


def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that model metadata contains all required fields.
    
    Args:
        metadata: The metadata dictionary to validate.
        
    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating if the metadata is valid,
            and a list of missing required fields.
    """
    missing_fields = []
    
    for field in ModelRegistryConfig.REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def generate_model_id(model_name: str, model_version: str, timestamp: str = None) -> str:
    """Generate a unique ID for a model.
    
    Args:
        model_name: The name of the model.
        model_version: The version of the model.
        timestamp: A timestamp string. If None, uses the current time.
        
    Returns:
        str: A unique model ID.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    # Combine model name, version, and timestamp to create a unique ID
    combined = f"{model_name}_{model_version}_{timestamp}"
    
    # Generate a hash of the combined string
    return hashlib.md5(combined.encode()).hexdigest()


def get_current_timestamp() -> str:
    """Get the current timestamp in ISO format.
    
    Returns:
        str: The current timestamp in ISO format.
    """
    return datetime.datetime.now().isoformat()


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.
    
    Args:
        file_path: The path to the JSON file.
        
    Returns:
        Dict[str, Any]: The loaded JSON data as a dictionary.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save a dictionary to a JSON file.
    
    Args:
        data: The dictionary to save.
        file_path: The path to the JSON file.
        
    Raises:
        IOError: If the file cannot be written.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def merge_metadata(base_metadata: Dict[str, Any], update_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two metadata dictionaries.
    
    Args:
        base_metadata: The base metadata dictionary.
        update_metadata: The metadata dictionary with updates.
        
    Returns:
        Dict[str, Any]: The merged metadata dictionary.
    """
    merged = base_metadata.copy()
    
    for key, value in update_metadata.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_metadata(merged[key], value)
        else:
            # Replace or add the value
            merged[key] = value
    
    return merged


def get_model_info_for_prediction(
    symbol: str,
    timeframe: str,
    target_type: str,
    forecast_horizon: int,
    model_type: Optional[str] = None
) -> Dict[str, str]:
    """Get model information for prediction.
    
    Args:
        symbol: The trading symbol (e.g., 'BTC-USD').
        timeframe: The timeframe (e.g., '1h', '1d').
        target_type: The prediction target type (e.g., 'price', 'return').
        forecast_horizon: The forecast horizon in periods.
        model_type: The model type (e.g., 'random_forest', 'lstm'). If None, uses any available model.
        
    Returns:
        Dict[str, str]: A dictionary containing model information.
    """
    # Construct a standardized model name based on the parameters
    model_name = f"{symbol}_{timeframe}_{target_type}_{forecast_horizon}"
    
    # If model_type is specified, append it to the model name
    if model_type is not None:
        model_name = f"{model_name}_{model_type}"
    
    return {
        "model_name": model_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "target_type": target_type,
        "forecast_horizon": str(forecast_horizon),
        "model_type": model_type if model_type is not None else "any"
    }


def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse a standardized model name into its components.
    
    Args:
        model_name: The standardized model name.
        
    Returns:
        Dict[str, str]: A dictionary containing the parsed components.
        
    Raises:
        ValueError: If the model name does not follow the expected format.
    """
    # Expected format: {symbol}_{timeframe}_{target_type}_{forecast_horizon}[_{model_type}]
    parts = model_name.split('_')
    
    if len(parts) < 4:
        raise ValueError(
            f"Invalid model name format: {model_name}. "
            f"Expected format: {{symbol}}_{{timeframe}}_{{target_type}}_{{forecast_horizon}}[_{{model_type}}]"
        )
    
    result = {
        "symbol": parts[0],
        "timeframe": parts[1],
        "target_type": parts[2],
        "forecast_horizon": parts[3]
    }
    
    # If model_type is included
    if len(parts) > 4:
        result["model_type"] = '_'.join(parts[4:])  # Join remaining parts in case model_type contains underscores
    
    return result