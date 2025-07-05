"""Configuration Manager for Friday AI Trading System.

This module provides a comprehensive configuration management system with support for:
1. Environment-specific configuration overrides
2. Configuration validation
3. Secure credential management
4. Configuration loading/saving from/to various formats
5. Dynamic configuration updates

Classes:
    ConfigurationManager: Main configuration manager class
    ConfigValidator: Configuration validation class
    CredentialManager: Secure credential management class
    EnvironmentManager: Environment-specific configuration manager

Usage:
    # Get configuration instance
    config = ConfigurationManager.get_instance()
    
    # Get a configuration value
    db_host = config.get('database.host')
    
    # Set a configuration value
    config.set('database.host', 'localhost')
    
    # Load configuration from file
    config.load_from_file('config.json')
    
    # Save configuration to file
    config.save_to_file('config.json')
"""

import os
import json
import yaml
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from copy import deepcopy
import re

# Set up logging
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validation class.
    
    This class provides methods to validate configuration values against schemas,
    types, ranges, and custom validation functions.
    
    Attributes:
        _schemas: Dictionary of validation schemas for configuration sections
    """
    
    def __init__(self):
        """Initialize the ConfigValidator."""
        self._schemas = {}
    
    def register_schema(self, section: str, schema: Dict[str, Any]) -> None:
        """Register a validation schema for a configuration section.
        
        Args:
            section: The configuration section name
            schema: The validation schema as a dictionary
        """
        self._schemas[section] = schema
    
    def validate_type(self, value: Any, expected_type: Union[type, List[type]]) -> bool:
        """Validate that a value is of the expected type.
        
        Args:
            value: The value to validate
            expected_type: The expected type or list of types
            
        Returns:
            bool: True if the value is of the expected type, False otherwise
        """
        if isinstance(expected_type, list):
            return any(isinstance(value, t) for t in expected_type)
        return isinstance(value, expected_type)
    
    def validate_range(self, value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                      max_value: Optional[Union[int, float]] = None) -> bool:
        """Validate that a numeric value is within the specified range.
        
        Args:
            value: The value to validate
            min_value: The minimum allowed value (inclusive)
            max_value: The maximum allowed value (inclusive)
            
        Returns:
            bool: True if the value is within the range, False otherwise
        """
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    
    def validate_string(self, value: str, pattern: Optional[str] = None, 
                        min_length: Optional[int] = None, max_length: Optional[int] = None) -> bool:
        """Validate a string value.
        
        Args:
            value: The string to validate
            pattern: Regular expression pattern to match
            min_length: Minimum string length
            max_length: Maximum string length
            
        Returns:
            bool: True if the string is valid, False otherwise
        """
        if not isinstance(value, str):
            return False
        
        if min_length is not None and len(value) < min_length:
            return False
        
        if max_length is not None and len(value) > max_length:
            return False
        
        if pattern is not None and not re.match(pattern, value):
            return False
        
        return True
    
    def validate_config(self, config: Dict[str, Any], section: Optional[str] = None) -> List[str]:
        """Validate the configuration against registered schemas.
        
        Args:
            config: The configuration dictionary to validate
            section: Optional section name to validate only that section
            
        Returns:
            List[str]: List of validation error messages, empty if valid
        """
        errors = []
        
        if section is not None:
            # Validate only the specified section
            if section not in self._schemas:
                return [f"No validation schema registered for section '{section}'"]
            
            if section not in config:
                return [f"Configuration section '{section}' not found"]
            
            return self._validate_section(section, config[section])
        
        # Validate all registered sections
        for section_name, schema in self._schemas.items():
            if section_name not in config:
                errors.append(f"Required configuration section '{section_name}' not found")
                continue
            
            section_errors = self._validate_section(section_name, config[section_name])
            errors.extend(section_errors)
        
        return errors
    
    def _validate_section(self, section_name: str, section_config: Dict[str, Any]) -> List[str]:
        """Validate a configuration section against its schema.
        
        Args:
            section_name: The name of the section
            section_config: The section configuration dictionary
            
        Returns:
            List[str]: List of validation error messages, empty if valid
        """
        errors = []
        schema = self._schemas[section_name]
        
        # Check for required fields
        for field_name, field_schema in schema.items():
            if field_schema.get('required', False) and field_name not in section_config:
                errors.append(f"Required field '{field_name}' not found in section '{section_name}'")
        
        # Validate each field
        for field_name, field_value in section_config.items():
            if field_name not in schema:
                if not schema.get('allow_extra_fields', False):
                    errors.append(f"Unknown field '{field_name}' in section '{section_name}'")
                continue
            
            field_schema = schema[field_name]
            field_type = field_schema.get('type')
            
            # Type validation
            if field_type and not self.validate_type(field_value, field_type):
                errors.append(f"Field '{field_name}' in section '{section_name}' should be of type {field_type.__name__}")
            
            # Range validation for numeric types
            if field_type in [int, float] and (field_schema.get('min') is not None or field_schema.get('max') is not None):
                if not self.validate_range(field_value, field_schema.get('min'), field_schema.get('max')):
                    min_val = field_schema.get('min', 'no minimum')
                    max_val = field_schema.get('max', 'no maximum')
                    errors.append(f"Field '{field_name}' in section '{section_name}' should be between {min_val} and {max_val}")
            
            # String validation
            if field_type == str:
                pattern = field_schema.get('pattern')
                min_length = field_schema.get('min_length')
                max_length = field_schema.get('max_length')
                
                if not self.validate_string(field_value, pattern, min_length, max_length):
                    errors.append(f"Field '{field_name}' in section '{section_name}' has invalid string format")
            
            # Custom validation function
            validator_func = field_schema.get('validator')
            if validator_func and callable(validator_func):
                if not validator_func(field_value):
                    errors.append(f"Field '{field_name}' in section '{section_name}' failed custom validation")
        
        return errors


class CredentialManager:
    """Secure credential management class.
    
    This class provides methods to securely store and retrieve sensitive credentials
    using encryption.
    
    Attributes:
        _key_file: Path to the encryption key file
        _credentials_file: Path to the encrypted credentials file
        _cipher_suite: Fernet cipher suite for encryption/decryption
        _credentials: Dictionary of credentials
    """
    
    def __init__(self, key_file: str, credentials_file: str):
        """Initialize the CredentialManager.
        
        Args:
            key_file: Path to the encryption key file
            credentials_file: Path to the encrypted credentials file
        """
        self._key_file = Path(key_file)
        self._credentials_file = Path(credentials_file)
        self._cipher_suite = None
        self._credentials = {}
        
        # Ensure directories exist
        self._key_file.parent.mkdir(parents=True, exist_ok=True)
        self._credentials_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load credentials if they exist
        if self._credentials_file.exists():
            self.load_credentials()
    
    def _initialize_encryption(self) -> None:
        """Initialize the encryption system.
        
        If the key file doesn't exist, a new key is generated.
        """
        if not self._key_file.exists():
            # Generate a new key
            key = Fernet.generate_key()
            with open(self._key_file, 'wb') as key_file:
                key_file.write(key)
            logger.info(f"Generated new encryption key at {self._key_file}")
        else:
            # Load existing key
            with open(self._key_file, 'rb') as key_file:
                key = key_file.read()
        
        self._cipher_suite = Fernet(key)
    
    def set_credential(self, name: str, value: str) -> None:
        """Set a credential value.
        
        Args:
            name: The credential name
            value: The credential value
        """
        self._credentials[name] = value
        self.save_credentials()
    
    def get_credential(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a credential value.
        
        Args:
            name: The credential name
            default: Default value to return if credential doesn't exist
            
        Returns:
            str: The credential value or default if not found
        """
        return self._credentials.get(name, default)
    
    def delete_credential(self, name: str) -> bool:
        """Delete a credential.
        
        Args:
            name: The credential name
            
        Returns:
            bool: True if the credential was deleted, False if it didn't exist
        """
        if name in self._credentials:
            del self._credentials[name]
            self.save_credentials()
            return True
        return False
    
    def has_credential(self, name: str) -> bool:
        """Check if a credential exists.
        
        Args:
            name: The credential name
            
        Returns:
            bool: True if the credential exists, False otherwise
        """
        return name in self._credentials
    
    def save_credentials(self) -> None:
        """Save credentials to the encrypted credentials file."""
        if not self._cipher_suite:
            raise ValueError("Encryption not initialized")
        
        # Convert credentials to JSON and encrypt
        credentials_json = json.dumps(self._credentials).encode('utf-8')
        encrypted_data = self._cipher_suite.encrypt(credentials_json)
        
        # Save to file
        with open(self._credentials_file, 'wb') as file:
            file.write(encrypted_data)
        
        logger.debug(f"Saved encrypted credentials to {self._credentials_file}")
    
    def load_credentials(self) -> None:
        """Load credentials from the encrypted credentials file."""
        if not self._cipher_suite:
            raise ValueError("Encryption not initialized")
        
        if not self._credentials_file.exists():
            logger.warning(f"Credentials file {self._credentials_file} does not exist")
            return
        
        try:
            # Read and decrypt the file
            with open(self._credentials_file, 'rb') as file:
                encrypted_data = file.read()
            
            decrypted_data = self._cipher_suite.decrypt(encrypted_data)
            self._credentials = json.loads(decrypted_data.decode('utf-8'))
            
            logger.debug(f"Loaded encrypted credentials from {self._credentials_file}")
        except Exception as e:
            logger.error(f"Failed to load credentials: {str(e)}")
            self._credentials = {}
    
    def clear_credentials(self) -> None:
        """Clear all credentials."""
        self._credentials = {}
        if self._credentials_file.exists():
            self._credentials_file.unlink()
            logger.info(f"Cleared credentials file {self._credentials_file}")
    
    def rotate_key(self) -> None:
        """Rotate the encryption key.
        
        This generates a new encryption key and re-encrypts all credentials.
        """
        # Save current credentials
        current_credentials = self._credentials.copy()
        
        # Generate new key
        key = Fernet.generate_key()
        with open(self._key_file, 'wb') as key_file:
            key_file.write(key)
        
        # Update cipher suite
        self._cipher_suite = Fernet(key)
        
        # Re-encrypt credentials
        self._credentials = current_credentials
        self.save_credentials()
        
        logger.info(f"Rotated encryption key at {self._key_file}")


class EnvironmentManager:
    """Environment-specific configuration manager.
    
    This class manages environment-specific configuration overrides.
    
    Attributes:
        _environment: Current environment name
        _env_configs: Dictionary of environment-specific configurations
    """
    
    # Default environments
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    def __init__(self, default_environment: str = DEVELOPMENT):
        """Initialize the EnvironmentManager.
        
        Args:
            default_environment: Default environment name
        """
        self._environment = os.environ.get("FRIDAY_ENV", default_environment)
        self._env_configs = {
            self.DEVELOPMENT: {},
            self.TESTING: {},
            self.STAGING: {},
            self.PRODUCTION: {}
        }
    
    def get_environment(self) -> str:
        """Get the current environment name.
        
        Returns:
            str: Current environment name
        """
        return self._environment
    
    def set_environment(self, environment: str) -> None:
        """Set the current environment.
        
        Args:
            environment: Environment name
        """
        if environment not in self._env_configs:
            self._env_configs[environment] = {}
        
        self._environment = environment
        logger.info(f"Set environment to {environment}")
    
    def register_environment_config(self, environment: str, config: Dict[str, Any]) -> None:
        """Register configuration for a specific environment.
        
        Args:
            environment: Environment name
            config: Environment-specific configuration dictionary
        """
        self._env_configs[environment] = config
    
    def get_environment_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific environment.
        
        Args:
            environment: Environment name, uses current environment if None
            
        Returns:
            Dict[str, Any]: Environment-specific configuration
        """
        env = environment or self._environment
        return self._env_configs.get(env, {})
    
    def is_development(self) -> bool:
        """Check if current environment is development.
        
        Returns:
            bool: True if current environment is development
        """
        return self._environment == self.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if current environment is testing.
        
        Returns:
            bool: True if current environment is testing
        """
        return self._environment == self.TESTING
    
    def is_staging(self) -> bool:
        """Check if current environment is staging.
        
        Returns:
            bool: True if current environment is staging
        """
        return self._environment == self.STAGING
    
    def is_production(self) -> bool:
        """Check if current environment is production.
        
        Returns:
            bool: True if current environment is production
        """
        return self._environment == self.PRODUCTION


class ConfigurationManager:
    """Main configuration manager class.
    
    This class provides a comprehensive configuration management system with
    support for environment-specific overrides, validation, and secure credential
    management.
    
    Attributes:
        _instance: Singleton instance of ConfigurationManager
        _config: Main configuration dictionary
        _validator: ConfigValidator instance
        _credential_manager: CredentialManager instance
        _env_manager: EnvironmentManager instance
        _lock: Thread lock for thread safety
        _change_listeners: Dictionary of configuration change listeners
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config_dir: Optional[str] = None) -> 'ConfigurationManager':
        """Get the singleton instance of ConfigurationManager.
        
        Args:
            config_dir: Optional configuration directory path
            
        Returns:
            ConfigurationManager: Singleton instance
        """
        if cls._instance is None:
            cls._instance = ConfigurationManager(config_dir)
        return cls._instance
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the ConfigurationManager.
        
        Args:
            config_dir: Optional configuration directory path
        """
        if config_dir is None:
            config_dir = os.environ.get("FRIDAY_CONFIG_DIR", "config")
        
        self._config_dir = Path(config_dir)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config = {}
        self._validator = ConfigValidator()
        
        # Set up credential manager
        security_dir = self._config_dir / "security"
        security_dir.mkdir(parents=True, exist_ok=True)
        self._credential_manager = CredentialManager(
            str(security_dir / "master.key"),
            str(security_dir / "credentials.enc")
        )
        
        # Set up environment manager
        self._env_manager = EnvironmentManager()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Change listeners
        self._change_listeners = {}
        
        # Load default configuration
        self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration from unified_config.py."""
        try:
            # Import the unified_config module
            import sys
            sys.path.append(str(Path.cwd()))
            from unified_config import (
                ZERODHA_CONFIG, MCP_CONFIG, KNOWLEDGE_CONFIG, TRADING_CONFIG,
                MODEL_CONFIG, DATA_CONFIG, LOGGING_CONFIG, BACKTESTING_CONFIG,
                NOTIFICATION_CONFIG, SECURITY_CONFIG, DATABASE_CONFIG, CACHE_CONFIG,
                VALIDATION_CONFIG, API_CONFIG, SYSTEM_CONFIG, FEATURES_CONFIG
            )
            
            # Create the configuration dictionary
            self._config = {
                "zerodha": ZERODHA_CONFIG,
                "mcp": MCP_CONFIG,
                "knowledge": KNOWLEDGE_CONFIG,
                "trading": TRADING_CONFIG,
                "model": MODEL_CONFIG,
                "data": DATA_CONFIG,
                "logging": LOGGING_CONFIG,
                "backtesting": BACKTESTING_CONFIG,
                "notification": NOTIFICATION_CONFIG,
                "security": SECURITY_CONFIG,
                "database": DATABASE_CONFIG,
                "cache": CACHE_CONFIG,
                "validation": VALIDATION_CONFIG,
                "api": API_CONFIG,
                "system": SYSTEM_CONFIG,
                "features": FEATURES_CONFIG
            }
            
            logger.info("Loaded default configuration from unified_config.py")
        except ImportError as e:
            logger.warning(f"Failed to import unified_config.py: {str(e)}")
            logger.warning("Using empty default configuration")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation path.
        
        Args:
            path: Configuration path in dot notation (e.g., 'database.host')
            default: Default value to return if path doesn't exist
            
        Returns:
            Any: Configuration value or default if not found
        """
        with self._lock:
            # Split the path into parts
            parts = path.split('.')
            
            # Start with the main config
            current = self._config
            
            # Check environment overrides first
            env_config = self._env_manager.get_environment_config()
            env_current = env_config
            env_found = True
            
            # Traverse the environment config path
            for part in parts:
                if env_found and isinstance(env_current, dict) and part in env_current:
                    env_current = env_current[part]
                else:
                    env_found = False
            
            # If found in environment config, return it
            if env_found:
                return env_current
            
            # Otherwise traverse the main config path
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            
            return current
    
    def set(self, path: str, value: Any, environment: Optional[str] = None) -> None:
        """Set a configuration value using dot notation path.
        
        Args:
            path: Configuration path in dot notation (e.g., 'database.host')
            value: Value to set
            environment: Optional environment name to set for specific environment
        """
        with self._lock:
            # Split the path into parts
            parts = path.split('.')
            
            # If setting for a specific environment
            if environment is not None:
                env_config = self._env_manager.get_environment_config(environment)
                current = env_config
                
                # Create nested dictionaries as needed
                for i, part in enumerate(parts[:-1]):
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = value
                
                # Update the environment config
                self._env_manager.register_environment_config(environment, env_config)
                
                logger.debug(f"Set configuration {path}={value} for environment {environment}")
            else:
                # Set in main config
                current = self._config
                
                # Create nested dictionaries as needed
                for i, part in enumerate(parts[:-1]):
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                old_value = current.get(parts[-1])
                current[parts[-1]] = value
                
                logger.debug(f"Set configuration {path}={value}")
                
                # Notify change listeners
                self._notify_change_listeners(path, old_value, value)
    
    def delete(self, path: str, environment: Optional[str] = None) -> bool:
        """Delete a configuration value using dot notation path.
        
        Args:
            path: Configuration path in dot notation (e.g., 'database.host')
            environment: Optional environment name to delete from specific environment
            
        Returns:
            bool: True if the value was deleted, False if it didn't exist
        """
        with self._lock:
            # Split the path into parts
            parts = path.split('.')
            
            # If deleting from a specific environment
            if environment is not None:
                env_config = self._env_manager.get_environment_config(environment)
                current = env_config
                
                # Traverse the path
                for i, part in enumerate(parts[:-1]):
                    if part not in current or not isinstance(current[part], dict):
                        return False
                    current = current[part]
                
                # Delete the value
                if parts[-1] in current:
                    del current[parts[-1]]
                    
                    # Update the environment config
                    self._env_manager.register_environment_config(environment, env_config)
                    
                    logger.debug(f"Deleted configuration {path} for environment {environment}")
                    return True
                
                return False
            else:
                # Delete from main config
                current = self._config
                
                # Traverse the path
                for i, part in enumerate(parts[:-1]):
                    if part not in current or not isinstance(current[part], dict):
                        return False
                    current = current[part]
                
                # Delete the value
                if parts[-1] in current:
                    old_value = current[parts[-1]]
                    del current[parts[-1]]
                    
                    logger.debug(f"Deleted configuration {path}")
                    
                    # Notify change listeners
                    self._notify_change_listeners(path, old_value, None)
                    
                    return True
                
                return False
    
    def has(self, path: str, environment: Optional[str] = None) -> bool:
        """Check if a configuration path exists.
        
        Args:
            path: Configuration path in dot notation (e.g., 'database.host')
            environment: Optional environment name to check in specific environment
            
        Returns:
            bool: True if the path exists, False otherwise
        """
        with self._lock:
            # Split the path into parts
            parts = path.split('.')
            
            # If checking in a specific environment
            if environment is not None:
                env_config = self._env_manager.get_environment_config(environment)
                current = env_config
            else:
                # Check environment overrides first
                env_config = self._env_manager.get_environment_config()
                env_current = env_config
                env_found = True
                
                # Traverse the environment config path
                for part in parts:
                    if env_found and isinstance(env_current, dict) and part in env_current:
                        env_current = env_current[part]
                    else:
                        env_found = False
                
                # If found in environment config, return True
                if env_found:
                    return True
                
                # Otherwise check main config
                current = self._config
            
            # Traverse the path
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            
            return True
    
    def get_all(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get the entire configuration.
        
        Args:
            environment: Optional environment name to get specific environment config
            
        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        with self._lock:
            if environment is not None:
                return deepcopy(self._env_manager.get_environment_config(environment))
            
            # Merge main config with environment overrides
            result = deepcopy(self._config)
            env_config = self._env_manager.get_environment_config()
            
            # Deep merge the environment config into the result
            self._deep_merge(result, env_config)
            
            return result
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge two dictionaries.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    
    def load_from_file(self, file_path: str, format: str = 'auto') -> None:
        """Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            format: File format ('json', 'yaml', or 'auto' to detect from extension)
        """
        with self._lock:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file {file_path} not found")
            
            # Detect format from extension if auto
            if format == 'auto':
                if path.suffix.lower() in ['.json', '.jsn']:
                    format = 'json'
                elif path.suffix.lower() in ['.yaml', '.yml']:
                    format = 'yaml'
                else:
                    raise ValueError(f"Cannot detect format from file extension: {path.suffix}")
            
            # Load the file
            with open(path, 'r') as file:
                if format == 'json':
                    config = json.load(file)
                elif format == 'yaml':
                    config = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            # Update the configuration
            self._config.update(config)
            
            logger.info(f"Loaded configuration from {file_path}")
    
    def save_to_file(self, file_path: str, format: str = 'auto', environment: Optional[str] = None) -> None:
        """Save configuration to a file.
        
        Args:
            file_path: Path to the configuration file
            format: File format ('json', 'yaml', or 'auto' to detect from extension)
            environment: Optional environment name to save specific environment config
        """
        with self._lock:
            path = Path(file_path)
            
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Detect format from extension if auto
            if format == 'auto':
                if path.suffix.lower() in ['.json', '.jsn']:
                    format = 'json'
                elif path.suffix.lower() in ['.yaml', '.yml']:
                    format = 'yaml'
                else:
                    raise ValueError(f"Cannot detect format from file extension: {path.suffix}")
            
            # Get the configuration to save
            if environment is not None:
                config = self._env_manager.get_environment_config(environment)
            else:
                config = self._config
            
            # Save the file
            with open(path, 'w') as file:
                if format == 'json':
                    json.dump(config, file, indent=2)
                elif format == 'yaml':
                    yaml.dump(config, file, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved configuration to {file_path}")
    
    def register_validator(self, section: str, schema: Dict[str, Any]) -> None:
        """Register a validation schema for a configuration section.
        
        Args:
            section: The configuration section name
            schema: The validation schema as a dictionary
        """
        self._validator.register_schema(section, schema)
    
    def validate(self, section: Optional[str] = None) -> List[str]:
        """Validate the configuration against registered schemas.
        
        Args:
            section: Optional section name to validate only that section
            
        Returns:
            List[str]: List of validation error messages, empty if valid
        """
        with self._lock:
            return self._validator.validate_config(self.get_all(), section)
    
    def set_credential(self, name: str, value: str) -> None:
        """Set a credential value.
        
        Args:
            name: The credential name
            value: The credential value
        """
        self._credential_manager.set_credential(name, value)
    
    def get_credential(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a credential value.
        
        Args:
            name: The credential name
            default: Default value to return if credential doesn't exist
            
        Returns:
            str: The credential value or default if not found
        """
        return self._credential_manager.get_credential(name, default)
    
    def delete_credential(self, name: str) -> bool:
        """Delete a credential.
        
        Args:
            name: The credential name
            
        Returns:
            bool: True if the credential was deleted, False if it didn't exist
        """
        return self._credential_manager.delete_credential(name)
    
    def has_credential(self, name: str) -> bool:
        """Check if a credential exists.
        
        Args:
            name: The credential name
            
        Returns:
            bool: True if the credential exists, False otherwise
        """
        return self._credential_manager.has_credential(name)
    
    def get_environment(self) -> str:
        """Get the current environment name.
        
        Returns:
            str: Current environment name
        """
        return self._env_manager.get_environment()
    
    def set_environment(self, environment: str) -> None:
        """Set the current environment.
        
        Args:
            environment: Environment name
        """
        with self._lock:
            self._env_manager.set_environment(environment)
    
    def is_development(self) -> bool:
        """Check if current environment is development.
        
        Returns:
            bool: True if current environment is development
        """
        return self._env_manager.is_development()
    
    def is_testing(self) -> bool:
        """Check if current environment is testing.
        
        Returns:
            bool: True if current environment is testing
        """
        return self._env_manager.is_testing()
    
    def is_staging(self) -> bool:
        """Check if current environment is staging.
        
        Returns:
            bool: True if current environment is staging
        """
        return self._env_manager.is_staging()
    
    def is_production(self) -> bool:
        """Check if current environment is production.
        
        Returns:
            bool: True if current environment is production
        """
        return self._env_manager.is_production()
    
    def register_change_listener(self, path: str, listener: Callable[[str, Any, Any], None]) -> str:
        """Register a listener for configuration changes.
        
        Args:
            path: Configuration path in dot notation (e.g., 'database.host')
            listener: Callback function that takes (path, old_value, new_value)
            
        Returns:
            str: Listener ID for unregistering
        """
        with self._lock:
            listener_id = str(hash(listener))
            
            if path not in self._change_listeners:
                self._change_listeners[path] = {}
            
            self._change_listeners[path][listener_id] = listener
            
            return listener_id
    
    def unregister_change_listener(self, path: str, listener_id: str) -> bool:
        """Unregister a configuration change listener.
        
        Args:
            path: Configuration path in dot notation
            listener_id: Listener ID returned from register_change_listener
            
        Returns:
            bool: True if the listener was unregistered, False if it didn't exist
        """
        with self._lock:
            if path in self._change_listeners and listener_id in self._change_listeners[path]:
                del self._change_listeners[path][listener_id]
                
                # Clean up empty dictionaries
                if not self._change_listeners[path]:
                    del self._change_listeners[path]
                
                return True
            
            return False
    
    def _notify_change_listeners(self, path: str, old_value: Any, new_value: Any) -> None:
        """Notify change listeners of a configuration change.
        
        Args:
            path: Configuration path that changed
            old_value: Previous value
            new_value: New value
        """
        # Notify listeners for the exact path
        if path in self._change_listeners:
            for listener in self._change_listeners[path].values():
                try:
                    listener(path, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in configuration change listener: {str(e)}")
        
        # Notify listeners for parent paths
        parts = path.split('.')
        for i in range(len(parts) - 1):
            parent_path = '.'.join(parts[:i+1])
            if parent_path in self._change_listeners:
                for listener in self._change_listeners[parent_path].values():
                    try:
                        listener(path, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Error in configuration change listener: {str(e)}")
    
    def load_environment_config(self, environment: str, file_path: str, format: str = 'auto') -> None:
        """Load configuration for a specific environment from a file.
        
        Args:
            environment: Environment name
            file_path: Path to the configuration file
            format: File format ('json', 'yaml', or 'auto' to detect from extension)
        """
        with self._lock:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Environment configuration file {file_path} not found")
            
            # Detect format from extension if auto
            if format == 'auto':
                if path.suffix.lower() in ['.json', '.jsn']:
                    format = 'json'
                elif path.suffix.lower() in ['.yaml', '.yml']:
                    format = 'yaml'
                else:
                    raise ValueError(f"Cannot detect format from file extension: {path.suffix}")
            
            # Load the file
            with open(path, 'r') as file:
                if format == 'json':
                    config = json.load(file)
                elif format == 'yaml':
                    config = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            # Register the environment configuration
            self._env_manager.register_environment_config(environment, config)
            
            logger.info(f"Loaded configuration for environment {environment} from {file_path}")
    
    def get_development_config(self) -> Dict[str, Any]:
        """Get the development environment configuration.
        
        Returns:
            Dict[str, Any]: Development environment configuration
        """
        return self._env_manager.get_environment_config(EnvironmentManager.DEVELOPMENT)
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get the testing environment configuration.
        
        Returns:
            Dict[str, Any]: Testing environment configuration
        """
        return self._env_manager.get_environment_config(EnvironmentManager.TESTING)
    
    def get_staging_config(self) -> Dict[str, Any]:
        """Get the staging environment configuration.
        
        Returns:
            Dict[str, Any]: Staging environment configuration
        """
        return self._env_manager.get_environment_config(EnvironmentManager.STAGING)
    
    def get_production_config(self) -> Dict[str, Any]:
        """Get the production environment configuration.
        
        Returns:
            Dict[str, Any]: Production environment configuration
        """
        return self._env_manager.get_environment_config(EnvironmentManager.PRODUCTION)
    
    def load_from_env_vars(self, prefix: str = "FRIDAY_") -> None:
        """Load configuration from environment variables.
        
        Environment variables should be in the format PREFIX_SECTION_KEY.
        For example, FRIDAY_DATABASE_HOST would set config['database']['host'].
        
        Args:
            prefix: Prefix for environment variables
        """
        with self._lock:
            # Get all environment variables with the prefix
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Remove prefix and split into parts
                    parts = key[len(prefix):].lower().split('_')
                    
                    if len(parts) < 2:
                        continue
                    
                    # Convert value to appropriate type
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                        value = float(value)
                    
                    # Build the path
                    path = '.'.join(parts)
                    
                    # Set the value
                    self.set(path, value)
            
            logger.info(f"Loaded configuration from environment variables with prefix {prefix}")
    
    def clear(self, environment: Optional[str] = None) -> None:
        """Clear the configuration.
        
        Args:
            environment: Optional environment name to clear specific environment
        """
        with self._lock:
            if environment is not None:
                self._env_manager.register_environment_config(environment, {})
                logger.info(f"Cleared configuration for environment {environment}")
            else:
                self._config = {}
                logger.info("Cleared main configuration")
    
    def reload(self) -> None:
        """Reload the default configuration."""
        with self._lock:
            self._load_default_config()
            logger.info("Reloaded default configuration")


# Convenience functions

def get_config_instance() -> ConfigurationManager:
    """Get the singleton instance of ConfigurationManager.
    
    Returns:
        ConfigurationManager: Singleton instance
    """
    return ConfigurationManager.get_instance()


def get_config(path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation path.
    
    Args:
        path: Configuration path in dot notation (e.g., 'database.host')
        default: Default value to return if path doesn't exist
        
    Returns:
        Any: Configuration value or default if not found
    """
    return get_config_instance().get(path, default)


def set_config(path: str, value: Any, environment: Optional[str] = None) -> None:
    """Set a configuration value using dot notation path.
    
    Args:
        path: Configuration path in dot notation (e.g., 'database.host')
        value: Value to set
        environment: Optional environment name to set for specific environment
    """
    get_config_instance().set(path, value, environment)


def get_credential(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get a credential value.
    
    Args:
        name: The credential name
        default: Default value to return if credential doesn't exist
        
    Returns:
        str: The credential value or default if not found
    """
    return get_config_instance().get_credential(name, default)


def set_credential(name: str, value: str) -> None:
    """Set a credential value.
    
    Args:
        name: The credential name
        value: The credential value
    """
    get_config_instance().set_credential(name, value)


def get_environment() -> str:
    """Get the current environment name.
    
    Returns:
        str: Current environment name
    """
    return get_config_instance().get_environment()


def set_environment(environment: str) -> None:
    """Set the current environment.
    
    Args:
        environment: Environment name
    """
    get_config_instance().set_environment(environment)


def is_production() -> bool:
    """Check if current environment is production.
    
    Returns:
        bool: True if current environment is production
    """
    return get_config_instance().is_production()


def is_development() -> bool:
    """Check if current environment is development.
    
    Returns:
        bool: True if current environment is development
    """
    return get_config_instance().is_development()


def validate_config(section: Optional[str] = None) -> List[str]:
    """Validate the configuration against registered schemas.
    
    Args:
        section: Optional section name to validate only that section
        
    Returns:
        List[str]: List of validation error messages, empty if valid
    """
    return get_config_instance().validate(section)


# Create alias for backward compatibility
ConfigManager = ConfigurationManager
