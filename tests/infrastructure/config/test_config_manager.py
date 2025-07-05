"""Unit tests for the configuration management system.

This module contains tests for the ConfigurationManager, ConfigValidator,
CredentialManager, and EnvironmentManager classes.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the configuration management system
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.infrastructure.config.config_manager import (
    ConfigurationManager, ConfigValidator, CredentialManager, EnvironmentManager,
    get_config, set_config, get_credential, set_credential, get_environment, set_environment,
    is_production, is_development, validate_config
)


class TestConfigValidator(unittest.TestCase):
    """Tests for the ConfigValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    def test_validate_type(self):
        """Test type validation."""
        # Test single type
        self.assertTrue(self.validator.validate_type(42, int))
        self.assertFalse(self.validator.validate_type("42", int))
        
        # Test multiple types
        self.assertTrue(self.validator.validate_type(42, [int, str]))
        self.assertTrue(self.validator.validate_type("42", [int, str]))
        self.assertFalse(self.validator.validate_type(42.0, [int, str]))
    
    def test_validate_range(self):
        """Test range validation."""
        # Test min only
        self.assertTrue(self.validator.validate_range(42, min_value=0))
        self.assertFalse(self.validator.validate_range(42, min_value=100))
        
        # Test max only
        self.assertTrue(self.validator.validate_range(42, max_value=100))
        self.assertFalse(self.validator.validate_range(42, max_value=10))
        
        # Test min and max
        self.assertTrue(self.validator.validate_range(42, min_value=0, max_value=100))
        self.assertFalse(self.validator.validate_range(42, min_value=50, max_value=100))
        self.assertFalse(self.validator.validate_range(42, min_value=0, max_value=40))
    
    def test_validate_string(self):
        """Test string validation."""
        # Test type check
        self.assertTrue(self.validator.validate_string("test"))
        self.assertFalse(self.validator.validate_string(42))
        
        # Test min length
        self.assertTrue(self.validator.validate_string("test", min_length=2))
        self.assertFalse(self.validator.validate_string("a", min_length=2))
        
        # Test max length
        self.assertTrue(self.validator.validate_string("test", max_length=10))
        self.assertFalse(self.validator.validate_string("test", max_length=3))
        
        # Test pattern
        self.assertTrue(self.validator.validate_string("test123", pattern=r"^\w+$"))
        self.assertFalse(self.validator.validate_string("test 123", pattern=r"^\w+$"))
        
        # Test all together
        self.assertTrue(self.validator.validate_string("test123", pattern=r"^\w+$", min_length=5, max_length=10))
        self.assertFalse(self.validator.validate_string("test", pattern=r"^\w+$", min_length=5, max_length=10))
        self.assertFalse(self.validator.validate_string("test1234567890", pattern=r"^\w+$", min_length=5, max_length=10))
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Register a schema
        schema = {
            "host": {
                "type": str,
                "required": True,
                "pattern": r"^[\w.-]+$"
            },
            "port": {
                "type": int,
                "required": True,
                "min": 1,
                "max": 65535
            },
            "debug": {
                "type": bool,
                "required": False
            }
        }
        self.validator.register_schema("database", schema)
        
        # Test valid config
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "debug": True
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 0)
        
        # Test missing required field
        config = {
            "database": {
                "host": "localhost"
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test invalid type
        config = {
            "database": {
                "host": "localhost",
                "port": "5432",
                "debug": True
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test invalid range
        config = {
            "database": {
                "host": "localhost",
                "port": 0,
                "debug": True
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test invalid pattern
        config = {
            "database": {
                "host": "localhost:5432",
                "port": 5432,
                "debug": True
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test unknown field
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "debug": True,
                "unknown": "value"
            }
        }
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test missing section
        config = {}
        errors = self.validator.validate_config(config)
        self.assertEqual(len(errors), 1)
        
        # Test validate specific section
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "debug": True
            }
        }
        errors = self.validator.validate_config(config, "database")
        self.assertEqual(len(errors), 0)
        
        # Test validate non-existent section
        errors = self.validator.validate_config(config, "non_existent")
        self.assertEqual(len(errors), 1)


class TestCredentialManager(unittest.TestCase):
    """Tests for the CredentialManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.key_file = os.path.join(self.temp_dir.name, "master.key")
        self.credentials_file = os.path.join(self.temp_dir.name, "credentials.enc")
        
        # Create credential manager
        self.credential_manager = CredentialManager(self.key_file, self.credentials_file)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_set_get_credential(self):
        """Test setting and getting credentials."""
        # Set a credential
        self.credential_manager.set_credential("api_key", "secret_value")
        
        # Get the credential
        value = self.credential_manager.get_credential("api_key")
        self.assertEqual(value, "secret_value")
        
        # Get a non-existent credential
        value = self.credential_manager.get_credential("non_existent")
        self.assertIsNone(value)
        
        # Get a non-existent credential with default
        value = self.credential_manager.get_credential("non_existent", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_delete_credential(self):
        """Test deleting credentials."""
        # Set a credential
        self.credential_manager.set_credential("api_key", "secret_value")
        
        # Delete the credential
        result = self.credential_manager.delete_credential("api_key")
        self.assertTrue(result)
        
        # Verify it's gone
        value = self.credential_manager.get_credential("api_key")
        self.assertIsNone(value)
        
        # Delete a non-existent credential
        result = self.credential_manager.delete_credential("non_existent")
        self.assertFalse(result)
    
    def test_has_credential(self):
        """Test checking if a credential exists."""
        # Set a credential
        self.credential_manager.set_credential("api_key", "secret_value")
        
        # Check if it exists
        self.assertTrue(self.credential_manager.has_credential("api_key"))
        
        # Check if a non-existent credential exists
        self.assertFalse(self.credential_manager.has_credential("non_existent"))
    
    def test_clear_credentials(self):
        """Test clearing all credentials."""
        # Set some credentials
        self.credential_manager.set_credential("api_key", "secret_value")
        self.credential_manager.set_credential("password", "another_secret")
        
        # Clear all credentials
        self.credential_manager.clear_credentials()
        
        # Verify they're gone
        self.assertFalse(self.credential_manager.has_credential("api_key"))
        self.assertFalse(self.credential_manager.has_credential("password"))
    
    def test_save_load_credentials(self):
        """Test saving and loading credentials."""
        # Set some credentials
        self.credential_manager.set_credential("api_key", "secret_value")
        self.credential_manager.set_credential("password", "another_secret")
        
        # Save credentials
        self.credential_manager.save_credentials()
        
        # Create a new credential manager that will load from the same files
        new_manager = CredentialManager(self.key_file, self.credentials_file)
        
        # Verify credentials were loaded
        self.assertEqual(new_manager.get_credential("api_key"), "secret_value")
        self.assertEqual(new_manager.get_credential("password"), "another_secret")
    
    def test_rotate_key(self):
        """Test rotating the encryption key."""
        # Set a credential
        self.credential_manager.set_credential("api_key", "secret_value")
        
        # Save the current key file content
        with open(self.key_file, "rb") as f:
            old_key = f.read()
        
        # Rotate the key
        self.credential_manager.rotate_key()
        
        # Verify the key changed
        with open(self.key_file, "rb") as f:
            new_key = f.read()
        
        self.assertNotEqual(old_key, new_key)
        
        # Verify credentials are still accessible
        self.assertEqual(self.credential_manager.get_credential("api_key"), "secret_value")
        
        # Create a new credential manager that will load from the same files
        new_manager = CredentialManager(self.key_file, self.credentials_file)
        
        # Verify credentials were loaded with the new key
        self.assertEqual(new_manager.get_credential("api_key"), "secret_value")


class TestEnvironmentManager(unittest.TestCase):
    """Tests for the EnvironmentManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Save original environment variable
        self.original_env = os.environ.get("FRIDAY_ENV")
        
        # Create environment manager
        self.env_manager = EnvironmentManager()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original environment variable
        if self.original_env is not None:
            os.environ["FRIDAY_ENV"] = self.original_env
        elif "FRIDAY_ENV" in os.environ:
            del os.environ["FRIDAY_ENV"]
    
    def test_default_environment(self):
        """Test default environment."""
        # Default should be development
        self.assertEqual(self.env_manager.get_environment(), EnvironmentManager.DEVELOPMENT)
        self.assertTrue(self.env_manager.is_development())
        self.assertFalse(self.env_manager.is_production())
    
    def test_set_environment(self):
        """Test setting the environment."""
        # Set to production
        self.env_manager.set_environment(EnvironmentManager.PRODUCTION)
        
        # Verify it changed
        self.assertEqual(self.env_manager.get_environment(), EnvironmentManager.PRODUCTION)
        self.assertTrue(self.env_manager.is_production())
        self.assertFalse(self.env_manager.is_development())
        
        # Set to testing
        self.env_manager.set_environment(EnvironmentManager.TESTING)
        
        # Verify it changed
        self.assertEqual(self.env_manager.get_environment(), EnvironmentManager.TESTING)
        self.assertTrue(self.env_manager.is_testing())
        self.assertFalse(self.env_manager.is_production())
        
        # Set to staging
        self.env_manager.set_environment(EnvironmentManager.STAGING)
        
        # Verify it changed
        self.assertEqual(self.env_manager.get_environment(), EnvironmentManager.STAGING)
        self.assertTrue(self.env_manager.is_staging())
        self.assertFalse(self.env_manager.is_testing())
    
    def test_environment_from_env_var(self):
        """Test setting environment from environment variable."""
        # Set environment variable
        os.environ["FRIDAY_ENV"] = EnvironmentManager.PRODUCTION
        
        # Create new environment manager
        env_manager = EnvironmentManager()
        
        # Verify it used the environment variable
        self.assertEqual(env_manager.get_environment(), EnvironmentManager.PRODUCTION)
        self.assertTrue(env_manager.is_production())
    
    def test_register_environment_config(self):
        """Test registering environment-specific configuration."""
        # Register configuration for production
        config = {"database": {"host": "prod-db"}}
        self.env_manager.register_environment_config(EnvironmentManager.PRODUCTION, config)
        
        # Verify it was registered
        env_config = self.env_manager.get_environment_config(EnvironmentManager.PRODUCTION)
        self.assertEqual(env_config, config)
        
        # Register configuration for development
        config = {"database": {"host": "dev-db"}}
        self.env_manager.register_environment_config(EnvironmentManager.DEVELOPMENT, config)
        
        # Verify it was registered
        env_config = self.env_manager.get_environment_config(EnvironmentManager.DEVELOPMENT)
        self.assertEqual(env_config, config)
    
    def test_get_environment_config(self):
        """Test getting environment-specific configuration."""
        # Register configurations
        dev_config = {"database": {"host": "dev-db"}}
        prod_config = {"database": {"host": "prod-db"}}
        
        self.env_manager.register_environment_config(EnvironmentManager.DEVELOPMENT, dev_config)
        self.env_manager.register_environment_config(EnvironmentManager.PRODUCTION, prod_config)
        
        # Set environment to development
        self.env_manager.set_environment(EnvironmentManager.DEVELOPMENT)
        
        # Get current environment config
        env_config = self.env_manager.get_environment_config()
        self.assertEqual(env_config, dev_config)
        
        # Get specific environment config
        env_config = self.env_manager.get_environment_config(EnvironmentManager.PRODUCTION)
        self.assertEqual(env_config, prod_config)


class TestConfigurationManager(unittest.TestCase):
    """Tests for the ConfigurationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Reset the singleton instance
        ConfigurationManager._instance = None
        
        # Create configuration manager
        self.config_manager = ConfigurationManager.get_instance(self.temp_dir.name)
        
        # Set up test configuration
        self.config_manager._config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "user",
                "password": "password"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
        # Reset the singleton instance
        ConfigurationManager._instance = None
    
    def test_singleton(self):
        """Test singleton pattern."""
        # Get another instance
        another_instance = ConfigurationManager.get_instance()
        
        # Verify it's the same instance
        self.assertIs(self.config_manager, another_instance)
    
    def test_get_config(self):
        """Test getting configuration values."""
        # Get existing values
        self.assertEqual(self.config_manager.get("database.host"), "localhost")
        self.assertEqual(self.config_manager.get("database.port"), 5432)
        self.assertEqual(self.config_manager.get("api.debug"), True)
        
        # Get non-existent value
        self.assertIsNone(self.config_manager.get("non.existent"))
        
        # Get non-existent value with default
        self.assertEqual(self.config_manager.get("non.existent", "default"), "default")
        
        # Get entire section
        database = self.config_manager.get("database")
        self.assertEqual(database, {
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "password": "password"
        })
    
    def test_set_config(self):
        """Test setting configuration values."""
        # Set existing value
        self.config_manager.set("database.host", "new-host")
        self.assertEqual(self.config_manager.get("database.host"), "new-host")
        
        # Set new value
        self.config_manager.set("database.ssl", True)
        self.assertEqual(self.config_manager.get("database.ssl"), True)
        
        # Set new nested value
        self.config_manager.set("database.pool.size", 10)
        self.assertEqual(self.config_manager.get("database.pool.size"), 10)
        
        # Set new section
        self.config_manager.set("logging.level", "INFO")
        self.assertEqual(self.config_manager.get("logging.level"), "INFO")
    
    def test_delete_config(self):
        """Test deleting configuration values."""
        # Delete existing value
        self.assertTrue(self.config_manager.delete("database.host"))
        self.assertIsNone(self.config_manager.get("database.host"))
        
        # Delete non-existent value
        self.assertFalse(self.config_manager.delete("non.existent"))
    
    def test_has_config(self):
        """Test checking if a configuration path exists."""
        # Check existing path
        self.assertTrue(self.config_manager.has("database.host"))
        
        # Check non-existent path
        self.assertFalse(self.config_manager.has("non.existent"))
    
    def test_get_all_config(self):
        """Test getting the entire configuration."""
        # Get all configuration
        config = self.config_manager.get_all()
        
        # Verify it matches the test configuration
        self.assertEqual(config, self.config_manager._config)
        
        # Verify it's a deep copy
        config["database"]["host"] = "modified"
        self.assertEqual(self.config_manager.get("database.host"), "localhost")
    
    def test_load_save_config(self):
        """Test loading and saving configuration to/from files."""
        # Save configuration to JSON file
        json_file = os.path.join(self.temp_dir.name, "config.json")
        self.config_manager.save_to_file(json_file)
        
        # Modify configuration
        self.config_manager.set("database.host", "modified")
        
        # Load configuration from JSON file
        self.config_manager.load_from_file(json_file)
        
        # Verify it was loaded correctly
        self.assertEqual(self.config_manager.get("database.host"), "localhost")
        
        # Save configuration to YAML file
        yaml_file = os.path.join(self.temp_dir.name, "config.yaml")
        self.config_manager.save_to_file(yaml_file)
        
        # Modify configuration
        self.config_manager.set("database.host", "modified")
        
        # Load configuration from YAML file
        self.config_manager.load_from_file(yaml_file)
        
        # Verify it was loaded correctly
        self.assertEqual(self.config_manager.get("database.host"), "localhost")
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        # Set up environment-specific configuration
        dev_config = {"database": {"host": "dev-db"}}
        prod_config = {"database": {"host": "prod-db"}}
        
        # Register environment configurations
        self.config_manager._env_manager.register_environment_config(EnvironmentManager.DEVELOPMENT, dev_config)
        self.config_manager._env_manager.register_environment_config(EnvironmentManager.PRODUCTION, prod_config)
        
        # Set environment to development
        self.config_manager.set_environment(EnvironmentManager.DEVELOPMENT)
        
        # Verify environment-specific value overrides main config
        self.assertEqual(self.config_manager.get("database.host"), "dev-db")
        
        # Verify other values still come from main config
        self.assertEqual(self.config_manager.get("database.port"), 5432)
        
        # Set environment to production
        self.config_manager.set_environment(EnvironmentManager.PRODUCTION)
        
        # Verify environment-specific value changed
        self.assertEqual(self.config_manager.get("database.host"), "prod-db")
        
        # Set environment-specific value
        self.config_manager.set("database.port", 5433, EnvironmentManager.PRODUCTION)
        
        # Verify it was set
        self.assertEqual(self.config_manager.get("database.port"), 5433)
        
        # Set environment back to development
        self.config_manager.set_environment(EnvironmentManager.DEVELOPMENT)
        
        # Verify environment-specific value is not applied
        self.assertEqual(self.config_manager.get("database.port"), 5432)
    
    def test_load_environment_config(self):
        """Test loading environment-specific configuration from files."""
        # Create environment-specific configuration
        prod_config = {
            "database": {
                "host": "prod-db",
                "port": 5433
            }
        }
        
        # Save to file
        prod_file = os.path.join(self.temp_dir.name, "production.json")
        with open(prod_file, "w") as f:
            json.dump(prod_config, f)
        
        # Load environment configuration
        self.config_manager.load_environment_config(EnvironmentManager.PRODUCTION, prod_file)
        
        # Set environment to production
        self.config_manager.set_environment(EnvironmentManager.PRODUCTION)
        
        # Verify environment-specific values are applied
        self.assertEqual(self.config_manager.get("database.host"), "prod-db")
        self.assertEqual(self.config_manager.get("database.port"), 5433)
    
    def test_validation(self):
        """Test configuration validation."""
        # Register validation schema
        schema = {
            "host": {
                "type": str,
                "required": True,
                "pattern": r"^[\w.-]+$"
            },
            "port": {
                "type": int,
                "required": True,
                "min": 1,
                "max": 65535
            },
            "debug": {
                "type": bool,
                "required": False
            }
        }
        self.config_manager.register_validator("api", schema)
        
        # Validate valid configuration
        errors = self.config_manager.validate("api")
        self.assertEqual(len(errors), 0)
        
        # Modify configuration to be invalid
        self.config_manager.set("api.port", 0)
        
        # Validate invalid configuration
        errors = self.config_manager.validate("api")
        self.assertEqual(len(errors), 1)
    
    def test_credential_management(self):
        """Test credential management."""
        # Set a credential
        self.config_manager.set_credential("api_key", "secret_value")
        
        # Get the credential
        value = self.config_manager.get_credential("api_key")
        self.assertEqual(value, "secret_value")
        
        # Delete the credential
        result = self.config_manager.delete_credential("api_key")
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertFalse(self.config_manager.has_credential("api_key"))
    
    def test_change_listeners(self):
        """Test configuration change listeners."""
        # Create a mock listener
        listener = MagicMock()
        
        # Register the listener
        listener_id = self.config_manager.register_change_listener("database.host", listener)
        
        # Change the value
        self.config_manager.set("database.host", "new-host")
        
        # Verify the listener was called
        listener.assert_called_once_with("database.host", "localhost", "new-host")
        
        # Reset the mock
        listener.reset_mock()
        
        # Unregister the listener
        result = self.config_manager.unregister_change_listener("database.host", listener_id)
        self.assertTrue(result)
        
        # Change the value again
        self.config_manager.set("database.host", "another-host")
        
        # Verify the listener was not called
        listener.assert_not_called()
    
    def test_load_from_env_vars(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["FRIDAY_DATABASE_HOST"] = "env-db"
        os.environ["FRIDAY_DATABASE_PORT"] = "5433"
        os.environ["FRIDAY_API_DEBUG"] = "false"
        
        # Load from environment variables
        self.config_manager.load_from_env_vars()
        
        # Verify values were loaded
        self.assertEqual(self.config_manager.get("database.host"), "env-db")
        self.assertEqual(self.config_manager.get("database.port"), 5433)
        self.assertEqual(self.config_manager.get("api.debug"), False)
        
        # Clean up
        del os.environ["FRIDAY_DATABASE_HOST"]
        del os.environ["FRIDAY_DATABASE_PORT"]
        del os.environ["FRIDAY_API_DEBUG"]
    
    def test_clear_config(self):
        """Test clearing configuration."""
        # Clear main configuration
        self.config_manager.clear()
        
        # Verify it's empty
        self.assertEqual(self.config_manager.get_all(), {})
        
        # Set up test configuration again
        self.config_manager._config = {
            "database": {"host": "localhost"}
        }
        
        # Set up environment-specific configuration
        self.config_manager._env_manager.register_environment_config(
            EnvironmentManager.PRODUCTION, {"database": {"host": "prod-db"}}
        )
        
        # Clear environment-specific configuration
        self.config_manager.clear(EnvironmentManager.PRODUCTION)
        
        # Set environment to production
        self.config_manager.set_environment(EnvironmentManager.PRODUCTION)
        
        # Verify environment-specific configuration is empty
        self.assertEqual(self.config_manager.get("database.host"), "localhost")
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test get_config
        self.assertEqual(get_config("database.host"), "localhost")
        
        # Test set_config
        set_config("database.host", "new-host")
        self.assertEqual(get_config("database.host"), "new-host")
        
        # Test get_credential and set_credential
        set_credential("api_key", "secret_value")
        self.assertEqual(get_credential("api_key"), "secret_value")
        
        # Test get_environment and set_environment
        set_environment(EnvironmentManager.PRODUCTION)
        self.assertEqual(get_environment(), EnvironmentManager.PRODUCTION)
        
        # Test is_production and is_development
        self.assertTrue(is_production())
        self.assertFalse(is_development())
        
        # Test validate_config
        schema = {
            "host": {
                "type": str,
                "required": True
            }
        }
        self.config_manager.register_validator("database", schema)
        self.assertEqual(len(validate_config("database")), 0)


if __name__ == "__main__":
    unittest.main()