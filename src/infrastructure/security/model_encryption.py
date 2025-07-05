"""Model Encryption for Friday AI Trading System.

This module provides functionality for encrypting and decrypting sensitive models.
"""

import os
import json
import base64
import hashlib
from typing import Dict, Optional, Any, Union, BinaryIO
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.infrastructure.logging import get_logger
from src.infrastructure.security import generate_key, encrypt_data, decrypt_data

# Create logger
logger = get_logger(__name__)


class ModelEncryptor:
    """Handles encryption and decryption of sensitive models."""
    
    def __init__(self, key_file: Optional[str] = None, salt: Optional[bytes] = None):
        """Initialize the model encryptor.
        
        Args:
            key_file: Path to the encryption key file. If None, a new key will be generated.
            salt: Salt for key derivation. If None, a random salt will be generated.
        """
        self.key_file = key_file
        
        if salt is None:
            # Generate a random salt if none is provided
            self.salt = os.urandom(16)
        else:
            self.salt = salt
        
        if key_file and os.path.exists(key_file):
            # Load the key from the file
            with open(key_file, "rb") as f:
                key_data = json.load(f)
                self.key = base64.urlsafe_b64decode(key_data["key"])
                # If salt is stored in the key file, use it
                if "salt" in key_data:
                    self.salt = base64.urlsafe_b64decode(key_data["salt"])
        else:
            # Generate a new key
            self.key = Fernet.generate_key()
            if key_file:
                # Save the key to the file
                self._save_key()
        
        # Initialize the Fernet cipher with the key
        self.cipher = Fernet(self.key)
        
        logger.info("Initialized ModelEncryptor")
    
    def _save_key(self) -> None:
        """Save the encryption key to the key file."""
        if self.key_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            # Save the key and salt to the file
            with open(self.key_file, "w") as f:
                key_data = {
                    "key": base64.urlsafe_b64encode(self.key).decode(),
                    "salt": base64.urlsafe_b64encode(self.salt).decode()
                }
                json.dump(key_data, f)
            
            logger.info(f"Saved encryption key to {self.key_file}")
    
    def derive_key_from_password(self, password: str) -> bytes:
        """Derive an encryption key from a password.
        
        Args:
            password: The password to derive the key from.
            
        Returns:
            bytes: The derived key.
        """
        # Convert the password to bytes
        password_bytes = password.encode()
        
        # Create a PBKDF2HMAC instance for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 32 bytes = 256 bits
            salt=self.salt,
            iterations=100000  # Number of iterations (higher is more secure but slower)
        )
        
        # Derive the key from the password
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        
        return key
    
    def set_key_from_password(self, password: str) -> None:
        """Set the encryption key from a password.
        
        Args:
            password: The password to derive the key from.
        """
        # Derive the key from the password
        self.key = self.derive_key_from_password(password)
        
        # Initialize the Fernet cipher with the new key
        self.cipher = Fernet(self.key)
        
        # Save the key to the file if a key file is specified
        if self.key_file:
            self._save_key()
        
        logger.info("Set encryption key from password")
    
    def encrypt_model_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Encrypt a model file.
        
        Args:
            input_file: Path to the model file to encrypt.
            output_file: Path to save the encrypted model file. If None, the input file will be
                         replaced with an encrypted version.
            
        Returns:
            str: Path to the encrypted model file.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Model file {input_file} not found")
        
        # If no output file is specified, use the input file with a .enc extension
        if output_file is None:
            output_file = f"{input_file}.enc"
        
        # Read the input file
        with open(input_file, "rb") as f:
            data = f.read()
        
        # Encrypt the data
        encrypted_data = self.cipher.encrypt(data)
        
        # Write the encrypted data to the output file
        with open(output_file, "wb") as f:
            f.write(encrypted_data)
        
        logger.info(f"Encrypted model file {input_file} to {output_file}")
        
        return output_file
    
    def decrypt_model_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Decrypt a model file.
        
        Args:
            input_file: Path to the encrypted model file.
            output_file: Path to save the decrypted model file. If None, the input file will be
                         replaced with a decrypted version (without the .enc extension if present).
            
        Returns:
            str: Path to the decrypted model file.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Encrypted model file {input_file} not found")
        
        # If no output file is specified, use the input file without the .enc extension
        if output_file is None:
            if input_file.endswith(".enc"):
                output_file = input_file[:-4]  # Remove the .enc extension
            else:
                output_file = f"{input_file}.dec"
        
        # Read the encrypted input file
        with open(input_file, "rb") as f:
            encrypted_data = f.read()
        
        # Decrypt the data
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt model file {input_file}: {e}")
            raise ValueError(f"Failed to decrypt model file: {e}")
        
        # Write the decrypted data to the output file
        with open(output_file, "wb") as f:
            f.write(decrypted_data)
        
        logger.info(f"Decrypted model file {input_file} to {output_file}")
        
        return output_file
    
    def is_file_encrypted(self, file_path: str) -> bool:
        """Check if a file is encrypted.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            bool: True if the file is encrypted, False otherwise.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Check if the file has a .enc extension
        if file_path.endswith(".enc"):
            return True
        
        # Try to decrypt the first few bytes of the file
        try:
            with open(file_path, "rb") as f:
                # Read the first 100 bytes (or less if the file is smaller)
                data = f.read(100)
            
            # Try to decrypt the data
            self.cipher.decrypt(data)
            
            # If decryption succeeds, the file is likely encrypted
            return True
        except Exception:
            # If decryption fails, the file is likely not encrypted
            return False
    
    def encrypt_model_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in model metadata.
        
        Args:
            metadata: The model metadata to encrypt.
            
        Returns:
            Dict[str, Any]: The metadata with sensitive fields encrypted.
        """
        # Define sensitive fields that should be encrypted
        sensitive_fields = ["api_key", "credentials", "password", "secret", "token"]
        
        # Create a copy of the metadata
        encrypted_metadata = metadata.copy()
        
        # Encrypt sensitive fields
        for field in sensitive_fields:
            if field in encrypted_metadata and encrypted_metadata[field] is not None:
                # Convert the value to a string if it's not already
                value = str(encrypted_metadata[field])
                
                # Encrypt the value
                encrypted_value = self.cipher.encrypt(value.encode()).decode()
                
                # Replace the value with the encrypted value
                encrypted_metadata[field] = f"ENCRYPTED:{encrypted_value}"
        
        # Mark the metadata as containing encrypted fields
        encrypted_metadata["_contains_encrypted_fields"] = True
        
        return encrypted_metadata
    
    def decrypt_model_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in model metadata.
        
        Args:
            metadata: The model metadata with encrypted fields.
            
        Returns:
            Dict[str, Any]: The metadata with sensitive fields decrypted.
        """
        # Check if the metadata contains encrypted fields
        if not metadata.get("_contains_encrypted_fields", False):
            return metadata
        
        # Create a copy of the metadata
        decrypted_metadata = metadata.copy()
        
        # Remove the marker
        if "_contains_encrypted_fields" in decrypted_metadata:
            del decrypted_metadata["_contains_encrypted_fields"]
        
        # Decrypt encrypted fields
        for field, value in decrypted_metadata.items():
            if isinstance(value, str) and value.startswith("ENCRYPTED:"):
                # Extract the encrypted value
                encrypted_value = value[len("ENCRYPTED:"):]
                
                # Decrypt the value
                try:
                    decrypted_value = self.cipher.decrypt(encrypted_value.encode()).decode()
                    
                    # Replace the encrypted value with the decrypted value
                    decrypted_metadata[field] = decrypted_value
                except Exception as e:
                    logger.error(f"Failed to decrypt field {field}: {e}")
                    # Keep the encrypted value if decryption fails
        
        return decrypted_metadata


class ModelEncryptionManager:
    """Manages encryption for models in the model registry."""
    
    def __init__(self, model_registry_path: str, key_file: Optional[str] = None):
        """Initialize the model encryption manager.
        
        Args:
            model_registry_path: Path to the model registry directory.
            key_file: Path to the encryption key file. If None, a default path will be used.
        """
        self.model_registry_path = model_registry_path
        
        # If no key file is specified, use a default path
        if key_file is None:
            key_file = os.path.join(model_registry_path, "encryption", "model_encryption_key.json")
        
        # Initialize the model encryptor
        self.encryptor = ModelEncryptor(key_file=key_file)
        
        logger.info("Initialized ModelEncryptionManager")
    
    def encrypt_model(self, model_name: str, model_version: str, password: Optional[str] = None) -> None:
        """Encrypt a model in the model registry.
        
        Args:
            model_name: The name of the model to encrypt.
            model_version: The version of the model to encrypt.
            password: Optional password for encryption. If provided, a key will be derived from it.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        # Set the key from the password if provided
        if password:
            self.encryptor.set_key_from_password(password)
        
        # Construct the path to the model file
        model_path = os.path.join(self.model_registry_path, model_name, model_version, "model.pkl")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Check if the model is already encrypted
        if self.encryptor.is_file_encrypted(model_path):
            logger.warning(f"Model {model_name} version {model_version} is already encrypted")
            return
        
        # Encrypt the model file
        encrypted_path = self.encryptor.encrypt_model_file(model_path)
        
        # Update the model metadata to indicate that the model is encrypted
        metadata_path = os.path.join(self.model_registry_path, model_name, model_version, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Update the metadata
            metadata["encrypted"] = True
            metadata["encryption_method"] = "fernet"
            
            # Encrypt sensitive fields in the metadata
            metadata = self.encryptor.encrypt_model_metadata(metadata)
            
            # Save the updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Encrypted model {model_name} version {model_version}")
    
    def decrypt_model(self, model_name: str, model_version: str, password: Optional[str] = None) -> None:
        """Decrypt a model in the model registry.
        
        Args:
            model_name: The name of the model to decrypt.
            model_version: The version of the model to decrypt.
            password: Optional password for decryption. If provided, a key will be derived from it.
            
        Raises:
            FileNotFoundError: If the encrypted model file does not exist.
            ValueError: If decryption fails.
        """
        # Set the key from the password if provided
        if password:
            self.encryptor.set_key_from_password(password)
        
        # Construct the path to the encrypted model file
        model_path = os.path.join(self.model_registry_path, model_name, model_version, "model.pkl.enc")
        
        # Check if the encrypted model file exists
        if not os.path.exists(model_path):
            # Try without the .enc extension
            model_path = os.path.join(self.model_registry_path, model_name, model_version, "model.pkl")
            
            # Check if the model file exists and is encrypted
            if not os.path.exists(model_path) or not self.encryptor.is_file_encrypted(model_path):
                raise FileNotFoundError(f"Encrypted model file for {model_name} version {model_version} not found")
        
        # Decrypt the model file
        decrypted_path = self.encryptor.decrypt_model_file(model_path)
        
        # Update the model metadata to indicate that the model is no longer encrypted
        metadata_path = os.path.join(self.model_registry_path, model_name, model_version, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Decrypt sensitive fields in the metadata
            metadata = self.encryptor.decrypt_model_metadata(metadata)
            
            # Update the metadata
            metadata["encrypted"] = False
            if "encryption_method" in metadata:
                del metadata["encryption_method"]
            
            # Save the updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Decrypted model {model_name} version {model_version}")
    
    def is_model_encrypted(self, model_name: str, model_version: str) -> bool:
        """Check if a model is encrypted.
        
        Args:
            model_name: The name of the model to check.
            model_version: The version of the model to check.
            
        Returns:
            bool: True if the model is encrypted, False otherwise.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        # Construct the path to the model file
        model_path = os.path.join(self.model_registry_path, model_name, model_version, "model.pkl")
        encrypted_path = f"{model_path}.enc"
        
        # Check if the encrypted model file exists
        if os.path.exists(encrypted_path):
            return True
        
        # Check if the model file exists and is encrypted
        if os.path.exists(model_path):
            return self.encryptor.is_file_encrypted(model_path)
        
        # If neither file exists, the model does not exist
        raise FileNotFoundError(f"Model file for {model_name} version {model_version} not found")
    
    def encrypt_all_models(self, password: Optional[str] = None) -> None:
        """Encrypt all models in the model registry.
        
        Args:
            password: Optional password for encryption. If provided, a key will be derived from it.
        """
        # Set the key from the password if provided
        if password:
            self.encryptor.set_key_from_password(password)
        
        # Walk through the model registry directory
        for model_name in os.listdir(self.model_registry_path):
            model_dir = os.path.join(self.model_registry_path, model_name)
            
            # Skip non-directory entries
            if not os.path.isdir(model_dir):
                continue
            
            # Iterate through model versions
            for version in os.listdir(model_dir):
                version_dir = os.path.join(model_dir, version)
                
                # Skip non-directory entries
                if not os.path.isdir(version_dir):
                    continue
                
                # Check if the model file exists
                model_path = os.path.join(version_dir, "model.pkl")
                if os.path.exists(model_path):
                    try:
                        # Encrypt the model if it's not already encrypted
                        if not self.encryptor.is_file_encrypted(model_path):
                            self.encrypt_model(model_name, version, password)
                    except Exception as e:
                        logger.error(f"Failed to encrypt model {model_name} version {version}: {e}")
        
        logger.info("Encrypted all models in the model registry")