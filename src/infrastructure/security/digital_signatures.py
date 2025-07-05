"""Digital Signatures for Friday AI Trading System.

This module provides functionality for creating and verifying digital signatures
for model files to ensure their authenticity and integrity.
"""

import os
import hashlib
import hmac
import base64
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class ModelSigner:
    """Class for signing and verifying model files.
    
    This class provides functionality for generating key pairs, signing model files,
    and verifying signatures to ensure model authenticity and integrity.
    """
    
    def __init__(self, keys_dir: Optional[str] = None):
        """Initialize the ModelSigner.
        
        Args:
            keys_dir: Directory to store key pairs. If None, defaults to
                'keys/model_signatures' in the current working directory.
        """
        self.keys_dir = keys_dir or os.path.join(os.getcwd(), "keys", "model_signatures")
        os.makedirs(self.keys_dir, exist_ok=True)
        logger.info(f"Initialized ModelSigner with keys directory at {self.keys_dir}")
    
    def generate_key_pair(self, key_name: str) -> Tuple[str, str]:
        """Generate a new RSA key pair.
        
        Args:
            key_name: Name for the key pair.
            
        Returns:
            Tuple[str, str]: Paths to the private and public key files.
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys to files
        private_key_path = os.path.join(self.keys_dir, f"{key_name}_private.pem")
        public_key_path = os.path.join(self.keys_dir, f"{key_name}_public.pem")
        
        with open(private_key_path, "wb") as f:
            f.write(private_pem)
        
        with open(public_key_path, "wb") as f:
            f.write(public_pem)
        
        logger.info(f"Generated key pair '{key_name}' and saved to {self.keys_dir}")
        
        return private_key_path, public_key_path
    
    def sign_model(self, model_path: str, private_key_path: str) -> str:
        """Sign a model file using a private key.
        
        Args:
            model_path: Path to the model file.
            private_key_path: Path to the private key file.
            
        Returns:
            str: The base64-encoded signature.
            
        Raises:
            FileNotFoundError: If the model file or private key file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        if not os.path.exists(private_key_path):
            raise FileNotFoundError(f"Private key file {private_key_path} not found")
        
        # Load private key
        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
        
        # Calculate hash of model file
        file_hash = self._calculate_file_hash(model_path)
        
        # Sign the hash
        signature = private_key.sign(
            file_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Encode signature as base64
        encoded_signature = base64.b64encode(signature).decode()
        
        # Save signature to file
        signature_path = f"{model_path}.sig"
        with open(signature_path, "w") as f:
            f.write(encoded_signature)
        
        logger.info(f"Signed model {model_path} and saved signature to {signature_path}")
        
        return encoded_signature
    
    def verify_signature(self, model_path: str, signature: str, public_key_path: str) -> bool:
        """Verify a model signature using a public key.
        
        Args:
            model_path: Path to the model file.
            signature: The base64-encoded signature.
            public_key_path: Path to the public key file.
            
        Returns:
            bool: True if the signature is valid, False otherwise.
            
        Raises:
            FileNotFoundError: If the model file or public key file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        if not os.path.exists(public_key_path):
            raise FileNotFoundError(f"Public key file {public_key_path} not found")
        
        # Load public key
        with open(public_key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())
        
        # Calculate hash of model file
        file_hash = self._calculate_file_hash(model_path)
        
        # Decode signature from base64
        decoded_signature = base64.b64decode(signature)
        
        try:
            # Verify the signature
            public_key.verify(
                decoded_signature,
                file_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            logger.info(f"Signature verification successful for model {model_path}")
            return True
        except InvalidSignature:
            logger.warning(f"Signature verification failed for model {model_path}")
            return False
    
    def verify_model(self, model_path: str, public_key_path: str) -> bool:
        """Verify a model using its signature file and a public key.
        
        Args:
            model_path: Path to the model file.
            public_key_path: Path to the public key file.
            
        Returns:
            bool: True if the signature is valid, False otherwise.
            
        Raises:
            FileNotFoundError: If the model file, signature file, or public key file does not exist.
        """
        signature_path = f"{model_path}.sig"
        
        if not os.path.exists(signature_path):
            raise FileNotFoundError(f"Signature file {signature_path} not found")
        
        # Load signature from file
        with open(signature_path, "r") as f:
            signature = f.read().strip()
        
        return self.verify_signature(model_path, signature, public_key_path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate the SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            str: The hexadecimal digest of the hash.
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()