"""Security module for the Friday AI Trading System.

This module provides functions for authentication, authorization, and secure
handling of sensitive information such as API keys and credentials.
"""

import base64
import hashlib
import hmac
import os
from typing import Dict, Optional, Tuple

from cryptography.fernet import Fernet


def generate_key() -> bytes:
    """Generate a new encryption key.

    Returns:
        bytes: The generated key.
    """
    return Fernet.generate_key()


def encrypt_data(data: str, key: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Encrypt data using Fernet symmetric encryption.

    Args:
        data: The data to encrypt.
        key: The encryption key. If None, a new key will be generated.

    Returns:
        Tuple[bytes, bytes]: A tuple containing the encrypted data and the key.
    """
    if key is None:
        key = generate_key()

    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data, key


def decrypt_data(encrypted_data: bytes, key: bytes) -> str:
    """Decrypt data using Fernet symmetric encryption.

    Args:
        encrypted_data: The encrypted data.
        key: The encryption key.

    Returns:
        str: The decrypted data.

    Raises:
        cryptography.fernet.InvalidToken: If the token is invalid or expired.
    """
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return decrypted_data.decode()


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Hash a password using PBKDF2.

    Args:
        password: The password to hash.
        salt: The salt to use. If None, a new salt will be generated.

    Returns:
        Tuple[bytes, bytes]: A tuple containing the hashed password and the salt.
    """
    if salt is None:
        salt = os.urandom(32)

    # Use PBKDF2 with SHA-256, 100,000 iterations
    hashed_password = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, 100000
    )

    return hashed_password, salt


def verify_password(password: str, hashed_password: bytes, salt: bytes) -> bool:
    """Verify a password against a hash.

    Args:
        password: The password to verify.
        hashed_password: The hashed password to compare against.
        salt: The salt used to hash the password.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    # Hash the provided password with the same salt
    new_hash, _ = hash_password(password, salt)

    # Compare the hashes using a constant-time comparison function
    return hmac.compare_digest(new_hash, hashed_password)


def generate_api_key() -> str:
    """Generate a new API key.

    Returns:
        str: The generated API key.
    """
    # Generate 32 random bytes and encode as base64
    random_bytes = os.urandom(32)
    api_key = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
    return api_key


def generate_api_secret() -> str:
    """Generate a new API secret.

    Returns:
        str: The generated API secret.
    """
    # Generate 64 random bytes and encode as base64
    random_bytes = os.urandom(64)
    api_secret = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
    return api_secret


def secure_api_credentials(api_key: str, api_secret: str) -> Dict[str, bytes]:
    """Securely store API credentials.

    Args:
        api_key: The API key.
        api_secret: The API secret.

    Returns:
        Dict[str, bytes]: A dictionary containing the encrypted credentials and keys.
    """
    # Encrypt API key
    encrypted_key, key_encryption_key = encrypt_data(api_key)

    # Encrypt API secret
    encrypted_secret, secret_encryption_key = encrypt_data(api_secret)

    return {
        "encrypted_key": encrypted_key,
        "key_encryption_key": key_encryption_key,
        "encrypted_secret": encrypted_secret,
        "secret_encryption_key": secret_encryption_key,
    }


def retrieve_api_credentials(
    encrypted_credentials: Dict[str, bytes]
) -> Tuple[str, str]:
    """Retrieve API credentials from encrypted storage.

    Args:
        encrypted_credentials: The encrypted credentials and keys.

    Returns:
        Tuple[str, str]: A tuple containing the API key and secret.
    """
    # Decrypt API key
    api_key = decrypt_data(
        encrypted_credentials["encrypted_key"],
        encrypted_credentials["key_encryption_key"],
    )

    # Decrypt API secret
    api_secret = decrypt_data(
        encrypted_credentials["encrypted_secret"],
        encrypted_credentials["secret_encryption_key"],
    )

    return api_key, api_secret