"""Authentication Manager for External Systems.

This module provides classes for managing authentication with external systems,
including secure credential storage, token management, and authentication flows.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from enum import Enum, auto
import json
import time
import logging
import base64
import hashlib
import hmac
import secrets
import datetime
from abc import ABC, abstractmethod
from urllib.parse import urlencode

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.security import SecureCredentialStore
from src.infrastructure.error import FridayError, ErrorSeverity, ErrorCode

# Create logger
logger = get_logger(__name__)


class AuthType(Enum):
    """Enum for authentication types."""
    NONE = auto()
    API_KEY = auto()
    OAUTH = auto()
    JWT = auto()
    BASIC = auto()
    HMAC = auto()
    CUSTOM = auto()


class AuthError(FridayError):
    """Exception raised for authentication errors.

    Attributes:
        message: Explanation of the error.
        system_id: The external system ID where the error occurred.
        auth_type: The authentication type that failed.
    """

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        auth_type: Optional[AuthType] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        details: Any = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        self.system_id = system_id
        self.auth_type = auth_type
        super().__init__(
            message=message,
            severity=severity,
            troubleshooting_guidance=self._generate_guidance(),
            context={"system_id": system_id, "auth_type": auth_type.name if auth_type else None},
            cause=cause,
            error_code=error_code
        )

    def _generate_guidance(self) -> str:
        """Generate troubleshooting guidance based on the error."""
        if not self.auth_type:
            return "Check authentication configuration and credentials."
            
        if self.auth_type == AuthType.API_KEY:
            return "Verify that the API key is correct and has not expired."
        elif self.auth_type == AuthType.OAUTH:
            return "Check OAuth credentials and ensure the token is valid and has not expired."
        elif self.auth_type == AuthType.JWT:
            return "Verify JWT token validity and expiration."
        elif self.auth_type == AuthType.BASIC:
            return "Check username and password credentials."
        elif self.auth_type == AuthType.HMAC:
            return "Verify HMAC signing key and algorithm."
        else:
            return "Check authentication configuration and credentials."


class AuthManager(ABC):
    """Base class for authentication managers.

    This abstract class defines the interface for all authentication managers
    and provides common functionality for credential management and token handling.

    Attributes:
        config_manager: Configuration manager for accessing system configs.
        credential_store: Secure storage for authentication credentials.
        tokens: Cache of authentication tokens.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        credential_store: Optional[SecureCredentialStore] = None
    ):
        """Initialize an authentication manager.

        Args:
            config_manager: Configuration manager for accessing system configs.
            credential_store: Secure storage for authentication credentials.
        """
        self.config_manager = config_manager
        self.credential_store = credential_store or SecureCredentialStore()
        self.tokens: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing authentication information.
        """
        pass

    def get_system_config(self, system_id: str) -> Dict[str, Any]:
        """Get the configuration for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, Any]: The system configuration.

        Raises:
            AuthError: If the system configuration is not found.
        """
        external_systems = self.config_manager.get("external_systems", {})
        system_config = external_systems.get(system_id)
        
        if not system_config:
            raise AuthError(
                message=f"Configuration for external system '{system_id}' not found",
                system_id=system_id
            )
            
        return system_config

    def get_auth_config(self, system_id: str) -> Dict[str, Any]:
        """Get the authentication configuration for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, Any]: The authentication configuration.

        Raises:
            AuthError: If the authentication configuration is not found.
        """
        system_config = self.get_system_config(system_id)
        auth_config = system_config.get("authentication", {})
        
        if not auth_config:
            raise AuthError(
                message=f"Authentication configuration for external system '{system_id}' not found",
                system_id=system_id
            )
            
        return auth_config

    def get_auth_type(self, system_id: str) -> AuthType:
        """Get the authentication type for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            AuthType: The authentication type.
        """
        auth_config = self.get_auth_config(system_id)
        auth_type_str = auth_config.get("type", "none").upper()
        
        try:
            return AuthType[auth_type_str]
        except KeyError:
            logger.warning(f"Unknown authentication type '{auth_type_str}' for system '{system_id}', using NONE")
            return AuthType.NONE

    def get_credential(self, system_id: str, credential_key: str) -> str:
        """Get a credential for the specified external system.

        Args:
            system_id: The ID of the external system.
            credential_key: The key of the credential to retrieve.

        Returns:
            str: The credential value.

        Raises:
            AuthError: If the credential is not found.
        """
        credential_id = f"{system_id}:{credential_key}"
        
        try:
            return self.credential_store.get_credential(credential_id)
        except Exception as e:
            raise AuthError(
                message=f"Failed to retrieve credential '{credential_key}' for system '{system_id}'",
                system_id=system_id,
                cause=e
            )

    def store_credential(self, system_id: str, credential_key: str, credential_value: str) -> bool:
        """Store a credential for the specified external system.

        Args:
            system_id: The ID of the external system.
            credential_key: The key of the credential to store.
            credential_value: The value of the credential to store.

        Returns:
            bool: True if the credential was stored successfully, False otherwise.
        """
        credential_id = f"{system_id}:{credential_key}"
        
        try:
            self.credential_store.store_credential(credential_id, credential_value)
            return True
        except Exception as e:
            logger.error(f"Failed to store credential '{credential_key}' for system '{system_id}': {str(e)}")
            return False

    def is_token_valid(self, system_id: str) -> bool:
        """Check if the authentication token for the specified system is valid.

        Args:
            system_id: The ID of the external system.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        if system_id not in self.tokens:
            return False
            
        token_info = self.tokens[system_id]
        
        # Check if token has expired
        if "expires_at" in token_info and token_info["expires_at"] <= time.time():
            return False
            
        return True

    def store_token(self, system_id: str, token: str, expires_in: Optional[int] = None) -> None:
        """Store an authentication token for the specified system.

        Args:
            system_id: The ID of the external system.
            token: The authentication token.
            expires_in: The number of seconds until the token expires.
        """
        token_info = {
            "token": token,
            "created_at": time.time()
        }
        
        if expires_in is not None:
            token_info["expires_at"] = time.time() + expires_in
            
        self.tokens[system_id] = token_info
        logger.info(f"Stored authentication token for system '{system_id}'")

    def get_token(self, system_id: str) -> Optional[str]:
        """Get the authentication token for the specified system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Optional[str]: The authentication token, or None if not found or expired.
        """
        if not self.is_token_valid(system_id):
            return None
            
        return self.tokens[system_id]["token"]

    def clear_token(self, system_id: str) -> None:
        """Clear the authentication token for the specified system.

        Args:
            system_id: The ID of the external system.
        """
        if system_id in self.tokens:
            del self.tokens[system_id]
            logger.info(f"Cleared authentication token for system '{system_id}'")


class ApiKeyAuthManager(AuthManager):
    """Authentication manager for API key authentication.

    This class implements the AuthManager interface for API key authentication,
    providing methods for retrieving and using API keys.
    """

    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system using an API key.

        For API key authentication, this method simply checks if the API key
        is available and returns True, as there is no actual authentication step.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if the API key is available, False otherwise.
        """
        try:
            auth_config = self.get_auth_config(system_id)
            key_name = auth_config.get("key_name", "api_key")
            self.get_credential(system_id, key_name)
            return True
        except AuthError:
            return False

    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing the API key.

        Raises:
            AuthError: If the API key is not found.
        """
        auth_config = self.get_auth_config(system_id)
        key_name = auth_config.get("key_name", "api_key")
        header_name = auth_config.get("header_name", "X-API-Key")
        
        api_key = self.get_credential(system_id, key_name)
        
        return {header_name: api_key}


class BasicAuthManager(AuthManager):
    """Authentication manager for basic authentication.

    This class implements the AuthManager interface for basic authentication,
    providing methods for retrieving and using username/password credentials.
    """

    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system using basic authentication.

        For basic authentication, this method simply checks if the username and
        password are available and returns True, as there is no actual authentication step.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if the username and password are available, False otherwise.
        """
        try:
            self.get_credential(system_id, "username")
            self.get_credential(system_id, "password")
            return True
        except AuthError:
            return False

    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing the basic authentication credentials.

        Raises:
            AuthError: If the username or password is not found.
        """
        username = self.get_credential(system_id, "username")
        password = self.get_credential(system_id, "password")
        
        auth_string = f"{username}:{password}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        return {"Authorization": f"Basic {encoded_auth}"}


class OAuthManager(AuthManager):
    """Authentication manager for OAuth authentication.

    This class implements the AuthManager interface for OAuth authentication,
    providing methods for obtaining and refreshing OAuth tokens.
    """

    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system using OAuth.

        This method obtains an OAuth token if one is not already available or
        if the current token has expired.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        # Check if we already have a valid token
        if self.is_token_valid(system_id):
            return True
            
        # Get OAuth configuration
        auth_config = self.get_auth_config(system_id)
        grant_type = auth_config.get("grant_type", "client_credentials")
        token_url = auth_config.get("token_url")
        
        if not token_url:
            logger.error(f"OAuth token URL not specified for system '{system_id}'")
            return False
            
        # Get client credentials
        try:
            client_id = self.get_credential(system_id, "client_id")
            client_secret = self.get_credential(system_id, "client_secret")
        except AuthError as e:
            logger.error(f"Failed to retrieve OAuth credentials for system '{system_id}': {str(e)}")
            return False
            
        # Prepare token request
        import requests
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": grant_type, "client_id": client_id, "client_secret": client_secret}
        
        # Add additional parameters based on grant type
        if grant_type == "password":
            try:
                data["username"] = self.get_credential(system_id, "username")
                data["password"] = self.get_credential(system_id, "password")
            except AuthError as e:
                logger.error(f"Failed to retrieve username/password for system '{system_id}': {str(e)}")
                return False
        elif grant_type == "refresh_token" and system_id in self.tokens:
            data["refresh_token"] = self.tokens[system_id].get("refresh_token")
            
        # Make token request
        try:
            response = requests.post(token_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in")
            refresh_token = token_data.get("refresh_token")
            
            if not access_token:
                logger.error(f"OAuth token response for system '{system_id}' did not contain access_token")
                return False
                
            # Store token
            token_info = {
                "token": access_token,
                "created_at": time.time()
            }
            
            if expires_in is not None:
                token_info["expires_at"] = time.time() + expires_in
                
            if refresh_token:
                token_info["refresh_token"] = refresh_token
                
            self.tokens[system_id] = token_info
            logger.info(f"Successfully obtained OAuth token for system '{system_id}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to obtain OAuth token for system '{system_id}': {str(e)}")
            return False

    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing the OAuth token.

        Raises:
            AuthError: If authentication fails or the token is not available.
        """
        # Ensure we have a valid token
        if not self.is_token_valid(system_id) and not self.authenticate(system_id):
            raise AuthError(
                message=f"Failed to authenticate with system '{system_id}'",
                system_id=system_id,
                auth_type=AuthType.OAUTH
            )
            
        token = self.get_token(system_id)
        return {"Authorization": f"Bearer {token}"}


class HmacAuthManager(AuthManager):
    """Authentication manager for HMAC authentication.

    This class implements the AuthManager interface for HMAC authentication,
    providing methods for signing requests with HMAC.
    """

    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system using HMAC.

        For HMAC authentication, this method simply checks if the signing key
        is available and returns True, as the actual authentication happens
        when signing individual requests.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if the signing key is available, False otherwise.
        """
        try:
            self.get_credential(system_id, "signing_key")
            return True
        except AuthError:
            return False

    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        For HMAC authentication, this method returns a timestamp and nonce
        that will be used for signing requests. The actual signature will
        need to be computed for each request based on its content.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing the timestamp and nonce.
        """
        auth_config = self.get_auth_config(system_id)
        timestamp_header = auth_config.get("timestamp_header", "X-Timestamp")
        nonce_header = auth_config.get("nonce_header", "X-Nonce")
        
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        
        return {
            timestamp_header: timestamp,
            nonce_header: nonce
        }

    def sign_request(
        self,
        system_id: str,
        method: str,
        path: str,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None
    ) -> Dict[str, str]:
        """Sign a request using HMAC.

        Args:
            system_id: The ID of the external system.
            method: The HTTP method of the request.
            path: The path of the request.
            query_params: The query parameters of the request.
            headers: The headers of the request.
            body: The body of the request.

        Returns:
            Dict[str, str]: Headers containing the HMAC signature.

        Raises:
            AuthError: If the signing key is not found or signing fails.
        """
        auth_config = self.get_auth_config(system_id)
        algorithm = auth_config.get("algorithm", "sha256")
        signature_header = auth_config.get("signature_header", "X-Signature")
        include_body = auth_config.get("include_body", True)
        
        # Get signing key
        signing_key = self.get_credential(system_id, "signing_key")
        
        # Get or create timestamp and nonce
        auth_headers = self.get_auth_headers(system_id)
        timestamp = auth_headers.get(auth_config.get("timestamp_header", "X-Timestamp"))
        nonce = auth_headers.get(auth_config.get("nonce_header", "X-Nonce"))
        
        # Build string to sign
        string_to_sign = f"{method.upper()}\n{path}\n"
        
        if query_params:
            sorted_params = sorted(query_params.items())
            query_string = urlencode(sorted_params)
            string_to_sign += f"{query_string}\n"
            
        string_to_sign += f"{timestamp}\n{nonce}\n"
        
        if include_body and body:
            string_to_sign += body
            
        # Compute signature
        try:
            if algorithm.lower() == "sha256":
                hash_func = hashlib.sha256
            elif algorithm.lower() == "sha512":
                hash_func = hashlib.sha512
            elif algorithm.lower() == "md5":
                hash_func = hashlib.md5
            else:
                raise AuthError(
                    message=f"Unsupported HMAC algorithm '{algorithm}' for system '{system_id}'",
                    system_id=system_id,
                    auth_type=AuthType.HMAC
                )
                
            signature = hmac.new(
                signing_key.encode(),
                string_to_sign.encode(),
                hash_func
            ).hexdigest()
            
            # Add signature to headers
            auth_headers[signature_header] = signature
            
            return auth_headers
            
        except Exception as e:
            raise AuthError(
                message=f"Failed to sign request for system '{system_id}'",
                system_id=system_id,
                auth_type=AuthType.HMAC,
                cause=e
            )


class JwtAuthManager(AuthManager):
    """Authentication manager for JWT authentication.

    This class implements the AuthManager interface for JWT authentication,
    providing methods for obtaining and using JWT tokens.
    """

    def authenticate(self, system_id: str) -> bool:
        """Authenticate with the specified external system using JWT.

        This method obtains a JWT token if one is not already available or
        if the current token has expired.

        Args:
            system_id: The ID of the external system to authenticate with.

        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        # Check if we already have a valid token
        if self.is_token_valid(system_id):
            return True
            
        # Get JWT configuration
        auth_config = self.get_auth_config(system_id)
        token_url = auth_config.get("token_url")
        
        if token_url:
            # Token is obtained from an endpoint
            return self._authenticate_with_endpoint(system_id, token_url, auth_config)
        else:
            # Token is generated locally
            return self._generate_jwt_token(system_id, auth_config)

    def _authenticate_with_endpoint(self, system_id: str, token_url: str, auth_config: Dict[str, Any]) -> bool:
        """Authenticate with a JWT token endpoint.

        Args:
            system_id: The ID of the external system.
            token_url: The URL of the token endpoint.
            auth_config: The authentication configuration.

        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        import requests
        
        # Get credentials
        try:
            username = self.get_credential(system_id, "username")
            password = self.get_credential(system_id, "password")
        except AuthError as e:
            logger.error(f"Failed to retrieve JWT credentials for system '{system_id}': {str(e)}")
            return False
            
        # Prepare token request
        headers = {"Content-Type": "application/json"}
        data = {"username": username, "password": password}
        
        # Add additional fields from config
        additional_fields = auth_config.get("additional_fields", {})
        data.update(additional_fields)
        
        # Make token request
        try:
            response = requests.post(token_url, headers=headers, json=data)
            response.raise_for_status()
            
            token_data = response.json()
            token_field = auth_config.get("token_field", "token")
            expires_field = auth_config.get("expires_field", "expires_in")
            
            token = token_data.get(token_field)
            expires_in = token_data.get(expires_field)
            
            if not token:
                logger.error(f"JWT token response for system '{system_id}' did not contain token field '{token_field}'")
                return False
                
            # Store token
            self.store_token(system_id, token, expires_in)
            logger.info(f"Successfully obtained JWT token for system '{system_id}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to obtain JWT token for system '{system_id}': {str(e)}")
            return False

    def _generate_jwt_token(self, system_id: str, auth_config: Dict[str, Any]) -> bool:
        """Generate a JWT token locally.

        Args:
            system_id: The ID of the external system.
            auth_config: The authentication configuration.

        Returns:
            bool: True if token generation was successful, False otherwise.
        """
        try:
            import jwt
            
            # Get signing key
            signing_key = self.get_credential(system_id, "signing_key")
            
            # Prepare token payload
            algorithm = auth_config.get("algorithm", "HS256")
            issuer = auth_config.get("issuer")
            audience = auth_config.get("audience")
            subject = auth_config.get("subject")
            expires_in = auth_config.get("expires_in", 3600)  # Default: 1 hour
            
            now = datetime.datetime.utcnow()
            payload = {
                "iat": now,
                "exp": now + datetime.timedelta(seconds=expires_in)
            }
            
            if issuer:
                payload["iss"] = issuer
            if audience:
                payload["aud"] = audience
            if subject:
                payload["sub"] = subject
                
            # Add additional claims from config
            additional_claims = auth_config.get("additional_claims", {})
            payload.update(additional_claims)
            
            # Generate token
            token = jwt.encode(payload, signing_key, algorithm=algorithm)
            
            # If token is bytes, convert to string (depends on jwt library version)
            if isinstance(token, bytes):
                token = token.decode("utf-8")
                
            # Store token
            self.store_token(system_id, token, expires_in)
            logger.info(f"Successfully generated JWT token for system '{system_id}'")
            
            return True
            
        except ImportError:
            logger.error("PyJWT library not installed. Install it with 'pip install pyjwt'")
            return False
        except Exception as e:
            logger.error(f"Failed to generate JWT token for system '{system_id}': {str(e)}")
            return False

    def get_auth_headers(self, system_id: str) -> Dict[str, str]:
        """Get authentication headers for the specified external system.

        Args:
            system_id: The ID of the external system.

        Returns:
            Dict[str, str]: Headers containing the JWT token.

        Raises:
            AuthError: If authentication fails or the token is not available.
        """
        # Ensure we have a valid token
        if not self.is_token_valid(system_id) and not self.authenticate(system_id):
            raise AuthError(
                message=f"Failed to authenticate with system '{system_id}'",
                system_id=system_id,
                auth_type=AuthType.JWT
            )
            
        token = self.get_token(system_id)
        return {"Authorization": f"Bearer {token}"}


class AuthManagerFactory:
    """Factory for creating authentication managers.

    This class provides methods for creating authentication managers based on
    the authentication type specified in the system configuration.
    """

    @staticmethod
    def create_auth_manager(
        auth_type: AuthType,
        config_manager: ConfigManager,
        credential_store: Optional[SecureCredentialStore] = None
    ) -> AuthManager:
        """Create an authentication manager for the specified authentication type.

        Args:
            auth_type: The authentication type.
            config_manager: Configuration manager for accessing system configs.
            credential_store: Secure storage for authentication credentials.

        Returns:
            AuthManager: An authentication manager for the specified type.

        Raises:
            ValueError: If the authentication type is not supported.
        """
        if auth_type == AuthType.API_KEY:
            return ApiKeyAuthManager(config_manager, credential_store)
        elif auth_type == AuthType.BASIC:
            return BasicAuthManager(config_manager, credential_store)
        elif auth_type == AuthType.OAUTH:
            return OAuthManager(config_manager, credential_store)
        elif auth_type == AuthType.HMAC:
            return HmacAuthManager(config_manager, credential_store)
        elif auth_type == AuthType.JWT:
            return JwtAuthManager(config_manager, credential_store)
        elif auth_type == AuthType.NONE:
            return AuthManager(config_manager, credential_store)  # type: ignore
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")

    @staticmethod
    def create_auth_manager_for_system(
        system_id: str,
        config_manager: ConfigManager,
        credential_store: Optional[SecureCredentialStore] = None
    ) -> AuthManager:
        """Create an authentication manager for the specified external system.

        Args:
            system_id: The ID of the external system.
            config_manager: Configuration manager for accessing system configs.
            credential_store: Secure storage for authentication credentials.

        Returns:
            AuthManager: An authentication manager for the specified system.

        Raises:
            AuthError: If the system configuration is not found or the authentication type is not supported.
        """
        # Create a temporary auth manager to get the auth type
        temp_manager = AuthManager(config_manager, credential_store)  # type: ignore
        
        try:
            auth_type = temp_manager.get_auth_type(system_id)
            return AuthManagerFactory.create_auth_manager(auth_type, config_manager, credential_store)
        except Exception as e:
            raise AuthError(
                message=f"Failed to create authentication manager for system '{system_id}'",
                system_id=system_id,
                cause=e
            )