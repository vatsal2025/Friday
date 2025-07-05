"""API Key Authentication implementation."""

import hashlib
import secrets
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader


# API key header field name
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Global instance of APIKeyAuth
_api_key_auth_instance = None


def get_api_key_auth() -> 'APIKeyAuth':
    """Get the global APIKeyAuth instance.
    
    Returns:
        Global APIKeyAuth instance
    """
    global _api_key_auth_instance
    if _api_key_auth_instance is None:
        _api_key_auth_instance = APIKeyAuth()
    return _api_key_auth_instance


async def api_key_auth(api_key: str = Security(api_key_header)):
    """Dependency for API key authentication.
    
    Args:
        api_key: API key from request header
        
    Returns:
        Validated API key information
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    auth = get_api_key_auth()
    key_info = auth.validate_api_key(api_key)
    
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return key_info


class APIKeyAuth:
    """API Key Authentication handler."""

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize API key authentication.
        
        Args:
            secret_key: Secret key for signing tokens
        """
        self._secret_key = secret_key or secrets.token_urlsafe(32)
        self._api_keys: Dict[str, Dict[str, Any]] = {}

    def generate_api_key(self, user_id: str, permissions: Optional[list] = None) -> str:
        """Generate a new API key for a user.
        
        Args:
            user_id: User identifier
            permissions: List of permissions for this key
            
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        self._api_keys[api_key] = {
            'user_id': user_id,
            'permissions': permissions or [],
            'created_at': datetime.utcnow(),
            'last_used': None,
            'active': True
        }
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key information if valid, None otherwise
        """
        if api_key not in self._api_keys:
            return None
            
        key_info = self._api_keys[api_key]
        if not key_info.get('active', False):
            return None
            
        # Update last used timestamp
        key_info['last_used'] = datetime.utcnow()
        return key_info

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        if api_key in self._api_keys:
            self._api_keys[api_key]['active'] = False
            return True
        return False

    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if an API key has a specific permission.
        
        Args:
            api_key: API key to check
            permission: Permission to check for
            
        Returns:
            True if key has permission, False otherwise
        """
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return False
            
        permissions = key_info.get('permissions', [])
        return permission in permissions or 'admin' in permissions
