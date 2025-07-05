"""Authentication middleware for request processing."""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from .api_key_auth import APIKeyAuth


logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Authentication middleware for handling API key validation."""

    def __init__(self, api_key_auth: Optional[APIKeyAuth] = None):
        """Initialize authentication middleware.
        
        Args:
            api_key_auth: API key authentication handler
        """
        self.api_key_auth = api_key_auth or APIKeyAuth()

    def validate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Validate authentication in request headers.
        
        Args:
            headers: Request headers
            
        Returns:
            User info if authenticated, None otherwise
        """
        # Check for API key in headers
        api_key = headers.get('X-API-Key') or headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            logger.warning("No API key provided in request")
            return None
            
        # Validate API key
        key_info = self.api_key_auth.validate_api_key(api_key)
        if not key_info:
            logger.warning(f"Invalid API key: {api_key[:8]}...")
            return None
            
        logger.info(f"Authenticated user: {key_info.get('user_id')}")
        return key_info

    def require_auth(self, permission: Optional[str] = None):
        """Decorator to require authentication for a function.
        
        Args:
            permission: Optional permission required
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This is a simplified implementation
                # In a real application, you'd extract headers from the request context
                headers = kwargs.get('headers', {})
                
                user_info = self.validate_request(headers)
                if not user_info:
                    raise PermissionError("Authentication required")
                    
                if permission and not self.api_key_auth.has_permission(
                    headers.get('X-API-Key', ''), permission
                ):
                    raise PermissionError(f"Permission '{permission}' required")
                    
                # Add user info to kwargs
                kwargs['user_info'] = user_info
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def check_permission(self, api_key: str, permission: str) -> bool:
        """Check if an API key has a specific permission.
        
        Args:
            api_key: API key to check
            permission: Permission to check for
            
        Returns:
            True if key has permission, False otherwise
        """
        return self.api_key_auth.has_permission(api_key, permission)
