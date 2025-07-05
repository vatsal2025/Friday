"""Authentication module for Friday AI Trading System.

This module provides authentication and authorization functionality.
"""

from .api_key_auth import APIKeyAuth, api_key_auth
from .middleware import AuthMiddleware

__all__ = ['APIKeyAuth', 'AuthMiddleware', 'api_key_auth']
