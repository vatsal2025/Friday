"""API Gateway for the Friday AI Trading System.

This module provides a FastAPI application with authentication middleware and rate limiting.
"""

import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import jwt
from pydantic import BaseModel

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.security import generate_api_key, generate_api_secret, hash_password, verify_password

# Create logger
logger = get_logger(__name__)

# Load API configuration
API_CONFIG = get_config("API_CONFIG")

# JWT settings
JWT_SECRET_KEY = API_CONFIG.get("jwt_secret_key", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = API_CONFIG.get("jwt_expiry_minutes", 60)

# Rate limiting settings
RATE_LIMIT_ENABLED = API_CONFIG.get("rate_limit_enabled", True)
RATE_LIMIT_WINDOW = API_CONFIG.get("rate_limit_window", 60)  # seconds
RATE_LIMIT_MAX_REQUESTS = API_CONFIG.get("rate_limit_max_requests", 100)  # requests per window

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# In-memory storage for rate limiting
# In a production environment, this should be replaced with Redis or similar
rate_limit_store: Dict[str, Dict[str, Union[int, float]]] = {}

# In-memory API key store
# In a production environment, this should be stored in a database
api_keys: Dict[str, Dict[str, Any]] = {}


class Token(BaseModel):
    """Token model for authentication responses."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: List[str] = []


class UserInDB(User):
    """User model with hashed password for database storage."""
    hashed_password: str
    salt: bytes


class APIKeyInfo(BaseModel):
    """API key information model."""
    key: str
    name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None


def create_api_gateway() -> FastAPI:
    """Create the API gateway application.

    Returns:
        FastAPI: The API gateway application.
    """
    app = FastAPI(
        title="Friday AI Trading System API",
        description="API for the Friday AI Trading System",
        version="1.0.0",
        docs_url=API_CONFIG.get("api_docs_path", "/docs") if API_CONFIG.get("api_docs_enabled", True) else None,
    )

    # Add CORS middleware
    if API_CONFIG.get("cors_enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=API_CONFIG.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add rate limiting middleware
    if RATE_LIMIT_ENABLED:
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next: Callable):
            """Rate limiting middleware.

            Args:
                request: The request object.
                call_next: The next middleware or route handler.

            Returns:
                The response from the next middleware or route handler.
            """
            # Get client identifier (IP address or API key)
            client_id = request.headers.get("X-API-Key", request.client.host)

            # Check rate limit
            current_time = time.time()
            if client_id in rate_limit_store:
                last_request_time = rate_limit_store[client_id].get("last_request_time", 0)
                request_count = rate_limit_store[client_id].get("request_count", 0)

                # Reset counter if window has passed
                if current_time - last_request_time > RATE_LIMIT_WINDOW:
                    request_count = 0

                # Check if rate limit exceeded
                if request_count >= RATE_LIMIT_MAX_REQUESTS:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "Rate limit exceeded. Please try again later."},
                    )

                # Update request count
                rate_limit_store[client_id] = {
                    "last_request_time": current_time,
                    "request_count": request_count + 1,
                }
            else:
                # First request from this client
                rate_limit_store[client_id] = {
                    "last_request_time": current_time,
                    "request_count": 1,
                }

            # Add rate limit headers to response
            response = await call_next(request)
            response.headers["X-Rate-Limit-Limit"] = str(RATE_LIMIT_MAX_REQUESTS)
            response.headers["X-Rate-Limit-Remaining"] = str(
                RATE_LIMIT_MAX_REQUESTS - rate_limit_store[client_id]["request_count"]
            )
            response.headers["X-Rate-Limit-Reset"] = str(
                int(rate_limit_store[client_id]["last_request_time"] + RATE_LIMIT_WINDOW)
            )

            return response

    # Add request logging middleware
    if API_CONFIG.get("log_requests", True):
        @app.middleware("http")
        async def log_requests_middleware(request: Request, call_next: Callable):
            """Request logging middleware.

            Args:
                request: The request object.
                call_next: The next middleware or route handler.

            Returns:
                The response from the next middleware or route handler.
            """
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"Request: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Process Time: {process_time:.4f}s"
            )
            return response

    return app


def get_api_key_from_header(api_key: str = Depends(API_KEY_HEADER)) -> Dict[str, Any]:
    """Get API key information from header.

    Args:
        api_key: The API key from the header.

    Returns:
        Dict[str, Any]: The API key information.

    Raises:
        HTTPException: If the API key is invalid or expired.
    """
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    api_key_info = api_keys[api_key]

    # Check if API key is expired
    if api_key_info.get("expires_at") and api_key_info["expires_at"] < datetime.now():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key_info


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: The data to encode in the token.
        expires_delta: The expiration time delta. Defaults to None.

    Returns:
        str: The encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token.

    Args:
        token: The token to verify.

    Returns:
        Dict[str, Any]: The decoded token payload.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_api_key_pair() -> Dict[str, str]:
    """Generate an API key and secret pair.

    Returns:
        Dict[str, str]: The API key and secret.
    """
    api_key = generate_api_key()
    api_secret = generate_api_secret()
    return {"api_key": api_key, "api_secret": api_secret}


def register_api_key(name: str, scopes: List[str], expires_in_days: Optional[int] = None) -> APIKeyInfo:
    """Register a new API key.

    Args:
        name: The name of the API key.
        scopes: The scopes of the API key.
        expires_in_days: The number of days until the API key expires. Defaults to None.

    Returns:
        APIKeyInfo: The API key information.
    """
    key_pair = generate_api_key_pair()
    api_key = key_pair["api_key"]
    api_secret = key_pair["api_secret"]

    created_at = datetime.now()
    expires_at = None
    if expires_in_days is not None:
        expires_at = created_at + timedelta(days=expires_in_days)

    api_key_info = {
        "key": api_key,
        "secret": api_secret,
        "name": name,
        "scopes": scopes,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    # Store API key info
    api_keys[api_key] = api_key_info

    return APIKeyInfo(
        key=api_key,
        name=name,
        scopes=scopes,
        created_at=created_at,
        expires_at=expires_at,
    )


def revoke_api_key(api_key: str) -> None:
    """Revoke an API key.

    Args:
        api_key: The API key to revoke.

    Raises:
        ValueError: If the API key does not exist.
    """
    if api_key not in api_keys:
        raise ValueError(f"API key {api_key} does not exist")

    del api_keys[api_key]