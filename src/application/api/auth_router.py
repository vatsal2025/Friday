"""Authentication router for the API Gateway.

This module provides endpoints for user authentication and API key management.
"""

from datetime import timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.application.api.api_gateway import (
    Token, User, UserInDB, APIKeyInfo,
    create_access_token, get_api_key_from_header,
    register_api_key, revoke_api_key
)
from src.infrastructure.security import hash_password, verify_password

# Create logger
logger = get_logger(__name__)

# Load API configuration
API_CONFIG = get_config("API_CONFIG")
JWT_EXPIRY_MINUTES = API_CONFIG.get("jwt_expiry_minutes", 60)

# OAuth2 password bearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# In-memory user store
# In a production environment, this should be stored in a database
users: Dict[str, Dict[str, Any]] = {}

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


class UserCreate(BaseModel):
    """User creation model."""
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    scopes: List[str] = []


class APIKeyCreate(BaseModel):
    """API key creation model."""
    name: str
    scopes: List[str] = []
    expires_in_days: Optional[int] = None


def get_user(username: str) -> Optional[UserInDB]:
    """Get a user by username.

    Args:
        username: The username to look up.

    Returns:
        Optional[UserInDB]: The user if found, None otherwise.
    """
    if username not in users:
        return None
    user_dict = users[username]
    return UserInDB(**user_dict)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user.

    Args:
        username: The username to authenticate.
        password: The password to verify.

    Returns:
        Optional[UserInDB]: The authenticated user if successful, None otherwise.
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password, user.salt):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from a token.

    Args:
        token: The JWT token.

    Returns:
        User: The current user.

    Raises:
        HTTPException: If the token is invalid or the user does not exist.
    """
    from src.application.api.api_gateway import verify_token

    payload = verify_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        scopes=user.scopes,
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user.

    Args:
        current_user: The current user.

    Returns:
        User: The current active user.

    Raises:
        HTTPException: If the user is disabled.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate) -> User:
    """Register a new user.

    Args:
        user_create: The user creation data.

    Returns:
        User: The created user.

    Raises:
        HTTPException: If the username already exists.
    """
    if user_create.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Hash the password
    hashed_password, salt = hash_password(user_create.password)

    # Create the user
    user_dict = {
        "username": user_create.username,
        "email": user_create.email,
        "full_name": user_create.full_name,
        "hashed_password": hashed_password,
        "salt": salt,
        "disabled": False,
        "scopes": user_create.scopes,
    }
    users[user_create.username] = user_dict

    logger.info(f"User {user_create.username} registered")

    return User(
        username=user_create.username,
        email=user_create.email,
        full_name=user_create.full_name,
        disabled=False,
        scopes=user_create.scopes,
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, str]:
    """Login for access token.

    Args:
        form_data: The login form data.

    Returns:
        Dict[str, str]: The access token and token type.

    Raises:
        HTTPException: If the username or password is incorrect.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=JWT_EXPIRY_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/api-keys", response_model=APIKeyInfo)
async def create_api_key(
    api_key_create: APIKeyCreate, current_user: User = Depends(get_current_active_user)
) -> APIKeyInfo:
    """Create a new API key.

    Args:
        api_key_create: The API key creation data.
        current_user: The current user.

    Returns:
        APIKeyInfo: The created API key information.
    """
    # Check if user has permission to create API keys
    if "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    api_key_info = register_api_key(
        name=api_key_create.name,
        scopes=api_key_create.scopes,
        expires_in_days=api_key_create.expires_in_days,
    )

    logger.info(f"API key {api_key_info.key} created by {current_user.username}")

    return api_key_info


@router.delete("/api-keys/{api_key}")
async def delete_api_key(
    api_key: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, str]:
    """Delete an API key.

    Args:
        api_key: The API key to delete.
        current_user: The current user.

    Returns:
        Dict[str, str]: A success message.

    Raises:
        HTTPException: If the API key does not exist or the user does not have permission.
    """
    # Check if user has permission to delete API keys
    if "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    try:
        revoke_api_key(api_key)
        logger.info(f"API key {api_key} deleted by {current_user.username}")
        return {"message": f"API key {api_key} deleted"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)) -> User:
    """Get the current user.

    Args:
        current_user: The current user.

    Returns:
        User: The current user.
    """
    return current_user


@router.get("/status")
async def check_auth_status(api_key_info: Dict[str, Any] = Depends(get_api_key_from_header)) -> Dict[str, Any]:
    """Check authentication status.

    Args:
        api_key_info: The API key information.

    Returns:
        Dict[str, Any]: The authentication status.
    """
    return {
        "authenticated": True,
        "key_name": api_key_info["name"],
        "scopes": api_key_info["scopes"],
        "created_at": api_key_info["created_at"],
        "expires_at": api_key_info.get("expires_at"),
    }