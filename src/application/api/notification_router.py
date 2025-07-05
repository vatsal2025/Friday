"""Notification router for the API Gateway.

This module provides endpoints for notification-related operations.
"""

from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from src.infrastructure.logging import get_logger
from src.services.notification.notification_service import NotificationService
from src.application.api.api_gateway import get_api_key_from_header

# Create logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/notifications", tags=["notifications"])

# Get notification service instance
notification_service = NotificationService()


class NotificationRequest(BaseModel):
    """Notification request model."""
    message: str
    notification_type: str = Field(..., description="trade, alert, system")
    level: str = Field("info", description="info, warning, error, critical")
    channels: List[str] = Field(["all"], description="email, telegram, push, dashboard, all")
    metadata: Optional[Dict[str, Any]] = None


class NotificationResponse(BaseModel):
    """Notification response model."""
    notification_id: str
    status: str
    message: str
    notification_type: str
    level: str
    channels: List[str]
    timestamp: str


class NotificationPreference(BaseModel):
    """Notification preference model."""
    channel: str
    enabled: bool
    notification_types: List[str] = ["trade", "alert", "system"]
    min_level: str = "info"
    metadata: Optional[Dict[str, Any]] = None


@router.post("/send", response_model=NotificationResponse)
async def send_notification(
    notification_request: NotificationRequest,
    background_tasks: BackgroundTasks,
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> Dict[str, Any]:
    """Send a notification.

    Args:
        notification_request: The notification request.
        background_tasks: Background tasks.
        api_key_info: The API key information.

    Returns:
        Dict[str, Any]: The notification response.

    Raises:
        HTTPException: If there is an error sending the notification.
    """
    # Check if API key has notification scope
    if "notification" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have notification scope",
        )

    try:
        # Prepare notification data
        notification_data = notification_request.dict()
        notification_data["api_key_name"] = api_key_info["name"]

        # Use background task to send notification
        if notification_request.notification_type == "trade":
            background_tasks.add_task(
                notification_service.send_trade_notification,
                notification_request.message,
                notification_request.level,
                notification_request.channels,
                notification_request.metadata,
            )
        elif notification_request.notification_type == "alert":
            background_tasks.add_task(
                notification_service.send_alert_notification,
                notification_request.message,
                notification_request.level,
                notification_request.channels,
                notification_request.metadata,
            )
        elif notification_request.notification_type == "system":
            background_tasks.add_task(
                notification_service.send_system_notification,
                notification_request.message,
                notification_request.level,
                notification_request.channels,
                notification_request.metadata,
            )
        else:
            raise ValueError(f"Invalid notification type: {notification_request.notification_type}")

        # Generate a placeholder notification ID
        import uuid
        notification_id = str(uuid.uuid4())

        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()

        return {
            "notification_id": notification_id,
            "status": "SENT",
            "message": notification_request.message,
            "notification_type": notification_request.notification_type,
            "level": notification_request.level,
            "channels": notification_request.channels,
            "timestamp": timestamp,
        }
    except ValueError as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending notification: {str(e)}",
        )


@router.get("/history", response_model=List[NotificationResponse])
async def get_notification_history(
    notification_type: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> List[Dict[str, Any]]:
    """Get notification history.

    Args:
        notification_type: Filter by notification type.
        level: Filter by notification level.
        limit: Maximum number of notifications to return.
        offset: Offset for pagination.
        api_key_info: The API key information.

    Returns:
        List[Dict[str, Any]]: The notification history.

    Raises:
        HTTPException: If there is an error fetching notification history.
    """
    # Check if API key has notification scope
    if "notification" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have notification scope",
        )

    try:
        # In a real implementation, we would fetch the notification history from a database
        # For now, return a placeholder response
        import uuid
        from datetime import datetime, timedelta

        # Generate some placeholder notifications
        notifications = []
        for i in range(limit):
            notification_id = str(uuid.uuid4())
            timestamp = (datetime.now() - timedelta(minutes=i)).isoformat()
            notification_type_value = notification_type or "system"
            level_value = level or "info"

            notifications.append({
                "notification_id": notification_id,
                "status": "SENT",
                "message": f"Placeholder notification {i + 1}",
                "notification_type": notification_type_value,
                "level": level_value,
                "channels": ["dashboard"],
                "timestamp": timestamp,
            })

        return notifications
    except Exception as e:
        logger.error(f"Error fetching notification history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching notification history: {str(e)}",
        )


@router.post("/preferences", response_model=Dict[str, Any])
async def set_notification_preferences(
    preferences: List[NotificationPreference],
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> Dict[str, Any]:
    """Set notification preferences.

    Args:
        preferences: The notification preferences.
        api_key_info: The API key information.

    Returns:
        Dict[str, Any]: A success message.

    Raises:
        HTTPException: If there is an error setting notification preferences.
    """
    # Check if API key has notification scope
    if "notification" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have notification scope",
        )

    try:
        # In a real implementation, we would store the notification preferences
        # For now, return a placeholder response
        return {
            "message": "Notification preferences updated",
            "preferences": [pref.dict() for pref in preferences],
        }
    except Exception as e:
        logger.error(f"Error setting notification preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting notification preferences: {str(e)}",
        )


@router.get("/preferences", response_model=List[NotificationPreference])
async def get_notification_preferences(
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> List[Dict[str, Any]]:
    """Get notification preferences.

    Args:
        api_key_info: The API key information.

    Returns:
        List[Dict[str, Any]]: The notification preferences.

    Raises:
        HTTPException: If there is an error fetching notification preferences.
    """
    # Check if API key has notification scope
    if "notification" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have notification scope",
        )

    try:
        # In a real implementation, we would fetch the notification preferences
        # For now, return a placeholder response
        return [
            {
                "channel": "email",
                "enabled": True,
                "notification_types": ["trade", "alert", "system"],
                "min_level": "info",
                "metadata": {"email": "user@example.com"},
            },
            {
                "channel": "telegram",
                "enabled": True,
                "notification_types": ["trade", "alert"],
                "min_level": "warning",
                "metadata": {"chat_id": "123456789"},
            },
            {
                "channel": "push",
                "enabled": False,
                "notification_types": ["alert", "system"],
                "min_level": "error",
                "metadata": {"device_id": "device123"},
            },
            {
                "channel": "dashboard",
                "enabled": True,
                "notification_types": ["trade", "alert", "system"],
                "min_level": "info",
                "metadata": None,
            },
        ]
    except Exception as e:
        logger.error(f"Error fetching notification preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching notification preferences: {str(e)}",
        )


@router.get("/channels", response_model=List[str])
async def get_available_channels(
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> List[str]:
    """Get available notification channels.

    Args:
        api_key_info: The API key information.

    Returns:
        List[str]: The available notification channels.

    Raises:
        HTTPException: If there is an error fetching available channels.
    """
    # Check if API key has notification scope
    if "notification" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have notification scope",
        )

    try:
        # In a real implementation, we would fetch the available channels
        # For now, return a placeholder response
        return ["email", "telegram", "push", "dashboard"]
    except Exception as e:
        logger.error(f"Error fetching available channels: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching available channels: {str(e)}",
        )