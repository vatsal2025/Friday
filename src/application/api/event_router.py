"""Event router for the API Gateway.

This module provides endpoints for event-related operations, including SSE (Server-Sent Events)
for real-time event streaming.
"""

from typing import Dict, List, Optional, Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from src.infrastructure.logging import get_logger
from src.infrastructure.event.event_system import EventSystem, Event
from src.application.api.api_gateway import get_api_key_from_header

# Create logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/events", tags=["events"])

# Get event system instance
event_system = EventSystem()


class EventFilter(BaseModel):
    """Event filter model."""
    event_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None


class EventResponse(BaseModel):
    """Event response model."""
    id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: str
    source: str


async def event_generator(
    request: Request, filter_params: EventFilter
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate events for SSE streaming.

    Args:
        request: The request object.
        filter_params: The event filter parameters.

    Yields:
        Dict[str, Any]: The event data.
    """
    # Create a queue for events
    import asyncio
    from datetime import datetime
    import json
    import uuid

    queue = asyncio.Queue()
    client_id = str(uuid.uuid4())

    # Define event handler
    async def handle_event(event: Event) -> None:
        # Apply filters
        if filter_params.event_types and event.event_type not in filter_params.event_types:
            return
        if filter_params.sources and event.source not in filter_params.sources:
            return
        if filter_params.start_timestamp:
            start_time = datetime.fromisoformat(filter_params.start_timestamp)
            if event.timestamp < start_time:
                return
        if filter_params.end_timestamp:
            end_time = datetime.fromisoformat(filter_params.end_timestamp)
            if event.timestamp > end_time:
                return

        # Put event in queue
        await queue.put({
            "id": event.id,
            "event": event.event_type,
            "data": json.dumps(event.data),
            "retry": 3000,  # Retry after 3 seconds
        })

    # Register event handler
    event_types = filter_params.event_types or ["*"]
    for event_type in event_types:
        event_system.register_handler(handle_event, event_type)

    try:
        # Keep connection alive with ping events
        ping_task = None

        async def send_ping() -> None:
            while True:
                if await request.is_disconnected():
                    break
                await queue.put({"event": "ping", "data": json.dumps({"time": datetime.now().isoformat()})})
                await asyncio.sleep(30)  # Send ping every 30 seconds

        ping_task = asyncio.create_task(send_ping())

        # Yield events from queue
        while True:
            if await request.is_disconnected():
                break

            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield event_data
                queue.task_done()
            except asyncio.TimeoutError:
                # No event received, continue waiting
                pass
    finally:
        # Clean up
        if ping_task:
            ping_task.cancel()

        # Unregister event handler
        for event_type in event_types:
            event_system.unregister_handler(handle_event, event_type)

        logger.info(f"Client {client_id} disconnected from event stream")


@router.get("/stream")
async def stream_events(
    request: Request,
    event_types: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> EventSourceResponse:
    """Stream events using Server-Sent Events (SSE).

    Args:
        request: The request object.
        event_types: Filter by event types.
        sources: Filter by event sources.
        start_timestamp: Filter by start timestamp (ISO format).
        end_timestamp: Filter by end timestamp (ISO format).
        api_key_info: The API key information.

    Returns:
        EventSourceResponse: The SSE response.

    Raises:
        HTTPException: If there is an error streaming events.
    """
    # Check if API key has events scope
    if "events" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have events scope",
        )

    try:
        # Create filter parameters
        filter_params = EventFilter(
            event_types=event_types,
            sources=sources,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

        # Return SSE response
        return EventSourceResponse(event_generator(request, filter_params))
    except Exception as e:
        logger.error(f"Error streaming events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming events: {str(e)}",
        )


@router.post("/emit", response_model=EventResponse)
async def emit_event(
    event_type: str,
    data: Dict[str, Any],
    source: str = "api",
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> Dict[str, Any]:
    """Emit an event.

    Args:
        event_type: The event type.
        data: The event data.
        source: The event source.
        api_key_info: The API key information.

    Returns:
        Dict[str, Any]: The emitted event.

    Raises:
        HTTPException: If there is an error emitting the event.
    """
    # Check if API key has events scope
    if "events" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have events scope",
        )

    try:
        # Create event
        event = Event(event_type=event_type, data=data, source=source)

        # Emit event
        event_system.emit(event)

        # Return event
        return {
            "id": event.id,
            "event_type": event.event_type,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
        }
    except Exception as e:
        logger.error(f"Error emitting event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error emitting event: {str(e)}",
        )


@router.get("/history", response_model=List[EventResponse])
async def get_event_history(
    event_filter: EventFilter,
    limit: int = 10,
    offset: int = 0,
    api_key_info: Dict[str, Any] = Depends(get_api_key_from_header),
) -> List[Dict[str, Any]]:
    """Get event history.

    Args:
        event_filter: The event filter.
        limit: Maximum number of events to return.
        offset: Offset for pagination.
        api_key_info: The API key information.

    Returns:
        List[Dict[str, Any]]: The event history.

    Raises:
        HTTPException: If there is an error fetching event history.
    """
    # Check if API key has events scope
    if "events" not in api_key_info["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have events scope",
        )

    try:
        # In a real implementation, we would fetch the event history from a database
        # For now, return a placeholder response
        import uuid
        from datetime import datetime, timedelta

        # Generate some placeholder events
        events = []
        for i in range(limit):
            event_id = str(uuid.uuid4())
            timestamp = (datetime.now() - timedelta(minutes=i)).isoformat()
            event_type = event_filter.event_types[0] if event_filter.event_types else "system.info"
            source = event_filter.sources[0] if event_filter.sources else "api"

            events.append({
                "id": event_id,
                "event_type": event_type,
                "data": {"message": f"Placeholder event {i + 1}"},
                "timestamp": timestamp,
                "source": source,
            })

        return events
    except Exception as e:
        logger.error(f"Error fetching event history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching event history: {str(e)}",
        )