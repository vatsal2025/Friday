"""External API Client module for the Friday AI Trading System.

This module provides the base ExternalApiClient class and protocol-specific
implementations for connecting to external systems.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
from enum import Enum, auto
import requests
import json
import time
import logging
import asyncio
import websockets
import aiohttp
from abc import ABC, abstractmethod
from urllib.parse import urljoin

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.error import FridayError, ErrorSeverity, ErrorCode

# Create logger
logger = get_logger(__name__)


class ApiProtocol(Enum):
    """Enum for API protocols."""
    REST = auto()
    GRAPHQL = auto()
    WEBSOCKET = auto()
    GRPC = auto()
    FIX = auto()


class ApiError(FridayError):
    """Exception raised for errors in API communication.

    Attributes:
        message: Explanation of the error.
        system_id: The external system ID where the error occurred.
        status_code: HTTP status code or other protocol-specific error code.
        details: Additional details about the error.
    """

    def __init__(
        self,
        message: str,
        system_id: str = None,
        status_code: Optional[int] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Any = None,
        cause: Optional[Exception] = None,
        error_code: Optional[ErrorCode] = None
    ):
        self.system_id = system_id
        self.status_code = status_code
        super().__init__(
            message=message,
            severity=severity,
            troubleshooting_guidance=self._generate_guidance(),
            context={"system_id": system_id, "status_code": status_code},
            cause=cause,
            error_code=error_code
        )

    def _generate_guidance(self) -> str:
        """Generate troubleshooting guidance based on the error."""
        if not self.status_code:
            return "Check network connectivity and external system availability."
            
        if 400 <= self.status_code < 500:
            return (
                "This appears to be a client error. Check your request parameters, "
                "authentication credentials, and API permissions."
            )
        elif 500 <= self.status_code < 600:
            return (
                "This appears to be a server error. The external system may be "
                "experiencing issues. Wait and retry later."
            )
        else:
            return "Check the API documentation for the specific error code."


class ExternalApiClient(ABC):
    """Base class for external API clients.

    This abstract class defines the interface for all external API clients
    and provides common functionality for authentication, request handling,
    and error management.

    Attributes:
        system_id: Unique identifier for the external system.
        config: Configuration for the API client.
        protocol: The API protocol used by this client.
        base_url: Base URL for API requests.
        headers: Default headers for API requests.
        auth_manager: Authentication manager for handling credentials.
        rate_limiter: Rate limiter for throttling requests.
    """

    def __init__(
        self,
        system_id: str,
        config: Dict[str, Any],
        protocol: ApiProtocol,
        auth_manager: Optional[Any] = None,
    ):
        """Initialize an external API client.

        Args:
            system_id: Unique identifier for the external system.
            config: Configuration for the API client.
            protocol: The API protocol used by this client.
            auth_manager: Authentication manager for handling credentials.
        """
        self.system_id = system_id
        self.config = config
        self.protocol = protocol
        self.base_url = config.get("connection", {}).get("base_url", "")
        self.headers = {"Content-Type": "application/json"}
        self.auth_manager = auth_manager
        
        # Configure rate limiting
        throttling_config = config.get("throttling", {})
        self.max_requests_per_second = throttling_config.get("max_requests_per_second", 5)
        self.max_requests_per_minute = throttling_config.get("max_requests_per_minute", 100)
        self.retry_attempts = throttling_config.get("retry_attempts", 3)
        self.retry_delay = throttling_config.get("retry_delay", 1.0)
        
        # Request tracking for rate limiting
        self._request_timestamps: List[float] = []
        
        # Initialize connection
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the API client connection.

        This method is called during initialization to set up any necessary
        resources or connections.
        """
        logger.info(f"Initializing API client for {self.system_id}")
        # Implement in subclasses if needed
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the external system.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the external system.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the client is connected to the external system.

        Returns:
            bool: True if connected, False otherwise.
        """
        pass

    def _check_rate_limit(self) -> bool:
        """Check if the request would exceed rate limits.

        Returns:
            bool: True if the request can proceed, False if it would exceed limits.
        """
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [ts for ts in self._request_timestamps 
                                  if current_time - ts < 60]
        
        # Check per-minute limit
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            return False
        
        # Check per-second limit
        recent_requests = [ts for ts in self._request_timestamps 
                          if current_time - ts < 1]
        if len(recent_requests) >= self.max_requests_per_second:
            return False
        
        # Request can proceed
        self._request_timestamps.append(current_time)
        return True

    def _wait_for_rate_limit(self) -> None:
        """Wait until the request can proceed within rate limits."""
        while not self._check_rate_limit():
            time.sleep(0.1)

    def _handle_response_error(self, response: requests.Response) -> None:
        """Handle error responses from the API.

        Args:
            response: The API response object.

        Raises:
            ApiError: If the response indicates an error.
        """
        try:
            error_data = response.json()
        except (ValueError, json.JSONDecodeError):
            error_data = {"text": response.text}
            
        raise ApiError(
            message=f"API error: {response.status_code} - {response.reason}",
            system_id=self.system_id,
            status_code=response.status_code,
            details=error_data
        )

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the API client.

        Returns:
            Dict[str, Any]: Status information including connection state,
                            rate limit information, and other metrics.
        """
        return {
            "system_id": self.system_id,
            "connected": self.is_connected(),
            "protocol": self.protocol.name,
            "rate_limit": {
                "max_requests_per_second": self.max_requests_per_second,
                "max_requests_per_minute": self.max_requests_per_minute,
                "current_requests_last_minute": len(self._request_timestamps),
            }
        }


class RestApiClient(ExternalApiClient):
    """REST API client for external systems.

    This class implements the ExternalApiClient interface for REST APIs,
    providing methods for making HTTP requests with proper error handling,
    retries, and rate limiting.
    """

    def __init__(
        self,
        system_id: str,
        config: Dict[str, Any],
        auth_manager: Optional[Any] = None,
    ):
        """Initialize a REST API client.

        Args:
            system_id: Unique identifier for the external system.
            config: Configuration for the API client.
            auth_manager: Authentication manager for handling credentials.
        """
        super().__init__(system_id, config, ApiProtocol.REST, auth_manager)
        self.session = requests.Session()
        
        # Add default headers from config
        headers = config.get("connection", {}).get("headers", {})
        if headers:
            self.session.headers.update(headers)

    def connect(self) -> bool:
        """Connect to the external system.

        For REST APIs, this method validates the connection by making a test request
        if a health check endpoint is specified in the configuration.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        health_check_endpoint = self.config.get("connection", {}).get("health_check_endpoint")
        if not health_check_endpoint:
            # No health check endpoint specified, assume connected
            return True
            
        try:
            response = self.request("GET", health_check_endpoint)
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Failed to connect to {self.system_id}: {str(e)}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the external system.

        For REST APIs, this method closes the session.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        try:
            self.session.close()
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from {self.system_id}: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if the client is connected to the external system.

        For REST APIs, this method checks if the session is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.session is not None

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        verify: bool = True,
    ) -> requests.Response:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.).
            endpoint: API endpoint path.
            params: Query parameters.
            data: Request body data.
            headers: Additional headers.
            timeout: Request timeout in seconds.
            verify: Whether to verify SSL certificates.

        Returns:
            requests.Response: The API response.

        Raises:
            ApiError: If the request fails or returns an error response.
        """
        # Apply rate limiting
        self._wait_for_rate_limit()
        
        # Prepare request
        url = urljoin(self.base_url, endpoint)
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
            
        # Apply authentication if available
        if self.auth_manager:
            auth_headers = self.auth_manager.get_auth_headers(self.system_id)
            if auth_headers:
                request_headers.update(auth_headers)
        
        # Convert data to JSON if it's a dict
        json_data = None
        if isinstance(data, dict):
            json_data = data
            data = None
            
        # Make request with retries
        for attempt in range(self.retry_attempts + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    timeout=timeout or 30,
                    verify=verify
                )
                
                # Check for successful response
                if response.status_code < 400:
                    return response
                    
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay * (attempt + 1)))
                    logger.warning(f"Rate limited by {self.system_id}, retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                    
                # Handle other errors
                self._handle_response_error(response)
                
            except (requests.RequestException, ConnectionError) as e:
                if attempt < self.retry_attempts:
                    retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request to {self.system_id} failed, retrying in {retry_delay} seconds: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    raise ApiError(
                        message=f"Request failed after {self.retry_attempts} attempts",
                        system_id=self.system_id,
                        cause=e
                    )
        
        # This should not be reached due to the error handling above
        raise ApiError(
            message="Request failed with unknown error",
            system_id=self.system_id
        )

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make a GET request to the API.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        response = self.request("GET", endpoint, params=params, **kwargs)
        return response.json()

    def post(self, endpoint: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Make a POST request to the API.

        Args:
            endpoint: API endpoint path.
            data: Request body data.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        response = self.request("POST", endpoint, data=data, **kwargs)
        return response.json()

    def put(self, endpoint: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Make a PUT request to the API.

        Args:
            endpoint: API endpoint path.
            data: Request body data.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        response = self.request("PUT", endpoint, data=data, **kwargs)
        return response.json()

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a DELETE request to the API.

        Args:
            endpoint: API endpoint path.
            **kwargs: Additional arguments to pass to the request method.

        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        response = self.request("DELETE", endpoint, **kwargs)
        return response.json()


class WebSocketApiClient(ExternalApiClient):
    """WebSocket API client for external systems.

    This class implements the ExternalApiClient interface for WebSocket APIs,
    providing methods for establishing WebSocket connections and handling
    real-time data streams.
    """

    def __init__(
        self,
        system_id: str,
        config: Dict[str, Any],
        auth_manager: Optional[Any] = None,
    ):
        """Initialize a WebSocket API client.

        Args:
            system_id: Unique identifier for the external system.
            config: Configuration for the API client.
            auth_manager: Authentication manager for handling credentials.
        """
        super().__init__(system_id, config, ApiProtocol.WEBSOCKET, auth_manager)
        self.ws_url = config.get("connection", {}).get("ws_url", "")
        self.ws_connection = None
        self.ws_connected = False
        self.message_handlers: Dict[str, Callable] = {}
        self._ws_task = None
        self._ws_loop = None

    async def _connect_async(self) -> bool:
        """Establish a WebSocket connection asynchronously.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            # Apply authentication if available
            headers = {}
            if self.auth_manager:
                auth_headers = self.auth_manager.get_auth_headers(self.system_id)
                if auth_headers:
                    headers.update(auth_headers)
            
            # Connect to WebSocket
            self.ws_connection = await websockets.connect(self.ws_url, extra_headers=headers)
            self.ws_connected = True
            
            # Start message handler
            self._ws_task = asyncio.create_task(self._message_handler())
            
            logger.info(f"Connected to WebSocket for {self.system_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket for {self.system_id}: {str(e)}")
            self.ws_connected = False
            return False

    async def _disconnect_async(self) -> bool:
        """Close the WebSocket connection asynchronously.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        try:
            if self._ws_task:
                self._ws_task.cancel()
                self._ws_task = None
                
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
                
            self.ws_connected = False
            logger.info(f"Disconnected from WebSocket for {self.system_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from WebSocket for {self.system_id}: {str(e)}")
            return False

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.ws_connection:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Determine message type
                    message_type = data.get("type", "default")
                    
                    # Call appropriate handler
                    if message_type in self.message_handlers:
                        await self.message_handlers[message_type](data)
                    elif "default" in self.message_handlers:
                        await self.message_handlers["default"](data)
                    else:
                        logger.warning(f"No handler for message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message from {self.system_id}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message from {self.system_id}: {str(e)}")
                    
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.error(f"WebSocket connection error for {self.system_id}: {str(e)}")
            self.ws_connected = False

    def connect(self) -> bool:
        """Connect to the WebSocket API.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        # Create a new event loop if needed
        if self._ws_loop is None:
            self._ws_loop = asyncio.new_event_loop()
            
        # Run the async connect method in the event loop
        return self._ws_loop.run_until_complete(self._connect_async())

    def disconnect(self) -> bool:
        """Disconnect from the WebSocket API.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        if self._ws_loop is None:
            return False
            
        # Run the async disconnect method in the event loop
        return self._ws_loop.run_until_complete(self._disconnect_async())

    def is_connected(self) -> bool:
        """Check if the client is connected to the WebSocket API.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.ws_connected

    async def send_message_async(self, message: Dict[str, Any]) -> bool:
        """Send a message to the WebSocket API asynchronously.

        Args:
            message: The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        if not self.is_connected():
            logger.error(f"Cannot send message: WebSocket for {self.system_id} is not connected")
            return False
            
        try:
            await self.ws_connection.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {self.system_id}: {str(e)}")
            return False

    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to the WebSocket API.

        Args:
            message: The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        if self._ws_loop is None:
            return False
            
        # Run the async send method in the event loop
        return self._ws_loop.run_until_complete(self.send_message_async(message))

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type.

        Args:
            message_type: The type of message to handle.
            handler: The async function to call when a message of this type is received.
        """
        self.message_handlers[message_type] = handler


class GraphQLApiClient(RestApiClient):
    """GraphQL API client for external systems.

    This class extends the RestApiClient to provide specialized handling
    for GraphQL queries and mutations.
    """

    def __init__(
        self,
        system_id: str,
        config: Dict[str, Any],
        auth_manager: Optional[Any] = None,
    ):
        """Initialize a GraphQL API client.

        Args:
            system_id: Unique identifier for the external system.
            config: Configuration for the API client.
            auth_manager: Authentication manager for handling credentials.
        """
        # Initialize with REST protocol but override internally
        super().__init__(system_id, config, auth_manager)
        self.protocol = ApiProtocol.GRAPHQL
        self.graphql_endpoint = config.get("connection", {}).get("graphql_endpoint", "/graphql")

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query.

        Args:
            query: The GraphQL query string.
            variables: Variables for the query.

        Returns:
            Dict[str, Any]: The query result.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
            
        response = self.request("POST", self.graphql_endpoint, data=payload)
        result = response.json()
        
        # Check for GraphQL errors
        if "errors" in result:
            raise ApiError(
                message="GraphQL query error",
                system_id=self.system_id,
                details=result["errors"]
            )
            
        return result["data"]

    def mutation(self, mutation: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL mutation.

        Args:
            mutation: The GraphQL mutation string.
            variables: Variables for the mutation.

        Returns:
            Dict[str, Any]: The mutation result.
        """
        # GraphQL mutations use the same endpoint and format as queries
        return self.query(mutation, variables)