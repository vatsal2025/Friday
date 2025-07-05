"""Mock Service API for the Friday AI Trading System.

This module provides a REST API for interacting with mock services.
"""

from typing import Any, Dict, List, Optional, Union, Type
import json
import logging
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Body, Query, Path
from pydantic import BaseModel, Field

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.integration.external_system_registry import SystemType, SystemStatus
from src.integration.mock.mock_registry import MockServiceRegistry, MockRegistryError
from src.integration.mock.mock_config import MockServiceConfig, MockConfigError
from src.integration.mock.mock_service import MockServiceError

# Create logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/mock", tags=["mock"])


# Pydantic models for request and response validation
class ServiceTypeEnum(str, Enum):
    """Enum for service types."""
    BROKER = "broker"
    MARKET_DATA = "market_data"
    FINANCIAL_DATA = "financial_data"
    ANALYTICS = "analytics"


class ServiceStatusEnum(str, Enum):
    """Enum for service statuses."""
    RUNNING = "running"
    STOPPED = "stopped"


class MockServiceCreate(BaseModel):
    """Model for creating a mock service."""
    service_id: str = Field(..., description="Unique identifier for the mock service")
    name: str = Field(..., description="Human-readable name of the mock service")
    service_type: ServiceTypeEnum = Field(..., description="Type of the mock service")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the mock service")


class MockServiceUpdate(BaseModel):
    """Model for updating a mock service."""
    name: Optional[str] = Field(None, description="Human-readable name of the mock service")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the mock service")


class MockServiceStatus(BaseModel):
    """Model for mock service status."""
    service_id: str = Field(..., description="Unique identifier for the mock service")
    name: str = Field(..., description="Human-readable name of the mock service")
    type: str = Field(..., description="Type of the mock service")
    status: ServiceStatusEnum = Field(..., description="Status of the mock service")
    endpoints: List[str] = Field(..., description="List of endpoints supported by the mock service")


class MockServiceRequest(BaseModel):
    """Model for a request to a mock service."""
    endpoint: str = Field(..., description="The endpoint to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the request")


class MockServiceResponse(BaseModel):
    """Model for a response from a mock service."""
    service_id: str = Field(..., description="Unique identifier for the mock service")
    endpoint: str = Field(..., description="The endpoint that was called")
    result: Any = Field(..., description="The result of the request")


class ErrorResponse(BaseModel):
    """Model for an error response."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")


# Helper functions
def _get_system_type(service_type: ServiceTypeEnum) -> SystemType:
    """Convert a ServiceTypeEnum to a SystemType.

    Args:
        service_type: The service type enum.

    Returns:
        SystemType: The corresponding system type.
    """
    if service_type == ServiceTypeEnum.BROKER:
        return SystemType.BROKER
    elif service_type == ServiceTypeEnum.MARKET_DATA:
        return SystemType.MARKET_DATA
    elif service_type == ServiceTypeEnum.FINANCIAL_DATA:
        return SystemType.FINANCIAL_DATA
    elif service_type == ServiceTypeEnum.ANALYTICS:
        return SystemType.ANALYTICS
    else:
        raise ValueError(f"Unsupported service type: {service_type}")


# API endpoints
@router.get("/services", response_model=List[MockServiceStatus], summary="Get all mock services")
async def get_services(
    service_type: Optional[ServiceTypeEnum] = Query(None, description="Filter by service type")
) -> List[MockServiceStatus]:
    """Get all mock services.

    Args:
        service_type: Filter by service type.

    Returns:
        List[MockServiceStatus]: List of mock service statuses.
    """
    try:
        system_type = _get_system_type(service_type) if service_type else None
        return MockServiceRegistry.get_all_service_statuses(system_type)
    except Exception as e:
        logger.error(f"Failed to get mock services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{service_id}", response_model=MockServiceStatus, summary="Get a mock service")
async def get_service(service_id: str = Path(..., description="The ID of the service to get")) -> MockServiceStatus:
    """Get a mock service.

    Args:
        service_id: The ID of the service to get.

    Returns:
        MockServiceStatus: The mock service status.
    """
    try:
        return MockServiceRegistry.get_service_status(service_id)
    except MockRegistryError as e:
        logger.error(f"Failed to get mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services", response_model=MockServiceStatus, summary="Create a mock service")
async def create_service(service: MockServiceCreate) -> MockServiceStatus:
    """Create a mock service.

    Args:
        service: The service to create.

    Returns:
        MockServiceStatus: The created mock service status.
    """
    try:
        # Convert service type
        system_type = _get_system_type(service.service_type)
        
        # Create config
        config = {
            "service_id": service.service_id,
            "name": service.name,
            "service_type": system_type.name,
            **service.config
        }
        
        # Create service
        from src.integration.mock.mock_service import create_mock_service
        service_id = create_mock_service(service.service_id, system_type, config)
        
        # Return status
        return MockServiceRegistry.get_service_status(service_id)
    except MockServiceError as e:
        logger.error(f"Failed to create mock service: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create mock service: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/services/{service_id}", response_model=MockServiceStatus, summary="Update a mock service")
async def update_service(
    service_id: str = Path(..., description="The ID of the service to update"),
    update: MockServiceUpdate = Body(...)
) -> MockServiceStatus:
    """Update a mock service.

    Args:
        service_id: The ID of the service to update.
        update: The update to apply.

    Returns:
        MockServiceStatus: The updated mock service status.
    """
    try:
        # Get the service
        service = MockServiceRegistry.get_service(service_id)
        
        # Update the service
        if update.name is not None:
            service.name = update.name
            
        if update.config is not None:
            service.config.update(update.config)
            
        # Return status
        return MockServiceRegistry.get_service_status(service_id)
    except MockRegistryError as e:
        logger.error(f"Failed to update mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/services/{service_id}", response_model=Dict[str, str], summary="Delete a mock service")
async def delete_service(service_id: str = Path(..., description="The ID of the service to delete")) -> Dict[str, str]:
    """Delete a mock service.

    Args:
        service_id: The ID of the service to delete.

    Returns:
        Dict[str, str]: A message indicating the service was deleted.
    """
    try:
        MockServiceRegistry.unregister_service(service_id)
        return {"message": f"Service '{service_id}' deleted"}
    except MockRegistryError as e:
        logger.error(f"Failed to delete mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/start", response_model=MockServiceStatus, summary="Start a mock service")
async def start_service(service_id: str = Path(..., description="The ID of the service to start")) -> MockServiceStatus:
    """Start a mock service.

    Args:
        service_id: The ID of the service to start.

    Returns:
        MockServiceStatus: The mock service status.
    """
    try:
        MockServiceRegistry.start_service(service_id)
        return MockServiceRegistry.get_service_status(service_id)
    except MockRegistryError as e:
        logger.error(f"Failed to start mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/stop", response_model=MockServiceStatus, summary="Stop a mock service")
async def stop_service(service_id: str = Path(..., description="The ID of the service to stop")) -> MockServiceStatus:
    """Stop a mock service.

    Args:
        service_id: The ID of the service to stop.

    Returns:
        MockServiceStatus: The mock service status.
    """
    try:
        MockServiceRegistry.stop_service(service_id)
        return MockServiceRegistry.get_service_status(service_id)
    except MockRegistryError as e:
        logger.error(f"Failed to stop mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/request", response_model=MockServiceResponse, summary="Send a request to a mock service")
async def send_request(
    service_id: str = Path(..., description="The ID of the service to send the request to"),
    request: MockServiceRequest = Body(...)
) -> MockServiceResponse:
    """Send a request to a mock service.

    Args:
        service_id: The ID of the service to send the request to.
        request: The request to send.

    Returns:
        MockServiceResponse: The response from the mock service.
    """
    try:
        result = MockServiceRegistry.handle_request(service_id, request.endpoint, request.params)
        return MockServiceResponse(
            service_id=service_id,
            endpoint=request.endpoint,
            result=result
        )
    except MockRegistryError as e:
        logger.error(f"Failed to send request to mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except MockServiceError as e:
        logger.error(f"Mock service '{service_id}' returned an error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to send request to mock service '{service_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/start-all", response_model=Dict[str, int], summary="Start all mock services")
async def start_all_services(
    service_type: Optional[ServiceTypeEnum] = Query(None, description="Filter by service type")
) -> Dict[str, int]:
    """Start all mock services.

    Args:
        service_type: Filter by service type.

    Returns:
        Dict[str, int]: The number of services started.
    """
    try:
        system_type = _get_system_type(service_type) if service_type else None
        count = MockServiceRegistry.start_all_services(system_type)
        return {"started": count}
    except Exception as e:
        logger.error(f"Failed to start all mock services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/stop-all", response_model=Dict[str, int], summary="Stop all mock services")
async def stop_all_services(
    service_type: Optional[ServiceTypeEnum] = Query(None, description="Filter by service type")
) -> Dict[str, int]:
    """Stop all mock services.

    Args:
        service_type: Filter by service type.

    Returns:
        Dict[str, int]: The number of services stopped.
    """
    try:
        system_type = _get_system_type(service_type) if service_type else None
        count = MockServiceRegistry.stop_all_services(system_type)
        return {"stopped": count}
    except Exception as e:
        logger.error(f"Failed to stop all mock services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/clear", response_model=Dict[str, int], summary="Clear all mock services")
async def clear_services() -> Dict[str, int]:
    """Clear all mock services.

    Returns:
        Dict[str, int]: The number of services cleared.
    """
    try:
        count = MockServiceRegistry.clear_registry()
        return {"cleared": count}
    except Exception as e:
        logger.error(f"Failed to clear mock services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-config", response_model=Dict[str, List[str]], summary="Load mock services from a configuration file")
async def load_config(file_path: str = Body(..., embed=True)) -> Dict[str, List[str]]:
    """Load mock services from a configuration file.

    Args:
        file_path: Path to the configuration file.

    Returns:
        Dict[str, List[str]]: The IDs of the created services.
    """
    try:
        config = MockServiceConfig.load_config_from_file(file_path)
        service_id = MockServiceConfig.create_mock_service_from_config(config)
        return {"services": [service_id]}
    except (MockConfigError, MockServiceError) as e:
        logger.error(f"Failed to load mock services from configuration file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load mock services from configuration file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-configs", response_model=Dict[str, List[str]], summary="Load mock services from a directory of configuration files")
async def load_configs(directory_path: str = Body(..., embed=True)) -> Dict[str, List[str]]:
    """Load mock services from a directory of configuration files.

    Args:
        directory_path: Path to the directory containing configuration files.

    Returns:
        Dict[str, List[str]]: The IDs of the created services.
    """
    try:
        configs = MockServiceConfig.load_configs_from_directory(directory_path)
        service_ids = MockServiceConfig.create_mock_services_from_configs(configs)
        return {"services": service_ids}
    except MockConfigError as e:
        logger.error(f"Failed to load mock services from directory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load mock services from directory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))