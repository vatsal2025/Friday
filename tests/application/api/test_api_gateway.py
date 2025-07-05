"""Integration tests for the API Gateway.

This module contains tests to verify that the API Gateway components work together correctly.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.application.api.main import create_app
from src.application.api.api_gateway import register_api_key


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key():
    """Create a test API key."""
    # Register a test API key with all scopes
    api_key_info = register_api_key(
        name="test_api_key",
        scopes=["trading", "market_data", "notification", "events", "broker_auth"],
        expires_in_days=1,
    )
    return api_key_info.key


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version(client):
    """Test the version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_auth_status_with_api_key(client, api_key):
    """Test the authentication status endpoint with a valid API key."""
    response = client.get("/auth/status", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_auth_status_without_api_key(client):
    """Test the authentication status endpoint without an API key."""
    response = client.get("/auth/status")
    assert response.status_code == 401  # Unauthorized


def test_auth_status_with_invalid_api_key(client):
    """Test the authentication status endpoint with an invalid API key."""
    response = client.get("/auth/status", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401  # Unauthorized


def test_place_order_with_api_key(client, api_key):
    """Test the place order endpoint with a valid API key."""
    order_data = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": "Market",
        "side": "Buy",
        "time_in_force": "DAY",
    }
    response = client.post("/broker/orders", json=order_data, headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"
    assert response.json()["status"] == "PENDING"


def test_place_order_without_api_key(client):
    """Test the place order endpoint without an API key."""
    order_data = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": "Market",
        "side": "Buy",
        "time_in_force": "DAY",
    }
    response = client.post("/broker/orders", json=order_data)
    assert response.status_code == 401  # Unauthorized


def test_send_notification_with_api_key(client, api_key):
    """Test the send notification endpoint with a valid API key."""
    notification_data = {
        "message": "Test notification",
        "notification_type": "system",
        "level": "info",
        "channels": ["dashboard"],
    }
    response = client.post(
        "/notifications/send", json=notification_data, headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Test notification"
    assert response.json()["status"] == "SENT"


def test_send_notification_without_api_key(client):
    """Test the send notification endpoint without an API key."""
    notification_data = {
        "message": "Test notification",
        "notification_type": "system",
        "level": "info",
        "channels": ["dashboard"],
    }
    response = client.post("/notifications/send", json=notification_data)
    assert response.status_code == 401  # Unauthorized


def test_emit_event_with_api_key(client, api_key):
    """Test the emit event endpoint with a valid API key."""
    event_data = {
        "event_type": "test.event",
        "data": {"message": "Test event"},
        "source": "test",
    }
    response = client.post("/events/emit", json=event_data, headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert response.json()["event_type"] == "test.event"
    assert response.json()["data"]["message"] == "Test event"


def test_emit_event_without_api_key(client):
    """Test the emit event endpoint without an API key."""
    event_data = {
        "event_type": "test.event",
        "data": {"message": "Test event"},
        "source": "test",
    }
    response = client.post("/events/emit", json=event_data)
    assert response.status_code == 401  # Unauthorized


def test_rate_limiting(client, api_key):
    """Test rate limiting."""
    # Make multiple requests to trigger rate limiting
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200

    # Check rate limit headers
    assert "X-Rate-Limit-Limit" in response.headers
    assert "X-Rate-Limit-Remaining" in response.headers
    assert "X-Rate-Limit-Reset" in response.headers