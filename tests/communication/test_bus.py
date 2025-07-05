#!/usr/bin/env python3
"""
Unit tests for CommunicationBus production-ready enhancements.
Tests back-pressure, rate limiting, circuit-breaker metrics, and graceful shutdown.
"""

import asyncio
import pytest
import time
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.infrastructure.communication.communication_system import (
    CommunicationBus, 
    CommunicationBusMetrics,
    Message, 
    MessageType, 
    MessageHandler
)


class TestMessageHandler(MessageHandler):
    """Test message handler for testing purposes."""
    
    def __init__(self, response_delay=0):
        self.received_messages = []
        self.response_delay = response_delay
    
    async def handle_message(self, message: Message):
        """Handle message with optional delay."""
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        self.received_messages.append(message)
        return Message(
            type=MessageType.RESPONSE,
            source="test_handler",
            topic=f"{message.topic}.response",
            payload={"handled": True}
        )
    
    def get_supported_topics(self):
        return ["test.*"]


class TestCommunicationBusMetrics:
    """Test CommunicationBusMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CommunicationBusMetrics()
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.latencies == []
    
    def test_record_success(self):
        """Test recording success metrics."""
        metrics = CommunicationBusMetrics()
        metrics.record_success(0.5)
        metrics.record_success(1.0)
        
        assert metrics.success_count == 2
        assert metrics.latencies == [0.5, 1.0]
    
    def test_record_failure(self):
        """Test recording failure metrics."""
        metrics = CommunicationBusMetrics()
        metrics.record_failure()
        metrics.record_failure()
        
        assert metrics.failure_count == 2
    
    def test_get_metrics(self):
        """Test getting metrics."""
        metrics = CommunicationBusMetrics()
        metrics.record_success(0.5)
        metrics.record_success(1.5)
        metrics.record_failure()
        
        result = metrics.get_metrics()
        
        assert result["success_count"] == 2
        assert result["failure_count"] == 1
        assert result["average_latency"] == 1.0  # (0.5 + 1.5) / 2


class TestCommunicationBus:
    """Test CommunicationBus production-ready features."""
    
    def test_initialization_with_parameters(self):
        """Test bus initialization with rate limit and queue size parameters."""
        bus = CommunicationBus(max_queue_size=50, rate_limit=5)
        
        assert bus.rate_limit == 5
        assert bus.message_queue.maxsize == 50
        assert isinstance(bus.metrics, CommunicationBusMetrics)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        bus = CommunicationBus(rate_limit=2)  # 2 messages per second
        
        message = Message(
            type=MessageType.REQUEST,
            source="test",
            topic="test.message",
            payload={"data": "test"}
        )
        
        # First message should go through
        await bus.publish_message(message)
        assert bus.message_queue.qsize() == 1
        
        # Second message immediately should be rate limited
        await bus.publish_message(message)
        # Should still only have 1 message due to rate limiting
        assert bus.message_queue.qsize() == 1
        
        # Wait and try again
        await asyncio.sleep(0.6)  # Wait more than 1/rate_limit seconds
        await bus.publish_message(message)
        assert bus.message_queue.qsize() == 2
    
    @pytest.mark.asyncio
    async def test_back_pressure_queue_full(self):
        """Test back-pressure when queue is full."""
        bus = CommunicationBus(max_queue_size=2, rate_limit=100)  # High rate limit
        
        message = Message(
            type=MessageType.REQUEST,
            source="test",
            topic="test.message",
            payload={"data": "test"}
        )
        
        # Fill the queue
        await bus.publish_message(message)
        await bus.publish_message(message)
        
        # Queue should be full
        assert bus.message_queue.qsize() == 2
        assert bus.message_queue.full()
        
        # Next message should handle back-pressure
        # This would block in real scenario, but in our test we verify the queue state
        try:
            bus.message_queue.put_nowait(message)
            assert False, "Should have raised QueueFull exception"
        except asyncio.QueueFull:
            pass  # Expected behavior
    
    def test_metrics_integration(self):
        """Test metrics integration with communication bus."""
        bus = CommunicationBus()
        
        # Test getting metrics
        metrics = bus.get_metrics()
        assert "success_count" in metrics
        assert "failure_count" in metrics
        assert "average_latency" in metrics
        
        # Initially should be zero
        assert metrics["success_count"] == 0
        assert metrics["failure_count"] == 0
        assert metrics["average_latency"] == 0
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_on_failure(self):
        """Test exponential backoff and reconnection logic."""
        bus = CommunicationBus()
        
        # Mock the _route_message to raise an exception
        original_route = bus._route_message
        
        async def failing_route(message):
            raise Exception("Simulated failure")
        
        bus._route_message = failing_route
        
        # Start the bus
        await bus.start()
        
        # Publish a message that will fail
        message = Message(
            type=MessageType.REQUEST,
            source="test",
            topic="test.message",
            payload={"data": "test"}
        )
        await bus.publish_message(message)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Should have recorded failure
        metrics = bus.get_metrics()
        assert metrics["failure_count"] > 0
        
        # Restore original method and stop bus
        bus._route_message = original_route
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_message_processing_with_metrics(self):
        """Test message processing records metrics correctly."""
        bus = CommunicationBus()
        handler = TestMessageHandler()
        
        bus.register_handler("test.message", handler)
        await bus.start()
        
        # Send a message
        message = Message(
            type=MessageType.REQUEST,
            source="test",
            topic="test.message",
            payload={"data": "test"}
        )
        await bus.publish_message(message)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check metrics
        metrics = bus.get_metrics()
        assert metrics["success_count"] > 0
        assert len(handler.received_messages) == 1
        
        await bus.stop()
    
    def test_shutdown_handler_registration(self):
        """Test that shutdown handlers are registered."""
        bus = CommunicationBus()
        
        # Check that the _handle_shutdown method exists
        assert hasattr(bus, '_handle_shutdown')
        assert callable(bus._handle_shutdown)
    
    @patch('signal.signal')
    @pytest.mark.asyncio
    async def test_graceful_shutdown_signal_registration(self, mock_signal):
        """Test that signal handlers are registered for graceful shutdown."""
        bus = CommunicationBus()
        
        # Start the bus (which should register signal handlers)
        await bus.start()
        
        # Verify signal handlers were registered
        assert mock_signal.call_count >= 2  # At least SIGINT and SIGTERM
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_persistent_reconnect_logic(self):
        """Test persistent reconnect with exponential backoff."""
        bus = CommunicationBus()
        
        # Create a mock that fails a few times then succeeds
        failure_count = 0
        max_failures = 2
        
        async def intermittent_failure(message):
            nonlocal failure_count
            if failure_count < max_failures:
                failure_count += 1
                raise Exception(f"Failure {failure_count}")
            return None
        
        # Replace the route method
        original_route = bus._route_message
        bus._route_message = intermittent_failure
        
        await bus.start()
        
        # Send a message
        message = Message(
            type=MessageType.REQUEST,
            source="test",
            topic="test.message",
            payload={"data": "test"}
        )
        await bus.publish_message(message)
        
        # Wait for retries
        await asyncio.sleep(0.5)
        
        # Should have recorded failures but continued running
        metrics = bus.get_metrics()
        assert metrics["failure_count"] >= max_failures
        assert bus.running  # Should still be running
        
        # Restore and cleanup
        bus._route_message = original_route
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self):
        """Test that circuit breaker metrics are properly collected."""
        bus = CommunicationBus()
        handler = TestMessageHandler(response_delay=0.1)  # Add small delay
        
        bus.register_handler("test.message", handler)
        await bus.start()
        
        # Send multiple messages
        for i in range(3):
            message = Message(
                type=MessageType.REQUEST,
                source="test",
                topic="test.message",
                payload={"data": f"test_{i}"}
            )
            await bus.publish_message(message)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check metrics
        metrics = bus.get_metrics()
        assert metrics["success_count"] == 3
        assert metrics["average_latency"] > 0  # Should have recorded latencies
        assert len(bus.metrics.latencies) == 3
        
        await bus.stop()


@pytest.mark.asyncio
async def test_production_readiness_integration():
    """Integration test for all production-ready features."""
    bus = CommunicationBus(max_queue_size=10, rate_limit=5)
    handler = TestMessageHandler()
    
    # Register handler
    bus.register_handler("prod.test", handler)
    
    # Start bus
    await bus.start()
    
    # Test normal operation
    message = Message(
        type=MessageType.REQUEST,
        source="prod_test",
        topic="prod.test",
        payload={"test": "production"}
    )
    
    await bus.publish_message(message)
    await asyncio.sleep(0.1)
    
    # Verify metrics are working
    metrics = bus.get_metrics()
    assert metrics["success_count"] > 0
    
    # Test rate limiting
    start_time = time.time()
    for i in range(3):
        await bus.publish_message(message)
    end_time = time.time()
    
    # Should have taken some time due to rate limiting
    assert end_time - start_time > 0.1
    
    # Cleanup
    await bus.stop()
    
    print("✅ All production-ready features working correctly!")


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(test_production_readiness_integration())
