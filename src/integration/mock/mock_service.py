"""Mock Service implementation for the Friday AI Trading System.

This module provides mock implementations of external systems for development and testing.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Set
import threading
import time
import json
import random
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.integration.external_system_registry import SystemType, SystemStatus

# Create logger
logger = get_logger(__name__)


class MockServiceError(FridayError):
    """Exception raised for mock service errors."""
    pass


class MockResponseType(Enum):
    """Enum for mock response types."""
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    RATE_LIMIT = auto()
    MAINTENANCE = auto()


class MockService(ABC):
    """Base class for mock external services.

    This class provides a foundation for creating mock implementations of external
    systems for development and testing purposes.

    Attributes:
        service_id: Unique identifier for the mock service.
        name: Human-readable name of the mock service.
        service_type: Type of the mock service.
        config: Configuration for the mock service.
        is_running: Flag indicating if the mock service is running.
        endpoints: Dictionary of registered endpoints and their handlers.
        _lock: Lock for thread-safe access to the service.
    """

    def __init__(
        self,
        service_id: str,
        name: str,
        service_type: SystemType,
        config: Dict[str, Any]
    ):
        """Initialize the mock service.

        Args:
            service_id: Unique identifier for the mock service.
            name: Human-readable name of the mock service.
            service_type: Type of the mock service.
            config: Configuration for the mock service.
        """
        self.service_id = service_id
        self.name = name
        self.service_type = service_type
        self.config = config
        self.is_running = False
        self.endpoints: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        # Register default endpoints
        self._register_default_endpoints()
        
        logger.info(f"Initialized mock service '{service_id}' of type {service_type.name}")

    def _register_default_endpoints(self) -> None:
        """Register default endpoints for the mock service."""
        self.register_endpoint("status", self.get_status)
        self.register_endpoint("authenticate", self.authenticate)

    def register_endpoint(self, endpoint: str, handler: Callable) -> None:
        """Register an endpoint with the mock service.

        Args:
            endpoint: The name of the endpoint.
            handler: The function to handle requests to the endpoint.
        """
        with self._lock:
            self.endpoints[endpoint] = handler
            logger.debug(f"Registered endpoint '{endpoint}' for mock service '{self.service_id}'")

    def handle_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """Handle a request to the mock service.

        Args:
            endpoint: The endpoint to call.
            params: Parameters for the request.

        Returns:
            Any: The response from the endpoint handler.

        Raises:
            MockServiceError: If the endpoint is not registered or the service is not running.
        """
        if not self.is_running:
            raise MockServiceError(f"Mock service '{self.service_id}' is not running")
            
        handler = self.endpoints.get(endpoint)
        if not handler:
            raise MockServiceError(f"Endpoint '{endpoint}' not found in mock service '{self.service_id}'")
            
        # Apply configured behavior (latency, errors, etc.)
        response_type = self._determine_response_type(endpoint)
        
        if response_type == MockResponseType.TIMEOUT:
            raise MockServiceError(f"Request to '{endpoint}' timed out")
        elif response_type == MockResponseType.ERROR:
            raise MockServiceError(f"Error processing request to '{endpoint}'")
        elif response_type == MockResponseType.RATE_LIMIT:
            raise MockServiceError("Rate limit exceeded", status_code=429)
        elif response_type == MockResponseType.MAINTENANCE:
            raise MockServiceError("Service is in maintenance mode", status_code=503)
            
        # Apply configured latency
        latency = self._get_configured_latency(endpoint)
        if latency > 0:
            time.sleep(latency)
            
        # Call the handler
        return handler(params)

    def _determine_response_type(self, endpoint: str) -> MockResponseType:
        """Determine the type of response to return for a request.

        Args:
            endpoint: The endpoint being called.

        Returns:
            MockResponseType: The type of response to return.
        """
        behavior = self.config.get("behavior", {})
        
        # Check for endpoint-specific behavior
        endpoint_behavior = behavior.get("endpoints", {}).get(endpoint, {})
        
        # Get error rates
        error_rate = endpoint_behavior.get("error_rate", behavior.get("error_rate", 0.0))
        timeout_rate = endpoint_behavior.get("timeout_rate", behavior.get("timeout_rate", 0.0))
        rate_limit_rate = endpoint_behavior.get("rate_limit_rate", behavior.get("rate_limit_rate", 0.0))
        maintenance_rate = endpoint_behavior.get("maintenance_rate", behavior.get("maintenance_rate", 0.0))
        
        # Determine response type based on probabilities
        rand = random.random()
        if rand < error_rate:
            return MockResponseType.ERROR
        elif rand < error_rate + timeout_rate:
            return MockResponseType.TIMEOUT
        elif rand < error_rate + timeout_rate + rate_limit_rate:
            return MockResponseType.RATE_LIMIT
        elif rand < error_rate + timeout_rate + rate_limit_rate + maintenance_rate:
            return MockResponseType.MAINTENANCE
        else:
            return MockResponseType.SUCCESS

    def _get_configured_latency(self, endpoint: str) -> float:
        """Get the configured latency for an endpoint.

        Args:
            endpoint: The endpoint being called.

        Returns:
            float: The latency in seconds.
        """
        behavior = self.config.get("behavior", {})
        
        # Check for endpoint-specific latency
        endpoint_behavior = behavior.get("endpoints", {}).get(endpoint, {})
        
        # Get latency configuration
        base_latency = endpoint_behavior.get("latency", behavior.get("latency", 0.0))
        latency_variation = endpoint_behavior.get("latency_variation", behavior.get("latency_variation", 0.0))
        
        # Calculate actual latency
        if latency_variation > 0:
            return max(0.0, base_latency + random.uniform(-latency_variation, latency_variation))
        else:
            return max(0.0, base_latency)

    def start(self) -> bool:
        """Start the mock service.

        Returns:
            bool: True if the service was started, False if it was already running.
        """
        with self._lock:
            if self.is_running:
                logger.warning(f"Mock service '{self.service_id}' is already running")
                return False
                
            self.is_running = True
            logger.info(f"Started mock service '{self.service_id}'")
            return True

    def stop(self) -> bool:
        """Stop the mock service.

        Returns:
            bool: True if the service was stopped, False if it was not running.
        """
        with self._lock:
            if not self.is_running:
                logger.warning(f"Mock service '{self.service_id}' is not running")
                return False
                
            self.is_running = False
            logger.info(f"Stopped mock service '{self.service_id}'")
            return True

    def get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get the status of the mock service.

        Args:
            params: Parameters for the request.

        Returns:
            Dict[str, Any]: The status of the mock service.
        """
        return {
            "service_id": self.service_id,
            "name": self.name,
            "type": self.service_type.name,
            "status": "running" if self.is_running else "stopped",
            "endpoints": list(self.endpoints.keys())
        }

    @abstractmethod
    def authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with the mock service.

        Args:
            params: Authentication parameters.

        Returns:
            Dict[str, Any]: Authentication result.
        """
        pass


class MockBrokerService(MockService):
    """Mock implementation of a broker service.

    This class provides a mock implementation of a broker service for development and testing.
    """

    def __init__(self, service_id: str, name: str, config: Dict[str, Any]):
        """Initialize the mock broker service.

        Args:
            service_id: Unique identifier for the mock service.
            name: Human-readable name of the mock service.
            config: Configuration for the mock service.
        """
        super().__init__(service_id, name, SystemType.BROKER, config)
        
        # Initialize broker-specific state
        self.accounts = config.get("accounts", [])
        self.positions = config.get("positions", {})
        self.orders = config.get("orders", [])
        self.order_id_counter = 1000
        
        # Register broker-specific endpoints
        self.register_endpoint("get_accounts", self.get_accounts)
        self.register_endpoint("get_positions", self.get_positions)
        self.register_endpoint("get_orders", self.get_orders)
        self.register_endpoint("place_order", self.place_order)
        self.register_endpoint("cancel_order", self.cancel_order)

    def authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with the mock broker service.

        Args:
            params: Authentication parameters.

        Returns:
            Dict[str, Any]: Authentication result.
        """
        api_key = params.get("api_key")
        api_secret = params.get("api_secret")
        
        # Check if credentials are valid
        valid_credentials = self.config.get("authentication", {}).get("valid_credentials", [])
        
        for cred in valid_credentials:
            if cred.get("api_key") == api_key and cred.get("api_secret") == api_secret:
                # Generate a token
                token = str(uuid.uuid4())
                
                return {
                    "success": True,
                    "token": token,
                    "expires_in": 3600,
                    "account_id": cred.get("account_id")
                }
                
        return {
            "success": False,
            "error": "Invalid credentials"
        }

    def get_accounts(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get accounts from the mock broker service.

        Args:
            params: Parameters for the request.

        Returns:
            List[Dict[str, Any]]: List of accounts.
        """
        return self.accounts

    def get_positions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get positions from the mock broker service.

        Args:
            params: Parameters for the request.

        Returns:
            Dict[str, Any]: Dictionary of positions.
        """
        account_id = params.get("account_id")
        if account_id and account_id in self.positions:
            return {"positions": self.positions[account_id]}
        elif not account_id:
            return {"positions": self.positions}
        else:
            return {"positions": []}

    def get_orders(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get orders from the mock broker service.

        Args:
            params: Parameters for the request.

        Returns:
            List[Dict[str, Any]]: List of orders.
        """
        account_id = params.get("account_id")
        status = params.get("status")
        
        filtered_orders = self.orders
        
        if account_id:
            filtered_orders = [order for order in filtered_orders if order.get("account_id") == account_id]
            
        if status:
            filtered_orders = [order for order in filtered_orders if order.get("status") == status]
            
        return filtered_orders

    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order with the mock broker service.

        Args:
            params: Order parameters.

        Returns:
            Dict[str, Any]: Order result.
        """
        # Generate order ID
        order_id = str(self.order_id_counter)
        self.order_id_counter += 1
        
        # Create order
        order = {
            "order_id": order_id,
            "account_id": params.get("account_id"),
            "symbol": params.get("symbol"),
            "side": params.get("side"),
            "quantity": params.get("quantity"),
            "order_type": params.get("order_type"),
            "price": params.get("price"),
            "status": "pending",
            "created_at": time.time()
        }
        
        # Add order to list
        self.orders.append(order)
        
        # Simulate order execution in a separate thread
        threading.Thread(target=self._simulate_order_execution, args=(order_id,), daemon=True).start()
        
        return {
            "success": True,
            "order_id": order_id,
            "status": "pending"
        }

    def cancel_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel an order with the mock broker service.

        Args:
            params: Cancellation parameters.

        Returns:
            Dict[str, Any]: Cancellation result.
        """
        order_id = params.get("order_id")
        
        for order in self.orders:
            if order.get("order_id") == order_id:
                if order.get("status") in ("pending", "open"):
                    order["status"] = "cancelled"
                    return {
                        "success": True,
                        "order_id": order_id,
                        "status": "cancelled"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Order {order_id} cannot be cancelled (status: {order.get('status')})"
                    }
                    
        return {
            "success": False,
            "error": f"Order {order_id} not found"
        }

    def _simulate_order_execution(self, order_id: str) -> None:
        """Simulate the execution of an order.

        Args:
            order_id: The ID of the order to execute.
        """
        # Find the order
        order = None
        for o in self.orders:
            if o.get("order_id") == order_id:
                order = o
                break
                
        if not order:
            return
            
        # Simulate order processing delay
        time.sleep(random.uniform(0.5, 2.0))
        
        # Update order status to open
        order["status"] = "open"
        
        # Simulate order execution delay
        time.sleep(random.uniform(1.0, 5.0))
        
        # Determine if order is filled or rejected
        if random.random() < 0.9:  # 90% chance of success
            order["status"] = "filled"
            order["filled_at"] = time.time()
            order["filled_price"] = order.get("price")
            
            # Update positions
            account_id = order.get("account_id")
            symbol = order.get("symbol")
            quantity = order.get("quantity")
            side = order.get("side")
            
            if account_id not in self.positions:
                self.positions[account_id] = {}
                
            if symbol not in self.positions[account_id]:
                self.positions[account_id][symbol] = {
                    "quantity": 0,
                    "average_price": 0.0
                }
                
            position = self.positions[account_id][symbol]
            
            if side == "buy":
                position["quantity"] += quantity
            else:  # sell
                position["quantity"] -= quantity
        else:
            order["status"] = "rejected"
            order["rejected_at"] = time.time()
            order["reject_reason"] = "Simulated rejection"


class MockMarketDataService(MockService):
    """Mock implementation of a market data service.

    This class provides a mock implementation of a market data service for development and testing.
    """

    def __init__(self, service_id: str, name: str, config: Dict[str, Any]):
        """Initialize the mock market data service.

        Args:
            service_id: Unique identifier for the mock service.
            name: Human-readable name of the mock service.
            config: Configuration for the mock service.
        """
        super().__init__(service_id, name, SystemType.MARKET_DATA, config)
        
        # Initialize market data-specific state
        self.symbols = config.get("symbols", [])
        self.quotes = config.get("quotes", {})
        self.bars = config.get("bars", {})
        self.subscribers = {}
        
        # Register market data-specific endpoints
        self.register_endpoint("get_symbols", self.get_symbols)
        self.register_endpoint("get_quote", self.get_quote)
        self.register_endpoint("get_bars", self.get_bars)
        self.register_endpoint("subscribe", self.subscribe)
        self.register_endpoint("unsubscribe", self.unsubscribe)

    def authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with the mock market data service.

        Args:
            params: Authentication parameters.

        Returns:
            Dict[str, Any]: Authentication result.
        """
        api_key = params.get("api_key")
        
        # Check if API key is valid
        valid_keys = self.config.get("authentication", {}).get("valid_keys", [])
        
        if api_key in valid_keys:
            # Generate a token
            token = str(uuid.uuid4())
            
            return {
                "success": True,
                "token": token,
                "expires_in": 3600
            }
            
        return {
            "success": False,
            "error": "Invalid API key"
        }

    def get_symbols(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get symbols from the mock market data service.

        Args:
            params: Parameters for the request.

        Returns:
            List[Dict[str, Any]]: List of symbols.
        """
        return self.symbols

    def get_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a quote from the mock market data service.

        Args:
            params: Parameters for the request.

        Returns:
            Dict[str, Any]: Quote data.
        """
        symbol = params.get("symbol")
        
        if symbol in self.quotes:
            quote = self.quotes[symbol].copy()
            
            # Add some randomness to simulate real-time data
            price = quote.get("price", 100.0)
            variation = price * 0.001  # 0.1% variation
            quote["price"] = price + random.uniform(-variation, variation)
            quote["timestamp"] = time.time()
            
            return quote
        else:
            return {
                "symbol": symbol,
                "error": "Symbol not found"
            }

    def get_bars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get bars from the mock market data service.

        Args:
            params: Parameters for the request.

        Returns:
            Dict[str, Any]: Bar data.
        """
        symbol = params.get("symbol")
        timeframe = params.get("timeframe", "1d")
        limit = params.get("limit", 10)
        
        key = f"{symbol}_{timeframe}"
        
        if key in self.bars:
            bars = self.bars[key][-limit:].copy()
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars
            }
        else:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "Bars not found"
            }

    def subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe to data from the mock market data service.

        Args:
            params: Subscription parameters.

        Returns:
            Dict[str, Any]: Subscription result.
        """
        symbol = params.get("symbol")
        channel = params.get("channel", "quotes")
        client_id = params.get("client_id")
        
        if not client_id:
            return {
                "success": False,
                "error": "Client ID is required"
            }
            
        key = f"{channel}_{symbol}"
        
        if key not in self.subscribers:
            self.subscribers[key] = set()
            
        self.subscribers[key].add(client_id)
        
        return {
            "success": True,
            "symbol": symbol,
            "channel": channel,
            "client_id": client_id
        }

    def unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unsubscribe from data from the mock market data service.

        Args:
            params: Unsubscription parameters.

        Returns:
            Dict[str, Any]: Unsubscription result.
        """
        symbol = params.get("symbol")
        channel = params.get("channel", "quotes")
        client_id = params.get("client_id")
        
        if not client_id:
            return {
                "success": False,
                "error": "Client ID is required"
            }
            
        key = f"{channel}_{symbol}"
        
        if key in self.subscribers and client_id in self.subscribers[key]:
            self.subscribers[key].remove(client_id)
            
            if not self.subscribers[key]:
                del self.subscribers[key]
                
            return {
                "success": True,
                "symbol": symbol,
                "channel": channel,
                "client_id": client_id
            }
        else:
            return {
                "success": False,
                "error": "Subscription not found"
            }


class MockFinancialDataService(MockService):
    """Mock implementation of a financial data service.

    This class provides a mock implementation of a financial data service for development and testing.
    """

    def __init__(self, service_id: str, name: str, config: Dict[str, Any]):
        """Initialize the mock financial data service.

        Args:
            service_id: Unique identifier for the mock service.
            name: Human-readable name of the mock service.
            config: Configuration for the mock service.
        """
        super().__init__(service_id, name, SystemType.FINANCIAL_DATA, config)
        
        # Initialize financial data-specific state
        self.companies = config.get("companies", [])
        self.financials = config.get("financials", {})
        self.news = config.get("news", [])
        
        # Register financial data-specific endpoints
        self.register_endpoint("get_companies", self.get_companies)
        self.register_endpoint("get_financials", self.get_financials)
        self.register_endpoint("get_news", self.get_news)

    def authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with the mock financial data service.

        Args:
            params: Authentication parameters.

        Returns:
            Dict[str, Any]: Authentication result.
        """
        api_key = params.get("api_key")
        
        # Check if API key is valid
        valid_keys = self.config.get("authentication", {}).get("valid_keys", [])
        
        if api_key in valid_keys:
            # Generate a token
            token = str(uuid.uuid4())
            
            return {
                "success": True,
                "token": token,
                "expires_in": 3600
            }
            
        return {
            "success": False,
            "error": "Invalid API key"
        }

    def get_companies(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get companies from the mock financial data service.

        Args:
            params: Parameters for the request.

        Returns:
            List[Dict[str, Any]]: List of companies.
        """
        return self.companies

    def get_financials(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get financials from the mock financial data service.

        Args:
            params: Parameters for the request.

        Returns:
            Dict[str, Any]: Financial data.
        """
        symbol = params.get("symbol")
        period = params.get("period", "annual")
        
        key = f"{symbol}_{period}"
        
        if key in self.financials:
            return {
                "symbol": symbol,
                "period": period,
                "financials": self.financials[key]
            }
        else:
            return {
                "symbol": symbol,
                "period": period,
                "error": "Financials not found"
            }

    def get_news(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get news from the mock financial data service.

        Args:
            params: Parameters for the request.

        Returns:
            List[Dict[str, Any]]: List of news articles.
        """
        symbol = params.get("symbol")
        limit = params.get("limit", 10)
        
        if symbol:
            filtered_news = [news for news in self.news if symbol in news.get("symbols", [])]
            return filtered_news[:limit]
        else:
            return self.news[:limit]


class MockServiceFactory:
    """Factory for creating mock services.

    This class provides methods for creating mock services of different types.
    """

    @staticmethod
    def create_mock_service(
        service_id: str,
        service_type: SystemType,
        config: Dict[str, Any]
    ) -> str:
        """Create a mock service.

        Args:
            service_id: Unique identifier for the mock service.
            service_type: Type of the mock service.
            config: Configuration for the mock service.

        Returns:
            str: The ID of the created mock service.

        Raises:
            MockServiceError: If the service type is not supported.
        """
        name = config.get("name", service_id)
        
        if service_type == SystemType.BROKER:
            service = MockBrokerService(service_id, name, config)
        elif service_type == SystemType.MARKET_DATA:
            service = MockMarketDataService(service_id, name, config)
        elif service_type == SystemType.FINANCIAL_DATA:
            service = MockFinancialDataService(service_id, name, config)
        else:
            raise MockServiceError(f"Unsupported mock service type: {service_type.name}")
            
        # Store service in registry
        from src.integration.mock.mock_registry import MockServiceRegistry
        MockServiceRegistry.register_service(service)
        
        # Start service if auto_start is enabled
        if config.get("auto_start", True):
            service.start()
            
        return service_id


# Convenience function for creating mock services
def create_mock_service(
    service_id: str,
    service_type: SystemType,
    config: Dict[str, Any]
) -> str:
    """Create a mock service.

    Args:
        service_id: Unique identifier for the mock service.
        service_type: Type of the mock service.
        config: Configuration for the mock service.

    Returns:
        str: The ID of the created mock service.
    """
    return MockServiceFactory.create_mock_service(service_id, service_type, config)