"""Portfolio Management System Integration for the Friday AI Trading System.

This module integrates the Portfolio Management System with other components
of the Friday AI Trading System, providing a unified interface for portfolio
management functionality.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import logging

# Import portfolio components
from .portfolio_factory import PortfolioFactory
from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager
from .allocation_manager import AllocationManager

# Optional imports for integration with other components
try:
    from ..infrastructure.event.event_system import EventSystem
    EVENT_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from infrastructure.event.event_system import EventSystem
        EVENT_SYSTEM_AVAILABLE = True
    except ImportError:
        EVENT_SYSTEM_AVAILABLE = False
        # Define placeholder class if event system is not available
        class EventSystem:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def publish(self, *args: Any, **kwargs: Any) -> None:
                pass

            def subscribe(self, *args: Any, **kwargs: Any) -> None:
                pass

try:
    from ..orchestration.trading_engine.integration import TradingEngineIntegrator
    TRADING_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from orchestration.trading_engine.integration import TradingEngineIntegrator
        TRADING_ENGINE_AVAILABLE = True
    except ImportError:
        TRADING_ENGINE_AVAILABLE = False
        # Define placeholder class if trading engine is not available
        class TradingEngineIntegrator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

try:
    from ..data.market_data_service import MarketDataService
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    try:
        from data.market_data_service import MarketDataService
        DATA_SERVICE_AVAILABLE = True
    except ImportError:
        DATA_SERVICE_AVAILABLE = False
        # Define placeholder class if data service is not available
        class MarketDataService:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def get_latest_prices(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
                return {}

try:
    from ..risk.advanced_risk_manager import AdvancedRiskManager
    RISK_MODULE_AVAILABLE = True
except ImportError:
    try:
        from risk.advanced_risk_manager import AdvancedRiskManager
        RISK_MODULE_AVAILABLE = True
    except ImportError:
        RISK_MODULE_AVAILABLE = False
        # Define placeholder class if risk module is not available
        class AdvancedRiskManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

logger = logging.getLogger(__name__)

class PortfolioIntegration:
    """Integration class for the Portfolio Management System.

    This class integrates the Portfolio Management System with other components
    of the Friday AI Trading System, including:

    - Event System: For publishing and subscribing to portfolio events
    - Trading Engine: For executing trades and receiving trade events
    - Market Data Service: For receiving market data updates
    - Risk Management System: For risk assessment and constraints
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 event_system: Optional[EventSystem] = None,
                 trading_engine: Optional[TradingEngineIntegrator] = None,
                 market_data_service: Optional[MarketDataService] = None,
                 auto_start: bool = False):
        """Initialize the Portfolio Integration.

        Args:
            config: Configuration dictionary for portfolio components
            event_system: Event system for publishing and subscribing to events
            trading_engine: Trading engine for executing trades
            market_data_service: Market data service for price updates
            auto_start: Whether to automatically start the integration
        """
        self.config = config or {}
        self.event_system = event_system
        self.trading_engine = trading_engine
        self.market_data_service = market_data_service

        # Create portfolio factory and components
        self.factory = PortfolioFactory(self.config)
        # Disable risk management for testing if needed
        if not RISK_MODULE_AVAILABLE:
            self.factory.risk_management_available = False

        # Get initial capital from config if available
        initial_capital = self.config.get("portfolio_manager", {}).get("initial_cash",
                                                                  self.config.get("portfolio_manager", {}).get("initial_capital", 100000.0))
        self.portfolio_system = self.factory.create_complete_portfolio_system(initial_capital=initial_capital)

        # Extract components for easier access
        self.portfolio_manager = self.portfolio_system["portfolio_manager"]
        self.performance_calculator = self.portfolio_system["performance_calculator"]
        self.tax_manager = self.portfolio_system["tax_manager"]
        self.allocation_manager = self.portfolio_system["allocation_manager"]

        # Risk manager may be optional
        self.risk_manager = self.portfolio_system.get("risk_manager")

        # Track integration status
        self._started = False

        # Auto-start if requested
        if auto_start:
            self.start()

        logger.info("Portfolio Integration initialized")

    def start(self) -> None:
        """Start the portfolio integration.

        This method sets up event subscriptions and initializes connections
        with other components.
        """
        if self._started:
            logger.warning("Portfolio Integration already started")
            return

        # Set up event subscriptions if event system is available
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            self._setup_event_subscriptions()

        # Set up trading engine integration if available
        if self.trading_engine and TRADING_ENGINE_AVAILABLE:
            self._setup_trading_engine_integration()

        # Set up market data service integration if available
        if self.market_data_service and DATA_SERVICE_AVAILABLE:
            self._setup_market_data_integration()

        self._started = True
        logger.info("Portfolio Integration started")

    def stop(self) -> None:
        """Stop the portfolio integration.

        This method cleans up event subscriptions and connections.
        """
        if not self._started:
            logger.warning("Portfolio Integration not started")
            return

        # Clean up event subscriptions if event system is available
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            self._cleanup_event_subscriptions()

        self._started = False
        logger.info("Portfolio Integration stopped")

    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions with the event system."""
        # Subscribe to market data updates
        self.event_system.subscribe(
            event_type="market_data_update",
            callback=self._handle_market_data_update
        )

        # Subscribe to trade execution events
        self.event_system.subscribe(
            event_type="trade_executed",
            callback=self._handle_trade_executed
        )

        # Subscribe to portfolio update requests
        self.event_system.subscribe(
            event_type="portfolio_update_request",
            callback=self._handle_portfolio_update_request
        )

        # Subscribe to rebalance requests
        self.event_system.subscribe(
            event_type="portfolio_rebalance_request",
            callback=self._handle_rebalance_request
        )

        logger.info("Portfolio event subscriptions set up")

    def _cleanup_event_subscriptions(self) -> None:
        """Clean up event subscriptions with the event system."""
        # Unsubscribe from all events
        if hasattr(self.event_system, "unsubscribe_all"):
            self.event_system.unsubscribe_all(subscriber=self)

        logger.info("Portfolio event subscriptions cleaned up")

    def _setup_trading_engine_integration(self) -> None:
        """Set up integration with the trading engine."""
        # Register portfolio manager with trading engine if supported
        if hasattr(self.trading_engine, "register_portfolio_manager"):
            self.trading_engine.register_portfolio_manager(self.portfolio_manager)

        # Register risk manager with trading engine if available and supported
        if self.risk_manager and hasattr(self.trading_engine, "register_risk_manager"):
            self.trading_engine.register_risk_manager(self.risk_manager)

        logger.info("Trading engine integration set up")

    def _setup_market_data_integration(self) -> None:
        """Set up integration with the market data service."""
        # Set up periodic price updates if supported
        if hasattr(self.market_data_service, "register_price_update_callback"):
            self.market_data_service.register_price_update_callback(
                callback=self._handle_market_data_update
            )

        logger.info("Market data service integration set up")

    def _handle_market_data_update(self, data: Dict[str, Any]) -> None:
        """Handle market data updates.

        Args:
            data: Market data update containing prices and other information
        """
        # Extract prices from the data
        if "prices" in data:
            prices = data["prices"]
            # Update portfolio prices
            self.portfolio_manager.update_prices(prices)

            # Publish portfolio value update event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                self.event_system.publish(
                    event_type="portfolio_value_update",
                    data={
                        "portfolio_id": self.portfolio_manager.portfolio_id,
                        "timestamp": datetime.now(),
                        "value": portfolio_value,
                        "positions": self.portfolio_manager.get_positions_summary()
                    }
                )

            logger.debug(f"Updated portfolio prices with {len(prices)} symbols")

    def _handle_trade_executed(self, data: Dict[str, Any]) -> None:
        """Handle trade execution events.

        Args:
            data: Trade execution data
        """
        # Extract trade details
        symbol = data.get("symbol")
        quantity = data.get("quantity")
        price = data.get("price")
        timestamp = data.get("timestamp", datetime.now())

        if symbol and quantity is not None and price is not None:
            # Execute the trade in the portfolio
            self.portfolio_manager.execute_trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                transaction_costs=data.get("commission", 0.0),
                sector=data.get("sector"),
                asset_class=data.get("asset_class")
            )

            # Update tax manager with the trade
            if quantity > 0:  # Buy trade
                self.tax_manager.add_tax_lot(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp
                )
            elif quantity < 0:  # Sell trade
                self.tax_manager.sell_shares(
                    symbol=symbol,
                    quantity=abs(quantity),
                    price=price,
                    timestamp=timestamp
                )

            # Update allocation manager
            self.allocation_manager.update_allocation_from_portfolio(
                self.portfolio_manager.get_positions_value(),
                self.portfolio_manager.get_portfolio_value()
            )

            # Publish portfolio update event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    event_type="portfolio_updated",
                    data={
                        "portfolio_id": self.portfolio_manager.portfolio_id,
                        "timestamp": datetime.now(),
                        "value": self.portfolio_manager.get_portfolio_value(),
                        "cash": self.portfolio_manager.cash,
                        "positions": self.portfolio_manager.get_positions_summary()
                    }
                )

            logger.info(f"Processed trade: {quantity} shares of {symbol} at ${price:.2f}")

    def _handle_portfolio_update_request(self, data: Dict[str, Any]) -> None:
        """Handle portfolio update requests.

        Args:
            data: Request data (unused)
        """
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            # Get current portfolio state
            portfolio_data = {
                "portfolio_id": self.portfolio_manager.portfolio_id,
                "timestamp": datetime.now(),
                "value": self.portfolio_manager.get_portfolio_value(),
                "cash": self.portfolio_manager.cash,
                "positions": self.portfolio_manager.get_positions_summary(),
                "transaction_history": self.portfolio_manager.get_transaction_history()
            }

            # Add performance metrics if available
            if hasattr(self.performance_calculator, "get_performance_metrics"):
                portfolio_data["performance_metrics"] = \
                    self.performance_calculator.get_performance_metrics()

            # Add tax information if available
            if hasattr(self.tax_manager, "get_tax_summary"):
                portfolio_data["tax_summary"] = self.tax_manager.get_tax_summary()

            # Add allocation information if available
            if hasattr(self.allocation_manager, "get_allocation_summary"):
                portfolio_data["allocation_summary"] = \
                    self.allocation_manager.get_allocation_summary()

            # Publish portfolio state
            self.event_system.publish(
                event_type="portfolio_state",
                data=portfolio_data
            )

            logger.debug("Published portfolio state in response to update request")

    def _handle_rebalance_request(self, data: Dict[str, Any]) -> None:
        """Handle portfolio rebalance requests.

        Args:
            data: Request data containing rebalance parameters
        """
        # Check if rebalancing is needed
        if self.allocation_manager.is_rebalance_needed():
            # Generate rebalance plan
            rebalance_plan = self.allocation_manager.generate_rebalance_plan(
                self.portfolio_manager.get_positions_value(),
                self.portfolio_manager.get_portfolio_value()
            )

            # Publish rebalance plan if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    event_type="portfolio_rebalance_plan",
                    data={
                        "portfolio_id": self.portfolio_manager.portfolio_id,
                        "timestamp": datetime.now(),
                        "rebalance_plan": rebalance_plan
                    }
                )

            logger.info(f"Generated rebalance plan with {len(rebalance_plan)} actions")
        else:
            # Publish no rebalance needed event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    event_type="portfolio_rebalance_not_needed",
                    data={
                        "portfolio_id": self.portfolio_manager.portfolio_id,
                        "timestamp": datetime.now(),
                        "message": "Rebalance not needed based on current allocation drift"
                    }
                )

            logger.info("Rebalance not needed based on current allocation drift")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the portfolio state.

        Returns:
            Dictionary containing portfolio summary information
        """
        summary = {
            "portfolio_id": self.portfolio_manager.portfolio_id,
            "timestamp": datetime.now(),
            "value": self.portfolio_manager.get_portfolio_value(),
            "cash": self.portfolio_manager.cash,
            "positions": self.portfolio_manager.get_positions_summary()
        }

        # Add performance metrics if available
        if hasattr(self.performance_calculator, "get_performance_metrics"):
            summary["performance_metrics"] = \
                self.performance_calculator.get_performance_metrics()

        # Add tax information if available
        if hasattr(self.tax_manager, "get_tax_summary"):
            summary["tax_summary"] = self.tax_manager.get_tax_summary()

        # Add allocation information if available
        if hasattr(self.allocation_manager, "get_allocation_summary"):
            summary["allocation_summary"] = \
                self.allocation_manager.get_allocation_summary()

        # Add risk metrics if available
        if self.risk_manager and hasattr(self.portfolio_manager, "get_risk_metrics"):
            summary["risk_metrics"] = self.portfolio_manager.get_risk_metrics()

        return summary


def create_portfolio_integration(
    config: Optional[Dict[str, Any]] = None,
    event_system: Optional[EventSystem] = None,
    trading_engine: Optional[TradingEngineIntegrator] = None,
    market_data_service: Optional[MarketDataService] = None,
    auto_start: bool = True
) -> PortfolioIntegration:
    """Create and initialize a portfolio integration instance.

    Args:
        config: Configuration dictionary for portfolio components
        event_system: Event system for publishing and subscribing to events
        trading_engine: Trading engine for executing trades
        market_data_service: Market data service for price updates
        auto_start: Whether to automatically start the integration

    Returns:
        Initialized PortfolioIntegration instance
    """
    integration = PortfolioIntegration(
        config=config,
        event_system=event_system,
        trading_engine=trading_engine,
        market_data_service=market_data_service,
        auto_start=auto_start
    )

    logger.info("Created portfolio integration")
    return integration


def integration_example() -> None:
    """Example of how to use the portfolio integration."""
    # Create configuration
    config = {
        "portfolio_manager": {
            "portfolio_id": "integrated-portfolio",
            "initial_cash": 100000.0
        },
        "performance_calculator": {
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02
        },
        "tax_manager": {
            "default_method": "FIFO",
            "wash_sale_window_days": 30
        },
        "allocation_manager": {
            "rebalance_method": "THRESHOLD",
            "default_threshold": 5.0,
            "rebalance_frequency_days": 90,
            "allocation_targets": [
                {"symbol": "AAPL", "target": 0.15},
                {"symbol": "MSFT", "target": 0.15},
                {"symbol": "GOOGL", "target": 0.10},
                {"symbol": "BND", "target": 0.30},
                {"symbol": "VTI", "target": 0.30}
            ]
        }
    }

    # Create event system if available
    event_system = None
    if EVENT_SYSTEM_AVAILABLE:
        try:
            from infrastructure.event.integration import EventSystemIntegration
            event_integration = EventSystemIntegration(auto_start=True)
            event_system = event_integration.event_system
        except ImportError:
            logger.warning("EventSystemIntegration not available")

    # Create portfolio integration
    integration = create_portfolio_integration(
        config=config,
        event_system=event_system,
        auto_start=True
    )

    # Simulate some trades
    portfolio = integration.portfolio_manager
    portfolio.execute_trade("AAPL", 50, 150.0)
    portfolio.execute_trade("MSFT", 40, 250.0)
    portfolio.execute_trade("GOOGL", 10, 1500.0)
    portfolio.execute_trade("BND", 300, 85.0)
    portfolio.execute_trade("VTI", 100, 200.0)

    # Update prices
    prices = {
        "AAPL": 160.0,
        "MSFT": 260.0,
        "GOOGL": 1550.0,
        "BND": 86.0,
        "VTI": 205.0
    }
    portfolio.update_prices(prices)

    # Get portfolio summary
    summary = integration.get_portfolio_summary()

    # Print summary
    logger.info(f"Portfolio ID: {summary['portfolio_id']}")
    logger.info(f"Portfolio Value: ${summary['value']:.2f}")
    logger.info(f"Cash: ${summary['cash']:.2f}")
    logger.info("Positions:")
    for symbol, position in summary['positions'].items():
        logger.info(f"  {symbol}: {position['quantity']} shares, ${position['value']:.2f}")

    # Stop integration
    integration.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run example
    integration_example()
