"""Multi-Portfolio Integration module.

This module provides the MultiPortfolioIntegration class, which extends the system
to support multiple portfolios simultaneously, with portfolio isolation,
portfolio-specific event routing, portfolio grouping, and cross-portfolio analysis.
"""

from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
import logging
from datetime import datetime

from .portfolio_registry import PortfolioRegistry
from .portfolio_integration import PortfolioIntegration

# Try to import event system components
try:
    from ..event.event_system import EventSystem, Event, EventType
    from ..event.event_types import (
        MarketDataEvent, TradeExecutionEvent, PortfolioUpdateRequestEvent,
        PortfolioUpdateEvent, RebalanceRequestEvent, RebalanceEvent
    )
    EVENT_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from event.event_system import EventSystem, Event, EventType
        from event.event_types import (
            MarketDataEvent, TradeExecutionEvent, PortfolioUpdateRequestEvent,
            PortfolioUpdateEvent, RebalanceRequestEvent, RebalanceEvent
        )
        EVENT_SYSTEM_AVAILABLE = True
    except ImportError:
        EVENT_SYSTEM_AVAILABLE = False
        # Define placeholder classes if event system is not available
        class EventSystem:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def subscribe(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def unsubscribe(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def publish(self, *args: Any, **kwargs: Any) -> None:
                pass
        
        class Event:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.event_type = None
                self.data = {}
        
        class EventType:
            MARKET_DATA = "MARKET_DATA"
            TRADE_EXECUTION = "TRADE_EXECUTION"
            PORTFOLIO_UPDATE_REQUEST = "PORTFOLIO_UPDATE_REQUEST"
            PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
            REBALANCE_REQUEST = "REBALANCE_REQUEST"
            REBALANCE = "REBALANCE"
        
        # Define placeholder event classes
        MarketDataEvent = Event
        TradeExecutionEvent = Event
        PortfolioUpdateRequestEvent = Event
        PortfolioUpdateEvent = Event
        RebalanceRequestEvent = Event
        RebalanceEvent = Event

# Try to import trading engine components
try:
    from ..trading.trading_engine import TradingEngine
    TRADING_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from trading.trading_engine import TradingEngine
        TRADING_ENGINE_AVAILABLE = True
    except ImportError:
        TRADING_ENGINE_AVAILABLE = False
        # Define placeholder class if trading engine is not available
        class TradingEngine:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def register_portfolio_manager(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def register_risk_manager(self, *args: Any, **kwargs: Any) -> None:
                pass

# Try to import market data service components
try:
    from ..market_data.market_data_service import MarketDataService
    MARKET_DATA_SERVICE_AVAILABLE = True
except ImportError:
    try:
        from market_data.market_data_service import MarketDataService
        MARKET_DATA_SERVICE_AVAILABLE = True
    except ImportError:
        MARKET_DATA_SERVICE_AVAILABLE = False
        # Define placeholder class if market data service is not available
        class MarketDataService:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            
            def register_price_update_callback(self, *args: Any, **kwargs: Any) -> None:
                pass

logger = logging.getLogger(__name__)

class MultiPortfolioIntegration:
    """Integration class for managing multiple portfolios.
    
    This class extends the system to support multiple portfolios simultaneously,
    with portfolio isolation, portfolio-specific event routing, portfolio grouping,
    and cross-portfolio analysis capabilities.
    """
    
    def __init__(self, 
                 event_system: Optional[EventSystem] = None,
                 trading_engine: Optional[TradingEngine] = None,
                 market_data_service: Optional[MarketDataService] = None,
                 default_config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-portfolio integration.
        
        Args:
            event_system: Optional event system for event-based communication
            trading_engine: Optional trading engine for trade execution
            market_data_service: Optional market data service for price updates
            default_config: Default configuration for new portfolios
        """
        self.event_system = event_system
        self.trading_engine = trading_engine
        self.market_data_service = market_data_service
        
        # Initialize the portfolio registry
        self.registry = PortfolioRegistry(default_config=default_config)
        
        # Dictionary to store portfolio integrations
        self.portfolio_integrations: Dict[str, PortfolioIntegration] = {}
        
        # Dictionary to store event subscriptions
        self.subscriptions: Dict[str, List[Any]] = {}
        
        # Set up event subscriptions if event system is available
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            self._setup_event_subscriptions()
        
        logger.info("MultiPortfolioIntegration initialized")
    
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for the multi-portfolio system."""
        if not self.event_system or not EVENT_SYSTEM_AVAILABLE:
            logger.warning("Event system not available, skipping event subscriptions")
            return
        
        # Subscribe to portfolio-related events
        self.subscriptions["portfolio_create"] = [
            self.event_system.subscribe(
                "PORTFOLIO_CREATE",
                self._handle_portfolio_create_event
            )
        ]
        
        self.subscriptions["portfolio_delete"] = [
            self.event_system.subscribe(
                "PORTFOLIO_DELETE",
                self._handle_portfolio_delete_event
            )
        ]
        
        self.subscriptions["portfolio_activate"] = [
            self.event_system.subscribe(
                "PORTFOLIO_ACTIVATE",
                self._handle_portfolio_activate_event
            )
        ]
        
        self.subscriptions["portfolio_deactivate"] = [
            self.event_system.subscribe(
                "PORTFOLIO_DEACTIVATE",
                self._handle_portfolio_deactivate_event
            )
        ]
        
        # Subscribe to group-related events
        self.subscriptions["group_create"] = [
            self.event_system.subscribe(
                "GROUP_CREATE",
                self._handle_group_create_event
            )
        ]
        
        self.subscriptions["group_delete"] = [
            self.event_system.subscribe(
                "GROUP_DELETE",
                self._handle_group_delete_event
            )
        ]
        
        # Subscribe to cross-portfolio analysis events
        self.subscriptions["cross_portfolio_analysis"] = [
            self.event_system.subscribe(
                "CROSS_PORTFOLIO_ANALYSIS",
                self._handle_cross_portfolio_analysis_event
            )
        ]
        
        # Subscribe to consolidated report events
        self.subscriptions["consolidated_report"] = [
            self.event_system.subscribe(
                "CONSOLIDATED_REPORT_REQUEST",
                self._handle_consolidated_report_request_event
            )
        ]
        
        logger.info("Event subscriptions set up for multi-portfolio integration")
    
    def cleanup_subscriptions(self) -> None:
        """Clean up all event subscriptions."""
        if not self.event_system:
            return
        
        for subscription_type, subscription_list in self.subscriptions.items():
            for subscription in subscription_list:
                try:
                    self.event_system.unsubscribe(subscription)
                except Exception as e:
                    logger.error(f"Error unsubscribing from {subscription_type}: {e}")
        
        # Clean up subscriptions for individual portfolio integrations
        for portfolio_id, integration in self.portfolio_integrations.items():
            try:
                integration.cleanup_subscriptions()
            except Exception as e:
                logger.error(f"Error cleaning up subscriptions for portfolio {portfolio_id}: {e}")
        
        self.subscriptions.clear()
        logger.info("All event subscriptions cleaned up")
    
    def create_portfolio(self, 
                        portfolio_id: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None,
                        initial_capital: float = 100000.0,
                        name: Optional[str] = None,
                        description: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        auto_activate: bool = True) -> str:
        """Create a new portfolio and its integration.
        
        Args:
            portfolio_id: Optional unique identifier (generated if not provided)
            config: Configuration for the portfolio
            initial_capital: Initial capital for the portfolio
            name: Display name for the portfolio
            description: Optional description of the portfolio
            tags: Optional list of tags for categorization
            auto_activate: Whether to automatically activate the portfolio
            
        Returns:
            Portfolio ID of the created portfolio
        """
        # Create the portfolio in the registry
        portfolio_id = self.registry.create_portfolio(
            portfolio_id=portfolio_id,
            config=config,
            initial_capital=initial_capital,
            name=name,
            description=description,
            tags=tags,
            auto_activate=auto_activate
        )
        
        # Get the portfolio system components
        portfolio_system = self.registry.get_portfolio_system(portfolio_id)
        
        # Create a portfolio integration for this portfolio
        integration = PortfolioIntegration(
            portfolio_manager=portfolio_system["portfolio_manager"],
            performance_calculator=portfolio_system["performance_calculator"],
            tax_manager=portfolio_system["tax_manager"],
            allocation_manager=portfolio_system["allocation_manager"],
            risk_manager=portfolio_system.get("risk_manager"),
            event_system=self.event_system,
            trading_engine=self.trading_engine,
            market_data_service=self.market_data_service
        )
        
        # Store the integration
        self.portfolio_integrations[portfolio_id] = integration
        
        # Publish portfolio creation event if event system is available
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            self.event_system.publish(
                Event(
                    event_type="PORTFOLIO_CREATED",
                    data={
                        "portfolio_id": portfolio_id,
                        "name": name or f"Portfolio {portfolio_id}",
                        "timestamp": datetime.now()
                    }
                )
            )
        
        logger.info(f"Created portfolio and integration: {name or portfolio_id} (ID: {portfolio_id})")
        return portfolio_id
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """Delete a portfolio and its integration.
        
        Args:
            portfolio_id: ID of the portfolio to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        # Check if portfolio exists
        if portfolio_id not in self.registry.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found for deletion")
            return False
        
        # Clean up the portfolio integration
        if portfolio_id in self.portfolio_integrations:
            try:
                self.portfolio_integrations[portfolio_id].cleanup_subscriptions()
                del self.portfolio_integrations[portfolio_id]
            except Exception as e:
                logger.error(f"Error cleaning up portfolio integration for {portfolio_id}: {e}")
        
        # Delete the portfolio from the registry
        success = self.registry.delete_portfolio(portfolio_id)
        
        # Publish portfolio deletion event if event system is available
        if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
            self.event_system.publish(
                Event(
                    event_type="PORTFOLIO_DELETED",
                    data={
                        "portfolio_id": portfolio_id,
                        "timestamp": datetime.now()
                    }
                )
            )
        
        logger.info(f"Deleted portfolio and integration: {portfolio_id}")
        return success
    
    def activate_portfolio(self, portfolio_id: str) -> bool:
        """Activate a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to activate
            
        Returns:
            True if activation was successful, False otherwise
        """
        # Check if portfolio exists
        if portfolio_id not in self.registry.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found for activation")
            return False
        
        # Activate the portfolio in the registry
        success = self.registry.activate_portfolio(portfolio_id)
        
        # Publish portfolio activation event if event system is available
        if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
            self.event_system.publish(
                Event(
                    event_type="PORTFOLIO_ACTIVATED",
                    data={
                        "portfolio_id": portfolio_id,
                        "timestamp": datetime.now()
                    }
                )
            )
        
        logger.info(f"Activated portfolio: {portfolio_id}")
        return success
    
    def deactivate_portfolio(self, portfolio_id: str) -> bool:
        """Deactivate a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to deactivate
            
        Returns:
            True if deactivation was successful, False otherwise
        """
        # Check if portfolio exists
        if portfolio_id not in self.registry.portfolios:
            logger.warning(f"Portfolio {portfolio_id} not found for deactivation")
            return False
        
        # Deactivate the portfolio in the registry
        success = self.registry.deactivate_portfolio(portfolio_id)
        
        # Publish portfolio deactivation event if event system is available
        if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
            self.event_system.publish(
                Event(
                    event_type="PORTFOLIO_DEACTIVATED",
                    data={
                        "portfolio_id": portfolio_id,
                        "timestamp": datetime.now()
                    }
                )
            )
        
        logger.info(f"Deactivated portfolio: {portfolio_id}")
        return success
    
    def set_active_portfolio(self, portfolio_id: str) -> bool:
        """Set the active portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to set as active
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.registry.set_active_portfolio(portfolio_id)
            
            # Publish active portfolio changed event if event system is available
            if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="ACTIVE_PORTFOLIO_CHANGED",
                        data={
                            "portfolio_id": portfolio_id,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Set active portfolio: {portfolio_id}")
            return success
        except (KeyError, ValueError) as e:
            logger.error(f"Error setting active portfolio {portfolio_id}: {e}")
            return False
    
    def get_active_portfolio_id(self) -> Optional[str]:
        """Get the ID of the currently active portfolio.
        
        Returns:
            ID of the active portfolio, or None if no portfolio is active
        """
        return self.registry.get_active_portfolio_id()
    
    def get_portfolio_integration(self, portfolio_id: Optional[str] = None) -> Optional[PortfolioIntegration]:
        """Get the portfolio integration for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Portfolio integration instance, or None if not found
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.registry.get_active_portfolio_id()
            if portfolio_id is None:
                logger.warning("No active portfolio")
                return None
        
        # Return the portfolio integration if it exists
        if portfolio_id in self.portfolio_integrations:
            return self.portfolio_integrations[portfolio_id]
        
        logger.warning(f"Portfolio integration for {portfolio_id} not found")
        return None
    
    def create_portfolio_group(self, 
                              name: str, 
                              portfolio_ids: Optional[List[str]] = None,
                              group_id: Optional[str] = None,
                              description: Optional[str] = None,
                              allocation: Optional[Dict[str, float]] = None) -> str:
        """Create a new portfolio group.
        
        Args:
            name: Display name for the group
            portfolio_ids: Optional list of portfolio IDs to include
            group_id: Optional unique identifier (generated if not provided)
            description: Optional description of the group
            allocation: Optional dictionary mapping portfolio IDs to allocations
            
        Returns:
            Group ID of the created group
        """
        # Create the group in the registry
        group_id = self.registry.create_portfolio_group(
            name=name,
            portfolio_ids=portfolio_ids,
            group_id=group_id,
            description=description,
            allocation=allocation
        )
        
        # Publish group creation event if event system is available
        if self.event_system and EVENT_SYSTEM_AVAILABLE:
            self.event_system.publish(
                Event(
                    event_type="GROUP_CREATED",
                    data={
                        "group_id": group_id,
                        "name": name,
                        "portfolio_ids": portfolio_ids or [],
                        "timestamp": datetime.now()
                    }
                )
            )
        
        logger.info(f"Created portfolio group: {name} (ID: {group_id})")
        return group_id
    
    def delete_portfolio_group(self, group_id: str) -> bool:
        """Delete a portfolio group.
        
        Args:
            group_id: ID of the group to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Delete the group from the registry
            success = self.registry.delete_portfolio_group(group_id)
            
            # Publish group deletion event if event system is available
            if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="GROUP_DELETED",
                        data={
                            "group_id": group_id,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Deleted portfolio group: {group_id}")
            return success
        except KeyError as e:
            logger.error(f"Error deleting portfolio group {group_id}: {e}")
            return False
    
    def add_to_group(self, group_id: str, portfolio_id: str, allocation: Optional[float] = None) -> bool:
        """Add a portfolio to a group.
        
        Args:
            group_id: ID of the group
            portfolio_id: ID of the portfolio to add
            allocation: Optional target allocation for this portfolio
            
        Returns:
            True if addition was successful, False otherwise
        """
        try:
            # Add the portfolio to the group in the registry
            success = self.registry.add_to_group(group_id, portfolio_id, allocation)
            
            # Publish portfolio added to group event if event system is available
            if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="PORTFOLIO_ADDED_TO_GROUP",
                        data={
                            "group_id": group_id,
                            "portfolio_id": portfolio_id,
                            "allocation": allocation,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Added portfolio {portfolio_id} to group {group_id}")
            return success
        except KeyError as e:
            logger.error(f"Error adding portfolio {portfolio_id} to group {group_id}: {e}")
            return False
    
    def remove_from_group(self, group_id: str, portfolio_id: str) -> bool:
        """Remove a portfolio from a group.
        
        Args:
            group_id: ID of the group
            portfolio_id: ID of the portfolio to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        try:
            # Remove the portfolio from the group in the registry
            success = self.registry.remove_from_group(group_id, portfolio_id)
            
            # Publish portfolio removed from group event if event system is available
            if success and self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="PORTFOLIO_REMOVED_FROM_GROUP",
                        data={
                            "group_id": group_id,
                            "portfolio_id": portfolio_id,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Removed portfolio {portfolio_id} from group {group_id}")
            return success
        except KeyError as e:
            logger.error(f"Error removing portfolio {portfolio_id} from group {group_id}: {e}")
            return False
    
    def compare_portfolios(self, portfolio_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple portfolios based on specified metrics.
        
        Args:
            portfolio_ids: List of portfolio IDs to compare
            metrics: Optional list of metrics to compare (default: all available)
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Compare portfolios using the registry
            comparison = self.registry.compare_portfolios(portfolio_ids, metrics)
            
            # Publish comparison results event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="PORTFOLIO_COMPARISON_RESULTS",
                        data={
                            "portfolio_ids": portfolio_ids,
                            "metrics": metrics,
                            "results": comparison,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Compared {len(portfolio_ids)} portfolios on {len(metrics or [])} metrics")
            return comparison
        except KeyError as e:
            logger.error(f"Error comparing portfolios: {e}")
            return {"error": str(e)}
    
    def calculate_correlation(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Calculate correlation between multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to analyze
            
        Returns:
            Dictionary containing correlation matrix
        """
        try:
            # Calculate correlation using the registry
            correlation = self.registry.calculate_correlation(portfolio_ids)
            
            # Publish correlation results event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="PORTFOLIO_CORRELATION_RESULTS",
                        data={
                            "portfolio_ids": portfolio_ids,
                            "results": correlation,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Calculated correlation for {len(portfolio_ids)} portfolios")
            return correlation
        except KeyError as e:
            logger.error(f"Error calculating correlation: {e}")
            return {"error": str(e)}
    
    def analyze_diversification(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Analyze diversification across multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to analyze
            
        Returns:
            Dictionary containing diversification metrics
        """
        try:
            # Analyze diversification using the registry
            diversification = self.registry.analyze_diversification(portfolio_ids)
            
            # Publish diversification results event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="PORTFOLIO_DIVERSIFICATION_RESULTS",
                        data={
                            "portfolio_ids": portfolio_ids,
                            "results": diversification,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Analyzed diversification for {len(portfolio_ids)} portfolios")
            return diversification
        except KeyError as e:
            logger.error(f"Error analyzing diversification: {e}")
            return {"error": str(e)}
    
    def generate_consolidated_report(self, portfolio_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a consolidated report for multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to include (default: all active)
            
        Returns:
            Dictionary containing consolidated report data
        """
        try:
            # Generate consolidated report using the registry
            report = self.registry.generate_consolidated_report(portfolio_ids)
            
            # Publish consolidated report event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="CONSOLIDATED_REPORT",
                        data={
                            "portfolio_ids": portfolio_ids or list(self.registry.active_portfolios),
                            "report": report,
                            "timestamp": datetime.now()
                        }
                    )
                )
            
            logger.info(f"Generated consolidated report for {report['portfolio_count']} portfolios")
            return report
        except Exception as e:
            logger.error(f"Error generating consolidated report: {e}")
            return {"error": str(e)}
    
    # Event handlers
    def _handle_portfolio_create_event(self, event: Event) -> None:
        """Handle portfolio creation events.
        
        Args:
            event: The portfolio creation event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid portfolio create event data")
            return
        
        try:
            # Extract portfolio creation parameters from event data
            portfolio_id = event.data.get("portfolio_id")
            config = event.data.get("config")
            initial_capital = event.data.get("initial_capital", 100000.0)
            name = event.data.get("name")
            description = event.data.get("description")
            tags = event.data.get("tags")
            auto_activate = event.data.get("auto_activate", True)
            
            # Create the portfolio
            self.create_portfolio(
                portfolio_id=portfolio_id,
                config=config,
                initial_capital=initial_capital,
                name=name,
                description=description,
                tags=tags,
                auto_activate=auto_activate
            )
        except Exception as e:
            logger.error(f"Error handling portfolio create event: {e}")
    
    def _handle_portfolio_delete_event(self, event: Event) -> None:
        """Handle portfolio deletion events.
        
        Args:
            event: The portfolio deletion event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid portfolio delete event data")
            return
        
        try:
            # Extract portfolio ID from event data
            portfolio_id = event.data.get("portfolio_id")
            if not portfolio_id:
                logger.warning("Portfolio ID not provided in delete event")
                return
            
            # Delete the portfolio
            self.delete_portfolio(portfolio_id)
        except Exception as e:
            logger.error(f"Error handling portfolio delete event: {e}")
    
    def _handle_portfolio_activate_event(self, event: Event) -> None:
        """Handle portfolio activation events.
        
        Args:
            event: The portfolio activation event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid portfolio activate event data")
            return
        
        try:
            # Extract portfolio ID from event data
            portfolio_id = event.data.get("portfolio_id")
            if not portfolio_id:
                logger.warning("Portfolio ID not provided in activate event")
                return
            
            # Activate the portfolio
            self.activate_portfolio(portfolio_id)
        except Exception as e:
            logger.error(f"Error handling portfolio activate event: {e}")
    
    def _handle_portfolio_deactivate_event(self, event: Event) -> None:
        """Handle portfolio deactivation events.
        
        Args:
            event: The portfolio deactivation event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid portfolio deactivate event data")
            return
        
        try:
            # Extract portfolio ID from event data
            portfolio_id = event.data.get("portfolio_id")
            if not portfolio_id:
                logger.warning("Portfolio ID not provided in deactivate event")
                return
            
            # Deactivate the portfolio
            self.deactivate_portfolio(portfolio_id)
        except Exception as e:
            logger.error(f"Error handling portfolio deactivate event: {e}")
    
    def _handle_group_create_event(self, event: Event) -> None:
        """Handle group creation events.
        
        Args:
            event: The group creation event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid group create event data")
            return
        
        try:
            # Extract group creation parameters from event data
            name = event.data.get("name")
            if not name:
                logger.warning("Group name not provided in create event")
                return
            
            portfolio_ids = event.data.get("portfolio_ids")
            group_id = event.data.get("group_id")
            description = event.data.get("description")
            allocation = event.data.get("allocation")
            
            # Create the group
            self.create_portfolio_group(
                name=name,
                portfolio_ids=portfolio_ids,
                group_id=group_id,
                description=description,
                allocation=allocation
            )
        except Exception as e:
            logger.error(f"Error handling group create event: {e}")
    
    def _handle_group_delete_event(self, event: Event) -> None:
        """Handle group deletion events.
        
        Args:
            event: The group deletion event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid group delete event data")
            return
        
        try:
            # Extract group ID from event data
            group_id = event.data.get("group_id")
            if not group_id:
                logger.warning("Group ID not provided in delete event")
                return
            
            # Delete the group
            self.delete_portfolio_group(group_id)
        except Exception as e:
            logger.error(f"Error handling group delete event: {e}")
    
    def _handle_cross_portfolio_analysis_event(self, event: Event) -> None:
        """Handle cross-portfolio analysis events.
        
        Args:
            event: The cross-portfolio analysis event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid cross-portfolio analysis event data")
            return
        
        try:
            # Extract analysis parameters from event data
            analysis_type = event.data.get("analysis_type")
            if not analysis_type:
                logger.warning("Analysis type not provided in event")
                return
            
            portfolio_ids = event.data.get("portfolio_ids")
            if not portfolio_ids:
                logger.warning("Portfolio IDs not provided in event")
                return
            
            # Perform the requested analysis
            if analysis_type == "comparison":
                metrics = event.data.get("metrics")
                results = self.compare_portfolios(portfolio_ids, metrics)
            elif analysis_type == "correlation":
                results = self.calculate_correlation(portfolio_ids)
            elif analysis_type == "diversification":
                results = self.analyze_diversification(portfolio_ids)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return
            
            # Publish analysis results event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type=f"CROSS_PORTFOLIO_ANALYSIS_RESULTS",
                        data={
                            "analysis_type": analysis_type,
                            "portfolio_ids": portfolio_ids,
                            "results": results,
                            "timestamp": datetime.now()
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error handling cross-portfolio analysis event: {e}")
    
    def _handle_consolidated_report_request_event(self, event: Event) -> None:
        """Handle consolidated report request events.
        
        Args:
            event: The consolidated report request event
        """
        if not hasattr(event, "data") or not isinstance(event.data, dict):
            logger.warning("Invalid consolidated report request event data")
            return
        
        try:
            # Extract report parameters from event data
            portfolio_ids = event.data.get("portfolio_ids")
            
            # Generate the consolidated report
            report = self.generate_consolidated_report(portfolio_ids)
            
            # Publish consolidated report event if event system is available
            if self.event_system and EVENT_SYSTEM_AVAILABLE:
                self.event_system.publish(
                    Event(
                        event_type="CONSOLIDATED_REPORT",
                        data={
                            "portfolio_ids": portfolio_ids or list(self.registry.active_portfolios),
                            "report": report,
                            "timestamp": datetime.now()
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error handling consolidated report request event: {e}")