"""Portfolio Registry for managing multiple portfolio instances.

This module provides the PortfolioRegistry class, which is responsible for managing
multiple portfolio instances, including creation, activation, deactivation, and deletion
of portfolios, as well as portfolio grouping and cross-portfolio operations.
"""

from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime
import logging
import uuid

from .portfolio_factory import PortfolioFactory
from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager
from .allocation_manager import AllocationManager

# Try to import risk components
try:
    from ..risk.portfolio_risk_manager import PortfolioRiskManager
    RISK_MODULE_AVAILABLE = True
except ImportError:
    try:
        from risk.portfolio_risk_manager import PortfolioRiskManager
        RISK_MODULE_AVAILABLE = True
    except ImportError:
        RISK_MODULE_AVAILABLE = False
        # Define placeholder class if risk module is not available
        class PortfolioRiskManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

logger = logging.getLogger(__name__)

class PortfolioGroup:
    """Class representing a group of portfolios.
    
    This class provides functionality for managing a collection of portfolios,
    including aggregated performance calculation and group-level operations.
    """
    
    def __init__(self, group_id: str, name: str, description: Optional[str] = None):
        """Initialize a portfolio group.
        
        Args:
            group_id: Unique identifier for the group
            name: Display name for the group
            description: Optional description of the group
        """
        self.group_id = group_id
        self.name = name
        self.description = description
        self.portfolio_ids: Set[str] = set()
        self.allocation: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
        logger.info(f"Portfolio group created: {name} (ID: {group_id})")
    
    def add_portfolio(self, portfolio_id: str, allocation: Optional[float] = None) -> None:
        """Add a portfolio to the group.
        
        Args:
            portfolio_id: ID of the portfolio to add
            allocation: Optional target allocation for this portfolio within the group
        """
        self.portfolio_ids.add(portfolio_id)
        if allocation is not None:
            self.allocation[portfolio_id] = allocation
        self.updated_at = datetime.now()
        logger.info(f"Added portfolio {portfolio_id} to group {self.name}")
    
    def remove_portfolio(self, portfolio_id: str) -> None:
        """Remove a portfolio from the group.
        
        Args:
            portfolio_id: ID of the portfolio to remove
        """
        if portfolio_id in self.portfolio_ids:
            self.portfolio_ids.remove(portfolio_id)
            if portfolio_id in self.allocation:
                del self.allocation[portfolio_id]
            self.updated_at = datetime.now()
            logger.info(f"Removed portfolio {portfolio_id} from group {self.name}")
        else:
            logger.warning(f"Portfolio {portfolio_id} not found in group {self.name}")
    
    def set_allocation(self, portfolio_id: str, allocation: float) -> None:
        """Set the target allocation for a portfolio within the group.
        
        Args:
            portfolio_id: ID of the portfolio
            allocation: Target allocation (0.0 to 1.0)
            
        Raises:
            ValueError: If allocation is not between 0 and 1
            KeyError: If portfolio_id is not in the group
        """
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")
            
        if portfolio_id not in self.portfolio_ids:
            raise KeyError(f"Portfolio {portfolio_id} not in group {self.name}")
            
        self.allocation[portfolio_id] = allocation
        self.updated_at = datetime.now()
        logger.info(f"Set allocation for portfolio {portfolio_id} to {allocation:.2%} in group {self.name}")
    
    def get_portfolio_ids(self) -> List[str]:
        """Get the list of portfolio IDs in this group.
        
        Returns:
            List of portfolio IDs
        """
        return list(self.portfolio_ids)
    
    def get_allocation(self) -> Dict[str, float]:
        """Get the current allocation map for portfolios in this group.
        
        Returns:
            Dictionary mapping portfolio IDs to their allocations
        """
        return self.allocation.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the group to a dictionary representation.
        
        Returns:
            Dictionary representation of the group
        """
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "portfolio_ids": list(self.portfolio_ids),
            "allocation": self.allocation,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class PortfolioRegistry:
    """Registry for managing multiple portfolio instances.
    
    This class provides functionality for:
    - Creating and managing multiple portfolio instances
    - Portfolio lifecycle management (activation, deactivation, deletion)
    - Portfolio grouping and organization
    - Cross-portfolio operations and analysis
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """Initialize the portfolio registry.
        
        Args:
            default_config: Default configuration for new portfolios
        """
        self.portfolios: Dict[str, Dict[str, Any]] = {}
        self.portfolio_systems: Dict[str, Dict[str, Any]] = {}
        self.active_portfolios: Set[str] = set()
        self.groups: Dict[str, PortfolioGroup] = {}
        self.default_config = default_config or {}
        self.factory = PortfolioFactory(self.default_config)
        self.active_portfolio_id: Optional[str] = None
        
        logger.info("Portfolio Registry initialized")
    
    def create_portfolio(self, 
                        portfolio_id: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None,
                        initial_capital: float = 100000.0,
                        name: Optional[str] = None,
                        description: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        auto_activate: bool = True) -> str:
        """Create a new portfolio instance.
        
        Args:
            portfolio_id: Optional unique identifier (generated if not provided)
            config: Configuration for the portfolio (merged with default_config)
            initial_capital: Initial capital for the portfolio
            name: Display name for the portfolio
            description: Optional description of the portfolio
            tags: Optional list of tags for categorization
            auto_activate: Whether to automatically activate the portfolio
            
        Returns:
            Portfolio ID of the created portfolio
            
        Raises:
            ValueError: If portfolio_id already exists
        """
        # Generate portfolio ID if not provided
        if portfolio_id is None:
            portfolio_id = str(uuid.uuid4())
        elif portfolio_id in self.portfolios:
            raise ValueError(f"Portfolio with ID {portfolio_id} already exists")
        
        # Merge with default config
        merged_config = self.default_config.copy()
        if config:
            # Deep merge the configurations
            for section, section_config in config.items():
                if section in merged_config and isinstance(merged_config[section], dict):
                    merged_config[section].update(section_config)
                else:
                    merged_config[section] = section_config
        
        # Ensure portfolio_id is set in the config
        if "portfolio_manager" not in merged_config:
            merged_config["portfolio_manager"] = {}
        merged_config["portfolio_manager"]["portfolio_id"] = portfolio_id
        
        # Create the portfolio system using the factory
        portfolio_system = self.factory.create_complete_portfolio_system(initial_capital=initial_capital)
        
        # Store the portfolio system
        self.portfolio_systems[portfolio_id] = portfolio_system
        
        # Store portfolio metadata
        self.portfolios[portfolio_id] = {
            "portfolio_id": portfolio_id,
            "name": name or f"Portfolio {portfolio_id}",
            "description": description or "",
            "tags": tags or [],
            "config": merged_config,
            "initial_capital": initial_capital,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "active": False,
            "metadata": {}
        }
        
        # Activate if requested
        if auto_activate:
            self.activate_portfolio(portfolio_id)
        
        logger.info(f"Created portfolio: {name or portfolio_id} (ID: {portfolio_id})")
        return portfolio_id
    
    def activate_portfolio(self, portfolio_id: str) -> bool:
        """Activate a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to activate
            
        Returns:
            True if activation was successful, False otherwise
            
        Raises:
            KeyError: If portfolio_id does not exist
        """
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        self.active_portfolios.add(portfolio_id)
        self.portfolios[portfolio_id]["active"] = True
        self.portfolios[portfolio_id]["updated_at"] = datetime.now()
        
        # Set as active portfolio if none is currently active
        if self.active_portfolio_id is None:
            self.active_portfolio_id = portfolio_id
        
        logger.info(f"Activated portfolio: {portfolio_id}")
        return True
    
    def deactivate_portfolio(self, portfolio_id: str) -> bool:
        """Deactivate a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to deactivate
            
        Returns:
            True if deactivation was successful, False otherwise
            
        Raises:
            KeyError: If portfolio_id does not exist
        """
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        if portfolio_id in self.active_portfolios:
            self.active_portfolios.remove(portfolio_id)
        
        self.portfolios[portfolio_id]["active"] = False
        self.portfolios[portfolio_id]["updated_at"] = datetime.now()
        
        # Clear active portfolio if this was the active one
        if self.active_portfolio_id == portfolio_id:
            self.active_portfolio_id = None
            # Set another portfolio as active if available
            if self.active_portfolios:
                self.active_portfolio_id = next(iter(self.active_portfolios))
        
        logger.info(f"Deactivated portfolio: {portfolio_id}")
        return True
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """Delete a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to delete
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            KeyError: If portfolio_id does not exist
        """
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        # Deactivate first
        if portfolio_id in self.active_portfolios:
            self.deactivate_portfolio(portfolio_id)
        
        # Remove from all groups
        for group in self.groups.values():
            if portfolio_id in group.portfolio_ids:
                group.remove_portfolio(portfolio_id)
        
        # Delete the portfolio
        del self.portfolios[portfolio_id]
        if portfolio_id in self.portfolio_systems:
            del self.portfolio_systems[portfolio_id]
        
        logger.info(f"Deleted portfolio: {portfolio_id}")
        return True
    
    def set_active_portfolio(self, portfolio_id: str) -> bool:
        """Set the active portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to set as active
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If portfolio is not active
        """
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        if portfolio_id not in self.active_portfolios:
            raise ValueError(f"Portfolio {portfolio_id} is not active")
        
        self.active_portfolio_id = portfolio_id
        logger.info(f"Set active portfolio: {portfolio_id}")
        return True
    
    def get_active_portfolio_id(self) -> Optional[str]:
        """Get the ID of the currently active portfolio.
        
        Returns:
            ID of the active portfolio, or None if no portfolio is active
        """
        return self.active_portfolio_id
    
    def get_portfolio_manager(self, portfolio_id: Optional[str] = None) -> PortfolioManager:
        """Get the portfolio manager for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Portfolio manager instance
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id]["portfolio_manager"]
    
    def get_performance_calculator(self, portfolio_id: Optional[str] = None) -> PerformanceCalculator:
        """Get the performance calculator for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Performance calculator instance
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id]["performance_calculator"]
    
    def get_tax_manager(self, portfolio_id: Optional[str] = None) -> TaxManager:
        """Get the tax manager for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Tax manager instance
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id]["tax_manager"]
    
    def get_allocation_manager(self, portfolio_id: Optional[str] = None) -> AllocationManager:
        """Get the allocation manager for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Allocation manager instance
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id]["allocation_manager"]
    
    def get_risk_manager(self, portfolio_id: Optional[str] = None) -> Optional[PortfolioRiskManager]:
        """Get the risk manager for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Risk manager instance, or None if not available
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id].get("risk_manager")
    
    def get_portfolio_system(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the complete portfolio system for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Dictionary containing all portfolio components
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolio_systems:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolio_systems[portfolio_id]
    
    def get_portfolio_metadata(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific portfolio.
        
        Args:
            portfolio_id: ID of the portfolio (uses active portfolio if None)
            
        Returns:
            Dictionary containing portfolio metadata
            
        Raises:
            KeyError: If portfolio_id does not exist
            ValueError: If no active portfolio and portfolio_id is None
        """
        # Use active portfolio if portfolio_id is None
        if portfolio_id is None:
            portfolio_id = self.active_portfolio_id
            if portfolio_id is None:
                raise ValueError("No active portfolio")
        
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        return self.portfolios[portfolio_id].copy()
    
    def get_all_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all portfolios.
        
        Returns:
            Dictionary mapping portfolio IDs to their metadata
        """
        return {pid: meta.copy() for pid, meta in self.portfolios.items()}
    
    def get_active_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all active portfolios.
        
        Returns:
            Dictionary mapping portfolio IDs to their metadata
        """
        return {pid: meta.copy() for pid, meta in self.portfolios.items() 
                if pid in self.active_portfolios}
    
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
            
        Raises:
            ValueError: If group_id already exists
            KeyError: If any portfolio_id does not exist
        """
        # Generate group ID if not provided
        if group_id is None:
            group_id = str(uuid.uuid4())
        elif group_id in self.groups:
            raise ValueError(f"Group with ID {group_id} already exists")
        
        # Create the group
        group = PortfolioGroup(group_id, name, description)
        
        # Add portfolios if provided
        if portfolio_ids:
            for pid in portfolio_ids:
                if pid not in self.portfolios:
                    raise KeyError(f"Portfolio {pid} not found")
                group.add_portfolio(pid)
        
        # Set allocations if provided
        if allocation:
            for pid, alloc in allocation.items():
                if pid not in self.portfolios:
                    raise KeyError(f"Portfolio {pid} not found")
                if pid not in group.portfolio_ids:
                    group.add_portfolio(pid)
                group.set_allocation(pid, alloc)
        
        # Store the group
        self.groups[group_id] = group
        
        logger.info(f"Created portfolio group: {name} (ID: {group_id})")
        return group_id
    
    def delete_portfolio_group(self, group_id: str) -> bool:
        """Delete a portfolio group.
        
        Args:
            group_id: ID of the group to delete
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            KeyError: If group_id does not exist
        """
        if group_id not in self.groups:
            raise KeyError(f"Group {group_id} not found")
        
        del self.groups[group_id]
        logger.info(f"Deleted portfolio group: {group_id}")
        return True
    
    def add_to_group(self, group_id: str, portfolio_id: str, allocation: Optional[float] = None) -> bool:
        """Add a portfolio to a group.
        
        Args:
            group_id: ID of the group
            portfolio_id: ID of the portfolio to add
            allocation: Optional target allocation for this portfolio
            
        Returns:
            True if addition was successful, False otherwise
            
        Raises:
            KeyError: If group_id or portfolio_id does not exist
        """
        if group_id not in self.groups:
            raise KeyError(f"Group {group_id} not found")
        
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Portfolio {portfolio_id} not found")
        
        self.groups[group_id].add_portfolio(portfolio_id, allocation)
        return True
    
    def remove_from_group(self, group_id: str, portfolio_id: str) -> bool:
        """Remove a portfolio from a group.
        
        Args:
            group_id: ID of the group
            portfolio_id: ID of the portfolio to remove
            
        Returns:
            True if removal was successful, False otherwise
            
        Raises:
            KeyError: If group_id does not exist
        """
        if group_id not in self.groups:
            raise KeyError(f"Group {group_id} not found")
        
        self.groups[group_id].remove_portfolio(portfolio_id)
        return True
    
    def get_group(self, group_id: str) -> PortfolioGroup:
        """Get a portfolio group.
        
        Args:
            group_id: ID of the group
            
        Returns:
            Portfolio group instance
            
        Raises:
            KeyError: If group_id does not exist
        """
        if group_id not in self.groups:
            raise KeyError(f"Group {group_id} not found")
        
        return self.groups[group_id]
    
    def get_all_groups(self) -> Dict[str, PortfolioGroup]:
        """Get all portfolio groups.
        
        Returns:
            Dictionary mapping group IDs to group instances
        """
        return self.groups.copy()
    
    def get_group_portfolios(self, group_id: str) -> List[Dict[str, Any]]:
        """Get metadata for all portfolios in a group.
        
        Args:
            group_id: ID of the group
            
        Returns:
            List of portfolio metadata dictionaries
            
        Raises:
            KeyError: If group_id does not exist
        """
        if group_id not in self.groups:
            raise KeyError(f"Group {group_id} not found")
        
        group = self.groups[group_id]
        return [self.portfolios[pid] for pid in group.portfolio_ids 
                if pid in self.portfolios]
    
    def compare_portfolios(self, portfolio_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple portfolios based on specified metrics.
        
        Args:
            portfolio_ids: List of portfolio IDs to compare
            metrics: Optional list of metrics to compare (default: all available)
            
        Returns:
            Dictionary containing comparison results
            
        Raises:
            KeyError: If any portfolio_id does not exist
        """
        # Validate portfolio IDs
        for pid in portfolio_ids:
            if pid not in self.portfolios:
                raise KeyError(f"Portfolio {pid} not found")
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ["total_return", "annualized_return", "volatility", "sharpe_ratio", 
                      "max_drawdown", "alpha", "beta"]
        
        # Collect performance metrics for each portfolio
        comparison = {}
        for pid in portfolio_ids:
            portfolio_name = self.portfolios[pid]["name"]
            performance_calc = self.get_performance_calculator(pid)
            
            # Get performance metrics
            if hasattr(performance_calc, "get_performance_metrics"):
                perf_metrics = performance_calc.get_performance_metrics()
                comparison[pid] = {
                    "name": portfolio_name,
                    "metrics": {}
                }
                
                # Extract requested metrics
                for metric in metrics:
                    if metric in perf_metrics:
                        comparison[pid]["metrics"][metric] = perf_metrics[metric]
        
        logger.info(f"Compared {len(portfolio_ids)} portfolios on {len(metrics)} metrics")
        return comparison
    
    def calculate_correlation(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Calculate correlation between multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to analyze
            
        Returns:
            Dictionary containing correlation matrix
            
        Raises:
            KeyError: If any portfolio_id does not exist
        """
        # Validate portfolio IDs
        for pid in portfolio_ids:
            if pid not in self.portfolios:
                raise KeyError(f"Portfolio {pid} not found")
        
        # Collect returns for each portfolio
        returns_data = {}
        for pid in portfolio_ids:
            portfolio_name = self.portfolios[pid]["name"]
            portfolio_mgr = self.get_portfolio_manager(pid)
            
            # Get historical returns
            if hasattr(portfolio_mgr, "get_returns"):
                returns = portfolio_mgr.get_returns()
                returns_data[pid] = {
                    "name": portfolio_name,
                    "returns": returns
                }
        
        # Calculate correlation matrix if we have pandas
        try:
            import pandas as pd
            import numpy as np
            
            # Create DataFrame with returns
            returns_df = {}
            for pid, data in returns_data.items():
                returns_series = pd.Series({ts: ret for ts, ret in data["returns"]}, name=data["name"])
                returns_df[pid] = returns_series
            
            if returns_df:
                df = pd.DataFrame(returns_df)
                correlation_matrix = df.corr().to_dict()
                
                logger.info(f"Calculated correlation matrix for {len(portfolio_ids)} portfolios")
                return {
                    "correlation_matrix": correlation_matrix,
                    "portfolio_names": {pid: data["name"] for pid, data in returns_data.items()}
                }
        except ImportError:
            logger.warning("Pandas not available for correlation calculation")
        
        # Fallback if pandas is not available or no returns data
        return {
            "correlation_matrix": {},
            "portfolio_names": {pid: self.portfolios[pid]["name"] for pid in portfolio_ids}
        }
    
    def analyze_diversification(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Analyze diversification across multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to analyze
            
        Returns:
            Dictionary containing diversification metrics
            
        Raises:
            KeyError: If any portfolio_id does not exist
        """
        # Validate portfolio IDs
        for pid in portfolio_ids:
            if pid not in self.portfolios:
                raise KeyError(f"Portfolio {pid} not found")
        
        # Collect sector and asset class exposures
        sector_exposure = {}
        asset_class_exposure = {}
        symbol_overlap = set()
        all_symbols = set()
        
        for pid in portfolio_ids:
            portfolio_mgr = self.get_portfolio_manager(pid)
            positions = portfolio_mgr.get_positions()
            
            # Track symbols for overlap calculation
            portfolio_symbols = set(positions.keys())
            if not symbol_overlap and portfolio_symbols:  # First portfolio
                symbol_overlap = portfolio_symbols
            else:
                symbol_overlap &= portfolio_symbols  # Intersection
            
            all_symbols |= portfolio_symbols  # Union
            
            # Aggregate sector exposure
            for symbol, position in positions.items():
                sector = position.get("sector", "Unknown")
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += position.get("market_value", 0)
                
                asset_class = position.get("asset_class", "Unknown")
                if asset_class not in asset_class_exposure:
                    asset_class_exposure[asset_class] = 0
                asset_class_exposure[asset_class] += position.get("market_value", 0)
        
        # Calculate total exposure
        total_exposure = sum(sector_exposure.values())
        
        # Normalize exposures
        if total_exposure > 0:
            sector_exposure = {sector: value / total_exposure 
                             for sector, value in sector_exposure.items()}
            asset_class_exposure = {asset_class: value / total_exposure 
                                  for asset_class, value in asset_class_exposure.items()}
        
        # Calculate overlap percentage
        overlap_percentage = len(symbol_overlap) / len(all_symbols) if all_symbols else 0
        
        logger.info(f"Analyzed diversification across {len(portfolio_ids)} portfolios")
        return {
            "sector_exposure": sector_exposure,
            "asset_class_exposure": asset_class_exposure,
            "symbol_overlap": list(symbol_overlap),
            "overlap_percentage": overlap_percentage,
            "unique_symbols": len(all_symbols),
            "common_symbols": len(symbol_overlap)
        }
    
    def generate_consolidated_report(self, portfolio_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a consolidated report for multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio IDs to include (default: all active)
            
        Returns:
            Dictionary containing consolidated report data
        """
        # Use all active portfolios if not specified
        if portfolio_ids is None:
            portfolio_ids = list(self.active_portfolios)
        
        # Validate portfolio IDs
        valid_portfolio_ids = [pid for pid in portfolio_ids if pid in self.portfolios]
        
        # Collect portfolio data
        portfolios_data = []
        total_value = 0
        total_cash = 0
        
        for pid in valid_portfolio_ids:
            portfolio_mgr = self.get_portfolio_manager(pid)
            portfolio_value = portfolio_mgr.get_portfolio_value()
            cash = portfolio_mgr.cash
            
            total_value += portfolio_value
            total_cash += cash
            
            portfolios_data.append({
                "portfolio_id": pid,
                "name": self.portfolios[pid]["name"],
                "value": portfolio_value,
                "cash": cash,
                "positions_count": len(portfolio_mgr.get_positions())
            })
        
        # Calculate allocations
        if total_value > 0:
            for portfolio_data in portfolios_data:
                portfolio_data["allocation"] = portfolio_data["value"] / total_value
        
        logger.info(f"Generated consolidated report for {len(valid_portfolio_ids)} portfolios")
        return {
            "timestamp": datetime.now(),
            "total_value": total_value,
            "total_cash": total_cash,
            "cash_percentage": (total_cash / total_value) * 100 if total_value > 0 else 0,
            "portfolios": portfolios_data,
            "portfolio_count": len(portfolios_data)
        }