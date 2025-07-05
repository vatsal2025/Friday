import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager, TaxLotMethod
from .allocation_manager import AllocationManager, RebalanceMethod

# Optional import for risk management integration
# Set a flag to indicate if risk module is available
RISK_MODULE_AVAILABLE = False

# Define placeholder classes if risk module is not available
class RiskManagementFactory:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
        
    @staticmethod
    def create_risk_manager(*args: Any, **kwargs: Any) -> None:
        return None
        
    def create_advanced_risk_manager(*args: Any, **kwargs: Any) -> None:
        return None
        
class AdvancedRiskManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

logger = logging.getLogger(__name__)

class PortfolioFactory:
    """
    Factory class for creating and configuring portfolio management components.

    This class provides a centralized way to create and configure:
    - PortfolioManager
    - PerformanceCalculator
    - TaxManager
    - AllocationManager
    - Integration with RiskManagementFactory (if available)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Portfolio Factory.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        logger.info("Portfolio Factory initialized")

    def create_portfolio_manager(self,
                               portfolio_id: Optional[str] = None,
                               initial_capital: float = 0.0,
                               risk_manager: Optional[Any] = None) -> PortfolioManager:
        """
        Create a PortfolioManager instance.

        Args:
            portfolio_id: Identifier for the portfolio (optional)
            initial_capital: Initial capital amount
            risk_manager: Risk manager instance (optional)

        Returns:
            Configured PortfolioManager instance
        """
        # Get configuration
        portfolio_config = self.config.get("portfolio_manager", {})

        # Create the portfolio manager
        portfolio_manager = PortfolioManager(
            portfolio_id=portfolio_id or portfolio_config.get("portfolio_id"),
            initial_capital=initial_capital if initial_capital > 0.0 else portfolio_config.get("initial_cash", portfolio_config.get("initial_capital", 100000.0)),
            risk_manager=risk_manager
        )

        logger.info(f"Created PortfolioManager with ID: {portfolio_manager.portfolio_id}")
        return portfolio_manager

    def create_performance_calculator(self,
                                    benchmark_symbol: Optional[str] = None,
                                    risk_free_rate: Optional[float] = None) -> PerformanceCalculator:
        """
        Create a PerformanceCalculator instance.

        Args:
            benchmark_symbol: Symbol for benchmark comparison (optional)
            risk_free_rate: Risk-free rate for risk-adjusted metrics (optional)

        Returns:
            Configured PerformanceCalculator instance
        """
        # Get configuration
        performance_config = self.config.get("performance_calculator", {})

        # Create the performance calculator
        performance_calculator = PerformanceCalculator(
            benchmark_symbol=benchmark_symbol or performance_config.get("benchmark_symbol"),
            risk_free_rate=risk_free_rate if risk_free_rate is not None else performance_config.get("risk_free_rate", 0.0)
        )

        logger.info(f"Created PerformanceCalculator with benchmark: {performance_calculator.benchmark_symbol}")
        return performance_calculator

    def create_tax_manager(self,
                         default_method: Optional[TaxLotMethod] = None,
                         wash_sale_window_days: Optional[int] = None) -> TaxManager:
        """
        Create a TaxManager instance.

        Args:
            default_method: Default tax lot selection method (optional)
            wash_sale_window_days: Number of days to look for wash sales (optional)

        Returns:
            Configured TaxManager instance
        """
        # Get configuration
        tax_config = self.config.get("tax_manager", {})

        # Determine default method
        if default_method is None:
            method_str = tax_config.get("default_tax_lot_method", "FIFO")
            try:
                default_method = TaxLotMethod[method_str]
            except KeyError:
                logger.warning(f"Invalid tax lot method '{method_str}', using FIFO")
                default_method = TaxLotMethod.FIFO

        # Determine wash sale window
        if wash_sale_window_days is None:
            wash_sale_window_days = tax_config.get("wash_sale_window_days", 30)

        # Create the tax manager
        tax_manager = TaxManager(
            default_method=default_method,
            wash_sale_window_days=wash_sale_window_days
        )

        logger.info(f"Created TaxManager with default method: {default_method.name} and wash sale window: {wash_sale_window_days} days")
        return tax_manager

    def create_allocation_manager(self,
                                rebalance_method: Optional[RebalanceMethod] = None,
                                default_threshold: Optional[float] = None,
                                rebalance_frequency_days: Optional[int] = None) -> AllocationManager:
        """
        Create an AllocationManager instance.

        Args:
            rebalance_method: Method for rebalancing (optional)
            default_threshold: Default threshold percentage for rebalancing (optional)
            rebalance_frequency_days: Days between rebalances (optional)

        Returns:
            Configured AllocationManager instance
        """
        # Get configuration
        allocation_config = self.config.get("allocation_manager", {})

        # Determine rebalance method
        if rebalance_method is None:
            method_str = allocation_config.get("rebalance_method", "THRESHOLD")
            try:
                rebalance_method = RebalanceMethod[method_str]
            except KeyError:
                logger.warning(f"Invalid rebalance method '{method_str}', using THRESHOLD")
                rebalance_method = RebalanceMethod.THRESHOLD

        # Create the allocation manager
        allocation_manager = AllocationManager(
            rebalance_method=rebalance_method,
            default_threshold=default_threshold or allocation_config.get("default_threshold", 5.0),
            rebalance_frequency_days=rebalance_frequency_days or allocation_config.get("rebalance_frequency_days", 90)
        )

        # Set allocation targets if provided in config
        targets = allocation_config.get("allocation_targets", [])
        if targets:
            # Convert 'symbol' field to 'name' field if needed
            for target in targets:
                if "symbol" in target and "name" not in target:
                    target["name"] = target["symbol"]
                if "target" in target and "target_percentage" not in target:
                    target["target_percentage"] = target["target"]
            allocation_manager.set_multiple_allocation_targets(targets)

        logger.info(f"Created AllocationManager with method: {rebalance_method.name}")
        return allocation_manager

    def create_risk_manager(self, config_override: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create a risk manager instance using RiskManagementFactory if available.

        Args:
            config_override: Override configuration for risk manager (optional)

        Returns:
            Risk manager instance or None if risk module not available
        """
        if not RISK_MODULE_AVAILABLE:
            logger.warning("Risk management module not available")
            return None

        # Get configuration
        risk_config = self.config.get("risk_manager", {})

        if config_override:
            # Merge configurations with override taking precedence
            merged_config = risk_config.copy()
            merged_config.update(config_override)
            risk_config = merged_config

        # Create the risk management factory
        # Convert dictionary to RiskManagementProductionConfig if needed
        from src.risk.production_config import RiskManagementProductionConfig
        if isinstance(risk_config, dict):
            # Create a config object with default values
            config_obj = RiskManagementProductionConfig()
            # Update with values from the dictionary
            for key, value in risk_config.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            risk_config = config_obj
        
        risk_factory = RiskManagementFactory(risk_config)

        # Create the risk manager
        risk_manager = risk_factory.create_advanced_risk_manager()

        logger.info("Created risk manager using RiskManagementFactory")
        return risk_manager

    def create_complete_portfolio_system(self,
                                       portfolio_id: Optional[str] = None,
                                       initial_capital: float = 0.0,
                                       include_risk_manager: bool = True) -> Dict[str, Any]:
        """
        Create a complete portfolio management system with all components.

        Args:
            portfolio_id: Identifier for the portfolio (optional)
            initial_capital: Initial capital amount
            include_risk_manager: Whether to include a risk manager (if available)

        Returns:
            Dict containing all created components
        """
        # Create risk manager if requested and available
        risk_manager = None
        if include_risk_manager and RISK_MODULE_AVAILABLE:
            risk_manager = self.create_risk_manager()

        # Create all components
        portfolio_manager = self.create_portfolio_manager(
            portfolio_id=portfolio_id,
            initial_capital=initial_capital,
            risk_manager=risk_manager
        )

        performance_calculator = self.create_performance_calculator()
        tax_manager = self.create_tax_manager()
        allocation_manager = self.create_allocation_manager()

        # Return all components
        components = {
            "portfolio_manager": portfolio_manager,
            "performance_calculator": performance_calculator,
            "tax_manager": tax_manager,
            "allocation_manager": allocation_manager
        }

        if risk_manager:
            components["risk_manager"] = risk_manager

        logger.info(f"Created complete portfolio system with ID: {portfolio_manager.portfolio_id}")
        return components

    @staticmethod
    def load_config_from_file(config_file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_file_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        import json
        import yaml
        import os

        if not os.path.exists(config_file_path):
            logger.error(f"Configuration file not found: {config_file_path}")
            return {}

        file_ext = os.path.splitext(config_file_path)[1].lower()

        try:
            with open(config_file_path, 'r') as file:
                if file_ext == '.json':
                    config = json.load(file)
                elif file_ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(file)
                else:
                    logger.error(f"Unsupported configuration file format: {file_ext}")
                    return {}

            logger.info(f"Loaded configuration from {config_file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
