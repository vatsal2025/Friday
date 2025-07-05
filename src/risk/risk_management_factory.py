import logging
from typing import Dict, Any, Optional, List, Tuple

from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss_manager import StopLossManager
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.risk.risk_metrics_calculator import RiskMetricsCalculator
from src.risk.advanced_risk_manager import AdvancedRiskManager
from src.risk.production_config import RiskManagementProductionConfig, load_production_config
from src.risk.circuit_breaker import CircuitBreakerManager, CircuitBreaker, CircuitBreakerType, MarketWideCircuitBreaker, AccountCircuitBreaker

logger = logging.getLogger(__name__)

class RiskManagementFactory:
    """
    Factory class for creating production-ready risk management components.

    This factory creates and configures risk management components based on the
    production configuration, ensuring consistent setup across the application.
    """

    def __init__(self, config: Optional[RiskManagementProductionConfig] = None):
        """
        Initialize the risk management factory.

        Args:
            config: The production configuration to use. If None, loads the default.
        """
        self.config = config if config is not None else load_production_config()
        logger.info("Initialized RiskManagementFactory with production configuration")

    def create_position_sizer(self) -> PositionSizer:
        """
        Create a production-ready position sizer.

        Returns:
            PositionSizer: A configured position sizer.
        """
        position_sizer = PositionSizer(
            default_risk_per_trade=self.config.risk_per_trade,
            max_position_size_percentage=self.config.max_position_size_percentage,
            max_position_value=self.config.max_position_value,
            volatility_lookback_days=self.config.volatility_lookback_days,
            position_sizing_method=self.config.position_sizing_method
        )

        logger.info(f"Created production PositionSizer with method: {self.config.position_sizing_method}")
        return position_sizer

    def create_stop_loss_manager(self) -> StopLossManager:
        """
        Create a production-ready stop loss manager.

        Returns:
            StopLossManager: A configured stop loss manager.
        """
        stop_loss_manager = StopLossManager(
            default_stop_loss_percent=self.config.default_stop_loss_percent,
            default_trailing_percent=self.config.default_trailing_percent,
            default_atr_multiplier=self.config.default_atr_multiplier,
            default_time_stop_days=self.config.default_time_stop_days,
            default_profit_target_ratio=self.config.default_profit_target_ratio
        )

        logger.info("Created production StopLossManager")
        return stop_loss_manager

    def create_portfolio_risk_manager(self) -> PortfolioRiskManager:
        """
        Create a production-ready portfolio risk manager.

        Returns:
            PortfolioRiskManager: A configured portfolio risk manager.
        """
        portfolio_risk_manager = PortfolioRiskManager(
            max_portfolio_var_percent=self.config.max_portfolio_var_percent,
            max_drawdown_percent=self.config.max_drawdown_percent,
            max_sector_allocation=self.config.max_sector_allocation,
            max_position_size=self.config.max_position_size,
            max_history_size=self.config.max_history_size
        )

        logger.info("Created production PortfolioRiskManager")
        return portfolio_risk_manager

    def create_risk_metrics_calculator(self) -> RiskMetricsCalculator:
        """
        Create a production-ready risk metrics calculator.

        Returns:
            RiskMetricsCalculator: A configured risk metrics calculator.
        """
        risk_metrics_calculator = RiskMetricsCalculator(
            confidence_level=self.config.var_confidence_level
        )

        logger.info("Created production RiskMetricsCalculator")
        return risk_metrics_calculator

    def create_circuit_breakers(self) -> List[CircuitBreaker]:
        """
        Create production-ready circuit breakers.

        Returns:
            List[CircuitBreaker]: A list of configured circuit breakers.
        """
        circuit_breakers = []

        # Add market circuit breaker if enabled
        if self.config.market_circuit_breaker_enabled:
            market_cb = MarketWideCircuitBreaker(
                market="SPY",  # Default to SPY as the market index
                level_1_percent=self.config.market_volatility_threshold,
                level_2_percent=self.config.market_volatility_threshold * 1.5,
                level_3_percent=self.config.market_volatility_threshold * 2.0,
                level_1_duration_minutes=15,
                level_2_duration_minutes=30,
                level_3_duration_minutes=60
            )
            circuit_breakers.append(market_cb)

        # Add account circuit breakers if enabled
        if self.config.account_circuit_breaker_enabled:
            # Convert config parameters to match AccountCircuitBreaker constructor
            account_cb = AccountCircuitBreaker(
                account_id="default",  # Default account ID
                daily_loss_percent_warning=self.config.daily_loss_limit_percent * 0.5,  # 50% of limit as warning
                daily_loss_percent_soft=self.config.daily_loss_limit_percent,
                daily_loss_percent_hard=self.config.daily_loss_limit_percent * 1.5,  # 150% of limit as hard stop
                enabled=True
            )
            # Set initial balance
            account_cb.update_starting_balance(100000.0)
            circuit_breakers.append(account_cb)

        logger.info(f"Created {len(circuit_breakers)} production circuit breakers")
        return circuit_breakers

    def create_circuit_breaker_manager(self, emergency_handlers: Optional[List[Any]] = None) -> CircuitBreakerManager:
        """
        Create a production-ready circuit breaker manager.

        Args:
            emergency_handlers: Optional list of emergency handlers to notify when circuit breakers trigger.

        Returns:
            CircuitBreakerManager: A configured circuit breaker manager.
        """
        circuit_breakers = self.create_circuit_breakers()

        circuit_breaker_manager = CircuitBreakerManager(
            circuit_breakers=circuit_breakers,
            emergency_handlers=emergency_handlers or []
        )

        logger.info("Created production CircuitBreakerManager")
        return circuit_breaker_manager

    def create_advanced_risk_manager(self, emergency_handlers: Optional[List[Any]] = None) -> AdvancedRiskManager:
        """
        Create a production-ready advanced risk manager with all components integrated.

        Args:
            emergency_handlers: Optional list of emergency handlers to notify when circuit breakers trigger.

        Returns:
            AdvancedRiskManager: A fully configured advanced risk manager.
        """
        position_sizer = self.create_position_sizer()
        stop_loss_manager = self.create_stop_loss_manager()
        portfolio_risk_manager = self.create_portfolio_risk_manager()
        risk_metrics_calculator = self.create_risk_metrics_calculator()
        circuit_breaker_manager = self.create_circuit_breaker_manager(emergency_handlers)
        
        # Get risk adjustment parameters from config
        risk_adjustment_factor = getattr(self.config, 'risk_adjustment_factor', 1.0)
        max_signal_strength = getattr(self.config, 'max_signal_strength', 1.0)
        min_signal_strength = getattr(self.config, 'min_signal_strength', 0.1)
        enable_signal_filtering = getattr(self.config, 'enable_signal_filtering', True)

        advanced_risk_manager = AdvancedRiskManager(
            position_sizer=position_sizer,
            stop_loss_manager=stop_loss_manager,
            portfolio_risk_manager=portfolio_risk_manager,
            risk_metrics_calculator=risk_metrics_calculator,
            circuit_breaker_manager=circuit_breaker_manager,
            risk_adjustment_factor=risk_adjustment_factor,
            max_signal_strength=max_signal_strength,
            min_signal_strength=min_signal_strength,
            enable_signal_filtering=enable_signal_filtering
        )

        # Configure position limits by asset class
        for asset_class, limits in self.config.position_limits_by_asset.items():
            advanced_risk_manager.set_position_limits_for_asset_class(
                asset_class=asset_class,
                max_position_percentage=limits.get("max_position_percentage"),
                max_position_value=limits.get("max_position_value")
            )

        # Configure logging
        if self.config.enable_detailed_logging:
            advanced_risk_manager.enable_detailed_logging()

        logger.info("Created production AdvancedRiskManager with all components integrated")
        return advanced_risk_manager
