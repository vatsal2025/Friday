"""Transaction cost modeling for backtesting framework.

This module provides tools for modeling transaction costs in trading simulations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class CostType(Enum):
    """Types of transaction costs."""
    FIXED = "fixed"                # Fixed cost per trade
    PERCENTAGE = "percentage"      # Percentage of trade value
    PER_SHARE = "per_share"        # Cost per share/unit
    SLIPPAGE = "slippage"          # Price slippage
    SPREAD = "spread"              # Bid-ask spread
    MARKET_IMPACT = "market_impact"  # Market impact
    EXCHANGE_FEE = "exchange_fee"  # Exchange fee
    CLEARING_FEE = "clearing_fee"  # Clearing fee
    TAX = "tax"                    # Transaction tax
    CUSTOM = "custom"              # Custom cost model


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""
    
    @abstractmethod
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate transaction cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Transaction cost
        """
        pass
    
    @abstractmethod
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        pass


class FixedCostModel(TransactionCostModel):
    """Fixed cost per trade model."""
    
    def __init__(self, fixed_cost: float = 0.0):
        """Initialize the fixed cost model.
        
        Args:
            fixed_cost: Fixed cost per trade (default: 0.0)
        """
        self.fixed_cost = fixed_cost
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate fixed cost for a trade.
        
        Args:
            price: Trade price (not used)
            quantity: Trade quantity (not used)
            **kwargs: Additional parameters (not used)
            
        Returns:
            Fixed cost
        """
        return self.fixed_cost
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.FIXED
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Fixed cost of {self.fixed_cost} per trade"


class PercentageCostModel(TransactionCostModel):
    """Percentage of trade value cost model."""
    
    def __init__(self, percentage: float = 0.0):
        """Initialize the percentage cost model.
        
        Args:
            percentage: Percentage of trade value (default: 0.0)
        """
        self.percentage = percentage / 100.0  # Convert to decimal
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate percentage cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters (not used)
            
        Returns:
            Percentage cost
        """
        return abs(price * quantity * self.percentage)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.PERCENTAGE
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Percentage cost of {self.percentage * 100}% of trade value"


class PerShareCostModel(TransactionCostModel):
    """Cost per share/unit model."""
    
    def __init__(self, per_share_cost: float = 0.0):
        """Initialize the per-share cost model.
        
        Args:
            per_share_cost: Cost per share/unit (default: 0.0)
        """
        self.per_share_cost = per_share_cost
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate per-share cost for a trade.
        
        Args:
            price: Trade price (not used)
            quantity: Trade quantity
            **kwargs: Additional parameters (not used)
            
        Returns:
            Per-share cost
        """
        return abs(quantity * self.per_share_cost)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.PER_SHARE
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Per-share cost of {self.per_share_cost} per unit"


class SlippageCostModel(TransactionCostModel):
    """Price slippage cost model."""
    
    def __init__(self, slippage_percentage: float = 0.0, random_seed: Optional[int] = None):
        """Initialize the slippage cost model.
        
        Args:
            slippage_percentage: Slippage percentage (default: 0.0)
            random_seed: Random seed for slippage calculation (default: None)
        """
        self.slippage_percentage = slippage_percentage / 100.0  # Convert to decimal
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate slippage cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
                is_buy: Whether the trade is a buy (True) or sell (False)
            
        Returns:
            Slippage cost
        """
        is_buy = kwargs.get("is_buy", True)
        
        # Generate random slippage within the specified percentage
        slippage_factor = self.rng.uniform(0, self.slippage_percentage)
        
        # For buys, slippage increases the price; for sells, slippage decreases the price
        slippage_direction = 1 if is_buy else -1
        slippage_amount = price * slippage_factor * slippage_direction
        
        return abs(slippage_amount * quantity)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.SLIPPAGE
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Slippage cost of up to {self.slippage_percentage * 100}% of trade value"


class SpreadCostModel(TransactionCostModel):
    """Bid-ask spread cost model."""
    
    def __init__(self, spread_percentage: float = 0.0):
        """Initialize the spread cost model.
        
        Args:
            spread_percentage: Spread percentage (default: 0.0)
        """
        self.spread_percentage = spread_percentage / 100.0  # Convert to decimal
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate spread cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters (not used)
            
        Returns:
            Spread cost
        """
        # Half the spread is applied to each side of the trade
        half_spread = self.spread_percentage / 2.0
        spread_cost = price * half_spread
        
        return abs(spread_cost * quantity)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.SPREAD
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Spread cost of {self.spread_percentage * 100}% of trade value"


class MarketImpactCostModel(TransactionCostModel):
    """Market impact cost model.
    
    Market impact is modeled as a function of trade size relative to average daily volume.
    """
    
    def __init__(self, impact_factor: float = 0.1):
        """Initialize the market impact cost model.
        
        Args:
            impact_factor: Market impact factor (default: 0.1)
        """
        self.impact_factor = impact_factor
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate market impact cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
                volume: Average daily volume
            
        Returns:
            Market impact cost
        """
        volume = kwargs.get("volume", None)
        
        if volume is None or volume == 0:
            logger.warning("Average daily volume not provided or zero, market impact cost will be zero")
            return 0.0
        
        # Calculate market impact as a function of trade size relative to volume
        participation_rate = abs(quantity) / volume
        impact = self.impact_factor * price * (participation_rate ** 0.5)
        
        return abs(impact * quantity)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.MARKET_IMPACT
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Market impact cost with impact factor of {self.impact_factor}"


class ExchangeFeeCostModel(TransactionCostModel):
    """Exchange fee cost model."""
    
    def __init__(self, fee_percentage: float = 0.0, min_fee: float = 0.0, max_fee: Optional[float] = None):
        """Initialize the exchange fee cost model.
        
        Args:
            fee_percentage: Fee percentage (default: 0.0)
            min_fee: Minimum fee (default: 0.0)
            max_fee: Maximum fee (default: None)
        """
        self.fee_percentage = fee_percentage / 100.0  # Convert to decimal
        self.min_fee = min_fee
        self.max_fee = max_fee
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate exchange fee for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters (not used)
            
        Returns:
            Exchange fee
        """
        # Calculate fee as a percentage of trade value
        fee = abs(price * quantity * self.fee_percentage)
        
        # Apply minimum fee
        fee = max(fee, self.min_fee)
        
        # Apply maximum fee if specified
        if self.max_fee is not None:
            fee = min(fee, self.max_fee)
        
        return fee
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.EXCHANGE_FEE
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        description = f"Exchange fee of {self.fee_percentage * 100}% of trade value"
        
        if self.min_fee > 0:
            description += f", minimum fee of {self.min_fee}"
        
        if self.max_fee is not None:
            description += f", maximum fee of {self.max_fee}"
        
        return description


class TaxCostModel(TransactionCostModel):
    """Transaction tax cost model."""
    
    def __init__(self, tax_percentage: float = 0.0, apply_to_buys: bool = False, apply_to_sells: bool = True):
        """Initialize the tax cost model.
        
        Args:
            tax_percentage: Tax percentage (default: 0.0)
            apply_to_buys: Whether to apply tax to buy trades (default: False)
            apply_to_sells: Whether to apply tax to sell trades (default: True)
        """
        self.tax_percentage = tax_percentage / 100.0  # Convert to decimal
        self.apply_to_buys = apply_to_buys
        self.apply_to_sells = apply_to_sells
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate tax cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
                is_buy: Whether the trade is a buy (True) or sell (False)
            
        Returns:
            Tax cost
        """
        is_buy = kwargs.get("is_buy", True)
        
        # Check if tax applies to this trade type
        if (is_buy and not self.apply_to_buys) or (not is_buy and not self.apply_to_sells):
            return 0.0
        
        # Calculate tax as a percentage of trade value
        return abs(price * quantity * self.tax_percentage)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.TAX
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        description = f"Transaction tax of {self.tax_percentage * 100}% of trade value"
        
        if self.apply_to_buys and self.apply_to_sells:
            description += " (applies to buys and sells)"
        elif self.apply_to_buys:
            description += " (applies to buys only)"
        elif self.apply_to_sells:
            description += " (applies to sells only)"
        
        return description


class CompositeCostModel(TransactionCostModel):
    """Composite cost model combining multiple cost models."""
    
    def __init__(self, cost_models: List[TransactionCostModel]):
        """Initialize the composite cost model.
        
        Args:
            cost_models: List of cost models
        """
        self.cost_models = cost_models
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate total cost for a trade using all cost models.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Total cost
        """
        total_cost = 0.0
        
        for model in self.cost_models:
            cost = model.calculate_cost(price, quantity, **kwargs)
            total_cost += cost
        
        return total_cost
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.CUSTOM
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        descriptions = [model.get_description() for model in self.cost_models]
        return "Composite cost model: " + "; ".join(descriptions)
    
    def get_cost_breakdown(self, price: float, quantity: float, **kwargs) -> Dict[str, float]:
        """Get a breakdown of costs by cost model.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Dict with cost breakdown
        """
        breakdown = {}
        
        for model in self.cost_models:
            cost = model.calculate_cost(price, quantity, **kwargs)
            breakdown[model.get_description()] = cost
        
        return breakdown


class ZeroCostModel(TransactionCostModel):
    """Zero cost model (no transaction costs)."""
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate cost for a trade (always zero).
        
        Args:
            price: Trade price (not used)
            quantity: Trade quantity (not used)
            **kwargs: Additional parameters (not used)
            
        Returns:
            Zero cost
        """
        return 0.0
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.CUSTOM
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return "Zero cost model (no transaction costs)"


class RealisticStockCostModel(CompositeCostModel):
    """Realistic stock trading cost model.
    
    This model combines commission, spread, and slippage costs.
    """
    
    def __init__(
        self,
        commission_percentage: float = 0.1,
        min_commission: float = 1.0,
        max_commission: Optional[float] = None,
        spread_percentage: float = 0.05,
        slippage_percentage: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        """Initialize the realistic stock cost model.
        
        Args:
            commission_percentage: Commission percentage (default: 0.1%)
            min_commission: Minimum commission (default: $1.0)
            max_commission: Maximum commission (default: None)
            spread_percentage: Spread percentage (default: 0.05%)
            slippage_percentage: Slippage percentage (default: 0.05%)
            random_seed: Random seed for slippage calculation (default: None)
        """
        # Create individual cost models
        commission_model = ExchangeFeeCostModel(
            fee_percentage=commission_percentage,
            min_fee=min_commission,
            max_fee=max_commission,
        )
        
        spread_model = SpreadCostModel(spread_percentage=spread_percentage)
        
        slippage_model = SlippageCostModel(
            slippage_percentage=slippage_percentage,
            random_seed=random_seed,
        )
        
        # Initialize composite model with individual models
        super().__init__([commission_model, spread_model, slippage_model])


class RealisticCryptoCostModel(CompositeCostModel):
    """Realistic cryptocurrency trading cost model.
    
    This model combines exchange fee, spread, and slippage costs.
    """
    
    def __init__(
        self,
        fee_percentage: float = 0.1,
        spread_percentage: float = 0.1,
        slippage_percentage: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """Initialize the realistic cryptocurrency cost model.
        
        Args:
            fee_percentage: Exchange fee percentage (default: 0.1%)
            spread_percentage: Spread percentage (default: 0.1%)
            slippage_percentage: Slippage percentage (default: 0.1%)
            random_seed: Random seed for slippage calculation (default: None)
        """
        # Create individual cost models
        fee_model = ExchangeFeeCostModel(fee_percentage=fee_percentage)
        
        spread_model = SpreadCostModel(spread_percentage=spread_percentage)
        
        slippage_model = SlippageCostModel(
            slippage_percentage=slippage_percentage,
            random_seed=random_seed,
        )
        
        # Initialize composite model with individual models
        super().__init__([fee_model, spread_model, slippage_model])


class RealisticFuturesCostModel(CompositeCostModel):
    """Realistic futures trading cost model.
    
    This model combines commission, spread, slippage, and exchange fee costs.
    """
    
    def __init__(
        self,
        commission_per_contract: float = 2.0,
        spread_ticks: float = 1.0,
        tick_size: float = 0.25,
        slippage_ticks: float = 1.0,
        exchange_fee_per_contract: float = 1.5,
        random_seed: Optional[int] = None,
    ):
        """Initialize the realistic futures cost model.
        
        Args:
            commission_per_contract: Commission per contract (default: $2.0)
            spread_ticks: Spread in ticks (default: 1.0)
            tick_size: Tick size in price units (default: 0.25)
            slippage_ticks: Slippage in ticks (default: 1.0)
            exchange_fee_per_contract: Exchange fee per contract (default: $1.5)
            random_seed: Random seed for slippage calculation (default: None)
        """
        # Create individual cost models
        commission_model = PerShareCostModel(per_share_cost=commission_per_contract)
        
        # Convert ticks to percentage for spread and slippage
        spread_model = CustomSpreadTickModel(spread_ticks=spread_ticks, tick_size=tick_size)
        
        slippage_model = CustomSlippageTickModel(
            slippage_ticks=slippage_ticks,
            tick_size=tick_size,
            random_seed=random_seed,
        )
        
        exchange_fee_model = PerShareCostModel(per_share_cost=exchange_fee_per_contract)
        
        # Initialize composite model with individual models
        super().__init__([commission_model, spread_model, slippage_model, exchange_fee_model])


class CustomSpreadTickModel(TransactionCostModel):
    """Custom spread model based on ticks."""
    
    def __init__(self, spread_ticks: float = 1.0, tick_size: float = 0.25):
        """Initialize the custom spread tick model.
        
        Args:
            spread_ticks: Spread in ticks (default: 1.0)
            tick_size: Tick size in price units (default: 0.25)
        """
        self.spread_ticks = spread_ticks
        self.tick_size = tick_size
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate spread cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters (not used)
            
        Returns:
            Spread cost
        """
        # Calculate spread cost in price units
        spread_cost = self.spread_ticks * self.tick_size / 2.0  # Half the spread
        
        return abs(spread_cost * quantity)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.SPREAD
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Spread cost of {self.spread_ticks} ticks (tick size: {self.tick_size})"


class CustomSlippageTickModel(TransactionCostModel):
    """Custom slippage model based on ticks."""
    
    def __init__(self, slippage_ticks: float = 1.0, tick_size: float = 0.25, random_seed: Optional[int] = None):
        """Initialize the custom slippage tick model.
        
        Args:
            slippage_ticks: Maximum slippage in ticks (default: 1.0)
            tick_size: Tick size in price units (default: 0.25)
            random_seed: Random seed for slippage calculation (default: None)
        """
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """Calculate slippage cost for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
                is_buy: Whether the trade is a buy (True) or sell (False)
            
        Returns:
            Slippage cost
        """
        is_buy = kwargs.get("is_buy", True)
        
        # Generate random slippage within the specified ticks
        slippage_factor = self.rng.uniform(0, self.slippage_ticks)
        
        # Calculate slippage cost in price units
        slippage_cost = slippage_factor * self.tick_size
        
        # For buys, slippage increases the price; for sells, slippage decreases the price
        slippage_direction = 1 if is_buy else -1
        
        return abs(slippage_cost * quantity * slippage_direction)
    
    def get_cost_type(self) -> CostType:
        """Get the cost type.
        
        Returns:
            Cost type
        """
        return CostType.SLIPPAGE
    
    def get_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return f"Slippage cost of up to {self.slippage_ticks} ticks (tick size: {self.tick_size})"


class TransactionCostCalculator:
    """Calculator for transaction costs across multiple trades."""
    
    def __init__(self, cost_model: TransactionCostModel):
        """Initialize the transaction cost calculator.
        
        Args:
            cost_model: Transaction cost model
        """
        self.cost_model = cost_model
        self.total_cost = 0.0
        self.trade_costs = []
    
    def add_trade(self, price: float, quantity: float, **kwargs) -> float:
        """Add a trade and calculate its cost.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters for cost calculation
            
        Returns:
            Trade cost
        """
        cost = self.cost_model.calculate_cost(price, quantity, **kwargs)
        self.total_cost += cost
        self.trade_costs.append(cost)
        
        return cost
    
    def get_total_cost(self) -> float:
        """Get the total cost of all trades.
        
        Returns:
            Total cost
        """
        return self.total_cost
    
    def get_average_cost(self) -> float:
        """Get the average cost per trade.
        
        Returns:
            Average cost
        """
        if not self.trade_costs:
            return 0.0
        
        return self.total_cost / len(self.trade_costs)
    
    def get_cost_model_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return self.cost_model.get_description()
    
    def reset(self) -> None:
        """Reset the calculator."""
        self.total_cost = 0.0
        self.trade_costs = []


class TransactionCostAnalyzer:
    """Analyzer for transaction costs impact on performance."""
    
    def __init__(
        self,
        trades: pd.DataFrame,
        cost_model: TransactionCostModel,
        initial_capital: float = 100000.0,
    ):
        """Initialize the transaction cost analyzer.
        
        Args:
            trades: DataFrame with trade details
            cost_model: Transaction cost model
            initial_capital: Initial capital (default: 100000.0)
        """
        self.trades = trades.copy()
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        
        # Calculate costs for each trade
        self._calculate_costs()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
    
    def _calculate_costs(self) -> None:
        """Calculate costs for each trade."""
        if self.trades.empty:
            return
        
        # Calculate cost for each trade
        self.trades["cost"] = self.trades.apply(
            lambda x: self.cost_model.calculate_cost(
                price=x["price"],
                quantity=x["quantity"],
                is_buy=(x["type"] == "buy"),
            ),
            axis=1,
        )
        
        # Calculate total cost
        self.total_cost = self.trades["cost"].sum()
        
        # Calculate net profit (profit - cost)
        if "profit" not in self.trades.columns:
            self.trades["profit"] = self.trades.apply(
                lambda x: (x["price"] * x["quantity"]) if x["type"] == "sell" else (-x["price"] * x["quantity"]),
                axis=1,
            )
        
        self.trades["net_profit"] = self.trades["profit"] - self.trades["cost"]
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics with and without costs."""
        if self.trades.empty:
            self.metrics = {
                "total_cost": 0.0,
                "cost_percentage": 0.0,
                "gross_profit": 0.0,
                "net_profit": 0.0,
                "cost_impact": 0.0,
                "break_even_improvement": 0.0,
            }
            return
        
        # Calculate total gross profit
        gross_profit = self.trades["profit"].sum()
        
        # Calculate total net profit
        net_profit = self.trades["net_profit"].sum()
        
        # Calculate cost as percentage of gross profit
        cost_percentage = (self.total_cost / abs(gross_profit)) * 100 if gross_profit != 0 else 0.0
        
        # Calculate cost impact on return
        gross_return = (gross_profit / self.initial_capital) * 100
        net_return = (net_profit / self.initial_capital) * 100
        cost_impact = gross_return - net_return
        
        # Calculate break-even improvement
        winning_trades = self.trades[self.trades["profit"] > 0]
        losing_trades = self.trades[self.trades["profit"] < 0]
        
        if not winning_trades.empty and not losing_trades.empty:
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = winning_trades["profit"].mean()
            avg_loss = abs(losing_trades["profit"].mean())
            
            # Break-even win rate without costs
            break_even_win_rate = avg_loss / (avg_win + avg_loss)
            
            # Break-even win rate with costs
            avg_win_net = winning_trades["net_profit"].mean()
            avg_loss_net = abs(losing_trades["net_profit"].mean())
            break_even_win_rate_net = avg_loss_net / (avg_win_net + avg_loss_net)
            
            break_even_improvement = break_even_win_rate_net - break_even_win_rate
        else:
            break_even_improvement = 0.0
        
        # Store metrics
        self.metrics = {
            "total_cost": self.total_cost,
            "cost_percentage": cost_percentage,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "cost_impact": cost_impact,
            "break_even_improvement": break_even_improvement,
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all calculated metrics.
        
        Returns:
            Dict with metrics
        """
        return self.metrics
    
    def get_metric(self, metric: str) -> float:
        """Get a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Metric value
        """
        return self.metrics.get(metric, 0.0)
    
    def get_cost_model_description(self) -> str:
        """Get a description of the cost model.
        
        Returns:
            Cost model description
        """
        return self.cost_model.get_description()
    
    def get_summary(self) -> Dict[str, float]:
        """Get a summary of key metrics.
        
        Returns:
            Dict with key metrics
        """
        return {
            "total_cost": self.get_metric("total_cost"),
            "cost_percentage": self.get_metric("cost_percentage"),
            "gross_profit": self.get_metric("gross_profit"),
            "net_profit": self.get_metric("net_profit"),
            "cost_impact": self.get_metric("cost_impact"),
        }