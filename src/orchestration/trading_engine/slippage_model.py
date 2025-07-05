"""Slippage Models for Trading Engine.

This module provides various slippage models to simulate realistic market impact
and execution costs for different order types and market conditions.
"""

import math
import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger(__name__)


class SlippageModelType(Enum):
    """Types of slippage models."""
    FIXED = "fixed"  # Fixed slippage percentage or amount
    PERCENTAGE = "percentage"  # Percentage of price
    VOLUME_BASED = "volume_based"  # Based on order size relative to volume
    MARKET_IMPACT = "market_impact"  # Square-root market impact model
    VOLATILITY_BASED = "volatility_based"  # Based on price volatility
    SPREAD_BASED = "spread_based"  # Based on bid-ask spread
    PROBABILISTIC = "probabilistic"  # Random slippage within a range
    CUSTOM = "custom"  # Custom slippage model


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketConditions:
    """Market conditions affecting slippage."""
    volume: float = 0.0  # Average daily volume
    volatility: float = 0.0  # Price volatility (e.g., standard deviation)
    spread: float = 0.0  # Bid-ask spread
    liquidity: float = 1.0  # Liquidity factor (1.0 = normal, <1.0 = less liquid, >1.0 = more liquid)
    is_stressed: bool = False  # Whether the market is in a stressed condition


@dataclass
class SlippageParameters:
    """Parameters for slippage models."""
    model_type: SlippageModelType = SlippageModelType.FIXED
    fixed_slippage: float = 0.0  # Fixed slippage amount
    percentage: float = 0.001  # 0.1% default slippage
    market_impact_factor: float = 0.1  # Market impact factor
    min_slippage: float = 0.0  # Minimum slippage
    max_slippage: float = 0.01  # Maximum slippage (1%)
    spread_factor: float = 0.5  # Portion of spread to use (0.5 = half spread)
    volatility_factor: float = 0.1  # Volatility impact factor
    random_seed: Optional[int] = None  # Seed for random number generation
    custom_model: Optional[Callable[[float, OrderSide, MarketConditions, Any], float]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)  # Parameters for custom model


class SlippageModel:
    """Base class for slippage models."""
    def __init__(self, params: SlippageParameters):
        """Initialize the slippage model.
        
        Args:
            params: Parameters for the slippage model
        """
        self.params = params
        
        # Set random seed if provided
        if params.random_seed is not None:
            random.seed(params.random_seed)
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate slippage for an order.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage amount (positive for price increase, negative for price decrease)
        """
        raise NotImplementedError("Subclasses must implement calculate_slippage")
    
    def calculate_execution_price(self, price: float, quantity: float, side: OrderSide, 
                                market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate execution price after slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Execution price after slippage
        """
        slippage = self.calculate_slippage(price, quantity, side, market_conditions)
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            # For buys, slippage increases the price
            return price + slippage
        else:
            # For sells, slippage decreases the price
            return price - slippage
    
    def calculate_slippage_cost(self, price: float, quantity: float, side: OrderSide, 
                              market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate the cost of slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Cost of slippage (always positive)
        """
        execution_price = self.calculate_execution_price(price, quantity, side, market_conditions)
        
        if side == OrderSide.BUY:
            # For buys, cost is the difference between execution price and order price
            return (execution_price - price) * quantity
        else:
            # For sells, cost is the difference between order price and execution price
            return (price - execution_price) * quantity


class FixedSlippageModel(SlippageModel):
    """Fixed slippage model.
    
    Applies a fixed amount of slippage regardless of order size or market conditions.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate fixed slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Fixed slippage amount
        """
        return self.params.fixed_slippage


class PercentageSlippageModel(SlippageModel):
    """Percentage slippage model.
    
    Applies slippage as a percentage of the order price.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate percentage slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage as a percentage of price
        """
        return price * self.params.percentage


class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model.
    
    Applies slippage based on the order size relative to average daily volume.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate volume-based slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage based on volume
        """
        if market_conditions is None or market_conditions.volume <= 0:
            logger.warning("No volume information available, using default percentage slippage")
            return price * self.params.percentage
        
        # Calculate order size as a percentage of daily volume
        volume_ratio = quantity / market_conditions.volume
        
        # Apply a non-linear impact based on volume ratio
        # Higher volume ratio = higher impact
        impact = self.params.market_impact_factor * math.pow(volume_ratio, 0.5)
        
        # Apply liquidity adjustment if available
        if market_conditions.liquidity > 0:
            impact = impact / market_conditions.liquidity
        
        # Apply stress adjustment if market is stressed
        if market_conditions.is_stressed:
            impact *= 2.0
        
        # Cap slippage at max_slippage
        slippage = min(price * impact, price * self.params.max_slippage)
        
        # Ensure minimum slippage
        slippage = max(slippage, price * self.params.min_slippage)
        
        return slippage


class MarketImpactSlippageModel(SlippageModel):
    """Market impact slippage model.
    
    Implements the square-root market impact model: impact = k * sigma * sqrt(Q/V)
    where k is a constant, sigma is volatility, Q is order quantity, and V is volume.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate market impact slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage based on market impact model
        """
        if market_conditions is None or market_conditions.volume <= 0:
            logger.warning("No volume information available, using default percentage slippage")
            return price * self.params.percentage
        
        # Get volatility, default to 1% if not available
        volatility = market_conditions.volatility if market_conditions.volatility > 0 else 0.01
        
        # Calculate market impact using square-root formula
        volume_ratio = quantity / market_conditions.volume
        impact = self.params.market_impact_factor * volatility * math.sqrt(volume_ratio)
        
        # Apply liquidity adjustment if available
        if market_conditions.liquidity > 0:
            impact = impact / market_conditions.liquidity
        
        # Apply stress adjustment if market is stressed
        if market_conditions.is_stressed:
            impact *= 2.0
        
        # Calculate slippage amount
        slippage = price * impact
        
        # Cap slippage at max_slippage
        slippage = min(slippage, price * self.params.max_slippage)
        
        # Ensure minimum slippage
        slippage = max(slippage, price * self.params.min_slippage)
        
        return slippage


class VolatilityBasedSlippageModel(SlippageModel):
    """Volatility-based slippage model.
    
    Applies slippage based on price volatility.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate volatility-based slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage based on volatility
        """
        if market_conditions is None or market_conditions.volatility <= 0:
            logger.warning("No volatility information available, using default percentage slippage")
            return price * self.params.percentage
        
        # Calculate slippage as a function of volatility
        slippage = price * market_conditions.volatility * self.params.volatility_factor
        
        # Apply volume adjustment if available
        if market_conditions.volume > 0:
            volume_ratio = quantity / market_conditions.volume
            slippage *= (1.0 + math.sqrt(volume_ratio))
        
        # Apply liquidity adjustment if available
        if market_conditions.liquidity > 0:
            slippage = slippage / market_conditions.liquidity
        
        # Apply stress adjustment if market is stressed
        if market_conditions.is_stressed:
            slippage *= 2.0
        
        # Cap slippage at max_slippage
        slippage = min(slippage, price * self.params.max_slippage)
        
        # Ensure minimum slippage
        slippage = max(slippage, price * self.params.min_slippage)
        
        return slippage


class SpreadBasedSlippageModel(SlippageModel):
    """Spread-based slippage model.
    
    Applies slippage based on the bid-ask spread.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate spread-based slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage based on spread
        """
        if market_conditions is None or market_conditions.spread <= 0:
            logger.warning("No spread information available, using default percentage slippage")
            return price * self.params.percentage
        
        # Base slippage is a portion of the spread
        slippage = market_conditions.spread * self.params.spread_factor
        
        # Apply volume adjustment if available
        if market_conditions.volume > 0:
            volume_ratio = quantity / market_conditions.volume
            slippage *= (1.0 + volume_ratio)
        
        # Apply liquidity adjustment if available
        if market_conditions.liquidity > 0:
            slippage = slippage / market_conditions.liquidity
        
        # Apply stress adjustment if market is stressed
        if market_conditions.is_stressed:
            slippage *= 2.0
        
        # Cap slippage at max_slippage
        slippage = min(slippage, price * self.params.max_slippage)
        
        # Ensure minimum slippage
        slippage = max(slippage, price * self.params.min_slippage)
        
        return slippage


class ProbabilisticSlippageModel(SlippageModel):
    """Probabilistic slippage model.
    
    Applies random slippage within a range based on market conditions.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate probabilistic slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Random slippage within a range
        """
        # Calculate base slippage range
        min_slip = price * self.params.min_slippage
        max_slip = price * self.params.max_slippage
        
        # Adjust range based on market conditions if available
        if market_conditions is not None:
            # Adjust for volume
            if market_conditions.volume > 0:
                volume_ratio = quantity / market_conditions.volume
                max_slip *= (1.0 + volume_ratio)
            
            # Adjust for volatility
            if market_conditions.volatility > 0:
                vol_factor = 1.0 + (market_conditions.volatility * self.params.volatility_factor)
                max_slip *= vol_factor
            
            # Adjust for liquidity
            if market_conditions.liquidity > 0:
                min_slip /= market_conditions.liquidity
                max_slip /= market_conditions.liquidity
            
            # Adjust for stress
            if market_conditions.is_stressed:
                min_slip *= 1.5
                max_slip *= 2.0
        
        # Generate random slippage within the range
        slippage = random.uniform(min_slip, max_slip)
        
        return slippage


class CustomSlippageModel(SlippageModel):
    """Custom slippage model.
    
    Uses a custom function to calculate slippage.
    """
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide, 
                          market_conditions: Optional[MarketConditions] = None) -> float:
        """Calculate custom slippage.
        
        Args:
            price: Order price
            quantity: Order quantity
            side: Order side (buy or sell)
            market_conditions: Market conditions affecting slippage
            
        Returns:
            Slippage calculated by custom function
        """
        if self.params.custom_model is None:
            logger.warning("No custom model provided, using default percentage slippage")
            return price * self.params.percentage
        
        try:
            slippage = self.params.custom_model(price, quantity, side, market_conditions, self.params.custom_params)
            
            # Cap slippage at max_slippage if specified
            if self.params.max_slippage > 0:
                slippage = min(slippage, price * self.params.max_slippage)
            
            # Ensure minimum slippage if specified
            if self.params.min_slippage > 0:
                slippage = max(slippage, price * self.params.min_slippage)
            
            return slippage
        except Exception as e:
            logger.error(f"Error in custom slippage model: {e}")
            return price * self.params.percentage


class SlippageModelFactory:
    """Factory for creating slippage models."""
    @staticmethod
    def create_model(params: SlippageParameters) -> SlippageModel:
        """Create a slippage model based on parameters.
        
        Args:
            params: Parameters for the slippage model
            
        Returns:
            SlippageModel instance
        """
        if params.model_type == SlippageModelType.FIXED:
            return FixedSlippageModel(params)
        elif params.model_type == SlippageModelType.PERCENTAGE:
            return PercentageSlippageModel(params)
        elif params.model_type == SlippageModelType.VOLUME_BASED:
            return VolumeBasedSlippageModel(params)
        elif params.model_type == SlippageModelType.MARKET_IMPACT:
            return MarketImpactSlippageModel(params)
        elif params.model_type == SlippageModelType.VOLATILITY_BASED:
            return VolatilityBasedSlippageModel(params)
        elif params.model_type == SlippageModelType.SPREAD_BASED:
            return SpreadBasedSlippageModel(params)
        elif params.model_type == SlippageModelType.PROBABILISTIC:
            return ProbabilisticSlippageModel(params)
        elif params.model_type == SlippageModelType.CUSTOM:
            return CustomSlippageModel(params)
        else:
            logger.warning(f"Unknown slippage model type: {params.model_type}, using percentage model")
            return PercentageSlippageModel(params)
    
    @staticmethod
    def create_default_model() -> SlippageModel:
        """Create a default slippage model.
        
        Returns:
            Default SlippageModel instance
        """
        params = SlippageParameters()
        return PercentageSlippageModel(params)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> SlippageModel:
        """Create a slippage model from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SlippageModel instance
        """
        # Parse model type
        model_type_str = config.get("model_type", "percentage")
        try:
            model_type = SlippageModelType(model_type_str)
        except ValueError:
            logger.warning(f"Invalid model type: {model_type_str}, using percentage model")
            model_type = SlippageModelType.PERCENTAGE
        
        # Create parameters
        params = SlippageParameters(
            model_type=model_type,
            fixed_slippage=config.get("fixed_slippage", 0.0),
            percentage=config.get("percentage", 0.001),
            market_impact_factor=config.get("market_impact_factor", 0.1),
            min_slippage=config.get("min_slippage", 0.0),
            max_slippage=config.get("max_slippage", 0.01),
            spread_factor=config.get("spread_factor", 0.5),
            volatility_factor=config.get("volatility_factor", 0.1),
            random_seed=config.get("random_seed")
        )
        
        # Add custom parameters if present
        if "custom_params" in config and isinstance(config["custom_params"], dict):
            params.custom_params = config["custom_params"]
        
        return SlippageModelFactory.create_model(params)


# Predefined slippage models for different market conditions
def get_low_impact_model() -> SlippageModel:
    """Get a slippage model for low market impact.
    
    Returns:
        SlippageModel for low impact
    """
    params = SlippageParameters(
        model_type=SlippageModelType.PERCENTAGE,
        percentage=0.0005,  # 0.05%
        min_slippage=0.0001,  # 0.01%
        max_slippage=0.002   # 0.2%
    )
    return PercentageSlippageModel(params)


def get_medium_impact_model() -> SlippageModel:
    """Get a slippage model for medium market impact.
    
    Returns:
        SlippageModel for medium impact
    """
    params = SlippageParameters(
        model_type=SlippageModelType.MARKET_IMPACT,
        market_impact_factor=0.1,
        min_slippage=0.0005,  # 0.05%
        max_slippage=0.005    # 0.5%
    )
    return MarketImpactSlippageModel(params)


def get_high_impact_model() -> SlippageModel:
    """Get a slippage model for high market impact.
    
    Returns:
        SlippageModel for high impact
    """
    params = SlippageParameters(
        model_type=SlippageModelType.MARKET_IMPACT,
        market_impact_factor=0.2,
        min_slippage=0.001,   # 0.1%
        max_slippage=0.01     # 1.0%
    )
    return MarketImpactSlippageModel(params)


def get_volatile_market_model() -> SlippageModel:
    """Get a slippage model for volatile markets.
    
    Returns:
        SlippageModel for volatile markets
    """
    params = SlippageParameters(
        model_type=SlippageModelType.VOLATILITY_BASED,
        volatility_factor=0.2,
        min_slippage=0.001,   # 0.1%
        max_slippage=0.02     # 2.0%
    )
    return VolatilityBasedSlippageModel(params)


def get_illiquid_market_model() -> SlippageModel:
    """Get a slippage model for illiquid markets.
    
    Returns:
        SlippageModel for illiquid markets
    """
    params = SlippageParameters(
        model_type=SlippageModelType.SPREAD_BASED,
        spread_factor=0.75,
        min_slippage=0.002,   # 0.2%
        max_slippage=0.03     # 3.0%
    )
    return SpreadBasedSlippageModel(params)


def get_realistic_model() -> SlippageModel:
    """Get a realistic slippage model that combines multiple factors.
    
    Returns:
        SlippageModel with realistic behavior
    """
    # Custom slippage function that combines multiple factors
    def realistic_slippage(price, quantity, side, market_conditions, custom_params):
        # Default values if market conditions not provided
        volume = 1000000.0
        volatility = 0.01
        spread = price * 0.001
        liquidity = 1.0
        is_stressed = False
        
        # Use market conditions if provided
        if market_conditions is not None:
            if market_conditions.volume > 0:
                volume = market_conditions.volume
            if market_conditions.volatility > 0:
                volatility = market_conditions.volatility
            if market_conditions.spread > 0:
                spread = market_conditions.spread
            if market_conditions.liquidity > 0:
                liquidity = market_conditions.liquidity
            is_stressed = market_conditions.is_stressed
        
        # Get parameters
        impact_factor = custom_params.get("impact_factor", 0.1)
        vol_factor = custom_params.get("vol_factor", 0.1)
        spread_factor = custom_params.get("spread_factor", 0.5)
        
        # Calculate components
        volume_ratio = quantity / volume
        impact_component = price * impact_factor * math.sqrt(volume_ratio)
        vol_component = price * volatility * vol_factor
        spread_component = spread * spread_factor
        
        # Combine components
        slippage = impact_component + vol_component + spread_component
        
        # Apply liquidity adjustment
        slippage = slippage / liquidity
        
        # Apply stress adjustment
        if is_stressed:
            slippage *= 1.5
        
        return slippage
    
    params = SlippageParameters(
        model_type=SlippageModelType.CUSTOM,
        min_slippage=0.0005,  # 0.05%
        max_slippage=0.01,    # 1.0%
        custom_model=realistic_slippage,
        custom_params={
            "impact_factor": 0.1,
            "vol_factor": 0.1,
            "spread_factor": 0.5
        }
    )
    
    return CustomSlippageModel(params)