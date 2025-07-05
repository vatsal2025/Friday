"""Utility functions for the Trading Engine.

This module provides utility functions for the trading engine.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, time, timedelta
import uuid
import json
import hashlib

from src.infrastructure.logging import get_logger

# Configure logger
logger = get_logger(__name__)


def generate_trade_id() -> str:
    """Generate a unique trade ID.
    
    Returns:
        str: A unique trade ID.
    """
    return str(uuid.uuid4())


def generate_order_id() -> str:
    """Generate a unique order ID.
    
    Returns:
        str: A unique order ID.
    """
    return f"ord-{uuid.uuid4().hex[:12]}"


def generate_signal_id() -> str:
    """Generate a unique signal ID.
    
    Returns:
        str: A unique signal ID.
    """
    return f"sig-{uuid.uuid4().hex[:12]}"


def generate_execution_id() -> str:
    """Generate a unique execution ID.
    
    Returns:
        str: A unique execution ID.
    """
    return f"exe-{uuid.uuid4().hex[:12]}"


def calculate_order_value(price: float, quantity: float) -> float:
    """Calculate the value of an order.
    
    Args:
        price: The price of the order.
        quantity: The quantity of the order.
        
    Returns:
        float: The value of the order.
    """
    return price * quantity


def calculate_slippage(price: float, quantity: float, side: str, 
                      slippage_model: str = "fixed", slippage_params: Dict[str, Any] = None) -> float:
    """Calculate slippage for an order.
    
    Args:
        price: The price of the order.
        quantity: The quantity of the order.
        side: The side of the order ("buy" or "sell").
        slippage_model: The slippage model to use.
        slippage_params: Parameters for the slippage model.
        
    Returns:
        float: The slippage amount.
    """
    if slippage_params is None:
        slippage_params = {}
    
    # Default parameters
    fixed_points = slippage_params.get("fixed_points", 1)
    percentage = slippage_params.get("percentage", 0.0005)  # 0.05%
    impact_factor = slippage_params.get("impact_factor", 0.1)
    
    # Direction multiplier (positive for buy, negative for sell)
    direction = 1 if side.lower() == "buy" else -1
    
    if slippage_model == "fixed":
        # Fixed slippage in price points
        return direction * fixed_points
    
    elif slippage_model == "percentage":
        # Percentage of price
        return direction * price * percentage
    
    elif slippage_model == "market_impact":
        # Simple market impact model based on quantity
        # Higher quantity = higher impact
        order_value = price * quantity
        return direction * price * percentage * (1 + impact_factor * (order_value / 10000))
    
    else:
        logger.warning(f"Unknown slippage model: {slippage_model}, using fixed")
        return direction * fixed_points


def is_market_open(current_time: Optional[datetime] = None, 
                  market_open: time = time(9, 30), 
                  market_close: time = time(16, 0),
                  pre_market_start: Optional[time] = None,
                  post_market_end: Optional[time] = None,
                  allow_pre_market: bool = False,
                  allow_post_market: bool = False,
                  weekend_trading: bool = False) -> bool:
    """Check if the market is open.
    
    Args:
        current_time: The current time. If None, uses the current time.
        market_open: The market open time.
        market_close: The market close time.
        pre_market_start: The pre-market start time.
        post_market_end: The post-market end time.
        allow_pre_market: Whether to allow pre-market trading.
        allow_post_market: Whether to allow post-market trading.
        weekend_trading: Whether to allow weekend trading.
        
    Returns:
        bool: True if the market is open, False otherwise.
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Check if it's a weekend
    if not weekend_trading and current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    current_time_only = current_time.time()
    
    # Regular market hours
    if market_open <= current_time_only < market_close:
        return True
    
    # Pre-market hours
    if allow_pre_market and pre_market_start is not None and pre_market_start <= current_time_only < market_open:
        return True
    
    # Post-market hours
    if allow_post_market and post_market_end is not None and market_close <= current_time_only < post_market_end:
        return True
    
    return False


def get_market_session(current_time: Optional[datetime] = None,
                      market_open: time = time(9, 30),
                      market_close: time = time(16, 0),
                      pre_market_start: Optional[time] = None,
                      post_market_end: Optional[time] = None) -> str:
    """Get the current market session.
    
    Args:
        current_time: The current time. If None, uses the current time.
        market_open: The market open time.
        market_close: The market close time.
        pre_market_start: The pre-market start time.
        post_market_end: The post-market end time.
        
    Returns:
        str: The current market session ("pre_market", "regular", "post_market", or "closed").
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Check if it's a weekend
    if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return "closed"
    
    current_time_only = current_time.time()
    
    # Regular market hours
    if market_open <= current_time_only < market_close:
        return "regular"
    
    # Pre-market hours
    if pre_market_start is not None and pre_market_start <= current_time_only < market_open:
        return "pre_market"
    
    # Post-market hours
    if post_market_end is not None and market_close <= current_time_only < post_market_end:
        return "post_market"
    
    return "closed"


def calculate_time_to_market_open(current_time: Optional[datetime] = None,
                                market_open: time = time(9, 30)) -> Optional[timedelta]:
    """Calculate the time until market open.
    
    Args:
        current_time: The current time. If None, uses the current time.
        market_open: The market open time.
        
    Returns:
        timedelta: The time until market open, or None if the market is already open.
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Create a datetime for today's market open
    today_market_open = datetime.combine(current_time.date(), market_open)
    
    # If current time is after today's market open, use tomorrow's market open
    if current_time >= today_market_open:
        tomorrow = current_time.date() + timedelta(days=1)
        today_market_open = datetime.combine(tomorrow, market_open)
    
    return today_market_open - current_time


def calculate_time_to_market_close(current_time: Optional[datetime] = None,
                                 market_close: time = time(16, 0)) -> Optional[timedelta]:
    """Calculate the time until market close.
    
    Args:
        current_time: The current time. If None, uses the current time.
        market_close: The market close time.
        
    Returns:
        timedelta: The time until market close, or None if the market is already closed.
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Create a datetime for today's market close
    today_market_close = datetime.combine(current_time.date(), market_close)
    
    # If current time is after today's market close, return None
    if current_time >= today_market_close:
        return None
    
    return today_market_close - current_time


def hash_dict(data: Dict[str, Any]) -> str:
    """Generate a hash for a dictionary.
    
    Args:
        data: The dictionary to hash.
        
    Returns:
        str: The hash of the dictionary.
    """
    # Convert dict to a stable string representation
    json_str = json.dumps(data, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(json_str.encode()).hexdigest()


def format_price(price: float, tick_size: float = 0.01) -> float:
    """Format a price to the nearest tick size.
    
    Args:
        price: The price to format.
        tick_size: The tick size to round to.
        
    Returns:
        float: The formatted price.
    """
    return round(price / tick_size) * tick_size


def format_quantity(quantity: float, lot_size: float = 1.0) -> float:
    """Format a quantity to the nearest lot size.
    
    Args:
        quantity: The quantity to format.
        lot_size: The lot size to round to.
        
    Returns:
        float: The formatted quantity.
    """
    return round(quantity / lot_size) * lot_size


def calculate_profit_loss(entry_price: float, exit_price: float, 
                         quantity: float, side: str) -> float:
    """Calculate profit or loss for a trade.
    
    Args:
        entry_price: The entry price.
        exit_price: The exit price.
        quantity: The quantity.
        side: The side of the trade ("buy" or "sell").
        
    Returns:
        float: The profit or loss.
    """
    if side.lower() == "buy":
        return (exit_price - entry_price) * quantity
    else:  # sell
        return (entry_price - exit_price) * quantity


def calculate_return_percentage(entry_price: float, exit_price: float, side: str) -> float:
    """Calculate return percentage for a trade.
    
    Args:
        entry_price: The entry price.
        exit_price: The exit price.
        side: The side of the trade ("buy" or "sell").
        
    Returns:
        float: The return percentage.
    """
    if side.lower() == "buy":
        return (exit_price - entry_price) / entry_price * 100
    else:  # sell
        return (entry_price - exit_price) / entry_price * 100


def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """Parse a timeframe string into a value and unit.
    
    Args:
        timeframe: The timeframe string (e.g., "1m", "5m", "1h", "1d").
        
    Returns:
        Tuple[int, str]: The value and unit of the timeframe.
    """
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    # Extract numeric part and unit
    for i, char in enumerate(timeframe):
        if not char.isdigit():
            value = int(timeframe[:i])
            unit = timeframe[i:]
            return value, unit
    
    raise ValueError(f"Invalid timeframe format: {timeframe}")


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe string to seconds.
    
    Args:
        timeframe: The timeframe string (e.g., "1m", "5m", "1h", "1d").
        
    Returns:
        int: The timeframe in seconds.
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 60 * 60
    elif unit == "d":
        return value * 60 * 60 * 24
    elif unit == "w":
        return value * 60 * 60 * 24 * 7
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convert a timeframe string to a timedelta.
    
    Args:
        timeframe: The timeframe string (e.g., "1m", "5m", "1h", "1d").
        
    Returns:
        timedelta: The timeframe as a timedelta.
    """
    seconds = timeframe_to_seconds(timeframe)
    return timedelta(seconds=seconds)


def round_to_significant_figures(value: float, sig_figs: int) -> float:
    """Round a value to a specified number of significant figures.
    
    Args:
        value: The value to round.
        sig_figs: The number of significant figures.
        
    Returns:
        float: The rounded value.
    """
    if value == 0:
        return 0
    
    import math
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)


def normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to a standard format.
    
    Args:
        symbol: The symbol to normalize.
        
    Returns:
        str: The normalized symbol.
    """
    # Remove any whitespace
    symbol = symbol.strip()
    
    # Convert to uppercase
    symbol = symbol.upper()
    
    # Remove any special characters or exchange suffixes
    # This is a simplified example and may need to be customized
    for suffix in [".NS", ".BSE", ".NYSE", ".NASDAQ"]:
        if symbol.endswith(suffix):
            symbol = symbol[:-len(suffix)]
    
    return symbol


def get_signal_strength_description(strength: float) -> str:
    """Get a description of a signal strength.
    
    Args:
        strength: The signal strength (0.0 to 1.0).
        
    Returns:
        str: The signal strength description.
    """
    if strength >= 0.8:
        return "Very Strong"
    elif strength >= 0.6:
        return "Strong"
    elif strength >= 0.4:
        return "Moderate"
    elif strength >= 0.2:
        return "Weak"
    else:
        return "Very Weak"


def get_confidence_description(confidence: float) -> str:
    """Get a description of a confidence level.
    
    Args:
        confidence: The confidence level (0.0 to 1.0).
        
    Returns:
        str: The confidence description.
    """
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Moderate"
    elif confidence >= 0.3:
        return "Low"
    else:
        return "Very Low"


def format_currency(value: float, currency: str = "USD", precision: int = 2) -> str:
    """Format a value as a currency string.
    
    Args:
        value: The value to format.
        currency: The currency code.
        precision: The number of decimal places.
        
    Returns:
        str: The formatted currency string.
    """
    if currency == "USD":
        return f"${value:.{precision}f}"
    elif currency == "EUR":
        return f"€{value:.{precision}f}"
    elif currency == "GBP":
        return f"£{value:.{precision}f}"
    elif currency == "JPY":
        return f"¥{value:.{precision}f}"
    elif currency == "INR":
        return f"₹{value:.{precision}f}"
    else:
        return f"{value:.{precision}f} {currency}"


def format_percentage(value: float, precision: int = 2, include_sign: bool = True) -> str:
    """Format a value as a percentage string.
    
    Args:
        value: The value to format.
        precision: The number of decimal places.
        include_sign: Whether to include a sign for positive values.
        
    Returns:
        str: The formatted percentage string.
    """
    if include_sign and value > 0:
        return f"+{value:.{precision}f}%"
    else:
        return f"{value:.{precision}f}%"