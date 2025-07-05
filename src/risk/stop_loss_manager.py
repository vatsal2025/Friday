import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StopLossType:
    """Enumeration of stop loss types."""
    FIXED = "fixed"              # Fixed price stop loss
    TRAILING = "trailing"        # Trailing stop loss that follows price
    VOLATILITY = "volatility"    # Volatility-based stop loss (ATR)
    TIME = "time"                # Time-based exit
    PROFIT_TARGET = "profit"     # Profit target (take profit)

class StopLossManager:
    """
    Stop-loss management system that implements multiple stop-loss strategies.

    This class provides functionality for:
    - Fixed stop losses
    - Trailing stop losses
    - Volatility-based stops (ATR)
    - Time-based exits
    - Profit targets
    """

    def __init__(self,
                 default_stop_loss_percent: float = 0.02,  # 2% default stop loss
                 default_trailing_percent: float = 0.015,   # 1.5% trailing stop
                 default_atr_multiplier: float = 2.0,       # 2x ATR for volatility stops
                 default_time_stop_days: int = 10,          # 10-day time stop
                 default_profit_target_ratio: float = 2.0,  # 2:1 reward-to-risk ratio
                 atr_period: int = 14):                     # 14-day ATR period
        """
        Initialize the stop loss manager.

        Args:
            default_stop_loss_percent: Default fixed stop loss percentage (0.02 = 2%)
            default_trailing_percent: Default trailing stop percentage (0.015 = 1.5%)
            default_atr_multiplier: Default ATR multiplier for volatility-based stops
            default_time_stop_days: Default number of days for time-based exits
            default_profit_target_ratio: Default profit target as a ratio of risk
            atr_period: Period for ATR calculation
        """
        self.default_stop_loss_percent = default_stop_loss_percent
        self.default_trailing_percent = default_trailing_percent
        self.default_atr_multiplier = default_atr_multiplier
        self.default_time_stop_days = default_time_stop_days
        self.default_profit_target_ratio = default_profit_target_ratio
        self.atr_period = atr_period

        # Store active stops
        self.active_stops = {}

        logger.info("Initialized StopLossManager")

    def calculate_fixed_stop(self,
                           entry_price: float,
                           direction: str,
                           stop_percent: Optional[float] = None) -> float:
        """
        Calculate a fixed stop loss price.

        Args:
            entry_price: Entry price of the position
            direction: Trade direction ("long" or "short")
            stop_percent: Stop loss percentage (0.02 = 2%)

        Returns:
            Stop loss price
        """
        if stop_percent is None:
            stop_percent = self.default_stop_loss_percent

        if direction.lower() == "long":
            return entry_price * (1 - stop_percent)
        else:  # short
            return entry_price * (1 + stop_percent)

    def calculate_trailing_stop(self,
                              entry_price: float,
                              current_price: float,
                              direction: str,
                              trailing_percent: Optional[float] = None,
                              current_stop: Optional[float] = None) -> float:
        """
        Calculate a trailing stop loss price.

        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            direction: Trade direction ("long" or "short")
            trailing_percent: Trailing stop percentage (0.015 = 1.5%)
            current_stop: Current stop loss price (if already set)

        Returns:
            Updated trailing stop loss price
        """
        if trailing_percent is None:
            trailing_percent = self.default_trailing_percent

        # Calculate new potential stop
        if direction.lower() == "long":
            new_stop = current_price * (1 - trailing_percent)

            # Only update if it would raise the stop (for long positions)
            if current_stop is None or new_stop > current_stop:
                return new_stop
            return current_stop
        else:  # short
            new_stop = current_price * (1 + trailing_percent)

            # Only update if it would lower the stop (for short positions)
            if current_stop is None or new_stop < current_stop:
                return new_stop
            return current_stop

    def calculate_volatility_stop(self,
                                entry_price: float,
                                direction: str,
                                historical_prices: pd.DataFrame,
                                atr_multiplier: Optional[float] = None) -> float:
        """
        Calculate a volatility-based stop loss using Average True Range (ATR).

        Args:
            entry_price: Entry price of the position
            direction: Trade direction ("long" or "short")
            historical_prices: DataFrame with OHLC price data
            atr_multiplier: Multiplier for ATR (default: 2.0)

        Returns:
            Volatility-based stop loss price
        """
        if atr_multiplier is None:
            atr_multiplier = self.default_atr_multiplier

        # Calculate ATR if we have enough data
        if historical_prices is None or len(historical_prices) < self.atr_period:
            # Fall back to fixed stop if not enough data
            return self.calculate_fixed_stop(entry_price, direction)

        # Calculate ATR
        atr = self._calculate_atr(historical_prices)

        # Apply ATR to determine stop distance
        if direction.lower() == "long":
            return entry_price - (atr * atr_multiplier)
        else:  # short
            return entry_price + (atr * atr_multiplier)

    def _calculate_atr(self, prices: pd.DataFrame) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            prices: DataFrame with OHLC price data

        Returns:
            ATR value
        """
        # Make sure we have high, low, close columns
        required_columns = ['high', 'low', 'close']
        if not all(col.lower() in map(str.lower, prices.columns) for col in required_columns):
            logger.warning("Price data missing required columns for ATR calculation")
            return 0.0

        # Standardize column names to lowercase
        prices_lower = prices.copy()
        prices_lower.columns = [col.lower() for col in prices.columns]

        # Calculate true range
        tr1 = prices_lower['high'] - prices_lower['low']
        tr2 = abs(prices_lower['high'] - prices_lower['close'].shift(1))
        tr3 = abs(prices_lower['low'] - prices_lower['close'].shift(1))

        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR as simple moving average of TR
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]

        return atr if not np.isnan(atr) else tr.mean()

    def calculate_time_stop(self,
                          entry_time: datetime,
                          days: Optional[int] = None) -> datetime:
        """
        Calculate a time-based exit.

        Args:
            entry_time: Entry time of the position
            days: Number of days to hold the position

        Returns:
            Exit time
        """
        if days is None:
            days = self.default_time_stop_days

        return entry_time + timedelta(days=days)

    def calculate_profit_target(self,
                              entry_price: float,
                              stop_price: float,
                              direction: str,
                              risk_reward_ratio: Optional[float] = None) -> float:
        """
        Calculate a profit target based on risk-reward ratio.

        Args:
            entry_price: Entry price of the position
            stop_price: Stop loss price
            direction: Trade direction ("long" or "short")
            risk_reward_ratio: Risk-reward ratio (default: 2.0)

        Returns:
            Profit target price
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.default_profit_target_ratio

        # Calculate risk in price terms
        if direction.lower() == "long":
            risk = entry_price - stop_price
            return entry_price + (risk * risk_reward_ratio)
        else:  # short
            risk = stop_price - entry_price
            return entry_price - (risk * risk_reward_ratio)

    def set_stop_loss(self,
                     trade_id: str,
                     entry_price: float,
                     entry_time: datetime,
                     direction: str,
                     stop_type: str = StopLossType.FIXED,
                     stop_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set a stop loss for a trade.

        Args:
            trade_id: Unique identifier for the trade
            entry_price: Entry price of the position
            entry_time: Entry time of the position
            direction: Trade direction ("long" or "short")
            stop_type: Type of stop loss (fixed, trailing, volatility, time, profit)
            stop_params: Additional parameters for the stop loss

        Returns:
            Stop loss details
        """
        if stop_params is None:
            stop_params = {}

        stop_details = {
            "trade_id": trade_id,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "direction": direction,
            "stop_type": stop_type,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "is_active": True
        }

        # Calculate stop price based on type
        if stop_type == StopLossType.FIXED:
            stop_percent = stop_params.get("stop_percent", self.default_stop_loss_percent)
            stop_details["stop_price"] = self.calculate_fixed_stop(entry_price, direction, stop_percent)
            stop_details["stop_percent"] = stop_percent

        elif stop_type == StopLossType.TRAILING:
            trailing_percent = stop_params.get("trailing_percent", self.default_trailing_percent)
            # Initialize trailing stop at the same level as a fixed stop
            stop_details["stop_price"] = self.calculate_fixed_stop(entry_price, direction, trailing_percent)
            stop_details["trailing_percent"] = trailing_percent
            stop_details["highest_price"] = entry_price if direction.lower() == "long" else float('inf')
            stop_details["lowest_price"] = entry_price if direction.lower() == "short" else 0.0

        elif stop_type == StopLossType.VOLATILITY:
            historical_prices = stop_params.get("historical_prices")
            atr_multiplier = stop_params.get("atr_multiplier", self.default_atr_multiplier)
            if historical_prices is not None:
                stop_details["stop_price"] = self.calculate_volatility_stop(
                    entry_price, direction, historical_prices, atr_multiplier
                )
            else:
                # Fall back to fixed stop if no historical prices provided
                stop_details["stop_price"] = self.calculate_fixed_stop(entry_price, direction)
            stop_details["atr_multiplier"] = atr_multiplier

        elif stop_type == StopLossType.TIME:
            days = stop_params.get("days", self.default_time_stop_days)
            stop_details["exit_time"] = self.calculate_time_stop(entry_time, days)
            stop_details["days"] = days

        elif stop_type == StopLossType.PROFIT_TARGET:
            # For profit targets, we need a stop loss to calculate the risk
            stop_price = stop_params.get("stop_price")
            if stop_price is None:
                # Calculate a default stop if none provided
                stop_price = self.calculate_fixed_stop(entry_price, direction)

            risk_reward_ratio = stop_params.get("risk_reward_ratio", self.default_profit_target_ratio)
            stop_details["target_price"] = self.calculate_profit_target(
                entry_price, stop_price, direction, risk_reward_ratio
            )
            stop_details["stop_price"] = stop_price
            stop_details["risk_reward_ratio"] = risk_reward_ratio

        # Store the stop loss
        self.active_stops[trade_id] = stop_details

        logger.info(f"Set {stop_type} stop loss for trade {trade_id}")
        return stop_details

    def update_stop_loss(self,
                        trade_id: str,
                        current_price: float,
                        current_time: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Update a stop loss based on current market conditions.

        Args:
            trade_id: Unique identifier for the trade
            current_price: Current market price
            current_time: Current time (default: now)

        Returns:
            Tuple of (is_triggered, updated_stop_details)
        """
        if current_time is None:
            current_time = datetime.now()

        if trade_id not in self.active_stops:
            logger.warning(f"No active stop loss found for trade {trade_id}")
            return False, {}

        stop_details = self.active_stops[trade_id]

        # Skip if stop is not active
        if not stop_details.get("is_active", True):
            return False, stop_details

        # Check if stop is triggered
        is_triggered = False
        direction = stop_details["direction"].lower()
        stop_type = stop_details["stop_type"]

        # Update stop details
        stop_details["updated_at"] = current_time

        # Check different stop types
        if stop_type == StopLossType.FIXED:
            # Fixed stops don't change, just check if triggered
            stop_price = stop_details["stop_price"]
            if (direction == "long" and current_price <= stop_price) or \
               (direction == "short" and current_price >= stop_price):
                is_triggered = True

        elif stop_type == StopLossType.TRAILING:
            # Update trailing stop if price moved favorably
            trailing_percent = stop_details["trailing_percent"]

            if direction == "long":
                # Update highest price seen
                if current_price > stop_details.get("highest_price", 0):
                    stop_details["highest_price"] = current_price
                    # Update trailing stop based on highest price
                    new_stop = self.calculate_trailing_stop(
                        stop_details["entry_price"],
                        current_price,
                        direction,
                        trailing_percent,
                        stop_details["stop_price"]
                    )
                    stop_details["stop_price"] = new_stop

                # Check if stop is triggered
                if current_price <= stop_details["stop_price"]:
                    is_triggered = True
            else:  # short
                # Update lowest price seen
                if current_price < stop_details.get("lowest_price", float('inf')):
                    stop_details["lowest_price"] = current_price
                    # Update trailing stop based on lowest price
                    new_stop = self.calculate_trailing_stop(
                        stop_details["entry_price"],
                        current_price,
                        direction,
                        trailing_percent,
                        stop_details["stop_price"]
                    )
                    stop_details["stop_price"] = new_stop

                # Check if stop is triggered
                if current_price >= stop_details["stop_price"]:
                    is_triggered = True

        elif stop_type == StopLossType.VOLATILITY:
            # Volatility stops are typically recalculated with new data
            # For simplicity, we'll just check if the current stop is triggered
            stop_price = stop_details["stop_price"]
            if (direction == "long" and current_price <= stop_price) or \
               (direction == "short" and current_price >= stop_price):
                is_triggered = True

        elif stop_type == StopLossType.TIME:
            # Check if current time is past the exit time
            exit_time = stop_details["exit_time"]
            if current_time >= exit_time:
                is_triggered = True

        elif stop_type == StopLossType.PROFIT_TARGET:
            # Check if profit target is reached
            target_price = stop_details["target_price"]
            if (direction == "long" and current_price >= target_price) or \
               (direction == "short" and current_price <= target_price):
                is_triggered = True

            # Also check if stop loss is triggered
            stop_price = stop_details["stop_price"]
            if (direction == "long" and current_price <= stop_price) or \
               (direction == "short" and current_price >= stop_price):
                is_triggered = True
                stop_details["triggered_by"] = "stop_loss"
            elif is_triggered:
                stop_details["triggered_by"] = "profit_target"

        # Update stop details if triggered
        if is_triggered:
            stop_details["is_active"] = False
            stop_details["triggered_at"] = current_time
            stop_details["exit_price"] = current_price
            logger.info(f"Stop loss triggered for trade {trade_id}")

        # Update stored stop details
        self.active_stops[trade_id] = stop_details

        return is_triggered, stop_details

    def get_stop_loss(self, trade_id: str) -> Dict[str, Any]:
        """
        Get stop loss details for a trade.

        Args:
            trade_id: Unique identifier for the trade

        Returns:
            Stop loss details or empty dict if not found
        """
        return self.active_stops.get(trade_id, {})

    def remove_stop_loss(self, trade_id: str) -> bool:
        """
        Remove a stop loss.

        Args:
            trade_id: Unique identifier for the trade

        Returns:
            True if removed, False if not found
        """
        if trade_id in self.active_stops:
            del self.active_stops[trade_id]
            logger.info(f"Removed stop loss for trade {trade_id}")
            return True
        return False

    def get_active_stops(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active stop losses.

        Returns:
            Dictionary of active stop losses {trade_id: stop_details}
        """
        return {tid: stop for tid, stop in self.active_stops.items() if stop.get("is_active", True)}

    def check_all_stops(self,
                       current_prices: Dict[str, float],
                       current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Check all active stops against current market prices.

        Args:
            current_prices: Dictionary of current prices {symbol: price}
            current_time: Current time (default: now)

        Returns:
            List of triggered stop details
        """
        if current_time is None:
            current_time = datetime.now()

        triggered_stops = []

        for trade_id, stop_details in self.get_active_stops().items():
            # Get symbol from trade_id or stop_details
            symbol = stop_details.get("symbol")
            if not symbol:
                # Try to extract symbol from trade_id (implementation-specific)
                continue

            # Get current price for this symbol
            current_price = current_prices.get(symbol)
            if current_price is None:
                logger.warning(f"No current price available for {symbol}")
                continue

            # Update stop and check if triggered
            is_triggered, updated_stop = self.update_stop_loss(trade_id, current_price, current_time)

            if is_triggered:
                triggered_stops.append(updated_stop)

        return triggered_stops
