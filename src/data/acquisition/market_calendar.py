"""Market calendar module for the Friday AI Trading System.

This module provides the MarketCalendar class for handling market schedules and trading days.
"""

from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import pytz

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager

# Create logger
logger = get_logger(__name__)


class MarketType(Enum):
    """Enum for market types."""

    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"
    CUSTOM = "custom"


class MarketCalendar:
    """Market calendar for handling market schedules and trading days.

    This class provides functionality for determining market open/close times,
    trading days, holidays, and other calendar-related operations.

    Attributes:
        market_type: The type of market.
        timezone: The timezone of the market.
        config: The configuration manager.
        holidays: Set of holiday dates.
        early_closes: Dictionary mapping dates to early close times.
        late_opens: Dictionary mapping dates to late open times.
        custom_schedule: Dictionary mapping dates to custom open/close times.
    """

    def __init__(
        self,
        market_type: MarketType,
        timezone: str = "UTC",
        config: Optional[ConfigManager] = None,
    ):
        """Initialize a market calendar.

        Args:
            market_type: The type of market.
            timezone: The timezone of the market. Defaults to "UTC".
            config: Configuration manager. If None, a new one will be created.
        """
        self.market_type = market_type
        self.timezone = pytz.timezone(timezone)
        self.config = config or ConfigManager()
        
        # Initialize holiday and special schedule data
        self.holidays: Set[datetime.date] = set()
        self.early_closes: Dict[datetime.date, time] = {}
        self.late_opens: Dict[datetime.date, time] = {}
        self.custom_schedule: Dict[datetime.date, Tuple[time, time]] = {}
        
        # Load default schedule based on market type
        self._load_default_schedule()
        
        # Load holidays and special schedules
        self._load_holidays()
        self._load_special_schedules()

    def _load_default_schedule(self) -> None:
        """Load the default schedule based on market type."""
        # Default schedules for different market types
        if self.market_type == MarketType.STOCK:
            # Default US stock market hours (9:30 AM - 4:00 PM Eastern Time)
            self.regular_open = time(9, 30)
            self.regular_close = time(16, 0)
            self.trading_days = [0, 1, 2, 3, 4]  # Monday to Friday
            
        elif self.market_type == MarketType.FOREX:
            # Forex markets are open 24/5 (Sunday 5 PM ET to Friday 5 PM ET)
            self.regular_open = time(17, 0)  # 5 PM Sunday
            self.regular_close = time(17, 0)  # 5 PM Friday
            self.trading_days = [0, 1, 2, 3, 4, 6]  # Monday to Friday + Sunday
            
        elif self.market_type == MarketType.CRYPTO:
            # Crypto markets are open 24/7
            self.regular_open = time(0, 0)
            self.regular_close = time(23, 59, 59)
            self.trading_days = [0, 1, 2, 3, 4, 5, 6]  # All days
            
        elif self.market_type == MarketType.FUTURES:
            # Default futures market hours (varies by contract)
            # Using CME hours as default (6:00 PM - 5:00 PM ET next day, Sunday-Friday)
            self.regular_open = time(18, 0)  # 6 PM
            self.regular_close = time(17, 0)  # 5 PM next day
            self.trading_days = [0, 1, 2, 3, 4, 6]  # Monday to Friday + Sunday
            
        else:  # OPTIONS or CUSTOM
            # Default to stock market hours
            self.regular_open = time(9, 30)
            self.regular_close = time(16, 0)
            self.trading_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        # Override with configuration if available
        config_prefix = f"market.{self.market_type.value}"
        self.regular_open = self._parse_time(self.config.get(f"{config_prefix}.regular_open", None)) or self.regular_open
        self.regular_close = self._parse_time(self.config.get(f"{config_prefix}.regular_close", None)) or self.regular_close
        self.trading_days = self.config.get(f"{config_prefix}.trading_days", self.trading_days)

    def _load_holidays(self) -> None:
        """Load holidays from configuration."""
        config_prefix = f"market.{self.market_type.value}.holidays"
        holiday_dates = self.config.get(config_prefix, [])
        
        for date_str in holiday_dates:
            try:
                holiday_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                self.holidays.add(holiday_date)
            except ValueError:
                logger.warning(f"Invalid holiday date format: {date_str}")

    def _load_special_schedules(self) -> None:
        """Load special schedules from configuration."""
        config_prefix = f"market.{self.market_type.value}"
        
        # Early closes
        early_closes = self.config.get(f"{config_prefix}.early_closes", {})
        for date_str, time_str in early_closes.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                close_time = self._parse_time(time_str)
                if close_time:
                    self.early_closes[date] = close_time
            except ValueError:
                logger.warning(f"Invalid early close format: {date_str} {time_str}")
        
        # Late opens
        late_opens = self.config.get(f"{config_prefix}.late_opens", {})
        for date_str, time_str in late_opens.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                open_time = self._parse_time(time_str)
                if open_time:
                    self.late_opens[date] = open_time
            except ValueError:
                logger.warning(f"Invalid late open format: {date_str} {time_str}")
        
        # Custom schedules
        custom_schedules = self.config.get(f"{config_prefix}.custom_schedules", {})
        for date_str, schedule in custom_schedules.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                open_time = self._parse_time(schedule.get("open"))
                close_time = self._parse_time(schedule.get("close"))
                if open_time and close_time:
                    self.custom_schedule[date] = (open_time, close_time)
            except ValueError:
                logger.warning(f"Invalid custom schedule format: {date_str} {schedule}")

    def _parse_time(self, time_str: Optional[str]) -> Optional[time]:
        """Parse a time string into a time object.

        Args:
            time_str: The time string to parse.

        Returns:
            Optional[time]: The parsed time, or None if parsing fails.
        """
        if not time_str:
            return None
            
        try:
            # Try HH:MM format
            t = datetime.strptime(time_str, "%H:%M").time()
            return t
        except ValueError:
            try:
                # Try HH:MM:SS format
                t = datetime.strptime(time_str, "%H:%M:%S").time()
                return t
            except ValueError:
                logger.warning(f"Invalid time format: {time_str}")
                return None

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if the market is open at the given datetime.

        Args:
            dt: The datetime to check. Defaults to current time if None.

        Returns:
            bool: True if the market is open, False otherwise.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            # Localize naive datetime
            dt = self.timezone.localize(dt)
        elif dt.tzinfo != self.timezone:
            # Convert to market timezone
            dt = dt.astimezone(self.timezone)
        
        # Check if it's a holiday
        if dt.date() in self.holidays:
            return False
        
        # Check if it's a trading day
        if dt.weekday() not in self.trading_days:
            return False
        
        # Get open and close times for this date
        open_time, close_time = self.get_market_hours(dt.date())
        
        # Create datetime objects for open and close times
        open_dt = datetime.combine(dt.date(), open_time).replace(tzinfo=self.timezone)
        close_dt = datetime.combine(dt.date(), close_time).replace(tzinfo=self.timezone)
        
        # Handle overnight markets (close time is earlier than open time)
        if close_time < open_time:
            # Market spans two days
            if dt.time() >= open_time:
                # After open on first day
                return True
            elif dt.time() < close_time:
                # Before close on second day
                # Check if previous day was a trading day and not a holiday
                prev_day = (dt - timedelta(days=1)).date()
                if prev_day.weekday() in self.trading_days and prev_day not in self.holidays:
                    return True
                return False
            else:
                return False
        else:
            # Regular market hours within the same day
            return open_dt <= dt < close_dt

    def get_market_hours(self, date: datetime.date) -> Tuple[time, time]:
        """Get the market open and close times for a specific date.

        Args:
            date: The date to get market hours for.

        Returns:
            Tuple[time, time]: The open and close times for the date.
        """
        # Check for custom schedule
        if date in self.custom_schedule:
            return self.custom_schedule[date]
        
        # Get regular open and close times
        open_time = self.regular_open
        close_time = self.regular_close
        
        # Check for late open
        if date in self.late_opens:
            open_time = self.late_opens[date]
        
        # Check for early close
        if date in self.early_closes:
            close_time = self.early_closes[date]
        
        return open_time, close_time

    def get_next_market_open(self, dt: Optional[datetime] = None) -> datetime:
        """Get the next market open time after the given datetime.

        Args:
            dt: The datetime to start from. Defaults to current time if None.

        Returns:
            datetime: The next market open time.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            # Localize naive datetime
            dt = self.timezone.localize(dt)
        elif dt.tzinfo != self.timezone:
            # Convert to market timezone
            dt = dt.astimezone(self.timezone)
        
        # If market is already open, return current time
        if self.is_market_open(dt):
            return dt
        
        # Start checking from the current date
        current_date = dt.date()
        current_time = dt.time()
        
        # Check up to 30 days ahead (to avoid infinite loop)
        for _ in range(30):
            # Get market hours for current date
            open_time, close_time = self.get_market_hours(current_date)
            
            # Check if market opens later today
            if current_date.weekday() in self.trading_days and current_date not in self.holidays and current_time < open_time:
                next_open = datetime.combine(current_date, open_time).replace(tzinfo=self.timezone)
                return next_open
            
            # Move to next day
            current_date += timedelta(days=1)
            current_time = time(0, 0)  # Reset time to beginning of day
            
            # Check if next day is a trading day and not a holiday
            if current_date.weekday() in self.trading_days and current_date not in self.holidays:
                open_time, _ = self.get_market_hours(current_date)
                next_open = datetime.combine(current_date, open_time).replace(tzinfo=self.timezone)
                return next_open
        
        # If no open time found within 30 days, return None or raise exception
        raise ValueError("No market open time found within the next 30 days")

    def get_next_market_close(self, dt: Optional[datetime] = None) -> datetime:
        """Get the next market close time after the given datetime.

        Args:
            dt: The datetime to start from. Defaults to current time if None.

        Returns:
            datetime: The next market close time.
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            # Localize naive datetime
            dt = self.timezone.localize(dt)
        elif dt.tzinfo != self.timezone:
            # Convert to market timezone
            dt = dt.astimezone(self.timezone)
        
        # Start checking from the current date
        current_date = dt.date()
        current_time = dt.time()
        
        # Check up to 30 days ahead (to avoid infinite loop)
        for _ in range(30):
            # Skip if not a trading day or is a holiday
            if current_date.weekday() not in self.trading_days or current_date in self.holidays:
                current_date += timedelta(days=1)
                current_time = time(0, 0)  # Reset time to beginning of day
                continue
            
            # Get market hours for current date
            open_time, close_time = self.get_market_hours(current_date)
            
            # Handle overnight markets
            if close_time < open_time:
                # Market spans two days
                if current_time < close_time:
                    # Before close on second day
                    next_close = datetime.combine(current_date, close_time).replace(tzinfo=self.timezone)
                    return next_close
                elif current_time >= open_time:
                    # After open on first day, close is on next day
                    next_day = current_date + timedelta(days=1)
                    if next_day.weekday() in self.trading_days and next_day not in self.holidays:
                        _, next_close_time = self.get_market_hours(next_day)
                        next_close = datetime.combine(next_day, next_close_time).replace(tzinfo=self.timezone)
                        return next_close
            else:
                # Regular market hours within the same day
                if current_time < close_time:
                    if current_time < open_time:
                        # Before open, return close on same day
                        next_close = datetime.combine(current_date, close_time).replace(tzinfo=self.timezone)
                        return next_close
                    else:
                        # After open, before close, return close on same day
                        next_close = datetime.combine(current_date, close_time).replace(tzinfo=self.timezone)
                        return next_close
            
            # Move to next day
            current_date += timedelta(days=1)
            current_time = time(0, 0)  # Reset time to beginning of day
        
        # If no close time found within 30 days, return None or raise exception
        raise ValueError("No market close time found within the next 30 days")

    def get_trading_days(self, start_date: datetime.date, end_date: datetime.date) -> List[datetime.date]:
        """Get all trading days between start_date and end_date (inclusive).

        Args:
            start_date: The start date.
            end_date: The end date.

        Returns:
            List[datetime.date]: List of trading days.
        """
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() in self.trading_days and current_date not in self.holidays:
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days

    def get_trading_hours(self, date: datetime.date) -> Optional[Tuple[datetime, datetime]]:
        """Get the trading hours for a specific date.

        Args:
            date: The date to get trading hours for.

        Returns:
            Optional[Tuple[datetime, datetime]]: Tuple of (open_datetime, close_datetime),
                or None if not a trading day.
        """
        # Check if it's a trading day
        if date.weekday() not in self.trading_days or date in self.holidays:
            return None
        
        # Get open and close times
        open_time, close_time = self.get_market_hours(date)
        
        # Create datetime objects
        open_dt = datetime.combine(date, open_time).replace(tzinfo=self.timezone)
        
        # Handle overnight markets
        if close_time < open_time:
            # Close time is on the next day
            next_day = date + timedelta(days=1)
            close_dt = datetime.combine(next_day, close_time).replace(tzinfo=self.timezone)
        else:
            close_dt = datetime.combine(date, close_time).replace(tzinfo=self.timezone)
        
        return (open_dt, close_dt)

    def is_trading_day(self, date: datetime.date) -> bool:
        """Check if a date is a trading day.

        Args:
            date: The date to check.

        Returns:
            bool: True if it's a trading day, False otherwise.
        """
        return date.weekday() in self.trading_days and date not in self.holidays

    def add_holiday(self, date: datetime.date) -> None:
        """Add a holiday to the calendar.

        Args:
            date: The holiday date to add.
        """
        self.holidays.add(date)

    def remove_holiday(self, date: datetime.date) -> bool:
        """Remove a holiday from the calendar.

        Args:
            date: The holiday date to remove.

        Returns:
            bool: True if the holiday was removed, False if it wasn't in the calendar.
        """
        if date in self.holidays:
            self.holidays.remove(date)
            return True
        return False

    def set_early_close(self, date: datetime.date, close_time: time) -> None:
        """Set an early close for a specific date.

        Args:
            date: The date to set early close for.
            close_time: The early close time.
        """
        self.early_closes[date] = close_time

    def set_late_open(self, date: datetime.date, open_time: time) -> None:
        """Set a late open for a specific date.

        Args:
            date: The date to set late open for.
            open_time: The late open time.
        """
        self.late_opens[date] = open_time

    def set_custom_schedule(self, date: datetime.date, open_time: time, close_time: time) -> None:
        """Set a custom schedule for a specific date.

        Args:
            date: The date to set custom schedule for.
            open_time: The custom open time.
            close_time: The custom close time.
        """
        self.custom_schedule[date] = (open_time, close_time)

    def get_current_market_status(self) -> Dict[str, Union[bool, str, datetime]]:
        """Get the current market status.

        Returns:
            Dict[str, Union[bool, str, datetime]]: Dictionary with market status information.
        """
        now = datetime.now(self.timezone)
        is_open = self.is_market_open(now)
        
        if is_open:
            next_event = "close"
            next_event_time = self.get_next_market_close(now)
        else:
            next_event = "open"
            next_event_time = self.get_next_market_open(now)
        
        # Calculate time until next event
        time_until_next_event = next_event_time - now
        
        return {
            "is_open": is_open,
            "current_time": now,
            "next_event": next_event,
            "next_event_time": next_event_time,
            "time_until_next_event": time_until_next_event,
            "market_type": self.market_type.value,
            "timezone": str(self.timezone),
        }