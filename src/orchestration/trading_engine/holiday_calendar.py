"""Holiday Calendar for Trading Engine.

This module provides functionality for managing market holidays, early closures,
and special trading hours for different exchanges and markets.
"""

import datetime
import logging
import json
import csv
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Status of a market on a given day."""
    OPEN = "open"  # Normal trading day
    CLOSED = "closed"  # Full holiday
    EARLY_CLOSE = "early_close"  # Early closing
    LATE_OPEN = "late_open"  # Late opening
    WEEKEND = "weekend"  # Regular weekend
    SPECIAL = "special"  # Special trading session


@dataclass
class TradingHours:
    """Trading hours for a market on a specific day."""
    open_time: datetime.time  # Market open time
    close_time: datetime.time  # Market close time
    pre_market_start: Optional[datetime.time] = None  # Pre-market trading start
    post_market_end: Optional[datetime.time] = None  # Post-market trading end
    lunch_break_start: Optional[datetime.time] = None  # Lunch break start (for markets with breaks)
    lunch_break_end: Optional[datetime.time] = None  # Lunch break end
    status: MarketStatus = MarketStatus.OPEN  # Market status
    description: str = ""  # Description or reason for special hours

    def is_trading_time(self, time: datetime.time) -> bool:
        """Check if the given time is during regular trading hours.
        
        Args:
            time: The time to check
            
        Returns:
            True if the time is during regular trading hours, False otherwise
        """
        # Check if market is closed
        if self.status == MarketStatus.CLOSED or self.status == MarketStatus.WEEKEND:
            return False
        
        # Check regular trading hours
        if time < self.open_time or time > self.close_time:
            return False
        
        # Check lunch break if applicable
        if (self.lunch_break_start is not None and 
            self.lunch_break_end is not None and 
            self.lunch_break_start <= time <= self.lunch_break_end):
            return False
        
        return True
    
    def is_extended_hours(self, time: datetime.time) -> bool:
        """Check if the given time is during extended trading hours.
        
        Args:
            time: The time to check
            
        Returns:
            True if the time is during extended trading hours, False otherwise
        """
        # Check if market is closed
        if self.status == MarketStatus.CLOSED or self.status == MarketStatus.WEEKEND:
            return False
        
        # Check pre-market hours
        if (self.pre_market_start is not None and 
            self.open_time is not None and 
            self.pre_market_start <= time < self.open_time):
            return True
        
        # Check post-market hours
        if (self.close_time is not None and 
            self.post_market_end is not None and 
            self.close_time < time <= self.post_market_end):
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trading hours to a dictionary.
        
        Returns:
            Dictionary representation of trading hours
        """
        result = {
            "status": self.status.value,
            "description": self.description
        }
        
        # Convert time objects to strings
        if self.open_time:
            result["open_time"] = self.open_time.strftime("%H:%M:%S")
        if self.close_time:
            result["close_time"] = self.close_time.strftime("%H:%M:%S")
        if self.pre_market_start:
            result["pre_market_start"] = self.pre_market_start.strftime("%H:%M:%S")
        if self.post_market_end:
            result["post_market_end"] = self.post_market_end.strftime("%H:%M:%S")
        if self.lunch_break_start:
            result["lunch_break_start"] = self.lunch_break_start.strftime("%H:%M:%S")
        if self.lunch_break_end:
            result["lunch_break_end"] = self.lunch_break_end.strftime("%H:%M:%S")
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingHours':
        """Create trading hours from a dictionary.
        
        Args:
            data: Dictionary representation of trading hours
            
        Returns:
            TradingHours instance
        """
        # Parse time strings to time objects
        open_time = datetime.time.fromisoformat(data.get("open_time", "09:30:00"))
        close_time = datetime.time.fromisoformat(data.get("close_time", "16:00:00"))
        
        # Parse optional time fields
        pre_market_start = None
        if "pre_market_start" in data:
            pre_market_start = datetime.time.fromisoformat(data["pre_market_start"])
        
        post_market_end = None
        if "post_market_end" in data:
            post_market_end = datetime.time.fromisoformat(data["post_market_end"])
        
        lunch_break_start = None
        if "lunch_break_start" in data:
            lunch_break_start = datetime.time.fromisoformat(data["lunch_break_start"])
        
        lunch_break_end = None
        if "lunch_break_end" in data:
            lunch_break_end = datetime.time.fromisoformat(data["lunch_break_end"])
        
        # Parse status
        status_str = data.get("status", "open")
        status = MarketStatus(status_str)
        
        return cls(
            open_time=open_time,
            close_time=close_time,
            pre_market_start=pre_market_start,
            post_market_end=post_market_end,
            lunch_break_start=lunch_break_start,
            lunch_break_end=lunch_break_end,
            status=status,
            description=data.get("description", "")
        )


@dataclass
class HolidayEntry:
    """Entry for a market holiday or special trading day."""
    date: datetime.date  # Date of the holiday
    name: str  # Name of the holiday
    status: MarketStatus  # Market status on this day
    trading_hours: Optional[TradingHours] = None  # Special trading hours if applicable
    affected_markets: List[str] = field(default_factory=list)  # Markets affected by this holiday
    affected_asset_classes: List[str] = field(default_factory=list)  # Asset classes affected
    description: str = ""  # Additional description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert holiday entry to a dictionary.
        
        Returns:
            Dictionary representation of holiday entry
        """
        result = {
            "date": self.date.isoformat(),
            "name": self.name,
            "status": self.status.value,
            "affected_markets": self.affected_markets,
            "affected_asset_classes": self.affected_asset_classes,
            "description": self.description
        }
        
        if self.trading_hours:
            result["trading_hours"] = self.trading_hours.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HolidayEntry':
        """Create a holiday entry from a dictionary.
        
        Args:
            data: Dictionary representation of holiday entry
            
        Returns:
            HolidayEntry instance
        """
        # Parse date
        date = datetime.date.fromisoformat(data["date"])
        
        # Parse status
        status_str = data.get("status", "closed")
        status = MarketStatus(status_str)
        
        # Parse trading hours if present
        trading_hours = None
        if "trading_hours" in data:
            trading_hours = TradingHours.from_dict(data["trading_hours"])
        
        return cls(
            date=date,
            name=data["name"],
            status=status,
            trading_hours=trading_hours,
            affected_markets=data.get("affected_markets", []),
            affected_asset_classes=data.get("affected_asset_classes", []),
            description=data.get("description", "")
        )


class HolidayCalendar:
    """Calendar for managing market holidays and special trading days."""
    def __init__(self, name: str, description: str = ""):
        """Initialize a holiday calendar.
        
        Args:
            name: Name of the calendar
            description: Description of the calendar
        """
        self.name = name
        self.description = description
        self.holidays: Dict[datetime.date, HolidayEntry] = {}
        self.default_trading_hours: Dict[int, TradingHours] = {}  # Weekday -> trading hours
        self.markets: Set[str] = set()  # Markets covered by this calendar
        self.asset_classes: Set[str] = set()  # Asset classes covered by this calendar
    
    def add_holiday(self, holiday: HolidayEntry) -> None:
        """Add a holiday to the calendar.
        
        Args:
            holiday: Holiday entry to add
        """
        self.holidays[holiday.date] = holiday
        
        # Update markets and asset classes sets
        self.markets.update(holiday.affected_markets)
        self.asset_classes.update(holiday.affected_asset_classes)
    
    def add_holidays(self, holidays: List[HolidayEntry]) -> None:
        """Add multiple holidays to the calendar.
        
        Args:
            holidays: List of holiday entries to add
        """
        for holiday in holidays:
            self.add_holiday(holiday)
    
    def set_default_trading_hours(self, weekday: int, trading_hours: TradingHours) -> None:
        """Set default trading hours for a weekday.
        
        Args:
            weekday: Day of the week (0=Monday, 6=Sunday)
            trading_hours: Trading hours for this weekday
        """
        if not 0 <= weekday <= 6:
            raise ValueError(f"Weekday must be between 0 and 6, got {weekday}")
        
        self.default_trading_hours[weekday] = trading_hours
    
    def is_holiday(self, date: datetime.date, market: Optional[str] = None) -> bool:
        """Check if a date is a holiday.
        
        Args:
            date: Date to check
            market: Optional market to check for
            
        Returns:
            True if the date is a holiday, False otherwise
        """
        # Check if date is a weekend
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return True
        
        # Check if date is in holidays
        if date in self.holidays:
            holiday = self.holidays[date]
            
            # If market is specified, check if it's affected
            if market is not None and holiday.affected_markets:
                return market in holiday.affected_markets
            
            # Check if it's a full holiday
            return holiday.status in [MarketStatus.CLOSED, MarketStatus.WEEKEND]
        
        return False
    
    def get_trading_hours(self, date: datetime.date, market: Optional[str] = None) -> Optional[TradingHours]:
        """Get trading hours for a date.
        
        Args:
            date: Date to get trading hours for
            market: Optional market to get trading hours for
            
        Returns:
            TradingHours for the date, or None if the market is closed
        """
        # Check if date is a holiday with special trading hours
        if date in self.holidays:
            holiday = self.holidays[date]
            
            # If market is specified and holiday has affected markets, check if it's affected
            if market is not None and holiday.affected_markets and market not in holiday.affected_markets:
                # Market not affected by this holiday, use default hours
                return self.default_trading_hours.get(date.weekday())
            
            # If holiday has special trading hours, return them
            if holiday.trading_hours is not None:
                return holiday.trading_hours
            
            # If holiday is a full closure, return None
            if holiday.status in [MarketStatus.CLOSED, MarketStatus.WEEKEND]:
                return None
        
        # Check if date is a weekend
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            weekend_hours = TradingHours(
                open_time=datetime.time(0, 0),
                close_time=datetime.time(0, 0),
                status=MarketStatus.WEEKEND,
                description="Weekend"
            )
            return weekend_hours
        
        # Return default trading hours for this weekday
        return self.default_trading_hours.get(date.weekday())
    
    def is_trading_time(self, dt: datetime.datetime, market: Optional[str] = None) -> bool:
        """Check if a datetime is during trading hours.
        
        Args:
            dt: Datetime to check
            market: Optional market to check for
            
        Returns:
            True if the datetime is during trading hours, False otherwise
        """
        trading_hours = self.get_trading_hours(dt.date(), market)
        
        if trading_hours is None:
            return False
        
        return trading_hours.is_trading_time(dt.time())
    
    def is_extended_hours(self, dt: datetime.datetime, market: Optional[str] = None) -> bool:
        """Check if a datetime is during extended trading hours.
        
        Args:
            dt: Datetime to check
            market: Optional market to check for
            
        Returns:
            True if the datetime is during extended trading hours, False otherwise
        """
        trading_hours = self.get_trading_hours(dt.date(), market)
        
        if trading_hours is None:
            return False
        
        return trading_hours.is_extended_hours(dt.time())
    
    def get_next_trading_day(self, date: datetime.date, market: Optional[str] = None) -> datetime.date:
        """Get the next trading day after a date.
        
        Args:
            date: Starting date
            market: Optional market to check for
            
        Returns:
            Next trading day
        """
        next_date = date + datetime.timedelta(days=1)
        
        # Keep incrementing until we find a trading day
        while self.is_holiday(next_date, market):
            next_date += datetime.timedelta(days=1)
        
        return next_date
    
    def get_previous_trading_day(self, date: datetime.date, market: Optional[str] = None) -> datetime.date:
        """Get the previous trading day before a date.
        
        Args:
            date: Starting date
            market: Optional market to check for
            
        Returns:
            Previous trading day
        """
        prev_date = date - datetime.timedelta(days=1)
        
        # Keep decrementing until we find a trading day
        while self.is_holiday(prev_date, market):
            prev_date -= datetime.timedelta(days=1)
        
        return prev_date
    
    def get_trading_days(self, start_date: datetime.date, end_date: datetime.date, 
                        market: Optional[str] = None) -> List[datetime.date]:
        """Get all trading days between two dates.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            market: Optional market to check for
            
        Returns:
            List of trading days
        """
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if not self.is_holiday(current_date, market):
                trading_days.append(current_date)
            
            current_date += datetime.timedelta(days=1)
        
        return trading_days
    
    def get_market_open_close(self, date: datetime.date, 
                             market: Optional[str] = None) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        """Get market open and close datetimes for a date.
        
        Args:
            date: Date to get market hours for
            market: Optional market to check for
            
        Returns:
            Tuple of (market open datetime, market close datetime), or None if market is closed
        """
        trading_hours = self.get_trading_hours(date, market)
        
        if trading_hours is None or trading_hours.status in [MarketStatus.CLOSED, MarketStatus.WEEKEND]:
            return None
        
        open_dt = datetime.datetime.combine(date, trading_hours.open_time)
        close_dt = datetime.datetime.combine(date, trading_hours.close_time)
        
        return (open_dt, close_dt)
    
    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Save the calendar to a JSON file.
        
        Args:
            file_path: Path to save the file to
        """
        data = {
            "name": self.name,
            "description": self.description,
            "markets": list(self.markets),
            "asset_classes": list(self.asset_classes),
            "default_trading_hours": {
                str(weekday): hours.to_dict() for weekday, hours in self.default_trading_hours.items()
            },
            "holidays": [
                holiday.to_dict() for holiday in self.holidays.values()
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, file_path: Union[str, Path]) -> 'HolidayCalendar':
        """Load a calendar from a JSON file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            HolidayCalendar instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        calendar = cls(name=data["name"], description=data.get("description", ""))
        
        # Load default trading hours
        for weekday_str, hours_data in data.get("default_trading_hours", {}).items():
            weekday = int(weekday_str)
            trading_hours = TradingHours.from_dict(hours_data)
            calendar.set_default_trading_hours(weekday, trading_hours)
        
        # Load holidays
        for holiday_data in data.get("holidays", []):
            holiday = HolidayEntry.from_dict(holiday_data)
            calendar.add_holiday(holiday)
        
        return calendar
    
    def load_holidays_from_csv(self, file_path: Union[str, Path], 
                              date_format: str = "%Y-%m-%d") -> None:
        """Load holidays from a CSV file.
        
        Expected CSV format:
        date,name,status,markets,asset_classes,description
        
        Args:
            file_path: Path to the CSV file
            date_format: Format string for parsing dates
        """
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse date
                date_str = row.get("date", "")
                if not date_str:
                    logger.warning(f"Skipping row with missing date: {row}")
                    continue
                
                try:
                    date = datetime.datetime.strptime(date_str, date_format).date()
                except ValueError:
                    logger.warning(f"Skipping row with invalid date format: {date_str}")
                    continue
                
                # Parse status
                status_str = row.get("status", "closed")
                try:
                    status = MarketStatus(status_str)
                except ValueError:
                    logger.warning(f"Invalid status '{status_str}', defaulting to CLOSED")
                    status = MarketStatus.CLOSED
                
                # Parse markets and asset classes
                markets_str = row.get("markets", "")
                markets = [m.strip() for m in markets_str.split(",") if m.strip()] if markets_str else []
                
                asset_classes_str = row.get("asset_classes", "")
                asset_classes = [a.strip() for a in asset_classes_str.split(",") if a.strip()] if asset_classes_str else []
                
                # Create holiday entry
                holiday = HolidayEntry(
                    date=date,
                    name=row.get("name", "Holiday"),
                    status=status,
                    affected_markets=markets,
                    affected_asset_classes=asset_classes,
                    description=row.get("description", "")
                )
                
                self.add_holiday(holiday)


class HolidayCalendarFactory:
    """Factory for creating holiday calendars."""
    @staticmethod
    def create_nyse_calendar() -> HolidayCalendar:
        """Create a calendar for NYSE holidays.
        
        Returns:
            HolidayCalendar for NYSE
        """
        calendar = HolidayCalendar(name="NYSE", description="New York Stock Exchange")
        
        # Set default trading hours
        # Monday to Friday: 9:30 AM to 4:00 PM
        for weekday in range(5):  # 0=Monday, 4=Friday
            trading_hours = TradingHours(
                open_time=datetime.time(9, 30),
                close_time=datetime.time(16, 0),
                pre_market_start=datetime.time(4, 0),
                post_market_end=datetime.time(20, 0),
                status=MarketStatus.OPEN
            )
            calendar.set_default_trading_hours(weekday, trading_hours)
        
        # Weekend trading hours (closed)
        weekend_hours = TradingHours(
            open_time=datetime.time(0, 0),
            close_time=datetime.time(0, 0),
            status=MarketStatus.WEEKEND,
            description="Weekend"
        )
        calendar.set_default_trading_hours(5, weekend_hours)  # Saturday
        calendar.set_default_trading_hours(6, weekend_hours)  # Sunday
        
        # Add standard NYSE holidays for the current year
        current_year = datetime.datetime.now().year
        
        # New Year's Day
        new_years = datetime.date(current_year, 1, 1)
        # If New Year's falls on a weekend, it's observed on the closest weekday
        if new_years.weekday() == 5:  # Saturday
            new_years = datetime.date(current_year, 1, 3)  # Following Monday
        elif new_years.weekday() == 6:  # Sunday
            new_years = datetime.date(current_year, 1, 2)  # Following Monday
        
        calendar.add_holiday(HolidayEntry(
            date=new_years,
            name="New Year's Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Martin Luther King Jr. Day (Third Monday in January)
        mlk_day = datetime.date(current_year, 1, 1)
        while mlk_day.weekday() != 0 or mlk_day.day <= 14:  # Find third Monday
            mlk_day += datetime.timedelta(days=1)
        
        calendar.add_holiday(HolidayEntry(
            date=mlk_day,
            name="Martin Luther King Jr. Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Presidents' Day (Third Monday in February)
        presidents_day = datetime.date(current_year, 2, 1)
        while presidents_day.weekday() != 0 or presidents_day.day <= 14:  # Find third Monday
            presidents_day += datetime.timedelta(days=1)
        
        calendar.add_holiday(HolidayEntry(
            date=presidents_day,
            name="Presidents' Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Good Friday (Date varies)
        # Note: This is a simplified calculation and may not be accurate for all years
        # For production use, consider using a library like workalendar or holidays
        # This is just a placeholder
        good_friday = datetime.date(current_year, 4, 15)  # Placeholder
        
        calendar.add_holiday(HolidayEntry(
            date=good_friday,
            name="Good Friday",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Memorial Day (Last Monday in May)
        memorial_day = datetime.date(current_year, 5, 31)
        while memorial_day.weekday() != 0:  # Find last Monday
            memorial_day -= datetime.timedelta(days=1)
        
        calendar.add_holiday(HolidayEntry(
            date=memorial_day,
            name="Memorial Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Juneteenth (June 19)
        juneteenth = datetime.date(current_year, 6, 19)
        # If Juneteenth falls on a weekend, it's observed on the closest weekday
        if juneteenth.weekday() == 5:  # Saturday
            juneteenth = datetime.date(current_year, 6, 18)  # Previous Friday
        elif juneteenth.weekday() == 6:  # Sunday
            juneteenth = datetime.date(current_year, 6, 20)  # Following Monday
        
        calendar.add_holiday(HolidayEntry(
            date=juneteenth,
            name="Juneteenth",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Independence Day (July 4)
        independence_day = datetime.date(current_year, 7, 4)
        # If Independence Day falls on a weekend, it's observed on the closest weekday
        if independence_day.weekday() == 5:  # Saturday
            independence_day = datetime.date(current_year, 7, 3)  # Previous Friday
        elif independence_day.weekday() == 6:  # Sunday
            independence_day = datetime.date(current_year, 7, 5)  # Following Monday
        
        calendar.add_holiday(HolidayEntry(
            date=independence_day,
            name="Independence Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Labor Day (First Monday in September)
        labor_day = datetime.date(current_year, 9, 1)
        while labor_day.weekday() != 0:  # Find first Monday
            labor_day += datetime.timedelta(days=1)
        
        calendar.add_holiday(HolidayEntry(
            date=labor_day,
            name="Labor Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Thanksgiving Day (Fourth Thursday in November)
        thanksgiving = datetime.date(current_year, 11, 1)
        while thanksgiving.weekday() != 3:  # Find first Thursday
            thanksgiving += datetime.timedelta(days=1)
        thanksgiving += datetime.timedelta(days=21)  # Add 3 weeks to get to the fourth Thursday
        
        calendar.add_holiday(HolidayEntry(
            date=thanksgiving,
            name="Thanksgiving Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Day after Thanksgiving (Early close)
        day_after_thanksgiving = thanksgiving + datetime.timedelta(days=1)
        
        early_close_hours = TradingHours(
            open_time=datetime.time(9, 30),
            close_time=datetime.time(13, 0),  # 1:00 PM close
            pre_market_start=datetime.time(4, 0),
            post_market_end=datetime.time(17, 0),
            status=MarketStatus.EARLY_CLOSE,
            description="Day after Thanksgiving (Early Close)"
        )
        
        calendar.add_holiday(HolidayEntry(
            date=day_after_thanksgiving,
            name="Day after Thanksgiving",
            status=MarketStatus.EARLY_CLOSE,
            trading_hours=early_close_hours,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Christmas Day (December 25)
        christmas = datetime.date(current_year, 12, 25)
        # If Christmas falls on a weekend, it's observed on the closest weekday
        if christmas.weekday() == 5:  # Saturday
            christmas = datetime.date(current_year, 12, 24)  # Previous Friday
        elif christmas.weekday() == 6:  # Sunday
            christmas = datetime.date(current_year, 12, 26)  # Following Monday
        
        calendar.add_holiday(HolidayEntry(
            date=christmas,
            name="Christmas Day",
            status=MarketStatus.CLOSED,
            affected_markets=["NYSE", "NASDAQ"],
            affected_asset_classes=["EQUITY", "OPTION", "ETF"]
        ))
        
        # Christmas Eve (Early close if weekday)
        christmas_eve = datetime.date(current_year, 12, 24)
        if christmas_eve.weekday() < 5:  # Weekday
            calendar.add_holiday(HolidayEntry(
                date=christmas_eve,
                name="Christmas Eve",
                status=MarketStatus.EARLY_CLOSE,
                trading_hours=early_close_hours,
                affected_markets=["NYSE", "NASDAQ"],
                affected_asset_classes=["EQUITY", "OPTION", "ETF"]
            ))
        
        return calendar
    
    @staticmethod
    def create_nasdaq_calendar() -> HolidayCalendar:
        """Create a calendar for NASDAQ holidays.
        
        Returns:
            HolidayCalendar for NASDAQ
        """
        # NASDAQ follows the same holiday schedule as NYSE
        return HolidayCalendarFactory.create_nyse_calendar()
    
    @staticmethod
    def create_lse_calendar() -> HolidayCalendar:
        """Create a calendar for London Stock Exchange holidays.
        
        Returns:
            HolidayCalendar for LSE
        """
        calendar = HolidayCalendar(name="LSE", description="London Stock Exchange")
        
        # Set default trading hours
        # Monday to Friday: 8:00 AM to 4:30 PM
        for weekday in range(5):  # 0=Monday, 4=Friday
            trading_hours = TradingHours(
                open_time=datetime.time(8, 0),
                close_time=datetime.time(16, 30),
                pre_market_start=datetime.time(7, 30),
                post_market_end=datetime.time(17, 15),
                status=MarketStatus.OPEN
            )
            calendar.set_default_trading_hours(weekday, trading_hours)
        
        # Weekend trading hours (closed)
        weekend_hours = TradingHours(
            open_time=datetime.time(0, 0),
            close_time=datetime.time(0, 0),
            status=MarketStatus.WEEKEND,
            description="Weekend"
        )
        calendar.set_default_trading_hours(5, weekend_hours)  # Saturday
        calendar.set_default_trading_hours(6, weekend_hours)  # Sunday
        
        # Add standard LSE holidays for the current year
        # This is a simplified version - for production use, consider using a library
        # like workalendar or holidays
        current_year = datetime.datetime.now().year
        
        # New Year's Day
        new_years = datetime.date(current_year, 1, 1)
        # If New Year's falls on a weekend, it's observed on the next Monday
        if new_years.weekday() >= 5:  # Weekend
            new_years = datetime.date(current_year, 1, 1 + (7 - new_years.weekday()) % 7)
        
        calendar.add_holiday(HolidayEntry(
            date=new_years,
            name="New Year's Day",
            status=MarketStatus.CLOSED,
            affected_markets=["LSE"],
            affected_asset_classes=["EQUITY", "ETF"]
        ))
        
        # Add more LSE holidays here
        # ...
        
        return calendar
    
    @staticmethod
    def create_tse_calendar() -> HolidayCalendar:
        """Create a calendar for Tokyo Stock Exchange holidays.
        
        Returns:
            HolidayCalendar for TSE
        """
        calendar = HolidayCalendar(name="TSE", description="Tokyo Stock Exchange")
        
        # Set default trading hours
        # Monday to Friday: 9:00 AM to 3:00 PM with lunch break
        for weekday in range(5):  # 0=Monday, 4=Friday
            trading_hours = TradingHours(
                open_time=datetime.time(9, 0),
                close_time=datetime.time(15, 0),
                lunch_break_start=datetime.time(11, 30),
                lunch_break_end=datetime.time(12, 30),
                status=MarketStatus.OPEN
            )
            calendar.set_default_trading_hours(weekday, trading_hours)
        
        # Weekend trading hours (closed)
        weekend_hours = TradingHours(
            open_time=datetime.time(0, 0),
            close_time=datetime.time(0, 0),
            status=MarketStatus.WEEKEND,
            description="Weekend"
        )
        calendar.set_default_trading_hours(5, weekend_hours)  # Saturday
        calendar.set_default_trading_hours(6, weekend_hours)  # Sunday
        
        # Add standard TSE holidays for the current year
        # This is a simplified version - for production use, consider using a library
        # like workalendar or holidays
        # ...
        
        return calendar
    
    @staticmethod
    def create_calendar(calendar_type: str) -> HolidayCalendar:
        """Create a calendar based on a predefined type.
        
        Args:
            calendar_type: Type of calendar to create (NYSE, NASDAQ, LSE, TSE)
            
        Returns:
            HolidayCalendar instance
        """
        if calendar_type.upper() == "NYSE":
            return HolidayCalendarFactory.create_nyse_calendar()
        elif calendar_type.upper() == "NASDAQ":
            return HolidayCalendarFactory.create_nasdaq_calendar()
        elif calendar_type.upper() == "LSE":
            return HolidayCalendarFactory.create_lse_calendar()
        elif calendar_type.upper() == "TSE":
            return HolidayCalendarFactory.create_tse_calendar()
        else:
            raise ValueError(f"Unknown calendar type: {calendar_type}")


class CalendarManager:
    """Manager for multiple holiday calendars."""
    def __init__(self):
        """Initialize the calendar manager."""
        self.calendars: Dict[str, HolidayCalendar] = {}
    
    def add_calendar(self, calendar: HolidayCalendar) -> None:
        """Add a calendar to the manager.
        
        Args:
            calendar: Calendar to add
        """
        self.calendars[calendar.name] = calendar
    
    def get_calendar(self, name: str) -> Optional[HolidayCalendar]:
        """Get a calendar by name.
        
        Args:
            name: Name of the calendar
            
        Returns:
            HolidayCalendar instance, or None if not found
        """
        return self.calendars.get(name)
    
    def is_market_open(self, dt: datetime.datetime, market: str) -> bool:
        """Check if a market is open at a datetime.
        
        Args:
            dt: Datetime to check
            market: Market to check
            
        Returns:
            True if the market is open, False otherwise
        """
        calendar = self.get_calendar(market)
        
        if calendar is None:
            logger.warning(f"No calendar found for market: {market}")
            return False
        
        return calendar.is_trading_time(dt, market)
    
    def get_market_hours(self, date: datetime.date, market: str) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        """Get market open and close times for a date.
        
        Args:
            date: Date to check
            market: Market to check
            
        Returns:
            Tuple of (market open datetime, market close datetime), or None if market is closed
        """
        calendar = self.get_calendar(market)
        
        if calendar is None:
            logger.warning(f"No calendar found for market: {market}")
            return None
        
        return calendar.get_market_open_close(date, market)
    
    def get_next_market_open(self, dt: datetime.datetime, market: str) -> Optional[datetime.datetime]:
        """Get the next market open time after a datetime.
        
        Args:
            dt: Starting datetime
            market: Market to check
            
        Returns:
            Next market open datetime, or None if no calendar found
        """
        calendar = self.get_calendar(market)
        
        if calendar is None:
            logger.warning(f"No calendar found for market: {market}")
            return None
        
        current_date = dt.date()
        
        # Check if market opens later today
        market_hours = calendar.get_market_open_close(current_date, market)
        
        if market_hours is not None:
            market_open, _ = market_hours
            
            if market_open > dt:
                return market_open
        
        # Find next trading day
        next_date = calendar.get_next_trading_day(current_date, market)
        market_hours = calendar.get_market_open_close(next_date, market)
        
        if market_hours is not None:
            market_open, _ = market_hours
            return market_open
        
        return None
    
    def get_next_market_close(self, dt: datetime.datetime, market: str) -> Optional[datetime.datetime]:
        """Get the next market close time after a datetime.
        
        Args:
            dt: Starting datetime
            market: Market to check
            
        Returns:
            Next market close datetime, or None if no calendar found
        """
        calendar = self.get_calendar(market)
        
        if calendar is None:
            logger.warning(f"No calendar found for market: {market}")
            return None
        
        current_date = dt.date()
        
        # Check if market closes later today
        market_hours = calendar.get_market_open_close(current_date, market)
        
        if market_hours is not None:
            _, market_close = market_hours
            
            if market_close > dt:
                return market_close
        
        # Find next trading day
        next_date = calendar.get_next_trading_day(current_date, market)
        market_hours = calendar.get_market_open_close(next_date, market)
        
        if market_hours is not None:
            _, market_close = market_hours
            return market_close
        
        return None
    
    def initialize_default_calendars(self) -> None:
        """Initialize default calendars for common markets."""
        self.add_calendar(HolidayCalendarFactory.create_nyse_calendar())
        self.add_calendar(HolidayCalendarFactory.create_nasdaq_calendar())
        self.add_calendar(HolidayCalendarFactory.create_lse_calendar())
        self.add_calendar(HolidayCalendarFactory.create_tse_calendar())


# Create a global calendar manager instance
default_calendar_manager = CalendarManager()
default_calendar_manager.initialize_default_calendars()


def get_calendar_manager() -> CalendarManager:
    """Get the default calendar manager instance.
    
    Returns:
        Default CalendarManager instance
    """
    return default_calendar_manager