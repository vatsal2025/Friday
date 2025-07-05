"""Timeframe management module for the Friday AI Trading System."""

class TimeframeManager:
    """Class for managing timeframes in the trading system.
    
    This class provides methods for converting between different timeframes,
    aligning data from different timeframes, and managing timeframe-related operations.
    """
    
    def __init__(self):
        """Initialize a timeframe manager."""
        self.timeframes = {}
        
    def register_timeframe(self, name, interval):
        """Register a new timeframe.
        
        Args:
            name: The name of the timeframe.
            interval: The interval of the timeframe in minutes.
        """
        self.timeframes[name] = interval
        
    def get_timeframe(self, name):
        """Get a timeframe by name.
        
        Args:
            name: The name of the timeframe.
            
        Returns:
            The interval of the timeframe in minutes.
        """
        return self.timeframes.get(name)