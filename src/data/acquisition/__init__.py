"""Data acquisition module for the Friday AI Trading System.

This module provides components for fetching historical and real-time market data.
"""

from src.data.acquisition.data_fetcher import (
    DataFetcher,
    DataSourceAdapter,
    DataValidationError,
    DataConnectionError,
)
from src.data.acquisition.historical_data_fetcher import HistoricalDataFetcher
from src.data.acquisition.real_time_data_stream import RealTimeDataStream
from src.data.acquisition.market_calendar import MarketCalendar

__all__ = [
    "DataFetcher",
    "HistoricalDataFetcher",
    "RealTimeDataStream",
    "MarketCalendar",
    "DataSourceAdapter",
    "DataValidationError",
    "DataConnectionError",
]