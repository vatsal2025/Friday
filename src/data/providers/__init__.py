"""Data providers module for the Friday AI Trading System.

This module provides classes for retrieving data from various sources.
"""

from src.data.providers.data_provider import DataProvider, ProviderError
from src.data.providers.market_data_provider import MarketDataProvider
from src.data.providers.alternative_data_provider import AlternativeDataProvider
from src.data.providers.news_provider import NewsProvider
from src.data.providers.social_media_provider import SocialMediaProvider

__all__ = [
    "DataProvider",
    "ProviderError",
    "MarketDataProvider",
    "AlternativeDataProvider",
    "NewsProvider",
    "SocialMediaProvider",
]