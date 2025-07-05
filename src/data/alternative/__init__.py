"""Alternative data integration module for the Friday AI Trading System.

This module provides classes and functions for integrating alternative data sources
such as news sentiment, social media, and economic data.
"""

from src.data.alternative.news_sentiment_analyzer import NewsSentimentAnalyzer
from src.data.alternative.social_media_analyzer import SocialMediaAnalyzer
from src.data.alternative.economic_data_provider import EconomicDataProvider
from src.data.alternative.alternative_data_normalizer import AlternativeDataNormalizer

__all__ = [
    'NewsSentimentAnalyzer',
    'SocialMediaAnalyzer',
    'EconomicDataProvider',
    'AlternativeDataNormalizer',
]