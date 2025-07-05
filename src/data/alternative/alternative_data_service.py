"""Alternative data integration service for the Friday AI Trading System.

This module provides a service for integrating alternative data sources
and making them available to the trading system.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one, update_one
)

from src.data.alternative.news_sentiment_analyzer import NewsSentimentAnalyzer
from src.data.alternative.social_media_analyzer import SocialMediaAnalyzer
from src.data.alternative.economic_data_provider import EconomicDataProvider
from src.data.alternative.alternative_data_normalizer import AlternativeDataNormalizer

# Create logger
logger = get_logger(__name__)

class AlternativeDataService:
    """Service for integrating alternative data sources.
    
    This class orchestrates the process of fetching, analyzing, normalizing,
    and integrating alternative data from various sources.
    
    Attributes:
        config_manager: Configuration manager.
        config: Configuration dictionary.
        news_analyzer: News sentiment analyzer.
        social_analyzer: Social media analyzer.
        economic_provider: Economic data provider.
        data_normalizer: Alternative data normalizer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AlternativeDataService with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('alternative_data')
        
        # Initialize component services
        self.news_analyzer = NewsSentimentAnalyzer(self.config)
        self.social_analyzer = SocialMediaAnalyzer(self.config)
        self.economic_provider = EconomicDataProvider(self.config)
        self.data_normalizer = AlternativeDataNormalizer(self.config)
        
        logger.info("Alternative Data Service initialized")
    
    def update_all_alternative_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Update all alternative data for the specified symbols.
        
        This method orchestrates the process of fetching and processing
        all types of alternative data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to update data for.
            
        Returns:
            Dictionary with status and summary of updated data.
        """
        logger.info(f"Updating all alternative data for symbols: {symbols}")
        
        results = {}
        errors = []
        
        try:
            # Update news sentiment data
            news_result = self.update_news_sentiment(symbols)
            results['news_sentiment'] = news_result
            
            if news_result.get('status') == 'error':
                errors.append(f"News sentiment error: {news_result.get('error')}")
        except Exception as e:
            error_msg = f"Error updating news sentiment data: {str(e)}"
            logger.error(error_msg)
            results['news_sentiment'] = {'status': 'error', 'error': error_msg}
            errors.append(error_msg)
        
        try:
            # Update social media data
            social_result = self.update_social_media_data(symbols)
            results['social_media'] = social_result
            
            if social_result.get('status') == 'error':
                errors.append(f"Social media error: {social_result.get('error')}")
        except Exception as e:
            error_msg = f"Error updating social media data: {str(e)}"
            logger.error(error_msg)
            results['social_media'] = {'status': 'error', 'error': error_msg}
            errors.append(error_msg)
        
        try:
            # Update economic data
            economic_result = self.update_economic_data()
            results['economic_data'] = economic_result
            
            if economic_result.get('status') == 'error':
                errors.append(f"Economic data error: {economic_result.get('error')}")
        except Exception as e:
            error_msg = f"Error updating economic data: {str(e)}"
            logger.error(error_msg)
            results['economic_data'] = {'status': 'error', 'error': error_msg}
            errors.append(error_msg)
        
        # Determine overall status
        if errors:
            overall_status = 'partial' if len(errors) < 3 else 'error'
        else:
            overall_status = 'success'
        
        # Create summary
        summary = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'results': results
        }
        
        if errors:
            summary['errors'] = errors
        
        # Store summary in MongoDB
        try:
            insert_one('alternative_data_updates', summary)
        except Exception as e:
            logger.error(f"Error storing update summary in MongoDB: {str(e)}")
        
        return summary
    
    def update_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Update news sentiment data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to update news sentiment for.
            
        Returns:
            Dictionary with status and summary of updated data.
        """
        logger.info(f"Updating news sentiment data for symbols: {symbols}")
        
        try:
            # Fetch news data
            news_data = self.news_analyzer.fetch_news(symbols)
            
            # Analyze sentiment
            sentiment_results = self.news_analyzer.analyze_sentiment(news_data)
            
            # Convert to DataFrame for normalization
            sentiment_df = pd.DataFrame(sentiment_results)
            
            # Normalize sentiment data
            normalized_df = self.data_normalizer.normalize_news_sentiment(sentiment_df)
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'news_count': len(news_data),
                'sentiment_count': len(sentiment_results),
                'normalized_count': len(normalized_df)
            }
        
        except Exception as e:
            error_msg = f"Error updating news sentiment data: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'error': error_msg
            }
    
    def update_social_media_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Update social media data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to update social media data for.
            
        Returns:
            Dictionary with status and summary of updated data.
        """
        logger.info(f"Updating social media data for symbols: {symbols}")
        
        try:
            # Fetch social media data
            social_data = self.social_analyzer.fetch_social_data(symbols)
            
            # Analyze sentiment
            sentiment_results = self.social_analyzer.analyze_social_sentiment(social_data)
            
            # Flatten sentiment results for normalization
            flattened_results = []
            for platform, platform_results in sentiment_results.items():
                flattened_results.extend(platform_results)
            
            # Convert to DataFrame for normalization
            sentiment_df = pd.DataFrame(flattened_results)
            
            # Normalize sentiment data
            normalized_df = self.data_normalizer.normalize_social_metrics(sentiment_df)
            
            # Calculate buzz scores
            buzz_scores = self.social_analyzer.calculate_social_buzz(symbols)
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'platforms': list(social_data.keys()),
                'post_count': sum(len(posts) for posts in social_data.values()),
                'sentiment_count': sum(len(results) for results in sentiment_results.values()),
                'normalized_count': len(normalized_df),
                'buzz_scores': buzz_scores
            }
        
        except Exception as e:
            error_msg = f"Error updating social media data: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'error': error_msg
            }
    
    def update_economic_data(self) -> Dict[str, Any]:
        """Update economic data.
        
        Returns:
            Dictionary with status and summary of updated data.
        """
        logger.info("Updating economic data")
        
        try:
            # Get economic indicators to fetch from config
            indicators = self.config.get('economic_indicators', ['gdp', 'inflation', 'unemployment', 'interest_rate'])
            
            # Fetch economic indicators
            economic_data = self.economic_provider.fetch_economic_indicators(indicators)
            
            # Fetch central bank data
            central_bank_data = self.economic_provider.fetch_central_bank_data()
            
            # Flatten economic indicator data for normalization
            flattened_indicators = []
            for indicator, countries_data in economic_data.items():
                for country, data in countries_data.items():
                    flattened_indicators.append(data)
            
            # Convert to DataFrame for normalization
            economic_df = pd.DataFrame(flattened_indicators)
            
            # Normalize economic data
            normalized_df = self.data_normalizer.normalize_economic_data(economic_df)
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'central_banks': list(central_bank_data.keys()),
                'indicator_count': len(flattened_indicators),
                'normalized_count': len(normalized_df)
            }
        
        except Exception as e:
            error_msg = f"Error updating economic data: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            }
    
    def get_alternative_data_features(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Get alternative data features for the specified symbols.
        
        Args:
            symbols: List of stock symbols to get features for.
            days: Number of days to look back.
            
        Returns:
            DataFrame with alternative data features.
        """
        logger.info(f"Getting alternative data features for symbols: {symbols} over last {days} days")
        
        try:
            # Get feature matrix
            feature_matrix, feature_names = self.data_normalizer.get_feature_matrix(symbols, days=days)
            
            if feature_matrix.empty:
                logger.warning(f"No alternative data features found for symbols: {symbols}")
            else:
                logger.info(f"Retrieved {len(feature_names)} alternative data features for {len(symbols)} symbols")
            
            return feature_matrix
        
        except Exception as e:
            logger.error(f"Error getting alternative data features: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_alternative_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get the latest alternative data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to get data for.
            
        Returns:
            Dictionary mapping symbols to dictionaries of latest data.
        """
        logger.info(f"Getting latest alternative data for symbols: {symbols}")
        
        latest_data = {}
        
        try:
            # Get latest news sentiment
            news_sentiment = self.news_analyzer.get_sentiment_for_symbols(symbols, days=1)
            
            # Get latest social media metrics
            social_metrics = self.social_analyzer.get_social_metrics(symbols, days=1)
            
            # Get latest economic indicators
            economic_indicators = self.economic_provider.get_economic_indicators(
                ['gdp', 'inflation', 'unemployment', 'interest_rate'], days=1)
            
            # Get social buzz scores
            buzz_scores = self.social_analyzer.calculate_social_buzz(symbols, days=1)
            
            # Organize data by symbol
            for symbol in symbols:
                symbol_data = {
                    'news_sentiment': {},
                    'social_metrics': {},
                    'economic_indicators': {},
                    'social_buzz': buzz_scores.get(symbol, 0.0)
                }
                
                # Add news sentiment if available
                if not news_sentiment.empty:
                    symbol_news = news_sentiment[news_sentiment['symbol'] == symbol]
                    if not symbol_news.empty:
                        latest_news = symbol_news.iloc[0].to_dict()
                        symbol_data['news_sentiment'] = latest_news
                
                # Add social metrics if available
                if not social_metrics.empty:
                    symbol_social = social_metrics[social_metrics['symbol'] == symbol]
                    if not symbol_social.empty:
                        # Group by platform and get latest for each
                        platforms = symbol_social['platform'].unique()
                        for platform in platforms:
                            platform_data = symbol_social[symbol_social['platform'] == platform]
                            if not platform_data.empty:
                                latest_platform = platform_data.iloc[0].to_dict()
                                symbol_data['social_metrics'][platform] = latest_platform
                
                latest_data[symbol] = symbol_data
            
            # Add economic indicators (same for all symbols)
            if not economic_indicators.empty:
                for symbol in symbols:
                    for _, row in economic_indicators.iterrows():
                        indicator = row['indicator']
                        country = row['country']
                        key = f"{indicator}_{country}"
                        latest_data[symbol]['economic_indicators'][key] = row.to_dict()
            
            return latest_data
        
        except Exception as e:
            logger.error(f"Error getting latest alternative data: {str(e)}")
            return {symbol: {} for symbol in symbols}