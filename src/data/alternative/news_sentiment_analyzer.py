"""News sentiment analysis module for the Friday AI Trading System.

This module provides functionality for analyzing news sentiment related to financial markets.
"""

import pandas as pd
import requests
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

# API client libraries
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one
)

# Create logger
logger = get_logger(__name__)

class NewsSentimentAnalyzer:
    """Class for analyzing news sentiment related to financial markets.
    
    This class provides methods for fetching news articles from various sources,
    analyzing sentiment, and storing the results in MongoDB.
    
    Attributes:
        config_manager: Configuration manager.
        config: Configuration dictionary.
        news_collection: MongoDB collection for storing news articles.
        sentiment_collection: MongoDB collection for storing sentiment analysis results.
        newsapi_client: NewsAPI client.
        alpha_vantage_api_key: Alpha Vantage API key.
        bloomberg_api_key: Bloomberg API key.
        sentiment_analyzer: VADER sentiment analyzer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NewsSentimentAnalyzer with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('alternative_data')
        
        # Initialize MongoDB collections
        self.news_collection = get_collection('news_data')
        self.sentiment_collection = get_collection('news_sentiment')
        
        # Initialize API clients
        self._init_api_clients()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info("News Sentiment Analyzer initialized")
    
    def _init_api_clients(self):
        """Initialize API clients for different news sources."""
        try:
            # Initialize NewsAPI client
            newsapi_key = self.config.get('news_sentiment', {}).get('api_keys', {}).get('newsapi')
            if newsapi_key:
                self.newsapi_client = NewsApiClient(api_key=newsapi_key)
                logger.info("NewsAPI client initialized")
            else:
                self.newsapi_client = None
                logger.warning("NewsAPI key not found in config")
            
            # Store Alpha Vantage API key
            self.alpha_vantage_api_key = self.config.get('news_sentiment', {}).get('api_keys', {}).get('alpha_vantage')
            if not self.alpha_vantage_api_key:
                logger.warning("Alpha Vantage API key not found in config")
            
            # Store Bloomberg API key
            self.bloomberg_api_key = self.config.get('news_sentiment', {}).get('api_keys', {}).get('bloomberg')
            if not self.bloomberg_api_key:
                logger.warning("Bloomberg API key not found in config")
            
        except Exception as e:
            logger.error(f"Error initializing API clients: {str(e)}")
            raise
    
    def fetch_news(self, symbols: List[str], days: int = 1) -> List[Dict[str, Any]]:
        """Fetch news articles for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch news for.
            days: Number of days to look back.
            
        Returns:
            List of news articles as dictionaries.
        """
        logger.info(f"Fetching news for symbols: {symbols} for the last {days} days")
        
        all_articles = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API calls
        end_date_str = end_date.strftime("%Y-%m-%d")
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Maximum articles per symbol from config
        max_articles = self.config.get('news_sentiment', {}).get('max_articles_per_symbol', 50)
        
        # Try NewsAPI first (has better search capabilities)
        if self.newsapi_client:
            try:
                for symbol in symbols:
                    # Create search query for the symbol
                    # Include company name if available (would need a mapping of symbols to company names)
                    query = f"{symbol} stock OR {symbol} finance OR {symbol} earnings OR {symbol} company"
                    
                    # Search for articles
                    response = self.newsapi_client.get_everything(
                        q=query,
                        from_param=start_date_str,
                        to=end_date_str,
                        language='en',
                        sort_by='relevancy',
                        page_size=min(max_articles, 100)  # NewsAPI limits to 100 per request
                    )
                    
                    articles = response.get('articles', [])
                    logger.info(f"Found {len(articles)} articles from NewsAPI for symbol {symbol}")
                    
                    for article in articles:
                        # Create article object
                        news_article = {
                            "symbol": symbol,
                            "source": "newsapi",
                            "source_name": article.get('source', {}).get('name', 'Unknown'),
                            "author": article.get('author', 'Unknown'),
                            "title": article.get('title', ''),
                            "description": article.get('description', ''),
                            "content": article.get('content', ''),
                            "url": article.get('url', ''),
                            "image_url": article.get('urlToImage', ''),
                            "published_at": article.get('publishedAt', ''),
                            "fetched_at": datetime.now().isoformat()
                        }
                        all_articles.append(news_article)
                    
                    # Rate limiting - NewsAPI has rate limits
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching from NewsAPI: {str(e)}")
        
        # If we have Alpha Vantage API key, fetch news from there as well
        if self.alpha_vantage_api_key and (not all_articles or len(all_articles) < max_articles * len(symbols)):
            try:
                for symbol in symbols:
                    # Alpha Vantage News Sentiment API endpoint
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_api_key}"
                    
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        feed = data.get('feed', [])
                        
                        # Limit to max_articles per symbol
                        remaining_articles = max_articles - len([a for a in all_articles if a['symbol'] == symbol])
                        feed = feed[:remaining_articles]
                        
                        logger.info(f"Found {len(feed)} articles from Alpha Vantage for symbol {symbol}")
                        
                        for item in feed:
                            # Parse time
                            time_published = item.get('time_published', '')
                            if time_published:
                                try:
                                    published_at = datetime.strptime(time_published, "%Y%m%dT%H%M%S").isoformat()
                                except ValueError:
                                    published_at = time_published
                            else:
                                published_at = ''
                            
                            # Get sentiment if available
                            sentiment = item.get('overall_sentiment_score')
                            sentiment_label = item.get('overall_sentiment_label')
                            
                            # Create article object
                            news_article = {
                                "symbol": symbol,
                                "source": "alpha_vantage",
                                "source_name": item.get('source', 'Alpha Vantage'),
                                "author": item.get('authors', []),
                                "title": item.get('title', ''),
                                "description": item.get('summary', ''),
                                "content": item.get('summary', ''),  # Alpha Vantage provides summary, not full content
                                "url": item.get('url', ''),
                                "image_url": item.get('banner_image', ''),
                                "published_at": published_at,
                                "fetched_at": datetime.now().isoformat()
                            }
                            
                            # Add sentiment data if available
                            if sentiment is not None:
                                news_article["alpha_vantage_sentiment_score"] = sentiment
                                news_article["alpha_vantage_sentiment_label"] = sentiment_label
                            
                            all_articles.append(news_article)
                    else:
                        logger.warning(f"Alpha Vantage API returned status code {response.status_code} for symbol {symbol}")
                    
                    # Rate limiting - Alpha Vantage has strict rate limits
                    time.sleep(15)  # Alpha Vantage free tier has 5 calls per minute limit
                
            except Exception as e:
                logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
        
        # Store news articles in MongoDB
        if all_articles:
            self._store_news_data(all_articles)
        
        return all_articles
    
    def analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for the provided news articles.
        
        Args:
            news_data: List of news articles as dictionaries.
            
        Returns:
            List of sentiment analysis results.
        """
        logger.info(f"Analyzing sentiment for {len(news_data)} news articles")
        
        sentiment_results = []
        
        for article in news_data:
            try:
                # Check if Alpha Vantage already provided sentiment
                if 'alpha_vantage_sentiment_score' in article:
                    sentiment_score = article['alpha_vantage_sentiment_score']
                    
                    # Alpha Vantage scores are between -1 and 1, normalize to 0-1
                    normalized_score = (sentiment_score + 1) / 2
                    
                    # Map Alpha Vantage sentiment label to our format
                    av_label = article.get('alpha_vantage_sentiment_label', '').lower()
                    if av_label == 'bullish' or av_label == 'somewhat_bullish':
                        sentiment_label = 'positive'
                    elif av_label == 'bearish' or av_label == 'somewhat_bearish':
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    # Confidence is fixed for Alpha Vantage provided sentiment
                    confidence = 0.8
                else:
                    # Use VADER for sentiment analysis
                    # Combine title and content for better analysis
                    text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                    
                    # Get sentiment scores from VADER
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                    
                    # Map compound score to a 0-1 scale (VADER compound is -1 to 1)
                    normalized_score = (sentiment_scores['compound'] + 1) / 2
                    
                    # Determine sentiment label
                    if sentiment_scores['compound'] >= 0.05:
                        sentiment_label = 'positive'
                    elif sentiment_scores['compound'] <= -0.05:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    # Confidence based on the magnitude of the compound score
                    confidence = abs(sentiment_scores['compound'])
                
                # Create sentiment result
                sentiment = {
                    "article_id": article.get("_id", ""),  # MongoDB ID if available
                    "symbol": article.get("symbol", ""),
                    "source": article.get("source", ""),
                    "source_name": article.get("source_name", ""),
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                    "sentiment_score": normalized_score,
                    "sentiment_label": sentiment_label,
                    "confidence": confidence,
                    "analyzed_at": datetime.now().isoformat()
                }
                sentiment_results.append(sentiment)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for article: {str(e)}")
        
        # Store sentiment results in MongoDB
        if sentiment_results:
            self._store_sentiment_data(sentiment_results)
        
        return sentiment_results
    
    def get_sentiment_for_symbols(self, symbols: List[str], days: int = 7) -> pd.DataFrame:
        """Get sentiment analysis results for the specified symbols over the last N days.
        
        Args:
            symbols: List of stock symbols to get sentiment for.
            days: Number of days to look back.
            
        Returns:
            DataFrame with sentiment analysis results.
        """
        logger.info(f"Getting sentiment for symbols: {symbols} over last {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query MongoDB for sentiment data
        query = {
            "symbol": {"$in": symbols},
            "published_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }
        
        # Find sentiment data in MongoDB
        sentiment_data = list(find("news_sentiment", query))
        
        # Convert to DataFrame
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            return df
        else:
            logger.warning(f"No sentiment data found for symbols: {symbols}")
            return pd.DataFrame()
    
    def calculate_sentiment_score(self, symbols: List[str], days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Calculate aggregated sentiment scores for the specified symbols.
        
        Args:
            symbols: List of stock symbols to calculate sentiment for.
            days: Number of days to look back.
            
        Returns:
            Dictionary mapping symbols to sentiment scores.
        """
        logger.info(f"Calculating sentiment scores for symbols: {symbols} over last {days} days")
        
        # Get sentiment data
        sentiment_df = self.get_sentiment_for_symbols(symbols, days)
        
        sentiment_scores = {}
        
        for symbol in symbols:
            # Filter sentiment for this symbol
            symbol_df = sentiment_df[sentiment_df['symbol'] == symbol] if not sentiment_df.empty else pd.DataFrame()
            
            if not symbol_df.empty:
                # Count articles
                article_count = len(symbol_df)
                
                # Calculate weighted average sentiment score
                weighted_sentiment = (symbol_df['sentiment_score'] * symbol_df['confidence']).sum() / symbol_df['confidence'].sum()
                
                # Count articles by sentiment label
                sentiment_counts = symbol_df['sentiment_label'].value_counts().to_dict()
                
                # Calculate percentages
                sentiment_percentages = {
                    label: count / article_count 
                    for label, count in sentiment_counts.items()
                }
                
                # Ensure all labels are present
                for label in ['positive', 'neutral', 'negative']:
                    if label not in sentiment_percentages:
                        sentiment_percentages[label] = 0.0
                
                # Create result
                sentiment_scores[symbol] = {
                    'sentiment_score': weighted_sentiment,
                    'article_count': article_count,
                    'positive_pct': sentiment_percentages.get('positive', 0.0),
                    'neutral_pct': sentiment_percentages.get('neutral', 0.0),
                    'negative_pct': sentiment_percentages.get('negative', 0.0),
                    'days': days
                }
            else:
                # No data for this symbol
                sentiment_scores[symbol] = {
                    'sentiment_score': 0.5,  # Neutral
                    'article_count': 0,
                    'positive_pct': 0.0,
                    'neutral_pct': 0.0,
                    'negative_pct': 0.0,
                    'days': days
                }
        
        return sentiment_scores
    
    def _store_news_data(self, news_data: List[Dict[str, Any]]) -> None:
        """Store news articles in MongoDB.
        
        Args:
            news_data: List of news articles as dictionaries.
        """
        try:
            # Insert news data into MongoDB
            insert_many("news_data", news_data)
            logger.info(f"Stored {len(news_data)} news articles in MongoDB")
        except Exception as e:
            logger.error(f"Error storing news data in MongoDB: {str(e)}")
    
    def _store_sentiment_data(self, sentiment_data: List[Dict[str, Any]]) -> None:
        """Store sentiment analysis results in MongoDB.
        
        Args:
            sentiment_data: List of sentiment analysis results as dictionaries.
        """
        try:
            # Insert sentiment data into MongoDB
            insert_many("news_sentiment", sentiment_data)
            logger.info(f"Stored {len(sentiment_data)} sentiment results in MongoDB")
        except Exception as e:
            logger.error(f"Error storing sentiment data in MongoDB: {str(e)}")