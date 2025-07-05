"""Social media analysis module for the Friday AI Trading System.

This module provides functionality for analyzing social media sentiment related to financial markets.
"""

import pandas as pd
import requests
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

# API client libraries
import tweepy
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one
)

# Create logger
logger = get_logger(__name__)

class SocialMediaAnalyzer:
    """Class for analyzing social media sentiment related to financial markets.
    
    This class provides methods for fetching social media posts from various platforms,
    analyzing sentiment, and storing the results in MongoDB.
    
    Attributes:
        config_manager: Configuration manager.
        config: Configuration dictionary.
        posts_collection: MongoDB collection for storing social media posts.
        metrics_collection: MongoDB collection for storing social media metrics.
        twitter_api: Twitter API client.
        reddit_api: Reddit API client.
        stocktwits_api_key: Stocktwits API key.
        sentiment_analyzer: VADER sentiment analyzer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SocialMediaAnalyzer with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('alternative_data')
        
        # Initialize MongoDB collections
        self.posts_collection = get_collection('social_media_posts')
        self.metrics_collection = get_collection('social_media_metrics')
        
        # Initialize API clients
        self._init_api_clients()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info("Social Media Analyzer initialized")
    
    def _init_api_clients(self):
        """Initialize API clients for different social media platforms."""
        try:
            # Initialize Twitter API client
            twitter_config = self.config.get('social_media', {}).get('api_keys', {}).get('twitter', {})
            if twitter_config.get('api_key') and twitter_config.get('api_secret') and \
               twitter_config.get('access_token') and twitter_config.get('access_token_secret'):
                auth = tweepy.OAuth1UserHandler(
                    twitter_config.get('api_key'),
                    twitter_config.get('api_secret'),
                    twitter_config.get('access_token'),
                    twitter_config.get('access_token_secret')
                )
                self.twitter_api = tweepy.API(auth)
                logger.info("Twitter API client initialized")
            else:
                self.twitter_api = None
                logger.warning("Twitter API credentials not found in config")
            
            # Initialize Reddit API client
            reddit_config = self.config.get('social_media', {}).get('api_keys', {}).get('reddit', {})
            if reddit_config.get('client_id') and reddit_config.get('client_secret') and reddit_config.get('user_agent'):
                self.reddit_api = praw.Reddit(
                    client_id=reddit_config.get('client_id'),
                    client_secret=reddit_config.get('client_secret'),
                    user_agent=reddit_config.get('user_agent')
                )
                logger.info("Reddit API client initialized")
            else:
                self.reddit_api = None
                logger.warning("Reddit API credentials not found in config")
            
            # Store Stocktwits API key
            self.stocktwits_api_key = self.config.get('social_media', {}).get('api_keys', {}).get('stocktwits', {})
            if not self.stocktwits_api_key:
                logger.warning("Stocktwits API key not found in config")
            
        except Exception as e:
            logger.error(f"Error initializing API clients: {str(e)}")
            raise
    
    def fetch_social_data(self, symbols: List[str], platforms: List[str] = None, days: int = 1) -> List[Dict[str, Any]]:
        """Fetch social media posts for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch posts for.
            platforms: List of social media platforms to fetch from. Defaults to ['twitter', 'reddit', 'stocktwits'].
            days: Number of days to look back.
            
        Returns:
            List of social media posts as dictionaries.
        """
        if platforms is None:
            platforms = ['twitter', 'reddit', 'stocktwits']
        
        logger.info(f"Fetching social media data for symbols: {symbols} from platforms: {platforms}")
        
        all_posts = []
        
        # Fetch from Twitter
        if 'twitter' in platforms and self.twitter_api:
            try:
                twitter_posts = self._fetch_twitter_data(symbols, days)
                all_posts.extend(twitter_posts)
                logger.info(f"Fetched {len(twitter_posts)} posts from Twitter")
            except Exception as e:
                logger.error(f"Error fetching from Twitter: {str(e)}")
        
        # Fetch from Reddit
        if 'reddit' in platforms and self.reddit_api:
            try:
                reddit_posts = self._fetch_reddit_data(symbols, days)
                all_posts.extend(reddit_posts)
                logger.info(f"Fetched {len(reddit_posts)} posts from Reddit")
            except Exception as e:
                logger.error(f"Error fetching from Reddit: {str(e)}")
        
        # Fetch from Stocktwits
        if 'stocktwits' in platforms:
            try:
                stocktwits_posts = self._fetch_stocktwits_data(symbols, days)
                all_posts.extend(stocktwits_posts)
                logger.info(f"Fetched {len(stocktwits_posts)} posts from Stocktwits")
            except Exception as e:
                logger.error(f"Error fetching from Stocktwits: {str(e)}")
        
        # Store social media posts in MongoDB
        if all_posts:
            self._store_social_posts(all_posts)
        
        return all_posts
    
    def _fetch_twitter_data(self, symbols: List[str], days: int) -> List[Dict[str, Any]]:
        """Fetch tweets for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch tweets for.
            days: Number of days to look back.
            
        Returns:
            List of tweets as dictionaries.
        """
        posts = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Maximum posts per symbol from config
        max_posts = self.config.get('social_media', {}).get('max_posts_per_symbol', 100)
        
        for symbol in symbols:
            try:
                # Search for tweets containing the symbol (with $ prefix for cashtags)
                query = f"${symbol} OR #{symbol} OR {symbol} stock OR {symbol} finance"
                
                # Search tweets
                tweets = tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=query,
                    lang="en",
                    tweet_mode="extended",
                    result_type="mixed",  # mixed, recent, or popular
                    count=100  # Maximum allowed by Twitter API
                ).items(max_posts)
                
                for tweet in tweets:
                    # Check if tweet is within the date range
                    created_at = tweet.created_at
                    if isinstance(created_at, str):
                        created_at = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                    
                    if created_at >= start_date and created_at <= end_date:
                        # Create post object
                        post = {
                            "symbol": symbol,
                            "platform": "twitter",
                            "post_id": tweet.id_str,
                            "user_id": tweet.user.id_str,
                            "username": tweet.user.screen_name,
                            "content": tweet.full_text,
                            "created_at": created_at.isoformat(),
                            "likes": tweet.favorite_count,
                            "retweets": tweet.retweet_count,
                            "followers": tweet.user.followers_count,
                            "verified": tweet.user.verified,
                            "fetched_at": datetime.now().isoformat()
                        }
                        posts.append(post)
                
                # Rate limiting - Twitter has rate limits
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching from Twitter for symbol {symbol}: {str(e)}")
        
        return posts
    
    def _fetch_reddit_data(self, symbols: List[str], days: int) -> List[Dict[str, Any]]:
        """Fetch Reddit posts for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch posts for.
            days: Number of days to look back.
            
        Returns:
            List of Reddit posts as dictionaries.
        """
        posts = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Maximum posts per symbol from config
        max_posts = self.config.get('social_media', {}).get('max_posts_per_symbol', 100)
        
        # Subreddits to search
        subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket", "options"]
        
        for symbol in symbols:
            try:
                for subreddit_name in subreddits:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    # Search for posts containing the symbol
                    search_query = f"{symbol}"
                    submissions = subreddit.search(search_query, limit=max_posts // len(subreddits))
                    
                    for submission in submissions:
                        # Check if submission is within the date range
                        created_at = datetime.fromtimestamp(submission.created_utc)
                        if created_at >= start_date and created_at <= end_date:
                            # Create post object
                            post = {
                                "symbol": symbol,
                                "platform": "reddit",
                                "post_id": submission.id,
                                "user_id": submission.author.name if submission.author else "[deleted]",
                                "username": submission.author.name if submission.author else "[deleted]",
                                "subreddit": subreddit_name,
                                "title": submission.title,
                                "content": submission.selftext,
                                "created_at": created_at.isoformat(),
                                "upvotes": submission.score,
                                "comments": submission.num_comments,
                                "url": submission.url,
                                "fetched_at": datetime.now().isoformat()
                            }
                            posts.append(post)
                    
                    # Rate limiting - Reddit has rate limits
                    time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching from Reddit for symbol {symbol}: {str(e)}")
        
        return posts
    
    def _fetch_stocktwits_data(self, symbols: List[str], days: int) -> List[Dict[str, Any]]:
        """Fetch Stocktwits messages for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch messages for.
            days: Number of days to look back.
            
        Returns:
            List of Stocktwits messages as dictionaries.
        """
        posts = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in symbols:
            try:
                # Stocktwits API endpoint for symbol streams
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
                
                # Add API key if available
                if self.stocktwits_api_key:
                    url += f"?access_token={self.stocktwits_api_key}"
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    messages = data.get('messages', [])
                    
                    for message in messages:
                        # Parse created_at
                        created_at = datetime.strptime(message.get('created_at'), "%Y-%m-%dT%H:%M:%SZ")
                        
                        # Check if message is within the date range
                        if created_at >= start_date and created_at <= end_date:
                            # Get user data
                            user = message.get('user', {})
                            
                            # Get sentiment if available
                            entities = message.get('entities', {})
                            sentiment = entities.get('sentiment', {})
                            sentiment_value = sentiment.get('basic', '')
                            
                            # Map Stocktwits sentiment to numeric value
                            sentiment_score = None
                            if sentiment_value == 'Bullish':
                                sentiment_score = 1.0
                            elif sentiment_value == 'Bearish':
                                sentiment_score = 0.0
                            
                            # Create post object
                            post = {
                                "symbol": symbol,
                                "platform": "stocktwits",
                                "post_id": str(message.get('id')),
                                "user_id": str(user.get('id')),
                                "username": user.get('username'),
                                "content": message.get('body', ''),
                                "created_at": created_at.isoformat(),
                                "likes": message.get('likes', {}).get('total', 0),
                                "followers": user.get('followers', 0),
                                "following": user.get('following', 0),
                                "user_posts_count": user.get('ideas', 0),
                                "user_since": user.get('join_date', ''),
                                "fetched_at": datetime.now().isoformat()
                            }
                            
                            # Add sentiment data if available
                            if sentiment_score is not None:
                                post["stocktwits_sentiment"] = sentiment_value
                                post["stocktwits_sentiment_score"] = sentiment_score
                            
                            posts.append(post)
                else:
                    logger.warning(f"Stocktwits API returned status code {response.status_code} for symbol {symbol}")
                
                # Rate limiting - Stocktwits has rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching from Stocktwits for symbol {symbol}: {str(e)}")
        
        return posts
    
    def analyze_social_sentiment(self, social_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for the provided social media posts.
        
        Args:
            social_data: List of social media posts as dictionaries.
            
        Returns:
            List of sentiment analysis results.
        """
        logger.info(f"Analyzing sentiment for {len(social_data)} social media posts")
        
        sentiment_results = []
        
        for post in social_data:
            try:
                # Check if Stocktwits already provided sentiment
                if 'stocktwits_sentiment_score' in post:
                    sentiment_score = post['stocktwits_sentiment_score']
                    sentiment_label = 'positive' if sentiment_score > 0.5 else 'negative'
                    confidence = 0.8  # Stocktwits sentiment is user-declared, so relatively high confidence
                else:
                    # Use VADER for sentiment analysis
                    text = post.get('content', '')
                    if post.get('platform') == 'reddit' and post.get('title'):
                        text = f"{post.get('title')} {text}"
                    
                    # Get sentiment scores from VADER
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                    
                    # Map compound score to a 0-1 scale (VADER compound is -1 to 1)
                    sentiment_score = (sentiment_scores['compound'] + 1) / 2
                    
                    # Determine sentiment label
                    if sentiment_scores['compound'] >= 0.05:
                        sentiment_label = 'positive'
                    elif sentiment_scores['compound'] <= -0.05:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    # Confidence based on the magnitude of the compound score
                    confidence = abs(sentiment_scores['compound'])
                
                # Calculate engagement score
                engagement = 0.0
                if post.get('platform') == 'twitter':
                    # Twitter engagement: likes + retweets * 2 (retweets have more impact)
                    engagement = post.get('likes', 0) + (post.get('retweets', 0) * 2)
                    # Adjust for user influence (followers)
                    if post.get('followers', 0) > 0:
                        engagement = engagement * (1 + min(post.get('followers', 0) / 10000, 5))
                    # Verified users get a boost
                    if post.get('verified', False):
                        engagement *= 1.5
                elif post.get('platform') == 'reddit':
                    # Reddit engagement: upvotes + comments * 3 (comments show more engagement)
                    engagement = post.get('upvotes', 0) + (post.get('comments', 0) * 3)
                elif post.get('platform') == 'stocktwits':
                    # Stocktwits engagement: likes + followers/100
                    engagement = post.get('likes', 0) + (post.get('followers', 0) / 100)
                
                # Create sentiment result
                sentiment = {
                    "post_id": post.get("post_id", ""),
                    "symbol": post.get("symbol", ""),
                    "platform": post.get("platform", ""),
                    "user_id": post.get("user_id", ""),
                    "username": post.get("username", ""),
                    "created_at": post.get("created_at", ""),
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "confidence": confidence,
                    "engagement": engagement,
                    "analyzed_at": datetime.now().isoformat()
                }
                sentiment_results.append(sentiment)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for post: {str(e)}")
        
        # Store sentiment results in MongoDB
        if sentiment_results:
            self._store_social_metrics(sentiment_results)
        
        return sentiment_results
    
    def get_social_metrics(self, symbols: List[str], platforms: List[str] = None, days: int = 7) -> pd.DataFrame:
        """Get social media metrics for the specified symbols over the last N days.
        
        Args:
            symbols: List of stock symbols to get metrics for.
            platforms: List of social media platforms to get metrics for. Defaults to all platforms.
            days: Number of days to look back.
            
        Returns:
            DataFrame with social media metrics.
        """
        if platforms is None:
            platforms = ['twitter', 'reddit', 'stocktwits']
        
        logger.info(f"Getting social metrics for symbols: {symbols} from platforms: {platforms} over last {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query MongoDB for social metrics
        query = {
            "symbol": {"$in": symbols},
            "platform": {"$in": platforms},
            "created_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }
        
        # Find social metrics in MongoDB
        metrics_data = list(find("social_media_metrics", query))
        
        # Convert to DataFrame
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            return df
        else:
            logger.warning(f"No social metrics found for symbols: {symbols} from platforms: {platforms}")
            return pd.DataFrame()
    
    def calculate_social_buzz(self, symbols: List[str], platforms: List[str] = None, days: int = 7) -> Dict[str, Dict[str, float]]:
        """Calculate social media buzz scores for the specified symbols.
        
        Args:
            symbols: List of stock symbols to calculate buzz for.
            platforms: List of social media platforms to include. Defaults to all platforms.
            days: Number of days to look back.
            
        Returns:
            Dictionary mapping symbols to buzz scores.
        """
        if platforms is None:
            platforms = ['twitter', 'reddit', 'stocktwits']
        
        logger.info(f"Calculating social buzz for symbols: {symbols} from platforms: {platforms} over last {days} days")
        
        # Get social metrics
        metrics_df = self.get_social_metrics(symbols, platforms, days)
        
        buzz_scores = {}
        
        for symbol in symbols:
            # Filter metrics for this symbol
            symbol_df = metrics_df[metrics_df['symbol'] == symbol] if not metrics_df.empty else pd.DataFrame()
            
            if not symbol_df.empty:
                # Calculate volume (number of posts)
                volume = len(symbol_df)
                
                # Calculate weighted average sentiment score
                weighted_sentiment = (symbol_df['sentiment_score'] * symbol_df['confidence'] * symbol_df['engagement']).sum() / \
                                     (symbol_df['confidence'] * symbol_df['engagement']).sum()
                
                # Calculate total engagement
                total_engagement = symbol_df['engagement'].sum()
                
                # Count posts by sentiment label
                sentiment_counts = symbol_df['sentiment_label'].value_counts().to_dict()
                
                # Calculate percentages
                sentiment_percentages = {
                    label: count / volume 
                    for label, count in sentiment_counts.items()
                }
                
                # Ensure all labels are present
                for label in ['positive', 'neutral', 'negative']:
                    if label not in sentiment_percentages:
                        sentiment_percentages[label] = 0.0
                
                # Calculate buzz score (combination of volume, engagement, and sentiment)
                # Higher volume, engagement, and positive sentiment = higher buzz
                buzz_score = (volume * 0.3) + (total_engagement * 0.4) + (weighted_sentiment * 100 * 0.3)
                
                # Normalize buzz score (0-100 scale)
                # This is a simple normalization, in a real system you might want to normalize across all symbols
                normalized_buzz = min(buzz_score / 1000, 100)
                
                # Create result
                buzz_scores[symbol] = {
                    'buzz_score': normalized_buzz,
                    'volume': volume,
                    'engagement': total_engagement,
                    'sentiment_score': weighted_sentiment,
                    'positive_pct': sentiment_percentages.get('positive', 0.0),
                    'neutral_pct': sentiment_percentages.get('neutral', 0.0),
                    'negative_pct': sentiment_percentages.get('negative', 0.0),
                    'platforms': platforms
                }
                
                # Add platform-specific metrics
                for platform in platforms:
                    platform_df = symbol_df[symbol_df['platform'] == platform]
                    if not platform_df.empty:
                        platform_volume = len(platform_df)
                        platform_engagement = platform_df['engagement'].sum()
                        platform_sentiment = (platform_df['sentiment_score'] * platform_df['confidence']).sum() / platform_df['confidence'].sum()
                        
                        buzz_scores[symbol][f'{platform}_volume'] = platform_volume
                        buzz_scores[symbol][f'{platform}_engagement'] = platform_engagement
                        buzz_scores[symbol][f'{platform}_sentiment'] = platform_sentiment
                    else:
                        buzz_scores[symbol][f'{platform}_volume'] = 0
                        buzz_scores[symbol][f'{platform}_engagement'] = 0
                        buzz_scores[symbol][f'{platform}_sentiment'] = 0.5  # Neutral
            else:
                # No data for this symbol
                buzz_scores[symbol] = {
                    'buzz_score': 0.0,
                    'volume': 0,
                    'engagement': 0,
                    'sentiment_score': 0.5,  # Neutral
                    'positive_pct': 0.0,
                    'neutral_pct': 0.0,
                    'negative_pct': 0.0,
                    'platforms': platforms
                }
                
                # Add empty platform-specific metrics
                for platform in platforms:
                    buzz_scores[symbol][f'{platform}_volume'] = 0
                    buzz_scores[symbol][f'{platform}_engagement'] = 0
                    buzz_scores[symbol][f'{platform}_sentiment'] = 0.5  # Neutral
        
        return buzz_scores
    
    def _store_social_posts(self, posts: List[Dict[str, Any]]) -> None:
        """Store social media posts in MongoDB.
        
        Args:
            posts: List of social media posts as dictionaries.
        """
        try:
            # Insert posts into MongoDB
            insert_many("social_media_posts", posts)
            logger.info(f"Stored {len(posts)} social media posts in MongoDB")
        except Exception as e:
            logger.error(f"Error storing social media posts in MongoDB: {str(e)}")
    
    def _store_social_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Store social media metrics in MongoDB.
        
        Args:
            metrics: List of social media metrics as dictionaries.
        """
        try:
            # Insert metrics into MongoDB
            insert_many("social_media_metrics", metrics)
            logger.info(f"Stored {len(metrics)} social media metrics in MongoDB")
        except Exception as e:
            logger.error(f"Error storing social media metrics in MongoDB: {str(e)}")