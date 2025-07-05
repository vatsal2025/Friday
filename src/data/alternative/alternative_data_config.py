"""Configuration for alternative data integration in the Friday AI Trading System.

This module provides default configuration settings for alternative data integration.
"""

# Default configuration for alternative data integration
ALTERNATIVE_DATA_CONFIG = {
    # News sentiment analysis configuration
    'news_sentiment': {
        'enabled': True,
        'update_frequency': 3600,  # in seconds (1 hour)
        'max_articles_per_symbol': 50,
        'sentiment_model': 'vader',  # options: vader, textblob, transformers
        'sources': [
            'bloomberg',
            'reuters',
            'cnbc',
            'wsj',
            'ft',
            'seekingalpha'
        ],
        'api_keys': {
            'newsapi': '',
            'alpha_vantage': '',
            'bloomberg': ''
        }
    },
    
    # Social media analysis configuration
    'social_media': {
        'enabled': True,
        'update_frequency': 1800,  # in seconds (30 minutes)
        'max_posts_per_symbol': 100,
        'sentiment_model': 'vader',  # options: vader, textblob, transformers
        'social_media_platforms': [
            'twitter',
            'reddit',
            'stocktwits'
        ],
        'api_keys': {
            'twitter': '',
            'reddit': '',
            'stocktwits': ''
        }
    },
    
    # Economic data configuration
    'economic_data': {
        'enabled': True,
        'update_frequency': 86400,  # in seconds (1 day)
        'economic_indicators': [
            'gdp',
            'inflation',
            'unemployment',
            'interest_rate',
            'retail_sales',
            'housing_starts',
            'pmi',
            'consumer_confidence'
        ],
        'default_countries': [
            'US',  # United States
            'EU',  # European Union
            'JP',  # Japan
            'CN',  # China
            'IN',  # India
            'UK',  # United Kingdom
            'DE',  # Germany
            'FR'   # France
        ],
        'default_central_banks': [
            'FED',  # Federal Reserve (US)
            'ECB',  # European Central Bank
            'BOJ',  # Bank of Japan
            'PBOC', # People's Bank of China
            'RBI',  # Reserve Bank of India
            'BOE',  # Bank of England
            'SNB',  # Swiss National Bank
            'BOC'   # Bank of Canada
        ],
        'api_keys': {
            'fred': '',
            'world_bank': '',
            'imf': '',
            'quandl': ''
        }
    },
    
    # Data normalization configuration
    'normalization': {
        'default_scaling': 'z-score',  # options: z-score, min-max, robust
        'outlier_detection': True,
        'outlier_method': 'iqr',  # options: iqr, z-score, isolation_forest
        'missing_value_strategy': 'interpolate',  # options: drop, mean, median, interpolate
        'feature_selection': {
            'enabled': True,
            'method': 'correlation',  # options: correlation, mutual_info, chi2
            'threshold': 0.7
        }
    },
    
    # Error handling and fallbacks
    'error_handling': {
        'max_retries': 3,
        'retry_delay': 5,  # in seconds
        'fallback_to_cache': True,
        'cache_expiry': 86400,  # in seconds (1 day)
        'alert_on_failure': True
    },
    
    # MongoDB collections
    'mongodb_collections': {
        'news_data': 'news_data',
        'news_sentiment': 'news_sentiment',
        'social_media_posts': 'social_media_posts',
        'social_media_metrics': 'social_media_metrics',
        'economic_indicators': 'economic_indicators',
        'central_bank_data': 'central_bank_data',
        'normalized_alternative_data': 'normalized_alternative_data',
        'alternative_data_updates': 'alternative_data_updates'
    }
}