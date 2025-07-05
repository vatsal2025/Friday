"""Alternative data normalizer module for the Friday AI Trading System.

This module provides functionality for normalizing and standardizing alternative data
from different sources to make it compatible with the trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one, update_one
)

# Create logger
logger = get_logger(__name__)

class AlternativeDataNormalizer:
    """Class for normalizing alternative data from different sources.
    
    This class provides methods for standardizing, scaling, and transforming
    alternative data to make it compatible with the trading system.
    
    Attributes:
        config_manager: Configuration manager.
        config: Configuration dictionary.
        normalized_data_collection: MongoDB collection for storing normalized data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AlternativeDataNormalizer with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('alternative_data')
        
        # Initialize MongoDB collections
        self.normalized_data_collection = get_collection('normalized_alternative_data')
        
        logger.info("Alternative Data Normalizer initialized")
    
    def normalize_news_sentiment(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize news sentiment data.
        
        Args:
            sentiment_data: DataFrame with news sentiment data.
            
        Returns:
            Normalized DataFrame.
        """
        logger.info(f"Normalizing news sentiment data with {len(sentiment_data)} records")
        
        if sentiment_data.empty:
            logger.warning("Empty sentiment data provided for normalization")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        normalized_df = sentiment_data.copy()
        
        # Ensure required columns exist
        required_columns = ['symbol', 'overall_sentiment', 'analyzed_at']
        for col in required_columns:
            if col not in normalized_df.columns:
                logger.error(f"Required column '{col}' missing from sentiment data")
                return pd.DataFrame()
        
        try:
            # Convert sentiment to standard scale (-1 to 1)
            if 'overall_sentiment' in normalized_df.columns:
                # Assuming original scale is 0 to 1
                normalized_df['normalized_sentiment'] = normalized_df['overall_sentiment'] * 2 - 1
            
            # Add data source identifier
            normalized_df['data_source'] = 'news_sentiment'
            
            # Add normalized timestamp
            normalized_df['normalized_timestamp'] = pd.to_datetime(normalized_df['analyzed_at'])
            
            # Create standardized feature name
            normalized_df['feature_name'] = 'news_sentiment'
            
            # Create standardized feature value
            normalized_df['feature_value'] = normalized_df['normalized_sentiment']
            
            # Store normalized data in MongoDB
            self._store_normalized_data(normalized_df, 'news_sentiment')
            
            return normalized_df
        
        except Exception as e:
            logger.error(f"Error normalizing news sentiment data: {str(e)}")
            return pd.DataFrame()
    
    def normalize_social_metrics(self, social_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize social media metrics data.
        
        Args:
            social_data: DataFrame with social media metrics data.
            
        Returns:
            Normalized DataFrame.
        """
        logger.info(f"Normalizing social media metrics data with {len(social_data)} records")
        
        if social_data.empty:
            logger.warning("Empty social media data provided for normalization")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        normalized_df = social_data.copy()
        
        # Ensure required columns exist
        required_columns = ['symbol', 'sentiment_score', 'analyzed_at']
        for col in required_columns:
            if col not in normalized_df.columns:
                logger.error(f"Required column '{col}' missing from social media data")
                return pd.DataFrame()
        
        try:
            # Convert sentiment to standard scale (-1 to 1)
            if 'sentiment_score' in normalized_df.columns:
                # Assuming original scale is 0 to 1
                normalized_df['normalized_sentiment'] = normalized_df['sentiment_score'] * 2 - 1
            
            # Add data source identifier
            normalized_df['data_source'] = 'social_media'
            
            # Add normalized timestamp
            normalized_df['normalized_timestamp'] = pd.to_datetime(normalized_df['analyzed_at'])
            
            # Create standardized feature name
            normalized_df['feature_name'] = 'social_sentiment'
            
            # Create standardized feature value
            normalized_df['feature_value'] = normalized_df['normalized_sentiment']
            
            # Store normalized data in MongoDB
            self._store_normalized_data(normalized_df, 'social_media')
            
            return normalized_df
        
        except Exception as e:
            logger.error(f"Error normalizing social media data: {str(e)}")
            return pd.DataFrame()
    
    def normalize_economic_data(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize economic data.
        
        Args:
            economic_data: DataFrame with economic data.
            
        Returns:
            Normalized DataFrame.
        """
        logger.info(f"Normalizing economic data with {len(economic_data)} records")
        
        if economic_data.empty:
            logger.warning("Empty economic data provided for normalization")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        normalized_df = economic_data.copy()
        
        # Ensure required columns exist
        required_columns = ['indicator', 'country', 'value', 'date']
        for col in required_columns:
            if col not in normalized_df.columns:
                logger.error(f"Required column '{col}' missing from economic data")
                return pd.DataFrame()
        
        try:
            # Add data source identifier
            normalized_df['data_source'] = 'economic_indicators'
            
            # Add normalized timestamp
            normalized_df['normalized_timestamp'] = pd.to_datetime(normalized_df['date'])
            
            # Create standardized feature name (indicator_country)
            normalized_df['feature_name'] = normalized_df['indicator'] + '_' + normalized_df['country']
            
            # Create standardized feature value
            normalized_df['feature_value'] = normalized_df['value']
            
            # Z-score normalization for numeric values
            # Group by indicator to normalize each indicator separately
            for indicator in normalized_df['indicator'].unique():
                indicator_mask = normalized_df['indicator'] == indicator
                values = normalized_df.loc[indicator_mask, 'value']
                
                if len(values) > 1:  # Need at least 2 values for std to be non-zero
                    mean = values.mean()
                    std = values.std()
                    if std > 0:  # Avoid division by zero
                        normalized_df.loc[indicator_mask, 'normalized_value'] = (values - mean) / std
                    else:
                        normalized_df.loc[indicator_mask, 'normalized_value'] = 0
                else:
                    normalized_df.loc[indicator_mask, 'normalized_value'] = 0
            
            # Store normalized data in MongoDB
            self._store_normalized_data(normalized_df, 'economic_indicators')
            
            return normalized_df
        
        except Exception as e:
            logger.error(f"Error normalizing economic data: {str(e)}")
            return pd.DataFrame()
    
    def combine_alternative_data(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Combine normalized alternative data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to get data for.
            days: Number of days to look back.
            
        Returns:
            DataFrame with combined alternative data.
        """
        logger.info(f"Combining alternative data for symbols: {symbols} over last {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query MongoDB for normalized alternative data
        query = {
            "symbol": {"$in": symbols},
            "normalized_timestamp": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }
        
        # Find normalized data in MongoDB
        normalized_data = list(find("normalized_alternative_data", query))
        
        # Convert to DataFrame
        if normalized_data:
            df = pd.DataFrame(normalized_data)
            
            # Pivot the data to create a wide format DataFrame
            # Each symbol-date combination will have columns for different features
            pivot_df = df.pivot_table(
                index=['symbol', 'normalized_timestamp'],
                columns='feature_name',
                values='feature_value',
                aggfunc='mean'  # Use mean if there are multiple values for the same feature
            ).reset_index()
            
            return pivot_df
        else:
            logger.warning(f"No normalized alternative data found for symbols: {symbols}")
            return pd.DataFrame()
    
    def get_feature_matrix(self, symbols: List[str], features: Optional[List[str]] = None, 
                         days: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """Get a feature matrix for machine learning models.
        
        Args:
            symbols: List of stock symbols to get features for.
            features: List of feature names to include. If None, includes all available features.
            days: Number of days to look back.
            
        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        logger.info(f"Creating feature matrix for symbols: {symbols} with features: {features} over last {days} days")
        
        # Get combined alternative data
        combined_df = self.combine_alternative_data(symbols, days)
        
        if combined_df.empty:
            logger.warning("No data available for feature matrix")
            return pd.DataFrame(), []
        
        # Filter features if specified
        if features:
            available_features = [col for col in combined_df.columns if col not in ['symbol', 'normalized_timestamp']]
            selected_features = [feat for feat in features if feat in available_features]
            
            if not selected_features:
                logger.warning(f"None of the requested features {features} are available")
                return pd.DataFrame(), []
            
            # Select only the specified features
            feature_cols = ['symbol', 'normalized_timestamp'] + selected_features
            feature_matrix = combined_df[feature_cols]
            feature_names = selected_features
        else:
            # Use all available features
            feature_names = [col for col in combined_df.columns if col not in ['symbol', 'normalized_timestamp']]
            feature_matrix = combined_df
        
        return feature_matrix, feature_names
    
    def _store_normalized_data(self, data: pd.DataFrame, data_source: str) -> None:
        """Store normalized alternative data in MongoDB.
        
        Args:
            data: DataFrame with normalized data.
            data_source: Source of the data (e.g., 'news_sentiment', 'social_media', 'economic_indicators').
        """
        try:
            # Convert DataFrame to list of dictionaries
            records = data.to_dict('records')
            
            # Add metadata
            for record in records:
                record['normalized_at'] = datetime.now().isoformat()
                record['data_source'] = data_source
            
            # Insert normalized data into MongoDB
            insert_many("normalized_alternative_data", records)
            
            logger.info(f"Stored {len(records)} normalized {data_source} records in MongoDB")
        except Exception as e:
            logger.error(f"Error storing normalized {data_source} data in MongoDB: {str(e)}")