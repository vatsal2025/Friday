"""Economic data provider module for the Friday AI Trading System.

This module provides functionality for fetching and analyzing economic data.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger
from src.infrastructure.database.mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one, update_one
)

# Create logger
logger = get_logger(__name__)

class EconomicDataProvider:
    """Class for providing economic data for financial analysis.
    
    This class provides methods for fetching economic indicators, central bank data,
    and other macroeconomic data, and storing the results in MongoDB.
    
    Attributes:
        config_manager: Configuration manager.
        config: Configuration dictionary.
        economic_indicators_collection: MongoDB collection for storing economic indicators.
        central_bank_data_collection: MongoDB collection for storing central bank data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the EconomicDataProvider with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('alternative_data')
        
        # Initialize MongoDB collections
        self.economic_indicators_collection = get_collection('economic_indicators')
        self.central_bank_data_collection = get_collection('central_bank_data')
        
        logger.info("Economic Data Provider initialized")
    
    def fetch_economic_indicators(self, indicators: List[str], 
                                countries: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch economic indicators for the specified countries.
        
        Args:
            indicators: List of economic indicators to fetch (e.g., 'gdp', 'inflation', 'unemployment').
            countries: List of country codes to fetch data for. If None, fetches for default countries.
            
        Returns:
            Dictionary mapping indicators to dictionaries of country data.
        """
        if countries is None:
            countries = self.config.get('default_countries', ['US', 'EU', 'JP', 'CN', 'IN'])
        
        logger.info(f"Fetching economic indicators: {indicators} for countries: {countries}")
        
        # Implementation would connect to economic data APIs here
        # For now, we'll just return placeholder data
        
        economic_data = {}
        for indicator in indicators:
            # Example placeholder data
            indicator_data = {
                country: {
                    "indicator": indicator,
                    "country": country,
                    "value": 5.0,  # Placeholder value
                    "previous_value": 4.8,  # Placeholder previous value
                    "change": 0.2,  # Placeholder change
                    "change_percent": 4.17,  # Placeholder percent change
                    "date": datetime.now().isoformat(),
                    "source": "Example Economic Data Source"
                } for country in countries
            }
            economic_data[indicator] = indicator_data
            
            # Store economic indicator data in MongoDB
            self._store_economic_indicators(indicator, indicator_data)
        
        return economic_data
    
    def fetch_central_bank_data(self, central_banks: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch central bank data for the specified central banks.
        
        Args:
            central_banks: List of central bank codes to fetch data for. If None, fetches for default central banks.
            
        Returns:
            Dictionary mapping central banks to dictionaries of data.
        """
        if central_banks is None:
            central_banks = self.config.get('default_central_banks', ['FED', 'ECB', 'BOJ', 'PBOC', 'RBI'])
        
        logger.info(f"Fetching central bank data for: {central_banks}")
        
        # Implementation would connect to central bank data APIs here
        # For now, we'll just return placeholder data
        
        central_bank_data = {}
        for bank in central_banks:
            # Example placeholder data
            bank_data = {
                "central_bank": bank,
                "interest_rate": 2.5,  # Placeholder interest rate
                "previous_interest_rate": 2.25,  # Placeholder previous interest rate
                "change": 0.25,  # Placeholder change
                "next_meeting_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "last_meeting_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "statement": f"Example statement from {bank}",
                "date": datetime.now().isoformat(),
                "source": "Example Central Bank Data Source"
            }
            central_bank_data[bank] = bank_data
            
            # Store central bank data in MongoDB
            self._store_central_bank_data(bank, bank_data)
        
        return central_bank_data
    
    def get_economic_indicators(self, indicators: List[str], countries: Optional[List[str]] = None, 
                              days: int = 30) -> pd.DataFrame:
        """Get economic indicators for the specified countries over the last N days.
        
        Args:
            indicators: List of economic indicators to get.
            countries: List of country codes to get data for. If None, gets for all countries.
            days: Number of days to look back.
            
        Returns:
            DataFrame with economic indicator data.
        """
        logger.info(f"Getting economic indicators: {indicators} for countries: {countries} over last {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Prepare query
        query = {
            "indicator": {"$in": indicators},
            "date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }
        
        if countries:
            query["country"] = {"$in": countries}
        
        # Find economic indicators in MongoDB
        indicators_data = list(find("economic_indicators", query))
        
        # Convert to DataFrame
        if indicators_data:
            df = pd.DataFrame(indicators_data)
            return df
        else:
            logger.warning(f"No economic indicators found for: {indicators}")
            return pd.DataFrame()
    
    def get_central_bank_data(self, central_banks: Optional[List[str]] = None, 
                           days: int = 90) -> pd.DataFrame:
        """Get central bank data for the specified central banks over the last N days.
        
        Args:
            central_banks: List of central bank codes to get data for. If None, gets for all central banks.
            days: Number of days to look back.
            
        Returns:
            DataFrame with central bank data.
        """
        logger.info(f"Getting central bank data for: {central_banks} over last {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Prepare query
        query = {
            "date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }
        
        if central_banks:
            query["central_bank"] = {"$in": central_banks}
        
        # Find central bank data in MongoDB
        bank_data = list(find("central_bank_data", query))
        
        # Convert to DataFrame
        if bank_data:
            df = pd.DataFrame(bank_data)
            return df
        else:
            logger.warning(f"No central bank data found for: {central_banks}")
            return pd.DataFrame()
    
    def calculate_economic_impact(self, symbols: List[str], indicators: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate the potential impact of economic indicators on the specified symbols.
        
        Args:
            symbols: List of stock symbols to calculate impact for.
            indicators: List of economic indicators to consider.
            
        Returns:
            Dictionary mapping symbols to dictionaries of indicator impacts.
        """
        logger.info(f"Calculating economic impact for symbols: {symbols} using indicators: {indicators}")
        
        # Implementation would use statistical models to calculate impact
        # For now, we'll just return placeholder impact scores
        
        impact_scores = {}
        for symbol in symbols:
            # Example placeholder impact scores
            symbol_impacts = {
                indicator: 0.5  # Placeholder impact score (0 to 1)
                for indicator in indicators
            }
            impact_scores[symbol] = symbol_impacts
        
        return impact_scores
    
    def _store_economic_indicators(self, indicator: str, data: Dict[str, Dict[str, Any]]) -> None:
        """Store economic indicator data in MongoDB.
        
        Args:
            indicator: The economic indicator name.
            data: Dictionary mapping countries to indicator data.
        """
        try:
            # Prepare documents for insertion
            documents = list(data.values())
            
            # Insert or update economic indicators in MongoDB
            for doc in documents:
                # Create query to find existing document
                query = {
                    "indicator": doc["indicator"],
                    "country": doc["country"],
                    "date": doc["date"]
                }
                
                # Update or insert document
                update_one("economic_indicators", query, {"$set": doc}, upsert=True)
            
            logger.info(f"Stored economic indicator data for {indicator} in MongoDB")
        except Exception as e:
            logger.error(f"Error storing economic indicator data in MongoDB: {str(e)}")
    
    def _store_central_bank_data(self, bank: str, data: Dict[str, Any]) -> None:
        """Store central bank data in MongoDB.
        
        Args:
            bank: The central bank code.
            data: Dictionary of central bank data.
        """
        try:
            # Create query to find existing document
            query = {
                "central_bank": bank,
                "date": data["date"]
            }
            
            # Update or insert document
            update_one("central_bank_data", query, {"$set": data}, upsert=True)
            
            logger.info(f"Stored central bank data for {bank} in MongoDB")
        except Exception as e:
            logger.error(f"Error storing central bank data in MongoDB: {str(e)}")