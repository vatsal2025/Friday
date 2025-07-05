"""Financial Data Service for the Friday AI Trading System.

This module provides a service for accessing financial data, including market data,
company information, financial statements, and news articles.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import json
import logging
import pandas as pd

from src.data.acquisition.adapters.financial_data_adapter import FinancialDataAdapter, FinancialDataType
from src.data.acquisition.data_fetcher import DataTimeframe
from src.infrastructure.error.data_errors import DataConnectionError, DataValidationError
from src.infrastructure.service.service_base import ServiceBase
from src.infrastructure.config.config_manager import ConfigManager

# Set up logging
logger = logging.getLogger(__name__)


class FinancialDataService(ServiceBase):
    """Service for accessing financial data.

    This service provides access to financial data, including market data,
    company information, financial statements, and news articles.
    
    Attributes:
        adapter: The financial data adapter used to fetch data.
        config: The configuration manager.
    """

    def __init__(self, adapter: Optional[FinancialDataAdapter] = None):
        """Initialize the financial data service.

        Args:
            adapter: The financial data adapter to use. If None, it will be created based on config.
        """
        super().__init__()
        self.config = ConfigManager()
        self.adapter = adapter
        
        # Register endpoints
        self.register_endpoint("authenticate", self.authenticate)
        self.register_endpoint("get_companies", self.get_companies)
        self.register_endpoint("get_financials", self.get_financials)
        self.register_endpoint("get_news", self.get_news)
        self.register_endpoint("get_market_data", self.get_market_data)

    def start(self) -> bool:
        """Start the financial data service.

        Returns:
            bool: True if the service started successfully, False otherwise.
        """
        try:
            if not self.adapter:
                # Create adapter based on config
                adapter_type = self.config.get("financial_data.adapter_type", "mock")
                
                if adapter_type == "mock":
                    from src.integration.mock.mock_service import MockFinancialDataService
                    self.adapter = MockFinancialDataService()
                else:
                    # Import the appropriate adapter class dynamically
                    adapter_module = __import__(f"src.data.acquisition.adapters.{adapter_type}_adapter", fromlist=[f"{adapter_type.capitalize()}Adapter"])
                    adapter_class = getattr(adapter_module, f"{adapter_type.capitalize()}Adapter")
                    self.adapter = adapter_class()
            
            # Connect to the data source
            if hasattr(self.adapter, "connect"):
                self.adapter.connect()
            
            logger.info(f"Financial data service started with {type(self.adapter).__name__}")
            return True
        except Exception as e:
            logger.error(f"Failed to start financial data service: {str(e)}")
            return False

    def stop(self) -> bool:
        """Stop the financial data service.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        try:
            if self.adapter and hasattr(self.adapter, "disconnect"):
                self.adapter.disconnect()
            
            logger.info("Financial data service stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop financial data service: {str(e)}")
            return False

    def authenticate(self, api_key: str) -> Dict[str, Any]:
        """Authenticate with the financial data service.

        Args:
            api_key: The API key to authenticate with.

        Returns:
            Dict[str, Any]: Authentication response with token.

        Raises:
            Exception: If authentication fails.
        """
        try:
            # Check if the adapter has an authenticate method
            if hasattr(self.adapter, "authenticate"):
                return self.adapter.authenticate(api_key)
            
            # Otherwise, use a simple authentication mechanism
            if api_key == self.config.get("financial_data.api_key"):
                import uuid
                import time
                
                # Generate a token that expires in 24 hours
                token = str(uuid.uuid4())
                expiry = int(time.time()) + 86400  # 24 hours
                
                return {
                    "token": token,
                    "expires_at": expiry,
                    "status": "success"
                }
            else:
                raise Exception("Invalid API key")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def get_companies(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get company information for the specified symbols.

        Args:
            symbols: List of symbols to get company information for. If None, get all available companies.

        Returns:
            List[Dict[str, Any]]: List of company information.

        Raises:
            Exception: If fetching company information fails.
        """
        try:
            # Check if the adapter has a get_company_info method
            if hasattr(self.adapter, "get_company_info"):
                if symbols:
                    # Get company info for each symbol
                    companies = []
                    for symbol in symbols:
                        try:
                            company = self.adapter.get_company_info(symbol)
                            companies.append(company)
                        except Exception as e:
                            logger.warning(f"Failed to get company info for {symbol}: {str(e)}")
                    
                    return companies
                else:
                    # Get all available symbols and then get company info for each
                    all_symbols = self.adapter.get_symbols()
                    companies = []
                    
                    for symbol in all_symbols[:100]:  # Limit to 100 companies to avoid overloading
                        try:
                            company = self.adapter.get_company_info(symbol)
                            companies.append(company)
                        except Exception as e:
                            logger.warning(f"Failed to get company info for {symbol}: {str(e)}")
                    
                    return companies
            elif hasattr(self.adapter, "get_companies"):
                # If the adapter has a get_companies method, use it directly
                return self.adapter.get_companies(symbols)
            else:
                raise Exception("Adapter does not support getting company information")
        except Exception as e:
            logger.error(f"Failed to get companies: {str(e)}")
            raise

    def get_financials(self, symbol: str, period_type: str = "quarterly") -> Dict[str, Any]:
        """Get financial statements for the specified symbol.

        Args:
            symbol: The symbol to get financial statements for.
            period_type: The period type (quarterly or annual).

        Returns:
            Dict[str, Any]: Financial statements.

        Raises:
            Exception: If fetching financial statements fails.
        """
        try:
            # Check if the adapter has a get_financial_statements method
            if hasattr(self.adapter, "get_financial_statements"):
                return self.adapter.get_financial_statements(symbol, period_type)
            elif hasattr(self.adapter, "get_financials"):
                # If the adapter has a get_financials method, use it directly
                return self.adapter.get_financials(symbol, period_type)
            else:
                raise Exception("Adapter does not support getting financial statements")
        except Exception as e:
            logger.error(f"Failed to get financials for {symbol}: {str(e)}")
            raise

    def get_news(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news articles for the specified symbol or general market news.

        Args:
            symbol: The symbol to get news for. If None, get general market news.
            limit: The maximum number of news articles to return.

        Returns:
            List[Dict[str, Any]]: List of news articles.

        Raises:
            Exception: If fetching news fails.
        """
        try:
            # Check if the adapter has a get_news method
            if hasattr(self.adapter, "get_news"):
                return self.adapter.get_news(symbol, limit)
            else:
                raise Exception("Adapter does not support getting news")
        except Exception as e:
            logger.error(f"Failed to get news: {str(e)}")
            raise

    def get_market_data(self, 
                       symbol: str, 
                       timeframe: Union[str, DataTimeframe], 
                       start_date: Optional[Union[str, datetime]] = None, 
                       end_date: Optional[Union[str, datetime]] = None, 
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """Get market data for the specified symbol.

        Args:
            symbol: The symbol to get market data for.
            timeframe: The timeframe of the data.
            start_date: The start date for the data. Defaults to None.
            end_date: The end date for the data. Defaults to None.
            limit: The maximum number of data points to fetch. Defaults to None.

        Returns:
            Dict[str, Any]: Market data.

        Raises:
            Exception: If fetching market data fails.
        """
        try:
            # Convert timeframe to DataTimeframe if it's a string
            if isinstance(timeframe, str):
                try:
                    timeframe = DataTimeframe[timeframe.upper()]
                except KeyError:
                    raise DataValidationError(f"Invalid timeframe: {timeframe}")
            
            # Convert dates to datetime if they're strings
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Check if the adapter has a fetch_data method
            if hasattr(self.adapter, "fetch_data"):
                df = self.adapter.fetch_data(symbol, timeframe, start_date, end_date, limit)
                
                # Convert DataFrame to dict for JSON serialization
                data = df.to_dict(orient="records")
                
                return {
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "data": data
                }
            else:
                raise Exception("Adapter does not support fetching market data")
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            raise