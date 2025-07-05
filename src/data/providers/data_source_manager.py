"""
Data Source Manager for Friday AI Trading System
Configures and manages both real-time (Zerodha) and historical data sources.
"""

import asyncio
import json
import logging
import pandas as pd
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from ...infrastructure.logging import get_logger
from ...infrastructure.config.config_manager import ConfigurationManager
from ...infrastructure.communication import CommunicationSystem, Message, MessageHandler
from .zerodha_connector import ZerodhaKiteConnector, ZerodhaDataAdapter
from .historical_data_loader import HistoricalDataLoader

# Create logger
logger = get_logger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    ALTERNATIVE = "alternative"


class DataPriority(Enum):
    """Data source priority levels."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    type: DataSourceType
    priority: DataPriority
    enabled: bool = True
    config_params: Dict[str, Any] = field(default_factory=dict)
    authentication: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)


class DataSourceManager:
    """
    Manages all data sources for the Friday AI Trading System.
    Coordinates between real-time (Zerodha) and historical data.
    """
    
    def __init__(self, comm_system: Optional[CommunicationSystem] = None):
        """Initialize the data source manager."""
        self.comm_system = comm_system
        self.config_manager = ConfigurationManager.get_instance()
        
        # Data sources
        self.zerodha_connector: Optional[ZerodhaKiteConnector] = None
        self.zerodha_adapter: Optional[ZerodhaDataAdapter] = None
        self.historical_loader: Optional[HistoricalDataLoader] = None
        
        # Configuration
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.active_subscriptions: Dict[str, List[Callable]] = {}
        self.data_cache: Dict[str, Any] = {}
        
        # Initialize default configurations
        self._setup_default_configurations()
        self._initialize_data_sources()
        
        logger.info("Data source manager initialized")
    
    def _setup_default_configurations(self):
        """Setup default data source configurations."""
        # Zerodha real-time data source
        self.data_sources["zerodha_realtime"] = DataSourceConfig(
            name="zerodha_realtime",
            type=DataSourceType.REAL_TIME,
            priority=DataPriority.PRIMARY,
            enabled=True,
            config_params={
                "base_url": "https://api.kite.trade",
                "websocket_url": "wss://ws.kite.trade",
                "timeout": 30
            },
            rate_limits={
                "requests_per_second": 10,
                "requests_per_minute": 200,
                "requests_per_day": 5000
            },
            retry_config={
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True
            }
        )
        
        # Historical data source
        self.data_sources["historical_local"] = DataSourceConfig(
            name="historical_local",
            type=DataSourceType.HISTORICAL,
            priority=DataPriority.PRIMARY,
            enabled=True,
            config_params={
                "cache_enabled": True,
                "auto_scan": True,
                "preferred_timeframes": ["5 min data", "15 min data", "Day data"]
            }
        )
        
        # Alternative data sources (placeholders for future)
        self.data_sources["alternative_data"] = DataSourceConfig(
            name="alternative_data", 
            type=DataSourceType.ALTERNATIVE,
            priority=DataPriority.SECONDARY,
            enabled=False,
            config_params={
                "sources": ["news", "social_sentiment", "economic_indicators"]
            }
        )
        
        logger.info("Default data source configurations set up")
    
    def _initialize_data_sources(self):
        """Initialize all enabled data sources."""
        try:
            # Initialize Zerodha connector if enabled
            if self.data_sources["zerodha_realtime"].enabled:
                self.zerodha_connector = ZerodhaKiteConnector()
                self.zerodha_adapter = ZerodhaDataAdapter(self.zerodha_connector)
                logger.info("Zerodha connector initialized")
            
            # Initialize historical data loader if enabled
            if self.data_sources["historical_local"].enabled:
                self.historical_loader = HistoricalDataLoader()
                logger.info("Historical data loader initialized")
            
            # Register with communication system if available
            if self.comm_system:
                self._register_communication_handlers()
                
        except Exception as e:
            logger.error(f"Failed to initialize data sources: {e}")
    
    def _register_communication_handlers(self):
        """Register message handlers with communication system."""
        # Create data request handler
        data_handler = DataSourceMessageHandler(self)
        
        # Register with communication bus
        comm_bus = self.comm_system.get_bus()
        topics = data_handler.get_supported_topics()
        
        for topic in topics:
            comm_bus.register_handler(topic, data_handler)
        
        logger.info("Data source message handlers registered")
    
    def authenticate_zerodha(self, api_key: str = None, api_secret: str = None, 
                           request_token: str = None) -> bool:
        """Authenticate with Zerodha Kite API."""
        try:
            if not self.zerodha_connector:
                raise Exception("Zerodha connector not initialized")
            
            # Use provided credentials or load from config
            if api_key and api_secret:
                self.zerodha_connector.config.api_key = api_key
                self.zerodha_connector.config.api_secret = api_secret
            
            # Generate session if request token provided
            if request_token:
                session_data = self.zerodha_connector.generate_session(request_token)
                logger.info("Zerodha authentication successful")
                return True
            else:
                # Check if access token already exists
                if self.zerodha_connector.config.access_token:
                    logger.info("Using existing Zerodha access token")
                    return True
                else:
                    login_url = self.zerodha_connector.get_login_url()
                    logger.info(f"Zerodha login required. Visit: {login_url}")
                    return False
                    
        except Exception as e:
            logger.error(f"Zerodha authentication failed: {e}")
            return False
    
    def get_available_instruments(self, source: str = "zerodha", 
                                exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available instruments from specified source."""
        try:
            if source == "zerodha" and self.zerodha_connector:
                instruments = self.zerodha_connector.get_instruments(exchange)
                if self.zerodha_adapter:
                    return self.zerodha_adapter.normalize_instrument_data(instruments)
                return instruments
            
            elif source == "historical" and self.historical_loader:
                symbols = self.historical_loader.get_available_symbols()
                # Convert symbols to instrument format
                instruments = []
                for symbol in symbols:
                    instruments.append({
                        "symbol": symbol,
                        "name": symbol,
                        "instrument_token": symbol,
                        "exchange": "NSE",  # Assuming NSE for historical data
                        "data_source": "historical"
                    })
                return instruments
            
            else:
                logger.warning(f"Data source '{source}' not available or not initialized")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get instruments from {source}: {e}")
            return []
    
    def get_real_time_quote(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for symbols."""
        try:
            if not self.zerodha_connector:
                raise Exception("Zerodha connector not available")
            
            quotes = self.zerodha_connector.get_quote(symbols)
            
            if self.zerodha_adapter:
                return self.zerodha_adapter.normalize_quote_data(quotes)
            
            return quotes
            
        except Exception as e:
            logger.error(f"Failed to get real-time quotes: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, interval: str = "1d", 
                                start_date: datetime = None, end_date: datetime = None,
                                from_date: datetime = None, to_date: datetime = None, 
                                source: str = "auto") -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for
            interval: Data interval
            start_date: Start date (optional for local data)
            end_date: End date (optional for local data)
            from_date: Start date (legacy parameter)
            to_date: End date (legacy parameter)
            source: Data source ("zerodha", "historical", "auto")
        """
        try:
            # Handle both new and legacy parameter names
            if start_date is not None:
                from_date = start_date
            if end_date is not None:
                to_date = end_date
                
            if source == "auto":
                # Determine best source based on availability and date range
                source = self._determine_best_historical_source(symbol, from_date, to_date)
            
            if source == "zerodha" and self.zerodha_connector:
                # Get from Zerodha API
                if not from_date:
                    from_date = datetime.now() - timedelta(days=30)
                if not to_date:
                    to_date = datetime.now()
                
                # Get instrument token for symbol
                instruments = self.zerodha_connector.get_instruments("NSE")
                instrument_token = None
                
                for inst in instruments:
                    if inst.get("tradingsymbol") == symbol:
                        instrument_token = str(inst.get("instrument_token"))
                        break
                
                if instrument_token:
                    df = self.zerodha_connector.get_historical_data(
                        instrument_token, from_date, to_date, interval
                    )
                    
                    if self.zerodha_adapter and not df.empty:
                        return self.zerodha_adapter.normalize_historical_data(df, symbol)
                    return df
                else:
                    logger.warning(f"Instrument token not found for symbol: {symbol}")
                    return None
            
            elif source == "historical" and self.historical_loader:
                # Get from local historical data
                timeframe = self._map_interval_to_timeframe(interval)
                return self.historical_loader.load_symbol_data(symbol, "auto", timeframe)
            
            else:
                logger.error(f"Historical data source '{source}' not available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _determine_best_historical_source(self, symbol: str, from_date: Optional[datetime], 
                                        to_date: Optional[datetime]) -> str:
        """Determine the best historical data source."""
        # If dates are recent (within last 30 days), prefer Zerodha
        if from_date and from_date > datetime.now() - timedelta(days=30):
            if self.zerodha_connector and self.zerodha_connector.config.access_token:
                return "zerodha"
        
        # For older data or no Zerodha access, use historical files
        if self.historical_loader:
            return "historical"
        
        # Fallback to Zerodha if available
        if self.zerodha_connector:
            return "zerodha"
        
        return "historical"
    
    def _map_interval_to_timeframe(self, interval: str) -> Optional[str]:
        """Map API interval to local data timeframe."""
        mapping = {
            "minute": "3 min data",
            "3minute": "3 min data", 
            "5minute": "5 min data",
            "10minute": "10 min data",
            "15minute": "15 min data",
            "30minute": "30 min data",
            "hour": "60 min data",
            "day": "Day data"
        }
        return mapping.get(interval, "Day data")
    
    def subscribe_real_time_data(self, symbols: List[str], 
                               callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to real-time data updates."""
        try:
            if not self.zerodha_connector:
                raise Exception("Zerodha connector not available")
            
            # Register callback for each symbol
            for symbol in symbols:
                if symbol not in self.active_subscriptions:
                    self.active_subscriptions[symbol] = []
                self.active_subscriptions[symbol].append(callback)
            
            # Subscribe via Zerodha WebSocket
            self.zerodha_connector.subscribe_quotes(symbols, self._handle_real_time_update)
            
            logger.info(f"Subscribed to real-time data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to real-time data: {e}")
            return False
    
    def _handle_real_time_update(self, data: Dict[str, Any]):
        """Handle real-time data updates from Zerodha."""
        try:
            instrument_token = str(data.get("instrument_token", ""))
            
            # Find symbol for this instrument token
            symbol = None
            # This would need to be mapped from instrument token to symbol
            # For now, use instrument_token as symbol
            symbol = instrument_token
            
            if symbol in self.active_subscriptions:
                for callback in self.active_subscriptions[symbol]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in real-time callback: {e}")
            
            # Cache the update
            self.data_cache[f"realtime_{symbol}"] = {
                "data": data,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error handling real-time update: {e}")
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        status = {}
        
        for name, config in self.data_sources.items():
            source_status = {
                "name": name,
                "type": config.type.value,
                "priority": config.priority.value,
                "enabled": config.enabled,
                "status": "unknown"
            }
            
            if name == "zerodha_realtime":
                if self.zerodha_connector:
                    source_status["status"] = "initialized"
                    source_status["authenticated"] = bool(self.zerodha_connector.config.access_token)
                    source_status["websocket_connected"] = self.zerodha_connector.is_connected
                else:
                    source_status["status"] = "not_initialized"
            
            elif name == "historical_local":
                if self.historical_loader:
                    source_status["status"] = "initialized"
                    cache_info = self.historical_loader.get_cache_info()
                    source_status["instruments_available"] = cache_info["total_instruments"]
                    source_status["cached_datasets"] = cache_info["cached_datasets"]
                else:
                    source_status["status"] = "not_initialized"
            
            status[name] = source_status
        
        return status
    
    def configure_data_source(self, source_name: str, config_updates: Dict[str, Any]):
        """Update configuration for a data source."""
        if source_name not in self.data_sources:
            logger.error(f"Data source '{source_name}' not found")
            return False
        
        try:
            # Update configuration
            source_config = self.data_sources[source_name]
            
            for key, value in config_updates.items():
                if hasattr(source_config, key):
                    setattr(source_config, key, value)
                elif key in source_config.config_params:
                    source_config.config_params[key] = value
                else:
                    logger.warning(f"Unknown config parameter: {key}")
            
            logger.info(f"Updated configuration for {source_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure data source {source_name}: {e}")
            return False
    
    def disconnect_all(self):
        """Disconnect all data sources."""
        if self.zerodha_connector:
            self.zerodha_connector.disconnect()
        
        self.active_subscriptions.clear()
        self.data_cache.clear()
        
        logger.info("All data sources disconnected")
    
    # Additional methods for integration test compatibility
    async def subscribe_real_time(self, symbols: List[str], callback: Callable) -> str:
        """Subscribe to real-time data with callback."""
        try:
            subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
            success = self.subscribe_real_time_data(symbols, callback)
            if success:
                return subscription_id
            else:
                raise Exception("Failed to subscribe to real-time data")
        except Exception as e:
            logger.error(f"Real-time subscription error: {e}")
            raise
    
    async def unsubscribe_real_time(self, subscription_id: str):
        """Unsubscribe from real-time data."""
        # For now, disconnect all (simplified implementation)
        try:
            if self.zerodha_connector:
                self.zerodha_connector.disconnect()
            logger.info(f"Unsubscribed from real-time data: {subscription_id}")
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status."""
        try:
            if self.zerodha_connector:
                # In a real implementation, this would call Zerodha API
                return {
                    "market": "NSE",
                    "status": "open",  # This would be fetched from API
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "market": "unknown",
                    "status": "unavailable",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Market status error: {e}")
            raise
    
    async def get_alternative_data(self, data_type: str) -> Dict[str, Any]:
        """Get alternative data."""
        try:
            # Placeholder implementation for alternative data
            return {
                "data_type": data_type,
                "status": "mock_data",
                "available": False,
                "message": "Alternative data sources not implemented yet",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Alternative data error: {e}")
            raise


class DataSourceMessageHandler(MessageHandler):
    """Message handler for data source requests."""
    
    def __init__(self, data_manager: DataSourceManager):
        self.data_manager = data_manager
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle data source related messages."""
        try:
            if message.topic == "data.instruments":
                # Get available instruments
                source = message.payload.get("source", "zerodha")
                exchange = message.payload.get("exchange")
                
                instruments = self.data_manager.get_available_instruments(source, exchange)
                
                return Message(
                    type=message.type,
                    source="data_source_manager",
                    topic=f"{message.topic}.response",
                    payload={
                        "instruments": instruments,
                        "count": len(instruments),
                        "source": source
                    }
                )
            
            elif message.topic == "data.quote":
                # Get real-time quotes
                symbols = message.payload.get("symbols", [])
                
                quotes = self.data_manager.get_real_time_quote(symbols)
                
                return Message(
                    type=message.type,
                    source="data_source_manager",
                    topic=f"{message.topic}.response",
                    payload={
                        "quotes": quotes,
                        "symbols": symbols
                    }
                )
            
            elif message.topic == "data.historical":
                # Get historical data
                symbol = message.payload.get("symbol")
                interval = message.payload.get("interval", "day")
                source = message.payload.get("source", "auto")
                
                from_date = None
                to_date = None
                
                if "from_date" in message.payload:
                    from_date = datetime.fromisoformat(message.payload["from_date"])
                if "to_date" in message.payload:
                    to_date = datetime.fromisoformat(message.payload["to_date"])
                
                df = self.data_manager.get_historical_data(symbol, from_date, to_date, interval, source)
                
                response_data = {"symbol": symbol, "status": "success"}
                if df is not None and not df.empty:
                    response_data["data"] = df.to_dict("records")
                    response_data["records"] = len(df)
                else:
                    response_data["status"] = "no_data"
                    response_data["data"] = []
                    response_data["records"] = 0
                
                return Message(
                    type=message.type,
                    source="data_source_manager",
                    topic=f"{message.topic}.response",
                    payload=response_data
                )
            
            elif message.topic == "data.status":
                # Get data source status
                status = self.data_manager.get_data_source_status()
                
                return Message(
                    type=message.type,
                    source="data_source_manager",
                    topic=f"{message.topic}.response",
                    payload={"status": status}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling data message: {e}")
            return Message(
                type=message.type,
                source="data_source_manager",
                topic=f"{message.topic}.error",
                payload={"error": str(e)}
            )
    
    def get_supported_topics(self) -> List[str]:
        """Get supported message topics."""
        return [
            "data.instruments",
            "data.quote", 
            "data.historical",
            "data.status",
            "data.subscribe",
            "data.configure"
        ]


# Factory function
def create_data_source_manager(comm_system: Optional[CommunicationSystem] = None) -> DataSourceManager:
    """Create a data source manager."""
    return DataSourceManager(comm_system)
