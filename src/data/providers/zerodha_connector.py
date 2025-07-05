"""
Zerodha Kite API Connector for Friday AI Trading System
Handles real-time and historical data from Zerodha Kite API.
"""

import asyncio
import json
import logging
import pandas as pd
import requests
import time
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
from urllib.parse import urljoin

from ...infrastructure.logging import get_logger
from ...infrastructure.config.config_manager import ConfigurationManager

# Create logger
logger = get_logger(__name__)


@dataclass
class KiteConfig:
    """Configuration for Kite API."""
    api_key: str
    api_secret: str
    access_token: Optional[str] = None
    request_token: Optional[str] = None
    base_url: str = "https://api.kite.trade"
    websocket_url: str = "wss://ws.kite.trade"
    timeout: int = 30


class ZerodhaKiteConnector:
    """
    Connector for Zerodha Kite API to fetch real-time and historical market data.
    Compatible with the Friday AI Trading System data pipeline.
    """
    
    def __init__(self, config: Optional[KiteConfig] = None):
        """Initialize the Zerodha Kite connector."""
        self.config = config or self._load_config()
        self.session = requests.Session()
        self.session.headers.update({
            'X-Kite-Version': '3',
            'User-Agent': 'Friday-AI-Trading-System'
        })
        
        # WebSocket connection for real-time data
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Data cache
        self.instrument_cache = {}
        self.quote_cache = {}
        
        logger.info("Zerodha Kite connector initialized")
    
    def _load_config(self) -> KiteConfig:
        """Load configuration from config manager."""
        try:
            config_manager = ConfigurationManager.get_instance()
            
            api_key = config_manager.get("zerodha.api_key", "")
            api_secret = config_manager.get("zerodha.api_secret", "")
            access_token = config_manager.get("zerodha.access_token")
            
            if not api_key or not api_secret:
                logger.warning("Zerodha API credentials not found in config")
            
            return KiteConfig(
                api_key=api_key,
                api_secret=api_secret,
                access_token=access_token
            )
        except Exception as e:
            logger.error(f"Failed to load Kite config: {e}")
            return KiteConfig(api_key="", api_secret="")
    
    def set_access_token(self, access_token: str):
        """Set the access token for API authentication."""
        self.config.access_token = access_token
        self.session.headers.update({
            'Authorization': f'token {self.config.api_key}:{access_token}'
        })
        logger.info("Access token set successfully")
    
    def get_login_url(self) -> str:
        """Get the login URL for obtaining request token."""
        return f"https://kite.trade/connect/login?api_key={self.config.api_key}"
    
    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Generate session using request token."""
        try:
            import hashlib
            
            # Generate checksum
            checksum_data = f"{self.config.api_key}{request_token}{self.config.api_secret}"
            checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
            
            # Make API call to generate session
            url = urljoin(self.config.base_url, "/session/token")
            payload = {
                "api_key": self.config.api_key,
                "request_token": request_token,
                "checksum": checksum
            }
            
            response = self.session.post(url, data=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            session_data = response.json()
            if session_data.get("status") == "success":
                access_token = session_data["data"]["access_token"]
                self.set_access_token(access_token)
                
                # Store in config
                config_manager = ConfigurationManager.get_instance()
                config_manager.set("zerodha.access_token", access_token)
                
                return session_data["data"]
            else:
                raise Exception(f"Session generation failed: {session_data}")
                
        except Exception as e:
            logger.error(f"Failed to generate session: {e}")
            raise
    
    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of instruments from Kite API."""
        try:
            if exchange:
                url = urljoin(self.config.base_url, f"/instruments/{exchange}")
            else:
                url = urljoin(self.config.base_url, "/instruments")
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Parse CSV response
            import io
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            instruments = df.to_dict('records')
            
            # Cache instruments
            if exchange:
                self.instrument_cache[exchange] = instruments
            else:
                self.instrument_cache['all'] = instruments
            
            logger.info(f"Retrieved {len(instruments)} instruments")
            return instruments
            
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return []
    
    def get_quote(self, instruments: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for instruments."""
        try:
            url = urljoin(self.config.base_url, "/quote")
            params = {"i": instruments}
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            quote_data = response.json()
            if quote_data.get("status") == "success":
                quotes = quote_data["data"]
                
                # Update cache
                self.quote_cache.update(quotes)
                
                return quotes
            else:
                raise Exception(f"Quote request failed: {quote_data}")
                
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}
    
    def get_historical_data(self, instrument_token: str, from_date: datetime, 
                          to_date: datetime, interval: str = "day") -> pd.DataFrame:
        """
        Get historical data for an instrument.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, 3minute, 5minute, 10minute, 15minute, 30minute, hour, day)
        """
        try:
            url = urljoin(self.config.base_url, f"/instruments/historical/{instrument_token}/{interval}")
            params = {
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            }
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            hist_data = response.json()
            if hist_data.get("status") == "success":
                candles = hist_data["data"]["candles"]
                
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                
                logger.info(f"Retrieved {len(df)} historical records for {instrument_token}")
                return df
            else:
                raise Exception(f"Historical data request failed: {hist_data}")
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_ohlc(self, instruments: List[str]) -> Dict[str, Any]:
        """Get OHLC data for instruments."""
        try:
            url = urljoin(self.config.base_url, "/ohlc")
            params = {"i": instruments}
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            ohlc_data = response.json()
            if ohlc_data.get("status") == "success":
                return ohlc_data["data"]
            else:
                raise Exception(f"OHLC request failed: {ohlc_data}")
                
        except Exception as e:
            logger.error(f"Failed to get OHLC data: {e}")
            return {}
    
    def subscribe_quotes(self, instruments: List[str], callback: Callable):
        """Subscribe to real-time quotes via WebSocket."""
        if not self.config.access_token:
            raise Exception("Access token required for WebSocket connection")
        
        # Register callback
        for instrument in instruments:
            if instrument not in self.callbacks:
                self.callbacks[instrument] = []
            self.callbacks[instrument].append(callback)
        
        # Start WebSocket connection if not already connected
        if not self.is_connected:
            self._start_websocket()
        
        # Subscribe to instruments
        self._subscribe_instruments(instruments)
        
        logger.info(f"Subscribed to quotes for {len(instruments)} instruments")
    
    def _start_websocket(self):
        """Start WebSocket connection in a separate thread."""
        def run_websocket():
            try:
                # WebSocket URL with access token
                ws_url = f"{self.config.websocket_url}?api_key={self.config.api_key}&access_token={self.config.access_token}"
                
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close
                )
                
                self.ws.run_forever()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.is_connected = False
        
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
    
    def _on_ws_open(self, ws):
        """WebSocket connection opened."""
        self.is_connected = True
        logger.info("WebSocket connection opened")
    
    def _on_ws_message(self, ws, message):
        """Handle WebSocket message."""
        try:
            # Parse binary message (Kite uses binary format)
            import struct
            
            # Basic parsing - actual implementation would need Kite's binary protocol
            # For now, handle as JSON for demo
            try:
                data = json.loads(message)
                self._process_tick_data(data)
            except json.JSONDecodeError:
                # Handle binary data
                logger.debug("Received binary tick data")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _process_tick_data(self, data: Dict[str, Any]):
        """Process tick data and call registered callbacks."""
        instrument_token = str(data.get("instrument_token", ""))
        
        if instrument_token in self.callbacks:
            for callback in self.callbacks[instrument_token]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")
    
    def _on_ws_error(self, ws, error):
        """WebSocket error handler."""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        logger.info("WebSocket connection closed")
        self.is_connected = False
    
    def _subscribe_instruments(self, instruments: List[str]):
        """Subscribe to instruments via WebSocket."""
        if self.ws and self.is_connected:
            # Format subscription message according to Kite WebSocket protocol
            subscription_msg = {
                "a": "subscribe",
                "v": instruments
            }
            
            self.ws.send(json.dumps(subscription_msg))
            logger.info(f"Sent subscription for {len(instruments)} instruments")
    
    def disconnect(self):
        """Disconnect WebSocket and cleanup."""
        if self.ws:
            self.ws.close()
        
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
        
        self.is_connected = False
        self.callbacks.clear()
        
        logger.info("Zerodha connector disconnected")
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            url = urljoin(self.config.base_url, "/user/profile")
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            profile_data = response.json()
            if profile_data.get("status") == "success":
                return profile_data["data"]
            else:
                raise Exception(f"Profile request failed: {profile_data}")
                
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return {}
    
    def get_margins(self) -> Dict[str, Any]:
        """Get account margins."""
        try:
            url = urljoin(self.config.base_url, "/user/margins")
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            margin_data = response.json()
            if margin_data.get("status") == "success":
                return margin_data["data"]
            else:
                raise Exception(f"Margins request failed: {margin_data}")
                
        except Exception as e:
            logger.error(f"Failed to get margins: {e}")
            return {}


class ZerodhaDataAdapter:
    """
    Adapter to convert Zerodha data to Friday AI Trading System format.
    """
    
    def __init__(self, connector: ZerodhaKiteConnector):
        self.connector = connector
        
    def normalize_instrument_data(self, kite_instruments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Kite instrument data to standard format."""
        normalized = []
        
        for instrument in kite_instruments:
            normalized_instrument = {
                "symbol": instrument.get("tradingsymbol", ""),
                "name": instrument.get("name", ""),
                "instrument_token": str(instrument.get("instrument_token", "")),
                "exchange": instrument.get("exchange", ""),
                "segment": instrument.get("segment", ""),
                "instrument_type": instrument.get("instrument_type", ""),
                "tick_size": instrument.get("tick_size", 0.05),
                "lot_size": instrument.get("lot_size", 1),
                "expiry": instrument.get("expiry", ""),
                "strike": instrument.get("strike", 0.0)
            }
            normalized.append(normalized_instrument)
        
        return normalized
    
    def normalize_quote_data(self, kite_quotes: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Kite quote data to standard format."""
        normalized_quotes = {}
        
        for instrument_token, quote in kite_quotes.items():
            normalized_quote = {
                "instrument_token": instrument_token,
                "timestamp": datetime.now().isoformat(),
                "last_price": quote.get("last_price", 0.0),
                "volume": quote.get("volume", 0),
                "buy_quantity": quote.get("depth", {}).get("buy", [{}])[0].get("quantity", 0),
                "sell_quantity": quote.get("depth", {}).get("sell", [{}])[0].get("quantity", 0),
                "ohlc": {
                    "open": quote.get("ohlc", {}).get("open", 0.0),
                    "high": quote.get("ohlc", {}).get("high", 0.0),
                    "low": quote.get("ohlc", {}).get("low", 0.0),
                    "close": quote.get("ohlc", {}).get("close", 0.0)
                }
            }
            normalized_quotes[instrument_token] = normalized_quote
        
        return normalized_quotes
    
    def normalize_historical_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize historical data format."""
        if df.empty:
            return df
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Ensure consistent column names
        df.columns = [col.lower() for col in df.columns]
        
        # Add additional metadata
        df["data_source"] = "zerodha_kite"
        df["last_updated"] = datetime.now()
        
        return df


# Factory function for easy instantiation
def create_zerodha_connector(api_key: str = None, api_secret: str = None, 
                           access_token: str = None) -> ZerodhaKiteConnector:
    """Create a Zerodha Kite connector with optional credentials."""
    if api_key and api_secret:
        config = KiteConfig(
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token
        )
        return ZerodhaKiteConnector(config)
    else:
        return ZerodhaKiteConnector()  # Use config from ConfigurationManager
