"""
Historical Data Loader for Friday AI Trading System
Loads historical market data from local files in src/data/market directories.
"""

import os
import pandas as pd
import glob
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ...infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class HistoricalDataLoader:
    """
    Loads historical market data from local CSV/JSON files.
    Supports the three data directories in src/data/market.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the historical data loader."""
        if base_path is None:
            # Default to src/data/market directory
            current_dir = Path(__file__).parent.parent
            self.base_path = current_dir / "market"
        else:
            self.base_path = Path(base_path)
        
        # Define data source paths
        self.data_sources = {
            "intraday": self.base_path / "market_data",
            "long_term": self.base_path / "NSE nifty50 stocks 20 year data", 
            "multi_timeframe": self.base_path / "nse_market_data"
        }
        
        # Cache for loaded data
        self.data_cache = {}
        self.instrument_metadata = {}
        
        logger.info(f"Historical data loader initialized with base path: {self.base_path}")
        self._scan_available_data()
    
    def _scan_available_data(self):
        """Scan available data files and build metadata."""
        for source_name, source_path in self.data_sources.items():
            if source_path.exists():
                logger.info(f"Scanning {source_name} data in {source_path}")
                
                if source_name == "intraday":
                    self._scan_intraday_data(source_path)
                elif source_name == "long_term":
                    self._scan_long_term_data(source_path)
                elif source_name == "multi_timeframe":
                    self._scan_multi_timeframe_data(source_path)
            else:
                logger.warning(f"Data source path not found: {source_path}")
    
    def _scan_intraday_data(self, path: Path):
        """Scan intraday data directory."""
        symbols = []
        for symbol_dir in path.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                symbols.append(symbol)
                
                # Check for available files
                csv_files = list(symbol_dir.glob("*.csv"))
                json_files = list(symbol_dir.glob("*.json"))
                summary_files = list(symbol_dir.glob("*summary.json"))
                
                self.instrument_metadata[symbol] = {
                    "source": "intraday",
                    "symbol": symbol,
                    "csv_files": [f.name for f in csv_files],
                    "json_files": [f.name for f in json_files],
                    "summary_files": [f.name for f in summary_files],
                    "path": str(symbol_dir)
                }
        
        logger.info(f"Found {len(symbols)} symbols in intraday data: {symbols[:10]}...")
    
    def _scan_long_term_data(self, path: Path):
        """Scan long-term historical data directory."""
        csv_files = list(path.glob("*.csv"))
        symbols = []
        
        for csv_file in csv_files:
            if "_fixed" not in csv_file.name:  # Skip fixed files for now, use originals
                symbol = csv_file.stem
                symbols.append(symbol)
                
                self.instrument_metadata[f"{symbol}_long_term"] = {
                    "source": "long_term",
                    "symbol": symbol,
                    "file": csv_file.name,
                    "path": str(csv_file),
                    "timeframe": "daily"
                }
        
        logger.info(f"Found {len(symbols)} symbols in long-term data: {symbols[:10]}...")
    
    def _scan_multi_timeframe_data(self, path: Path):
        """Scan multi-timeframe data directory."""
        timeframe_dirs = []
        for timeframe_dir in path.iterdir():
            if timeframe_dir.is_dir():
                timeframe = timeframe_dir.name
                timeframe_dirs.append(timeframe)
                
                csv_files = list(timeframe_dir.glob("*.csv"))
                symbols = []
                
                for csv_file in csv_files:
                    if "_fixed" not in csv_file.name:  # Skip fixed files
                        symbol = csv_file.stem
                        symbols.append(symbol)
                        
                        key = f"{symbol}_{timeframe.replace(' ', '_')}"
                        self.instrument_metadata[key] = {
                            "source": "multi_timeframe",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "file": csv_file.name,
                            "path": str(csv_file)
                        }
                
                logger.info(f"Found {len(symbols)} symbols in {timeframe} timeframe")
        
        logger.info(f"Available timeframes: {timeframe_dirs}")
    
    def get_available_symbols(self, source: Optional[str] = None) -> List[str]:
        """Get list of available symbols."""
        symbols = set()
        
        for key, metadata in self.instrument_metadata.items():
            if source is None or metadata["source"] == source:
                symbols.add(metadata["symbol"])
        
        return sorted(list(symbols))
    
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes."""
        timeframes = set()
        
        for metadata in self.instrument_metadata.values():
            if "timeframe" in metadata:
                timeframes.add(metadata["timeframe"])
        
        return sorted(list(timeframes))
    
    def load_symbol_data(self, symbol: str, source: str = "auto", 
                        timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: Symbol to load
            source: Data source ("intraday", "long_term", "multi_timeframe", "auto")
            timeframe: Specific timeframe for multi_timeframe source
        """
        try:
            if source == "auto":
                # Try to find the best available source
                source = self._determine_best_source(symbol, timeframe)
            
            if source == "intraday":
                return self._load_intraday_data(symbol)
            elif source == "long_term":
                return self._load_long_term_data(symbol)
            elif source == "multi_timeframe":
                return self._load_multi_timeframe_data(symbol, timeframe)
            else:
                logger.error(f"Unknown data source: {source}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return None
    
    def _determine_best_source(self, symbol: str, timeframe: Optional[str] = None) -> str:
        """Determine the best data source for a symbol."""
        # Priority: multi_timeframe > long_term > intraday
        
        if timeframe:
            # Check if multi-timeframe data exists for this timeframe
            key = f"{symbol}_{timeframe.replace(' ', '_')}"
            if key in self.instrument_metadata:
                return "multi_timeframe"
        
        # Check long-term data
        if f"{symbol}_long_term" in self.instrument_metadata:
            return "long_term"
        
        # Check intraday data
        if symbol in self.instrument_metadata:
            return "intraday"
        
        # Default fallback
        return "intraday"
    
    def _load_intraday_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load intraday data for a symbol."""
        cache_key = f"intraday_{symbol}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        if symbol not in self.instrument_metadata:
            logger.warning(f"Symbol {symbol} not found in intraday data")
            return None
        
        metadata = self.instrument_metadata[symbol]
        symbol_path = Path(metadata["path"])
        
        # Try to load CSV file (prefer _fixed version if available)
        csv_files = [f for f in metadata["csv_files"] if "intraday" in f]
        if not csv_files:
            logger.warning(f"No intraday CSV files found for {symbol}")
            return None
        
        # Prefer fixed version
        fixed_files = [f for f in csv_files if "_fixed" in f]
        if fixed_files:
            csv_file = fixed_files[0]
        else:
            csv_file = csv_files[0]
        
        file_path = symbol_path / csv_file
        
        try:
            df = pd.read_csv(file_path)
            df = self._normalize_dataframe(df, symbol, "intraday")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            logger.info(f"Loaded {len(df)} intraday records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load intraday CSV for {symbol}: {e}")
            return None
    
    def _load_long_term_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load long-term historical data for a symbol."""
        cache_key = f"long_term_{symbol}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        key = f"{symbol}_long_term"
        if key not in self.instrument_metadata:
            logger.warning(f"Symbol {symbol} not found in long-term data")
            return None
        
        metadata = self.instrument_metadata[key]
        file_path = Path(metadata["path"])
        
        try:
            df = pd.read_csv(file_path)
            df = self._normalize_dataframe(df, symbol, "long_term")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            logger.info(f"Loaded {len(df)} long-term records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load long-term CSV for {symbol}: {e}")
            return None
    
    def _load_multi_timeframe_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load multi-timeframe data for a symbol."""
        cache_key = f"multi_{symbol}_{timeframe}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        key = f"{symbol}_{timeframe.replace(' ', '_')}"
        if key not in self.instrument_metadata:
            logger.warning(f"Symbol {symbol} not found in {timeframe} timeframe data")
            return None
        
        metadata = self.instrument_metadata[key]
        file_path = Path(metadata["path"])
        
        try:
            df = pd.read_csv(file_path)
            df = self._normalize_dataframe(df, symbol, f"multi_timeframe_{timeframe}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            logger.info(f"Loaded {len(df)} {timeframe} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {timeframe} CSV for {symbol}: {e}")
            return None
    
    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """Normalize DataFrame format and add metadata."""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure consistent column names (lowercase)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Handle different timestamp formats
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                if timestamp_col != 'timestamp':
                    df['timestamp'] = df[timestamp_col]
                    df.drop(columns=[timestamp_col], inplace=True)
                
                # Set timestamp as index if not already
                if 'timestamp' in df.columns and df.index.name != 'timestamp':
                    df.set_index('timestamp', inplace=True)
                    
            except Exception as e:
                logger.warning(f"Failed to parse timestamp for {symbol}: {e}")
        
        # Ensure OHLCV columns are numeric
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata columns
        df['symbol'] = symbol
        df['data_source'] = source
        df['loaded_at'] = datetime.now()
        
        # Sort by timestamp
        if 'timestamp' in df.index.names or df.index.name == 'timestamp':
            df.sort_index(inplace=True)
        
        return df
    
    def load_multiple_symbols(self, symbols: List[str], source: str = "auto", 
                            timeframe: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            df = self.load_symbol_data(symbol, source, timeframe)
            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"Failed to load data for {symbol}")
        
        logger.info(f"Successfully loaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_symbol_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary information for a symbol."""
        summary = {
            "symbol": symbol,
            "available_sources": [],
            "timeframes": [],
            "date_ranges": {}
        }
        
        for key, metadata in self.instrument_metadata.items():
            if metadata["symbol"] == symbol:
                source = metadata["source"]
                summary["available_sources"].append(source)
                
                if "timeframe" in metadata:
                    summary["timeframes"].append(metadata["timeframe"])
                
                # Try to get date range by loading a small sample
                try:
                    if source == "intraday":
                        df = self._load_intraday_data(symbol)
                    elif source == "long_term":
                        df = self._load_long_term_data(symbol)
                    elif source == "multi_timeframe":
                        df = self._load_multi_timeframe_data(symbol, metadata.get("timeframe", ""))
                    
                    if df is not None and not df.empty:
                        summary["date_ranges"][source] = {
                            "start": df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
                            "end": df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max()),
                            "records": len(df)
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get date range for {symbol} from {source}: {e}")
        
        return summary
    
    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_info = {
            "cached_datasets": len(self.data_cache),
            "total_instruments": len(self.instrument_metadata),
            "available_sources": list(self.data_sources.keys()),
            "cache_keys": list(self.data_cache.keys())
        }
        return cache_info


# Factory function for easy instantiation
def create_historical_loader(base_path: Optional[str] = None) -> HistoricalDataLoader:
    """Create a historical data loader."""
    return HistoricalDataLoader(base_path)
