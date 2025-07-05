"""Data module for backtesting framework.

This module provides classes for handling different data sources and formats
for the backtesting framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import datetime

import pandas as pd
import numpy as np

from src.backtesting.engine import Event, EventType
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    CSV = "csv"              # CSV file
    PARQUET = "parquet"      # Parquet file
    SQL = "sql"              # SQL database
    API = "api"              # API (REST, WebSocket, etc.)
    CUSTOM = "custom"        # Custom data source


class DataField(Enum):
    """Standard data fields."""
    OPEN = "open"            # Open price
    HIGH = "high"            # High price
    LOW = "low"              # Low price
    CLOSE = "close"          # Close price
    VOLUME = "volume"        # Volume
    TIMESTAMP = "timestamp"  # Timestamp
    SYMBOL = "symbol"        # Symbol


class DataFrequency(Enum):
    """Data frequencies."""
    TICK = "tick"            # Tick data
    SECOND = "1s"            # 1 second
    MINUTE = "1m"            # 1 minute
    FIVE_MINUTE = "5m"       # 5 minutes
    FIFTEEN_MINUTE = "15m"   # 15 minutes
    THIRTY_MINUTE = "30m"    # 30 minutes
    HOUR = "1h"              # 1 hour
    FOUR_HOUR = "4h"         # 4 hours
    DAY = "1d"               # 1 day
    WEEK = "1w"              # 1 week
    MONTH = "1M"             # 1 month


class DataSource(ABC):
    """Abstract base class for data sources.
    
    This class defines the interface for data sources in the backtesting
    framework. Concrete data sources should inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, name: str = None):
        """Initialize the data source.
        
        Args:
            name: Data source name (default: None, uses class name)
        """
        self.name = name or self.__class__.__name__
        self.data = {}
        self.symbols = []
        self.start_date = None
        self.end_date = None
        self.frequency = None
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load data from the source.
        
        This method must be implemented by concrete data sources to load
        data from the source.
        
        Returns:
            Dictionary of DataFrames (symbol -> DataFrame)
        """
        pass
    
    def get_data(self, symbol: str = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Get the loaded data.
        
        Args:
            symbol: Symbol to get data for (default: None, returns all data)
            
        Returns:
            Dictionary of DataFrames or a single DataFrame
        """
        if symbol is not None:
            if symbol in self.data:
                return self.data[symbol]
            else:
                raise ValueError(f"Symbol {symbol} not found in data")
        else:
            return self.data
    
    def get_symbols(self) -> List[str]:
        """Get the list of symbols.
        
        Returns:
            List of symbols
        """
        return self.symbols
    
    def get_start_date(self) -> pd.Timestamp:
        """Get the start date of the data.
        
        Returns:
            Start date
        """
        return self.start_date
    
    def get_end_date(self) -> pd.Timestamp:
        """Get the end date of the data.
        
        Returns:
            End date
        """
        return self.end_date
    
    def get_frequency(self) -> DataFrequency:
        """Get the frequency of the data.
        
        Returns:
            Data frequency
        """
        return self.frequency
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and standardize a DataFrame.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol of the DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(f"DataFrame for symbol {symbol} is empty")
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime for symbol {symbol}: {e}")
        
        # Sort by index
        df = df.sort_index()
        
        # Check for required columns
        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for symbol {symbol}: {missing_columns}")
        
        # Add symbol column if not present
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        
        return df
    
    def _update_metadata(self) -> None:
        """Update metadata based on loaded data."""
        # Update symbols
        self.symbols = list(self.data.keys())
        
        # Update start and end dates
        if self.symbols:
            start_dates = [df.index.min() for df in self.data.values() if not df.empty]
            end_dates = [df.index.max() for df in self.data.values() if not df.empty]
            
            if start_dates:
                self.start_date = min(start_dates)
            
            if end_dates:
                self.end_date = max(end_dates)
        
        # Infer frequency if not set
        if self.frequency is None and self.symbols:
            for df in self.data.values():
                if not df.empty and len(df) > 1:
                    # Calculate median time delta
                    time_deltas = df.index.to_series().diff().dropna()
                    if not time_deltas.empty:
                        median_delta = time_deltas.median()
                        
                        # Map to DataFrequency
                        if median_delta <= pd.Timedelta(seconds=1):
                            self.frequency = DataFrequency.TICK
                        elif median_delta <= pd.Timedelta(seconds=5):
                            self.frequency = DataFrequency.SECOND
                        elif median_delta <= pd.Timedelta(minutes=3):
                            self.frequency = DataFrequency.MINUTE
                        elif median_delta <= pd.Timedelta(minutes=10):
                            self.frequency = DataFrequency.FIVE_MINUTE
                        elif median_delta <= pd.Timedelta(minutes=20):
                            self.frequency = DataFrequency.FIFTEEN_MINUTE
                        elif median_delta <= pd.Timedelta(minutes=45):
                            self.frequency = DataFrequency.THIRTY_MINUTE
                        elif median_delta <= pd.Timedelta(hours=2):
                            self.frequency = DataFrequency.HOUR
                        elif median_delta <= pd.Timedelta(hours=12):
                            self.frequency = DataFrequency.FOUR_HOUR
                        elif median_delta <= pd.Timedelta(days=3):
                            self.frequency = DataFrequency.DAY
                        elif median_delta <= pd.Timedelta(days=10):
                            self.frequency = DataFrequency.WEEK
                        else:
                            self.frequency = DataFrequency.MONTH
                        
                        break


class CSVDataSource(DataSource):
    """CSV data source.
    
    This class loads data from CSV files.
    """
    
    def __init__(self, name: str = None):
        """Initialize the CSV data source.
        
        Args:
            name: Data source name (default: None, uses class name)
        """
        super().__init__(name)
    
    def load(
        self,
        file_paths: Union[str, List[str]],
        symbols: Optional[List[str]] = None,
        date_column: str = "date",
        symbol_column: Optional[str] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        frequency: Optional[Union[str, DataFrequency]] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files.
        
        Args:
            file_paths: Path(s) to CSV file(s)
            symbols: List of symbols (default: None, inferred from files or data)
            date_column: Name of the date column (default: "date")
            symbol_column: Name of the symbol column (default: None)
            start_date: Start date for filtering data (default: None)
            end_date: End date for filtering data (default: None)
            frequency: Data frequency (default: None, inferred from data)
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            Dictionary of DataFrames (symbol -> DataFrame)
        """
        # Convert file_paths to list if it's a string
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Convert frequency to DataFrequency if it's a string
        if isinstance(frequency, str):
            try:
                frequency = DataFrequency(frequency)
            except ValueError:
                raise ValueError(f"Invalid frequency: {frequency}")
        
        self.frequency = frequency
        
        # Convert start_date and end_date to pd.Timestamp if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Load data from each file
        for file_path in file_paths:
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load CSV file
            df = pd.read_csv(file_path, **kwargs)
            
            # Convert date column to datetime and set as index
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            else:
                raise ValueError(f"Date column '{date_column}' not found in file: {file_path}")
            
            # Filter by date range if specified
            if start_date is not None:
                df = df[df.index >= start_date]
            
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # Handle symbol(s)
            if symbol_column is not None and symbol_column in df.columns:
                # Multiple symbols in one file
                unique_symbols = df[symbol_column].unique()
                for symbol in unique_symbols:
                    symbol_df = df[df[symbol_column] == symbol].copy()
                    self.data[symbol] = self._validate_dataframe(symbol_df, symbol)
            else:
                # Single symbol per file
                if symbols is not None and len(symbols) == len(file_paths):
                    # Use provided symbol
                    symbol = symbols[file_paths.index(file_path)]
                else:
                    # Use filename as symbol
                    symbol = os.path.splitext(os.path.basename(file_path))[0]
                
                self.data[symbol] = self._validate_dataframe(df, symbol)
        
        # Update metadata
        self._update_metadata()
        
        return self.data


class ParquetDataSource(DataSource):
    """Parquet data source.
    
    This class loads data from Parquet files.
    """
    
    def __init__(self, name: str = None):
        """Initialize the Parquet data source.
        
        Args:
            name: Data source name (default: None, uses class name)
        """
        super().__init__(name)
    
    def load(
        self,
        file_paths: Union[str, List[str]],
        symbols: Optional[List[str]] = None,
        date_column: str = "date",
        symbol_column: Optional[str] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        frequency: Optional[Union[str, DataFrequency]] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Load data from Parquet files.
        
        Args:
            file_paths: Path(s) to Parquet file(s)
            symbols: List of symbols (default: None, inferred from files or data)
            date_column: Name of the date column (default: "date")
            symbol_column: Name of the symbol column (default: None)
            start_date: Start date for filtering data (default: None)
            end_date: End date for filtering data (default: None)
            frequency: Data frequency (default: None, inferred from data)
            **kwargs: Additional arguments for pd.read_parquet()
            
        Returns:
            Dictionary of DataFrames (symbol -> DataFrame)
        """
        # Convert file_paths to list if it's a string
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Convert frequency to DataFrequency if it's a string
        if isinstance(frequency, str):
            try:
                frequency = DataFrequency(frequency)
            except ValueError:
                raise ValueError(f"Invalid frequency: {frequency}")
        
        self.frequency = frequency
        
        # Convert start_date and end_date to pd.Timestamp if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Load data from each file
        for file_path in file_paths:
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load Parquet file
            df = pd.read_parquet(file_path, **kwargs)
            
            # Convert date column to datetime and set as index
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            else:
                raise ValueError(f"Date column '{date_column}' not found in file: {file_path}")
            
            # Filter by date range if specified
            if start_date is not None:
                df = df[df.index >= start_date]
            
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # Handle symbol(s)
            if symbol_column is not None and symbol_column in df.columns:
                # Multiple symbols in one file
                unique_symbols = df[symbol_column].unique()
                for symbol in unique_symbols:
                    symbol_df = df[df[symbol_column] == symbol].copy()
                    self.data[symbol] = self._validate_dataframe(symbol_df, symbol)
            else:
                # Single symbol per file
                if symbols is not None and len(symbols) == len(file_paths):
                    # Use provided symbol
                    symbol = symbols[file_paths.index(file_path)]
                else:
                    # Use filename as symbol
                    symbol = os.path.splitext(os.path.basename(file_path))[0]
                
                self.data[symbol] = self._validate_dataframe(df, symbol)
        
        # Update metadata
        self._update_metadata()
        
        return self.data


class SQLDataSource(DataSource):
    """SQL data source.
    
    This class loads data from SQL databases.
    """
    
    def __init__(self, name: str = None):
        """Initialize the SQL data source.
        
        Args:
            name: Data source name (default: None, uses class name)
        """
        super().__init__(name)
    
    def load(
        self,
        connection_string: str,
        query: str,
        symbols: Optional[List[str]] = None,
        date_column: str = "date",
        symbol_column: Optional[str] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        frequency: Optional[Union[str, DataFrequency]] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Load data from a SQL database.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            symbols: List of symbols (default: None, inferred from data)
            date_column: Name of the date column (default: "date")
            symbol_column: Name of the symbol column (default: None)
            start_date: Start date for filtering data (default: None)
            end_date: End date for filtering data (default: None)
            frequency: Data frequency (default: None, inferred from data)
            **kwargs: Additional arguments for pd.read_sql()
            
        Returns:
            Dictionary of DataFrames (symbol -> DataFrame)
        """
        # Convert frequency to DataFrequency if it's a string
        if isinstance(frequency, str):
            try:
                frequency = DataFrequency(frequency)
            except ValueError:
                raise ValueError(f"Invalid frequency: {frequency}")
        
        self.frequency = frequency
        
        # Convert start_date and end_date to pd.Timestamp if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Modify query to include date range if specified
        if start_date is not None or end_date is not None:
            # Check if query already has a WHERE clause
            if "WHERE" in query.upper():
                where_clause = " AND "
            else:
                where_clause = " WHERE "
            
            if start_date is not None:
                where_clause += f"{date_column} >= '{start_date}'"
                
                if end_date is not None:
                    where_clause += f" AND {date_column} <= '{end_date}'"
            elif end_date is not None:
                where_clause += f"{date_column} <= '{end_date}'"
            
            # Add WHERE clause to query
            query += where_clause
        
        # Load data from database
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load data from database: {e}")
        
        # Convert date column to datetime and set as index
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        else:
            raise ValueError(f"Date column '{date_column}' not found in query result")
        
        # Handle symbol(s)
        if symbol_column is not None and symbol_column in df.columns:
            # Multiple symbols in one query result
            unique_symbols = df[symbol_column].unique()
            for symbol in unique_symbols:
                symbol_df = df[df[symbol_column] == symbol].copy()
                self.data[symbol] = self._validate_dataframe(symbol_df, symbol)
        else:
            # Single symbol
            if symbols is not None and len(symbols) == 1:
                # Use provided symbol
                symbol = symbols[0]
            else:
                # Use a default symbol
                symbol = "data"
            
            self.data[symbol] = self._validate_dataframe(df, symbol)
        
        # Update metadata
        self._update_metadata()
        
        return self.data


class APIDataSource(DataSource):
    """API data source.
    
    This class loads data from APIs (REST, WebSocket, etc.).
    """
    
    def __init__(self, name: str = None):
        """Initialize the API data source.
        
        Args:
            name: Data source name (default: None, uses class name)
        """
        super().__init__(name)
    
    def load(
        self,
        api_url: str,
        symbols: List[str],
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        frequency: Optional[Union[str, DataFrequency]] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        date_format: Optional[str] = None,
        response_type: str = "json",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Load data from an API.
        
        Args:
            api_url: API URL
            symbols: List of symbols
            start_date: Start date for filtering data (default: None)
            end_date: End date for filtering data (default: None)
            frequency: Data frequency (default: None, inferred from data)
            api_key: API key (default: None)
            headers: HTTP headers (default: None)
            params: Query parameters (default: None)
            date_format: Date format for parsing timestamps (default: None)
            response_type: Response type (default: "json")
            **kwargs: Additional arguments for API requests
            
        Returns:
            Dictionary of DataFrames (symbol -> DataFrame)
        """
        # Convert frequency to DataFrequency if it's a string
        if isinstance(frequency, str):
            try:
                frequency = DataFrequency(frequency)
            except ValueError:
                raise ValueError(f"Invalid frequency: {frequency}")
        
        self.frequency = frequency
        
        # Convert start_date and end_date to pd.Timestamp if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Initialize headers and params
        if headers is None:
            headers = {}
        
        if params is None:
            params = {}
        
        # Add API key to headers or params if provided
        if api_key is not None:
            # Try to guess where to put the API key
            if "key" in params or "apikey" in params or "api_key" in params:
                # API key goes in params
                pass
            elif "Authorization" in headers or "X-API-Key" in headers:
                # API key goes in headers
                pass
            else:
                # Default to params
                params["apikey"] = api_key
        
        # Load data for each symbol
        import requests
        
        for symbol in symbols:
            # Add symbol to params
            params["symbol"] = symbol
            
            # Add date range to params if specified
            if start_date is not None:
                params["start_date"] = start_date.strftime("%Y-%m-%d")
            
            if end_date is not None:
                params["end_date"] = end_date.strftime("%Y-%m-%d")
            
            # Make API request
            try:
                response = requests.get(api_url, headers=headers, params=params, **kwargs)
                response.raise_for_status()  # Raise exception for HTTP errors
            except Exception as e:
                raise ValueError(f"Failed to load data from API for symbol {symbol}: {e}")
            
            # Parse response
            if response_type.lower() == "json":
                try:
                    data = response.json()
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON response for symbol {symbol}: {e}")
                
                # Convert to DataFrame
                try:
                    # Handle different JSON structures
                    if isinstance(data, list):
                        # List of records
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Dictionary with data field
                        if "data" in data:
                            df = pd.DataFrame(data["data"])
                        elif "results" in data:
                            df = pd.DataFrame(data["results"])
                        elif "prices" in data:
                            df = pd.DataFrame(data["prices"])
                        else:
                            # Try to find a list field
                            list_fields = [k for k, v in data.items() if isinstance(v, list)]
                            if list_fields:
                                df = pd.DataFrame(data[list_fields[0]])
                            else:
                                # Use the whole dict as a single row
                                df = pd.DataFrame([data])
                    else:
                        raise ValueError(f"Unexpected JSON structure for symbol {symbol}")
                except Exception as e:
                    raise ValueError(f"Failed to convert JSON to DataFrame for symbol {symbol}: {e}")
            elif response_type.lower() == "csv":
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                except Exception as e:
                    raise ValueError(f"Failed to parse CSV response for symbol {symbol}: {e}")
            else:
                raise ValueError(f"Unsupported response type: {response_type}")
            
            # Convert timestamp column to datetime and set as index
            timestamp_columns = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
            if timestamp_columns:
                timestamp_column = timestamp_columns[0]
                try:
                    if date_format is not None:
                        df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=date_format)
                    else:
                        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                    
                    df.set_index(timestamp_column, inplace=True)
                except Exception as e:
                    raise ValueError(f"Failed to convert timestamp column for symbol {symbol}: {e}")
            else:
                raise ValueError(f"No timestamp column found for symbol {symbol}")
            
            # Validate and store DataFrame
            self.data[symbol] = self._validate_dataframe(df, symbol)
        
        # Update metadata
        self._update_metadata()
        
        return self.data


class DataHandler:
    """Data handler for backtesting.
    
    This class handles data for backtesting, including loading data from
    different sources, preprocessing, and generating events.
    """
    
    def __init__(self):
        """Initialize the data handler."""
        self.data_sources = {}
        self.data = {}
        self.symbols = []
        self.start_date = None
        self.end_date = None
        self.current_date = None
        self.events = []


class OHLCVDataHandler(DataHandler):
    """Data handler specifically for OHLCV (Open, High, Low, Close, Volume) data.
    
    This class extends the base DataHandler with specific functionality for
    handling OHLCV data commonly used in financial backtesting.
    """
    
    def __init__(self):
        """Initialize the OHLCV data handler."""
        super().__init__()
        self.latest_data = {}
        self.bars_processed = 0
        
    def update_bars(self):
        """Update the latest bars for all symbols.
        
        This method updates the latest_data dictionary with the latest bars
        for each symbol based on the current date.
        
        Returns:
            True if new bars were added, False otherwise
        """
        if self.current_date is None:
            if self.start_date is None:
                raise ValueError("No start date set")
            self.current_date = self.start_date
        
        # Check if we have reached the end date
        if self.end_date is not None and self.current_date > self.end_date:
            return False
        
        # Update latest data for each symbol
        new_bars = False
        for symbol in self.symbols:
            # Get data for the symbol
            symbol_data = self.data[symbol]
            
            # Get data for the current date
            try:
                # Get data on or before the current date
                latest_bars = symbol_data[symbol_data.index <= self.current_date]
                
                if not latest_bars.empty:
                    # Store the latest bars
                    self.latest_data[symbol] = latest_bars
                    new_bars = True
            except Exception as e:
                logger.error(f"Error updating bars for {symbol}: {e}")
        
        # Increment bars processed
        if new_bars:
            self.bars_processed += 1
        
        return new_bars
    
    def get_latest_bars(self, symbol: str, N: int = 1) -> pd.DataFrame:
        """Get the latest N bars for a symbol.
        
        Args:
            symbol: Symbol to get bars for
            N: Number of bars to get (default: 1)
            
        Returns:
            DataFrame with the latest N bars
        """
        if symbol not in self.latest_data:
            raise ValueError(f"Symbol {symbol} not found in latest data")
        
        # Get the latest N bars
        return self.latest_data[symbol].iloc[-N:]
    
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Get the latest bar for a symbol.
        
        Args:
            symbol: Symbol to get bar for
            
        Returns:
            Series with the latest bar
        """
        return self.get_latest_bars(symbol, 1).iloc[0]
    
    def get_latest_bar_value(self, symbol: str, field: Union[str, DataField]) -> float:
        """Get the latest bar value for a symbol and field.
        
        Args:
            symbol: Symbol to get bar value for
            field: Field to get value for (can be string or DataField enum)
            
        Returns:
            Value for the field
        """
        # Convert DataField enum to string if needed
        if isinstance(field, DataField):
            field = field.value
        
        # Get the latest bar
        latest_bar = self.get_latest_bar(symbol)
        
        # Return the value for the field
        if field in latest_bar:
            return latest_bar[field]
        else:
            raise ValueError(f"Field {field} not found in latest bar for {symbol}")
    
    def get_latest_bars_values(self, symbol: str, field: Union[str, DataField], N: int = 1) -> np.ndarray:
        """Get the latest N bar values for a symbol and field.
        
        Args:
            symbol: Symbol to get bar values for
            field: Field to get values for (can be string or DataField enum)
            N: Number of bars to get (default: 1)
            
        Returns:
            Array with the latest N bar values
        """
        # Convert DataField enum to string if needed
        if isinstance(field, DataField):
            field = field.value
        
        # Get the latest N bars
        latest_bars = self.get_latest_bars(symbol, N)
        
        # Return the values for the field
        if field in latest_bars.columns:
            return latest_bars[field].values
        else:
            raise ValueError(f"Field {field} not found in latest bars for {symbol}")
    
    def add_data_source(self, data_source: DataSource, name: str = None) -> None:
        """Add a data source.
        
        Args:
            data_source: Data source to add
            name: Name of the data source (default: None, uses data source name)
        """
        name = name or data_source.name
        self.data_sources[name] = data_source
        
        # Add data from the source
        for symbol, df in data_source.get_data().items():
            self.data[symbol] = df
            if symbol not in self.symbols:
                self.symbols.append(symbol)
        
        # Update start and end dates
        source_start_date = data_source.get_start_date()
        source_end_date = data_source.get_end_date()
        
        if source_start_date is not None:
            if self.start_date is None or source_start_date < self.start_date:
                self.start_date = source_start_date
        
        if source_end_date is not None:
            if self.end_date is None or source_end_date > self.end_date:
                self.end_date = source_end_date
    
    def get_data(self, symbol: str = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Get the loaded data.
        
        Args:
            symbol: Symbol to get data for (default: None, returns all data)
            
        Returns:
            Dictionary of DataFrames or a single DataFrame
        """
        if symbol is not None:
            if symbol in self.data:
                return self.data[symbol]
            else:
                raise ValueError(f"Symbol {symbol} not found in data")
        else:
            return self.data
    
    def get_symbols(self) -> List[str]:
        """Get the list of symbols.
        
        Returns:
            List of symbols
        """
        return self.symbols
    
    def get_start_date(self) -> pd.Timestamp:
        """Get the start date of the data.
        
        Returns:
            Start date
        """
        return self.start_date
    
    def get_end_date(self) -> pd.Timestamp:
        """Get the end date of the data.
        
        Returns:
            End date
        """
        return self.end_date
    
    def get_current_date(self) -> pd.Timestamp:
        """Get the current date of the backtest.
        
        Returns:
            Current date
        """
        return self.current_date
    
    def set_current_date(self, date: pd.Timestamp) -> None:
        """Set the current date of the backtest.
        
        Args:
            date: Current date
        """
        self.current_date = date
    
    def generate_events(self, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> List[Event]:
        """Generate events from the data.
        
        Args:
            start_date: Start date for generating events (default: None, uses data start date)
            end_date: End date for generating events (default: None, uses data end date)
            
        Returns:
            List of events
        """
        # Use data start and end dates if not specified
        if start_date is None:
            start_date = self.start_date
        
        if end_date is None:
            end_date = self.end_date
        
        # Check if we have data
        if not self.data:
            raise ValueError("No data loaded")
        
        # Generate events
        events = []
        
        # Get all timestamps from all symbols
        all_timestamps = set()
        for symbol, df in self.data.items():
            # Filter by date range
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
            all_timestamps.update(filtered_df.index)
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        # Generate events for each timestamp
        for timestamp in all_timestamps:
            for symbol, df in self.data.items():
                if timestamp in df.index:
                    # Create bar event
                    event = Event(
                        type=EventType.BAR,
                        time=timestamp,
                        symbol=symbol,
                        data=df.loc[timestamp].to_dict(),
                    )
                    events.append(event)
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.time)
        
        self.events = events
        return events
    
    def get_events(self) -> List[Event]:
        """Get the generated events.
        
        Returns:
            List of events
        """
        return self.events
    
    def get_latest_data(self, symbol: str, lookback: int = 1) -> pd.DataFrame:
        """Get the latest data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback: Number of bars to look back (default: 1)
            
        Returns:
            DataFrame with the latest data
        """
        if symbol not in self.data:
            raise ValueError(f"Symbol {symbol} not found in data")
        
        if self.current_date is None:
            raise ValueError("Current date not set")
        
        # Get data up to current date
        df = self.data[symbol]
        df = df[df.index <= self.current_date]
        
        # Get the latest bars
        if lookback > len(df):
            lookback = len(df)
        
        return df.iloc[-lookback:]
    
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Get the latest bar for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Series with the latest bar
        """
        latest_data = self.get_latest_data(symbol, lookback=1)
        if latest_data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        return latest_data.iloc[0]
    
    def get_latest_bar_value(self, symbol: str, field: Union[str, DataField]) -> float:
        """Get the latest bar value for a symbol and field.
        
        Args:
            symbol: Symbol to get data for
            field: Field to get value for
            
        Returns:
            Value of the field
        """
        if isinstance(field, DataField):
            field = field.value
        
        latest_bar = self.get_latest_bar(symbol)
        if field not in latest_bar:
            raise ValueError(f"Field {field} not found in latest bar for symbol {symbol}")
        
        return latest_bar[field]
    
    def get_latest_bars_values(self, symbol: str, field: Union[str, DataField], lookback: int = 1) -> np.ndarray:
        """Get the latest bars values for a symbol and field.
        
        Args:
            symbol: Symbol to get data for
            field: Field to get values for
            lookback: Number of bars to look back (default: 1)
            
        Returns:
            Array of values
        """
        if isinstance(field, DataField):
            field = field.value
        
        latest_data = self.get_latest_data(symbol, lookback=lookback)
        if field not in latest_data.columns:
            raise ValueError(f"Field {field} not found in latest data for symbol {symbol}")
        
        return latest_data[field].values