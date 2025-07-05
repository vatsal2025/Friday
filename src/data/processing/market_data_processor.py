"""Market data processor module for the Friday AI Trading System.

This module provides functionality for processing market data from the market directory,
including loading, cleaning, and feature engineering for model training.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from typing import Dict, List, Optional, Tuple, Union
import traceback

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.data.processing.data_processor import DataProcessor, ProcessingStep, DataProcessingError
from src.data.processing.data_cleaner import DataCleaner, CleaningStrategy, OutlierDetectionMethod
from src.data.processing.feature_engineering import FeatureEngineer
from src.data.processing.multi_timeframe_processor import TimeframeConverter
from src.data.acquisition.data_fetcher import DataTimeframe

# Create logger
logger = get_logger(__name__)


class MarketDataProcessor:
    """Class for processing market data from the market directory.

    This class provides methods for loading, cleaning, and processing market data
    from CSV files in the market directory for model training.

    Attributes:
        config: Configuration manager.
        market_data_dir: Directory containing market data files.
        processed_data_dir: Directory to save processed data files.
        data_cleaner: Data cleaner for cleaning market data.
        feature_engineer: Feature engineer for creating features from market data.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        market_data_dir: Optional[str] = None,
        processed_data_dir: Optional[str] = None,
    ):
        """Initialize a market data processor.

        Args:
            config: Configuration manager. If None, a new one will be created.
            market_data_dir: Directory containing market data files. If None, will use config value.
            processed_data_dir: Directory to save processed data files. If None, will use config value.
        """
        self.config = config or ConfigManager()
        
        # Set directories
        self.market_data_dir = market_data_dir or self.config.get("data.market.directory", "src/data/market")
        self.processed_data_dir = processed_data_dir or self.config.get("data.processed.directory", "src/data/processed")
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize data processors
        self.data_cleaner = DataCleaner(
            config=self.config,
            default_missing_strategy=CleaningStrategy.FILL_FORWARD,
            default_outlier_strategy=CleaningStrategy.WINSORIZE,
            outlier_detection_method=OutlierDetectionMethod.IQR,
            outlier_threshold=1.5,
        )
        
        self.feature_engineer = FeatureEngineer(
            config=self.config,
            enable_all_features=False,
        )
        
        logger.info(f"Initialized MarketDataProcessor with market data directory: {self.market_data_dir}")

    def discover_data_files(self, subdirectory: Optional[str] = None) -> Dict[str, List[str]]:
        """Discover data files in the market data directory.

        Args:
            subdirectory: Optional subdirectory within market_data_dir to search.

        Returns:
            Dict[str, List[str]]: Dictionary mapping symbols to lists of file paths.
        """
        search_dir = os.path.join(self.market_data_dir, subdirectory) if subdirectory else self.market_data_dir
        logger.info(f"Discovering data files in {search_dir}")
        
        # Find all CSV files recursively
        csv_files = glob.glob(os.path.join(search_dir, "**/*.csv"), recursive=True)
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Group files by symbol
        symbol_files: Dict[str, List[str]] = {}
        for file_path in csv_files:
            # Extract symbol from filename
            filename = os.path.basename(file_path)
            symbol = filename.split("_")[0].split(".")[0]  # Handle both SYMBOL.csv and SYMBOL_timeframe.csv formats
            
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(file_path)
        
        logger.info(f"Grouped files for {len(symbol_files)} symbols")
        return symbol_files

    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            DataProcessingError: If an error occurs during loading.
        """
        try:
            logger.debug(f"Loading file: {file_path}")
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
            
            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in {file_path}: {missing_columns}")
            
            return df
            
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise DataProcessingError(error_msg)

    def process_symbol_data(self, symbol: str, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Process data for a single symbol.

        Args:
            symbol: The symbol to process.
            file_paths: List of file paths for the symbol.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping timeframes to processed DataFrames.

        Raises:
            DataProcessingError: If an error occurs during processing.
        """
        try:
            logger.info(f"Processing data for symbol: {symbol} with {len(file_paths)} files")
            
            # Dictionary to store processed data by timeframe
            processed_data: Dict[str, pd.DataFrame] = {}
            
            # Process each file
            for file_path in file_paths:
                # Determine timeframe from file path or content
                timeframe = self._determine_timeframe(file_path)
                logger.debug(f"Determined timeframe {timeframe} for {file_path}")
                
                # Load the data
                df = self.load_csv_file(file_path)
                
                # Skip empty dataframes
                if df.empty:
                    logger.warning(f"Empty dataframe for {file_path}, skipping")
                    continue
                
                # Clean the data
                cleaned_df = self.data_cleaner.process_data(df)
                
                # Generate features
                processed_df = self.feature_engineer.process_data(cleaned_df)
                
                # Store processed data
                if timeframe in processed_data:
                    # If we already have data for this timeframe, concatenate and remove duplicates
                    processed_data[timeframe] = pd.concat([processed_data[timeframe], processed_df])
                    processed_data[timeframe] = processed_data[timeframe].loc[~processed_data[timeframe].index.duplicated(keep='last')]
                    processed_data[timeframe] = processed_data[timeframe].sort_index()
                else:
                    processed_data[timeframe] = processed_df
            
            return processed_data
            
        except Exception as e:
            error_msg = f"Error processing data for symbol {symbol}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise DataProcessingError(error_msg)

    def _determine_timeframe(self, file_path: str) -> str:
        """Determine the timeframe of a data file.

        Args:
            file_path: Path to the data file.

        Returns:
            str: The determined timeframe.
        """
        # Extract timeframe from file path if possible
        filename = os.path.basename(file_path)
        dirname = os.path.basename(os.path.dirname(file_path))
        
        # Check if timeframe is in the directory name
        if any(tf in dirname.lower() for tf in ['day', 'daily', '1d']):
            return '1d'
        elif any(tf in dirname.lower() for tf in ['60min', '60 min', '1h', '1hour']):
            return '1h'
        elif any(tf in dirname.lower() for tf in ['30min', '30 min', '30m']):
            return '30m'
        elif any(tf in dirname.lower() for tf in ['15min', '15 min', '15m']):
            return '15m'
        elif any(tf in dirname.lower() for tf in ['10min', '10 min', '10m']):
            return '10m'
        elif any(tf in dirname.lower() for tf in ['5min', '5 min', '5m']):
            return '5m'
        elif any(tf in dirname.lower() for tf in ['1min', '1 min', '1m']):
            return '1m'
        
        # Check if timeframe is in the filename
        if '_1d' in filename or '_day' in filename.lower() or '_daily' in filename.lower():
            return '1d'
        elif '_1h' in filename or '_60min' in filename or '_60m' in filename:
            return '1h'
        elif '_30m' in filename or '_30min' in filename:
            return '30m'
        elif '_15m' in filename or '_15min' in filename:
            return '15m'
        elif '_10m' in filename or '_10min' in filename:
            return '10m'
        elif '_5m' in filename or '_5min' in filename:
            return '5m'
        elif '_1m' in filename or '_1min' in filename:
            return '1m'
        
        # Default to daily if we can't determine
        return '1d'

    def save_processed_data(self, symbol: str, processed_data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files.

        Args:
            symbol: The symbol of the processed data.
            processed_data: Dictionary mapping timeframes to processed DataFrames.
        """
        # Create directory for the symbol if it doesn't exist
        symbol_dir = os.path.join(self.processed_data_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save each timeframe's data
        for timeframe, df in processed_data.items():
            file_path = os.path.join(symbol_dir, f"{symbol}_{timeframe}_processed.csv")
            df.to_csv(file_path)
            logger.info(f"Saved processed data for {symbol} ({timeframe}) to {file_path}")

    def process_all_market_data(self, subdirectories: Optional[List[str]] = None) -> None:
        """Process all market data files.

        Args:
            subdirectories: Optional list of subdirectories within market_data_dir to process.
                If None, will process all files in market_data_dir.
        """
        if subdirectories:
            # Process each subdirectory
            for subdir in subdirectories:
                symbol_files = self.discover_data_files(subdir)
                self._process_symbol_files(symbol_files)
        else:
            # Process all files
            symbol_files = self.discover_data_files()
            self._process_symbol_files(symbol_files)

    def _process_symbol_files(self, symbol_files: Dict[str, List[str]]) -> None:
        """Process files for multiple symbols.

        Args:
            symbol_files: Dictionary mapping symbols to lists of file paths.
        """
        total_symbols = len(symbol_files)
        logger.info(f"Processing data for {total_symbols} symbols")
        
        for i, (symbol, file_paths) in enumerate(symbol_files.items(), 1):
            try:
                logger.info(f"Processing symbol {i}/{total_symbols}: {symbol}")
                processed_data = self.process_symbol_data(symbol, file_paths)
                self.save_processed_data(symbol, processed_data)
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                continue


def main():
    """Main function to run the market data processor."""
    # Initialize the market data processor
    processor = MarketDataProcessor()
    
    # Process all market data
    processor.process_all_market_data()
    
    logger.info("Market data processing completed")


if __name__ == "__main__":
    main()