#!/usr/bin/env python3
"""
Process All Market Data for Friday AI Trading System

This script processes all market data files from the three folders under src/data/market:
1. market_data/
2. NSE nifty50 stocks 20 year data/
3. nse_market_data/

It uses the MarketDataPipeline to ensure all data goes through the complete data pipeline:
- Data validation
- Data cleaning
- Feature engineering
- Storage (Parquet format)

Usage:
    python process_all_market_data.py [--dry-run] [--verbose] [--outdir OUTPUT_DIR] [--folder FOLDER]
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import pandas as pd
import traceback
import time
import logging
from datetime import datetime
import json

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem
from pipelines.market_data_pipeline import initialize_pipeline

# Create logger
logger = get_logger(__name__)

def discover_market_data_files(folder=None):
    """
    Discover all market data files from the three folders under src/data/market.

    Args:
        folder (str, optional): Process only this specific folder. Can be any folder name
                               under src/data/market or one of the shortcuts: 'market_data',
                               'nifty50', 'nse_market_data'

    Returns:
        list: List of file paths to all market data CSV files.
    """
    logger.info("Discovering market data files...")

    # Get base market data directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "data", "market")

    # Define known folder shortcuts
    market_data_dirs = {
        "market_data": "market_data",
        "nifty50": "NSE nifty50 stocks 20 year data",
        "nse_market_data": "nse_market_data"
    }

    all_files = []

    if folder:
        # Check if the folder is a shortcut or direct subfolder name
        if folder in market_data_dirs:
            target_folder = os.path.join(base_dir, market_data_dirs[folder])
        else:
            # Try as direct subfolder
            target_folder = os.path.join(base_dir, folder)

        if not os.path.exists(target_folder):
            logger.error(f"Invalid folder '{folder}'. Folder not found at {target_folder}")
            logger.info(f"Valid shortcut options are: {', '.join(market_data_dirs.keys())} or any subfolder under src/data/market")
            return []

        directories = [target_folder]
        logger.info(f"Processing single folder: {target_folder}")
    else:
        # Process all known market data directories
        directories = [os.path.join(base_dir, d) for d in market_data_dirs.values()]
        logger.info(f"Processing all folders: {len(directories)} folders")

    for directory in directories:
        # Find all CSV files in the directory and its subdirectories
        csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
        logger.info(f"Found {len(csv_files)} CSV files in {directory}")
        all_files.extend(csv_files)

    # Filter out any summary or already processed files
    filtered_files = [f for f in all_files if "_summary" not in f and "_processed" not in f]

    logger.info(f"Total market data files to process: {len(filtered_files)}")
    return filtered_files

def get_symbol_from_filepath(filepath):
    """
    Extract symbol from file path.

    Args:
        filepath: Path to the CSV file.

    Returns:
        str: The extracted symbol.
    """
    # Extract the filename without extension
    filename = os.path.basename(filepath)
    symbol = os.path.splitext(filename)[0]

    # Some files may have special formats, handle them
    if "_" in symbol:
        # Handle common formats like "AAPL_data.csv" or "AAPL_1d.csv"
        symbol = symbol.split("_")[0]

    return symbol

def process_file(file_path, pipeline):
    """
    Process a single market data file through the pipeline.

    Args:
        file_path: Path to the CSV file.
        pipeline: Initialized data pipeline.

    Returns:
        pd.DataFrame or None: Processed data if successful, None otherwise.
    """
    try:
        logger.info(f"Processing file: {file_path}")

        # Extract symbol from file path
        symbol = get_symbol_from_filepath(file_path)
        logger.info(f"Symbol extracted: {symbol}")

        try:
            # Load the data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data shape: {df.shape}")

            # Add symbol information if not present
            if "symbol" not in df.columns:
                df["symbol"] = symbol

        except Exception as e:
            logger.error(f"Error loading market data from {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

        # Process the data through the pipeline
        processed_data = pipeline.process_data(df)

        # Get basic statistics
        logger.info(f"Processed data shape: {processed_data.shape}")

        return processed_data

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to process all market data."""
    parser = argparse.ArgumentParser(description="Process all market data files.")

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without storing data'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="storage/data/processed",
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--folder',
        type=str,
        help='Process only a specific market data folder (market_data, NSE nifty50 stocks 20 year data, nse_market_data)'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        # Start timing
        start_time = datetime.now()

        if args.folder:
            logger.info(f"Starting to process market data from folder '{args.folder}' at {start_time}")
        else:
            logger.info(f"Starting to process all market data at {start_time}")

        # Create output directory
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Initialize the pipeline
        pipeline = initialize_pipeline(
            warn_only=True,  # Use warn_only to prevent validation failures from stopping the pipeline
            output_dir=str(output_dir),
            enable_all_features=True,
            enable_storage=not args.dry_run
        )

        # Discover market data files (for specific folder or all)
        all_files = discover_market_data_files(args.folder)

        # Process statistics
        total_files = len(all_files)
        processed_files = 0
        failed_files = 0

        # Generate folder-specific summary name
        if args.folder:
            folder_suffix = f"_{args.folder}"
        else:
            folder_suffix = "_all"

        # Process each file
        for i, file_path in enumerate(all_files):
            logger.info(f"Processing file {i+1}/{total_files}: {file_path}")

            try:
                # Process the file
                result = process_file(file_path, pipeline)

                if result is not None:
                    processed_files += 1
                    logger.info(f"Successfully processed file {i+1}/{total_files}")
                else:
                    failed_files += 1
                    logger.warning(f"Failed to process file {i+1}/{total_files}")

            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                failed_files += 1

        # End timing and calculate duration
        end_time = datetime.now()
        duration = end_time - start_time

        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        summary['total_files'] = total_files
        summary['processed_files'] = processed_files
        summary['failed_files'] = failed_files
        summary['start_time'] = start_time.isoformat()
        summary['end_time'] = end_time.isoformat()
        summary['duration_seconds'] = duration.total_seconds()

        # Add folder information to summary
        if args.folder:
            summary['processed_folder'] = args.folder
        else:
            summary['processed_folder'] = "all"

        logger.info("Pipeline Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        # Save summary to file
        summary_file = output_dir / f"market_data_pipeline_summary{folder_suffix}.json"

        # Convert non-serializable values
        summary_serializable = {}
        for k, v in summary.items():
            try:
                json.dumps(v)
                summary_serializable[k] = v
            except (TypeError, ValueError):
                summary_serializable[k] = str(v)

        with open(summary_file, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        logger.info(f"Pipeline summary saved to {summary_file}")

        logger.info(f"Processing completed in {duration}!")
        logger.info(f"Processed {processed_files} files successfully, {failed_files} files failed")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
