#!/usr/bin/env python3
"""
Script to prepare raw data files with the correct column format for the pipeline.
Converts 'date' to 'timestamp' and adds 'symbol' column.
"""

import pandas as pd
import os
from pathlib import Path

def prepare_csv_file(input_file: str, output_file: str, symbol: str):
    """
    Prepare a CSV file with the correct format for the pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        symbol: Symbol to add to the data
    """
    print(f"Processing {input_file} -> {output_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"  Original columns: {df.columns.tolist()}")
    
    # Create standardized column mapping
    column_mapping = {}
    
    # Map timestamp column
    if 'DateTime' in df.columns:
        column_mapping['DateTime'] = 'timestamp'
    elif 'date' in df.columns:
        column_mapping['date'] = 'timestamp'
    elif 'Date' in df.columns:
        column_mapping['Date'] = 'timestamp'
    
    # Map symbol column
    if 'Symbol' in df.columns:
        column_mapping['Symbol'] = 'symbol'
    elif 'symbol' not in df.columns:
        # Add symbol if not present
        df['symbol'] = symbol
    
    # Map OHLCV columns
    ohlcv_mapping = {
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    column_mapping.update(ohlcv_mapping)
    
    # Apply column renaming
    df.rename(columns=column_mapping, inplace=True)
    
    # Ensure timestamp is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert timezone-aware timestamps to UTC then remove timezone for Parquet compatibility
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        raise ValueError(f"No timestamp column found in {input_file}. Available columns: {df.columns.tolist()}")
    
    # Select only the required columns
    standard_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in standard_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
    
    df = df[standard_columns]
    
    # Save the prepared file
    df.to_csv(output_file, index=False)
    
    print(f"  Processed {len(df)} rows with symbol '{symbol}'")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return len(df)

def main():
    """Main function to prepare all raw data files."""
    print("Preparing raw data files for the pipeline...")
    
    raw_dir = Path("data/raw")
    
    # File mappings: (current_file, new_file, symbol)
    file_mappings = [
        ("NYSE_BRITANNIA_202301.csv", "NYSE_BRITANNIA_202301.csv", "BRITANNIA"),
        ("NYSE_TCS_202301.csv", "NYSE_TCS_202301.csv", "TCS"),
        ("NYSE_INFY_202301.csv", "NYSE_INFY_202301.csv", "INFY"),
        ("NYSE_AXISBANK_202301.csv", "NYSE_AXISBANK_202301.csv", "AXISBANK"),
    ]
    
    total_rows = 0
    
    for current_file, new_file, symbol in file_mappings:
        input_path = raw_dir / current_file
        output_path = raw_dir / f"prepared_{new_file}"
        
        if input_path.exists():
            rows = prepare_csv_file(str(input_path), str(output_path), symbol)
            total_rows += rows
        else:
            print(f"Warning: {input_path} not found")
    
    print(f"\nTotal rows prepared: {total_rows:,}")
    print("Raw data preparation complete!")

if __name__ == "__main__":
    main()
