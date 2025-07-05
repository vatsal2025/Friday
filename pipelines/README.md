# Market Data Pipeline

This directory contains the complete market data processing pipeline for the Friday AI Trading System.

## Overview

The `market_data_pipeline.py` module provides a comprehensive data processing pipeline that includes:

1. **Enhanced Validation Stage** - Comprehensive market data validation with configurable `warn_only` mode
2. **Data Cleaning Stage** - Using `MarketDataCleaner` for outlier detection, duplicate removal, and missing value imputation
3. **Feature Engineering Stage** - Technical indicators and derived features
4. **Storage Stage** - Partitioned Parquet storage using `LocalParquetStorage`

## Features

### Pipeline Stages

1. **Enhanced Validation**
   - OHLCV column presence and type validation
   - Price consistency checks (High ≥ Open/Close/Low)
   - No negative prices/volumes
   - Timestamp uniqueness and monotonicity
   - Configurable `warn_only` mode for live data streams

2. **Data Cleaning**
   - Type coercion to float64
   - Duplicate removal based on timestamp & symbol
   - Missing value imputation (forward-fill then back-fill)
   - Outlier capping using z-score or IQR methods

3. **Feature Engineering**
   - Price-derived features (typical price, returns, etc.)
   - Moving averages (SMA, EMA)
   - Volatility indicators (ATR, Bollinger Bands, Keltner Channels)
   - Momentum indicators (RSI, Stochastic, MACD)
   - Volume indicators (OBV, VWAP, volume ratios)
   - Trend indicators (ADX, Aroon, CCI)

4. **Storage**
   - Partitioned Parquet files by symbol/date
   - Metadata tracking
   - Append/overwrite modes

### CLI Interface

The pipeline can be executed from the command line:

```bash
python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed
```

#### Command Line Options

- `--input, -i`: Input file path (CSV or Parquet format)
- `--outdir, -o`: Output directory for processed data (default: storage/data/processed)
- `--warn-only`: Enable warn-only mode for validation (warnings instead of errors)
- `--no-features`: Disable feature engineering (only basic processing)
- `--no-storage`: Disable storage stage (process data but do not store)
- `--sample`: Use sample data instead of input file (for testing)
- `--dry-run`: Perform dry run without storing data
- `--verbose, -v`: Enable verbose logging

#### Usage Examples

```bash
# Process a CSV file with all features
python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed

# Process with validation warnings only
python -m pipelines.market_data_pipeline --input data.parquet --outdir /tmp/processed --warn-only

# Process without feature engineering
python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed --no-features

# Process without storage (for testing)
python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed --no-storage

# Use sample data for testing
python -m pipelines.market_data_pipeline --sample --outdir test_output

# Dry run with verbose logging
python -m pipelines.market_data_pipeline --sample --outdir test_output --dry-run --verbose
```

## Input Data Requirements

The input data must contain the following columns:
- `symbol`: Stock symbol/ticker
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume
- `timestamp` or datetime index: Timestamp for each data point

## Output

### Processed Data
The pipeline outputs enhanced market data with:
- Original OHLCV columns (cleaned)
- Technical indicators and derived features
- Proper data types and formatting

### Metadata
- Pipeline execution summary saved as `pipeline_summary.json`
- Detailed validation and processing metrics
- Performance statistics

### Storage Structure
When storage is enabled, data is stored in partitioned Parquet files:
```
storage/data/processed/
├── AAPL/
│   ├── 2023-01-01/
│   │   ├── market_data_processed.parquet
│   │   └── market_data_processed.json
│   └── 2023-01-02/
│       ├── market_data_processed.parquet
│       └── market_data_processed.json
└── GOOGL/
    └── ...
```

## Programming Interface

### MarketDataPipeline Class

```python
from pipelines.market_data_pipeline import MarketDataPipeline

# Create pipeline
pipeline = MarketDataPipeline(
    warn_only=True,
    output_dir='storage/data/processed',
    enable_all_features=True,
    enable_storage=True
)

# Process data
processed_data = pipeline.process_data(raw_data)

# Get pipeline metadata
summary = pipeline.get_pipeline_summary()
```

### Key Methods

- `process_data(data)`: Process a DataFrame through the complete pipeline
- `process_file(file_path)`: Load and process data from a file
- `get_pipeline_summary()`: Get pipeline configuration and status
- `get_pipeline_metadata()`: Get detailed execution metadata

## Dependencies

The pipeline requires the following components from the Friday AI Trading System:
- `src.data.integration.data_pipeline.DataPipeline`
- `src.data.processing.market_data_cleaner.MarketDataCleaner`
- `src.data.processing.feature_engineering.FeatureEngineer`
- `src.data.storage.local_parquet_storage.LocalParquetStorage`
- `src.data.processing.data_validator.build_default_market_validator`

## Performance

- Processes ~1,800 rows per second on typical hardware
- Memory efficient with chunked processing
- Optimized for both batch and streaming data

## Error Handling

- Comprehensive validation with detailed error messages
- Configurable warn-only mode for production environments
- Graceful handling of missing or malformed data
- Detailed logging and error reporting

## Testing

Run the example demonstration:
```bash
python example_usage.py
```

Test CLI functionality:
```bash
python test_cli.py
```

Test full pipeline:
```bash
python test_full_pipeline.py
```
