#!/usr/bin/env python3
"""Market Data Pipeline for the Friday AI Trading System.

This module provides a complete market data pipeline that includes:
1. Enhanced validation stage with configurable warn_only mode
2. Cleaning stage using MarketDataCleaner
3. Feature engineering stage
4. Storage stage using LocalParquetStorage

The pipeline can be executed from the command line:
python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem
from src.data.integration.data_pipeline import DataPipeline
from src.data.processing.data_validator import build_default_market_validator
from src.data.processing.market_data_cleaner import MarketDataCleaner
from src.data.processing.feature_engineering import FeatureEngineer
from src.data.storage.local_parquet_storage import LocalParquetStorage

# Create logger
logger = get_logger(__name__)


def initialize_pipeline(
    config: Optional[ConfigManager] = None,
    event_system: Optional[EventSystem] = None,
    warn_only: bool = False,
    output_dir: str = "storage/data/processed",
    enable_all_features: bool = True,
    enable_storage: bool = True
) -> 'MarketDataPipeline':
    """Initialize and configure the market data pipeline with all stages.
    
    This function creates a complete market data processing pipeline that:
    1. Instantiates DataPipeline with optional EventSystem
    2. Adds stages in order: enhanced_validation → market_data_cleaning → feature_engineering → parquet_storage
    3. Publishes events to EventSystem during execution
    4. Ensures stage names match task file requirements for future orchestration
    
    Args:
        config: Configuration manager. If None, a new one will be created.
        event_system: Event system for publishing pipeline events. If None, a new one will be created and started.
        warn_only: If True, validation failures will be warnings, not errors
        output_dir: Output directory for processed data
        enable_all_features: Whether to enable all feature engineering by default
        enable_storage: Whether to enable the storage stage
        
    Returns:
        MarketDataPipeline: Configured pipeline ready for processing
        
    Example:
        >>> # Initialize with default settings
        >>> pipeline = initialize_pipeline()
        >>> 
        >>> # Initialize with custom configuration
        >>> config = ConfigManager()
        >>> event_system = EventSystem()
        >>> event_system.start()
        >>> pipeline = initialize_pipeline(
        ...     config=config,
        ...     event_system=event_system,
        ...     warn_only=True,
        ...     output_dir="/custom/output"
        ... )
        >>> 
        >>> # Process data
        >>> result = pipeline.process_data(input_data)
    """
    # Initialize configuration if not provided
    if config is None:
        config = ConfigManager()
    
    # Initialize and start event system if not provided
    if event_system is None:
        event_system = EventSystem()
        event_system.start()
        logger.info("Started new EventSystem for pipeline")
    
    # Create the underlying data pipeline with event system
    data_pipeline = DataPipeline(
        name="market_data_pipeline",
        config=config,
        event_system=event_system
    )
    
    # Add stages in the required order with exact stage names
    logger.info("Setting up pipeline stages in required order")
    
    # Stage 1: Enhanced Validation
    logger.info("Adding enhanced_validation stage")
    data_pipeline.add_enhanced_validation_stage(
        warn_only=warn_only,
        stage_name="enhanced_validation"
    )
    
    # Stage 2: Market Data Cleaning
    logger.info("Adding market_data_cleaning stage")
    market_cleaner = MarketDataCleaner(
        config=config,
        symbol_column="symbol",
        timestamp_column="timestamp",
        numeric_columns=['open', 'high', 'low', 'close', 'volume'],
        z_score_threshold=3.0,
        iqr_multiplier=1.5
    )
    data_pipeline.add_cleaning_stage(
        market_cleaner,
        stage_name="market_data_cleaning"
    )
    
    # Stage 3: Feature Engineering
    logger.info("Adding feature_engineering stage")
    feature_engineer = FeatureEngineer(
        config=config,
        enable_all_features=enable_all_features
    )
    
    # Enable specific feature sets for market data if not enabling all
    if not enable_all_features:
        feature_engineer.enable_feature_set("price_derived")
        feature_engineer.enable_feature_set("moving_averages")
        feature_engineer.enable_feature_set("volatility")
        feature_engineer.enable_feature_set("momentum")
    
    data_pipeline.add_feature_engineering_stage(
        feature_engineer,
        stage_name="feature_engineering"
    )
    
    # Stage 4: Parquet Storage (optional)
    if enable_storage:
        logger.info("Adding parquet_storage stage")
        parquet_storage = LocalParquetStorage(
            config=config,
            base_dir=str(output_dir)
        )
        data_pipeline.add_storage_stage(
            parquet_storage,
            stage_name="parquet_storage",
            table_name="market_data_processed",
            if_exists="append"
        )
    else:
        logger.info("Skipping parquet_storage stage (disabled)")
    
    # Create MarketDataPipeline wrapper with the configured pipeline
    market_pipeline = MarketDataPipeline(
        config=config,
        warn_only=warn_only,
        output_dir=output_dir,
        enable_all_features=enable_all_features,
        enable_storage=enable_storage
    )
    
    # Replace the internal pipeline with our configured one
    market_pipeline.pipeline = data_pipeline
    market_pipeline.event_system = event_system
    
    logger.info(f"Successfully initialized MarketDataPipeline with {len(data_pipeline.stages)} stages")
    logger.info(f"Stage names: {[stage['name'] for stage in data_pipeline.stages]}")
    
    return market_pipeline


class MarketDataPipeline:
    """Complete market data processing pipeline.
    
    This pipeline orchestrates the complete processing workflow for market data:
    1. Enhanced validation with comprehensive market data rules
    2. Data cleaning using MarketDataCleaner
    3. Feature engineering with technical indicators
    4. Storage in partitioned Parquet format
    
    Attributes:
        config: Configuration manager
        pipeline: Underlying DataPipeline instance
        event_system: Event system for publishing pipeline events
        warn_only: Whether validation should warn only or fail on errors
        output_dir: Directory for storing processed data
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        event_system: Optional[EventSystem] = None,
        warn_only: bool = False,
        output_dir: str = "storage/data/processed",
        enable_all_features: bool = True,
        enable_storage: bool = True
    ):
        """Initialize the market data pipeline.
        
        Args:
            config: Configuration manager
            event_system: Event system for publishing pipeline events
            warn_only: If True, validation failures will be warnings, not errors
            output_dir: Output directory for processed data
            enable_all_features: Whether to enable all feature engineering by default
            enable_storage: Whether to enable the storage stage
        """
        self.config = config or ConfigManager()
        self.event_system = event_system
        self.warn_only = warn_only
        self.output_dir = Path(output_dir)
        self.enable_all_features = enable_all_features
        self.enable_storage = enable_storage
        
        # Create the underlying data pipeline with event system
        self.pipeline = DataPipeline(
            name="market_data_pipeline",
            config=self.config,
            event_system=self.event_system
        )
        
        # Setup pipeline stages
        self._setup_pipeline_stages()
        
        logger.info(f"Initialized MarketDataPipeline with output_dir: {self.output_dir}")
    
    def _setup_pipeline_stages(self) -> None:
        """Setup all pipeline stages in the correct order."""
        
        # Stage 1: Enhanced Validation
        logger.info("Setting up enhanced validation stage")
        self.pipeline.add_enhanced_validation_stage(
            warn_only=self.warn_only,
            stage_name="enhanced_validation"
        )
        
        # Stage 2: Data Cleaning using MarketDataCleaner
        logger.info("Setting up data cleaning stage")
        market_cleaner = MarketDataCleaner(
            config=self.config,
            symbol_column="symbol",
            timestamp_column="timestamp",
            numeric_columns=['open', 'high', 'low', 'close', 'volume'],
            z_score_threshold=3.0,
            iqr_multiplier=1.5
        )
        self.pipeline.add_cleaning_stage(
            market_cleaner,
            stage_name="market_data_cleaning"
        )
        
        # Stage 3: Feature Engineering
        logger.info("Setting up feature engineering stage")
        feature_engineer = FeatureEngineer(
            config=self.config,
            enable_all_features=self.enable_all_features
        )
        
        # Enable specific feature sets for market data
        if not self.enable_all_features:
            # Enable commonly used feature sets
            feature_engineer.enable_feature_set("price_derived")
            feature_engineer.enable_feature_set("moving_averages")
            feature_engineer.enable_feature_set("volatility")
            feature_engineer.enable_feature_set("momentum")
        
        self.pipeline.add_feature_engineering_stage(
            feature_engineer,
            stage_name="feature_engineering"
        )
        
        # Stage 4: Storage using LocalParquetStorage (optional)
        if self.enable_storage:
            logger.info("Setting up storage stage")
            parquet_storage = LocalParquetStorage(
                config=self.config,
                base_dir=str(self.output_dir)
            )
            self.pipeline.add_storage_stage(
                parquet_storage,
                stage_name="parquet_storage",
                table_name="market_data_processed",
                if_exists="append"
            )
        else:
            logger.info("Skipping storage stage (disabled)")
    
    def process_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process market data through the complete pipeline.
        
        Args:
            input_data: Raw market data to process
            
        Returns:
            Processed market data with features
            
        Raises:
            Exception: If pipeline execution fails
        """
        try:
            logger.info(f"Starting market data pipeline processing for {len(input_data)} rows")
            
            # Validate input data has required columns
            required_columns = {'symbol', 'open', 'high', 'low', 'close', 'volume'}
            missing_columns = required_columns - set(input_data.columns)
            if missing_columns:
                raise ValueError(f"Input data missing required columns: {missing_columns}")
            
            # Ensure timestamp column exists (use index if datetime)
            if 'timestamp' not in input_data.columns:
                if isinstance(input_data.index, pd.DatetimeIndex):
                    input_data = input_data.reset_index()
                    if 'index' in input_data.columns:
                        input_data.rename(columns={'index': 'timestamp'}, inplace=True)
                else:
                    raise ValueError("Input data must have a 'timestamp' column or datetime index")
            
            # Ensure timestamp column is properly formatted for Parquet storage
            if 'timestamp' in input_data.columns:
                input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
                # Convert timezone-aware timestamps to UTC then remove timezone for Parquet compatibility
                if input_data['timestamp'].dt.tz is not None:
                    input_data['timestamp'] = input_data['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Execute the pipeline
            processed_data = self.pipeline.execute(input_data)
            
            logger.info(f"Successfully processed {len(processed_data)} rows")
            return processed_data
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    def process_file(self, input_file: str) -> pd.DataFrame:
        """Process market data from a file.
        
        Args:
            input_file: Path to input file (CSV or Parquet)
            
        Returns:
            Processed market data
            
        Raises:
            Exception: If file processing fails
        """
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            logger.info(f"Loading data from {input_file}")
            
            # Load data based on file extension
            if input_path.suffix.lower() == '.csv':
                data = pd.read_csv(input_file)
            elif input_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(input_file)
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
            logger.info(f"Loaded {len(data)} rows from {input_file}")
            
            # Return the loaded data (will be processed by caller)
            return data
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            raise
    
    def get_pipeline_metadata(self) -> Dict[str, Any]:
        """Get metadata about the pipeline execution.
        
        Returns:
            Dictionary with pipeline metadata
        """
        return self.pipeline.get_metadata()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline configuration and status.
        
        Returns:
            Dictionary with pipeline summary
        """
        metadata = self.get_pipeline_metadata()
        
        return {
            "pipeline_name": self.pipeline.name,
            "total_stages": len(self.pipeline.stages),
            "stage_names": [stage["name"] for stage in self.pipeline.stages],
            "warn_only_mode": self.warn_only,
            "output_directory": str(self.output_dir),
            "enable_all_features": self.enable_all_features,
            "last_run_status": metadata.get("last_run_status"),
            "last_run_duration": metadata.get("last_run_duration"),
            "total_runs": len(metadata.get("runs", []))
        }


def load_sample_data() -> pd.DataFrame:
    """Load sample market data for testing.
    
    Returns:
        Sample DataFrame with OHLCV data
    """
    import numpy as np
    
    # Generate sample market data
    np.random.seed(42)
    n_rows = 1000
    
    # Create date range
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='1min')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.002, n_rows).cumsum()
    closes = base_price * np.exp(price_changes)
    
    # Generate OHLC from closes
    opens = np.roll(closes, 1)
    opens[0] = base_price
    
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
    
    volumes = np.random.lognormal(8, 1, n_rows).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return data


def example_usage_with_event_system():
    """Example of using initialize_pipeline with EventSystem integration."""
    logger.info("=== Market Data Pipeline with EventSystem Example ===")
    
    try:
        # Create an event system and start it
        event_system = EventSystem()
        event_system.start()
        
        # Register an event handler to monitor pipeline events
        def pipeline_event_handler(event):
            logger.info(f"Event received: {event.event_type} from {event.source}")
            if event.event_type == "pipeline.start":
                logger.info(f"Pipeline started: {event.data['pipeline_name']}")
            elif event.event_type == "pipeline.stage.start":
                logger.info(f"Stage started: {event.data['stage_name']} ({event.data['stage_type']})")
            elif event.event_type == "pipeline.stage.success":
                logger.info(f"Stage completed: {event.data['stage_name']} in {event.data['duration']:.2f}s")
            elif event.event_type == "pipeline.success":
                logger.info(f"Pipeline completed successfully in {event.data['duration']:.2f}s")
            elif event.event_type == "pipeline.validation.metrics":
                metrics = event.data['validation_metrics']
                logger.info(f"Validation metrics: {metrics['rules_passed']}/{metrics['rules_tested']} rules passed")
        
        # Register the handler for all pipeline events
        event_system.register_handler(
            pipeline_event_handler,
            event_types=[
                "pipeline.start",
                "pipeline.stage.start", 
                "pipeline.stage.success",
                "pipeline.success",
                "pipeline.validation.metrics"
            ]
        )
        
        # Initialize pipeline with the event system
        pipeline = initialize_pipeline(
            event_system=event_system,
            warn_only=True,  # Use warn-only mode for demonstration
            output_dir="demo_output",
            enable_storage=False  # Disable storage for demo
        )
        
        # Generate sample data and process it
        logger.info("Generating sample market data...")
        sample_data = load_sample_data()
        
        logger.info("Processing data through pipeline...")
        processed_data = pipeline.process_data(sample_data)
        
        # Display results
        logger.info(f"Successfully processed {len(processed_data)} rows")
        logger.info(f"Original columns: {len(sample_data.columns)}")
        logger.info(f"Processed columns: {len(processed_data.columns)}")
        logger.info(f"New features added: {len(processed_data.columns) - len(sample_data.columns)}")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        logger.info("Pipeline Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Stop the event system
        event_system.stop()
        logger.info("Event system stopped")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise


def example_basic_usage():
    """Example of basic pipeline usage without custom event system."""
    logger.info("=== Basic Market Data Pipeline Example ===")
    
    try:
        # Initialize pipeline with default settings
        # This automatically creates and starts an EventSystem
        pipeline = initialize_pipeline(
            warn_only=False,
            output_dir="basic_output",
            enable_all_features=True,
            enable_storage=True
        )
        
        # Generate and process sample data
        sample_data = load_sample_data()
        processed_data = pipeline.process_data(sample_data)
        
        logger.info(f"Processing complete: {len(processed_data)} rows with {len(processed_data.columns)} columns")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Basic example failed: {str(e)}")
        raise


def main():
    """Main CLI entry point for the market data pipeline."""
    parser = argparse.ArgumentParser(
        description="Market Data Pipeline - Process market data with validation, cleaning, and feature engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipelines.market_data_pipeline --input raw_data.csv --outdir storage/data/processed
  python -m pipelines.market_data_pipeline --input data.parquet --outdir /tmp/processed --warn-only
  python -m pipelines.market_data_pipeline --sample --outdir test_output
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (CSV or Parquet format)'
    )
    parser.add_argument(
        '--outdir', '-o',
        type=str,
        default='storage/data/processed',
        help='Output directory for processed data (default: storage/data/processed)'
    )
    
    # Pipeline configuration
    parser.add_argument(
        '--warn-only',
        action='store_true',
        help='Enable warn-only mode for validation (warnings instead of errors)'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Disable feature engineering (only basic processing)'
    )
    parser.add_argument(
        '--no-storage',
        action='store_true',
        help='Disable storage stage (process data but do not store)'
    )
    
    # Special modes
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data instead of input file (for testing)'
    )
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
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting Market Data Pipeline CLI")
        
        # Validate arguments
        if not args.sample and not args.input:
            parser.error("Either --input or --sample must be specified")
        
        # Create output directory
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize pipeline using the new initialize_pipeline function
        pipeline = initialize_pipeline(
            warn_only=args.warn_only,
            output_dir=str(output_dir),
            enable_all_features=not args.no_features,
            enable_storage=not args.no_storage
        )
        
        # Get input data
        if args.sample:
            logger.info("Using sample data")
            data = load_sample_data()
        else:
            logger.info(f"Loading data from {args.input}")
            data = pipeline.process_file(args.input)
        
        # Display input data info
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input columns: {list(data.columns)}")
        if 'symbol' in data.columns:
            symbols = data['symbol'].nunique()
            logger.info(f"Number of unique symbols: {symbols}")
        
        # Process data (unless dry run)
        if args.dry_run:
            logger.info("Dry run mode - skipping actual processing")
            processed_data = data  # Just use original data
        else:
            processed_data = pipeline.process_data(data)
        
        # Display results
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Output columns: {list(processed_data.columns)}")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        logger.info("Pipeline Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Save summary to file
        summary_file = output_dir / "pipeline_summary.json"
        import json
        with open(summary_file, 'w') as f:
            # Convert non-serializable values
            summary_serializable = {}
            for k, v in summary.items():
                try:
                    json.dumps(v)
                    summary_serializable[k] = v
                except (TypeError, ValueError):
                    summary_serializable[k] = str(v)
            
            json.dump(summary_serializable, f, indent=2)
        logger.info(f"Pipeline summary saved to {summary_file}")
        
        logger.info("Market Data Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
