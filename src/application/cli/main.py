#!/usr/bin/env python
"""Command Line Interface for Friday AI Trading System.

This module provides a command-line interface for interacting with the
Friday AI Trading System, allowing users to start services, run backtests,
manage configurations, and more.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.infrastructure.logging import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments. Defaults to None, which uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Friday AI Trading System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # MCP Servers command
    mcp_parser = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP command")

    # MCP start command
    mcp_start_parser = mcp_subparsers.add_parser("start", help="Start MCP servers")
    mcp_start_parser.add_argument(
        "--memory", action="store_true", help="Start only the Memory MCP server"
    )
    mcp_start_parser.add_argument(
        "--thinking", action="store_true", help="Start only the Sequential Thinking MCP server"
    )

    # MCP stop command
    mcp_stop_parser = mcp_subparsers.add_parser("stop", help="Stop MCP servers")
    mcp_stop_parser.add_argument(
        "--memory", action="store_true", help="Stop only the Memory MCP server"
    )
    mcp_stop_parser.add_argument(
        "--thinking", action="store_true", help="Stop only the Sequential Thinking MCP server"
    )

    # MCP status command
    mcp_subparsers.add_parser("status", help="Check MCP servers status")

    # Trading command
    trading_parser = subparsers.add_parser("trading", help="Trading operations")
    trading_subparsers = trading_parser.add_subparsers(
        dest="trading_command", help="Trading command"
    )

    # Trading start command
    trading_start_parser = trading_subparsers.add_parser(
        "start", help="Start trading engine"
    )
    trading_start_parser.add_argument(
        "--strategy", type=str, required=True, help="Trading strategy to use"
    )
    trading_start_parser.add_argument(
        "--symbols", type=str, nargs="+", required=True, help="Symbols to trade"
    )
    trading_start_parser.add_argument(
        "--paper", action="store_true", help="Use paper trading"
    )

    # Trading stop command
    trading_subparsers.add_parser("stop", help="Stop trading engine")

    # Trading status command
    trading_subparsers.add_parser("status", help="Check trading engine status")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtests")
    backtest_parser.add_argument(
        "--strategy", type=str, required=True, help="Trading strategy to backtest"
    )
    backtest_parser.add_argument(
        "--symbols", type=str, nargs="+", required=True, help="Symbols to backtest"
    )
    backtest_parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--timeframe", type=str, default="1d", help="Timeframe (e.g., 1m, 5m, 1h, 1d)"
    )

    # Feature Engineering command
    features_parser = subparsers.add_parser("features", help="Manage feature engineering")
    features_subparsers = features_parser.add_subparsers(
        dest="features_command", help="Feature engineering command"
    )

    # Features list command
    features_list_parser = features_subparsers.add_parser(
        "list", help="List available feature sets"
    )
    features_list_parser.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled feature sets"
    )

    # Features enable command
    features_enable_parser = features_subparsers.add_parser(
        "enable", help="Enable feature sets"
    )
    features_enable_parser.add_argument(
        "feature_sets", nargs="+", 
        choices=["price_derived", "moving_averages", "volatility", "momentum", "volume", "trend", "all"],
        help="Feature sets to enable"
    )

    # Features disable command
    features_disable_parser = features_subparsers.add_parser(
        "disable", help="Disable feature sets"
    )
    features_disable_parser.add_argument(
        "feature_sets", nargs="+", 
        choices=["price_derived", "moving_averages", "volatility", "momentum", "volume", "trend", "all"],
        help="Feature sets to disable"
    )

    # Features benchmark command
    features_benchmark_parser = features_subparsers.add_parser(
        "benchmark", help="Benchmark feature generation performance"
    )
    features_benchmark_parser.add_argument(
        "--dataset-size", type=str, default="1month", 
        choices=["1month", "1year", "custom"],
        help="Dataset size for benchmarking"
    )
    features_benchmark_parser.add_argument(
        "--custom-rows", type=int, default=10000,
        help="Number of rows for custom dataset size"
    )
    features_benchmark_parser.add_argument(
        "--feature-sets", nargs="*", 
        choices=["price_derived", "moving_averages", "volatility", "momentum", "volume", "trend"],
        help="Specific feature sets to benchmark (default: all enabled)"
    )
    features_benchmark_parser.add_argument(
        "--output", type=str, help="Output file for benchmark results"
    )

    # Features validate command
    features_validate_parser = features_subparsers.add_parser(
        "validate", help="Validate required columns for enabled features"
    )
    features_validate_parser.add_argument(
        "--data-file", type=str, help="Optional data file to validate against"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Configuration command"
    )

    # Config show command
    config_show_parser = config_subparsers.add_parser(
        "show", help="Show current configuration"
    )
    config_show_parser.add_argument(
        "--section", type=str, help="Configuration section to show"
    )

    # Config update command
    config_update_parser = config_subparsers.add_parser(
        "update", help="Update configuration"
    )
    config_update_parser.add_argument(
        "--section", type=str, required=True, help="Configuration section to update"
    )
    config_update_parser.add_argument(
        "--key", type=str, required=True, help="Configuration key to update"
    )
    config_update_parser.add_argument(
        "--value", type=str, required=True, help="New value"
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    return parser.parse_args(args)


def main() -> None:
    """Main entry point for the CLI application."""
    # Set up logging
    setup_logging()

    # Parse arguments
    args = parse_args()

    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        sys.exit(1)

    # Handle MCP commands
    if args.command == "mcp":
        handle_mcp_command(args)
    # Handle trading commands
    elif args.command == "trading":
        handle_trading_command(args)
    # Handle backtest command
    elif args.command == "backtest":
        handle_backtest_command(args)
    # Handle features commands
    elif args.command == "features":
        handle_features_command(args)
    # Handle config commands
    elif args.command == "config":
        handle_config_command(args)
    # Handle version command
    elif args.command == "version":
        show_version()


def handle_mcp_command(args: argparse.Namespace) -> None:
    """Handle MCP server commands.

    Args:
        args: Parsed command line arguments.
    """
    from src.mcp_servers import start_servers, stop_servers, check_status

    if args.mcp_command == "start":
        # Determine which servers to start
        memory_only = args.memory and not args.thinking
        thinking_only = args.thinking and not args.memory
        # If neither flag is set, start both servers
        if not args.memory and not args.thinking:
            memory_only = False
            thinking_only = False

        start_servers(memory_only=memory_only, thinking_only=thinking_only)
    elif args.mcp_command == "stop":
        # Determine which servers to stop
        memory_only = args.memory and not args.thinking
        thinking_only = args.thinking and not args.memory
        # If neither flag is set, stop both servers
        if not args.memory and not args.thinking:
            memory_only = False
            thinking_only = False

        stop_servers(memory_only=memory_only, thinking_only=thinking_only)
    elif args.mcp_command == "status":
        check_status()
    else:
        print("Error: Unknown MCP command. Use --help for usage information.")
        sys.exit(1)


def handle_trading_command(args: argparse.Namespace) -> None:
    """Handle trading commands.

    Args:
        args: Parsed command line arguments.
    """
    # This will be implemented in a future task
    print("Trading commands not yet implemented.")


def handle_backtest_command(args: argparse.Namespace) -> None:
    """Handle backtest command.

    Args:
        args: Parsed command line arguments.
    """
    # This will be implemented in a future task
    print("Backtest command not yet implemented.")


def handle_features_command(args: argparse.Namespace) -> None:
    """Handle feature engineering commands.

    Args:
        args: Parsed command line arguments.
    """
    from src.data.processing.feature_engineering import FeatureEngineer
    from src.infrastructure.config import ConfigManager
    from src.infrastructure.logging import get_logger
    import pandas as pd
    import json
    import time
    import psutil
    import os
    from datetime import datetime, timedelta
    import numpy as np

    logger = get_logger(__name__)
    config = ConfigManager()
    
    if args.features_command == "list":
        # List available feature sets
        feature_engineer = FeatureEngineer(config=config)
        
        print("\n=== Available Feature Sets ===")
        for name, feature_set in feature_engineer.feature_sets.items():
            enabled = all(feature in feature_engineer.enabled_features for feature in feature_set.features)
            status = "✓ ENABLED" if enabled else "✗ DISABLED"
            
            if args.enabled_only and not enabled:
                continue
                
            print(f"\n{status} {name}")
            print(f"  Category: {feature_set.category.name}")
            print(f"  Description: {feature_set.description}")
            print(f"  Features ({len(feature_set.features)}): {', '.join(feature_set.features)}")
            print(f"  Dependencies: {', '.join(feature_set.dependencies)}")
    
    elif args.features_command == "enable":
        # Enable feature sets
        feature_engineer = FeatureEngineer(config=config)
        
        for feature_set_name in args.feature_sets:
            if feature_set_name == "all":
                feature_engineer.enable_all_features()
                print("✓ Enabled all feature sets")
                break
            else:
                try:
                    feature_engineer.enable_feature_set(feature_set_name)
                    print(f"✓ Enabled feature set: {feature_set_name}")
                except ValueError as e:
                    print(f"✗ Failed to enable {feature_set_name}: {e}")
        
        # Save to config
        enabled_sets = feature_engineer.get_enabled_feature_sets()
        config.set("features.enabled_feature_sets", enabled_sets)
        print(f"\nTotal enabled feature sets: {len(enabled_sets)}")
    
    elif args.features_command == "disable":
        # Disable feature sets
        feature_engineer = FeatureEngineer(config=config)
        
        # First enable sets from config if they exist
        enabled_sets = config.get("features.enabled_feature_sets", [])
        for feature_set in enabled_sets:
            try:
                feature_engineer.enable_feature_set(feature_set)
            except ValueError:
                pass  # Ignore if feature set doesn't exist
        
        for feature_set_name in args.feature_sets:
            if feature_set_name == "all":
                feature_engineer.disable_all_features()
                print("✓ Disabled all feature sets")
                break
            else:
                try:
                    feature_engineer.disable_feature_set(feature_set_name)
                    print(f"✓ Disabled feature set: {feature_set_name}")
                except ValueError as e:
                    print(f"✗ Failed to disable {feature_set_name}: {e}")
        
        # Save to config
        enabled_sets = feature_engineer.get_enabled_feature_sets()
        config.set("features.enabled_feature_sets", enabled_sets)
        print(f"\nTotal enabled feature sets: {len(enabled_sets)}")
    
    elif args.features_command == "benchmark":
        # Benchmark feature generation
        feature_engineer = FeatureEngineer(config=config)
        
        # Load enabled sets from config
        enabled_sets = config.get("features.enabled_feature_sets", [])
        for feature_set in enabled_sets:
            try:
                feature_engineer.enable_feature_set(feature_set)
            except ValueError:
                pass
        
        # Override with specific feature sets if provided
        if args.feature_sets:
            feature_engineer.disable_all_features()
            for feature_set in args.feature_sets:
                try:
                    feature_engineer.enable_feature_set(feature_set)
                except ValueError as e:
                    print(f"Warning: {e}")
        
        # Generate sample data based on dataset size
        if args.dataset_size == "1month":
            num_rows = 30 * 24 * 60  # 1 month of minute data
            description = "1-month dataset (30 days of minute data)"
        elif args.dataset_size == "1year":
            num_rows = 365 * 24 * 60  # 1 year of minute data
            description = "1-year dataset (365 days of minute data)"
        else:  # custom
            num_rows = args.custom_rows
            description = f"Custom dataset ({num_rows} rows)"
        
        print(f"\n=== Feature Generation Benchmark ===")
        print(f"Dataset: {description}")
        print(f"Enabled feature sets: {feature_engineer.get_enabled_feature_sets()}")
        print(f"Total features: {len(feature_engineer.enabled_features)}")
        
        # Generate sample data
        print("\nGenerating sample data...")
        data = _generate_sample_market_data(num_rows)
        print(f"Generated {len(data)} rows of market data")
        
        # Benchmark feature generation
        print("\nBenchmarking feature generation...")
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure processing time
        start_time = time.time()
        processed_data = feature_engineer.process_data(data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Calculate metrics
        original_size = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        processed_size = processed_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        size_increase = processed_size - original_size
        
        new_features = len(processed_data.columns) - len(data.columns)
        processing_rate = num_rows / processing_time if processing_time > 0 else float('inf')
        
        results = {
            'dataset_size': args.dataset_size,
            'num_rows': num_rows,
            'enabled_feature_sets': feature_engineer.get_enabled_feature_sets(),
            'total_features': len(feature_engineer.enabled_features),
            'new_features_generated': new_features,
            'processing_time_seconds': processing_time,
            'processing_rate_rows_per_second': processing_rate,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_increase,
            'original_data_size_mb': original_size,
            'processed_data_size_mb': processed_size,
            'data_size_increase_mb': size_increase,
            'memory_per_feature_mb': memory_increase / new_features if new_features > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n=== Benchmark Results ===")
        print(f"Processing Time: {processing_time:.4f} seconds")
        print(f"Processing Rate: {processing_rate:.0f} rows/second")
        print(f"Memory Usage: {memory_before:.2f} MB → {memory_after:.2f} MB (+{memory_increase:.2f} MB)")
        print(f"Data Size: {original_size:.2f} MB → {processed_size:.2f} MB (+{size_increase:.2f} MB)")
        print(f"Features Generated: {new_features}")
        print(f"Memory per Feature: {memory_increase / new_features:.4f} MB/feature" if new_features > 0 else "No features generated")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nBenchmark results saved to: {args.output}")
    
    elif args.features_command == "validate":
        # Validate required columns
        feature_engineer = FeatureEngineer(config=config)
        
        # Load enabled sets from config
        enabled_sets = config.get("features.enabled_feature_sets", [])
        for feature_set in enabled_sets:
            try:
                feature_engineer.enable_feature_set(feature_set)
            except ValueError:
                pass
        
        required_columns = feature_engineer.get_required_columns()
        
        print(f"\n=== Required Columns Validation ===")
        print(f"Enabled feature sets: {feature_engineer.get_enabled_feature_sets()}")
        print(f"Required columns: {sorted(required_columns)}")
        
        # Standard OHLCV columns
        standard_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_standard = required_columns - standard_columns
        
        print(f"\n=== Column Analysis ===")
        print(f"Standard OHLCV columns: {sorted(standard_columns)}")
        print(f"Additional required: {sorted(missing_standard) if missing_standard else 'None'}")
        
        if args.data_file:
            # Validate against actual data file
            try:
                if args.data_file.endswith('.csv'):
                    data = pd.read_csv(args.data_file)
                elif args.data_file.endswith(('.json', '.jsonl')):
                    data = pd.read_json(args.data_file)
                else:
                    print(f"Error: Unsupported file format for {args.data_file}")
                    return
                
                available_columns = set(data.columns)
                missing_columns = required_columns - available_columns
                extra_columns = available_columns - required_columns
                
                print(f"\n=== Data File Validation ===")
                print(f"File: {args.data_file}")
                print(f"Available columns: {sorted(available_columns)}")
                print(f"Missing columns: {sorted(missing_columns) if missing_columns else 'None'}")
                print(f"Extra columns: {sorted(extra_columns) if extra_columns else 'None'}")
                
                if missing_columns:
                    print(f"\n⚠️  Warning: Missing required columns. Feature generation may fail.")
                else:
                    print(f"\n✅ All required columns are available.")
                    
            except Exception as e:
                print(f"Error reading data file: {e}")
    
    else:
        print("Error: Unknown features command. Use --help for usage information.")
        sys.exit(1)


def _generate_sample_market_data(num_rows: int):
    """Generate sample market data for benchmarking.
    
    Args:
        num_rows: Number of rows to generate.
        
    Returns:
        pd.DataFrame: Sample market data.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    # Generate datetime index
    start_date = datetime.now() - timedelta(days=num_rows // (24 * 60))
    date_range = pd.date_range(start=start_date, periods=num_rows, freq='1min')
    
    # Generate realistic market data
    np.random.seed(42)  # For reproducibility
    
    # Base price
    base_price = 100.0
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0.0001, 0.02, num_rows)  # Small positive trend with volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=date_range)
    
    # Close prices
    data['close'] = prices
    
    # Open prices (close of previous period with small gap)
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.001, num_rows))
    data['open'].iloc[0] = base_price
    
    # High and low prices
    volatility = np.random.uniform(0.005, 0.03, num_rows)
    data['high'] = np.maximum(data['open'], data['close']) * (1 + volatility)
    data['low'] = np.minimum(data['open'], data['close']) * (1 - volatility)
    
    # Volume (correlated with price volatility)
    base_volume = 10000
    volume_multiplier = 1 + np.abs(returns) * 10  # Higher volume during high volatility
    data['volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, num_rows)).astype(int)
    
    # Ensure no negative prices
    data = data.clip(lower=0.01)
    
    return data


def handle_config_command(args: argparse.Namespace) -> None:
    """Handle configuration commands.

    Args:
        args: Parsed command line arguments.
    """
    from src.infrastructure.config import ConfigManager
    import json
    
    config = ConfigManager()
    
    if args.config_command == "show":
        if args.section:
            # Show specific section
            section_config = config.get(args.section, {})
            print(f"\n=== Configuration Section: {args.section} ===")
            print(json.dumps(section_config, indent=2))
        else:
            # Show all configuration
            all_config = config.get_all()
            print("\n=== All Configuration ===")
            print(json.dumps(all_config, indent=2))
    
    elif args.config_command == "update":
        # Convert value to appropriate type
        value = args.value
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
            value = float(value)
        
        # Build the path
        path = f"{args.section}.{args.key}"
        
        # Set the value
        config.set(path, value)
        print(f"✓ Updated configuration: {path} = {value}")
    
    else:
        print("Error: Unknown config command. Use --help for usage information.")
        sys.exit(1)


def show_version() -> None:
    """Show version information."""
    # Get version from setup.py or package metadata
    try:
        from importlib.metadata import version

        version_str = version("friday-ai-trading")
    except ImportError:
        # Fallback for Python < 3.8
        try:
            import pkg_resources

            version_str = pkg_resources.get_distribution("friday-ai-trading").version
        except pkg_resources.DistributionNotFound:
            version_str = "0.1.0 (development)"

    print(f"Friday AI Trading System v{version_str}")


if __name__ == "__main__":
    main()