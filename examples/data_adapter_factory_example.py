"""Example demonstrating the usage of DataAdapterFactory with HistoricalDataFetcher.

This example shows how to use the DataAdapterFactory to create different data adapters
and use them with HistoricalDataFetcher to fetch historical market data.
"""

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from src.data.acquisition.data_adapter_factory import DataAdapterFactory
from src.data.acquisition.data_fetcher import DataSourceType, DataTimeframe
from src.data.acquisition.historical_data_fetcher import HistoricalDataFetcher
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


def fetch_and_plot_data(symbol, adapter_name, api_key=None):
    """Fetch historical data using the specified adapter and plot it.
    
    Args:
        symbol: The stock symbol to fetch data for.
        adapter_name: The name of the adapter to use.
        api_key: Optional API key for the data provider.
    """
    # Create configuration and event system
    config = ConfigManager()
    event_system = EventSystem()
    
    # Create data adapter factory
    factory = DataAdapterFactory(config, event_system)
    
    # Create the appropriate adapter based on the adapter name
    kwargs = {}
    if api_key:
        kwargs['api_key'] = api_key
    
    # Determine the source type based on the adapter name
    if adapter_name in ["alpha_vantage", "yahoo_finance", "financial_data"]:
        source_type = DataSourceType.MARKET_DATA_PROVIDER
    elif adapter_name == "zerodha":
        source_type = DataSourceType.BROKER_API
    elif adapter_name == "polygon":
        source_type = DataSourceType.REST_API
    else:
        raise ValueError(f"Unknown adapter name: {adapter_name}")
    
    # Create the adapter
    adapter = factory.create_adapter(
        source_type=source_type,
        adapter_name=adapter_name,
        **kwargs
    )
    
    # Create historical data fetcher
    fetcher = HistoricalDataFetcher(
        source_type=source_type,
        adapter=adapter,
        cache_enabled=True,
        config=config
    )
    
    # Connect to the data source
    connected = fetcher.connect()
    if not connected:
        logger.error(f"Failed to connect to {adapter_name}")
        return None
    
    # Set date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {symbol} using {adapter_name}")
    data = fetcher.fetch_data(
        symbol=symbol,
        timeframe=DataTimeframe.ONE_DAY,
        start_date=start_date,
        end_date=end_date
    )
    
    # Plot the data
    if data is not None and not data.empty:
        logger.info(f"Successfully fetched {len(data)} data points")
        plot_data(data, symbol, adapter_name)
        return data
    else:
        logger.error(f"Failed to fetch data for {symbol} using {adapter_name}")
        return None


def plot_data(data, symbol, adapter_name):
    """Plot the historical data.
    
    Args:
        data: The historical data as a pandas DataFrame.
        symbol: The stock symbol.
        adapter_name: The name of the adapter used to fetch the data.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot closing prices
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.title(f"{symbol} Close Price - Data from {adapter_name}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    # Plot volume
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['volume'], label='Volume')
    plt.title(f"{symbol} Trading Volume - Data from {adapter_name}")
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_{adapter_name}_historical_data.png")
    plt.close()
    
    logger.info(f"Saved plot to {symbol}_{adapter_name}_historical_data.png")


def compare_data_sources(symbol, api_keys=None):
    """Compare data from different sources for the same symbol.
    
    Args:
        symbol: The stock symbol to fetch data for.
        api_keys: Dictionary of API keys for different data providers.
    """
    if api_keys is None:
        api_keys = {}
    
    # Fetch data from different sources
    data_sources = {}
    
    # Yahoo Finance (doesn't require API key)
    yahoo_data = fetch_and_plot_data(symbol, "yahoo_finance")
    if yahoo_data is not None:
        data_sources["yahoo_finance"] = yahoo_data
    
    # Alpha Vantage (requires API key)
    if "alpha_vantage" in api_keys:
        alpha_vantage_data = fetch_and_plot_data(
            symbol, "alpha_vantage", api_keys["alpha_vantage"]
        )
        if alpha_vantage_data is not None:
            data_sources["alpha_vantage"] = alpha_vantage_data
    
    # Polygon (requires API key)
    if "polygon" in api_keys:
        polygon_data = fetch_and_plot_data(
            symbol, "polygon", api_keys["polygon"]
        )
        if polygon_data is not None:
            data_sources["polygon"] = polygon_data
    
    # Compare the data if we have multiple sources
    if len(data_sources) > 1:
        compare_and_plot_data_sources(symbol, data_sources)


def compare_and_plot_data_sources(symbol, data_sources):
    """Compare and plot data from different sources.
    
    Args:
        symbol: The stock symbol.
        data_sources: Dictionary of data sources with their data.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot closing prices from different sources
    for source_name, data in data_sources.items():
        plt.plot(data.index, data['close'], label=f"{source_name} Close Price")
    
    plt.title(f"{symbol} Close Price Comparison")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_comparison.png")
    plt.close()
    
    logger.info(f"Saved comparison plot to {symbol}_comparison.png")
    
    # Calculate and print statistics
    print(f"\nStatistics for {symbol}:")
    for source_name, data in data_sources.items():
        print(f"\n{source_name}:")
        print(f"  Average Close: ${data['close'].mean():.2f}")
        print(f"  Min Close: ${data['close'].min():.2f}")
        print(f"  Max Close: ${data['close'].max():.2f}")
        print(f"  Close Range: ${data['close'].max() - data['close'].min():.2f}")
        print(f"  Average Volume: {data['volume'].mean():.0f}")


def demonstrate_batch_fetching():
    """Demonstrate batch fetching of historical data for multiple symbols."""
    # Create configuration and event system
    config = ConfigManager()
    event_system = EventSystem()
    
    # Create data adapter factory
    factory = DataAdapterFactory(config, event_system)
    
    # Create Yahoo Finance adapter (doesn't require API key)
    adapter = factory.create_adapter(
        source_type=DataSourceType.MARKET_DATA_PROVIDER,
        adapter_name="yahoo_finance"
    )
    
    # Create historical data fetcher
    fetcher = HistoricalDataFetcher(
        source_type=DataSourceType.MARKET_DATA_PROVIDER,
        adapter=adapter,
        cache_enabled=True,
        config=config
    )
    
    # Connect to the data source
    connected = fetcher.connect()
    if not connected:
        logger.error("Failed to connect to Yahoo Finance")
        return
    
    # Set date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    # List of symbols to fetch
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Fetch historical data for multiple symbols
    logger.info(f"Fetching historical data for {len(symbols)} symbols")
    data_dict = fetcher.fetch_multiple_symbols(
        symbols=symbols,
        timeframe=DataTimeframe.ONE_DAY,
        start_date=start_date,
        end_date=end_date
    )
    
    # Plot the data for each symbol
    for symbol, data in data_dict.items():
        if data is not None and not data.empty:
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            plot_data(data, symbol, "yahoo_finance")
        else:
            logger.error(f"Failed to fetch data for {symbol}")


def main():
    """Main function to demonstrate the usage of DataAdapterFactory with HistoricalDataFetcher."""
    print("\n=== Demonstrating DataAdapterFactory with HistoricalDataFetcher ===\n")
    
    # Example 1: Fetch and plot data for a single symbol using Yahoo Finance
    print("\nExample 1: Fetching data for AAPL using Yahoo Finance\n")
    fetch_and_plot_data("AAPL", "yahoo_finance")
    
    # Example 2: Compare data from different sources
    # Note: You need to provide your own API keys for Alpha Vantage and Polygon
    print("\nExample 2: Comparing data from different sources\n")
    api_keys = {
        "alpha_vantage": "YOUR_ALPHA_VANTAGE_API_KEY",  # Replace with your API key
        "polygon": "YOUR_POLYGON_API_KEY"  # Replace with your API key
    }
    # Uncomment the following line and replace the placeholder API keys with your own
    # compare_data_sources("AAPL", api_keys)
    
    # Example 3: Demonstrate batch fetching for multiple symbols
    print("\nExample 3: Demonstrating batch fetching for multiple symbols\n")
    demonstrate_batch_fetching()


if __name__ == "__main__":
    main()