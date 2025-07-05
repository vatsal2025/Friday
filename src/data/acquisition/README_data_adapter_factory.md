# Data Adapter Factory

The `DataAdapterFactory` is a factory class that simplifies the creation of data source adapters for the Friday AI Trading System. It provides methods for registering adapter classes and creating adapter instances.

## Overview

The `DataAdapterFactory` class is designed to centralize the creation of data source adapters, making it easier to switch between different data sources without changing the client code. It supports various data sources, including market data providers, broker APIs, and REST APIs.

## Features

- Register adapter classes for different data source types
- Create adapter instances with appropriate configuration
- Built-in support for common data adapters:
  - Alpha Vantage
  - Yahoo Finance
  - Zerodha
  - Polygon
  - Financial Data

## Usage

### Creating a Data Adapter Factory

```python
from src.data.acquisition.data_adapter_factory import DataAdapterFactory
from src.infrastructure.config import ConfigManager
from src.infrastructure.event import EventSystem

# Create configuration and event system
config = ConfigManager()
event_system = EventSystem()

# Create data adapter factory
factory = DataAdapterFactory(config, event_system)
```

### Creating a Data Adapter

```python
from src.data.acquisition.data_fetcher import DataSourceType

# Create a Yahoo Finance adapter
yahoo_adapter = factory.create_adapter(
    source_type=DataSourceType.MARKET_DATA_PROVIDER,
    adapter_name="yahoo_finance"
)

# Create an Alpha Vantage adapter with API key
alpha_vantage_adapter = factory.create_adapter(
    source_type=DataSourceType.MARKET_DATA_PROVIDER,
    adapter_name="alpha_vantage",
    api_key="YOUR_API_KEY"
)

# Create a Zerodha adapter with API key and secret
zerodha_adapter = factory.create_adapter(
    source_type=DataSourceType.BROKER_API,
    adapter_name="zerodha",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET"
)
```

### Using with HistoricalDataFetcher

```python
from datetime import datetime, timedelta
from src.data.acquisition.data_fetcher import DataTimeframe
from src.data.acquisition.historical_data_fetcher import HistoricalDataFetcher

# Create a Yahoo Finance adapter
adapter = factory.create_adapter(
    source_type=DataSourceType.MARKET_DATA_PROVIDER,
    adapter_name="yahoo_finance"
)

# Create a historical data fetcher
fetcher = HistoricalDataFetcher(
    source_type=DataSourceType.MARKET_DATA_PROVIDER,
    adapter=adapter,
    cache_enabled=True,
    config=config
)

# Connect to the data source
fetcher.connect()

# Set date range for historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Last 30 days

# Fetch historical data
data = fetcher.fetch_data(
    symbol="AAPL",
    timeframe=DataTimeframe.ONE_DAY,
    start_date=start_date,
    end_date=end_date
)
```

### Registering a Custom Adapter

```python
from src.data.acquisition.data_fetcher import DataSourceAdapter, DataSourceType

# Define a custom adapter class
class CustomAdapter(DataSourceAdapter):
    # Implement the required methods
    ...

# Register the custom adapter
DataAdapterFactory.register_adapter(
    source_type=DataSourceType.CUSTOM,
    adapter_class=CustomAdapter,
    name="custom_adapter"
)

# Create an instance of the custom adapter
custom_adapter = factory.create_adapter(
    source_type=DataSourceType.CUSTOM,
    adapter_name="custom_adapter"
)
```

## Built-in Adapters

The `DataAdapterFactory` comes with several built-in adapters:

### Alpha Vantage Adapter

- **Source Type**: `DataSourceType.MARKET_DATA_PROVIDER`
- **Name**: `"alpha_vantage"`
- **Required Parameters**: `api_key`

### Yahoo Finance Adapter

- **Source Type**: `DataSourceType.MARKET_DATA_PROVIDER`
- **Name**: `"yahoo_finance"`
- **Required Parameters**: None

### Zerodha Adapter

- **Source Type**: `DataSourceType.BROKER_API`
- **Name**: `"zerodha"`
- **Required Parameters**: `api_key`, `api_secret`

### Polygon Adapter

- **Source Type**: `DataSourceType.REST_API`
- **Name**: `"polygon"`
- **Required Parameters**: `api_key`

### Financial Data Adapter

- **Source Type**: `DataSourceType.MARKET_DATA_PROVIDER`
- **Name**: `"financial_data"`
- **Required Parameters**: `api_key`

## Examples

For more examples, see the following files:

- `examples/data_adapter_factory_example.py`: Demonstrates how to use the `DataAdapterFactory` with `HistoricalDataFetcher`.
- `tests/data/acquisition/test_data_adapter_factory.py`: Contains tests for the `DataAdapterFactory` class.