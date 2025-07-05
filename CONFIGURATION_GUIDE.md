# Friday AI Trading System - Configuration Guide

## Overview

This document provides a comprehensive guide to configuring the Friday AI Trading System. The system uses a unified configuration approach, with all settings centralized in the `unified_config.py` file.

## Table of Contents

1. [Configuration Structure](#configuration-structure)
2. [Broker Configuration](#broker-configuration)
3. [MCP Server Configuration](#mcp-server-configuration)
4. [Knowledge Extraction Configuration](#knowledge-extraction-configuration)
5. [Database Configuration](#database-configuration)
6. [Cache Configuration](#cache-configuration)
7. [Validation Configuration](#validation-configuration)
8. [Alternative Data Configuration](#alternative-data-configuration)
9. [Custom Functions](#custom-functions)
10. [Environment Variables](#environment-variables)
11. [Configuration Best Practices](#configuration-best-practices)

## Configuration Structure

The `unified_config.py` file is organized into sections, each containing related configuration settings. The main sections include:

- Broker Configuration
- MCP Server Configuration
- Knowledge Extraction Configuration
- MongoDB Configuration
- Database Configuration
- Cache Configuration
- Validation Configuration
- Alternative Data Configuration

## Broker Configuration

The broker configuration section contains settings for connecting to trading brokers. Currently, the system supports Zerodha:

```python
# Zerodha API Configuration
ZERODHA_CONFIG = {
    'api_key': '',  # Your Zerodha API Key
    'api_secret': '',  # Your Zerodha API Secret
    'user_id': '',  # Your Zerodha User ID
    'password': '',  # Your Zerodha Password
    'totp_key': '',  # Your TOTP Key for 2FA
    'redirect_url': 'http://localhost:8080'  # Default redirect URL
}
```

### Required Settings

To use the Zerodha broker, you must provide:

- `api_key`: Your Zerodha API key
- `api_secret`: Your Zerodha API secret
- `user_id`: Your Zerodha user ID
- `password`: Your Zerodha password
- `totp_key`: Your TOTP key for two-factor authentication

## MCP Server Configuration

The MCP (Model Context Protocol) server configuration contains settings for the Memory and Sequential Thinking servers:

```python
# Redis Configuration
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True,
    'decode_responses': True
}

# Model Context Protocol server configurations
MCP_CONFIG = {
    # Memory MCP server configuration
    "memory": {
        "enabled": True,
        "host": "localhost",
        "port": 8081,
        "endpoints": { ... },
        "persistence": { ... }
    },

    # Sequential Thinking MCP server configuration
    "sequential_thinking": {
        "enabled": True,
        "host": "localhost",
        "port": 8082,
        "endpoints": { ... },
        "parameters": { ... }
    },

    # Common settings
    "common": {
        "log_level": "INFO",
        "request_timeout": 60,  # Seconds
        "auto_restart": True
    }
}
```

### Key MCP Settings

- **Memory Server**:
  - `enabled`: Enable/disable the Memory MCP server
  - `host`: Host address for the Memory server
  - `port`: Port for the Memory server
  - `persistence`: Settings for persisting memory data

- **Sequential Thinking Server**:
  - `enabled`: Enable/disable the Sequential Thinking MCP server
  - `host`: Host address for the Sequential Thinking server
  - `port`: Port for the Sequential Thinking server
  - `parameters`: Parameters for the Sequential Thinking process

## Knowledge Extraction Configuration

The knowledge extraction configuration contains settings for extracting trading knowledge:

```python
# Knowledge extraction configuration
KNOWLEDGE_CONFIG = {
    # Models to use for extraction
    "extraction": {
        "default_model": "gpt-4",
        "fast_model": "gpt-3.5-turbo",
        "embeddings_model": "text-embedding-ada-002"
    },

    # Categories of knowledge to extract
    "categories": {
        "trading_rules": True,
        "patterns": True,
        "psychological_insights": True,
        "strategies": True
    },

    # Storage configuration
    "storage": {
        "format": "json",
        "compress": False
    },

    # Processing parameters
    "processing": {
        "chunk_size": 8000,
        "chunk_overlap": 200,
    }
}
```

### Key Knowledge Extraction Settings

- **Extraction Models**:
  - `default_model`: The default model for knowledge extraction
  - `fast_model`: A faster model for less complex extraction tasks
  - `embeddings_model`: The model used for generating embeddings

- **Categories**:
  - `trading_rules`: Extract trading rules
  - `patterns`: Extract trading patterns
  - `psychological_insights`: Extract psychological insights
  - `strategies`: Extract trading strategies

## Database Configuration

The system uses both MongoDB and SQLite for data storage:

```python
# MongoDB configuration for the system
MONGODB_CONFIG = {
    "enabled": True,
    "host": "localhost",
    "port": 27017,
    "db_name": "friday",
    "username": "",  # Leave empty if no authentication
    "password": "",  # Leave empty if no authentication
    "auth_source": "admin",
    "connect_timeout_ms": 5000,
    "server_selection_timeout_ms": 5000,
    "collections": { ... }
}

# SQLite Database configuration
DATABASE_CONFIG = {
    'connection_string': 'sqlite:///data/friday.db',
    'echo': False,
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'backup_enabled': True,
    'backup_interval_hours': 24,
    'backup_path': 'data/backups',
    'max_backups': 7,
    'migration_path': 'src/data/migrations'
}
```

### Key Database Settings

- **MongoDB**:
  - `enabled`: Enable/disable MongoDB
  - `host`: MongoDB host address
  - `port`: MongoDB port
  - `db_name`: MongoDB database name
  - `collections`: MongoDB collection names

- **SQLite**:
  - `connection_string`: Database connection string
  - `backup_enabled`: Enable/disable database backups
  - `backup_interval_hours`: Backup interval in hours
  - `backup_path`: Path to store backups

## Cache Configuration

The cache configuration contains settings for Redis caching:

```python
CACHE_CONFIG = {
    'enabled': True,
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'redis_password': None,
    'default_ttl': 3600,
    'prediction_ttl': 300,
    'instrument_ttl': 86400,
    'market_data_ttl': 60,
    'config_ttl': None,
    'compression_enabled': True,
    'compression_level': 6,
    'max_memory_mb': 512,
    'eviction_policy': 'allkeys-lru'
}
```

### Key Cache Settings

- `enabled`: Enable/disable caching
- `redis_host`: Redis host address
- `redis_port`: Redis port
- `default_ttl`: Default time-to-live in seconds
- `compression_enabled`: Enable/disable data compression
- `max_memory_mb`: Maximum Redis memory in MB
- `eviction_policy`: Redis eviction policy

## Validation Configuration

The validation configuration contains settings for data validation:

```python
VALIDATION_CONFIG = {
    'price_outlier_threshold': 3.0,
    'feature_outlier_threshold': 5.0,
    'max_nan_percentage': 0.05,
    'max_outlier_percentage': 0.01,
    'min_symbol_length': 1,
    'max_symbol_length': 20,
    'max_validation_errors': 100,
    'validate_on_save': True
}
```

### Key Validation Settings

- `price_outlier_threshold`: Z-score threshold for price outliers
- `feature_outlier_threshold`: Z-score threshold for feature outliers
- `max_nan_percentage`: Maximum percentage of NaN values
- `max_outlier_percentage`: Maximum percentage of outliers
- `validate_on_save`: Validate data before saving

## Alternative Data Configuration

The alternative data configuration contains settings for news, social media, and economic data:

```python
ALTERNATIVE_DATA_CONFIG = {
    # News analysis configuration
    'news': {
        'enabled': True,
        'update_frequency': 3600,  # Seconds
        'sources': ['bloomberg', 'reuters', 'cnbc', 'financial-times'],
        'max_articles_per_request': 100,
        'sentiment_model': 'vader',
        'custom_model_path': 'models/sentiment/financial_sentiment.pkl',
        'cache_expiry': 86400,  # Seconds (24 hours)
        'mongodb_collection': 'news_sentiment'
    },
    
    # Social media analysis configuration
    'social_media': { ... },
    
    # Economic data configuration
    'economic_data': { ... },
    
    # Data normalization configuration
    'normalization': { ... },
    
    # Error handling configuration
    'error_handling': { ... }
}
```

### Key Alternative Data Settings

- **News**:
  - `enabled`: Enable/disable news analysis
  - `update_frequency`: Update frequency in seconds
  - `sources`: News sources to analyze
  - `sentiment_model`: Sentiment analysis model

- **Social Media**:
  - `enabled`: Enable/disable social media analysis
  - `platforms`: Social media platforms to analyze
  - `sentiment_analysis`: Sentiment analysis settings

- **Economic Data**:
  - `enabled`: Enable/disable economic data analysis
  - `sources`: Economic data sources
  - `indicators`: Economic indicators to track

## Custom Functions

The configuration file includes several custom functions:

```python
def is_expiry_day(date):
    """Custom function to determine if a date is an expiry day"""
    # Default logic: Thursday is expiry day
    is_thursday = date.weekday() == 3
    return is_thursday

def calculate_position_size(capital, risk_per_trade, stop_loss_percent):
    """Calculate position size based on risk parameters"""
    if not capital or not risk_per_trade or not stop_loss_percent:
        return 0

    risk_amount = float(capital) * float(risk_per_trade)
    position_size = risk_amount / float(stop_loss_percent)
    return position_size

def validate_config():
    """Validate the configuration settings."""
    # Basic validation logic
    valid = True
    return valid
```

## Environment Variables

Many configuration settings can be overridden using environment variables. This is particularly useful for sensitive information like API keys and passwords.

To use environment variables, create a `.env` file in the project root directory with the following format:

```
# MongoDB
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=
MONGODB_PASSWORD=

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Zerodha
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_USER_ID=your_user_id
ZERODHA_PASSWORD=your_password
ZERODHA_TOTP_KEY=your_totp_key
```

## Configuration Best Practices

1. **Security**: Never commit sensitive information like API keys and passwords to version control. Use environment variables or a `.env` file instead.

2. **Validation**: Always validate your configuration before using it. The `validate_config()` function can be extended to include detailed validation logic.

3. **Documentation**: Keep this configuration guide updated when adding new configuration settings.

4. **Defaults**: Provide sensible defaults for all configuration settings to ensure the system works out of the box.

5. **Environment-Specific Configuration**: Use environment variables to override configuration settings for different environments (development, testing, production).

6. **Backup**: Regularly backup your configuration file, especially if it contains custom settings.

7. **Version Control**: Keep track of configuration changes in version control, but exclude sensitive information.

---

For more information, refer to the `README.md` and `SETUP_INSTRUCTIONS.md` files in the project root directory.