# =========================================================
# Unified Configuration file for Friday AI Trading System
# =========================================================
# This file combines all configuration settings required for the trading system
# Last updated: May 24, 2025

# =========================
# BROKER CONFIGURATION
# =========================

# Zerodha API Configuration
ZERODHA_CONFIG = {
    'api_key': '',  # Your Zerodha API Key (to be filled by user)
    'api_secret': '',  # Your Zerodha API Secret (to be filled by user)
    'user_id': '',  # Your Zerodha User ID (to be filled by user)
    'password': '',  # Your Zerodha Password (to be filled by user)
    'totp_key': '',  # Your TOTP Key for 2FA (to be filled by user)
    'redirect_url': 'http://localhost:8080'  # Default redirect URL
}

# =========================
# MCP SERVER CONFIGURATION
# =========================

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
        "endpoints": {
            "get": "/memory",
            "update": "/memory/update",
            "search": "/memory/search",
            "delete": "/memory/delete"
        },
        "persistence": {
            "enabled": True,
            "storage_path": "./storage/memory",
            "backup_interval": 3600  # Seconds
        }
    },

    # Sequential Thinking MCP server configuration
    "sequential_thinking": {
        "enabled": True,
        "host": "localhost",
        "port": 8082,
        "endpoints": {
            "think": "/think",
            "status": "/status",
            "history": "/history"
        },
        "parameters": {
            "max_steps": 10,
            "timeout": 300,  # Seconds
            "detailed_output": True
        }
    },

    # Common settings
    "common": {
        "log_level": "INFO",
        "request_timeout": 60,  # Seconds
        "auto_restart": True
    }
}

# =========================
# KNOWLEDGE EXTRACTION
# =========================

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

# =========================
# MONGODB CONFIGURATION
# =========================

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
    "collections": {
        "market_data": "market_data",
        "trades": "trades",
        "orders": "orders",
        "strategies": "strategies",
        "backtest_results": "backtest_results",
        "user_preferences": "user_preferences",
        "system_logs": "system_logs",
        # Alternative data collections
        "news_sentiment": "news_sentiment",
        "social_media_data": "social_media_data",
        "economic_indicators": "economic_indicators",
        "alternative_data_normalized": "alternative_data_normalized",
        "error_reports": "error_reports",
        "alternative_data_cache": "alternative_data_cache"
    }
}

# =========================
# DATABASE CONFIGURATION
# =========================

DATABASE_CONFIG = {
    'connection_string': 'sqlite:///data/friday.db',  # Database connection string
    'echo': False,                                     # Echo SQL commands
    'pool_size': 10,                                   # Connection pool size
    'max_overflow': 20,                                # Maximum overflow connections
    'pool_timeout': 30,                                # Connection pool timeout in seconds
    'pool_recycle': 3600,                              # Connection recycle time in seconds
    'backup_enabled': True,                            # Enable database backups
    'backup_interval_hours': 24,                       # Backup interval in hours
    'backup_path': 'data/backups',                     # Path to store backups
    'max_backups': 7,                                  # Maximum number of backups to keep
    'migration_path': 'src/data/migrations'            # Path to database migrations
}

# =========================
# CACHE CONFIGURATION
# =========================

CACHE_CONFIG = {
    'enabled': True,                # Enable caching
    'redis_host': 'localhost',      # Redis host
    'redis_port': 6379,             # Redis port
    'redis_db': 0,                  # Redis database number
    'redis_password': None,         # Redis password
    'default_ttl': 3600,            # Default TTL in seconds
    'prediction_ttl': 300,          # Prediction TTL in seconds
    'instrument_ttl': 86400,        # Instrument info TTL in seconds
    'market_data_ttl': 60,          # Market data TTL in seconds
    'config_ttl': None,             # Config TTL in seconds (None = no expiry)
    'compression_enabled': True,    # Enable data compression
    'compression_level': 6,         # Compression level (1-9)
    'max_memory_mb': 512,           # Maximum Redis memory in MB
    'eviction_policy': 'allkeys-lru'  # Redis eviction policy
}

# =========================
# FEATURE ENGINEERING CONFIGURATION
# =========================

FEATURES_CONFIG = {
    # Default enabled feature sets
    'default_enabled': [
        'price_derived',     # Basic price-derived features
        'moving_averages',   # Simple and exponential moving averages
        'volatility',        # Volatility-based indicators
        'momentum',          # Momentum-based indicators
        'volume',            # Volume-based indicators
        'trend'              # Trend-based indicators
    ],
    
    # Feature set configurations
    'feature_sets': {
        'price_derived': {
            'enabled': True,
            'description': 'Basic price-derived features like typical price, price changes',
            'computational_complexity': 'low'
        },
        'moving_averages': {
            'enabled': True,
            'description': 'Simple and exponential moving averages',
            'computational_complexity': 'medium'
        },
        'volatility': {
            'enabled': True,
            'description': 'Volatility-based indicators like ATR, Bollinger Bands',
            'computational_complexity': 'medium-high'
        },
        'momentum': {
            'enabled': True,
            'description': 'Momentum indicators like RSI, MACD, Stochastic',
            'computational_complexity': 'medium'
        },
        'volume': {
            'enabled': True,
            'description': 'Volume-based indicators like OBV, VWAP',
            'computational_complexity': 'medium'
        },
        'trend': {
            'enabled': True,
            'description': 'Trend indicators like ADX, Aroon, CCI',
            'computational_complexity': 'high'
        }
    },
    
    # Performance and benchmarking settings
    'benchmarking': {
        'enabled': True,
        'default_dataset_sizes': {
            '1month': 43200,    # 30 days * 24 hours * 60 minutes
            '1year': 525600     # 365 days * 24 hours * 60 minutes
        },
        'memory_monitoring': True,
        'performance_logging': True,
        'save_benchmark_results': True,
        'benchmark_output_dir': 'benchmarks/features'
    },
    
    # Validation settings
    'validation': {
        'require_ohlcv': True,
        'allow_missing_volume': False,
        'validate_on_generation': True,
        'strict_column_validation': True
    },
    
    # Processing settings
    'processing': {
        'batch_size': 10000,
        'parallel_processing': False,  # Enable when thread-safe
        'cache_intermediate_results': True,
        'memory_optimization': True
    }
}

# =========================
# VALIDATION CONFIGURATION
# =========================

VALIDATION_CONFIG = {
    # General validation settings
    'price_outlier_threshold': 3.0,   # Z-score threshold for price outliers
    'feature_outlier_threshold': 5.0, # Z-score threshold for feature outliers
    'max_nan_percentage': 0.05,       # Maximum percentage of NaN values
    'max_outlier_percentage': 0.01,   # Maximum percentage of outliers
    'min_symbol_length': 1,           # Minimum symbol length
    'max_symbol_length': 20,          # Maximum symbol length
    'max_validation_errors': 100,     # Maximum validation errors to report
    'validate_on_save': True,         # Validate data before saving
    
    # Market data validation rules parameters
    'market_data': {
        # Timestamp validation
        'max_timestamp_gap_minutes': 5,      # Maximum allowed gap between consecutive timestamps
        'require_monotonic_timestamps': True, # Require timestamps to be monotonic increasing
        'allow_duplicate_timestamps': False, # Allow duplicate timestamps
        
        # Price validation
        'min_price_bound': 0.01,            # Minimum allowed price value
        'max_price_bound': 100000.0,        # Maximum allowed price value
        'allow_negative_prices': False,     # Allow negative prices
        'allow_zero_prices': False,         # Allow zero prices
        
        # Volume validation
        'allow_negative_volume': False,     # Allow negative volume
        'allow_zero_volume': True,          # Allow zero volume (common in some markets)
        'min_volume_bound': 0.0,            # Minimum allowed volume
        'max_volume_bound': 1e12,           # Maximum allowed volume
        
        # Data type validation
        'strict_type_validation': False,    # Require exact float types vs. numeric convertible
        'required_columns': ['open', 'high', 'low', 'close', 'volume'],  # Required OHLCV columns
        
        # Trading hours validation
        'enforce_trading_hours': False,     # Enable trading hours validation
        'trading_hours_start': '09:30',     # Trading hours start time (HH:MM format)
        'trading_hours_end': '16:00',       # Trading hours end time (HH:MM format)
        'timezone': 'America/New_York',     # Timezone for trading hours
        
        # Symbol whitelist validation
        'enable_symbol_whitelist': False,   # Enable symbol whitelist validation
        'symbol_whitelist': [               # List of allowed symbols (empty = allow all)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'INTC', 'CSCO', 'PEP', 'AVGO', 'TXN', 'QCOM'
        ],
        
        # Warn-only mode for real-time streams
        'warn_only_mode': False,            # Enable warn-only mode (log warnings but don't fail)
        'warn_only_for_realtime': True,     # Auto-enable warn-only for real-time data
        
        # Performance settings
        'enable_detailed_logging': True,    # Enable detailed validation logging
        'max_validation_time_seconds': 30,  # Maximum time allowed for validation
        'batch_validation_size': 10000,     # Batch size for large dataset validation
    },
    
    # Legacy validation settings
    'strict_validation': False,       # Strict validation mode
    'schema_path': 'data/schemas'     # Path to validation schemas
}

# =========================
# API CONFIGURATION
# =========================

API_CONFIG = {
    'host': '127.0.0.1',              # API host
    'port': 5000,                      # API port
    'debug': False,                    # Debug mode
    'threaded': True,                  # Run in threaded mode
    'cors_enabled': True,              # Enable CORS
    'cors_origins': ['*'],             # CORS allowed origins
    'rate_limit_enabled': True,        # Enable rate limiting
    'rate_limit_window': 60,           # Rate limit window in seconds
    'rate_limit_max_requests': 100,    # Maximum requests per window
    'jwt_secret_key': '',              # JWT secret key (to be filled by user)
    'jwt_expiry_minutes': 60,          # JWT expiry in minutes
    'max_content_length': 10 * 1024 * 1024,  # Maximum content length in bytes
    'api_docs_enabled': True,          # Enable API docs
    'api_docs_path': '/docs',          # API docs path
    'log_requests': True,              # Log API requests
    'request_log_file': 'logs/api/requests.log'  # Request log file
}

# =========================
# NOTIFICATION CONFIGURATION
# =========================

NOTIFICATION_CONFIG = {
    # Email notifications
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',  # To be filled by user
        'sender_password': '',  # To be filled by user
        'default_recipients': [],  # To be filled by user
        'use_tls': True,
        'email_template_path': 'templates/email',
        'notification_types': ['trade_alert', 'system_alert', 'error_alert']
    },

    # Telegram notifications
    'telegram': {
        'enabled': False,
        'bot_token': '',  # To be filled by user
        'default_chat_ids': [],  # To be filled by user
        'notification_types': ['trade_alert', 'system_alert']
    },

    # Mobile app notifications
    'mobile_app': {
        'enabled': False,
        'api_key': '',  # To be filled by user
        'api_url': 'https://example.com/push',  # To be replaced with actual API URL
        'default_user_ids': [],  # To be filled by user
        'notification_types': ['trade_alert', 'system_alert', 'error_alert']
    }
}

# =========================
# SYSTEM CONFIGURATION
# =========================

SYSTEM_CONFIG = {
    'app_name': 'Friday AI Trading System',
    'version': '1.0.0',
    'auto_start': False,           # Whether to auto-start the system
    'auto_connect_broker': False,  # Whether to auto-connect to broker
    'development_mode': True,      # Development mode
    'max_threads': 10,             # Maximum worker threads
    'memory_limit_mb': 2048,       # Memory limit in MB
    'log_system_metrics': True,    # Whether to log system metrics
    'metrics_interval': 60,        # Interval to log metrics in seconds
    'backup_enabled': True,        # Enable system backups
    'backup_interval_days': 7,     # Backup interval in days
    'backup_path': 'backups',      # Path to store backups
    'max_backups': 5,              # Maximum number of backups to keep
    'timezone': 'Asia/Kolkata',    # System timezone
    'enable_crash_reporting': True,  # Enable crash reporting
    'crash_report_path': 'logs/crashes',  # Path to store crash reports
    'update_check_enabled': True,  # Enable update checking
    'update_check_interval_days': 7  # Update check interval in days
}

# =========================
# DATA PROCESSING CONFIGURATION
# =========================

# Data processing and cleaning configuration
DATA_CONFIG = {
    # Data cleaning configuration
    'cleaning': {
        # Column specifications
        'symbol_column': 'symbol',
        'timestamp_column': 'timestamp',
        'numeric_columns': ['open', 'high', 'low', 'close', 'volume', 'price'],
        
        # Outlier detection thresholds
        'z_score_threshold': 3.0,           # Z-score threshold for outlier detection
        'iqr_multiplier': 1.5,              # IQR multiplier for outlier detection
        'outlier_method': 'iqr',            # Default outlier detection method ('zscore' or 'iqr')
        'max_outlier_percentage': 0.05,     # Maximum acceptable outlier percentage (5%)
        
        # Numeric cast validation
        'min_numeric_cast_success_rate': 0.95,  # Minimum success rate for numeric conversions (95%)
        
        # Gap filling configuration
        'gap_fill_max_consecutive': 5,      # Maximum consecutive gaps to warn about
        'enable_forward_fill': True,        # Enable forward-fill for gaps
        'enable_back_fill': True,           # Enable back-fill for remaining gaps
        
        # Logging and reporting
        'enable_detailed_logging': True,    # Enable detailed cleaning operation logging
        'store_cleaning_metrics': True,     # Store detailed cleaning metrics
        'emit_events': True,                # Emit cleaning events to EventSystem
        
        # Data quality thresholds
        'min_data_quality_score': 0.7,     # Minimum acceptable data quality score
        'max_processing_time_seconds': 300, # Maximum processing time per cleaning operation
    },
    
    # Data validation configuration (enhanced from existing VALIDATION_CONFIG)
    'validation': {
        'price_outlier_threshold': 3.0,
        'feature_outlier_threshold': 5.0,
        'max_nan_percentage': 0.05,
        'max_outlier_percentage': 0.01,
        'min_symbol_length': 1,
        'max_symbol_length': 20,
        'max_validation_errors': 100,
        'validate_on_save': True,
        'strict_validation': False,
        'schema_path': 'data/schemas'
    },
    
    # Data storage configuration
    'storage': {
        # Default storage backend
        'default_backend': 'local_parquet',  # Options: 'local_parquet', 'mongodb', 'postgresql'
        'default_format': 'parquet',
        'compression': 'snappy',
        'backup_enabled': True,
        'backup_retention_days': 30,
        
        # LocalParquetStorage configuration
        'local_parquet': {
            'base_dir': 'data/market_data',
            'partition_by': ['symbol', 'date'],
            'compression': 'snappy',
            'metadata_enabled': True,
            'file_rotation': {
                'enabled': True,
                'max_file_size_mb': 100,
                'max_files_per_partition': 10
            },
            'auto_create_dirs': True,
            'validate_partitioning': True
        },
        
        # MongoDB storage configuration
        'mongodb': {
            'connection_string': 'mongodb://localhost:27017',
            'database_name': 'friday_trading',
            'collections': {
                'market_data': 'market_data',
                'processed_data': 'processed_data',
                'features': 'features',
                'model_training_data': 'model_training_data',
                'metadata': 'storage_metadata'
            },
            'indexes': {
                'market_data': [
                    [('symbol', 1), ('timestamp', 1)],
                    [('symbol', 1), ('date', 1)]
                ]
            },
            'chunk_size': 1000,
            'connection_timeout': 5000
        },
        
        # PostgreSQL storage configuration
        'postgresql': {
            'connection_string': 'postgresql://localhost:5432/friday_trading',
            'schema': 'market_data',
            'tables': {
                'market_data': 'market_data',
                'processed_data': 'processed_data',
                'features': 'features',
                'model_training_data': 'model_training_data',
                'metadata': 'storage_metadata'
            },
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'auto_create_tables': True
        },
        
        # Retrieval utilities configuration
        'retrieval': {
            'batch_size': 10000,
            'parallel_loading': True,
            'max_workers': 4,
            'cache_enabled': True,
            'cache_ttl': 3600,  # seconds
            'prefetch_enabled': True,
            'optimize_for_ml': True
        },
        
        # Automatic directory creation
        'auto_directory_creation': {
            'enabled': True,
            'permissions': '755',
            'create_parents': True
        },
        
        # File rotation settings
        'file_rotation': {
            'enabled': True,
            'strategy': 'size_based',  # 'size_based', 'time_based', 'count_based'
            'max_file_size_mb': 100,
            'max_age_days': 30,
            'max_files': 1000,
            'compress_old_files': True
        },
        
        # Metadata logging
        'metadata_logging': {
            'enabled': True,
            'log_operations': True,
            'log_performance': True,
            'log_errors': True,
            'metadata_retention_days': 90,
            'detailed_logging': False
        }
    }
}

# =========================
# ALTERNATIVE DATA CONFIGURATION
# =========================

# Configuration for alternative data integration
ALTERNATIVE_DATA_CONFIG = {
    # News sentiment analysis configuration
    'news_sentiment': {
        'enabled': True,
        'api_key': '',  # To be filled by user
        'api_url': 'https://newsapi.org/v2/everything',
        'update_frequency': 3600,  # Seconds
        'sources': ['bloomberg', 'reuters', 'cnbc', 'financial-times'],
        'max_articles_per_request': 100,
        'sentiment_model': 'vader',  # Options: 'vader', 'textblob', 'custom'
        'custom_model_path': 'models/sentiment/financial_sentiment.pkl',
        'cache_expiry': 86400,  # Seconds (24 hours)
        'mongodb_collection': 'news_sentiment'
    },
    
    # Social media analysis configuration
    'social_media': {
        'enabled': True,
        'platforms': {
            'twitter': {
                'enabled': True,
                'api_key': '',  # To be filled by user
                'api_secret': '',  # To be filled by user
                'access_token': '',  # To be filled by user
                'access_token_secret': '',  # To be filled by user
                'update_frequency': 1800,  # Seconds
                'search_terms': ['#stocks', '#trading', '#investing'],
                'influencers': ['@jimcramer', '@thestreet', '@cnbc'],
                'max_tweets_per_request': 100
            },
            'reddit': {
                'enabled': True,
                'client_id': '',  # To be filled by user
                'client_secret': '',  # To be filled by user
                'user_agent': 'Friday AI Trading System',
                'update_frequency': 3600,  # Seconds
                'subreddits': ['wallstreetbets', 'investing', 'stocks'],
                'max_posts_per_request': 50,
                'time_filter': 'day'  # Options: 'hour', 'day', 'week', 'month', 'year', 'all'
            }
        },
        'sentiment_analysis': {
            'enabled': True,
            'model': 'vader',  # Options: 'vader', 'textblob', 'custom'
            'custom_model_path': 'models/sentiment/social_sentiment.pkl'
        },
        'cache_expiry': 43200,  # Seconds (12 hours)
        'mongodb_collection': 'social_media_data'
    },
    
    # Economic data configuration
    'economic_data': {
        'enabled': True,
        'sources': {
            'fred': {
                'enabled': True,
                'api_key': '',  # To be filled by user
                'update_frequency': 86400,  # Seconds (24 hours)
                'indicators': [
                    'GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'T10Y2Y',
                    'INDPRO', 'HOUST', 'PCE', 'M2', 'RETAILSALES'
                ]
            },
            'world_bank': {
                'enabled': True,
                'update_frequency': 604800,  # Seconds (7 days)
                'indicators': [
                    'NY.GDP.MKTP.CD', 'FP.CPI.TOTL.ZG', 'FR.INR.RINR',
                    'NE.TRD.GNFS.ZS', 'BN.CAB.XOKA.GD.ZS'
                ],
                'countries': ['US', 'CN', 'IN', 'JP', 'DE', 'GB']
            }
        },
        'cache_expiry': 86400,  # Seconds (24 hours)
        'mongodb_collection': 'economic_indicators'
    },
    
    # Data normalization configuration
    'normalization': {
        'enabled': True,
        'methods': {
            'numerical': 'min_max',  # Options: 'min_max', 'z_score', 'robust'
            'categorical': 'one_hot',  # Options: 'one_hot', 'label', 'binary'
            'text': 'tfidf'  # Options: 'tfidf', 'count', 'word2vec', 'bert'
        },
        'output_format': 'dataframe',
        'mongodb_collection': 'alternative_data_normalized'
    },
    
    # Error handling configuration
    'error_handling': {
        'retry_attempts': 3,
        'retry_delay': 5,  # Seconds
        'fallback_to_cache': True,
        'cache_expiry': 604800,  # Seconds (7 days)
        'log_errors': True,
        'error_report_collection': 'error_reports',
        'cache_collection': 'alternative_data_cache'
    }
}

# Custom functions
def is_expiry_day(date):
    """Custom function to determine if a date is an expiry day"""
    # Default logic: Thursday is expiry day
    is_thursday = date.weekday() == 3

    # Special handling for holidays (to be expanded with a holiday calendar)
    # Example: If Thursday is a holiday, Wednesday is expiry day
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
    # This will be expanded with detailed validation in Phase 2
    valid = True
    return valid
