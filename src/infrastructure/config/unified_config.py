"""Unified configuration for the Friday AI Trading System.

This module contains all the configuration settings for the application.
It is imported by the config module to provide a centralized configuration.
"""

from typing import Dict, Any

# Zerodha configuration
ZERODHA_CONFIG: Dict[str, Any] = {
    "api_key": "",  # Will be overridden by environment variable if set
    "api_secret": "",  # Will be overridden by environment variable if set
    "redirect_uri": "http://localhost:8080/zerodha/callback",
    "api_timeout": 10,  # seconds
    "retry_count": 3,
    "retry_delay": 1,  # seconds
    "default_exchange": "NSE",
    "default_product": "CNC",  # Cash and Carry
    "default_order_type": "MARKET",
    "default_validity": "DAY",
    "default_variety": "regular",
    "default_quantity": 1,
    "default_disclosed_quantity": 0,
    "default_trigger_price": 0,
    "default_squareoff": 0,
    "default_stoploss": 0,
    "default_trailing_stoploss": 0,
    "default_tag": "FRIDAY_AI",
}

# Broker configuration (alias for Zerodha)
BROKER_CONFIG: Dict[str, Any] = ZERODHA_CONFIG

# MCP (Multi-Chain Protocol) configuration
MCP_CONFIG: Dict[str, Any] = {
    "server_host": "localhost",
    "server_port": 8000,
    "client_timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1,  # seconds
    "log_level": "INFO",
    "servers": [
        {
            "name": "mcp.config.usrlocalmcp.Puppeteer",
            "enabled": True,
            "auto_start": False,
        },
        {
            "name": "mcp.config.usrlocalmcp.File System",
            "enabled": True,
            "auto_start": False,
        },
        {
            "name": "mcp.config.usrlocalmcp.Multi Fetch",
            "enabled": True,
            "auto_start": False,
        },
        {
            "name": "mcp.config.usrlocalmcp.Persistent Knowledge Graph",
            "enabled": True,
            "auto_start": False,
        },
        {
            "name": "mcp.config.usrlocalmcp.Excel",
            "enabled": True,
            "auto_start": False,
        },
        {
            "name": "mcp.config.usrlocalmcp.Docker",
            "enabled": True,
            "auto_start": False,
        },
    ],
}

# Knowledge Graph configuration
KNOWLEDGE_CONFIG: Dict[str, Any] = {
    "storage_path": "storage/knowledge",
    "backup_interval": 3600,  # seconds
    "max_backups": 10,
    "entity_types": [
        "Stock",
        "Sector",
        "Industry",
        "Company",
        "Event",
        "News",
        "Indicator",
        "Strategy",
        "Trade",
        "Portfolio",
        "Market",
        "Economy",
    ],
    "relation_types": [
        "belongs_to",
        "affects",
        "correlates_with",
        "causes",
        "precedes",
        "follows",
        "implements",
        "contains",
        "triggers",
        "signals",
    ],
}

# Database configuration
DATABASE_CONFIG: Dict[str, Any] = {
    "type": "sqlite",  # sqlite, mysql, postgresql
    "path": "storage/data/friday.db",  # For SQLite
    "host": "localhost",  # For MySQL/PostgreSQL
    "port": 3306,  # For MySQL/PostgreSQL
    "username": "",  # For MySQL/PostgreSQL
    "password": "",  # For MySQL/PostgreSQL
    "database": "friday",  # For MySQL/PostgreSQL
    "pool_size": 5,
    "max_overflow": 10,
    "timeout": 30,  # seconds
}

# API configuration
API_CONFIG: Dict[str, Any] = {
    "host": "localhost",
    "port": 8000,
    "debug": False,
    "reload": False,
    "workers": 4,
    "timeout": 60,  # seconds
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "api_prefix": "/api/v1",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json",
    "title": "Friday AI Trading System API",
    "description": "API for the Friday AI Trading System",
    "version": "0.1.0",
}

# Dashboard configuration
DASHBOARD_CONFIG: Dict[str, Any] = {
    "host": "localhost",
    "port": 8501,
    "theme": "light",  # light, dark
    "wide_mode": True,
    "sidebar_state": "expanded",  # expanded, collapsed
    "initial_sidebar_state": "expanded",  # expanded, collapsed, auto
    "page_title": "Friday AI Trading System",
    "page_icon": "📈",
    "layout": "wide",  # wide, centered
    "menu_items": {
        "Get Help": "https://github.com/yourusername/friday/issues",
        "Report a bug": "https://github.com/yourusername/friday/issues/new",
        "About": "# Friday AI Trading System\nAn AI-powered trading system.",
    },
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": "storage/logs/friday.log",
    "max_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
    "console": True,
}

# Trading configuration
TRADING_CONFIG: Dict[str, Any] = {
    "mode": "paper",  # paper, live
    "capital": 100000,  # Initial capital for paper trading
    "max_position_size": 0.1,  # Maximum position size as a fraction of capital
    "max_positions": 10,  # Maximum number of open positions
    "default_stop_loss": 0.05,  # Default stop loss as a fraction of entry price
    "default_take_profit": 0.1,  # Default take profit as a fraction of entry price
    "default_trailing_stop": 0.02,  # Default trailing stop as a fraction of entry price
    "risk_per_trade": 0.01,  # Risk per trade as a fraction of capital
    "max_risk_per_day": 0.05,  # Maximum risk per day as a fraction of capital
    "trading_hours": {
        "start": "09:15",  # Market open time (IST)
        "end": "15:30",  # Market close time (IST)
    },
    "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday (0 = Monday, 6 = Sunday)
    "holidays": [],  # List of holiday dates in YYYY-MM-DD format
}

# Backtesting configuration
BACKTESTING_CONFIG: Dict[str, Any] = {
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "commission": 0.0003,  # 0.03%
    "slippage": 0.0001,  # 0.01%
    "data_source": "yahoo",  # yahoo, alpha_vantage, zerodha
    "timeframe": "1d",  # 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M
    "benchmark": "^NSEI",  # NIFTY 50 index
    "metrics": [
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
    ],
}

# Data configuration
DATA_CONFIG: Dict[str, Any] = {
    "sources": {
        "yahoo": {
            "enabled": True,
            "rate_limit": 2000,  # requests per hour
        },
        "alpha_vantage": {
            "enabled": False,
            "api_key": "",
            "rate_limit": 500,  # requests per day (free tier)
        },
        "zerodha": {
            "enabled": True,
            "rate_limit": 3,  # requests per second
        },
    },
    "cache": {
        "enabled": True,
        "expiry": {
            "1m": 3600,  # 1 hour in seconds
            "5m": 3600 * 6,  # 6 hours in seconds
            "15m": 3600 * 12,  # 12 hours in seconds
            "30m": 3600 * 24,  # 24 hours in seconds
            "1h": 3600 * 24 * 3,  # 3 days in seconds
            "1d": 3600 * 24 * 30,  # 30 days in seconds
            "1w": 3600 * 24 * 90,  # 90 days in seconds
            "1M": 3600 * 24 * 180,  # 180 days in seconds
        },
    },
    "storage": {
        "type": "sqlite",  # sqlite, csv, parquet
        "path": "storage/data",
    },
}

# Redis configuration
REDIS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": "",  # Leave empty if no password
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": True,
    "max_connections": 10,
    "namespaces": {
        "market_data": "market:",
        "user_data": "user:",
        "system": "system:",
        "trading": "trading:",
    },
}

# MongoDB configuration
MONGODB_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "host": "localhost",
    "port": 27017,
    "database": "friday",
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
    },
}

# Strategy configuration
STRATEGY_CONFIG: Dict[str, Any] = {
    "default": "moving_average_crossover",
    "strategies": {
        "moving_average_crossover": {
            "short_window": 50,
            "long_window": 200,
        },
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2,
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        },
    },
}

# Notification configuration
NOTIFICATION_CONFIG: Dict[str, Any] = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": "",
        "from_email": "",
        "to_email": "",
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "headers": {},
    },
    "desktop": {
        "enabled": True,
    },
    "events": {
        "trade_executed": True,
        "trade_closed": True,
        "stop_loss_hit": True,
        "take_profit_hit": True,
        "error": True,
        "warning": False,
        "info": False,
    },
}

# AI configuration
AI_CONFIG: Dict[str, Any] = {
    "models": {
        "sentiment_analysis": {
            "type": "huggingface",
            "model_name": "finbert-sentiment",
            "threshold": 0.6,
        },
        "news_classification": {
            "type": "huggingface",
            "model_name": "finbert-news",
            "threshold": 0.7,
        },
        "price_prediction": {
            "type": "custom",
            "model_path": "storage/models/price_prediction.pkl",
            "features": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "bollinger_bands",
            ],
            "target": "close",
            "horizon": 5,  # days
        },
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "early_stopping": True,
        "patience": 10,
        "validation_split": 0.2,
        "test_split": 0.1,
    },
}

# System configuration
SYSTEM_CONFIG: Dict[str, Any] = {
    "timezone": "Asia/Kolkata",
    "locale": "en_IN",
    "currency": "INR",
    "date_format": "%Y-%m-%d",
    "time_format": "%H:%M:%S",
    "datetime_format": "%Y-%m-%d %H:%M:%S",
    "decimal_places": 2,
    "thousand_separator": ",",
    "decimal_separator": ".",
    "memory_limit": 1024 * 1024 * 1024,  # 1 GB
    "cpu_limit": 0.8,  # 80% of CPU
    "disk_limit": 1024 * 1024 * 1024 * 10,  # 10 GB
    "temp_dir": "storage/temp",
    "cache_dir": "storage/cache",
}

# Knowledge Extraction Configuration
KNOWLEDGE_EXTRACTION_CONFIG: Dict[str, Any] = {
    # OCR Configuration
    'ocr': {
        'enabled': True,
        'engine': 'tesseract',  # 'tesseract' or 'easyocr'
        'language': 'eng',  # Language for OCR
        'dpi': 300,  # DPI for image processing
        'preprocessing': {
            'enabled': True,
            'denoise': True,
            'contrast_enhancement': True,
            'deskew': True,
            'threshold_method': 'adaptive',  # 'simple', 'adaptive', or 'otsu'
        },
        'confidence_threshold': 0.65,  # Minimum confidence for OCR results
        'batch_size': 10,  # Number of pages to process in a batch
        'timeout': 300,  # Timeout in seconds for OCR processing
        'table_detection': {
            'enabled': True,
            'method': 'hough',  # 'hough', 'morphology', or 'neural'
            'confidence_threshold': 0.7,
        },
        'image_detection': {
            'enabled': True,
            'min_size': 100,  # Minimum size in pixels
            'save_format': 'png',  # Format to save extracted images
        },
        'formula_detection': {
            'enabled': True,
            'method': 'pattern',  # 'pattern' or 'neural'
            'confidence_threshold': 0.75,
        },
        'performance': {
            'threads': 4,  # Number of threads for parallel processing
            'gpu_acceleration': False,  # Use GPU acceleration if available
            'memory_limit': 2048,  # Memory limit in MB
        },
        'output': {
            'save_intermediate': False,  # Save intermediate processing results
            'format': 'json',  # Output format: 'json', 'text', 'xml'
            'include_confidence': True,  # Include confidence scores in output
        },
    },
    
    # Multimodal Content Processing Configuration
    'multimodal_processing': {
        'enabled': True,
        'text_processing': {
            'enabled': True,
            'language': 'en',  # Language for text processing
            'spacy_model': 'en_core_web_lg',  # Spacy model to use
            'min_sentence_length': 5,  # Minimum sentence length to process
            'max_sentence_length': 500,  # Maximum sentence length to process
            'preprocessing': {
                'remove_stopwords': True,
                'lemmatize': True,
                'remove_punctuation': False,
                'lowercase': True,
            },
        },
        'table_processing': {
            'enabled': True,
            'max_rows': 1000,  # Maximum number of rows to process
            'max_columns': 50,  # Maximum number of columns to process
            'header_detection': True,  # Detect table headers
            'data_type_detection': True,  # Detect data types in table columns
        },
        'image_processing': {
            'enabled': True,
            'chart_detection': True,  # Detect charts in images
            'chart_types': ['line', 'bar', 'candlestick', 'scatter', 'pie'],
            'extract_chart_data': True,  # Extract data from charts
            'max_image_size': 5000000,  # Maximum image size in bytes
            'min_image_resolution': [100, 100],  # Minimum width and height
        },
        'formula_processing': {
            'enabled': True,
            'parse_latex': True,  # Parse LaTeX formulas
            'convert_to_code': True,  # Convert formulas to executable code
            'supported_symbols': ['sum', 'prod', 'int', 'frac', 'sqrt'],
        },
        'cross_reference_resolution': {
            'enabled': True,
            'resolve_internal_references': True,  # Resolve references within the document
            'resolve_external_references': False,  # Resolve references to external sources
        },
    },
    
    # Knowledge Extraction Configuration
    'knowledge_extraction': {
        'enabled': True,
        'extraction_methods': {
            'rule_based': {
                'enabled': True,
                'rules_file': 'rules/trading_knowledge_rules.json',
                'confidence_threshold': 0.8,
            },
            'pattern_based': {
                'enabled': True,
                'patterns_file': 'patterns/trading_patterns.json',
                'min_pattern_length': 3,  # Minimum token length for patterns
                'confidence_threshold': 0.75,
            },
            'ml_based': {
                'enabled': True,
                'model_path': 'models/knowledge_extraction_model',
                'confidence_threshold': 0.7,
                'batch_size': 32,
            },
        },
        'entity_types': [
            'trading_strategy', 'indicator', 'pattern', 'rule', 'parameter',
            'market_condition', 'asset_class', 'risk_management', 'formula'
        ],
        'relation_types': [
            'uses', 'requires', 'generates_signal_on', 'applies_to',
            'has_parameter', 'is_effective_in', 'contradicts', 'enhances'
        ],
        'context_window_size': 5,  # Number of sentences for context
        'min_entity_occurrences': 2,  # Minimum occurrences for entity extraction
        'max_relation_distance': 10,  # Maximum token distance for relation extraction
    },
    
    # Knowledge Base Configuration
    'knowledge_base': {
        'enabled': True,
        'storage': {
            'type': 'mongodb',  # 'mongodb', 'neo4j', or 'file'
            'connection_string': 'mongodb://localhost:27017/',
            'database_name': 'friday_knowledge',
            'collections': {
                'entities': 'knowledge_entities',
                'relations': 'knowledge_relations',
                'sources': 'knowledge_sources',
                'strategies': 'extracted_strategies',
            },
        },
        'indexing': {
            'enabled': True,
            'method': 'vector',  # 'vector', 'inverted', or 'hybrid'
            'vector_dimensions': 384,  # Dimensions for vector embeddings
            'update_frequency': 'realtime',  # 'realtime', 'batch', or 'scheduled'
        },
        'versioning': {
            'enabled': True,
            'keep_history': True,  # Keep history of changes
            'max_versions': 10,  # Maximum number of versions to keep
        },
        'validation': {
            'enabled': True,
            'schema_validation': True,  # Validate against schema
            'consistency_check': True,  # Check for consistency
            'duplicate_detection': True,  # Detect duplicates
        },
    },
    
    # Strategy Generation Configuration
    'strategy_generation': {
        'enabled': True,
        'generation_methods': {
            'template_based': {
                'enabled': True,
                'templates_directory': 'templates/strategy_templates',
                'parameter_inference': True,  # Infer parameters from knowledge
            },
            'ml_based': {
                'enabled': True,
                'model_path': 'models/strategy_generation_model',
                'temperature': 0.7,  # Creativity parameter
                'max_tokens': 2000,  # Maximum tokens for generation
            },
        },
        'validation': {
            'enabled': True,
            'syntax_check': True,  # Check syntax of generated code
            'logic_check': True,  # Check logic of generated strategy
            'backtest': True,  # Run backtest on generated strategy
        },
        'output': {
            'format': 'python',  # 'python', 'pine', or 'json'
            'include_comments': True,  # Include comments in generated code
            'include_references': True,  # Include references to knowledge sources
        },
    },
    
    # Integration Configuration
    'integration': {
        'database': {
            'connection_string': 'mongodb://localhost:27017/',
            'database_name': 'friday_knowledge',
            'job_collection': 'extraction_jobs',
            'stats_collection': 'extraction_stats',
        },
        'event_system': {
            'enabled': True,
            'events': {
                'job_created': True,
                'job_updated': True,
                'job_completed': True,
                'job_failed': True,
                'knowledge_added': True,
                'strategy_generated': True,
            },
        },
        'logging': {
            'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
            'file': 'logs/knowledge_extraction.log',
            'max_size': 10485760,  # 10 MB
            'backup_count': 5,
        },
        'performance': {
            'max_concurrent_jobs': 4,
            'job_timeout': 3600,  # Timeout in seconds
            'batch_size': 5,  # Number of books to process in a batch
        },
    },
}